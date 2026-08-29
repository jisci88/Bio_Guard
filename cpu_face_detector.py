"""CPU-only SCRFD + YOLOv8 frontend for the existing hybrid face fusion."""

from __future__ import annotations

import math
from pathlib import Path
import threading
import time
from typing import Callable, Mapping, Sequence

import numpy as np

from hailo_face_detector import (
    Box,
    DetectionCandidate,
    FaceObservation,
    HybridFaceFusion,
    LatestFrameSlot,
    LetterboxTransform,
    _confidence,
    _finite_number,
    _label_map,
    _load_labels,
    _nms,
    _required_yolo_labels,
    _yolo_frame_box,
    decode_personface_nms,
    decode_scrfd,
)


def resolve_onnx(explicit_path: object | None, candidate_paths: Sequence[object]) -> Path:
    """Resolve one CPU ONNX model without importing any accelerator runtime."""
    candidates = [Path(path).expanduser() for path in candidate_paths]
    if explicit_path is not None:
        explicit = Path(explicit_path).expanduser()
        if explicit.is_file():
            return explicit.resolve()
        raise RuntimeError(f"CPU ONNX model not found at explicit path: {explicit}")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    rendered = ", ".join(str(path) for path in candidates)
    raise RuntimeError(f"CPU ONNX model not found; searched paths: {rendered}")


def _model_candidates(model: str, roots: Sequence[object] | None) -> list[Path]:
    project = Path(__file__).resolve().parent / "models" / "cpu"
    names = (f"{model}.onnx", f"{model.replace('_', '-')}.onnx")
    candidates: list[Path] = []
    for root in (*tuple(roots or ()), project):
        path = Path(root).expanduser()
        candidates.extend(path / name for name in names)
    return list(dict.fromkeys(candidates))


def _load_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for CPU face detection. Install python3-opencv or opencv-python."
        ) from exc
    return cv2


def _cpu_net(cv2, path: Path):
    try:
        net = cv2.dnn.readNetFromONNX(str(path))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return net
    except Exception as exc:
        raise RuntimeError(f"failed to load CPU ONNX model {path}: {exc}") from exc


def _blob(frame: np.ndarray, *, scale: float, mean: float) -> np.ndarray:
    numeric = frame.astype(np.float32)
    if mean:
        numeric -= mean
    if scale != 1.0:
        numeric *= scale
    return np.ascontiguousarray(np.moveaxis(numeric, -1, 0)[None])


def _forward(net: object, blob: np.ndarray) -> object:
    net.setInput(blob)
    names_getter = getattr(net, "getUnconnectedOutLayersNames", None)
    names = names_getter() if callable(names_getter) else ()
    return net.forward(names) if names else net.forward()


def _opencv_scrfd_tensors(raw: object, transform: LetterboxTransform) -> object:
    """Restore OpenCV's flattened two-anchor SCRFD outputs to HWC tensors."""
    if isinstance(raw, Mapping) or not isinstance(raw, (list, tuple)):
        return raw

    model_h, model_w = (int(round(value)) for value in transform.model_shape)
    formatted: dict[str, np.ndarray] = {}
    for stride in (8, 16, 32):
        height, width = model_h // stride, model_w // stride
        rows = height * width * 2
        tensors: dict[int, np.ndarray] = {}
        for tensor in raw:
            array = np.asarray(tensor)
            while array.ndim > 2 and array.shape[0] == 1:
                array = array[0]
            if array.ndim == 2 and array.shape[0] == rows and array.shape[1] in (1, 4, 10):
                tensors.setdefault(array.shape[1], array)
        if 1 not in tensors or 4 not in tensors:
            continue
        for channels, label in ((1, "score"), (4, "bbox"), (10, "kps")):
            if channels in tensors:
                array = tensors[channels]
                formatted[f"{label}_stride{stride}"] = array.reshape(
                    height, width, 2, channels
                ).reshape(height, width, 2 * channels)
    return formatted if formatted else raw


def decode_opencv_scrfd(
    raw: object,
    transform: LetterboxTransform,
    timestamp: object,
    confidence: float = 0.55,
    nms_iou: float = 0.40,
) -> list[DetectionCandidate]:
    """Decode SCRFD outputs returned by OpenCV DNN's named-output API."""
    return decode_scrfd(
        _opencv_scrfd_tensors(raw, transform), transform, timestamp, confidence, nms_iou
    )


def decode_yolov8(
    raw: object,
    timestamp: object,
    transform: LetterboxTransform,
    labels: object | None = None,
    confidence: float = 0.25,
    nms_iou: float = 0.45,
) -> tuple[list[DetectionCandidate], list[DetectionCandidate]]:
    """Decode standard YOLOv8 ONNX output (1x84xN) plus exported Nx6 variants."""
    threshold = _confidence(confidence)
    stamp = _finite_number(timestamp, "timestamp")
    mapping = _required_yolo_labels(labels)
    while isinstance(raw, (list, tuple)) and len(raw) == 1:
        raw = raw[0]
    array = np.asarray(raw)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 2 and array.shape[1] == 6:
        return decode_personface_nms(array, stamp, transform, mapping)
    if array.ndim != 2:
        raise RuntimeError(f"YOLOv8 CPU output must be 1x(4+C)xN or Nx6; got shape {array.shape}")
    if array.shape[0] >= 6 and array.shape[0] <= 128 and array.shape[1] != 6:
        array = array.T
    if array.shape[1] < 6:
        raise RuntimeError(f"YOLOv8 CPU output has fewer than 2 classes; got shape {array.shape}")

    faces: list[DetectionCandidate] = []
    persons: list[DetectionCandidate] = []
    for row in np.asarray(array, dtype=float):
        if not np.isfinite(row).all():
            continue
        class_id = int(np.argmax(row[4:]))
        kind = mapping.get(class_id)
        score = float(row[4 + class_id])
        if kind not in ("person", "face") or score < threshold:
            continue
        center_x, center_y, width, height = row[:4]
        box = _yolo_frame_box(
            (center_x - width / 2.0, center_y - height / 2.0,
             center_x + width / 2.0, center_y + height / 2.0),
            xyxy=True,
            transform=transform,
        )
        if box is not None:
            candidate = DetectionCandidate(box, score, kind, stamp, "YOLO")
            (faces if kind == "face" else persons).append(candidate)
    return _nms(faces, nms_iou), _nms(persons, nms_iou)


class CpuHybridFaceDetector:
    """Latest-frame CPU workers running lightweight SCRFD and YOLOv8 ONNX models."""

    def __init__(
        self,
        *,
        scrfd_onnx: object | None = None,
        yolo_onnx: object | None = None,
        model_roots: Sequence[object] | None = None,
        net_factory: Callable[[Path], object] | None = None,
        cv2_module: object | None = None,
        clock: Callable[[], float] = time.monotonic,
        fusion: HybridFaceFusion | None = None,
        labels: object | None = None,
        scrfd_input_size: int = 320,
        yolo_input_size: int = 320,
        yolo_idle_hz: float = 1.0,
        yolo_recovery_hz: float = 2.0,
    ) -> None:
        self.clock = clock
        self.scrfd_path = resolve_onnx(
            scrfd_onnx, _model_candidates("det_500m", model_roots)
        )
        self.yolo_path = resolve_onnx(yolo_onnx, _model_candidates("yolov8n", model_roots))
        for name, size in (("scrfd_input_size", scrfd_input_size), ("yolo_input_size", yolo_input_size)):
            if not isinstance(size, int) or size <= 0 or size % 32:
                raise ValueError(f"{name} must be a positive multiple of 32")

        sidecar_labels, label_path = _load_labels(self.yolo_path)
        self.labels = _required_yolo_labels(
            labels if labels is not None else (sidecar_labels or {0: "person"})
        )
        cv2 = cv2_module or _load_cv2()
        factory = net_factory or (lambda path: _cpu_net(cv2, path))
        try:
            self._nets = {"scrfd": factory(self.scrfd_path), "yolo": factory(self.yolo_path)}
        except Exception as exc:
            raise RuntimeError(f"CPU ONNX detector startup failed: {exc}") from exc

        self.fusion = fusion or HybridFaceFusion()
        self._input_sizes = {"scrfd": scrfd_input_size, "yolo": yolo_input_size}
        self.yolo_idle_hz = float(yolo_idle_hz)
        self.yolo_recovery_hz = float(yolo_recovery_hz)
        if self.yolo_idle_hz <= 0.0 or self.yolo_recovery_hz <= 0.0:
            raise ValueError("YOLO scheduling frequencies must be positive")
        self.model_paths = {"scrfd": self.scrfd_path, "yolo": self.yolo_path}
        self.startup_diagnostics = {
            "architecture": "CPU",
            "scrfd_path": str(self.scrfd_path),
            "personface_path": str(self.yolo_path),
            "label_path": None if label_path is None else str(label_path),
            "input_sizes": dict(self._input_sizes),
        }
        self._slots = {"scrfd": LatestFrameSlot(), "yolo": LatestFrameSlot()}
        self._state_lock = threading.Lock()
        self._close_event = threading.Event()
        self._closed = False
        self._next_frame_id = 1
        self._last_yolo_submit = -math.inf
        self._pending_error: BaseException | None = None
        self.last_error: BaseException | None = None
        self.completed_frame_ids = {"scrfd": -1, "yolo": -1}
        self.completion_counts = {"scrfd": 0, "yolo": 0}
        self.latest_scrfd: list[DetectionCandidate] = []
        self.latest_yolo_faces: list[DetectionCandidate] = []
        self.latest_persons: list[DetectionCandidate] = []
        self._threads = [
            threading.Thread(target=self._worker, args=(model,), name=f"cpu-{model}", daemon=True)
            for model in ("scrfd", "yolo")
        ]
        for thread in self._threads:
            thread.start()

    def _store_error(self, model: str, exc: BaseException) -> None:
        error = RuntimeError(f"{model.upper()} CPU inference failed: {exc}")
        error.__cause__ = exc
        with self._state_lock:
            self.last_error = error
            if self._pending_error is None:
                self._pending_error = error

    def _publish_result(self, model: str, frame_id: int, value: object) -> None:
        with self._state_lock:
            if frame_id <= self.completed_frame_ids[model]:
                return
            self.completed_frame_ids[model] = frame_id
            self.completion_counts[model] += 1
            if model == "scrfd":
                self.latest_scrfd = list(value)
            else:
                faces, persons = value
                self.latest_yolo_faces = list(faces)
                self.latest_persons = list(persons)

    def _worker(self, model: str) -> None:
        slot, net = self._slots[model], self._nets[model]
        while not self._close_event.is_set():
            item = slot.get(timeout=0.1)
            if item is None:
                if slot.closed:
                    return
                continue
            frame_id, frame, timestamp = item
            try:
                prepared, transform = self._letterbox(frame, self._input_sizes[model])
                if model == "scrfd":
                    raw = _forward(net, _blob(prepared, scale=1.0 / 128.0, mean=127.5))
                    decoded: object = decode_opencv_scrfd(raw, transform, timestamp)
                else:
                    raw = _forward(net, _blob(prepared, scale=1.0 / 255.0, mean=0.0))
                    decoded = decode_yolov8(raw, timestamp, transform, self.labels)
                self._publish_result(model, frame_id, decoded)
            except Exception as exc:
                self._store_error(model, exc)

    @staticmethod
    def _letterbox(frame: np.ndarray, input_size: int) -> tuple[np.ndarray, LetterboxTransform]:
        transform = LetterboxTransform.create(frame.shape[:2], (input_size, input_size))
        target_h = max(1, int(round(frame.shape[0] * transform.scale)))
        target_w = max(1, int(round(frame.shape[1] * transform.scale)))
        y_indices = np.minimum((np.arange(target_h) / transform.scale).astype(int), frame.shape[0] - 1)
        x_indices = np.minimum((np.arange(target_w) / transform.scale).astype(int), frame.shape[1] - 1)
        result = np.zeros((input_size, input_size, 3), dtype=np.uint8)
        top, left = (input_size - target_h) // 2, (input_size - target_w) // 2
        result[top:top + target_h, left:left + target_w] = frame[y_indices[:, None], x_indices[None, :]]
        return result, transform

    def _raise_pending_error(self) -> None:
        with self._state_lock:
            error, self._pending_error = self._pending_error, None
        if error is not None:
            raise error

    def submit(
        self,
        frame_rgb: object,
        now: object | None = None,
        flow_box: Box | None = None,
        flow_quality: float = 0.0,
    ) -> FaceObservation:
        if self._closed:
            raise RuntimeError("CpuHybridFaceDetector is closed")
        self._raise_pending_error()
        frame = np.asarray(frame_rgb)
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("frame_rgb must be an HxWx3 uint8 RGB array")
        owned = np.array(frame, dtype=np.uint8, order="C", copy=True)
        stamp = _finite_number(self.clock() if now is None else now, "now")
        with self._state_lock:
            scrfd, yolo_faces, persons = (
                list(self.latest_scrfd), list(self.latest_yolo_faces), list(self.latest_persons)
            )
            frame_id = self._next_frame_id
            self._next_frame_id += 1
        observation = self.fusion.update(
            scrfd, yolo_faces, persons, flow_box, flow_quality, owned.shape[:2], stamp
        )
        self._slots["scrfd"].put(frame_id, owned, stamp)
        rate = self.yolo_idle_hz if observation.source == "SCRFD" else self.yolo_recovery_hz
        if stamp - self._last_yolo_submit >= 1.0 / rate:
            self._slots["yolo"].put(frame_id, owned, stamp)
            self._last_yolo_submit = stamp
        return observation

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._close_event.set()
        for slot in self._slots.values():
            slot.close()
        for thread in self._threads:
            thread.join(timeout=2.0)
