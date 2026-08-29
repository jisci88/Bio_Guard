"""Hardware-independent face detection observations and pixel-box geometry."""

from dataclasses import dataclass
import importlib
import inspect
import json
import math
from pathlib import Path
import re
import threading
import time
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

Box = tuple[tuple[float, float], tuple[float, float]]


def _box(value: object, *, allow_none: bool = False) -> Optional[Box]:
    if value is None and allow_none:
        return None
    try:
        y, x = value  # type: ignore[misc]
        y1, y2 = y
        x1, x2 = x
        values = tuple(float(v) for v in (y1, y2, x1, x2))
    except (TypeError, ValueError, IndexError, OverflowError):
        raise ValueError("box must be ((y1, y2), (x1, x2))") from None
    if not all(math.isfinite(v) for v in values):
        raise ValueError("box coordinates must be finite")
    return ((values[0], values[1]), (values[2], values[3]))


def _shape(value: object) -> tuple[float, float]:
    try:
        h, w = (float(v) for v in value)  # type: ignore[misc]
    except (TypeError, ValueError, OverflowError):
        raise ValueError("shape must be (height, width)") from None
    if not math.isfinite(h) or not math.isfinite(w) or h <= 0 or w <= 0:
        raise ValueError("shape dimensions must be positive and finite")
    return h, w


def _confidence(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError("confidence must be finite") from None
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    return result


def _finite_number(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"{name} must be finite") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class DetectionCandidate:
    box: Box
    confidence: float
    kind: str
    timestamp: float
    source: str
    track_id: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "box", _box(self.box))
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        if self.kind not in ("face", "person"):
            raise ValueError("kind must be face or person")
        if self.source not in ("SCRFD", "YOLO"):
            raise ValueError("source must be SCRFD or YOLO")
        object.__setattr__(self, "timestamp", _finite_number(self.timestamp, "timestamp"))
        if self.track_id is not None and (isinstance(self.track_id, bool) or not isinstance(self.track_id, int)):
            raise ValueError("track_id must be an integer or None")


@dataclass(frozen=True)
class FaceObservation:
    sensor_box: Box | None
    motor_box: Box | None
    face_confidence: float
    motor_confidence: float
    source: str
    timestamp: float
    track_id: int | None
    result_age: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "sensor_box", _box(self.sensor_box, allow_none=True))
        object.__setattr__(self, "motor_box", _box(self.motor_box, allow_none=True))
        object.__setattr__(self, "face_confidence", _confidence(self.face_confidence))
        object.__setattr__(self, "motor_confidence", _confidence(self.motor_confidence))
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be non-empty")
        object.__setattr__(self, "timestamp", _finite_number(self.timestamp, "timestamp"))
        object.__setattr__(self, "result_age", _finite_number(self.result_age, "result_age"))
        if self.track_id is not None and (isinstance(self.track_id, bool) or not isinstance(self.track_id, int)):
            raise ValueError("track_id must be an integer or None")


@dataclass(frozen=True)
class LetterboxTransform:
    frame_shape: tuple[float, float]
    model_shape: tuple[float, float]
    scale: float
    pad_y: float
    pad_x: float

    @classmethod
    def create(cls, frame_shape: object, model_shape: object) -> "LetterboxTransform":
        fh, fw = _shape(frame_shape)
        mh, mw = _shape(model_shape)
        scale = min(mh / fh, mw / fw)
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("letterbox scale must be positive and finite")
        return cls((fh, fw), (mh, mw), scale, (mh - fh * scale) / 2.0, (mw - fw * scale) / 2.0)

    def to_model_box(self, box: Box) -> Box:
        b = _box(box)
        return ((b[0][0] * self.scale + self.pad_y, b[0][1] * self.scale + self.pad_y),
                (b[1][0] * self.scale + self.pad_x, b[1][1] * self.scale + self.pad_x))

    def to_frame_box(self, box: Box) -> Box:
        b = _box(box)
        return (((b[0][0] - self.pad_y) / self.scale, (b[0][1] - self.pad_y) / self.scale),
                ((b[1][0] - self.pad_x) / self.scale, (b[1][1] - self.pad_x) / self.scale))


def clip_box(box: Box, frame_shape: object) -> Box | None:
    b = _box(box)
    h, w = _shape(frame_shape)
    y1, y2 = max(0.0, min(h, b[0][0])), max(0.0, min(h, b[0][1]))
    x1, x2 = max(0.0, min(w, b[1][0])), max(0.0, min(w, b[1][1]))
    return ((y1, y2), (x1, x2)) if y2 > y1 and x2 > x1 else None


def box_iou(first: Box, second: Box) -> float:
    a, b = _box(first), _box(second)
    try:
        height_a, width_a = max(0.0, a[0][1] - a[0][0]), max(0.0, a[1][1] - a[1][0])
        height_b, width_b = max(0.0, b[0][1] - b[0][0]), max(0.0, b[1][1] - b[1][0])
        iy = max(0.0, min(a[0][1], b[0][1]) - max(a[0][0], b[0][0]))
        ix = max(0.0, min(a[1][1], b[1][1]) - max(a[1][0], b[1][0]))
    except OverflowError:
        raise ValueError("box geometry is not representable") from None
    dimensions = (height_a, width_a, height_b, width_b, iy, ix)
    if not all(math.isfinite(value) for value in dimensions):
        raise ValueError("box geometry is not representable")
    scale_y = max(height_a, height_b, iy)
    scale_x = max(width_a, width_b, ix)
    if scale_y == 0.0 or scale_x == 0.0:
        return 0.0
    iy /= scale_y; height_a /= scale_y; height_b /= scale_y
    ix /= scale_x; width_a /= scale_x; width_b /= scale_x
    inter = iy * ix
    area_a = height_a * width_a
    area_b = height_b * width_b
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def edge_complete_box(visible: Box, previous_full_size: tuple[float, float], frame_shape: object, edge_margin: float) -> Box:
    b = _box(visible)
    h, w = _shape(frame_shape)
    try:
        full_h, full_w = (float(v) for v in previous_full_size)
        margin = float(edge_margin)
    except (TypeError, ValueError):
        raise ValueError("invalid previous size or edge margin") from None
    if not all(math.isfinite(v) for v in (full_h, full_w, margin)) or full_h <= 0 or full_w <= 0 or margin < 0:
        raise ValueError("invalid previous size or edge margin")
    y1, y2 = b[0]; x1, x2 = b[1]
    if y1 <= h * margin: y1 = y2 - full_h
    elif y2 >= h * (1 - margin): y2 = y1 + full_h
    else: y1, y2 = (y1 + y2 - full_h) / 2, (y1 + y2 + full_h) / 2
    if x1 <= w * margin: x1 = x2 - full_w
    elif x2 >= w * (1 - margin): x2 = x1 + full_w
    else: x1, x2 = (x1 + x2 - full_w) / 2, (x1 + x2 + full_w) / 2
    return ((y1, y2), (x1, x2))


def person_head_box(person_box: Box, frame_shape: object) -> Box | None:
    visible = clip_box(person_box, frame_shape)
    if visible is None:
        return None
    y1, y2 = visible[0]; x1, x2 = visible[1]
    height, width = y2 - y1, x2 - x1
    head_h = height / 3.0
    head_w = width * 0.7
    cx = (x1 + x2) / 2.0
    return ((y1, y1 + head_h), (cx - head_w / 2.0, cx + head_w / 2.0))


def _area(box: Box) -> float:
    height = max(0.0, box[0][1] - box[0][0])
    width = max(0.0, box[1][1] - box[1][0])
    return height * width


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


class HybridFaceFusion:
    """Deterministically choose one safe face observation per frame."""

    def __init__(
        self,
        *,
        scrfd_confidence: float = 0.55,
        person_confidence: float = 0.50,
        association_threshold: float = 0.35,
        visible_area_threshold: float = 0.70,
        max_face_age_sec: float = 0.20,
        flow_hold_sec: float = 0.25,
        person_hold_sec: float = 0.60,
        edge_margin: float = 0.12,
        flow_min_quality: float = 0.45,
    ) -> None:
        self.scrfd_confidence = _confidence(scrfd_confidence)
        self.person_confidence = _confidence(person_confidence)
        self.association_threshold = _confidence(association_threshold)
        self.visible_area_threshold = _confidence(visible_area_threshold)
        self.flow_min_quality = _confidence(flow_min_quality)
        self.edge_margin = _confidence(edge_margin)
        self.max_face_age_sec = self._duration(max_face_age_sec, "max_face_age_sec")
        self.flow_hold_sec = self._duration(flow_hold_sec, "flow_hold_sec")
        self.person_hold_sec = self._duration(person_hold_sec, "person_hold_sec")
        self._active_track_id: int | None = None
        self._next_track_id = 1
        self._track_box: Box | None = None
        self._last_full_face_size: tuple[float, float] | None = None
        self._last_fresh_scrfd_time: float | None = None
        self._cached_person: tuple[Box, float, float] | None = None

    @staticmethod
    def _duration(value: object, name: str) -> float:
        result = _finite_number(value, name)
        if result < 0.0:
            raise ValueError(f"{name} must be non-negative")
        return result

    def _track_id_for(self, candidate: DetectionCandidate) -> int:
        if candidate.track_id is not None:
            return candidate.track_id
        if self._active_track_id is None:
            result = self._next_track_id
            self._next_track_id += 1
            return result
        return self._active_track_id

    @staticmethod
    def _candidates(
        values: Sequence[DetectionCandidate], *, kind: str, source: str, now: float
    ) -> list[DetectionCandidate]:
        try:
            result = list(values)
        except (TypeError, ValueError, OverflowError):
            raise ValueError("candidates must be a sequence") from None
        for candidate in result:
            if not isinstance(candidate, DetectionCandidate):
                raise ValueError("candidates must be DetectionCandidate instances")
            if candidate.kind != kind or candidate.source != source:
                raise ValueError("candidate kind or source is invalid for this input")
            if candidate.timestamp > now:
                raise ValueError("candidate timestamp cannot be in the future")
        return result

    @staticmethod
    def _is_valid_box(box: Box) -> bool:
        return box[0][1] > box[0][0] and box[1][1] > box[1][0]

    @staticmethod
    def _score(first: Box, second: Box) -> float:
        """Association score using finite, bounded overlap, center, and size terms."""
        if not HybridFaceFusion._is_valid_box(first) or not HybridFaceFusion._is_valid_box(second):
            return 0.0
        try:
            iou = box_iou(first, second)
            cy1 = (first[0][0] + first[0][1]) / 2.0
            cx1 = (first[1][0] + first[1][1]) / 2.0
            cy2 = (second[0][0] + second[0][1]) / 2.0
            cx2 = (second[1][0] + second[1][1]) / 2.0
            h1, w1 = first[0][1] - first[0][0], first[1][1] - first[1][0]
            h2, w2 = second[0][1] - second[0][0], second[1][1] - second[1][0]
            diagonal = max(math.hypot(h1, w1), math.hypot(h2, w2))
            distance = math.hypot(cy1 - cy2, cx1 - cx2)
            center = 1.0 if distance == 0.0 else 1.0 - distance / diagonal
            size = min(h1 * w1, h2 * w2) / max(h1 * w1, h2 * w2)
        except (ValueError, OverflowError, ZeroDivisionError):
            return 0.0
        return _clamp(0.55 * _clamp(iou) + 0.30 * _clamp(center) + 0.15 * _clamp(size))

    def _select(self, candidates: Sequence[DetectionCandidate]) -> DetectionCandidate | None:
        if not candidates:
            return None
        if self._track_box is None:
            return max(enumerate(candidates), key=lambda item: (item[1].confidence, -item[0]))[1]
        scored = [(self._score(self._track_box, candidate.box),
                   candidate.confidence, -index, candidate)
                  for index, candidate in enumerate(candidates)]
        score, _, _, selected = max(scored, key=lambda item: item[:3])
        return selected if score >= self.association_threshold else None

    def _select_person(
        self,
        candidates: Sequence[DetectionCandidate],
        yolo_faces: Sequence[DetectionCandidate],
    ) -> tuple[DetectionCandidate | None, Box | None]:
        if not candidates:
            return None, None
        if self._track_box is not None:
            matched_face = self._select(yolo_faces)
            if matched_face is not None:
                containing = [
                    (self._score(self._track_box, candidate.box), candidate.confidence,
                     -index, candidate)
                    for index, candidate in enumerate(candidates)
                    if (candidate.box[0][0] <= matched_face.box[0][0]
                        and matched_face.box[0][1] <= candidate.box[0][1]
                        and candidate.box[1][0] <= matched_face.box[1][0]
                        and matched_face.box[1][1] <= candidate.box[1][1])
                ]
                if containing:
                    selected = max(containing, key=lambda item: item[:3])[3]
                    return selected, matched_face.box
        return self._select(candidates), None

    def _observation(
        self,
        sensor_box: Box | None,
        motor_box: Box | None,
        face_confidence: float,
        motor_confidence: float,
        source: str,
        timestamp: float,
        now: float,
    ) -> FaceObservation:
        return FaceObservation(
            sensor_box, motor_box, face_confidence, motor_confidence, source,
            timestamp, self._active_track_id, max(0.0, now - timestamp),
        )

    def _none(self, now: float) -> FaceObservation:
        return FaceObservation(None, None, 0.0, 0.0, "NONE", now, self._active_track_id, 0.0)

    def _touches_edge(self, box: Box, frame_shape: tuple[float, float]) -> bool:
        h, w = frame_shape
        return (box[0][0] <= h * self.edge_margin or box[0][1] >= h * (1.0 - self.edge_margin)
                or box[1][0] <= w * self.edge_margin or box[1][1] >= w * (1.0 - self.edge_margin))

    def _cache_person(
        self,
        candidate: DetectionCandidate,
        frame_shape: tuple[float, float],
        association_box: Box | None = None,
    ) -> None:
        head = person_head_box(candidate.box, frame_shape)
        if head is not None:
            self._cached_person = (head, candidate.confidence, candidate.timestamp)
            self._track_box = candidate.box if association_box is None else association_box

    def update(
        self,
        scrfd_faces: Sequence[DetectionCandidate],
        yolo_faces: Sequence[DetectionCandidate],
        persons: Sequence[DetectionCandidate],
        flow_box: Box | None,
        flow_quality: float,
        frame_shape: tuple[int, int],
        now: float,
    ) -> FaceObservation:
        now_value = _finite_number(now, "now")
        quality = _confidence(flow_quality)
        shape = _shape(frame_shape)
        scrfd = self._candidates(scrfd_faces, kind="face", source="SCRFD", now=now_value)
        yolo = self._candidates(yolo_faces, kind="face", source="YOLO", now=now_value)
        people = self._candidates(persons, kind="person", source="YOLO", now=now_value)
        if flow_box is not None:
            flow = _box(flow_box)
            if not self._is_valid_box(flow):
                raise ValueError("flow_box must have positive area")
        else:
            flow = None

        fresh_faces = [candidate for candidate in scrfd
                       if candidate.confidence >= self.scrfd_confidence
                       and now_value - candidate.timestamp <= self.max_face_age_sec]
        selected_face = self._select(fresh_faces)
        if selected_face is not None:
            self._active_track_id = self._track_id_for(selected_face)
            visible = clip_box(selected_face.box, shape)
            edge = self._touches_edge(selected_face.box, shape)
            if edge:
                motor = (
                    edge_complete_box(selected_face.box, self._last_full_face_size, shape, self.edge_margin)
                    if self._last_full_face_size is not None else selected_face.box
                )
                self._track_box = motor
                return self._observation(None, motor, selected_face.confidence, selected_face.confidence,
                                         "SCRFD_EDGE", selected_face.timestamp, now_value)
            self._track_box = selected_face.box
            if not edge:
                self._last_fresh_scrfd_time = selected_face.timestamp
                self._last_full_face_size = (
                    selected_face.box[0][1] - selected_face.box[0][0],
                    selected_face.box[1][1] - selected_face.box[1][0],
                )
            visible_ratio = 0.0 if visible is None or _area(selected_face.box) <= 0 else _area(visible) / _area(selected_face.box)
            sensor = visible if visible_ratio >= self.visible_area_threshold else None
            return self._observation(sensor, selected_face.box, selected_face.confidence, selected_face.confidence,
                                     "SCRFD", selected_face.timestamp, now_value)

        fresh_people = [candidate for candidate in people
                        if candidate.confidence >= self.person_confidence
                        and now_value - candidate.timestamp < self.person_hold_sec]
        fresh_yolo = [candidate for candidate in yolo
                      if now_value - candidate.timestamp < self.person_hold_sec]
        selected_person, association_box = self._select_person(fresh_people, fresh_yolo)
        if selected_person is not None:
            self._active_track_id = self._track_id_for(selected_person)
            self._cache_person(selected_person, shape, association_box)

        if (self._last_fresh_scrfd_time is not None and flow is not None
                and quality >= self.flow_min_quality
                and now_value - self._last_fresh_scrfd_time <= self.flow_hold_sec):
            self._track_box = flow
            return self._observation(None, flow, 0.0, quality, "FLOW", now_value, now_value)

        if self._cached_person is not None:
            head, confidence, timestamp = self._cached_person
            if now_value - timestamp < self.person_hold_sec:
                return self._observation(None, head, 0.0, confidence, "PERSON_HEAD", timestamp, now_value)
            self._cached_person = None
        return self._none(now_value)


def resolve_hef(explicit_path: object | None, candidate_paths: Sequence[object]) -> Path:
    """Resolve a HEF without importing Hailo; the first existing path wins."""
    if explicit_path is not None:
        explicit = Path(explicit_path).expanduser()
        if explicit.is_file():
            return explicit.resolve()
        raise RuntimeError(
            f"HAILO8 HEF not found at {explicit}; the explicit path is terminal and no fallback "
            "model was selected. Pass an existing HAILO8 (not HAILO8L) HEF."
        )
    searched = [Path(path).expanduser() for path in candidate_paths]
    for path in searched:
        if path.is_file():
            return path.resolve()
    rendered = ", ".join(str(path) for path in searched) or "<none>"
    model = searched[0].stem if searched else "model"
    raise RuntimeError(
        f"HAILO8 HEF model not found: {model}; searched paths: {rendered}. "
        "Install the HAILO8 (not HAILO8L) model or pass an explicit HEF path."
    )


def _metadata_architecture(metadata: object) -> str | None:
    for name in ("architecture", "arch", "target_arch", "device_architecture", "hw_arch"):
        value = getattr(metadata, name, None)
        if value is not None:
            return str(value)
    for name in ("get_architecture", "get_target_arch", "get_device_architecture", "get_hw_arch"):
        getter = getattr(metadata, name, None)
        if callable(getter):
            value = getter()
            if value is not None:
                return str(value)
    if isinstance(metadata, (str, bytes)):
        return str(metadata)
    return None


def validate_hef_arch(hef_or_metadata: object, path: object, expected: str = "HAILO8") -> None:
    """Reject a HEF that positively identifies an architecture other than HAILO8."""
    actual = _metadata_architecture(hef_or_metadata)
    if actual is None:
        return
    expected_name = expected.upper().replace("-", "").replace("_", "")
    actual_name = actual.upper().replace("-", "").replace("_", "")
    if expected_name not in actual_name or f"{expected_name}L" in actual_name:
        raise RuntimeError(
            f"HEF {path} targets {actual}; expected {expected} for the installed Hailo-8 device "
            "(HAILO8L HEFs are incompatible)."
        )


def _unwrap_result(raw: object) -> object:
    current = raw
    for _ in range(4):
        if isinstance(current, Mapping):
            for key in ("outputs", "result", "results", "payload"):
                if key in current and len(current) == 1:
                    current = current[key]
                    break
            else:
                return current
            continue
        for name in ("outputs", "result", "results", "payload"):
            value = getattr(current, name, None)
            if value is not None:
                current = value() if callable(value) else value
                break
        else:
            return current
    return current


def _to_hwc(tensor: object, height: int, width: int) -> np.ndarray | None:
    array = np.asarray(tensor)
    while array.ndim > 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 2 and array.shape[0] == height * width:
        return array.reshape(height, width, -1)
    if array.ndim != 3:
        return None
    if array.shape[:2] == (height, width):
        return array
    if array.shape[1:] == (height, width):
        return np.moveaxis(array, 0, -1)
    return None


def _raw_named_tensors(raw: object) -> list[tuple[str, object]]:
    value = _unwrap_result(raw)
    if isinstance(value, Mapping):
        return [(str(name), tensor) for name, tensor in value.items()]
    if isinstance(value, np.ndarray) and value.dtype == object:
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [(str(index), tensor) for index, tensor in enumerate(value)]
    return [("0", value)]


def _nms(candidates: Sequence[DetectionCandidate], threshold: float) -> list[DetectionCandidate]:
    ordered = sorted(enumerate(candidates), key=lambda item: (-item[1].confidence, item[0]))
    kept: list[DetectionCandidate] = []
    for _, candidate in ordered:
        if all(box_iou(candidate.box, existing.box) <= threshold for existing in kept):
            kept.append(candidate)
    return kept


def decode_scrfd(
    raw: object,
    transform: LetterboxTransform,
    timestamp: object,
    confidence: float = 0.55,
    nms_iou: float = 0.40,
) -> list[DetectionCandidate]:
    """Decode common two-anchor SCRFD stride 8/16/32 Hailo tensors."""
    if not isinstance(transform, LetterboxTransform):
        raise ValueError("transform must be a LetterboxTransform")
    stamp = _finite_number(timestamp, "timestamp")
    threshold = _confidence(confidence)
    iou_threshold = _confidence(nms_iou)
    model_h, model_w = (int(round(value)) for value in transform.model_shape)
    tensors = _raw_named_tensors(raw)
    decoded: list[DetectionCandidate] = []
    matched_stride = False

    for stride in (8, 16, 32):
        height, width = model_h // stride, model_w // stride
        shaped: list[tuple[str, np.ndarray]] = []
        for name, tensor in tensors:
            array = _to_hwc(tensor, height, width)
            if array is not None and array.shape[-1] > 0:
                shaped.append((name.lower(), array))
        if len(shaped) < 2:
            continue
        score_hints = [item for item in shaped if "score" in item[0] or "cls" in item[0]]
        possible_scores = score_hints or shaped
        pairs: list[tuple[tuple[str, np.ndarray], tuple[str, np.ndarray]]] = []
        for score_item in possible_scores:
            score_channels = score_item[1].shape[-1]
            for bbox_item in shaped:
                bbox_name = bbox_item[0]
                if bbox_item is score_item or any(
                    marker in bbox_name for marker in ("landmark", "keypoint", "kps")
                ):
                    continue
                if bbox_item[1].shape[-1] == score_channels * 4:
                    pairs.append((score_item, bbox_item))
        if not pairs:
            continue
        (_, scores), (_, bbox) = min(
            pairs,
            key=lambda pair: (
                0 if "bbox" in pair[1][0] or "box" in pair[1][0] else 1,
                pair[0][1].shape[-1],
            ),
        )
        anchors = bbox.shape[-1] // 4
        channels = scores.shape[-1]
        if channels == anchors:
            foreground = scores
        elif channels == anchors * 2:
            foreground = scores[..., anchors:]
        else:
            continue
        matched_stride = True
        for y, x, anchor in np.argwhere(foreground >= threshold):
            score = float(foreground[y, x, anchor])
            distances = np.asarray(bbox[y, x, anchor * 4:(anchor + 1) * 4], dtype=float)
            if distances.shape != (4,) or not np.isfinite(distances).all():
                continue
            left, top, right, bottom = distances * stride
            center_x, center_y = float(x) * stride, float(y) * stride
            model_box = ((center_y - top, center_y + bottom), (center_x - left, center_x + right))
            frame_box = clip_box(transform.to_frame_box(model_box), transform.frame_shape)
            if frame_box is not None:
                decoded.append(DetectionCandidate(frame_box, score, "face", stamp, "SCRFD"))
    if not matched_stride:
        shapes = [tuple(np.asarray(tensor).shape) for _, tensor in tensors]
        raise RuntimeError(
            f"SCRFD callback outputs do not contain score/bbox tensors for strides 8/16/32; got {shapes}"
        )
    return _nms(decoded, iou_threshold)


def _label_map(labels: object | None) -> dict[int, str]:
    if labels is None:
        return {}
    if isinstance(labels, Mapping):
        result: dict[int, str] = {}
        for key, value in labels.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                continue
            result[index] = str(value).strip().lower()
        return result
    if isinstance(labels, (list, tuple)):
        return {index: str(value).strip().lower() for index, value in enumerate(labels)}
    raise RuntimeError("YOLO labels must be a class-id mapping or sequence containing person and face")


def _required_yolo_labels(
    labels: object | None, *, require_face: bool = False
) -> dict[int, str]:
    mapping = _label_map(labels)
    person_ids = [class_id for class_id, name in mapping.items() if name == "person"]
    face_ids = [class_id for class_id, name in mapping.items() if name == "face"]
    if not person_ids or (require_face and not face_ids):
        raise RuntimeError(
            "YOLO person/face labels are unresolved: labels must map a class to person"
            + (" and a class to face" if require_face else "")
            + "; provide the HEF label sidecar or explicit labels."
        )
    return mapping


def _required_personface_labels(labels: object | None) -> dict[int, str]:
    mapping = _label_map(labels)
    if mapping.get(0) != "person" or mapping.get(1) != "face":
        raise RuntimeError(
            "YOLO person/face labels are unresolved: class 0 and class 1 must map to person and face; "
            "provide the HEF label sidecar or labels={0: 'person', 1: 'face'}."
        )
    return mapping


def _yolo_frame_box(
    coordinates: Sequence[object], *, xyxy: bool, transform: LetterboxTransform | None
) -> Box | None:
    values = np.asarray(coordinates, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all():
        return None
    if xyxy:
        x1, y1, x2, y2 = values
    else:
        y1, x1, y2, x2 = values
    if transform is not None and float(np.max(np.abs(values))) <= 1.5:
        model_h, model_w = transform.model_shape
        y1, y2 = y1 * model_h, y2 * model_h
        x1, x2 = x1 * model_w, x2 * model_w
    model_box: Box = ((float(y1), float(y2)), (float(x1), float(x2)))
    if transform is None:
        return model_box if y2 > y1 and x2 > x1 else None
    return clip_box(transform.to_frame_box(model_box), transform.frame_shape)


def decode_personface_nms(
    raw: object,
    timestamp: object,
    transform: LetterboxTransform | None = None,
    labels: object | None = None,
) -> tuple[list[DetectionCandidate], list[DetectionCandidate]]:
    """Decode Hailo NMS-by-class, named dictionaries, or YOLO Nx6 rows."""
    stamp = _finite_number(timestamp, "timestamp")
    value = _unwrap_result(raw)
    faces: list[DetectionCandidate] = []
    persons: list[DetectionCandidate] = []

    def add(kind: str, row: object, *, xyxy: bool) -> None:
        array = np.asarray(row, dtype=float).reshape(-1)
        if array.size < 5:
            return
        try:
            score = _confidence(array[4])
        except ValueError:
            return
        box = _yolo_frame_box(array[:4], xyxy=xyxy, transform=transform)
        if box is None:
            return
        candidate = DetectionCandidate(box, score, kind, stamp, "YOLO")
        (faces if kind == "face" else persons).append(candidate)

    if isinstance(value, Mapping):
        if len(value) == 1:
            only_name, only_value = next(iter(value.items()))
            if str(only_name).lower() not in ("person", "face", "0", "1"):
                return decode_personface_nms(only_value, stamp, transform, labels)
        mapping = _label_map(labels)
        recognized = False
        for key, rows in value.items():
            key_name = str(key).strip().lower()
            if key_name in ("person", "face"):
                kind = key_name
            else:
                try:
                    kind = mapping.get(int(key_name), "")
                except ValueError:
                    kind = ""
            if kind not in ("person", "face"):
                continue
            recognized = True
            for row in np.asarray(rows, dtype=float).reshape(-1, 5):
                add(kind, row, xyxy=False)
        if recognized:
            return faces, persons
        raise RuntimeError("YOLO output dictionary has no person/face class keys and labels could not resolve them")

    try:
        array = np.asarray(value)
    except ValueError:
        array = np.asarray(value, dtype=object)
    if array.dtype == object:
        value = array.tolist()
        array = np.asarray(value, dtype=object)
    elif array.ndim == 3 and array.shape[-1] == 5:
        value = list(array)
    if isinstance(value, (list, tuple)) and not (
        array.ndim == 2 and array.shape[-1] == 6 and array.dtype != object
    ):
        mapping = _required_yolo_labels(labels)
        for class_id, rows in enumerate(value):
            kind = mapping.get(class_id)
            if kind not in ("person", "face"):
                continue
            numeric = np.asarray(rows, dtype=float)
            if numeric.size == 0:
                continue
            for row in numeric.reshape(-1, numeric.shape[-1]):
                add(kind, row, xyxy=False)
        return faces, persons

    numeric = np.asarray(value, dtype=float)
    if numeric.ndim == 3 and numeric.shape[0] == 1:
        numeric = numeric[0]
    if numeric.ndim != 2 or numeric.shape[1] != 6:
        raise RuntimeError(f"YOLO callback output must be NMS-by-class or Nx6; got shape {numeric.shape}")
    mapping = _required_yolo_labels(labels)
    for row in numeric:
        class_value = float(row[5])
        class_id = int(class_value)
        if class_value != class_id or class_id not in mapping:
            continue
        kind = mapping[class_id]
        if kind in ("person", "face"):
            add(kind, row, xyxy=True)
    return faces, persons


class LatestFrameSlot:
    """A closeable capacity-one slot; put replaces an unconsumed frame."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._item: tuple[int, np.ndarray, float] | None = None
        self._closed = False

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def put(self, frame_id: int, frame: np.ndarray, timestamp: float) -> bool:
        with self._condition:
            if self._closed:
                return False
            self._item = (frame_id, frame, timestamp)
            self._condition.notify()
            return True

    def get(self, timeout: float | None = None) -> tuple[int, np.ndarray, float] | None:
        with self._condition:
            if self._item is None and not self._closed:
                self._condition.wait(timeout)
            if self._closed:
                return None
            item, self._item = self._item, None
            return item

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._item = None
            self._condition.notify_all()


def _load_hailo_runtime() -> type:
    """Load the Hailo Apps inference helper only on a hardware runtime path."""
    failures: list[str] = []
    for module_name in (
        "hailo_apps.python.core.common.hailo_inference",
        "hailo_apps_infra.hailo_inference",
        "hailo_apps_infra.hailo_rpi_common",
    ):
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            failures.append(f"{module_name}: {exc}")
            continue
        for class_name in ("HailoInfer", "HailoAsyncInference"):
            runner = getattr(module, class_name, None)
            if runner is not None:
                return runner
    raise RuntimeError(
        "Hailo runtime is unavailable. Install HailoRT 4.23 Python bindings and Hailo Apps, "
        "then source the Hailo Apps setup environment. Tried: " + "; ".join(failures)
    )


def _model_filenames(model: str) -> tuple[str, ...]:
    aliases = {model, model.replace("_", "-"), model.replace("-", "_")}
    names: list[str] = []
    for alias in sorted(aliases):
        names.extend((f"{alias}.hef", f"{alias}_h8.hef", f"{alias}_hailo8.hef"))
    return tuple(dict.fromkeys(names))


def _model_candidates(model: str, model_roots: Sequence[object] | None) -> list[Path]:
    project = Path(__file__).resolve().parent / "models" / "hailo8"
    roots = [Path("/usr/local/hailo/resources/models/hailo8")]
    roots.extend(Path(root) for root in (model_roots or ()))
    roots.extend((Path("/usr/local/hailo/resources"), Path("/usr/share/hailo-apps"), Path("/opt/hailo")))
    roots.append(project)
    filenames = _model_filenames(model)
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(root / name for name in filenames)
        if root.is_dir() and root not in (roots[0], project):
            for path in sorted(root.rglob("*.hef"), key=lambda item: str(item)):
                normalized = path.stem.lower().replace("-", "_")
                if model.lower().replace("-", "_") in normalized:
                    candidates.append(path)
    return list(dict.fromkeys(candidates))


def _load_labels(path: Path) -> tuple[object | None, Path | None]:
    for suffix in (".json", ".txt", ".yaml", ".yml"):
        sidecar = path.with_suffix(suffix)
        if not sidecar.is_file():
            continue
        text = sidecar.read_text(encoding="utf-8")
        if suffix == ".json":
            value = json.loads(text)
            if isinstance(value, Mapping) and "names" in value:
                value = value["names"]
            return value, sidecar
        lines = [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
        if suffix == ".txt":
            return lines, sidecar
        parsed: dict[int, str] = {}
        in_names = False
        for line in lines:
            if line.rstrip(":").strip() == "names":
                in_names = True
                continue
            match = re.match(r"(?:-\s*)?(\d+)\s*:\s*['\"]?([^'\"]+)['\"]?$", line)
            if match and (in_names or line[0].isdigit()):
                parsed[int(match.group(1))] = match.group(2).strip()
            elif in_names and line.startswith("-"):
                parsed[len(parsed)] = line[1:].strip().strip("'\"")
        if parsed:
            return parsed, sidecar
    return None, None


def _construct_runner(factory: Callable[..., object], path: Path, priority: int) -> object:
    optional = {
        "batch_size": 1,
        "output_type": "FLOAT32",
        "priority": priority,
        "group_id": "SHARED",
        "scheduling_group": "SHARED",
    }
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        kwargs = {"priority": priority}
    else:
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        kwargs = optional if accepts_kwargs else {
            name: value for name, value in optional.items() if name in signature.parameters
        }
    try:
        return factory(str(path), **kwargs)
    except Exception as exc:
        raise RuntimeError(f"failed to initialize HAILO8 inference for {path}: {exc}") from exc


def _input_hw(runner: object) -> tuple[int, int]:
    getter = getattr(runner, "get_input_shape", None)
    if not callable(getter):
        raise RuntimeError("Hailo inference helper does not expose get_input_shape()")
    shape = tuple(int(value) for value in getter())
    if len(shape) == 4 and shape[0] == 1:
        shape = shape[1:]
    if len(shape) != 3:
        raise RuntimeError(f"Hailo input shape must be HxWx3; got {shape}")
    if shape[-1] == 3:
        return shape[0], shape[1]
    if shape[0] == 3:
        return shape[1], shape[2]
    raise RuntimeError(f"Hailo input shape must have three RGB channels; got {shape}")


def _letterbox_frame(frame: np.ndarray, model_shape: tuple[int, int]) -> tuple[np.ndarray, LetterboxTransform]:
    transform = LetterboxTransform.create(frame.shape[:2], model_shape)
    model_h, model_w = model_shape
    resized_h = max(1, min(model_h, int(round(frame.shape[0] * transform.scale))))
    resized_w = max(1, min(model_w, int(round(frame.shape[1] * transform.scale))))
    ys = np.minimum((np.arange(resized_h) / transform.scale).astype(int), frame.shape[0] - 1)
    xs = np.minimum((np.arange(resized_w) / transform.scale).astype(int), frame.shape[1] - 1)
    resized = frame[ys[:, None], xs[None, :]]
    result = np.zeros((model_h, model_w, 3), dtype=np.uint8)
    top, left = (model_h - resized_h) // 2, (model_w - resized_w) // 2
    result[top:top + resized_h, left:left + resized_w] = resized
    return np.ascontiguousarray(result), transform


class HailoHybridFaceDetector:
    """Non-blocking two-model Hailo-8 frontend for HybridFaceFusion."""

    def __init__(
        self,
        *,
        scrfd_hef: object | None = None,
        personface_hef: object | None = None,
        model_roots: Sequence[object] | None = None,
        runner_factory: Callable[..., object] | None = None,
        clock: Callable[[], float] = time.monotonic,
        fusion: HybridFaceFusion | None = None,
        labels: object | None = None,
        yolo_idle_hz: float = 5.0,
        yolo_recovery_hz: float = 15.0,
    ) -> None:
        self.clock = clock
        self.scrfd_path = resolve_hef(scrfd_hef, _model_candidates("scrfd_10g", model_roots))
        yolo_candidates = _model_candidates("yolov8n_personface", model_roots)
        yolo_candidates.extend(_model_candidates("yolov8m", model_roots))
        self.personface_path = resolve_hef(
            personface_hef, yolo_candidates
        )
        try:
            sidecar_labels, label_path = _load_labels(self.personface_path)
        except Exception as exc:
            raise RuntimeError(
                f"failed to load YOLO label sidecar for model {self.personface_path}: {exc}"
            ) from exc
        normalized_yolo_stem = self.personface_path.stem.lower().replace("-", "_")
        official_coco_yolo = normalized_yolo_stem in (
            "yolov8m", "yolov8m_h8", "yolov8m_hailo8",
        )
        selected_labels = labels if labels is not None else sidecar_labels
        if selected_labels is None and official_coco_yolo:
            selected_labels = {0: "person"}
        if labels is not None:
            label_source = "explicit labels"
        elif label_path is not None:
            label_source = f"label sidecar {label_path}"
        elif official_coco_yolo:
            label_source = "built-in YOLOv8m COCO contract (class 0=person)"
        else:
            label_source = f"label sidecar not found next to {self.personface_path}"
        try:
            self.labels = (
                _required_yolo_labels(selected_labels)
                if official_coco_yolo
                else _required_personface_labels(selected_labels)
            )
        except RuntimeError as exc:
            required_contract = (
                "required a class mapped to person"
                if official_coco_yolo
                else "required class 0=person and class 1=face"
            )
            raise RuntimeError(
                f"YOLO class contract invalid for model {self.personface_path} using {label_source}: "
                f"{required_contract}. "
                f"{exc}"
            ) from exc
        factory = runner_factory or _load_hailo_runtime()
        runners: dict[str, object] = {}
        try:
            runners["scrfd"] = _construct_runner(factory, self.scrfd_path, 31)
            runners["yolo"] = _construct_runner(factory, self.personface_path, 20)
            self._runners = runners
            for name, runner in runners.items():
                getter = getattr(runner, "get_hef", None)
                if callable(getter):
                    validate_hef_arch(getter(), self.scrfd_path if name == "scrfd" else self.personface_path)
            self._input_shapes = {name: _input_hw(runner) for name, runner in runners.items()}
        except Exception as exc:
            cleanup_errors: list[str] = []
            for runner in runners.values():
                close = getattr(runner, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as cleanup_exc:
                        cleanup_errors.append(str(cleanup_exc))
            if cleanup_errors and hasattr(exc, "add_note"):
                exc.add_note("runner cleanup errors: " + "; ".join(cleanup_errors))
            raise
        self.fusion = fusion or HybridFaceFusion()
        self.yolo_idle_hz = float(yolo_idle_hz)
        self.yolo_recovery_hz = float(yolo_recovery_hz)
        if self.yolo_idle_hz <= 0 or self.yolo_recovery_hz <= 0:
            raise ValueError("YOLO scheduling frequencies must be positive")
        self.model_paths = {"scrfd": self.scrfd_path, "yolo": self.personface_path}
        self.startup_diagnostics = {
            "architecture": "HAILO8",
            "scrfd_path": str(self.scrfd_path),
            "personface_path": str(self.personface_path),
            "label_path": None if label_path is None else str(label_path),
            "priorities": {"scrfd": 31, "yolo": 20},
        }
        self._slots = {"scrfd": LatestFrameSlot(), "yolo": LatestFrameSlot()}
        self._state_lock = threading.Lock()
        self._runner_close_lock = threading.Lock()
        self._closed_runners: set[str] = set()
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
        self.shutdown_diagnostics = {
            "deferred_runner_closes": [],
            "hailo_helper_close_timeout_controlled": False,
        }
        self._threads = [
            threading.Thread(target=self._worker, args=(name,), name=f"hailo-{name}", daemon=True)
            for name in ("scrfd", "yolo")
        ]
        for thread in self._threads:
            thread.start()

    def _store_error(self, model: str, exc: BaseException) -> None:
        error = RuntimeError(f"{model.upper()} callback failed: {exc}")
        error.__cause__ = exc
        with self._state_lock:
            self.last_error = error
            if self._pending_error is None:
                self._pending_error = error

    def _publish_result(self, model: str, frame_id: int, value: object, timestamp: float) -> None:
        with self._state_lock:
            if frame_id <= self.completed_frame_ids[model]:
                return
            self.completed_frame_ids[model] = frame_id
            self.completion_counts[model] += 1
            if model == "scrfd":
                self.latest_scrfd = list(value)  # type: ignore[arg-type]
            else:
                faces, persons = value  # type: ignore[misc]
                self.latest_yolo_faces = list(faces)
                self.latest_persons = list(persons)

    def _binding_outputs(self, bindings: object, runner: object) -> object:
        batch = list(bindings) if isinstance(bindings, (list, tuple)) else [bindings]
        if not batch:
            return {}
        binding = batch[0]
        names: list[str] = []
        get_info = getattr(runner, "get_vstream_info", None)
        if callable(get_info):
            _, outputs = get_info()
            names = [str(getattr(info, "name", info)) for info in outputs]
        result: dict[str, object] = {}
        for name in names:
            output = binding.output(name)
            buffer = output.get_buffer() if hasattr(output, "get_buffer") else output
            result[name] = buffer
        if result:
            return result
        for name in ("outputs", "result"):
            value = getattr(binding, name, None)
            if value is not None:
                return value() if callable(value) else value
        return binding

    @staticmethod
    def _raise_completion_failure(completion_info: object) -> None:
        exception = getattr(completion_info, "exception", None)
        if callable(exception):
            exception = exception()
        if exception:
            if isinstance(exception, BaseException):
                raise RuntimeError(f"Hailo inference completion failed: {exception}") from exception
            raise RuntimeError(f"Hailo inference completion failed: {exception}")
        status = getattr(completion_info, "status", None)
        if callable(status):
            status = status()
        if status is None:
            return
        if isinstance(status, bool):
            succeeded = status
        elif isinstance(status, int):
            succeeded = status == 0
        else:
            normalized = str(status).strip().upper()
            succeeded = normalized in ("0", "OK", "SUCCESS", "HAILO_SUCCESS") or "SUCCESS" in normalized
        if not succeeded:
            raise RuntimeError(f"Hailo inference completion status: {status}")

    def _callback_payload(self, runner: object, args: tuple[object, ...], kwargs: dict[str, object]) -> object:
        if args and (
            hasattr(args[0], "exception") or hasattr(args[0], "status")
        ):
            self._raise_completion_failure(args[0])
        for name in ("outputs", "result", "results", "payload"):
            if name in kwargs:
                return kwargs[name]
        if "bindings_list" in kwargs:
            return self._binding_outputs(kwargs["bindings_list"], runner)
        if len(args) == 1:
            return args[0]
        if args:
            return args[-1]
        raise RuntimeError("Hailo callback returned no output payload")

    def _close_runner(self, model: str) -> None:
        with self._runner_close_lock:
            if model in self._closed_runners:
                return
            self._closed_runners.add(model)
        close = getattr(self._runners[model], "close", None)
        if callable(close):
            close()

    def _worker(self, model: str) -> None:
        try:
            self._worker_loop(model)
        finally:
            if self._close_event.is_set():
                self._close_runner(model)

    def _worker_loop(self, model: str) -> None:
        slot, runner = self._slots[model], self._runners[model]
        while not self._close_event.is_set():
            item = slot.get(timeout=0.1)
            if item is None:
                if slot.closed:
                    return
                continue
            frame_id, frame, timestamp = item
            try:
                prepared, transform = _letterbox_frame(frame, self._input_shapes[model])
            except Exception as exc:
                self._store_error(model, exc)
                continue
            completed = threading.Event()

            def callback(*args: object, **kwargs: object) -> None:
                try:
                    raw = self._callback_payload(runner, args, dict(kwargs))
                    if model == "scrfd":
                        decoded: object = decode_scrfd(raw, transform, timestamp)
                    else:
                        decoded = decode_personface_nms(raw, timestamp, transform, self.labels)
                    self._publish_result(model, frame_id, decoded, timestamp)
                except Exception as exc:
                    self._store_error(model, exc)
                finally:
                    completed.set()

            try:
                runner.run([prepared], callback)
            except Exception as exc:
                self._store_error(model, exc)
                completed.set()
            while not completed.wait(0.05):
                if self._close_event.is_set():
                    return

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
            raise RuntimeError("HailoHybridFaceDetector is closed")
        self._raise_pending_error()
        array = np.asarray(frame_rgb)
        if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
            raise ValueError("frame_rgb must be an HxWx3 uint8 RGB array")
        owned = np.array(array, dtype=np.uint8, order="C", copy=True)
        stamp = _finite_number(self.clock() if now is None else now, "now")
        with self._state_lock:
            scrfd = list(self.latest_scrfd)
            yolo_faces = list(self.latest_yolo_faces)
            persons = list(self.latest_persons)
            frame_id = self._next_frame_id
            self._next_frame_id += 1
        observation = self.fusion.update(
            scrfd, yolo_faces, persons, flow_box, flow_quality, owned.shape[:2], stamp
        )
        self._slots["scrfd"].put(frame_id, owned, stamp)
        healthy = observation.source == "SCRFD"
        rate = self.yolo_idle_hz if healthy else self.yolo_recovery_hz
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
        deferred: list[str] = []
        for model, thread in zip(("scrfd", "yolo"), self._threads):
            if thread.is_alive():
                deferred.append(model)
            else:
                self._close_runner(model)
        self.shutdown_diagnostics["deferred_runner_closes"] = deferred
