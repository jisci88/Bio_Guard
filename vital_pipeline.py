"""
vital_pipeline.py - 계측 파이프라인 단일 구현   (D-1)

예전에는 vital_monitor.run() 과 vital_run.VitalWorker.run() 이 같은
파이프라인을 각각 구현했고, 이미 다섯 군데가 갈라져 있었다:

                     vital_monitor        vital_run
  RR 앵커 실패       rr_ready 플래그      continue  <- A-1 버그
  motion-hold        있음                 없음
  Reporter SQI       0.14                 0.03
  체온               raw skin             SkinToCore(+2.0)  <- A-5 불일치
  detector RR conf   실측 SQI             0/1 이진

버그를 한쪽만 고치는 상황이 실제로 벌어지고 있었다. 이제 계측은 전부 여기서
하고, vital_monitor 는 CLI/HUD 렌더러, vital_run 은 Tk 렌더러로만 남는다.

    pipeline = VitalPipeline(model, ...)
    with model.video_capture(camera_id):
        for frame_rgb, box in model.preview:
            snap = pipeline.step(frame_rgb, box)
            ...  # 렌더링만
    pipeline.close()
"""

import csv
import functools
import inspect
import os
import time

import cv2
import numpy as np

import config
import rr
from config import get_logger
from csv_uploader import CsvUploader
from face_tracker_motor import FaceTrackerMotor
from cpu_face_detector import CpuHybridFaceDetector, FaceObservation, HybridFaceFusion
from vital_anomaly import VitalAnomalyDetector
from vital_sensors import (
    AlarmBuzzer, NeoPixelAlert, OpenRppgVitalTracker, ThermalFaceTracker,
)

log = get_logger("pipe")


# ══════════════════════════════════════════════════════
#  피부 -> 심부 체온 보정
# ══════════════════════════════════════════════════════

try:
    from temp_calib import SkinToCore

    HAVE_TEMP_CALIB = True
except ImportError:      # pragma: no cover - 파이에는 있고 개발 PC 에는 없다
    HAVE_TEMP_CALIB = False

    class SkinToCore:
        """
        temp_calib.py 가 없을 때 쓰는 대체 구현.

        vital_run.py 가 `from temp_calib import SkinToCore` 를 하는데 그 파일이
        없으면 import 단계에서 프로그램이 통째로 죽는다. 그래서 temp_calib 의
        계약과 기본값을 그대로 따르는 폴백을 둔다. ROI 가 얼굴에서 벗어난
        프레임은 '차가운 얼굴'이 아니라 '실패'이므로 보정하지 않고 기각한다.

            core = skin + offset + ambient_gain * (ambient_ref - ambient)

        ※ 이것은 '측정된' 보정이 아니다. 진짜 값은 기준 체온계와 짝지어
          temp_calib.py --fit 으로 구해야 한다.
        """

        OFFSET_C = 2.0
        AMBIENT_GAIN = 0.20
        AMBIENT_REF_C = 24.0
        MIN_SKIN_C = 30.0
        MEDIAN_N = 5
        MAX_STEP_C = 0.7

        def __init__(self, offset=OFFSET_C, ambient_gain=AMBIENT_GAIN,
                     ambient_ref=AMBIENT_REF_C, min_skin=MIN_SKIN_C,
                     median_n=MEDIAN_N, max_step_c=MAX_STEP_C):
            self.offset = offset
            self.ambient_gain = ambient_gain
            self.ambient_ref = ambient_ref
            self.min_skin = min_skin
            self.median_n = median_n
            self.max_step_c = max_step_c
            self.hist = []
            self.ambient = None
            self.rejected = 0

        @staticmethod
        def ambient_from_frame(frame):
            """열화상 프레임의 차가운 쪽 = 방 온도."""
            if frame is None:
                return None
            values = np.asarray(frame, dtype=float).reshape(-1)
            values = values[np.isfinite(values)]
            return float(np.percentile(values, 10)) if values.size else None

        def update(self, skin_c, frame=None):
            ambient = self.ambient_from_frame(frame)
            if ambient is not None:
                self.ambient = ambient

            if (skin_c is None or not np.isfinite(skin_c)
                    or skin_c < self.min_skin):
                self.rejected += 1
                return None, self.ambient

            core = skin_c + self.offset
            if self.ambient is not None:
                core += self.ambient_gain * (self.ambient_ref - self.ambient)

            if (self.hist and self.max_step_c is not None
                    and abs(core - float(np.median(self.hist))) > self.max_step_c):
                self.rejected += 1
                return None, self.ambient

            self.hist.append(core)
            del self.hist[:-self.median_n]
            return float(np.median(self.hist)), self.ambient


# ══════════════════════════════════════════════════════
#  파형 정규화  (C-5)
# ══════════════════════════════════════════════════════

def robust_normalize(values, max_points=None):
    """
    2~98 백분위 기준으로 0..1 스케일한 리스트를 돌려준다.

    C-5: min-max 정규화는 스파이크 하나가 파형 전체를 납작하게 만든다.
    백분위로 자르고 클립하면 형태가 보존된다.
    """
    if values is None:
        return None
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return None

    max_points = config.WAVE_POINTS if max_points is None else int(max_points)
    if x.size > max_points:
        idx = np.linspace(0, x.size - 1, max_points).astype(int)
        x = x[idx]

    lo, hi = (float(v) for v in np.percentile(x, (2.0, 98.0)))
    if hi - lo < 1e-9:
        lo, hi = float(x.min()), float(x.max())
        if hi - lo < 1e-9:
            return None
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0).tolist()


# ══════════════════════════════════════════════════════
#  파이프라인
# ══════════════════════════════════════════════════════

def _close_owned_face_detector_on_init_failure(initializer):
    """Roll back only the CPU detector lifecycle this pipeline owns."""
    @functools.wraps(initializer)
    def guarded(self, *args, **kwargs):
        try:
            return initializer(self, *args, **kwargs)
        except BaseException:
            detector = getattr(self, "face_detector", None)
            if getattr(self, "_face_detector_owned", False) and detector is not None:
                self._face_detector_owned = False
                close = getattr(detector, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as exc:
                        log.warning("CPU 얼굴 검출기 시작 롤백 실패: %s", exc)
            raise
    return guarded


class VitalPipeline:
    """프레임 하나를 받아 HR / RR / 체온 / 추적 / 이상탐지를 모두 갱신한다."""

    @_close_owned_face_detector_on_init_failure
    def __init__(
        self,
        model,
        use_thermal=True,
        thermal_stat=None,
        thermal_freq=None,
        use_tracker=True,
        dynamixel_port=None,
        pan_sign=None,
        tilt_sign=None,
        dxl_baud=None,
        require_frame_size=None,
        use_anomaly=True,
        calib_sec=None,
        min_conf=None,
        out_pct=None,
        rr_sigma=None,
        temp_sigma=None,
        temp_offset=None,
        temp_ambient_gain=None,
        temp_ambient_ref=None,
        temp_dist_gain=None,
        temp_face_px_ref=None,
        use_buzzer=False,
        buzzer_pin=17,
        rr_sqi_display=None,
        rr_sqi_learn=None,
        rr_raw=False,
        rr_motion_rate=None,
        rr_mute_sec=None,
        rr_roi_from_face=None,
        patient_id=None,
        use_baseline_store=True,
        session_csv=None,
        remote_host=None,
        remote_user=None,
        remote_key=None,
        remote_dir=None,
        remote_interval=None,
        dump=None,
        face_detector_mode=None,
        scrfd_onnx=None,
        yolo_onnx=None,
        face_detector=None,
    ):
        self.model = model
        self.face_detector_mode = (
            config.FACE_DETECTOR if face_detector_mode is None else face_detector_mode
        )
        if self.face_detector_mode not in ("cpu-hybrid", "rppg"):
            raise ValueError(
                "face_detector_mode must be 'cpu-hybrid' or 'rppg'"
            )

        # Construct CPU inference before any other pipeline resource, in particular before
        # the delayed motor initialization can ever enable torque.  An injected
        # detector is deliberately caller-owned; close() only closes an instance
        # this pipeline constructed itself.
        self.face_detector = face_detector
        self._face_detector_owned = False
        self.observation = FaceObservation(
            None, None, 0.0, 0.0, "NONE", 0.0, None, 0.0,
        )
        self._last_sensor_box = None
        self._last_sensor_stamp = -float("inf")
        if self.face_detector is None and self.face_detector_mode == "cpu-hybrid":
            try:
                fusion = HybridFaceFusion(
                    scrfd_confidence=config.CPU_SCRFD_CONF,
                    person_confidence=config.CPU_PERSON_CONF,
                    max_face_age_sec=config.CPU_RESULT_MAX_AGE,
                    flow_hold_sec=config.CPU_FLOW_HOLD_SEC,
                    person_hold_sec=config.CPU_PERSON_HOLD_SEC,
                    edge_margin=config.CPU_EDGE_MARGIN,
                )
                self.face_detector = CpuHybridFaceDetector(
                    scrfd_onnx=scrfd_onnx,
                    yolo_onnx=yolo_onnx,
                    fusion=fusion,
                    scrfd_input_size=config.CPU_SCRFD_INPUT_SIZE,
                    yolo_input_size=config.CPU_YOLO_INPUT_SIZE,
                    yolo_idle_hz=config.CPU_YOLO_IDLE_HZ,
                    yolo_recovery_hz=config.CPU_YOLO_RECOVERY_HZ,
                )
                self._face_detector_owned = True
            except Exception as exc:
                raise RuntimeError(
                    f"CPU hybrid detector startup failed: {exc}"
                ) from exc

        self.thermal_stat = config.THERMAL_STAT if thermal_stat is None else thermal_stat
        self.require_frame_size = (
            config.MOTOR_REQUIRE_FRAME_SIZE
            if require_frame_size is None else bool(require_frame_size)
        )
        self.rr_raw = bool(rr_raw)
        self.rr_sqi_display = (
            config.RR_SQI_DISPLAY if rr_sqi_display is None else rr_sqi_display
        )
        self.rr_sqi_learn = (
            config.RR_SQI_LEARN if rr_sqi_learn is None else rr_sqi_learn
        )
        self.rr_roi_from_face = (
            config.RR_ROI_FROM_FACE if rr_roi_from_face is None else rr_roi_from_face
        )
        self.rr_motion_rate = (
            rr.MOTION_RATE if rr_motion_rate is None else rr_motion_rate
        )
        self.rr_mute_sec = rr.MUTE_SEC if rr_mute_sec is None else rr_mute_sec
        self.patient_id = patient_id
        self.dump = dump

        # ── 심박수 ──
        self.vitals = OpenRppgVitalTracker(model)

        # ── 열화상 ──
        self.thermal = None
        if use_thermal:
            log.info("MLX90640 열화상 센서 초기화 중...")
            thermal = ThermalFaceTracker(refresh_rate_hz=thermal_freq)
            self.thermal = thermal if thermal.available else None
            if self.thermal is not None:
                log.info("MLX90640 준비 완료")
        else:
            log.info("얼굴 온도 측정을 사용하지 않습니다.")

        self.use_temp = self.thermal is not None
        if not HAVE_TEMP_CALIB and self.use_temp:
            log.warning(
                "temp_calib.py 가 없어 내장 폴백 보정을 씁니다. "
                "기준 체온계로 실측한 값이 아닙니다."
            )

        # temp_calib.SkinToCore 는 자체 기본값(ambient_gain=0.20,
        # ambient_ref=24.0)을 갖는다. None 을 '명시적으로' 넘기면 그 기본값을
        # 덮어써서 core += None * (...) 로 터진다. 값이 있을 때만 넘긴다.
        skin_kwargs = {}
        for key, given, fallback in (
            ("offset", temp_offset, config.TEMP_OFFSET_DEFAULT),
            ("ambient_gain", temp_ambient_gain, config.TEMP_AMBIENT_GAIN),
            ("ambient_ref", temp_ambient_ref, config.TEMP_AMBIENT_REF),
            ("dist_gain", temp_dist_gain, config.TEMP_DIST_GAIN),
            ("face_px_ref", temp_face_px_ref, config.TEMP_FACE_PX_REF),
            ("median_n", None, config.TEMP_FILTER_SAMPLES),
            ("max_step_c", None, config.TEMP_MAX_STEP_C),
        ):
            value = fallback if given is None else given
            if value is not None:
                skin_kwargs[key] = value
        try:
            self.skin2core = SkinToCore(**skin_kwargs)
        except TypeError:
            # 구버전 temp_calib.py 는 거리/안정화 항을 모른다. 나머지만 넘긴다.
            for key in ("dist_gain", "face_px_ref", "median_n", "max_step_c"):
                skin_kwargs.pop(key, None)
            self.skin2core = SkinToCore(**skin_kwargs)
            log.warning("temp_calib.py 가 거리 보정을 지원하지 않습니다 "
                        "(구버전). 거리 항 없이 진행합니다.")
        # update() 가 face_px 를 받는지 한 번만 확인해 둔다.
        self._skin2core_face_px = "face_px" in inspect.signature(
            self.skin2core.update).parameters
        log.info("체온 보정: %s  %s  거리항=%s",
                 "temp_calib.SkinToCore" if HAVE_TEMP_CALIB else "내장 폴백",
                 skin_kwargs or "(모듈 기본값)",
                 "ON" if (self._skin2core_face_px
                          and getattr(self.skin2core, "dist_gain", 0)) else "OFF")

        # ── 이상탐지 ──
        self.detector = None
        if use_anomaly:
            self.detector = VitalAnomalyDetector(
                calib_sec=calib_sec,
                min_conf=min_conf,
                out_pct=out_pct,
                rr_sigma=rr_sigma,
                temp_sigma=temp_sigma,
                use_temp=self.use_temp,
                # A-5: 파이프라인은 항상 SkinToCore 를 거친 값을 넣는다.
                # 어느 물리량인지 detector 에 명시해서 임계를 맞춘다.
                temp_scale=config.TEMP_SCALE,
                patient_id=patient_id,
            )
            log.info(
                "이상탐지 활성  축=%s  calib=%.0fs  min_samples=%d  temp_scale=%s",
                "HR+RR+TEMP" if self.use_temp else "HR+RR",
                self.detector.calib_sec, self.detector.min_samples,
                self.detector.temp_scale,
            )
            # C-8: 저장된 기준선이 있으면 180초 재학습을 건너뛴다.
            self._baseline_path = (
                VitalAnomalyDetector.baseline_path(patient_id)
                if (patient_id and use_baseline_store) else None
            )
            self._baseline_saved = False
            if self._baseline_path:
                self._baseline_saved = self.detector.load_baseline(
                    self._baseline_path
                )
        else:
            log.info("이상탐지 비활성. 기준 학습이 진행되지 않습니다.")
            self._baseline_path = None
            self._baseline_saved = False
        self.use_baseline_store = bool(use_baseline_store)

        self.buzzer = AlarmBuzzer(pin=buzzer_pin, enabled=use_buzzer)
        self.neopixel = NeoPixelAlert()

        # ── 얼굴 추적 모터 (첫 프레임 이후 지연 초기화) ──
        self.tracker = None
        self.tracker_fault = None
        self._tracker_pending = bool(use_tracker)
        self._tracker_opts = dict(
            device_name=dynamixel_port or config.DXL_PORT,
            pan_sign=pan_sign,
            tilt_sign=tilt_sign,
            baudrate=dxl_baud,
        )
        if not use_tracker:
            log.info("얼굴 추적 모터를 사용하지 않습니다.")

        # ── 상태 ──
        self.hr_bpm = 0.0
        self.hr_conf = 0.0
        self.hr_reason = "WARMING UP"

        self.rr_bpm = 0.0
        self.rr_conf = 0.0
        self.rr_sqi = 0.0
        self.rr_valid = False
        self.rr_fresh = False
        self.rr_learn_ok = False
        self.rr_state = "UNAVAILABLE"
        self.rr_source = "NONE"
        self.rr_quality = {}
        self.rr_reason = "WARMING UP"
        self.rr_updated = False
        self.resp_wave = None
        self._rr_face_present = None

        self.temp_c = 0.0
        self.temp_conf = 0.0
        self.temp_skin = None
        self.temp_ambient = None
        self.temp_face_px = None       # 충전율(거리) 대용값
        self.temp_sigma = None         # temp_calib 이 추정한 1시그마
        self.temp_reason = ""          # 기각 사유 (no_skin / cold_roi / too_far)
        self.temp_update_seq = 0       # JSON-safe actual thermal update marker

        self.anomaly = None
        self.motion_hold = False
        self.fps = 0.0
        self.frame_size = None

        # ── 내부 ──
        self._rr_estimator = None
        self._next_rr_flow = None
        self._last_rr_report = None

        self._last_hr = 0.0
        self._last_temp = 0.0
        self._last_motion = 0.0
        self._last_face_source = None

        # 첫 step() 에서 초기화한다. 생성 시각(time.time())으로 잡아 두면
        # 호출자가 다른 시계(테스트의 모의 시각, 영상 파일의 PTS 등)를 넘길 때
        # 경과시간이 음수가 되어 FPS 가 영영 갱신되지 않는다.
        self._fps_timer = None
        self._frame_count = 0
        self._fps_warned = False
        self._reported_shape = False

        self._log_t, self._log_d, self._log_v = [], [], []

        # ── C-9: 세션 CSV + 원격 전송 ──
        self._csv_file = None
        self._csv_writer = None
        self.session_csv_path = None
        self.uploader = None
        self._remote = dict(host=remote_host, user=remote_user,
                            key=remote_key, directory=remote_dir,
                            interval=remote_interval)
        if session_csv:
            self._open_session_csv(session_csv)

    # ------------------------------------------------------------ 세션 로그

    CSV_COLUMNS = (
        "iso_time", "t", "hr", "hr_conf", "hr_reason",
        "rr", "rr_conf", "rr_sqi", "rr_valid", "rr_learn_ok", "rr_state",
        "rr_source", "rr_reason",
        "temp_skin", "temp_core", "temp_ambient", "face_px", "temp_sigma",
        "temp_conf",
        "state", "score", "threshold", "alert", "alert_reason", "critical",
        "accepted", "motion_hold", "tracker_state", "fps",
    )

    def _open_session_csv(self, path):
        """
        C-9: HR/RR/TEMP/score/state 를 타임스탬프와 함께 남긴다.
        stdout 텍스트만 있으면 temp_offset, rr_sigma 같은 파라미터를 사후에
        튜닝할 근거가 남지 않는다.
        """
        if path is True or path == "auto":
            stamp = time.strftime("%Y%m%d_%H%M%S")
            name = f"session_{self.patient_id or 'unknown'}_{stamp}.csv"
            path = os.path.join(config.SESSION_LOG_DIR, name)
        try:
            parent = os.path.dirname(os.path.abspath(path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._csv_file = open(path, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(self.CSV_COLUMNS)
            self.session_csv_path = path
            log.info("세션 로그: %s", path)
        except Exception as exc:
            log.warning("세션 로그를 열 수 없습니다 (%s): %s", path, exc)
            self._csv_file = None
            self._csv_writer = None

        self._start_uploader(path)

    def _start_uploader(self, local_path):
        """측정 중 실시간으로 서버에 같은 행을 보낸다 (--remote).

        업로더는 별도 스레드이고 put() 은 논블로킹이다. 서버가 죽어 있어도
        계측 루프는 영향을 받지 않는다. 로컬 CSV 가 원본이므로 전송이
        실패해도 데이터는 남는다.
        """
        if not self._remote.get("host"):
            return
        directory = self._remote.get("directory") or config.REMOTE_DIR
        name = os.path.basename(local_path or "session.csv")
        try:
            self.uploader = CsvUploader(
                host=self._remote["host"],
                key_path=self._remote.get("key"),
                user=self._remote.get("user"),
                interval=self._remote.get("interval"),
                remote_path=f"{directory}/{name}",
                header=list(self.CSV_COLUMNS),
            )
            self.uploader.start()
            log.info("원격 전송 활성: %s -> %s/%s (%.0f초 주기)",
                     self._remote["host"], directory, name,
                     self.uploader.interval)
        except Exception as exc:
            log.warning("원격 전송을 시작하지 못했습니다: %s", exc)
            self.uploader = None

    def _write_csv_row(self, now):
        if self._csv_writer is None and self.uploader is None:
            return
        anomaly = self.anomaly or {}
        try:
            values = [
                time.strftime("%Y-%m-%d %H:%M:%S"),
                f"{now:.3f}",
                f"{self.hr_bpm:.1f}", f"{self.hr_conf:.3f}", self.hr_reason,
                f"{self.rr_bpm:.1f}", f"{self.rr_conf:.3f}",
                f"{self.rr_sqi:.4f}", int(self.rr_valid), int(self.rr_learn_ok),
                self.rr_state, self.rr_source,
                self.rr_reason,
                "" if self.temp_skin is None else f"{self.temp_skin:.2f}",
                f"{self.temp_c:.2f}",
                "" if self.temp_ambient is None else f"{self.temp_ambient:.2f}",
                "" if self.temp_face_px is None else f"{self.temp_face_px:.1f}",
                "" if self.temp_sigma is None else f"{self.temp_sigma:.3f}",
                f"{self.temp_conf:.2f}",
                anomaly.get("state", ""),
                "" if anomaly.get("score") is None else f"{anomaly['score']:.4f}",
                "" if anomaly.get("threshold") is None
                else f"{anomaly['threshold']:.4f}",
                int(bool(anomaly.get("alert"))),
                anomaly.get("alert_reason") or "",
                anomaly.get("critical") or "",
                anomaly.get("accepted", 0),
                int(self.motion_hold),
                self.tracker.tracking_state if self.tracker is not None else "OFF",
                f"{self.fps:.1f}",
            ]
        except Exception as exc:
            log.warning("세션 로그 행 구성 실패: %s", exc)
            return

        # 원격 전송이 먼저다. put() 은 논블로킹이라 로컬 쓰기를 지연시키지 않는다.
        if self.uploader is not None:
            self.uploader.put(values)

        if self._csv_writer is None:
            return
        try:
            self._csv_writer.writerow(values)
            self._csv_file.flush()
        except Exception as exc:
            log.warning("세션 로그 기록 실패 -> 이후 기록을 중단합니다: %s", exc)
            self._csv_writer = None

    # ------------------------------------------------------------- 메인 스텝

    def step(self, frame_rgb, box=None, now=None):
        """프레임 1장을 처리하고 상태 스냅샷을 돌려준다."""
        now = time.time() if now is None else float(now)
        if self.face_detector_mode == "cpu-hybrid":
            # Preview's legacy rPPG rectangle must never become evidence in
            # hybrid mode.  Detector errors are operational failures, not a
            # reason to silently downgrade to rPPG.
            flow = None
            if self.tracker is not None:
                latest_flow = getattr(self.tracker, "latest_flow", None)
                if callable(latest_flow):
                    flow = latest_flow()
            flow_box, flow_quality = (None, 0.0) if flow is None else flow
            observation = self.face_detector.submit(
                frame_rgb, now=now, flow_box=flow_box, flow_quality=flow_quality,
            )
        else:
            observation = FaceObservation(
                box, box, 1.0 if box is not None else 0.0,
                1.0 if box is not None else 0.0, "RPPG", now, None, 0.0,
            )
        self.observation = observation
        previous_source = getattr(self, "_last_face_source", None)
        if previous_source != observation.source:
            log.info(
                "FACE source=%s conf=%.2f age=%.3f track=%s",
                observation.source, observation.motor_confidence,
                observation.result_age, observation.track_id,
            )
            if observation.source == "FLOW":
                log.info("FACE FLOW acquired")
            elif previous_source == "FLOW":
                log.info("FACE FLOW lost")
            self._last_face_source = observation.source
        sensor_box = self._measurement_sensor_box(observation.sensor_box, now)
        motor_box = observation.motor_box
        face_visible = sensor_box is not None

        if not self._reported_shape:
            self._report_frame_shape(frame_rgb)

        # 회색조는 한 번만 만들어 RR 과 얼굴 추적이 공유한다.
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

        if self._tracker_pending:
            # 모터 초기화에 걸린 실제 시간만큼만 now 를 민다. time.time() 으로
            # 덮어쓰면 호출자가 넘긴 시계와 뒤섞인다. (B-7 로 이 구간은 예전
            # 2초에서 보통 0.1~0.3초로 줄었다.)
            init_start = time.monotonic()
            self._init_tracker()
            now += time.monotonic() - init_start
            self._fps_timer, self._frame_count = now, 0

        self._update_tracker(
            motor_box, frame_rgb, gray, now,
            face_confidence=observation.motor_confidence,
            source=observation.source,
            evidence_age=observation.result_age,
            track_id=observation.track_id,
        )
        self._update_fps(now)

        if not face_visible:
            self.vitals.invalidate()

        self._update_rr(gray, sensor_box, now)

        if now - self._last_temp >= config.THERMAL_UPDATE_SEC:
            self._last_temp = now
            self._update_temp(sensor_box, frame_rgb.shape)
            self.temp_update_seq = getattr(self, "temp_update_seq", 0) + 1

        if now - self._last_hr >= config.HR_UPDATE_SEC:
            self._last_hr = now
            self._update_hr_and_anomaly(face_visible, now)
            self._write_csv_row(now)

        return self.snapshot()

    def _measurement_sensor_box(self, sensor_box, now):
        """Bridge one asynchronous detector gap with verified face geometry.

        The cached box is only used by HR/RR/thermal measurement.  Motor
        control still consumes the current observation's motor box and source.
        """
        if sensor_box is not None:
            self._last_sensor_box = sensor_box
            self._last_sensor_stamp = float(now)
            return sensor_box
        last_box = getattr(self, "_last_sensor_box", None)
        last_stamp = getattr(self, "_last_sensor_stamp", -float("inf"))
        if (last_box is not None
                and float(now) - last_stamp
                <= config.MEASUREMENT_ROI_HOLD_SEC):
            return last_box
        return None

    def _report_frame_shape(self, frame_rgb):
        self._reported_shape = True
        size = (frame_rgb.shape[1], frame_rgb.shape[0])
        self.frame_size = size
        log.info("카메라 실제 입력: %dx%d", size[0], size[1])
        if self.require_frame_size and size != tuple(config.EXPECTED_FRAME_SIZE):
            log.warning(
                "출전 튜닝 기준은 %dx%d 입니다. 추적 모터를 초기화하지 않습니다.",
                config.EXPECTED_FRAME_SIZE[0], config.EXPECTED_FRAME_SIZE[1],
            )
            self._tracker_pending = False

    # ------------------------------------------------------------- 얼굴 추적

    def _init_tracker(self):
        self._tracker_pending = False
        log.info("얼굴 추적 모터(FaceTrackerMotor) 초기화 중...")
        # 첫 프레임이 확보된 뒤에만 토크를 켠다. 카메라 시작 지연 동안에는
        # 모터가 무전원 상태다.
        try:
            tracker = FaceTrackerMotor(
                require_optical_flow=config.MOTOR_REQUIRE_FLOW,
                enable_bus_watchdog=False,
                raise_on_fault=True,
                **self._tracker_opts,
            )
        except Exception as exc:
            log.error("얼굴 추적 모터 생성 실패 -> 계측만 계속합니다: %s", exc)
            self.tracker_fault = str(exc)
            return

        if tracker.enabled:
            self.tracker = tracker
            log.info("얼굴 추적 모터 준비 완료")
            return

        # 초기화 실패. 사유를 남겨 대시보드에서 보여줄 수 있게 한다 (A-7).
        self.tracker_fault = tracker.tracking_reason or "INIT_FAILED"
        if tracker.torque_off_confirmed is False:
            log.critical("모터 토크 OFF 를 확인하지 못했습니다. %s", tracker.status())
            log.critical("사람이 회전 범위에서 벗어난 뒤 물리 전원을 차단하세요.")
        elif self.tracker_fault == "FIXED_SIGN_REQUIRED":
            log.warning(
                "--pan-sign / --tilt-sign 이 지정되지 않아 추적을 끕니다. "
                "face_tracker_motor.py --selftest 로 방향을 확인하세요."
            )
        else:
            log.warning("얼굴 추적 모터 초기화 실패 (%s) - 모터는 움직이지 않습니다.",
                        self.tracker_fault)

    def _update_tracker(
        self, box, frame_rgb, gray, now, *, face_confidence=None,
        source=None, evidence_age=None, track_id=None,
    ):
        """Forward the motor ROI and its source-qualified evidence."""
        if self.tracker is None:
            self.motion_hold = False
            return
        try:
            self.tracker.update(
                box, frame_rgb.shape, frame_rgb=frame_rgb, frame_gray=gray,
                external_confidence=(
                    1.0 if face_confidence is None else face_confidence
                ),
                external_source="RPPG" if source is None else source,
                evidence_age=0.0 if evidence_age is None else evidence_age,
                external_track_id=track_id,
            )
        except Exception as exc:
            self.tracker_fault = str(exc)
            self.motion_hold = False
            if getattr(self.tracker, "raise_on_fault", False):
                log.exception("얼굴 추적 모터 런타임 오류 -> 계측을 중단하고 모터를 종료합니다")
                raise
            # Non-fatal software-only tracker implementations retain the legacy
            # measurement-only fallback. Hardware trackers use raise_on_fault.
            log.error("얼굴 추적 모터 오류 -> 계측만 계속합니다: %s", exc)
            return

        if self.tracker.is_moving():
            self._last_motion = now
        # C-2 / A-6: 이동 중과 직후는 HR 도 RR 도 믿지 않는다.
        self.motion_hold = (now - self._last_motion) < config.MOTION_HOLD_SEC

    def _update_fps(self, now):
        if self._fps_timer is None:
            self._fps_timer = now
        self._frame_count += 1
        elapsed = now - self._fps_timer
        if elapsed < config.FPS_REPORT_SEC:
            return
        self.fps = self._frame_count / elapsed
        self._frame_count = 0
        self._fps_timer = now

        lo, hi = config.FPS_OK_RANGE
        out_of_range = not lo <= self.fps <= hi
        if out_of_range and not self._fps_warned:
            log.warning(
                "전체 앱 처리 속도 %.1f FPS; 카메라 설정과 별개의 값이며 "
                "추적은 계속합니다.", self.fps,
            )
        self._fps_warned = out_of_range

    # ------------------------------------------------------------------ RR

    def _update_rr(self, gray, box, now):
        """Feed adaptive RR at its own cadence without delaying per-frame work."""
        if getattr(self, "_rr_estimator", None) is None:
            self._rr_estimator = rr.AdaptiveRespirationEstimator(
                gray.shape, motion_rate=self.rr_motion_rate,
                mute_sec=self.rr_mute_sec,
            )
        if self._next_rr_flow is None:
            self._next_rr_flow = now
        if now >= self._next_rr_flow:
            self._rr_estimator.update_frame(
                gray, face_box=box,
                person_box=getattr(self.observation, "person_box", None),
                now=now, motion_hold=self.motion_hold,
            )
            while self._next_rr_flow <= now:
                self._next_rr_flow += self._rr_estimator.flow_period

        self.rr_updated = False
        if self._last_rr_report is None:
            self._last_rr_report = now
        elif now - self._last_rr_report >= rr.HOP_SEC:
            self._last_rr_report = now
            self.rr_updated = True
            result = self._rr_estimator.report(now, motion_hold=self.motion_hold)
            self.rr_state = "HOLD" if (self.motion_hold and result.state == "LOCKED") else result.state
            self.rr_source = result.source
            self.rr_reason = result.reason
            self.rr_conf = result.confidence
            self.rr_sqi = result.confidence
            self.rr_fresh = result.fresh and not self.motion_hold
            self.rr_learn_ok = result.learn_valid and not self.motion_hold
            self.rr_valid = result.state in ("LOCKED", "HOLD")
            self.rr_quality = self._rr_estimator.diagnostics()
            acquisition = self.rr_quality.get("acquisition", {})
            active_seconds = float(acquisition.get("active_seconds", 0.0))
            target_seconds = float(acquisition.get("target_seconds", 0.0))
            if result.state == "UNAVAILABLE" and result.reason == "NO_CANDIDATE":
                if target_seconds > 0.0 and active_seconds < target_seconds:
                    self.rr_state = "FILLING"
                    self.rr_reason = (
                        f"FILLING {active_seconds:.0f}/{target_seconds:.0f}s"
                    )
                else:
                    self.rr_state = "SEARCHING"
                    self.rr_reason = "SEARCHING FOR PERIODIC BREATHING"
            if result.state in ("ACQUIRING", "LOCKED", "HOLD"):
                self.rr_bpm = result.rate_bpm
            if result.wave is not None and result.state == "LOCKED":
                self.resp_wave = robust_normalize(result.wave)

    # ---------------------------------------------------------------- 체온

    def _update_temp(self, box, rgb_shape):
        skin = None
        face_px = None
        if self.thermal is not None and box is not None:
            measured = self.thermal.measure_face(
                box, rgb_shape, stat=self.thermal_stat,
            )
            if measured is not None:
                skin, face_px, _sample_px = measured

        # A-5: 두 실행 경로가 항상 같은 물리량(심부 추정치)을 쓰도록 여기서만
        # 보정한다. detector 에는 temp_scale 로 어느 쪽인지 알려 두었다.
        # face_px 는 충전율이고, 거리에 따른 과소측정을 되돌리는 데 쓰인다.
        frame = self.thermal.raw_frame() if self.thermal is not None else None
        if self._skin2core_face_px:
            core, ambient = self.skin2core.update(skin, frame, face_px=face_px)
        else:
            core, ambient = self.skin2core.update(skin, frame)

        self.temp_skin = skin
        self.temp_ambient = ambient
        self.temp_face_px = face_px

        diag = getattr(self.skin2core, "last", None) or {}
        self.temp_sigma = diag.get("sigma")
        self.temp_reason = diag.get("reason") or ""

        if core is not None:
            self.temp_c = core
            self.temp_conf = 1.0
        else:
            self.temp_conf = 0.0

        if self.thermal is not None:
            summary = getattr(self.skin2core, "summary", None)
            if callable(summary):
                log.debug("%s  stat=%s rejected=%d",
                          summary(), self.thermal_stat, self.skin2core.rejected)
            else:
                log.debug(
                    "TEMP skin=%s amb=%s px=%s -> core=%s  stat=%s rejected=%d",
                    "  --  " if skin is None else f"{skin:6.2f}",
                    "  --  " if ambient is None else f"{ambient:6.2f}",
                    "--" if face_px is None else f"{face_px:.0f}",
                    "REJECT" if core is None else f"{core:6.2f}",
                    self.thermal_stat, self.skin2core.rejected,
                )

    # ----------------------------------------------------- 심박수 + 이상탐지

    def _update_hr_and_anomaly(self, face_visible, now):
        self.vitals.update_hr(face_visible)
        self.hr_bpm = self.vitals.hr_bpm
        self.hr_conf = self.vitals.hr_conf
        self.hr_reason = self.vitals.hr_reason

        estimator = getattr(self, "_rr_estimator", None)
        if estimator is not None and not self.motion_hold:
            try:
                bvp, stamps = self.model.bvp(start=-rr.WIN_SEC * 1.2)
            except Exception as exc:
                # BVP is optional evidence.  A model/window failure must not
                # erase a useful optical-flow candidate.
                log.debug("RR BVP unavailable: %s", exc)
            else:
                estimator.update_bvp(
                    stamps, bvp,
                    confidence=self.hr_conf if face_visible else 0.0,
                    now=now,
                )

        # C-2 / A-6: 모터 이동 중/직후 샘플은 HR 도 RR 도 무효다. 예전 주석은
        # "RR 은 rr.py 가 처리하므로 건드리지 않는다" 라고 써 놓고 코드는
        # 건드리고 있었다. 이제 둘 다 명시적으로 막는다.
        if self.motion_hold:
            self.hr_conf = 0.0
            self.hr_reason = "MOTOR MOVING"

        if self.detector is None:
            if not hasattr(self, "rr_fresh"):
                # Compatibility for focused legacy harnesses constructed with
                # __new__; real pipeline state is mapped by _update_rr.
                self.rr_learn_ok = bool(
                    self.rr_valid and self.rr_sqi >= self.rr_sqi_learn
                    and self.rr_conf >= config.MIN_CONF and not self.motion_hold
                )
            else:
                self.rr_learn_ok = bool(self.rr_learn_ok and not self.motion_hold)
            return

        # RR 숫자가 화면에 유효하게 표시되면 동일한 SQI 문턱을 학습에도 쓴다.
        # 실제 신뢰도를 그대로 넘겨 detector.min_conf 가 최종 검증하게 한다.
        # 과거의 별도 SQI 0.14 문턱 + 강제 1.0 조합은 측정값이 보여도 샘플을
        # 영원히 거절하거나, 반대로 경계 신호를 완전 신뢰로 위장했다.
        if not hasattr(self, "rr_fresh"):
            self.rr_learn_ok = bool(
                self.rr_valid and self.rr_sqi >= self.rr_sqi_learn
                and self.rr_conf >= self.detector.min_conf and not self.motion_hold
            )
        else:
            self.rr_learn_ok = bool(self.rr_learn_ok and not self.motion_hold)
        self.anomaly = self.detector.push(
            self.hr_bpm, self.hr_conf,
            self.rr_bpm, self.rr_conf,
            temp=self.temp_c, temp_conf=self.temp_conf,
            now=now,
            sample_fresh=(
                getattr(self, "rr_fresh", self.rr_learn_ok)
                and self.rr_learn_ok
            ),
        )

        neopixel = getattr(self, "neopixel", None)
        if self.anomaly["critical"]:
            self.buzzer.set_level("critical")
            if neopixel is not None:
                neopixel.set_alert(True)
        elif self.anomaly["alert"]:
            self.buzzer.set_level("alert")
            if neopixel is not None:
                neopixel.set_alert(True)
        else:
            self.buzzer.set_level(None)
            if neopixel is not None:
                neopixel.set_alert(False)

        # C-8: 학습이 끝나는 순간 기준선을 저장한다. 다음 실행부터 즉시 시작.
        if (self._baseline_path and not self._baseline_saved
                and self.detector.model is not None):
            self._baseline_saved = self.detector.save_baseline(self._baseline_path)

        if self.anomaly["baseline"] is None:
            log.info(
                "AN accepted=%d/%d state=%s reason=%s crit=%s  "
                "hr=%5.1f(%.2f) rr=%5.1f(%.2f) temp=%5.1f(%.2f)  %s",
                self.anomaly["accepted"], self.detector.min_samples,
                self.anomaly["state"], self.anomaly["reason"],
                self.anomaly["critical"],
                self.hr_bpm, self.hr_conf,
                self.rr_bpm, self.rr_conf,
                self.temp_c, self.temp_conf, self.detector.stats(),
            )

    # ------------------------------------------------------------- 스냅샷

    def ppg_wave(self):
        """PPG 파형 (C-5 로버스트 정규화)."""
        try:
            bvp, _ts = self.model.bvp(start=-6.0)
        except Exception:
            return None
        return robust_normalize(bvp)

    def tracker_info(self):
        if self.tracker is not None:
            return dict(
                active=True,
                state=self.tracker.tracking_state,
                confidence=self.tracker.tracking_confidence,
                reason=self.tracker.tracking_reason,
                status=self.tracker.status(),
            )
        # A-7: 왜 꺼져 있는지 렌더러가 보여줄 수 있게 사유를 함께 준다.
        return dict(
            active=False,
            state="OFF",
            confidence=0.0,
            reason=self.tracker_fault or "",
            status="",
        )

    def upload_info(self):
        if self.uploader is None:
            return {"on": False, "connected": False, "sent": 0,
                    "queue": 0, "status": ""}
        with self.uploader._lock:
            queued = len(self.uploader._buffer)
        return {
            "on": True,
            "connected": self.uploader.connected,
            "sent": self.uploader.sent,
            "queue": queued,
            "status": self.uploader.status(),
        }

    def snapshot(self):
        observation = self.observation
        def json_safe(value):
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, dict):
                return {
                    str(key): json_safe(item)
                    for key, item in value.items()
                    if not isinstance(item, np.ndarray)
                }
            if isinstance(value, (tuple, list)):
                return [json_safe(item) for item in value]
            return value

        quality = {}
        for name, value in getattr(self, "rr_quality", {}).items():
            if isinstance(value, np.ndarray):
                continue
            quality[name] = json_safe(value)
        return {
            "hr": self.hr_bpm,
            "hr_conf": self.hr_conf,
            "hr_reason": self.hr_reason,
            "rr": self.rr_bpm,
            "rr_conf": self.rr_conf,
            "rr_sqi": self.rr_sqi,
            "rr_valid": self.rr_valid,
            "rr_fresh": getattr(self, "rr_fresh", False),
            "rr_learn_ok": self.rr_learn_ok,
            "rr_state": getattr(self, "rr_state", "UNAVAILABLE"),
            "rr_source": getattr(self, "rr_source", "NONE"),
            "rr_reason": self.rr_reason,
            "rr_quality": quality,
            "rr_updated": self.rr_updated,
            "resp": self.resp_wave,
            "temp": self.temp_c,
            "temp_conf": self.temp_conf,
            "temp_skin": self.temp_skin,
            "temp_ambient": self.temp_ambient,
            "temp_face_px": self.temp_face_px,
            "temp_sigma": self.temp_sigma,
            "temp_reason": self.temp_reason,
            "temp_update_seq": getattr(self, "temp_update_seq", 0),
            "anomaly": self.anomaly,
            "use_temp": self.use_temp,
            "motion_hold": self.motion_hold,
            "fps": self.fps,
            "tracker": self.tracker_info(),
            "face_detector": {
                "mode": self.face_detector_mode,
                "source": observation.source,
                "face_conf": observation.face_confidence,
                "motor_conf": observation.motor_confidence,
                "age_ms": observation.result_age * 1000.0,
                "track_id": observation.track_id,
            },
            "upload": self.upload_info(),
            "min_samples": self.detector.min_samples if self.detector else 0,
        }

    # ---------------------------------------------------------------- 종료

    def close(self):
        if self.dump:
            try:
                payload = {
                    "t": np.array(self._log_t),
                    "disp": np.array(self._log_d),
                    "valid": np.array(self._log_v),
                }
                estimator = getattr(self, "_rr_estimator", None)
                if estimator is not None:
                    payload.update(estimator.dump_arrays())
                np.savez(self.dump, **payload)
                log.info("호흡 원신호 저장: %s (%d 프레임)",
                         self.dump, len(self._log_t))
            except Exception as exc:
                log.warning("원신호 저장 실패: %s", exc)

        # 학습이 끝났는데 아직 저장 못 한 기준선이 있으면 여기서 저장한다.
        if (self._baseline_path and not self._baseline_saved
                and self.detector is not None and self.detector.model is not None):
            self.detector.save_baseline(self._baseline_path)

        if self.uploader is not None:
            self.uploader.stop()
            self.uploader = None

        if self._csv_file is not None:
            try:
                self._csv_file.close()
            except Exception:
                pass
            self._csv_file = None
            self._csv_writer = None

        if self.tracker is not None:
            try:
                self.tracker.close()
            except Exception as exc:
                log.warning("얼굴 추적 모터 종료 실패: %s", exc)
            self.tracker = None
        # The detector is stopped after frame processing and before thermal
        # teardown.  Injected detectors remain caller-owned (see __init__).
        if self._face_detector_owned and self.face_detector is not None:
            try:
                self.face_detector.close()
            except Exception as exc:
                log.warning("CPU 얼굴 검출기 종료 실패: %s", exc)
            self.face_detector = None
        self.buzzer.close()
        neopixel = getattr(self, "neopixel", None)
        if neopixel is not None:
            neopixel.close()
        if self.thermal is not None:
            try:
                self.thermal.close()
            except Exception as exc:
                log.warning("열화상 센서 종료 실패: %s", exc)
