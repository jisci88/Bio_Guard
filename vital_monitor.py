"""
실시간 심박수 + 호흡수 + 얼굴 온도 측정 + 환자별 개인화 이상 탐지
+ 얼굴 팬/틸트 모터 트래킹 (Dynamixel, PID)

  심박수  : open-rppg의 BVP 주파수 분석
  호흡수  : rr.py의 Lucas-Kanade 옵티컬 플로우 기반 어깨 움직임 추적
  얼굴온도: MLX90640 열화상 센서, RGB 카메라의 얼굴 박스를 32x24
            열화상 격자로 비율 매핑해서 그 영역 온도를 추정
  이상탐지: Isolation Forest (환자별 기준 분포 학습) → vital_anomaly.py
            HR + RR + TEMP 세 축을 함께 학습한다
  얼굴추적: 얼굴 박스 중심이 화면 중앙에 오도록 팬(ID1)/틸트(ID2)
            다이나믹셀 서보를 PID로 구동 → face_tracker_motor.py
  알림    : 이상 판정 시 액티브 부저(GPIO) 논블로킹 비프

설치:
    pip install open-rppg opencv-python numpy scikit-learn adafruit-circuitpython-mlx90640 dynamixel-sdk gpiozero
    (호흡수 추적을 위해 동일 디렉터리에 rr.py 파일이 필요합니다)

실행:
    # 라즈베리파이 데스크톱
    python vital_monitor.py --camera 0

    # SSH 또는 VS Code Remote
    python vital_monitor.py --camera 0 --headless
"""

import argparse
import os
import threading
import time
from collections import deque

import cv2
import numpy as np
import rppg
import rr  # rr.py 모듈 임포트

from vital_anomaly import VitalAnomalyDetector
from face_tracker_motor import FaceTrackerMotor


def should_show_window(headless, environ=None):
    """X11 디스플레이가 있고 headless가 아닐 때만 창을 표시한다."""
    env = os.environ if environ is None else environ
    return not headless and bool(env.get("DISPLAY"))


# ══════════════════════════════════════════════════════
#  open-rppg 심박수 (호흡수 분리)
# ══════════════════════════════════════════════════════

class OpenRppgVitalTracker:
    """open-rppg로 심박수만 갱신하도록 축소된 트래커."""

    HR_MIN, HR_MAX = 40.0, 200.0
    HR_HOLD_SEC = 6.0

    def __init__(self, model, hr_window_sec=10):
        self.model = model
        self.hr_window_sec = hr_window_sec

        self.hr_bpm = 0.0
        self.hr_conf = 0.0
        self.hr_reason = "WARMING UP"

        # 마지막으로 유효했던 (값, 신뢰도, 시각)
        self._hr_last = None

        self.diag = {
            "hr_calls": 0, "hr_ok": 0, "hr_err": 0,
            "sqi_max": 0.0,
        }

    @staticmethod
    def _finite_float(value, default=0.0):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default

        return value if np.isfinite(value) else default

    @staticmethod
    def _hold(last, now, hold_sec):
        """마지막 유효값을 신뢰도 선형 감쇠로 유지한다. 만료면 None."""
        if last is None:
            return None

        value, conf, stamp = last
        age = now - stamp

        if age > hold_sec:
            return None

        return value, conf * max(0.0, 1.0 - age / hold_sec)

    def invalidate(self):
        """얼굴이 보이지 않을 때 기존 결과의 신뢰도를 무효화한다."""
        self.hr_conf = 0.0
        self.hr_reason = "NO FACE"

    def update_hr(self, face_visible):
        """심박수를 갱신한다."""
        now = time.time()

        if not face_visible:
            self.hr_conf = 0.0
            self.hr_reason = "NO FACE"
            return

        self.diag["hr_calls"] += 1

        try:
            result = self.model.hr(
                start=-self.hr_window_sec,
                return_hrv=False,
            )
        except Exception as exc:
            self.diag["hr_err"] += 1
            self.hr_conf = 0.0
            self.hr_reason = f"ERR {type(exc).__name__}"
            return

        result = result if isinstance(result, dict) else {}
        hr = self._finite_float(result.get("hr"))
        sqi = self._finite_float(result.get("SQI"))

        self.diag["sqi_max"] = max(self.diag["sqi_max"], sqi)

        if not self.HR_MIN <= hr <= self.HR_MAX:
            held = self._hold(self._hr_last, now, self.HR_HOLD_SEC)
            if held is not None:
                self.hr_bpm, self.hr_conf = held
                self.hr_reason = "HOLD"
            else:
                self.hr_conf = 0.0
                self.hr_reason = "WARMING UP"
            return

        self.hr_bpm = hr
        self.hr_conf = float(np.clip(sqi, 0.0, 1.0))
        self.diag["hr_ok"] += 1

        if self.hr_conf > 0.0:
            self._hr_last = (hr, self.hr_conf, now)

        self.hr_reason = "OK" if self.hr_conf >= 0.3 else f"SQI LOW {sqi:.2f}"


# ══════════════════════════════════════════════════════
#  MLX90640 열화상 - 얼굴 온도
# ══════════════════════════════════════════════════════

MLX_COLS = 32
MLX_ROWS = 24

MLX_FOV_SCALE_X = 1.0
MLX_FOV_SCALE_Y = 1.0
MLX_OFFSET_X = 0.0
MLX_OFFSET_Y = 0.0
MLX_STALE_SEC = 3.0


class ThermalFaceTracker:
    def __init__(self, i2c_frequency=400000, refresh_rate_hz=4):
        self.available = False
        self._frame = None
        self._frame_time = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

        try:
            import adafruit_mlx90640
            import board
            import busio

            i2c = busio.I2C(
                board.SCL,
                board.SDA,
                frequency=i2c_frequency,
            )
            self.mlx = adafruit_mlx90640.MLX90640(i2c)

            rate_map = {
                1: adafruit_mlx90640.RefreshRate.REFRESH_1_HZ,
                2: adafruit_mlx90640.RefreshRate.REFRESH_2_HZ,
                4: adafruit_mlx90640.RefreshRate.REFRESH_4_HZ,
                8: adafruit_mlx90640.RefreshRate.REFRESH_8_HZ,
            }
            self.mlx.refresh_rate = rate_map.get(
                refresh_rate_hz,
                adafruit_mlx90640.RefreshRate.REFRESH_4_HZ,
            )

            self.available = True

        except Exception as exc:
            print(
                "[WARN] MLX90640 초기화 실패, "
                f"얼굴 온도 측정을 비활성화합니다: {exc}"
            )
            return

        self._raw = [0.0] * (MLX_COLS * MLX_ROWS)
        self._running = True
        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
        )
        self._thread.start()

    def _loop(self):
        while self._running:
            try:
                self.mlx.getFrame(self._raw)
            except ValueError:
                continue
            except Exception as exc:
                print(f"[WARN] MLX90640 읽기 오류: {exc}")
                time.sleep(0.5)
                continue

            with self._lock:
                self._frame = list(self._raw)
                self._frame_time = time.time()

    def get_face_temperature(self, box, rgb_shape, stat="p90"):
        if not self.available or box is None:
            return None

        with self._lock:
            frame = self._frame
            frame_time = self._frame_time

        if frame is None:
            return None

        if time.time() - frame_time > MLX_STALE_SEC:
            return None

        h, w = rgb_shape[0], rgb_shape[1]
        (y1, y2), (x1, x2) = box

        cx = (
            ((x1 + x2) / 2.0 / w - 0.5) * MLX_FOV_SCALE_X
            + 0.5
            + MLX_OFFSET_X
        )
        cy = (
            ((y1 + y2) / 2.0 / h - 0.5) * MLX_FOV_SCALE_Y
            + 0.5
            + MLX_OFFSET_Y
        )
        half_w = (x2 - x1) / 2.0 / w * MLX_FOV_SCALE_X
        half_h = (y2 - y1) / 2.0 / h * MLX_FOV_SCALE_Y

        tx1 = int(np.clip((cx - half_w) * MLX_COLS, 0, MLX_COLS - 1))
        tx2 = int(np.clip((cx + half_w) * MLX_COLS, 0, MLX_COLS - 1))
        ty1 = int(np.clip((cy - half_h) * MLX_ROWS, 0, MLX_ROWS - 1))
        ty2 = int(np.clip((cy + half_h) * MLX_ROWS, 0, MLX_ROWS - 1))

        if tx2 <= tx1 or ty2 <= ty1:
            return None

        region = []
        for row in range(ty1, ty2 + 1):
            row_start = row * MLX_COLS
            region.extend(frame[row_start + tx1: row_start + tx2 + 1])

        if not region:
            return None

        region = np.asarray(region, dtype=float)

        if stat == "max":
            return float(region.max())

        k = max(1, int(len(region) * 0.1))
        return float(np.sort(region)[-k:].mean())

    def raw_frame(self):
        """Latest full thermal frame, for ambient estimation."""
        with self._lock:
            return None if self._frame is None else list(self._frame)

    def close(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)


# ══════════════════════════════════════════════════════
#  부저 알림
# ══════════════════════════════════════════════════════

class AlarmBuzzer:
    PATTERNS = {
        "critical": dict(on_time=0.12, off_time=0.12),
        "alert": dict(on_time=0.35, off_time=0.9),
    }

    def __init__(self, pin=17, enabled=True):
        self.buzzer = None
        self.level = None

        if not enabled:
            print("[INFO] 부저 알림을 사용하지 않습니다 (--no-buzzer).")
            return

        try:
            from gpiozero import Buzzer

            self.buzzer = Buzzer(pin)
            self.buzzer.off()
            print(f"[INFO] 부저 준비 완료 (GPIO{pin})")
        except Exception as exc:
            print(f"[WARN] 부저 초기화 실패, 소리 알림 없이 진행합니다: {exc}")
            self.buzzer = None

    def set_level(self, level):
        if self.buzzer is None or level == self.level:
            return

        self.level = level

        try:
            if level is None:
                self.buzzer.off()
            else:
                self.buzzer.beep(background=True, **self.PATTERNS[level])
        except Exception as exc:
            print(f"[WARN] 부저 제어 실패, 비활성화합니다: {exc}")
            self.buzzer = None

    def close(self):
        if self.buzzer is None:
            return
        try:
            self.buzzer.off()
            self.buzzer.close()
        except Exception:
            pass
        finally:
            self.buzzer = None


# ══════════════════════════════════════════════════════
#  HUD
# ══════════════════════════════════════════════════════

def draw_hud(frame, hr, rr, face_temp, fps, anomaly):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (355, 290), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    def block(label, value, conf, unit, y, hue):
        col = hue if conf > 0.3 else (130, 130, 130)
        cv2.putText(frame, label, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        text = f"{value:.1f} {unit}" if conf > 0.3 else "warming up..."
        cv2.putText(frame, text, (10, y + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.25, col, 2)
        cv2.rectangle(frame, (10, y + 44), (100, y + 52), (50, 50, 50), -1)
        cv2.rectangle(
            frame,
            (10, y + 44),
            (10 + int(min(conf, 1.0) * 90), y + 52),
            col,
            -1,
        )

    block("HEART RATE  (open-rppg)", hr[0], hr[1], "BPM", 26, (80, 220, 80))
    block("RESP RATE   (Optical Flow)", rr[0], rr[1], "BrPM", 108, (80, 180, 255))
    block("FACE TEMP   (MLX90640)", face_temp[0], face_temp[1], "C", 190, (255, 200, 80))

    if anomaly is not None:
        if anomaly["critical"]:
            col = (40, 40, 255)
            text = f"CRITICAL: {anomaly['critical']}"
        elif anomaly["state"] == "signal_lost":
            col = (0, 165, 255)
            text = "SIGNAL LOST"
        elif anomaly["alert"]:
            col = (60, 60, 255)
            text = anomaly["alert_reason"] or "ANOMALY"
        elif anomaly["baseline"] is None:
            col = (150, 150, 150)
            text = f"BASELINE {anomaly['progress'] * 100:.0f}%"
        elif anomaly["state"] in ("invalid", "warmup"):
            col = (0, 200, 255)
            text = "NO VALID SIGNAL - HOLD"
        else:
            col = (80, 220, 80)
            text = "NORMAL"

        cv2.putText(frame, text, (10, 277), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 1)

    cv2.putText(frame, f"FPS: {fps:.0f}", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    cv2.putText(
        frame,
        "HR:60-100  RR:12-20  Temp:36-37.5",
        (w - 250, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (120, 120, 120),
        1,
    )

    return frame


# ══════════════════════════════════════════════════════
#  메인 루프
# ══════════════════════════════════════════════════════

def run(
    camera_id=0,
    calib_sec=180.0,
    min_conf=0.30,
    out_pct=1.0,
    rr_sigma=4.0,
    temp_sigma=0.4,
    headless=False,
    use_thermal=True,
    thermal_stat="p90",
    thermal_freq=4,
    use_tracker=True,
    use_buzzer=True,
    buzzer_pin=17,
):
    print("[INFO] open-rppg 모델 로딩 중...")
    model = rppg.Model("ME-flow.rlap")
    print("[INFO] 모델 로딩 완료")

    show_window = should_show_window(headless)
    if headless:
        print("[INFO] 헤드리스 모드로 실행합니다.")
    elif not show_window:
        print("[WARN] DISPLAY가 없어 자동으로 헤드리스 모드로 실행합니다.")

    vitals = OpenRppgVitalTracker(model)

    thermal = None
    if use_thermal:
        print("[INFO] MLX90640 열화상 센서 초기화 중...")
        thermal = ThermalFaceTracker(refresh_rate_hz=thermal_freq)
        if thermal.available:
            print("[INFO] MLX90640 준비 완료")
    else:
        print("[INFO] 얼굴 온도 측정을 사용하지 않습니다 (--no-thermal).")

    use_temp = bool(thermal is not None and thermal.available)
    if use_temp:
        print("[INFO] 이상탐지: HR + RR + TEMP 3축으로 학습합니다.")
    else:
        print("[INFO] 이상탐지: HR + RR 2축으로 학습합니다 (체온 축 제외).")

    face_temp_c = 0.0
    face_temp_conf = 0.0

    detector = VitalAnomalyDetector(
        calib_sec=calib_sec,
        min_conf=min_conf,
        out_pct=out_pct,
        rr_sigma=rr_sigma,
        temp_sigma=temp_sigma,
        use_temp=use_temp,
    )

    anomaly = None

    tracker = None
    if use_tracker:
        print("[INFO] 얼굴 추적 모터(FaceTrackerMotor) 초기화 중...")
        tracker = FaceTrackerMotor()
        if tracker.enabled:
            print("[INFO] 얼굴 추적 모터 준비 완료")
        else:
            print("[WARN] 얼굴 추적 모터 초기화 실패, 추적 없이 진행합니다.")
    else:
        print("[INFO] 얼굴 추적 모터를 사용하지 않습니다 (--no-tracker).")

    buzzer = AlarmBuzzer(pin=buzzer_pin, enabled=use_buzzer)

    # 심박수 업데이트 변수
    hr_update_interval = 2.0
    last_hr_update = time.time()

    # 호흡수(rr.py) 초기화 상태 변수
    rr_tracker = None
    rr_reporter = rr.Reporter()
    buf_t, buf_d, buf_v = [], [], []
    last_rr_report = None
    current_rr_bpm = 0.0
    current_rr_conf = 0.0
    rr_reason = "WARMING UP"

    fps_timer = time.time()
    frame_count = 0
    measured_fps = 0.0

    print("=" * 62)
    print(f"  open-rppg 심박수 + 옵티컬 플로우 호흡수 + 얼굴 온도 측정 시작 (기준 학습 {calib_sec:.0f}초)")
    if show_window:
        print("  종료: Q / ESC 또는 Ctrl+C")
    else:
        print("  종료: Ctrl+C")
    print("=" * 62)

    try:
        with model.video_capture(camera_id):
            for frame_rgb, box in model.preview:
                now = time.time()
                face_visible = box is not None

                if tracker is not None:
                    tracker.update(box, frame_rgb.shape)

                frame_count += 1
                if now - fps_timer >= 2.0:
                    measured_fps = frame_count / (now - fps_timer)
                    frame_count = 0
                    fps_timer = now

                if not face_visible:
                    vitals.invalidate()

                if thermal is not None and face_visible:
                    temp = thermal.get_face_temperature(box, frame_rgb.shape, stat=thermal_stat)
                    if temp is not None:
                        face_temp_c = temp
                        face_temp_conf = 1.0
                    else:
                        face_temp_conf = 0.0
                else:
                    face_temp_conf = 0.0

                # ── 호흡수: 옵티컬 플로우 추적 (매 프레임) ──
                gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                if rr_tracker is None:
                    roi = rr.clip_roi(rr.default_roi(gray.shape), gray.shape)
                    rr_tracker = rr.KeyframeTracker(gray.shape, roi)

                if rr_tracker.needs_anchor(now):
                    rr_tracker.anchor(gray, now)

                out = rr_tracker.track(gray, now)
                if out is not None:
                    d, v = out
                    buf_t.append(now)
                    buf_d.append(d.copy())
                    buf_v.append(v)
                    while buf_t[-1] - buf_t[0] > rr.WIN_SEC * 1.2:
                        buf_t.pop(0)
                        buf_d.pop(0)
                        buf_v.pop(0)

                rr_updated = False
                if last_rr_report is None:
                    last_rr_report = now
                elif now - last_rr_report >= rr.HOP_SEC:
                    last_rr_report = now
                    if len(buf_t) > 0:
                        res = rr.estimate_rr(buf_t, np.array(buf_d), np.array(buf_v))
                        if res is None:
                            rr_reason = f"FILLING ({buf_t[-1]-buf_t[0]:.0f}/{rr.WIN_SEC:.0f}s)"
                            current_rr_conf = 0.0
                        elif not np.isfinite(res["rr"]):
                            rr_reason = "MOTION"
                            current_rr_conf = 0.0
                        else:
                            rep = rr_reporter.update(now, res["rr"], res["sqi"])
                            if rep is not None:
                                current_rr_bpm = rep[0]
                                current_rr_conf = res["sqi"]
                                rr_reason = f"OK (n={rep[1]})"
                            else:
                                rr_reason = "ACQUIRING"
                                current_rr_conf = 0.0
                    rr_updated = True
                # ────────────────────────────────────

                # ── 심박수 갱신 및 로그 출력 ──
                if now - last_hr_update >= hr_update_interval:
                    vitals.update_hr(face_visible)

                    anomaly = detector.push(
                        vitals.hr_bpm,
                        vitals.hr_conf,
                        current_rr_bpm,
                        current_rr_conf,
                        temp=face_temp_c,
                        temp_conf=face_temp_conf,
                        now=now,
                    )

                    if anomaly["critical"]:
                        buzzer.set_level("critical")
                    elif anomaly["alert"]:
                        buzzer.set_level("alert")
                    else:
                        buzzer.set_level(None)

                    rr_mark = "*" if rr_updated else " "
                    temp_text = f"{face_temp_c:5.1f}C" if face_temp_conf > 0.3 else "  -- "

                    print(
                        f"[HR] {vitals.hr_bpm:6.1f} ({vitals.hr_conf:.2f})"
                        f" {vitals.hr_reason:<14}"
                        f"  [RR{rr_mark}] {current_rr_bpm:5.1f} ({current_rr_conf:.2f})"
                        f" OF/{rr_reason:<14}"
                        f"  [TEMP] {temp_text} ({face_temp_conf:.2f})"
                        f"  [FPS] {measured_fps:4.1f}"
                    )

                    score = anomaly["score"]
                    threshold = anomaly["threshold"]

                    score_text = "score=-" if score is None else f"score={score:.3f}"
                    threshold_text = "" if threshold is None else f" / 임계 {threshold:.3f}"

                    stale = anomaly["state"] in ("invalid", "warmup")

                    if anomaly["critical"]:
                        status = f"** 절대범위 이탈: {anomaly['critical']}"
                    elif anomaly["state"] == "signal_lost":
                        status = "** 신호 소실 (측정 중단)"
                    elif anomaly["alert"]:
                        status = f"** 이상징후: {anomaly['alert_reason']}  {score_text}{threshold_text}"
                        if stale:
                            status += "  (현재 샘플 무효 - 알림 유지)"
                    elif anomaly["baseline"] is None:
                        status = (
                            f"기준 학습: 시간 {anomaly['progress_time'] * 100:3.0f}%"
                            f" / 샘플 {anomaly['accepted']}/{detector.min_samples}"
                            f"  {detector.stats()}"
                        )
                    elif stale:
                        status = f"판정 보류 - 유효 신호 없음 (원인={anomaly['reason'] or anomaly['state']})"
                    else:
                        baseline = anomaly["baseline"]
                        base_text = f"기준선 HR {baseline[0]:.0f} / RR {baseline[1]:.0f}"
                        if baseline[2] is not None:
                            base_text += f" / TEMP {baseline[2]:.1f}"
                        status = f"NORMAL  {score_text}{threshold_text}  {base_text}"

                    print(f"  {status}")
                    last_hr_update = now

                if show_window:
                    frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                    if box is not None:
                        (y1, y2), (x1, x2) = (box[0], box[1])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 220, 80), 2)

                    frame = draw_hud(
                        frame,
                        (vitals.hr_bpm, vitals.hr_conf),
                        (current_rr_bpm, current_rr_conf),
                        (face_temp_c, face_temp_conf),
                        measured_fps,
                        anomaly,
                    )

                    cv2.imshow("Vital Monitor  (HR + RR + Temp)", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        print("[INFO] 종료")
                        break

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C 입력으로 종료합니다.")
    finally:
        buzzer.close()
        if thermal is not None:
            thermal.close()
        if tracker is not None:
            tracker.close()
        if show_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="심박수 + 호흡수 + 얼굴 온도 실시간 측정")

    parser.add_argument("--camera", type=int, default=0, help="카메라 ID")
    parser.add_argument("--calib", type=float, default=180.0, help="기준 학습 초")
    parser.add_argument("--min-conf", type=float, default=0.30, help="SQI 게이팅 임계")
    parser.add_argument("--out-pct", type=float, default=1.0, help="학습 분포 하위 몇 %%를 이상 경계로 사용할지")
    parser.add_argument("--rr-sigma", type=float, default=4.0, help="RR 측정 불확실성(BPM)")
    parser.add_argument(
        "--temp-sigma",
        type=float,
        default=0.4,
        help="얼굴 온도 측정 불확실성(C).",
    )
    parser.add_argument("--headless", action="store_true", help="OpenCV 영상 창 없이 실행")
    parser.add_argument("--no-thermal", action="store_true", help="MLX90640 얼굴 온도 측정을 비활성화")
    parser.add_argument(
        "--thermal-stat",
        choices=["p90", "max"],
        default="p90",
        help="얼굴 영역 온도 대표값: 상위10%% 평균(p90) 또는 최고값(max)",
    )
    parser.add_argument(
        "--thermal-freq",
        type=int,
        default=4,
        choices=[1, 2, 4, 8],
        help="MLX90640 갱신 주파수(Hz)",
    )
    parser.add_argument(
        "--no-tracker",
        action="store_true",
        help="얼굴 팬/틸트 모터 추적을 비활성화",
    )
    parser.add_argument(
        "--no-buzzer",
        action="store_true",
        help="이상 알림 부저를 비활성화",
    )
    parser.add_argument(
        "--buzzer-pin",
        type=int,
        default=17,
        help="액티브 부저가 연결된 GPIO 번호(BCM)",
    )

    args, _ = parser.parse_known_args()

    run(
        camera_id=args.camera,
        calib_sec=args.calib,
        min_conf=args.min_conf,
        out_pct=args.out_pct,
        rr_sigma=args.rr_sigma,
        temp_sigma=args.temp_sigma,
        headless=args.headless,
        use_thermal=not args.no_thermal,
        thermal_stat=args.thermal_stat,
        thermal_freq=args.thermal_freq,
        use_tracker=not args.no_tracker,
        use_buzzer=not args.no_buzzer,
        buzzer_pin=args.buzzer_pin,
    )

# python3 vital_run.py --thermal-stat max --no-tracker