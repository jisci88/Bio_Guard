"""
실시간 심박수 + 호흡수 측정 + 환자별 개인화 이상 탐지

  심박수  : open-rppg의 BVP 주파수 분석
  호흡수  : open-rppg의 PRV/HRV 기반 breathingrate 추정
  이상탐지: Isolation Forest (환자별 기준 분포 학습) → vital_anomaly.py

주의:
  open-rppg의 hrv["breathingrate"] 값은 Hz 단위이므로 60을 곱해
  분당 호흡수(BrPM)로 변환한다. 호흡수는 충분한 BVP 구간과 높은 SQI가
  필요하므로 심박수보다 준비 시간이 길 수 있다.

  rppg.Model("ME-flow.rlap")은 pickle로 저장할 수 없다.

설치:
    pip install open-rppg opencv-python numpy scikit-learn

실행:
    # 라즈베리파이 데스크톱
    python vital_monitor.py --camera 0

    # SSH 또는 VS Code Remote
    python vital_monitor.py --camera 0 --headless
"""

import argparse
import os
import time

import cv2
import numpy as np
import rppg

from vital_anomaly import VitalAnomalyDetector


def should_show_window(headless, environ=None):
    """X11 디스플레이가 있고 headless가 아닐 때만 창을 표시한다."""
    env = os.environ if environ is None else environ
    return not headless and bool(env.get("DISPLAY"))


# ══════════════════════════════════════════════════════
#  open-rppg 심박수 + 호흡수
# ══════════════════════════════════════════════════════

class OpenRppgVitalTracker:
    """open-rppg 하나로 심박수와 호흡수를 갱신한다.

    심박수는 짧은 BVP 창에서 계산하고, 호흡수는 더 긴 창의 PRV/HRV에서
    추정된 ``breathingrate``를 사용한다. open-rppg가 반환하는 SQI를 각
    측정값의 신뢰도로 사용한다.
    """

    HR_MIN, HR_MAX = 40.0, 200.0
    RR_MIN, RR_MAX = 4.0, 40.0

    def __init__(self, model, hr_window_sec=10, rr_window_sec=60):
        self.model = model
        self.hr_window_sec = hr_window_sec
        self.rr_window_sec = rr_window_sec

        self.hr_bpm = 0.0
        self.hr_conf = 0.0
        self.rr_bpm = 0.0
        self.rr_conf = 0.0

    @staticmethod
    def _finite_float(value, default=0.0):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return default

        return value if np.isfinite(value) else default

    def invalidate(self):
        """얼굴이 보이지 않을 때 기존 결과의 신뢰도를 무효화한다."""
        self.hr_conf = 0.0
        self.rr_conf = 0.0

    def update_hr(self, face_visible):
        """심박수를 갱신한다."""
        if not face_visible:
            self.hr_conf = 0.0
            return

        result = self.model.hr(
            start=-self.hr_window_sec,
            return_hrv=False,
        )

        hr = self._finite_float((result or {}).get("hr"))
        sqi = self._finite_float((result or {}).get("SQI"))

        if self.HR_MIN <= hr <= self.HR_MAX:
            self.hr_bpm = hr
            self.hr_conf = float(np.clip(sqi, 0.0, 1.0))
        else:
            self.hr_conf = 0.0

    def update_rr(self, face_visible):
        """호흡수를 갱신한다."""
        if not face_visible:
            self.rr_conf = 0.0
            return

        result = self.model.hr(start=-self.rr_window_sec)

        hrv = (result or {}).get("hrv") or {}
        breathing_hz = self._finite_float(
            hrv.get("breathingrate")
        )

        # open-rppg의 breathingrate는 Hz이므로 분당 횟수로 변환한다.
        rr = breathing_hz * 60.0
        sqi = self._finite_float((result or {}).get("SQI"))

        if self.RR_MIN <= rr <= self.RR_MAX:
            self.rr_bpm = rr
            self.rr_conf = float(np.clip(sqi, 0.0, 1.0))
        else:
            self.rr_conf = 0.0


# ══════════════════════════════════════════════════════
#  HUD
# ══════════════════════════════════════════════════════

def draw_hud(frame, hr, rr, fps, anomaly):
    """hr/rr = (값, 신뢰도). cv2.putText는 한글 미지원이므로 ASCII 사용."""
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (0, 0),
        (355, 225),
        (10, 10, 10),
        -1,
    )
    cv2.addWeighted(
        overlay,
        0.55,
        frame,
        0.45,
        0,
        frame,
    )

    def block(label, value, conf, unit, y, hue):
        col = hue if conf > 0.3 else (130, 130, 130)

        cv2.putText(
            frame,
            label,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            col,
            1,
        )

        text = (
            f"{value:.0f} {unit}"
            if conf > 0.3
            else "warming up..."
        )

        cv2.putText(
            frame,
            text,
            (10, y + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.25,
            col,
            2,
        )

        cv2.rectangle(
            frame,
            (10, y + 44),
            (100, y + 52),
            (50, 50, 50),
            -1,
        )

        cv2.rectangle(
            frame,
            (10, y + 44),
            (10 + int(min(conf, 1.0) * 90), y + 52),
            col,
            -1,
        )

    block(
        "HEART RATE  (open-rppg)",
        hr[0],
        hr[1],
        "BPM",
        26,
        (80, 220, 80),
    )

    block(
        "RESP RATE   (open-rppg)",
        rr[0],
        rr[1],
        "BrPM",
        108,
        (80, 180, 255),
    )

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
            text = (
                f"BASELINE "
                f"{anomaly['progress'] * 100:.0f}%"
            )

        elif anomaly["state"] in ("invalid", "warmup"):
            col = (0, 200, 255)
            text = "NO VALID SIGNAL - HOLD"

        else:
            col = (80, 220, 80)
            text = "NORMAL"

        cv2.putText(
            frame,
            text,
            (10, 212),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            col,
            1,
        )

    cv2.putText(
        frame,
        f"FPS: {fps:.0f}",
        (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (160, 160, 160),
        1,
    )

    cv2.putText(
        frame,
        "HR:60-100  RR:12-20",
        (w - 185, h - 10),
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
    headless=False,
):
    print("[INFO] open-rppg 모델 로딩 중...")

    # 내부 지역 함수가 포함돼 있으므로 pickle로 저장하지 않는다.
    model = rppg.Model("ME-flow.rlap")

    print("[INFO] 모델 로딩 완료")

    show_window = should_show_window(headless)

    if headless:
        print("[INFO] 헤드리스 모드로 실행합니다.")

    elif not show_window:
        print(
            "[WARN] DISPLAY가 없어 자동으로 "
            "헤드리스 모드로 실행합니다."
        )

    vitals = OpenRppgVitalTracker(model)

    detector = VitalAnomalyDetector(
        calib_sec=calib_sec,
        min_conf=min_conf,
        out_pct=out_pct,
        rr_sigma=rr_sigma,
    )

    anomaly = None

    # HRV 계산이 상대적으로 무거우므로 서로 다른 주기를 사용한다.
    hr_update_interval = 2.0
    rr_update_interval = 10.0

    last_hr_update = time.time()
    last_rr_update = time.time()

    fps_timer = time.time()
    frame_count = 0
    measured_fps = 0.0

    print("=" * 62)
    print(
        f"  open-rppg 심박수 + 호흡수 측정 시작 "
        f"(기준 학습 {calib_sec:.0f}초)"
    )

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

                frame_count += 1

                if now - fps_timer >= 2.0:
                    measured_fps = (
                        frame_count / (now - fps_timer)
                    )
                    frame_count = 0
                    fps_timer = now

                if not face_visible:
                    vitals.invalidate()

                # 호흡수는 10초마다 갱신한다.
                rr_updated = False

                if now - last_rr_update >= rr_update_interval:
                    vitals.update_rr(face_visible)
                    last_rr_update = now
                    rr_updated = True

                # 심박수와 이상탐지는 2초마다 갱신한다.
                if now - last_hr_update >= hr_update_interval:
                    vitals.update_hr(face_visible)

                    anomaly = detector.push(
                        vitals.hr_bpm,
                        vitals.hr_conf,
                        vitals.rr_bpm,
                        vitals.rr_conf,
                        now=now,
                    )

                    rr_mark = "*" if rr_updated else " "

                    print(
                        f"[HR] {vitals.hr_bpm:6.1f} "
                        f"({vitals.hr_conf:.2f})"
                        f"  [RR{rr_mark}] "
                        f"{vitals.rr_bpm:5.1f} "
                        f"({vitals.rr_conf:.2f})"
                        "  source=open-rppg"
                    )

                    score = anomaly["score"]
                    threshold = anomaly["threshold"]

                    score_text = (
                        "score=-"
                        if score is None
                        else f"score={score:.3f}"
                    )

                    threshold_text = (
                        ""
                        if threshold is None
                        else f" / 임계 {threshold:.3f}"
                    )

                    stale = (
                        anomaly["state"]
                        in ("invalid", "warmup")
                    )

                    if anomaly["critical"]:
                        status = (
                            f"** 절대범위 이탈: "
                            f"{anomaly['critical']}"
                        )

                    elif anomaly["state"] == "signal_lost":
                        status = "** 신호 소실 (측정 중단)"

                    elif anomaly["alert"]:
                        status = (
                            f"** 이상징후: "
                            f"{anomaly['alert_reason']}  "
                            f"{score_text}{threshold_text}"
                        )

                        if stale:
                            status += (
                                "  (현재 샘플 무효 - 알림 유지)"
                            )

                    elif anomaly["baseline"] is None:
                        status = (
                            f"기준 학습: 시간 "
                            f"{anomaly['progress_time'] * 100:3.0f}%"
                            f" / 샘플 "
                            f"{anomaly['accepted']}"
                            f"/{detector.min_samples}"
                            f"  {detector.stats()}"
                        )

                    elif stale:
                        status = (
                            "판정 보류 - 유효 신호 없음"
                            f" (원인="
                            f"{anomaly['reason'] or anomaly['state']})"
                        )

                    else:
                        baseline = anomaly["baseline"]

                        status = (
                            f"NORMAL  "
                            f"{score_text}{threshold_text}"
                            f"  기준선 HR {baseline[0]:.0f}"
                            f" / RR {baseline[1]:.0f}"
                        )

                    print(f"  {status}")
                    last_hr_update = now

                # GUI가 허용된 경우에만 OpenCV 창 관련 함수를 호출한다.
                if show_window:
                    frame = cv2.cvtColor(
                        frame_rgb,
                        cv2.COLOR_RGB2BGR,
                    )

                    if box is not None:
                        (y1, y2), (x1, x2) = (
                            box[0],
                            box[1],
                        )

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (80, 220, 80),
                            2,
                        )

                    frame = draw_hud(
                        frame,
                        (
                            vitals.hr_bpm,
                            vitals.hr_conf,
                        ),
                        (
                            vitals.rr_bpm,
                            vitals.rr_conf,
                        ),
                        measured_fps,
                        anomaly,
                    )

                    cv2.imshow(
                        "Vital Monitor  "
                        "(HR + RR: open-rppg)",
                        frame,
                    )

                    key = cv2.waitKey(1) & 0xFF

                    if key in (ord("q"), 27):
                        print("[INFO] 종료")
                        break

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C 입력으로 종료합니다.")

    finally:
        if show_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="심박수 + 호흡수 실시간 측정"
    )

    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="카메라 ID",
    )

    parser.add_argument(
        "--calib",
        type=float,
        default=180.0,
        help="기준 학습 초",
    )

    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.30,
        help="open-rppg SQI 게이팅 임계",
    )

    parser.add_argument(
        "--out-pct",
        type=float,
        default=1.0,
        help="학습 분포 하위 몇 %%를 이상 경계로 사용할지",
    )

    parser.add_argument(
        "--rr-sigma",
        type=float,
        default=4.0,
        help="RR 측정 불확실성(BPM). 클수록 RR 오탐 감소",
    )

    parser.add_argument(
        "--headless",
        action="store_true",
        help="OpenCV 영상 창 없이 실행",
    )

    args, _ = parser.parse_known_args()

    run(
        camera_id=args.camera,
        calib_sec=args.calib,
        min_conf=args.min_conf,
        out_pct=args.out_pct,
        rr_sigma=args.rr_sigma,
        headless=args.headless,
    )