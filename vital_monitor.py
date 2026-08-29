"""
실시간 심박수 + 호흡수 + 얼굴 온도 측정 + 환자별 개인화 이상 탐지
+ 얼굴 팬/틸트 모터 트래킹 (Dynamixel, PID)  -- CLI / OpenCV HUD 렌더러

계측 자체는 전부 vital_pipeline.VitalPipeline 이 한다 (D-1). 이 파일은
화면에 그리고 콘솔에 찍는 일만 한다. vital_run.py 의 Tk 대시보드도 같은
파이프라인을 쓰므로 두 실행 경로의 계측 결과가 항상 동일하다.

  심박수  : open-rppg 의 BVP 주파수 분석
  호흡수  : rr.py 의 Lucas-Kanade 옵티컬 플로우 기반 어깨 움직임 추적
            (어깨 ROI 는 검출된 얼굴 박스에서 유도한다)
  얼굴온도: MLX90640 열화상. 얼굴 박스 중심부만 32x24 격자에 매핑한 뒤
            SkinToCore 로 심부 추정치까지 환산한다
  이상탐지: Isolation Forest + robust z (환자별 기준 분포) -> vital_anomaly.py
  얼굴추적: 팬(ID1)/틸트(ID2) 다이나믹셀 PID 구동 -> face_tracker_motor.py
  알림    : 이상 판정 시 액티브 부저(GPIO) 논블로킹 비프

설치:
    pip install open-rppg opencv-python numpy scipy scikit-learn adafruit-circuitpython-mlx90640 dynamixel-sdk gpiozero

실행 (각 줄을 통째로 한 줄에 복사해서 쓴다. 줄바꿈용 백슬래시를 쓰지 않는
이유는, 한 줄로 붙여넣을 때 그게 인자로 들어가 argparse 가 죽기 때문이다):

    # 라즈베리파이 데스크톱
    python3 vital_monitor.py --camera 0 --dynamixel-port /dev/ttyACM0 --pan-sign 1 --tilt-sign -1

    # SSH 또는 VS Code Remote
    python3 vital_monitor.py --camera 0 --headless

    # 열화상 화각 보정 (C-7). 결과를 config.py 에 붙여넣는다
    python3 vital_monitor.py --thermal-calib

    # 환자별 기준선 저장/복원 + 세션 CSV 로깅
    python3 vital_monitor.py --camera 0 --patient-id kim --session-log auto
"""

import argparse
import os
import time

import cv2

import config
import rppg
from config import get_logger
from vital_pipeline import VitalPipeline
from vital_sensors import (          # 하위 호환 재export (D-1 로 모듈만 이동)
    AlarmBuzzer,
    OpenRppgVitalTracker,
    ThermalFaceTracker,
    fit_thermal_alignment,
    picamera_rgb888_to_rgb,
)

log = get_logger("monitor")

__all__ = [
    "AlarmBuzzer",
    "OpenRppgVitalTracker",
    "ThermalFaceTracker",
    "should_show_window",
    "draw_hud",
    "run",
]


def should_show_window(headless, environ=None):
    """X11 디스플레이가 있고 headless 가 아닐 때만 창을 표시한다."""
    env = os.environ if environ is None else environ
    return not headless and bool(env.get("DISPLAY"))


# ══════════════════════════════════════════════════════
#  HUD
# ══════════════════════════════════════════════════════

TRACK_COLORS = {
    "LOCKED": (80, 220, 80),
    "VERIFYING": (0, 215, 255),
    "HOLD": (0, 140, 255),
    "LOST": (120, 120, 120),
    "FAULT": (40, 40, 255),
    "SEARCHING": (160, 160, 160),
    "DIRECTION_UNCONFIRMED": (180, 80, 255),
    "OFF": (110, 110, 110),
}


def draw_hud(frame, snap):
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

    # RR 카드는 reporter 가 값을 들고 있으면 켠다 (SQI 자체는 따로 표시).
    rr_shown_conf = 1.0 if snap["rr_valid"] else 0.0
    block("HEART RATE  (open-rppg)", snap["hr"], snap["hr_conf"], "BPM", 26,
          (80, 220, 80))
    block("RESP RATE   (Optical Flow)", snap["rr"], rr_shown_conf, "BrPM", 108,
          (80, 180, 255))
    block("FACE TEMP   (MLX90640)", snap["temp"], snap["temp_conf"], "C", 190,
          (255, 200, 80))

    anomaly = snap["anomaly"]
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

    fps_text = f"FPS: {snap['fps']:.0f}"
    if snap["motion_hold"]:
        fps_text += "   MOTOR MOVING"
    cv2.putText(frame, fps_text, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (160, 160, 160), 1)

    cv2.putText(
        frame,
        "HR:60-100  RR:12-20  Temp:36-37.5",
        (w - 250, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (120, 120, 120),
        1,
    )

    tracker = snap["tracker"]
    color = TRACK_COLORS.get(tracker["state"], (160, 160, 160))
    cv2.putText(
        frame,
        f"TRACK {tracker['state']}  q={tracker['confidence']:.2f}",
        (max(10, w - 285), 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )
    if tracker["reason"]:
        cv2.putText(
            frame,
            tracker["reason"][:34],
            (max(10, w - 285), 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
        )

    return frame


def _console_line(snap):
    temp_text = (
        f"{snap['temp']:5.1f}C" if snap["temp_conf"] > 0.3 else "  -- "
    )
    rr_mark = "*" if snap["rr_updated"] else " "
    tracker_status = snap["tracker"]["status"]
    face = snap.get("face_detector", {})
    source = face.get("source", "NONE")
    motor_conf = float(face.get("motor_conf", 0.0) or 0.0)
    age_ms = float(face.get("age_ms", 0.0) or 0.0)
    track_id = face.get("track_id")
    face_text = (
        f"face={source} q={motor_conf:.2f} age={age_ms:.0f}ms "
        f"id={'-' if track_id is None else track_id}"
    )
    return (
        f"[HR] {snap['hr']:6.1f} ({snap['hr_conf']:.2f}) {snap['hr_reason']:<14}"
        f"  [RR{rr_mark}] {snap['rr']:5.1f} ({snap['rr_conf']:.2f})"
        f" {snap.get('rr_source', 'NONE')}/{snap.get('rr_state', 'UNAVAILABLE'):<14}"
        f"  [TEMP] {temp_text} ({snap['temp_conf']:.2f})"
        f"  [FPS] {snap['fps']:4.1f}"
        f"  {face_text}"
        + (f"  {tracker_status}" if tracker_status else "")
    )


def _status_line(snap, pipeline):
    anomaly = snap["anomaly"]
    if anomaly is None:
        return "이상탐지 비활성 - 계측만 표시합니다"

    score = anomaly["score"]
    threshold = anomaly["threshold"]
    score_text = "score=-" if score is None else f"score={score:.3f}"
    threshold_text = "" if threshold is None else f" / 임계 {threshold:.3f}"
    stale = anomaly["state"] in ("invalid", "warmup")

    if anomaly["critical"]:
        return f"** 절대범위 이탈: {anomaly['critical']}"
    if anomaly["state"] == "signal_lost":
        return "** 신호 소실 (측정 중단)"
    if anomaly["alert"]:
        text = (f"** 이상징후: {anomaly['alert_reason']}  "
                f"{score_text}{threshold_text}")
        return text + ("  (현재 샘플 무효 - 알림 유지)" if stale else "")
    if anomaly["baseline"] is None:
        return (f"기준 학습: 시간 {anomaly['progress_time'] * 100:3.0f}%"
                f" / 샘플 {anomaly['accepted']}/{snap['min_samples']}"
                f"  {pipeline.detector.stats()}")
    if stale:
        return ("판정 보류 - 유효 신호 없음 "
                f"(원인={anomaly['reason'] or anomaly['state']})")

    baseline = anomaly["baseline"]
    base_text = f"기준선 HR {baseline[0]:.0f} / RR {baseline[1]:.0f}"
    if baseline[2] is not None:
        base_text += f" / TEMP {baseline[2]:.1f}"
    return f"NORMAL  {score_text}{threshold_text}  {base_text}"


# ══════════════════════════════════════════════════════
#  열화상 화각 보정  (C-7)
# ══════════════════════════════════════════════════════

def thermal_calib(camera_id=0, thermal_freq=None, min_hot_c=30.0):
    """
    RGB 카메라와 MLX90640 의 화각 정합을 실측한다.

    config.py 의 MLX_FOV_SCALE_* / MLX_OFFSET_* 기본값 1.0 / 0.0 은 "두 센서
    화각이 같다" 는 가정일 뿐 보정된 값이 아니다. 그대로 두면 얼굴 박스를
    실제보다 넓은 영역에 매핑해 머리카락과 배경이 섞이고 온도가 낮게 나온다.

    사용법:
      1) 이 모드를 실행한다.
      2) 얼굴을 화면 **중앙**에 두고 3초 정도 머문다.
      3) 좌 -> 우 -> 위 -> 아래 가장자리 가까이로 하나씩 천천히 옮기며
         각각 3초씩 머문다. (모서리까지 갈수록 추정이 정확해진다)
      4) Ctrl+C 로 끝내면 fit 결과가 출력된다.
      5) 출력된 4줄을 config.py 에 그대로 붙여넣는다.

    표본은 (RGB 얼굴 중심, 열화상 hotspot 무게중심) 쌍이며 둘 다 0..1 정규화
    좌표다. 얼굴이 아닌 열원(라디에이터 등)을 잡지 않도록 격자 최고 온도가
    min_hot_c 이상일 때만 표본으로 인정한다.
    """
    log.info("open-rppg 모델 로딩 중...")
    model = rppg.Model(config.RPPG_MODEL)

    thermal = ThermalFaceTracker(refresh_rate_hz=thermal_freq)
    if not thermal.available:
        print("열화상 센서를 열 수 없습니다. 보정을 진행할 수 없습니다.")
        return

    samples = []
    last_sample = 0.0

    print("=" * 62)
    print("  열화상 화각 보정 (C-7)")
    print("  중앙 -> 좌 -> 우 -> 위 -> 아래 순서로 3초씩 머무세요.")
    print("  Ctrl+C 로 종료하면 결과가 출력됩니다.")
    print("=" * 62)

    try:
        with model.video_capture(camera_id):
            for frame_rgb, box in model.preview:
                now = time.time()
                if box is None or now - last_sample < 0.5:
                    continue

                grid = thermal.raw_grid()
                hotspot = thermal.hotspot_norm()
                if grid is None or hotspot is None:
                    continue
                if float(grid.max()) < min_hot_c:
                    continue

                (y1, y2), (x1, x2) = box
                h, w = frame_rgb.shape[0], frame_rgb.shape[1]
                rgb_nx = (float(x1) + float(x2)) * 0.5 / w
                rgb_ny = (float(y1) + float(y2)) * 0.5 / h

                samples.append((rgb_nx, rgb_ny, hotspot[0], hotspot[1]))
                last_sample = now
                print(
                    f"  n={len(samples):3d}  RGB=({rgb_nx:.3f},{rgb_ny:.3f})"
                    f"  THERMAL=({hotspot[0]:.3f},{hotspot[1]:.3f})"
                    f"  max={float(grid.max()):.1f}C",
                    flush=True,
                )
    except KeyboardInterrupt:
        print()
    finally:
        thermal.close()

    print("=" * 62)
    fit = fit_thermal_alignment(samples)
    if fit is None:
        print(f"표본이 부족하거나 한쪽에 몰려 있습니다 (n={len(samples)}).")
        print("얼굴을 화면 좌우/상하 끝까지 충분히 옮기며 다시 측정하세요.")
        return

    print(f"표본 {fit['n']}개, 잔차 RMS x={fit['rms_x']:.4f} y={fit['rms_y']:.4f}")
    print("아래 4줄을 config.py 에 그대로 붙여넣으세요:")
    print()
    print(f"MLX_FOV_SCALE_X = {fit['scale_x']:.4f}")
    print(f"MLX_FOV_SCALE_Y = {fit['scale_y']:.4f}")
    print(f"MLX_OFFSET_X = {fit['offset_x']:.4f}")
    print(f"MLX_OFFSET_Y = {fit['offset_y']:.4f}")
    print()
    if max(fit["rms_x"], fit["rms_y"]) > 0.06:
        print("경고: 잔차가 큽니다. 얼굴이 아닌 열원을 잡았거나 표본이 한쪽에")
        print("      몰렸을 수 있습니다. 다시 측정하는 것을 권합니다.")


# ══════════════════════════════════════════════════════
#  메인 루프
# ══════════════════════════════════════════════════════

def run(
    camera_id=0,
    calib_sec=None,
    min_conf=None,
    out_pct=None,
    rr_sigma=None,
    temp_sigma=None,
    temp_offset=None,
    temp_dist_gain=None,
    temp_face_px_ref=None,
    headless=False,
    use_thermal=True,
    thermal_stat=None,
    thermal_freq=None,
    use_tracker=True,
    use_buzzer=True,
    buzzer_pin=17,
    dynamixel_port=None,
    dxl_baud=None,
    pan_sign=None,
    tilt_sign=None,
    patient_id=None,
    use_baseline_store=True,
    session_csv=None,
    remote_host=None,
    remote_user=None,
    remote_key=None,
    remote_dir=None,
    remote_interval=None,
    face_detector_mode=None,
    scrfd_onnx=None,
    yolo_onnx=None,
):
    log.info("open-rppg 모델 로딩 중...")
    model = rppg.Model(config.RPPG_MODEL)
    log.info("모델 로딩 완료")

    show_window = should_show_window(headless)
    if headless:
        log.info("헤드리스 모드로 실행합니다.")
    elif not show_window:
        log.warning("DISPLAY 가 없어 자동으로 헤드리스 모드로 실행합니다.")

    pipeline = VitalPipeline(
        model,
        use_thermal=use_thermal,
        thermal_stat=thermal_stat,
        thermal_freq=thermal_freq,
        use_tracker=use_tracker,
        dynamixel_port=dynamixel_port,
        pan_sign=pan_sign,
        tilt_sign=tilt_sign,
        dxl_baud=dxl_baud,
        calib_sec=calib_sec,
        min_conf=min_conf,
        out_pct=out_pct,
        rr_sigma=rr_sigma,
        temp_sigma=temp_sigma,
        temp_offset=temp_offset,
        temp_dist_gain=temp_dist_gain,
        temp_face_px_ref=temp_face_px_ref,
        use_buzzer=use_buzzer,
        buzzer_pin=buzzer_pin,
        patient_id=patient_id,
        use_baseline_store=use_baseline_store,
        session_csv=session_csv,
        remote_host=remote_host,
        remote_user=remote_user,
        remote_key=remote_key,
        remote_dir=remote_dir,
        remote_interval=remote_interval,
        face_detector_mode=face_detector_mode,
        scrfd_onnx=scrfd_onnx,
        yolo_onnx=yolo_onnx,
    )

    header = "  심박수 + 호흡수 + 얼굴 온도 측정 시작"
    if pipeline.detector is not None:
        header += f"  (기준 학습 {pipeline.detector.calib_sec:.0f}초)"

    print("=" * 62)
    print(header)
    print("  종료: Q / ESC 또는 Ctrl+C" if show_window else "  종료: Ctrl+C")
    print("=" * 62)

    last_print = 0.0

    try:
        with model.video_capture(camera_id):
            for frame_bgr, box in model.preview:
                frame_rgb = picamera_rgb888_to_rgb(frame_bgr)
                snap = pipeline.step(frame_rgb, box)
                now = time.time()

                # 콘솔 로그는 이상탐지 주기와 맞춘다.
                if now - last_print >= config.HR_UPDATE_SEC:
                    last_print = now
                    print(_console_line(snap), flush=True)
                    print(f"  {_status_line(snap, pipeline)}", flush=True)

                if not show_window:
                    continue

                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                if box is not None:
                    (y1, y2), (x1, x2) = box
                    color = TRACK_COLORS.get(
                        snap["tracker"]["state"], (150, 150, 150)
                    )
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                if pipeline.tracker is not None:
                    target = pipeline.tracker.target_box()
                    if target is not None:
                        (ty1, ty2), (tx1, tx2) = target
                        cv2.rectangle(
                            frame,
                            (int(tx1), int(ty1)),
                            (int(tx2), int(ty2)),
                            (255, 220, 60),
                            1,
                        )

                frame = draw_hud(frame, snap)
                cv2.imshow("Vital Monitor  (HR + RR + Temp)", frame)
                if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                    log.info("종료")
                    break

    except KeyboardInterrupt:
        print()
        log.info("Ctrl+C 입력으로 종료합니다.")
    finally:
        pipeline.close()
        if show_window:
            cv2.destroyAllWindows()


def build_parser():
    parser = argparse.ArgumentParser(
        description="심박수 + 호흡수 + 얼굴 온도 실시간 측정"
    )

    parser.add_argument("--camera", type=int, default=config.CAMERA_ID,
                        help="카메라 ID")
    parser.add_argument(
        "--face-detector",
        choices=["cpu-hybrid", "rppg"],
        default=config.FACE_DETECTOR,
        help="얼굴 검출기: CPU SCRFD+YOLOv8 또는 명시적 rPPG 진단 폴백",
    )
    parser.add_argument("--scrfd-onnx", default=None,
                        help="SCRFD CPU ONNX 경로")
    parser.add_argument("--yolo-onnx", default=None,
                        help="YOLOv8 CPU ONNX 경로")
    parser.add_argument("--calib", type=float, default=config.CALIB_SEC,
                        help="기준 학습 초")
    parser.add_argument("--min-conf", type=float, default=config.MIN_CONF,
                        help="SQI 게이팅 임계")
    parser.add_argument("--out-pct", type=float, default=config.OUT_PCT,
                        help="학습 분포 하위 몇 %%를 이상 경계로 사용할지")
    parser.add_argument("--rr-sigma", type=float, default=config.RR_SIGMA,
                        help="RR 측정 불확실성(BPM)")
    parser.add_argument("--temp-sigma", type=float, default=config.TEMP_SIGMA,
                        help="얼굴 온도 측정 불확실성(C)")
    parser.add_argument("--temp-offset", type=float,
                        default=config.TEMP_OFFSET_DEFAULT,
                        help=("피부 표면 -> 심부 체온 보정(C). 기본 "
                              f"{config.TEMP_OFFSET_DEFAULT} 는 임시값이므로 "
                              "기준 체온계로 실측해서 정할 것"))
    parser.add_argument("--temp-dist-gain", type=float, default=None,
                        help="거리(충전율) 보정 계수. temp_calib.py --fit 로 산출")
    parser.add_argument("--temp-face-px-ref", type=float, default=None,
                        help="보정 당시 얼굴이 덮던 열화상 화소 수")
    parser.add_argument("--headless", action="store_true",
                        help="OpenCV 영상 창 없이 실행")
    parser.add_argument("--no-thermal", action="store_true",
                        help="MLX90640 얼굴 온도 측정을 비활성화")
    parser.add_argument("--thermal-stat", choices=["p90", "max"],
                        default=config.THERMAL_STAT,
                        help="얼굴 영역 온도 대표값: 상위10%% 평균(p90) 또는 최고값(max)")
    parser.add_argument("--thermal-freq", type=int, default=config.MLX_REFRESH_HZ,
                        choices=[1, 2, 4, 8], help="MLX90640 갱신 주파수(Hz)")
    parser.add_argument("--thermal-calib", action="store_true",
                        help="C-7: RGB<->열화상 화각을 실측하고 종료")
    parser.add_argument("--no-tracker", action="store_true",
                        help="얼굴 팬/틸트 모터 추적을 비활성화")
    parser.add_argument("--no-buzzer", action="store_true",
                        help="이상 알림 부저를 비활성화")
    parser.add_argument("--buzzer-pin", type=int, default=17,
                        help="액티브 부저가 연결된 GPIO 번호(BCM)")
    parser.add_argument("--dynamixel-port", default=config.DXL_PORT,
                        help="Dynamixel USB 시리얼 장치")
    parser.add_argument("--dxl-baud", type=int, default=None,
                        help=(f"통신 속도(기본 {config.DXL_BAUDRATE}). 응답이 "
                              "없으면 후보 속도를 자동 탐색한다"))
    parser.add_argument("--pan-sign", type=float, choices=[-1.0, 1.0],
                        default=None,
                        help="필수: 자가점검으로 확정한 팬 장착 방향(-1 또는 +1)")
    parser.add_argument("--tilt-sign", type=float, choices=[-1.0, 1.0],
                        default=None,
                        help="필수: 자가점검으로 확정한 틸트 장착 방향(-1 또는 +1)")
    parser.add_argument("--patient-id", default=None,
                        help="C-8: 환자 식별자. 지정하면 기준선을 저장/복원한다")
    parser.add_argument("--no-baseline-store", action="store_true",
                        help="기준선 저장/복원을 사용하지 않고 매번 새로 학습")
    parser.add_argument("--session-log", nargs="?", const="auto", default=None,
                        help="C-9: 세션 CSV 경로. 값 없이 주면 자동 이름")
    parser.add_argument("--remote", action="store_true",
                        help=f"측정 CSV 를 {config.REMOTE_HOST} 로 실시간 전송 "
                             "(--session-log 가 없으면 자동으로 켜진다)")
    parser.add_argument("--remote-host", default=config.REMOTE_HOST,
                        help="전송 대상 서버")
    parser.add_argument("--remote-user", default=None,
                        help=f"서버 계정. 미지정 시 자동 탐색: "
                             f"{', '.join(config.REMOTE_USERS)}")
    parser.add_argument("--remote-key", default=config.REMOTE_KEY,
                        help="SSH 개인키 파일")
    parser.add_argument("--remote-dir", default=config.REMOTE_DIR,
                        help="서버 저장 디렉터리 (홈 기준 상대경로)")
    parser.add_argument("--remote-interval", type=float,
                        default=config.REMOTE_INTERVAL,
                        help=f"전송 주기(초). 기본 {config.REMOTE_INTERVAL:.0f}")
    parser.add_argument("--log-file", default=None,
                        help="로그를 파일로도 저장")
    parser.add_argument("--log-level", default=config.LOG_LEVEL,
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="로그 레벨")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args(config.clean_argv())

    config.setup_logging(level=args.log_level, logfile=args.log_file)

    # --remote 는 보낼 행이 있어야 의미가 있으므로 세션 로그를 자동으로 켠다.
    session_csv = args.session_log
    if args.remote and not session_csv:
        session_csv = "auto"
    remote_host = args.remote_host if args.remote else None

    if args.thermal_calib:
        thermal_calib(camera_id=args.camera, thermal_freq=args.thermal_freq)
        raise SystemExit

    run(
        camera_id=args.camera,
        calib_sec=args.calib,
        min_conf=args.min_conf,
        out_pct=args.out_pct,
        rr_sigma=args.rr_sigma,
        temp_sigma=args.temp_sigma,
        temp_offset=args.temp_offset,
        temp_dist_gain=args.temp_dist_gain,
        temp_face_px_ref=args.temp_face_px_ref,
        headless=args.headless,
        use_thermal=not args.no_thermal,
        thermal_stat=args.thermal_stat,
        thermal_freq=args.thermal_freq,
        use_tracker=not args.no_tracker,
        use_buzzer=not args.no_buzzer,
        buzzer_pin=args.buzzer_pin,
        dynamixel_port=args.dynamixel_port,
        dxl_baud=args.dxl_baud,
        pan_sign=args.pan_sign,
        tilt_sign=args.tilt_sign,
        patient_id=args.patient_id,
        use_baseline_store=not args.no_baseline_store,
        session_csv=session_csv,
        remote_host=remote_host,
        remote_user=args.remote_user,
        remote_key=args.remote_key,
        remote_dir=args.remote_dir,
        remote_interval=args.remote_interval,
        face_detector_mode=args.face_detector,
        scrfd_onnx=args.scrfd_onnx,
        yolo_onnx=args.yolo_onnx,
    )

# python3 vital_monitor.py --camera 0 --dynamixel-port /dev/ttyACM0 --pan-sign 1 --tilt-sign -1
