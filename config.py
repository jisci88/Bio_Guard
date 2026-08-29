"""
config.py - Bio-Guardian 튜닝 상수 중앙 저장소   (D-2 / D-3)

현장 튜닝은 이 파일만 고치면 된다. 다른 모듈에는 매직 넘버를 두지 않는다.
우선순위는 항상  CLI 인자 > 이 파일의 기본값  이다.

로깅(D-3):
    from config import get_logger
    log = get_logger(__name__)
    log.info("...")

    - 레벨/타임스탬프가 자동으로 붙는다
    - setup_logging(logfile=...) 로 파일 저장까지 한 번에
"""

import logging
import os
import sys

# ══════════════════════════════════════════════════════
#  로깅  (D-3)
# ══════════════════════════════════════════════════════

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s %(levelname).1s [%(name)s] %(message)s"
LOG_DATEFMT = "%H:%M:%S"

_LOG_CONFIGURED = False


def setup_logging(level=None, logfile=None):
    """루트 로거를 한 번만 구성한다. 두 번째 호출부터는 무시된다."""
    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return
    _LOG_CONFIGURED = True

    # 라즈베리파이는 UTF-8 이지만 윈도우 콘솔은 cp949 라서 한글 로그가 깨진다.
    # 콘솔 인코딩을 UTF-8 로 올릴 수 있으면 올리고, 안 되면 조용히 넘어간다.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    handlers = [logging.StreamHandler(sys.stdout)]
    if logfile:
        try:
            parent = os.path.dirname(os.path.abspath(logfile))
            os.makedirs(parent, exist_ok=True)
            handlers.append(logging.FileHandler(logfile, encoding="utf-8"))
        except Exception as exc:  # 로그 파일 실패가 계측을 막으면 안 된다
            print(f"[WARN] 로그 파일을 열 수 없습니다 ({logfile}): {exc}")

    logging.basicConfig(
        level=getattr(logging, str(level or LOG_LEVEL).upper(), logging.INFO),
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
        handlers=handlers,
    )


def get_logger(name):
    setup_logging()
    return logging.getLogger(name)


def clean_argv(argv=None):
    """줄바꿈용 백슬래시가 인자로 섞여 들어온 것을 걸러낸다.

    여러 줄로 적힌 예제를 한 줄로 붙여넣으면 이렇게 된다:

        python3 vital_run.py --camera 0 \\ --dynamixel-port /dev/ttyACM0
                                       ^^ 줄 끝이 아니라 그냥 인자가 된다

    백슬래시는 줄 **맨 끝**에 있을 때만 줄바꿈 기호다. 뒤에 공백이 오면
    셸이 `\\` 를 인자 하나로 넘기고, argparse 는
    "unrecognized arguments: \\" 로 죽는다. 흔한 복붙 사고라 여기서 막는다.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    kept = [a for a in argv if a.strip("\\") != "" or a == ""]
    dropped = len(argv) - len(kept)
    if dropped:
        print(f"[WARN] 줄바꿈용 백슬래시 {dropped}개를 인자에서 제외했습니다. "
              "명령을 한 줄로 쓸 때는 '\\' 를 빼세요.")
    return kept


# ══════════════════════════════════════════════════════
#  카메라 / 파이프라인
# ══════════════════════════════════════════════════════

CAMERA_ID = 0
EXPECTED_FRAME_SIZE = (640, 480)   # 출전 튜닝 기준 해상도
FPS_OK_RANGE = (24.0, 36.0)        # 이 밖이면 경고 1회
FPS_REPORT_SEC = 2.0

RPPG_MODEL = "ME-flow.rlap"


# ══════════════════════════════════════════════════════
#  Raspberry Pi 5 CPU 하이브리드 얼굴 검출
# ══════════════════════════════════════════════════════

FACE_DETECTOR = "cpu-hybrid"
CPU_SCRFD_MODEL = "det_500m"
CPU_YOLO_MODEL = "yolov8n"
CPU_SCRFD_INPUT_SIZE = 320
CPU_YOLO_INPUT_SIZE = 320
CPU_SCRFD_CONF = 0.55
CPU_PERSON_CONF = 0.50
CPU_EDGE_MARGIN = 0.12
CPU_RESULT_MAX_AGE = 0.20
CPU_FLOW_HOLD_SEC = 0.25
CPU_PERSON_HOLD_SEC = 0.60
CPU_YOLO_IDLE_HZ = 1.0
CPU_YOLO_RECOVERY_HZ = 2.0
CPU_SOURCE_SPEED = {"SCRFD": 1.0, "SCRFD_EDGE": 0.8, "FLOW": 0.65, "PERSON_HEAD": 0.50}
# SCRFD and YOLO complete asynchronously.  Keep the last verified face ROI
# only long enough to bridge one detector cycle; it is never motor evidence.
MEASUREMENT_ROI_HOLD_SEC = 0.75


# ══════════════════════════════════════════════════════
#  심박수 (open-rppg)
# ══════════════════════════════════════════════════════

HR_WINDOW_SEC = 10.0     # model.hr 분석 창
HR_UPDATE_SEC = 2.0      # 갱신 주기 = 이상탐지 push 주기
HR_MIN, HR_MAX = 40.0, 200.0
HR_HOLD_SEC = 6.0        # 마지막 유효값을 신뢰도 감쇠로 유지하는 시간
HR_OK_CONF = 0.30        # 이 이상이면 "OK", 미만이면 SQI LOW


# ══════════════════════════════════════════════════════
#  호흡수 (rr.py 옵티컬 플로우)
# ══════════════════════════════════════════════════════

# ── C-3: 표시용 게이트와 학습용 게이트를 분리한다 ──
# 화면과 기준선 학습은 같은 최소 SQI 를 사용한다. 대신 학습기에는 1.0을
# 강제로 넣지 않고 실제 RR 신뢰도를 전달해 MIN_CONF 에서 다시 검증한다.
# 서로 다른 두 문턱을 쓰면 RR 숫자는 보이지만 학습 샘플은 영원히 0개가 된다.
RR_SQI_DISPLAY = 0.03
RR_SQI_LEARN = RR_SQI_DISPLAY

# SQI 를 0~1 신뢰도로 환산할 때의 기준값 (이 카메라 경로에서 견고한 락의 점수)
RR_CONF_REF = 0.10

# ── C-1: 얼굴 박스에서 어깨 ROI 를 유도할 때 쓰는 배율 ──
# 얼굴 박스 폭/높이의 배수. 팬/틸트가 얼굴을 중앙에 두므로 어깨는 항상
# 얼굴 아래 일정 배율에 있다. 고정 비율 ROI 는 특정 촬영거리에만 맞았다.
RR_ROI_FROM_FACE = True
RR_ROI_WIDTH_FACES = 3.2     # 어깨 밴드 가로 = 얼굴폭 x 3.2
RR_ROI_TOP_FACES = 1.15      # 밴드 상단 = 얼굴 상단 + 얼굴높이 x 1.15
RR_ROI_BOTTOM_FACES = 2.35   # 밴드 하단 = 얼굴 상단 + 얼굴높이 x 2.35
RR_ROI_MIN_PX = (120, 60)    # 이보다 작아지면 고정 ROI 로 폴백
RR_ROI_MOVE_FRAC = 0.22      # ROI 중심이 폭의 이 비율 이상 움직여야 재설정

# Adaptive respiration optical-flow geometry.  The person box supplies a
# stable torso band; face geometry supplies a small fallback motion region.
RR_FLOW_HZ = 15.0
RR_TORSO_TOP = 0.22
RR_TORSO_BOTTOM = 0.58
RR_TORSO_SIDE_INSET = 0.18
RR_FACE_INSET = 0.12
RR_ROI_MIN_VISIBLE = 0.70
RR_REGION_MIN_PTS = 12
RESP_WAVE_SEC = 12.0         # C-4: 화면에 그릴 대역통과 호흡 파형 길이

# Adaptive respiration candidate quality.  Hard gates reject evidence which
# cannot support a fresh rate; the weighted terms then keep moderate, separate
# evidence from collapsing to zero as a product would.
RR_MIN_VALID_FRACTION = 0.35
RR_MIN_PERIODICITY = 0.04
RR_MIN_COVERAGE = 0.50
RR_SNR_FULL = 4.0
RR_PERIODICITY_FULL = 0.60
RR_CONCENTRATION_FULL = 0.35
RR_QUALITY_WEIGHTS = (0.24, 0.26, 0.20, 0.18, 0.12)
# Optical-flow motion is measured in pixels after respiration-band filtering.
# It needs both a visible absolute signal and substantially stronger spectral
# evidence than BVP before it can become temporal-lock input.
# Lucas–Kanade chest motion is commonly only a few thousandths of a pixel
# after respiratory-band filtering at 640x480.  Keep zero/jitter rejection,
# but admit clean sub-pixel periodic motion for the later spectral/temporal
# gates to judge.
RR_FLOW_MIN_MOTION_ENERGY = 0.0002
RR_FLOW_MIN_SPECTRAL_SNR = 0.015
RR_FLOW_MIN_CONCENTRATION = 0.20
RR_BVP_MIN_CONF = 0.15
RR_BVP_MIN_PERIODICITY = 0.50
RR_BVP_MIN_CONCENTRATION = 0.35
RR_BVP_MAX_AGE = 3.0

# Candidate fusion and temporal lock.  A displayed value becomes fresh only
# after independently-selected estimates have remained stable over this many
# reporting updates.
RR_LOCK_UPDATES = 2
RR_LOCK_STABILITY_BPM = 3.5
RR_AGREEMENT_BPM = 2.0
RR_HOLD_SEC = 10.0
RR_LEARN_CONF = 0.15

# Display-only fallback: a weak but finite respiratory peak is shown as
# ACQUIRING. It never enters the lock sequence, baseline learning, or alarm.
RR_DISPLAY_MIN_VALID_FRACTION = 0.20
RR_DISPLAY_MIN_PERIODICITY = 0.00
RR_DISPLAY_MIN_COVERAGE = 0.35
RR_DISPLAY_FLOW_MIN_MOTION_ENERGY = 0.00005
RR_DISPLAY_FLOW_MIN_SPECTRAL_SNR = 0.005
RR_DISPLAY_FLOW_MIN_CONCENTRATION = 0.08
RR_DISPLAY_BVP_MIN_CONF = 0.05
RR_DISPLAY_BVP_MIN_PERIODICITY = 0.10
RR_DISPLAY_BVP_MIN_CONCENTRATION = 0.10


# ══════════════════════════════════════════════════════
#  모터 이동 중 게이팅  (C-2)
# ══════════════════════════════════════════════════════

# 카메라가 팬/틸트하면 rPPG ROI 가 흔들려 HR 이 오염되고, 어깨 ROI 도
# 시차(parallax) 때문에 배경 특징점 보상만으로는 완전히 제거되지 않는다.
# 이동 중/직후 샘플은 HR 과 RR 양쪽 모두 무효로 본다.
MOTION_HOLD_SEC = 1.5


# ══════════════════════════════════════════════════════
#  열화상 (MLX90640)
# ══════════════════════════════════════════════════════

MLX_COLS, MLX_ROWS = 32, 24
MLX_REFRESH_HZ = 2
MLX_I2C_FREQ = 100000
# MLX90640 is wired to Pi 5 I2C3 on GPIO22/23 (physical pins 15/16).
# Use 1 when rewired to the standard GPIO2/3 I2C header pins instead.
MLX_I2C_BUS = 3
MLX_STALE_SEC = 3.0
MLX_READ_ERROR_SLEEP = 0.05   # B-4: CRC 실패 시 busy-spin 방지

# ── C-7: RGB <-> 열화상 화각 정합 ──
# 아래 1.0 / 0.0 은 "두 센서 화각이 같다" 는 가정이며 보정된 값이 아니다.
# MLX90640 은 BAB 55x35deg, BAA 110x75deg 이고 Pi 카메라는 보통 그 중간이
# 아니다. 실제 값은 다음으로 측정한다:
#     python3 vital_monitor.py --thermal-calib
# 측정이 끝나면 출력된 4개 숫자를 그대로 여기에 붙여넣는다.
MLX_FOV_SCALE_X = 1.0
MLX_FOV_SCALE_Y = 1.0
MLX_OFFSET_X = 0.0
MLX_OFFSET_Y = 0.0

# 얼굴 박스를 그대로 매핑하면 머리카락과 배경이 섞여 온도가 낮게 나온다.
# 이마/볼 중심부만 남기도록 박스를 축소한 뒤 매핑한다.
MLX_FACE_SHRINK = 0.60
MLX_FACE_Y_BIAS = -0.05       # 살짝 위(이마) 쪽으로 치우치게

THERMAL_STAT = "p90"          # p90 = 상위 10% 평균, max = 최고값
THERMAL_UPDATE_SEC = 2.0
# The thermal ROI can momentarily include background when face geometry moves.
# A 9-sample median plus this per-update gate removes those discontinuities;
# it does not change the separately calibrated absolute offset.
TEMP_FILTER_SAMPLES = 9
TEMP_MAX_STEP_C = 0.7


# ══════════════════════════════════════════════════════
#  체온 스케일  (A-5)
# ══════════════════════════════════════════════════════

# 예전에는 vital_monitor 가 "피부 표면 온도"를, vital_run 이 "심부 추정치"를
# 같은 임계값에 넣고 있었다. 이제 파이프라인이 항상 SkinToCore 를 적용하고
# 이상탐지에는 TEMP_SCALE 을 함께 알려준다.
TEMP_SCALE = "core"           # "core" | "skin"

# 피부 표면 -> 심부 보정(C). 기준 체온계와 짝지어 실측해서 정할 것:
#     python3 temp_calib.py --fit pairs.csv     # skin,ambient,ear 한 줄씩
# 그 출력이 --temp-offset / --temp-ambient-gain / --temp-ambient-ref 를 준다.
#
# None 은 "temp_calib.SkinToCore 자체 기본값을 쓴다" 는 뜻이다. 파이프라인은
# None 인 항목을 아예 넘기지 않는다 - None 을 명시적으로 넘기면 그 모듈의
# 기본값(ambient_gain=0.20, ambient_ref=24.0)을 덮어써서 계산이 깨진다.
TEMP_OFFSET_DEFAULT = 3.2
TEMP_AMBIENT_GAIN = None      # 실온 1C 하락당 추가 보정
TEMP_AMBIENT_REF = None       # 오프셋을 측정한 기준 실온

# ── 거리(충전율) 보정 ──
# MLX90640 화소 하나는 거리 d 에서 약 0.03*d 를 덮는다. 멀어지면 화소마다
# 얼굴과 배경이 섞여 온도가 실제보다 **낮게** 나온다. 거리를 직접 재지 않고,
# 얼굴이 덮은 열화상 화소 수(= 충전율)로 되돌린다.
#
#   d_ratio   = sqrt(face_px_ref / face_px)
#   dist_term = dist_gain * (skin - ambient) * (d_ratio - 1.0)
#
# TEMP_DIST_GAIN = None 은 "temp_calib 의 기본값(0.0 = 미보정)을 쓴다" 는 뜻.
# 맞추지 않은 기울기를 켜 두는 것은 꺼 두는 것보다 나쁘다. 실측해서 정할 것:
#     python3 temp_calib.py --fit pairs.csv     # skin,ambient,face_px,ear
#     python3 temp_calib.py --demo              # 항의 크기만 미리 보기
TEMP_DIST_GAIN = None
TEMP_FACE_PX_REF = None       # 보정 당시 얼굴이 덮던 열화상 화소 수


# ══════════════════════════════════════════════════════
#  이상탐지 (Isolation Forest + robust z)
# ══════════════════════════════════════════════════════

CALIB_SEC = 60.0
MIN_SAMPLES = 25
ANOM_WINDOW, ANOM_TRIGGER = 5, 3
OUT_PCT = 1.0
MIN_CONF = 0.30
SIGNAL_LOST_SEC = 45.0

HR_SIGMA = 2.5
RR_SIGMA = 4.0
TEMP_SIGMA = 0.4

BASELINE_DIR = "baselines"    # C-8: 환자별 기준선 저장 위치
SESSION_LOG_DIR = "sessions"  # C-9: 세션 CSV 저장 위치


# ══════════════════════════════════════════════════════
#  원격 전송 (오라클 서버)
# ══════════════════════════════════════════════════════

# 세션 CSV 를 측정 중 실시간으로 서버에 append 한다. --remote 로 켠다.
# 전송이 실패해도 계측은 계속되고, 로컬 CSV 가 항상 원본으로 남는다.
REMOTE_HOST = "161.33.204.105"
REMOTE_PORT = 22
REMOTE_KEY = "ssh-key-2026-07-29.key"

# 오라클 클라우드 이미지마다 기본 계정이 다르다(Oracle Linux=opc,
# Ubuntu=ubuntu). 순서대로 접속을 시도하고 성공한 계정을 기억한다.
# --remote-user 로 직접 지정하면 탐색을 건너뛴다.
REMOTE_USERS = ("opc", "ubuntu", "oracle", "root")

REMOTE_DIR = "vitals"         # 서버 홈 기준 상대경로
REMOTE_INTERVAL = 5.0         # 전송 주기(초). 짧을수록 실시간, 길수록 부하 적음
REMOTE_TIMEOUT = 15.0         # ssh 한 번의 제한 시간


# ══════════════════════════════════════════════════════
#  얼굴 추적 모터 (Dynamixel XL330)
# ══════════════════════════════════════════════════════

DXL_PORT = "/dev/ttyACM0"

# ── B-6: 통신 속도 ──
# XL330 은 1Mbps 를 지원하고, 올리면 20Hz SyncWrite + 0.5s 피드백의 시리얼
# 점유가 약 1/17 로 줄어 A-4 의 타임아웃 위험도 함께 낮아진다.
# 다만 서보 EEPROM 의 Baud Rate 를 먼저 바꿔야 하므로 기본값은 그대로 두고,
# 설정한 속도로 응답이 없으면 아래 목록을 자동 탐색한 뒤 어떤 속도로 붙었는지
# 로그에 남긴다. 실제로 올리려면:
#   1) Dynamixel Wizard 2.0 으로 ID1/ID2 의 Baud Rate 를 1000000 으로 변경
#   2) 여기 DXL_BAUDRATE 를 1000000 으로 변경 (또는 --dxl-baud 1000000)
DXL_BAUDRATE = 57600
DXL_BAUD_FALLBACKS = (57600, 1000000, 115200, 2000000, 9600)

# B-7: 예전에는 포트 오픈 후 무조건 time.sleep(2) 였고, 그게 카메라 프레임
# 루프 안에서 돌아 rppg 큐를 2초간 밀리게 했다. 이제는 모델 번호가 읽힐
# 때까지만 폴링한다.
DXL_READY_TIMEOUT = 2.0
DXL_READY_POLL = 0.05

PAN_SPAN, TILT_SPAN = 900, 400      # 시작 위치 기준 +-틱
PROFILE_VELOCITY = 5                # 약 78 tick/s (~6.9 deg/s). 안전 우선값
PROFILE_ACCELERATION = 2
MOTOR_UPDATE_SEC = 0.05             # 20Hz

PAN_PID = dict(kp=250.0, ki=4.0, kd=2.5)
TILT_PID = dict(kp=200.0, ki=3.0, kd=2.0)
AXIS_MAX_SPEED = 70.0               # tick/s. PROFILE_VELOCITY 상한과 맞춘 값
AXIS_MAX_ACCEL = 200.0
DEADZONE_ENTER, DEADZONE_EXIT = 0.04, 0.07

# ── B-2: 얼굴 검증용 LK 파라미터 ──
# 얼굴은 프레임 간 변위가 작아서 rr.py 만큼 큰 창/피라미드가 필요 없다.
# (21,21)/3 -> (15,15)/2 로 낮추면 Pi 에서 눈에 띄게 싸진다.
FLOW_LK_WINSIZE = (15, 15)
FLOW_LK_MAXLEVEL = 2
FLOW_MAX_CORNERS = 60
FLOW_MIN_POINTS = 8
FLOW_FB_ERR_PX = 1.5

# ── C-6: flow 품질 정의와 임계 ──
# 예전 quality = inlier / 시드된 코너 수 였다. 저조도나 매끈한 얼굴에서
# goodFeaturesToTrack 이 60개를 못 채우면 그것만으로 점수가 깎여
# FLOW_UNCERTAIN 으로 모터가 계속 멈췄다. 이제는
#     quality = 합의율 x (0.5 + 0.5 x 추적생존율)
# 로 두 성분을 분리한다. 임계는 로그(of q= 항목)를 보고 조정할 것.
FLOW_MIN_QUALITY = 0.45
FLOW_MIN_INLIER_FRACTION = 0.55
FLOW_MAX_DISAGREEMENT = 0.35        # 얼굴 크기 대비 검출-플로우 중심 불일치

# ── A-4: 통신 실패 내성 ──
# 예전에는 이동 중 위치 읽기가 단 1회 실패해도 즉시 fault -> RuntimeError ->
# 워커 스레드 종료 -> HR/RR/체온 전부 정지였다. USB 시리얼 글리치 한 번에
# 시연이 끝난다. 이제 연속 실패가 이 횟수를 넘어야 fault 로 본다.
MAX_COMM_FAILURES = 3
MAX_FEEDBACK_FAILURES = 3
MAX_MOTION_FEEDBACK_FAILURES = 3
FEEDBACK_INTERVAL = 0.50
MAX_TRACKING_ERROR_TICKS = 180
MAX_STALL_SAMPLES = 4

ACQUIRE_FRAMES = 1                  # 원래 8. 첫 검출부터 바로 움직인다
LOST_FRAMES = 6


# ══════════════════════════════════════════════════════
#  모터 동작 게이트  (강제 동작 모드)
# ══════════════════════════════════════════════════════
#
# 아래는 원래 "이 조건이 아니면 모터를 움직이지 않는다" 는 안전 게이트였다.
# 요청에 따라 기본값을 전부 '통과' 쪽으로 열어 두었다. 각 줄의 주석에 원래
# 값이 있으니 되돌리려면 그 값으로 바꾸면 된다. 코드는 지우지 않았다.
#
# ★ 딱 하나만 다시 켜야 한다면 MOTOR_REQUIRE_SIGN 이다 ★
#   장착 방향(부호)이 반대면 PID 가 음의 피드백이 아니라 **양의 피드백**이
#   된다. 모터가 얼굴을 쫓는 게 아니라 얼굴 반대로 돌고, 가동 한계에 박힌
#   채 계속 토크를 준다. 이건 "추적이 잘 안 되는" 게 아니라 고장이다.
#   지금은 부호를 안 주면 아래 기본값으로 진행하되 경고를 크게 찍는다.
#   face_tracker_motor.py --selftest 로 한 번만 확인해 두면 된다.

MOTOR_REQUIRE_SIGN = False          # 원래 True
MOTOR_DEFAULT_PAN_SIGN = 1.0        # 부호 미지정 시 사용할 값
MOTOR_DEFAULT_TILT_SIGN = 1.0

# 카메라 해상도가 640x480 이 아니어도 추적을 켠다.
MOTOR_REQUIRE_FRAME_SIZE = False    # 원래 True

# 광학흐름 합의 검증 없이도 움직인다. 검출기가 잡은 박스를 그대로 믿는다.
# (오검출을 쫓아갈 수 있다는 뜻이다)
MOTOR_REQUIRE_FLOW = False          # 원래 True

# 얼굴 검출기가 잠깐 놓쳐도 마지막 optical-flow 궤적을 이어가는 시간(초).
# 너무 길면 사람을 완전히 놓친 뒤에도 잘못된 방향으로 갈 수 있으므로 짧게 둔다.
MOTOR_FACE_LOST_GRACE_SEC = 0.8

# 얼굴 크기/종횡비/급격한 점프 검사를 건너뛴다.
MOTOR_STRICT_GEOMETRY = False       # 원래 True

# 지원 모델(XL330)이 아니어도, 레지스터 설정이 일부 실패해도 진행한다.
MOTOR_REQUIRE_MODEL_MATCH = False   # 원래 True
MOTOR_REQUIRE_SETUP_OK = False      # 원래 True

# 통신/추종 오류를 '영구 중단'으로 바꾸지 않는다. 카운터를 리셋하고 계속
# 시도한다. 서보가 막혔을 때(SERVO_NOT_FOLLOWING)는 목표를 실제 위치로
# 되돌려 물린 채로 계속 힘 주는 상태를 푼 뒤 이어서 추적한다.
MOTOR_FAULT_DISABLES = False        # 원래 True
MOTOR_RETRY_SEC = 10.0              # 초기화/연결 실패 시 재시도 간격


# ══════════════════════════════════════════════════════
#  대시보드 (Tk)
# ══════════════════════════════════════════════════════

DASH_REFRESH_MS = 200
# B-3: 파형 포인트 수. 예전에는 260 포인트 x 3겹 x 3패널 을 200ms 마다
# delete/create 했다. 이제 아이템을 재사용하고 포인트도 줄인다.
WAVE_POINTS = 120
WAVE_LAYERS = ((7, 0.28), (4, 0.62), (2, 1.0))   # (두께, 색상 혼합비)

HR_DISPLAY_RANGE = (60, 100)
RR_DISPLAY_RANGE = (12, 20)
TEMP_DISPLAY_RANGE = (36.1, 37.2)
