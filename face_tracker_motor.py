"""
face_tracker_motor.py

얼굴 박스(box) 좌표를 받아서 팬(ID 1) / 틸트(ID 2) 다이나믹셀 서보를
PID로 제어해 얼굴이 항상 화면 중앙에 오도록 카메라(또는 헤드)를
움직이는 모듈.

vital_monitor.py 쪽에서는 아래처럼 3줄만 추가하면 됩니다:

    from face_tracker_motor import FaceTrackerMotor
    tracker = FaceTrackerMotor()                     # 루프 시작 전 1회
    ...
    tracker.update(box, frame_rgb.shape)              # 매 프레임, box 나온 직후
    ...
    tracker.close()                                   # 종료 시
"""

from dynamixel_sdk import *
import time


class PID:
    """단순 PID 컨트롤러 (한 축용)."""

    def __init__(self, kp, ki, kd, output_limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit

        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time = None

    def compute(self, error, now=None):
        now = time.time() if now is None else now

        if self._prev_time is None:
            dt = 0.0
        else:
            dt = now - self._prev_time
        self._prev_time = now

        self._integral += error * dt
        derivative = 0.0 if dt <= 0 else (error - self._prev_error) / dt
        self._prev_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        return max(-self.output_limit, min(self.output_limit, output))


class FaceTrackerMotor:
    """얼굴 box를 받아 팬/틸트 서보를 PID로 구동한다."""

    # ===== 환경에 맞게 수정 =====
    DEVICENAME = '/dev/ttyACM0'
    BAUDRATE = 57600
    PROTOCOL_VERSION = 2.0

    PAN_ID = 1
    TILT_ID = 2

    ADDR_TORQUE_ENABLE = 64
    ADDR_GOAL_POSITION = 116
    ADDR_PRESENT_POSITION = 132

    CENTER_POSITION = 2048     # 정면을 바라보는 중립 위치 (필요시 실측값으로 수정)
    # 서보가 과회전해서 케이블이 꼬이지 않도록 이동 가능 범위를 제한
    PAN_MIN, PAN_MAX = 1024, 3072
    TILT_MIN, TILT_MAX = 1536, 2560

    # PID 게인 - 처음엔 보수적으로 시작해서 튜닝 권장
    PAN_PID_GAINS = dict(kp=0.15, ki=0.0, kd=0.02, output_limit=60)
    TILT_PID_GAINS = dict(kp=0.15, ki=0.0, kd=0.02, output_limit=60)

    MIN_WRITE_INTERVAL = 0.05  # 초당 최대 20회로 시리얼 쓰기 제한

    def __init__(self, enable=True):
        self.enabled = enable
        self._last_write_time = 0.0
        self._pan_pos = self.CENTER_POSITION
        self._tilt_pos = self.CENTER_POSITION

        if not enable:
            return

        self.portHandler = PortHandler(self.DEVICENAME)
        self.packetHandler = PacketHandler(self.PROTOCOL_VERSION)

        if not self.portHandler.openPort():
            print(f"[FaceTrackerMotor] 포트 열기 실패: {self.DEVICENAME}")
            self.enabled = False
            return

        if not self.portHandler.setBaudRate(self.BAUDRATE):
            print(f"[FaceTrackerMotor] 보레이트 설정 실패: {self.BAUDRATE}")
            self.enabled = False
            return

        time.sleep(2)  # 보드 리셋/부팅 대기

        for dxl_id in (self.PAN_ID, self.TILT_ID):
            result, error = self.packetHandler.write1ByteTxRx(
                self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE, 1
            )
            if result != COMM_SUCCESS or error != 0:
                print(f"[FaceTrackerMotor] [ID {dxl_id}] 토크 ON 실패")

        # 시작 시 중앙으로 정렬
        self._write_position(self.PAN_ID, self.CENTER_POSITION)
        self._write_position(self.TILT_ID, self.CENTER_POSITION)

        self.pan_pid = PID(**self.PAN_PID_GAINS)
        self.tilt_pid = PID(**self.TILT_PID_GAINS)

        print("[FaceTrackerMotor] 초기화 완료, 중앙 위치로 정렬")

    def _write_position(self, dxl_id, position):
        if not self.enabled:
            return
        self.packetHandler.write4ByteTxRx(
            self.portHandler, dxl_id, self.ADDR_GOAL_POSITION, int(position)
        )

    def update(self, box, frame_shape):
        """
        box: ((y1, y2), (x1, x2)) - 얼굴이 없으면 None
        frame_shape: frame_rgb.shape (h, w, ...)
        """
        if not self.enabled:
            return

        if box is None:
            # 얼굴이 안 보이면 PID 누적 오차 리셋 (얼굴 다시 나타났을 때
            # 갑자기 튀는 움직임 방지)
            self.pan_pid.reset()
            self.tilt_pid.reset()
            return

        now = time.time()
        if now - self._last_write_time < self.MIN_WRITE_INTERVAL:
            return  # 너무 자주 쓰지 않도록 제한

        h, w = frame_shape[0], frame_shape[1]
        (y1, y2), (x1, x2) = box

        face_cx = (x1 + x2) / 2.0
        face_cy = (y1 + y2) / 2.0
        frame_cx = w / 2.0
        frame_cy = h / 2.0

        # 오차: 화면 중심 - 얼굴 중심 (얼굴이 오른쪽에 있으면 음수 -> 팬을 오른쪽으로)
        error_x = frame_cx - face_cx
        error_y = frame_cy - face_cy

        pan_output = self.pan_pid.compute(error_x, now=now)
        tilt_output = self.tilt_pid.compute(error_y, now=now)

        # 화면에서 얼굴이 오른쪽(양수 x)에 있으면 pan 위치를 줄이는 방향으로 이동한다고
        # 가정 - 실제 서보 장착 방향에 따라 부호가 반대일 수 있으니 실측 후 조정
        self._pan_pos = self._pan_pos - pan_output
        self._tilt_pos = self._tilt_pos + tilt_output

        self._pan_pos = max(self.PAN_MIN, min(self.PAN_MAX, self._pan_pos))
        self._tilt_pos = max(self.TILT_MIN, min(self.TILT_MAX, self._tilt_pos))

        self._write_position(self.PAN_ID, self._pan_pos)
        self._write_position(self.TILT_ID, self._tilt_pos)

        self._last_write_time = now

    def close(self):
        if not self.enabled:
            return
        # 종료 시 중앙으로 복귀 후 포트 닫기
        self._write_position(self.PAN_ID, self.CENTER_POSITION)
        self._write_position(self.TILT_ID, self.CENTER_POSITION)
        time.sleep(0.5)
        self.portHandler.closePort()
        print("[FaceTrackerMotor] 종료, 포트 닫음")