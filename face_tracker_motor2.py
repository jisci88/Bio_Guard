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

--------------------------------------------------------------------------
속도 관련 메모
--------------------------------------------------------------------------
속도는 두 군데에서 제한된다. 둘 다 걸어야 안전하다.

  1) 하드웨어 상한: PROFILE_VELOCITY (서보 펌웨어가 강제)
     PROFILE_VELOCITY = 5  ->  5 * 0.229 rpm = 1.145 rpm = 6.87 deg/s
     이 값이 0이면 "무제한"이라 Velocity Limit(기본 445, 약 611 deg/s)까지
     튀어나간다. 이전 버전이 급발진하던 원인이 바로 이것.

  2) 소프트웨어 상한: MAX_TICKS_PER_SEC (PID 출력 클램프)
     70 ticks/s = 70/4096 rev/s = 6.15 deg/s

소프트웨어(6.15)를 하드웨어(6.87)보다 살짝 낮게 잡아서, 평소에는 PID가
속도를 지배하고 하드웨어는 안전망 역할만 하게 했다.
"""

from dynamixel_sdk import (
    PortHandler,
    PacketHandler,
    GroupSyncWrite,
    COMM_SUCCESS,
    DXL_LOBYTE,
    DXL_HIBYTE,
    DXL_LOWORD,
    DXL_HIWORD,
)
import time


class PID:
    """단순 PID 컨트롤러 (한 축용). 출력 단위는 '틱/초'."""

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

        derivative = 0.0 if dt <= 0 else (error - self._prev_error) / dt
        self._prev_error = error

        # 적분 anti-windup: 포화 상태에서는 적분을 더 쌓지 않는다.
        candidate = self._integral + error * dt
        unsat = self.kp * error + self.ki * candidate + self.kd * derivative
        if abs(unsat) <= self.output_limit:
            self._integral = candidate
            output = unsat
        else:
            output = self.kp * error + self.ki * self._integral + self.kd * derivative

        return max(-self.output_limit, min(self.output_limit, output))


class FaceTrackerMotor:
    """얼굴 box를 받아 팬/틸트 서보를 PID로 구동한다."""

    # ===== 환경에 맞게 수정 =====
    DEVICENAME = '/dev/ttyACM0'
    BAUDRATE = 57600
    PROTOCOL_VERSION = 2.0

    PAN_ID = 1   # 좌우
    TILT_ID = 2  # 상하

    # ----- XL330 컨트롤 테이블 (주소 확인 필수) -----
    ADDR_DRIVE_MODE = 10           # EEPROM, 1 byte
    ADDR_OPERATING_MODE = 11       # EEPROM, 1 byte
    ADDR_TORQUE_ENABLE = 64        # RAM,    1 byte
    ADDR_PROFILE_ACCELERATION = 108  # RAM,  4 byte
    ADDR_PROFILE_VELOCITY = 112      # RAM,  4 byte
    ADDR_GOAL_POSITION = 116       # RAM,    4 byte
    ADDR_PRESENT_POSITION = 132    # RAM,    4 byte
    LEN_GOAL_POSITION = 4

    OPERATING_MODE_POSITION = 3    # 위치 제어 모드
    DRIVE_MODE_NORMAL = 0

    # ----- 속도 제한 -----
    PROFILE_VELOCITY = 5       # 0.229 rpm 단위 -> 약 6.87 deg/s (하드웨어 상한)
    PROFILE_ACCELERATION = 2   # 214.577 rev/min^2 단위, 부드러운 가감속
    MAX_TICKS_PER_SEC = 70.0   # 약 6.15 deg/s (소프트웨어 상한)

    HOMING_PROFILE_VELOCITY = 40  # 종료 시 중앙 복귀만 조금 빠르게

    CENTER_POSITION = 2048     # 정면을 바라보는 중립 위치 (필요시 실측값으로 수정)
    # 서보가 과회전해서 케이블이 꼬이지 않도록 이동 가능 범위를 제한
    PAN_MIN, PAN_MAX = 1024, 3072
    TILT_MIN, TILT_MAX = 1536, 2560

    # PID 게인. error는 화면 반폭/반높이로 정규화된 -1.0 ~ +1.0 값이고,
    # 출력 단위는 틱/초다. kp=60이면 얼굴이 화면 가장자리에 있을 때 60 ticks/s.
    PAN_PID_GAINS = dict(kp=60.0, ki=0.0, kd=0.0, output_limit=MAX_TICKS_PER_SEC)
    TILT_PID_GAINS = dict(kp=60.0, ki=0.0, kd=0.0, output_limit=MAX_TICKS_PER_SEC)

    # 서보 장착 방향에 따라 부호가 반대일 수 있음. 실측 후 조정.
    PAN_SIGN = -1.0
    TILT_SIGN = +1.0

    # 히스테리시스 데드존 (정규화 오차 기준).
    # 정지 상태에서는 START를 넘어야 움직이기 시작하고,
    # 움직이는 중에는 STOP 아래로 들어와야 멈춘다.
    # 중앙 근처에서 서보가 미세하게 떠는 것을 막는다 (rPPG SQI에 직결).
    DEADZONE_START = 0.10
    DEADZONE_STOP = 0.04

    MIN_WRITE_INTERVAL = 0.05  # 초당 최대 20회로 시리얼 쓰기 제한

    # 목표 위치를 float으로 누적하다가, 마지막으로 '쓴' 값과 이만큼 벌어졌을 때만
    # 실제로 Goal Position을 쓴다.
    # 1~2틱짜리 명령은 정지 마찰을 못 이겨서 서보가 제자리에 멈춰 있는다.
    MIN_STEP_TICKS = 6.0

    def __init__(self, enable=True):
        self.enabled = enable
        self._last_write_time = 0.0
        self._pan_pos = float(self.CENTER_POSITION)
        self._tilt_pos = float(self.CENTER_POSITION)
        self._pan_moving = False
        self._tilt_moving = False
        self._written_pan = self._pan_pos
        self._written_tilt = self._tilt_pos

        self.pan_pid = PID(**self.PAN_PID_GAINS)
        self.tilt_pid = PID(**self.TILT_PID_GAINS)

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

        self.syncWrite = GroupSyncWrite(
            self.portHandler,
            self.packetHandler,
            self.ADDR_GOAL_POSITION,
            self.LEN_GOAL_POSITION,
        )

        for dxl_id in (self.PAN_ID, self.TILT_ID):
            self._setup_servo(dxl_id)

        # 현재 위치를 읽어서 목표값을 초기화한다.
        # 읽기에 실패했을 때 CENTER_POSITION으로 가정해 버리면, 토크를 켜는
        # 순간 서보가 2048로 달려간다. "저장된 원점으로 움직인다"의 전형적인
        # 증상이므로 추측하지 말고 중단한다.
        pan_now = self._read_position(self.PAN_ID)
        tilt_now = self._read_position(self.TILT_ID)
        if pan_now is None or tilt_now is None:
            print("[FaceTrackerMotor] 현재 위치를 못 읽음 -> 초기화 중단 (토크 OFF 유지)")
            self.enabled = False
            return

        if not (self.PAN_MIN <= pan_now <= self.PAN_MAX):
            print(
                f"[FaceTrackerMotor] 팬 현재 위치 {pan_now}가 허용 범위 "
                f"[{self.PAN_MIN}, {self.PAN_MAX}] 밖 -> 초기화 중단. "
                f"헤드를 손으로 범위 안에 맞추거나 PAN_MIN/PAN_MAX를 조정하세요."
            )
            self.enabled = False
            return
        if not (self.TILT_MIN <= tilt_now <= self.TILT_MAX):
            print(
                f"[FaceTrackerMotor] 틸트 현재 위치 {tilt_now}가 허용 범위 "
                f"[{self.TILT_MIN}, {self.TILT_MAX}] 밖 -> 초기화 중단. "
                f"헤드를 손으로 범위 안에 맞추거나 TILT_MIN/TILT_MAX를 조정하세요."
            )
            self.enabled = False
            return

        self._pan_pos = float(pan_now)
        self._tilt_pos = float(tilt_now)

        # 토크를 켜기 전에 목표를 현재 위치로 맞춰둬야 토크 ON 순간 튀지 않는다.
        self._write_positions(self._pan_pos, self._tilt_pos)

        for dxl_id in (self.PAN_ID, self.TILT_ID):
            result, error = self.packetHandler.write1ByteTxRx(
                self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE, 1
            )
            if result != COMM_SUCCESS or error != 0:
                print(f"[FaceTrackerMotor] [ID {dxl_id}] 토크 ON 실패")

        print(
            f"[FaceTrackerMotor] 초기화 완료 "
            f"(pan={self._pan_pos:.0f}, tilt={self._tilt_pos:.0f}, "
            f"최대 {self.MAX_TICKS_PER_SEC:.0f} ticks/s)"
        )

    # ------------------------------------------------------------------
    # 초기 설정
    # ------------------------------------------------------------------
    def _setup_servo(self, dxl_id):
        """EEPROM 설정을 확인/복구하고 속도 프로파일을 건다."""
        # EEPROM은 토크가 꺼져 있어야 쓸 수 있다.
        self.packetHandler.write1ByteTxRx(
            self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE, 0
        )

        # 이전 버전이 주소 10에 4바이트를 써서 Drive Mode / Operating Mode를
        # 덮어썼을 수 있다. 값이 틀렸을 때만 복구한다 (EEPROM 수명 보호).
        self._fix_eeprom_byte(
            dxl_id, self.ADDR_DRIVE_MODE, self.DRIVE_MODE_NORMAL, "Drive Mode"
        )
        self._fix_eeprom_byte(
            dxl_id,
            self.ADDR_OPERATING_MODE,
            self.OPERATING_MODE_POSITION,
            "Operating Mode",
        )

        # 속도 프로파일 (RAM). 여기가 하드웨어 속도 상한.
        self.packetHandler.write4ByteTxRx(
            self.portHandler,
            dxl_id,
            self.ADDR_PROFILE_ACCELERATION,
            self.PROFILE_ACCELERATION,
        )
        self.packetHandler.write4ByteTxRx(
            self.portHandler, dxl_id, self.ADDR_PROFILE_VELOCITY, self.PROFILE_VELOCITY
        )

    def _fix_eeprom_byte(self, dxl_id, addr, expected, name):
        value, result, error = self.packetHandler.read1ByteTxRx(
            self.portHandler, dxl_id, addr
        )
        if result != COMM_SUCCESS or error != 0:
            print(f"[FaceTrackerMotor] [ID {dxl_id}] {name} 읽기 실패")
            return
        if value == expected:
            return
        print(f"[FaceTrackerMotor] [ID {dxl_id}] {name} {value} -> {expected} 복구")
        self.packetHandler.write1ByteTxRx(self.portHandler, dxl_id, addr, expected)
        time.sleep(0.05)  # EEPROM 쓰기 반영 대기

    def _read_position(self, dxl_id):
        value, result, error = self.packetHandler.read4ByteTxRx(
            self.portHandler, dxl_id, self.ADDR_PRESENT_POSITION
        )
        if result != COMM_SUCCESS or error != 0:
            print(f"[FaceTrackerMotor] [ID {dxl_id}] 현재 위치 읽기 실패")
            return None
        if value > 0x7FFFFFFF:  # 부호 있는 32비트로 해석
            value -= 0x100000000
        return value

    # ------------------------------------------------------------------
    # 위치 쓰기
    # ------------------------------------------------------------------
    def _write_positions(self, pan_pos, tilt_pos):
        """팬/틸트를 한 패킷으로 동시에 쓴다 (상태 패킷 왕복 없음)."""
        if not self.enabled:
            return
        self.syncWrite.clearParam()
        for dxl_id, pos in ((self.PAN_ID, pan_pos), (self.TILT_ID, tilt_pos)):
            pos = int(pos)
            param = [
                DXL_LOBYTE(DXL_LOWORD(pos)),
                DXL_HIBYTE(DXL_LOWORD(pos)),
                DXL_LOBYTE(DXL_HIWORD(pos)),
                DXL_HIBYTE(DXL_HIWORD(pos)),
            ]
            self.syncWrite.addParam(dxl_id, param)
        self.syncWrite.txPacket()
        self.syncWrite.clearParam()
        self._written_pan = float(pan_pos)
        self._written_tilt = float(tilt_pos)

    # ------------------------------------------------------------------
    # 메인 루프에서 호출
    # ------------------------------------------------------------------
    def update(self, box, frame_shape):
        """
        box: ((y1, y2), (x1, x2)) - 얼굴이 없으면 None
        frame_shape: frame_rgb.shape (h, w, ...)
        """
        if not self.enabled:
            return

        if box is None:
            # 얼굴이 안 보이면 PID 누적 오차와 이동 상태를 리셋한다.
            # (얼굴이 다시 나타났을 때 갑자기 튀는 것 방지)
            self.pan_pid.reset()
            self.tilt_pid.reset()
            self._pan_moving = False
            self._tilt_moving = False
            return

        now = time.time()
        dt = now - self._last_write_time
        if dt < self.MIN_WRITE_INTERVAL:
            return  # 너무 자주 쓰지 않도록 제한
        if dt > 1.0:
            dt = self.MIN_WRITE_INTERVAL  # 오래 멈췄다 돌아온 경우 튐 방지

        h, w = frame_shape[0], frame_shape[1]
        (y1, y2), (x1, x2) = box

        face_cx = (x1 + x2) / 2.0
        face_cy = (y1 + y2) / 2.0

        # 화면 반폭/반높이로 정규화 -> 해상도가 바뀌어도 게인을 다시 안 잡아도 된다.
        error_x = (w / 2.0 - face_cx) / (w / 2.0)
        error_y = (h / 2.0 - face_cy) / (h / 2.0)

        # 히스테리시스 데드존
        self._pan_moving = self._apply_deadzone(
            self._pan_moving, error_x, self.pan_pid
        )
        self._tilt_moving = self._apply_deadzone(
            self._tilt_moving, error_y, self.tilt_pid
        )

        # PID 출력은 '틱/초'. dt를 곱해서 실제 이동량으로 바꾼다.
        if self._pan_moving:
            vel_x = self.pan_pid.compute(error_x, now=now)
            self._pan_pos += self.PAN_SIGN * vel_x * dt
        if self._tilt_moving:
            vel_y = self.tilt_pid.compute(error_y, now=now)
            self._tilt_pos += self.TILT_SIGN * vel_y * dt

        self._pan_pos = max(self.PAN_MIN, min(self.PAN_MAX, self._pan_pos))
        self._tilt_pos = max(self.TILT_MIN, min(self.TILT_MAX, self._tilt_pos))

        if (
            abs(self._pan_pos - self._written_pan) >= self.MIN_STEP_TICKS
            or abs(self._tilt_pos - self._written_tilt) >= self.MIN_STEP_TICKS
        ):
            self._write_positions(self._pan_pos, self._tilt_pos)

        self._last_write_time = now

    def _apply_deadzone(self, moving, error, pid):
        mag = abs(error)
        if moving:
            if mag < self.DEADZONE_STOP:
                pid.reset()
                return False
            return True
        if mag > self.DEADZONE_START:
            pid.reset()  # 시간 기준을 새로 잡아 첫 스텝의 dt 폭주 방지
            return True
        return False

    def is_moving(self):
        """서보가 움직이는 중인지. rPPG 샘플 무효화(motion-hold)에 사용."""
        return self._pan_moving or self._tilt_moving

    # ------------------------------------------------------------------
    # 종료
    # ------------------------------------------------------------------
    def close(self):
        if not self.enabled:
            return
        # 추적 속도는 매우 느리므로 복귀 때만 프로파일 속도를 올린다.
        for dxl_id in (self.PAN_ID, self.TILT_ID):
            self.packetHandler.write4ByteTxRx(
                self.portHandler,
                dxl_id,
                self.ADDR_PROFILE_VELOCITY,
                self.HOMING_PROFILE_VELOCITY,
            )
        self._write_positions(self.CENTER_POSITION, self.CENTER_POSITION)

        deadline = time.time() + 5.0
        while time.time() < deadline:
            pan_now = self._read_position(self.PAN_ID)
            tilt_now = self._read_position(self.TILT_ID)
            if pan_now is None or tilt_now is None:
                break
            if (
                abs(pan_now - self.CENTER_POSITION) < 20
                and abs(tilt_now - self.CENTER_POSITION) < 20
            ):
                break
            time.sleep(0.1)

        for dxl_id in (self.PAN_ID, self.TILT_ID):
            self.packetHandler.write1ByteTxRx(
                self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE, 0
            )
        self.portHandler.closePort()
        print("[FaceTrackerMotor] 종료, 포트 닫음")


if __name__ == "__main__":
    # 테스트용: 얼굴이 화면 왼쪽 위에 치우쳐 있다고 가정하고
    # 서보가 얼마나 '느리게' 따라가는지 눈으로 확인한다.
    tracker = FaceTrackerMotor(enable=True)
    frame_shape = (480, 640, 3)  # h, w, c
    box = ((80, 200), (100, 220))  # (y1, y2), (x1, x2) - 중앙에서 벗어남

    try:
        t0 = time.time()
        while True:
            tracker.update(box, frame_shape)
            print(
                f"t={time.time() - t0:5.1f}s  "
                f"pan={tracker._pan_pos:7.1f}  tilt={tracker._tilt_pos:7.1f}  "
                f"moving={tracker.is_moving()}"
            )
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        tracker.close()
