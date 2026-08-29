"""
face_tracker_motor.py

얼굴 ROI 박스를 받아 팬(ID 1, 좌우) / 틸트(ID 2, 상하) 다이나믹셀 서보를
움직여 얼굴이 화면 중앙에 오도록 카메라를 향하게 한다.

vital_monitor.py 연동:

    from face_tracker_motor import FaceTrackerMotor
    tracker = FaceTrackerMotor(                   # 루프 시작 전 1회
        pan_sign=1, tilt_sign=-1)                 # selftest로 검증한 값
    tracker.update(box, frame_rgb.shape, frame_rgb)  # 매 프레임
    tracker.close()                               # 종료 시

하드웨어 점검:

    python3 face_tracker_motor.py --selftest

--------------------------------------------------------------------------
가동 범위
--------------------------------------------------------------------------
2048이 중립이라고 가정하지 않는다. 시작할 때 읽은 실제 위치를 원점으로
삼고 거기서 +-PAN_SPAN / +-TILT_SPAN 만큼을 가동 범위로 잡는다.

이렇게 하면 서보가 어디에 있든 시작 시 움직이지 않고, 가동 범위가 항상
현재 위치를 포함한다. 고정 범위를 쓰면 서보가 그 밖에 있을 때 시작하자마자
한계값으로 끌려가고 거기 붙어서 추적이 불가능해진다.

--------------------------------------------------------------------------
추적 안전 구조
--------------------------------------------------------------------------
  - 8프레임 연속 검증 후에만 LOCKED
  - 박스 기하/점프/크기 변화 및 전후방 LK 광학 흐름 합의 검증
  - alpha-beta 위치·속도 필터와 짧은 지연 예측
  - 시간 기반 PID, 데드존 히스테리시스, 속도·가속도 제한

통신 장애 또는 이동 중 위치 피드백 단절이면 fail-closed로 양축 토크를 끈다.
"""

import argparse
import time
from dataclasses import dataclass
from math import isfinite
from statistics import median

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

import config
from config import get_logger

# D-3: 런타임 진단은 로깅으로 보낸다. 짧은 별칭을 쓰는 이유는 기존 print 문의
# 줄바꿈 정렬을 그대로 유지하기 위해서다. selftest() 의 출력은 프로그램의
# 실제 결과물이므로 print 그대로 둔다.
log = get_logger("trk")
warn = log.warning
info = log.info


@dataclass(frozen=True)
class TargetDecision:
    allowed: bool
    state: str
    box: object = None
    confidence: float = 0.0
    reason: str = ""


class AlphaBetaFilter:
    """Small constant-velocity filter suitable for real-time image coordinates."""

    def __init__(self, alpha=0.55, beta=0.08):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.value = None
        self.velocity = 0.0
        self._last_time = None

    def reset(self):
        self.value = None
        self.velocity = 0.0
        self._last_time = None

    def update(self, measurement, now):
        measurement = float(measurement)
        now = float(now)
        if self.value is None or self._last_time is None:
            self.value = measurement
            self.velocity = 0.0
            self._last_time = now
            return self.value

        dt = max(1e-3, min(0.25, now - self._last_time))
        prediction = self.value + self.velocity * dt
        residual = measurement - prediction
        self.value = prediction + self.alpha * residual
        self.velocity += self.beta * residual / dt
        self._last_time = now
        return self.value

    def predict(self, horizon):
        if self.value is None:
            return None
        return self.value + self.velocity * max(0.0, float(horizon))


class RobustFlowEstimator:
    """Estimate face translation from the median consensus of tracked points."""

    def __init__(
        self,
        min_points=None,
        min_inlier_radius_px=1.5,
        max_residual_px=6.0,
        min_inlier_fraction=None,
    ):
        if min_points is None:
            min_points = config.FLOW_MIN_POINTS
        if min_inlier_fraction is None:
            min_inlier_fraction = config.FLOW_MIN_INLIER_FRACTION
        self.min_points = max(1, int(min_points))
        self.min_inlier_radius_px = float(min_inlier_radius_px)
        self.max_residual_px = max(
            self.min_inlier_radius_px,
            float(max_residual_px),
        )
        self.min_inlier_fraction = max(
            0.0,
            min(1.0, float(min_inlier_fraction)),
        )
        # 진단용: 마지막 추정의 두 성분 (로그 of= 항목으로 노출된다)
        self.last_track_fraction = 0.0
        self.last_inlier_fraction = 0.0

    def estimate(self, previous, current, status, box):
        samples = []
        total = min(len(previous), len(current), len(status))
        for old, new, good in zip(previous, current, status):
            if good:
                samples.append((float(new[0] - old[0]), float(new[1] - old[1])))

        # C-6: 추적 생존율과 합의율은 서로 다른 것을 뜻한다. 예전에는 quality 를
        # inlier / 시드된 코너 수 하나로 계산해서, 저조도나 매끈한 얼굴이라
        # goodFeaturesToTrack 이 코너를 적게 잡았다는 이유만으로 점수가 깎이고
        # FLOW_UNCERTAIN 이 떠 모터가 계속 멈춰 있었다. 두 성분을 분리한다.
        self.last_track_fraction = len(samples) / float(max(1, total))
        self.last_inlier_fraction = 0.0

        if len(samples) < self.min_points:
            return None, 0.0

        dx = median(item[0] for item in samples)
        dy = median(item[1] for item in samples)
        residuals = [
            ((item_dx - dx) ** 2 + (item_dy - dy) ** 2) ** 0.5
            for item_dx, item_dy in samples
        ]
        residual_median = median(residuals)
        residual_mad = median(abs(value - residual_median) for value in residuals)
        radius = min(
            self.max_residual_px,
            max(
                self.min_inlier_radius_px,
                residual_median + 2.5 * 1.4826 * residual_mad,
            ),
        )
        inlier_samples = [
            sample
            for sample, residual in zip(samples, residuals)
            if residual <= radius
        ]
        inlier_fraction = len(inlier_samples) / float(len(samples))
        self.last_inlier_fraction = inlier_fraction
        if (
            len(inlier_samples) < self.min_points
            or inlier_fraction < self.min_inlier_fraction
        ):
            return None, 0.0

        dx = median(item[0] for item in inlier_samples)
        dy = median(item[1] for item in inlier_samples)

        (y1, y2), (x1, x2) = box
        predicted = ((y1 + dy, y2 + dy), (x1 + dx, x2 + dx))
        # 합의율이 주된 신호, 추적 생존율은 감쇠 계수. 생존율이 0이어도 점수는
        # 합의율의 절반까지만 떨어지므로 코너 수 부족만으로 게이트가 닫히지 않는다.
        quality = inlier_fraction * (0.5 + 0.5 * self.last_track_fraction)
        return predicted, quality


class SparseOpticalFlow:
    """One-frame forward/backward LK validation of the detected face ROI."""

    def __init__(
        self,
        cv2_module=None,
        numpy_module=None,
        max_corners=None,
        min_points=None,
        fb_error_px=None,
        lost_grace_sec=None,
    ):
        if max_corners is None:
            max_corners = config.FLOW_MAX_CORNERS
        if min_points is None:
            min_points = config.FLOW_MIN_POINTS
        if fb_error_px is None:
            fb_error_px = config.FLOW_FB_ERR_PX
        if lost_grace_sec is None:
            lost_grace_sec = config.MOTOR_FACE_LOST_GRACE_SEC
        self.lost_grace_sec = max(0.0, float(lost_grace_sec))
        self.cv2 = cv2_module
        self.np = numpy_module
        if cv2_module is None or numpy_module is None:
            try:
                if cv2_module is None:
                    import cv2 as imported_cv2

                    self.cv2 = imported_cv2
                if numpy_module is None:
                    import numpy as imported_numpy

                    self.np = imported_numpy
            except ImportError:
                self.cv2 = None
                self.np = None
        elif cv2_module is False or numpy_module is False:
            self.cv2 = None
            self.np = None

        self.available = self.cv2 is not None and self.np is not None
        self.max_corners = max(8, int(max_corners))
        self.fb_error_px = float(fb_error_px)
        self.estimator = RobustFlowEstimator(min_points=min_points)
        self._previous_gray = None
        self._previous_points = None
        self._previous_box = None
        self._last_detector_time = None

        # B-2: 얼굴은 프레임 간 변위가 작아 rr.py 만큼 큰 창/피라미드가 필요
        # 없다. (21,21)/3 -> (15,15)/2 로 낮추면 Pi 에서 눈에 띄게 싸진다.
        # 매 프레임 dict 를 새로 만들 이유도 없으므로 여기서 한 번만 만든다.
        self._lk = None
        if self.available:
            self._lk = dict(
                winSize=tuple(config.FLOW_LK_WINSIZE),
                maxLevel=int(config.FLOW_LK_MAXLEVEL),
                criteria=(
                    self.cv2.TERM_CRITERIA_EPS | self.cv2.TERM_CRITERIA_COUNT,
                    20,
                    0.01,
                ),
            )

    def reset(self):
        self._previous_gray = None
        self._previous_points = None
        self._previous_box = None
        self._last_detector_time = None

    @staticmethod
    def _point_pairs(points):
        pairs = []
        for point in points:
            value = point
            try:
                if len(value) == 1:
                    value = value[0]
            except TypeError:
                pass
            pairs.append((float(value[0]), float(value[1])))
        return pairs

    def _seed(self, gray, box):
        h, w = gray.shape[:2]
        (y1, y2), (x1, x2) = box
        x1 = max(0, min(w - 1, int(x1)))
        x2 = max(x1 + 1, min(w, int(x2)))
        y1 = max(0, min(h - 1, int(y1)))
        y2 = max(y1 + 1, min(h, int(y2)))
        inset_x = int((x2 - x1) * 0.10)
        inset_y = int((y2 - y1) * 0.10)
        crop_x1, crop_x2 = x1 + inset_x, x2 - inset_x
        crop_y1, crop_y2 = y1 + inset_y, y2 - inset_y
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
            return None
        crop = gray[crop_y1:crop_y2, crop_x1:crop_x2]
        points = self.cv2.goodFeaturesToTrack(
            crop,
            mask=None,
            maxCorners=self.max_corners,
            qualityLevel=0.01,
            minDistance=6,
            blockSize=7,
        )
        if points is None:
            return None
        points = points.copy()
        for point in points:
            value = point[0] if len(point) == 1 else point
            value[0] += crop_x1
            value[1] += crop_y1
        return points

    def update(self, frame_rgb, box, frame_gray=None, now=None):
        if not self.available:
            return None, None
        if frame_rgb is None and frame_gray is None:
            self.reset()
            return None, None

        now = time.monotonic() if now is None else float(now)
        detector_visible = box is not None
        if detector_visible:
            self._last_detector_time = now
        elif (
            self._previous_gray is None
            or self._previous_points is None
            or self._previous_box is None
            or self._last_detector_time is None
            or now - self._last_detector_time > self.lost_grace_sec
        ):
            self.reset()
            return None, None

        try:
            gray = (
                frame_gray
                if frame_gray is not None
                else self.cv2.cvtColor(frame_rgb, self.cv2.COLOR_RGB2GRAY)
            )

            # 검출기가 살아 있으면 detector ROI에서 새 특징점을 심고,
            # 잠깐 놓친 동안에는 직전 optical-flow 박스를 이어서 사용한다.
            seed_box = box if detector_visible else self._previous_box
            if self._previous_gray is None:
                self._previous_gray = gray
                self._previous_points = self._seed(gray, seed_box)
                self._previous_box = seed_box
                return None, None

            if self._previous_points is None or len(self._previous_points) == 0:
                result = (self._previous_box, 0.0)
            else:
                lk = self._lk
                forward, forward_status, _ = self.cv2.calcOpticalFlowPyrLK(
                    self._previous_gray,
                    gray,
                    self._previous_points,
                    None,
                    **lk,
                )
                if forward is None or forward_status is None:
                    result = (self._previous_box, 0.0)
                else:
                    backward, backward_status, _ = self.cv2.calcOpticalFlowPyrLK(
                        gray,
                        self._previous_gray,
                        forward,
                        None,
                        **lk,
                    )
                    previous_pairs = self._point_pairs(self._previous_points)
                    forward_pairs = self._point_pairs(forward)
                    backward_pairs = (
                        [] if backward is None else self._point_pairs(backward)
                    )
                    status_forward = forward_status.reshape(-1).tolist()
                    status_backward = (
                        []
                        if backward_status is None
                        else backward_status.reshape(-1).tolist()
                    )
                    valid = []
                    for index, old in enumerate(previous_pairs):
                        good = (
                            index < len(status_forward)
                            and index < len(status_backward)
                            and bool(status_forward[index])
                            and bool(status_backward[index])
                            and index < len(backward_pairs)
                        )
                        if good:
                            back = backward_pairs[index]
                            fb_error = (
                                (back[0] - old[0]) ** 2 + (back[1] - old[1]) ** 2
                            ) ** 0.5
                            good = fb_error <= self.fb_error_px
                        valid.append(good)
                    result = self.estimator.estimate(
                        previous_pairs,
                        forward_pairs,
                        valid,
                        self._previous_box,
                    )

            predicted_box, quality = result
            next_box = box if detector_visible else predicted_box
            if next_box is None:
                next_box = self._previous_box
            self._previous_gray = gray
            self._previous_points = self._seed(gray, next_box)
            self._previous_box = next_box
            return result
        except Exception:
            previous_box = self._previous_box
            # 짧은 검출 유실 중에는 상태를 지워버리지 않는다. 다음 프레임에서
            # 다시 optical flow를 시도할 수 있게 마지막 박스를 유지한다.
            if detector_visible:
                self.reset()
            return previous_box, 0.0


class SafeAxisController:
    """Time-based PID controller with a quiet hysteresis band around center."""

    def __init__(
        self,
        kp=85.0,
        ki=4.0,
        kd=2.5,
        integral_limit=0.25,
        derivative_alpha=0.75,
        deadzone_enter=None,
        deadzone_exit=None,
        max_speed=None,
        max_acceleration=None,
    ):
        # D-2: 값을 주지 않으면 config.py 를 따른다.
        if deadzone_enter is None:
            deadzone_enter = config.DEADZONE_ENTER
        if deadzone_exit is None:
            deadzone_exit = config.DEADZONE_EXIT
        if max_speed is None:
            max_speed = config.AXIS_MAX_SPEED
        if max_acceleration is None:
            max_acceleration = config.AXIS_MAX_ACCEL
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.integral_limit = abs(float(integral_limit))
        self.derivative_alpha = max(0.0, min(1.0, float(derivative_alpha)))
        self.deadzone_enter = float(deadzone_enter)
        self.deadzone_exit = float(deadzone_exit)
        self.max_speed = abs(float(max_speed))
        self.max_acceleration = abs(float(max_acceleration))
        self._centered = True
        self._speed = 0.0
        self.integral = 0.0
        self._last_error = None
        self._derivative = 0.0

    def reset(self):
        self._centered = True
        self._speed = 0.0
        self.integral = 0.0
        self._last_error = None
        self._derivative = 0.0

    def update(self, error, dt, confidence=1.0):
        error = float(error)
        dt = max(1e-3, float(dt))
        confidence = max(0.0, min(1.0, float(confidence)))

        if self._centered:
            if abs(error) < self.deadzone_exit:
                self._speed = 0.0
                self.integral = 0.0
                self._last_error = error
                self._derivative = 0.0
                return 0.0
            self._centered = False
        elif abs(error) <= self.deadzone_enter:
            self._centered = True
            self._speed = 0.0
            self.integral = 0.0
            self._last_error = error
            self._derivative = 0.0
            return 0.0

        self.integral = max(
            -self.integral_limit,
            min(self.integral_limit, self.integral + error * dt),
        )
        raw_derivative = (
            0.0 if self._last_error is None else (error - self._last_error) / dt
        )
        self._derivative = (
            self.derivative_alpha * self._derivative
            + (1.0 - self.derivative_alpha) * raw_derivative
        )
        self._last_error = error
        pid_speed = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * self._derivative
        ) * confidence
        desired_speed = max(
            -self.max_speed,
            min(self.max_speed, pid_speed),
        )
        max_speed_change = self.max_acceleration * dt
        speed_change = max(
            -max_speed_change,
            min(max_speed_change, desired_speed - self._speed),
        )
        self._speed += speed_change
        return self._speed * dt


class FaceTargetFilter:
    """Conservative temporal gate in front of all motor commands."""

    def __init__(
        self,
        acquire_frames=None,
        lost_frames=None,
        min_face_area_ratio=0.008,
        max_face_area_ratio=0.65,
        min_aspect_ratio=0.55,
        max_aspect_ratio=1.90,
        max_center_jump_ratio=0.08,
        max_area_change_ratio=1.80,
        min_flow_quality=None,
        require_flow=False,
        max_flow_disagreement_ratio=None,
        prediction_horizon=0.08,
        max_lead_face_ratio=0.25,
    ):
        # D-2: 값을 주지 않으면 config.py 를 따른다.
        if acquire_frames is None:
            acquire_frames = config.ACQUIRE_FRAMES
        if lost_frames is None:
            lost_frames = config.LOST_FRAMES
        if min_flow_quality is None:
            min_flow_quality = config.FLOW_MIN_QUALITY
        if max_flow_disagreement_ratio is None:
            max_flow_disagreement_ratio = config.FLOW_MAX_DISAGREEMENT
        self.acquire_frames = max(1, int(acquire_frames))
        self.lost_frames = max(1, int(lost_frames))
        self.min_face_area_ratio = float(min_face_area_ratio)
        self.max_face_area_ratio = float(max_face_area_ratio)
        self.min_aspect_ratio = float(min_aspect_ratio)
        self.max_aspect_ratio = float(max_aspect_ratio)
        self.max_center_jump_ratio = float(max_center_jump_ratio)
        self.max_area_change_ratio = float(max_area_change_ratio)
        self.min_flow_quality = float(min_flow_quality)
        self.require_flow = bool(require_flow)
        self.max_flow_disagreement_ratio = float(max_flow_disagreement_ratio)
        self.prediction_horizon = max(0.0, float(prediction_horizon))
        self.max_lead_face_ratio = max(0.0, float(max_lead_face_ratio))
        self._valid_streak = 0
        self._bad_streak = 0
        self._has_lock = False
        self._last_box = None
        self._center_x = AlphaBetaFilter()
        self._center_y = AlphaBetaFilter()
        self._width = AlphaBetaFilter(alpha=0.45, beta=0.04)
        self._height = AlphaBetaFilter(alpha=0.45, beta=0.04)
        self.state = "SEARCHING"

    def _reject(self, reason):
        self._valid_streak = 0
        self._bad_streak += 1
        if self._has_lock and self._bad_streak < self.lost_frames:
            self.state = "HOLD"
        elif self._has_lock:
            self.state = "LOST"
            self._has_lock = False
            self._last_box = None
            self._reset_motion_filters()
        else:
            self.state = "SEARCHING"
            self._last_box = None
            self._reset_motion_filters()
        return TargetDecision(False, self.state, reason=reason)

    def update(
        self,
        box,
        frame_shape,
        now=None,
        flow_box=None,
        flow_quality=None,
    ):
        now = time.monotonic() if now is None else float(now)
        if box is None:
            return self._reject("NO_FACE")

        try:
            (raw_y1, raw_y2), (raw_x1, raw_x2) = box
            y1, y2, x1, x2 = (
                float(raw_y1),
                float(raw_y2),
                float(raw_x1),
                float(raw_x2),
            )
            h, w = int(frame_shape[0]), int(frame_shape[1])
        except (TypeError, ValueError, IndexError):
            return self._reject("INVALID_BOX")
        if not all(isfinite(value) for value in (y1, y2, x1, x2)):
            return self._reject("INVALID_BOX")
        if y2 <= y1 or x2 <= x1:
            return self._reject("INVALID_BOX")
        if y1 < 0 or x1 < 0 or y2 > h or x2 > w:
            return self._reject("OUT_OF_FRAME")

        frame_area = float(frame_shape[0] * frame_shape[1])
        face_h = float(y2 - y1)
        face_w = float(x2 - x1)
        face_ratio = (face_h * face_w) / frame_area if frame_area > 0.0 else 0.0
        if face_ratio < self.min_face_area_ratio:
            return self._reject("FACE_TOO_SMALL")
        if face_ratio > self.max_face_area_ratio:
            return self._reject("FACE_TOO_LARGE")
        aspect = face_w / face_h
        if not self.min_aspect_ratio <= aspect <= self.max_aspect_ratio:
            return self._reject("INVALID_ASPECT")

        flow_check_required = self.require_flow or (
            self._has_lock and flow_quality is not None
        )
        if flow_check_required:
            if flow_quality is None:
                return self._reject("FLOW_UNAVAILABLE")
            try:
                checked_flow_quality = float(flow_quality)
            except (TypeError, ValueError):
                return self._reject("FLOW_UNCERTAIN")
            if (
                not isfinite(checked_flow_quality)
                or flow_box is None
                or checked_flow_quality < self.min_flow_quality
            ):
                return self._reject("FLOW_UNCERTAIN")

            try:
                (raw_fy1, raw_fy2), (raw_fx1, raw_fx2) = flow_box
                fy1, fy2, fx1, fx2 = (
                    float(raw_fy1),
                    float(raw_fy2),
                    float(raw_fx1),
                    float(raw_fx2),
                )
            except (TypeError, ValueError):
                return self._reject("FLOW_UNCERTAIN")
            if not all(isfinite(value) for value in (fy1, fy2, fx1, fx2)):
                return self._reject("FLOW_UNCERTAIN")
            detector_cx = (x1 + x2) * 0.5
            detector_cy = (y1 + y2) * 0.5
            flow_cx = (fx1 + fx2) * 0.5
            flow_cy = (fy1 + fy2) * 0.5
            disagreement = (
                (detector_cx - flow_cx) ** 2 + (detector_cy - flow_cy) ** 2
            ) ** 0.5
            if disagreement > max(face_w, face_h) * self.max_flow_disagreement_ratio:
                return self._reject("FLOW_MISMATCH")

        if self._last_box is not None:
            (last_y1, last_y2), (last_x1, last_x2) = self._last_box
            last_cx = (last_x1 + last_x2) * 0.5
            last_cy = (last_y1 + last_y2) * 0.5
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            distance = ((cx - last_cx) ** 2 + (cy - last_cy) ** 2) ** 0.5
            diagonal = (frame_shape[0] ** 2 + frame_shape[1] ** 2) ** 0.5
            if distance > diagonal * self.max_center_jump_ratio:
                return self._reject("TARGET_JUMP")

            last_area = float((last_y2 - last_y1) * (last_x2 - last_x1))
            area_change = max(face_h * face_w, last_area) / min(
                face_h * face_w, last_area
            )
            if area_change > self.max_area_change_ratio:
                return self._reject("SIZE_JUMP")

        self._last_box = ((y1, y2), (x1, x2))
        self._bad_streak = 0
        self._valid_streak += 1
        filtered_cx = self._center_x.update((x1 + x2) * 0.5, now)
        filtered_cy = self._center_y.update((y1 + y2) * 0.5, now)
        filtered_w = self._width.update(face_w, now)
        filtered_h = self._height.update(face_h, now)
        if self._valid_streak < self.acquire_frames:
            self.state = "VERIFYING"
            return TargetDecision(
                False,
                self.state,
                confidence=self._valid_streak / float(self.acquire_frames),
                reason="ACQUIRING",
            )

        self.state = "LOCKED"
        self._has_lock = True
        predicted_cx = self._center_x.predict(self.prediction_horizon)
        predicted_cy = self._center_y.predict(self.prediction_horizon)
        lead_x = max(
            -filtered_w * self.max_lead_face_ratio,
            min(
                filtered_w * self.max_lead_face_ratio,
                predicted_cx - filtered_cx,
            ),
        )
        lead_y = max(
            -filtered_h * self.max_lead_face_ratio,
            min(
                filtered_h * self.max_lead_face_ratio,
                predicted_cy - filtered_cy,
            ),
        )
        filtered_cx += lead_x
        filtered_cy += lead_y
        filtered_box = (
            (filtered_cy - filtered_h * 0.5, filtered_cy + filtered_h * 0.5),
            (filtered_cx - filtered_w * 0.5, filtered_cx + filtered_w * 0.5),
        )
        confidence = 1.0 if flow_quality is None else float(flow_quality)
        return TargetDecision(
            True,
            self.state,
            box=filtered_box,
            confidence=max(0.0, min(1.0, confidence)),
        )

    def _reset_motion_filters(self):
        self._center_x.reset()
        self._center_y.reset()
        self._width.reset()
        self._height.reset()


class FaceTrackerMotor:

    # ===== 환경에 맞게 수정: 값은 config.py 에 모여 있다 (D-2) =====
    DEVICENAME = config.DXL_PORT
    BAUDRATE = config.DXL_BAUDRATE
    BAUD_FALLBACKS = config.DXL_BAUD_FALLBACKS
    PROTOCOL_VERSION = 2.0

    PAN_ID = 1   # 좌우
    TILT_ID = 2  # 상하

    # ----- XL330 컨트롤 테이블 -----
    ADDR_MODEL_NUMBER = 0            # EEPROM, 2 byte, read-only
    ADDR_DRIVE_MODE = 10             # EEPROM, 1 byte
    ADDR_OPERATING_MODE = 11         # EEPROM, 1 byte
    ADDR_TORQUE_ENABLE = 64          # RAM,    1 byte
    ADDR_BUS_WATCHDOG = 98           # RAM,    1 byte (20ms unit)
    ADDR_PROFILE_ACCELERATION = 108  # RAM,    4 byte
    ADDR_PROFILE_VELOCITY = 112      # RAM,    4 byte
    ADDR_GOAL_POSITION = 116         # RAM,    4 byte
    ADDR_PRESENT_POSITION = 132      # RAM,    4 byte
    LEN_GOAL_POSITION = 4
    SUPPORTED_MODEL_NUMBERS = (1190, 1200)  # XL330-M077, XL330-M288

    OPERATING_MODE_POSITION = 3
    DRIVE_MODE_NORMAL = 0
    BUS_WATCHDOG_VALUE = 50          # 1.0s controller silence -> hardware stop

    POS_MIN, POS_MAX = 0, 4095       # 서보 절대 한계

    # ----- 속도 -----
    PROFILE_VELOCITY = config.PROFILE_VELOCITY
    PROFILE_ACCELERATION = config.PROFILE_ACCELERATION
    UPDATE_INTERVAL = config.MOTOR_UPDATE_SEC

    # ----- 가동 범위 (시작 위치 기준 상대값) -----
    PAN_SPAN = config.PAN_SPAN     # 약 +-79도
    TILT_SPAN = config.TILT_SPAN   # 약 +-35도

    # ----- 장착 방향 -----
    # 출전 장비는 PAN_SIGN/TILT_SIGN을 사전에 검증해야 한다. AUTO_SIGN=True는
    # 안전상 정상 운전 중 학습하지 않고 DIRECTION_UNCONFIRMED로 정지한다.
    AUTO_SIGN = False

    # --selftest에서 +100 tick의 실제 방향을 확인한 뒤 고정한다.
    PAN_SIGN = 1.0
    TILT_SIGN = 1.0
    MAX_COMM_FAILURES = config.MAX_COMM_FAILURES
    FEEDBACK_INTERVAL = config.FEEDBACK_INTERVAL
    MAX_FEEDBACK_FAILURES = config.MAX_FEEDBACK_FAILURES
    # A-4: 이동 중 피드백 실패도 이 횟수까지는 견딘다. 예전에는 단 1회 실패로
    # 즉시 fault -> RuntimeError -> 워커 종료 -> 계측 전체 정지였다.
    MAX_MOTION_FEEDBACK_FAILURES = config.MAX_MOTION_FEEDBACK_FAILURES
    MAX_TRACKING_ERROR_TICKS = config.MAX_TRACKING_ERROR_TICKS
    MAX_STALL_SAMPLES = config.MAX_STALL_SAMPLES

    def __init__(
        self,
        enable=True,
        verbose=True,
        require_optical_flow=True,
        device_name=None,
        pan_sign=None,
        tilt_sign=None,
        calibration_mode=False,
        enable_bus_watchdog=True,
        raise_on_fault=False,
        baudrate=None,
    ):
        self.enabled = False
        self.portHandler = None
        self.packetHandler = None
        self.syncWrite = None
        self.verbose = verbose
        self.device_name = self.DEVICENAME if device_name is None else str(device_name)
        self.baudrate = int(self.BAUDRATE if baudrate is None else baudrate)
        self.calibration_mode = bool(calibration_mode)
        self.enable_bus_watchdog = bool(enable_bus_watchdog)
        self.raise_on_fault = bool(raise_on_fault)
        self.last_error = (0.0, 0.0)
        self._moving = False
        self._last_update = 0.0
        self._written = (None, None)
        self._warned = {"pan": False, "tilt": False}
        self.require_optical_flow = bool(require_optical_flow)
        self.target_filter = FaceTargetFilter(
            require_flow=self.require_optical_flow,
        )
        self.optical_flow = SparseOpticalFlow()
        self.pan_controller = SafeAxisController(**config.PAN_PID)
        self.tilt_controller = SafeAxisController(**config.TILT_PID)
        self.tracking_state = "SEARCHING"
        self.tracking_confidence = 0.0
        self.tracking_reason = "NO_FACE"
        self.last_tracking_box = None
        self.external_evidence_source = "RPPG"
        self.external_evidence_confidence = 1.0
        self.external_evidence_age = 0.0
        self.applied_command_scale = 1.0
        self._latest_flow = None
        self._flow_track_id = None
        self._tracking_was_allowed = False
        self._comm_failures = 0
        self._faulted = False
        self._last_feedback = time.monotonic()
        self._feedback_failures = 0
        self._motion_feedback_failures = 0   # A-4
        self._stall_samples = 0
        self.torque_off_confirmed = None
        self.watchdog_armed = False

        configured_signs = {
            "pan": pan_sign,
            "tilt": tilt_sign,
        }
        parsed_signs = {}
        self._sign_ok = {}
        for axis, configured in configured_signs.items():
            try:
                value = float(configured)
            except (TypeError, ValueError):
                value = 0.0
            parsed_signs[axis] = value
            self._sign_ok[axis] = (
                not self.AUTO_SIGN and value in (-1.0, 1.0)
            )
        self.pan_sign = parsed_signs["pan"]
        self.tilt_sign = parsed_signs["tilt"]

        if not enable:
            return

        if not all(self._sign_ok.values()) and not self.calibration_mode:
            self.tracking_state = "DIRECTION_UNCONFIRMED"
            self.tracking_reason = "FIXED_SIGN_REQUIRED"
            print(
                "[TRK] PAN_SIGN/TILT_SIGN은 각각 -1 또는 +1로 "
                "사전 검증되어야 합니다 -> 하드웨어 비활성화"
            )
            return

        try:
            self.portHandler = PortHandler(self.device_name)
            self.packetHandler = PacketHandler(self.PROTOCOL_VERSION)
        except Exception as exc:
            warn(f"SDK 핸들 생성 실패: {exc}")
            self._close_port_safely()
            return

        try:
            port_opened = self.portHandler.openPort()
        except Exception as exc:
            warn(f"포트 열기 예외: {exc}")
            self._close_port_safely()
            return
        if not port_opened:
            warn(f"포트 열기 실패: {self.device_name}")
            return

        # B-6 / B-7: 설정한 속도로 응답이 없으면 후보 속도를 순서대로 시도하고,
        # 예전의 무조건 time.sleep(2) 대신 모델 번호가 실제로 읽힐 때까지만
        # 기다린다. 그 sleep 은 카메라 프레임 루프 안에서 돌면서 rppg 큐를
        # 2초간 통째로 밀리게 하고 있었다.
        if not self._negotiate_baud():
            self._close_port_safely()
            return

        try:
            self.syncWrite = GroupSyncWrite(
                self.portHandler, self.packetHandler,
                self.ADDR_GOAL_POSITION, self.LEN_GOAL_POSITION,
            )
        except Exception as exc:
            warn(f"SyncWrite 생성 실패: {exc}")
            self._close_port_safely()
            return

        setup_ok = True
        try:
            for dxl_id in (self.PAN_ID, self.TILT_ID):
                setup_ok = self._setup_servo(dxl_id) and setup_ok
        except Exception as exc:
            setup_ok = False
            warn(f"서보 설정 중 SDK 예외: {exc}")
        if not setup_ok:
            self._best_effort_shutdown()
            warn("서보 설정 검증 실패 -> 추적 비활성화")
            return

        pan_now = self._read_position(self.PAN_ID)
        tilt_now = self._read_position(self.TILT_ID)
        if pan_now is None or tilt_now is None:
            warn("현재 위치를 못 읽음 -> 추적 비활성화")
            self._best_effort_shutdown()
            return

        # 중앙으로 이동하려면 토크가 먼저 켜져 있어야 한다.
        torque_ok = True
        for dxl_id in (self.PAN_ID, self.TILT_ID):
            try:
                result, error = self.packetHandler.write1ByteTxRx(
                    self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE, 1)
            except Exception as exc:
                torque_ok = False
                warn(f"[ID {dxl_id}] 토크 ON 예외: {exc}")
                continue
            if result != COMM_SUCCESS or error != 0:
                torque_ok = False
                warn(f"[ID {dxl_id}] 토크 ON 실패 "
                      f"(result={result}, error={error})")

        if not torque_ok:
            self._best_effort_shutdown()
            warn("양축 토크 준비 실패 -> 추적 비활성화")
            return

        # Fail-closed startup: 얼굴이 확정되기 전에는 목표 위치를 쓰지 않는다.
        # 현재 위치를 안전 원점으로 사용하므로 전원을 켜도 갑자기 2048로 가지 않는다.
        self.pan_home, self.tilt_home = float(pan_now), float(tilt_now)
        self.pan_min = max(self.POS_MIN, self.pan_home - self.PAN_SPAN)
        self.pan_max = min(self.POS_MAX, self.pan_home + self.PAN_SPAN)
        self.tilt_min = max(self.POS_MIN, self.tilt_home - self.TILT_SPAN)
        self.tilt_max = min(self.POS_MAX, self.tilt_home + self.TILT_SPAN)

        self._pan_target = self.pan_home
        self._tilt_target = self.tilt_home
        self._written = (int(self.pan_home), int(self.tilt_home))

        self.enabled = True

        if self.verbose:
            info(f"준비 완료  안전 원점 pan={pan_now:.0f} tilt={tilt_now:.0f}")
            info(f"가동 범위  pan [{self.pan_min:.0f}, {self.pan_max:.0f}]"
                  f"  tilt [{self.tilt_min:.0f}, {self.tilt_max:.0f}]")

    # ---------------------------------------------------------------- setup

    def _close_port_safely(self):
        port = getattr(self, "portHandler", None)
        if port is None:
            return False
        try:
            port.closePort()
            return True
        except Exception as exc:
            warn(f"포트 닫기 예외: {exc}")
            return False

    def _best_effort_shutdown(self):
        torque_off_confirmed = True
        packet = getattr(self, "packetHandler", None)
        port = getattr(self, "portHandler", None)
        if packet is not None and port is not None:
            for dxl_id in (self.PAN_ID, self.TILT_ID):
                try:
                    result, error = packet.write1ByteTxRx(
                        port, dxl_id, self.ADDR_TORQUE_ENABLE, 0)
                    if result != COMM_SUCCESS or error != 0:
                        torque_off_confirmed = False
                except Exception as exc:
                    torque_off_confirmed = False
                    warn(f"[ID {dxl_id}] 정리 토크 OFF 예외: {exc}")
        else:
            torque_off_confirmed = False
        port_closed = self._close_port_safely()
        self.enabled = False
        self.torque_off_confirmed = torque_off_confirmed
        if not torque_off_confirmed:
            self._faulted = True
            self._moving = False
            self.tracking_state = "FAULT"
            self.tracking_confidence = 0.0
            self.tracking_reason = "INITIALIZATION_ROLLBACK_UNCONFIRMED"
            print(
                "[TRK] 초기화 롤백 토크 OFF 확인 실패 -> "
                "TORQUE_OFF_UNCONFIRMED; 외부 비상 전원 차단 필요"
            )
        return torque_off_confirmed and port_closed

    def _wait_until_ready(self, timeout):
        """
        B-7: 모델 번호가 실제로 읽힐 때까지만 기다린다.

        OpenRB-150 의 USB CDC 가 올라오는 시간은 흡수해야 하지만, 예전처럼
        무조건 2초를 잡아먹을 이유는 없다. 보통 100~300ms 안에 끝난다.
        """
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            try:
                _model, result, error = self.packetHandler.read2ByteTxRx(
                    self.portHandler, self.PAN_ID, self.ADDR_MODEL_NUMBER)
                if result == COMM_SUCCESS and error == 0:
                    return True
            except Exception:
                pass
            if time.monotonic() >= deadline:
                return False
            time.sleep(config.DXL_READY_POLL)

    def _negotiate_baud(self):
        """
        B-6: 설정한 통신 속도를 먼저 시도하고, 응답이 없으면 후보 속도를
        차례로 시도한다. 서보 EEPROM 의 Baud Rate 를 바꾼 뒤 config.py 를
        안 고쳤거나 그 반대인 경우에도 그냥 붙고, 어떤 속도로 붙었는지
        로그에 남기므로 다음 실행에서 맞춰 놓을 수 있다.
        """
        candidates = [self.baudrate]
        for value in self.BAUD_FALLBACKS:
            if int(value) not in candidates:
                candidates.append(int(value))

        for index, baud in enumerate(candidates):
            try:
                if not self.portHandler.setBaudRate(baud):
                    warn(f"보레이트 설정 실패: {baud}")
                    continue
            except Exception as exc:
                warn(f"보레이트 설정 예외 ({baud}): {exc}")
                continue

            # 설정한 속도에는 충분히, 폴백 후보에는 짧게 준다. 그렇지 않으면
            # 전원이 꺼진 서보를 상대로 후보 수 x 2초를 낭비한다.
            timeout = config.DXL_READY_TIMEOUT if index == 0 else 0.3
            if not self._wait_until_ready(timeout):
                continue

            if not all(
                self._verify_model(dxl_id)
                for dxl_id in (self.PAN_ID, self.TILT_ID)
            ):
                # 응답은 오는데 지원 모델이 아니면 속도를 바꿔도 소용없다.
                warn("지원하지 않는 모터 -> 레지스터 쓰기 없이 비활성화")
                return False

            self.baudrate = baud
            if index == 0:
                info(f"통신 속도 {baud} bps 로 연결")
            else:
                warn(
                    f"설정한 {candidates[0]} bps 는 응답이 없어 {baud} bps 로 "
                    f"연결했습니다. config.DXL_BAUDRATE 를 {baud} 으로 맞추면 "
                    "시작이 빨라집니다."
                )
            return True

        warn(f"어떤 보레이트로도 서보가 응답하지 않습니다 (시도: {candidates})")
        return False

    def _verify_model(self, dxl_id):
        try:
            model, result, error = self.packetHandler.read2ByteTxRx(
                self.portHandler, dxl_id, self.ADDR_MODEL_NUMBER)
        except Exception as exc:
            warn(f"[ID {dxl_id}] 모델 번호 읽기 예외: {exc}")
            return False
        if result != COMM_SUCCESS or error != 0:
            warn(f"[ID {dxl_id}] 모델 번호 읽기 실패")
            return False
        if model not in self.SUPPORTED_MODEL_NUMBERS:
            print(
                f"[TRK] [ID {dxl_id}] 지원하지 않는 모델 번호 {model}; "
                f"지원={self.SUPPORTED_MODEL_NUMBERS}"
            )
            return False
        if self.verbose:
            info(f"[ID {dxl_id}] XL330 모델 확인: {model}")
        return True

    def _setup_servo(self, dxl_id):
        result, error = self.packetHandler.write1ByteTxRx(
            self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE, 0)
        ok = result == COMM_SUCCESS and error == 0
        result, error = self.packetHandler.write1ByteTxRx(
            self.portHandler, dxl_id, self.ADDR_BUS_WATCHDOG, 0)
        ok = (result == COMM_SUCCESS and error == 0) and ok
        ok = self._fix_eeprom(
            dxl_id, self.ADDR_DRIVE_MODE,
            self.DRIVE_MODE_NORMAL, "Drive Mode",
        ) and ok
        ok = self._fix_eeprom(
            dxl_id, self.ADDR_OPERATING_MODE,
            self.OPERATING_MODE_POSITION, "Operating Mode",
        ) and ok
        result, error = self.packetHandler.write4ByteTxRx(
            self.portHandler, dxl_id,
            self.ADDR_PROFILE_ACCELERATION, self.PROFILE_ACCELERATION)
        ok = (result == COMM_SUCCESS and error == 0) and ok
        result, error = self.packetHandler.write4ByteTxRx(
            self.portHandler, dxl_id,
            self.ADDR_PROFILE_VELOCITY, self.PROFILE_VELOCITY)
        ok = (result == COMM_SUCCESS and error == 0) and ok
        if not ok:
            warn(f"[ID {dxl_id}] 안전 프로파일 설정 실패")
        return ok

    def _fix_eeprom(self, dxl_id, addr, expected, name):
        value, result, error = self.packetHandler.read1ByteTxRx(
            self.portHandler, dxl_id, addr)
        if result != COMM_SUCCESS or error != 0:
            warn(f"[ID {dxl_id}] {name} 읽기 실패")
            return False
        if value == expected:
            return True
        warn(f"[ID {dxl_id}] {name} {value} -> {expected} 복구")
        result, error = self.packetHandler.write1ByteTxRx(
            self.portHandler, dxl_id, addr, expected)
        time.sleep(0.05)
        return result == COMM_SUCCESS and error == 0

    def _read_position(self, dxl_id):
        try:
            value, result, error = self.packetHandler.read4ByteTxRx(
                self.portHandler, dxl_id, self.ADDR_PRESENT_POSITION)
        except Exception as exc:
            warn(f"[ID {dxl_id}] 위치 읽기 예외: {exc}")
            return None
        if result != COMM_SUCCESS or error != 0:
            warn(f"[ID {dxl_id}] 위치 읽기 실패 "
                  f"(result={result}, error={error})")
            return None
        if value > 0x7FFFFFFF:
            value -= 0x100000000
        if not self.POS_MIN <= value <= self.POS_MAX:
            warn(f"[ID {dxl_id}] 비정상 위치값 {value}")
            return None
        return value

    def _read_watchdog(self, dxl_id):
        try:
            value, result, error = self.packetHandler.read1ByteTxRx(
                self.portHandler, dxl_id, self.ADDR_BUS_WATCHDOG)
        except Exception as exc:
            warn(f"[ID {dxl_id}] Bus Watchdog 읽기 예외: {exc}")
            return None
        if result != COMM_SUCCESS or error != 0:
            print(
                f"[TRK] [ID {dxl_id}] Bus Watchdog 읽기 실패 "
                f"(result={result}, error={error})"
            )
            return None
        return int(value) & 0xFF

    def _arm_bus_watchdog(self):
        if self.watchdog_armed:
            return True
        for dxl_id in (self.PAN_ID, self.TILT_ID):
            try:
                result, error = self.packetHandler.write1ByteTxRx(
                    self.portHandler,
                    dxl_id,
                    self.ADDR_BUS_WATCHDOG,
                    self.BUS_WATCHDOG_VALUE,
                )
            except Exception as exc:
                warn(f"[ID {dxl_id}] Bus Watchdog 예외: {exc}")
                return False
            if result != COMM_SUCCESS or error != 0:
                print(
                    f"[TRK] [ID {dxl_id}] Bus Watchdog 설정 실패 "
                    f"(result={result}, error={error})"
                )
                return False
        self.watchdog_armed = True
        self._last_feedback = time.monotonic()
        if self.verbose:
            info("런타임 Bus Watchdog 시작 (1.0초)")
        return True

    def _clamp_pan(self, v):
        return float(max(self.pan_min, min(self.pan_max, v)))

    def _clamp_tilt(self, v):
        return float(max(self.tilt_min, min(self.tilt_max, v)))

    # ---------------------------------------------------------------- write

    def _write_targets(self):
        pan, tilt = int(self._pan_target), int(self._tilt_target)
        if (pan, tilt) == self._written:
            return True
        try:
            self.syncWrite.clearParam()
            params_ok = True
            for dxl_id, pos in ((self.PAN_ID, pan), (self.TILT_ID, tilt)):
                added = self.syncWrite.addParam(dxl_id, [
                    DXL_LOBYTE(DXL_LOWORD(pos)), DXL_HIBYTE(DXL_LOWORD(pos)),
                    DXL_LOBYTE(DXL_HIWORD(pos)), DXL_HIBYTE(DXL_HIWORD(pos)),
                ])
                params_ok = params_ok and bool(added)
            result = self.syncWrite.txPacket() if params_ok else -1
            self.syncWrite.clearParam()
        except Exception as exc:
            try:
                self.syncWrite.clearParam()
            except Exception:
                pass
            warn(f"SyncWrite SDK 예외: {exc}")
            self._enter_fault("SYNC_WRITE_EXCEPTION")
            return False

        if not params_ok or result != COMM_SUCCESS:
            self._comm_failures += 1
            self._moving = False
            print(
                f"[TRK] SyncWrite 실패 {self._comm_failures}/"
                f"{self.MAX_COMM_FAILURES} (result={result})"
            )
            if self._comm_failures >= self.MAX_COMM_FAILURES:
                self._enter_fault("SYNC_WRITE_FAILED")
            return False

        self._comm_failures = 0
        self._written = (pan, tilt)
        return True

    def _enter_fault(self, reason):
        if self._faulted:
            return
        self._faulted = True
        self.watchdog_armed = False
        self._moving = False
        self.tracking_state = "FAULT"
        self.tracking_confidence = 0.0
        self.tracking_reason = str(reason)
        self.pan_controller.reset()
        self.tilt_controller.reset()

        # Competition continuous mode: never convert a recoverable runtime
        # communication fault into a persistent, internally torque-disabled
        # state.  Let the application unwind through its normal finally block
        # instead, so the original fault is visible as a RuntimeError.
        if self.raise_on_fault:
            warn(f"MOTOR ERROR: {reason} -> RuntimeError")
            raise RuntimeError(f"FaceTrackerMotor fault: {reason}")

        self.torque_off_confirmed = True
        for dxl_id in (self.PAN_ID, self.TILT_ID):
            try:
                result, error = self.packetHandler.write1ByteTxRx(
                    self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE, 0)
                if result != COMM_SUCCESS or error != 0:
                    self.torque_off_confirmed = False
                    print(
                        f"[TRK] [ID {dxl_id}] 토크 OFF 확인 실패 "
                        f"(result={result}, error={error})"
                    )
            except Exception as exc:
                self.torque_off_confirmed = False
                warn(f"[ID {dxl_id}] 토크 OFF 예외: {exc}")
        if self.torque_off_confirmed:
            warn(f"FAULT: {reason} -> 양축 토크 OFF 확인")
        else:
            print(
                f"[TRK] FAULT: {reason} -> TORQUE_OFF_UNCONFIRMED; "
                "외부 비상 전원 차단 필요"
            )

    def _refresh_feedback(self, now):
        if now - self._last_feedback < self.FEEDBACK_INTERVAL:
            return True
        self._last_feedback = now

        pan_now = self._read_position(self.PAN_ID)
        tilt_now = self._read_position(self.TILT_ID)
        if pan_now is None or tilt_now is None:
            self._feedback_failures += 1
            # ── A-4 ──────────────────────────────────────────────────
            # 이동 중 피드백 단절은 여전히 가장 위험한 상태다. 다만 예전처럼
            # 1회 실패로 즉시 fault 를 내면 USB 시리얼 글리치 한 번에 계측
            # 전체가 죽는다(raise_on_fault=True -> 워커 스레드 종료).
            # 실패가 이어지는 동안 update() 는 어차피 새 목표를 쓰지 않고
            # 빠져나가므로 모터는 마지막 목표를 유지할 뿐이고, 눈 감고
            # 명령하는 구간은 생기지 않는다. 연속 실패가 한도를 넘을 때만
            # fail-closed 로 간다.
            if self._tracking_was_allowed or self._moving:
                self._motion_feedback_failures += 1
                if (self._motion_feedback_failures
                        >= self.MAX_MOTION_FEEDBACK_FAILURES):
                    self._enter_fault("POSITION_FEEDBACK_LOST_DURING_MOTION")
                else:
                    warn(f"이동 중 위치 피드백 실패 "
                         f"{self._motion_feedback_failures}/"
                         f"{self.MAX_MOTION_FEEDBACK_FAILURES} - 목표 유지")
                return False
            if self._feedback_failures >= self.MAX_FEEDBACK_FAILURES:
                self._enter_fault("POSITION_FEEDBACK_LOST")
            return False

        if self.enable_bus_watchdog:
            watchdog_values = (
                self._read_watchdog(self.PAN_ID),
                self._read_watchdog(self.TILT_ID),
            )
            if any(value == 0xFF for value in watchdog_values):
                self._enter_fault("BUS_WATCHDOG_EXPIRED")
                return False
            if any(value is None for value in watchdog_values):
                self._feedback_failures += 1
                if self._tracking_was_allowed or self._moving:
                    # A-4: 위와 같은 이유로 연속 실패 한도를 준다.
                    self._motion_feedback_failures += 1
                    if (self._motion_feedback_failures
                            >= self.MAX_MOTION_FEEDBACK_FAILURES):
                        self._enter_fault("WATCHDOG_STATUS_LOST_DURING_MOTION")
                    else:
                        warn(f"이동 중 watchdog 상태 읽기 실패 "
                             f"{self._motion_feedback_failures}/"
                             f"{self.MAX_MOTION_FEEDBACK_FAILURES} - 목표 유지")
                elif self._feedback_failures >= self.MAX_FEEDBACK_FAILURES:
                    self._enter_fault("WATCHDOG_STATUS_LOST")
                return False

        self._feedback_failures = 0
        self._motion_feedback_failures = 0
        tracking_error = max(
            abs(float(pan_now) - self._pan_target),
            abs(float(tilt_now) - self._tilt_target),
        )
        if tracking_error > self.MAX_TRACKING_ERROR_TICKS:
            self._stall_samples += 1
            if self._stall_samples >= self.MAX_STALL_SAMPLES:
                self._enter_fault("SERVO_NOT_FOLLOWING")
                return False
        else:
            self._stall_samples = 0
        return not self._faulted

    def _hold_current_position(self):
        pan_now = self._read_position(self.PAN_ID)
        tilt_now = self._read_position(self.TILT_ID)
        if pan_now is None or tilt_now is None:
            self._enter_fault("HOLD_POSITION_UNKNOWN")
            return False
        self._pan_target = self._clamp_pan(float(pan_now))
        self._tilt_target = self._clamp_tilt(float(tilt_now))
        written = self._write_targets()
        self._moving = False
        if not written:
            self._enter_fault("SAFETY_HOLD_WRITE_FAILED")
        return written

    # --------------------------------------------------------------- update

    @staticmethod
    def _fit_box_in_frame(box, frame_shape):
        """박스 크기는 최대한 유지하면서 화면 안으로 평행 이동한다."""
        if box is None:
            return None
        try:
            (y1, y2), (x1, x2) = box
            y1, y2, x1, x2 = map(float, (y1, y2, x1, x2))
            h, w = float(frame_shape[0]), float(frame_shape[1])
        except (TypeError, ValueError, IndexError):
            return None
        if y2 <= y1 or x2 <= x1 or h <= 1 or w <= 1:
            return None

        box_h = min(y2 - y1, h)
        box_w = min(x2 - x1, w)
        cy = (y1 + y2) * 0.5
        cx = (x1 + x2) * 0.5

        y1 = cy - box_h * 0.5
        y2 = cy + box_h * 0.5
        x1 = cx - box_w * 0.5
        x2 = cx + box_w * 0.5

        if y1 < 0.0:
            y2 -= y1
            y1 = 0.0
        if y2 > h:
            y1 -= y2 - h
            y2 = h
        if x1 < 0.0:
            x2 -= x1
            x1 = 0.0
        if x2 > w:
            x1 -= x2 - w
            x2 = w

        return ((max(0.0, y1), min(h, y2)),
                (max(0.0, x1), min(w, x2)))

    @staticmethod
    def _evidence_scale(source, confidence, evidence_age):
        """Return ``(scale, reason)`` without accepting unsafe public evidence."""
        sources = {"SCRFD", "SCRFD_EDGE", "FLOW", "PERSON_HEAD", "RPPG"}
        if not isinstance(source, str) or source not in sources:
            return 0.0, "UNKNOWN_SOURCE"
        if isinstance(confidence, bool):
            return 0.0, "INVALID_CONFIDENCE"
        try:
            confidence = float(confidence)
        except (TypeError, ValueError, OverflowError):
            return 0.0, "INVALID_CONFIDENCE"
        if not isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
            return 0.0, "INVALID_CONFIDENCE"
        if confidence == 0.0:
            return 0.0, "ZERO_CONFIDENCE"
        if isinstance(evidence_age, bool):
            return 0.0, "INVALID_EVIDENCE_AGE"
        try:
            evidence_age = float(evidence_age)
        except (TypeError, ValueError, OverflowError):
            return 0.0, "INVALID_EVIDENCE_AGE"
        if not isfinite(evidence_age) or evidence_age < 0.0:
            return 0.0, "INVALID_EVIDENCE_AGE"
        source_limits = {
            "SCRFD": config.CPU_RESULT_MAX_AGE,
            "SCRFD_EDGE": config.CPU_RESULT_MAX_AGE,
            "FLOW": config.CPU_FLOW_HOLD_SEC,
            "PERSON_HEAD": config.CPU_PERSON_HOLD_SEC,
            "RPPG": 0.0,
        }
        if evidence_age > source_limits[source]:
            return 0.0, "STALE_EVIDENCE"
        speed = 1.0 if source == "RPPG" else config.CPU_SOURCE_SPEED[source]
        return speed * confidence, ""

    @classmethod
    def _command_scale(cls, source, confidence, evidence_age):
        """Return the safe source/confidence multiplier, or zero when held."""
        return cls._evidence_scale(source, confidence, evidence_age)[0]

    @staticmethod
    def _diagnostic_number(value):
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if isfinite(number) else None

    def _record_evidence(self, source, confidence, evidence_age, scale):
        self.external_evidence_source = source if isinstance(source, str) else "INVALID"
        self.external_evidence_confidence = self._diagnostic_number(confidence)
        self.external_evidence_age = self._diagnostic_number(evidence_age)
        self.applied_command_scale = float(scale)

    def _hold_for_evidence(self, reason):
        """Hold the last servo goal; uncertain evidence is not a bus fault."""
        self._tracking_was_allowed = False
        self._moving = False
        self.tracking_state = "HOLD"
        self.tracking_confidence = 0.0
        self.tracking_reason = reason
        self.pan_controller.reset()
        self.tilt_controller.reset()

    def update(
        self,
        box,
        frame_shape,
        frame_rgb=None,
        frame_gray=None,
        external_confidence=1.0,
        external_source="RPPG",
        evidence_age=0.0,
        external_track_id=None,
    ):
        """Move only for current, source-qualified face evidence."""
        if not self.enabled or self._faulted:
            return

        command_scale, evidence_reason = self._evidence_scale(
            external_source, external_confidence, evidence_age,
        )
        self._record_evidence(
            external_source, external_confidence, evidence_age, command_scale,
        )
        if evidence_reason:
            # An armed Dynamixel bus watchdog must see real bus traffic even
            # while uncertain evidence holds the last goal.  Feedback reads
            # are the existing non-goal heartbeat and also preserve fault
            # detection; never claim the watchdog is armed without them.
            if self.enable_bus_watchdog and self.watchdog_armed:
                if not self._refresh_feedback(time.monotonic()):
                    self._moving = False
                    return
            self._hold_for_evidence(evidence_reason)
            return

        if self.enable_bus_watchdog and not self._arm_bus_watchdog():
            self._enter_fault("BUS_WATCHDOG_ARM_FAILED")
            return

        now = time.monotonic()
        if not self._refresh_feedback(now):
            self._moving = False
            return
        if self._flow_track_id is not None and external_track_id != self._flow_track_id:
            self.optical_flow.reset()
            self._latest_flow = None
            self._flow_track_id = None
        flow_box, flow_quality = self.optical_flow.update(
            frame_rgb,
            box,
            frame_gray=frame_gray,
            now=now,
        )
        if (
            external_source == "SCRFD"
            and flow_box is not None
            and isinstance(flow_quality, (int, float))
            and not isinstance(flow_quality, bool)
            and isfinite(flow_quality)
            and 0.45 <= float(flow_quality) <= 1.0
        ):
            self._latest_flow = (flow_box, float(flow_quality), now)
            self._flow_track_id = external_track_id
        else:
            self._latest_flow = None
            self._flow_track_id = None

        # detector가 화면 가장자리에서 얼굴을 잠깐 놓치면 optical flow가 이어받는다.
        # 예측 박스가 프레임 밖으로 조금 나간 경우에는 크기를 유지한 채 화면 안으로
        # 평행 이동해서, 얼굴이 아래쪽 경계에 있어도 충분한 y 오차를 계속 만든다.
        flow_fallback = False
        tracking_box = box
        if external_source in ("SCRFD_EDGE", "FLOW", "PERSON_HEAD"):
            # Hybrid flow/head estimates may intentionally project just outside
            # the image. Keep their size while shifting them into the existing
            # temporal gate's valid frame.
            tracking_box = self._fit_box_in_frame(tracking_box, frame_shape)
        if tracking_box is None and flow_box is not None:
            try:
                flow_ok = flow_quality is not None and float(flow_quality) > 0.0
            except (TypeError, ValueError):
                flow_ok = False
            if flow_ok:
                tracking_box = self._fit_box_in_frame(flow_box, frame_shape)
                flow_fallback = tracking_box is not None

        decision = self.target_filter.update(
            tracking_box,
            frame_shape,
            now=now,
            # fallback 박스 자체가 optical-flow 결과이므로 detector-vs-flow
            # 일치 검사를 다시 걸지 않는다.
            flow_box=None if flow_fallback else flow_box,
            flow_quality=None if flow_fallback else flow_quality,
        )
        self.tracking_state = decision.state
        self.tracking_confidence = (
            min(decision.confidence, float(flow_quality))
            if flow_fallback else decision.confidence
        )
        self.tracking_reason = (
            "FLOW_FALLBACK"
            if flow_fallback and decision.allowed
            else (external_source if decision.allowed else decision.reason)
        )
        self.last_tracking_box = decision.box if decision.allowed else None

        if not decision.allowed:
            # 얼굴 미검출/검증 실패만으로 현재 위치를 새 Goal Position으로 쓰지 않는다.
            # 새 목표를 보내지 않고 마지막 목표를 그대로 둔다.
            self._tracking_was_allowed = False
            self._moving = False
            self.pan_controller.reset()
            self.tilt_controller.reset()
            return

        if not all(self._sign_ok.values()):
            if self._tracking_was_allowed:
                self._hold_current_position()
            self._tracking_was_allowed = False
            self._moving = False
            self.tracking_state = "DIRECTION_UNCONFIRMED"
            self.tracking_confidence = 0.0
            self.tracking_reason = "FIXED_SIGN_REQUIRED"
            self.pan_controller.reset()
            self.tilt_controller.reset()
            return

        self._tracking_was_allowed = True

        if now - self._last_update < self.UPDATE_INTERVAL:
            return
        dt = (
            self.UPDATE_INTERVAL
            if self._last_update <= 0.0
            else max(0.01, min(0.20, now - self._last_update))
        )
        self._last_update = now

        h, w = frame_shape[0], frame_shape[1]
        (y1, y2), (x1, x2) = decision.box
        face_cx = (x1 + x2) / 2.0
        face_cy = (y1 + y2) / 2.0

        error_x = (w / 2.0 - face_cx) / (w / 2.0)
        error_y = (h / 2.0 - face_cy) / (h / 2.0)
        self.last_error = (error_x, error_y)

        command_confidence = self.tracking_confidence * command_scale
        step_x = self.pan_controller.update(
            error_x, dt=dt, confidence=command_confidence)
        step_y = self.tilt_controller.update(
            error_y, dt=dt, confidence=command_confidence)

        previous_targets = (int(self._pan_target), int(self._tilt_target))
        self._pan_target = self._clamp_pan(
            self._pan_target + self.pan_sign * step_x)
        self._tilt_target = self._clamp_tilt(
            self._tilt_target + self.tilt_sign * step_y)
        new_targets = (int(self._pan_target), int(self._tilt_target))
        self._moving = new_targets != previous_targets

        self._check_stuck()
        if not self._write_targets():
            self._moving = False

    def _check_stuck(self):
        for axis, err, target, lo, hi in (
            ("pan", self.last_error[0], self._pan_target,
             self.pan_min, self.pan_max),
            ("tilt", self.last_error[1], self._tilt_target,
             self.tilt_min, self.tilt_max),
        ):
            if self._warned[axis]:
                continue
            if abs(err) > 0.35 and target in (lo, hi):
                warn(f"경고: {axis}이 가동 한계 {target:.0f}에 붙었는데 "
                      f"오차가 {err:+.2f}입니다. "
                      f"{'PAN_SPAN' if axis == 'pan' else 'TILT_SPAN'}을 "
                      f"늘리거나 헤드를 정면으로 맞추고 다시 시작하세요.")
                self._warned[axis] = True

    # ----------------------------------------------------------------- API

    def is_moving(self):
        return self._moving

    def target_box(self):
        return self.last_tracking_box

    def latest_flow(self, now=None):
        """Return the most recent SCRFD-seeded LK estimate during the fusion hold."""
        if self._latest_flow is None:
            return None
        checked_at = time.monotonic() if now is None else float(now)
        box, quality, timestamp = self._latest_flow
        if checked_at - timestamp > config.CPU_FLOW_HOLD_SEC:
            self._latest_flow = None
            self._flow_track_id = None
            return None
        return box, quality

    def fail_safe_stop(self, reason="EXTERNAL_SAFETY_STOP"):
        """Public integration hook: stop tracking and attempt confirmed torque-off."""
        if self.enabled:
            self._enter_fault(str(reason))
        else:
            self._faulted = True
            self._moving = False
            self.tracking_state = "FAULT"
            self.tracking_confidence = 0.0
            self.tracking_reason = str(reason)
        return self.torque_off_confirmed is True

    def status(self):
        if self._faulted and self.torque_off_confirmed is False:
            return (
                f"[TRK] FAULT {self.tracking_reason} "
                "TORQUE_OFF_UNCONFIRMED"
            )
        if not self.enabled:
            return "[TRK] DISABLED"
        sign_mark = "" if all(self._sign_ok.values()) else " sign?"
        reason = f" {self.tracking_reason}" if self.tracking_reason else ""
        # C-6: flow 품질의 두 성분을 그대로 노출한다. FLOW_MIN_QUALITY 를
        # 조정할 때 이 값(t=추적생존율, i=합의율)을 보고 정하면 된다.
        estimator = self.optical_flow.estimator
        flow_mark = (
            f"ON t={estimator.last_track_fraction:.2f}"
            f"/i={estimator.last_inlier_fraction:.2f}"
            if self.optical_flow.available
            else "OFF"
        )
        torque_mark = (
            " TORQUE_OFF_UNCONFIRMED"
            if self._faulted and self.torque_off_confirmed is False
            else ""
        )
        evidence_confidence = (
            "invalid"
            if self.external_evidence_confidence is None
            else f"{self.external_evidence_confidence:.2f}"
        )
        evidence_age = (
            "invalid"
            if self.external_evidence_age is None
            else f"{self.external_evidence_age:.3f}"
        )
        return (
            f"[TRK] {self.tracking_state} q={self.tracking_confidence:.2f}{reason}"
            f" src={self.external_evidence_source} conf={evidence_confidence}"
            f" age={evidence_age} scale={self.applied_command_scale:.2f}"
            f" e=({self.last_error[0]:+.2f},{self.last_error[1]:+.2f})"
            f" tgt={self._pan_target:4.0f}/{self._tilt_target:4.0f}"
            f" s=({self.pan_sign:+.0f},{self.tilt_sign:+.0f}){sign_mark}"
            f" of={flow_mark} {'MOVING' if self._moving else 'idle'}{torque_mark}"
        )

    def close(self):
        """토크만 끈다. 원위치 복귀는 하지 않는다."""
        if not self.enabled:
            self._close_port_safely()
            return
        torque_off_confirmed = True
        for dxl_id in (self.PAN_ID, self.TILT_ID):
            try:
                result, error = self.packetHandler.write1ByteTxRx(
                    self.portHandler, dxl_id, self.ADDR_TORQUE_ENABLE, 0)
                if result != COMM_SUCCESS or error != 0:
                    torque_off_confirmed = False
            except Exception as exc:
                torque_off_confirmed = False
                warn(f"[ID {dxl_id}] 종료 토크 OFF 예외: {exc}")
        port_closed = self._close_port_safely()
        self.enabled = False
        self.watchdog_armed = False
        self.torque_off_confirmed = torque_off_confirmed
        if self.verbose:
            if torque_off_confirmed and port_closed:
                info("종료, 토크 OFF 확인")
            else:
                warn("종료 불완전, 외부 비상 전원 차단 필요")


# ------------------------------------------------------------------ 진단

def selftest(device_name=None, baudrate=None):
    """서보가 위치 명령을 실제로 실행하는지 확인한다."""
    print("=" * 60)
    print("서보 하드웨어 점검")
    print("=" * 60)

    trk = FaceTrackerMotor(
        device_name=device_name,
        baudrate=baudrate,
        calibration_mode=True,
    )
    if not trk.enabled:
        print("\n결과: 초기화 실패. 위 메시지가 원인입니다.")
        print("  - 포트 실패      -> DEVICENAME, USB 연결, 권한(dialout) 확인")
        print("  - 위치 읽기 실패 -> ID(1/2), BAUDRATE, 전원 확인")
        return

    if not trk._arm_bus_watchdog():
        trk.fail_safe_stop("BUS_WATCHDOG_ARM_FAILED")
        trk.close()
        print("\n결과: Bus Watchdog 시작 실패")
        return

    def write_test_goal(dxl_id, goal):
        try:
            result, error = trk.packetHandler.write4ByteTxRx(
                trk.portHandler, dxl_id, trk.ADDR_GOAL_POSITION, int(goal))
        except Exception as exc:
            print(f"  >> 목표 쓰기 예외: {exc}")
            return False
        if result != COMM_SUCCESS or error != 0:
            print(
                f"  >> 목표 쓰기 실패: result={result}, error={error}"
            )
            return False
        return True

    def wait_for_goal(dxl_id, goal, attempts=30):
        latest = None
        for _ in range(attempts):
            time.sleep(0.1)
            # watchdog은 ID별로 동작하므로 대기 중에도 양축에 instruction을 보낸다.
            pan_now = trk._read_position(trk.PAN_ID)
            tilt_now = trk._read_position(trk.TILT_ID)
            latest = pan_now if dxl_id == trk.PAN_ID else tilt_now
            if latest is not None and abs(latest - goal) < 10:
                return latest
        return latest

    try:
        for name, dxl_id, lo, hi in (
            ("팬 (ID 1, 좌우)", trk.PAN_ID, trk.pan_min, trk.pan_max),
            ("틸트(ID 2, 상하)", trk.TILT_ID, trk.tilt_min, trk.tilt_max),
        ):
            start = trk._read_position(dxl_id)
            if start is None:
                print(f"\n{name}: 위치 읽기 실패")
                continue

            delta = 100 if start + 100 <= hi else -100
            goal = start + delta
            print(f"\n{name}")
            print(f"  현재 위치 {start} -> 목표 {goal} ({delta:+d} 틱) 명령")

            if not write_test_goal(dxl_id, goal):
                continue
            end = wait_for_goal(dxl_id, goal)
            moved = abs(end - start) if end is not None else 0
            print(f"  실제 위치 {end}  (이동량 {moved} 틱)")

            if moved < 20:
                print("  >> 실패: 서보가 명령을 실행하지 않았습니다.")
                print("     Operating Mode / 토크 / 전원 / ID 를 확인하세요.")
            else:
                print("  >> 정상: 위치 제어가 동작합니다.")
                if write_test_goal(dxl_id, start):
                    returned = wait_for_goal(dxl_id, start)
                    if returned is None or abs(returned - start) >= 10:
                        print("  >> 경고: 시작 위치 복귀를 확인하지 못했습니다.")
    finally:
        trk.close()
    print("\n점검 끝. 두 축 모두 '정상'이면 하드웨어는 문제 없습니다.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="서보가 위치 명령을 실행하는지 점검")
    ap.add_argument(
        "--device-name",
        default=None,
        help=f"Dynamixel USB 시리얼 장치(기본: {config.DXL_PORT})",
    )
    ap.add_argument(
        "--dxl-baud",
        type=int,
        default=None,
        help=(f"통신 속도(기본 {config.DXL_BAUDRATE}). 응답이 없으면 "
              f"{list(config.DXL_BAUD_FALLBACKS)} 를 자동 탐색한다."),
    )
    args = ap.parse_args()

    if args.selftest:
        selftest(device_name=args.device_name, baudrate=args.dxl_baud)
    else:
        ap.print_help()
