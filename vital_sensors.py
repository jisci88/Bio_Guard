"""
vital_sensors.py - 센서 어댑터 계층   (D-1 분리)

예전에는 이 클래스들이 vital_monitor.py 안에 있었고 vital_run.py 가 거기서
import 했다. 그 구조 때문에 파이프라인을 공유 모듈로 뽑으면 순환 import 가
생긴다. 그래서 센서만 여기로 내렸다.

  OpenRppgVitalTracker : open-rppg BVP 주파수 분석 기반 심박수
  ThermalFaceTracker   : MLX90640 열화상 -> 얼굴 영역 온도
  AlarmBuzzer          : GPIO 액티브 부저 논블로킹 비프

vital_monitor.py 는 하위 호환을 위해 이 세 이름을 그대로 재export 한다.
"""

import threading
import time

import numpy as np

import config
from config import get_logger

log = get_logger("sensors")


def picamera_rgb888_to_rgb(frame_bgr):
    """Picamera2 RGB888 메모리(BGR 순서)를 RGB 프레임으로 정규화한다."""
    return np.ascontiguousarray(np.asarray(frame_bgr)[:, :, ::-1])


# ══════════════════════════════════════════════════════
#  open-rppg 심박수
# ══════════════════════════════════════════════════════

class OpenRppgVitalTracker:
    """open-rppg 로 심박수만 갱신하는 트래커."""

    def __init__(self, model, hr_window_sec=None):
        self.model = model
        self.hr_window_sec = (
            config.HR_WINDOW_SEC if hr_window_sec is None else hr_window_sec
        )

        self.hr_bpm = 0.0
        self.hr_conf = 0.0
        self.hr_reason = "WARMING UP"

        # 마지막으로 유효했던 (값, 신뢰도, 시각)
        self._hr_last = None

        self.diag = {"hr_calls": 0, "hr_ok": 0, "hr_err": 0, "sqi_max": 0.0}

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

        if not config.HR_MIN <= hr <= config.HR_MAX:
            held = self._hold(self._hr_last, now, config.HR_HOLD_SEC)
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

        self.hr_reason = (
            "OK" if self.hr_conf >= config.HR_OK_CONF else f"SQI LOW {sqi:.2f}"
        )


# ══════════════════════════════════════════════════════
#  MLX90640 열화상 - 얼굴 온도
# ══════════════════════════════════════════════════════

class ThermalFaceTracker:
    """MLX90640 백그라운드 리더 + RGB 얼굴 박스 -> 열화상 격자 매핑."""

    def __init__(self, i2c_frequency=None, refresh_rate_hz=None, i2c_bus=None):
        self.available = False
        self._frame = None          # (ROWS, COLS) float32
        self._frame_time = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self.read_errors = 0

        i2c_frequency = (
            config.MLX_I2C_FREQ if i2c_frequency is None else i2c_frequency
        )
        refresh_rate_hz = (
            config.MLX_REFRESH_HZ if refresh_rate_hz is None else refresh_rate_hz
        )
        i2c_bus = config.MLX_I2C_BUS if i2c_bus is None else i2c_bus

        try:
            import adafruit_mlx90640

            if i2c_bus == 1:
                import board
                import busio

                i2c = busio.I2C(board.SCL, board.SDA, frequency=i2c_frequency)
            else:
                from adafruit_extended_bus import ExtendedI2C

                i2c = ExtendedI2C(i2c_bus, frequency=i2c_frequency)
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
            log.warning("MLX90640 초기화 실패, 얼굴 온도 측정을 비활성화합니다: %s", exc)
            return

        # adafruit 드라이버는 길이 768 의 파이썬 list 를 요구한다.
        self._raw = [0.0] * (config.MLX_COLS * config.MLX_ROWS)
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    # ---------------------------------------------------------------- 읽기

    def _loop(self):
        while self._running:
            try:
                self.mlx.getFrame(self._raw)
            except ValueError:
                # B-4: CRC/재동기 실패. 예전에는 sleep 없이 즉시 재시도해서
                # 코어 하나를 통째로 태울 수 있었다.
                self.read_errors += 1
                time.sleep(config.MLX_READ_ERROR_SLEEP)
                continue
            except Exception as exc:
                self.read_errors += 1
                log.warning("MLX90640 읽기 오류: %s", exc)
                time.sleep(0.5)
                continue

            # B-5: 파이썬 list 복사 대신 numpy 2D 로 한 번에 저장한다.
            grid = np.asarray(self._raw, dtype=np.float32).reshape(
                config.MLX_ROWS, config.MLX_COLS
            )
            with self._lock:
                self._frame = grid
                self._frame_time = time.time()

    def _fresh_grid(self):
        """유효하고 신선한 (ROWS, COLS) 배열 또는 None."""
        with self._lock:
            grid = self._frame
            stamp = self._frame_time
        if grid is None:
            return None
        if time.time() - stamp > config.MLX_STALE_SEC:
            return None
        return grid

    # ---------------------------------------------------------------- 매핑

    @staticmethod
    def _face_extent(box, rgb_shape, shrink=1.0, y_bias=0.0):
        """
        RGB 얼굴 박스 -> 열화상 격자 좌표계의 (cx, cy, half_w, half_h).

        단위는 격자 셀이고, 정수화 전의 **연속값**이다. 정수 셀 수로 세면
        얼굴이 셀 경계를 넘나들 때마다 값이 튀어서 거리 보정이 흔들린다.
        """
        h, w = float(rgb_shape[0]), float(rgb_shape[1])
        (y1, y2), (x1, x2) = box
        face_w = float(x2) - float(x1)
        face_h = float(y2) - float(y1)

        cx_px = (float(x1) + float(x2)) * 0.5
        cy_px = (float(y1) + float(y2)) * 0.5 + face_h * y_bias

        cols, rows = config.MLX_COLS, config.MLX_ROWS
        cx = ((cx_px / w - 0.5) * config.MLX_FOV_SCALE_X
              + 0.5 + config.MLX_OFFSET_X) * cols
        cy = ((cy_px / h - 0.5) * config.MLX_FOV_SCALE_Y
              + 0.5 + config.MLX_OFFSET_Y) * rows
        half_w = face_w * 0.5 * shrink / w * config.MLX_FOV_SCALE_X * cols
        half_h = face_h * 0.5 * shrink / h * config.MLX_FOV_SCALE_Y * rows
        return cx, cy, half_w, half_h

    def face_pixel_count(self, box, rgb_shape):
        """
        얼굴 **전체**가 덮는 열화상 화소 수 (연속값).

        이것이 곧 충전율이고 거리의 제곱에 반비례한다. temp_calib 의 거리
        보정이 이 값을 쓴다 (거리를 직접 재지 않는 이유는 temp_calib 문서 참고).
        """
        if box is None:
            return None
        _cx, _cy, half_w, half_h = self._face_extent(box, rgb_shape, shrink=1.0)
        return float(max(0.0, 4.0 * half_w * half_h))

    def measure_face(self, box, rgb_shape, stat=None):
        """
        (skin_c, face_px, sample_px) 또는 None.

        skin_c    : 얼굴 중심부 대표 온도
        face_px   : 얼굴 전체가 덮은 열화상 화소 수 = 충전율(거리) 대용
        sample_px : 실제로 통계를 낸 중심부 화소 수

        C-7: 얼굴 박스를 그대로 매핑하면 머리카락과 배경이 함께 들어가
        p90 를 써도 온도가 낮게 나온다. MLX_FACE_SHRINK 로 중심부만 남기고
        MLX_FACE_Y_BIAS 로 이마 쪽에 살짝 치우치게 한다.
        """
        if not self.available or box is None:
            return None

        grid = self._fresh_grid()
        if grid is None:
            return None

        stat = config.THERMAL_STAT if stat is None else stat
        cx, cy, half_w, half_h = self._face_extent(
            box, rgb_shape,
            shrink=config.MLX_FACE_SHRINK,
            y_bias=config.MLX_FACE_Y_BIAS,
        )

        cols, rows = config.MLX_COLS, config.MLX_ROWS
        tx1 = int(np.clip(cx - half_w, 0, cols - 1))
        tx2 = int(np.clip(cx + half_w, 0, cols - 1))
        ty1 = int(np.clip(cy - half_h, 0, rows - 1))
        ty2 = int(np.clip(cy + half_h, 0, rows - 1))
        if tx2 < tx1 or ty2 < ty1:
            return None

        region = grid[ty1:ty2 + 1, tx1:tx2 + 1].reshape(-1)
        if region.size == 0:
            return None

        if stat == "max":
            skin = float(region.max())
        else:
            k = max(1, int(region.size * 0.1))
            # 전체 정렬 대신 부분 정렬. 768 화소라 차이는 작지만 공짜다.
            skin = float(np.partition(region, -k)[-k:].mean())

        return skin, self.face_pixel_count(box, rgb_shape), int(region.size)

    def get_face_temperature(self, box, rgb_shape, stat=None):
        """대표 온도만 필요한 호출자를 위한 얇은 래퍼 (기존 계약 유지)."""
        measured = self.measure_face(box, rgb_shape, stat=stat)
        return None if measured is None else measured[0]

    # ------------------------------------------------------ 보정 / 진단용

    def hotspot_norm(self, top_k=12):
        """
        가장 뜨거운 픽셀들의 무게중심을 정규화 좌표 (nx, ny) 로 반환한다.
        C-7 의 --thermal-calib 이 RGB 얼굴 중심과 짝지어 화각을 역산한다.
        """
        grid = self._fresh_grid()
        if grid is None:
            return None

        flat = grid.reshape(-1)
        k = int(min(max(1, top_k), flat.size))
        idx = np.argpartition(flat, -k)[-k:]
        weights = flat[idx] - float(flat.min())
        if float(weights.sum()) <= 1e-6:
            return None

        ys, xs = np.divmod(idx, config.MLX_COLS)
        nx = float(np.average(xs, weights=weights) + 0.5) / config.MLX_COLS
        ny = float(np.average(ys, weights=weights) + 0.5) / config.MLX_ROWS
        return nx, ny

    def raw_frame(self):
        """평탄한 768개 float list. temp_calib.SkinToCore 의 기존 계약 유지."""
        grid = self._fresh_grid()
        return None if grid is None else grid.reshape(-1).tolist()

    def raw_grid(self):
        """(ROWS, COLS) numpy 배열 사본 또는 None."""
        grid = self._fresh_grid()
        return None if grid is None else grid.copy()

    def close(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def fit_thermal_alignment(samples):
    """
    (rgb_nx, rgb_ny, th_nx, th_ny) 표본들로 화각 scale/offset 을 최소자승 추정.

    모델:  th_n - 0.5 = scale * (rgb_n - 0.5) + offset
    반환:  dict(scale_x, scale_y, offset_x, offset_y, n, rms_x, rms_y)
    """
    arr = np.asarray(samples, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 4:
        return None

    out = {"n": int(arr.shape[0])}
    for name, rgb_col, th_col in (("x", 0, 2), ("y", 1, 3)):
        u = arr[:, rgb_col] - 0.5
        v = arr[:, th_col] - 0.5
        if float(np.ptp(u)) < 0.05:
            # 표본이 화면 중앙에만 몰려 있으면 기울기를 못 정한다.
            return None
        scale, offset = np.polyfit(u, v, 1)
        residual = v - (scale * u + offset)
        out[f"scale_{name}"] = float(scale)
        out[f"offset_{name}"] = float(offset)
        out[f"rms_{name}"] = float(np.sqrt(np.mean(residual ** 2)))
    return out


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
            log.info("부저 알림을 사용하지 않습니다.")
            return

        try:
            from gpiozero import Buzzer

            self.buzzer = Buzzer(pin)
            self.buzzer.off()
            log.info("부저 준비 완료 (GPIO%s)", pin)
        except Exception as exc:
            log.warning("부저 초기화 실패, 소리 알림 없이 진행합니다: %s", exc)
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
            log.warning("부저 제어 실패, 비활성화합니다: %s", exc)
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
#  NeoPixel 이상 경보
# ══════════════════════════════════════════════════════

class NeoPixelAlert:
    """Default WS2812B ring alert: red only for an active anomaly."""

    def __init__(self, count=12, brightness=0.2):
        self.pixels = None
        self.active = False
        try:
            import board
            import neopixel

            self.pixels = neopixel.NeoPixel(
                board.D18, int(count), brightness=float(brightness),
                auto_write=False,
            )
            self.pixels.fill((0, 0, 0))
            self.pixels.show()
            log.info("네오픽셀 경보 준비 완료 (GPIO18, %d LEDs)", count)
        except Exception as exc:
            log.warning("네오픽셀 초기화 실패, LED 경보 없이 진행합니다: %s", exc)

    def set_alert(self, active):
        active = bool(active)
        if self.pixels is None or active == self.active:
            return
        self.active = active
        self.pixels.fill((255, 0, 0) if active else (0, 0, 0))
        self.pixels.show()

    def close(self):
        if self.pixels is None:
            return
        try:
            self.pixels.fill((0, 0, 0))
            self.pixels.show()
            deinit = getattr(self.pixels, "deinit", None)
            if callable(deinit):
                deinit()
        finally:
            self.pixels = None
