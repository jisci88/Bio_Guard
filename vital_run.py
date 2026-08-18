"""
open-rppg + MLX90640 실측값을 Tkinter 대시보드로 표시한다.
+ 얼굴 팬/틸트 모터 트래킹 (Dynamixel, PID) - face_tracker_motor.py
+ 환자별 개인화 이상 탐지 (Isolation Forest) - vital_anomaly.py

  심박수    : vital_monitor.OpenRppgVitalTracker (model.hr)
  호흡수    : rr.py의 Lucas-Kanade 옵티컬 플로우 기반 어깨 움직임 추적
  PPG 파형  : model.bvp(start=-6)
  호흡 파형 : 어깨 움직임의 원시 변위(Displacement) 평균값을 실시간으로 표시
  체온      : vital_monitor.ThermalFaceTracker (MLX90640).
              센서가 없거나 초기화에 실패하면 값은 N/A, 트렌드는
              NO SENSOR로 유지된다.
  얼굴추적  : face_tracker_motor.FaceTrackerMotor (OpenRB-150 + Dynamixel).
              얼굴 박스 중심이 화면 중앙에 오도록 팬(ID1)/틸트(ID2) 서보를
              PID로 구동한다. 미연결/초기화 실패 시 자동 비활성화되고
              나머지 측정은 그대로 동작한다.
  이상탐지  : vital_anomaly.VitalAnomalyDetector.
              HR + RR + TEMP 세 축을 환자별로 학습한 뒤 이탈을 판정한다.
              열화상 센서가 없으면 HR + RR 2축으로 자동 축소된다.

화면 하단:
  학습 중  - 기준 학습 진행률(시간 조건 / 샘플 조건 중 느린 쪽)을 게이지로 표시
  학습 후  - NORMAL / ANOMALY / CRITICAL / SIGNAL LOST 판정과 사유를 표시
  이상 시  - 창 배경과 카드가 적색으로 맥동한다 (등급에 따라 속도/채도 차이)

스레드 규칙:
  rppg / OpenCV / MLX90640 / 모터(dynamixel) / IsolationForest 호출은
  VitalWorker 스레드에서만, Tk 위젯 갱신은 메인 스레드에서만 한다.
  둘 사이는 락으로 보호된 스냅샷 dict 하나로만 주고받는다.

실행:
    python vital_run.py --camera 0

    # 열화상 센서 없이 심박수/호흡수만 (이상탐지도 2축으로 자동 축소)
    python vital_run.py --camera 0 --no-thermal

    # 얼굴 추적 모터 없이 (OpenRB-150 미연결 상태 테스트용)
    python vital_run.py --camera 0 --no-tracker

    # 이상탐지 없이 계측만
    python vital_run.py --camera 0 --no-anomaly

    # 기준 학습을 60초로 줄여서 빠르게 확인
    python vital_run.py --camera 0 --calib 60

    # 이상 판정 시 부저도 울리기 (GPIO17, 기본은 무음)
    python vital_run.py --camera 0 --buzzer

    F11 전체화면 전환 / ESC 전체화면 해제
"""

import argparse
import math
import threading
import time
import tkinter as tk
from collections import deque
from tkinter import font as tkfont

import cv2
import numpy as np

import rppg
import rr

from vital_monitor import (
    AlarmBuzzer,
    OpenRppgVitalTracker,
    ThermalFaceTracker,
)
from vital_anomaly import VitalAnomalyDetector
from temp_calib import SkinToCore
from face_tracker_motor import FaceTrackerMotor

# ══════════════════════════════════════════════════════
#  팔레트
# ══════════════════════════════════════════════════════

BG = "#0A0D11"
CARD = "#151A21"
PANEL = "#121821"
GRID = "#1B242F"
HAIRLINE = "#232C38"

FG_TITLE = "#E6EDF5"
FG_LABEL = "#7E8B9A"
FG_DIM = "#3F4A57"
FG_WARN = "#E8A33D"
FG_OK = "#35F07F"

HR_COLOR = "#35F07F"
RR_COLOR = "#58C8F5"

# The dashboard shows a number whenever the reporter holds one, so estimates
# are admitted to its median far more freely than rr.py's own CLI default.
# An ach <= 0 still scores exactly 0, so harmonic misreads stay rejected.
RR_SQI_MIN = 0.03
# What a solid lock scores in this camera path; conf is reported against it and
# stays honest, because the anomaly detector gates on it even when the card does not.
RR_CONF_REF = 0.10
TEMP_COLOR = "#FF8A7A"

# Skin surface sits below core temperature, so a non-contact reading is low by
# design and needs an offset the way a clinical IR thermometer applies one.
# 2.0 is a placeholder standing in for a measurement against a reference
# thermometer, not a calibrated figure -- see --temp-offset.
TEMP_OFFSET_DEFAULT = 2.0

CORNER = 12

ALARM_THEMES = {
    "warn": {
        "period": 2.4,
        "bg": ("#0C0E10", "#1E1708"),
        "card": ("#171A1C", "#2C2413"),
        "panel": ("#14181C", "#271F11"),
        "accent": FG_WARN,
        "title": "#F2E6CE",
        "label": "#B9A683",
        "dim": "#7A6C50",
    },
    "alert": {
        "period": 1.6,
        "bg": ("#150406", "#3F0D13"),
        "card": ("#1F0A0D", "#4A161B"),
        "panel": ("#1B080B", "#401217"),
        "accent": "#FF6B6B",
        "title": "#FFE8E8",
        "label": "#E0A8A8",
        "dim": "#9E6E6E",
    },
    "critical": {
        "period": 0.8,
        "bg": ("#360A10", "#8C1420"),
        "card": ("#3C1116", "#901D27"),
        "panel": ("#360D13", "#851A25"),
        "accent": "#FFE2E2",
        "title": "#FFFFFF",
        "label": "#FFD2D2",
        "dim": "#E9A8A8",
    },
}


# ══════════════════════════════════════════════════════
#  그리기 도우미
# ══════════════════════════════════════════════════════

def blend(fg, bg, ratio):
    fr, fg_, fb = (int(fg[i:i + 2], 16) for i in (1, 3, 5))
    br, bg_, bb = (int(bg[i:i + 2], 16) for i in (1, 3, 5))

    return "#%02x%02x%02x" % (
        int(br + (fr - br) * ratio),
        int(bg_ + (fg_ - bg_) * ratio),
        int(bb + (fb - bb) * ratio),
    )


def round_rect(canvas, x0, y0, x1, y1, radius, **kwargs):
    points = [
        x0 + radius, y0, x1 - radius, y0, x1, y0,
        x1, y0 + radius, x1, y1 - radius, x1, y1,
        x1 - radius, y1, x0 + radius, y1, x0, y1,
        x0, y1 - radius, x0, y0 + radius, x0, y0,
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


def pick_font(candidates, fallback="Helvetica"):
    available = {name.lower() for name in tkfont.families()}
    for name in candidates:
        if name.lower() in available:
            return name
    return fallback


# ══════════════════════════════════════════════════════
#  측정 워커
# ══════════════════════════════════════════════════════

class VitalWorker(threading.Thread):
    PPG_WINDOW_SEC = 6.0
    HR_INTERVAL = 2.0
    WAVE_INTERVAL = 0.25

    TEMP_INTERVAL = 2.0
    TEMP_TREND_MAXLEN = 150

    def __init__(self, camera_id=0, use_thermal=True,
                 thermal_stat="p90", thermal_freq=4, use_tracker=True,
                 use_anomaly=True, calib_sec=180.0, min_conf=0.30,
                 out_pct=1.0, rr_sigma=4.0, temp_sigma=0.4,
                 temp_offset=TEMP_OFFSET_DEFAULT,
                 temp_ambient_gain=None, temp_ambient_ref=None,
                 use_buzzer=False, buzzer_pin=17, dump=None,
                 rr_sqi_min=RR_SQI_MIN, rr_raw=False,
                 rr_motion_rate=rr.MOTION_RATE, rr_mute_sec=rr.MUTE_SEC):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.dump = dump
        self.rr_sqi_min = rr_sqi_min
        self.rr_raw = rr_raw
        self.rr_motion_rate = rr_motion_rate
        self.rr_mute_sec = rr_mute_sec
        self.use_thermal = use_thermal
        self.thermal_stat = thermal_stat
        self.thermal_freq = thermal_freq
        self.use_tracker = use_tracker

        self.use_anomaly = use_anomaly
        self.calib_sec = calib_sec
        self.min_conf = min_conf
        self.out_pct = out_pct
        self.rr_sigma = rr_sigma
        self.temp_sigma = temp_sigma
        # Skin surface runs below core temperature, so a non-contact reading is
        # low by design. Measure this against a reference; do not guess it.
        self.temp_offset = temp_offset
        self.temp_ambient_gain = temp_ambient_gain
        self.temp_ambient_ref = temp_ambient_ref

        self.use_buzzer = use_buzzer
        self.buzzer_pin = buzzer_pin

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._temp_history = deque(maxlen=self.TEMP_TREND_MAXLEN)
        self._resp_history = deque(maxlen=260)

        self._snapshot = {
            "status": "모델 로딩 중...",
            "face": False,
            "hr": 0.0,
            "hr_conf": 0.0,
            "hr_reason": "WARMING UP",
            "rr": 0.0,
            "rr_conf": 0.0,
            "rr_valid": False,
            "rr_reason": "WARMING UP",
            "rr_source": "-",
            "temp": 0.0,
            "temp_conf": 0.0,
            "temp_trend": None,
            "ppg": None,
            "resp": None,
            "tracker_active": False,
            "anomaly": None,
            "anomaly_on": use_anomaly,
            "min_samples": 0,
            "use_temp": False,
        }

    def snapshot(self):
        with self._lock:
            return dict(self._snapshot)

    def stop(self):
        self._stop.set()

    def _publish(self, **kwargs):
        with self._lock:
            self._snapshot.update(kwargs)

    @staticmethod
    def _normalize(signal, max_points=260):
        x = np.asarray(signal, dtype=float)
        x = x[np.isfinite(x)]
        if x.size < 16:
            return None
        if x.size > max_points:
            idx = np.linspace(0, x.size - 1, max_points).astype(int)
            x = x[idx]
        lo, hi = float(x.min()), float(x.max())
        if hi - lo < 1e-9:
            return None
        return ((x - lo) / (hi - lo)).tolist()

    def _ppg_wave(self, model):
        try:
            bvp, _ts = model.bvp(start=-self.PPG_WINDOW_SEC)
        except Exception:
            return None
        return self._normalize(bvp)

    def run(self):
        try:
            model = rppg.Model("ME-flow.rlap")
        except Exception as exc:
            self._publish(status=f"모델 로딩 실패: {exc}")
            return

        vitals = OpenRppgVitalTracker(model)

        thermal = None
        if self.use_thermal:
            thermal = ThermalFaceTracker(refresh_rate_hz=self.thermal_freq)
            if not thermal.available:
                thermal = None

        tracker = None
        if self.use_tracker:
            tracker = FaceTrackerMotor()
            if not tracker.enabled:
                tracker = None

        detector = None
        if self.use_anomaly:
            detector = VitalAnomalyDetector(
                calib_sec=self.calib_sec,
                min_conf=self.min_conf,
                out_pct=self.out_pct,
                rr_sigma=self.rr_sigma,
                temp_sigma=self.temp_sigma,
                use_temp=thermal is not None,
            )

        skin2core = SkinToCore(
            offset=self.temp_offset,
            **{k: v for k, v in (("ambient_gain", self.temp_ambient_gain),
                                 ("ambient_ref", self.temp_ambient_ref))
               if v is not None})

        buzzer = AlarmBuzzer(pin=self.buzzer_pin, enabled=self.use_buzzer)

        if detector is None:
            print("[AN] 이상탐지 비활성 (--no-anomaly). 기준 학습이 진행되지 않습니다.",
                  flush=True)
        else:
            print(f"[AN] 이상탐지 활성  calib={self.calib_sec:.0f}s"
                  f"  min_samples={detector.min_samples}"
                  f"  min_conf={self.min_conf}"
                  f"  use_temp={thermal is not None}", flush=True)

        self._publish(
            status="카메라 연결 중...",
            tracker_active=tracker is not None,
            use_temp=thermal is not None,
            min_samples=detector.min_samples if detector else 0,
        )

        last_hr = last_wave = last_temp = 0.0
        current_temp = 0.0
        temp_conf = 0.0
        anomaly = None

        rr_tracker = None
        rr_reporter = rr.Reporter(sqi_min=self.rr_sqi_min)
        buf_t, buf_d, buf_v = [], [], []
        last_rr_report = None
        current_rr_bpm = 0.0
        current_rr_conf = 0.0
        current_rr_valid = False
        rr_reason = "WARMING UP"
        rr_steps = []
        last_mutes = 0
        log_t, log_d, log_v = [], [], []

        try:
            with model.video_capture(self.camera_id):
                self._publish(status="MONITORING")

                for frame_rgb, box in model.preview:
                    if self._stop.is_set():
                        break

                    now = time.time()
                    face_visible = box is not None

                    if tracker is not None:
                        tracker.update(box, frame_rgb.shape)

                    if not face_visible:
                        vitals.invalidate()

                    # ── 호흡수 갱신 로직 (rr.py 연동) ──
                    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                    if rr_tracker is None:
                        roi = rr.clip_roi(rr.default_roi(gray.shape), gray.shape)
                        rr_tracker = rr.KeyframeTracker(
                            gray.shape, roi,
                            motion_rate=self.rr_motion_rate,
                            mute_sec=self.rr_mute_sec)
                        print(f"[RR] frame={gray.shape} roi={roi}", flush=True)
                        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                        cv2.rectangle(vis, (roi[0], roi[1]),
                                      (roi[0] + roi[2], roi[1] + roi[3]), (0, 255, 0), 2)
                        cv2.imwrite("rr_roi.png", vis)

                    if rr_tracker.needs_anchor(now):
                        if not rr_tracker.anchor(gray, now):
                            continue

                    out = rr_tracker.track(gray, now)
                    if out is not None:
                        d, v = out
                        rr_steps.append(rr_tracker.step)
                        if self.dump:
                            log_t.append(now)
                            log_d.append(d.copy())
                            log_v.append(v)
                        buf_t.append(now)
                        buf_d.append(d.copy())
                        buf_v.append(v)
                        self._resp_history.append(float(np.mean(d)))
                        while buf_t[-1] - buf_t[0] > rr.WIN_SEC * 1.2:
                            buf_t.pop(0)
                            buf_d.pop(0)
                            buf_v.pop(0)

                    if last_rr_report is None:
                        last_rr_report = now
                    elif now - last_rr_report >= rr.HOP_SEC:
                        last_rr_report = now
                        if len(buf_t) > 0:
                            eff = len(buf_t) / max(buf_t[-1] - buf_t[0], 1e-6)
                            s = np.array(rr_steps) if rr_steps else np.zeros(1)
                            rr_steps = []
                            nmute = rr_tracker.mutes - last_mutes
                            last_mutes = rr_tracker.mutes
                            diag = (f"fps={eff:4.1f} pts={rr_tracker.frac:3.0%} "
                                    f"bg={rr_tracker.nbg:3d} mute={nmute:2d} "
                                    f"step p50={np.median(s):.2f} p90={np.percentile(s, 90):.2f}")

                            res = rr.estimate_rr(buf_t, np.array(buf_d), np.array(buf_v))
                            if res is None:
                                rr_reason = f"FILLING ({buf_t[-1]-buf_t[0]:.0f}/{rr.WIN_SEC:.0f}s)"
                                if not rr_reporter.hist:
                                    current_rr_conf = 0.0
                                    current_rr_valid = current_rr_valid and self.rr_raw
                                print(f"[RR] FILLING     {buf_t[-1]-buf_t[0]:4.0f}/{rr.WIN_SEC:.0f}s"
                                      f"  {diag}", flush=True)
                            elif not np.isfinite(res["rr"]):
                                rr_reason = "MOTION"
                                if not rr_reporter.hist:
                                    current_rr_conf = 0.0
                                    current_rr_valid = current_rr_valid and self.rr_raw
                                print(f"[RR] MOTION      vfrac={res['vfrac']:3.0%}  {diag}",
                                      flush=True)
                            else:
                                rep = rr_reporter.update(now, res["rr"], res["sqi"])
                                current_rr_conf = float(
                                    np.clip(res["sqi"] / RR_CONF_REF, 0.0, 1.0))
                                if self.rr_raw:
                                    current_rr_bpm = res["rr"]
                                    current_rr_valid = True
                                    rr_reason = f"RAW ({res['sqi']:.3f})"
                                elif rep is not None:
                                    current_rr_bpm = rep[0]
                                    current_rr_valid = True
                                    rr_reason = f"OK (n={rep[1]})"
                                else:
                                    current_rr_valid = False
                                    rr_reason = "ACQUIRING"
                                print(f"[RR] {rr_reason:<11} rr={res['rr']:5.1f} "
                                      f"sqi={res['sqi']:.3f} evr={res['evr']:.2f} "
                                      f"ach={res['ach']:.2f} prom={res['prom']:.3f} "
                                      f"vfrac={res['vfrac']:3.0%} "
                                      f"nkeep={res['nkeep']}/{rr.NCELL}  {diag}", flush=True)

                    if now - last_temp >= self.TEMP_INTERVAL:
                        last_temp = now
                        temp_val = None
                        if thermal is not None and face_visible:
                            temp_val = thermal.get_face_temperature(
                                box, frame_rgb.shape, stat=self.thermal_stat,
                            )
                        core, amb = skin2core.update(
                            temp_val,
                            thermal.raw_frame() if thermal is not None else None)
                        if core is not None:
                            current_temp = core
                            temp_conf = 1.0
                            self._temp_history.append(current_temp)
                        else:
                            temp_conf = 0.0
                        if thermal is not None:
                            raw_txt = "  --  " if temp_val is None else f"{temp_val:6.2f}"
                            amb_txt = "  --  " if amb is None else f"{amb:6.2f}"
                            core_txt = "REJECT" if core is None else f"{core:6.2f}"
                            print(f"[TEMP] skin={raw_txt}C amb={amb_txt}C "
                                  f"-> core={core_txt}C "
                                  f"stat={self.thermal_stat} "
                                  f"rejected={skin2core.rejected}", flush=True)

                    if now - last_hr >= self.HR_INTERVAL:
                        vitals.update_hr(face_visible)
                        last_hr = now

                        if detector is not None:
                            # The detector gates on the same thing the card does:
                            # if a number is good enough to show, it is good enough
                            # to learn. rr_conf stays honest for the SQI readout.
                            anomaly = detector.push(
                                vitals.hr_bpm,
                                vitals.hr_conf,
                                current_rr_bpm,
                                1.0 if current_rr_valid else 0.0,
                                temp=current_temp,
                                temp_conf=temp_conf,
                                now=now,
                            )

                            if anomaly["baseline"] is None:
                                print(f"[AN] accepted={anomaly['accepted']}/{detector.min_samples}"
                                      f" state={anomaly['state']}"
                                      f" reason={anomaly['reason']}"
                                      f" crit={anomaly['critical']}"
                                      f"  hr={vitals.hr_bpm:5.1f}({vitals.hr_conf:.2f})"
                                      f" rr={current_rr_bpm:5.1f}"
                                      f"({1.0 if current_rr_valid else 0.0:.2f})"
                                      f" temp={current_temp:5.1f}({temp_conf:.2f})"
                                      f"  {detector.stats()}", flush=True)

                            if anomaly["critical"]:
                                buzzer.set_level("critical")
                            elif anomaly["alert"]:
                                buzzer.set_level("alert")
                            else:
                                buzzer.set_level(None)

                    if now - last_wave >= self.WAVE_INTERVAL:
                        last_wave = now
                        temp_trend = self._normalize(list(self._temp_history))

                        self._publish(
                            face=face_visible,
                            hr=vitals.hr_bpm,
                            hr_conf=vitals.hr_conf,
                            hr_reason=vitals.hr_reason,
                            rr=current_rr_bpm,
                            rr_conf=current_rr_conf,
                            rr_valid=current_rr_valid,
                            rr_reason=rr_reason,
                            rr_source="OF",
                            temp=current_temp,
                            temp_conf=temp_conf,
                            temp_trend=temp_trend,
                            ppg=self._ppg_wave(model),
                            resp=self._normalize(list(self._resp_history)),
                            anomaly=anomaly,
                            status=(
                                "MONITORING"
                                if face_visible
                                else "NO SUBJECT IN FRAME"
                            ),
                        )

        except Exception as exc:
            self._publish(status=f"측정 중단: {exc}")
        finally:
            if self.dump and log_t:
                np.savez(self.dump, t=np.array(log_t), disp=np.array(log_d),
                         valid=np.array(log_v))
                print(f"[RR] saved {self.dump}  ({len(log_t)} frames)", flush=True)
            buzzer.close()
            if thermal is not None:
                thermal.close()
            if tracker is not None:
                tracker.close()


# ══════════════════════════════════════════════════════
#  수치 카드
# ══════════════════════════════════════════════════════

class ValueCard(tk.Canvas):
    SEGMENTS = 5

    def __init__(self, parent, fonts, label, unit, color, alarm_text):
        super().__init__(parent, bg=BG, highlightthickness=0)
        self.fonts = fonts
        self.label = label
        self.unit = unit
        self.color = color
        self.alarm_text = alarm_text
        self.value_text = "--"
        self.active = False
        self.in_range = True
        self.conf = 0.0
        self.reason = ""
        self._fill = CARD
        self._items = {}
        self.bind("<Configure>", lambda _e: self._layout())

    def _layout(self):
        width = self.winfo_width()
        height = self.winfo_height()

        if width < 40 or height < 40:
            return

        self.delete("all")
        self._items.clear()

        self._items["bg"] = round_rect(
            self, 1, 1, width - 1, height - 1, CORNER,
            fill=self._fill, outline=HAIRLINE,
        )
        self.create_text(
            16, 18, text=self.label, anchor="w",
            fill=FG_LABEL, font=self.fonts["label"],
        )
        self._items["reason"] = self.create_text(
            width / 2, height * 0.46 - self.fonts["value"][1] * 0.62,
            text="", fill=FG_WARN, font=self.fonts["tiny"],
        )
        self._items["value"] = self.create_text(
            width / 2, height * 0.46, text=self.value_text,
            fill=FG_DIM, font=self.fonts["value"],
        )
        self.create_text(
            width / 2, height * 0.46 + self.fonts["value"][1] * 0.72,
            text=self.unit, fill=FG_LABEL, font=self.fonts["unit"],
        )
        self._items["alarm"] = self.create_text(
            16, height - 34, text=self.alarm_text, anchor="w",
            fill=FG_DIM, font=self.fonts["tiny"],
        )

        gauge_w = (width - 32 - (self.SEGMENTS - 1) * 4) / self.SEGMENTS
        self._items["segments"] = []
        for i in range(self.SEGMENTS):
            x = 16 + i * (gauge_w + 4)
            self._items["segments"].append(
                self.create_rectangle(
                    x, height - 20, x + gauge_w, height - 16,
                    fill=FG_DIM, outline="",
                )
            )
        self._apply()

    def _apply(self):
        if not self._items:
            return
        self.itemconfigure(
            self._items["value"],
            text=self.value_text,
            fill=self.color if self.active else FG_DIM,
        )
        self.itemconfigure(
            self._items["reason"],
            text="" if self.active else self.reason,
        )
        self.itemconfigure(
            self._items["alarm"],
            fill=FG_WARN if (self.active and not self.in_range) else FG_DIM,
        )
        filled = int(round(self.conf * self.SEGMENTS))
        for i, item in enumerate(self._items["segments"]):
            self.itemconfigure(
                item, fill=self.color if i < filled else FG_DIM
            )

    def set_theme(self, outer, fill):
        self._fill = fill
        self.configure(bg=outer)
        if "bg" in self._items:
            self.itemconfigure(self._items["bg"], fill=fill)

    def update_state(self, value_text, active, conf, in_range=True, reason=""):
        self.value_text = value_text
        self.active = active
        self.conf = conf
        self.in_range = in_range
        self.reason = reason
        self._apply()


# ══════════════════════════════════════════════════════
#  파형 패널
# ══════════════════════════════════════════════════════

class WavePanel(tk.Canvas):
    GRID_STEP = 26

    def __init__(self, parent, fonts, label, color,
                 placeholder="AWAITING SIGNAL"):
        super().__init__(parent, bg=BG, highlightthickness=0)
        self.fonts = fonts
        self.label = label
        self.color = color
        self.placeholder = placeholder
        self.glow = blend(color, PANEL, 0.28)
        self.mid = blend(color, PANEL, 0.62)
        self.active = False
        self._fill = PANEL
        self._dot = None
        self._bg_item = None
        self.bind("<Configure>", lambda _e: self._layout())

    def _layout(self):
        width = self.winfo_width()
        height = self.winfo_height()

        if width < 40 or height < 40:
            return

        self.delete("chrome")
        self._bg_item = round_rect(
            self, 1, 1, width - 1, height - 1, CORNER,
            fill=self._fill, outline=HAIRLINE, tags="chrome",
        )
        for x in range(self.GRID_STEP, int(width) - 14, self.GRID_STEP):
            self.create_line(x, 34, x, height - 12, fill=GRID, tags="chrome")
        for y in range(34, int(height) - 12, self.GRID_STEP):
            self.create_line(14, y, width - 14, y, fill=GRID, tags="chrome")
        self.create_text(
            16, 18, text=self.label, anchor="w",
            fill=FG_LABEL, font=self.fonts["label"], tags="chrome",
        )
        self._dot = self.create_oval(
            width - 26, 12, width - 18, 20,
            fill=FG_DIM, outline="", tags="chrome",
        )
        self.tag_lower("chrome")
        self.set_active(self.active)

    def set_active(self, active):
        self.active = active
        if self._dot is not None:
            self.itemconfigure(self._dot, fill=self.color if active else FG_DIM)

    def set_theme(self, outer, fill):
        self._fill = fill
        self.configure(bg=outer)
        if self._bg_item is not None:
            self.itemconfigure(self._bg_item, fill=fill)

    def draw(self, values):
        self.delete("wave")
        width = self.winfo_width()
        height = self.winfo_height()

        if width < 40 or height < 40:
            return

        if not values or len(values) < 2:
            self.create_text(
                width / 2, height / 2 + 10, text=self.placeholder,
                fill=FG_DIM, font=self.fonts["tiny"], tags="wave",
            )
            return

        top = 42
        bottom = height - 20
        left = 18
        span_x = width - 36
        span_y = bottom - top
        last = len(values) - 1
        points = []

        for i, value in enumerate(values):
            points.append(left + span_x * i / last)
            points.append(bottom - span_y * value)

        for color, thickness in ((self.glow, 7), (self.mid, 4), (self.color, 2)):
            self.create_line(
                *points, fill=color, width=thickness,
                capstyle="round", joinstyle="round", tags="wave",
            )

        cx, cy = points[-2], points[-1]
        self.create_oval(
            cx - 6, cy - 6, cx + 6, cy + 6,
            fill=self.glow, outline="", tags="wave",
        )
        self.create_oval(
            cx - 2.5, cy - 2.5, cx + 2.5, cy + 2.5,
            fill=self.color, outline="", tags="wave",
        )


# ══════════════════════════════════════════════════════
#  하단 상태바
# ══════════════════════════════════════════════════════

class StatusBar(tk.Canvas):
    HEIGHT = 62
    GAUGE_W = 260

    def __init__(self, parent, fonts):
        super().__init__(parent, bg=BG, highlightthickness=0, height=self.HEIGHT)
        self.fonts = fonts
        self._fill = CARD
        self._title_fg = FG_TITLE
        self._label_fg = FG_LABEL
        self._items = {}
        self.badge = "STANDBY"
        self.detail = ""
        self.sub = ""
        self.accent = FG_DIM
        self.progress = None
        self.bind("<Configure>", lambda _e: self._layout())

    def _layout(self):
        width = self.winfo_width()
        height = self.winfo_height()

        if width < 40 or height < 30:
            return

        self.delete("all")
        self._items.clear()

        self._items["bg"] = round_rect(
            self, 1, 1, width - 1, height - 1, CORNER,
            fill=self._fill, outline=HAIRLINE,
        )
        self._items["lamp"] = self.create_oval(
            18, height / 2 - 5, 28, height / 2 + 5,
            fill=self.accent, outline="",
        )
        self._items["badge"] = self.create_text(
            40, height / 2, text=self.badge, anchor="w",
            fill=self.accent, font=self.fonts["badge"],
        )
        self._items["detail"] = self.create_text(
            210, height / 2 - 9, text=self.detail, anchor="w",
            fill=self._title_fg, font=self.fonts["detail"],
        )
        self._items["sub"] = self.create_text(
            210, height / 2 + 11, text=self.sub, anchor="w",
            fill=self._label_fg, font=self.fonts["tiny"],
        )

        gx1 = width - 22
        gx0 = gx1 - self.GAUGE_W
        self._items["gauge_bg"] = self.create_rectangle(
            gx0, height / 2 - 3, gx1, height / 2 + 3,
            fill=FG_DIM, outline="",
        )
        self._items["gauge_fg"] = self.create_rectangle(
            gx0, height / 2 - 3, gx0, height / 2 + 3,
            fill=self.accent, outline="",
        )
        self._items["gauge_pct"] = self.create_text(
            gx1, height / 2 - 15, text="", anchor="e",
            fill=self._label_fg, font=self.fonts["tiny"],
        )
        self._gauge_span = (gx0, gx1)
        self._apply()

    def _apply(self):
        if not self._items:
            return

        height = self.winfo_height()
        self.itemconfigure(self._items["lamp"], fill=self.accent)
        self.itemconfigure(self._items["badge"], text=self.badge, fill=self.accent)
        self.itemconfigure(self._items["detail"], text=self.detail)
        self.itemconfigure(self._items["sub"], text=self.sub)

        bbox = self.bbox(self._items["badge"])
        text_x = (bbox[2] + 24) if bbox else 210
        self.coords(self._items["detail"], text_x, height / 2 - 9)
        self.coords(self._items["sub"], text_x, height / 2 + 11)

        gx0, gx1 = self._gauge_span
        if self.progress is None:
            self.itemconfigure(self._items["gauge_bg"], state="hidden")
            self.itemconfigure(self._items["gauge_fg"], state="hidden")
            self.itemconfigure(self._items["gauge_pct"], text="")
            return

        pct = max(0.0, min(1.0, self.progress))
        self.itemconfigure(self._items["gauge_bg"], state="normal")
        self.itemconfigure(self._items["gauge_fg"], state="normal", fill=self.accent)
        self.coords(
            self._items["gauge_fg"],
            gx0, height / 2 - 3,
            gx0 + (gx1 - gx0) * pct, height / 2 + 3,
        )
        self.itemconfigure(self._items["gauge_pct"], text=f"{pct * 100:.0f}%")

    def set_theme(self, outer, fill, title=FG_TITLE, label=FG_LABEL):
        self._fill = fill
        self._title_fg = title
        self._label_fg = label
        self.configure(bg=outer)
        if not self._items:
            return
        self.itemconfigure(self._items["bg"], fill=fill)
        self.itemconfigure(self._items["detail"], fill=title)
        self.itemconfigure(self._items["sub"], fill=label)
        self.itemconfigure(self._items["gauge_pct"], fill=label)

    def update_state(self, badge, detail, sub, accent, progress=None):
        self.badge = badge
        self.detail = detail
        self.sub = sub
        self.accent = accent
        self.progress = progress
        self._apply()


# ══════════════════════════════════════════════════════
#  대시보드
# ══════════════════════════════════════════════════════

class Dashboard:
    MIN_CONF = 0.3
    REFRESH_MS = 200

    HR_RANGE = (60, 100)
    RR_RANGE = (12, 20)
    TEMP_RANGE = (36.1, 37.2)

    def __init__(self, root, worker):
        self.root = root
        self.worker = worker
        self.started = time.time()

        root.title("Bio-Guardian Patient Monitor")
        root.geometry("1024x640")
        root.minsize(820, 540)
        root.configure(bg=BG)

        display = pick_font(["DejaVu Sans Mono", "Consolas", "Roboto Mono", "Menlo"], "Courier")
        ui = pick_font(["Inter", "Segoe UI", "DejaVu Sans", "Helvetica Neue"], "Helvetica")

        self.fonts = {
            "value": (display, 46, "bold"),
            "unit": (ui, 10),
            "label": (ui, 11),
            "tiny": (display, 9),
            "badge": (display, 15, "bold"),
            "detail": (ui, 11),
        }

        self._bg_widgets = []
        self._build_header(ui, display)

        root.grid_columnconfigure(0, weight=0, minsize=196)
        root.grid_columnconfigure(1, weight=1)

        for row in (1, 2, 3):
            root.grid_rowconfigure(row, weight=1)

        self.hr_card = ValueCard(root, self.fonts, "HEART RATE", "BPM", HR_COLOR, "60 - 100")
        self.rr_card = ValueCard(root, self.fonts, "RESPIRATION", "BrPM", RR_COLOR, "12 - 20")
        self.temp_card = ValueCard(root, self.fonts, "TEMPERATURE", "°C", TEMP_COLOR, "36.1 - 37.2")

        self.ppg_panel = WavePanel(root, self.fonts, "PPG WAVEFORM", HR_COLOR)
        self.resp_panel = WavePanel(root, self.fonts, "RESPIRATION  ·  OPTICAL FLOW", RR_COLOR)
        self.temp_panel = WavePanel(root, self.fonts, "TEMPERATURE TREND", TEMP_COLOR, placeholder="NO SENSOR CONNECTED")

        self.cards = (self.hr_card, self.rr_card, self.temp_card)
        self.panels = (self.ppg_panel, self.resp_panel, self.temp_panel)

        for row, (card, panel) in enumerate(zip(self.cards, self.panels), start=1):
            card.grid(row=row, column=0, sticky="nsew", padx=(10, 5), pady=5)
            panel.grid(row=row, column=1, sticky="nsew", padx=(5, 10), pady=5)

        self.status_bar = StatusBar(root, self.fonts)
        self.status_bar.grid(row=4, column=0, columnspan=2, sticky="ew", padx=10, pady=(6, 2))

        self._build_footer(ui, display)
        self._theme_key = None

        root.bind("<F11>", self._toggle_fullscreen)
        root.bind("<Escape>", lambda _e: root.attributes("-fullscreen", False))
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh()

    def _build_header(self, ui, display):
        header = tk.Frame(self.root, bg=BG)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=12, pady=(10, 4))
        title = tk.Label(header, text="BIO-GUARDIAN", fg=FG_TITLE, bg=BG, font=(ui, 15, "bold"))
        title.pack(side="left")
        subtitle = tk.Label(header, text="   PATIENT MONITOR", fg=FG_LABEL, bg=BG, font=(ui, 11))
        subtitle.pack(side="left")
        self.clock = tk.Label(header, text="", fg=FG_LABEL, bg=BG, font=(display, 11))
        self.clock.pack(side="right")
        self._bg_widgets += [header, title, subtitle, self.clock]

    def _build_footer(self, ui, display):
        footer = tk.Frame(self.root, bg=BG)
        footer.grid(row=5, column=0, columnspan=2, sticky="ew", padx=14, pady=(2, 10))
        self.status = tk.Label(footer, text="", fg=FG_LABEL, bg=BG, font=(ui, 10))
        self.status.pack(side="left")
        self.sqi = tk.Label(footer, text="", fg=FG_DIM, bg=BG, font=(display, 10))
        self.sqi.pack(side="right")
        self._bg_widgets += [footer, self.status, self.sqi]

    def _toggle_fullscreen(self, _event=None):
        current = bool(self.root.attributes("-fullscreen"))
        self.root.attributes("-fullscreen", not current)

    def _describe(self, snap):
        anomaly = snap.get("anomaly")

        if not snap.get("anomaly_on"):
            return ("MONITOR ONLY", "이상탐지 비활성 (--no-anomaly)", "계측값만 표시합니다", FG_DIM, None, None)
        if anomaly is None:
            return ("STANDBY", "측정 시작 대기 중", "얼굴이 화면에 들어오면 기준 학습이 시작됩니다", FG_DIM, None, None)

        axes = "HR + RR + TEMP" if snap.get("use_temp") else "HR + RR"

        if anomaly["critical"]:
            return ("CRITICAL", f"절대 안전범위 이탈  ·  {anomaly['critical']}", "환자별 기준선과 무관한 즉시 경보입니다",
                    ALARM_THEMES["critical"]["accent"], None, "critical")
        if anomaly["state"] == "signal_lost":
            return ("SIGNAL LOST", "측정 신호가 끊겼습니다", f"마지막 기각 사유: {anomaly['reason'] or '-'}",
                    FG_WARN, None, "warn")
        if anomaly["alert"]:
            score = anomaly["score"]
            thr = anomaly["threshold"]
            sub = "판정 보류 중 - 현재 샘플 무효 (알림 유지)" if anomaly["state"] in ("invalid", "warmup") else (
                f"score {score:.3f} / 임계 {thr:.3f}" if score is not None and thr is not None else "")
            return ("ANOMALY", f"개인 기준선 이탈  ·  {anomaly['alert_reason']}", sub,
                    ALARM_THEMES["alert"]["accent"], None, "alert")
        if anomaly["baseline"] is None:
            return ("BASELINE LEARNING", f"환자별 기준 분포 학습 중  ·  {axes}",
                    f"시간 {anomaly['progress_time'] * 100:.0f}%   샘플 {anomaly['accepted']}/{snap.get('min_samples', 0)}   (둘 다 100%가 되어야 완료)",
                    RR_COLOR, anomaly["progress"], None)
        if anomaly["state"] in ("invalid", "warmup"):
            return ("HOLD", "판정 보류  ·  유효 신호 없음", f"기각 사유: {anomaly['reason'] or anomaly['state']}",
                    FG_WARN, None, None)

        base = anomaly["baseline"]
        base_text = f"기준선 HR {base[0]:.0f}  RR {base[1]:.0f}"
        if base[2] is not None:
            base_text += f"  TEMP {base[2]:.1f}"

        score = anomaly["score"]
        thr = anomaly["threshold"]
        sub = (f"score {score:.3f} / 임계 {thr:.3f}" if score is not None and thr is not None else "")
        return ("NORMAL", base_text, sub, FG_OK, None, None)

    def _apply_alarm_theme(self, level):
        if level is None:
            key = ("none",)
            if self._theme_key == key:
                return
            self._theme_key = key
            outer, card_fill, panel_fill = BG, CARD, PANEL
            title, label, dim = FG_TITLE, FG_LABEL, FG_DIM
        else:
            theme = ALARM_THEMES[level]
            phase = 0.5 - 0.5 * math.cos(2 * math.pi * (time.time() % theme["period"]) / theme["period"])
            outer = blend(theme["bg"][1], theme["bg"][0], phase)
            card_fill = blend(theme["card"][1], theme["card"][0], phase)
            panel_fill = blend(theme["panel"][1], theme["panel"][0], phase)
            title, label, dim = theme["title"], theme["label"], theme["dim"]
            key = (level, outer)
            if self._theme_key == key:
                return
            self._theme_key = key

        self.root.configure(bg=outer)
        for widget in self._bg_widgets:
            widget.configure(bg=outer)
        self.clock.configure(fg=label)
        self.status.configure(fg=label)
        self.sqi.configure(fg=dim)
        for card in self.cards:
            card.set_theme(outer, card_fill)
        for panel in self.panels:
            panel.set_theme(outer, panel_fill)
        self.status_bar.set_theme(outer, card_fill, title=title, label=label)

    def refresh(self):
        snap = self.worker.snapshot()

        hr_ok = snap["hr_conf"] >= self.MIN_CONF
        rr_ok = snap.get("rr_valid", False)
        temp_ok = snap["temp_conf"] >= self.MIN_CONF

        self.hr_card.update_state(
            f"{snap['hr']:.0f}" if hr_ok else "--",
            active=hr_ok,
            conf=snap["hr_conf"],
            in_range=self.HR_RANGE[0] <= snap["hr"] <= self.HR_RANGE[1],
            reason=snap.get("hr_reason", ""),
        )
        self.rr_card.update_state(
            f"{snap['rr']:.0f}" if rr_ok else "--",
            active=rr_ok,
            conf=snap["rr_conf"],
            in_range=self.RR_RANGE[0] <= snap["rr"] <= self.RR_RANGE[1],
            reason=snap.get("rr_reason", ""),
        )
        self.temp_card.update_state(
            f"{snap['temp']:.1f}" if temp_ok else "N/A",
            active=temp_ok,
            conf=snap["temp_conf"],
            in_range=(self.TEMP_RANGE[0] <= snap["temp"] <= self.TEMP_RANGE[1]),
            reason="NO SENSOR" if not snap.get("use_temp") else "NO FACE",
        )

        self.ppg_panel.set_active(hr_ok)
        self.resp_panel.set_active(rr_ok)
        self.temp_panel.set_active(temp_ok)

        self.ppg_panel.draw(snap["ppg"])
        self.resp_panel.draw(snap["resp"])
        self.temp_panel.draw(snap["temp_trend"])

        badge, detail, sub, accent, progress, level = self._describe(snap)
        self.status_bar.update_state(badge, detail, sub, accent, progress)
        self._apply_alarm_theme(level)

        elapsed = int(time.time() - self.started)
        self.clock.configure(
            text=f"{time.strftime('%H:%M:%S')}    ELAPSED {elapsed // 60:02d}:{elapsed % 60:02d}"
        )

        tracker_text = " · TRACK ON" if snap.get("tracker_active") else ""
        self.status.configure(text=snap["status"] + tracker_text)

        self.sqi.configure(
            text=(f"SQI   HR {snap['hr_conf']:.2f}   "
                  f"RR {snap['rr_conf']:.2f} [{snap.get('rr_source', '-')}]   "
                  f"TEMP {snap['temp_conf']:.2f}")
        )
        self.root.after(self.REFRESH_MS, self.refresh)

    def close(self):
        self.worker.stop()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="환자 모니터 대시보드")
    parser.add_argument("--camera", type=int, default=0, help="카메라 ID")
    parser.add_argument("--no-thermal", action="store_true", help="MLX90640 체온 측정을 비활성화")
    parser.add_argument("--thermal-stat", choices=["p90", "max"], default="p90", help="얼굴 영역 온도 대표값: 상위10%% 평균(p90) 또는 최고값(max)")
    parser.add_argument("--thermal-freq", type=int, default=4, choices=[1, 2, 4, 8], help="MLX90640 갱신 주파수(Hz)")
    parser.add_argument("--no-tracker", action="store_true", help="얼굴 팬/틸트 모터 추적을 비활성화 (OpenRB-150 미연결 상태 테스트용)")
    parser.add_argument("--no-anomaly", action="store_true", help="이상탐지를 비활성화하고 계측만 표시")
    parser.add_argument("--calib", type=float, default=180.0, help="기준 학습 초")
    parser.add_argument("--min-conf", type=float, default=0.30, help="open-rppg SQI 게이팅 임계")
    parser.add_argument("--out-pct", type=float, default=1.0, help="학습 분포 하위 몇 %%를 이상 경계로 사용할지")
    parser.add_argument("--rr-sigma", type=float, default=4.0, help="RR 측정 불확실성(BPM). 클수록 RR 오탐 감소")
    parser.add_argument("--temp-offset", type=float, default=TEMP_OFFSET_DEFAULT,
                        help=f"피부 표면 -> 심부 체온 보정(C). 기본 {TEMP_OFFSET_DEFAULT} "
                             "는 임시값이므로 기준 체온계로 실측해서 정할 것")
    parser.add_argument("--temp-ambient-gain", type=float, default=None,
                        help="실온 1C 하락당 추가 보정(C). temp_calib.py --fit 로 산출")
    parser.add_argument("--temp-ambient-ref", type=float, default=None,
                        help="오프셋을 측정한 기준 실온(C)")
    parser.add_argument("--temp-sigma", type=float, default=0.4, help="얼굴 온도 측정 불확실성(C). 기본 0.4 = 기준선 +-0.8C 봉투")
    parser.add_argument("--buzzer", action="store_true", help="이상 판정 시 액티브 부저를 울린다 (기본: 울리지 않음)")
    parser.add_argument("--buzzer-pin", type=int, default=17, help="액티브 부저가 연결된 GPIO 번호(BCM)")
    parser.add_argument("--dump", help="호흡 원신호를 .npz로 저장 (오프라인 튜닝용)")
    parser.add_argument("--rr-sqi-min", type=float, default=RR_SQI_MIN,
                        help=f"RR 추정치를 median에 넣는 SQI 문턱 (기본 {RR_SQI_MIN})")
    parser.add_argument("--rr-raw", action="store_true",
                        help="[테스트] median을 건너뛰고 매 초 원추정치를 그대로 표시")
    parser.add_argument("--rr-motion-rate", type=float, default=rr.MOTION_RATE,
                        help=f"움직임으로 간주할 어깨 이동 속도(px/s). 기본 {rr.MOTION_RATE:.0f}")
    parser.add_argument("--rr-mute-sec", type=float, default=rr.MUTE_SEC,
                        help=f"움직임 감지 후 무효화 시간(초). 기본 {rr.MUTE_SEC}")

    args = parser.parse_args()

    root = tk.Tk()

    worker = VitalWorker(
        camera_id=args.camera,
        use_thermal=not args.no_thermal,
        thermal_stat=args.thermal_stat,
        thermal_freq=args.thermal_freq,
        use_tracker=not args.no_tracker,
        use_anomaly=not args.no_anomaly,
        calib_sec=args.calib,
        min_conf=args.min_conf,
        out_pct=args.out_pct,
        rr_sigma=args.rr_sigma,
        temp_sigma=args.temp_sigma,
        temp_offset=args.temp_offset,
        temp_ambient_gain=args.temp_ambient_gain,
        temp_ambient_ref=args.temp_ambient_ref,
        use_buzzer=args.buzzer,
        buzzer_pin=args.buzzer_pin,
        dump=args.dump,
        rr_sqi_min=args.rr_sqi_min,
        rr_raw=args.rr_raw,
        rr_motion_rate=args.rr_motion_rate,
        rr_mute_sec=args.rr_mute_sec,
    )
    worker.start()

    Dashboard(root, worker)
    root.mainloop()


if __name__ == "__main__":
    main()