"""
open-rppg 실측값을 Tkinter 대시보드로 표시한다.

  심박수    : vital_monitor.OpenRppgVitalTracker (model.hr)
  호흡수    : 동일 tracker (hrv["breathingrate"] * 60)
  PPG 파형  : model.bvp(start=-6)
  호흡 파형 : model.bvp(raw=True, start=-30)을 0.1~0.5Hz 대역통과.
              open-rppg는 호흡 파형을 직접 주지 않으므로,
              BVP의 호흡성 진폭 변동(RIIV)을 뽑아 파형으로 쓴다.
  체온      : 측정 수단 없음. 값은 N/A, 트렌드는 NO SENSOR로 유지한다.

스레드 규칙:
  rppg / OpenCV 호출은 VitalWorker 스레드에서만,
  Tk 위젯 갱신은 메인 스레드에서만 한다. 둘 사이는 락으로 보호된
  스냅샷 dict 하나로만 주고받는다.

실행:
    python vital_dashboard.py --camera 0

    F11 전체화면 전환 / ESC 전체화면 해제
"""

import argparse
import threading
import time
import tkinter as tk
from tkinter import font as tkfont

import numpy as np
from scipy.signal import butter, filtfilt

import rppg

from vital_monitor import OpenRppgVitalTracker

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

HR_COLOR = "#35F07F"
RR_COLOR = "#58C8F5"
TEMP_COLOR = "#FF8A7A"

CORNER = 12


# ══════════════════════════════════════════════════════
#  그리기 도우미
# ══════════════════════════════════════════════════════

def blend(fg, bg, ratio):
    """Canvas에 알파가 없으므로 색을 미리 섞어 반투명을 흉내낸다."""
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
    """시스템에 실제로 설치된 첫 후보를 고른다."""
    available = {name.lower() for name in tkfont.families()}

    for name in candidates:
        if name.lower() in available:
            return name

    return fallback


# ══════════════════════════════════════════════════════
#  측정 워커
# ══════════════════════════════════════════════════════

class VitalWorker(threading.Thread):
    """카메라와 open-rppg를 담당하고 최신 상태를 스냅샷으로 노출한다."""

    PPG_WINDOW_SEC = 6.0
    RESP_WINDOW_SEC = 30.0

    HR_INTERVAL = 2.0
    RR_INTERVAL = 10.0
    WAVE_INTERVAL = 0.25

    def __init__(self, camera_id=0):
        super().__init__(daemon=True)
        self.camera_id = camera_id

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._snapshot = {
            "status": "모델 로딩 중...",
            "face": False,
            "hr": 0.0,
            "hr_conf": 0.0,
            "rr": 0.0,
            "rr_conf": 0.0,
            "ppg": None,
            "resp": None,
        }

    def snapshot(self):
        with self._lock:
            return dict(self._snapshot)

    def stop(self):
        self._stop.set()

    def _publish(self, **kwargs):
        with self._lock:
            self._snapshot.update(kwargs)

    # ── 파형 추출 ──────────────────────────────────────

    @staticmethod
    def _normalize(signal, max_points=260):
        """파형을 0~1로 정규화하고 그리기 좋게 다운샘플한다."""
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

    def _resp_wave(self, model):
        """raw BVP를 0.1~0.5Hz로 대역통과해 호흡성 변동을 뽑는다."""
        try:
            bvp, ts = model.bvp(raw=True, start=-self.RESP_WINDOW_SEC)
        except Exception:
            return None

        bvp = np.asarray(bvp, dtype=float)
        ts = np.asarray(ts, dtype=float)

        if bvp.size < 64 or ts.size != bvp.size:
            return None

        span = ts[-1] - ts[0]

        if span <= 0:
            return None

        fs = (ts.size - 1) / span

        if not 5.0 <= fs <= 120.0:
            return None

        nyq = fs / 2.0
        b, a = butter(2, [0.1 / nyq, 0.5 / nyq], btype="band")

        try:
            filtered = filtfilt(b, a, bvp)
        except ValueError:
            return None

        return self._normalize(filtered)

    # ── 메인 루프 ──────────────────────────────────────

    def run(self):
        try:
            model = rppg.Model("ME-flow.rlap")
        except Exception as exc:
            self._publish(status=f"모델 로딩 실패: {exc}")
            return

        vitals = OpenRppgVitalTracker(model)
        self._publish(status="카메라 연결 중...")

        last_hr = last_rr = last_wave = 0.0

        try:
            with model.video_capture(self.camera_id):
                self._publish(status="MONITORING")

                for _frame, box in model.preview:
                    if self._stop.is_set():
                        break

                    now = time.time()
                    face_visible = box is not None

                    if not face_visible:
                        vitals.invalidate()

                    if now - last_rr >= self.RR_INTERVAL:
                        vitals.update_rr(face_visible)
                        last_rr = now

                    if now - last_hr >= self.HR_INTERVAL:
                        vitals.update_hr(face_visible)
                        last_hr = now

                    if now - last_wave >= self.WAVE_INTERVAL:
                        last_wave = now

                        self._publish(
                            face=face_visible,
                            hr=vitals.hr_bpm,
                            hr_conf=vitals.hr_conf,
                            rr=vitals.rr_bpm,
                            rr_conf=vitals.rr_conf,
                            ppg=self._ppg_wave(model),
                            resp=self._resp_wave(model),
                            status=(
                                "MONITORING"
                                if face_visible
                                else "NO SUBJECT IN FRAME"
                            ),
                        )

        except Exception as exc:
            self._publish(status=f"측정 중단: {exc}")


# ══════════════════════════════════════════════════════
#  수치 카드
# ══════════════════════════════════════════════════════

class ValueCard(tk.Canvas):
    """좌측 열의 수치 카드. 라운드 배경 + 대형 숫자 + SQI 게이지."""

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

        self._items = {}
        self.bind("<Configure>", lambda _e: self._layout())

    def _layout(self):
        width = self.winfo_width()
        height = self.winfo_height()

        if width < 40 or height < 40:
            return

        self.delete("all")
        self._items.clear()

        round_rect(
            self, 1, 1, width - 1, height - 1, CORNER,
            fill=CARD, outline=HAIRLINE,
        )

        self.create_text(
            16, 18, text=self.label, anchor="w",
            fill=FG_LABEL, font=self.fonts["label"],
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
            self._items["alarm"],
            fill=FG_WARN if (self.active and not self.in_range) else FG_DIM,
        )

        filled = int(round(self.conf * self.SEGMENTS))

        for i, item in enumerate(self._items["segments"]):
            self.itemconfigure(
                item, fill=self.color if i < filled else FG_DIM
            )

    def update_state(self, value_text, active, conf, in_range=True):
        self.value_text = value_text
        self.active = active
        self.conf = conf
        self.in_range = in_range
        self._apply()


# ══════════════════════════════════════════════════════
#  파형 패널
# ══════════════════════════════════════════════════════

class WavePanel(tk.Canvas):
    """우측 열의 파형 패널. ECG 그리드 위에 글로우 트레이스를 그린다."""

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
        self._dot = None

        self.bind("<Configure>", lambda _e: self._layout())

    def _layout(self):
        width = self.winfo_width()
        height = self.winfo_height()

        if width < 40 or height < 40:
            return

        self.delete("chrome")

        round_rect(
            self, 1, 1, width - 1, height - 1, CORNER,
            fill=PANEL, outline=HAIRLINE, tags="chrome",
        )

        for x in range(self.GRID_STEP, int(width) - 14, self.GRID_STEP):
            self.create_line(
                x, 34, x, height - 12, fill=GRID, tags="chrome"
            )

        for y in range(34, int(height) - 12, self.GRID_STEP):
            self.create_line(
                14, y, width - 14, y, fill=GRID, tags="chrome"
            )

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
            self.itemconfigure(
                self._dot, fill=self.color if active else FG_DIM
            )

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

        for color, thickness in (
            (self.glow, 7),
            (self.mid, 4),
            (self.color, 2),
        ):
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
#  대시보드
# ══════════════════════════════════════════════════════

class Dashboard:
    MIN_CONF = 0.3
    REFRESH_MS = 200

    HR_RANGE = (60, 100)
    RR_RANGE = (12, 20)

    def __init__(self, root, worker):
        self.root = root
        self.worker = worker
        self.started = time.time()

        root.title("Bio-Guardian Patient Monitor")
        root.geometry("1024x600")
        root.minsize(820, 500)
        root.configure(bg=BG)

        display = pick_font(
            ["DejaVu Sans Mono", "Consolas", "Roboto Mono", "Menlo"],
            "Courier",
        )
        ui = pick_font(
            ["Inter", "Segoe UI", "DejaVu Sans", "Helvetica Neue"],
            "Helvetica",
        )

        self.fonts = {
            "value": (display, 46, "bold"),
            "unit": (ui, 10),
            "label": (ui, 11),
            "tiny": (display, 9),
        }

        self._build_header(ui, display)

        root.grid_columnconfigure(0, weight=0, minsize=196)
        root.grid_columnconfigure(1, weight=1)

        for row in (1, 2, 3):
            root.grid_rowconfigure(row, weight=1)

        self.hr_card = ValueCard(
            root, self.fonts, "HEART RATE", "BPM", HR_COLOR, "60 - 100"
        )
        self.rr_card = ValueCard(
            root, self.fonts, "RESPIRATION", "BrPM", RR_COLOR, "12 - 20"
        )
        self.temp_card = ValueCard(
            root, self.fonts, "TEMPERATURE", "\u00b0C", TEMP_COLOR,
            "36.1 - 37.2",
        )

        self.ppg_panel = WavePanel(
            root, self.fonts, "PPG WAVEFORM", HR_COLOR
        )
        self.resp_panel = WavePanel(
            root, self.fonts, "RESPIRATION  \u00b7  RIIV DERIVED", RR_COLOR
        )
        self.temp_panel = WavePanel(
            root, self.fonts, "TEMPERATURE TREND", TEMP_COLOR,
            placeholder="NO SENSOR CONNECTED",
        )

        cards = (self.hr_card, self.rr_card, self.temp_card)
        panels = (self.ppg_panel, self.resp_panel, self.temp_panel)

        for row, (card, panel) in enumerate(zip(cards, panels), start=1):
            card.grid(row=row, column=0, sticky="nsew", padx=(10, 5), pady=5)
            panel.grid(row=row, column=1, sticky="nsew", padx=(5, 10), pady=5)

        self.temp_card.update_state("N/A", active=False, conf=0.0)

        self._build_footer(ui, display)

        root.bind("<F11>", self._toggle_fullscreen)
        root.bind("<Escape>", lambda _e: root.attributes("-fullscreen", False))
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.refresh()

    def _build_header(self, ui, display):
        header = tk.Frame(self.root, bg=BG)
        header.grid(
            row=0, column=0, columnspan=2,
            sticky="ew", padx=12, pady=(10, 4),
        )

        tk.Label(
            header, text="BIO-GUARDIAN", fg=FG_TITLE, bg=BG,
            font=(ui, 15, "bold"),
        ).pack(side="left")

        tk.Label(
            header, text="   PATIENT MONITOR", fg=FG_LABEL, bg=BG,
            font=(ui, 11),
        ).pack(side="left")

        self.clock = tk.Label(
            header, text="", fg=FG_LABEL, bg=BG, font=(display, 11)
        )
        self.clock.pack(side="right")

    def _build_footer(self, ui, display):
        footer = tk.Frame(self.root, bg=BG)
        footer.grid(
            row=4, column=0, columnspan=2,
            sticky="ew", padx=14, pady=(2, 10),
        )

        self.status = tk.Label(
            footer, text="", fg=FG_LABEL, bg=BG, font=(ui, 10)
        )
        self.status.pack(side="left")

        self.sqi = tk.Label(
            footer, text="", fg=FG_DIM, bg=BG, font=(display, 10)
        )
        self.sqi.pack(side="right")

    def _toggle_fullscreen(self, _event=None):
        current = bool(self.root.attributes("-fullscreen"))
        self.root.attributes("-fullscreen", not current)

    def refresh(self):
        snap = self.worker.snapshot()

        hr_ok = snap["hr_conf"] >= self.MIN_CONF
        rr_ok = snap["rr_conf"] >= self.MIN_CONF

        self.hr_card.update_state(
            f"{snap['hr']:.0f}" if hr_ok else "--",
            active=hr_ok,
            conf=snap["hr_conf"],
            in_range=self.HR_RANGE[0] <= snap["hr"] <= self.HR_RANGE[1],
        )

        self.rr_card.update_state(
            f"{snap['rr']:.0f}" if rr_ok else "--",
            active=rr_ok,
            conf=snap["rr_conf"],
            in_range=self.RR_RANGE[0] <= snap["rr"] <= self.RR_RANGE[1],
        )

        self.ppg_panel.set_active(hr_ok)
        self.resp_panel.set_active(rr_ok)

        self.ppg_panel.draw(snap["ppg"])
        self.resp_panel.draw(snap["resp"])
        self.temp_panel.draw(None)

        elapsed = int(time.time() - self.started)

        self.clock.configure(
            text=(
                f"{time.strftime('%H:%M:%S')}    "
                f"ELAPSED {elapsed // 60:02d}:{elapsed % 60:02d}"
            )
        )

        self.status.configure(text=snap["status"])

        self.sqi.configure(
            text=(
                f"SQI   HR {snap['hr_conf']:.2f}   "
                f"RR {snap['rr_conf']:.2f}"
            )
        )

        self.root.after(self.REFRESH_MS, self.refresh)

    def close(self):
        self.worker.stop()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="환자 모니터 대시보드")
    parser.add_argument("--camera", type=int, default=0, help="카메라 ID")
    args = parser.parse_args()

    worker = VitalWorker(camera_id=args.camera)
    worker.start()

    root = tk.Tk()
    Dashboard(root, worker)
    root.mainloop()


if __name__ == "__main__":
    main()
