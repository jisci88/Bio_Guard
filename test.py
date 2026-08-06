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
"""

import argparse
import threading
import time
import tkinter as tk

import numpy as np
from scipy.signal import butter, filtfilt

import rppg

from vital_monitor import OpenRppgVitalTracker

# ══════════════════════════════════════════════════════
#  색상
# ══════════════════════════════════════════════════════

BG = "#12151a"
PANEL = "#232830"
CARD_HR = "#1c2b1f"
CARD_RR = "#1b2733"
CARD_TEMP = "#2c1f21"

FG_LABEL = "#c5cdd8"
FG_DIM = "#6b7280"

HR_COLOR = "#3ce85a"
RR_COLOR = "#5eb3f5"
TEMP_COLOR = "#f08080"


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
    def _normalize(signal, max_points=320):
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
                self._publish(status="측정 중")

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
                                "측정 중"
                                if face_visible
                                else "얼굴을 찾는 중"
                            ),
                        )

        except Exception as exc:
            self._publish(status=f"측정 중단: {exc}")


# ══════════════════════════════════════════════════════
#  위젯
# ══════════════════════════════════════════════════════

class ValueCard(tk.Frame):
    """왼쪽 열의 수치 카드."""

    def __init__(self, parent, title, unit, color, bg):
        super().__init__(parent, bg=bg, padx=8, pady=10)

        self.color = color

        tk.Label(
            self,
            text=title,
            fg=FG_LABEL,
            bg=bg,
            font=("Helvetica", 13),
        ).pack()

        self._value = tk.Label(
            self,
            text="--",
            fg=FG_DIM,
            bg=bg,
            font=("Helvetica", 38, "bold"),
        )
        self._value.pack()

        tk.Label(
            self,
            text=unit,
            fg=FG_LABEL,
            bg=bg,
            font=("Helvetica", 12),
        ).pack()

    def set_value(self, text, active=True):
        self._value.configure(
            text=text,
            fg=self.color if active else FG_DIM,
        )


class WavePanel(tk.Frame):
    """오른쪽 열의 파형 패널."""

    def __init__(self, parent, title, color, placeholder="신호 대기 중"):
        super().__init__(parent, bg=PANEL, padx=10, pady=8)

        self.color = color
        self.placeholder = placeholder

        header = tk.Frame(self, bg=PANEL)
        header.pack(fill="x")

        tk.Label(
            header,
            text=title,
            fg=FG_LABEL,
            bg=PANEL,
            font=("Helvetica", 12),
        ).pack(side="left")

        self._dot = tk.Label(
            header,
            text="\u25cf",
            fg=FG_DIM,
            bg=PANEL,
            font=("Helvetica", 12),
        )
        self._dot.pack(side="right")

        self.canvas = tk.Canvas(
            self,
            bg=PANEL,
            height=100,
            highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)

    def set_active(self, active):
        self._dot.configure(fg=self.color if active else FG_DIM)

    def draw(self, values):
        self.canvas.delete("wave")

        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()

        if width < 20 or height < 20:
            return

        if not values or len(values) < 2:
            self.canvas.create_text(
                width / 2,
                height / 2,
                text=self.placeholder,
                fill=FG_DIM,
                font=("Helvetica", 11),
                tags="wave",
            )
            return

        pad = 8
        span_x = width - 2 * pad
        span_y = height - 2 * pad
        last = len(values) - 1

        points = []

        for i, value in enumerate(values):
            points.append(pad + span_x * i / last)
            points.append(height - pad - span_y * value)

        self.canvas.create_line(
            *points,
            fill=self.color,
            width=2,
            tags="wave",
        )


# ══════════════════════════════════════════════════════
#  대시보드
# ══════════════════════════════════════════════════════

class Dashboard:
    MIN_CONF = 0.3
    REFRESH_MS = 150

    def __init__(self, root, worker):
        self.root = root
        self.worker = worker

        root.title("Bio-Guardian Patient Monitor")
        root.geometry("900x580")
        root.configure(bg=BG)

        tk.Label(
            root,
            text="Bio-Guardian Patient Monitor",
            fg="#a8c7fa",
            bg=BG,
            font=("Helvetica", 16, "bold"),
            anchor="w",
            padx=12,
            pady=8,
        ).grid(row=0, column=0, columnspan=2, sticky="ew")

        root.grid_columnconfigure(0, weight=0, minsize=200)
        root.grid_columnconfigure(1, weight=1)

        for row in (1, 2, 3):
            root.grid_rowconfigure(row, weight=1)

        self.hr_card = ValueCard(root, "Heart Rate", "BPM", HR_COLOR, CARD_HR)
        self.rr_card = ValueCard(root, "Respiration", "RR", RR_COLOR, CARD_RR)
        self.temp_card = ValueCard(
            root, "Temperature", "\u00b0C", TEMP_COLOR, CARD_TEMP
        )

        self.ppg_panel = WavePanel(root, "PPG Wave", HR_COLOR)
        self.resp_panel = WavePanel(root, "Respiration Wave", RR_COLOR)
        self.temp_panel = WavePanel(
            root, "Temperature Trend", TEMP_COLOR, placeholder="NO SENSOR"
        )

        cards = (self.hr_card, self.rr_card, self.temp_card)
        panels = (self.ppg_panel, self.resp_panel, self.temp_panel)

        for row, (card, panel) in enumerate(zip(cards, panels), start=1):
            card.grid(row=row, column=0, sticky="nsew", padx=(8, 4), pady=4)
            panel.grid(row=row, column=1, sticky="nsew", padx=(4, 8), pady=4)

        self.temp_card.set_value("N/A", active=False)

        self.status = tk.Label(
            root,
            text="",
            fg=FG_DIM,
            bg=BG,
            font=("Helvetica", 10),
            anchor="w",
            padx=12,
            pady=4,
        )
        self.status.grid(row=4, column=0, columnspan=2, sticky="ew")

        root.protocol("WM_DELETE_WINDOW", self.close)
        self.refresh()

    def refresh(self):
        snap = self.worker.snapshot()

        hr_ok = snap["hr_conf"] >= self.MIN_CONF
        rr_ok = snap["rr_conf"] >= self.MIN_CONF

        self.hr_card.set_value(
            f"{snap['hr']:.0f}" if hr_ok else "--", active=hr_ok
        )
        self.rr_card.set_value(
            f"{snap['rr']:.0f}" if rr_ok else "--", active=rr_ok
        )

        self.ppg_panel.set_active(hr_ok)
        self.resp_panel.set_active(rr_ok)

        self.ppg_panel.draw(snap["ppg"])
        self.resp_panel.draw(snap["resp"])
        self.temp_panel.draw(None)

        self.status.configure(
            text=(
                f"{snap['status']}   "
                f"SQI  HR {snap['hr_conf']:.2f} / RR {snap['rr_conf']:.2f}"
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
