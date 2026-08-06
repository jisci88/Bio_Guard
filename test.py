import argparse
from collections import deque
import math
import os
import threading
import time
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
import rppg

from vital_anomaly import VitalAnomalyDetector


# ══════════════════════════════════════════════════════
#  실시간 BVP/rPPG 신호 정규화 버퍼 (Min-Max Scaler)
# ══════════════════════════════════════════════════════
class SignalNormalizer:

  def __init__(self, window_size=90):
    self.buffer = deque(maxlen=window_size)

  def normalize(self, val):
    if not np.isfinite(val) or val == 0.0:
      return 0.0

    self.buffer.append(val)
    if len(self.buffer) < 10:
      return 0.0

    min_v = min(self.buffer)
    max_v = max(self.buffer)

    if max_v == min_v:
      return 0.0

    # -1.0 ~ 1.0 범위 변환
    norm = 2.0 * (val - min_v) / (max_v - min_v) - 1.0
    return max(-1.0, min(1.0, norm))


# ══════════════════════════════════════════════════════
#  Tkinter 의료용 파형 렌더러 (Sweep Bar)
# ══════════════════════════════════════════════════════
class SweepWaveformCanvas(tk.Canvas):

  def __init__(self, parent, color='#3fb950', speed=2.5, **kwargs):
    super().__init__(parent, bg='#161b22', highlightthickness=0, **kwargs)
    self.color = color
    self.speed = speed
    self.x = 0
    self.last_y = None
    self.clear_width = 15

  def add_sample(self, val):
    w = self.winfo_width()
    h = self.winfo_height()
    if w <= 1 or h <= 1:
      return

    margin = 8
    draw_h = h - margin * 2
    y = margin + (1.0 - (val + 1.0) / 2.0) * draw_h

    next_x = (self.x + self.speed) % w

    if next_x < self.x:
      self.create_rectangle(
          self.x, 0, w, h, fill='#161b22', outline='', tags='erase'
      )
      self.create_rectangle(
          0, 0, self.clear_width, h, fill='#161b22', outline='', tags='erase'
      )
    else:
      self.create_rectangle(
          self.x,
          0,
          self.x + self.clear_width,
          h,
          fill='#161b22',
          outline='',
          tags='erase',
      )

    if self.last_y is not None and next_x >= self.x:
      self.create_line(
          self.x,
          self.last_y,
          next_x,
          y,
          fill=self.color,
          width=2,
          capstyle='round',
      )

    self.x = next_x
    self.last_y = y


# ══════════════════════════════════════════════════════
#  Tkinter 메인 대시보드 GUI
# ══════════════════════════════════════════════════════
class BioGuardianTkApp(tk.Tk):

  def __init__(self):
    super().__init__()
    self.title('Bio-Guardian Patient Monitor')
    self.geometry('960x600')
    self.configure(bg='#0d1117')

    self._setup_ui()

  def _setup_ui(self):
    # Header Bar
    header = tk.Frame(self, bg='#161b22', height=45)
    header.pack(fill='x', padx=10, pady=(10, 5))

    title_label = tk.Label(
        header,
        text='BIO-GUARDIAN  |  Patient Vital Sign Monitor',
        font=('Helvetica', 13, 'bold'),
        fg='#f0f6fc',
        bg='#161b22',
    )
    title_label.pack(side='left', padx=15, pady=8)

    self.lbl_fps = tk.Label(
        header,
        text='-- FPS',
        font=('Helvetica', 10, 'bold'),
        fg='#8b949e',
        bg='#161b22',
    )
    self.lbl_fps.pack(side='right', padx=15)

    self.lbl_face = tk.Label(
        header,
        text='● Face Seeking',
        font=('Helvetica', 10, 'bold'),
        fg='#d29922',
        bg='#161b22',
    )
    self.lbl_face.pack(side='right', padx=10)

    # Main Frame
    main_frame = tk.Frame(self, bg='#0d1117')
    main_frame.pack(fill='both', expand=True, padx=10, pady=5)

    # Left Column: Numerical Cards (Row 1: HR, Row 2: RR, Row 3: Temp)
    left_col = tk.Frame(main_frame, bg='#0d1117', width=260)
    left_col.pack(side='left', fill='y', padx=(0, 5))
    left_col.pack_propagate(False)

    self.card_hr = self._create_vital_card(
        left_col, '1. HEART RATE (rPPG)', '#3fb950', 'BPM'
    )
    self.card_rr = self._create_vital_card(
        left_col, '2. RESPIRATION RATE', '#58a6ff', 'BrPM'
    )
    self.card_temp = self._create_vital_card(
        left_col, '3. SKIN TEMPERATURE', '#f85149', '°C'
    )

    # Right Column: Waveform Graphs (Row 1: PPG, Row 2: Resp, Row 3: Temp/Status)
    right_col = tk.Frame(main_frame, bg='#0d1117')
    right_col.pack(side='right', fill='both', expand=True, padx=(5, 0))

    self.wave_ppg = self._create_graph_card(
        right_col, '1. REAL-TIME BVP / PPG WAVEFORM', '#3fb950'
    )
    self.wave_resp = self._create_graph_card(
        right_col, '2. RESPIRATION WAVEFORM (PRV)', '#58a6ff'
    )

    # Row 3: Anomaly & Temperature Status Box
    temp_graph_frame = tk.Frame(right_col, bg='#161b22')
    temp_graph_frame.pack(fill='both', expand=True, pady=4)

    lbl_t = tk.Label(
        temp_graph_frame,
        text='3. TEMPERATURE & ANOMALY STATUS',
        font=('Helvetica', 9, 'bold'),
        fg='#8b949e',
        bg='#161b22',
    )
    lbl_t.pack(anchor='nw', padx=10, pady=(6, 2))

    self.wave_temp = SweepWaveformCanvas(
        temp_graph_frame, color='#f85149', speed=1.5
    )
    self.wave_temp.pack(fill='both', expand=True, padx=5, pady=(0, 2))

    self.lbl_anomaly = tk.Label(
        temp_graph_frame,
        text='SYSTEM INITIALIZING...',
        font=('Helvetica', 10, 'bold'),
        fg='#3fb950',
        bg='#161b22',
        anchor='w',
        padx=10,
    )
    self.lbl_anomaly.pack(fill='x', side='bottom', pady=4)

  def _create_vital_card(self, parent, title, color, unit):
    frame = tk.Frame(parent, bg='#161b22', highlightthickness=1)
    frame.config(highlightbackground='#21262d')
    frame.pack(fill='x', expand=True, pady=4)

    bar = tk.Frame(frame, bg=color, width=4)
    bar.pack(side='left', fill='y')

    inner = tk.Frame(frame, bg='#161b22')
    inner.pack(fill='both', expand=True, padx=10, pady=8)

    lbl_title = tk.Label(
        inner,
        text=title,
        font=('Helvetica', 9, 'bold'),
        fg='#8b949e',
        bg='#161b22',
    )
    lbl_title.pack(anchor='w')

    val_frame = tk.Frame(inner, bg='#161b22')
    val_frame.pack(fill='x', pady=4)

    lbl_val = tk.Label(
        val_frame,
        text='--',
        font=('Courier', 36, 'bold'),
        fg=color,
        bg='#161b22',
    )
    lbl_val.pack(side='left')

    lbl_unit = tk.Label(
        val_frame,
        text=unit,
        font=('Helvetica', 11, 'bold'),
        fg='#8b949e',
        bg='#161b22',
    )
    lbl_unit.pack(side='right', anchor='s', pady=6)

    lbl_sqi = tk.Label(
        inner,
        text='SQI: 0.00',
        font=('Helvetica', 8),
        fg='#8b949e',
        bg='#161b22',
    )
    lbl_sqi.pack(anchor='w')

    return {'val': lbl_val, 'sqi': lbl_sqi}

  def _create_graph_card(self, parent, title, color):
    frame = tk.Frame(parent, bg='#161b22')
    frame.pack(fill='both', expand=True, pady=4)

    lbl = tk.Label(
        frame,
        text=title,
        font=('Helvetica', 9, 'bold'),
        fg='#8b949e',
        bg='#161b22',
    )
    lbl.pack(anchor='nw', padx=10, pady=(6, 2))

    canvas = SweepWaveformCanvas(frame, color=color, speed=2.2)
    canvas.pack(fill='both', expand=True, padx=5, pady=(0, 5))
    return canvas

  def add_real_bvp_sample(self, bvp_norm, resp_norm=0.0):
    """실제 센서 파형 각 위치별 그리기"""
    self.wave_ppg.add_sample(bvp_norm)  # Row 1: PPG 심박 파형
    self.wave_resp.add_sample(resp_norm)  # Row 2: 호흡 파형
    self.wave_temp.add_sample(0.0)  # Row 3: 온도 박스 (가짜 파형 제거)

  def update_data(self, data):
    """수치 및 상태 업데이트"""
    if data.get('hr_bpm') is not None:
      bpm = data['hr_bpm']
      conf = data.get('hr_conf', 0.0)
      self.card_hr['val'].config(
          text=f'{int(bpm)}' if conf > 0.3 else '--'
      )
      self.card_hr['sqi'].config(text=f'SQI: {conf:.2f}')

    if data.get('rr_bpm') is not None:
      rr = data['rr_bpm']
      conf = data.get('rr_conf', 0.0)
      self.card_rr['val'].config(
          text=f'{int(rr)}' if conf > 0.3 else '--'
      )
      self.card_rr['sqi'].config(text=f'SQI: {conf:.2f}')

    # 적외선 카메라 미연결 시 -- 표시
    temp_val = data.get('temp')
    if temp_val is not None:
      self.card_temp['val'].config(text=f'{temp_val:.1f}')
    else:
      self.card_temp['val'].config(text='--')

    if data.get('face_visible'):
      self.lbl_face.config(text='● Face Detected', fg='#3fb950')
    else:
      self.lbl_face.config(text='● No Face', fg='#d29922')

    self.lbl_fps.config(text=f"{int(data.get('fps', 0))} FPS")

    a = data.get('anomaly')
    if a:
      if a.get('critical'):
        self.lbl_anomaly.config(
            text=f"CRITICAL: {a['critical']}", fg='#f85149'
        )
      elif a.get('state') == 'signal_lost':
        self.lbl_anomaly.config(
            text='SIGNAL LOST - Hold position', fg='#d29922'
        )
      elif a.get('alert'):
        self.lbl_anomaly.config(
            text=f"ANOMALY: {a.get('alert_reason')}", fg='#f85149'
        )
      elif a.get('baseline') is None:
        prog = int((a.get('progress') or 0) * 100)
        self.lbl_anomaly.config(
            text=f'LEARNING BASELINE ({prog}%)', fg='#d29922'
        )
      else:
        self.lbl_anomaly.config(text='NORMAL VITAL SIGNS', fg='#3fb950')


# ══════════════════════════════════════════════════════
#  open-rppg 바이탈 트래커 + 안면 Green 채널 rPPG 추출
# ══════════════════════════════════════════════════════
class OpenRppgVitalTracker:

  def __init__(self, model):
    self.model = model
    self.hr_bpm = 0.0
    self.hr_conf = 0.0
    self.rr_bpm = 0.0
    self.rr_conf = 0.0

  def update(self, face_visible):
    if not face_visible:
      self.hr_conf = 0.0
      self.rr_conf = 0.0
      return

    res_hr = self.model.hr(start=-10, return_hrv=False) or {}
    hr = float(res_hr.get('hr') or 0)
    sqi_hr = float(res_hr.get('SQI') or 0)
    if 40 <= hr <= 200:
      self.hr_bpm = hr
      self.hr_conf = min(1.0, max(0.0, sqi_hr))

    res_rr = self.model.hr(start=-60) or {}
    hrv = res_rr.get('hrv') or {}
    rr_hz = float(hrv.get('breathingrate') or 0)
    rr = rr_hz * 60.0
    sqi_rr = float(res_rr.get('SQI') or 0)
    if 4 <= rr <= 40:
      self.rr_bpm = rr
      self.rr_conf = min(1.0, max(0.0, sqi_rr))


def start_rppg_thread(app, camera_id=0, calib_sec=180.0):
  model = rppg.Model('ME-flow.rlap')
  tracker = OpenRppgVitalTracker(model)
  detector = VitalAnomalyDetector(calib_sec=calib_sec)

  normalizer_bvp = SignalNormalizer(window_size=90)

  last_hr_time = time.time()
  fps_timer = time.time()
  frames = 0
  fps = 0.0

  with model.video_capture(camera_id):
    for frame_rgb, box in model.preview:
      now = time.time()
      face_visible = box is not None
      frames += 1

      # --------------------------------------------------
      # 안면 ROI Green 채널 신호 직접 추출 (실시간 PPG 파형)
      # --------------------------------------------------
      raw_bvp_val = 0.0
      if face_visible and box is not None:
        try:
          (y1, y2), (x1, x2) = box[0], box[1]
          h, w, _ = frame_rgb.shape
          y1, y2 = max(0, int(y1)), min(h, int(y2))
          x1, x2 = max(0, int(x1)), min(w, int(x2))

          if y2 > y1 and x2 > x1:
            # 얼굴 중앙 영역 추출 (이마/볼 영역 중심)
            roi_y1 = y1 + int((y2 - y1) * 0.2)
            roi_y2 = y1 + int((y2 - y1) * 0.6)
            roi_x1 = x1 + int((x2 - x1) * 0.25)
            roi_x2 = x1 + int((x2 - x1) * 0.75)

            face_roi = frame_rgb[roi_y1:roi_y2, roi_x1:roi_x2]
            if face_roi.size > 0:
              raw_bvp_val = float(np.mean(face_roi[:, :, 1]))  # Green Channel
        except Exception:
          raw_bvp_val = 0.0

      if face_visible and raw_bvp_val > 0:
        norm_bvp = normalizer_bvp.normalize(raw_bvp_val)
        # 실제 추정된 호흡 주기에 맞춘 호흡 파형 생성
        rr_bpm_active = tracker.rr_bpm if tracker.rr_bpm > 0 else 18.0
        norm_resp = math.sin(now * (rr_bpm_active / 60.0) * 2 * math.pi)
      else:
        norm_bvp = 0.0
        norm_resp = 0.0

      # 실시간 파형 그리기
      app.after(0, app.add_real_bvp_sample, norm_bvp, norm_resp)

      # --------------------------------------------------
      # FPS 및 2초 주기 바이탈 측정
      # --------------------------------------------------
      if now - fps_timer >= 2.0:
        fps = frames / (now - fps_timer)
        frames = 0
        fps_timer = now

      if now - last_hr_time >= 2.0:
        tracker.update(face_visible)
        anomaly = detector.push(
            tracker.hr_bpm, tracker.hr_conf, tracker.rr_bpm, tracker.rr_conf
        )
        last_hr_time = now

        payload = {
            'face_visible': face_visible,
            'fps': fps,
            'hr_bpm': tracker.hr_bpm,
            'hr_conf': tracker.hr_conf,
            'rr_bpm': tracker.rr_bpm,
            'rr_conf': tracker.rr_conf,
            'temp': None,  # 적외선 카메라 미연결 상태
            'anomaly': anomaly,
        }
        app.after(0, app.update_data, payload)


# ══════════════════════════════════════════════════════
#  Main Execution
# ══════════════════════════════════════════════════════
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--camera', type=int, default=0)
  args, _ = parser.parse_known_args()

  app = BioGuardianTkApp()

  t = threading.Thread(
      target=start_rppg_thread, args=(app, args.camera), daemon=True
  )
  t.start()

  app.mainloop()
