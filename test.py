import argparse
import asyncio
import json
import os
import threading
import time

import cv2
import numpy as np
import rppg
import websockets

from vital_anomaly import VitalAnomalyDetector


# ══════════════════════════════════════════════════════
#  WebSocket 스트리밍 서버 (비동기 스레드)
# ══════════════════════════════════════════════════════
class VitalWebSocketServer:

  def __init__(self, host="0.0.0.0", port=8765):
    self.host = host
    self.port = port
    self.clients = set()
    self.latest_data = {}
    self.loop = None
    self.thread = None

  def start(self):
    self.thread = threading.Thread(target=self._run_server, daemon=True)
    self.thread.start()

  def _run_server(self):
    self.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self.loop)

    async def handler(websocket):
      self.clients.add(websocket)
      try:
        await websocket.wait_closed()
      finally:
        self.clients.remove(websocket)

    async def main():
      async with websockets.serve(handler, self.host, self.port):
        await asyncio.Future()  # run forever

    self.loop.run_until_complete(main())

  def broadcast(self, data):
    self.latest_data = data
    if self.loop and self.clients:
      message = json.dumps(data)
      asyncio.run_coroutine_threadsafe(
          self._send_to_all(message), self.loop
      )

  async def _send_to_all(self, message):
    if self.clients:
      await asyncio.gather(
          *[client.send(message) for client in self.clients],
          return_exceptions=True,
      )


def should_show_window(headless, environ=None):
  env = os.environ if environ is None else environ
  return not headless and bool(env.get("DISPLAY"))


# ══════════════════════════════════════════════════════
#  open-rppg 바이탈 트래커
# ══════════════════════════════════════════════════════
class OpenRppgVitalTracker:

  HR_MIN, HR_MAX = 40.0, 200.0
  RR_MIN, RR_MAX = 4.0, 40.0

  def __init__(self, model, hr_window_sec=10, rr_window_sec=60):
    self.model = model
    self.hr_window_sec = hr_window_sec
    self.rr_window_sec = rr_window_sec

    self.hr_bpm = 0.0
    self.hr_conf = 0.0
    self.rr_bpm = 0.0
    self.rr_conf = 0.0

  @staticmethod
  def _finite_float(value, default=0.0):
    try:
      value = float(value)
    except (TypeError, ValueError):
      return default
    return value if np.isfinite(value) else default

  def invalidate(self):
    self.hr_conf = 0.0
    self.rr_conf = 0.0

  def update_hr(self, face_visible):
    if not face_visible:
      self.hr_conf = 0.0
      return

    result = self.model.hr(start=-self.hr_window_sec, return_hrv=False)
    hr = self._finite_float((result or {}).get("hr"))
    sqi = self._finite_float((result or {}).get("SQI"))

    if self.HR_MIN <= hr <= self.HR_MAX:
      self.hr_bpm = hr
      self.hr_conf = float(np.clip(sqi, 0.0, 1.0))
    else:
      self.hr_conf = 0.0

  def update_rr(self, face_visible):
    if not face_visible:
      self.rr_conf = 0.0
      return

    result = self.model.hr(start=-self.rr_window_sec)
    hrv = (result or {}).get("hrv") or {}
    breathing_hz = self._finite_float(hrv.get("breathingrate"))
    rr = breathing_hz * 60.0
    sqi = self._finite_float((result or {}).get("SQI"))

    if self.RR_MIN <= rr <= self.RR_MAX:
      self.rr_bpm = rr
      self.rr_conf = float(np.clip(sqi, 0.0, 1.0))
    else:
      self.rr_conf = 0.0


# ══════════════════════════════════════════════════════
#  메인 실행 루프 (웹소켓 연동)
# ══════════════════════════════════════════════════════
def run(
    camera_id=0,
    calib_sec=180.0,
    min_conf=0.30,
    out_pct=1.0,
    rr_sigma=4.0,
    headless=False,
):
  print("[INFO] open-rppg 모델 및 웹소켓 서버 로딩 중...")
  model = rppg.Model("ME-flow.rlap")

  # 웹소켓 서버 시작 (포트 8765)
  ws_server = VitalWebSocketServer(port=8765)
  ws_server.start()
  print("[INFO] 웹소켓 서버 가동 완료 (ws://0.0.0.0:8765)")

  show_window = should_show_window(headless)
  vitals = OpenRppgVitalTracker(model)
  detector = VitalAnomalyDetector(
      calib_sec=calib_sec, min_conf=min_conf, out_pct=out_pct, rr_sigma=rr_sigma
  )

  anomaly = None
  hr_update_interval = 2.0
  rr_update_interval = 10.0

  last_hr_update = time.time()
  last_rr_update = time.time()

  fps_timer = time.time()
  frame_count = 0
  measured_fps = 0.0

  try:
    with model.video_capture(camera_id):
      for frame_rgb, box in model.preview:
        now = time.time()
        face_visible = box is not None
        frame_count += 1

        if now - fps_timer >= 2.0:
          measured_fps = frame_count / (now - fps_timer)
          frame_count = 0
          fps_timer = now

        if not face_visible:
          vitals.invalidate()

        # 호흡수 갱신 (10초 주기)
        if now - last_rr_update >= rr_update_interval:
          vitals.update_rr(face_visible)
          last_rr_update = now

        # 심박수 및 이상탐지 갱신 (2초 주기)
        if now - last_hr_update >= hr_update_interval:
          vitals.update_hr(face_visible)
          anomaly = detector.push(
              vitals.hr_bpm,
              vitals.hr_conf,
              vitals.rr_bpm,
              vitals.rr_conf,
              now=now,
          )
          last_hr_update = now

        # ----------------------------------------------------
        # 프론트엔드 웹소켓 전송 패킷 조립
        # ----------------------------------------------------
        ws_payload = {
            "timestamp": now,
            "face_visible": face_visible,
            "fps": measured_fps,
            "hr_bpm": vitals.hr_bpm,
            "hr_conf": vitals.hr_conf,
            "rr_bpm": vitals.rr_bpm,
            "rr_conf": vitals.rr_conf,
            "temp": 36.8,  # 열화상 센서 연동 시 실제 온도값으로 대체
            "anomaly": anomaly,
        }
        ws_server.broadcast(ws_payload)

        # GUI 모드 시 OpenCV 창 출력
        if show_window:
          frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
          if box is not None:
            (y1, y2), (x1, x2) = box[0], box[1]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 220, 80), 2)
          cv2.imshow("Bio-Guardian Live Feed", frame)
          if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
            break

  except KeyboardInterrupt:
    print("\n[INFO] 종료 중...")
  finally:
    if show_window:
      cv2.destroyAllWindows()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--camera", type=int, default=0)
  parser.add_argument("--calib", type=float, default=180.0)
  parser.add_argument("--min-conf", type=float, default=0.30)
  parser.add_argument("--out-pct", type=float, default=1.0)
  parser.add_argument("--rr-sigma", type=float, default=4.0)
  parser.add_argument("--headless", action="store_true")
  args, _ = parser.parse_known_args()

  run(
      camera_id=args.camera,
      calib_sec=args.calib,
      min_conf=args.min_conf,
      out_pct=args.out_pct,
      rr_sigma=args.rr_sigma,
      headless=args.headless,
  )
