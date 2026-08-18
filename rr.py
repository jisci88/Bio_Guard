#!/usr/bin/env python3
"""Respiratory rate from a NoIR camera via keyframe Lucas-Kanade optical flow.

v2 changes vs v1:
  - displacement is measured against a keyframe, not integrated frame-to-frame
    (v1's cumsum turned LK jitter into a random walk that swamped the low end
     of the respiration band)
  - the ROI follows the subject via the median displacement of tracked points
  - large body motion is detected per frame, marked invalid, and forces a
    re-anchor so the resulting step never enters the analysis window
  - forward-backward consistency check rejects mistracked points

Usage:
    python3 rr.py --selftest                  # DSP path only, no camera
    python3 rr.py --snapshot roi.png          # check ROI placement
    python3 rr.py --roi 130,270,380,160       # override the default band
    python3 rr.py --dump sig.npz              # save raw signals for analysis
    python3 rr.py --video clip.mp4
"""

import argparse
import signal as pysignal
import time

import cv2
import numpy as np
from scipy import ndimage, signal

PROC_W, PROC_H = 640, 480
FS = 15.0                      # resample rate (Hz)
WIN_SEC = 30.0                 # analysis window
HOP_SEC = 1.0                  # report interval
RR_BAND = (0.1, 0.5)           # Hz -> 6..30 breaths/min
GRID = (3, 2)                  # cols, rows of shoulder patches
NCELL = GRID[0] * GRID[1]
DEFAULT_ROI = (0.20, 0.55, 0.60, 0.35)   # fractions of the frame

KEYFRAME_SEC = 12.0            # scheduled re-anchor interval
MIN_TRACK_FRAC = 0.6           # re-anchor if fewer points survive
FB_ERR_PX = 1.0                # forward-backward consistency threshold
MOTION_PX = 1.0                # per-frame body shift that counts as motion
MOTION_RATE = MOTION_PX * 30.0 # same threshold as px/s; MOTION_PX was tuned at 30 fps
MUTE_SEC = 0.4                 # invalidate this long after a motion event
MIN_PTS = 30
PROM_REF = 0.12               # whitened-peak prominence treated as full quality
ACF_MIN = 0.10                # a candidate period must actually correlate there
SQI_MIN = 0.14                # estimates below this are not reported
HOLD_SEC = 20.0               # median over accepted estimates in this window

LK = dict(winSize=(21, 21), maxLevel=3,
          criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))


# ---------------------------------------------------------------- ROI

def default_roi(shape):
    h, w = shape
    fx, fy, fw, fh = DEFAULT_ROI
    return (int(fx * w), int(fy * h), int(fw * w), int(fh * h))


def bg_rois(shape):
    h, w = shape
    return [(0, 0, 80, 80), (w - 80, 0, 80, 80)]


def clip_roi(roi, shape):
    h, w = shape
    x, y, rw, rh = (int(round(v)) for v in roi)
    x, y = max(0, min(x, w - 40)), max(0, min(y, h - 30))
    return (x, y, min(w - x, rw), min(h - y, rh))


# ------------------------------------------------------------ tracking

class KeyframeTracker:
    """Shoulder displacement measured against a periodically reset keyframe."""

    def __init__(self, shape, roi):
        self.shape = shape
        self.roi0 = roi
        self.bg = bg_rois(shape)
        self.roi = roi
        self.roi_off = np.zeros(2)      # cumulative subject translation
        self.carry = np.zeros(NCELL)    # displacement accrued before this keyframe
        self.rel = np.zeros(NCELL)      # displacement relative to this keyframe
        self.body = np.zeros(2)         # subject translation since this keyframe
        self.prev_body = None
        self.prev_t = None
        self.step = 0.0                  # last inter-frame body shift (px)
        self.nbg = 0                     # background points used for compensation
        self.ref = None
        self.ref_pts = None
        self.labels = None
        self.n0 = 0
        self.frac = 0.0
        self.t_anchor = -1e9
        self.mute_until = -1e9
        self.force = False

    def needs_anchor(self, t):
        return (self.ref is None or self.force
                or t - self.t_anchor > KEYFRAME_SEC
                or self.frac < MIN_TRACK_FRAC)

    def anchor(self, gray, t):
        """Reset the keyframe. Returns False if the ROI has too little texture."""
        roi = clip_roi((self.roi0[0] + self.roi_off[0],
                        self.roi0[1] + self.roi_off[1],
                        self.roi0[2], self.roi0[3]), self.shape)
        sx, sy, sw, sh = roi

        p = cv2.goodFeaturesToTrack(gray[sy:sy + sh, sx:sx + sw], 200, 0.01, 6)
        if p is None or len(p) < MIN_PTS:
            return False
        p = p.reshape(-1, 2) + np.float32([sx, sy])
        col = np.clip(((p[:, 0] - sx) / sw * GRID[0]).astype(int), 0, GRID[0] - 1)
        row = np.clip(((p[:, 1] - sy) / sh * GRID[1]).astype(int), 0, GRID[1] - 1)

        pts, labels = [p], [row * GRID[0] + col]
        for bx, by, bw, bh in self.bg:
            q = cv2.goodFeaturesToTrack(gray[by:by + bh, bx:bx + bw], 40, 0.01, 6)
            if q is not None:
                pts.append(q.reshape(-1, 2) + np.float32([bx, by]))
                labels.append(np.full(len(q), -1))

        # commit only on success, so a failed anchor never double-counts carry
        self.carry = self.carry + self.rel
        self.roi_off = self.roi_off + self.body
        self.roi = roi
        self.rel = np.zeros(NCELL)
        self.body = np.zeros(2)
        self.prev_body = None
        self.prev_t = None
        self.ref = gray
        self.ref_pts = np.vstack(pts).reshape(-1, 1, 2).astype(np.float32)
        self.labels = np.concatenate(labels)
        self.n0 = int((self.labels >= 0).sum())
        self.frac = 1.0
        self.t_anchor = t
        self.force = False
        return True

    def track(self, gray, t):
        """Return (cell_displacement, valid) or None if tracking collapsed."""
        cur, st1, _ = cv2.calcOpticalFlowPyrLK(self.ref, gray,
                                               self.ref_pts, None, **LK)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.ref,
                                                cur, None, **LK)
        fb = np.linalg.norm(back - self.ref_pts, axis=2).ravel()
        good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < FB_ERR_PX)

        d = (cur - self.ref_pts)[:, 0, :]          # (N, 2) dx, dy

        bgm = good & (self.labels == -1)
        self.nbg = int(bgm.sum())
        if self.nbg >= 5:
            d = d - np.median(d[bgm], axis=0)      # camera motion compensation

        shoulder = good & (self.labels >= 0)
        self.frac = shoulder.sum() / max(self.n0, 1)
        if shoulder.sum() < 20:
            self.force = True
            return None

        body = np.median(d[shoulder], axis=0)
        for c in range(NCELL):
            m = good & (self.labels == c)
            if m.sum() >= 3:
                self.rel[c] = float(np.median(d[m, 1]))

        if self.prev_body is not None:
            dt = max(t - self.prev_t, 1e-3)
            self.step = float(np.linalg.norm(body - self.prev_body))
            if self.step / dt > MOTION_RATE:
                self.mute_until = t + MUTE_SEC
                self.force = True                  # absorb the step via carry
        valid = t >= self.mute_until
        self.prev_body = body
        self.prev_t = t
        self.body = body

        return self.carry + self.rel, valid


# ------------------------------------------------------------------ DSP

def _parabolic(f, p, k):
    if k <= 0 or k >= len(p) - 1:
        return f[k]
    a, b, c = p[k - 1], p[k], p[k + 1]
    d = a - 2 * b + c
    if d == 0:
        return f[k]
    return f[k] + 0.5 * (a - c) / d * (f[1] - f[0])


def estimate_rr(times, disp, valid):
    """times (N,), disp (N, NCELL) displacement, valid (N,) bool -> dict|None."""
    times = np.asarray(times, float)
    disp = np.asarray(disp, float)
    valid = np.asarray(valid, bool)
    if times[-1] - times[0] < WIN_SEC * 0.9:
        return None

    vfrac = float(valid.mean())
    if valid.sum() < 0.4 * len(valid):
        return dict(rr=np.nan, sqi=0.0, rr_spec=np.nan, rr_time=np.nan,
                    evr=0.0, prom=0.0, vfrac=vfrac, nkeep=0, ach=0.0)

    # interpolate across motion-invalidated samples onto a uniform grid
    grid = np.arange(times[0], times[-1], 1.0 / FS)
    tv, dv = times[valid], disp[valid]
    X = np.stack([np.interp(grid, tv, dv[:, c]) for c in range(NCELL)], axis=1)

    X = signal.detrend(X, axis=0)
    sos = signal.butter(3, [RR_BAND[0] / (FS / 2), RR_BAND[1] / (FS / 2)],
                        btype="band", output="sos")
    X = signal.sosfiltfilt(sos, X, axis=0)

    # Drop cells with no real in-band motion (static background inside the
    # ROI). Unit-normalising those would give pure noise the same weight as
    # a genuine shoulder cell in the SVD.
    sd = X.std(axis=0)
    keep = sd > max(1e-9, 0.25 * np.median(sd))
    if keep.sum() < 2:
        return None
    nkeep = int(keep.sum())
    X = X[:, keep] / sd[keep]

    # PC1 preserves sign -> no rectification, no frequency doubling
    U, S, _ = np.linalg.svd(X - X.mean(0), full_matrices=False)
    evr = float(S[0] ** 2 / np.sum(S ** 2))
    pc1 = U[:, 0] * S[0]

    n = len(pc1)
    f, pxx = signal.welch(pc1, fs=FS, nperseg=n, nfft=4 * n)
    band = (f >= RR_BAND[0]) & (f <= RR_BAND[1])
    fb, pb = f[band], pxx[band]

    # Whiten: divide out the 1/f baseline. Postural sway puts large power at
    # the low edge of the band; without this it outranks the real peak and the
    # estimate sticks near RR_BAND[0]*60.
    kern = max(9, (len(pb) // 4) * 2 + 1)
    base = ndimage.median_filter(pb, size=kern, mode="nearest")
    base = np.maximum(base, pb.max() * 1e-6)
    wh = pb / base

    kk = int(np.argmax(wh))
    f_peak = _parabolic(fb, wh, kk)
    prom = float(wh[kk] / wh.sum())

    # Octave correction. Breathing is not sinusoidal, so at low rates its 2nd
    # and 3rd harmonics fall inside the 6-30 brpm band and can outrank the
    # fundamental (measured at a true 10 brpm: peaks at 10, 20 and 30). The
    # autocorrelation disambiguates: it has local maxima at T, 2T, 3T but not
    # at T/2 or T/3. So requiring a local ACF maximum rejects harmonics, and
    # then taking the fastest survivor rejects subharmonics.
    f_sel, lags_ok = f_peak, []
    x = pc1 - pc1.mean()
    ac = np.correlate(x, x, mode="full")[n - 1:]
    if ac[0] > 0:
        ac = ac / ac[0]
        apk, _ = signal.find_peaks(ac)
        if len(apk):
            for r in (1 / 3, 1 / 2, 1, 2, 3):
                fc = f_peak * r
                if not (RR_BAND[0] <= fc <= RR_BAND[1]):
                    continue
                lag = FS / fc
                j = int(np.argmin(np.abs(apk - lag)))
                # find_peaks returns every local maximum, including ones at a
                # negative ACF. Those are the opposite of periodic, and letting
                # them through is what lets max() below lock onto a harmonic.
                if abs(apk[j] - lag) <= 0.12 * lag and ac[apk[j]] > ACF_MIN:
                    lags_ok.append(fc)
            if lags_ok:
                f_sel = max(lags_ok)
    rr_spec = f_sel * 60.0

    # Independent time-domain estimate, kept for diagnostics only. It is NOT
    # used in SQI: at low rates find_peaks counts harmonic shoulders as extra
    # breaths (measured median 15.5 against a true 10 brpm, a factor of 1.55),
    # so an agreement term built on it rewarded exactly the harmonic errors.
    pk, _ = signal.find_peaks(pc1, distance=int(FS / RR_BAND[1]))
    rr_time = float(60.0 / (np.median(np.diff(pk)) / FS)) if len(pk) >= 3 else np.nan

    # Periodicity confidence: the normalised autocorrelation at the selected
    # period. High only if the signal really does repeat at that rate, and
    # unlike the peak-counting check it carries no harmonic bias.
    ach = float(np.interp(FS / f_sel, np.arange(len(ac)), ac)) if ac[0] > 0 else 0.0

    sqi = float(np.clip(evr, 0, 1) * np.clip(prom / PROM_REF, 0, 1)
                * np.clip(ach, 0, 1) * vfrac)

    return dict(rr=rr_spec, sqi=sqi, rr_spec=rr_spec, rr_time=rr_time,
                evr=evr, prom=prom, vfrac=vfrac, nkeep=nkeep, ach=ach)


class Reporter:
    """Suppresses low-quality estimates and reports a short-window median.

    Tuned on two recordings (71 s and 90 s) with a ground truth of 18 brpm.
    Reporting every estimate gave MAE 3.99 / 53% within 2 brpm on the first,
    because early estimates are contaminated by the subject settling. With
    SQI_MIN=0.20 and a HOLD_SEC median: MAE 0.62 / 100% and 1.10 / 100%.
    Retuned again on a 10 brpm session: SQI_MIN=0.14 with the autocorrelation
    based quality term gives MAE 0.18 / 100% at 10 brpm and 1.16 / 100% at
    18 brpm, at the cost of showing a value about half the time.
    """

    def __init__(self):
        self.hist = []

    def update(self, t, rr, sqi):
        if np.isfinite(rr) and sqi >= SQI_MIN:
            self.hist.append((t, rr))
        self.hist = [h for h in self.hist if t - h[0] <= HOLD_SEC]
        if not self.hist:
            return None
        return float(np.median([h[1] for h in self.hist])), len(self.hist)


# ----------------------------------------------------------------- I/O

def frames_picamera():
    from picamera2 import Picamera2
    cam = Picamera2()
    cam.configure(cam.create_video_configuration(
        main={"size": (PROC_W, PROC_H), "format": "YUV420"},
        controls={"FrameRate": 30}))
    cam.start()
    try:
        while True:
            buf = cam.capture_array()
            yield time.monotonic(), np.ascontiguousarray(buf[:PROC_H, :PROC_W])
    finally:
        cam.stop()


def frames_video(path):
    cap = cv2.VideoCapture(path)
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        yield t, cv2.resize(gray, (PROC_W, PROC_H))
    cap.release()


def snapshot(frames, path, roi=None):
    t, gray = next(frames)
    shoulder = clip_roi(roi or default_roi(gray.shape), gray.shape)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    sx, sy, sw, sh = shoulder
    cv2.rectangle(vis, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
    for c in range(1, GRID[0]):
        x = sx + c * sw // GRID[0]
        cv2.line(vis, (x, sy), (x, sy + sh), (0, 160, 0), 1)
    for r in range(1, GRID[1]):
        y = sy + r * sh // GRID[1]
        cv2.line(vis, (sx, y), (sx + sw, y), (0, 160, 0), 1)
    for bx, by, bw, bh in bg_rois(gray.shape):
        cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (255, 0, 0), 2)

    p = cv2.goodFeaturesToTrack(gray[sy:sy + sh, sx:sx + sw], 200, 0.01, 6)
    n = 0 if p is None else len(p)
    if p is not None:
        for x, y in p.reshape(-1, 2):
            cv2.circle(vis, (int(x) + sx, int(y) + sy), 2, (0, 0, 255), -1)

    cv2.imwrite(path, vis)
    print(f"saved {path}  shoulder={shoulder}  features={n}")
    if n < MIN_PTS:
        print(f"WARNING: fewer than {MIN_PTS} features. Move the ROI onto "
              "textured clothing, or add contrast.")


# ----------------------------------------------------------------- main

def _stop_on_sigint():
    """First Ctrl+C ends the loop cleanly; a second one aborts immediately."""
    state = {"stop": False}

    def handler(signum, frame):
        state["stop"] = True
        pysignal.signal(pysignal.SIGINT, pysignal.default_int_handler)
        print("\nstopping after this frame... (Ctrl+C again to abort)")

    pysignal.signal(pysignal.SIGINT, handler)
    return state


def run(frames, roi=None, dump=None):
    tracker = None
    reporter = Reporter()
    stop = _stop_on_sigint()
    buf_t, buf_d, buf_v = [], [], []
    log_t, log_d, log_v = [], [], []
    last_report = None

    for t, gray in frames:
        if stop["stop"]:
            break
        if tracker is None:
            tracker = KeyframeTracker(
                gray.shape, clip_roi(roi or default_roi(gray.shape), gray.shape))
            print(f"shoulder ROI = {tracker.roi}   bg = {tracker.bg}")

        if tracker.needs_anchor(t):
            if not tracker.anchor(gray, t):
                continue

        out = tracker.track(gray, t)
        if out is None:
            continue
        d, v = out

        buf_t.append(t)
        buf_d.append(d.copy())
        buf_v.append(v)
        if dump:
            log_t.append(t)
            log_d.append(d.copy())
            log_v.append(v)
        while buf_t[-1] - buf_t[0] > WIN_SEC * 1.2:
            buf_t.pop(0)
            buf_d.pop(0)
            buf_v.pop(0)

        if last_report is None:
            last_report = t
        if t - last_report < HOP_SEC:
            continue
        last_report = t

        r = estimate_rr(buf_t, np.array(buf_d), np.array(buf_v))
        if r is None:
            print(f"t={t:6.1f}s  filling window "
                  f"({buf_t[-1] - buf_t[0]:.0f}/{WIN_SEC:.0f}s)", flush=True)
        elif not np.isfinite(r["rr"]):
            print(f"t={t:6.1f}s  motion  (valid={r['vfrac']:.0%})", flush=True)
        else:
            rep = reporter.update(t, r["rr"], r["sqi"])
            shown = f"RR={rep[0]:5.1f} brpm (n={rep[1]})" if rep else "RR=  --  acquiring"
            print(f"t={t:6.1f}s  {shown}  SQI={r['sqi']:.2f}  "
                  f"[raw={r['rr']:.1f} spec={r['rr_spec']:.1f} "
                  f"time={r['rr_time']:.1f} evr={r['evr']:.2f} acf={r['ach']:.2f} "
                  f"prom={r['prom']:.2f} valid={r['vfrac']:.0%} "
                  f"cells={r['nkeep']}/{NCELL} pts={tracker.frac:.0%}]",
                  flush=True)

    if dump and log_t:
        np.savez(dump, t=np.array(log_t), disp=np.array(log_d),
                 valid=np.array(log_v))
        print(f"saved {dump}  ({len(log_t)} frames)")


def selftest():
    fs_cam, dur, rr_true = 30.0, 32.0, 15.0
    t = np.arange(0, dur, 1 / fs_cam)
    gains = np.array([1.0, 0.8, 0.6, -0.5, 0.3, 0.1])
    rng = np.random.default_rng(0)
    disp = (np.sin(2 * np.pi * (rr_true / 60) * t)[:, None] * gains[None, :]
            + rng.normal(0, 0.15, (len(t), NCELL)))
    valid = np.ones(len(t), bool)

    r = estimate_rr(t, disp, valid)
    assert r is not None, "estimator returned None"
    print(f"selftest: true={rr_true:.1f}  est={r['rr']:.2f}  "
          f"SQI={r['sqi']:.2f}  evr={r['evr']:.2f}")
    assert abs(r["rr"] - rr_true) < 0.5, f"error too large: {r['rr']}"

    valid[int(10 * fs_cam):int(14 * fs_cam)] = False    # 4 s of motion
    r2 = estimate_rr(t, disp, valid)
    print(f"with 4s motion gap: est={r2['rr']:.2f}  SQI={r2['sqi']:.2f}  "
          f"valid={r2['vfrac']:.0%}")
    assert abs(r2["rr"] - rr_true) < 1.0, f"gap handling failed: {r2['rr']}"
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="run on a video file instead of the camera")
    ap.add_argument("--roi", help="shoulder ROI in pixels: x,y,w,h")
    ap.add_argument("--snapshot", help="save one annotated frame and exit")
    ap.add_argument("--dump", help="save raw signals to an .npz")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    roi = tuple(int(v) for v in args.roi.split(",")) if args.roi else None
    if roi is not None and len(roi) != 4:
        ap.error("--roi needs exactly 4 values: x,y,w,h")

    if args.selftest:
        selftest()
        raise SystemExit

    frames = frames_video(args.video) if args.video else frames_picamera()
    if args.snapshot:
        snapshot(frames, args.snapshot, roi)
    else:
        run(frames, roi, args.dump)