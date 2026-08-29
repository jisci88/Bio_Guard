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
from collections import deque
from dataclasses import dataclass, replace

import cv2
import numpy as np
from scipy import ndimage, signal

import config

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


def shoulder_roi_from_face(box, shape):
    """
    C-1: 검출된 얼굴 박스에서 어깨 밴드 ROI 를 유도한다.

    DEFAULT_ROI 는 프레임의 고정 비율이라 특정 촬영거리·앉은 높이에서만
    어깨에 맞았다. 그런데 이 시스템은 팬/틸트로 얼굴을 화면 중앙에 두므로
    얼굴의 위치와 크기를 이미 알고 있고, 어깨는 항상 얼굴 아래 일정 배율에
    있다. 그걸 그대로 쓴다.

    box 는 ((y1, y2), (x1, x2)) 형식(open-rppg preview 와 동일).
    ROI 가 프레임 밖으로 밀려 너무 작아지면 None 을 돌려 고정 ROI 로 폴백한다.
    """
    if box is None:
        return None
    try:
        (y1, y2), (x1, x2) = box
        y1, y2, x1, x2 = float(y1), float(y2), float(x1), float(x2)
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(v) for v in (y1, y2, x1, x2)):
        return None

    face_w, face_h = x2 - x1, y2 - y1
    if face_w <= 1.0 or face_h <= 1.0:
        return None

    h, w = shape
    cx = (x1 + x2) * 0.5
    roi_w = face_w * config.RR_ROI_WIDTH_FACES
    roi_x = cx - roi_w * 0.5
    roi_y = y1 + face_h * config.RR_ROI_TOP_FACES
    roi_h = face_h * (config.RR_ROI_BOTTOM_FACES - config.RR_ROI_TOP_FACES)

    # 프레임 경계로 잘라낸 뒤에도 쓸 만한 크기인지 확인한다.
    roi_x = max(0.0, min(roi_x, float(w) - 1.0))
    roi_y = max(0.0, min(roi_y, float(h) - 1.0))
    roi_w = min(roi_w, float(w) - roi_x)
    roi_h = min(roi_h, float(h) - roi_y)
    if roi_w < config.RR_ROI_MIN_PX[0] or roi_h < config.RR_ROI_MIN_PX[1]:
        return None

    return clip_roi((roi_x, roi_y, roi_w, roi_h), shape)


def _box_rect(box):
    """Convert the preview-style ``((y1, y2), (x1, x2))`` box to xywh."""
    if box is None:
        return None
    try:
        (y1, y2), (x1, x2) = box
        x1, y1, x2, y2 = (float(value) for value in (x1, y1, x2, y2))
    except (TypeError, ValueError):
        return None
    if not all(np.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    width, height = x2 - x1, y2 - y1
    return (x1, y1, width, height) if width > 1.0 and height > 1.0 else None


def _visible_roi(rect, shape):
    """Return a usable clipped ROI, rejecting materially clipped proposals."""
    x, y, width, height = rect
    if width <= 0.0 or height <= 0.0:
        return None
    frame_h, frame_w = shape
    left, top = max(0.0, x), max(0.0, y)
    right, bottom = min(float(frame_w), x + width), min(float(frame_h), y + height)
    clipped_width, clipped_height = right - left, bottom - top
    if clipped_width <= 0.0 or clipped_height <= 0.0:
        return None
    visible = clipped_width * clipped_height / (width * height)
    if (visible < config.RR_ROI_MIN_VISIBLE
            or clipped_width < config.RR_ROI_MIN_PX[0]
            or clipped_height < config.RR_ROI_MIN_PX[1]):
        return None
    return (int(round(left)), int(round(top)),
            int(round(clipped_width)), int(round(clipped_height)))


def respiration_rois(face_box, person_box, frame_shape):
    """Build respiration regions from current face/person geometry.

    No fixed-frame fallback is used once detector geometry is available: a
    clipped or undersized proposal is omitted until a valid region returns.
    """
    rois = {}
    face = _box_rect(face_box)
    person = _box_rect(person_box)

    if person is not None:
        px, py, pw, ph = person
        torso = _visible_roi((
            px + pw * config.RR_TORSO_SIDE_INSET,
            py + ph * config.RR_TORSO_TOP,
            pw * (1.0 - 2.0 * config.RR_TORSO_SIDE_INSET),
            ph * (config.RR_TORSO_BOTTOM - config.RR_TORSO_TOP),
        ), frame_shape)
        if torso is not None:
            rois["TORSO"] = torso

        if face is not None:
            fx, fy, fw, fh = face
            shoulder = _visible_roi((
                (fx + fx + fw) * 0.5 - fw * config.RR_ROI_WIDTH_FACES * 0.5,
                fy + fh * config.RR_ROI_TOP_FACES,
                fw * config.RR_ROI_WIDTH_FACES,
                fh * (config.RR_ROI_BOTTOM_FACES - config.RR_ROI_TOP_FACES),
            ), frame_shape)
            if shoulder is not None:
                rois["SHOULDER"] = shoulder

    if face is not None:
        fx, fy, fw, fh = face
        # Keep the face-motion proposal large enough to meet the common ROI
        # texture floor.  A detector box may itself be exactly the floor wide.
        inset_x = min(fw * config.RR_FACE_INSET,
                      max(0.0, (fw - config.RR_ROI_MIN_PX[0]) * 0.5))
        inset_y = min(fh * config.RR_FACE_INSET,
                      max(0.0, (fh - config.RR_ROI_MIN_PX[1]) * 0.5))
        face_motion = _visible_roi((
            fx + inset_x,
            fy + inset_y,
            fw - 2.0 * inset_x,
            fh - 2.0 * inset_y,
        ), frame_shape)
        if face_motion is not None:
            rois["FACE_MOTION"] = face_motion
    return rois


# ------------------------------------------------------------ tracking

class KeyframeTracker:
    """Shoulder displacement measured against a periodically reset keyframe."""

    def __init__(self, shape, roi, motion_rate=MOTION_RATE, mute_sec=MUTE_SEC):
        self.motion_rate = motion_rate
        self.mute_sec = mute_sec
        self.mutes = 0                  # how many times the gate has fired
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
        self.target_roi = None          # C-1: 얼굴에서 유도한 절대 ROI
        self.roi_changes = 0            # 진단용: ROI 재설정 횟수

    def set_face_roi(self, roi, t):
        """
        C-1: 얼굴 박스에서 유도한 어깨 ROI 를 적용한다.

        매 프레임 호출해도 되지만 실제로 갱신되는 것은 중심이
        RR_ROI_MOVE_FRAC 이상 움직였거나 크기가 그만큼 변했을 때뿐이다.
        갱신될 때는 키프레임을 다시 잡아야 하므로 force 를 세우고, 새 셀이
        아직 다른 신체 지점을 재는 과도구간을 mute_sec 만큼 무효 처리한다.

        anchor() 가 carry += rel 로 연속성을 유지하므로 ROI 가 바뀌어도
        누적 변위 신호에 계단은 생기지 않는다.
        """
        if roi is None:
            return False
        x, y, w, h = (float(v) for v in roi)
        if w < config.RR_ROI_MIN_PX[0] or h < config.RR_ROI_MIN_PX[1]:
            return False

        if self.target_roi is not None:
            ox, oy, ow, oh = self.target_roi
            moved = max(
                abs((x + w * 0.5) - (ox + ow * 0.5)),
                abs((y + h * 0.5) - (oy + oh * 0.5)),
            )
            resized = abs(w - ow) + abs(h - oh)
            threshold = ow * config.RR_ROI_MOVE_FRAC
            if moved < threshold and resized < threshold:
                return False

        self.target_roi = (x, y, w, h)
        self.roi_changes += 1
        self.force = True
        self.mute_until = max(self.mute_until, t + self.mute_sec)
        return True

    def needs_anchor(self, t):
        return (self.ref is None or self.force
                or t - self.t_anchor > KEYFRAME_SEC
                or self.frac < MIN_TRACK_FRAC)

    def restart(self):
        """Discard a lost face's keyframe while retaining ROI configuration."""
        self.roi_off = np.zeros(2)
        self.carry = np.zeros(NCELL)
        self.rel = np.zeros(NCELL)
        self.body = np.zeros(2)
        self.prev_body = None
        self.prev_t = None
        self.step = 0.0
        self.nbg = 0
        self.ref = None
        self.ref_pts = None
        self.labels = None
        self.n0 = 0
        self.frac = 0.0
        self.t_anchor = -1e9
        self.mute_until = -1e9
        self.force = True

    def anchor(self, gray, t):
        """Reset the keyframe. Returns False if the ROI has too little texture."""
        # C-1: 얼굴에서 유도한 ROI 가 있으면 그것이 절대 기준이다. 없으면
        # 기존처럼 고정 ROI + 누적 피사체 이동으로 따라간다.
        use_face_roi = self.target_roi is not None
        if use_face_roi:
            roi = clip_roi(self.target_roi, self.shape)
        else:
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
        if use_face_roi:
            # 얼굴 박스가 절대 기준이므로 누적 피사체 오프셋은 의미가 없다.
            self.roi_off = np.zeros(2)
        else:
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
            if self.step / dt > self.motion_rate:
                self.mute_until = t + self.mute_sec
                self.mutes += 1
                self.force = True                  # absorb the step via carry
        valid = t >= self.mute_until
        self.prev_body = body
        self.prev_t = t
        self.body = body

        return self.carry + self.rel, valid


class MultiRegionKeyframeTracker:
    """Track labeled respiration regions with one shared LK point cloud."""

    def __init__(self, shape, regions, motion_rate=MOTION_RATE, mute_sec=MUTE_SEC):
        self.shape = tuple(shape)
        self.motion_rate = float(motion_rate)
        self.mute_sec = float(mute_sec)
        self.regions = dict(regions)
        self.ref = self.ref_pts = None
        self.point_regions = self.point_cells = None
        self.carry = {name: np.zeros(NCELL) for name in self.regions}
        self.rel = {name: np.zeros(NCELL) for name in self.regions}
        self.prev_body = self.prev_t = None
        self.t_anchor = self.mute_until = -1e9
        self.force = True
        self.mutes = 0
        self.step = 0.0
        self.nbg = 0
        self.region_counts = {name: 0 for name in self.regions}

    def set_regions(self, regions, now):
        """Apply changed geometry and require a muted re-anchor when needed."""
        regions = dict(regions)
        changed = set(regions) != set(self.regions)
        if not changed:
            for name, (x, y, width, height) in regions.items():
                old_x, old_y, old_width, old_height = self.regions[name]
                center_shift = max(
                    abs((x + width * 0.5) - (old_x + old_width * 0.5)),
                    abs((y + height * 0.5) - (old_y + old_height * 0.5)),
                )
                size_shift = abs(width - old_width) + abs(height - old_height)
                scale = max(float(old_width), float(old_height), 1.0)
                if max(center_shift, size_shift) >= scale * config.RR_ROI_MOVE_FRAC:
                    changed = True
                    break
        if changed:
            old_carry, old_rel = self.carry, self.rel
            self.regions = regions
            self.carry = {
                name: old_carry.get(name, np.zeros(NCELL)).copy()
                for name in self.regions
            }
            self.rel = {
                name: old_rel.get(name, np.zeros(NCELL)).copy()
                for name in self.regions
            }
            self.region_counts = {name: 0 for name in self.regions}
            self.force = True
            self.mute_until = max(self.mute_until, now + self.mute_sec)
        return changed

    def needs_anchor(self, now):
        return self.ref is None or self.force or now - self.t_anchor > KEYFRAME_SEC

    def _seed_labeled_points(self, gray):
        points, region_labels, cell_labels = [], [], []
        for name, (x, y, width, height) in self.regions.items():
            roi = _visible_roi((x, y, width, height), self.shape)
            if roi is None:
                continue
            rx, ry, rw, rh = roi
            seeded = cv2.goodFeaturesToTrack(gray[ry:ry + rh, rx:rx + rw],
                                              120, 0.01, 6)
            if seeded is None:
                continue
            seeded = seeded.reshape(-1, 2)[:120]
            if not len(seeded):
                continue
            seeded += np.float32([rx, ry])
            col = np.clip(((seeded[:, 0] - rx) / rw * GRID[0]).astype(int),
                          0, GRID[0] - 1)
            row = np.clip(((seeded[:, 1] - ry) / rh * GRID[1]).astype(int),
                          0, GRID[1] - 1)
            points.append(seeded)
            region_labels.extend([name] * len(seeded))
            cell_labels.extend((row * GRID[0] + col).tolist())

        for bx, by, bw, bh in bg_rois(self.shape):
            seeded = cv2.goodFeaturesToTrack(gray[by:by + bh, bx:bx + bw],
                                              40, 0.01, 6)
            if seeded is None:
                continue
            seeded = seeded.reshape(-1, 2)[:40] + np.float32([bx, by])
            points.append(seeded)
            region_labels.extend(["BACKGROUND"] * len(seeded))
            cell_labels.extend([-1] * len(seeded))

        if not points:
            return None, None, None
        region_counts = {
            name: sum(label == name for label in region_labels)
            for name in self.regions
        }
        if not any(count >= config.RR_REGION_MIN_PTS
                   for count in region_counts.values()):
            return None, None, None
        return (np.vstack(points).reshape(-1, 1, 2).astype(np.float32),
                region_labels, cell_labels)

    def anchor(self, gray, now):
        """Seed all active regions and preserve preceding keyframe offsets."""
        points, region_labels, cell_labels = self._seed_labeled_points(gray)
        if points is None:
            return False
        # Only a successful anchor transfers relative displacement into carry.
        for name in self.regions:
            self.carry[name] = self.carry[name] + self.rel[name]
            self.rel[name] = np.zeros(NCELL)
        self.ref = gray
        self.ref_pts = points
        self.point_regions = np.asarray(region_labels, dtype=object)
        self.point_cells = np.asarray(cell_labels, dtype=int)
        self.region_counts = {
            name: int((self.point_regions == name).sum()) for name in self.regions
        }
        self.prev_body = self.prev_t = None
        self.t_anchor, self.force = now, False
        return True

    def _summarize(self, displacement, good):
        result = {}
        for name in self.regions:
            region = good & (self.point_regions == name)
            if int(region.sum()) < config.RR_REGION_MIN_PTS:
                continue
            values = self.rel[name].copy()
            for cell in range(NCELL):
                chosen = region & (self.point_cells == cell)
                if int(chosen.sum()) >= 3:
                    values[cell] = float(np.median(displacement[chosen, 1]))
            self.rel[name] = values
            result[name] = self.carry[name] + values
        return result

    def track(self, gray, now):
        """Return surviving-region displacement arrays and the motion validity."""
        if self.ref is None or self.ref_pts is None or not len(self.ref_pts):
            return None
        cur, st1, _ = cv2.calcOpticalFlowPyrLK(self.ref, gray,
                                                self.ref_pts, None, **LK)
        if cur is None or st1 is None:
            self.force = True
            return None
        back, st2, _ = cv2.calcOpticalFlowPyrLK(gray, self.ref,
                                                 cur, None, **LK)
        if back is None or st2 is None:
            self.force = True
            return None
        fb = np.linalg.norm(back - self.ref_pts, axis=2).ravel()
        good = (st1.ravel() == 1) & (st2.ravel() == 1) & (fb < FB_ERR_PX)
        displacement = (cur - self.ref_pts)[:, 0, :]
        background = good & (self.point_regions == "BACKGROUND")
        self.nbg = int(background.sum())
        if self.nbg >= 5:
            displacement = displacement - np.median(displacement[background], axis=0)

        values = self._summarize(displacement, good)
        if not values:
            self.force = True
            return None
        active = good & np.isin(self.point_regions, tuple(values))
        body = np.median(displacement[active], axis=0)
        if self.prev_body is not None:
            dt = max(now - self.prev_t, 1e-3)
            self.step = float(np.linalg.norm(body - self.prev_body))
            if self.step / dt > self.motion_rate:
                self.mute_until = now + self.mute_sec
                self.mutes += 1
                self.force = True
        self.prev_body, self.prev_t = body, now
        return values, now >= self.mute_until


# ------------------------------------------------------------------ DSP


@dataclass(frozen=True)
class RespirationCandidate:
    """One independently-scored respiration estimate.

    The waveform is copied and made read-only so that an accepted candidate is
    a stable observation rather than a view into a tracker buffer.
    """

    rate_bpm: float
    source: str
    confidence: float
    spectral_snr: float
    periodicity: float
    concentration: float
    valid_fraction: float
    coverage: float
    wave: np.ndarray

    def __post_init__(self):
        numeric = (
            "rate_bpm", "confidence", "spectral_snr", "periodicity",
            "concentration", "valid_fraction", "coverage",
        )
        for name in numeric:
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be a non-empty string")
        wave = np.asarray(self.wave, dtype=float).copy()
        if wave.ndim != 1 or not len(wave) or not np.all(np.isfinite(wave)):
            raise ValueError("wave must be a finite one-dimensional signal")
        wave.setflags(write=False)
        object.__setattr__(self, "wave", wave)


SOURCE_PRIORITY = {
    "TORSO": 0,
    "SHOULDER": 1,
    "FACE_MOTION": 2,
    "BVP_AM": 3,
    "BVP_FM": 4,
}

FLOW_SOURCES = frozenset(("TORSO", "SHOULDER", "FACE_MOTION"))


def _evidence_family(source):
    """Group correlated measurement paths before labeling fusion evidence."""
    return "OPTICAL" if source in FLOW_SOURCES else "BVP"


@dataclass(frozen=True)
class RespirationResult:
    """The selected respiratory estimate and its temporal lock state."""

    rate_bpm: float
    confidence: float
    state: str
    source: str
    fresh: bool
    learn_valid: bool
    reason: str
    wave: np.ndarray | None
    candidates: tuple[RespirationCandidate, ...]

    def __post_init__(self):
        object.__setattr__(self, "rate_bpm", float(self.rate_bpm))
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "candidates", tuple(self.candidates))
        if self.wave is not None:
            wave = np.asarray(self.wave, dtype=float).copy()
            wave.setflags(write=False)
            object.__setattr__(self, "wave", wave)


def _weighted_median(values, weights):
    order = np.argsort(values)
    sorted_values = np.asarray(values, dtype=float)[order]
    sorted_weights = np.asarray(weights, dtype=float)[order]
    cutoff = sorted_weights.sum() * 0.5
    return float(sorted_values[np.searchsorted(np.cumsum(sorted_weights), cutoff)])


class RespirationFusion:
    """Select agreeing estimates, then require a stable temporal lock."""

    def __init__(self, *, lock_updates=None, agreement_bpm=None,
                 stability_bpm=None, hold_sec=None, learn_conf=None):
        self.lock_updates = (config.RR_LOCK_UPDATES if lock_updates is None
                             else int(lock_updates))
        self.agreement_bpm = (config.RR_AGREEMENT_BPM if agreement_bpm is None
                              else float(agreement_bpm))
        self.stability_bpm = (config.RR_LOCK_STABILITY_BPM if stability_bpm is None
                              else float(stability_bpm))
        self.hold_sec = config.RR_HOLD_SEC if hold_sec is None else float(hold_sec)
        self.learn_conf = config.RR_LEARN_CONF if learn_conf is None else float(learn_conf)
        self.history = deque(maxlen=self.lock_updates)
        self.last_locked = None

    @staticmethod
    def _family_representatives(candidates):
        """Aggregate correlated paths before their rates compete in fusion."""
        families = {}
        for candidate in candidates:
            families.setdefault(_evidence_family(candidate.source), []).append(candidate)
        representatives = []
        for family_candidates in families.values():
            ranked = tuple(sorted(
                family_candidates,
                key=lambda candidate: (
                    -candidate.confidence, SOURCE_PRIORITY[candidate.source],
                ),
            ))
            weights = [candidate.confidence for candidate in ranked]
            rate = _weighted_median(
                [candidate.rate_bpm for candidate in ranked], weights,
            )
            # A family has one vote: retain its strongest observed confidence
            # while using all agreeing paths to make its rate representative.
            representatives.append(replace(ranked[0], rate_bpm=rate))
        return tuple(representatives)

    def _select(self, candidates):
        ranked = tuple(sorted(
            candidates,
            key=lambda candidate: (-candidate.confidence, SOURCE_PRIORITY[candidate.source]),
        ))
        by_source = {}
        for candidate in ranked:
            by_source.setdefault(candidate.source, candidate)
        ranked = self._family_representatives(tuple(by_source.values()))
        by_rate = tuple(sorted(ranked, key=lambda candidate: candidate.rate_bpm))
        clusters = [
            tuple(by_rate[start:end + 1])
            for start in range(len(by_rate))
            for end in range(start, len(by_rate))
            if by_rate[end].rate_bpm - by_rate[start].rate_bpm <= self.agreement_bpm
        ]
        cluster = max(clusters, key=lambda group: (
            sum(candidate.confidence for candidate in group), len(group),
            -min(SOURCE_PRIORITY[candidate.source] for candidate in group),
        ))
        if len(ranked) > 1 and len(cluster) == 1:
            return None
        weights = [candidate.confidence for candidate in cluster]
        rate = _weighted_median([candidate.rate_bpm for candidate in cluster], weights)
        confidence = float(np.average(weights, weights=weights))
        families = {_evidence_family(candidate.source) for candidate in cluster}
        source = ("FUSED" if len(families) > 1 else
                  max(cluster, key=lambda candidate: (
                      candidate.confidence,
                      -SOURCE_PRIORITY[candidate.source],
                  )).source)
        wave = max(cluster, key=lambda candidate: candidate.confidence).wave
        return rate, confidence, source, frozenset(candidate.source for candidate in cluster), wave

    @staticmethod
    def preview(candidates):
        """Expose the strongest weak candidate without changing lock state."""
        candidates = tuple(candidates)
        candidate = min(
            candidates,
            key=lambda item: (-item.confidence, SOURCE_PRIORITY[item.source]),
        )
        return RespirationResult(
            candidate.rate_bpm, candidate.confidence, "ACQUIRING", candidate.source,
            False, False, "LOW_QUALITY", candidate.wave, candidates,
        )

    def update(self, now, candidates):
        candidates = tuple(candidates)
        selected = self._select(candidates) if candidates else None
        if selected is None:
            self.history.clear()
            reason = "CANDIDATE_CONFLICT" if len(candidates) > 1 else "NO_CANDIDATE"
            if self.last_locked is not None:
                rate, confidence, source, stamp, wave = self.last_locked
                age = max(0.0, float(now) - stamp)
                if age <= self.hold_sec:
                    return RespirationResult(
                        rate, confidence * (1.0 - age / self.hold_sec), "HOLD",
                        source, False, False, reason, wave, candidates,
                    )
            return RespirationResult(
                0.0, 0.0, "UNAVAILABLE", "NONE", False, False,
                reason, None, candidates,
            )

        rate, confidence, source, identity, wave = selected
        if self.history and self.history[-1][1] != identity:
            self.history.clear()
        self.history.append((rate, identity))
        stable = (len(self.history) == self.lock_updates
                  and np.ptp([item[0] for item in self.history]) <= self.stability_bpm)
        if not stable:
            return RespirationResult(
                rate, confidence, "ACQUIRING", source, False, False,
                "TEMPORAL_LOCK", wave, candidates,
            )
        self.last_locked = (rate, confidence, source, float(now), wave)
        return RespirationResult(
            rate, confidence, "LOCKED", source, True,
            confidence >= self.learn_conf, "OK", wave, candidates,
        )


class AdaptiveRespirationEstimator:
    """Own adaptive RR evidence across framing changes and short gaps.

    Flow samples are intentionally kept in separate source buffers.  Geometry
    changes therefore pause only the affected source instead of joining torso,
    shoulder, or face displacement into an artificial step waveform.
    """

    _FLOW_SOURCES = ("TORSO", "SHOULDER", "FACE_MOTION")

    def __init__(self, frame_shape, *, flow_hz=config.RR_FLOW_HZ,
                 motion_rate=MOTION_RATE, mute_sec=MUTE_SEC):
        self.flow_period = 1.0 / float(flow_hz)
        self.tracker = MultiRegionKeyframeTracker(
            frame_shape, {}, motion_rate=motion_rate, mute_sec=mute_sec,
        )
        self.flow_buffers = {}
        self.bvp_candidates = ()
        self.bvp_stamp = -1e9
        self._bvp_input = None
        self.fusion = RespirationFusion()
        self._latest_flow_candidates = ()
        self._motion_active = False
        self.last_result = RespirationResult(
            0.0, 0.0, "UNAVAILABLE", "NONE", False, False,
            "WARMING UP", None, (),
        )

    def _buffer(self, source):
        buffer = self.flow_buffers.get(source)
        if buffer is None:
            buffer = {
                "active_time": 0.0,
                "last_wall": None,
                "samples": deque(maxlen=max(8, int(np.ceil(
                    WIN_SEC * config.RR_FLOW_HZ * 1.2,
                )))),
            }
            self.flow_buffers[source] = buffer
        return buffer

    def _append_flow(self, now, values, valid):
        """Append continuous source evidence without counting gap wall time."""
        now = float(now)
        values = dict(values)
        for source in set(self.flow_buffers) | set(values):
            buffer = self._buffer(source)
            displacement = values.get(source)
            if not valid or displacement is None:
                # Do not delete history: the next valid frame starts a new
                # active-time segment and cannot bridge this wall-clock gap.
                buffer["last_wall"] = None
                continue
            displacement = np.asarray(displacement, dtype=float).copy()
            if displacement.ndim != 1 or not np.all(np.isfinite(displacement)):
                buffer["last_wall"] = None
                continue
            last_wall = buffer["last_wall"]
            if last_wall is None:
                # The first sample after every gap anchors a new segment; it
                # has no preceding interval and must not become a sample.
                buffer["last_wall"] = now
                continue
            buffer["active_time"] += max(0.0, now - last_wall)
            buffer["last_wall"] = now
            buffer["samples"].append((
                float(buffer["active_time"]), displacement, True,
            ))

    def update_frame(self, gray, *, face_box, person_box, now,
                     motion_hold=False):
        regions = respiration_rois(face_box, person_box, gray.shape)
        self.tracker.set_regions(regions, now)
        if self.tracker.needs_anchor(now) and not self.tracker.anchor(gray, now):
            self._append_flow(now, {}, False)
            return
        tracked = self.tracker.track(gray, now)
        if tracked is None:
            self._append_flow(now, {}, False)
            return
        values, valid = tracked
        # `_append_flow` also marks every known source absent from `values`
        # invalid, so its active clock cannot leap across a framing gap.
        self._append_flow(now, values, bool(valid) and not motion_hold)

    def update_bvp(self, times, values, *, confidence, now):
        self._bvp_input = (
            np.asarray(times, dtype=float).copy(),
            np.asarray(values, dtype=float).copy(),
            float(confidence),
        )
        self.bvp_candidates = estimate_bvp_candidates(
            times, values, bvp_confidence=confidence,
        )
        self.bvp_stamp = float(now)

    def _flow_candidates(self, *, display=False):
        candidates = []
        for source, buffer in self.flow_buffers.items():
            samples = tuple(buffer["samples"])
            if (buffer["active_time"] < WIN_SEC * 0.9 or len(samples) < 8):
                continue
            times = np.asarray([sample[0] for sample in samples], dtype=float)
            displacement = np.asarray(
                [sample[1] for sample in samples], dtype=float,
            )
            valid = np.asarray([sample[2] for sample in samples], dtype=bool)
            if len(times) < 2:
                continue
            span = max(float(times[-1] - times[0]), 1e-6)
            coverage = min(1.0, span / max(WIN_SEC * 0.9, 1e-6))
            candidate = estimate_flow_candidate(
                times, displacement, valid, source, coverage, display=display,
            )
            if candidate is not None:
                candidates.append(candidate)
        return tuple(candidates)

    def report(self, now, *, motion_hold=False):
        if motion_hold:
            if not self._motion_active:
                # Motion makes both image paths correlated with camera motion.
                # Drop their windows and the lock sequence so the next fresh
                # result must be rebuilt from post-motion evidence.
                for buffer in self.flow_buffers.values():
                    buffer["active_time"] = 0.0
                    buffer["last_wall"] = None
                    buffer["samples"].clear()
                self.bvp_candidates = ()
                self.bvp_stamp = -1e9
                self._bvp_input = None
                self.fusion.history.clear()
                self._motion_active = True
            self._latest_flow_candidates = ()
            self.last_result = self.fusion.update(now, ())
            return self.last_result

        self._motion_active = False
        self._latest_flow_candidates = self._flow_candidates()
        bvp = (self.bvp_candidates
               if float(now) - self.bvp_stamp <= config.RR_BVP_MAX_AGE else ())
        self.last_result = self.fusion.update(
            now, self._latest_flow_candidates + bvp,
        )
        # A strict disagreement is meaningful diagnostic evidence.  Only
        # synthesize a display-only preview when *no* strict candidate exists;
        # otherwise a weak candidate could hide a real source conflict.
        if self.last_result.state != "UNAVAILABLE" or self._latest_flow_candidates or bvp:
            return self.last_result

        display_bvp = ()
        if (self._bvp_input is not None
                and float(now) - self.bvp_stamp <= config.RR_BVP_MAX_AGE):
            times, values, confidence = self._bvp_input
            display_bvp = estimate_display_bvp_candidates(times, values, confidence)
        display_candidates = self._flow_candidates(display=True) + display_bvp
        if display_candidates:
            self.last_result = self.fusion.preview(display_candidates)
        return self.last_result

    def _diagnostic_snapshot(self):
        source_buffers = {}
        for source, buffer in self.flow_buffers.items():
            samples = tuple(buffer["samples"])
            source_buffers[source] = {
                "active_time": float(buffer["active_time"]),
                "span": (float(samples[-1][0] - samples[0][0])
                         if len(samples) > 1 else 0.0),
                "samples": len(samples),
                "valid_fraction": (float(np.mean([sample[2] for sample in samples]))
                                   if samples else 0.0),
            }
        target_seconds = float(WIN_SEC * 0.9)
        active_seconds = max(
            (float(buffer["active_time"])
             for buffer in self.flow_buffers.values()),
            default=0.0,
        )
        return {
            "sources": source_buffers,
            "acquisition": {
                "active_seconds": active_seconds,
                "target_seconds": target_seconds,
                "fraction": min(1.0, active_seconds / target_seconds),
            },
            "candidates": [
                {"source": candidate.source, "rate": float(candidate.rate_bpm),
                 "confidence": float(candidate.confidence)}
                for candidate in self.last_result.candidates
            ],
            "tracking_points": int(sum(self.tracker.region_counts.values())),
            "reason": self.last_result.reason,
        }

    def diagnostics(self):
        return self._diagnostic_snapshot()

    def _numeric_dump_arrays(self):
        arrays = {}
        for source in self._FLOW_SOURCES:
            samples = tuple(self.flow_buffers.get(source, {}).get("samples", ()))
            prefix = f"rr_{source.lower()}"
            arrays[f"{prefix}_t"] = np.asarray(
                [sample[0] for sample in samples], dtype=float,
            )
            arrays[f"{prefix}_disp"] = np.asarray(
                [sample[1] for sample in samples], dtype=float,
            ) if samples else np.empty((0, NCELL), dtype=float)
            arrays[f"{prefix}_valid"] = np.asarray(
                [sample[2] for sample in samples], dtype=bool,
            )
        candidates = self.last_result.candidates
        arrays["rr_candidate_rate"] = np.asarray(
            [candidate.rate_bpm for candidate in candidates], dtype=float,
        )
        arrays["rr_candidate_conf"] = np.asarray(
            [candidate.confidence for candidate in candidates], dtype=float,
        )
        arrays["rr_selected_source"] = np.asarray(
            [self.last_result.source], dtype="<U16",
        )
        arrays["rr_selected_state"] = np.asarray(
            [self.last_result.state], dtype="<U16",
        )
        return arrays

    def dump_arrays(self):
        return self._numeric_dump_arrays()


def combine_quality(*, spectral_snr, periodicity, concentration,
                    valid_fraction, coverage):
    """Return weighted normalized candidate quality without product collapse."""
    values = np.clip(
        [spectral_snr, periodicity, concentration, valid_fraction, coverage],
        0.0, 1.0,
    )
    weights = np.asarray(config.RR_QUALITY_WEIGHTS, dtype=float)
    return float(np.dot(values, weights) / weights.sum())


def _sample_rate(times):
    steps = np.diff(times)
    steps = steps[np.isfinite(steps) & (steps > 0.0)]
    return float(1.0 / np.median(steps)) if len(steps) else 0.0


def _bandpass(values, sample_rate, band):
    if sample_rate <= 2.0 * band[1]:
        return None
    sos = signal.butter(3, [band[0] / (sample_rate / 2.0),
                            band[1] / (sample_rate / 2.0)],
                        btype="band", output="sos")
    try:
        return signal.sosfiltfilt(sos, values, axis=0)
    except ValueError:
        return None


def _normalised_acf(values):
    centered = values - np.mean(values)
    acf = np.correlate(centered, centered, mode="full")[len(centered) - 1:]
    if not len(acf) or acf[0] <= 0.0:
        return None
    return acf / acf[0]


def _select_periodic_frequency(freqs, powers, wave, sample_rate,
                               octave_correct=False):
    band = (freqs >= RR_BAND[0]) & (freqs <= RR_BAND[1])
    if not np.any(band):
        return None
    in_band_freqs, in_band_powers = freqs[band], powers[band]
    peak_index = int(np.argmax(in_band_powers))
    peak_frequency = _parabolic(in_band_freqs, in_band_powers, peak_index)
    acf = _normalised_acf(wave)
    if acf is None:
        return None
    selected = peak_frequency
    if octave_correct:
        peaks, _ = signal.find_peaks(acf)
        candidates = []
        for multiple in (1 / 3, 1 / 2, 1, 2, 3):
            frequency = peak_frequency * multiple
            if not RR_BAND[0] <= frequency <= RR_BAND[1]:
                continue
            target_lag = sample_rate / frequency
            if len(peaks):
                nearest = peaks[int(np.argmin(np.abs(peaks - target_lag)))]
                if (abs(nearest - target_lag) <= 0.12 * target_lag
                        and acf[nearest] > config.RR_MIN_PERIODICITY):
                    candidates.append(frequency)
        if candidates:
            selected = max(candidates)
    period_lag = sample_rate / selected
    periodicity = float(np.interp(period_lag, np.arange(len(acf)), acf))
    return selected, in_band_powers, peak_index, periodicity


def _candidate_from_wave(times, wave, *, source, valid_fraction, coverage,
                         octave_correct=False, display=False):
    times = np.asarray(times, dtype=float)
    wave = np.asarray(wave, dtype=float)
    valid_fraction = float(valid_fraction)
    coverage = float(coverage)
    min_valid = (config.RR_DISPLAY_MIN_VALID_FRACTION if display
                 else config.RR_MIN_VALID_FRACTION)
    min_coverage = (config.RR_DISPLAY_MIN_COVERAGE if display
                    else config.RR_MIN_COVERAGE)
    min_periodicity = (config.RR_DISPLAY_MIN_PERIODICITY if display
                       else config.RR_MIN_PERIODICITY)
    if (times.ndim != 1 or wave.ndim != 1 or len(times) != len(wave)
            or len(times) < 8 or not np.all(np.isfinite(times))
            or not np.all(np.isfinite(wave))
            or times[-1] - times[0] < WIN_SEC * 0.9
            or valid_fraction < min_valid
            or coverage < min_coverage):
        return None
    sample_rate = _sample_rate(times)
    if sample_rate <= 0.0:
        return None
    freqs, powers = signal.welch(wave, fs=sample_rate, nperseg=len(wave),
                                 nfft=4 * len(wave))
    selected = _select_periodic_frequency(
        freqs, powers, wave, sample_rate, octave_correct=octave_correct,
    )
    if selected is None:
        return None
    frequency, in_band_powers, peak_index, periodicity = selected
    rate_bpm = frequency * 60.0
    if not np.isfinite(rate_bpm) or not RR_BAND[0] * 60.0 <= rate_bpm <= RR_BAND[1] * 60.0:
        return None
    peak_power = float(in_band_powers[peak_index])
    total_power = float(np.sum(in_band_powers))
    remaining_power = max(total_power - peak_power, np.finfo(float).eps)
    spectral_snr = peak_power / remaining_power
    concentration = peak_power / max(total_power, np.finfo(float).eps)
    if not np.isfinite(periodicity) or periodicity < min_periodicity:
        return None
    normalized_snr = np.clip(spectral_snr / config.RR_SNR_FULL, 0.0, 1.0)
    normalized_periodicity = np.clip(periodicity / config.RR_PERIODICITY_FULL, 0.0, 1.0)
    normalized_concentration = np.clip(
        concentration / config.RR_CONCENTRATION_FULL, 0.0, 1.0,
    )
    confidence = combine_quality(
        spectral_snr=normalized_snr, periodicity=normalized_periodicity,
        concentration=normalized_concentration, valid_fraction=valid_fraction,
        coverage=coverage,
    )
    tail = int(min(len(wave), max(2, round(config.RESP_WAVE_SEC * sample_rate))))
    return RespirationCandidate(
        rate_bpm=rate_bpm, source=source, confidence=confidence,
        spectral_snr=float(normalized_snr), periodicity=float(normalized_periodicity),
        concentration=float(normalized_concentration),
        valid_fraction=float(np.clip(valid_fraction, 0.0, 1.0)),
        coverage=float(np.clip(coverage, 0.0, 1.0)), wave=wave[-tail:],
    )


def estimate_signal_candidate(times, signal_values, source, valid_fraction,
                              coverage, *, display=False):
    """Score a uniform respiration-band waveform using shared hard/soft gates."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(signal_values, dtype=float)
    if times.ndim != 1 or values.ndim != 1 or len(times) != len(values):
        return None
    sample_rate = _sample_rate(times)
    if sample_rate <= 0.0 or not np.all(np.isfinite(values)):
        return None
    filtered = _bandpass(signal.detrend(values), sample_rate, RR_BAND)
    if filtered is None:
        return None
    return _candidate_from_wave(
        times, filtered, source=source, valid_fraction=valid_fraction,
        coverage=coverage, display=display,
    )


def estimate_display_signal_candidate(times, signal_values, source, valid_fraction,
                                      coverage):
    """Create a display-only respiration candidate using relaxed quality gates."""
    return estimate_signal_candidate(
        times, signal_values, source, valid_fraction, coverage, display=True,
    )


def estimate_flow_candidate(times, displacement, valid, source, coverage, *, display=False):
    """Convert multi-cell optical flow into one common-quality candidate."""
    times = np.asarray(times, dtype=float)
    displacement = np.asarray(displacement, dtype=float)
    valid = np.asarray(valid, dtype=bool)
    if (times.ndim != 1 or displacement.ndim != 2 or len(times) != len(displacement)
            or len(valid) != len(times) or valid.sum() < 2):
        return None
    valid_fraction = float(valid.mean())
    min_valid = (config.RR_DISPLAY_MIN_VALID_FRACTION if display
                 else config.RR_MIN_VALID_FRACTION)
    min_energy = (config.RR_DISPLAY_FLOW_MIN_MOTION_ENERGY if display
                  else config.RR_FLOW_MIN_MOTION_ENERGY)
    min_snr = (config.RR_DISPLAY_FLOW_MIN_SPECTRAL_SNR if display
               else config.RR_FLOW_MIN_SPECTRAL_SNR)
    min_concentration = (config.RR_DISPLAY_FLOW_MIN_CONCENTRATION if display
                         else config.RR_FLOW_MIN_CONCENTRATION)
    if valid_fraction < min_valid:
        return None
    grid = np.arange(times[0], times[-1], 1.0 / FS)
    if len(grid) < 8:
        return None
    valid_times = times[valid]
    cells = displacement[valid]
    if not np.all(np.isfinite(cells)):
        return None
    matrix = np.stack([np.interp(grid, valid_times, cells[:, cell])
                       for cell in range(cells.shape[1])], axis=1)
    filtered = _bandpass(signal.detrend(matrix, axis=0), FS, RR_BAND)
    if filtered is None:
        return None
    motion_energy = float(np.sqrt(np.mean(filtered ** 2)))
    if (not np.isfinite(motion_energy)
            or motion_energy < min_energy):
        return None
    deviation = filtered.std(axis=0)
    keep = deviation > max(1e-9, 0.25 * np.median(deviation))
    if keep.sum() < 2:
        return None
    normalised = filtered[:, keep] / deviation[keep]
    centered = normalised - normalised.mean(axis=0)
    left, singular, _ = np.linalg.svd(centered, full_matrices=False)
    wave = left[:, 0] * singular[0]
    if float(np.dot(wave, centered.mean(axis=1))) < 0.0:
        wave = -wave
    candidate = _candidate_from_wave(
        grid, wave, source=source, valid_fraction=valid_fraction,
        coverage=coverage, octave_correct=True, display=display,
    )
    if (candidate is None
            or candidate.spectral_snr < min_snr
            or candidate.concentration < min_concentration):
        return None
    return candidate


def estimate_display_flow_candidate(times, displacement, valid, source, coverage):
    """Create a display-only optical-flow candidate using relaxed quality gates."""
    return estimate_flow_candidate(
        times, displacement, valid, source, coverage, display=True,
    )


def estimate_bvp_candidates(times, bvp, bvp_confidence, *, display=False):
    """Derive respiration candidates from BVP amplitude and frequency modulation."""
    confidence = float(bvp_confidence)
    min_conf = config.RR_DISPLAY_BVP_MIN_CONF if display else config.RR_BVP_MIN_CONF
    min_periodicity = (config.RR_DISPLAY_BVP_MIN_PERIODICITY if display
                       else config.RR_BVP_MIN_PERIODICITY)
    min_concentration = (config.RR_DISPLAY_BVP_MIN_CONCENTRATION if display
                         else config.RR_BVP_MIN_CONCENTRATION)
    if not np.isfinite(confidence) or confidence < min_conf:
        return ()
    times = np.asarray(times, dtype=float)
    bvp = np.asarray(bvp, dtype=float)
    if (times.ndim != 1 or bvp.ndim != 1 or len(times) != len(bvp)
            or not np.all(np.isfinite(times)) or not np.all(np.isfinite(bvp))
            or not np.all(np.diff(times) > 0.0)):
        return ()
    sample_rate = _sample_rate(times)
    uniform_times = np.arange(times[0], times[-1], 1.0 / sample_rate)
    if len(uniform_times) < 8:
        return ()
    uniform_bvp = np.interp(uniform_times, times, bvp)
    pulse = _bandpass(signal.detrend(uniform_bvp), sample_rate, (0.7, 3.5))
    if pulse is None:
        return ()
    analytic = signal.hilbert(pulse)
    modulation = {
        "BVP_AM": np.abs(analytic),
        "BVP_FM": np.gradient(np.unwrap(np.angle(analytic))) * sample_rate / (2.0 * np.pi),
    }
    scale = float(np.clip(confidence, 0.0, 1.0))
    candidates = []
    for source, values in modulation.items():
        candidate = estimate_signal_candidate(
            uniform_times, values, source=source, valid_fraction=1.0,
            coverage=1.0, display=display,
        )
        # BVP phase derivatives can manufacture weak low-frequency structure
        # from envelope-only pulses or noise.  Require a strongly repeating,
        # concentrated modulation before fusion.
        if (candidate is not None
                and candidate.periodicity >= min_periodicity
                and candidate.concentration >= min_concentration):
            candidates.append(replace(
                candidate, confidence=candidate.confidence * scale,
            ))
    return tuple(candidates)


def estimate_display_bvp_candidates(times, bvp, bvp_confidence):
    """Create display-only BVP respiration candidates using relaxed quality gates."""
    return estimate_bvp_candidates(times, bvp, bvp_confidence, display=True)


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
                    evr=0.0, prom=0.0, vfrac=vfrac, nkeep=0, ach=0.0,
                    wave=None, wave_fs=FS)

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
    Xc = X - X.mean(0)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    evr = float(S[0] ** 2 / np.sum(S ** 2))
    pc1 = U[:, 0] * S[0]

    # C-4: SVD 의 부호는 임의라 창마다 파형이 통째로 뒤집힐 수 있다. 스펙트럼
    # 과 자기상관은 부호에 무관하지만 화면 파형은 깜빡인다. 셀 평균과 양의
    # 상관을 갖도록 고정한다.
    if float(np.dot(pc1, Xc.mean(axis=1))) < 0.0:
        pc1 = -pc1

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

    # C-4: 화면에 그릴 "진짜" 호흡 파형. 원시 변위(carry+rel)는 키프레임마다
    # 계단이 생기고 저주파 드리프트가 지배해서 호흡처럼 보이지 않았다.
    # pc1 은 이미 0.1-0.5Hz 대역통과 + SVD 를 거친 신호다.
    tail = int(min(len(pc1), max(2, round(config.RESP_WAVE_SEC * FS))))

    return dict(rr=rr_spec, sqi=sqi, rr_spec=rr_spec, rr_time=rr_time,
                evr=evr, prom=prom, vfrac=vfrac, nkeep=nkeep, ach=ach,
                wave=pc1[-tail:], wave_fs=FS)


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

    def __init__(self, sqi_min=SQI_MIN):
        self.sqi_min = sqi_min
        self.hist = []

    def update(self, t, rr, sqi):
        if np.isfinite(rr) and sqi >= self.sqi_min:
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
