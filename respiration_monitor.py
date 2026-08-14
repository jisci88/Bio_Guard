from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class EstimatorConfig:
    sample_rate: float = 15.0
    min_window_seconds: float = 20.0
    analysis_window_seconds: float = 35.0
    min_rr_bpm: float = 6.0
    max_rr_bpm: float = 42.0
    filter_order: int = 4
    max_motion_fraction: float = 0.22
    min_tracking_quality: float = 0.45
    accept_confidence: float = 0.58


@dataclass(frozen=True)
class RespirationResult:
    rr_bpm: Optional[float]
    confidence: float
    quality: str
    snr_db: float = 0.0
    spectral_bpm: Optional[float] = None
    autocorr_bpm: Optional[float] = None
    peak_bpm: Optional[float] = None
    tracking_quality: float = 0.0
    motion_fraction: float = 0.0
    diagnostics: Dict[str, float] = field(default_factory=dict)


class RespirationEstimator:
    """Timestamp-aware respiratory-rate estimator for a 1-D chest-motion signal."""

    def __init__(self, config: EstimatorConfig | None = None):
        self.config = config or EstimatorConfig()
        max_samples = int(self.config.sample_rate * self.config.analysis_window_seconds * 3)
        self._samples: Deque[tuple[float, float, float, bool]] = deque(maxlen=max_samples)

    def add_motion(
        self,
        timestamp: float,
        motion: float,
        tracking_quality: float = 1.0,
        motion_artifact: bool = False,
    ) -> None:
        if not np.isfinite(timestamp) or not np.isfinite(motion):
            return
        self._samples.append(
            (float(timestamp), float(motion), float(np.clip(tracking_quality, 0.0, 1.0)), bool(motion_artifact))
        )

    def estimate(self) -> RespirationResult:
        if len(self._samples) < 4:
            return RespirationResult(None, 0.0, "WARMING_UP")

        samples = np.asarray(self._samples, dtype=float)
        times = samples[:, 0]
        values = samples[:, 1]
        track_q = samples[:, 2]
        motion_flags = samples[:, 3] > 0.5

        order = np.argsort(times)
        times, values, track_q, motion_flags = (
            times[order], values[order], track_q[order], motion_flags[order]
        )
        keep = np.r_[True, np.diff(times) > 1e-6]
        times, values, track_q, motion_flags = (
            times[keep], values[keep], track_q[keep], motion_flags[keep]
        )

        end_t = times[-1]
        start_t = max(times[0], end_t - self.config.analysis_window_seconds)
        use = times >= start_t
        times, values, track_q, motion_flags = (
            times[use], values[use], track_q[use], motion_flags[use]
        )

        duration = times[-1] - times[0]
        if duration < self.config.min_window_seconds:
            return RespirationResult(None, 0.0, "WARMING_UP")

        mean_track_q = float(np.mean(track_q))
        motion_fraction = float(np.mean(motion_flags))
        if mean_track_q < self.config.min_tracking_quality:
            return RespirationResult(
                None, 0.0, "TRACKING_POOR", tracking_quality=mean_track_q, motion_fraction=motion_fraction
            )
        if motion_fraction > self.config.max_motion_fraction:
            return RespirationResult(
                None, 0.0, "MOTION", tracking_quality=mean_track_q, motion_fraction=motion_fraction
            )

        uniform_t, raw = self._resample(times, values, motion_flags)
        if raw is None:
            return RespirationResult(
                None, 0.0, "UNRELIABLE", tracking_quality=mean_track_q, motion_fraction=motion_fraction
            )

        filtered = self._preprocess(raw)
        if filtered is None:
            return RespirationResult(
                None, 0.0, "UNRELIABLE", tracking_quality=mean_track_q, motion_fraction=motion_fraction
            )

        spectral_bpm, snr_db, spectral_concentration, spectral_entropy_score = self._spectral_estimate(filtered)
        autocorr_bpm, ac_strength = self._autocorr_estimate(filtered)
        peak_bpm, peak_consistency = self._peak_estimate(filtered)

        fused, agreement, n_agree = self._fuse(
            [
                (spectral_bpm, max(0.15, spectral_concentration)),
                (autocorr_bpm, max(0.15, ac_strength)),
                (peak_bpm, max(0.10, peak_consistency)),
            ]
        )

        periodicity = float(np.clip(
            0.33 * self._scale(snr_db, 2.5, 12.0)
            + 0.27 * np.clip(ac_strength, 0.0, 1.0)
            + 0.20 * np.clip(spectral_entropy_score, 0.0, 1.0)
            + 0.20 * np.clip(peak_consistency, 0.0, 1.0),
            0.0,
            1.0,
        ))
        confidence = float(np.clip(
            0.42 * agreement
            + 0.33 * periodicity
            + 0.15 * mean_track_q
            + 0.10 * (1.0 - motion_fraction / max(self.config.max_motion_fraction, 1e-6)),
            0.0,
            1.0,
        ))

        peak_evidence = bool(
            peak_consistency >= 0.30
            or (ac_strength >= 0.75 and spectral_entropy_score >= 0.22 and snr_db >= 8.0)
        )
        accepted = (
            fused is not None
            and n_agree >= 2
            and snr_db >= 3.0
            and ac_strength >= 0.20
            and agreement >= 0.55
            and periodicity >= 0.50
            and peak_evidence
            and confidence >= self.config.accept_confidence
        )

        quality = "GOOD" if accepted and confidence >= 0.76 else "FAIR" if accepted else "UNRELIABLE"
        return RespirationResult(
            rr_bpm=float(fused) if accepted else None,
            confidence=confidence,
            quality=quality,
            snr_db=float(snr_db),
            spectral_bpm=spectral_bpm,
            autocorr_bpm=autocorr_bpm,
            peak_bpm=peak_bpm,
            tracking_quality=mean_track_q,
            motion_fraction=motion_fraction,
            diagnostics={
                "agreement": float(agreement),
                "periodicity": periodicity,
                "autocorr_strength": float(ac_strength),
                "peak_consistency": float(peak_consistency),
                "spectral_concentration": float(spectral_concentration),
                "spectral_entropy_score": float(spectral_entropy_score),
                "duration_seconds": float(uniform_t[-1] - uniform_t[0]),
            },
        )

    def _resample(self, times: np.ndarray, values: np.ndarray, motion_flags: np.ndarray):
        fs = self.config.sample_rate
        dt = 1.0 / fs
        uniform_t = np.arange(times[0], times[-1] + 0.5 * dt, dt)
        if len(uniform_t) < int(self.config.min_window_seconds * fs):
            return uniform_t, None

        good = ~motion_flags
        if np.count_nonzero(good) < max(8, int(0.7 * len(times))):
            return uniform_t, None

        cleaned = values.copy()
        cleaned[motion_flags] = np.nan
        valid = np.isfinite(cleaned)
        if np.count_nonzero(valid) < 4:
            return uniform_t, None
        interp = np.interp(uniform_t, times[valid], cleaned[valid])
        return uniform_t, interp

    def _preprocess(self, x: np.ndarray) -> Optional[np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        if len(x) < 32 or np.ptp(x) < 1e-8:
            return None

        # Hampel-like replacement of isolated spikes using a short running median.
        k = 5 if len(x) >= 5 else 3
        med = signal.medfilt(x, kernel_size=k)
        residual = x - med
        mad = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-9
        x = x.copy()
        spikes = np.abs(residual) > 6.0 * mad
        x[spikes] = med[spikes]

        x = signal.detrend(x, type="linear")
        fs = self.config.sample_rate
        lo = self.config.min_rr_bpm / 60.0
        hi = self.config.max_rr_bpm / 60.0
        if not (0 < lo < hi < fs / 2):
            return None
        sos = signal.butter(self.config.filter_order, [lo, hi], btype="bandpass", fs=fs, output="sos")
        try:
            y = signal.sosfiltfilt(sos, x)
        except ValueError:
            return None
        scale = np.std(y)
        if not np.isfinite(scale) or scale < 1e-8:
            return None
        return y / scale

    def _spectral_estimate(self, x: np.ndarray):
        fs = self.config.sample_rate
        n = len(x)
        nperseg = min(n, max(128, int(fs * 24.0)))
        nfft = 1
        while nfft < nperseg:
            nfft <<= 1
        nfft *= 8
        f, pxx = signal.welch(
            x,
            fs=fs,
            window="hann_periodic",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            nfft=nfft,
            detrend="constant",
            scaling="density",
            average="median",
        )
        lo = self.config.min_rr_bpm / 60.0
        hi = self.config.max_rr_bpm / 60.0
        mask = (f >= lo) & (f <= hi)
        fb, pb = f[mask], pxx[mask]
        if len(fb) < 5 or not np.any(pb > 0):
            return None, 0.0, 0.0, 0.0

        idx = int(np.argmax(pb))
        freq = self._quadratic_peak_frequency(fb, pb, idx)
        peak_power = float(pb[idx])
        exclusion = np.abs(fb - fb[idx]) <= max(0.04, 2.0 * (fb[1] - fb[0]))
        noise = pb[~exclusion]
        noise_floor = float(np.median(noise)) if len(noise) else 1e-12
        snr_db = 10.0 * np.log10((peak_power + 1e-12) / (noise_floor + 1e-12))

        total = float(np.sum(pb)) + 1e-12
        local = np.abs(fb - freq) <= 0.05
        concentration = float(np.clip(np.sum(pb[local]) / total * 2.2, 0.0, 1.0))
        prob = pb / total
        entropy = -float(np.sum(prob * np.log(prob + 1e-12))) / np.log(len(prob))
        entropy_score = float(np.clip(1.0 - entropy, 0.0, 1.0))
        return float(freq * 60.0), float(snr_db), concentration, entropy_score

    def _autocorr_estimate(self, x: np.ndarray):
        fs = self.config.sample_rate
        ac = signal.correlate(x, x, mode="full", method="fft")[len(x) - 1 :]
        overlap = np.arange(len(x), 0, -1, dtype=float)
        ac = ac / overlap
        if ac[0] <= 0:
            return None, 0.0
        ac /= ac[0]

        min_lag = max(1, int(fs * 60.0 / self.config.max_rr_bpm))
        max_lag = min(len(ac) - 2, int(fs * 60.0 / self.config.min_rr_bpm))
        if max_lag <= min_lag:
            return None, 0.0
        segment = ac[min_lag : max_lag + 1]
        peaks, props = signal.find_peaks(segment, prominence=0.04)
        if len(peaks) == 0:
            return None, 0.0
        heights = segment[peaks]
        max_h = float(np.max(heights))
        eligible = peaks[heights >= max_h * 0.82]
        p = int(eligible[0] if len(eligible) else peaks[int(np.argmax(heights))])
        lag_index = p + min_lag
        lag = self._quadratic_peak_index(ac, lag_index)
        bpm = 60.0 * fs / lag
        strength = float(np.clip(ac[lag_index], 0.0, 1.0))
        return float(bpm), strength

    def _peak_estimate(self, x: np.ndarray):
        fs = self.config.sample_rate
        min_distance = max(1, int(fs * 60.0 / self.config.max_rr_bpm * 0.75))
        prominence = max(0.25, 0.28 * np.std(x))
        candidates = []
        consistencies = []
        for sig in (x, -x):
            peaks, _ = signal.find_peaks(sig, distance=min_distance, prominence=prominence)
            if len(peaks) < 4:
                continue
            intervals = np.diff(peaks) / fs
            rr = 60.0 / intervals
            valid = (rr >= self.config.min_rr_bpm) & (rr <= self.config.max_rr_bpm)
            rr = rr[valid]
            if len(rr) < 3:
                continue
            med = float(np.median(rr))
            mad = 1.4826 * float(np.median(np.abs(rr - med)))
            consistency = float(np.clip(1.0 - mad / max(0.12 * med, 1.0), 0.0, 1.0))
            candidates.append(med)
            consistencies.append(consistency)
        if not candidates:
            return None, 0.0
        weights = np.asarray(consistencies, dtype=float) + 0.05
        bpm = float(np.average(candidates, weights=weights))
        consistency = float(np.average(consistencies, weights=weights))
        return bpm, consistency

    def _fuse(self, candidates):
        valid = [(float(v), float(w)) for v, w in candidates if v is not None and np.isfinite(v)]
        if len(valid) < 2:
            return None, 0.0, len(valid)
        values = np.array([v for v, _ in valid], dtype=float)
        weights = np.array([max(w, 0.05) for _, w in valid], dtype=float)
        center = float(np.median(values))
        tol = max(2.2, 0.11 * center)
        agreeing = np.abs(values - center) <= tol
        if np.count_nonzero(agreeing) < 2:
            # Try the closest pair, useful when one method locks onto a harmonic.
            best_pair = None
            best_delta = np.inf
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    d = abs(values[i] - values[j])
                    if d < best_delta:
                        best_delta, best_pair = d, (i, j)
            if best_pair is None or best_delta > max(2.5, 0.12 * np.mean(values[list(best_pair)])):
                return None, 0.0, 0
            agreeing[:] = False
            agreeing[list(best_pair)] = True

        av = values[agreeing]
        aw = weights[agreeing]
        fused = float(np.average(av, weights=aw))
        spread = float(np.max(av) - np.min(av)) if len(av) > 1 else 99.0
        agreement = float(np.clip(1.0 - spread / max(3.0, 0.16 * fused), 0.0, 1.0))
        return fused, agreement, int(np.count_nonzero(agreeing))

    @staticmethod
    def _quadratic_peak_frequency(freqs: np.ndarray, power: np.ndarray, idx: int) -> float:
        if idx <= 0 or idx >= len(power) - 1:
            return float(freqs[idx])
        y0, y1, y2 = np.log(power[idx - 1 : idx + 2] + 1e-18)
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) < 1e-12:
            return float(freqs[idx])
        delta = float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))
        return float(freqs[idx] + delta * (freqs[idx + 1] - freqs[idx]))

    @staticmethod
    def _quadratic_peak_index(y: np.ndarray, idx: int) -> float:
        if idx <= 0 or idx >= len(y) - 1:
            return float(idx)
        y0, y1, y2 = y[idx - 1], y[idx], y[idx + 1]
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) < 1e-12:
            return float(idx)
        return float(idx + np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))

    @staticmethod
    def _scale(value: float, low: float, high: float) -> float:
        if high <= low:
            return 0.0
        return float(np.clip((value - low) / (high - low), 0.0, 1.0))


@dataclass(frozen=True)
class VisionConfig:
    process_fps: float = 15.0
    chest_max_corners: int = 160
    reference_max_corners: int = 100
    min_chest_tracks: int = 18
    min_reference_tracks: int = 12
    reseed_interval_frames: int = 45
    quality_level: float = 0.01
    min_distance: float = 5.0
    fb_error_threshold: float = 1.2
    lk_error_threshold: float = 35.0
    max_point_flow_px: float = 6.0
    gross_motion_threshold_px: float = 2.4
    relative_motion_threshold_px: float = 2.2
    auto_face_fallback_frames: int = 45


@dataclass(frozen=True)
class VisionSample:
    displacement: float
    delta_displacement: float
    expansion_displacement: float
    delta_expansion: float
    tracking_quality: float
    motion_artifact: bool
    chest_tracks: int
    reference_tracks: int
    chest_roi: tuple[int, int, int, int]
    reference_roi: tuple[int, int, int, int]


class RespirationVisionTracker:
    """Extract relative chest displacement with sparse KLT optical flow.

    The chest motion is measured relative to a face/reference ROI. This cancels
    camera translation and most whole-body translation before signal analysis.
    """

    def __init__(
        self,
        config: VisionConfig | None = None,
        chest_roi: tuple[int, int, int, int] | None = None,
        reference_roi: tuple[int, int, int, int] | None = None,
    ):
        import cv2

        self.cv2 = cv2
        self.config = config or VisionConfig()
        self.chest_roi = chest_roi
        self.reference_roi = reference_roi
        self.prev_gray: Optional[np.ndarray] = None
        self.chest_points: Optional[np.ndarray] = None
        self.reference_points: Optional[np.ndarray] = None
        self.cumulative_displacement = 0.0
        self.cumulative_expansion = 0.0
        self.frame_index = 0
        self._calibration_frames = 0
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_detector = cv2.CascadeClassifier(cascade_path)

        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 25, 0.01),
            minEigThreshold=1e-4,
        )

    def reset(self) -> None:
        self.prev_gray = None
        self.chest_points = None
        self.reference_points = None
        self.cumulative_displacement = 0.0
        self.cumulative_expansion = 0.0
        self.frame_index = 0
        self._calibration_frames = 0

    def process(self, frame: np.ndarray, timestamp: float | None = None) -> Optional[VisionSample]:
        del timestamp  # timestamps are consumed by RespirationEstimator.
        gray = self._to_gray(frame)
        gray = self.cv2.GaussianBlur(gray, (3, 3), 0)

        if self.chest_roi is None or self.reference_roi is None:
            self._calibration_frames += 1
            rois = self._locate_rois(gray)
            if rois is None:
                if self._calibration_frames < self.config.auto_face_fallback_frames:
                    return None
                rois = self._normalized_fallback_rois(gray.shape)
            self.reference_roi, self.chest_roi = rois

        self.chest_roi = self._clip_roi(self.chest_roi, gray.shape)
        self.reference_roi = self._clip_roi(self.reference_roi, gray.shape)

        if self.prev_gray is None:
            self.chest_points = self._seed_points(gray, self.chest_roi, self.config.chest_max_corners)
            self.reference_points = self._seed_points(gray, self.reference_roi, self.config.reference_max_corners)
            self.prev_gray = gray
            self.frame_index += 1
            return None

        chest_flow, chest_points, chest_raw_n = self._track_group(self.prev_gray, gray, self.chest_points)
        ref_flow, ref_points, ref_raw_n = self._track_group(self.prev_gray, gray, self.reference_points)

        chest_n = 0 if chest_points is None else len(chest_points)
        ref_n = 0 if ref_points is None else len(ref_points)

        chest_motion = self._robust_flow(chest_flow)
        ref_motion = self._robust_flow(ref_flow)

        self.chest_points = chest_points
        self.reference_points = ref_points

        need_reseed = (
            self.frame_index % self.config.reseed_interval_frames == 0
            or chest_n < self.config.min_chest_tracks
            or ref_n < self.config.min_reference_tracks
        )
        if need_reseed:
            self.chest_points = self._seed_points(gray, self.chest_roi, self.config.chest_max_corners)
            self.reference_points = self._seed_points(gray, self.reference_roi, self.config.reference_max_corners)

        self.prev_gray = gray
        self.frame_index += 1

        if chest_motion is None or ref_motion is None:
            return VisionSample(
                displacement=self.cumulative_displacement,
                delta_displacement=0.0,
                expansion_displacement=self.cumulative_expansion,
                delta_expansion=0.0,
                tracking_quality=0.0,
                motion_artifact=True,
                chest_tracks=chest_n,
                reference_tracks=ref_n,
                chest_roi=self.chest_roi,
                reference_roi=self.reference_roi,
            )

        chest_dx, chest_dy = chest_motion
        ref_dx, ref_dy = ref_motion
        relative_dy = chest_dy - ref_dy
        relative_expansion = self._radial_expansion(chest_flow, chest_points, (ref_dx, ref_dy))

        chest_retention = chest_n / max(chest_raw_n, self.config.min_chest_tracks, 1)
        ref_retention = ref_n / max(ref_raw_n, self.config.min_reference_tracks, 1)
        chest_density = min(1.0, chest_n / max(self.config.min_chest_tracks * 2, 1))
        ref_density = min(1.0, ref_n / max(self.config.min_reference_tracks * 2, 1))
        tracking_quality = float(np.clip(
            0.35 * min(1.0, chest_retention)
            + 0.25 * min(1.0, ref_retention)
            + 0.25 * chest_density
            + 0.15 * ref_density,
            0.0,
            1.0,
        ))

        gross_ref_motion = float(np.hypot(ref_dx, ref_dy))
        gross_chest_motion = float(np.hypot(chest_dx, chest_dy))
        motion_artifact = bool(
            gross_ref_motion > self.config.gross_motion_threshold_px
            or gross_chest_motion > self.config.gross_motion_threshold_px * 1.35
            or abs(relative_dy) > self.config.relative_motion_threshold_px
            or abs(relative_expansion) > self.config.relative_motion_threshold_px
            or tracking_quality < 0.30
        )

        if not motion_artifact:
            self.cumulative_displacement += float(relative_dy)
            self.cumulative_expansion += float(relative_expansion)

        return VisionSample(
            displacement=float(self.cumulative_displacement),
            delta_displacement=float(relative_dy),
            expansion_displacement=float(self.cumulative_expansion),
            delta_expansion=float(relative_expansion),
            tracking_quality=tracking_quality,
            motion_artifact=motion_artifact,
            chest_tracks=chest_n,
            reference_tracks=ref_n,
            chest_roi=self.chest_roi,
            reference_roi=self.reference_roi,
        )

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            gray = frame
        elif frame.ndim == 3 and frame.shape[2] == 4:
            gray = self.cv2.cvtColor(frame, self.cv2.COLOR_RGBA2GRAY)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            gray = self.cv2.cvtColor(frame, self.cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError(f"Unsupported frame shape: {frame.shape}")
        if gray.dtype != np.uint8:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        return gray

    def _seed_points(self, gray: np.ndarray, roi: tuple[int, int, int, int], max_corners: int):
        x, y, w, h = roi
        mask = np.zeros_like(gray, dtype=np.uint8)
        inset_x = max(2, int(w * 0.04))
        inset_y = max(2, int(h * 0.04))
        self.cv2.rectangle(
            mask,
            (x + inset_x, y + inset_y),
            (x + w - inset_x - 1, y + h - inset_y - 1),
            255,
            -1,
        )
        pts = self.cv2.goodFeaturesToTrack(
            gray,
            maxCorners=max_corners,
            qualityLevel=self.config.quality_level,
            minDistance=self.config.min_distance,
            mask=mask,
            blockSize=5,
            useHarrisDetector=False,
        )
        return pts.astype(np.float32) if pts is not None else None

    def _track_group(self, prev: np.ndarray, curr: np.ndarray, points: Optional[np.ndarray]):
        if points is None or len(points) < 4:
            return None, None, 0
        raw_n = len(points)
        next_pts, st1, err1 = self.cv2.calcOpticalFlowPyrLK(prev, curr, points, None, **self._lk_params)
        if next_pts is None or st1 is None:
            return None, None, raw_n
        back_pts, st2, _ = self.cv2.calcOpticalFlowPyrLK(curr, prev, next_pts, None, **self._lk_params)
        if back_pts is None or st2 is None:
            return None, None, raw_n

        p0 = points.reshape(-1, 2)
        p1 = next_pts.reshape(-1, 2)
        pb = back_pts.reshape(-1, 2)
        fb = np.linalg.norm(p0 - pb, axis=1)
        flow = p1 - p0
        mag = np.linalg.norm(flow, axis=1)
        lk_err = err1.reshape(-1) if err1 is not None else np.zeros(raw_n)
        good = (
            (st1.reshape(-1) == 1)
            & (st2.reshape(-1) == 1)
            & np.isfinite(fb)
            & (fb <= self.config.fb_error_threshold)
            & np.isfinite(lk_err)
            & (lk_err <= self.config.lk_error_threshold)
            & (mag <= self.config.max_point_flow_px)
        )
        if np.count_nonzero(good) < 4:
            return None, None, raw_n
        good_flow = flow[good]
        good_points = p1[good].reshape(-1, 1, 2).astype(np.float32)
        return good_flow, good_points, raw_n

    def _robust_flow(self, flow: Optional[np.ndarray]) -> Optional[tuple[float, float]]:
        if flow is None or len(flow) < 4:
            return None
        flow = np.asarray(flow, dtype=np.float64)
        center = np.median(flow, axis=0)
        residual = np.linalg.norm(flow - center, axis=1)
        med = float(np.median(residual))
        mad = 1.4826 * float(np.median(np.abs(residual - med))) + 1e-6
        keep = residual <= med + 3.5 * mad
        if np.count_nonzero(keep) < 4:
            keep = np.ones(len(flow), dtype=bool)
        robust = np.median(flow[keep], axis=0)
        return float(robust[0]), float(robust[1])


    def _radial_expansion(
        self,
        chest_flow: np.ndarray,
        chest_points: Optional[np.ndarray],
        reference_flow: tuple[float, float],
    ) -> float:
        if chest_points is None or chest_flow is None or len(chest_flow) < 4:
            return 0.0
        pts = chest_points.reshape(-1, 2).astype(np.float64)
        rel_flow = np.asarray(chest_flow, dtype=np.float64) - np.asarray(reference_flow, dtype=np.float64)
        cx, cy, cw, ch = self.chest_roi
        center = np.array([cx + cw / 2.0, cy + ch / 2.0], dtype=np.float64)
        radial = pts - center
        radius = np.linalg.norm(radial, axis=1)
        valid = radius > max(4.0, 0.08 * min(cw, ch))
        if np.count_nonzero(valid) < 4:
            return 0.0
        unit = radial[valid] / radius[valid, None]
        components = np.sum(rel_flow[valid] * unit, axis=1)
        med = float(np.median(components))
        mad = 1.4826 * float(np.median(np.abs(components - med))) + 1e-6
        keep = np.abs(components - med) <= 3.5 * mad
        if np.count_nonzero(keep) >= 4:
            med = float(np.median(components[keep]))
        return med

    def _locate_rois(self, gray: np.ndarray):
        if self.face_detector.empty():
            return None
        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(45, 45),
            flags=self.cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        reference = self._clip_roi((int(x), int(y), int(w), int(h)), gray.shape)
        cx = x + w / 2.0
        chest = (
            int(cx - 1.20 * w),
            int(y + 1.20 * h),
            int(2.40 * w),
            int(2.00 * h),
        )
        chest = self._clip_roi(chest, gray.shape)
        if chest[2] < 40 or chest[3] < 40:
            return None
        return reference, chest

    @staticmethod
    def _normalized_fallback_rois(shape: tuple[int, int]):
        h, w = shape[:2]
        reference = (int(0.36 * w), int(0.10 * h), int(0.28 * w), int(0.24 * h))
        chest = (int(0.24 * w), int(0.43 * h), int(0.52 * w), int(0.38 * h))
        return reference, chest

    @staticmethod
    def _clip_roi(roi: tuple[int, int, int, int], shape: tuple[int, ...]):
        h_img, w_img = shape[:2]
        x, y, w, h = [int(v) for v in roi]
        x = max(0, min(x, w_img - 2))
        y = max(0, min(y, h_img - 2))
        w = max(2, min(w, w_img - x))
        h = max(2, min(h, h_img - y))
        return x, y, w, h


def _parse_roi(text: str) -> tuple[int, int, int, int]:
    parts = text.split(",")
    if len(parts) != 4:
        raise ValueError("ROI must be x,y,w,h")
    roi = tuple(int(v.strip()) for v in parts)
    if roi[2] <= 0 or roi[3] <= 0:
        raise ValueError("ROI width and height must be positive")
    return roi


def build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="High-reliability camera-based respiratory-rate monitor for Raspberry Pi NoIR cameras."
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--min-rr", type=float, default=6.0, dest="min_rr")
    parser.add_argument("--max-rr", type=float, default=42.0, dest="max_rr")
    parser.add_argument("--window", type=float, default=35.0, help="Analysis window in seconds")
    parser.add_argument("--min-window", type=float, default=20.0, dest="min_window")
    parser.add_argument("--chest-roi", type=_parse_roi, default=None, metavar="X,Y,W,H")
    parser.add_argument("--reference-roi", type=_parse_roi, default=None, metavar="X,Y,W,H")
    parser.add_argument("--lock-exposure", action="store_true", help="Lock AE after camera warm-up")
    parser.add_argument("--display", action="store_true", default=True)
    parser.add_argument("--no-display", action="store_false", dest="display")
    parser.add_argument("--print-every", type=float, default=1.0, dest="print_every")
    return parser


def fuse_channel_results(translation: RespirationResult, expansion: RespirationResult) -> RespirationResult:
    """Fuse independent chest-translation and radial-expansion RR estimates.

    Agreement between mechanically different optical-flow signals is strong evidence
    that the periodic component is respiration rather than motion artifact.
    """
    valid = [r for r in (translation, expansion) if r.rr_bpm is not None]
    if not valid:
        best = max((translation, expansion), key=lambda r: r.confidence)
        return RespirationResult(
            None, best.confidence, best.quality if best.quality != "GOOD" else "UNRELIABLE",
            snr_db=best.snr_db, tracking_quality=best.tracking_quality,
            motion_fraction=best.motion_fraction, diagnostics={"valid_channels": 0.0}
        )
    if len(valid) == 1:
        r = valid[0]
        return RespirationResult(
            r.rr_bpm, r.confidence * 0.94, "FAIR" if r.quality == "GOOD" else r.quality,
            snr_db=r.snr_db, spectral_bpm=r.spectral_bpm, autocorr_bpm=r.autocorr_bpm,
            peak_bpm=r.peak_bpm, tracking_quality=r.tracking_quality, motion_fraction=r.motion_fraction,
            diagnostics={**r.diagnostics, "valid_channels": 1.0},
        )

    a, b = valid
    center = 0.5 * (a.rr_bpm + b.rr_bpm)
    delta = abs(a.rr_bpm - b.rr_bpm)
    tolerance = max(2.0, 0.12 * center)
    if delta > tolerance:
        return RespirationResult(
            None, min(a.confidence, b.confidence) * 0.5, "UNRELIABLE",
            snr_db=min(a.snr_db, b.snr_db),
            tracking_quality=min(a.tracking_quality, b.tracking_quality),
            motion_fraction=max(a.motion_fraction, b.motion_fraction),
            diagnostics={"channel_disagreement_bpm": float(delta), "valid_channels": 2.0},
        )

    weights = np.array([max(a.confidence, 0.05), max(b.confidence, 0.05)], dtype=float)
    rr = float(np.average([a.rr_bpm, b.rr_bpm], weights=weights))
    agreement = float(np.clip(1.0 - delta / tolerance, 0.0, 1.0))
    confidence = float(np.clip(0.55 * np.average([a.confidence, b.confidence], weights=weights) + 0.45 * agreement, 0.0, 1.0))
    quality = "GOOD" if confidence >= 0.76 and a.quality == "GOOD" and b.quality == "GOOD" else "FAIR"
    return RespirationResult(
        rr, confidence, quality, snr_db=float(max(a.snr_db, b.snr_db)),
        tracking_quality=float(min(a.tracking_quality, b.tracking_quality)),
        motion_fraction=float(max(a.motion_fraction, b.motion_fraction)),
        diagnostics={
            "channel_disagreement_bpm": float(delta),
            "channel_agreement": agreement,
            "translation_rr_bpm": float(a.rr_bpm),
            "expansion_rr_bpm": float(b.rr_bpm),
            "valid_channels": 2.0,
        },
    )


class PicameraRespirationMonitor:
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: float = 15.0,
        min_rr_bpm: float = 6.0,
        max_rr_bpm: float = 42.0,
        analysis_window_seconds: float = 35.0,
        min_window_seconds: float = 20.0,
        chest_roi: tuple[int, int, int, int] | None = None,
        reference_roi: tuple[int, int, int, int] | None = None,
        display: bool = True,
        lock_exposure: bool = False,
        print_every: float = 1.0,
    ):
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.display = bool(display)
        self.lock_exposure = bool(lock_exposure)
        self.print_every = max(0.2, float(print_every))
        estimator_config = EstimatorConfig(
                sample_rate=self.fps,
                min_window_seconds=float(min_window_seconds),
                analysis_window_seconds=float(analysis_window_seconds),
                min_rr_bpm=float(min_rr_bpm),
                max_rr_bpm=float(max_rr_bpm),
            )
        self.estimator = RespirationEstimator(estimator_config)
        self.expansion_estimator = RespirationEstimator(estimator_config)
        self.tracker = RespirationVisionTracker(
            VisionConfig(process_fps=self.fps),
            chest_roi=chest_roi,
            reference_roi=reference_roi,
        )
        self._recent_rr: Deque[float] = deque(maxlen=5)

    def run(self) -> None:
        import time
        import cv2

        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "Picamera2/libcamera is unavailable. On Raspberry Pi OS install python3-picamera2 "
                "with apt. If you use a venv, create it with --system-site-packages so libcamera is visible."
            ) from exc

        picam2 = Picamera2()
        camera_config = picam2.create_video_configuration(
            main={"format": "YUV420", "size": (self.width, self.height)},
            controls={"FrameRate": self.fps},
            buffer_count=4,
        )
        picam2.configure(camera_config)
        picam2.start()
        time.sleep(2.0)

        if self.lock_exposure:
            self._lock_current_exposure(picam2)

        last_print = 0.0
        last_result = RespirationResult(None, 0.0, "WARMING_UP")
        try:
            while True:
                request = picam2.capture_request()
                try:
                    yuv = request.make_array("main")
                    metadata = request.get_metadata()
                finally:
                    request.release()
                frame = yuv[: self.height, : self.width]
                sensor_timestamp = metadata.get("SensorTimestamp")
                now = float(sensor_timestamp) * 1e-9 if sensor_timestamp is not None else time.monotonic()
                sample = self.tracker.process(frame, now)
                if sample is not None:
                    self.estimator.add_motion(
                        now, sample.displacement, sample.tracking_quality, sample.motion_artifact
                    )
                    self.expansion_estimator.add_motion(
                        now, sample.expansion_displacement, sample.tracking_quality, sample.motion_artifact
                    )

                if now - last_print >= self.print_every:
                    translation_result = self.estimator.estimate()
                    expansion_result = self.expansion_estimator.estimate()
                    last_result = fuse_channel_results(translation_result, expansion_result)
                    if last_result.rr_bpm is not None:
                        self._recent_rr.append(last_result.rr_bpm)
                        stable_rr = float(np.median(self._recent_rr))
                        print(
                            f"RR={stable_rr:5.2f} bpm  quality={last_result.quality:<4} "
                            f"confidence={last_result.confidence:.2f}  snr={last_result.snr_db:.1f} dB"
                        )
                    else:
                        print(
                            f"RR=--  quality={last_result.quality:<13} "
                            f"confidence={last_result.confidence:.2f}"
                        )
                    last_print = now

                if self.display:
                    shown = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    self._draw_overlay(shown, sample, last_result)
                    cv2.imshow("Respiration Monitor", shown)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        break
                    if key == ord("r"):
                        self.tracker.reset()
                        cfg = self.estimator.config
                        self.estimator = RespirationEstimator(cfg)
                        self.expansion_estimator = RespirationEstimator(cfg)
                        self._recent_rr.clear()
        finally:
            picam2.stop()
            if self.display:
                cv2.destroyAllWindows()

    def _lock_current_exposure(self, picam2) -> None:
        import time

        metadata = picam2.capture_metadata()
        controls = {"AeEnable": False}
        exposure = metadata.get("ExposureTime")
        gain = metadata.get("AnalogueGain")
        if exposure is not None:
            controls["ExposureTime"] = int(exposure)
        if gain is not None:
            controls["AnalogueGain"] = float(gain)
        try:
            picam2.set_controls(controls)
            time.sleep(0.2)
        except Exception as exc:
            print(f"Warning: could not lock exposure: {exc}")

    def _draw_overlay(self, frame: np.ndarray, sample: Optional[VisionSample], result: RespirationResult) -> None:
        import cv2

        if sample is not None:
            rx, ry, rw, rh = sample.reference_roi
            cx, cy, cw, ch = sample.chest_roi
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (0, 200, 255), 2)
            cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 2)
            cv2.putText(frame, "REFERENCE", (rx, max(18, ry - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
            cv2.putText(frame, "CHEST", (cx, max(18, cy - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(
                frame,
                f"tracks chest/ref: {sample.chest_tracks}/{sample.reference_tracks}",
                (12, self.height - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        else:
            cv2.putText(frame, "CALIBRATING...", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        rr_text = "RR: --" if result.rr_bpm is None else f"RR: {result.rr_bpm:.1f} bpm"
        cv2.putText(frame, rr_text, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(
            frame,
            f"{result.quality}  conf={result.confidence:.2f}",
            (12, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    monitor = PicameraRespirationMonitor(
        width=args.width,
        height=args.height,
        fps=args.fps,
        min_rr_bpm=args.min_rr,
        max_rr_bpm=args.max_rr,
        analysis_window_seconds=args.window,
        min_window_seconds=args.min_window,
        chest_roi=args.chest_roi,
        reference_roi=args.reference_roi,
        display=args.display,
        lock_exposure=args.lock_exposure,
        print_every=args.print_every,
    )
    monitor.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
