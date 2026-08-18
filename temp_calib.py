#!/usr/bin/env python3
"""Turn an MLX90640 skin reading into a tympanic-referenced estimate.

Three things stand between the sensor and a usable number, and they matter in
this order:

1. Frames where the ROI slips off the face read the wall instead of skin. Those
   are not cold faces, they are misses, and averaging them in is what produced
   the 2.7 C swing in the field log. They get rejected, not corrected.
2. The skin-to-core gap widens in a cold room and narrows in a warm one, so a
   constant offset is only right at the ambient it was measured in. The MLX
   frame contains its own ambient reference -- the background -- so the term
   costs no extra hardware.
3. What is left is a fixed offset, which is the only part a single-point
   calibration can give you.

Fit A and B against paired ear-thermometer readings before trusting the output:

    python3 temp_calib.py --fit pairs.csv     # skin,ambient,ear per row
"""

import numpy as np

# Literature-scale starting values, NOT a calibration. Replace via --fit.
OFFSET_C = 2.0          # skin -> tympanic at the reference ambient
AMBIENT_GAIN = 0.20     # extra offset per degree the room is below reference
AMBIENT_REF_C = 24.0    # ambient the offset was established at
MIN_SKIN_C = 30.0       # below this the ROI is not on skin
MEDIAN_N = 5            # rolling median over accepted samples


class SkinToCore:
    def __init__(self, offset=OFFSET_C, ambient_gain=AMBIENT_GAIN,
                 ambient_ref=AMBIENT_REF_C, min_skin=MIN_SKIN_C, median_n=MEDIAN_N):
        self.offset = offset
        self.ambient_gain = ambient_gain
        self.ambient_ref = ambient_ref
        self.min_skin = min_skin
        self.hist = []
        self.median_n = median_n
        self.ambient = None
        self.rejected = 0

    @staticmethod
    def ambient_from_frame(frame):
        """Room temperature from the cold end of the thermal frame."""
        if frame is None:
            return None
        a = np.asarray(frame, dtype=float)
        a = a[np.isfinite(a)]
        return float(np.percentile(a, 10)) if a.size else None

    def update(self, skin_c, frame=None):
        """Return (core_estimate, ambient) or (None, ambient) if rejected."""
        amb = self.ambient_from_frame(frame)
        if amb is not None:
            self.ambient = amb

        if skin_c is None or not np.isfinite(skin_c) or skin_c < self.min_skin:
            self.rejected += 1
            return None, self.ambient

        core = skin_c + self.offset
        if self.ambient is not None:
            core += self.ambient_gain * (self.ambient_ref - self.ambient)

        self.hist.append(core)
        del self.hist[:-self.median_n]
        return float(np.median(self.hist)), self.ambient


def fit(skin, ambient, ear):
    """Least squares for offset and ambient gain against ear readings.

    Falls back to a plain offset when the samples span too little ambient
    range to identify the gain -- fitting a slope to a 1 C spread produces a
    number that looks precise and generalises to nothing.
    """
    skin, ambient, ear = (np.asarray(v, float) for v in (skin, ambient, ear))
    resid = ear - skin
    span = float(ambient.max() - ambient.min())

    if span < 3.0:
        off = float(resid.mean())
        rms = float(np.sqrt(np.mean((resid - off) ** 2)))
        return dict(offset=off, ambient_gain=0.0, ambient_ref=float(ambient.mean()),
                    rms=rms, n=len(skin), ambient_span=span, gain_fitted=False)

    ref = float(ambient.mean())
    A = np.stack([np.ones_like(ambient), ref - ambient], axis=1)
    (off, gain), *_ = np.linalg.lstsq(A, resid, rcond=None)
    rms = float(np.sqrt(np.mean((resid - A @ [off, gain]) ** 2)))
    return dict(offset=float(off), ambient_gain=float(gain), ambient_ref=ref,
                rms=rms, n=len(skin), ambient_span=span, gain_fitted=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", required=True,
                    help="CSV with skin,ambient,ear per row (header optional)")
    args = ap.parse_args()

    rows = np.genfromtxt(args.fit, delimiter=",", skip_header=0)
    rows = rows[np.isfinite(rows).all(axis=1)]
    r = fit(rows[:, 0], rows[:, 1], rows[:, 2])

    print(f"n={r['n']}  ambient span {r['ambient_span']:.1f} C  "
          f"residual RMS {r['rms']:.2f} C")
    if not r["gain_fitted"]:
        print("  ambient span too small to fit a gain; offset only")
    print(f"\n--temp-offset {r['offset']:.2f} "
          f"--temp-ambient-gain {r['ambient_gain']:.3f} "
          f"--temp-ambient-ref {r['ambient_ref']:.1f}")
    print(f"\nresidual RMS is your measurement uncertainty; "
          f"set --temp-sigma to about {max(r['rms'], 0.1):.1f}")