#!/usr/bin/env python3
"""Turn an MLX90640 skin reading into a tympanic-referenced estimate.

먼저 솔직하게: 비접촉 피부 온도로 고막 온도를 **측정**할 수는 없다. 추정만
가능하고, 제대로 보정해도 잔차 RMS 0.3~0.5 C 가 현실적인 바닥이다
(IEC 80601-2-59 / ISO-TR 13154 의 발열 스크리닝 장비도 같은 수준이다).
이 파일이 하는 일은 그 바닥에 최대한 가깝게 가는 것이지, 체온계를 대체하는
것이 아니다.

센서와 쓸 만한 숫자 사이에 있는 네 가지를, 영향이 큰 순서대로:

1. ROI 가 얼굴에서 벗어난 프레임은 피부가 아니라 벽을 잰다. 그건 차가운
   얼굴이 아니라 '실패'다. 평균에 섞으면 현장 로그의 2.7 C 흔들림이 나온다.
   보정하지 않고 기각한다.

2. 거리.  ★이번에 추가된 항★
   MLX90640 은 32x24 화소로 55x35도(BAB 렌즈)를 본다. 화소 하나가 약
   1.7x1.5도, 즉 거리 d 에서 약 0.03*d 를 덮는다:

       거리     화소 한 변    얼굴(16x22cm)이 덮는 화소 수
       0.5 m     1.5 cm        약 157
       1.0 m     3.0 cm        약  39
       1.5 m     4.5 cm        약  17
       2.0 m     6.0 cm        약  10

   멀어질수록 화소 하나에 얼굴과 배경이 함께 들어간다(충전율 저하). 그래서
   측정값은 실제보다 **낮게** 나오고, 그 손실은 얼굴-배경 대비에 비례한다.
   대기 흡수는 수 미터 안에서 무시할 수준이라, 거리 오차의 실체는 복사가
   아니라 이 기하학이다.

   보정은 거리를 직접 재지 않는다. 얼굴이 실제로 덮은 열화상 화소 수가
   충전율 그 자체이므로 그것을 쓴다:

       d_ratio   = sqrt(face_px_ref / face_px)      # 1.0 = 보정 거리
       dist_term = dist_gain * (skin - ambient) * (d_ratio - 1.0)

   보정 거리에서 정확히 0 이 되고, 멀어질수록 대비에 비례해 커진다.

3. 실온. 피부-심부 격차는 추운 방에서 벌어지고 더운 방에서 좁아진다. 고정
   오프셋은 그것을 측정한 실온에서만 맞다. MLX 프레임이 배경으로 자기 기준
   실온을 갖고 있으므로 추가 하드웨어가 들지 않는다.

4. 남는 것이 고정 오프셋이고, 1점 보정이 줄 수 있는 유일한 부분이다.

전체 모델:

    core = skin
         + offset
         + ambient_gain * (ambient_ref - ambient)
         + dist_gain   * (skin - ambient) * (d_ratio - 1.0)

★ 반드시 실측으로 맞춘 뒤에 믿을 것 ★

    python3 temp_calib.py --fit pairs.csv

    pairs.csv 는 한 줄에  skin,ambient,face_px,ear
    (구버전 3열 skin,ambient,ear 도 그대로 읽는다. 그 경우 거리 항은 0)

    측정 절차:
      - 같은 사람을 **여러 거리**(예: 0.6 / 1.0 / 1.5 m)에서 재고, 각 회마다
        곧바로 고막 체온계로 잰 값을 함께 적는다.
      - 가능하면 **실온도 여러 조건**(예: 20 C / 26 C)에서 반복한다.
      - skin / ambient / face_px 는 세션 CSV(--session-log)의
        temp_skin / temp_ambient / face_px 열에 그대로 찍힌다.
      - 스팬이 좁은 항은 자동으로 0 으로 남는다. 1 C 스팬에 기울기를 맞추면
        정밀해 보이는 숫자가 나오지만 아무 데도 일반화되지 않는다.

    python3 temp_calib.py --demo    # 하드웨어 없이 거리 항 크기 확인

더 올라가려면(하드웨어 필요): 화면 안에 온도가 알려진 흑체(blackbody)를 같이
두는 것이 스크리닝 장비의 표준이다(IEC 80601-2-59). 그러면 센서 드리프트와
실온 항이 통째로 사라지고 0.2 C 급으로 내려간다.
"""

import math

import numpy as np

# Literature-scale starting values, NOT a calibration. Replace via --fit.
OFFSET_C = 2.0          # skin -> tympanic at the reference ambient
AMBIENT_GAIN = 0.20     # extra offset per degree the room is below reference
AMBIENT_REF_C = 24.0    # ambient the offset was established at

# 거리 항. 기본 0.0 = "아직 보정 안 됨". 맞추지 않은 기울기를 켜 두는 것은
# 꺼 두는 것보다 나쁘다. --fit 이 값을 주면 그때 넣는다.
DIST_GAIN = 0.0
FACE_PX_REF = 40        # 보정할 때 얼굴이 덮던 열화상 화소 수 (55도 렌즈 1 m)

MIN_SKIN_C = 30.0       # below this the ROI is not on skin
MIN_FACE_PX = 8         # 이보다 적게 덮으면 화소 대부분이 배경이다 -> 기각
WARN_FACE_PX = 16       # 기각까지는 아니지만 신뢰도를 크게 깎는 구간
MEDIAN_N = 5            # rolling median over accepted samples
MAX_STEP_C = 0.7         # impossible change between two 2-second MLX samples

BASE_SIGMA_C = 0.30     # MLX NETD + ROI 매핑 근사의 바닥 불확실성


class SkinToCore:
    """
    피부 표면 온도 -> 고막 기준 심부 추정치.

    호환: update(skin_c, frame) 2인자 호출은 예전과 똑같이 동작한다.
    face_px 를 함께 주면 거리 보정과 충전율 게이트가 켜진다.
    매 호출 뒤 self.last 에 진단값이 들어간다.
    """

    def __init__(self, offset=OFFSET_C, ambient_gain=AMBIENT_GAIN,
                 ambient_ref=AMBIENT_REF_C, min_skin=MIN_SKIN_C,
                 median_n=MEDIAN_N, dist_gain=DIST_GAIN,
                 face_px_ref=FACE_PX_REF, min_face_px=MIN_FACE_PX,
                 warn_face_px=WARN_FACE_PX, max_step_c=MAX_STEP_C):
        self.offset = offset
        self.ambient_gain = ambient_gain
        self.ambient_ref = ambient_ref
        self.min_skin = min_skin
        self.dist_gain = dist_gain
        self.face_px_ref = face_px_ref
        self.min_face_px = min_face_px
        self.warn_face_px = warn_face_px
        self.max_step_c = max_step_c

        self.hist = []
        self.median_n = median_n
        self.ambient = None
        self.rejected = 0
        self.reject_counts = {
            "no_skin": 0, "cold_roi": 0, "too_far": 0, "jump": 0,
        }
        self.last = {}

    # ------------------------------------------------------------------

    @staticmethod
    def ambient_from_frame(frame):
        """Room temperature from the cold end of the thermal frame."""
        if frame is None:
            return None
        a = np.asarray(frame, dtype=float)
        a = a[np.isfinite(a)]
        return float(np.percentile(a, 10)) if a.size else None

    def distance_ratio(self, face_px):
        """
        보정 거리 대비 상대 거리.

        얼굴이 덮는 화소 수는 거리의 제곱에 반비례하므로 sqrt 를 취하면
        거리에 비례하는 양이 된다. 1.0 = 보정 거리, 2.0 = 그 두 배 거리.
        """
        if not face_px or not self.face_px_ref:
            return 1.0
        return math.sqrt(float(self.face_px_ref) / max(float(face_px), 1.0))

    def _uncertainty(self, ambient_term, dist_term):
        """
        이 추정치를 얼마나 믿을 수 있는지 (1시그마, C).

        vital_monitor 의 --temp-sigma 에 넣을 값이고, 보정 항이 클수록,
        최근 값이 흔들릴수록 커진다. 보정을 많이 할수록 그 보정 자체의
        불확실성도 같이 커진다는 점을 반영한다.
        """
        parts = [BASE_SIGMA_C,
                 0.30 * abs(ambient_term),
                 0.50 * abs(dist_term)]
        if len(self.hist) >= 3:
            parts.append(0.5 * float(np.ptp(self.hist)))
        return float(math.sqrt(sum(p * p for p in parts)))

    def _reject(self, reason, skin_c, face_px, d_ratio):
        self.rejected += 1
        self.reject_counts[reason] = self.reject_counts.get(reason, 0) + 1
        self.last = dict(ok=False, reason=reason, skin=skin_c,
                         ambient=self.ambient, face_px=face_px,
                         d_ratio=d_ratio, core=None, sigma=None,
                         offset_term=None, ambient_term=None, dist_term=None)
        return None, self.ambient

    # ------------------------------------------------------------------

    def update(self, skin_c, frame=None, face_px=None):
        """Return (core_estimate, ambient) or (None, ambient) if rejected."""
        amb = self.ambient_from_frame(frame)
        if amb is not None:
            self.ambient = amb

        d_ratio = self.distance_ratio(face_px)

        if skin_c is None or not np.isfinite(skin_c):
            return self._reject("no_skin", skin_c, face_px, d_ratio)
        if skin_c < self.min_skin:
            # 차가운 얼굴이 아니라 얼굴이 아닌 곳을 잰 것이다.
            return self._reject("cold_roi", skin_c, face_px, d_ratio)
        if face_px is not None and face_px < self.min_face_px:
            # 화소 대부분이 배경이라 어떤 보정으로도 복구할 수 없다.
            return self._reject("too_far", skin_c, face_px, d_ratio)

        ambient_term = 0.0
        if self.ambient is not None:
            ambient_term = self.ambient_gain * (self.ambient_ref - self.ambient)

        # 거리(충전율) 보정. 손실은 얼굴-배경 대비에 비례한다.
        dist_term = 0.0
        if self.dist_gain and face_px is not None:
            reference = (self.ambient if self.ambient is not None
                         else self.ambient_ref)
            contrast = skin_c - reference
            dist_term = self.dist_gain * contrast * (d_ratio - 1.0)

        core = skin_c + self.offset + ambient_term + dist_term

        if (self.hist and self.max_step_c is not None
                and abs(core - float(np.median(self.hist))) > self.max_step_c):
            # A physiological core temperature cannot jump by this amount in
            # one thermal update.  This is almost always a transient ROI/FOV
            # mismatch, so retain the prior accepted filter history.
            return self._reject("jump", skin_c, face_px, d_ratio)

        self.hist.append(core)
        del self.hist[:-self.median_n]
        median_core = float(np.median(self.hist))

        sigma = self._uncertainty(ambient_term, dist_term)
        if face_px is not None and face_px < self.warn_face_px:
            # 기각 문턱은 넘었지만 충전율이 낮다. 숫자는 주되 믿음은 깎는다.
            sigma *= 1.0 + (self.warn_face_px - face_px) / float(self.warn_face_px)

        self.last = dict(ok=True, reason="", skin=skin_c, ambient=self.ambient,
                         face_px=face_px, d_ratio=d_ratio, core=median_core,
                         sigma=sigma, offset_term=self.offset,
                         ambient_term=ambient_term, dist_term=dist_term)
        return median_core, self.ambient

    def summary(self):
        """한 줄 진단 문자열."""
        if not self.last:
            return "temp: (측정 전)"
        if not self.last["ok"]:
            return f"temp: REJECT({self.last['reason']}) rejected={self.rejected}"
        amb = self.last["ambient"]
        return (f"temp: skin={self.last['skin']:.2f} "
                f"amb={'--' if amb is None else format(amb, '.2f')} "
                f"px={self.last['face_px']} d={self.last['d_ratio']:.2f} "
                f"(+{self.last['offset_term']:.2f}"
                f"{self.last['ambient_term']:+.2f}"
                f"{self.last['dist_term']:+.2f})"
                f" -> {self.last['core']:.2f} +-{self.last['sigma']:.2f}")


# ══════════════════════════════════════════════════════
#  보정 (fit)
# ══════════════════════════════════════════════════════

def fit(skin, ambient, ear, face_px=None, face_px_ref=None):
    """Least squares for offset, ambient gain, and distance gain vs ear readings.

    각 항은 자기 스팬이 충분할 때만 맞춘다. 1 C 스팬에 기울기를 맞추면
    정밀해 보이는 숫자가 나오지만 아무 데도 일반화되지 않는다. 거리 항도
    마찬가지라, 얼굴 화소 수가 거의 일정하면 0 으로 남긴다.
    """
    skin, ambient, ear = (np.asarray(v, float) for v in (skin, ambient, ear))
    resid = ear - skin
    amb_span = float(ambient.max() - ambient.min())

    columns = [np.ones_like(skin)]
    names = ["offset"]

    ref = float(ambient.mean())
    if amb_span >= 3.0:
        columns.append(ref - ambient)
        names.append("ambient_gain")

    px_ref = float(face_px_ref) if face_px_ref else None
    d_span = 0.0
    if face_px is not None:
        face_px = np.asarray(face_px, float)
        if px_ref is None:
            px_ref = float(np.median(face_px))
        d_ratio = np.sqrt(px_ref / np.maximum(face_px, 1.0))
        d_span = float(d_ratio.max() - d_ratio.min())
        if d_span >= 0.35:          # 대략 1.35배 이상의 거리 변화
            columns.append((skin - ambient) * (d_ratio - 1.0))
            names.append("dist_gain")

    design = np.stack(columns, axis=1)
    coeffs, *_ = np.linalg.lstsq(design, resid, rcond=None)
    rms = float(np.sqrt(np.mean((resid - design @ coeffs) ** 2)))

    out = dict(offset=0.0, ambient_gain=0.0, dist_gain=0.0,
               ambient_ref=ref, face_px_ref=px_ref or FACE_PX_REF,
               rms=rms, n=len(skin), ambient_span=amb_span,
               distance_span=d_span,
               gain_fitted="ambient_gain" in names,
               dist_fitted="dist_gain" in names)
    for name, value in zip(names, coeffs):
        out[name] = float(value)
    return out


def _demo():
    """하드웨어 없이 거리 항의 크기를 확인한다."""
    print("거리에 따른 충전율 손실 (55도 렌즈, 얼굴 16x22cm, 대비 13 C 가정)")
    print(f"{'거리':>7} {'화소':>7} {'d_ratio':>9} {'dist_gain=0.12 일 때 보정':>26}")
    for distance in (0.5, 0.75, 1.0, 1.5, 2.0):
        pixel_cm = 3.0 * distance
        px = max(1.0, (16.0 / pixel_cm) * (22.0 / pixel_cm))
        ratio = math.sqrt(FACE_PX_REF / px)
        print(f"{distance:6.2f}m {px:7.0f} {ratio:9.2f} "
              f"{0.12 * 13.0 * (ratio - 1.0):+24.2f} C")
    print(f"\nFACE_PX_REF={FACE_PX_REF:.0f} (보정 거리)에서 정확히 0 이고, "
          "멀어질수록 커진다.")
    print("dist_gain 은 --fit 으로 정해야 하는 값이다. 위 0.12 는 예시일 뿐이다.")
    print(f"화소가 {MIN_FACE_PX} 개 미만이면 어떤 보정으로도 복구할 수 없어 기각한다.")


if __name__ == "__main__":
    import argparse
    import sys

    # 라즈베리파이 콘솔은 UTF-8 이지만 윈도우 cmd 는 cp949 라 한글이 깨진다.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(
        description="피부 표면 온도 -> 고막 기준 심부 추정 보정")
    ap.add_argument("--fit",
                    help="CSV: skin,ambient,face_px,ear "
                         "(구버전 skin,ambient,ear 도 가능)")
    ap.add_argument("--face-px-ref", type=float, default=None,
                    help="거리 기준이 될 얼굴 열화상 화소 수 (기본: 표본 중앙값)")
    ap.add_argument("--demo", action="store_true",
                    help="거리 항의 크기를 표로 보여준다 (하드웨어 불필요)")
    args = ap.parse_args()

    if args.demo:
        _demo()
        raise SystemExit
    if not args.fit:
        ap.error("--fit 또는 --demo 중 하나가 필요합니다")

    rows = np.atleast_2d(np.genfromtxt(args.fit, delimiter=",", skip_header=0))
    rows = rows[np.isfinite(rows).all(axis=1)]
    if rows.shape[0] < 3:
        ap.error(f"유효한 행이 {rows.shape[0]}개뿐입니다. 최소 3개가 필요합니다.")

    if rows.shape[1] >= 4:
        r = fit(rows[:, 0], rows[:, 1], rows[:, 3], face_px=rows[:, 2],
                face_px_ref=args.face_px_ref)
    elif rows.shape[1] == 3:
        print("[알림] 3열 파일입니다. 거리 항은 맞출 수 없습니다.")
        print("       세션 CSV 의 face_px 열을 3번째로 넣으면 거리 보정이 켜집니다.\n")
        r = fit(rows[:, 0], rows[:, 1], rows[:, 2])
    else:
        ap.error("열이 부족합니다: skin,ambient[,face_px],ear 형식이어야 합니다")

    print(f"n={r['n']}  실온 스팬 {r['ambient_span']:.1f} C  "
          f"거리 스팬 {r['distance_span']:.2f}  잔차 RMS {r['rms']:.2f} C")
    if not r["gain_fitted"]:
        print("  실온 스팬이 좁아 ambient_gain 은 맞추지 않았습니다 (오프셋만).")
    if not r["dist_fitted"]:
        print("  거리 스팬이 좁아 dist_gain 은 맞추지 않았습니다.")
        print("  여러 거리(예: 0.6 / 1.0 / 1.5 m)에서 다시 측정하세요.")

    print(f"\n--temp-offset {r['offset']:.2f} "
          f"--temp-ambient-gain {r['ambient_gain']:.3f} "
          f"--temp-ambient-ref {r['ambient_ref']:.1f} "
          f"--temp-dist-gain {r['dist_gain']:.4f} "
          f"--temp-face-px-ref {r['face_px_ref']:.0f}")
    print(f"\n잔차 RMS 가 곧 측정 불확실성입니다. "
          f"--temp-sigma 를 약 {max(r['rms'], 0.1):.1f} 로 두세요.")
    if r["rms"] > 0.6:
        print("경고: RMS 0.6 C 초과. 표본이 부족하거나 측정 조건이 흔들렸을 수")
        print("      있습니다. 이 상태로는 발열 판정을 신뢰하기 어렵습니다.")
