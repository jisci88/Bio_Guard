"""
환자별 개인화 생체신호 이상 탐지 (Isolation Forest)

대상 신호: 심박수(HR) + 호흡수(RR) + 얼굴 온도(TEMP)

3개 레이어로 구성:
  1) 절대 안전범위  - 개인 기준선과 무관. 캘리브레이션 전에도, 신뢰도가 낮아도 동작
  2) Isolation Forest - 패턴 불안정 탐지 (특징 조합의 이례성)
     + robust z 수준검정 - HR/RR/TEMP 수준 이탈. IF 는 점수가 포화되어 수준
       변화를 잡지 못하므로 반드시 병행해야 한다
  3) 신호 소실      - 측정 자체가 끊긴 상태

레이어 1이 필요한 이유: 캘리브레이션 중 환자가 이미 이상 상태였다면 IF 는 그
상태를 정상으로 학습한다. 개인화는 절대적 위험 수치 개념을 갖지 못한다.

── 체온 축에 대하여 ────────────────────────────────────────────
MLX90640 이 재는 값은 심부체온이 아니라 "얼굴 피부 표면 온도" 다. 둘은
다르며, 피부 온도는 실내 온도·촬영 거리·각도·땀·머리카락 가림에 따라
1~2°C 씩 쉽게 움직인다. 그래서 이 파일에서 체온을 다루는 방식은:

  · 절대 안전범위(TEMP_CRITICAL)는 참고용 기본값일 뿐, 반드시 현장의
    실내 온도와 센서 설치 상태에서 재측정해 조정해야 한다. 세 신호 중
    절대 임계의 신뢰도가 가장 낮은 축이다.
  · 실질적인 탐지력은 개인 기준선 대비 상대 변화(robust z)에서 나온다.
    같은 사람이 같은 자리에서 재는 한, 환경 편향은 기준선에 함께
    흡수되므로 "이 환자의 평소보다 0.8°C 높다" 는 판정은 유효하다.
  · 열화상 센서가 없거나(--no-thermal) 초기화에 실패하면 use_temp=False
    로 생성해야 한다. 그러면 특징벡터에서 체온 축이 통째로 빠지고
    HR/RR 만으로 기존과 동일하게 동작한다.
"""

import json
import os
import time
from collections import deque

import numpy as np
from sklearn.ensemble import IsolationForest

import config
from config import get_logger

log = get_logger("anomaly")


class VitalAnomalyDetector:
    """
    Parameters
    ----------
    calib_sec       : 기준 학습 구간 길이 (초)
    min_samples     : 학습에 필요한 최소 채택 샘플 수 (미달 시 구간 자동 연장)
    window, trigger : 지속성 판정 - 최근 window 개 중 trigger 개 이상이면 ALERT
    out_pct         : 학습 점수 분포의 하위 몇 % 를 이상 경계로 볼지 (↑ 민감)
    min_conf        : 이 미만 SQI 는 측정 실패로 간주해 폐기
                      0.40 = 합성 백색잡음 통과율 1.7% 지점 (실측)
    hr_sigma        : HR 추정의 측정 불확실성 (BPM, 1시그마)
    rr_sigma        : RR 추정의 측정 불확실성 (BPM, 1시그마)
                      학습 분포를 이만큼 넓혀 겹치는 창의 과소분산을 보정한다.
                      정상 판정 범위는 대략 기준선 +- 2sigma 가 된다.
                      기본 4.0 은 실측 세션의 RR 표준편차(약 3.9)에서 정했다
    temp_sigma      : 얼굴 온도 추정의 측정 불확실성 (°C, 1시그마).
                      MLX90640 자체 노이즈(NETD)는 0.1°C 수준이지만, 실제
                      불확실성을 지배하는 것은 센서가 아니라 "얼굴 박스를
                      32x24 격자에 비율 매핑한 근사" 와 머리 움직임이다.
                      기본 0.4 는 z=3.0 에서 기준선 +-0.8°C 봉투에 해당한다.
                      오탐이 잦으면 키우고, 미열을 놓치면 줄인다.
    use_temp        : False 면 체온 축을 특징벡터·절대범위·수준검정에서
                      모두 제외한다. 열화상 센서가 없을 때 반드시 False.
    signal_lost_sec : 유효 샘플을 한 번이라도 받은 뒤, 이 시간 이상 끊기면
                      signal_lost. EVM 창(60초)보다 짧게 두면 초기 획득
                      구간을 소실로 오판하므로 주의
    """

    # 물리적으로 불가능한 값 = 측정 실패
    HR_RANGE = (40.0, 200.0)
    RR_RANGE = (6.0, 40.0)

    # ★ 절대 안전범위 ★ 개인 기준선을 무시하고 즉시 알린다.
    #   여기 값은 성인 안정 시 일반 기준일 뿐이다.
    #   실제 임상 적용 시 반드시 의료진이 환자별로 재설정해야 한다.
    HR_CRITICAL = (45.0, 130.0)
    RR_CRITICAL = (8.0, 30.0)

    # ── A-5: 체온 임계는 "어느 물리량이냐" 에 따라 다르다 ────────────
    # 예전에는 vital_monitor 가 보정 없는 피부 표면 온도를, vital_run 이
    # SkinToCore 로 +2.0C 보정한 심부 추정치를 **같은 임계값**에 넣고 있었다.
    # 이제 temp_scale 로 어느 쪽인지 명시하고 그에 맞는 상수를 고른다.
    #
    # skin : MLX90640 이 실제로 재는 값. 실내온도/거리/각도에 1~2C 씩 흔들린다.
    #        상한 37.5 는 열화상 발열 스크리닝에서 흔히 쓰는 피부온 임계이고,
    #        하한 34.0 은 저체온보다 "얼굴이 아닌 곳을 재고 있다" 는 신호다.
    # core : SkinToCore 보정 후의 심부 추정치. 성인 저체온/발열 통상 경계.
    #
    # ※ A-3: 예전 코드의 하한은 디버그용 3.0 이 그대로 남아 있어서 저체온과
    #    배경 오매핑 경보가 사실상 꺼져 있었다. 아래 값으로 복구했다.
    # ※ 둘 다 현장의 실내 온도와 센서 설치 상태에서 재측정해 조정할 것.
    TEMP_LIMITS = {
        # scale: (물리적 가능범위, 절대 안전범위)
        "skin": ((30.0, 42.0), (34.0, 37.5)),
        "core": ((32.0, 43.0), (35.0, 38.0)),
    }

    # 하위 호환용 기본값 (temp_scale 을 주지 않고 만든 경우)
    TEMP_RANGE = TEMP_LIMITS["core"][0]
    TEMP_CRITICAL = TEMP_LIMITS["core"][1]

    BASELINE_VERSION = 1

    AUG_K = 8         # 캘리브레이션 샘플당 생성할 불확실성 복제본 수
    STD_N = 10        # 이동 표준편차 창 (채택 샘플 개수, 2초 간격이면 약 20초)
    MIN_STD_N = 3     # 이 개수만 모이면 계산 시작. 창이 꽉 찰 때까지
                      # 기다리면 멀쩡한 통과 샘플을 버리게 된다.
                      # 초기 몇 샘플의 표준편차는 다소 거칠다 (허용 가능한 대가)
    LEVEL_Z = 3.0     # 수준 이탈 판정 임계 (robust z).
                      # MAD 하한이 0.6745*sigma 이므로 z=3.0 은 기준선 +-2sigma
                      # 봉투에 대응한다.
    CRIT_WINDOW = 3   # 절대범위 지속성 창
    CRIT_TRIGGER = 2

    def __init__(self, calib_sec=None, min_samples=None, window=None,
                 trigger=None, out_pct=None, min_conf=None,
                 signal_lost_sec=None, hr_sigma=None, rr_sigma=None,
                 temp_sigma=None, use_temp=True, temp_scale=None,
                 temp_range=None, temp_critical=None, patient_id=None):
        # D-2: 값을 주지 않으면 config.py 의 현장 튜닝값을 따른다.
        calib_sec = config.CALIB_SEC if calib_sec is None else calib_sec
        min_samples = config.MIN_SAMPLES if min_samples is None else min_samples
        window = config.ANOM_WINDOW if window is None else window
        trigger = config.ANOM_TRIGGER if trigger is None else trigger
        out_pct = config.OUT_PCT if out_pct is None else out_pct
        min_conf = config.MIN_CONF if min_conf is None else min_conf
        signal_lost_sec = (
            config.SIGNAL_LOST_SEC if signal_lost_sec is None else signal_lost_sec
        )
        hr_sigma = config.HR_SIGMA if hr_sigma is None else hr_sigma
        rr_sigma = config.RR_SIGMA if rr_sigma is None else rr_sigma
        temp_sigma = config.TEMP_SIGMA if temp_sigma is None else temp_sigma
        temp_scale = config.TEMP_SCALE if temp_scale is None else temp_scale

        self._kw = dict(calib_sec=calib_sec, min_samples=min_samples,
                        window=window, trigger=trigger, out_pct=out_pct,
                        min_conf=min_conf, signal_lost_sec=signal_lost_sec,
                        hr_sigma=hr_sigma, rr_sigma=rr_sigma,
                        temp_sigma=temp_sigma, use_temp=use_temp,
                        temp_scale=temp_scale, temp_range=temp_range,
                        temp_critical=temp_critical, patient_id=patient_id)
        self.hr_sigma = hr_sigma
        self.rr_sigma = rr_sigma
        self.temp_sigma = temp_sigma
        self.use_temp = bool(use_temp)
        self.patient_id = patient_id

        # ── A-5: 체온 임계를 물리량에 맞춰 고른다 ──
        scale = str(temp_scale).lower()
        if scale not in self.TEMP_LIMITS:
            log.warning("알 수 없는 temp_scale=%r -> 'core' 로 처리합니다.", temp_scale)
            scale = "core"
        self.temp_scale = scale
        default_range, default_critical = self.TEMP_LIMITS[scale]
        self.TEMP_RANGE = tuple(temp_range or default_range)
        self.TEMP_CRITICAL = tuple(temp_critical or default_critical)

        self.calib_sec = calib_sec
        self.min_samples = min_samples
        self.trigger = trigger
        self.out_pct = out_pct
        self.min_conf = min_conf
        self.signal_lost_sec = signal_lost_sec

        self.model = None
        self.threshold = None
        self.calib_feats = []
        self.calib_start = None

        self.hr_med = self.rr_med = self.temp_med = 0.0
        self.hr_mad = self.rr_mad = self.temp_mad = 1.0

        # 최근 채택 샘플 (hr, rr, temp). use_temp=False 면 temp 열은 0 으로 채워
        # 두고 특징벡터에서 제외한다 (열 구조를 고정해 두면 인덱싱이 단순해진다).
        self.recent = deque(maxlen=self.STD_N)
        self.hist = deque(maxlen=window)            # 최근 이상 여부
        self.crit_hist = deque(maxlen=self.CRIT_WINDOW)
        self.alerting = False
        self.alert_reason = None     # 알림 사유. reason(기각 원인)과 별도로 유지
        self.critical = None
        self.last_valid = None       # None = 유효 샘플을 아직 한 번도 못 받음

        self._now = 0.0
        self.reject = {"hr_conf": 0, "rr_conf": 0, "temp_conf": 0,
                       "hr_range": 0, "rr_range": 0, "temp_range": 0}

        self._build_feature_layout()

    # ── 특징벡터 구성 ─────────────────────────────────

    def _build_feature_layout(self):
        """
        use_temp 에 따라 특징벡터 길이가 달라지므로 인덱스를 여기서 확정한다.

        use_temp=True  : [HR, RR, TEMP, HRsd, RRsd, TEMPsd, HR/RR]   (7차원)
        use_temp=False : [HR, RR, HRsd, RRsd, HR/RR]                 (5차원)

        체온은 HR/RR 과 달리 비(ratio) 특징을 만들지 않는다. HR/RR 비는
        "빈맥인데 호흡은 그대로" 같은 생리적으로 의미 있는 조합이지만,
        HR/TEMP 같은 비는 단위가 섞인 무의미한 축이라 IF 에 잡음만 준다.
        체온이 기여하는 정보는 수준(TEMP)과 변동성(TEMPsd) 두 가지다.
        """
        if self.use_temp:
            self.I_HR, self.I_RR, self.I_TP = 0, 1, 2
            self.I_HRSD, self.I_RRSD, self.I_TPSD = 3, 4, 5
            self.I_RATIO = 6
            self.n_features = 7
            self._jitter_plan = [
                (self.I_HR, self.I_HRSD, self.hr_sigma),
                (self.I_RR, self.I_RRSD, self.rr_sigma),
                (self.I_TP, self.I_TPSD, self.temp_sigma),
            ]
        else:
            self.I_HR, self.I_RR = 0, 1
            self.I_TP = self.I_TPSD = None
            self.I_HRSD, self.I_RRSD = 2, 3
            self.I_RATIO = 4
            self.n_features = 5
            self._jitter_plan = [
                (self.I_HR, self.I_HRSD, self.hr_sigma),
                (self.I_RR, self.I_RRSD, self.rr_sigma),
            ]

    def _vital_specs(self):
        """수준검정·사유표시에 쓰는 (이름, 중앙값, MAD, 소수자리) 목록."""
        specs = [("HR", self.hr_med, self.hr_mad, 0),
                 ("RR", self.rr_med, self.rr_mad, 0)]
        if self.use_temp:
            specs.append(("TEMP", self.temp_med, self.temp_mad, 1))
        return specs

    # ── 공개 API ──────────────────────────────────────

    def push(self, hr, hr_conf, rr, rr_conf,
             temp=None, temp_conf=None, now=None, sample_fresh=True):
        """2초마다 1회 호출. 상태 dict 반환.

        temp / temp_conf 는 use_temp=False 일 때 무시된다. use_temp=True 인데
        temp_conf 가 낮으면(얼굴 미검출, 열화상 프레임 없음) 그 샘플은 통째로
        기각된다 - 세 축을 함께 학습한 모델이라 한 축만 빠진 벡터는 넣을 수 없다.
        """
        self._now = time.time() if now is None else now
        if self.calib_start is None:
            self.calib_start = self._now

        temp = 0.0 if temp is None else float(temp)
        temp_conf = 0.0 if temp_conf is None else float(temp_conf)

        # 레이어 1: 신뢰도·캘리브레이션과 무관하게 항상 검사
        self.crit_hist.append(self._critical_now(hr, rr, temp, temp_conf))
        hits = [c for c in self.crit_hist if c]
        self.critical = hits[-1] if len(hits) >= self.CRIT_TRIGGER else None

        # 레이어 3 / 품질 게이팅
        why = self._reject_reason(hr, hr_conf, rr, rr_conf, temp, temp_conf)
        if why is not None:
            self.reject[why] += 1
            # last_valid is None = 초기 획득 중. 아직 '소실' 이 아니다.
            if (self.last_valid is not None
                    and self._now - self.last_valid > self.signal_lost_sec):
                return self._status("signal_lost", reason=why)
            return self._status("invalid", reason=why)

        # A held RR may be displayed and assessed for critical ranges, but it
        # must not be treated as a new physiological observation.
        if not sample_fresh:
            return self._status("hold", reason="rr_hold")

        self.last_valid = self._now
        self.recent.append((hr, rr, temp))
        if len(self.recent) == 1:
            self.calib_start = self._now
        if len(self.recent) < self.MIN_STD_N:
            return self._status("warmup")

        # 레이어 2
        feat = self._features(hr, rr, temp)
        if self.model is None:
            self.calib_feats.append(feat)
            if (self._now - self.calib_start >= self.calib_sec
                    and len(self.calib_feats) >= self.min_samples):
                self._fit()
                # 학습 직후에도 판정까지 진행한다. "calibrating" 을 반환하면
                # baseline 은 채워졌는데 score 는 None 인 모순 상태가 생긴다.
                return self._detect(feat)
            return self._status("calibrating")

        return self._detect(feat)

    def stats(self):
        return dict(self.reject, accepted=len(self.calib_feats))

    def reset(self):
        """환자 교체 또는 기준선 재학습. 자동 재학습은 의도적으로 넣지 않았다 —
        서서히 악화되는 환자를 정상으로 흡수해버리기 때문."""
        self.__init__(**self._kw)

    # ── C-8: 기준선 저장 / 복원 ───────────────────────

    def baseline_text(self):
        if self.model is None:
            return "미학습"
        text = f"HR {self.hr_med:.0f} / RR {self.rr_med:.0f}"
        if self.use_temp:
            text += f" / TEMP {self.temp_med:.1f}({self.temp_scale})"
        return text

    @staticmethod
    def baseline_path(patient_id, directory=None):
        directory = config.BASELINE_DIR if directory is None else directory
        safe = "".join(
            c if (c.isalnum() or c in "-_") else "_" for c in str(patient_id)
        )
        return os.path.join(directory, f"baseline_{safe}.npz")

    def _baseline_meta(self):
        return {
            "version": self.BASELINE_VERSION,
            "patient_id": self.patient_id,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_features": self.n_features,
            "use_temp": self.use_temp,
            "temp_scale": self.temp_scale,
            "hr_sigma": self.hr_sigma,
            "rr_sigma": self.rr_sigma,
            "temp_sigma": self.temp_sigma,
            "out_pct": self.out_pct,
            "std_n": self.STD_N,
            "aug_k": self.AUG_K,
            "n_samples": len(self.calib_feats),
        }

    def save_baseline(self, path):
        """
        학습이 끝난 기준선을 파일로 저장한다. 매 실행마다 180초를 다시
        기다리지 않아도 된다.

        IsolationForest 객체를 pickle 하지 않고 **캘리브레이션 특징벡터**만
        저장한다. _augment 가 RandomState(0), IsolationForest 가
        random_state=0 이라 재학습이 완전히 결정적이기 때문이다. sklearn
        버전이 바뀌어도 깨지지 않고 파일도 작다.
        """
        if self.model is None:
            log.warning("아직 학습이 끝나지 않아 기준선을 저장하지 않습니다.")
            return False
        try:
            parent = os.path.dirname(os.path.abspath(path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            np.savez(
                path,
                feats=np.asarray(self.calib_feats, dtype=np.float64),
                meta=np.array(
                    json.dumps(self._baseline_meta(), ensure_ascii=False)
                ),
            )
        except Exception as exc:
            log.warning("기준선 저장 실패 (%s): %s", path, exc)
            return False
        log.info("기준선 저장: %s  (샘플 %d개)  %s",
                 path, len(self.calib_feats), self.baseline_text())
        return True

    def load_baseline(self, path):
        """저장된 기준선을 읽어 즉시 '학습 완료' 상태로 만든다."""
        if not path or not os.path.exists(path):
            log.info("기준선 파일이 없습니다 (%s). 새로 학습합니다.", path)
            return False
        try:
            data = np.load(path, allow_pickle=False)
            meta = json.loads(str(data["meta"]))
            feats = np.asarray(data["feats"], dtype=np.float64)
        except Exception as exc:
            log.warning("기준선 읽기 실패 (%s): %s", path, exc)
            return False

        # 특징벡터 레이아웃이 다르면 저장된 숫자들의 의미가 다르다. 섞으면 안 된다.
        mismatch = [
            f"{key}: 저장={meta.get(key)!r} 현재={current!r}"
            for key, current in (("n_features", self.n_features),
                                 ("use_temp", self.use_temp),
                                 ("temp_scale", self.temp_scale))
            if meta.get(key) != current
        ]
        if mismatch or feats.ndim != 2 or feats.shape[1] != self.n_features:
            log.warning(
                "기준선이 현재 설정과 호환되지 않습니다 -> 새로 학습합니다. %s",
                "; ".join(mismatch) or f"shape={feats.shape}",
            )
            return False

        self.calib_feats = [list(map(float, row)) for row in feats]
        # sigma / out_pct 는 '지금' 값으로 다시 계산된다. 저장 당시 값과
        # 달라도 되며, 오히려 그게 맞다.
        self._fit()
        log.info(
            "기준선 복원: %s  (환자=%s, 저장=%s, 샘플 %d개)  %s",
            path, meta.get("patient_id"), meta.get("saved_at"),
            len(self.calib_feats), self.baseline_text(),
        )
        return True

    # ── 레이어 1 ──────────────────────────────────────

    def _critical_now(self, hr, rr, temp, temp_conf):
        """측정값이 물리적으로 가능한 범위일 때만 절대범위를 판정."""
        out = []
        if self.HR_RANGE[0] <= hr <= self.HR_RANGE[1]:
            if not self.HR_CRITICAL[0] <= hr <= self.HR_CRITICAL[1]:
                out.append(f"HR {hr:.0f}")
        if self.RR_RANGE[0] <= rr <= self.RR_RANGE[1]:
            if not self.RR_CRITICAL[0] <= rr <= self.RR_CRITICAL[1]:
                out.append(f"RR {rr:.0f}")
        # 체온은 신뢰도까지 함께 본다. HR/RR 과 달리 "얼굴이 안 보이면 0.0" 같은
        # 값이 그대로 들어오는데, 0.0 은 TEMP_RANGE 밖이라 어차피 걸러지긴 하지만
        # 열화상 프레임이 낡았을 때의 오경보를 막기 위해 명시적으로 게이팅한다.
        if self.use_temp and temp_conf >= self.min_conf:
            if self.TEMP_RANGE[0] <= temp <= self.TEMP_RANGE[1]:
                if not self.TEMP_CRITICAL[0] <= temp <= self.TEMP_CRITICAL[1]:
                    out.append(f"TEMP {temp:.1f}")
        return " + ".join(out) if out else None

    # ── 레이어 2 ──────────────────────────────────────

    def _reject_reason(self, hr, hr_conf, rr, rr_conf, temp, temp_conf):
        if hr_conf < self.min_conf:
            return "hr_conf"
        if rr_conf < self.min_conf:
            return "rr_conf"
        if not self.HR_RANGE[0] <= hr <= self.HR_RANGE[1]:
            return "hr_range"
        if not self.RR_RANGE[0] <= rr <= self.RR_RANGE[1]:
            return "rr_range"
        if self.use_temp:
            if temp_conf < self.min_conf:
                return "temp_conf"
            if not self.TEMP_RANGE[0] <= temp <= self.TEMP_RANGE[1]:
                return "temp_range"
        return None

    def _features(self, hr, rr, temp):
        """
        use_temp=True  : [HR, RR, TEMP, HRsd, RRsd, TEMPsd, HR/RR]
        use_temp=False : [HR, RR, HRsd, RRsd, HR/RR]

        단일 샘플 차분 대신 이동 표준편차를 쓴다. 차분은 노이즈에 지배되고
        연속 두 샘플을 요구해 간헐적 측정 실패 시 샘플이 누적되지 않았다.
        HR/RR 비는 축 정렬 분할만 하는 IF 가 표현할 수 없는 조합이라 별도로 준다.

        TEMPsd 는 체온 자체보다 오히려 유용할 때가 있다. 얼굴이 흔들리거나
        센서 매핑이 어긋나기 시작하면 수준은 그대로인데 변동성만 튀기 때문에,
        "측정이 무너지고 있다" 를 수준 이탈보다 먼저 잡아낸다.
        """
        arr = np.asarray(self.recent, dtype=np.float64)
        hr_sd = float(arr[:, 0].std())
        rr_sd = float(arr[:, 1].std())
        if self.use_temp:
            return [hr, rr, temp, hr_sd, rr_sd, float(arr[:, 2].std()), hr / rr]
        return [hr, rr, hr_sd, rr_sd, hr / rr]

    def _augment(self, X):
        """
        측정 불확실성을 학습 분포에 반영한다.

        RR 은 60초 창을 2초씩 밀며 추정하므로 연속 샘플이 97% 겹친다. 그 결과
        캘리브레이션 구간의 관측 분산이 실제 불확실성보다 훨씬 작게 나온다
        (실측 로그에서 30샘플 연속 var=0.0). 그대로 학습하면 정상 범위 안의
        완만한 변동조차 이상으로 판정된다. 180초 캘리브레이션의 실질 독립
        관측 수는 RR 이 약 3개, HR 이 약 18개에 불과하다.

        체온은 이 문제가 더 심하다. 피부 온도는 분 단위로 거의 움직이지 않아
        180초 캘리브레이션 안에서 사실상 상수로 관측된다 (관측 표준편차가
        0.05°C 미만인 경우가 흔하다). 증강 없이 학습하면 0.2°C 변화만으로도
        이상 판정이 나온다. 반드시 temp_sigma 만큼 넓혀줘야 한다.

        그래서 알려진 측정 오차만큼 복제본을 만들어 분포를 넓힌다. 정보를
        추가하는 것이 아니라 "이 방향은 불확실하다" 를 모델에 알리는 것이다.
        """
        rng = np.random.RandomState(0)
        out = [X]
        n = len(X)
        # 이동 표준편차의 불확실성: 오차 전파로 sigma / sqrt(2(N-1))
        sd_scale = np.sqrt(2.0 * (self.STD_N - 1))

        def jitter(sigma):
            # 노이즈를 +-2sigma 로 절단한다. 절단하지 않으면 4sigma 급 복제본이
            # 생겨 임계값이 그 극단점 기준으로 잡히고, RR 불확실성을 키우는 것이
            # HR 민감도까지 함께 떨어뜨린다 (실측으로 확인된 결합 문제).
            # 절단하면 증강 구름이 "측정 불확실성 봉투" 가 되고 임계가 그 경계에
            # 놓이므로, 정상 판정 범위가 대략 기준선 +- 2sigma 로 해석된다.
            return np.clip(rng.normal(0, sigma, n), -2.0 * sigma, 2.0 * sigma)

        for _ in range(self.AUG_K):
            Y = X.copy()
            for i_level, i_std, sigma in self._jitter_plan:
                Y[:, i_level] = Y[:, i_level] + jitter(sigma)
                Y[:, i_std] = np.maximum(
                    Y[:, i_std] + jitter(sigma / sd_scale), 0.0)
            # 비율은 교란된 값으로 재계산 (원래 값을 두면 물리적으로 모순된 점이 됨)
            Y[:, self.I_RATIO] = Y[:, self.I_HR] / Y[:, self.I_RR]
            out.append(Y)
        return np.vstack(out)

    def _fit(self):
        X = np.asarray(self.calib_feats, dtype=np.float64)
        Xa = self._augment(X)

        self.model = IsolationForest(
            n_estimators=300,
            max_samples=min(256, len(Xa)),
            contamination="auto",
            random_state=0,
        ).fit(Xa)

        self.threshold = float(np.percentile(
            self.model.score_samples(Xa), self.out_pct))

        # 방향 판정용 MAD 도 측정 불확실성 아래로는 내려가지 않게 한다.
        # 그렇지 않으면 0.5 BPM 차이에도 "RR LOW" 라고 단정한다.
        # 정규분포에서 MAD ~= 0.6745 * sigma.
        def med_mad(col, sigma):
            med = float(np.median(X[:, col]))
            mad = max(float(np.median(np.abs(X[:, col] - med))), 0.6745 * sigma)
            return med, mad

        self.hr_med, self.hr_mad = med_mad(self.I_HR, self.hr_sigma)
        self.rr_med, self.rr_mad = med_mad(self.I_RR, self.rr_sigma)
        if self.use_temp:
            # 체온은 캘리브레이션 관측 MAD 가 0 에 가깝게 나오는 것이 정상이라
            # 사실상 항상 이 하한(0.6745*temp_sigma)이 채택된다. 의도된 동작이다.
            self.temp_med, self.temp_mad = med_mad(self.I_TP, self.temp_sigma)

    def _levels(self):
        """
        [(이름, 최근중앙값, z, 소수자리), ...]

        수준 이탈은 단일 샘플이 아니라 최근 채택 샘플의 중앙값으로 판정한다.
        순간값으로 보면 측정 잡음이 봉투 경계를 스치기만 해도 알림이 뜬다
        (실측: 기준선 +4 BPM 유지 시 20샘플 중 14회 오탐).
        MAD 하한 덕분에 측정 불확실성보다 작은 차이는 큰 z 를 만들지 못한다.
        """
        arr = np.asarray(self.recent, dtype=np.float64)
        levels = []
        for col, (name, med, mad, digits) in enumerate(self._vital_specs()):
            cur = float(np.median(arr[:, col]))
            levels.append((name, cur, (cur - med) / mad, digits))
        return levels

    def _detect(self, feat):
        score = float(self.model.score_samples([feat])[0])

        # Isolation Forest 단독으로는 수준 이탈을 잡지 못한다. 판별 축이 분할에
        # 선택되는 비율이 낮아 점수가 포화되기 때문이다 (실측: HR 이 기준선
        # +38 BPM 이어도 임계를 넘지 못함). 특징이 5개에서 7개로 늘면 축당 선택
        # 확률이 더 떨어지므로 이 경향은 체온 추가 후 오히려 강해진다.
        # 따라서 IF 는 패턴 불안정 탐지에, 수준 이탈은 robust z 검정에 맡긴다.
        levels = self._levels()
        outlier = (score < self.threshold
                   or any(abs(z) >= self.LEVEL_Z for _, _, z, _ in levels))
        self.hist.append(outlier)

        n_out = sum(self.hist)
        if not self.alerting and n_out >= self.trigger:
            self.alerting = True
        elif self.alerting and n_out == 0:      # 완전 정상일 때만 해제 (히스테리시스)
            self.alerting = False

        self.alert_reason = self._reason(levels) if self.alerting else None
        return self._status("anomaly" if self.hist[-1] else "ok",
                            score=score, reason=self.alert_reason)

    def _reason(self, levels):
        """수준 중앙값 기준 방향 설명. cv2.putText 가 한글을 못 그리므로 ASCII."""
        parts = []
        for (name, cur, z, digits), (_, med, _, _) in zip(levels, self._vital_specs()):
            if abs(z) >= self.LEVEL_Z:
                parts.append(f"{name} {'HIGH' if z > 0 else 'LOW'} "
                             f"{cur:.{digits}f}/{med:.{digits}f}")
        return " + ".join(parts) if parts else "UNSTABLE PATTERN"

    # ── 상태 ──────────────────────────────────────────

    def _progress(self):
        """(전체, 시간기준, 샘플기준). 두 조건이 AND 이므로 성분을 따로 노출한다 —
        합쳐서 보여주면 어느 쪽이 병목인지 알 수 없다."""
        if self.model is not None:
            return 1.0, 1.0, 1.0
        by_time = (1.0 if self.calib_sec <= 0.0 else
                   min((self._now - self.calib_start) / self.calib_sec, 1.0))
        by_count = min(len(self.calib_feats) / self.min_samples, 1.0)
        return max(0.0, min(by_time, by_count)), by_time, by_count

    def _status(self, state, score=None, reason=None):
        prog, p_time, p_count = self._progress()
        if self.model is None:
            baseline = None
        else:
            baseline = (self.hr_med, self.rr_med,
                        self.temp_med if self.use_temp else None)
        return {
            "state": state,       # invalid|warmup|calibrating|ok|anomaly|signal_lost
            "alert": self.alerting,
            "alert_reason": self.alert_reason,
            "critical": self.critical,
            "score": score,
            "threshold": self.threshold,
            "reason": reason,
            "progress": prog,
            "progress_time": p_time,
            "progress_count": p_count,
            "accepted": len(self.calib_feats),
            "baseline": baseline,      # (HR, RR, TEMP) - TEMP 는 미사용 시 None
            "use_temp": self.use_temp,
        }
