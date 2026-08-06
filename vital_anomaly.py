"""
환자별 개인화 생체신호 이상 탐지 (Isolation Forest)

3개 레이어로 구성:
  1) 절대 안전범위  - 개인 기준선과 무관. 캘리브레이션 전에도, 신뢰도가 낮아도 동작
  2) Isolation Forest - 패턴 불안정 탐지 (특징 조합의 이례성)
     + robust z 수준검정 - HR/RR 수준 이탈. IF 는 점수가 포화되어 수준
       변화를 잡지 못하므로 반드시 병행해야 한다
  3) 신호 소실      - 측정 자체가 끊긴 상태

레이어 1이 필요한 이유: 캘리브레이션 중 환자가 이미 이상 상태였다면 IF 는 그
상태를 정상으로 학습한다. 개인화는 절대적 위험 수치 개념을 갖지 못한다.
"""

import time
from collections import deque

import numpy as np
from sklearn.ensemble import IsolationForest


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

    def __init__(self, calib_sec=180.0, min_samples=40, window=5, trigger=3,
                 out_pct=1.0, min_conf=0.40, signal_lost_sec=45.0,
                 hr_sigma=2.5, rr_sigma=4.0):
        self._kw = dict(calib_sec=calib_sec, min_samples=min_samples,
                        window=window, trigger=trigger, out_pct=out_pct,
                        min_conf=min_conf, signal_lost_sec=signal_lost_sec,
                        hr_sigma=hr_sigma, rr_sigma=rr_sigma)
        self.hr_sigma = hr_sigma
        self.rr_sigma = rr_sigma

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

        self.hr_med = self.rr_med = 0.0
        self.hr_mad = self.rr_mad = 1.0

        self.recent = deque(maxlen=self.STD_N)      # 최근 채택 (hr, rr)
        self.hist = deque(maxlen=window)            # 최근 IF 이상 여부
        self.crit_hist = deque(maxlen=self.CRIT_WINDOW)
        self.alerting = False
        self.alert_reason = None     # 알림 사유. reason(기각 원인)과 별도로 유지
        self.critical = None
        self.last_valid = None       # None = 유효 샘플을 아직 한 번도 못 받음

        self._now = 0.0
        self.reject = {"hr_conf": 0, "rr_conf": 0,
                       "hr_range": 0, "rr_range": 0}

    # ── 공개 API ──────────────────────────────────────

    def push(self, hr, hr_conf, rr, rr_conf, now=None):
        """2초마다 1회 호출. 상태 dict 반환."""
        self._now = time.time() if now is None else now
        if self.calib_start is None:
            self.calib_start = self._now

        # 레이어 1: 신뢰도·캘리브레이션과 무관하게 항상 검사
        self.crit_hist.append(self._critical_now(hr, rr))
        hits = [c for c in self.crit_hist if c]
        self.critical = hits[-1] if len(hits) >= self.CRIT_TRIGGER else None

        # 레이어 3 / 품질 게이팅
        why = self._reject_reason(hr, hr_conf, rr, rr_conf)
        if why is not None:
            self.reject[why] += 1
            # last_valid is None = 초기 획득 중. 아직 '소실' 이 아니다.
            if (self.last_valid is not None
                    and self._now - self.last_valid > self.signal_lost_sec):
                return self._status("signal_lost", reason=why)
            return self._status("invalid", reason=why)

        self.last_valid = self._now
        self.recent.append((hr, rr))
        if len(self.recent) < self.MIN_STD_N:
            return self._status("warmup")

        # 레이어 2
        feat = self._features(hr, rr)
        if self.model is None:
            self.calib_feats.append(feat)
            if (self._now - self.calib_start >= self.calib_sec
                    and len(self.calib_feats) >= self.min_samples):
                self._fit()
                # 학습 직후에도 판정까지 진행한다. "calibrating" 을 반환하면
                # baseline 은 채워졌는데 score 는 None 인 모순 상태가 생긴다.
                return self._detect(feat, hr, rr)
            return self._status("calibrating")

        return self._detect(feat, hr, rr)

    def stats(self):
        return dict(self.reject, accepted=len(self.calib_feats))

    def reset(self):
        """환자 교체 또는 기준선 재학습. 자동 재학습은 의도적으로 넣지 않았다 —
        서서히 악화되는 환자를 정상으로 흡수해버리기 때문."""
        self.__init__(**self._kw)

    # ── 레이어 1 ──────────────────────────────────────

    def _critical_now(self, hr, rr):
        """측정값이 물리적으로 가능한 범위일 때만 절대범위를 판정."""
        out = []
        if self.HR_RANGE[0] <= hr <= self.HR_RANGE[1]:
            if not self.HR_CRITICAL[0] <= hr <= self.HR_CRITICAL[1]:
                out.append(f"HR {hr:.0f}")
        if self.RR_RANGE[0] <= rr <= self.RR_RANGE[1]:
            if not self.RR_CRITICAL[0] <= rr <= self.RR_CRITICAL[1]:
                out.append(f"RR {rr:.0f}")
        return " + ".join(out) if out else None

    # ── 레이어 2 ──────────────────────────────────────

    def _reject_reason(self, hr, hr_conf, rr, rr_conf):
        if hr_conf < self.min_conf:
            return "hr_conf"
        if rr_conf < self.min_conf:
            return "rr_conf"
        if not self.HR_RANGE[0] <= hr <= self.HR_RANGE[1]:
            return "hr_range"
        if not self.RR_RANGE[0] <= rr <= self.RR_RANGE[1]:
            return "rr_range"
        return None

    def _features(self, hr, rr):
        """
        [HR, RR, HR 이동표준편차, RR 이동표준편차, HR/RR]

        단일 샘플 차분 대신 이동 표준편차를 쓴다. 차분은 노이즈에 지배되고
        연속 두 샘플을 요구해 간헐적 측정 실패 시 샘플이 누적되지 않았다.
        HR/RR 비는 축 정렬 분할만 하는 IF 가 표현할 수 없는 조합이라 별도로 준다.
        """
        arr = np.asarray(self.recent, dtype=np.float64)
        return [hr, rr, float(arr[:, 0].std()), float(arr[:, 1].std()), hr / rr]

    def _augment(self, X):
        """
        측정 불확실성을 학습 분포에 반영한다.

        RR 은 60초 창을 2초씩 밀며 추정하므로 연속 샘플이 97% 겹친다. 그 결과
        캘리브레이션 구간의 관측 분산이 실제 불확실성보다 훨씬 작게 나온다
        (실측 로그에서 30샘플 연속 var=0.0). 그대로 학습하면 정상 범위 안의
        완만한 변동조차 이상으로 판정된다. 180초 캘리브레이션의 실질 독립
        관측 수는 RR 이 약 3개, HR 이 약 18개에 불과하다.

        그래서 알려진 측정 오차만큼 복제본을 만들어 분포를 넓힌다. 정보를
        추가하는 것이 아니라 "이 방향은 불확실하다" 를 모델에 알리는 것이다.
        """
        rng = np.random.RandomState(0)
        out = [X]
        n = len(X)

        def jitter(sigma):
            # 노이즈를 +-2sigma 로 절단한다. 절단하지 않으면 4sigma 급 복제본이
            # 생겨 임계값이 그 극단점 기준으로 잡히고, RR 불확실성을 키우는 것이
            # HR 민감도까지 함께 떨어뜨린다 (실측으로 확인된 결합 문제).
            # 절단하면 증강 구름이 "측정 불확실성 봉투" 가 되고 임계가 그 경계에
            # 놓이므로, 정상 판정 범위가 대략 기준선 +- 2sigma 로 해석된다.
            return np.clip(rng.normal(0, sigma, n), -2.0 * sigma, 2.0 * sigma)

        for _ in range(self.AUG_K):
            Y = X.copy()
            Y[:, 0] = Y[:, 0] + jitter(self.hr_sigma)
            Y[:, 1] = Y[:, 1] + jitter(self.rr_sigma)
            # 이동 표준편차의 불확실성: 오차 전파로 sigma / sqrt(2(N-1))
            sd = np.sqrt(2.0 * (self.STD_N - 1))
            Y[:, 2] = np.maximum(Y[:, 2] + jitter(self.hr_sigma / sd), 0.0)
            Y[:, 3] = np.maximum(Y[:, 3] + jitter(self.rr_sigma / sd), 0.0)
            # 비율은 교란된 값으로 재계산 (원래 값을 두면 물리적으로 모순된 점이 됨)
            Y[:, 4] = Y[:, 0] / Y[:, 1]
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

        self.hr_med = float(np.median(X[:, 0]))
        self.rr_med = float(np.median(X[:, 1]))
        # 방향 판정용 MAD 도 측정 불확실성 아래로는 내려가지 않게 한다.
        # 그렇지 않으면 0.5 BPM 차이에도 "RR LOW" 라고 단정한다.
        # 정규분포에서 MAD ~= 0.6745 * sigma.
        self.hr_mad = max(float(np.median(np.abs(X[:, 0] - self.hr_med))),
                          0.6745 * self.hr_sigma)
        self.rr_mad = max(float(np.median(np.abs(X[:, 1] - self.rr_med))),
                          0.6745 * self.rr_sigma)

    def _levels(self):
        """
        (hr_중앙값, rr_중앙값, z_hr, z_rr)

        수준 이탈은 단일 샘플이 아니라 최근 채택 샘플의 중앙값으로 판정한다.
        순간값으로 보면 측정 잡음이 봉투 경계를 스치기만 해도 알림이 뜬다
        (실측: 기준선 +4 BPM 유지 시 20샘플 중 14회 오탐).
        MAD 하한 덕분에 측정 불확실성보다 작은 차이는 큰 z 를 만들지 못한다.
        """
        arr = np.asarray(self.recent, dtype=np.float64)
        hr_m = float(np.median(arr[:, 0]))
        rr_m = float(np.median(arr[:, 1]))
        return (hr_m, rr_m,
                (hr_m - self.hr_med) / self.hr_mad,
                (rr_m - self.rr_med) / self.rr_mad)

    def _detect(self, feat, hr, rr):
        score = float(self.model.score_samples([feat])[0])

        # Isolation Forest 단독으로는 수준 이탈을 잡지 못한다. 5개 특징 중
        # 판별 축이 분할에 선택되는 비율이 낮아 점수가 포화되기 때문이다
        # (실측: HR 이 기준선 +38 BPM 이어도 임계를 넘지 못함).
        # 따라서 IF 는 패턴 불안정 탐지에, 수준 이탈은 robust z 검정에 맡긴다.
        hr_m, rr_m, z_hr, z_rr = self._levels()
        outlier = (score < self.threshold
                   or abs(z_hr) >= self.LEVEL_Z
                   or abs(z_rr) >= self.LEVEL_Z)
        self.hist.append(outlier)

        n_out = sum(self.hist)
        if not self.alerting and n_out >= self.trigger:
            self.alerting = True
        elif self.alerting and n_out == 0:      # 완전 정상일 때만 해제 (히스테리시스)
            self.alerting = False

        self.alert_reason = self._reason(hr_m, rr_m) if self.alerting else None
        return self._status("anomaly" if self.hist[-1] else "ok",
                            score=score, reason=self.alert_reason)

    def _reason(self, hr, rr):
        """수준 중앙값 기준 방향 설명. cv2.putText 가 한글을 못 그리므로 ASCII."""
        parts = []
        for name, val, med, mad in (("HR", hr, self.hr_med, self.hr_mad),
                                    ("RR", rr, self.rr_med, self.rr_mad)):
            z = (val - med) / mad
            if abs(z) >= self.LEVEL_Z:
                parts.append(f"{name} {'HIGH' if z > 0 else 'LOW'} "
                             f"{val:.0f}/{med:.0f}")
        return " + ".join(parts) if parts else "UNSTABLE PATTERN"

    # ── 상태 ──────────────────────────────────────────

    def _progress(self):
        """(전체, 시간기준, 샘플기준). 두 조건이 AND 이므로 성분을 따로 노출한다 —
        합쳐서 보여주면 어느 쪽이 병목인지 알 수 없다."""
        if self.model is not None:
            return 1.0, 1.0, 1.0
        by_time = min((self._now - self.calib_start) / self.calib_sec, 1.0)
        by_count = min(len(self.calib_feats) / self.min_samples, 1.0)
        return max(0.0, min(by_time, by_count)), by_time, by_count

    def _status(self, state, score=None, reason=None):
        prog, p_time, p_count = self._progress()
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
            "baseline": (self.hr_med, self.rr_med) if self.model else None,
        }