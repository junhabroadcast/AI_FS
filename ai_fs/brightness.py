"""밝기(노출) 판정 — SDI/파일/시뮬레이터 프레임 → 밝다/정상/어둡다.

2단 구조:
  1) 통계 엔진 — mean/median/p95 Y, 하이라이트·섀도우 클리핑, 중앙 ROI
  2) 경량 AI   — 동일 피처를 로지스틱 회귀로 점수화 (합성 데이터로 캘리브레이션)

DeckLink로 SDI를 받아 오면 프레임 RGB/Y만 넘기면 실시간 판정 가능.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from .color import rgb_to_ycc


class BrightnessLabel(str, Enum):
    DARK = "DARK"  # 어둡다
    NORMAL = "NORMAL"  # 정상
    BRIGHT = "BRIGHT"  # 밝다
    OVER = "OVER"  # 과다노출 (하이라이트 깨짐)
    UNDER = "UNDER"  # 과소노출 (섀도우 뭉개짐)
    BLACK = "BLACK"  # 무신호/블랙


@dataclass
class BrightnessFeatures:
    mean_y: float  # 전역 평균 Y [0,1]
    median_y: float
    p05_y: float
    p95_y: float
    center_y: float  # 중앙 50% ROI 평균
    highlight_frac: float  # Y > 0.98 비율
    shadow_frac: float  # Y < 0.02 비율
    skew: float  # 히스토그램 왜도 (양수=밝은 꼬리)

    def as_vector(self) -> np.ndarray:
        """AI 스코어용 고정 순서 피처 벡터."""
        return np.array(
            [
                self.mean_y,
                self.median_y,
                self.p95_y,
                self.center_y,
                self.highlight_frac,
                self.shadow_frac,
                self.skew,
            ],
            dtype=np.float64,
        )


@dataclass
class BrightnessResult:
    label: BrightnessLabel
    score: float  # -1(매우 어두움) ~ +1(매우 밝음), 0=정상
    mean_y_pct: float  # 평균 Y %
    center_y_pct: float
    p95_y_pct: float
    highlight_pct: float
    shadow_pct: float
    confidence: float  # 0..1
    features: BrightnessFeatures
    rule_label: BrightnessLabel
    ai_score: float

    def korean(self) -> str:
        return {
            BrightnessLabel.DARK: "어둡다",
            BrightnessLabel.NORMAL: "정상",
            BrightnessLabel.BRIGHT: "밝다",
            BrightnessLabel.OVER: "과다노출",
            BrightnessLabel.UNDER: "과소노출",
            BrightnessLabel.BLACK: "블랙/무신호",
        }[self.label]


def extract_features(rgb: np.ndarray, colorimetry: str = "BT709") -> BrightnessFeatures:
    y = rgb_to_ycc(rgb, colorimetry)[..., 0].astype(np.float64)
    return _features_from_y(y)


def extract_features_fast_bgr(bgr: np.ndarray, max_width: int = 160) -> BrightnessFeatures:
    """실시간용 대충 판정 — 작은 BGR에서 근사 Y만 보고 통계.

    풀프레임 float 변환·median/percentile 전체 정렬을 피해서 ~30fps를 노린다.
    """
    if bgr.ndim != 3 or bgr.shape[2] < 3:
        raise ValueError("BGR HxWx3 required")
    h, w = bgr.shape[:2]
    if w > max_width:
        nh = max(1, int(h * max_width / w))
        bgr = cv2.resize(bgr, (max_width, nh), interpolation=cv2.INTER_AREA)

    # Rec.709 근사 휘도 (uint8 → [0,1]). 색행렬 풀 변환보다 훨씬 싸다.
    y = (
        0.0722 * bgr[:, :, 0].astype(np.float32)
        + 0.7152 * bgr[:, :, 1].astype(np.float32)
        + 0.2126 * bgr[:, :, 2].astype(np.float32)
    ) * (1.0 / 255.0)
    return _features_from_y(y, rough=True)


def _features_from_y(y: np.ndarray, rough: bool = False) -> BrightnessFeatures:
    h, w = y.shape
    y0, y1 = h // 4, 3 * h // 4
    x0, x1 = w // 4, 3 * w // 4
    center = y[y0:y1, x0:x1]
    flat = y.ravel()
    mean_y = float(flat.mean())

    if rough:
        # 서브샘플로 분위수·중앙값 근사 (정렬 비용 축소)
        step = max(1, flat.size // 2048)
        sample = flat[::step]
        median_y = float(np.median(sample))
        p05, p95 = (float(v) for v in np.percentile(sample, [5, 95]))
        highlight = float((sample > 0.98).mean())
        shadow = float((sample < 0.02).mean())
        std = float(sample.std()) + 1e-8
        skew = float(((sample - mean_y) ** 3).mean() / (std**3))
    else:
        median_y = float(np.median(flat))
        p05, p95 = (float(v) for v in np.percentile(flat, [5, 95]))
        highlight = float((flat > 0.98).mean())
        shadow = float((flat < 0.02).mean())
        std = float(flat.std()) + 1e-8
        skew = float(((flat - mean_y) ** 3).mean() / (std**3))

    return BrightnessFeatures(
        mean_y=mean_y,
        median_y=median_y,
        p05_y=p05,
        p95_y=p95,
        center_y=float(center.mean()),
        highlight_frac=highlight,
        shadow_frac=shadow,
        skew=skew,
    )


def rule_classify(f: BrightnessFeatures) -> BrightnessLabel:
    """방송 QC식 규칙 판정 — 설명 가능한 1차 판정."""
    if f.mean_y < 0.03 and f.p95_y < 0.08:
        return BrightnessLabel.BLACK
    # 과다노출: 하이라이트 깨짐이 뚜렷하고 전체도 밝은 경우
    if f.highlight_frac > 0.20 or (f.highlight_frac > 0.10 and f.mean_y > 0.68):
        return BrightnessLabel.OVER
    if f.shadow_frac > 0.40 and f.mean_y < 0.18:
        return BrightnessLabel.UNDER
    if f.mean_y < 0.12 and f.p95_y < 0.25:
        return BrightnessLabel.UNDER
    if f.center_y > 0.58 or f.mean_y > 0.55:
        return BrightnessLabel.BRIGHT
    if f.center_y < 0.28 or f.mean_y < 0.25:
        return BrightnessLabel.DARK
    return BrightnessLabel.NORMAL


# 합성 노출 스윕으로 lstsq 캘리브레이션한 로지스틱 가중치
# 피처: [mean, median, p95, center, highlight, shadow, skew]
# 출력: exposure score ≈ tanh(w·x + b) ∈ (-1, 1)
_AI_WEIGHTS = np.array(
    [-7.1731, -1.0079, 0.3385, 10.9931, -1.192, -1.132, -0.6695],
    dtype=np.float64,
)
_AI_BIAS = -1.5148


def ai_exposure_score(f: BrightnessFeatures) -> float:
    """경량 AI 스코어: -1(어두움) ~ +1(밝음)."""
    z = float(_AI_WEIGHTS @ f.as_vector() + _AI_BIAS)
    return float(np.tanh(z))


def _label_from_score(score: float, f: BrightnessFeatures) -> BrightnessLabel:
    if f.mean_y < 0.03 and f.p95_y < 0.08:
        return BrightnessLabel.BLACK
    if f.highlight_frac > 0.22 or (score > 0.75 and f.highlight_frac > 0.12):
        return BrightnessLabel.OVER
    if score < -0.75 and (f.shadow_frac > 0.22 or f.mean_y < 0.12):
        return BrightnessLabel.UNDER
    if score > 0.30:
        return BrightnessLabel.BRIGHT
    if score < -0.30:
        return BrightnessLabel.DARK
    return BrightnessLabel.NORMAL


def _fuse(rule: BrightnessLabel, ai_score: float, f: BrightnessFeatures) -> tuple[BrightnessLabel, float, float]:
    """규칙 + AI 융합. 블랙/과다·과소는 규칙 우선, 일반 구간은 점수 중심."""
    if rule == BrightnessLabel.BLACK:
        return rule, -1.0, 0.99

    ai_label = _label_from_score(ai_score, f)

    if rule in (BrightnessLabel.OVER, BrightnessLabel.UNDER):
        score = 0.65 * ai_score + 0.35 * (0.9 if rule == BrightnessLabel.OVER else -0.9)
        return rule, float(np.clip(score, -1, 1)), 0.9

    # 규칙·AI 일치 → 높은 신뢰도
    if rule == ai_label:
        conf = 0.55 + 0.4 * min(1.0, abs(ai_score) / 0.55)
        return ai_label, ai_score, float(conf)

    # 불일치: 중앙대(정상 근처)에서는 규칙을 더 신뢰
    if abs(ai_score) < 0.35:
        return rule, 0.5 * ai_score, 0.6

    rule_score = {
        BrightnessLabel.BRIGHT: 0.45,
        BrightnessLabel.DARK: -0.45,
        BrightnessLabel.NORMAL: 0.0,
    }.get(rule, 0.0)
    blended = 0.7 * ai_score + 0.3 * rule_score
    return _label_from_score(blended, f), float(blended), 0.55


class BrightnessJudge:
    """프레임 단위 밝기 판정기. EMA로 라벨 깜빡임 억제 (실시간용)."""

    def __init__(self, alpha: float = 0.35, colorimetry: str = "BT709"):
        self.alpha = alpha
        self.colorimetry = colorimetry
        self._ema_score: float | None = None

    def reset(self) -> None:
        self._ema_score = None

    def judge(self, rgb: np.ndarray) -> BrightnessResult:
        f = extract_features(rgb, self.colorimetry)
        return self._result_from_features(f)

    def judge_bgr_fast(self, bgr: np.ndarray, max_width: int = 160) -> BrightnessResult:
        """실시간 경로 — 작은 BGR에서 대충 판정."""
        f = extract_features_fast_bgr(bgr, max_width=max_width)
        return self._result_from_features(f)

    def _result_from_features(self, f: BrightnessFeatures) -> BrightnessResult:
        rule = rule_classify(f)
        ai = ai_exposure_score(f)
        label, score, conf = _fuse(rule, ai, f)

        if self._ema_score is None:
            self._ema_score = score
        else:
            a = self.alpha
            self._ema_score = (1 - a) * self._ema_score + a * score

        smooth_label = _label_from_score(self._ema_score, f)
        # 블랙은 즉시 반영
        if rule == BrightnessLabel.BLACK:
            smooth_label = BrightnessLabel.BLACK

        return BrightnessResult(
            label=smooth_label,
            score=float(self._ema_score),
            mean_y_pct=f.mean_y * 100.0,
            center_y_pct=f.center_y * 100.0,
            p95_y_pct=f.p95_y * 100.0,
            highlight_pct=f.highlight_frac * 100.0,
            shadow_pct=f.shadow_frac * 100.0,
            confidence=conf,
            features=f,
            rule_label=rule,
            ai_score=ai,
        )
