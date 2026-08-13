"""기준 프레임 대비 밝기·색 판정.

사용자가 캡처한 골든 프레임의 Y/Cb/Cr 통계를 기준으로 두고,
실시간 프레임과의 델타로 밝아짐/어두워짐·색 틀어짐을 판정한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class BrightnessDelta(str, Enum):
    MATCH = "MATCH"  # 정상
    BRIGHTER = "BRIGHTER"  # 밝아짐
    MUCH_BRIGHTER = "MUCH_BRIGHTER"  # 많이 밝아짐
    DARKER = "DARKER"  # 어두워짐
    MUCH_DARKER = "MUCH_DARKER"  # 많이 어두워짐
    NO_REF = "NO_REF"  # 기준 없음


class ColorDelta(str, Enum):
    MATCH = "MATCH"
    SHIFT = "SHIFT"
    NO_REF = "NO_REF"


@dataclass
class RefFeatures:
    mean_y: float
    center_y: float
    p95_y: float
    mean_cb: float
    mean_cr: float


@dataclass
class ReferenceResult:
    has_reference: bool
    brightness: BrightnessDelta
    color: ColorDelta
    dy: float  # live - ref (Y, 0..1)
    d_cb: float
    d_cr: float
    dc: float  # hypot(dCb, dCr)
    live_mean_y: float
    ref_mean_y: float | None
    confidence: float

    def brightness_korean(self) -> str:
        return {
            BrightnessDelta.MATCH: "정상",
            BrightnessDelta.BRIGHTER: "밝아짐",
            BrightnessDelta.MUCH_BRIGHTER: "많이 밝아짐",
            BrightnessDelta.DARKER: "어두워짐",
            BrightnessDelta.MUCH_DARKER: "많이 어두워짐",
            BrightnessDelta.NO_REF: "기준 없음",
        }[self.brightness]

    def color_korean(self) -> str:
        return {
            ColorDelta.MATCH: "정상",
            ColorDelta.SHIFT: "틀어짐",
            ColorDelta.NO_REF: "기준 없음",
        }[self.color]

    def status_line(self) -> str:
        if not self.has_reference:
            return "기준을 먼저 캡처하세요"
        return (
            f"밝기: {self.brightness_korean()}(ΔY {self.dy * 100:+.1f}%)  |  "
            f"색: {self.color_korean()}(ΔC {self.dc:.3f})"
        )


def extract_ref_features_bgr(bgr: np.ndarray, max_width: int = 160) -> RefFeatures:
    """축소 BGR에서 Rec.709 근사 Y·Cb·Cr 평균 통계."""
    if bgr.ndim != 3 or bgr.shape[2] < 3:
        raise ValueError("BGR HxWx3 required")
    h, w = bgr.shape[:2]
    if w > max_width:
        nh = max(1, int(h * max_width / w))
        bgr = cv2.resize(bgr, (max_width, nh), interpolation=cv2.INTER_AREA)

    b = bgr[:, :, 0].astype(np.float32) * (1.0 / 255.0)
    g = bgr[:, :, 1].astype(np.float32) * (1.0 / 255.0)
    r = bgr[:, :, 2].astype(np.float32) * (1.0 / 255.0)

    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    cb = -0.1146 * r - 0.3854 * g + 0.5000 * b
    cr = 0.5000 * r - 0.4542 * g - 0.0458 * b

    hh, ww = y.shape
    y0, y1 = hh // 4, 3 * hh // 4
    x0, x1 = ww // 4, 3 * ww // 4
    flat = y.ravel()
    step = max(1, flat.size // 2048)
    sample = flat[::step]
    p95 = float(np.percentile(sample, 95))

    return RefFeatures(
        mean_y=float(y.mean()),
        center_y=float(y[y0:y1, x0:x1].mean()),
        p95_y=p95,
        mean_cb=float(cb.mean()),
        mean_cr=float(cr.mean()),
    )


def _classify_brightness(dy: float, soft: float, hard: float) -> BrightnessDelta:
    ady = abs(dy)
    if ady < soft:
        return BrightnessDelta.MATCH
    if dy >= hard:
        return BrightnessDelta.MUCH_BRIGHTER
    if dy <= -hard:
        return BrightnessDelta.MUCH_DARKER
    if dy > 0:
        return BrightnessDelta.BRIGHTER
    return BrightnessDelta.DARKER


def _classify_color(dc: float, soft: float) -> ColorDelta:
    if dc < soft:
        return ColorDelta.MATCH
    return ColorDelta.SHIFT


class ReferenceJudge:
    """기준 프레임 대비 실시간 밝기·색 판정기."""

    def __init__(
        self,
        alpha: float = 0.45,
        y_soft: float = 0.04,
        y_hard: float = 0.10,
        c_soft: float = 0.025,
        max_width: int = 160,
    ):
        self.alpha = alpha
        self.y_soft = y_soft
        self.y_hard = y_hard
        self.c_soft = c_soft
        self.max_width = max_width
        self._ref: RefFeatures | None = None
        self._ema_dy: float | None = None
        self._ema_dcb: float | None = None
        self._ema_dcr: float | None = None

    @property
    def has_reference(self) -> bool:
        return self._ref is not None

    @property
    def reference(self) -> RefFeatures | None:
        return self._ref

    def set_reference_bgr(self, bgr: np.ndarray) -> RefFeatures:
        self._ref = extract_ref_features_bgr(bgr, max_width=self.max_width)
        self._ema_dy = 0.0
        self._ema_dcb = 0.0
        self._ema_dcr = 0.0
        return self._ref

    def clear_reference(self) -> None:
        self._ref = None
        self._ema_dy = None
        self._ema_dcb = None
        self._ema_dcr = None

    def reset(self) -> None:
        """스트림 재시작 시 EMA만 리셋. 기준은 유지하지 않고 함께 지움."""
        self.clear_reference()

    def judge_bgr(self, bgr: np.ndarray) -> ReferenceResult:
        live = extract_ref_features_bgr(bgr, max_width=self.max_width)
        if self._ref is None:
            return ReferenceResult(
                has_reference=False,
                brightness=BrightnessDelta.NO_REF,
                color=ColorDelta.NO_REF,
                dy=0.0,
                d_cb=0.0,
                d_cr=0.0,
                dc=0.0,
                live_mean_y=live.mean_y,
                ref_mean_y=None,
                confidence=0.0,
            )

        raw_dy = live.mean_y - self._ref.mean_y
        raw_dcb = live.mean_cb - self._ref.mean_cb
        raw_dcr = live.mean_cr - self._ref.mean_cr

        a = self.alpha
        if self._ema_dy is None:
            self._ema_dy, self._ema_dcb, self._ema_dcr = raw_dy, raw_dcb, raw_dcr
        else:
            self._ema_dy = (1 - a) * self._ema_dy + a * raw_dy
            self._ema_dcb = (1 - a) * self._ema_dcb + a * raw_dcb
            self._ema_dcr = (1 - a) * self._ema_dcr + a * raw_dcr

        dy = float(self._ema_dy)
        d_cb = float(self._ema_dcb)
        d_cr = float(self._ema_dcr)
        dc = float(np.hypot(d_cb, d_cr))

        bright = _classify_brightness(dy, self.y_soft, self.y_hard)
        color = _classify_color(dc, self.c_soft)

        # 이탈이 클수록 신뢰도↑ (허용치 대비)
        conf_y = min(1.0, abs(dy) / max(self.y_soft, 1e-6))
        conf_c = min(1.0, dc / max(self.c_soft, 1e-6))
        conf = 0.55 + 0.45 * max(conf_y, conf_c) * 0.5 if bright == BrightnessDelta.MATCH and color == ColorDelta.MATCH else 0.55 + 0.4 * max(conf_y, conf_c)

        return ReferenceResult(
            has_reference=True,
            brightness=bright,
            color=color,
            dy=dy,
            d_cb=d_cb,
            d_cr=d_cr,
            dc=dc,
            live_mean_y=live.mean_y,
            ref_mean_y=self._ref.mean_y,
            confidence=float(min(1.0, conf)),
        )
