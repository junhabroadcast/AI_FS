"""AI 자동 색/밝기 교정 엔진.

정렬된 입력 프레임과 기준 프레임의 YCbCr 통계를 비교해 드리프트를 추정한다.
사람이 프로크앰프 노브를 돌리는 대신, 프레임마다 최소제곱으로 파라미터를
풀고 시간축으로 지수평활(EMA)해 안정적으로 추종한다.

  Y 채널:  y_in ≈ g·y_ref + o          → 1차 선형회귀
  C 채널:  (cb+j·cr)_in ≈ a·(cb+j·cr)_ref → 복소 최소제곱
           |a| = 채도 게인, arg(a) = 색상(hue) 회전각
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .color import rgb_to_ycc, ycc_to_rgb
from .drift import DriftParams


@dataclass
class Estimate:
    params: DriftParams
    valid: bool  # 추정에 쓸 수 있는 프레임이었는지 (블랙/프리즈 프레임 제외)


def estimate_frame_drift(
    input_rgb: np.ndarray,
    reference_rgb: np.ndarray,
    colorimetry: str = "BT709",
) -> Estimate:
    """한 프레임 쌍에서 드리프트 파라미터를 추정."""
    ycc_in = rgb_to_ycc(input_rgb, colorimetry)
    ycc_ref = rgb_to_ycc(reference_rgb, colorimetry)

    y_in, y_ref = ycc_in[..., 0].ravel(), ycc_ref[..., 0].ravel()

    # 클리핑된 픽셀(0/1 근처)은 선형 모델을 왜곡하므로 제외
    mask = (y_in > 0.02) & (y_in < 0.98)
    if mask.sum() < 500 or float(y_in.mean()) < 0.03:
        return Estimate(DriftParams(), valid=False)

    # Y: 최소제곱 선형회귀 — 2-pass 강인 추정.
    # 1차 적합 후 잔차가 큰 픽셀(내용 불일치: 프리즈 프레임, 움직이는 물체,
    # RGB 클리핑으로 비선형이 된 픽셀)을 제외하고 다시 적합한다.
    def fit_y(m: np.ndarray) -> tuple[float, float]:
        a = np.stack([y_ref[m], np.ones(int(m.sum()), dtype=np.float32)], axis=1)
        sol, *_ = np.linalg.lstsq(a, y_in[m], rcond=None)
        return float(sol[0]), float(sol[1])

    y_gain, y_offset = fit_y(mask)
    resid = np.abs(y_in - (y_gain * y_ref + y_offset))
    inlier = mask & (resid < max(0.01, 3.0 * float(resid[mask].std())))
    if inlier.sum() > 500:
        mask = inlier
        y_gain, y_offset = fit_y(mask)

    # C: 복소 최소제곱 (스케일+회전 동시 추정) — Y 인라이어만 사용
    c_in = (ycc_in[..., 1] + 1j * ycc_in[..., 2]).ravel()[mask]
    c_ref = (ycc_ref[..., 1] + 1j * ycc_ref[..., 2]).ravel()[mask]
    denom = float(np.vdot(c_ref, c_ref).real)
    if denom < 1e-6:
        return Estimate(DriftParams(float(y_gain), float(y_offset)), valid=True)
    coeff = complex(np.vdot(c_ref, c_in)) / denom

    return Estimate(
        DriftParams(
            y_gain=float(y_gain),
            y_offset=float(y_offset),
            c_gain=float(abs(coeff)),
            hue_deg=float(np.rad2deg(np.angle(coeff))),
        ),
        valid=True,
    )


def apply_correction(rgb: np.ndarray, p: DriftParams, colorimetry: str = "BT709") -> np.ndarray:
    """추정된 드리프트의 역변환을 적용해 신호를 복원."""
    ycc = rgb_to_ycc(rgb, colorimetry)
    y = (ycc[..., 0] - p.y_offset) / max(p.y_gain, 1e-6)
    th = np.deg2rad(-p.hue_deg)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=np.float32)
    c = ycc[..., 1:] @ (rot / max(p.c_gain, 1e-6)).T
    out = np.concatenate([y[..., None], c], axis=-1)
    return np.clip(ycc_to_rgb(out, colorimetry), 0.0, 1.0)


class AdaptiveCorrector:
    """프레임별 추정치를 EMA로 평활해 추종하는 실시간형 교정기.

    이상 프레임(블랙 등)에서는 추정을 건너뛰고 마지막 유효 파라미터를 유지
    — 실제 FS가 신호 단절 시 마지막 설정을 홀드하는 것과 같은 동작.
    """

    def __init__(self, alpha: float = 0.25, colorimetry: str = "BT709"):
        self.alpha = alpha
        self.colorimetry = colorimetry
        self.state: DriftParams | None = None

    def process(self, input_rgb: np.ndarray, reference_rgb: np.ndarray) -> tuple[np.ndarray, DriftParams]:
        est = estimate_frame_drift(input_rgb, reference_rgb, self.colorimetry)
        if est.valid:
            if self.state is None:
                self.state = est.params
            else:
                a = self.alpha
                s, p = self.state, est.params
                self.state = DriftParams(
                    y_gain=(1 - a) * s.y_gain + a * p.y_gain,
                    y_offset=(1 - a) * s.y_offset + a * p.y_offset,
                    c_gain=(1 - a) * s.c_gain + a * p.c_gain,
                    hue_deg=(1 - a) * s.hue_deg + a * p.hue_deg,
                )
        current = self.state if self.state is not None else DriftParams()
        return apply_correction(input_rgb, current, self.colorimetry), current
