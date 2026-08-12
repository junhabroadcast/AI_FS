"""BT.601 / BT.709 색 행렬 — WFM 프로젝트의 ColorMatrix를 파이썬으로 포팅.

내부 표현: float32, Y' ∈ [0,1], Cb/Cr ∈ [-0.5, +0.5].
"""

from __future__ import annotations

import numpy as np

# WFM ColorMatrix.cpp 와 동일한 계수
COEFFS = {
    "BT601": (0.299, 0.587, 0.114),
    "BT709": (0.2126, 0.7152, 0.0722),
}


def resolve(colorimetry: str, height: int) -> str:
    """WFM과 동일한 Auto 규칙: 높이 >= 720 → BT.709, 그 외 BT.601."""
    if colorimetry != "Auto":
        return colorimetry
    return "BT709" if height >= 720 else "BT601"


def rgb_to_ycc_matrix(colorimetry: str) -> np.ndarray:
    kr, kg, kb = COEFFS[colorimetry]
    return np.array(
        [
            [kr, kg, kb],
            [-kr / (2 * (1 - kb)), -kg / (2 * (1 - kb)), (1 - kb) / (2 * (1 - kb))],
            [(1 - kr) / (2 * (1 - kr)), -kg / (2 * (1 - kr)), -kb / (2 * (1 - kr))],
        ],
        dtype=np.float32,
    )


def rgb_to_ycc(rgb: np.ndarray, colorimetry: str = "BT709") -> np.ndarray:
    """rgb (H,W,3) [0,1] → ycc (H,W,3) with Y [0,1], Cb/Cr [-0.5,0.5]."""
    m = rgb_to_ycc_matrix(colorimetry)
    return rgb.astype(np.float32) @ m.T


def ycc_to_rgb(ycc: np.ndarray, colorimetry: str = "BT709") -> np.ndarray:
    m = rgb_to_ycc_matrix(colorimetry)
    inv = np.linalg.inv(m).astype(np.float32)
    return ycc.astype(np.float32) @ inv.T


# 75%/100% 컬러바 6색 (WFM kBarColors와 동일 순서: R, Mg, B, Cy, G, Yl)
BAR_COLORS = {
    "R": (1.0, 0.0, 0.0),
    "Mg": (1.0, 0.0, 1.0),
    "B": (0.0, 0.0, 1.0),
    "Cy": (0.0, 1.0, 1.0),
    "G": (0.0, 1.0, 0.0),
    "Yl": (1.0, 1.0, 0.0),
}


def vector_targets(bars75: bool = True, colorimetry: str = "BT709") -> dict[str, tuple[float, float]]:
    """벡터스코프 타깃 (Cb, Cr) — WFM vectorTargets 포팅."""
    amp = 0.75 if bars75 else 1.0
    m = rgb_to_ycc_matrix(colorimetry)
    out = {}
    for name, (r, g, b) in BAR_COLORS.items():
        ycc = m @ np.array([r * amp, g * amp, b * amp], dtype=np.float32)
        out[name] = (float(ycc[1]), float(ycc[2]))
    return out
