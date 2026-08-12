"""드리프트 시뮬레이터 — 실제 장비/케이블/카메라에서 발생하는 열화를 재현.

FS 입력단에서 흔히 보는 문제를 YCbCr 도메인에서 모델링한다:
  Y'  = y_gain * Y + y_offset            (밝기 게인/블랙 레벨 이동)
  C'  = c_gain * R(hue) @ [Cb, Cr]       (채도 게인 + 색상(위상) 회전)
그리고 프레임 오프셋(비동기 소스), 이상 프레임(프리즈/블랙)을 주입한다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .color import rgb_to_ycc, ycc_to_rgb


@dataclass
class DriftParams:
    y_gain: float = 1.0
    y_offset: float = 0.0
    c_gain: float = 1.0
    hue_deg: float = 0.0

    def describe(self) -> str:
        return (
            f"Y gain={self.y_gain:+.3f}, black={self.y_offset * 100:+.1f}%, "
            f"chroma gain={self.c_gain:+.3f}, hue={self.hue_deg:+.1f}\u00b0"
        )


def apply_drift(rgb: np.ndarray, p: DriftParams, colorimetry: str = "BT709") -> np.ndarray:
    """RGB 프레임에 프로크앰프형 드리프트를 적용 (클리핑 포함)."""
    ycc = rgb_to_ycc(rgb, colorimetry)
    y = ycc[..., 0] * p.y_gain + p.y_offset
    th = np.deg2rad(p.hue_deg)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=np.float32)
    c = ycc[..., 1:] @ (p.c_gain * rot).T
    out = np.concatenate([y[..., None], c], axis=-1)
    return np.clip(ycc_to_rgb(out, colorimetry), 0.0, 1.0)


def drift_at(frame_index: int, num_frames: int, severity: float = 1.0) -> DriftParams:
    """시간에 따라 서서히 진행되는 드리프트 곡선 (장비 온도 드리프트 재현)."""
    t = frame_index / max(1, num_frames - 1)
    ramp = 0.3 + 0.7 * t  # 시작부터 어긋나 있고 점점 심해짐
    return DriftParams(
        y_gain=1.0 + 0.18 * severity * ramp,
        y_offset=0.05 * severity * ramp,
        c_gain=1.0 - 0.25 * severity * ramp,
        hue_deg=12.0 * severity * ramp,
    )


def make_degraded_stream(
    reference: np.ndarray,
    frame_offset: int = 7,
    severity: float = 1.0,
    freeze_at: tuple[int, int] | None = (40, 44),
    black_at: tuple[int, int] | None = (70, 72),
    colorimetry: str = "BT709",
) -> tuple[np.ndarray, list[DriftParams]]:
    """기준 스트림으로부터 '현장 입력' 스트림을 만든다.

    - frame_offset 프레임만큼 지연 (비동기 소스)
    - 프레임별 진행성 색/밝기 드리프트
    - freeze_at / black_at 구간에 이상 프레임 주입
    반환: (열화 스트림, 프레임별 실제 드리프트 파라미터 — 검증용 정답지)
    """
    n = len(reference)
    truth: list[DriftParams] = []
    frames = []
    for i in range(n):
        src = reference[max(0, i - frame_offset)]
        p = drift_at(i, n, severity)
        truth.append(p)
        frames.append(apply_drift(src, p, colorimetry))

    degraded = np.stack(frames)
    if freeze_at is not None:
        a, b = freeze_at
        degraded[a:b] = degraded[a]
    if black_at is not None:
        a, b = black_at
        degraded[a:b] = 0.0
    return degraded, truth
