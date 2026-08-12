"""QC 이상 감지 — 방송 QC에서 보는 대표 이벤트를 실시간 판정.

- BLACK  : 평균 휘도가 임계 이하
- FREEZE : 직전 프레임과의 차이가 임계 이하 (모션 정지)
- GAMUT  : RGB 합법 범위를 벗어난 픽셀 비율 초과 (클리핑 잔량으로 추정)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .color import rgb_to_ycc


@dataclass
class QcEvent:
    frame: int
    kind: str
    detail: str


class QcMonitor:
    def __init__(
        self,
        black_luma: float = 0.03,
        freeze_diff: float = 1e-4,
        gamut_frac: float = 0.02,
    ):
        self.black_luma = black_luma
        self.freeze_diff = freeze_diff
        self.gamut_frac = gamut_frac
        self._prev: np.ndarray | None = None
        self.events: list[QcEvent] = []

    def inspect(self, frame_index: int, rgb: np.ndarray) -> list[QcEvent]:
        found: list[QcEvent] = []
        y = rgb_to_ycc(rgb)[..., 0]

        mean_y = float(y.mean())
        if mean_y < self.black_luma:
            found.append(QcEvent(frame_index, "BLACK", f"mean Y={mean_y * 100:.1f}%"))

        if self._prev is not None:
            diff = float(np.abs(rgb - self._prev).mean())
            if diff < self.freeze_diff and mean_y >= self.black_luma:
                found.append(QcEvent(frame_index, "FREEZE", f"frame diff={diff:.2e}"))
        self._prev = rgb.copy()

        clipped = float(((rgb <= 0.0005) | (rgb >= 0.9995)).any(axis=-1).mean())
        # 화면 구성상 순백/순흑 요소가 있으므로 비율이 클 때만 경보
        if clipped > 0.35:
            found.append(QcEvent(frame_index, "GAMUT", f"clipped px={clipped * 100:.1f}%"))

        self.events.extend(found)
        return found
