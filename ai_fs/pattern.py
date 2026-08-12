"""테스트 신호 발생기 — 75% 컬러바 + 움직이는 장면(모션 확인용)."""

from __future__ import annotations

import numpy as np

from .color import BAR_COLORS

# 화면 상단 바 순서 (SMPTE 스타일: 백 → 노 → 시안 → 녹 → 마젠타 → 적 → 청)
_BAR_ORDER = [
    (1.0, 1.0, 1.0),
    BAR_COLORS["Yl"],
    BAR_COLORS["Cy"],
    BAR_COLORS["G"],
    BAR_COLORS["Mg"],
    BAR_COLORS["R"],
    BAR_COLORS["B"],
]


def color_bars(width: int, height: int, amp: float = 0.75) -> np.ndarray:
    """75% 컬러바 풀프레임 RGB (H,W,3) [0,1]."""
    frame = np.zeros((height, width, 3), dtype=np.float32)
    n = len(_BAR_ORDER)
    for i, rgb in enumerate(_BAR_ORDER):
        x0 = width * i // n
        x1 = width * (i + 1) // n
        frame[:, x0:x1] = np.array(rgb, dtype=np.float32) * amp
    return frame


def test_scene(width: int, height: int, frame_index: int) -> np.ndarray:
    """움직임이 있는 테스트 장면.

    상단 60%: 75% 컬러바 (교정 기준 통계 확보용)
    하단: 수평 램프 + 좌우로 움직이는 흰 사각형 (프레임 싱크 검출용 모션)
    """
    frame = color_bars(width, height, amp=0.75)

    split = int(height * 0.6)
    ramp = np.linspace(0.0, 1.0, width, dtype=np.float32)
    frame[split:] = ramp[None, :, None]

    # 움직이는 사각형 — 프레임마다 위치가 달라 시간 시그니처를 만든다
    box_w = max(8, width // 12)
    box_h = max(8, (height - split) // 2)
    span = width - box_w
    t = frame_index / 30.0
    x = int((0.5 + 0.5 * np.sin(2 * np.pi * 0.23 * t)) * span)
    y0 = split + (height - split - box_h) // 2
    frame[y0 : y0 + box_h, x : x + box_w] = 1.0
    return frame


def generate_stream(width: int, height: int, num_frames: int) -> np.ndarray:
    """(N,H,W,3) float32 RGB 스트림."""
    return np.stack([test_scene(width, height, i) for i in range(num_frames)])
