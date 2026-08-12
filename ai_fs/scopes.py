"""스코프 렌더러 — WFM의 CRT 녹색 톤 파형/벡터스코프를 numpy로 재현.

출력은 cv2 규약의 uint8 BGR 이미지.
"""

from __future__ import annotations

import cv2
import numpy as np

from .color import rgb_to_ycc, vector_targets

_GRID = (60, 70, 60)  # 계수선 색 (BGR)


def _tonemap_green(acc: np.ndarray) -> np.ndarray:
    """가산 누적 버퍼 → CRT 녹색 (WFM tonemap.frag 느낌)."""
    v = np.log1p(acc)
    v = v / max(float(v.max()), 1e-6)
    img = np.zeros((*acc.shape, 3), dtype=np.uint8)
    img[..., 1] = (v * 255).astype(np.uint8)          # G
    img[..., 0] = (v * 90).astype(np.uint8)           # B 살짝 → 형광 느낌
    img[..., 2] = (np.clip(v * 1.8 - 0.8, 0, 1) * 255).astype(np.uint8)  # 밝은 곳 흰색
    return img


def waveform(rgb: np.ndarray, size: tuple[int, int] = (512, 360)) -> np.ndarray:
    """휘도 파형 (x=화면 가로, y=레벨 0~100%)."""
    w, h = size
    y = rgb_to_ycc(rgb)[..., 0]
    src_h, src_w = y.shape

    cols = (np.arange(src_w) * w // src_w)
    cols = np.broadcast_to(cols, (src_h, src_w)).ravel()
    rows = np.clip(((1.0 - y) * (h - 1)).astype(np.int32), 0, h - 1).ravel()

    acc = np.zeros((h, w), dtype=np.float32)
    np.add.at(acc, (rows, cols), 1.0)
    img = _tonemap_green(acc)

    for pct in (0, 25, 50, 75, 100):
        yy = int((1.0 - pct / 100.0) * (h - 1))
        cv2.line(img, (0, yy), (w - 1, yy), _GRID, 1)
        cv2.putText(img, f"{pct}", (4, max(12, yy - 4)), cv2.FONT_HERSHEY_PLAIN, 0.9, _GRID, 1)
    return img


def vectorscope(rgb: np.ndarray, size: int = 360, colorimetry: str = "BT709") -> np.ndarray:
    """벡터스코프 (x=Cb, y=Cr, 위가 +Cr). 75% 타깃 박스 포함."""
    s = size
    ycc = rgb_to_ycc(rgb, colorimetry)
    cb = ycc[..., 1].ravel()
    cr = ycc[..., 2].ravel()

    # [-0.5, 0.5] → 픽셀 (스케일 0.9로 여유)
    scale = s * 0.9
    xs = np.clip((cb * scale + s / 2).astype(np.int32), 0, s - 1)
    ys = np.clip((-cr * scale + s / 2).astype(np.int32), 0, s - 1)

    acc = np.zeros((s, s), dtype=np.float32)
    np.add.at(acc, (ys, xs), 1.0)
    img = _tonemap_green(acc)

    c = s // 2
    cv2.line(img, (c, 0), (c, s - 1), _GRID, 1)
    cv2.line(img, (0, c), (s - 1, c), _GRID, 1)
    cv2.circle(img, (c, c), int(0.45 * s), _GRID, 1)

    for name, (tcb, tcr) in vector_targets(bars75=True, colorimetry=colorimetry).items():
        x = int(tcb * scale + s / 2)
        y = int(-tcr * scale + s / 2)
        half = max(3, s // 60)
        cv2.rectangle(img, (x - half, y - half), (x + half, y + half), (80, 200, 255), 1)
        cv2.putText(img, name, (x + half + 2, y + 4), cv2.FONT_HERSHEY_PLAIN, 0.9, (80, 200, 255), 1)
    return img


def _labeled(img: np.ndarray, label: str) -> np.ndarray:
    bar = np.zeros((26, img.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar, label, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    return np.vstack([bar, img])


def rgb_to_bgr8(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((np.clip(rgb, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def comparison_panel(
    frames: list[tuple[str, np.ndarray]],
    scope_h: int = 300,
    colorimetry: str = "BT709",
) -> np.ndarray:
    """[(라벨, RGB 프레임)] → 각 열 = 픽처 + 파형 + 벡터스코프."""
    columns = []
    for label, rgb in frames:
        pic_w = 420
        pic_h = int(pic_w * rgb.shape[0] / rgb.shape[1])
        pic = cv2.resize(rgb_to_bgr8(rgb), (pic_w, pic_h), interpolation=cv2.INTER_NEAREST)
        wfm = cv2.resize(waveform(rgb), (pic_w, scope_h))
        vec = cv2.resize(vectorscope(rgb, colorimetry=colorimetry), (pic_w, pic_w))
        col = np.vstack([
            _labeled(pic, label),
            _labeled(wfm, "Waveform (Y)"),
            _labeled(vec, "Vectorscope"),
        ])
        columns.append(col)

    gap = np.zeros((columns[0].shape[0], 12, 3), dtype=np.uint8)
    out = columns[0]
    for col in columns[1:]:
        out = np.hstack([out, gap, col])
    return out
