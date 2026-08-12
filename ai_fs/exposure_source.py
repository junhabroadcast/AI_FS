"""밝기 판정용 노출 시나리오 스트림 + DeckLink/파일 입력 어댑터 골격."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .drift import DriftParams, apply_drift
from .pattern import test_scene


def exposure_scene(width: int, height: int, frame_index: int, exposure: float) -> np.ndarray:
    """exposure: -1(매우 어두움) ~ 0(정상) ~ +1(매우 밝음)."""
    base = test_scene(width, height, frame_index)
    # Y gain/offset으로 노출 재현
    gain = 1.0 + 0.85 * exposure
    offset = 0.12 * exposure if exposure > 0 else 0.18 * exposure
    return apply_drift(base, DriftParams(y_gain=max(gain, 0.05), y_offset=offset))


def exposure_stream(
    width: int,
    height: int,
    frames_per_level: int = 20,
) -> tuple[np.ndarray, list[str]]:
    """DARK → UNDER → NORMAL → BRIGHT → OVER → BLACK 구간 스트림.

    반환: (N,H,W,3), 프레임별 정답 라벨 문자열
    """
    schedule = [
        ("DARK", -0.55),
        ("UNDER", -0.85),
        ("NORMAL", 0.0),
        ("BRIGHT", 0.28),
        ("OVER", 0.95),
        ("BLACK", None),
    ]
    frames = []
    labels = []
    idx = 0
    for name, exp in schedule:
        for _ in range(frames_per_level):
            if name == "BLACK":
                frames.append(np.zeros((height, width, 3), dtype=np.float32))
            else:
                frames.append(exposure_scene(width, height, idx, float(exp)))
            labels.append(name)
            idx += 1
    return np.stack(frames), labels


def load_video_rgb(path: str | Path, max_frames: int | None = 300) -> np.ndarray:
    """파일/장치 입력을 float RGB [0,1] 스트림으로. DeckLink는 추후 동일 인터페이스로 연결."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"열 수 없음: {path}")
    out = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out.append(rgb)
        if max_frames is not None and len(out) >= max_frames:
            break
    cap.release()
    if not out:
        raise RuntimeError(f"프레임 없음: {path}")
    return np.stack(out)
