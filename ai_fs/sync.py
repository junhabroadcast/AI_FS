"""프레임 동기 검출 — FS의 본업.

각 프레임을 드리프트에 강인한 시그니처(정규화된 블록 평균 휘도)로 요약하고,
입력 스트림과 기준(하우스 레퍼런스) 스트림의 시그니처를 교차상관해
정수 프레임 오프셋을 추정한다. 색/밝기가 틀어져 있어도 동작해야 하므로
시그니처는 프레임마다 평균 0, 노름 1로 정규화한다.
"""

from __future__ import annotations

import numpy as np

from .color import rgb_to_ycc


def frame_signature(rgb: np.ndarray, grid: int = 8) -> np.ndarray:
    """(H,W,3) → 정규화된 grid*grid 휘도 시그니처."""
    y = rgb_to_ycc(rgb)[..., 0]
    h, w = y.shape
    ys = y[: h - h % grid, : w - w % grid]
    blocks = ys.reshape(grid, h // grid, grid, w // grid).mean(axis=(1, 3)).ravel()
    blocks -= blocks.mean()
    norm = np.linalg.norm(blocks)
    return blocks / norm if norm > 1e-8 else blocks


def estimate_offset(
    input_stream: np.ndarray,
    reference: np.ndarray,
    max_lag: int = 30,
) -> tuple[int, float]:
    """입력이 기준보다 몇 프레임 늦는지 추정. 반환: (offset, 신뢰도 0..1)."""
    sig_in = np.stack([frame_signature(f) for f in input_stream])
    sig_ref = np.stack([frame_signature(f) for f in reference])

    n = min(len(sig_in), len(sig_ref))
    best_lag, best_score = 0, -1.0
    scores = []
    for lag in range(0, max_lag + 1):
        # input[i] ≈ reference[i - lag]
        a = sig_in[lag:n]
        b = sig_ref[: n - lag]
        score = float((a * b).sum(axis=1).mean())
        scores.append(score)
        if score > best_score:
            best_score, best_lag = score, lag

    # 신뢰도: 최고 점수와 차순위의 간격
    ranked = sorted(scores, reverse=True)
    margin = ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0]
    confidence = float(np.clip(best_score * (0.5 + 5.0 * margin), 0.0, 1.0))
    return best_lag, confidence


def align(input_stream: np.ndarray, offset: int) -> np.ndarray:
    """프레임 버퍼 시뮬레이션: offset만큼 당겨 기준 타임라인에 정렬.

    실제 FS처럼 앞부분(버퍼가 채워지기 전)은 첫 유효 프레임을 유지한다.
    """
    n = len(input_stream)
    idx = np.clip(np.arange(n) + offset, 0, n - 1)
    return input_stream[idx]
