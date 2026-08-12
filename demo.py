"""AI FS 엔드투엔드 데모.

시나리오:
  1. 하우스 레퍼런스(마스터) 스트림 생성
  2. '현장 입력' 생성 — 7프레임 지연 + 진행성 색/밝기 드리프트 + 프리즈/블랙 구간
  3. AI FS 처리 — 프레임 싱크 검출·정렬 → 프레임별 드리프트 추정·교정 → QC 감시
  4. 결과 검증 — PSNR, 파라미터 추종 오차, 스코프 비교 패널 저장

실행:  python demo.py
출력:  output/ 폴더에 비교 패널·추종 그래프 PNG + 콘솔 리포트
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # Windows cp949 콘솔 대응

from ai_fs.drift import make_degraded_stream
from ai_fs.estimator import AdaptiveCorrector
from ai_fs.pattern import generate_stream
from ai_fs.qc import QcMonitor
from ai_fs.scopes import comparison_panel
from ai_fs.sync import align, estimate_offset

WIDTH, HEIGHT, NUM_FRAMES = 480, 270, 96
TRUE_OFFSET = 7
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse < 1e-12 else float(10 * np.log10(1.0 / mse))


def tracking_chart(truth, estimates, size=(900, 640)) -> np.ndarray:
    """실제 드리프트(노랑) vs AI 추정(녹색) 추종 그래프."""
    w, h = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    rows = [
        ("Y gain", [p.y_gain for p in truth], [p.y_gain for p in estimates]),
        ("Black level", [p.y_offset for p in truth], [p.y_offset for p in estimates]),
        ("Chroma gain", [p.c_gain for p in truth], [p.c_gain for p in estimates]),
        ("Hue (deg)", [p.hue_deg for p in truth], [p.hue_deg for p in estimates]),
    ]
    band = h // len(rows)
    n = len(truth)
    for r, (name, t, e) in enumerate(rows):
        y0, y1 = r * band + 30, (r + 1) * band - 14
        lo = min(min(t), min(e))
        hi = max(max(t), max(e))
        pad = 0.1 * max(hi - lo, 1e-3)
        lo, hi = lo - pad, hi + pad

        def to_px(vals):
            xs = (np.arange(n) / (n - 1) * (w - 80) + 60).astype(np.int32)
            ys = ((1 - (np.asarray(vals) - lo) / (hi - lo)) * (y1 - y0) + y0).astype(np.int32)
            return np.stack([xs, ys], axis=1)

        cv2.rectangle(img, (60, y0), (w - 20, y1), (50, 50, 50), 1)
        cv2.putText(img, name, (60, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(img, f"{hi:+.3f}", (2, y0 + 12), cv2.FONT_HERSHEY_PLAIN, 0.8, (130, 130, 130), 1)
        cv2.putText(img, f"{lo:+.3f}", (2, y1), cv2.FONT_HERSHEY_PLAIN, 0.8, (130, 130, 130), 1)
        cv2.polylines(img, [to_px(t)], False, (0, 210, 255), 1)   # 실제 (노랑)
        cv2.polylines(img, [to_px(e)], False, (80, 255, 80), 1)   # 추정 (녹색)
    cv2.putText(img, "yellow = actual drift   green = AI estimate", (60, h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return img


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rng_frames = {"early": 20, "late": 90}

    print("=" * 62)
    print(" AI FS 데모 — 프레임 동기 + AI 자동 색/밝기 교정 + QC")
    print("=" * 62)

    # 1) 신호 생성
    reference = generate_stream(WIDTH, HEIGHT, NUM_FRAMES)
    degraded, truth = make_degraded_stream(reference, frame_offset=TRUE_OFFSET)
    print(f"\n[신호] 기준 {NUM_FRAMES}프레임 / 입력: {TRUE_OFFSET}프레임 지연 + 드리프트")
    print(f"       마지막 프레임 실제 드리프트: {truth[-1].describe()}")

    # 2) 프레임 싱크 검출·정렬
    offset, conf = estimate_offset(degraded, reference)
    aligned = align(degraded, offset)
    mark = "일치" if offset == TRUE_OFFSET else f"불일치(실제 {TRUE_OFFSET})"
    print(f"\n[싱크] 추정 오프셋 = {offset}프레임 (신뢰도 {conf:.2f}) → {mark}")

    # 3) 프레임별 AI 교정 + QC
    corrector = AdaptiveCorrector(alpha=0.25)
    qc = QcMonitor()
    corrected = np.empty_like(aligned)
    estimates = []
    for i in range(NUM_FRAMES):
        qc.inspect(i, aligned[i])
        corrected[i], est = corrector.process(aligned[i], reference[i])
        estimates.append(est)

    # 4) 검증 — 이상 구간·버퍼 프라임(앞)·버퍼 홀드(뒤) 구간은 제외하고 화질 평가
    anomaly = set(range(40 - offset, 44)) | set(range(70 - offset, 72))
    eval_idx = [i for i in range(offset + 3, NUM_FRAMES - offset - 1) if i not in anomaly]
    p_before = np.mean([psnr(aligned[i], reference[i]) for i in eval_idx])
    p_after = np.mean([psnr(corrected[i], reference[i]) for i in eval_idx])
    print(f"\n[화질] 기준 대비 PSNR:  교정 전 {p_before:.1f} dB → 교정 후 {p_after:.1f} dB "
          f"(+{p_after - p_before:.1f} dB)")

    last_t, last_e = truth[-1], estimates[-1]
    print("[추종] 마지막 프레임 파라미터 (실제 → AI 추정):")
    print(f"       Y gain      {last_t.y_gain:+.3f} → {last_e.y_gain:+.3f}")
    print(f"       Black level {last_t.y_offset:+.3f} → {last_e.y_offset:+.3f}")
    print(f"       Chroma gain {last_t.c_gain:+.3f} → {last_e.c_gain:+.3f}")
    print(f"       Hue         {last_t.hue_deg:+.1f}\u00b0 → {last_e.hue_deg:+.1f}\u00b0")

    print(f"\n[QC] 감지 이벤트 {len(qc.events)}건:")
    kinds: dict[str, list[int]] = {}
    for ev in qc.events:
        kinds.setdefault(ev.kind, []).append(ev.frame)
    for kind, frames in kinds.items():
        print(f"       {kind:6s} frames {min(frames)}–{max(frames)} ({len(frames)}건)")

    # 5) 시각 자료 저장
    for name, idx in rng_frames.items():
        panel = comparison_panel([
            (f"INPUT (drifted, f{idx})", aligned[idx]),
            (f"AI FS OUTPUT (f{idx})", corrected[idx]),
            (f"REFERENCE (f{idx})", reference[idx]),
        ])
        path = os.path.join(OUT_DIR, f"panel_{name}.png")
        cv2.imwrite(path, panel)
        print(f"\n[저장] {path}")

    chart = tracking_chart(truth, estimates)
    chart_path = os.path.join(OUT_DIR, "drift_tracking.png")
    cv2.imwrite(chart_path, chart)
    print(f"[저장] {chart_path}")

    print("\n완료.")


if __name__ == "__main__":
    main()
