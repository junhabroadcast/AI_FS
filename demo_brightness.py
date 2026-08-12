"""밝기 판정 데모.

시나리오: DARK → UNDER → NORMAL → BRIGHT → OVER → BLACK 노출 스윕
판정: 통계 규칙 + 경량 AI 스코어 융합 → 한글 라벨 + Y%

실행:  python demo_brightness.py
       python demo_brightness.py --video path/to/clip.mp4   # 파일 입력 (DeckLink 대체 테스트)
출력:  output/brightness_*.png
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from ai_fs.brightness import BrightnessJudge, BrightnessLabel
from ai_fs.exposure_source import exposure_stream, load_video_rgb
from ai_fs.scopes import rgb_to_bgr8, waveform

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

_LABEL_COLOR = {
    BrightnessLabel.DARK: (80, 140, 255),  # 주황
    BrightnessLabel.UNDER: (40, 40, 220),  # 빨강
    BrightnessLabel.NORMAL: (80, 220, 80),  # 녹
    BrightnessLabel.BRIGHT: (0, 220, 255),  # 노랑
    BrightnessLabel.OVER: (0, 80, 255),  # 주황-빨강
    BrightnessLabel.BLACK: (160, 160, 160),
}


def annotate(rgb: np.ndarray, result, frame_i: int) -> np.ndarray:
    img = rgb_to_bgr8(rgb)
    h, w = img.shape[:2]
    scale = max(w / 480, 1.0)
    color = _LABEL_COLOR[result.label]
    lines = [
        f"f{frame_i}  {result.korean()} ({result.label.value})",
        f"Y avg {result.mean_y_pct:.1f}%  center {result.center_y_pct:.1f}%  p95 {result.p95_y_pct:.1f}%",
        f"score {result.score:+.2f}  AI {result.ai_score:+.2f}  conf {result.confidence:.2f}",
        f"HL {result.highlight_pct:.1f}%  SH {result.shadow_pct:.1f}%",
    ]
    y = int(22 * scale)
    for line in lines:
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * scale, color, 1, cv2.LINE_AA)
        y += int(20 * scale)

    # 점수 바: 왼쪽=어두움, 오른쪽=밝음
    bar_w, bar_h = w - 20, max(8, int(10 * scale))
    x0, y0 = 10, h - 20
    cv2.rectangle(img, (x0, y0), (x0 + bar_w, y0 + bar_h), (60, 60, 60), -1)
    mid = x0 + bar_w // 2
    cv2.line(img, (mid, y0 - 2), (mid, y0 + bar_h + 2), (120, 120, 120), 1)
    pos = int(np.clip((result.score + 1) * 0.5, 0, 1) * bar_w)
    cv2.rectangle(img, (x0, y0), (x0 + pos, y0 + bar_h), color, -1)
    return img


def timeline_chart(labels_true: list[str] | None, results, size=(960, 320)) -> np.ndarray:
    w, h = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    n = len(results)
    scores = np.array([r.score for r in results])
    means = np.array([r.mean_y_pct for r in results])

    def poly(vals, lo, hi, color, band_top, band_bot):
        xs = (np.arange(n) / max(n - 1, 1) * (w - 80) + 60).astype(np.int32)
        ys = ((1 - (vals - lo) / max(hi - lo, 1e-6)) * (band_bot - band_top) + band_top).astype(np.int32)
        pts = np.stack([xs, ys], axis=1)
        cv2.polylines(img, [pts], False, color, 2)

    # 상단: exposure score
    cv2.putText(img, "Exposure score (-1 dark .. +1 bright)", (60, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.rectangle(img, (60, 30), (w - 20, 150), (40, 40, 40), 1)
    cv2.line(img, (60, 90), (w - 20, 90), (80, 80, 80), 1)
    poly(scores, -1.0, 1.0, (80, 255, 80), 30, 150)

    # 하단: mean Y %
    cv2.putText(img, "Mean Y %", (60, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.rectangle(img, (60, 186), (w - 20, 300), (40, 40, 40), 1)
    poly(means, 0.0, 100.0, (0, 210, 255), 186, 300)

    # 정답 구간 라벨
    if labels_true:
        i = 0
        while i < n:
            j = i
            while j < n and labels_true[j] == labels_true[i]:
                j += 1
            x = int(i / max(n - 1, 1) * (w - 80) + 60)
            cv2.putText(img, labels_true[i], (x, h - 8), cv2.FONT_HERSHEY_PLAIN, 0.9, (180, 180, 180), 1)
            i = j
    return img


def confusion_report(truth: list[str], predicted: list[str]) -> None:
    # 유사 그룹: DARK≈UNDER, BRIGHT≈OVER 는 부분 정답으로도 집계
    soft = {
        "DARK": {"DARK", "UNDER"},
        "UNDER": {"DARK", "UNDER"},
        "NORMAL": {"NORMAL"},
        "BRIGHT": {"BRIGHT", "OVER"},
        "OVER": {"BRIGHT", "OVER"},
        "BLACK": {"BLACK"},
    }
    hard = sum(t == p for t, p in zip(truth, predicted))
    soft_ok = sum(p in soft.get(t, {t}) for t, p in zip(truth, predicted))
    n = len(truth)
    print(f"[정확도] 엄격 {hard}/{n} ({100 * hard / n:.1f}%)  /  유사허용 {soft_ok}/{n} ({100 * soft_ok / n:.1f}%)")


def run_synthetic() -> None:
    width, height, fpl = 480, 270, 18
    stream, truth = exposure_stream(width, height, frames_per_level=fpl)
    judge = BrightnessJudge(alpha=0.4)
    results = [judge.judge(frame) for frame in stream]
    pred = [r.label.value for r in results]

    print("=" * 62)
    print(" AI FS — 밝기 판정 데모 (합성 노출 스윕)")
    print("=" * 62)
    print(f"프레임 {len(stream)}  (구간당 {fpl})")
    confusion_report(truth, pred)

    # 구간별 대표 프레임 요약
    print("\n[구간 요약]")
    i = 0
    while i < len(truth):
        j = i
        while j < len(truth) and truth[j] == truth[i]:
            j += 1
        mid = (i + j) // 2
        r = results[mid]
        ok = "OK" if r.label.value in {truth[i], "DARK" if truth[i] == "UNDER" else "", "BRIGHT" if truth[i] == "OVER" else "", truth[i]} or (
            (truth[i] == "DARK" and r.label.value in ("DARK", "UNDER"))
            or (truth[i] == "UNDER" and r.label.value in ("DARK", "UNDER"))
            or (truth[i] == "BRIGHT" and r.label.value in ("BRIGHT", "OVER"))
            or (truth[i] == "OVER" and r.label.value in ("BRIGHT", "OVER"))
            or (truth[i] == r.label.value)
        ) else "MISS"
        print(
            f"  {truth[i]:6s} → 판정 {r.korean():6s} ({r.label.value:6s})  "
            f"Y={r.mean_y_pct:5.1f}%  score={r.score:+.2f}  [{ok}]"
        )
        i = j

    os.makedirs(OUT_DIR, exist_ok=True)

    # 각 구간 대표 프레임: 픽처+파형
    picks = []
    i = 0
    while i < len(truth):
        j = i
        while j < len(truth) and truth[j] == truth[i]:
            j += 1
        mid = (i + j) // 2
        picks.append(mid)
        i = j

    tiles = []
    for mid in picks:
        pic = annotate(stream[mid], results[mid], mid)
        pic = cv2.resize(pic, (320, 180))
        wfm = cv2.resize(waveform(stream[mid], size=(320, 140)), (320, 140))
        tiles.append(np.vstack([pic, wfm]))
    # 2x3 그리드
    while len(tiles) < 6:
        tiles.append(np.zeros_like(tiles[0]))
    row1 = np.hstack(tiles[0:3])
    row2 = np.hstack(tiles[3:6])
    grid = np.vstack([row1, row2])
    grid_path = os.path.join(OUT_DIR, "brightness_grid.png")
    cv2.imwrite(grid_path, grid)
    print(f"\n[저장] {grid_path}")

    chart = timeline_chart(truth, results)
    chart_path = os.path.join(OUT_DIR, "brightness_timeline.png")
    cv2.imwrite(chart_path, chart)
    print(f"[저장] {chart_path}")

    # 스트립: 시간순 샘플
    step = max(1, len(stream) // 12)
    strip = np.hstack([cv2.resize(annotate(stream[i], results[i], i), (160, 90)) for i in range(0, len(stream), step)])
    strip_path = os.path.join(OUT_DIR, "brightness_strip.png")
    cv2.imwrite(strip_path, strip)
    print(f"[저장] {strip_path}")
    print("\n완료.")


def run_video(path: str) -> None:
    print("=" * 62)
    print(f" AI FS — 밝기 판정 (파일 입력: {path})")
    print("=" * 62)
    stream = load_video_rgb(path)
    judge = BrightnessJudge(alpha=0.35)
    results = [judge.judge(frame) for frame in stream]

    from collections import Counter

    counts = Counter(r.label.value for r in results)
    print(f"프레임 {len(stream)}")
    for k, v in counts.most_common():
        print(f"  {k:6s}: {v} ({100 * v / len(stream):.1f}%)")

    os.makedirs(OUT_DIR, exist_ok=True)
    chart = timeline_chart(None, results)
    cv2.imwrite(os.path.join(OUT_DIR, "brightness_timeline.png"), chart)

    # 평균 score 기준 대표 프레임 3장
    scores = [r.score for r in results]
    idxs = [int(np.argmin(scores)), int(np.argsort(np.abs(scores))[len(scores) // 2]), int(np.argmax(scores))]
    tiles = []
    for i in idxs:
        pic = cv2.resize(annotate(stream[i], results[i], i), (360, 203))
        tiles.append(pic)
    cv2.imwrite(os.path.join(OUT_DIR, "brightness_grid.png"), np.hstack(tiles))
    print(f"[저장] {OUT_DIR}\\brightness_*.png")
    print("완료.")


def main() -> None:
    ap = argparse.ArgumentParser(description="AI FS 밝기 판정 데모")
    ap.add_argument("--video", type=str, default=None, help="입력 영상 파일 (미지정 시 합성 스윕)")
    args = ap.parse_args()
    if args.video:
        run_video(args.video)
    else:
        run_synthetic()


if __name__ == "__main__":
    main()
