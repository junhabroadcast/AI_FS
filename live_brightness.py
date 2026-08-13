"""실시간 밝기 판정 모니터.

캡처 화면이 재생되는 동안 매 프레임 밝기를 판정해 오버레이합니다.

  python live_brightness.py --decklink          # SDI (DeckLink, Busy 아닌 첫 포트)
  python live_brightness.py --decklink 1        # SDI 포트 인덱스 지정
  python live_brightness.py --list-decklink     # DeckLink 장치 목록
  python live_brightness.py --demo              # 합성 노출 (장비 없이)
  python live_brightness.py --camera 0
  python live_brightness.py --video clip.mp4
  python live_brightness.py --screen

종료: q 또는 ESC
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

if sys.stdout is not None and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # 창 모드 exe에서는 stdout이 None

from ai_fs.brightness import BrightnessJudge, BrightnessLabel, BrightnessResult
from ai_fs.exposure_source import exposure_scene
from ai_fs.reference_judge import BrightnessDelta, ColorDelta, ReferenceResult

_LABEL_BGR = {
    BrightnessLabel.DARK: (80, 140, 255),
    BrightnessLabel.UNDER: (40, 40, 220),
    BrightnessLabel.NORMAL: (80, 220, 80),
    BrightnessLabel.BRIGHT: (0, 220, 255),
    BrightnessLabel.OVER: (0, 80, 255),
    BrightnessLabel.BLACK: (160, 160, 160),
}

_REF_BRIGHT_BGR = {
    BrightnessDelta.MATCH: (80, 220, 80),
    BrightnessDelta.BRIGHTER: (0, 220, 255),
    BrightnessDelta.MUCH_BRIGHTER: (0, 80, 255),
    BrightnessDelta.DARKER: (80, 140, 255),
    BrightnessDelta.MUCH_DARKER: (40, 40, 220),
    BrightnessDelta.NO_REF: (160, 160, 160),
}


def _find_font(size: int = 22) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\malgunbd.ttf",
        r"C:\Windows\Fonts\gulim.ttc",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


_FONT = _find_font(26)
_FONT_SM = _find_font(18)


def overlay(bgr: np.ndarray, result: BrightnessResult, fps: float) -> np.ndarray:
    """30fps용 경량 오버레이 (OpenCV). 한글 판정은 상태바에 표시."""
    out = bgr.copy()
    h, w = out.shape[:2]
    color = _LABEL_BGR[result.label]
    box_h, box_w = 78, min(w - 16, 480)
    x0, y0 = 8, 8
    roi = out[y0 : y0 + box_h, x0 : x0 + box_w]
    cv2.addWeighted(roi, 0.40, np.zeros_like(roi), 0.60, 0, roi)
    cv2.putText(
        out,
        f"{result.label.value}",
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        f"Y {result.mean_y_pct:.1f}%  score {result.score:+.2f}  {fps:.0f}fps",
        (16, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    bar_y = y0 + box_h - 12
    cv2.rectangle(out, (16, bar_y), (x0 + box_w - 8, bar_y + 6), (50, 50, 50), -1)
    mid = (16 + x0 + box_w - 8) // 2
    cv2.line(out, (mid, bar_y - 2), (mid, bar_y + 8), (120, 120, 120), 1)
    pos = int(16 + (result.score + 1) * 0.5 * (box_w - 24))
    cv2.rectangle(out, (16, bar_y), (max(16, pos), bar_y + 6), color, -1)
    return out


def overlay_reference(bgr: np.ndarray, result: ReferenceResult, fps: float) -> np.ndarray:
    """기준 대비 판정 오버레이. 한글은 상태바에 표시."""
    out = bgr.copy()
    h, w = out.shape[:2]
    box_h, box_w = 92, min(w - 16, 520)
    x0, y0 = 8, 8
    roi = out[y0 : y0 + box_h, x0 : x0 + box_w]
    cv2.addWeighted(roi, 0.40, np.zeros_like(roi), 0.60, 0, roi)

    if not result.has_reference:
        cv2.putText(
            out,
            "NO REFERENCE",
            (16, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (160, 160, 160),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            f"Capture baseline first   {fps:.0f}fps",
            (16, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        return out

    color = _REF_BRIGHT_BGR[result.brightness]
    bright_txt = {
        BrightnessDelta.MATCH: "OK",
        BrightnessDelta.BRIGHTER: "BRIGHTER",
        BrightnessDelta.MUCH_BRIGHTER: "MUCH BRIGHTER",
        BrightnessDelta.DARKER: "DARKER",
        BrightnessDelta.MUCH_DARKER: "MUCH DARKER",
        BrightnessDelta.NO_REF: "NO REF",
    }[result.brightness]
    color_txt = "COLOR OK" if result.color == ColorDelta.MATCH else "COLOR SHIFT"
    color_bgr = (80, 220, 80) if result.color == ColorDelta.MATCH else (0, 140, 255)

    cv2.putText(
        out,
        bright_txt,
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        color_txt,
        (16, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color_bgr,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        f"dY {result.dy * 100:+.1f}%  dC {result.dc:.3f}  {fps:.0f}fps",
        (16, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    # ΔY 바: 중앙=기준, 오른쪽=밝아짐
    bar_y = y0 + box_h - 10
    cv2.rectangle(out, (16, bar_y), (x0 + box_w - 8, bar_y + 6), (50, 50, 50), -1)
    mid = (16 + x0 + box_w - 8) // 2
    cv2.line(out, (mid, bar_y - 2), (mid, bar_y + 8), (120, 120, 120), 1)
    span = box_w - 24
    # dy ±0.20 → 풀스케일
    t = float(np.clip(result.dy / 0.20, -1.0, 1.0))
    pos = int(mid + t * (span * 0.5))
    x_a, x_b = (mid, pos) if pos >= mid else (pos, mid)
    cv2.rectangle(out, (x_a, bar_y), (max(x_a + 1, x_b), bar_y + 6), color, -1)
    return out


class CameraSource:
    def __init__(self, index: int = 0):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"카메라 {index} 를 열 수 없습니다.")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self) -> None:
        self.cap.release()


class VideoSource:
    def __init__(self, path: str, loop: bool = True):
        self.path = path
        self.loop = loop
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"영상을 열 수 없습니다: {path}")

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        if not ok and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        return frame if ok else None

    def release(self) -> None:
        self.cap.release()


class ScreenSource:
    def __init__(self, monitor: int = 1, max_width: int = 1280):
        import mss

        self.sct = mss.mss()
        if monitor >= len(self.sct.monitors):
            monitor = 1
        self.mon = self.sct.monitors[monitor]
        self.max_width = max_width

    def read(self) -> np.ndarray | None:
        import mss.tools  # noqa: F401

        shot = np.asarray(self.sct.grab(self.mon))  # BGRA
        bgr = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
        h, w = bgr.shape[:2]
        if w > self.max_width:
            scale = self.max_width / w
            bgr = cv2.resize(bgr, (self.max_width, int(h * scale)))
        return bgr

    def release(self) -> None:
        self.sct.close()


class DemoSource:
    """장비 없이 실시간 노출 변화를 보여 주는 합성 소스 (30fps용 캐시)."""

    def __init__(self, width: int = 854, height: int = 480):
        self.w, self.h = width, height
        self.i = 0
        self.levels = [
            ("DARK", -0.55),
            ("UNDER", -0.85),
            ("NORMAL", 0.0),
            ("BRIGHT", 0.28),
            ("OVER", 0.95),
            ("BLACK", None),
        ]
        self.frames_per = 45  # ~1.5초 @30fps
        # 레벨별 프레임을 미리 만들어 매프레임 exposure_scene 비용 제거
        self._cache: dict[str, np.ndarray] = {}
        for name, exp in self.levels:
            if name == "BLACK":
                self._cache[name] = np.zeros((height, width, 3), dtype=np.uint8)
            else:
                rgb = exposure_scene(width, height, 0, float(exp))
                self._cache[name] = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)[:, :, ::-1]

    def read(self) -> np.ndarray | None:
        seg = (self.i // self.frames_per) % len(self.levels)
        name, _ = self.levels[seg]
        self.i += 1
        return self._cache[name].copy()

    def release(self) -> None:
        pass


def run(source, title: str = "AI FS Live Brightness") -> None:
    judge = BrightnessJudge(alpha=0.55)
    win = title
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    print("=" * 56)
    print(" 실시간 밝기 판정 — 창에서 영상이 재생되며 계속 판정합니다")
    if hasattr(source, "name"):
        print(f" 입력: {source.name}")
    print(" 목표 30fps  |  종료: q 또는 ESC")
    print("=" * 56)

    t0 = time.perf_counter()
    n = 0
    fps = 0.0
    last_print = 0.0
    target_dt = 1.0 / 30.0

    try:
        while True:
            loop_t0 = time.perf_counter()
            frame = source.read()
            if frame is None:
                print("입력 종료")
                break

            result = judge.judge_bgr_fast(frame, max_width=160)

            n += 1
            elapsed = time.perf_counter() - t0
            if elapsed > 0.5:
                fps = n / elapsed
                t0 = time.perf_counter()
                n = 0

            view = overlay(frame, result, fps)
            cv2.imshow(win, view)

            now = time.perf_counter()
            if now - last_print > 0.5:
                sig = ""
                if hasattr(source, "has_signal"):
                    mode = getattr(source, "mode_name", "") or ""
                    sig = f"  {'LOCK '+mode if source.has_signal else 'NO SIGNAL'}"
                print(
                    f"\r[{result.korean():6s}] Y={result.mean_y_pct:5.1f}%  "
                    f"score={result.score:+.2f}  {fps:4.1f} fps{sig}   ",
                    end="",
                    flush=True,
                )
                last_print = now

            spent = time.perf_counter() - loop_t0
            wait_ms = max(1, int((target_dt - spent) * 1000))
            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
    finally:
        print()
        source.release()
        cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description="실시간 밝기 판정 모니터")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--decklink", type=int, nargs="?", const=-1, default=None,
                   help="DeckLink SDI 입력 (인덱스 생략 시 Busy 아닌 첫 포트)")
    g.add_argument("--list-decklink", action="store_true", help="DeckLink 장치 목록만 출력")
    g.add_argument("--camera", type=int, nargs="?", const=0, default=None, help="웹캠 인덱스 (기본 0)")
    g.add_argument("--video", type=str, help="영상 파일 경로 (반복 재생)")
    g.add_argument("--screen", action="store_true", help="모니터 화면 캡처")
    g.add_argument("--demo", action="store_true", help="합성 노출 데모 (장비 불필요)")
    args = ap.parse_args()

    if args.list_decklink:
        from ai_fs.decklink_capture import list_devices
        devices = list_devices()
        if not devices:
            print("DeckLink 장치 없음 (Desktop Video 설치 / 카드 연결 확인)")
            return
        print("DeckLink 장치:")
        for d in devices:
            state = "BUSY" if d.busy else "free"
            print(f"  [{d.index}] {d.name}  ({state})")
        print("\n예: python live_brightness.py --decklink 1")
        return

    if args.decklink is not None:
        from ai_fs.decklink_capture import DeckLinkSource, pick_free_device
        idx = pick_free_device(None if args.decklink < 0 else args.decklink)
        source = DeckLinkSource(idx)
        title = f"AI FS Live — SDI {source.name}"
    elif args.demo:
        source = DemoSource()
        title = "AI FS Live — DEMO"
    elif args.video:
        source = VideoSource(args.video)
        title = f"AI FS Live — {os.path.basename(args.video)}"
    elif args.screen:
        source = ScreenSource()
        title = "AI FS Live — SCREEN"
    elif args.camera is not None:
        source = CameraSource(args.camera)
        title = f"AI FS Live — CAMERA {args.camera}"
    else:
        # 기본: DeckLink 있으면 SDI, 없으면 웹캠, 그것도 없으면 데모
        try:
            from ai_fs.decklink_capture import DeckLinkSource, list_devices, pick_free_device
            if list_devices():
                idx = pick_free_device()
                source = DeckLinkSource(idx)
                title = f"AI FS Live — SDI {source.name}"
            else:
                raise RuntimeError("no decklink")
        except Exception:
            try:
                source = CameraSource(0)
                title = "AI FS Live — CAMERA 0"
            except RuntimeError as e:
                print(f"{e}")
                print("카메라/SDI 없어 합성 데모(--demo)로 전환합니다.")
                source = DemoSource()
                title = "AI FS Live — DEMO (fallback)"

    run(source, title)


if __name__ == "__main__":
    main()
