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

_LABEL_BGR = {
    BrightnessLabel.DARK: (80, 140, 255),
    BrightnessLabel.UNDER: (40, 40, 220),
    BrightnessLabel.NORMAL: (80, 220, 80),
    BrightnessLabel.BRIGHT: (0, 220, 255),
    BrightnessLabel.OVER: (0, 80, 255),
    BrightnessLabel.BLACK: (160, 160, 160),
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
    """한글 라벨 + 점수 바를 프레임 위에 그린다.

    호출 전에 표시 크기로 줄인 프레임을 넘기는 것을 권장한다.
    (풀 HD에서 매 프레임 PIL alpha_composite 하면 ~수 fps로 떨어진다.)
    """
    # 표시용으로 너무 크면 한번 더 줄여 오버레이 비용을 제한
    h, w = bgr.shape[:2]
    max_w = 960
    work = bgr
    if w > max_w:
        nh = max(1, int(h * max_w / w))
        work = cv2.resize(bgr, (max_w, nh), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img, "RGBA")
    color = _LABEL_BGR[result.label]
    fill = (color[2], color[1], color[0])
    lines = [
        f"{result.korean()}  ({result.label.value})",
        f"Y 평균 {result.mean_y_pct:.1f}%   중앙 {result.center_y_pct:.1f}%   p95 {result.p95_y_pct:.1f}%",
        f"score {result.score:+.2f}   AI {result.ai_score:+.2f}   conf {result.confidence:.2f}   {fps:.1f} fps",
        f"하이라이트 {result.highlight_pct:.1f}%   섀도우 {result.shadow_pct:.1f}%",
    ]
    pad = 10
    line_h = 30
    box_h = pad * 2 + line_h * len(lines) + 18
    box_w = min(img.width - 20, 620)
    # 전체 프레임 alpha_composite 대신 박스만 반투명 사각형
    draw.rectangle([8, 8, 8 + box_w, 8 + box_h], fill=(0, 0, 0, 160))

    y = 14
    for i, line in enumerate(lines):
        font = _FONT if i == 0 else _FONT_SM
        draw.text((18, y), line, font=font, fill=fill if i == 0 else (230, 230, 230, 255))
        y += line_h if i == 0 else 24

    bar_y = 8 + box_h - 14
    bar_x0, bar_x1 = 18, 8 + box_w - 10
    draw.rectangle([bar_x0, bar_y, bar_x1, bar_y + 8], fill=(50, 50, 50, 255))
    mid = (bar_x0 + bar_x1) // 2
    draw.line([mid, bar_y - 2, mid, bar_y + 10], fill=(120, 120, 120, 255), width=1)
    pos = int(bar_x0 + (result.score + 1) * 0.5 * (bar_x1 - bar_x0))
    draw.rectangle([bar_x0, bar_y, pos, bar_y + 8], fill=(*fill, 255))

    out = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    if work is not bgr and (out.shape[1] != w or out.shape[0] != h):
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
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
    """장비 없이 실시간 노출 변화를 보여 주는 합성 소스."""

    def __init__(self, width: int = 960, height: int = 540):
        self.w, self.h = width, height
        self.i = 0
        # 약 6초 주기로 노출 스윕
        self.levels = [
            ("DARK", -0.55),
            ("UNDER", -0.85),
            ("NORMAL", 0.0),
            ("BRIGHT", 0.28),
            ("OVER", 0.95),
            ("BLACK", None),
        ]
        self.frames_per = 45  # ~1.5초 @30fps

    def read(self) -> np.ndarray | None:
        seg = (self.i // self.frames_per) % len(self.levels)
        name, exp = self.levels[seg]
        if name == "BLACK":
            rgb = np.zeros((self.h, self.w, 3), dtype=np.float32)
        else:
            rgb = exposure_scene(self.w, self.h, self.i, float(exp))
        self.i += 1
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)[:, :, ::-1]  # RGB→BGR

    def release(self) -> None:
        pass


def run(source, title: str = "AI FS Live Brightness") -> None:
    judge = BrightnessJudge(alpha=0.45)
    win = title
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    print("=" * 56)
    print(" 실시간 밝기 판정 — 창에서 영상이 재생되며 계속 판정합니다")
    if hasattr(source, "name"):
        print(f" 입력: {source.name}")
    print(" 종료: q 또는 ESC")
    print("=" * 56)

    t0 = time.perf_counter()
    n = 0
    fps = 0.0
    last_print = 0.0

    try:
        while True:
            frame = source.read()
            if frame is None:
                print("입력 종료")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            # 판정 속도: 큰 프레임은 축소 분석
            h, w = rgb.shape[:2]
            if w > 640:
                scale = 640 / w
                small = cv2.resize(rgb, (640, int(h * scale)), interpolation=cv2.INTER_AREA)
            else:
                small = rgb
            result = judge.judge(small)

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

            key = cv2.waitKey(1) & 0xFF
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
