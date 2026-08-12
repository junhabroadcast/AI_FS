r"""AI FS Monitor — WFM 스타일 GUI 실시간 밝기 판정.

상단에서 캡처카드(DeckLink 포트)를 선택하고 Start를 누르면
SDI 영상이 재생되면서 밝기(어둡다/정상/밝다/과다·과소/블랙)를 실시간 판정한다.

  python ai_fs_monitor.py            # GUI 실행
  python ai_fs_monitor.py --selftest # 자동 검증 후 종료 (빌드 확인용)

exe 빌드:  .\tools\build_exe.ps1  →  dist\AiFsMonitor\AiFsMonitor.exe
"""

from __future__ import annotations

import os
import sys
import time
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

if sys.stdout is not None and sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from ai_fs.brightness import BrightnessJudge
from live_brightness import CameraSource, DemoSource, ScreenSource, overlay

try:
    from ai_fs.decklink_capture import DeckLinkSource, list_devices

    HAVE_DECKLINK = True
except Exception:
    HAVE_DECKLINK = False


APP_TITLE = "AI FS Monitor"
VIEW_W, VIEW_H = 1120, 630
TARGET_FPS = 30
JUDGE_WIDTH = 160  # 대충 판정용 축소 폭
DISPLAY_MAX_W = 960  # 오버레이/표시용 상한


class MonitorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title(APP_TITLE)
        root.configure(bg="#161616")
        root.geometry(f"{VIEW_W + 24}x{VIEW_H + 130}")

        self.source = None
        self.judge = BrightnessJudge(alpha=0.45)
        self._options: list[tuple[str, int | None, str]] = []
        self._photo = None  # GC 방지
        self._fps = 0.0
        self._fps_n = 0
        self._fps_t0 = time.perf_counter()
        self._running = False

        self._build_ui()
        self.refresh_devices()

    # --- UI -----------------------------------------------------------
    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background="#161616")
        style.configure("TLabel", background="#161616", foreground="#dddddd")
        style.configure("Status.TLabel", background="#0d0d0d", foreground="#9ef09e")

        top = ttk.Frame(self.root, padding=(8, 8, 8, 4))
        top.pack(fill="x")

        ttk.Label(top, text="Device:").pack(side="left")
        self.device_combo = ttk.Combobox(top, state="readonly", width=52)
        self.device_combo.pack(side="left", padx=(6, 8))

        self.refresh_btn = ttk.Button(top, text="Refresh", command=self.refresh_devices)
        self.refresh_btn.pack(side="left", padx=(0, 16))

        self.start_btn = ttk.Button(top, text="Start", command=self.on_start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(top, text="Stop", command=self.on_stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(6, 0))

        # 영상 뷰
        self.view = tk.Label(self.root, bg="black")
        self.view.pack(fill="both", expand=True, padx=12, pady=8)

        # 상태바 (WFM 스타일)
        self.status = ttk.Label(
            self.root, text="Ready — 장치를 선택하고 Start를 누르세요", style="Status.TLabel",
            padding=(10, 5),
        )
        self.status.pack(fill="x", side="bottom")

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # --- 장치 목록 -----------------------------------------------------
    def refresh_devices(self) -> None:
        self._options = []
        labels = []
        if HAVE_DECKLINK:
            try:
                for d in list_devices():
                    state = "BUSY" if d.busy else "free"
                    labels.append(f"[SDI {d.index}] {d.name}  ({state})")
                    self._options.append(("decklink", d.index, d.name))
            except Exception as e:
                labels.append(f"(DeckLink 오류: {e})")
                self._options.append(("none", None, ""))
        if not self._options or self._options[0][0] == "none":
            if not HAVE_DECKLINK:
                labels.insert(0, "(No DeckLink — Desktop Video 미설치)")
                self._options.insert(0, ("none", None, ""))

        labels.append("Demo — 내장 노출 시뮬레이터")
        self._options.append(("demo", None, "DEMO"))
        labels.append("Webcam 0")
        self._options.append(("camera", 0, "CAMERA 0"))
        labels.append("Screen — 모니터 화면 캡처")
        self._options.append(("screen", None, "SCREEN"))

        self.device_combo["values"] = labels
        # 기본 선택: free SDI 포트 → 없으면 Demo
        pick = 0
        for i, (kind, idx, _) in enumerate(self._options):
            if kind == "decklink":
                try:
                    devs = {d.index: d for d in list_devices()}
                    if idx in devs and not devs[idx].busy:
                        pick = i
                        break
                except Exception:
                    pass
        else:
            for i, (kind, _, _) in enumerate(self._options):
                if kind == "demo":
                    pick = i
                    break
        self.device_combo.current(pick)

    # --- Start / Stop --------------------------------------------------
    def on_start(self) -> None:
        if self._running:
            return
        sel = self.device_combo.current()
        if sel < 0 or sel >= len(self._options):
            return
        kind, param, name = self._options[sel]

        try:
            if kind == "decklink":
                # 표시·판정에 풀 HD가 필요 없음 — 캡처 단계에서 줄여 CPU 절약
                self.source = DeckLinkSource(param, max_width=DISPLAY_MAX_W)
            elif kind == "camera":
                self.source = CameraSource(param)
            elif kind == "screen":
                self.source = ScreenSource()
            elif kind == "demo":
                self.source = DemoSource()
            else:
                self.status.config(text="사용 가능한 장치가 없습니다. Demo를 선택하세요.")
                return
        except Exception as e:
            self.status.config(text=f"시작 실패: {e}")
            self.source = None
            return

        self.judge.reset()
        self._running = True
        self._fps_n = 0
        self._fps_t0 = time.perf_counter()
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.device_combo.config(state="disabled")
        self.refresh_btn.config(state="disabled")
        self.status.config(text=f"Starting — {name or kind} ...")
        self._tick()

    def on_stop(self) -> None:
        self._running = False
        if self.source is not None:
            try:
                self.source.release()
            except Exception:
                pass
            self.source = None
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.device_combo.config(state="readonly")
        self.refresh_btn.config(state="normal")
        self.status.config(text="Stopped")

    def on_close(self) -> None:
        self.on_stop()
        self.root.destroy()

    # --- 프레임 루프 ----------------------------------------------------
    def _tick(self) -> None:
        if not self._running or self.source is None:
            return
        t0 = time.perf_counter()
        try:
            frame = self.source.read()
            if frame is None:
                self.status.config(text="입력 종료")
                self.on_stop()
                return

            # 1) 대충 판정: 아주 작은 BGR에서 Y 통계만 (풀 float RGB 변환 없음)
            result = self.judge.judge_bgr_fast(frame, max_width=JUDGE_WIDTH)

            self._fps_n += 1
            elapsed_fps = time.perf_counter() - self._fps_t0
            if elapsed_fps > 0.5:
                self._fps = self._fps_n / elapsed_fps
                self._fps_n = 0
                self._fps_t0 = time.perf_counter()

            no_signal = hasattr(self.source, "has_signal") and not self.source.has_signal

            # 2) 표시 크기로 먼저 줄인 뒤 오버레이 (풀해상도 PIL이 예전의 3fps 병목)
            self.root.update_idletasks()
            vw = self.view.winfo_width()
            vh = self.view.winfo_height()
            if vw < 64:
                vw = VIEW_W
            if vh < 64:
                vh = VIEW_H
            scale = min(vw / frame.shape[1], vh / frame.shape[0], 1.0)
            nw = max(1, int(frame.shape[1] * scale))
            nh = max(1, int(frame.shape[0] * scale))
            if nw != frame.shape[1] or nh != frame.shape[0]:
                disp = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
            else:
                disp = frame

            view = disp if no_signal else overlay(disp, result, self._fps)

            img = Image.fromarray(cv2.cvtColor(view, cv2.COLOR_BGR2RGB))
            self._photo = ImageTk.PhotoImage(img)
            self.view.config(image=self._photo)

            if no_signal:
                self.status.config(
                    text=f"NO SIGNAL — {getattr(self.source, 'name', '')}  "
                    f"| 이 포트에 SDI를 꽂거나 Device에서 Demo로 테스트하세요  |  {self._fps:.1f} fps"
                )
            else:
                sig = ""
                if hasattr(self.source, "has_signal"):
                    mode = getattr(self.source, "mode_name", "") or ""
                    sig = f"LOCKED {mode}" if self.source.has_signal else ""
                self.status.config(
                    text=f"{sig}   |   판정: {result.korean()} ({result.label.value})   |   "
                    f"Y {result.mean_y_pct:.1f}%   score {result.score:+.2f}   |   {self._fps:.1f} fps"
                )
        except Exception as e:
            self.status.config(text=f"표시 오류: {e}")
            _log_crash(e)

        # 목표 30fps: 작업이 빨리 끝나면 남은 시간만큼 쉬고, 늦으면 즉시 다음 틱
        spent_ms = (time.perf_counter() - t0) * 1000.0
        delay = max(1, int(round(1000.0 / TARGET_FPS - spent_ms)))
        self.root.after(delay, self._tick)


def selftest() -> int:
    """빌드/동작 자동 검증: Demo 소스 2초 실행 후 마커 파일 기록."""
    out_dir = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "output")
    os.makedirs(out_dir, exist_ok=True)
    marker = os.path.join(out_dir, "selftest_ok.txt")
    if os.path.exists(marker):
        os.remove(marker)

    root = tk.Tk()
    app = MonitorApp(root)
    # Demo 소스 강제 선택 후 시작
    for i, (kind, _, _) in enumerate(app._options):
        if kind == "demo":
            app.device_combo.current(i)
            break
    app.on_start()

    result = {"frames": 0}

    def check() -> None:
        if app._photo is not None:
            result["frames"] += 1
        if result["frames"] >= 10:
            dl = "no-decklink"
            if HAVE_DECKLINK:
                try:
                    dl = "; ".join(
                        f"[{d.index}] {d.name} ({'BUSY' if d.busy else 'free'})" for d in list_devices()
                    )
                except Exception as e:
                    dl = f"decklink error: {e}"
            with open(marker, "w", encoding="utf-8") as f:
                f.write(f"selftest ok, frames={result['frames']}, status={app.status.cget('text')}\n")
                f.write(f"decklink: {dl}\n")
            app.on_close()
        else:
            root.after(100, check)

    root.after(300, check)
    root.after(15000, app.on_close)  # 안전 타임아웃
    root.mainloop()
    return 0 if os.path.exists(marker) else 1


def _log_crash(exc: BaseException) -> None:
    """창 모드 exe에서는 콘솔이 없으므로 크래시를 파일로 남긴다."""
    import traceback

    out_dir = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), "output")
    try:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "crash.log"), "w", encoding="utf-8") as f:
            traceback.print_exception(exc, file=f)
    except Exception:
        pass


def main() -> None:
    try:
        if "--selftest" in sys.argv:
            sys.exit(selftest())
        root = tk.Tk()
        MonitorApp(root)
        root.mainloop()
    except SystemExit:
        raise
    except BaseException as e:
        _log_crash(e)
        raise


if __name__ == "__main__":
    main()
