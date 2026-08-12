"""DeckLink SDI 캡처 — WFM과 동일한 COM 경로.

핵심 (WFM DeckLinkCapture.cpp 정렬):
  - CoInitializeEx(MTA) 전용 워커 스레드에서만 DeckLink API 호출
    (Tkinter STA 메인 스레드와 분리 — 신호 록/포맷감지가 여기서 깨지던 원인)
  - EnableVideoInput: 10-bit YUV 우선 → 8-bit 폴백
  - VideoInputFormatChanged 에서 감지 모드로 Stop→Enable→Start 재설정
  - 프레임은 GetPixelFormat() 기준으로 UYVY / v210 디코드
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import comtypes
import comtypes.client
import cv2
import numpy as np
from comtypes import COMObject

comtypes.client.GetModule(
    r"C:\Program Files\Blackmagic Design\Desktop Video\DeckLinkAPI64.dll"
)
from comtypes.gen import DeckLinkAPI as api  # noqa: E402

BMD_NO_INPUT = 0x80000000  # bmdFrameHasNoInputSource (WFM과 동일)


@dataclass
class DeckLinkDeviceInfo:
    index: int
    name: str
    busy: bool


def _coinit_mta() -> None:
    try:
        comtypes.CoInitializeEx(comtypes.COINIT_MULTITHREADED)
    except OSError:
        pass


def list_devices() -> list[DeckLinkDeviceInfo]:
    _coinit_mta()
    it = comtypes.client.CreateObject(api.CDeckLinkIterator)
    out: list[DeckLinkDeviceInfo] = []
    idx = 0
    while True:
        try:
            deck = it.Next()
        except ValueError:
            break
        if not deck:
            break
        name = str(deck.GetDisplayName())
        busy = False
        try:
            st = deck.QueryInterface(api.IDeckLinkStatus)
            busy = int(st.GetInt(api.bmdDeckLinkStatusBusy)) != 0
        except Exception:
            pass
        out.append(DeckLinkDeviceInfo(idx, name, busy))
        idx += 1
    return out


def _uyvy_to_bgr(raw: np.ndarray, width: int, height: int, row_bytes: int) -> np.ndarray:
    """8-bit UYVY → BGR. OpenCV 5는 (H,W,2) 입력이 필요."""
    row = raw.reshape(height, row_bytes)
    packed = np.ascontiguousarray(row[:, : width * 2].reshape(height, width, 2))
    return cv2.cvtColor(packed, cv2.COLOR_YUV2BGR_UYVY)


def _unpack_v210_to_bgr(raw: np.ndarray, width: int, height: int, row_bytes: int) -> np.ndarray:
    """v210 → BGR8 (프리뷰용, 벡터화)."""
    # 6 pixels / 16 bytes
    y8 = np.empty((height, width), dtype=np.uint8)
    u8 = np.empty((height, width), dtype=np.uint8)
    v8 = np.empty((height, width), dtype=np.uint8)
    words_per_group = 4
    for y in range(height):
        row = np.frombuffer(raw[y * row_bytes : (y + 1) * row_bytes], dtype=np.uint32)
        x = 0
        gi = 0
        while x + 5 < width and gi + words_per_group <= row.size:
            w0, w1, w2, w3 = row[gi : gi + 4]
            cb0, y0, cr0 = int(w0) & 0x3FF, (int(w0) >> 10) & 0x3FF, (int(w0) >> 20) & 0x3FF
            y1, cb1, y2 = int(w1) & 0x3FF, (int(w1) >> 10) & 0x3FF, (int(w1) >> 20) & 0x3FF
            cr1, y3, cb2 = int(w2) & 0x3FF, (int(w2) >> 10) & 0x3FF, (int(w2) >> 20) & 0x3FF
            y4, cr2, y5 = int(w3) & 0x3FF, (int(w3) >> 10) & 0x3FF, (int(w3) >> 20) & 0x3FF
            y8[y, x : x + 6] = np.array([y0, y1, y2, y3, y4, y5], dtype=np.uint16) >> 2
            u8[y, x : x + 6] = np.array([cb0, cb0, cb1, cb1, cb2, cb2], dtype=np.uint16) >> 2
            v8[y, x : x + 6] = np.array([cr0, cr0, cr1, cr1, cr2, cr2], dtype=np.uint16) >> 2
            x += 6
            gi += words_per_group
    # BT.709 limited → BGR (간단 변환, OpenCV 채널수 이슈 회피)
    yf = y8.astype(np.float32)
    uf = u8.astype(np.float32) - 128.0
    vf = v8.astype(np.float32) - 128.0
    r = np.clip(yf + 1.5748 * vf, 0, 255)
    g = np.clip(yf - 0.1873 * uf - 0.4681 * vf, 0, 255)
    b = np.clip(yf + 1.8556 * uf, 0, 255)
    return cv2.merge([b, g, r]).astype(np.uint8)


class _InputCallback(COMObject):
    _com_interfaces_ = [api.IDeckLinkInputCallback]

    def __init__(self, owner: "_CaptureWorker"):
        super().__init__()
        self._owner = owner

    def VideoInputFormatChanged(self, notificationEvents, newDisplayMode, detectedSignalFlags):
        try:
            self._owner.on_format_changed(newDisplayMode)
        except Exception as e:
            self._owner.last_error = f"formatChanged: {e}"
        return 0

    def VideoInputFrameArrived(self, videoFrame, audioPacket):
        try:
            self._owner.on_frame(videoFrame)
        except Exception as e:
            self._owner.last_error = f"frameArrived: {e}"
        return 0


class _CaptureWorker:
    """MTA 스레드 안에서만 동작하는 실제 캡처 엔진."""

    def __init__(self, device_index: int, max_width: int, frame_q: queue.Queue, ctrl_q: queue.Queue):
        self.device_index = device_index
        self.max_width = max_width
        self.frame_q = frame_q
        self.ctrl_q = ctrl_q
        self.name = ""
        self.mode_name = ""
        self.has_signal = False
        self.frames = 0
        self.sig_frames = 0
        self.last_error = ""
        self.ready = threading.Event()
        self.failed: str | None = None
        self._input = None
        self._deck = None
        self._callback = None
        self._pixel_format = api.bmdFormat10BitYUV

    def run(self) -> None:
        _coinit_mta()
        try:
            self._open_and_start()
            self.ready.set()
            # 제어 큐: "stop" 올 때까지 대기 (콜백이 프레임 공급)
            while True:
                try:
                    cmd = self.ctrl_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if cmd == "stop":
                    break
        except Exception as e:
            self.failed = str(e)
            self.ready.set()
        finally:
            self._cleanup()
            try:
                comtypes.CoUninitialize()
            except Exception:
                pass

    def _open_and_start(self) -> None:
        devices = list_devices()
        if self.device_index < 0 or self.device_index >= len(devices):
            raise RuntimeError(f"장치 인덱스 {self.device_index} 없음")
        info = devices[self.device_index]
        if info.busy:
            raise RuntimeError(
                f"{info.name} 이 Busy 입니다. WFM/Media Express 에서 Stop 한 뒤 다시 시도하세요."
            )

        it = comtypes.client.CreateObject(api.CDeckLinkIterator)
        deck = None
        for _ in range(self.device_index + 1):
            deck = it.Next()
        self._deck = deck
        self.name = str(deck.GetDisplayName())

        try:
            cfg = deck.QueryInterface(api.IDeckLinkConfiguration)
            cfg.SetInt(api.bmdDeckLinkConfigVideoInputConnection, api.bmdVideoConnectionSDI)
        except Exception:
            pass

        self._input = deck.QueryInterface(api.IDeckLinkInput)
        self._callback = _InputCallback(self)
        self._input.SetCallback(self._callback)

        # 프리뷰 30fps: 8-bit 우선 (10-bit v210 언팩은 느림)
        attempts = [
            (api.bmdModeHD1080i5994, api.bmdFormat8BitYUV),
            (api.bmdModeHD1080p5994, api.bmdFormat8BitYUV),
            (api.bmdModeHD1080i5994, api.bmdFormat10BitYUV),
            (api.bmdModeHD1080p5994, api.bmdFormat10BitYUV),
            (api.bmdModeHD1080p30, api.bmdFormat8BitYUV),
            (api.bmdModeHD720p5994, api.bmdFormat8BitYUV),
        ]
        last_err: Exception | None = None
        enabled = False
        for mode, fmt in attempts:
            try:
                self._input.EnableVideoInput(mode, fmt, api.bmdVideoInputEnableFormatDetection)
                self._pixel_format = fmt
                enabled = True
                break
            except Exception as e:
                last_err = e
        if not enabled:
            raise RuntimeError(f"EnableVideoInput 실패: {last_err}")

        self._input.StartStreams()

    def on_format_changed(self, newDisplayMode) -> None:
        # WFM: Stop → Enable(감지모드) → Start
        # 프리뷰 FPS를 위해 8-bit를 우선 (WFM 스코프는 10-bit 우선)
        name = str(newDisplayMode.GetName())
        mode = newDisplayMode.GetDisplayMode()
        self.mode_name = name
        self._input.StopStreams()
        try:
            self._input.EnableVideoInput(mode, api.bmdFormat8BitYUV, api.bmdVideoInputEnableFormatDetection)
            self._pixel_format = api.bmdFormat8BitYUV
        except Exception:
            self._input.EnableVideoInput(mode, api.bmdFormat10BitYUV, api.bmdVideoInputEnableFormatDetection)
            self._pixel_format = api.bmdFormat10BitYUV
        self._input.StartStreams()

    def on_frame(self, videoFrame) -> None:
        if videoFrame is None:
            return
        self.frames += 1
        flags = int(videoFrame.GetFlags())
        if flags & BMD_NO_INPUT:
            self.has_signal = False
            return

        w = int(videoFrame.GetWidth())
        h = int(videoFrame.GetHeight())
        rb = int(videoFrame.GetRowBytes())
        pf = int(videoFrame.GetPixelFormat())
        ptr = videoFrame.GetBytes()
        if ptr is None:
            return
        addr = int(ptr)
        raw = np.ctypeslib.as_array((comtypes.c_ubyte * (rb * h)).from_address(addr)).copy()

        if pf == api.bmdFormat8BitYUV or pf == int(api.bmdFormat8BitYUV):
            bgr = _uyvy_to_bgr(raw, w, h, rb)
        elif pf == api.bmdFormat10BitYUV or pf == int(api.bmdFormat10BitYUV):
            bgr = _unpack_v210_to_bgr(raw, w, h, rb)
        elif pf == api.bmdFormat8BitBGRA or pf == int(api.bmdFormat8BitBGRA):
            bgra = np.ascontiguousarray(raw.reshape(h, rb)[:, : w * 4]).reshape(h, w, 4)
            bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        else:
            # 알 수 없는 포맷 — rowBytes 보고 UYVY 가정
            if rb >= w * 2:
                bgr = _uyvy_to_bgr(raw, w, h, rb)
            else:
                self.last_error = f"unsupported pixelFormat={pf} {w}x{h} rb={rb}"
                return

        if self.max_width and bgr.shape[1] > self.max_width:
            scale = self.max_width / bgr.shape[1]
            bgr = cv2.resize(bgr, (self.max_width, int(bgr.shape[0] * scale)))

        self.has_signal = True
        self.sig_frames += 1
        # 최신 프레임만 유지
        try:
            while True:
                self.frame_q.get_nowait()
        except queue.Empty:
            pass
        self.frame_q.put((bgr, self.mode_name, True))

    def _cleanup(self) -> None:
        try:
            if self._input is not None:
                try:
                    self._input.StopStreams()
                except Exception:
                    pass
                try:
                    self._input.DisableVideoInput()
                except Exception:
                    pass
                try:
                    self._input.SetCallback(None)
                except Exception:
                    pass
        finally:
            self._input = None
            self._callback = None
            self._deck = None


class DeckLinkSource:
    """UI 스레드에서 쓰는 래퍼 — 내부적으로 MTA 워커가 SDI를 잡는다."""

    def __init__(self, device_index: int = 0, max_width: int = 1280):
        self.device_index = device_index
        self.max_width = max_width
        self._frame_q: queue.Queue = queue.Queue(maxsize=2)
        self._ctrl_q: queue.Queue = queue.Queue()
        self._latest: np.ndarray | None = None
        self._has_signal = False
        self._mode_name = ""
        self._name = f"DeckLink[{device_index}]"
        self._worker = _CaptureWorker(device_index, max_width, self._frame_q, self._ctrl_q)
        self._thread = threading.Thread(target=self._worker.run, name="DeckLinkMTA", daemon=True)
        self._thread.start()
        if not self._worker.ready.wait(timeout=8.0):
            self.release()
            raise RuntimeError("DeckLink 시작 타임아웃")
        if self._worker.failed:
            err = self._worker.failed
            self.release()
            raise RuntimeError(err)
        self._name = self._worker.name or self._name

        # 신호 록 대기 (최대 3초) — 포맷 감지 시간
        t0 = time.time()
        while time.time() - t0 < 3.0:
            self._drain()
            if self._has_signal:
                break
            time.sleep(0.05)

    def _drain(self) -> None:
        while True:
            try:
                bgr, mode, sig = self._frame_q.get_nowait()
                self._latest = bgr
                self._mode_name = mode
                self._has_signal = sig
            except queue.Empty:
                break
        if self._worker.frames > 0 and self._worker.sig_frames == 0:
            self._has_signal = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def mode_name(self) -> str:
        return self._mode_name or self._worker.mode_name

    @property
    def has_signal(self) -> bool:
        self._drain()
        return self._has_signal

    @property
    def frames(self) -> int:
        return self._worker.frames

    def read(self) -> np.ndarray | None:
        self._drain()
        if self._latest is not None and self._has_signal:
            return self._latest.copy()
        return self._nosignal_frame()

    def _nosignal_frame(self) -> np.ndarray:
        from PIL import Image, ImageDraw, ImageFont

        h, w = 540, 960
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (28, 28, 28)
        pil = Image.fromarray(img[:, :, ::-1])
        draw = ImageDraw.Draw(pil)
        try:
            font_l = ImageFont.truetype(r"C:\Windows\Fonts\malgun.ttf", 42)
            font_s = ImageFont.truetype(r"C:\Windows\Fonts\malgun.ttf", 20)
        except OSError:
            font_l = ImageFont.load_default()
            font_s = font_l
        err = self._worker.last_error
        lines = [
            "NO SIGNAL",
            self._name,
            f"콜백 {self._worker.frames} / 유효 {self._worker.sig_frames}",
            "WFM 이 같은 포트를 쓰고 있으면 Stop 하세요.",
            "그래도 안 되면 Refresh 후 같은 Device 다시 Start.",
        ]
        if err:
            lines.append(str(err)[:80])
        y = h // 2 - 100
        for i, line in enumerate(lines):
            fill = (255, 80, 80) if i == 0 else (220, 220, 220)
            font = font_l if i == 0 else font_s
            bbox = draw.textbbox((0, 0), line, font=font)
            tw = bbox[2] - bbox[0]
            draw.text(((w - tw) // 2, y), line, font=font, fill=fill)
            y += 48 if i == 0 else 28
        return np.asarray(pil)[:, :, ::-1].copy()

    def release(self) -> None:
        try:
            self._ctrl_q.put("stop")
        except Exception:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)


def pick_free_device(preferred: int | None = None) -> int:
    devices = list_devices()
    if preferred is not None:
        return preferred
    for d in devices:
        if not d.busy:
            return d.index
    raise RuntimeError("모든 DeckLink 포트가 Busy 입니다. WFM/Media Express 를 종료하세요.")
