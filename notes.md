# 노트 — AI FS v1.0.0

## 범위

v1.0.0은 **DeckLink SDI 실시간 밝기 판정 GUI/exe**와, FS 자동화 가능성을 보여주는 **소프트웨어 데모**(싱크·색교정·QC)를 포함합니다.

포함:

- AiFsMonitor (장치 선택, Start/Stop, 30fps 판정)
- DeckLink COM 캡처 (MTA 워커)
- 밝기 규칙 + 경량 AI 스코어
- demo 파이프라인 (오프셋·드리프트·스코프 PNG)

미포함 (의도적 / 로드맵):

- 실장비급 서브프레임 젠록·라인 위상
- 딥러닝 기반 비선형 3D LUT 교정
- 오디오/립싱크
- WFM급 멀티 스코프 타일 UI (Waveform/Vector/Lightning)
- C++/Qt 네이티브 포트 (현재는 Python 프로토타입)

## 구조 (요약)

```
SDI → DeckLink COM (MTA thread)
    → UYVY/v210 → BGR
    → BrightnessJudge (fast)
    → Tk 프리뷰 + 상태바

데모: pattern → drift → sync → estimator → qc / scopes → output/*.png
```

| 경로 | 역할 |
|------|------|
| `ai_fs/decklink_capture.py` | DeckLink 목록·MTA 캡처·UYVY/v210 디코드 |
| `ai_fs/brightness.py` | 밝기 피처·규칙·AI 스코어·EMA |
| `ai_fs/color.py` | BT.601/709 행렬 (WFM ColorMatrix 포팅) |
| `ai_fs/sync.py` / `estimator.py` / `drift.py` | 오프셋·교정·열화 시뮬 |
| `ai_fs/qc.py` / `scopes.py` | QC 이벤트·스코프 렌더 |
| `ai_fs_monitor.py` | GUI |
| `tools/build_exe.ps1` | PyInstaller 빌드 |

## OpenCV를 쓰는 이유

WFM은 C++/Qt/OpenGL로 디코드·표시를 자체 구현합니다.  
AI FS는 **파이썬 프로토타입**이라 색변환·리사이즈·간단 오버레이에 OpenCV를 사용합니다.  
SDI 캡처 자체는 DeckLink COM이며, OpenCV는 필수가 아닙니다. 이후 네이티브 포트에서는 WFM처럼 제거할 수 있습니다.

## 사용 팁

1. Desktop Video가 설치되어 `DeckLinkAPI64.dll`이 있어야 합니다.
2. **같은 SDI 포트는 한 앱만** 사용하세요. WFM이 Start 중이면 AI FS는 Busy/무신호입니다.
3. exe는 **`dist\AiFsMonitor\AiFsMonitor.exe`** 만 실행하세요. `build\` 아래 exe는 중간 산출물입니다.
4. OpenCV 5: UYVY는 `(H, W, 2)` shape여야 합니다. (v1.0.0에서 수정)
5. Demo 장치로 UI/판정만 먼저 확인한 뒤 SDI로 전환하세요.
6. Qt 경로는 WFM과 무관합니다. AI FS GUI는 Tkinter입니다.

## 성능

- 목표 표시·판정: **30 fps**
- 판정은 폭 160px 근사 Y 통계 (~1 ms)
- 캡처 표시 상한 약 854px 폭
- DeckLink는 최신 프레임만 유지 (저지연 큐)

## 컬러메트리

- Auto: 높이 ≥720 → BT.709 (교정/스코프 데모)
- 밝기 고속 경로는 Rec.709 근사 휘도 가중
- 공인 legalizer / 교정기 대체품이 아닙니다

## 알려진 제한

- Python + OpenCV 프로토타입이며 방송망 교정 기준이 아닙니다.
- 10-bit v210 프리뷰 언팩은 단순화되어 있으며, 실시간은 8-bit UYVY를 권장합니다.
- PyInstaller 배포본은 Desktop Video 런타임이 대상 PC에 필요합니다.
- `build/`·대용량 바이너리는 저장소에 올리지 않는 것을 권장합니다 (`dist`는 로컬 빌드).
