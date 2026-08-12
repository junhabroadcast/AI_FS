# AI FS — AI Frame Synchronizer 프로토타입

**버전:** v0.2.0

FS(Frame Synchronizer)는 비동기 소스의 프레임 타이밍을 하우스 레퍼런스에 맞추고,
색·밝기가 틀어진 신호를 교정하는 방송 장비입니다. 이 프로젝트는 그 위에
**AI 자동화 계층**을 얹으면 무엇이 가능한지 소프트웨어로 증명하는 프로토타입입니다.

[WFM 프로젝트](../WFM)의 BT.601/709 색 행렬, 벡터스코프 타깃, 컬러바 시뮬레이터
지식을 그대로 이어받았습니다.

## AI FS가 하는 일

| 기능 | 기존 FS | AI FS (이 프로토타입) |
|------|---------|----------------------|
| 프레임 동기 | 젠록 기준 재타이밍 | 영상 내용 시그니처 교차상관으로 **오프셋 자동 검출** 후 정렬 |
| 색/밝기 교정 | 사람이 프로크앰프 노브 조작 | 기준 대비 드리프트를 **프레임마다 최소제곱으로 추정**해 자동 역보정 |
| **밝기 판정** | 사람이 스코프를 보고 판단 | Y 통계 + 경량 AI로 **어둡다/정상/밝다/과다·과소** 실시간 판정 |
| 신호 감시 | 별도 QC 장비 | BLACK / FREEZE / GAMUT 이벤트 **실시간 감지** 내장 |
| 이상 시 동작 | 수동 개입 | 이상 프레임에서 추정을 중단하고 마지막 유효 파라미터 **홀드** |

### 자동 교정 모델

입력 열화를 프로크앰프 파라미터로 모델링하고 역변환을 풉니다.

```
Y'  = y_gain · Y + y_offset            (밝기 게인 / 블랙 레벨)
C'  = c_gain · R(hue) · [Cb, Cr]       (채도 게인 / 색상 위상 회전)
```

- **Y 채널**: 1차 선형회귀 + 잔차 기반 아웃라이어 제거(2-pass) — 움직이는 물체,
  프리즈 프레임, RGB 클리핑 픽셀이 추정을 오염시키지 않음
- **C 채널**: 복소 최소제곱 한 번으로 채도 게인(|a|)과 hue 회전(arg a)을 동시 추정
- **시간축**: EMA 평활로 노이즈에 흔들리지 않게 추종 (실시간형)

## 데모 실행

```powershell
pip install -r requirements.txt
python demo.py                 # 프레임 싱크 + 색 교정
python demo_brightness.py      # 밝기 판정 배치 데모 (합성 노출 스윕)
python demo_brightness.py --video clip.mp4   # 파일 입력으로 밝기 판정
python live_brightness.py --demo             # 실시간 창: 합성 노출이 재생되며 계속 판정
python live_brightness.py                    # 실시간 창: 웹캠
python live_brightness.py --screen           # 실시간 창: 모니터 화면 캡처
python live_brightness.py --video clip.mp4   # 실시간 창: 영상 반복 재생 + 판정
```

창이 뜨면 영상이 재생되는 동안 좌상단에 **밝다/정상/어둡다** 가 실시간으로 갱신됩니다. 종료는 `q` 또는 `ESC`.

### 밝기 판정 (v0.2.0)

SDI(DeckLink) / 파일 / 시뮬레이터 프레임을 받아 **밝다·정상·어둡다**를 판정합니다.

```
프레임 → Y 피처(평균·중앙 ROI·p95·하이라이트/섀도우) → 규칙 판정
                                                      ↘ 경량 AI 스코어(-1~+1)
                                                      → 융합 + EMA → 한글 라벨
```

합성 노출 스윕 실측: **엄격 96.3% / 유사허용 97.2%**

`output/brightness_grid.png`, `brightness_timeline.png`, `brightness_strip.png`

DeckLink 연동 시에는 캡처 콜백에서 `BrightnessJudge.judge(rgb)`만 호출하면 됩니다.
파일 입력(`--video`)으로 동일 파이프라인을 미리 검증할 수 있습니다.

### GUI 앱 / exe (WFM 스타일)

장치 선택 콤보 + Start/Stop이 있는 GUI 버전입니다.

```powershell
python ai_fs_monitor.py        # 파이썬으로 실행
.\tools\build_exe.ps1          # exe 빌드
.\dist\AiFsMonitor\AiFsMonitor.exe   # exe 실행
```

- 상단 **Device** 콤보에서 SDI 포트(BUSY/free 표시), Demo, Webcam, Screen 선택
- **Start / Stop / Refresh** 버튼, 하단 상태바에 `LOCKED …` / `NO SIGNAL` + 판정 표시
- 다른 PC 배포 시 `dist\AiFsMonitor` 폴더째 복사 (Desktop Video 드라이버는 별도 설치)

### SDI (DeckLink) 실시간 연결

이미 Desktop Video가 설치되어 있고 **DeckLink Quad 2** 가 잡히는 환경입니다.

```powershell
cd C:\Users\user\cursor\AI_FS
pip install -r requirements.txt

python live_brightness.py --list-decklink   # 포트 목록 / BUSY 확인
python live_brightness.py --decklink        # Busy 아닌 첫 SDI 포트
python live_brightness.py --decklink 1      # Quad (2) 지정
```

1. SDI 케이블을 **BUSY가 아닌 포트**에 연결 (Media Express / WFM이 쓰는 포트는 피하기)
2. 위 명령으로 창을 띄우면 SDI 영상이 재생되며 밝기가 실시간 판정됩니다
3. 상태줄에 `LOCK …` 이면 신호 록, `NO SIGNAL` 이면 케이블/포트를 확인하세요

구현: `ai_fs/decklink_capture.py` (WFM과 동일 `DeckLinkAPI64.dll` COM)

### 프레임 싱크·색 교정 실측 (v0.1.0)

- 프레임 오프셋: **7/7 정확 검출**
- PSNR: 18.8 dB → **32.0 dB (+13.2 dB)**
- 최종 파라미터 추종 오차: Y gain 0.001, black 0.000, chroma 0.001, hue 0.0°
- QC: 주입한 FREEZE / BLACK 구간 전건 감지

`output/` 폴더에 저장되는 자료:

- `panel_early.png` / `panel_late.png` — 입력 / AI FS 출력 / 기준의
  픽처 + 파형 + 벡터스코프 3열 비교
- `drift_tracking.png` — 실제 드리프트(노랑) vs AI 추정(녹색) 추종 그래프

## 구조

```
입력 스트림 ─► 싱크 검출(sync) ─► 프레임 정렬 ─► 드리프트 추정·교정(estimator) ─► 출력
                                        │
하우스 레퍼런스 ────────────────────────┴─► QC 감시(qc) ─► 이벤트 로그
```

| 경로 | 역할 |
|------|------|
| `ai_fs/color.py` | BT.601/709 행렬, 벡터스코프 타깃 (WFM ColorMatrix 포팅) |
| `ai_fs/pattern.py` | 컬러바 + 모션 테스트 장면 발생기 |
| `ai_fs/drift.py` | 드리프트/이상 프레임 주입 시뮬레이터 (검증용 정답지 포함) |
| `ai_fs/sync.py` | 드리프트 불변 시그니처 교차상관 프레임 오프셋 검출 |
| `ai_fs/estimator.py` | 강인 최소제곱 드리프트 추정 + EMA 적응 교정기 |
| `ai_fs/qc.py` | BLACK / FREEZE / GAMUT 이벤트 감지 |
| `ai_fs/brightness.py` | 밝기 판정 (통계 규칙 + 경량 AI 스코어) |
| `ai_fs/exposure_source.py` | 노출 스윕 시나리오 / 파일 입력 어댑터 |
| `ai_fs/scopes.py` | CRT 녹색 톤 파형/벡터스코프 렌더러 (WFM 스타일) |
| `demo.py` | 싱크·색 교정 엔드투엔드 데모 |
| `demo_brightness.py` | 밝기 판정 데모 |

## 로드맵 (실장비화 방향)

- **DeckLink 실시간 밝기 모니터** — WFM 캡처 경로에 `BrightnessJudge` 연결
- **레퍼런스 없는 교정** — 컬러바 자동 인식 또는 장면 통계 학습 기반 교정
  (현재는 하우스 레퍼런스 대비 카메라 매칭 시나리오)
- **서브프레임 위상** — 정수 프레임을 넘어 라인/샘플 단위 타이밍 보간
- **AI 프레임 보간** — 드롭 프레임을 광류/신경망 보간으로 은닉 (에러 컨실먼트)
- **립싱크 감지** — 음성-입모양 상관으로 A/V 오프셋 자동 보정
- **딥러닝 열화 추정** — 비선형 열화(감마, 렌즈 셰이딩)까지 3D LUT로 학습 교정
- **실 SDI 연동** — WFM의 DeckLink 캡처 경로를 재사용해 실신호 입출력

## 요구 사항

- Python 3.10+
- numpy, opencv-python (`requirements.txt`)

## 한계

- 내장 시뮬레이터 기준의 개념 증명이며 방송망 교정 기준이 아닙니다.
- 교정 모델은 선형 프로크앰프 범위이며, 비선형(감마) 열화는 로드맵 항목입니다.
- 스트림 끝의 FREEZE 이벤트는 유한 데모 스트림의 버퍼 홀드에 의한 것입니다.
