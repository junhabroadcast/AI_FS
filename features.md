# 기능 — AI FS v1.1.0

## 캡처 / 입력

- Blackmagic **DeckLink** SDI 입력 (Desktop Video COM API, `comtypes`)
- 장치 목록 / Busy·free 표시 / 포트 선택
- **MTA 전용 워커 스레드**에서 Enable·콜백·디코드 (Tk STA와 분리)
- 8-bit UYVY 우선, 10-bit v210 폴백 (프리뷰용)
- OpenCV 5 호환 UYVY `(H,W,2)` 변환
- 포맷 자동 감지 후 Stop → Enable → Start 재설정 (WFM과 동일 패턴)
- 폴백 입력: **Demo** / Webcam / Screen 캡처
- 무신호 시 NO SIGNAL 안내 프레임 + 콜백/유효 카운트

## 기준 대비 밝기·색 판정 (모니터 기본)

- **기준 캡처**: 현재 프레임을 골든 레퍼런스로 저장
- **기준 지우기**: 대기 상태(판정 안내만)로 복귀
- 통계: 축소 BGR → 근사 Y / Cb / Cr 평균 (중심 ROI·p95 포함)
- 밝기: `ΔY = live − ref` → 정상 / 밝아짐 / 많이 밝아짐 / 어두워짐 / 많이 어두워짐
- 색: `ΔC = hypot(ΔCb, ΔCr)` → 정상 / 틀어짐
- EMA로 깜빡임 억제, 목표 **30 fps**
- 기준 없을 때: 「기준을 먼저 캡처하세요」 (절대 DARK/BRIGHT 미사용)

## 절대 밝기 판정 (CLI / 배치)

- 통계 + 경량 AI 스코어 + 규칙 융합 (`ai_fs/brightness.py`)
- 라벨: DARK / NORMAL / BRIGHT / OVER / UNDER / BLACK
- `demo_brightness.py` / `live_brightness.py`에서 사용

## GUI — AiFsMonitor

- Device 콤보 + Refresh / Start / Stop (WFM UX)
- **기준 캡처** / **기준 지우기**
- 실시간 프리뷰 + 하단 상태바 (`LOCKED` / `NO SIGNAL` / 기준 대비 판정 / fps)
- 오버레이: `OK` / `BRIGHTER` / `DARKER` + `COLOR OK` / `SHIFT`
- PyInstaller onedir → `dist/AiFsMonitor/AiFsMonitor.exe`
- `tools/build_exe.ps1` 빌드 스크립트
- `--selftest` 자동 검증 (기준 캡처 포함)

## 프레임 동기 · 색 교정 (데모)

- 드리프트 불변 시그니처 교차상관으로 정수 프레임 오프셋 추정
- Y 선형회귀 + C 복소 최소제곱으로 프로크앰프형 드리프트 추정·역보정
- AdaptiveCorrector EMA 추종, 이상 프레임에서 파라미터 홀드
- QC: BLACK / FREEZE / GAMUT 이벤트
- CRT형 파형·벡터스코프 비교 패널 PNG 출력

## 도구

- `demo.py` — 싱크·교정 엔드투엔드
- `demo_brightness.py` — 노출 스윕 정확도 / `--video` 파일 입력
- `live_brightness.py` — CLI 실시간 (`--decklink` / `--demo` / `--camera` / `--screen`)
- `ai_fs_monitor.py` — GUI 본체
- `ai_fs/reference_judge.py` — 기준 대비 판정 엔진
- `requirements.txt` — numpy, opencv-python, Pillow, mss, comtypes
