# Changelog

**AI FS**의 주요 변경 사항을 기록합니다.

## [1.1.0] — 2026-08-13

### 추가

- **기준 캡처 기반 밝기·색 판정** — 사용자가 지정한 골든 프레임 대비 ΔY/ΔC로 판정
- GUI: **기준 캡처** / **기준 지우기** 버튼 (`ai_fs_monitor.py`)
- `ai_fs/reference_judge.py` — 기준 피처 저장, EMA, 밝아짐/어두워짐·색 틀어짐 라벨
- 기준 대비 오버레이 (`overlay_reference`)

### 변경

- 모니터 기본 경로를 절대 밝기 판정에서 **기준 대비 판정**으로 전환 (기준 없으면 안내만 표시)
- 절대 규칙 엔진(`ai_fs/brightness.py`)은 유지 (CLI/배치 데모용)

### 문서

- v1.1.0 기준으로 README / features / notes / changelog 갱신

## [1.0.0] — 2026-08-12

### 추가

- **AiFsMonitor** GUI — Device 콤보, Refresh / Start / Stop, 실시간 밝기 판정
- DeckLink SDI 캡처 (`ai_fs/decklink_capture.py`) — MTA 워커, 포맷 감지, Busy 표시
- 밝기 판정 엔진 — 통계 규칙 + 경량 AI 스코어 + EMA (`ai_fs/brightness.py`)
- 프레임 싱크·색/밝기 자동 교정 데모 (`demo.py`)
- 밝기 배치/파일 데모 (`demo_brightness.py`), CLI 라이브 (`live_brightness.py`)
- PyInstaller exe 빌드 (`tools/build_exe.ps1` → `dist/AiFsMonitor/`)
- 문서: README / features / notes / changelog

### 수정

- OpenCV 5 UYVY 입력 shape `(H,W,2)` 호환 — 유효 프레임 0(NO SIGNAL 오인) 수정
- Tk STA와 DeckLink MTA 분리로 신호 록 실패 개선
- 30 fps 목표: 고속 판정·경량 오버레이·Demo 프레임 캐시

### 문서

- v1.0.0 기준으로 README / features / notes / changelog 정리
