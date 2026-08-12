# Changelog

**AI FS**의 주요 변경 사항을 기록합니다.

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
