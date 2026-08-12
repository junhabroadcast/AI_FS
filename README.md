# AI FS — AI Frame Synchronizer

**버전:** v1.0.0

방송용 FS(Frame Synchronizer) 개념에 **AI 자동화**를 얹은 Windows 프로토타입입니다.  
Blackmagic DeckLink SDI 입력을 받아 **실시간 밝기 판정**하고, 소프트웨어 데모로는 **프레임 동기·색/밝기 자동 교정·QC**까지 검증합니다.

WFM(웨이브폼 모니터) 프로젝트의 BT.601/709 색 행렬·컬러바·스코프 지식을 이어받았습니다.

## 주요 기능

- **AiFsMonitor GUI / exe** — Device 콤보로 DeckLink 포트 선택, Start/Stop (WFM 스타일)
- DeckLink SDI 실시간 캡처 (COM API, MTA 워커 스레드)
- 밝기 판정: 어둡다 / 정상 / 밝다 / 과다·과소노출 / 블랙 (목표 **30 fps**)
- Demo / Webcam / Screen 입력 폴백
- `demo.py` — 프레임 오프셋 검출 + 드리프트 자동 교정 엔드투엔드
- `demo_brightness.py` — 합성 노출 스윕 정확도 검증

## 요구 사항

- Windows 10/11 x64
- Python 3.12+ (소스 실행 시) 또는 배포된 `AiFsMonitor.exe`
- Blackmagic **Desktop Video** + DeckLink 카드 (SDI 사용 시)
- OpenGL/디스플레이는 GUI용 (Tk + OpenCV)

## 빠른 실행 (exe)

```powershell
.\dist\AiFsMonitor\AiFsMonitor.exe
```

1. **WFM / Media Express**에서 같은 포트를 쓰고 있으면 Stop
2. Device에서 `DeckLink Quad (n) (free)` 선택 → **Start**
3. 상태바에 `LOCKED …` + 판정 문구가 보이면 정상

빌드:

```powershell
pip install -r requirements.txt
.\tools\build_exe.ps1
```

## 소스 실행

```powershell
pip install -r requirements.txt
python ai_fs_monitor.py          # GUI
python demo.py                   # 싱크·색 교정 데모
python demo_brightness.py        # 밝기 판정 배치 데모
python live_brightness.py --demo # OpenCV 창 실시간 데모
```

## 문서

| 파일 | 설명 |
|------|------|
| [features.md](features.md) | v1.0.0 기능 목록 |
| [notes.md](notes.md) | 구조, 제한, 사용 팁 |
| [changelog.md](changelog.md) | 변경 이력 |

## 라이선스

저장소 소유자가 별도로 명시하지 않는 한 저작권은 소유자에게 있습니다.
