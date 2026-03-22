# Nasdaq All Multi-Zone Support / Resistance Scanner

이 패키지는 **Nasdaq 전체 종목**을 대상으로, 업로드한 Pine Script의 `S1 / S2 support zone` 로직을 파이썬 스캐너로 옮기고 `R1 / R2 resistance` 및 다중 존 관리를 추가한 **나스닥 전용 GitHub Actions 패키지**입니다.

## 포함 파일

```text
.
├─ .github/
│  └─ workflows/
│     └─ weekly_nasdaq_all_scan.yml
├─ support_resistance_scanner.py
├─ requirements.txt
└─ README.md
```

## 핵심 반영 내용

- 다중 support / resistance zone 유지
- `S1 / S2 / R1 / R2` 이벤트 로그 생성
- `isBear` 실제 반영
- `S2 / R2` 재무장(re-arm) 수정
- S2 / R2 볼륨 강화 조건
- RSI cross 확인
- 캔들 패턴 필터 반영

## GitHub Actions 실행

- 워크플로: `.github/workflows/weekly_nasdaq_all_scan.yml`
- 유니버스: `nasdaq_all`
- 결과 폴더: `results/nasdaq_all/`

## 직접 실행 예시

```bash
python -m pip install -r requirements.txt
python support_resistance_scanner.py \
  --universe nasdaq_all \
  --period 2y \
  --interval 1d \
  --output-dir results/nasdaq_all
```

## 생성 결과

- `latest_summary.csv`
- `latest_bar_signal_hits.csv`
- `all_signal_events.csv`
- `zones_catalog.csv`
- `errors.csv`
- `summary.md`
- `universe_membership.csv`
- `charts/*.png`
