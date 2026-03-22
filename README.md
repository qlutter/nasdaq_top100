# Nasdaq Top 100 Support / Resistance Scanner

이 패키지는 업로드한 Pine 기반 로직을 파이썬 스캐너로 옮긴 버전이며, 아래 기능을 포함합니다.

- 다중 `supportBox` / `resistanceBox`
- 시그널 4종: `S1`, `S2`, `R1`, `R2`
- 날짜 구간 필터링 결과
- 박스 시작일 / 종료일 포함 결과 CSV
- GitHub Actions 주 1회 실행

## 핵심 결과 파일

실행 결과는 `results/nasdaq_top100/` 아래에 생성됩니다.

- `latest_summary.csv`
- `latest_bar_signal_hits.csv`
- `all_signal_events.csv`
- `zones_catalog.csv`
- `signal_window_hits.csv`
- `structure_window_hits.csv`
- `window_scan_results.csv`
- `summary.md`
- `errors.csv`
- `charts/*.png`

## 새로 추가된 날짜/박스 결과 파일 설명

### 1) signal_window_hits.csv
요청한 날짜 구간에 실제 발생한 시그널만 저장합니다.

컬럼:
- `ticker`
- `signal_date`
- `signal_type`
- `box_type`
- `box_start_date`
- `box_end_date`
- `zone_id`
- `zone_kind`
- `zone_pivot_date`

### 2) structure_window_hits.csv
요청한 날짜 구간과 겹치는 `supportBox` / `resistanceBox` 구조만 저장합니다.

컬럼:
- `ticker`
- `box_type`
- `box_start_date`
- `box_end_date`
- `zone_id`
- `zone_kind`
- `zone_pivot_date`
- `zone_price`
- `zone_top`
- `zone_bottom`
- `active`

### 3) window_scan_results.csv
시그널과 구조를 한 파일로 합친 통합 결과입니다.

컬럼:
- `ticker`
- `row_type` (`signal` / `structure`)
- `item_type` (`S1`, `S2`, `R1`, `R2`, `supportBox`, `resistanceBox`)
- `signal_date`
- `box_start_date`
- `box_end_date`
- `zone_id`
- `zone_kind`
- `zone_pivot_date`

## 날짜 의미

- `zone_pivot_date`: 피봇이 형성된 날짜
- `box_start_date`: 박스가 실제로 생성된 날짜 (`created_date`)
- `box_end_date`: 박스가 무효화된 날짜 (`invalidated_date`)
  - 아직 살아 있는 박스는 빈칸으로 남습니다.

## GitHub Actions 동작

- 기본 스케줄: 매주 1회
- 수동 실행(`Run workflow`) 가능
- 수동 실행 시 `signal_start_date`, `signal_end_date`를 직접 넣을 수 있음
- 스케줄 실행 시에는 최근 20일 구간을 자동 계산

## 수동 실행 예시

```bash
python -m pip install -r requirements.txt
python support_resistance_scanner.py \
  --universe nasdaq_top100 \
  --top-n 100 \
  --period 2y \
  --interval 1d \
  --output-dir results/nasdaq_top100 \
  --signal-start-date 2026-03-01 \
  --signal-end-date 2026-03-20
```

## 리포 구조

```text
.
├─ .github/
│  └─ workflows/
│     └─ weekly_nasdaq_top100_scan.yml
├─ support_resistance_scanner.py
├─ requirements.txt
└─ README.md
```
