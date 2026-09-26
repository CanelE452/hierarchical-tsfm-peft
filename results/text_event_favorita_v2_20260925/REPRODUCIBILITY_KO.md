# 검산 및 재현 범위

## 원자료 없이 확인 가능한 것

1. REPORT_KO.md의 집계와 EVENT_COUNTS.json, figures/*.csv를 대조한다.
2. C0_EVENT_STORE_DETAILS.csv의 passed_c0=true 행을 골라 (concept,eval_date,store_nbr)를 C0_INDEPENDENT.json의 kept_store_events와 비교한다. 양쪽 모두823개이며 집합 차이는0이다.
3. ARTIFACT_MANIFEST.json의 경로별 SHA-256을 실제 파일 바이트와 비교한다. manifest 자체는 자기 해시 목록에 포함하지 않는다. 경로는 저장소 루트 기준이다.
4. T0_REPORT.json의 status와 실패한 t0_chronos.py 호출을 확인한다. 검증 실패 기록과 미실행 항목이 있으며 성능 점수는 없다.

## 원자료가 필요한 것

[확인] Kaggle 원본 ZIP은 게시하지 않는다. 다음 위치에 동일한 파일이 있어야 한다.

```text
experiments/text_event_favorita_v2_20260925/.cache/store-sales-time-series-forecasting.zip
SHA-256: 12e9c1dc4833cc804b3ff1515bd7b688f45fd8216fb2b340525036b006d625be
```

Windows 저장소 루트에서, 기존 Python 환경에 pandas/matplotlib가 있는 경우 다음 명령은 모델을 호출하지 않고 원본 해시·C0 저장 결과 검산 및 그림 재생성을 수행한다.

```powershell
& .venv/Scripts/python.exe experiments/text_event_favorita_v2_20260925/publish_report.py
```

[확인] 이 명령은 이 실험의 figures와 PUBLICATION_AUDIT.json을 다시 쓴다. SVG에는 생성 시각/내부 ID가 들어갈 수 있어 그림 재생성 후에는 기존 manifest와 바이트가 달라질 수 있다. 수치 재현과 파일 바이트 동일성을 구별한다.

[확인] C0 원본 계산은 c0_count.py가, 별도 구현은 independent_recount.py가 수행했다. 기존 폴더에서 자동 재실행하지 않는다. c0_count.py는 기존 결과를 덮어쓰고, independent_recount.py는 기존 JSON이 있으면 배타적 생성(open x)에서 거부하므로 별도 검증 사본에서 실행해야 한다. 보고서 게시 작업에서는 기존 과학적 결과 파일을 재생성하지 않았다.

## 실행하지 않은 검증

T0 전체, 데이터 누출/LOCO/PCA 감사, encoder 결정성, UniCA 공식 재현, 본 모델 점수의 독립 수식 재계산은 미완료다. C0·게시 검산 PASS를 위 항목의 PASS로 읽으면 안 된다.

T0를 재개하려면 공변량 arm과 target-only arm을 별도 predict_quantiles 호출로 분리해야 한다. 이번 게시에는 수정·재실행이나 새로운 학습이 포함되지 않는다.
