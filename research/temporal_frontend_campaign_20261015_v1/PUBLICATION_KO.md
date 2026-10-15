# 시간 앞단 캠페인 공개 자산

최신 설명은 [전체 보고서 v3](REPORT_KO_v3.md), [교수님 설명안 v3](PROFESSOR_BRIEF_v3.md), [판단 기록 v3](DECISION_LOG_v3.md)이다. [최신 감사 연결](AUDIT_LATEST.json)은 보존한 첫 감사와 통과한 v2 감사를 구분한다.

80개 neural gradient fit과 6개 deterministic ridge logical fit을 실행했다. 실제 fitting configurations는 80+10 ridge alpha=90이며 연장 endpoint를 새 fit으로 중복 세지 않는다. neural 80개 중 12개는 미수렴·미해결 상태다. G-only와 LoRA 반례를 유지하며 성공 방법론·새 독립 일반화를 주장하지 않는다. Robin의 2048행 lookback 경로 A13은 합법 공통 origin이 부족해 BLOCKED다. 시간 판단은 balanced general, 메모리는 exclusive-process peak다.

## 공개 범위와 재현 제약

이 디렉터리의 과학·보고서 소스, JSON/JSONL receipt, 작은 CSV, PNG/SVG, 설명·감사 로그를 게시한다. 원자료, 보호 TEST, 원 모델 가중치, 로컬 `.cache`의 선택/last 체크포인트와 예측 NPZ는 포함하지 않는다. 공개 자료로 문서·비용 원표와 파일 무결성을 확인할 수 있지만 저장 예측 재감사와 원 실험 재실행에는 허용된 원 local cache 및 실행환경이 추가로 필요하다. 이 복원 명령은 데이터나 모델을 내려받거나 학습하지 않는다.

[공개 manifest](PUBLICATION_FILES.json)와 [정확한 파일 목록](PUBLICATION_FILES.txt)은 허용 확장자의 정규파일만 열거한다. __pycache__/pyc, 확장자 없는 guard lock, 디렉터리, 큰 CSV 원본은 제외한다. 선택·비용 잠금 JSON은 포함한다. 순환 해시를 피하기 위해 manifest/list 자체 두 파일만 목록·해시에서 명시적으로 제외하므로 게시자는 두 파일을 별도로 포함해야 한다. 원 `.log` 두 개는 목록에 포함하며 Git ignore 여부와 별도로 root가 명시적으로 추가한다.

큰 비용 CSV 네 개는 원본 바이트와 SHA를 유지하는 mtime=0 gzip 사본으로 공개한다. 압축 원본의 Git stage 대신 다음 `.csv.gz` 파일을 사용한다. 다른 CSV는 그대로 포함한다.

- `cost/beijing/RAW_COST_TRIALS.csv` → `.csv.gz`; 원본 SHA256 `69dd79cc05d185f7a90b6b804d5d630bcccf1b6c833e96377bc8cc61e90dd475`.
- `cost/jena/RAW_COST_TRIALS.csv` → `.csv.gz`; 원본 SHA256 `6936dd501991c42838f0fbe28eb5bd3abbfe5926bb060f262d50e4cd41a347bc`.
- `cost_exclusive/primary_beijing_Raw_Ridge/RAW_COST_TRIALS.csv` → `.csv.gz`; 원본 SHA256 `c3cabd94edf0db64c381181bc06ec23277f112fb4d44b8051a8eb82a55400738`.
- `cost_exclusive/primary_jena_Raw_Ridge/RAW_COST_TRIALS.csv` → `.csv.gz`; 원본 SHA256 `b53d60ce277da02806cd87bda2264890b2fdc1a9a331833dcb63283d3d652a26`.

manifest의 compressed_originals에 원본·gzip의 정확한 repo 경로, byte size 및 SHA256이 모두 있다. 저장소 루트에서 Python 표준 라이브러리만으로 실행한다.

```sh
python research/temporal_frontend_campaign_20261015_v1/publication_assets.py verify
python research/temporal_frontend_campaign_20261015_v1/publication_assets.py restore
```

verify는 공개 파일의 SHA/size와 정확한 목록을 확인한다. restore는 gzip을 원 경로로 복원하기 전에 원 SHA를 검사하고 배타적으로 새 파일을 만든다. 기존 원 CSV가 있으면 일치 여부만 확인하며 덮어쓰지 않는다. 불일치하면 중단한다. 복원 후 verify도 같은 공개 목록을 검사하며 로컬 복원 원본은 게시 대상으로 바꾸지 않는다.

공개 링크 수선은 DATA_ASSET_REVIEW.md의 Beijing contract 링크 한 개를 [byte 동일 공개 사본](references/beijing_contract.json)으로 변경한 것이다. [수선 전 설명 원문 gzip](references/DATA_ASSET_REVIEW_at_run.md.gz)을 보존한다. 원문 SHA256은 `fa96a466b112eb84f9c5e2698053b5298e3ceed488764b72f494a719b21a1ca7`, 링크 수선본은 `a40c9c120a05ee3926c124bc48f4f863404b0bd654f9bf4ab4ebef5f74da25eb`이다. 이 보존 gzip은 현재 설명문을 덮어쓰는 restore 대상이 아니다. 과학 소스·선정·예측·비용 원수치는 바꾸지 않았다.

동결 receipt의 no-push 문구는 실험 종료 시점의 동작 범위다. 이번 게시 작업은 그 이후 사용자의 명시 승인에 따른 별도 수동 main push이며 receipt를 소급 수정하지 않는다. 이 자산 도구는 Git stage/commit/push를 실행하지 않는다.

## 최신 상대경로 그림

![R1 시간 변환](figures/01_time_transforms_v2.png)

![R2 시간 앞단과 C/K](figures/02_temporal_rank_v2.png)

![R3 같은 G 비교](figures/03_matched_residual_v2.png)

![R5 학습과 시간 재적합](figures/04_learning_time_refit_v2.png)

![정확도·general 시간·exclusive 메모리](figures/05_error_time_memory_v3.png)

![사전 고정 및 사후 사례](figures/06_forecast_cases_v2.png)
