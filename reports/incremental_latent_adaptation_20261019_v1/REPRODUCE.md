# 공개 표·그림 재현

저장소 루트에서 실행한다. 아래 재생성은 공개 CSV/JSON/충분통계만 읽고 raw data, private 예측, 모델 또는 GPU를 사용하지 않는다.

```powershell
python -X utf8 -m research.incremental_latent_adaptation_20261019_v1.report --report-dir reports/incremental_latent_adaptation_20261019_v1 --output-dir reports/incremental_latent_adaptation_20261019_v1/replayed_public
python -X utf8 -m research.incremental_latent_adaptation_20261019_v1.report --report-dir reports/incremental_latent_adaptation_20261019_v1 --verify-replay
```

학습 재개는 실제 [RESUME_QUEUE.md](RESUME_QUEUE.md) 명령과 sealed EXPERIMENT_PLAN을 따른다. 이 보고서가 새 학습/선택 명령을 추정해 만들지 않는다.
이번 실행 환경은 Python 3.11, PyTorch 2.10+cu128, RTX 4070 12 GiB다. 공개 그림은 NumPy와 Matplotlib 및 저장된 공개 파일로 재생성한다. 실제 학습·원 예측 재검산은 공개하지 않은 원자료/A60 부모/checkpoint/F 자산과 .cache의 봉인·manifest가 있어야 한다. 공개 충분통계 재생성은 원 예측의 신규 실행과 구분한다.
등록 시계와 deadline은 재개해도 초기화하지 않는다. 이후 날짜의 새 학습은 같은 캠페인의 시계를 재설정하는 방식으로 실행하지 않는다. 이미 완료된 결과와 현재 공개 표·그림의 재생성은 이 학습 상한과 구분한다.
배포 비용 fresh worker: `python -X utf8 -m research.incremental_latent_adaptation_20261019_v1.costs --request <sealed_request.json> --asset-root <local_asset_root> --output <new_receipt.json>`.
비용 요청에는 정확한 artifact SHA와 final prediction reference SHA가 필요하며 C_DECISIONS의 SEALED_BEFORE_V 봉인 전에는 V loader가 거부한다.
