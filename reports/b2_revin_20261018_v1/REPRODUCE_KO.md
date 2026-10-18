# 공개 표·그림 재생성

이 명령은 공개 CSV/JSON 입력과 선택적인 기존 그림 출처 메타데이터만 읽으며 모델·원자료·private NPZ·GPU를 사용하지 않는다. 실험의 재학습과 공개 집계의 재생성은 다른 작업이다. 프로젝트 Python 환경에서 실행한다.

```powershell
python -m research.b2_revin_20261018_v1.reporting --report-root reports/b2_revin_20261018_v1 --output-root .cache/b2_revin_public_replay
```

matplotlib Agg backend, 고정 SVG hash salt, timestamp 없는 PNG/SVG metadata를 사용한다. 동일 환경과 입력 bytes에서 표·문서·그림 bytes를 대사한다. 보고 generator는 실험 상태·선택·검산 입력을 수정하지 않는다.

입력 SHA256:

```text
RUNS.csv  5775d2d1dae8f06cd3ae09eaf873e2f4839266c776c6c4512a96ba615e70e365
COST.csv  8e41210ef53d7d08e85468f135f9191ea02d5cd8fa71c90de1efa339a21af22c
TRAINING_HISTORY.csv  27dac8bf630f147211294fb7fbec24ab8c63c54cb8e839d748ae739436bf9e7c
COMPARISONS.csv  e40088cb40e48218551ef34b5e181517f3ed4674e2612cc256e4d3eab2a8c8d5
ERROR_SUFFICIENT_STATS.csv  9d9a7e7698c405a781afe45af353c3a8748d747b5843ae5d5e3d0353619f7c21
PAIR_ORIGIN_DIFFS.csv  f8ec39f3aac1947050976977907d7f67bece73e3829a854bf6fd44efb6a0d4d8
CAMPAIGN_CONTRACT.json  b758cb50104b5a675600beae9f1d2e83fbf865ab613a19c9dc0ca85c31d5c011
DECISIONS.json  300850d51b5db0d09cc6cd367a90e09d02b51597c72d67c1787f8f9be74e0d8e
HISTORICAL_ARTIFACTS.json  40378ce410cf401155e2d48afb72cae29ef7dead99aeb4eeb10c72799269af8e
```

V는 반복 개발 평가로 이미 사용한 자료이며 독립 테스트 결과가 아니다. 새 선택에 M_cal/T를 사용하지 않았으며, 기존 T 노출과 pretraining overlap UNKNOWN의 범위는 원 계약을 따른다. 두 seed·고정 K의 결과를 모든 데이터·압축률의 일반화로 확대하지 않는다.
