# 재현 안내

과학 실행 계획 SHA: `cab1bbc423e7ae7e5cf7029443a7be9a4e01246d76ec6af3a16d72b16fa7693f`. 보고서 producer SHA: `dfa332fa872260d6277b0a79c5bc2e17606b86d796f91524c4315bf4cac1d575`.

공개 CSV/JSON에서 표·그림만 별도 경로에 재생성:

```powershell
python -m research.adapter_improvements_20261018_v1.reporting --public-input reports/adapter_improvements_20261018_v1 --output .cache/adapter_improvements_public_replay
```

위 명령은 F/model/private NPZ를 열지 않는다. RUNS/COMPARISONS/COST/CHANNEL_HORIZON_STATS/SAME_UPDATE/PAIRED_ORIGIN_EFFECTS 및 figure source CSV만 읽는다. output을 원 report 폴더와 다르게 지정한다. 정확한 full precision 데이터·레이블을 대사하며 PNG/SVG renderer/font 버전 차이는 별도 범위다. 게시 완료 여부는 이 로컬 build가 판단하지 않으며 실제 remote commit·payload 근거는 별도 REMOTE_PROOF.json을 따른다.

CHANNEL_HORIZON_STATS.csv의 6열 통계는 RUNS.csv의 row_id에 조인한다. metadata를 통계 행마다 반복하지 않으며 producer_row_id와 실제 source hash는 RUNS에 남긴다. 과거 선택 결과 59행과 과거 공통 endpoint 88행은 contextual inventory이고 신규 실험 수가 아니다.

등록 runner의 실제 인터페이스:

```powershell
python -m research.adapter_improvements_20261018_v1.run --help
python -m research.adapter_improvements_20261018_v1.run --mode verify --repo . --asset-root PRIVATE_ASSET_ROOT --cache-root PRIVATE_CAMPAIGN_CACHE --report-root reports/adapter_improvements_20261018_v1 --plan reports/adapter_improvements_20261018_v1/EXPERIMENT_PLAN.json
```

이번 캠페인에서 사용한 실행 명령은 아래 두 개다. private 경로만 이식 가능한 자리표시자로 표시했다. 첫 명령은 원래 37개 job, 두 번째는 이미 등록된 seed92602의 CPU/GPU 비용 2개 job을 처리한다. 각 명령이 처리한 계획과 실제 과학 arm 완료 수는 다르며, 미수렴·의존성 미실행은 receipt에 보존한다. 명령별 실제 종료코드·환경·검증 범위는 `EXECUTION_VALIDATION.json`을 따른다.

```powershell
python -m research.adapter_improvements_20261018_v1.run --mode execute --asset-root PRIVATE_ASSET_ROOT --cache-root PRIVATE_CAMPAIGN_CACHE --report-root reports/adapter_improvements_20261018_v1 --plan reports/adapter_improvements_20261018_v1/EXPERIMENT_PLAN.json
python -m research.adapter_improvements_20261018_v1.run --mode execute --asset-root PRIVATE_ASSET_ROOT --cache-root PRIVATE_CAMPAIGN_CACHE --report-root reports/adapter_improvements_20261018_v1 --plan reports/adapter_improvements_20261018_v1/SUPPLEMENTAL_PLANS/COST_JENA_SEED2_v1.json
```

Python3.11의 기존 프로젝트 가상환경과 실제 기록된 NumPy/OpenBLAS·PyTorch·SciPy 버전을 사용했다. 기본 Anaconda 환경으로 바꾸면 과거 B03 FP32 검산과 달라질 수 있다. CPU BLAS/OMP thread는 2개이며 기존 동결 F snapshot을 offline으로 읽는다. 공개 replay가 private 학습·저장 예측 검산까지 재현했다는 뜻은 아니다. 실제 공개 재생성 결과는 `PUBLIC_REPLAY_VERIFICATION.json`, 시각 확인은 `VISUAL_INSPECTION.json`, 원격 payload 확인은 `REMOTE_PROOF.json`을 별도로 확인한다.

학습/비공개 검산은 동일 private assets·persistent ledger·봉인 plan을 필요로 한다. --only는 기존 등록 job ID만 선택하며 과학 source를 바꿔 같은 identity를 재사용하지 않는다. 인자 없이 runner는 안내만 출력한다. 미완료 명령·의존성·예상시간은 RESUME_QUEUE.json에 있다.

현재 독립 검증 상태는 PASS다. 구현 테스트 통과, 저장 예측 FP64 재검산, whole-path parity, 공개 그림 재생성, main 게시를 각각 구분한다. 과거 test 통과 개수를 이번 새 실험 검증 수로 합치지 않는다.
