# E0: 기존 12개 미해결의 이어학습 장부와 실행기

[확인] 기존 `REPORT_KO_v3.md`의 정확한 12개 ID를 `UNRESOLVED_RESOLUTION.csv`에 등록했다. 각각의 원 `fits/<ID>_2400.json`과 CPU `weights_only=True`로 읽은 작은 `last_2400.pt`를 대조했다. 12/12에서 마지막 가중치, AdamW 모멘트와 step 2400, carry 2400행 및 체크포인트 목록이 존재하고 원 영수증과 일치한다. 원 2400회 sampler hash도 12/12 재구성했다. 선택 CP는 일부 경로에서 2400보다 이르므로 **선택 가중치로 학습을 재시작하지 않고 마지막 가중치와 그 Adam 상태를 함께 사용한다.**

[확인] 과거 RNG/sampler state는 0/12에 저장되어 있다. 원 코드에는 scheduler가 없다. 따라서 출발 상태는 `RESTARTED_NOT_EXACT_RESUME`이다. 이는 가중치나 optimizer를 무작위로 초기화한다는 뜻이 아니다. 둘은 실제로 불러오며, 과거 RNG까지 일치하는 완전 재개를 주장할 수 없다는 구분이다. 아직 실행하지 않은 경로는 `PENDING_QUEUE`이고 준비 상태는 `READY_RESTARTED_NOT_EXACT_RESUME`이다. 예산 중단을 확인하기 전에 `DEFERRED_BUDGET`으로 바꾸지 않는다.

원 12개는 다음과 같다.

```text
primary_beijing_B0_r11_s92602
primary_beijing_B0_r2_s92601
primary_beijing_B3_r2_s92601
primary_beijing_B4_r2_s92601_without_seasonal_mix
primary_beijing_B4_r2_s92601_without_trend_mix
primary_jena_B0_r5_s92601
primary_jena_B0_r5_s92601_G_s92601
primary_jena_B0_r5_s92602
primary_jena_B2_r5_s92602
primary_jena_B3_r5_s92602
primary_jena_B4_r5_s92602_without_trend_mix
primary_jena_Current_G_s92601
```

## 실행 계약

이 E0는 **옛 primary A/B, L512/H48**의 개발 후속이다. 새 INFORMS A_fit/A_stop/M/V/T 및 L364/H12/H28 계약과 분리한다. 기존 B/C를 이미 관찰한 뒤의 추가 개발이므로 독립 확증이나 새로운 일반화 근거가 아니다. 실행기는 C/TEST를 채점하지 않는다.

`legacy_continuation.py inventory`와 `plan`은 원 배열이나 F를 불러오지 않는다. inventory는 원문/JSON/작은 adapter PT만 읽는다. `run --execute-training`만 원 FIT allowlist 로더와 frozen F를 호출한다. 이는 root의 순차 GPU queue에서만 실행한다.

```text
python -X utf8 -B research/informs_broad_validation_20261015_v1/legacy_continuation.py inventory --asset-root <asset-root>
python -X utf8 -B research/informs_broad_validation_20261015_v1/legacy_continuation.py plan --asset-root <asset-root> --include-controls
python -X utf8 -B research/informs_broad_validation_20261015_v1/legacy_continuation.py run --asset-root <asset-root> --id primary_jena_B0_r5_s92601 --endpoint 4800 --execute-training
```

`INFORMS_ASSET_ROOT` 환경변수로 `--asset-root`를 대신할 수 있다. 9600은 새 4800 last/carry, 19200은 새 9600 last/carry가 있어야 한다. ID 하나당 **새 Python subprocess**를 사용한다. `plan --include-controls`가 출력한 28개 고유 경로를 root queue에 등록하고, 같은 비교군을 같은 최종 endpoint로 실행한다. 개별 matched-control ID도 `run --id`에 사용할 수 있다. 원 12개만 실행하고 옛 대조군 점수와 공정한 최종 비교라고 하지 않는다. 28개 전부 19200까지의 추가 예산은 470400 updates이며, 이는 실행 완료 수치가 아니다.

## 보존·재개·선정

- 실행 전 old `SELECTION_LOCK.json`에 고정된 models/runtime/campaign/data SHA와 통계 JSON SHA를 확인한다. 원 모델 클래스를 그대로 import하며 파일을 패치하지 않는다. G 경로는 마지막 state의 frozen parent까지 복원한다.
- 원 sampler의 2400×4 prefix를 그대로 보존한다. 이후에는 `SeedSequence([seed,2400,224])`로 A legal-origin 전체의 고정 순열을 이어 붙인다. 4800/9600/19200은 공통 prefix이며 요청 endpoint가 앞 구간을 바꾸지 않는다. 순열·prefix SHA와 새 RNG seed를 영수증에 남긴다.
- 원 batch4, FP32, AdamW 원 learning rate/weight_decay0, gradient clipping1, mask가 적용된 complete-A channel-macro loss를 유지한다. optimizer.step이 시작·종료 update와 실제 일치해야 한다. F는 optimizer에서 제외되고 전후 state hash가 같아야 한다.
- 새 B 체크포인트는 400 update 간격이다. A는 원래 시간 균등 최대32 origins monitor다. 선택은 원 CP와 새 CP 전체의 `min(B_mse,update)`이다. 계속 학습할 상태는 별도의 마지막 state/optimizer/RNG/sampler로 저장한다. 새 구간의 후속 재개는 저장된 RNG를 복구한다.
- 마지막 CP가 best이거나 최근 B 3개가 엄격히 감소하면 `STILL_IMPROVING_UNRESOLVED`, 그 외에는 `CONVERGENCE_UNPROVEN`이다. 단지 예산 상한에 도달했다는 이유로 수렴/해결을 선언하지 않는다. 조기중단을 실제 구현·실행하지 않았으므로 `RESOLVED_EARLY_STOP`을 쓰지 않는다. `research_success=False`를 유지한다.

영수증은 새 campaign의 `legacy_e0/`, 체크포인트는 `<asset-root>/.cache/informs_broad_validation_20261015_v1/legacy_e0/` 아래에만 쓴다. `--cache-root`는 이 새 private cache 안에서만 허용한다. runtime/data의 ROOT를 모두 명시적으로 asset-root로 고정하고, data.HERE는 새 campaign root로 두어 원 allowlist의 새 경로 허용 범위를 그대로 유지한다. guard는 Python open 감사이며 OS sandbox는 아니다. 옛 NPZ 로더는 안전 archive를 통째로 역직렬화한 뒤 허용 FIT 행만 반환한다는 원 한계도 유지한다. 옛 source/영수증/가중치는 수정하지 않는다.

기존 endpoint 출력은 덮어쓰지 않는다. 실패는 새 `FAILURE_*.json`과 이미 생성된 출력에 보존된다. 중간 CP가 있는 실패를 자동 재학습하지 않으며, 원인과 별도 재시도 경로는 root가 판정한다. 실행 전역 lock은 새 cache의 E0 동시 실행을 막는다. 다른 GPU 작업과의 배제는 root queue 책임이다.

## 이번 검증 범위

[확인] CPU 합성 5 tests RED→GREEN: old2400/nested schedule, 경로/guard binding, ablation·G spec, 작은 checkpoint의 weights_only 저장/복원과 다음 sampler minibatch/RNG/Adam 다음 step 동일성, carry counter와 B tie-break. CPU inventory 12개 및 metadata-only 28개 대조군 queue plan이 통과했다. 실제 원 배열/F/GPU/이어학습 실행은 하지 않았다. 실제 endpoint 성공은 root가 생성하는 영수증과 checkpoint로 별도 확인해야 한다.
