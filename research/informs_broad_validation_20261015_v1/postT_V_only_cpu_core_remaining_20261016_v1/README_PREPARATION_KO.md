# 원 core56의 조건부 잔여29: fresh CPU V 전용 준비 도구

이 폴더는 **ROOT가 코드·경계 시험22개·실제 원 메타데이터29개를 대조하고 PLAN/REG를 봉인한 준비 도구**다. 실제 probe·fit·V·비용은 아직0개이며, 기존 실행 중15조건의 종료 증거가 없으면 실행을 거부한다. 작성·합성 경계 검사와 실제 모델 실행을 구분한다. ROOT가 원본 메타데이터를 대조하고 별도 `PLAN.json`과 `REGISTRY.csv`를 봉인하기 전에는 어떤 조건도 실행할 수 없다. 기본 CLI는 실험 입력을 읽거나 파일을 쓰지 않으며 NumPy·Torch·Task·모델·자식을 만들지 않는다.

원 [core56 대응표](../diagnostics/postT_core56_scope_20261016_v1/CORE56_CORRESPONDENCE.csv)의 `remaining_after_all_planned8_success_SCENARIO_ONLY=True`인 29개 원 ID만 대응한다. 현재 실제 core19/56, 잔여37과 구분한다. 실행 중인 15개 fresh fit이 모두 성공했을 때 새로 채우는 원 core8개를 가정하면 27/56과 잔여29가 된다는 **조건부 범위**다. 이 도구가 29개를 준비했다는 사실은 현재 15개나 새 29개의 완료 증거가 아니다. 원 E2/E3/E4, Electricity 공유 K, 추가 seed·subset·cutoff, M5 등 MASTER의 다른 잔여도 그대로 남는다.

## 원 조건과 CPU 변경

원 `runs/<initialID>/SOURCE_CONTRACT.json`의 spec·data·원 source6와 고정 PCA PLAN의 네 자료 계약을 읽는다. source39, 원 REG/selection/점수/비용 등을 포함한 immutable9, foundation revision·tensor SHA, 실제 instance-normalization 소스를 대조한다. B0/B2/B4, AdaPTS port, LoRA 모두 원 `fit_base.fit_base`를 사용한다. `fit_control`, 중단 체크포인트, `deepen`, 원 선택 gate를 우회하는 실행은 없다. 새 post-T 개발 namespace에서 fresh 초기화만 허용하며 원 GPU 장부를 수정하지 않는다.

spec 변경은 새 ID·stage·provenance·CPU·후보 epoch `[0.25,1,3]`와 Electricity/PeMS microbatch에 한정한다. Beijing/Jena는 원 micro4를 유지한다. Electricity/PeMS는 **micro1 후보**, effective16·원 순열·학습 손실·seed·원 LR·epoch 노출·optimizer update 수를 유지한다. 부동소수점 누적 순서가 달라 bitwise 동등 학습을 주장하지 않는다. LoRA의 등록 K=0과 실제 foundation 입력 C를 구분하고 LR=0.0001을 유지한다. 나머지 LR=0.001, F row128, `checkpoint_inputs=False`다. K=C의 원 identity 초기화도 원 API에 맡긴다.

micro1의 실제 3GiB 준수 여부와 전체 소요시간은 미측정이다. 예상 RSS만으로 BLOCKED 처리하지 않는다. 향후 probe 실패 뒤 checkpointing을 검토하려면 별도 명시 조건과 provenance가 필요하다. 원 checkpoint=True는 backward 재계산으로 physical F 호출이 증가하므로 현재 False 결과와 카운터를 혼합할 수 없다.

## 실행 순서와 경계

1. `propose_core_remaining_plan.py --write-proposal`은 새 private 디렉터리에만 `DRAFT_PLAN.json`, `REGISTRY_PROPOSAL.csv`를 만든다. ROOT가 전체 검토 후 status만 `SEALED_BY_ROOT`로 바꾸고 새 public PLAN/REG를 exclusive 생성한다. ROOT seal metadata를 추가할 경우 허용 key는 `ROOT_seal`, `ROOT_seal_unix`다.
2. ROOT는 기존 15-fit·검산·정밀비용 및 관련 owned launcher/runtime 전부의 실제 종료를 확인한다. 별도 SHA 고정 closure JSON의 형식은 `{"status":"ALL_PRIOR_CPU_OWNERS_CLOSED","remaining_owned_processes":[],"owned_bindings":[{"pid":...,"create_time":...}]}`다. runner는 exact PID/create_time가 아직 살아 있거나 조회 불명확하면 실행을 거부한다. 이 proof는 ROOT가 실제 관측으로 만들며 도구가 추정 생성하지 않는다.
3. 조건별 `probe`는 원 전체 A50 legal-origin 손실 정규화로 3 effective update·48 sample을 학습한 새 모델을 폐기한다. B0/B2/B4/AdaPTS는 **F module 전용 full-backward hook**의 실제 입력 gradient, LoRA는 adapter parameter gradient와 hooks close·base hash 복구를 검사한다. F 입력의 일반 z gradient만으로 분기 통과를 주장하지 않는다. frozen F parameter gradient는 없어야 한다. probe artifact·optimizer·RNG는 fit에서 재사용하지 않는다.
4. `fit`은 동일 ID의 SHA 고정 actual probe RESULT가 있어야 한다. 원 A50 search0/.25/1/3 → A_stop 선택 → fresh A60 selected-epoch refit → 전체 V만 실제 실행한다. 선택0 refit도 유효하다. search3 완료가 수렴이나 selected refit3을 뜻하지 않는다. 조건의 기존 run/prediction/claim이 있으면 보존하고 거부한다. 실패 조건을 rename·재개·덮어쓰지 않는다.
5. `rescore`는 원 독립 FP64 auditor로 저장 full V를 검산한다. `replay`는 원 route replay로 6 V origin을 CPU 재추론한다. 다른 28개의 미완료가 선택된 유효 조건의 검산을 막지 않는다. exact REG byte alias만 새 EXREG로 생성할 수 있다. 기존 replay 재사용은 fresh CPU 추론으로 부르지 않는다. 원 공차 atol=1e-5/rtol=1e-4를 유지한다.

전체 V는 Beijing/Jena 43×H48, Electricity 50×H48, PeMS 148×H12다. actual logical/physical streams는 V origins×실제 K(LoRA는 C), row calls는 각 microbatch의 `ceil(batch*K/128)` 합이다. 원 저장 prediction cost의 F_calls=1은 bundle bookkeeping이므로 실제 row-call 증거로 사용하지 않는다. replay도 micro1/4와 variable C/K에 맞춰 실제 카운터를 검산한다.

## ROOT 호출 형태

환경 root는 사용자가 지정한다. 아래의 `$R`, `$C`, `$A`, `$PY`, `$PLAN_SHA`, `$RUN_ID`, `$CLOSURE`, `$CLOSURE_SHA`, `$PROBE`, `$PROBE_SHA`, `$DEADLINE`, `$NEW_PRIVATE`, `$NEW_PUBLIC_DIAG`는 ROOT가 실제 존재·SHA·종료 증거를 채울 값이며 실행된 명령이라는 주장이 아니다.

```powershell
& $PY -B "$R/postT_V_only_cpu_core_remaining_20261016_v1/propose_core_remaining_plan.py"
& $PY -B "$R/postT_V_only_cpu_core_remaining_20261016_v1/propose_core_remaining_plan.py" --write-proposal --public-root $R --cache-root $C --asset-root $A --output $NEW_PRIVATE
& $PY -B "$R/postT_V_only_cpu_core_remaining_20261016_v1/core_remaining_runner.py" --execute --public-root $R --cache-root $C --asset-root $A --plan-sha $PLAN_SHA --run-id $RUN_ID --mode probe --deadline $DEADLINE --closure-proof $CLOSURE --closure-sha $CLOSURE_SHA --output $NEW_PUBLIC_DIAG
# fit은 위와 같은 봉인 인자에 --mode fit --probe-result $PROBE --probe-sha $PROBE_SHA 사용
# rescore/replay도 각 --mode와 새로운 output을 지정; T mode는 존재하지 않음
```

CPU 전용 환경은 CUDA_VISIBLE_DEVICES=-1, OMP/MKL/OPENBLAS_NUM_THREADS=2, HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE=1, INFORMS_ASSET_ROOT=$A, INFORMS_CACHE_ROOT=$C/새namespace다. GPU authorization 환경은 없어야 한다. ROOT는 기존 CPU group supervisor로 조건당 RSS3GiB·CPU2·global 최대2slot/합계6GiB를 집행한다. runner 자체 watcher는 자기 RSS·deadline 초과 때 자기 process만 exit98하며, 최종 RESULT 없이 STARTED/RESOURCE_STOP만 남을 수 있다. ROOT가 실제 runtime/launcher exit와 childtree closure를 남겨야 한다.

원 combined C+R 30GiB, 추가 reserve256MiB, deadline1792222550 안에서만 admission한다. 다른 작업 종료·패키지 설치·큰 backbone 학습·자동 Git 동작은 없다. 원 safeFIT loader가 허용 archive의 T byte를 물리 역직렬화할 수 있다는 이력은 그대로 기록한다. 요청한 학습·선택·평가는 A50/A_stop/A60/V만이며 **새 T target 계산·재선택은 없다**. post-T V 전용 단일 seed 개발의 결과를 독립 일반화 검증으로 확대하지 않는다.

현재 이 폴더의 합성 검사 범위는 메타데이터·경계 함수이며 실제 학습·수렴·예측 PASS·RSS 준수·소요시간 증거가 아니다. 명시한 두 모듈의 stdlib 검사 **22개가 실제 통과**했다. 원 전체 suite를 실행했다는 주장은 아니다. 실제 원 source39/immutable9와 29개 SOURCE_CONTRACT를 읽는 메타데이터 대조는 29/29 compatible이었고, ROOT가 [PLAN](PLAN.json)과 [REGISTRY](REGISTRY.csv)를 작성·봉인했다. 실제 probe·fit·V·비용은 여전히 0이다. 기본 CLI를 별도 fresh Python에서 확인했으며 NumPy/Torch/psutil/data_contract/fit_base는 import되지 않았다. CPU 비용 실행기는 별도 ROOT 소유이며 이 runner는 비용 숫자를 만들지 않는다.

```powershell
# 이 폴더를 working directory로 지정한 명시 합성 검사; 원 root pytest 실행 금지
& $PY -B -m unittest -v test_propose_core_remaining_plan test_core_remaining_runner
```
