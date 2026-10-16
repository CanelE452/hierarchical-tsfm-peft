# Beijing 원 multi-K 저장점 재개: 실행 전 준비

현재 상태는 **준비 코드만 작성, 실제 재개·학습·추론·비용 측정 0**이다. `PLAN_DRAFT.json`은 실행 승인이 아니며, ROOT가 독립 검토하고 남은 원 예산 안에서 별도 `PLAN.json`을 봉인·입장시켜야 한다. 기존 큰 큐나 다른 조건을 시작하지 않는다.

## 무엇을 이어가는가

원 `postT_V_only_cpu_multik_20261016_v1`의 Beijing B0 두 부모만 대상으로 한다.

- `postT_cpu_beijing_E4_ORDERED_MULTI_K_k11_s92601_v1`: 학습 K 순환 `[3, 6, 11]`.
- `postT_cpu_beijing_E4_MAXK_PCA_TRAINED_k11_s92601_v1`: 학습 K `[11]`.

원 run ID, spec 전체, `stage_label`, provenance, 가족 선택 lock, 원 배열 fingerprint를 config에 넣는 axes와 checkpoint contract를 그대로 유지한다. 새 namespace가 운송·재개 영수증을 소유한다. 새 run ID로 checkpoint를 고치거나 계약을 느슨하게 만들지 않는다. 특히 max-only의 `[11]`을 없애면 K=C 초기화가 identity로 바뀌므로 허용하지 않는다.

두 부모 모두 원 search의 저장점은 0/.25/1/3 epoch, 마지막 1868 update·29880 exposure·선택 epoch 3이다. 원 refit 저장점은 0 update·0 exposure의 `checkpoint_000000.pt`이다. 원본과 중단 전 백업 각각 다섯 파일의 byte 수·SHA를 확인한 뒤 새 private run 폴더에 같은 basename으로 복사한다. 기존 파일을 이동하거나 덮지 않는다. 원 SOURCE_CONTRACT/RUN_CONTRACT도 byte 그대로 복사한다.

중단 직전 미저장 진행량은 `UNKNOWN_NOT_LOGGED`이다. 이 코드는 임의의 중단 순간부터의 완전한 복구가 아니라 **마지막 정상 refit CP0에서의 재개**를 준비한다. sampler의 NumPy Generator와 Torch CPU RNG는 원 checkpoint에 있으나 Python/NumPy 전역 RNG는 없고, 중단 없는 실행과 bitwise 동일함을 실제로 검증한 결과도 없다.

## 단계와 판정

1. 원 `_stage`에 search의 A50 끝 10519, A_stop `[10519, 12622]`, 원 epoch 목록을 넘긴다. 복원된 완료 endpoint는 전부 건너뛰어야 한다. F의 네 counter 증가, search checkpoint/event 및 byte 변경이 모두 0인지 검사한다. 새 optimizer update나 A_stop 추론은 허용하지 않는다.
2. 별도 initialization 검사를 한다. 원 `training.verify_contract`가 확인하는 다섯 필드 밖의 `initialization_sha`도 확인한다. search `a878e0…709a7`, refit `88eb4e…914d`가 일치해야 한다. 원 _stage를 수정하지 않는다.
3. 복사한 refit CP0를 원 `_stage`가 복원한다. A60 끝 12622, 원 선택 `[3]`, 12063 origin, 36189 exposure, 2262 update가 완료 기준이다. 부족하면 `BUDGET_LIMIT_PRESERVED`로 남긴다. 원본을 다시 시작하지 않는다.
4. 기존 CP extra의 curve/선택/누적 F 기록을 복원하고 새 namespace에만 RECEIPT/CURVE/selected bundle을 쓴다. search의 `seconds`는 이번 복원 호출 시간이다. 과거 search 전체 wall로 해석하지 않는다. 과거 저장된 segment 시간·누적 F, 이번 refit 비용, 미확인 손실을 구분한다.
5. `predict`는 두 부모에서 K3/6/11의 여섯 출력을 각각 독립 명령으로 실행한다. 원 `deployment.Route` 생성자와 canonical fit end 12622를 사용한다. shared와 max-only/posthoc 부모를 구분한다. 기존 nested PCA 세 결과는 다시 만들지 않는다.
6. 각 출력은 full V43, H48, C11, 실제 F 입력 `43×K`를 확인한다. 보관된 예측은 별도 no-Torch `rescore`에서 FP64로 원 통계·점수와 대사한다. `replay`는 실제 원 Route와 선택 checkpoint로 6 origin을 4+2로 재추론한다.
7. 해당 출력의 성공한 rescore/replay RESULT SHA와 새 quiet 증명 뒤에만 B1/B8 비용을 한 조건씩 측정한다. 원 프로토콜은 warmup 10, timed 20, 별도 stage 20이며 모델 추론의 전체 경로를 측정한다. 비용 실행은 예측 NPZ/Z/예측 cache를 읽지 않는다. 비용과 학습을 겹치지 않는다.

이 두 부모를 이어가는 것만으로 독립 K3/6/11 학습 대조가 완료되지는 않는다. 여섯 V 출력도 새 데이터 일반화 검증이나 수렴의 증거가 아니다. 원 선택이 마지막 epoch이므로 완료되어도 학습 상태는 `UNDERTRAINED_OR_BUDGET_LIMIT`로 표시한다.

## 데이터와 자원 경계

실제 숫자 해석은 검토된 prefix loader로 Beijing 원 자료의 0:16830 행에 한정한다. 원 full Task의 data/array SHA는 **상속값, 재계산 아님**으로 명시하고 실제 prefix fingerprint·A50/A60 scaler·V43 origin/target/mask를 별도로 검사한다. 압축 파일 전체의 opaque SHA/ZIP 버퍼가 tail byte를 읽을 수 있으나 보호 T의 숫자를 파싱·평가·선택하지 않는다. `fit_base.fit_base`, `ordered_pilot.predict_prefix`, prediction_cache의 전체 Task/split 재계산 진입점은 호출하지 않는다.

CPU2/worker, RSS3GiB/worker, 모델 최대2·합계6GiB, GPU0, 기존 저장30GiB+256MiB 여유, worker 마감 Unix1792222550을 유지한다. 부모 ROOT의 실제 PID 생성 시각 기반 supervisor가 이를 강제해야 한다. 실행기 자체는 worker 수를 추측하거나 다른 프로세스를 종료하지 않는다. soft deadline에 원 _stage가 optimizer 경계 checkpoint를 남기도록 ROOT가 admission과 마감을 정한다. 강제 종료 시 RESULT가 없을 수 있으므로 ROOT가 원 CP와 새 partial의 실제 상태를 보고해야 한다.

## 명령 형식

기본 호출은 0 input read·0 write·0 NumPy/Torch import이다. 아래 `<...>`는 ROOT가 현재 자원과 파일을 확인한 뒤 채우는 값이다. DRAFT에 `--execute`를 적용하면 거부된다. ROOT는 별도 새 PLAN의 SHA와 runner SHA를 봉인하고 원 DRAFT를 보존한다.

```text
python -X utf8 -u -B multik_resume_runner.py

python -X utf8 -u -B multik_resume_runner.py --execute --root-triggered
  --public-root <campaign-public-root> --cache-root <campaign-cache-root>
  --asset-root <asset-root> --plan-sha <new-sealed-PLAN-sha>
  --run-id <one-original-parent-ID> --mode resume
  --deadline <live-original-budget-deadline> --output <new-namespace/diagnostics/fresh-output>
```

`predict/rescore/replay/cost`에서는 `--run-id`에 여섯 output alias 중 하나를 사용한다. `cost`는 추가로 `--batch 1` 또는 `8`, `--quiet-proof/--quiet-sha`, `--rescore-proof/--rescore-sha`, `--replay-proof/--replay-sha`가 필요하다. ROOT가 namespace cache를 준비하고 원 CPU2/offline/GPU 비가시 환경과 단독 비용 창을 강제한다. 자동 큐·별도 child 실행은 없다.

## 검증 범위와 남은 점

합성 stdlib 시험은 계약 변형·원 ID 변경·max grid 삭제·초기화 차이·원/백업 byte 변조·복사 덮기·경로 탈출·search 추가 F/event/checkpoint·불완전 refit·prefix identity 바꾸기·model config 변조·cost 선행 증명 누락을 거부하는지 확인한다. 기본 fresh-process 호출도 확인한다. 이 시험은 실제 tensor optimizer 복원, 원 모델 학습, 실제 예측 parity, 비용, 수렴을 입증하지 않는다. ROOT의 전체 코드 검토, 실제 실행 후 별도 수치 검산과 checkpoint 재추론이 남아 있다. 현재 이 adapter를 실행·게시 완료로 계산하면 안 된다.
