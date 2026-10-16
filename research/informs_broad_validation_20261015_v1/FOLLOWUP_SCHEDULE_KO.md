# 후속 실행 순서와 자원 경계

`train_followups.py`는 현재 initial integration controller가 **종료된 뒤** ROOT가 실행하는 순차 실행기다. 기존 PID를 기다리거나 종료하지 않는다. 실행 중인 GPU owner가 있으면 새 child를 시작하지 않는다. controller 자체는 Torch·원 데이터·모델을 로드하지 않으며 outer GPU lease를 갖지 않는다. 각 postfit/deepen/legacy_queue child가 자신의 기존 resource guard와 lease를 사용한다.

```text
python train_followups.py --stage all --execute
```

실제 실행에는 `--execute`가 필요하다. 이를 빼면 stage별 명령과 누락 이유를 기록하는 plan-only다. ROOT가 기존 실행과 같은 `INFORMS_ASSET_ROOT`·`INFORMS_CACHE_ROOT`를 전달해야 한다. Python executable과 전체 환경을 이어받고 CPU thread 변수는 2로 유지한다. 공개 명령에는 로컬 개인 절대경로 대신 `{ASSET_ROOT}`·`{CACHE_ROOT}`를 남기고, 실제 child argv에서만 해석한다.

## 실행 순서

1. **recovery**: Beijing `predict → combine → ordered(.25) → controls(.25)`. 이전 실패 command에 대한 복구 시도이며 종료코드 0을 과학 실험 완료로 세지 않는다. 하위 실행기의 완료 artifact 검증·재사용을 유지한다.
2. **e0**: 기존 등록 unresolved·matched control에 대해 endpoint4800을 `legacy_queue.py run --include-controls --execute-training`으로 실행한다. 이 묶음의 total-GPU 상한은 `min(34200, 해당 stage 시작의 closed GPU usage + 7200)`이며 reserve0, 원래48시간 wall을 유지한다. 기존 endpoint/실패/부분 checkpoint는 queue가 검증하고 보존한다. old 모델의 초기 fidelity는 `RESTARTED_NOT_EXACT_RESUME`이며 새 queue 실행을 원 학습의 정확한 resume라고 바꾸어 말하지 않는다.
3. **deepen1**: Beijing·Jena·Electricity·PeMS08의 G를 먼저 시도한다. 이어 B0/B2를 같은 K의 한 child command에 묶어 quarter 전체 자료 → half 전체 자료 → C 전체 자료 순서로 시도한다. B4 quarter/C, 나머지 준비된 minimum E1(LoRA·ADAPTS·direct temporal·추가 등록 K 등), 같은 K의 causal/bidirectional lag pair quarter→half→C, 준비된 E4 shared/max pair 순서다. Robin main deepening은 포함하지 않는다.
4. **deepen3**: 준비된 epoch1 부모만 사용한다. Beijing·Jena의 G와 B0/B2 matched K pair를 먼저, 그다음 PeMS08·Electricity의 같은 비교를 시도한다. ID는 `원ID__e1__e3`이며 원 .25 부모에서 별도의 `원ID__e3`를 만들어 같은1epoch 구간을 다시 실행하지 않는다.
5. 자원이 남으면 E0 endpoint9600→19200. 직전 endpoint가 완료되고 selected/last SHA가 맞는 matched group만 명시적으로 열거한다. 이미 열거한 group의 control을 다시 확장해서 불완전한 다른 group을 끌어오지 않는다.

E1/E3 deepening command 뒤에는 관련 자료의 `postfit --phase predict`를 넣어 실제 V 캐시를 남긴다. E4는 기존 `deepen.py`가 같은 artifact의 prefix V 추론을 직접 수행한다. 그 뒤의 일반 postfit predict는 E1/E3 결과도 다시 검증·재사용한다.

`--stage recovery`, `e0`, `deepen1`, `deepen3`, `all`을 지원한다. `deepen3`에는 이후 가능한 E0 다음 endpoint 시도도 포함된다. all은 stage가 끝난 뒤 실제 새 receipt를 다시 읽어 다음 stage의 명령을 확정한다. plan-only에서는 아직 생기지 않은 후속 receipt를 추정하지 않으므로 다음 stage가 MISSING으로 남을 수 있다.

## 실제 부모와 중복 실행 방지

registry의 primary·screen seed92601·기본 ID만 사용한다. 기존 extension 행이 같은 family/K의 원래 행을 덮어 선택되지 않도록 `__e...`·prefix alias를 기본 ID lookup에서 제외한다. 각 부모의 COMPLETE_VALID receipt, SOURCE_CONTRACT, selected artifact의 실제 SHA를 대조한다. 점수·개선 여부는 읽거나 실행 gate에 쓰지 않는다.

원래 최초 세 Beijing 부모의 scientific source가 현재와 다르면 기존 `deepen.py`의 fresh restart/provenance 분기를 그대로 사용한다. controller가 source equivalence를 추정하거나 부모 optimizer를 임의 이식하지 않는다.

같은 목표 extension이 이미 완료됐다면 SHA를 확인해 다시 fit하지 않는다. pair의 한쪽만 완료됐다면 아직 시작하지 않은 반대쪽만 명시적 child argv에 넣고 완료된 비교 대상을 기록한다. 실행 직전에 partial directory가 발견되거나 저장 계획 이후 일부만 완료된 경우는 자동으로 덮거나 재개하지 않는다. 새 계획을 만들거나 별도의 명시적 partial-resume 절차가 필요하다. 준비되지 않은 pair는 이유를 남기고 독립적인 다른 비교를 계속한다.

## 계획·재개·영수증

각 stage의 `execution_batches/followup_plan_*.json`에 human-readable argv, 소스 SHA, 부모 증거, 누락 이유와 시작 시 자원량을 **새 child 전에** 기록한다. child별 PRELAUNCH와 exit 영수증도 새 파일이며 stdout/stderr 원문은 private `followup_commands/`에 둔다. 공개 영수증은 log SHA·시간·종료코드만 담고 개인 경로·환경 인증값·원 데이터는 내보내지 않는다.

저장 계획의 인덱스를 재개하려면 해당 계획을 지정한다.

```text
python train_followups.py --resume-plan execution_batches/followup_plan_deepen1_<stamp>.json --from-command 8 --execute
```

`--from-command`만으로 새로 계산한 목록의 앞부분을 건너뛰지 못한다. 이미 완료된 fit 때문에 새 목록의 인덱스가 달라질 수 있기 때문이다. 재개 시에도 같은 소스와 부모 artifact인지 다시 대조한다. partial target을 발견하면 실행하지 않고 명시적인 상태를 기록한다.

## 상한의 정확한 의미

closed GPU usage가 `43200−9000=34200초` 이상이면 **새 training child를 시작하지 않는다**. E0에는 위의 더 작은2시간 subcap을 전달한다. 원48시간 wall이 지나면 inference를 포함한 다음 command를 시작하지 않는다. 중단 시 아직 남은 명령을 DEFERRED_RESOURCE 영수증에 기록한다. 직전 fit의 V 저장 command는 training 시작 기준과 구분한다.

9000초는 이 controller의 **새 command 시작 기준**이며, 긴 학습을 수행하는 `deepen.py` 내부에도 같은9000초 hard reserve를 적용한다. LR 대조·G A+M 연장도 같은 reserve를 사용하고 E0에는 위의 더 작은 subcap을 전달한다. `postfit.py`의 저장 예측·결합 작업은 기존3600초 reserve를 유지하므로 전체 파이프라인에 엄격한9000초 잔여를 보장한다고 주장하지 않는다. 별도 비용 controller는 기본 누적 ceiling39600초를 사용해 최종 재추론·T에3600초를 남긴다. 종료코드0·명령 목록 종료·epoch1 종료를 전체 연구 완료, 충분 학습 또는 수렴으로 해석하지 않는다.

T 평가·선택 잠금·비용 측정·자동 commit/push를 실행하지 않는다. 기존 실행·기존 결과를 종료하거나 덮지 않는다.

## CPU 확인

임시 synthetic registry/receipt/artifact로 4자료×quarter/half/C의 B0/B2 pair12개와 G·lag pair를 포함한 ready training group28개를 만들고 순서를 확인했다. 완료 target 재실행 제외, partial 보존, 원 registry와 extension 행의 구분, `__e1__e3` 부모 연결, E0 cap 및 복구4명령을 확인했다. mocked child 실패 뒤 독립 V 저장 command 진행과 다음 training의34200초 경계도 확인했다. Torch import·실제 child·GPU·원 데이터·실제 registry 수정은0이었다.
