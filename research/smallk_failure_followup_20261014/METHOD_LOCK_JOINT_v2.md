# Joint E/D 목적 비교 잠금 v2: 자원 검사 cadence 보정

상태: **SOURCE_READY / ACTUAL_NOT_STARTED**. root가 소스·검사를 검토한 뒤 명시 실행한다. 이 문서는 새 비교의 고정 조건이며 결과가 나쁜 경우 조건을 사후 변경하지 않는다.

목적은 동결 E/F의 출력 head 재적합에서 해결되지 않은 성능 격차에 대해, E/D를 함께 예측 목적에 맞춰 갱신하면 차이가 생기는지 확인하는 것이다. 기존 중단 campaign을 재개하지 않고 새로운 output·optimizer·학습 기록을 만든다.

자료는 감사된 `input_bridge/old_conditions_actual_v1/RESULT.json`의 원 조건이다. Robin C17/K4·FIT256/TUNE64/기존 VAL30, Beijing C11/K2·FIT256/TUNE64/이미 노출된 EVAL146, L512/H48, 원 FIT mean/std·target/mask/origin을 유지한다. 보호 TEST와 새 EVAL 개봉을 하지 않는다. 두 평가 split은 개발 증거이며 독립 일반화를 주장하지 않는다.

각 자료·seed92601/92602의 **원 이미 Y-TUNE-selected MAIN CP**를 Y/RAW/MIXED 세 목적에 동일하게 복원한다. 원 CP file SHA·codec state SHA와 초기 FIT/TUNE 출력 parity를 검사한다. optimizer state를 새로 만들고 원 구조·현재 C anchor·K·hidden32·causal encoder를 유지한다. 엄격한 K-only 압축이 아니다.

```text
조건 수                 2자료 ×2원seed ×3목적 =12
공통 새 optimizer       AdamW lr0.001 weight_decay0 clip_global_norm1
새 업데이트 수          600; batch4
batch order             default_rng(seed).integers(0,FIT_count,(600,4)); 세 목적 동일
checkpoint              새 update0,100,200,400,600
Y                       실제 Y의 정규화 observed channel-macro MSE
RAW                     cached 원 RAW F(past) forecast의 같은 observed-mask MSE
MIXED                   0.5*Y loss +0.5*RAW loss
checkpoint 선택         세 목적 모두 실제 Y TUNE normalized MSE; 가장 이른 exact tie
주 지표                 normalized actual-Y MSE; MAE와 채널별 원단위 오차 함께 기록
F                       같은 로컬 Chronos-Bolt-small FP32/median0.5; weights 동결
갱신 파라미터           E와 D만; F input gradient는 실제 계산
fit 프로세스 cap        process birth 기준150초, RSS/allocatedVRAM8GiB
공유 상한               FollowupLimit으로 기존 Bi 잔여 GPU/CPU/campaign/storage 계속 사용
```

RAW teacher는 FIT의 과거만으로 이미 계산된 원 forecast다. 실제 미래 정답 oracle이나 protected TEST 출력이 아니다. 학습에는 FIT teacher만 사용하며 TUNE teacher 오차는 설명용 지표다. TUNE 선택 대상은 실제 Y를 유지한다. MIXED의 reported objective loss는 두 손실을 명시적으로 반씩 더한 값이다. FIT 실제 Y의 관측/비관측 finite 여부·비관측 nonfinite 수·채널별 observed count를 기록한다. 비관측 Y가 NaN이어도 backward를 오염시키지 않도록 잔차를 mask한 뒤 제곱한다. 유한 데이터에서는 원 macro_loss의 값·gradient와 같다.

새 실행 receipt에는 원 full-C teacher를 과거에 만들었던 비용이 들어가지 않는다. 현재 RAW 파일 hash/load/비교 시간은 따로 기록하지만, bridge에 원 FIT forecast 생성의 별도 timer가 없어 역사적 teacher 생성 CPU/GPU 시간은 unknown/null로 남긴다. 이미 계산됐다는 이유로 비용을 0으로 쓰거나 총비용 우위를 주장하지 않는다.

원 CP가 이미 Y로 선택됐으므로 new update0도 후보에 포함한다. update0이 선택되면 “새 joint 학습이 개선됐다”고 쓰지 않는다. u600 파라미터 갱신과 selected CP의 성능을 분리하고, checkpoint·FIT/TUNE 배열·최종 선택 전 split 실제 추론을 모두 보존한다. VAL/EVAL 최저 방법을 사후 선택하지 않는다.

새 update0/99/199/399/599에서 loss의 F future upstream gradient와 F branch의 input gradient를 따로 계산한다. 원 weights를 갱신하지 않은 실제 frozen F gradient, E/D의 유한·비영 gradient, u600 E/D 파라미터 변화, F 전후 두 hash와 parameter.grad 없음, fixed_basis 불변을 기록한다. 600실제 optimizer update·E/D 변화·비영 F input gradient를 모두 확인한 full_learning_proof_pass를 별도로 남긴다. 구조상 anchor에서 상쇄되는 D bias의 zero/FP32 반올림 수준 gradient를 실패로 오인하지 않는다.

원 600-update MAIN의 생산 JSON train_wall은 Robin20.30/20.47초, Beijing19.06/19.07초다. 새 fit는 모델 로드·초기 parity·gradient 증거·최종 추론·기록을 포함해 약35–60초, 전체12개 약500–900초의 GPU process wall을 예상한다. 이는 실행 결과가 아니라 추정이다. fit별150초 cap의 합1,800초와 원 공유 상한을 함께 적용한다. 추가 비용 timing은 학습/CPU 보고서와 겹치지 않는 별도 측정이다.

CLI는 한 fit만 실행한다. `$bridge`와 fresh tag를 root의 실제 완료 bridge로 지정한다.

```powershell
$bridge = 'E:/CODING/proj/hierarchical-tsfm-peft/.cache/smallk_failure_followup_20261014/input_bridge/old_conditions_actual_v1/RESULT.json'
.venv/Scripts/python.exe research/smallk_failure_followup_20261014/run_joint_objectives_v2.py --execute --dataset robin --seed 92601 --objective Y --input-bridge $bridge --tag joint_robin_y_s92601_v2
```

같은 형식으로 두 자료·두 seed·Y/RAW/MIXED 전부를 실행한다. 출력은 `.cache/smallk_failure_followup_20261014/joint_objectives/<tag>`이며 기존 output과 source를 덮어쓰지 않는다. FAILED/BUDGET_HELD의 실제 이유·부분 결과도 보존한다. 자동 commit/push·논문/PPT 작성은 하지 않는다.


## v1 보존과 v2의 변경 범위

v1 최초 Robin/Y/seed92601 fit는 process cap150초로 중단됐다. 실제510 optimizer updates와 u0/100/200/400 checkpoint, F 전후 동일 hash를 보존했다. GPU process wall150.941753초는 이미 원 공유 ledger에 청구됐다. v1 runner·lock·결과를 수정하거나 u400을 재개하지 않는다. v2의 모든 조건은 원 MAIN CP부터 새로 시작한다.

[JOINT_GUARD_TIMING_DIAGNOSTIC_v1.json](validation_v1/JOINT_GUARD_TIMING_DIAGNOSTIC_v1.json)의 실제 CPU 계측에서 full guard 다섯 표본의 median은0.3870044초, followup storage scan median은0.1500048초였다. 이는 guard의 CPU 비용을 확인한 값이며 GPU 학습 실패 원인의 전부를 확정한 증거가 아니다. v2는 guard 호출 빈도만 분리하고 호출 수·누적 wall을 실제로 기록한다.

예측 batch마다 local150초/RSS 확인과 기존 shared.Limit.check를 계속 수행한다. GPU process wall·campaign·RSS/VRAM·Bi storage guard는 유지한다. 추가 CPU 누계와 combined followup storage/disk를 포함하는 full FollowupLimit.check는10 optimizer updates, 각 checkpoint, CP/NPZ 쓰기 전후, 선택 출력 쓰기 전후, 최종 source 확인에서 수행한다. 새 source snapshot 전에도 full check를 수행한다. 생성자·종료 guard는 whole receipt에 포함되며 별도 explicit cadence 카운터에서는 제외한다.

Y/RAW/MIXED 수식·masked 손실·그라디언트 증거·batch order·원 CP·600updates·AdamW·clip·checkpoint 선택·허용오차·local150 및 모든 원 공유 상한은 v1과 같다. 독립 AST 검사는 숫자 계산 helper와 고정 상수의 동일성을 확인한다. v1 실패를 지우거나 상한을 늘리는 보정이 아니다. 원 실패 결과 SHA는0c123056900e4e2c4f3bc1cc6f853106145afe0b04a5a29ec250f6e6da7da6f5이며 guard diagnostic SHA는d6eace339b2e762440e5ea8b1f4937e7c476eee284bd1ece022f4a57bfd55370이다.
