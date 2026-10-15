# TRAIN 자산 확인 — 미실행 후속을 위한 읽기 전용 조사

작성일: 2026-10-15. **상태: SOURCE·JSON·파일 메타데이터 조사 완료 / 새 실험 미실행.** 최신 사용자의 마무리·push 요청에 따라 연구 확장과 새 학습을 시작하지 않았다. 이 조사에서는 실제 NPZ·Arrow·Parquet·CSV 데이터나 모델을 역직렬화하지 않았으며, F 실행·학습·원배열 검산을 하지 않았다. 아래 데이터 SHA는 기존 계약에 기록된 값이고, 이 조사에서 데이터 파일을 다시 해시한 값이 아니다. SOURCE와 계약 JSON SHA만 직접 계산했다.

## 확인된 자산과 시간 범위

**[확인·계약/파일 메타데이터] Robin:** `.cache/tsfm_peft_level_confirmation_v6_20260928/data/trainval.npz`가 존재하며 111,474 bytes다. 기존 계약 SHA는 `04523b70fa2df7b37514ab0f12e0cb25048c35fd120a11d4c6ab56e0ae464e14`다. [계약](../../level_matched_learned_ed_v18_20261006/data_contract_robin.json)의 `array_contract.trainval.x_shape`는 `[3312,17]`이고, `bounds.train`은 `[528,2520)`, `bounds.val`은 `[2568,3312)`다. 간격은 1시간이므로 TRAIN은 1,992시간이다. 이것은 TRAIN/VAL 결합 파일이며 TRAIN만으로 물리적으로 분리된 원시값 파일은 아니다. 보호된 TEST 값은 별도 `test.npz`에 있다.

[기존 로더](../../past_only_repr_diagnosis_20261012/common.py)의 150–187행은 TRAIN의 80% 경계로 FIT `[528,2121)` 1,593시간과 TUNE `[2121,2520)` 399시간을 만든다. 기존 FIT 256창의 origin은 1040–2073이며, 과거의 합집합은 `[528,2073)`, 정답의 합집합은 `[1040,2121)`다. 기존 TUNE까지 TRAIN 전체로 합쳐도 별도의 긴 신규 관측 기간이 생기는 것은 아니다. 이미 선택에 노출된 TUNE의 역할을 바꾸는 것이며 독립 검증으로 부를 수 없다. 창 수만 늘리는 것도 독립 시간 표본 증가와 같지 않다.

**[확인·계약/파일 메타데이터] Beijing Aotizhongxin:** `.cache/smallk_predictive_alignment_20261013/beijing_repair/raw_dataset/beijing_aotizhongxin/b04da438b2f331ac/fit_tune_payload.npz`가 존재하며 1,524,625 bytes다. 기존 계약 SHA는 `78063851dd3e65512e3e3ced22210839697553f2d652fc0be9513736af598277`다. 같은 폴더의 `contract.json`은 FIT `[0,21038)`, TUNE `[21038,28051)`, EVAL `[28051,35064)`를 기록한다. 채널은 11개, 간격은 1시간이다. 따라서 이 물리값 payload는 FIT+TUNE 28,051행을 보관하며, 원래 FIT만으로도 21,038시간이다.

Beijing 계약의 FIT origin 메타데이터는 256개, 처음 512, 마지막 20990이다. 현재 창들이 이미 넓은 FIT 시간대를 가로지르므로 추가 창을 뽑는 것을 그대로 ‘새로운 장기간 자료를 추가했다’고 해석하면 안 된다. 실제 합집합·간격·겹침은 후속 실행자가 배열로 따로 검산해야 하며 이 조사는 그 작업을 수행하지 않았다.

## 원시 관측값과 이미 적합된 통계의 구분

**[확인·SOURCE] Beijing은 원 단위 관측값과 mask를 별도 보관한다.** [newdata.py](../../smallk_predictive_alignment_20261013/newdata.py)의 436–465행은 `physical[:tune_end]`, `mask[:tune_end]`, `filled`, `x`를 분리하고 550–565행에서 저장한다. 689–743행의 `load_training`은 FIT/TUNE payload만 역직렬화하며 EVAL 정답을 열지 않는다. 다만 이 함수도 현재 full-FIT의 `x/filled/mean/std`를 함께 반환한다. 새 A-only 비교에서는 이 통계와 기존 E/D를 그대로 재사용하면 전체 경로가 A-only가 되지 않는다. 새 경계에 맞춰 `physical/mask`에서 scaler·결측 처리·E·D·특징 RMS·보정을 다시 적합해야 한다.

**[확인·SOURCE] Robin의 현재 safe archive에는 원래 물리값 배열이 없다.** [prepare_v6.py](../../tsfm_peft_level_confirmation_v6_20260928/prepare_v6.py)의 253–295행은 전체 TRAIN 평균·표준편차로 표준화한 FP32 `x`와 `finite/mean/std`를 저장한다. 원래 [standardize](../../tsfm_peft_development_20260926/data_prepare.py)의 99–110행은 관측값은 그대로 표준화하고 결측값은 평균으로 채운다. [common.py](../../past_only_repr_diagnosis_20261012/common.py) 156–165행은 `x*std+mean`으로 물리값을 FP32 역변환하고 원래 mask로 결측을 복구한다. 이는 기존 safe artifact에서 회수한 값이며 원 CSV의 무손실 복사라고 주장할 수 없다. 특히 A 이후를 포함한 원래 TRAIN 통계를 거친 FP32 표현이라는 계보를 유지해야 한다. A에서 통계를 다시 적합하더라도 이 역변환을 ‘아무 선행 변환도 없는 원시 A 값’이라고 부르지 않는다.

Robin 원 CSV `E:/CODING/proj/mltimeseries/data_external/bdg2_coarse_supervision_v1/raw/electricity.csv`도 메타데이터상 존재한다(174,239,039 bytes). 그러나 전체 기간 원본이므로 이번 조사에서 열지 않았다. 이를 새로 읽어 정확한 A-only 물리값을 추출하려면 TRAIN 이전/이내 행만 실제로 읽는 별도 스트리밍·종료 경계 계약이 먼저 필요하다. 기존 `prepare_v6.py` 전체를 재실행하는 것은 전체 원본과 평가용 배열을 함께 만들기 때문에 이 목적의 안전한 대체가 아니다.

## 로더의 부작용과 접근 가드

현재 [prepare_inputs.py](../../smallk_failure_followup_20261014/prepare_inputs.py)의 `install_protected_test_guard`(748–770행)는 두 명시 경로와 파일명 `test.npz`·`jena_test.npz`의 Python `open`을 막는다.

```text
.cache/tsfm_peft_level_confirmation_v6_20260928/data/test.npz
.cache/tsfm_peft_practical_controls_v7_20260928/data/jena_test.npz
```

**이 가드만으로 모든 미래 자료의 접근이 차단되는 것은 아니다.** Beijing의 `eval_future_target_payload.npz`, 전체 ZIP/CSV, 다른 이름의 데이터 파일까지 포괄적으로 막는 가드는 아니다. TRAIN-only 후속을 실제로 실행한다면 위 기존 TEST 가드를 설치한 뒤, 허용한 trainval/FIT-TUNE 파일의 정확한 경로만 데이터 역직렬화 allowlist에 넣고 Beijing EVAL 정답·전체 원본의 접근도 별도로 차단해야 한다. 관련 파일의 존재 확인과 데이터 값 역직렬화는 구분해서 기록해야 한다.

[newdata.py](../../smallk_predictive_alignment_20261013/newdata.py)의 `_contract_from_registry`/`_verify_contract_files`(650–686행)는 `include_eval_target=False`일 때 EVAL 정답의 존재만 확인하고 값이나 해시는 읽지 않는다. 하지만 EVAL 과거 입력은 해시한다. [newdata_repair.py](../../smallk_predictive_alignment_20261013/newdata_repair.py)의 `load_training`(116–136행)은 `_ensure_eval_past_physical`도 호출한다. 기존 repair 파일이 없거나 손상되면 재파싱 경로로 갈 수 있으므로 엄격한 TRAIN-only 후속에서 무조건 이 래퍼를 호출하지 않는 것이 좋다. `prepare(..., source_path=.../allow_download=True)`도 실행하지 않는다.

따라서 권장 가능한 미실행 경로는 **기존 계약 JSON과 source SHA를 고정하고, 허용 NPZ 한 개의 필요한 키만 직접 읽은 뒤 허용 시간으로 즉시 잘라 새 경계별 통계를 적합하는 별도 로더**다. NPZ 자체는 결합 아카이브이므로 파일의 일부 키를 읽는 것과 특정 행만 압축 해제하는 것은 다르다. TRAIN 이후 값을 물리적으로 전혀 역직렬화하지 않았다고 주장하려면 이를 지원하는 별도 분리 자산이 필요하며, 현재 조사에서는 그런 자산을 확인하지 못했다.

## 조사 결론과 미실행 사항

**[판단] 실제 재적합에 사용할 수 있는 기존 safe 자산은 있다.** 특히 Beijing에는 긴 FIT 구간의 원 단위 관측값과 mask가 보관되어 있다. Robin은 기존 safe archive로 제한한 조건부 재적합은 가능하나 기간이 짧고 FP32 역변환 계보가 남는다. 단순히 새 ridge를 더 적합하는 것만으로 학습 시간 부족·통계 이동·예측 표현 문제를 분리했다고 할 수 없다.

이번 단계에서 새 로더, 새 scaler/E/D, A/B/C 경계, 신경망 학습, F 추론, 예측 배열, 새 비용 결과는 **생성하지 않았다**. ‘후속 실험 완료’나 ‘새 독립 일반화 검증’으로 인용하면 안 된다. 실제 실행 여부와 비용은 기존 실험 보고서와 별개다.

직접 확인한 SOURCE/JSON SHA256:

```text
research/past_only_repr_diagnosis_20261012/common.py
  1bb0dc92a37d3bb3af525c2e72b62ab52430237ac4653086da6bf361a8c353c2
research/smallk_failure_followup_20261014/prepare_inputs.py
  af5c26c0742b4b68e49ae9b3cf4c1e614bd72745af6a40aa8132b652c7ed1f50
research/smallk_predictive_alignment_20261013/newdata.py
  ccef24b64c9beef0e6e31f9bc49dbad4e8e6ad57a6ce7c99973ccbce691de3bd
research/smallk_predictive_alignment_20261013/newdata_repair.py
  8cc75ffc4c86329939ae5155192afa03494f7bfff70fcf1ab2946c8c13f83900
research/level_matched_learned_ed_v18_20261006/data_contract_robin.json
  b7c02be4f69617e1c60dd874ef6337facde1a4165f8cbd30233aefb0601a85b2
.cache/smallk_predictive_alignment_20261013/beijing_repair/raw_dataset/beijing_aotizhongxin/b04da438b2f331ac/contract.json
  9116703c22b3dea317cb1ff750fd45d05c5d6e4a226946a41334e0794484849f
```
