제안은 **기존 방법의 E/D 적응 방식부터 바꾸고, 같은 변경을 받은 RAW 대조와 병합 LoRA까지 비교하는 후속 개발 회차**입니다. 핵심 질문은 “압축에서 손실된 정확도를 얼마나 회복했는가”와 “실제 대안 대신 선택할 가치가 있는가”를 분리하는 것입니다.

조사 결과에 따라 두 가지를 조정하겠습니다.

- **CURRENT 네 결과는 재사용**하고, FIXED_ED/SLOW_ED의 신규 8fit부터 시작합니다.
- 새 확인 후보인 로컬 Jena는 TEST 사용 이력이 있어 제외합니다. 그 여유를 RAW에도 두 학습 방식을 제공하는 데 사용해, **기본 신규 16fit + 조건부 비교·복구 최대 4attempt**로 구성합니다.

지금까지는 코드·메타데이터·기존 집계·체크포인트 구조·선행을 읽었습니다. **파일 수정, 학습, 모델 추론, 새로운 성능 채점, 설치, commit/push는 하지 않았습니다.**

**1. 현재 계약과 재사용할 기준선**

[확인] 작업 저장소는 요청하신 `CanelE452/hierarchical-tsfm-peft`이며, 로컬과 원격 `main` 모두 기준 커밋 `ef59f1e921bc70c93b919e095c4d3d6f048ca319`입니다. 이후 변경은 없습니다. 추적 파일은 깨끗하고 다음 기존 미추적 항목만 있습니다.

```text
_docs/history/.last-compact-resume.md
_docs/history/.pending.md
chronos-2-finetuned/
research/transient_peft_plan_20260923/
```

이를 보존합니다. 저장소·상위 경로에서 별도 프로젝트 지침은 찾지 못했으며, 전역 [AGENTS.md](C:/Users/User/.codex/AGENTS.md)와 이번 요청을 적용합니다. 과거 실험의 단발 실행 계약과 v1 봉인은 원본 그대로 유지합니다.

[확인] CURRENT의 실제 선택은 다음과 같습니다.

```text
자료          seed    run ID                                      선택/종료 epoch
Electricity  92601   revision1_rank32_residual_92601_0.001          41 / 47
Electricity  92602   revision1_rank32_residual_92602_0.001          15 / 21
Bull         92601   bull_final_residual_92601_0.001                1 / 7
Bull         92602   bull_final_residual_92602_0.001                1 / 7
```

네 `best.pt`와 `best_val.npz`, Electricity DEV 예측, Bull E1/E2 예측이 실제로 있고, 체크포인트·평가 예측의 hash가 기존 기록과 일치합니다. 데이터 NPZ도 존재하고 hash가 일치합니다. 캐시 루트는 [기존 실험 캐시](E:/CODING/proj/hierarchical-tsfm-peft/.cache/tsfm_peft_development_20260926/)입니다.

반면 **초기·모든 epoch의 가중치, 고정 TRAIN probe 예측, 분기별 출력은 저장되어 있지 않습니다.** 선택 가중치와 최초 PCA basis는 있으므로, 승인 후 필요한 초기 상태만 동일 생성 순서로 복원하고 최소 추론하겠습니다. 진단표를 채우기 위한 전체 재학습은 하지 않습니다.

초기 `protocol.json`의 rank8·40epoch·microbatch1을 복사하지 않습니다. CURRENT 실제 설정은 **G rank32·최대120epoch·microbatch4·LR 0.001**입니다. 기존 best checkpoint는 scheduler 갱신 전 저장되므로, 이를 완전한 epoch 종료 상태로 간주해 이어 학습하지 않습니다. 새 비교는 대응되는 초기값부터 시작합니다. 근거: [선택 파일](E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_development_20260926/revision1_rank32_selection.json), [실제 학습 루프](E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_development_20260926/run.py:214).

**2. 출발 근거와 검증할 가설**

[확인] 기존 집계의 두 seed **손실 평균**을 재확인했습니다. 예측 ensemble 점수가 아닙니다.

```text
자료·기간          CURRENT    RAW        FACTOR     미압축 LoRA
Electricity DEV    0.165850   0.180489   0.178457   0.161096
Bull E1            0.267849   0.568600   0.273212   0.162050
Bull E2            1.156667   1.271360   1.368564   1.191976
```

근거: [Electricity 집계](E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_development_20260926/revision1_rank32_development.json), [Bull 집계](E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_development_20260926/protected_evaluation.json).

Bull E1의 채널별 `CURRENT−LoRA` MSE 차이를 기존 공개 집계에서 계산하면, Mai·Marco의 합은 **전체 채널 순초과오차의 76.15%**입니다. 분모는 양의 손해만 더한 값이 아니라 모든 채널의 부호 있는 차이 합입니다. 또한 **14/16채널에서 CURRENT가 더 나쁩니다.** 두 채널을 삭제하거나 문제 전체를 두 채널로 한정할 근거는 아닙니다.

특히 기존 VAL 곡선에는 다음이 남아 있습니다.

```text
Bull CURRENT 초기 함수       0.490083
Bull CURRENT epoch1          0.402632 / 0.407504
미압축 F0에 해당하는 초기값  0.360104
```

즉 **압축 초기 함수부터 손해가 있었고, 첫 epoch는 그 손해를 줄였습니다.** 이후 VAL 악화가 관찰됐다고 해서 공동학습의 불안정만을 원인으로 확정할 수 없습니다.

검증할 가설은 다음 세 가지로 제한합니다.

- **적응 방식 가설:** E/D의 빠른 변경이 G 학습과 결합해 불리한 해를 만들 수 있다. FIXED_ED/SLOW_ED로 직접 비교합니다.
- **표현력 배분 가설:** 작은 decoder 공간과 공유 선형 G의 조합으로 보완하기 어려운 오차가 있을 수 있다. 같은 decoder 공간에서 오차 위치를 진단합니다.
- **TSFM 선택 가치 가설:** 잔차 경로는 유용하더라도, 일부 자료에서는 강한 선형 방법이나 미압축 모델이 더 적절할 수 있다. 정확도와 정상 배포 비용을 함께 비교합니다.

방법 해석은 기존 식을 유지합니다.

\[
\hat Y=D(F_0(E(X)))+G(X-D(E(X))).
\]

현재 G는 **과거 재구성 잔차를 입력으로 받으며 전체 미래 예측 MSE로 학습**합니다. 별도의 ‘진짜 미래 잔차’ 정답은 없습니다. 선형·채널공유 조건에서

\[
\hat Y=G(X)+D\{F_0(E(X))-G(E(X))\}
\]

가 성립하지만, 이것이 성공·실패 원인을 증명하지는 않습니다. 학습된 \(DE\)도 직교투영이라고 가정하지 않습니다.

또한 FIXED_ED의 PCA 공간이 고정되고 G가 공유 선형이면, G는 그 공간 바깥의 성분을 예측합니다. **PCA 공간 안의 TSFM 오차를 G가 자유롭게 고칠 수 있는 구성은 아닙니다.** 고정이 무조건 유리하리라고 예상하지 않는 이유입니다.

선행은 필요한 부분만 확인했습니다.

- **AdaPTS, ICML 2025:** 저자 원문 §3–5의 관련 부분과 공식 코드의 예측 목적함수·E/D 공동학습을 확인했습니다. 잠재 압축·PCA·E/D 공동학습 자체는 차별점이 아닙니다. COMPRESS를 전체 재현이라고 부르지 않습니다. 게재 상태는 공식 논문집으로 확인했고 최종 게재본 전체를 재현한 것은 아닙니다. [논문집](https://proceedings.mlr.press/v267/benechehab25a.html), [확인한 공식 코드](https://github.com/abenechehab/AdaPTS/blob/8bf57c7ee3b97bfd3f1852ad8dc8d0695a806278/src/adapts/adapts.py#L373)
- **Deep Factors, ICML 2019:** 원문 §1.2·§3.1–3.2의 전역 요인과 local 성분 결합을 확인했습니다. 현재의 동결 TSFM 압축 설계를 검증한 논문은 아닙니다. [원문](https://proceedings.mlr.press/v97/wang19k/wang19k.pdf)
- **L2-SP, ICML 2018:** 초기 파라미터 주변의 명시적 규제를 확인했습니다. 작은 학습률과 동일하지 않으며, PCA 초기값이나 1/10 학습률의 성공 근거로 쓰지 않습니다. 이번에는 규제 항까지 추가하지 않습니다. [원문](https://proceedings.mlr.press/v80/li18a/li18a.pdf)

**3. 승인 후 최소 진단 → 첫 직접 비교**

진단은 다음 학습을 해석하는 데 필요한 범위까지만 수행합니다.

1. **손해 위치:** 기존 예측으로 모든 채널의 CURRENT−LoRA·CURRENT−FACTOR 차이와 앞·뒤 기간을 계산합니다. Mai·Marco는 관측·예측·오차 시점을 함께 살펴보되 전체표를 유지합니다. 수준 변화·결측·계측 변화가 의심되더라도 확인되지 않으면 원인 미확정으로 남깁니다.
2. **공간 안팎:** 선택된 CURRENT의 D에서 수치 rank를 확인하고 \(Q=DD^+\)를 계산합니다. 같은 seed의 CURRENT와 비교군을 **같은 Q**로 투영합니다. 결측이 있는 시점은 완전 관측 벡터만의 별도 진단으로 구분하며, 이를 원래 masked macro MSE의 자동 가산 분해라고 하지 않습니다.
3. **학습과 분기:** 기존 curve를 읽고, 초기·선택 모델을 같은 고정 TRAIN probe와 전체 VAL에서 비교합니다. TRAIN probe는 24개 시작 위상에서 각 4개씩 고정한 **96원점**으로 제안합니다. 주경로와 G 출력 크기를 함께 봅니다. 매 epoch 바뀐 표본의 기존 TRAIN loss는 고정 probe 곡선과 구분합니다.

초기 상태나 분기 기록이 없다는 이유로 직접 비교를 막지 않습니다. 필요한 복원·추론만 GPU 시간에 포함하고, 누수·단위 불일치·핵심 구현 오류만 관련 실험의 중단 사유로 삼습니다.

첫 비교는 제안하신 세 구성을 유지합니다.

```text
구성       E/D                                  G
CURRENT    기존 공동학습, LR 1e-3                LR 1e-3
FIXED_ED   해당 자료 TRAIN PCA 초기값에 고정     LR 1e-3
SLOW_ED    공동학습, LR 1e-4                    LR 1e-3
```

FIXED_ED와 SLOW_ED는 **[추정·미검증] 대비**입니다. 두 자료×두 seed×세 구성의 12개 결과 중 CURRENT 4개를 재사용하므로 **신규 8fit**입니다.

다음 조건은 공통으로 유지합니다.

- 동일 Bolt-small 가중치와 median 점예측 경로, L512/H48.
- Electricity K8, Bull K4, G rank32.
- 동일 PCA basis·G 초기값·원점 schedule·정규화·관측 마스크.
- AdamW, weight decay 0, 전체 학습 파라미터 gradient clipping 1.
- epoch당 512 TRAIN 원점, effective batch4·microbatch4, float32.
- 최대120epoch, 전체 VAL의 strict 개선이 6회 없으면 종료.
- step0 포함 최소 VAL 선택, 동률이면 먼저 나온 checkpoint.
- 기존 plateau scheduler의 factor 0.5·patience2 및 세부 설정 유지.

SLOW_ED에서는 scheduler가 G의 LR을 줄일 때 E/D를 그 **0.1배로 동기화**하고 두 값을 기록합니다. 아주 작은 LR에서 scheduler의 수치 epsilon 때문에 비율이 깨지는 경우도 검사합니다.

구현 확인은 다음에 집중합니다.

- 세 구성의 대응되는 초기 함수가 같은지.
- FIXED_ED의 E/D는 불변이고 G는 실제 갱신되는지.
- CURRENT/SLOW_ED에서 동결 F0를 통과하는 E의 gradient가 유지되는지.
- FIXED_ED의 계산 절약이 출력·손실·G gradient를 바꾸지 않는지.
- 결측 effective-batch 손실, 샘플 순서, checkpoint 복원이 맞는지.

현재 [저장 함수](E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_development_20260926/model.py:82)는 `requires_grad` 파라미터만 다룹니다. v2에서는 **고정 E/D도 복원 가능한 방법 상태로 기록**합니다. G의 출력층이 0인 첫 backward에서 G의 앞층 gradient가 0일 수 있으므로, 모든 파라미터가 첫 step부터 갱신돼야 한다는 잘못된 검사는 만들지 않습니다.

**4. 결과별 후속 분기**

첫 비교 후에는 유리해진 방식을 RAW에도 먼저 제공합니다. 이번에는 새 확인 실험을 넣지 않으므로, 필요하면 다른 E/D 방식까지 제공해 **RESIDUAL과 RAW 모두 CURRENT/FIXED/SLOW의 같은 선택 기회**를 갖게 합니다. RAW 신규 실행 상한은 **8fit**입니다. 결과가 충분해 더 실행할 이유가 없으면 남은 실험을 채우지 않습니다.

각 자료·방법의 checkpoint와 LR·모드 선택은 **두 seed 평균 VAL**을 기준으로 합니다. E1/E2에서 좋아 보이는 seed나 LR로 바꾸지 않습니다. 같은 모드끼리의 비교와 각 방법의 VAL 선택 결과를 모두 보고합니다.

- **FIXED_ED가 유리하면:** 고정 표현과 G만으로 손해를 줄이고 학습 비용을 낮출 수 있는지 판단합니다. 같은 FIXED_RAW와 차이가 사라지면 잔차 입력의 추가 가치 주장을 낮춥니다.
- **SLOW_ED가 유리하면:** 같은 SLOW_RAW에도 이득이 나타나는지 확인합니다. 모두 비슷하게 좋아지면 개선의 상당 부분을 학습 방식의 효과로 보고합니다.
- **둘 다 불리하거나 큰 손해가 남으면:** 아래에서 판단을 바꿀 **한 가지 후속 비교만** 선택합니다.
- **변동이 크면:** 불안정한 결과를 성공·실패로 단정하지 않고, 아래 4attempt를 비교의 충분성 보완에 쓸지 결정합니다.

마지막 **최대 4attempt**의 우선순위는 다음과 같습니다.

1. **필수 복구·재사용 불가 기준선·학습량 보완.** 원본 실패와 diff를 보존합니다. 최대120에서 계속 개선 중인 핵심 비교가 있다면, 한 자료의 선택 RESIDUAL/RAW×두 seed를 같은 종료 규칙으로 재비교하는 데 사용할 수 있습니다. 이 경우 최대240epoch로 한 번만 확장하고 기존 결과를 보존합니다. 자원 상한은 그대로입니다.
2. **공간 바깥의 손해가 중요한 단서라면:** Bull에서 K4→8의 **RESIDUAL/RAW×두 seed=4fit**. K4의 같은 학습 방식과 비교하여 backbone 연산 증가의 효과와 같은 K에서의 잔차 입력 효과를 나눕니다. 이 비교만으로 G의 존재 자체에 대한 완전한 절제를 했다고 하지는 않습니다.
3. **학습형 E/D를 둔 TSFM의 추가 가치가 핵심 공백이라면:** 선택된 E/D 학습 방식을 유지하고 F0를 공유 선형 시간 경로로 바꾼 대조를 두 자료×두 seed로 비교합니다. 시간 경로는 편향 없는 512→32→48을 출발점으로 삼고, G와 함께 학습합니다. 초기 함수·학습 파라미터 수가 달라지므로 **사전학습 가중치만의 인과 절제라고 부르지 않습니다.** 학습이 불충분하면 그 음성을 TSFM의 필요성 증거로 사용하지 않습니다.

이 셋을 모두 실행하지 않습니다. 실행 전에 **관찰 → 선택한 가설 → 바꿀 한 가지 → 어떤 결과가 설명을 지지하는지**를 짧게 남깁니다.

이번 자동 범위에서는 비선형 G·공유 방식 변경·여러 규제·stop-gradient·무작위 backbone을 제외합니다. 단순 잠재 표준화 추가도 기본 분기에서 제외합니다. 설치된 Bolt는 이미 각 잠재 계열을 instance normalization하고 같은 문맥 통계로 출력을 역변환합니다. 이것이 분포 문제 전체를 해결한다는 뜻은 아니지만, 동일한 스케일 처리를 이유 없이 중복할 근거는 없습니다. [설치 코드](E:/CODING/proj/hierarchical-tsfm-peft/.venv/Lib/site-packages/chronos/chronos_bolt.py:292)

**5. 데이터·선택·새 확인 계약**

기존 [데이터 계약](E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_development_20260926/data_contract.json)을 그대로 연결합니다.

- **Electricity:** 첫32채널, 기존 70/10/20 경계, TRAIN/VAL/DEV 원점 17,853/108/218.
- **Bull:** 기존 적격16채널과 날짜 경계 유지. TRAIN/VAL/E1/E2 원점 1,945/30/40/40.
- 모델 업데이트는 TRAIN만 사용합니다. 정규화·PCA·입력 결측 처리 통계도 TRAIN만 사용합니다.
- 주 지표는 기존 정규화 후 채널별 MSE 동일 가중 평균입니다. MAE·채널별·앞뒤 기간 값을 함께 보고합니다.
- **Bull E1/E2는 v2에서 노출된 개발 평가**입니다. v1 결과·봉인·판정을 덮어쓰거나 다시 미노출 test로 부르지 않습니다.
- LoRA의 native quantile 학습과 후보의 MSE 학습 차이를 표시합니다. 실용적 대안 비교로 사용하며 모든 차이를 구조만의 효과로 해석하지 않습니다.
- 목표가 겹치는 원점과 같은 site의 채널·seed를 독립 데이터셋으로 세지 않습니다. 기존 방식의 시간 블록 비교를 쓰되, 두 seed에 조건부인 구간이며 반복 선택의 불확실성을 포함하지 않는다고 명시합니다.
- 데이터 오류가 확인되면 영향받은 비교를 함께 다룹니다. 실제 관측값이나 포함 집단의 실질적 변경은 자동 처리하지 않고 변경안으로 보고합니다.

유한한 VAL을 반복 선택에 사용하면 선택 편향이 생길 수 있으므로, **개발에서 고른 결과와 새 확인을 구분**합니다. 이번에는 Cawley–Talbot의 초록·서론과 평가 절차에 대한 설명을 확인했으며 전체 실험을 재현하지 않았습니다. [원문](https://www.jmlr.org/papers/volume11/cawley10a/cawley10a.pdf)

**새 확인 자료는 이번 자동 범위에서 미확보로 둡니다.** 확인한 로컬 Jena MPI roof 2024는 10분 간격·52,704행·21변수 자료지만, 이미 과거 TEST 예측·평가에 사용됐습니다. 기존 manifest도 이를 명시합니다. [노출 기록](E:/CODING/proj/hierarchical-tsfm-peft/results/emulator_response_gonogo_v1_20260922/DATA_MANIFEST.json:89)

이를 새 확인으로 재명명하거나 다른 자료를 연속해서 찾지 않습니다. 이번 회차의 결과는 **기존 두 자료에서의 개발·비용 검증**이며, 새 도메인 확증까지 확보했다고 주장하지 않습니다.

**6. 비용 측정 계약**

기존 단회 측정은 참고 기록으로만 남깁니다. 승인 후 다음 정상 경로를 비교합니다.

- 미적응 F0.
- **병합 LoRA**.
- CURRENT와 최종 선택 후보.
- 같은 변경을 받은 RAW, 적절한 COMPRESS·선형 대조.

설치 PEFT 0.21.0에서 `merge_and_unload(safe_merge=True)` 경로를 확인했습니다. 기존 adapter를 복원한 별도 인스턴스를 병합하고, 반환 backbone을 사용합니다. 원래 가중치·체크포인트는 덮어쓰지 않습니다. 공식 문서 역시 병합을 배포 시 adapter 계산을 합치는 경로로 설명합니다. [PEFT 문서](https://huggingface.co/docs/peft/main/conceptual_guides/lora#merge-lora-weights-into-the-base-model)

병합 전후 예측 정합성은 아직 실행하지 않았습니다. 승인 후 같은 VAL 입력에서 확인하며, FP32 수치 검사의 출발값은 `atol=1e-5, rtol=1e-4`로 제안합니다. 불일치하면 원인을 확인하고 유리한 경로를 골라 보고하지 않습니다.

측정 조건은 다음과 같이 고정합니다.

- 동일 장치·float32·L/H·원입력·전체 채널 출력.
- 각 자료 VAL에서 시간 순서로 고르게 고른 **24원점**, 두 학습 seed 모두 사용.
- **batch1 지연**과 **batch4 처리량**을 따로 보고.
- 모델별 예열10회, 24원점 전체 처리를 측정20회, 실행 순서를 바꾼 3개 블록. 이는 운영 출발값이며 독립 표본이나 성능 보장이 아닙니다.
- CPU의 공통 표준화 입력부터 시작해 입력 전송·전체 출력의 CPU 반환까지 포함. 모델 load·디스크 I/O·지표 계산·로그 저장은 측정 밖에 둡니다.
- CUDA 동기화, CPU thread 수 고정, 중앙값·사분위 범위·순서 블록별 결과를 기록합니다. 측정 해상도가 부족하면 결과 비교 전에 공통 반복 단위만 조정합니다.
- 선형 대조는 명시한 CPU 경로와 같은 입력·출력 범위로 측정합니다.
- 모델 교체 시 객체 참조·GC·allocated 기준점을 확인합니다. allocated/reserved/프로세스 메모리를 구분합니다.
- FIXED_ED의 학습 중 계산 절약은 인정하되, 배포 지연에서 TSFM 출력을 미리 캐시해 비용을 숨기지 않습니다.

예열·동기화·측정 오버헤드의 필요성은 PyTorch 안내에서 확인했습니다. 위 반복 수와 입력 집합은 이번 계획의 제안값입니다. [PyTorch benchmark](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html)

정확도와 비용을 데이터별로 나란히 제시합니다. 비용이 늘어 정확도가 좋아졌으면 그 절충을 그대로 보고하고, 허용 손해를 정하지 않은 비열등성 주장은 하지 않습니다.

**7. 신규 회차 예산과 자율 실행 경계**

[확인] 현재 머신은 RTX 4070 12,282MiB, RAM 약31.8GiB, E 드라이브 여유 약457GiB입니다. 조회 시 Python 학습 프로세스는 없었지만 실행 때 다시 확인합니다. 기존 캐시는 약130.75MiB입니다.

**신규 20attempt 또는 GPU 작업 점유 4시간 중 먼저 도달하는 상한**, 로컬 GPU1장·동시 신경망 학습1개·신규 저장10GiB를 유지하겠습니다.

```text
신규 신경망 실행                         최대 attempt
FIXED_ED/SLOW_ED × 두 자료 × 두 seed          8
동일 두 방식의 RAW × 두 자료 × 두 seed       8
조건부 비교·CURRENT 재현·실패 복구 통합       4
합계                                        20
```

CURRENT 재사용에 문제가 생기거나 실패가 발생하면 마지막 4칸을 먼저 사용합니다. 필요 대조를 빼고 후보 탐색만 늘리지 않습니다. 별도 optimizer 업데이트를 하는 검사 실행도 attempt에 계상하며, 가능하면 실제 첫 fit의 초기 검사에 통합합니다.

GPU 시간 배분은 다음 **[추정·미검증] 운영안**입니다.

```text
최소 진단·복원·구현 검사             20분
첫 직접 비교 8fit                   70분
RAW 비교 최대8fit                   70분
조건부 비교·재현·복구 최대4attempt   40분
병합 정합성·재평가·반복 비용 측정    40분
합계                               240분
```

기존 CURRENT의 실제 fit 시간은 Electricity 약87–192초, Bull 약26초였지만, 새 느린 학습이 같은 epoch에서 종료된다는 보장은 없습니다. 최초 실측 후 항목 간 시간은 재배분하되 총 상한을 늘리지 않습니다. 학습·추론·smoke·실패·복원·profile 모두 GPU 원장에 포함합니다.

기본 추가 CPU 회귀 적합은 필요하지 않으며 기존 선형 대조를 재사용합니다. 공간 진단·집계·그림의 CPU 시간과 추가 적합이 있다면 그 횟수를 따로 기록합니다. 신규 저장은 대략0.1–1GiB를 예상하지만 미실측이며, 10GiB 상한과 원자료·백본 중복 저장 금지를 적용합니다.

충돌 대기는 회당30분·누적60분입니다. 진행 없는 실행은 10분부터 원인과 I/O를 확인하고, 30분간 진전이나 합리적 연산 근거가 없으면 자신의 작업만 안전 중단합니다. 다른 프로젝트 프로세스는 종료하지 않습니다.

승인 후에는 이 범위의 코드·API·경로·직렬화·게시 오류를 원본과 diff를 남겨 수정하고 계속합니다. 총예산·자료 계약의 실질적 변경·새 서버·유료 서비스·전역 환경 변경은 별도 결정입니다. Codex 토큰 상한은 추가하지 않습니다.

**8. 완료 판정과 기록·게시**

이번 회차는 다음 중 하나를 실제 결과로 판정해야 끝납니다.

- **진전:** CURRENT의 손해 또는 비용을 줄인 구성이 있고, 같은 학습 기회를 받은 RAW와 비교한 뒤 잔차 입력의 추가 가치와 적용 범위를 설명할 수 있음.
- **부정적이지만 유용한 결과:** 단순 RAW·선형 대안으로 충분하거나 큰 손해가 남아 현재 구성을 교체·보류할 근거가 생김. 개선된 방법 확보라고 부르지 않음.
- **판단 불가:** 필수 비교의 자료·권한·유효성 문제 또는 예산 종료로 결론을 낼 수 없음. 실제 완료와 미실행을 구분함.

한 번의 음성이나 epoch1 선택만으로 폐기하지 않습니다. 반대로 두 E/D 대비와 필요한 후속 비교 뒤에 같은 질문의 반복만 남으면 예산을 다 쓰지 않고 종료합니다. 임의의 개선율 합격선은 만들지 않습니다.

[확인] 제안하신 다음 경로는 현재 없으므로 승인 후 사용할 수 있습니다.

`E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_followup_v2_20260926/`

새 `PLAN.md`에는 승인된 계획 원문을 보존하고, `STATUS.md`에는 관찰·변경 이유·실행·예산을, `TOPIC_DECISION.md`에는 **CURRENT 대비 무엇이 바뀌었고 실제로 무엇이 좋아졌거나 나빠졌는지**를 남깁니다. v1의 동명 파일·코드·봉인은 덮어쓰지 않습니다. v1 보호 실행기의 제한을 풀어 재사용하지 않고, v2의 노출 개발 계약과 새 실행 원장을 분리합니다.

최종 보고에는 전체 채널의 손해 분포, 필요한 시계열 사례, 학습 곡선, 데이터별 정확도–비용 비교 그림과 수치 출처를 포함합니다. 실행 주체의 검산이나 에이전트의 동의를 독립 재현으로 부르지 않습니다.

승인된 작업이 끝나면 관련 파일만 검사·일반 commit·`origin/main` push하고 실제 반영을 확인합니다. 원자료·대용량 가중치·예측 캐시·무관한 dirty 파일은 게시하지 않습니다.

**승인 대상은 위 v2 계획 전체입니다. 승인 전에는 실행하지 않겠습니다.** 이는 요청하신 새 승인 경계이며, 적용한 [research-plan-loop](C:/Users/User/.claude/skills/synced/a4a85a5d-7fa9-4c8d-866c-fbdbba10af00_ee36e183-ea13-46c0-83bf-ec9a51839098/research-plan-loop/SKILL.md)의 “합의 전 실행 금지” 원칙과도 일치합니다.
