제안은 **기존 TSFM 결과를 재사용하고, 문맥 정규화를 포함한 full-rank 시간 선형 대체를 신규 8fit로 비교한 뒤, 필요한 분기 하나와 비용 진단으로 결론을 좁히는 것**입니다.

이번에는 코드·기록·공식 자료와 자원 상태만 읽었습니다. **파일 수정, 모델 실행·학습·채점·benchmark, 설치, commit/push는 하지 않았습니다.**

**1. 확인한 출발점과 재사용 범위**

[확인] 저장소는 `CanelE452/hierarchical-tsfm-peft`, branch는 `main`이며 로컬 HEAD와 원격 main 모두 `e15c913520d9ec5100459077dfcc58518b22a363`입니다. 더 최근 커밋은 없습니다. 추적 파일은 깨끗하고 다음 기존 미추적 항목만 있습니다.

```text
_docs/history/.last-compact-resume.md
_docs/history/.pending.md
chronos-2-finetuned/
research/transient_peft_plan_20260923/
```

프로젝트·상위 경로의 별도 AGENTS.md는 없었습니다. 전역 [AGENTS.md](C:/Users/User/.codex/AGENTS.md)와 이번 지시를 적용합니다. 새 후보를 늘리는 일반적인 스킬 절차는 이번 요청에 적용하지 않습니다.

[확인] 아래 선택 checkpoint와 필요한 기존 예측 파일은 존재합니다. 이번에는 원시 배열을 다시 채점하거나 모든 파일을 재감사하지 않았습니다.

```text
자료          TSFM_RES 재사용 구성       선택 run
Electricity  CURRENT, C32/K8            revision1_rank32_residual_92601_0.001
                                       revision1_rank32_residual_92602_0.001
Bull         FIXED_ED, C16/K4           v2_bull_residual_fixed_ed_92601
                                       v2_bull_residual_fixed_ed_92602

기존 선택 결과                         개발 MSE
Electricity DEV                         0.165850
Bull E1 / E2                            0.210365 / 1.246580
```

Electricity checkpoint는 `.cache/tsfm_peft_development_20260926/runs/`, Bull은 `.cache/tsfm_peft_followup_v2_20260926/runs/`의 해당 run 아래에 있습니다. F0·LoRA·선택 RES·SHARED/FACTOR의 관련 예측 receipt 21개도 모두 파일이 존재했습니다. 정확도는 이 기록을 재사용하고, 비용 측정에 필요한 복원·출력 정합성만 승인 후 확인합니다. 근거: [최종 선택](E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_followup_v2_20260926/final_selection.json), [기존 평가와 예측 경로](E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_followup_v2_20260926/final_development.json).

Bull의 **선택 RAW는 CURRENT K4**, FIXED RES에 대응하는 RAW는 **FIXED K4**라는 구분을 유지합니다.

새 폴더 후보 `research/tsfm_peft_backbone_value_v3_20260927/`는 현재 없습니다. 승인 후 생성합니다. v2의 [22/20attempt 계상 이탈](E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_followup_v2_20260926/attempt_accounting_correction.json)은 그대로 보존합니다.

**2. 비교할 주장과 선형 대체의 정확한 정의**

주장은 다음으로 한정합니다.

> 현재 자료별 E/D 모드와 G 구조에서, 정상적으로 학습한 작은 선형 대체보다 동결 TSFM의 예측 가치가 남으며 그 비용을 감수할 이유가 있는가?

최종 E/D/G 가중치를 서로 공유하는 비교가 아닙니다. **구조·학습 여부·가능한 초기값·표본 순서를 맞추고 각 경로를 독립적으로 학습한 실용적 대체 비교**입니다.

\[
\begin{aligned}
\mathrm{TSFM\_RES}&=D(F_0(E(X)))+G(X-D(E(X))),\\
\mathrm{LINEAR\_RES}&=D(\widetilde T_\phi(E(X)))+G(X-D(E(X))).
\end{aligned}
\]

[제안·미검증] 주 대조는 다음으로 고정합니다.

- `T_phi`: 잠재 채널들이 공유하는 **`Linear(512,48,bias=True)`**, 24,624개 파라미터.
- `G`: 기존의 공유·무편향·무활성화 `512→32→48`, 17,920개.
- Electricity: K8, E/D/T/G 모두 학습.
- Bull: K4, TRAIN PCA E/D 고정, T/G 학습.
- T와 G는 별도 객체·별도 파라미터이며 weight tying을 하지 않습니다.
- F0 출력으로 증류하거나 TSFM과 학습한 최종 E/D/G를 초기값으로 복사하지 않습니다.

설계상 파라미터 수는 다음과 같습니다. 구현 후 실제 등록 수와 갱신 수를 별도로 확인합니다.

```text
LINEAR_RES       전체 파라미터    학습 가능
Electricity          43,056         43,056
Bull                 42,672         42,544
```

**주 전처리는 Bolt의 입력 문맥 정규화와 역변환을 공유하는 방식**을 선택합니다.

[확인] 설치 Bolt는 잠재 계열별 과거 문맥에서 FP32 평균과 모집단 표준편차를 계산합니다. all-NaN이면 평균 0·scale 1, scale이 정확히 0이면 `1e-5`로 대체합니다. 기본 arcsinh는 꺼져 있고 평균·scale에 `detach`가 없습니다. [설치된 InstanceNorm](E:/CODING/proj/hierarchical-tsfm-peft/.venv/Lib/site-packages/chronos/chronos_bolt.py:95)

따라서 제안 경로는 다음입니다.

\[
\widetilde T_\phi(z)
=s(z)\left[W\frac{z-\mu(z)}{s(z)}+b\right]+\mu(z).
\]

μ와 s는 **과거 512개 입력만** 사용합니다. 이 경로는 입력 의존적인 \(s(z)b\)를 포함하므로 **“정규화를 포함한 선형 대체”**라고 부르고, 전역적으로 순수한 선형 함수라고 하지 않습니다. 원래 잠재값을 직접 넣는 방식은 기본 실행에 추가하지 않고, 정규화가 판단을 막는 구체적 단서가 있을 때만 후속 분기로 검토합니다.

기존 입력은 TRAIN 표준화 후 결측을 0으로 채운 값이므로 잠재 입력도 finite입니다. 원채널 정답 mask를 새로운 잠재 입력 mask로 투영하지 않습니다.

초기화는 다음처럼 고정합니다.

- E/D/G는 해당 자료·seed·K의 **학습 전 v2 `initial.pt`**에서 가져옵니다. 네 파일의 존재와 기록된 hash 일치를 확인했습니다.
- T는 별도 CPU RNG 문맥에서 PyTorch 기본 비영 초기화를 사용합니다. T 생성이 G의 난수 순서를 바꾸지 않게 합니다.
- 두 T 학습률은 같은 초기 tensor를 사용하고 hash로 확인합니다.
- G 출력층은 기존처럼 0입니다. T까지 0으로 만들지 않습니다.
- Electricity v1은 당시 원본 초기 checkpoint가 없습니다. 동일 constructor·seed라는 근거는 있지만 **v1 초기 G의 bitwise 일치가 확인됐다고 주장하지 않습니다.** 이를 이유로 TSFM을 자동 재학습하지 않습니다.

가까운 선행과의 관계도 제한합니다. AdaPTS의 잠재 어댑터·PCA·공동학습은 기존 원리입니다. 이번에는 v2의 원문·코드 검토 기록을 재사용하고 공식 게재 정보만 재확인했습니다. [AdaPTS, ICML 2025](https://proceedings.mlr.press/v267/benechehab25a.html)

LTSF-Linear는 작은 선형 대조를 진지하게 비교할 근거입니다. 이번에 공식 초록과 저자 `Linear.py`·`NLinear.py`를 확인했습니다. 제안한 평균·scale 정규화는 NLinear의 마지막 값 정규화와 다르며, 해당 논문의 전체 재현이 아닙니다. 당시 결과를 현재 TSFM 전체로 일반화하지 않습니다. [AAAI 2023 논문](https://ojs.aaai.org/index.php/AAAI/article/view/26317), [공식 Linear 구현](https://raw.githubusercontent.com/cure-lab/LTSF-Linear/main/models/Linear.py)

**3. 구현에서 반드시 분리할 부분**

[확인] 기존 wrapper는 backbone 전체를 동결하고, FIXED_ED에서는 주경로 전체를 `no_grad()`로 감쌉니다. 저장 목록에서도 `backbone.*`를 제외합니다. 그대로 교체하면 T가 학습되지 않거나 저장에서 빠집니다. [현재 wrapper](E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_followup_v2_20260926/model.py:28)

승인 후 F0를 생성하지 않는 별도 선형 모델 경로를 구현합니다.

- T를 명시적인 학습 parameter group에 넣습니다.
- Bull의 E/D는 동결하되 `D(T(z))`의 autograd는 유지합니다.
- E/D/G/T 전체와 정규화 설정을 저장·복원합니다.
- 추론 인스턴스에는 Chronos 가중치를 로드하거나 남기지 않습니다.
- 선택 checkpoint와 재시작 checkpoint를 구분합니다. 재시작 상태에는 optimizer·scheduler·정체 횟수·RNG·학습 위치를 일관된 시점으로 저장합니다. 기존 v2 best 저장 시점은 scheduler 갱신 전이므로 그대로 재시작 규약으로 복사하지 않습니다. [학습·저장 순서](E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_followup_v2_20260926/run.py:381)

검사는 T의 gradient·실제 갱신, 고정 E/D 불변, Electricity E/D 갱신, G의 초기 0과 후속 갱신, mask 축, 저장·재시작 복원에 집중합니다. 초기 첫 step의 G 앞층 gradient 0은 허용하되 영구 단절과 구분합니다.

상수 입력에서 출력뿐 아니라 gradient도 유한한지 확인합니다. 설치 normalization의 `sqrt(0)` 경계는 **현재 실패가 확인된 것이 아닌 검사 대상**입니다. 실제 문제가 생기면 원본을 보존하고, 동일 forward를 유지하는 국소적인 경계 처리만 근거와 함께 수정합니다.

**4. 최초 8fit와 평가 계약**

[확인] 기존 두 자료는 시간당 관측을 전제로 준비·검사됐고, 평가 원점 간격은 24시간입니다. 동일 데이터 NPZ와 hash를 사용합니다. [데이터 계약](E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_development_20260926/data_contract.json), [기존 시간 간격 검사](E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_development_20260926/test_data.py:46)

```text
자료          채널/K   TRAIN 원점   VAL 원점   개발 평가 원점
Electricity   32/8       17,853       108       DEV 218
Bull          16/4        1,945        30       E1 40 / E2 40
```

표준화·PCA·결측 입력 통계·학습에는 TRAIN만 사용합니다. L512/H48, FP32, seed92601/92602를 유지합니다. 미래 정답 mask를 적용한 **원채널 동일 가중 MSE**가 주 지표이고 MAE·기간별·채널별 값은 보조입니다.

첫 실행은 다음 8fit입니다.

```text
자료별 고정 모드              T 초기 LR        seed             fit
Electricity CURRENT K8       1e-3, 1e-4      92601, 92602       4
Bull FIXED_ED K4             1e-3, 1e-4      92601, 92602       4
합계                                                            8
```

공통 출발 레시피는 다음과 같습니다.

- 학습되는 E/D와 G의 초기 LR `1e-3`; T는 별도 group.
- AdamW, weight decay 0, gradient clipping 1.
- 최대120epoch, 전체 VAL의 strict 개선이 6회 없으면 종료.
- epoch당 위상균형 TRAIN512원점, effective/microbatch4.
- 기존 plateau scheduler 설정을 유지하되 공통 감쇠 배율을 적용해 T와 G의 LR 비율을 보존하고 실제 값을 기록.
- 동일 seed·epoch의 원점 순서를 맞춥니다.

각 seed는 step0을 포함한 최소 VAL checkpoint를 선택하고 동률이면 먼저 나온 상태를 고릅니다. **두 seed의 best VAL 평균으로 자료별 T 학습률 하나를 선택한 뒤 개발 평가를 읽습니다.** DEV/E1/E2로 LR·epoch를 바꾸지 않습니다.

고정 TRAIN96원점 probe와 전체 VAL의 초기·선택 점수를 남깁니다. 실제 fit 내부에서 갱신을 확인해 별도 실자료 optimizer smoke를 줄입니다. step0 선택만으로 실패라 하지 않으며, 정상 갱신·TRAIN 개선과 VAL 일반화 문제를 구분합니다.

최종 주요 비교는 선택 LINEAR_RES와 기존 선택 TSFM_RES입니다. F0·LoRA·SHARED/FACTOR·RAW는 기존 점수를 재사용해 실용적 위치를 설명합니다. SHARED/FACTOR는 ridge 적합·bias·전처리·G 용량·E/D 학습 여부가 달라 이번 대조를 대신하지 못합니다. [기존 선형 적합](E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_development_20260926/masked_linear.py:58)

주 비교는 원수치와 \(\Delta=\mathrm{MSE}_{LINEAR}-\mathrm{MSE}_{TSFM}\)를 함께 보고합니다. 양수면 TSFM이 유리합니다. 두 seed의 **손실 평균**을 사용하며 예측 ensemble 점수로 바꾸지 않습니다.

불확실성은 기존처럼 모든 채널을 함께 움직이는 연속7원점 블록·2,000회 재표본, 개별 seed·시간 앞뒤 절반을 함께 봅니다. 겹치는48시간 목표를 독립 표본으로 세지 않습니다. 이 구간은 학습 seed가 고정된 조건부 구간이며 반복 선택 불확실성을 포함하지 않습니다. 모든 평가는 노출 개발 자료입니다.

**5. 첫 비교 이후 허용할 분기**

첫8fit 뒤 `관찰 → 남은 의문 → 바꿀 한 가지 → 결과별 해석`을 기록하고 **후속 최대4fit에서는 필요한 분기 하나를 우선**합니다.

- **학습·전처리 보완:** T가 정상 갱신되지 않거나, TRAIN/VAL 곡선·상수 처리·정규화에서 구체적인 문제가 확인되면 그 원인을 수정합니다. 과학적 변경은 초기화·정규화·학습량 중 한 가지로 제한하고 해당 자료의 두 seed를 재비교합니다. 최대epoch에서도 개선 중이라면 선택 LR의 두 seed를 최대240epoch까지 한 차례 보완할 수 있습니다. 재시작도 새 attempt로 계상합니다.
- **LINEAR_RAW:** 잔차 입력의 역할을 알아야 다음 판단이 달라질 때만 `D(T(E(X)))+G(X)`를 추가합니다. 같은 T·E/D 모드·초기값·고정 레시피로 자료당 두 seed를 비교합니다. RAW 전체 LR 탐색을 하지 않았다면 “대응 레시피 비교”라고 제한합니다.
- **추가 학습 없음:** 이미 판단이 충분하면 비용 진단으로 진행합니다.

별도 마지막4attempt는 실패 복구·재사용 불가 기준선·필수 학습량 보완에 남깁니다. 필요한 복구가 이 범위를 넘으면 비교 불충분으로 종료하며 총 상한을 늘리지 않습니다. 한 음성을 모두 버그로 돌리지 않고, 설명 없는 LR·rank 탐색도 하지 않습니다.

**6. 비용 변동과 병목을 분리하는 측정**

비용 문제는 첫8fit의 진입 조건으로 두지 않습니다.

대상은 자료별 F0 1개, 병합 LoRA 두 seed, 선택 TSFM_RES 두 seed, 선택 LINEAR_RES 두 seed로 **총14개**입니다. 과거26개 변형을 다시 측정하지 않습니다.

공통 조건은 RTX4070, FP32, TF32 off, CPU thread4, 동일 VAL24원점, batch1/4, 예열10회, 전체24원점 처리20회, 순서를 바꾼3블록입니다. 두 범위를 분리합니다.

1. **배포 시간:** 표준화 CPU 입력 → GPU 전송 → 전체 원채널48시간 출력의 CPU 반환까지 동기화한 wall time.
2. **모델 실행 진단:** 입력·출력을 GPU에 둔 전체 forward의 CUDA Event 시간과 동기화 wall time.

모델 load·디스크·채점·원장 쓰기는 성능표 시간에서 제외하지만 GPU 작업 점유 예산에는 포함합니다. Event 시간은 CPU의 늦은 연산 제출로 생긴 대기도 포함할 수 있으므로 순수 kernel 시간이라 하지 않습니다. 두 범위의 차이도 곧바로 전송 비용으로 단정하지 않습니다. [PyTorch CUDA timing](https://docs.pytorch.org/docs/2.10/notes/cuda.html#asynchronous-execution)

각 자료·블록의 **첫 F0를 기준선 겸 앞쪽 반복 기준으로**, 마지막에 같은 F0를 다시 측정합니다. 나머지 모델 순서를 회전하고 TSFM/LINEAR의 대응 seed 순서도 바꿉니다. 이 설계의 기본 GPU timing 행은 192개입니다. F0 기준 점수는 사전 지정한 첫 측정으로 집계하고 뒤쪽 측정은 drift 확인용으로 구분합니다.

- 앞뒤 분포가 분리되거나 변동 폭이 주장하려는 방법 차이보다 크면 불안정 표시를 남깁니다.
- 블록·반복·순서·예열·온도·읽을 수 있는 clock/사용률을 보존합니다.
- 원인 단서가 있을 때만 대표 문제 블록을 최대1회 재확인합니다.
- 반복해서 좋은 블록을 찾거나 자동 보정하지 않습니다.

변동이 재발하거나 두 측정 범위의 상대 순서가 달라지면 Bull batch1의 F0·TSFM_RES·LINEAR_RES 대표 각1개를 짧게 profile합니다. CPU/CUDA 활동을 보되 처음에는 shape·stack·memory 추적을 끕니다. profiler 시간은 성능표에 섞지 않습니다. [PyTorch Profiler](https://docs.pytorch.org/docs/2.10/profiler.html)

출력 동등 구현 개선은 trace가 지지하는 **변경 한 차례**만 허용합니다. 불필요한 복사 제거, 적용 가능한 inference mode, 인접 선형층 합성 등이 대상입니다. 문맥 정규화를 가로질러 전체 경로를 순수 선형으로 합치지는 않습니다. 비교군에도 적용 가능한 동일 기회를 주고 원본·수정 비용을 모두 남깁니다.

LoRA는 반환된 merged model을 사용하고 기존 `atol=1e-5, rtol=1e-4` 출발 기준으로 전체 출력 정합성을 검사합니다. `safe_merge` 자체를 출력 동등성 검사로 대신하지 않습니다. [PEFT 병합 문서](https://huggingface.co/docs/peft/v0.21.0/package_reference/lora#merge-lora-weights-into-the-base-model)

LINEAR_RES는 CPU FP32 경로도 측정합니다. 기존 SHARED/FACTOR의 가중치를 재사용해 같은 CPU 입·출력 조건으로 함께 측정하되 추가 적합은 하지 않습니다. GPU 모델 간 비교와 CPU/GPU 배포 선택을 분리합니다. 파라미터 수·실제 갱신 수·전체 저장 크기·학습 메모리·추론 allocated/reserved·프로세스 메모리·시간을 각각 보고하며, WDDM에서 확인되지 않는 전체 GPU 프로세스 메모리는 미측정으로 남깁니다.

**7. 신규 예산과 재발 방지**

[확인] 현재 RTX4070은 12,282MiB, RAM은 약31.77GiB, E 드라이브 여유는 약457GiB입니다. Python 학습 프로세스는 없었지만 데스크톱 앱의 GPU 사용은 있으므로 무부하 상태라고 가정하지 않습니다.

사용자 제안 상한을 유지합니다.

```text
실자료 적합       최대16attempt
                  최초8 + 필요한 후속 최대4 + 복구·보완 최대4

GPU               로컬1장, 동시 학습1개
                  작업 점유 누적10,800초 = 3시간

CPU 검사          누적15분
합성 optimizer    최대6세션, 세션 전체 optimizer 업데이트 합계10step 이하

신규 저장         최대5GiB
```

계상은 실행 전에 예약합니다.

- CPU/GPU 실자료 적합, closed-form 적합, optimizer smoke, 실패·중단·재시작은 각각 실자료 attempt에 기록합니다.
- 한 명령의 여러 설정을 하나로 세지 않습니다.
- 합성 optimizer 검사는 재실행할 때마다 새 세션입니다. 여러 test case의 업데이트를 합쳐 세션당10step 이하로 제한합니다.
- 학습 없는 assert/복원 검사는 CPU 검사 시간에 포함하고 실자료 fit로 세지 않습니다.
- 실자료 추론·benchmark는 fit가 아니지만 GPU 사용 시 점유 원장에 포함합니다. CPU 배포 benchmark·집계 시간도 별도 기록합니다.
- 원장 잠금과 사전 예약은 하나의 실행 주체가 관리합니다. worker가 임의로 테스트·학습을 시작하지 못하게 합니다.
- 실행 개시 후 실패한 예약은 소진으로 남깁니다. 미착수 예약만 사유를 남기고 취소합니다.

[추정·미검증] GPU 운영 배분은 복원·검사10분, 최초8fit40분, 후속4fit20분, 복구·보완25분, 비용 측정·진단75분, 최종 평가·검산10분으로 합계180분입니다. 항목 간 조정은 가능하지만 총상한은 늘리지 않습니다.

기존 선택 TSFM fit은 Electricity 약87–192초, Bull FIXED 약27–30초였습니다. 새 T의 시간은 아직 미측정입니다. 첫8fit가 모두120epoch까지 가면 정규 VAL 평가는 최대66,792원점이며, 이는 계산량 설명이지 완료시간 예측이 아닙니다. v2 반복 비용156행은 약23.1분이었으므로 이번 본측정은 약35–45분을 운영 추정으로 두고 진단 여유를 포함해75분을 배분했습니다.

새 저장은 작은 checkpoint·예측·제한된 trace를 포함해 약0.1–0.5GiB로 예상하지만 미실측입니다. raw와 백본을 복제하지 않고 실제 누적 크기로5GiB를 제한합니다.

충돌 대기는 회당30분·누적60분입니다. 진행 없는 작업은10분부터 상태를 확인하고30분 동안 근거 있는 진전이 없으면 자신의 작업만 중단합니다. 다른 프로젝트 프로세스·전력·clock·드라이버·전역 환경은 변경하지 않습니다.

**8. 결과별 결정과 완료·게시**

- **TSFM 우위가 남음:** 정상 학습된 선형 대체보다 유리한 자료·기간과 비용을 명시해 유지 또는 조건부 유지합니다. 다음 단일 결정은 그 범위를 고정한 새 보호 확인의 설계입니다. 새 자료 확보·평가는 이번에 실행하지 않습니다.
- **선형 대체로 충분함:** 현재 구조·자료에서 TSFM 추가 비용을 정당화하는 근거가 약하다고 보고합니다. 이를 일반 선형모델 연구의 성공이나 모든 TSFM PEFT의 불필요성으로 바꾸지 않습니다.
- **기간별 방향이 반대임:** 전체 손익과 불확실성을 남깁니다. 사후 채널 제외나 라우팅 정책을 만들지 않습니다.
- **학습·측정 문제가 남음:** 유효한 정확도 비교와 미확정 비용·학습 부분을 구분합니다. 넓은 구간을 동등성 증명으로 쓰지 않습니다.

승인 후 새 폴더에 PLAN.md·STATUS.md·TOPIC_DECISION.md와 필요한 실행 원장만 둡니다. 승인 원문, 초기/선택/재시작 상태, 코드·자료·가중치 hash, seed·원점·선택·예측·지표·자원을 연결합니다. 검산은 실행팀의 확인이며 독립 재현으로 부르지 않습니다.

관련 변경만 일반 commit·`origin/main` push하고 원격 반영을 확인합니다. v1/v2 원본과 기존 dirty 항목은 보존하고 raw/cache/가중치·비밀정보는 게시하지 않습니다. 재개할 때는 실제 프로세스와 원장을 먼저 확인합니다.

**승인 대상은 위 v3 계획 전체입니다. 승인 전에는 실행하지 않고 기다리겠습니다.** 이는 이번 사용자 지시의 승인 경계이며, 적용한 [research-plan-loop](C:/Users/User/.claude/skills/synced/a4a85a5d-7fa9-4c8d-866c-fbdbba10af00_ee36e183-ea13-46c0-83bf-ec9a51839098/research-plan-loop/SKILL.md)의 “합의 전 실행 금지”와도 일치합니다.
