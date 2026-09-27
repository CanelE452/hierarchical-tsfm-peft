제안은 **BDG2 Hog의 사무용 전력 32계열(C32/K8)에서 기존 구조와 선택 규칙을 고정해 확인하는 것**입니다. 잔차 입력의 추가 가치와 정확도–비용 절충을 따로 판단하겠습니다.

이번에는 코드·기록·공식 자료·메타데이터와 최소 TRAIN 품질만 읽었습니다. **파일 변경, 모델 실행, 새 예측·채점, 학습, benchmark, 설치, commit/push는 하지 않았습니다.** 아래 실행 내용은 모두 승인 전 제안입니다.

**1. 확인한 출발점과 적용 지침**

[확인] 작업 저장소는 `CanelE452/hierarchical-tsfm-peft`, branch는 `main`이며, 로컬 HEAD와 원격 main은 모두 `5eaeb55fda2d980055a33b3721b99cd3eab9e9d2`입니다. 추적 파일은 깨끗하고 기존 무관한 미추적 네 항목은 그대로입니다.

프로젝트·상위 경로의 별도 AGENTS.md는 없었습니다. 전역 [AGENTS.md](C:/Users/User/.codex/AGENTS.md)와 이번 사용자 지시를 적용합니다. 과거 실험별 실행 제한을 v4로 자동 확장하지 않으며, v1/v2/v3 원본과 **v2의 22/20회 위반 기록**을 보존합니다.

다음 두 경로는 현재 없으며 승인 후 생성합니다.

```text
E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_fresh_group_v4_20260927/
E:/CODING/proj/hierarchical-tsfm-peft/.cache/tsfm_peft_fresh_group_v4_20260927/
```

[확인] v3는 기존 자료에서 TSFM 경로의 조건부 정확도 가치를 남겼지만, 잔차 PEFT의 새 집단 가치와 안정적인 비용 우위를 확보하지는 못했습니다. 이번에는 그 남은 질문을 확인합니다. [v3 최종 판단](E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_backbone_value_v3_20260927/TOPIC_DECISION.md)

**2. 새 집단: Hog 32계열로 고정**

[확인] 메타데이터의 site ID 순서와 TRAIN 적격성·기존 노출 기록을 적용하면 다음과 같습니다.

```text
Bobcat     적격 2개: 16개 미만
Cockatoo   적격 0개
Eagle      기존 개발 평가 노출
Fox        기존 cold-start 실험의 실제 학습·예측·채점 기록 확인
Gator      적격 8개: 16개 미만
Hog        73개 중 TRAIN 적격 69개 → ID 정렬 앞 32개 선택
```

Fox를 제외하는 근거는 사전학습 가능성이 아니라, `Fox_office_Easter/Israel`의 실제 과거 실행 기록입니다. [해당 실험의 자료 구성](E:/CODING/proj/tsfm-peft-method-screen/experiments/building_coldstart_coverage_v1_20260915/data.py:15), [예측 기록](E:/CODING/proj/tsfm-peft-method-screen/results/building_coldstart_coverage_v1_20260915/prediction_manifest.csv:42)

Hog에서는 **전체 73개를 먼저 적격 판정하고, 통과한 ID를 정렬한 뒤 32개를 선택**했습니다. TRAIN 내 상수 0인 네 계열이 제외됐습니다. 선택한 32개는 모두 비상수이며, 관측률은 `1991/1992`부터 `1.0`입니다. 한 관측이 없는 계열은 `Hog_office_Elizbeth`입니다.

```text
Hog_office_Alexis       Hog_office_Cornell
Hog_office_Almeda       Hog_office_Cortney
Hog_office_Bessie       Hog_office_Darline
Hog_office_Betsy        Hog_office_Denita
Hog_office_Bill         Hog_office_Elizbeth
Hog_office_Buford        Hog_office_Elke
Hog_office_Byron        Hog_office_Elnora
Hog_office_Candi        Hog_office_Eloise
Hog_office_Carri        Hog_office_Elsy
Hog_office_Catalina     Hog_office_Emmanuel
Hog_office_Catharine    Hog_office_Garrett
Hog_office_Charla       Hog_office_Guadalupe
Hog_office_Clemencia    Hog_office_Gustavo
Hog_office_Cordelia     Hog_office_Joey
Hog_office_Corey        Hog_office_Josefina
Hog_office_Corie        Hog_office_Lanell
```

따라서 **C=32, K=ceil(C/4)=8, C/K=4**입니다. PCA 설명분산·상관·예측 성능은 선정에 사용하지 않았습니다.

자료는 기존 로컬 원천을 재사용합니다.

- [raw electricity.csv](E:/CODING/proj/mltimeseries/data_external/bdg2_coarse_supervision_v1/raw/electricity.csv): 174,239,039 bytes
- [metadata.csv](E:/CODING/proj/mltimeseries/data_external/bdg2_coarse_supervision_v1/raw/metadata.csv): 272,024 bytes
- 기록된 원천 revision: `9b97ccbe90096aff42ed4fd6493bf7ae692d7118`
- 두 파일의 SHA256은 [기존 자료 계약](E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_development_20260926/data_contract.json:94) 및 공식 LFS 식별자와 일치합니다.

**추가 데이터·가중치 다운로드 상한은 0 bytes**로 제안합니다. 기존 원천과 Chronos 가중치가 모두 있습니다.

노출 확인 범위는 다음으로 제한했습니다.

- `mltimeseries`, `tsfm-peft-method-screen`, `covariate-trust-pilot`, 현재 저장소의 관련 `results/research/_docs/experiments`.
- 원자료·캐시·배열을 제외한 문서·코드·manifest·결과 텍스트, 파일당 5MiB 이하.
- 기존 노출 원장과 관련 후보 manifest.

이 범위에서 **Hog의 직접 예측·성능·방법 선택 사용 기록은 발견하지 못했습니다.** 발견한 Hog 언급은 사전학습 자료 목록에 관한 같은 메모와 그 인용이었습니다. 원격 전체 이력·미게시 결과·현재 Bolt revision의 사전학습 비중복까지 확인한 것은 아닙니다. 이 한계를 명시하고 진행하며, 사전학습 중복 미확정만으로 Hog를 제외하지 않습니다. [관련 메모](E:/CODING/proj/mltimeseries/_docs/notes/tsfm_topics/04_objective_observation/17_coarse_supervision_data_entry_plan_20260908.md:11)

**3. 시간·전처리·주 평가 계약**

[확인] Hog 메타데이터 timezone은 `US/Central`입니다. CSV는 timezone 정보가 붙지 않은 현지 시각이며, 확인한 TRAIN은 1시간 격자입니다. 저장된 시각을 그대로 사용하고 UTC 변환이나 DST 시각 재배치를 하지 않습니다. 공식 원문은 전력을 한 시간 동안의 에너지량인 kWh로 설명합니다. 공개 `raw`도 원천 통합·초기 정제를 거친 자료이며, 추가 정제본 `cleaned`와 구분합니다. [BDG2 원문](https://www.nature.com/articles/s41597-020-00712-x)

분할은 다음으로 고정합니다. 원점 수는 관측값을 채점하지 않고 시간 인덱스로 확인했습니다.

```text
구간       반개방 날짜 구간                  원점 수   간격
문맥       [2016-01-01, 2016-01-23)              —       —
TRAIN      [2016-01-23, 2016-04-15)          1,945       1시간
VAL        [2016-04-17, 2016-05-18)             30      24시간
TEST-A     [2016-05-20, 2016-06-30)            40      24시간
TEST-B     [2016-06-30, 2016-08-10)            40      24시간
```

- L512/H48이며, 미래 48시간 전체가 해당 분할 안에 있는 원점만 사용합니다.
- TRAIN 평균·모집단 표준편차, PCA, 입력 결측 대체 통계는 **명시 TRAIN의 1,992시간만** 사용합니다. 사전 문맥은 통계 적합에 포함하지 않습니다.
- 입력 결측은 TRAIN 평균으로 대체하고, 미래 관측 mask는 대체 전에 보존합니다. 0을 자동 결측으로 바꾸지 않습니다.
- TEST 후속 원점의 과거 관측은 입력으로 사용할 수 있지만 가중치는 업데이트하지 않습니다.
- TEST 값·분포·그림·관측률을 선정이나 조정에 사용하지 않습니다.

이 전처리는 기존 구현과 연결됩니다. [TRAIN 통계와 적격 판정](E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_development_20260926/data_prepare.py:200)

주 지표는 **TEST-A/B 전체 80원점의 채널 동일 가중 정규화 MSE**입니다. seed별로 각 채널의 두 기간 오차합과 관측수를 먼저 합쳐 계산한 다음 채널 평균을 냅니다. 마지막에 두 seed의 손실을 평균합니다. 기간 MSE의 단순 평균이나 예측 ensemble로 대체하지 않습니다.

MAE·각 기간·각 seed·전체 채널별 결과도 보존합니다. TEST에서 어떤 채널의 관측 정답이 전혀 없다면, 유리하게 채널을 제거하지 않고 해당 집계의 계산 불가 범위를 표시합니다.

**4. 구조·비교군·선택 규칙**

새 구조는 추가하지 않습니다.

```text
TSFM_RES   = D(F0(E(X))) + G(X - D(E(X)))
TSFM_RAW   = D(F0(E(X))) + G(X)
COMPRESS   = D(F0(E(X)))
LINEAR_RES = D(T̃(E(X)))  + G(X - D(E(X)))
```

- F0: 동결 Chronos-Bolt-small, revision `772f3d25d38aec6d914c8949dab4462e2d46f5d8`.
- E/D: C32↔K8 채널 선형층, 새 Hog TRAIN PCA 초기값.
- G: 채널 공유·무편향·무활성화 `512→32→48`, 출력층 0 초기화.
- T: G와 독립인 `Linear(512,48,bias=True)`, 24,624개 파라미터.
- T̃: v3의 문맥 평균·scale 정규화와 역변환을 포함합니다. 평균·scale을 detach하지 않고, 0분산 경계 수리도 재사용합니다. 전역적으로 순수 선형 함수라고 부르지 않습니다. [v3 구현](E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_backbone_value_v3_20260927/model.py:24)

기본 적합 수는 다음과 같습니다.

```text
방법             모드/설정                         실자료 attempt
TSFM_RES         CURRENT/FIXED_ED × 2seed                  4
TSFM_RAW         CURRENT/FIXED_ED × 2seed                  4
LINEAR_RES       CURRENT/FIXED_ED × 2seed                  4
미압축 LoRA      고정 대표 레시피 × 2seed                 2
COMPRESS         CURRENT × 2seed                          2
COMPRESS FIXED   TRAIN PCA + 동결 F0                      0
미압축 F0        공식 동결 가중치                        0
SHARED-LINEAR    ridge penalty 3개                       3
합계             신경망 16 + CPU ridge 3                19
```

seed는 `92601/92602`입니다.

선택은 다음으로 사전 고정합니다.

1. 각 seed는 **step0을 포함한 최소 전체 VAL MSE checkpoint**를 선택합니다. 동률이면 먼저 나온 상태입니다.
2. RES·RAW·LINEAR는 각각 두 seed의 best VAL 평균으로 모드를 독립 선택합니다. 정확한 동률이면 `FIXED_ED`를 선택합니다.
3. COMPRESS도 CURRENT의 두 seed 평균과 FIXED의 단일 결정론적 값을 비교합니다. FIXED 값을 복제해 독립 실행처럼 세지 않습니다.
4. **잔차 설계의 주 대조는 RES가 선택한 모드에서의 RES–RAW**입니다. 각 방법이 독립 선택한 구성끼리의 비교도 별도로 보고합니다.
5. LINEAR·COMPRESS도 자체 선택 결과와 RES 선택 모드에서의 대응 결과를 구분합니다.

SHARED는 전체 TRAIN의 관측 정답을 사용하는 기존 affine ridge를 재사용하고, penalty는 **`0.001, 0.1, 10.0`**으로 고정합니다. VAL MSE로 선택하며 정확한 동률이면 이 나열 순서를 따릅니다.

[확인] 기존 ridge는 horizon별 관측 origin×channel 쌍의 **평균 제곱오차 + λ‖w‖²**이며 절편에는 벌점을 주지 않습니다. 결측 수가 다르면 적합 목적의 채널 가중이 평가의 macro MSE와 달라질 수 있다는 제한을 보고합니다. [penalty 목록](E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_development_20260926/linear_fulltrain.py:20), [masked ridge 정의](E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_development_20260926/masked_linear.py:55)

E/D 공동학습과 잠재 어댑터는 AdaPTS의 기존 원리입니다. 이번 COMPRESS는 가까운 구조 대조이며 AdaPTS 전체 확률모형의 재현이 아닙니다. G는 최종 미래 예측 손실로 학습하고, 별도 미래 잔차 정답을 사용하지 않습니다. [AdaPTS 공식 논문·코드](https://proceedings.mlr.press/v267/benechehab25a.html)

**5. 학습 레시피와 구현 확인**

기존 레시피를 다음과 같이 고정합니다.

- FP32, 백본 dropout 비활성, TF32 비활성.
- E/D·G·T 초기 LR `1e-3`; FIXED에서는 E/D만 고정.
- LoRA: `r8`, `q/v`, `alpha16`, dropout0, bias 없음, LR `1e-4`.
- AdamW, weight decay0, clip1, batch4, epoch당 TRAIN 512원점.
- 최대120epoch, strict VAL 개선이 없는 6epoch 후 종료.
- 기존 plateau scheduler: factor0.5, patience2, relative threshold `1e-4`. checkpoint의 strict 개선 판정과 scheduler 판정은 구분합니다.

[확인] 1,945 TRAIN 원점은 위상별 81/82개이므로, 기존 위상균형 sampler의 21/22개 비복원 추출이 가능합니다. 같은 seed·epoch의 원점 순서를 모든 대응 모델에 공유합니다. [sampler](E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_backbone_value_v3_20260927/runtime.py:180)

어댑터·LINEAR는 원채널 masked macro MSE로 학습합니다. LoRA는 기존 native quantile loss로 학습하고 median 출력의 VAL MSE로 선택합니다. 이 목적함수 차이는 실용 비교의 제한으로 남깁니다. [LoRA 구성](E:/CODING/proj/hierarchical-tsfm-peft/src/hier_peft/lora.py:9)

새 Hog PCA와 seed별 G 초기값을 명시적으로 생성·공유하고 tensor hash를 기록합니다. T는 별도 RNG(`seed+3,000,000`)로 초기화합니다. 이전 site에서 학습된 어댑터는 복사하지 않습니다.

승인 후 구현은 **v4 전용 데이터 계약·실행기·원장**으로 분리합니다. 기존 두 자료 전용 guard를 해제하지 않고, 재사용 모듈은 명시적 경로와 고유 alias로 읽어 실제 import 위치를 기록합니다.

검사는 다음에 집중합니다.

- CURRENT_TSFM에서 동결 F0를 통과한 gradient가 E까지 도달하는지.
- FIXED_LINEAR에서 E/D는 불변이고 T/G는 실제 갱신되는지.
- G의 초기 출력 0, 첫 step의 down gradient 0과 영구 단절의 구분.
- mask·축·유한 loss, 초기 대응 함수, 선택 checkpoint와 optimizer/scheduler/RNG 복원.
- LINEAR의 저장 상태에 T가 포함되고, 배포 인스턴스가 F0를 로드하지 않는지.

고정 TRAIN probe와 전체 VAL의 초기·선택 점수를 기록합니다. 합성 검사 통과나 짧은 smoke를 충분한 학습의 증거로 사용하지 않습니다.

**6. TEST 전후 변경과 자동 진행 경계**

승인 후에는 자료 준비 → 구현 확인 → 기본19fit → VAL 선택 → 설정 봉인 → 고정 TEST 평가 → 비용·검산 → 판단·게시까지 자동 진행합니다. 정상적인 음성 VAL 결과도 예정된 TEST 확인에서 제외하지 않습니다.

TEST 전에는 재현 가능한 API·경로·저장·gradient·mask 오류를 근거와 diff를 남겨 수리할 수 있습니다. 영향을 받은 fit과 선택은 무효 표시하고 같은 규칙으로 재실행합니다.

학습 상한 보완은 다음으로 제한합니다.

- 120epoch에 도달했고 patience 종료 조건에 이르지 않아 최근 6epoch 안에 best VAL이 갱신된 실행을 대상으로 합니다.
- 해당 설정의 두 seed에 동일하게 최대240epoch 규칙을 제공하되, 이미 patience로 종료한 seed는 그대로 둡니다.
- 미종료 실행만 저장된 상태에서 한 번 재개하며, 재개마다 새 attempt로 셉니다.
- 같은 조건에 해당하는 보완을 남은 예산으로 모두 수용할 수 있을 때 적용합니다. 일부 방법만 유리하게 연장하지 않습니다.
- 낮은 VAL 점수만을 이유로 LR·모드·구조를 추가하지 않습니다.

TEST 전에는 자료·채널·원점·주 지표·선택 결과·checkpoint hash·비교 역할·비용 조건을 고정합니다. **16개 신경망 checkpoint, 고정 COMPRESS, F0, VAL 선택 SHARED의 최대19개 고유 모델**을 한 평가 단계에서 확인합니다. 중복 함수는 재사용하고 TEST로 모드를 재선택하지 않습니다.

TEST 후에는 점수에 반응하는 학습·구조·기간·채널 변경을 금지합니다. 재현 가능한 채점·복원 버그는 관련 비교 전체를 함께 정정하고 최초 노출 시점을 남깁니다. 모델 선택까지 바뀌는 문제는 미노출 확인 지위를 유지한다고 주장하지 않고 별도 변경 승인을 받습니다.

이 구분은 VAL 선택 자체의 과적합 가능성과 TEST 선택 금지를 다루는 근거에 따릅니다. 이번 시계열에 무작위 교차검증을 추가하는 계획은 아닙니다. [Cawley–Talbot, JMLR 2010](https://jmlr.org/papers/v11/cawley10a.html)

**7. 정확도 불확실성과 비용 측정**

정확도 차이는 같은 원점에서 절대차·비교군을 분모로 한 상대차·seed별 값으로 보고합니다.

조건부 구간은 기존 방식에 맞춰 **연속7원점 블록, 2,000회, RNG seed `9262026`**으로 계산합니다. 모든 채널과 비교 모델에 같은 재표집을 적용하고, TEST-A/B 안에서 각각 뽑아 오차합·관측수로 합칩니다. 경계를 가로지르는 블록은 만들지 않습니다. 이는 고정된 두 학습 seed에 대한 기간 변동 요약이며, 전체 학습·선택 불확실성이나 동등성 증명이 아닙니다.

비용의 사전 관심은 **batch4 처리량과 peak allocated GPU 메모리**입니다. batch1 지연·전체 배포 크기·학습 시간도 함께 보고합니다.

측정 조건은 다음으로 고정합니다.

- 현재 RTX 4070, FP32, TF32 off, CPU thread4, 공통 inference mode.
- VAL 30원점 중 시간순으로 균등 선택한 동일24원점.
- batch1/4, 예열10회, 전체24원점 처리20회, 순서3블록.
- GPU 범위: **표준화된 CPU 입력 → GPU 전송 → 전체 `[B,48,32]` CPU 출력**, 동기화된 wall time.
- 로드·디스크·채점·원장 기록은 측정 밖에 둡니다.
- VAL 선택 RES/RAW/LINEAR/LoRA/COMPRESS와 F0를 측정합니다. 각 seed를 보존하며 GPU 주 대상은 최대11개입니다.
- LINEAR 두 seed와 선택 SHARED의 CPU 경로는 별도 실용 비교로 측정합니다.
- LoRA는 병합 반환 모델을 사용하고 동일 VAL에서 병합 전후 출력 정합성을 확인합니다.
- 모델 교체 시 참조·allocator·workspace를 정리하고 shape별 예열 뒤 peak를 초기화합니다. allocated/reserved/프로세스 메모리/모델 bytes를 구분합니다.

순서는 고정 목록·역순·순환 이동의 세 블록으로 구성하고 F0를 각 블록 앞뒤에 둡니다. 블록별 원자료와 분포를 보존합니다. 예열·thread·CUDA 동기화는 [PyTorch 공식 안내](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html), 병합은 [PEFT 공식 문서](https://huggingface.co/docs/peft/main/conceptual_guides/lora)와 설치된 0.21.0 구현을 기준으로 합니다.

F0 앞뒤 분포가 분리되거나 블록마다 비용 순위가 뒤집히면, 대표 F0/RES/LINEAR의 GPU 상주 진단을 **한 차례, GPU 예산 안에서 최대10분** 허용합니다. profiler 시간은 성능표에 넣지 않습니다. 원인이 남으면 비용 우열을 미확정으로 보고하며 유효한 정확도 비교는 유지합니다.

**8. 예산과 완료 판단**

[확인] 현재 장치는 RTX 4070 약12GiB, RAM 약32GiB이며 E 드라이브 여유는 약457GiB입니다. 조회 시 GPU 사용률이 있었으므로 실행 시 충돌 여부를 다시 확인합니다.

운영 상한은 요청안대로 유지합니다.

```text
실자료 적합        최대24 attempt = 기본19 + 복구/상한 보완5
GPU                로컬1장, 동시 신경망 학습1개, 누적4시간
합성 optimizer     최대6세션, 세션 전체 업데이트10step 이하
CPU 검사           누적15분
신규 저장          최대10GiB
추가 다운로드      0 bytes
충돌 대기          회당30분, 누적60분
```

[추정·미검증] GPU 배분 출발안은 학습·복구160분, 평가·복원20분, 비용50분, 나머지10분입니다. 실제 측정 후 내부 배분은 조정할 수 있지만 총4시간은 늘리지 않습니다. 기존 v2의 GPU58.16분, v3의34.90분은 규모 참고이며 새 자료의 완료시간 보장이 아닙니다.

계상은 실행 전에 공통 원장에 예약합니다.

- 실자료 한 설정·한 seed의 적합은 CPU/GPU와 관계없이 1attempt.
- ridge penalty 세 개는 3attempt; 내부48개 horizon solve는 별도 기록.
- 실자료 optimizer smoke·실패·재시작도 attempt에 포함.
- TRAIN 표준화/PCA는 데이터 준비 통계로 시간·횟수를 별도 기록.
- F0/고정 COMPRESS는 fit0이지만 추론 시간을 기록.
- GPU 상위 job과 내부 fit 시간을 이중 가산하지 않음.
- CPU ridge·CPU 비용 측정은 합성 검사15분에 숨겨 넣지 않고 별도 기록.

다른 프로젝트 프로세스 종료, 전역 환경·드라이버·전력 설정 변경은 하지 않습니다. 필수 비교를 상한 안에 완료하지 못하면 TEST를 성급히 개봉하지 않고 미완료 범위를 보고합니다.

결정은 세 질문을 분리합니다.

- **RES가 대응 RAW보다 유리하고 실제 비용 절충도 남음:** Hog 범위에서 잔차 설계를 유지할 근거 강화.
- **잔차 효과는 있으나 F0/LoRA/선형 대비 선택 가치가 약함:** 제한적 유지, 정확도 손해를 줄이는 개발 필요.
- **TSFM은 선형보다 좋지만 RES가 RAW/COMPRESS보다 낫지 않음:** TSFM 가치와 잔차 방법의 근거를 구분하고 후자는 약화로 판단.
- **기간별 반대·큰 불확실성:** 적용 조건 미확정 또는 보류.
- **정상 비교에서 단순 대안으로 충분함:** 현재 구성의 보류·축소 근거.
- **필수 자료·구현·측정 부재:** 방법 실패와 구분해 판단 불가 범위 표시.

임의 개선율·비열등성 허용폭·단일 정확도–비용 점수는 만들지 않습니다. 좋은 결과가 나오면 무엇이 새로 확인됐는지 명시하되, 한 site 확인을 범용 우수성이나 논문 전체 검증 완료로 확대하지 않습니다.

승인 후 `PLAN.md`, `STATUS.md`, `TOPIC_DECISION.md`와 필요한 실행·선택·예산 기록을 남깁니다. 방법 경로, 전체/기간 정확도, 정확도–처리량·메모리 그림은 원수치에 연결합니다. 관련 코드·문서·요약 결과만 검증해 일반 commit과 `origin/main` push 후 원격 반영을 확인합니다. raw·가중치·예측 배열과 무관한 dirty 파일은 게시하지 않습니다.

**이 전체 계획의 승인을 기다리겠습니다. 승인 후에는 위 범위의 구현·실험·수리·고정 평가·보고·게시를 이어가겠습니다.**
