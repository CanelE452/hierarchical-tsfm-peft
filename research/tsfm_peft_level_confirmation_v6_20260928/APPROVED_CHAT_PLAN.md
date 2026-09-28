**v6는 Robin 사무용 전력 17채널에서 LEVEL을 고정 비교하고, 그 결과로 영문 논문 초안까지 완성하는 회차로 설계합니다. 기본 12fit, 복구·학습상한 보완용 예비 4fit, 총 최대 16attempt입니다.**

[확인] 로컬 `main`과 실제 원격 `origin/main`은 모두 `9ddc0368716d32007b9af89859c7e91aa66be6fa`입니다. 추적 파일 변경은 없고, 기존 미추적 파일 6개는 보존했습니다. 이번에는 읽기와 계획만 수행했으며 파일 수정·학습·새 예측·채점·benchmark·push는 하지 않았습니다.

한 가지 접근 한계는 있습니다. `TSFM_PEFT_V6_PLAN_20260928.txt` 실물은 Downloads와 첨부 디렉터리에서 찾지 못했습니다. 아래는 **현재 메시지의 요구사항과 실제 v4/v5 코드·기록을 근거로 작성한 실행계획**입니다. 첨부 원문을 읽었다고 주장하지 않습니다. 아래 실행 규칙과 예산은 승인 대상인 **[제안·미검증]**입니다.

**1. 목표와 주장할 기여**

목표는 이미 선택한 LEVEL의 **기여를 선행과 구분하고, 방법 선택에 사용하지 않은 한 집단에서 고정 확인한 뒤 실제 결과를 담은 원고를 만드는 것**입니다. Hog 추가 튜닝이나 새 PEFT 탐색으로 돌아가지 않습니다.

[확인] v5에서는 Hog MSE가 기존 RES `3.220290`에서 LEVEL RES `2.666809`로 낮아졌고, matched RAW는 `2.949584`였습니다. 그러나 Bull에서는 matched RAW의 전체 MSE가 더 낮았고, Electricity에서는 LEVEL이 기존 RES보다 악화했습니다. 따라서 보편적 개선을 전제할 수 없습니다. [v5 판단 원문](E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_accuracy_recovery_v5_20260927/TOPIC_DECISION.md)

논문에서 검증할 기여는 다음 세 가지입니다.

- **설계:** 압축 TSFM이 제외한 채널 재구성 잔차에 최근 수준 복원과 작은 시간 예측기를 결합한다.
- **구조적 성질:** 고정 직교 PCA와 공유·무편향 시간 선형 G라는 조건에서, 보정항은 제외된 채널 부분공간에 놓이며 보유 PCA 공간의 TSFM 예측을 바꾸지 않는다.
- **실증:** 기존 개발 결과와 새 Robin 고정 확인을 분리하고, OLD_RES·matched RAW·LEVEL_ONLY·실용 대안으로 효과를 구분하면서 정확도와 같은 회차 비용을 보고한다.

가까운 선행과의 경계도 고정합니다.

- **AdaPTS, ICML 2025:** 잠재 입출력 어댑터, 차원 축소, 동결 단변량 FM 적용은 중복입니다. 관련 방법절·공식 코드 검토 범위를 근거로 쓰며, `COMPRESS`를 AdaPTS 전체 확률모형과 최적 튜닝의 재현이라고 부르지 않습니다. [공식 논문집](https://proceedings.mlr.press/v267/benechehab25a.html)
- **NLinear, AAAI 2023:** 마지막 입력값을 빼고 예측 후 복원하는 원리는 기존입니다. 차이는 이를 **원입력 단독 예측이 아니라 PCA 재구성 잔차 분기**에 적용한다는 점입니다. [저자 코드](https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py)
- **RevIN, ICLR 2022:** 문맥 평균·표준편차와 선택적 affine을 사용하는 방식입니다. LEVEL은 마지막 잔차 수준만 처리합니다. 저자 코드를 확인했으며 원문 재열람 제한은 기록합니다. [저자 코드](https://github.com/ts-kim/RevIN/blob/master/RevIN.py)
- **ELF, ICML 2025 / EPOC, 2026 preprint:** 동결 모델의 작은 보정기, 예측 잔차·끝점 보존이라는 넓은 표현은 인접합니다. 이들은 온라인 피드백·완료된 예측 오차를 사용하는 반면, 이번 방법은 **입력의 채널 재구성 잔차를 사용하고 TEST 중 갱신하지 않습니다.** 읽은 깊이는 ELF 공식 초록, EPOC 초록·방법절로 표시합니다. [ELF](https://proceedings.mlr.press/v267/lee25ag.html), [EPOC](https://arxiv.org/abs/2609.30929)

PCA, 정규화, 잔차 보정, 마지막 값 복원의 최초성을 주장하지 않습니다. 검색에서 같은 수식을 못 찾았다는 사실도 최초성 근거로 쓰지 않습니다.

**2. 새 집단·원천·노출 범위**

새 집단은 **BDG2 Robin site의 Office electricity 17계열**, `C=17`, `K=ceil(17/4)=5`로 고정합니다. 메타데이터상 교육·대학 시설의 사무용 건물군이므로 일반 상업 건물 전체의 확인으로 확대하지 않습니다.

선택 목록은 다음과 같습니다.

```text
Robin_office_Addie       Robin_office_Adolph
Robin_office_Antonina    Robin_office_Dina
Robin_office_Donald      Robin_office_Erma
Robin_office_Gayle       Robin_office_Lindsay
Robin_office_Maryann     Robin_office_Sammie
Robin_office_Saul        Robin_office_Serena
Robin_office_Shirlene    Robin_office_Soledad
Robin_office_Victor      Robin_office_Wai
Robin_office_Zelma
```

[확인] 기존 로컬 raw 원천을 사용하며 추가 다운로드는 없습니다.

- [electricity.csv](E:/CODING/proj/mltimeseries/data_external/bdg2_coarse_supervision_v1/raw/electricity.csv): 174,239,039bytes
- [metadata.csv](E:/CODING/proj/mltimeseries/data_external/bdg2_coarse_supervision_v1/raw/metadata.csv): 272,024bytes
- [기존 다운로드 receipt](E:/CODING/proj/mltimeseries/data_external/bdg2_coarse_supervision_v1/receipts/electricity.json)

```text
Source revision
9b97ccbe90096aff42ed4fd6493bf7ae692d7118

electricity SHA256
039d909d8981e2d69eaeb366144e6ab7e84fa5e7e216aee42bddd95384a66418

metadata SHA256
992d0b29f24f96ad4332bc4dbb534b7bdd7dd2689aad093f94e93068ecddca02
```

성능과 무관한 site ID 순서, 이전 노출 기록, TRAIN 관측률≥0.95·비상수·최소 16채널 조건으로 좁혔습니다. Cockatoo는 TRAIN 적격 0개, Panther는 3개였고, Eagle·Fox·Lamb·Rat에는 기존 사용 기록이 있었습니다. Robin의 17개는 TRAIN 1,992행에서 모두 비상수이고 관측률은 `1991/1992~1.0`이었습니다. 이 최소 TRAIN 확인에 새 모델 성능은 사용하지 않았습니다.

노출 검색은 현재 저장소와 접근 가능한 `mltimeseries`, `covariate-trust-pilot`, `tsfm-peft-method-screen`의 관련 결과·manifest·문서 범위에서 수행했습니다. 근거에는 [기존 노출 원장](E:/CODING/proj/mltimeseries/results/peft_paper_closure_v1/data_exposure_ledger.csv)과 [prediction manifest](E:/CODING/proj/tsfm-peft-method-screen/results/building_peft_topic_decision_20260916/prediction_manifest.json)가 포함됩니다.

정확한 표현은 **“검색한 로컬 기록 범위에서 Robin을 이번 방법 선택에 사용한 흔적을 찾지 못했다”**입니다. 전체 과거 이력이나 Chronos 사전학습 중복 부재의 증명은 아닙니다. 자료 부적격이나 실제 노출이 추가 확인되면 이를 보고하고, 승인 없이 다른 site로 교체하지 않습니다.

**3. 시간·전처리·평가 단위**

BDG2의 시간당 기록과 기존 raw 계약을 유지합니다. Robin 메타데이터는 `Europe/London`이지만 제공된 local-naive 시간 격자를 그대로 사용하며 UTC 변환이나 DST 재구성을 하지 않습니다. [BDG2 저자 자료 설명](https://github.com/buds-lab/building-data-genome-project-2)

```text
과거 문맥  [2016-01-01, 2016-01-23)
TRAIN      [2016-01-23, 2016-04-15)  stride 1   1,945원점
VAL        [2016-04-17, 2016-05-18)  stride 24     30원점
TEST-A     [2016-05-20, 2016-06-30)  stride 24     40원점
TEST-B     [2016-06-30, 2016-08-10)  stride 24     40원점

L=512, H=48
각 목표 48개 전체가 해당 구간 안에 있어야 함
```

원점 수는 날짜·시간 격자에 따른 값이며, 승인 후 준비 단계에서 실제 인덱스와 대조하고 목록을 저장합니다.

- 평균·표준편차·PCA는 TRAIN만 사용합니다.
- 기존 코드처럼 입력 결측은 TRAIN 평균으로 대치합니다. 표준화 뒤에는 0이며, 보간하지 않습니다.
- 원래 관측 mask를 유지하고 대치한 미래를 정답으로 채점하지 않습니다. 0값을 자동 결측 처리하지 않습니다.
- 마지막 입력이 결측이면 그 대치값으로 LEVEL의 마지막 수준을 계산합니다.
- 후속 TEST 원점에서 이미 관측 가능한 과거 TEST 입력은 사용할 수 있지만 온라인 학습은 하지 않습니다.
- 주 평가는 TEST-A/B의 **채널별 오차합과 관측수를 먼저 합친 다음 채널 평균**을 냅니다. 기간 MSE의 단순 평균으로 대체하지 않습니다.

승인 후 TRAIN/VAL과 TEST 접근을 분리한 v6 전용 loader를 만듭니다. 준비 중 TEST 통계·그림·오차를 출력하지 않으며, 채널·기간·원점은 성능을 보고 바꾸지 않습니다.

**4. 고정 수식과 파라미터**

\(X\in\mathbb R^{L\times C}\), TRAIN PCA \(U\in\mathbb R^{C\times K}\), \(U^\top U=I\)로 둡니다.

\[
E(X)=XU,\quad D(Z)=ZU^\top,\quad
r=X-D(E(X)),\quad b=D(F_0(E(X))).
\]

\(P_n(u)\)는 마지막 입력 행을 `detach`하고 \(n\)번 반복한 값입니다.

\[
\begin{aligned}
\mathrm{LEVEL\_RES}&=b+P_H(r)+G_{\rm res}(r-P_L(r)),\\
\mathrm{LEVEL\_RAW}&=b+P_H(r)+G_{\rm raw}(X-P_L(X)),\\
\mathrm{OLD\_RES}&=b+G_{\rm old}(r),\\
\mathrm{LEVEL\_ONLY}&=b+P_H(r),\\
\mathrm{COMPRESS}&=b.
\end{aligned}
\]

[확인] 이 LEVEL·matched RAW 정의는 [v5 실제 구현](E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_accuracy_recovery_v5_20260927/model_v5.py:154)과 연결됩니다.

- 모든 압축군은 **새 Robin TRAIN PCA E/D를 고정**합니다. CURRENT·K 탐색은 없습니다.
- G는 채널 공유·무편향·무활성화 `512→32→48`, 출력층 0 초기화입니다.
- G 학습 파라미터는 `17,920`, 고정 E/D는 `170`개입니다.
- LEVEL_RES와 RAW는 같은 초기 텐서와 초기 함수 \(b+P_H(r)\)를 갖습니다. OLD_RES의 초기 함수는 \(b\)이므로 동일하지 않습니다.
- F0는 동결 Chronos-Bolt-small, revision `772f3d25d38aec6d914c8949dab4462e2d46f5d8`, FP32·dropout off이며 중앙값 예측의 앞 H48을 사용합니다.
- 이전 Hog/Bull/Electricity의 학습된 E/D/G/T/LoRA는 복사하지 않습니다.

선형 대체도 같은 LEVEL 구조를 사용합니다.

\[
\mathrm{LEVEL\_LINEAR\_RES}
=D(\widetilde T(E(X)))+P_H(r)+G(r-P_L(r)).
\]

\(\widetilde T\)는 [v3 구현](E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_backbone_value_v3_20260927/model.py:69)의 문맥 평균·표준편차 정규화 및 역변환을 포함한 공유 `Linear(512,48,bias=True)`입니다. T는 24,624개, T+G는 42,544개 학습 파라미터입니다. 0분산 처리도 기존 구현을 유지하고, T와 G는 독립 객체·독립 초기화 난수 문맥을 사용합니다. 선형 모델에는 F0를 로드하지 않습니다.

방법 설명에서는 두 가지를 분명히 합니다.

- 고정 PCA·공유 선형 G에서 보정항이 제외된 공간에 놓인다는 것은 구조적 성질이지 성능 원인의 인과증명이 아닙니다.
- LEVEL은 학습 파라미터를 늘리지 않지만 함수공간과 초기 함수를 바꿉니다. G의 시간 행렬 rank가 최대 32여도 수준 복원까지 포함한 유효 행렬 rank는 최대 33입니다. 이를 동일 용량·순수 정규화 효과라고 부르지 않습니다.

**5. 비교군과 기본 12fit**

```text
비교군                 설정                         실자료 fit
LEVEL_RES              FIXED_ED K5 × 두 seed              2
OLD_RES                FIXED_ED K5 × 두 seed              2
LEVEL_RAW              FIXED_ED K5 × 두 seed              2
LEVEL_LINEAR_RES       FIXED_ED K5 × 두 seed              2
미압축 LoRA            기존 대표 레시피 × 두 seed          2
SHARED-LINEAR ridge    λ=0.001, 0.1 각각 전체 TRAIN 적합    2
기본 합계                                                 12

F0 / COMPRESS / LEVEL_ONLY                                 0
```

주 비교는 **LEVEL_RES 대 OLD_RES**입니다. 나머지는 서로 다른 질문에 답합니다.

- matched RAW: 공통 수준복원 조건에서 잔차 입력이 추가 가치를 남기는가?
- LEVEL_ONLY: 학습 G가 최근 잔차 수준 유지보다 더 필요한가?
- COMPRESS: 수준복원·보정 전체가 압축 주경로에 무엇을 더하는가?
- F0·병합 LoRA: 미압축 실용 대안 대비 남은 정확도 손해와 비용은 무엇인가?
- LEVEL_LINEAR_RES: 같은 LEVEL 구조에서도 TSFM 주경로가 필요한가?
- SHARED ridge: 더 단순하고 전체 TRAIN에 적합한 선형 대안으로 충분한가?

SHARED는 공유 affine `512→48`, bias 포함이며, horizon별 관측 `(origin, channel)` 표본의 평균제곱오차에 \(\lambda\|w\|^2\)를 더하고 intercept는 규제하지 않습니다. penalty마다 1fit이며 내부 48개 horizon 풀이는 따로 기록합니다.

두 penalty는 기존 `{0.001, 0.1, 10}`에서 과거 개발 결과를 근거로 미리 좁힌 것입니다. 새 VAL MSE로 선택하고 정확한 동률이면 `0.001`을 선택합니다. [기존 선택 기록](E:/CODING/proj/hierarchical-tsfm-peft/research/tsfm_peft_fresh_group_v4_20260927/shared_ridge_selection.json)

OLD_RAW의 신규 fit은 제외합니다. 따라서 LEVEL 유무×잔차 유무의 완전한 2×2 상호작용은 추정하지 않습니다. 과거 Hog OLD_RAW를 Robin의 대체 결과로 사용하지도 않습니다.

**6. 학습·선택·예비 4fit**

```text
seed                    92601 / 92602
G·T 초기 LR             1e-3
LoRA 초기 LR            1e-4
optimizer               AdamW, weight decay 0
gradient clipping       1
batch                   4
epoch당 TRAIN 원점       512, modulo-24 위상균형
최대 epoch              120
early stopping          strict VAL best 미갱신 6회
scheduler               ReduceLROnPlateau
                        factor 0.5 / patience 2 / relative threshold 1e-4
checkpoint              step0 포함 최소 전체 VAL MSE
정확한 동률              먼저 나온 상태
```

동일 seed·epoch의 TRAIN 순서는 `SeedSequence([seed, epoch])` 기반으로 맞추고 원점 순서 hash를 저장합니다. PCA/G 초기 텐서도 명시적으로 공유하고 hash를 확인합니다.

LoRA는 기존 `r8 / alpha16 / q,v / dropout0 / bias none`, native quantile loss를 유지합니다. 다른 학습군은 원채널 masked macro MSE입니다. checkpoint 선택은 모두 VAL MSE지만 **학습 목적 차이가 있는 실용 비교**임을 명시합니다. [기존 LoRA 코드](E:/CODING/proj/hierarchical-tsfm-peft/src/hier_peft/lora.py)

예비 4attempt는 다음 순서로만 사용합니다.

1. API·저장·복원·데이터 처리 오류로 무효가 된 필수 실행 복구.
2. TEST 전, **120epoch 도달·조기 종료 아님·최근 6epoch 안 strict best 갱신**을 모두 만족한 실행의 최대 240epoch까지 한 차례 연장.

연장 자격은 모든 신경망 비교군에 같습니다. 적격 실행 전체가 남은 예산에 들어갈 때만 수행하며, 주 방법만 골라 연장하지 않습니다. 낮은 VAL 점수 자체는 새 LR·K·구조를 추가할 이유가 아닙니다. 재시작·연장도 각각 attempt로 차감합니다.

검사는 실제 G/T 갱신, E/D·F0 동결, 초기 대응, 유한 손실·gradient, mask·축, checkpoint 재생, optimizer/scheduler/RNG 재시작 복원에 집중합니다. FIXED_LINEAR의 T를 `no_grad`로 끊거나 저장 목록에서 빠뜨리지 않습니다. G down의 첫 step gradient가 0인 것과 영구 단절을 구분합니다.

**7. TEST 개봉과 평가 계약**

자료·수식·비교 목록·지표·원점·비용 규칙을 학습 전에 기록하고, 모든 적합과 VAL 선택을 마친 뒤 checkpoint·설정·코드 hash를 TEST 전에 봉인합니다.

예정된 비교는 TEST-A/B에서 함께 평가합니다. 나쁜 VAL이나 먼저 보인 불리한 TEST 결과 때문에 남은 비교를 취소하지 않습니다. 이 분리는 유한한 선택 자료에 대한 과적합과 최종 평가의 선택 편향을 구분하기 위한 것입니다. [Cawley·Talbot, JMLR 2010](https://jmlr.org/papers/v11/cawley10a.html)

주 지표는 TRAIN 표준편차로 정규화된 관측 미래의 채널 동일 가중 MSE입니다. 다음을 함께 저장합니다.

- MSE·MAE, 관측 분모, 전체·기간·채널·seed별 결과.
- OLD_RES·RAW·LEVEL_ONLY·F0·LoRA·선형 대안 대비 절대차와 상대 변화율.
- 두 seed **손실의 평균**. 예측 ensemble과 혼동하지 않습니다.
- 기간 안의 연속 7원점 블록, 2,000회, seed `9262026`의 대응 재표집 구간. 채널·방법을 함께 움직이고 A/B 경계를 넘지 않습니다.

구간은 고정된 두 학습 seed에 조건부이며 전체 학습·선택 불확실성을 포괄하지 않습니다. 작은 차이나 넓은 구간으로 동등성·비열등성을 선언하지 않습니다. 관측수가 0인 평가 채널이나 비유한 예측은 조용히 제외하지 않고 평가 불능으로 기록합니다.

TEST 이후 구조·loss·LR·K·mode·정규화·채널·기간·주 지표를 변경하지 않습니다. 명백한 채점·복원 버그는 최초 결과와 수리 시점을 보존하고 영향받은 비교 전체를 정정합니다. 모델 선택까지 달라져야 한다면 미노출 확인이 유지된다고 주장하지 않고 그 변경만 별도 결정 대상으로 둡니다.

**8. 비용과 전체 예산**

비용은 결과가 긍정인 모델만 골라 재는 단계가 아닙니다. 유효하게 완료된 F0, 병합 LoRA, LEVEL_RES, OLD_RES, LEVEL_RAW, LEVEL_LINEAR_RES와 LEVEL_ONLY를 같은 회차에서 측정합니다. SHARED와 선형 대체의 CPU 배포 경로도 별도로 측정합니다.

```text
장치·설정     로컬 RTX 4070, FP32, TF32 off, CPU thread 4
입력          공통 Robin VAL 24원점
              30원점 중 linspace(0,29,24,dtype=int)
batch         1 / 4
예열          10회
반복          전체 24원점 처리 20회
측정 순서     3블록, F0 앞·뒤 기준 측정
배포 범위     표준화 CPU 입력 → H2D → 전체 [B,48,17] CPU 출력
제외          model load, disk, 채점, 원장 기록
```

LoRA는 반환된 병합 모델을 사용하고 VAL에서 병합 전후 정합성을 먼저 확인합니다. CUDA 동기화·정상 예열 뒤 peak를 초기화하며, batch4 처리량·batch1 지연·allocated·reserved·프로세스 메모리·파라미터 bytes를 구분합니다. [PyTorch benchmark](https://docs.pytorch.org/tutorials/recipes/recipes/benchmark.html), [PEFT 병합 안내](https://huggingface.co/docs/peft/main/conceptual_guides/lora)

모든 블록·seed·반복값을 남깁니다. 기준 모델 전후 IQR 분리나 순위 역전이 나타나면 GPU 상주 입력 진단을 최대 한 차례, 600초 이내로 제한합니다. 이 시간도 예산에 포함합니다. 원인이 남으면 비용 우열을 제한하되 정확도 결과를 폐기하지 않습니다. 과거 timing과 새 timing을 나눠 개선율을 만들지 않습니다.

[확인] 현재 RTX 4070은 12,282MiB이며 조회 시 Python 학습 프로세스는 없었습니다. E 드라이브 여유는 약 484GB입니다. v5의 GPU 사용 1,295.975초는 참고 기록이며 v6 완료시간 예측은 아닙니다.

```text
실자료 적합       기본 12 + 예비 4 = 최대 16attempt
GPU               로컬 1장, 동시 신경망 학습 1개
GPU 작업 점유     누적 최대 4시간
합성 optimizer    최대 6세션, 세션 전체 10update 이하
합성·검사 CPU     누적 최대 15분
신규 저장         최대 5GiB
추가 다운로드     0
GPU 충돌 대기     회당 30분 / 누적 60분
```

실자료 smoke·실패·재개·closed-form 적합도 시작 전에 원장에 예약합니다. TRAIN 통계/PCA는 준비 통계로 별도 횟수·시간을 기록하고, 예측모델 fit과 구분합니다. CPU ridge·자료 준비·분석 시간도 별도로 보고합니다. GPU 상위 job과 내부 fit 시간을 이중 가산하지 않습니다.

GPU 시간의 운영 배분안은 학습·복구 150분, 평가 30분, 비용 40분, 잔여 20분입니다. 실측에 따라 내부 배분은 조정할 수 있지만 총상한은 자동 증액하지 않습니다.

**9. 영문 원고·검산·게시**

승인 후 생성할 폴더는 다음입니다. [확인] 현재 같은 경로는 없어 충돌하지 않습니다.

```text
E:/CODING/proj/hierarchical-tsfm-peft/
  research/tsfm_peft_level_confirmation_v6_20260928/
  .cache/tsfm_peft_level_confirmation_v6_20260928/
```

기존 v4/v5 guard를 해제하지 않고 v6 전용 진입점·데이터 계약·TEST 보호를 만듭니다. 공통 코드를 재사용할 때는 고유 import 이름과 실제 파일 위치를 확인합니다.

```text
PLAN.md / APPROVAL.txt / STATUS.md
protocol.json / data_contract.json / exposure_search.json
공통 예산 원장 / pretest_seal.json
실행별 result / curve / receipt
final_comparison.json / final_comparison.csv
cost 결과와 반복 원본 / figures 및 source manifest
CLAIMS_PRIOR_ART.md / references.bib
PAPER_DRAFT.md
TOPIC_DECISION.md / final_checks.json / publication_receipt.json
```

원고 가제는 **“Level-Preserving Residual Adaptation for Compressed Time-Series Foundation Models”**입니다.

학습 중에는 선행·방법·프로토콜을 작성하고 결과는 `PENDING`으로 둡니다. 실제 결과가 나오면 Abstract, Introduction, Related Work, Method, Structural Properties, Experimental Protocol, Development Results, Robin Fixed Confirmation, Accuracy–Cost Tradeoff, Limitations, Conclusion을 **완성된 영문 문장**으로 채웁니다.

원고에서는 다음을 분리합니다.

- Hog/Bull/Electricity: 노출된 개발 근거.
- Robin: 이번 방법 선택 이후의 한 집단 고정 확인.
- 새 확인 완료, 원고 초안 완료, 투고 준비 완료: 서로 다른 상태.

부정 결과도 전체 초안에 반영합니다. 논문을 직접 제출하거나 저자·소속·채택 가능성을 만들어내지 않습니다. 표·그림·본문 수치와 원본 JSON/CSV·분모·hash를 대조하고, 실행팀 검산을 독립 재현으로 부르지 않습니다.

검산 후 관련 신규 폴더·결과 색인·분리된 v6 이력만 개별 staging하여 일반 commit, `origin/main` push, 원격 SHA 확인까지 수행합니다. raw·가중치·예측 배열·비밀정보·무관한 dirty 파일은 게시하지 않습니다. v1–v5 원본·실패·봉인과 v2 위반 기록은 유지합니다.

**10. 결과별 결정과 승인 범위**

- **LEVEL이 OLD_RES보다 개선하고 RAW·LEVEL_ONLY 대비 가치도 남음:** 수준 보존 잔차 설계의 새 집단 근거를 강화합니다. F0/LoRA/선형 대비 비용·오차 손익으로 적용 범위를 정합니다.
- **OLD_RES보다 개선하지만 RAW 또는 LEVEL_ONLY로 충분함:** 수준 처리 효과와 학습 잔차 설계의 기여를 분리하고 방법 주장을 축소합니다.
- **TSFM은 선형보다 낫지만 LEVEL 추가 효과가 없음:** TSFM 가치와 LEVEL 기여를 구분합니다.
- **단순 선형 대안으로 충분하거나 정상 비교에서 LEVEL이 불리함:** 현재 구성의 제한·반례로 보고합니다. 다른 site나 추가 모듈로 성공을 찾지 않습니다.
- **기간별 상반·큰 불확실성:** 모든 기간을 남기고 적용 조건 미확정으로 정리합니다.
- **필수 차단·예산 도달:** 유효한 범위와 미완료 비교를 명시하고 가능한 선행 정리·원고를 마칩니다.

승인 후에는 자료 준비→구현·검사→12fit→필요한 예비 사용→VAL 선택 봉인→TEST→비용→원고·검산→게시를 같은 범위에서 계속합니다. 일상적인 수리마다 재승인을 요구하지 않습니다. 새 자료·방법·총예산·서버·유료서비스·전역환경 변경과 TEST 이후 모델 선택 변경만 별도 결정이 필요합니다.

**이 전체 v6 실행계획을 승인하시겠습니까?**
