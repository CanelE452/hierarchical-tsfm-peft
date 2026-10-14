# 작은 K의 예측 손실 개선 → 미래 표현 정렬 → 새 자료 검증

2026-10-13. 사용자가 active goal에서 다음 실험의 우선순위를 정하고 실제 실행한 뒤, ExoTimer에서 착안한 미래 표현 정렬과 독립 새 평가자료 검증까지 묻지 않고 이어가도록 승인했다. 목적과 데이터 계약을 먼저 고정하고 이 전체 범위를 수행한다. 성능 개선을 보장하지 않으며, 음성 결과로 뒤의 정렬·새 자료 블록을 생략하지 않는다.

## 목적과 우선순위

소비처는 사용자와 지도교수가 압축 연구의 목적·방법·후속 기여를 판단하는 것이다. 최상위 목적은 과거 다변량 입력에서 동결 TSFM이 사용할 작은 연속 성분을 만들고 정확도와 실제 비용의 교환을 확인하는 것이다.

[확인] 기존 작은 K의 D는 모든 원점과 horizon에 공통인 affine Linear(K,C)다. 원채널 전체 미래를 한 K차원 affine 채널 부분공간에서 복원해야 한다. 이는 K가 커질수록 오차가 줄었던 결과와 양립하지만 원인이 확정됐다는 뜻은 아니다.

가장 먼저 채널별 현재 상태 C개를 보존하고 미래 변화만 K개 TSFM 성분으로 예측한다. 그다음 미래 예측에 관련된 FIT teacher output의 부분공간을 초기화에 사용한다. 마지막으로 같은 배포 구조에 연속 미래 정렬과 학습된 미래 패턴 분할의 보조 목적을 추가한다. nonlinear D와 pointwise E를 대조해 단순 구조로 충분한지 함께 본다.

가능성이 낮은 대안인 이진 hash를 직접 Bolt 입력으로 넣는 방식은 연속 시계열 입력을 깨고 hash bit와 채널 K를 혼동할 우려가 크다. 이진 경계 자체가 regime을 안정적으로 분리할 가능성은 있지만 현재 본 문제의 비용·예측 근거가 없어 이번에는 보조 학습에서만 분할 아이디어를 사용한다.

## 선행 근거와 변경 범위

- [NLinear 공식 코드](https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py)는 마지막 관측값을 빼고 예측 뒤 더하는 방법을 사용한다. 이번 구조는 동결 F의 K채널 변화량을 원자료 마지막 C채널 상태에 더하므로 같은 모델의 재현이 아니다.
- [ExoTimer 공식 논문](https://ojs.aaai.org/index.php/AAAI/article/view/39908/43869)의 학습 가능한 모달리티별 hash 경계·연속 표현 결합을 확인했다. 본문에 명시되지 않은 대조 손실이나 예측 압축 보장을 보충하지 않는다. 이번 future-regime alignment는 두 연속 표현을 목적에 맞게 연결하는 관점에서 착안한 변형이며 Bi-Hash 재현이 아니다.
- [CPC 원저](https://arxiv.org/abs/1807.03748)는 미래 표현에 관한 학습 목적의 선행 사례다. 고정 미래 요약에 맞추는 MSE는 변환된 정답에 대한 보조 회귀이며 새로운 원리로 주장하지 않는다.

시작 HEAD는 `95585946f99da9b50cbdadcadd7441ccbf6ff2e5`다. 기존 source/result/cache/실행 중 작업을 덮거나 종료하지 않는다. 이번 출력은 새 연구 폴더와 같은 이름의 `.cache`에만 쓴다. 보호 TEST 접근·대형 백본 재학습·이전 중단 캠페인 재개·자동 commit/push는 하지 않는다.

## 공통 비교 계약

기존 개발 자료는 Jena C21·Robin C17, FIT256/TUNE64 및 이미 반복 사용한 개발 VAL이다. 기존 FIT/TUNE bounds·통계·origin을 정확히 재사용한다. L512/H48, quarter K=floor(C/4), 두 seed92601/92602, batch4,600 updates, AdamW lr0.001/decay0, gradient clipping1, 체크포인트0/100/200/400/600을 사용한다. 각 비교는 초기 공유 파라미터와 배치 index 순서를 같게 맞춘다.

주지표는 사용자가 고른 FIT-only 정규화 observed channel-macro MSE다. checkpoint·lambda 선택은 미래 TUNE 주지표만 사용한다. MAE 및 원단위 MSE/MAE·채널·시간구간도 보고한다. 기존 VAL은 개발 진단이며 새 독립 평가의 증거로 바꾸어 부르지 않는다.

동결 Chronos-Bolt-small은 실제 forward/input gradient를 사용하되 백본 weight를 갱신하지 않는다. 모든 새 군에서 동일한 zero-variance backward convention을 적용하고 원 forward/양수분산 gradient/상수 fixture를 검증한다. 이 convention은 var=0에서 유일한 수학적 미분을 복구한다는 주장이 아니다.

## S0 — 원인 분리와 구현 검사

1. 현재 저장 teacher checkpoint·source·데이터 SHA와 장비/실행 작업을 확인한다.
2. 각 고정 linear D의 열공간에 실제 미래 Y 또는 변화량 Y−x_last를 관측가중 최소제곱으로 투영한다. 원점별 실제 미래를 쓴 oracle 진단은 배포 모델·선택·정확도 순위에 넣지 않는다. 오차를 보편적 lower bound라고 부르지 않고 고정 D와 해당 자료의 projection residual로 기록한다.
3. C=identity 초기 상태에서 anchor 경로와 RAW Bolt 실제 출력 parity, causal E prefix/미래 경계 perturbation, 모든 경로의 gradient/update와 백본 hash, missing-window reset을 확인한다.
4. 기존 direct PCA를 새 동일 조건에서 다시 실행해 재사용 결과와 허용 오차 내에서 일치하는지 확인한다.

## S1 — 가장 직접적인 작은 K 개선

배포 anchor는 정규화 공간에서 다음과 같다.

\[
z=E(X),\qquad \widehat Y=x_{last}+D(F(z))-D(z_{last}).
\]

원자료 현재값 C개가 bypass되므로 strict K-only compression으로 부르지 않는다. TSFM 입력은 K채널이고 원자료 상태 보존과 변화 예측을 결합한 구조다. affine D의 bias는 취소되며 active/registered parameter를 구분한다.

다음6개 조건을 두 자료×두 seed에서600 updates씩 수행한다.

```text
DIRECT_PCA          기존 causal E + PCA 초기 linear D, 전체 미래 직접 복원
ANCHOR_PCA          같은 E/D + 원채널 현재값 보존, 변화량 예측
DIRECT_TEACHER      FIT teacher 미래 변화 부분공간으로 E/D 초기화, 직접 복원
ANCHOR_TEACHER      같은 teacher 부분공간 + anchor
ANCHOR_POINTWISE    teacher 부분공간 + pointwise linear E + anchor
ANCHOR_NONLINEAR_D  teacher 부분공간 + causal E + linear D/zero-init MLP residual D + anchor
```

teacher는 기존 전체 C 항등 초기화 PRED의 TUNE 선택 checkpoint다. Jena는600/400, Robin은0/0이므로 Robin을 더 좋은 학습 teacher라고 부르지 않는다. teacher 예측은 과거 입력만으로 만들고 FIT origin output−x_last의 covariance eigenbasis를 K개 선택한다. basis/평균은 FIT에서만 적합한다. teacher 선정에 기존 TUNE가 쓰인 비용·개발 노출을 기록한다.

nonlinear D는 K→32→C residual을 linear D에 더하고 최종 residual weight/bias를0으로 초기화해 최초 forward를 linear D와 맞춘다. 용량이 커진 효과를 숨기지 않는다. pointwise E는 RF1이며 causal E와 파라미터 개수가 다른 비교다. u0 고정 PCA/anchor 성능과 학습 이후의 추가 효과를 분리한다.

S1 architecture는 자료별 두 seed의 TUNE MSE/RAW Bolt TUNE MSE 비율의 log 평균을 사용해 전 자료에서 하나로 선택한다. VAL을 사용하지 않는다. 1e-12 이내 tie는 위 조건 순서대로 단순 조건을 우선한다. 전체 24 fits를 보고하고 선택한 조건만 좋은 결과로 제시하지 않는다.

## S2 — ExoTimer에서 착안한 미래 표현 정렬

S1에서 고른 배포 구조를 새 동일 초기화에서 다시 학습한다. 앞 블록에서 개선이 없어도 수행한다. FIT의 실제 미래 Y는 보조 정답에만 쓰고 추론에는 미래 branch·보조 head·prototype을 제거한다.

고정 teacher basis B에 대해 teacher 미래 변화 좌표는 완전 관측에서는 `(Y−x_last)B`, student 좌표는 실제 F 미래 출력−z_last다. 일부 채널이 결측이면 관측된 B 행으로 변화량을 최소제곱 적합한 K좌표를 사용한다. 관측 채널 수가 K 이상이고 해당 B 행의 rank가 K인 horizon만 유효하다. 완전 관측·정규직교 B에서는 위 projection과 같음을 fixture로 확인한다. 공통 보조 head q는 Linear(K,K), identity/zero-bias 초기화다. mean·고정 마지막 horizon·관측가중 LS slope의3K 요약을 FIT 통계로 표준화한다. summary MSE/partition의 보조 손실은 전체3K 요약이 유효한 origin만 사용해 shuffled군과 유효 수를 맞춘다. LATENT_FUTURE_MSE는 유효 horizon 전체를 쓴다. 미래 결측값을 forward-fill해 정답으로 사용하지 않는다. 관측이 부족하거나 rank가 모자란 horizon은 보조 손실에서 제외하고 수를 보고한다.

```text
BASE_Q              main forecast MSE, 같은 q 등록, 보조 weight0
LATENT_FUTURE_MSE   q의 H×K 미래 변화와 고정 실제 미래 좌표의 보조 MSE
SUMMARY_MSE         동일 q의3K 요약과 실제 미래 요약의 보조 MSE
SHUFFLED_SUMMARY    같은 미래 요약과 head를 유지하고 non-overlap pair로 정답 shuffle
FIXED_REGIME        고정 random prototype의 미래 soft partition KL
LEARNED_REGIME      FIT 미래 요약에서 학습 후 고정한 prototype의 soft partition KL
SHUFFLED_REGIME     학습 prototype 유지, 실제 미래가 겹치지 않는 FIT pair로 label shuffle
```

보조 weight 후보는0.03/0.1, BASE_Q는0이다. partition P=8, KMeans n_init10/random_state=seed, softmax distance temperature1을 고정한다. 동일 q·E/D·init/order·updates/F 호출 수를 사용하며 prototype fitting과 보조 head 비용을 별도 기록한다. learned partition과 fixed partition은 같은 prototype 수와 추론 구조다. 미래 target overlap이 없는 shuffle(|origin_i−origin_j|≥48)은 전체3K 요약이 유효한 FIT origins subset 안에서 고정 permutation으로 만들고 bijection·거리 조건을 검증한다. invalid origin은 두 비교군에서 모두 제외해 매 배치 유효 수를 같게 유지한다. 유효 subset에서 가능한 permutation이 없으면 해당 shuffle 경로만 미실행 이유를 기록한다.

Gaussian 고정 중심과 KMeans 중심의 norm·거리·teacher entropy·보조 gradient scale은 같다고 보장되지 않는다. 해당 원수치를 기록하며 두 군 차이를 학습된 경계 하나의 순수 인과효과라고 주장하지 않는다. 결측 채널 LS의 condition number와 teacher latent 극값/분위수를 보고해 rank 통과를 안정성 보장으로 바꾸어 말하지 않는다.

teacher가 함께 상수로 학습되는 collapse를 막기 위해 basis·FIT 표준화·prototype을 고정한다. prototype 점유율·entropy·student variance/effective rank·constant-distribution baseline과 본 예측 오차를 보고한다. 보조 loss 감소만으로 미래 예측 개선을 선언하지 않는다. BASE_Q의 q가 active gradient를 받지 않는 차이도 명시한다.

S2는 BASE_Q4 fits와 나머지6방법×2 weights×4 자료/seed=48 fits, 총52 fits를 계획한다. 방법별 weight는 같은 TUNE ratio log 평균으로 한 개를 선택한다. 데이터별/seed별 VAL로 유리한 weight를 바꾸지 않는다. 같은 계열 추가 weight·prototype 탐색은 하지 않는다.

자료·구현·수치 문제는 해당 case의 실패 기록을 보존하고 다른 독립 case를 계속한다. 각 후보는 두 자료×두 seed가 모두 성공한 weight만 선택에 사용하며 coverage와 제외 이유를 남긴다. S1에서 완전 후보가 하나도 없으면 S2 architecture 선택은 미완료다. S2의 어떤 방법에 완전 weight 후보가 없으면 해당 방법의 개발 선택은 미완료로 표시하고, 새 자료에서는 사전 기본값0.03으로 독립 시도한다. 이 기본값을 개발에서 선택한 weight라고 쓰지 않는다. prototype과 shuffle 요구 조건은 사용하는 보조 방법에만 적용하며 BASE_Q나 latent 회귀를 함께 막지 않는다. 필수 case 누락이 있으면 전체 상태는 PARTIAL로 남긴다.

## S3 — 개발 선택을 봉인하고 새 두 자료에서 검증

기존 ETTm2가 사용된 기록이 있어 ETT family를 독립 새 자료의 중심 근거로 삼지 않는다. repo 검색에서 직접 사용 흔적이 없는 두 공식 UCI 자료를 사전 선택한다.

- [Appliances Energy Prediction](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction): date 제외 수치 채널, 공식 random variables rv1/rv2는 사전 제외. 약4.5개월의10분 간격 자료다.
- [Beijing Multi-Site Air Quality](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data): 사전 고정 Aotizhongxin 관측소, PM2.5/PM10/SO2/NO2/CO/O3/TEMP/PRES/DEWP/RAIN/WSPM의11채널. 시간당2013-03-01~2017-02-28, 결측NA. wd와날짜/행번호/station은 제외한다.

두 자료는 CC BY4.0으로 출처·원본 SHA를 기록한다. 새 자료의 원본 다운로드·값 materialization은 S1/S2의 METHOD_LOCK이 기록된 뒤 수행한다. header/timestamp/schema와 원자료 품질만 검사하고 미래 EVAL 성능을 미리 읽지 않는다.

시간순 FIT60%/TUNE20%/EVAL20%, L512/H48을 동일하게 사용한다. FIT256/TUNE64는 구간 내 균등선정, EVAL은48 stride의 모든 유효 origin으로 정답 horizon이 중첩되지 않게 한다. FIT-only mean/std, 입력은 각 실제512창 시작을 FIT mean으로 reset해 causal FF, 미래 관측 마스크로 loss/metric을 계산한다. 시간 단위가 다르므로512/48이 같은 실제 시간이라고 쓰지 않는다.

방법·architecture·loss·weight·K 규칙·선택 규칙·보고 비교를 METHOD_LOCK으로 봉인한 뒤, 새 자료에서는 FIT/TUNE만 사용해 같은 규칙의 모델을 새로 적합한다. 기존 가중치의 zero-shot transfer가 아니라 개발 선택과 분리한 자료별 재학습 검증이다. foundation 사전학습 자료와 겹치지 않는다는 보장은 없으며 이 불확실성을 기록한다.

새 자료 teacher는 직접 전체 C 항등 PRED를600 updates 학습하고 TUNE로 선택해 고정한다. EVAL 선택 없이 개발에서 봉인한 S1 기본 대조·선택 architecture와 S2 seven 방법의 선택 weight를 두 seed에서 수행한다. 각 학습은600 updates다. 새 자료에 맞는 K는floor(C/4)로 고정하고 최소1이다. teacher subspace는 각 새 자료의 FIT teacher outputs로만 적합한다.

새 자료별·seed별 정확한 fit 목록은 전체 C teacher 1개, S1 DIRECT_PCA/ANCHOR_PCA와 선택 architecture(같은 이름이면 중복 fit 없음), S2 BASE_Q 및6개 보조 방법 각1개다. S2 weight는 개발 자료에서 봉인한 한 개만 적용해 새 EVAL로 재선택하지 않는다. 자료·seed당10~11개 fit이며 두 자료·두 seed로40~44개다. native/R 비교에는 S1 선택 baseline과 S2 LEARNED_REGIME의 두 고정 조건만 사용한다. EVAL에서 가장 좋아 보이는 다른 정렬 방법으로 바꾸지 않는다.

새자료 materialization은 FIT_TUNE payload, EVAL 과거 입력창 payload, EVAL 미래 target/mask payload를 별도 파일로 분리한다. 모델 적합 loader는 FIT/TUNE만 반환한다. EVAL target/mask를 여는 API는 유효한 MODEL_SELECTION_LOCK의 model/source/계약 SHA를 확인해야 동작한다. ingest에서 원 CSV를 split 저장하는 행위와 학습 중 EVAL target 접근·평가를 구분하며, ingest가 raw EVAL 값도 읽었다는 사실을 숨기지 않는다. EVAL input은 원점 이전 관측만 포함하는 rolling past다. EVAL 정답은 최종 평가 pass에서 한 번 load해 모든 고정 모델에 공유하며 그 뒤 결과를 보고 추가 학습하지 않는다.

모든 모델·checkpoint를 TUNE로 선택해 MODEL_SELECTION_LOCK을 쓰고 나서 EVAL target store를 최초 개봉해 한 번 평가한다. 이후 EVAL 결과로 추가 학습·weight·architecture·station 선택을 바꾸지 않는다. independent EVAL은 기존 보호 TEST를 이름만 바꾼 자료가 아니다.

## 예측기·비용·검산·보고

각 자료의 RAW Bolt, raw SELF33 및 raw joint ridge를 강한 기준군으로 실행한다. 선택된 기본·정렬 E/D에는 같은 anchor/direct 규칙의 native shared linear를 추가600 updates로 학습해 비교한다. native 전용으로 E/D를 최적화한 비교는 아니며 추가 학습비를 기록한다. 원자료 보정 단독과 같은 R 결합은 선택된 기본·정렬 조건에서만 고정 특징/동일 FIT/TUNE lambda 규칙으로 별도 비용과 함께 보고한다.

모든 Python GPU 학습이 끝난 뒤 batch1/8·warmup2·repeat7·synchronize로 CPU window FF/정규화/H2D/E/F/D/anchor·역정규화/D2H의 전체 배포 경로를 측정한다. 모델 로드·disk read는 제외하고 로드시간은 별도다. 추론 시 제거한 alignment head/teacher를 배포 비용에 넣지 않되 학습 비용에는 포함한다. 공존 메모리를 방법별 exclusive 절감이라고 주장하지 않는다.

독립 검산은 raw→FIT 통계→window/mask/origin→저장 예측→점수→TUNE 선택→새 EVAL lock/timestamp→source/모델 hash를 추적한다. 한국어 보고서·원수치CSV·기전/정확도/MAE/비용/시간·채널·collapse 그림을 생성하고 실제 열어 본다. 모든 음성·실패·미완료 경로의 이유와 재개 조건을 남긴다.

## 자원·중단·완료 조건

새 작업의 보수적 상한은 GPU 사용 process wall2시간, CPU-only검산/보고 wall2시간, 전체wall8시간, 새 저장12GiB, process RAM/allocatedVRAM8GiB다. 이전 resource ledger는 수정하지 않는다. 초기 profiler의 실제 속도로 ETA를 갱신한다. 실행 중 resource JSON을 읽지 않고 append-only stdout을 모니터링한다. 상한·누출·자료/구현 문제가 난 경로는 보존하며 독립 가능한 다른 블록은 계속한다.

완료는 S0 실제 진단/검사, S1 신규24 fits, S2 신규52 fits 및 모든 대조, METHOD_LOCK 이후 새자료2개 FIT/TUNE 학습과 최초EVAL, native/raw/cost비교, 독립 검산 및 한국어 보고·그림으로 증명한다. 계획이나 smoke만으로 완료하지 않는다. 작은 K의 정확도가 좋아지지 않아도 실패와 남은 판단을 보고하면 실험 목적을 달성할 수 있지만, 필수 실행 누락을 완료로 바꾸어 말하지 않는다.
