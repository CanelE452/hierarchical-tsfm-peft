# V6: 수준 보존 잔차 PEFT의 Robin 고정 확인

[확인] 예정된 정확도·비용 비교를 완료했다. 아래 수치는 저장 결과에서 확인한 측정값이며, 실행팀의 최종 검산은 [final_checks.json](final_checks.json), 게시 상태는 [STATUS.md](STATUS.md)와 publication receipt로 구분한다.

## 무엇을 새로 확인했나

**수준 보존 잔차 PEFT를 후속 논문의 제한된 방법 기여로 유지할 근거가 강화됐다.** Robin 전체 MSE는 LEVEL_RES 0.275196, 기존 OLD_RES 0.538776이었다. 상대 변화는 -48.92%이며, 대응 LEVEL_RAW보다20.61%, LEVEL_ONLY보다21.16% 낮았다. 두 seed와 두 기간의 방향이 일치한다. 과거 Hog에서 선택한 수정을 새 집단에서 고정한 뒤 확인한 결과이며, Hog를 더 튜닝한 결과가 아니다.

미압축 F0 0.229095, LoRA 0.223610은 여전히 더 정확하다. LEVEL의 MSE 손해는 각각20.12%,23.07%이다. MAE도 같은 손해 방향이다. 따라서 최고 정확도나 미압축 대체의 보편적 우수성을 주장하지 않는다. 논문 제출 준비나 채택이 완료됐다는 판정도 아니다.

## 집단·정보·선택

BDG2 raw electricity의 Robin Office17채널, K5를 사용했다. 원문 metadata상 Education/College-University 시설이며, 제공된 Europe/London 지역의 local-naive 시간 인덱스를 그대로 사용했다. 선택은 site ID 순서·기존 노출 기록·TRAIN 관측률>=.95·비상수·16채널 이상 규칙에 따른다. 모델 점수로 집단을 고르지 않았다. 실제 채널 목록, source revision/파일 hash, 제외 집단과 조회 범위는 [data_contract.json](data_contract.json), [exposure_search.json](exposure_search.json), [PLAN.md](PLAN.md)에 있다.

조회한 로컬 기록 범위에서는 Robin의 직접 방법 선택 사용 흔적이 없었다. 이는 모든 대화/전체 과거 이력 또는 Chronos 사전학습 자료와의 비중복 증명이 아니다. 기존 Hog/Bull/Electricity는 노출 개발 자료다. 이번 한 site의 선택 후 확인을 독립 외부 재현이나 다른 도메인 전체의 확증이라고 부르지 않는다.

TRAIN [2016-01-23,2016-04-15), VAL [2016-04-17,2016-05-18), TEST-A [2016-05-20,2016-06-30), TEST-B [2016-06-30,2016-08-10). 과거512/미래48 관측, TRAIN1945·VAL30·A40·B40원점이다. 목표48관측은 분할 안에 완전히 포함되며 A/B 경계 목표를 제외했다. 이후 TEST 원점에서 당시 관측 가능한 과거만 입력으로 사용했고 온라인 업데이트는 없다.

표준화·PCA·입력 결측 평균 대체는 TRAIN에서만 적합했다. 0은 관측값이며 원래 유한 target mask로 채점했다. 10개 신경망은 seed92601/92602의 각 최소 VAL checkpoint(step0 포함, 동률 먼저)를 사용한다. 모든 실행이120epoch 전 조기 종료해 연장 자격은 없었다. 전체 TRAIN SHARED ridge는 사전 고정한 두 penalty0.001/0.1 중 VAL로0.001을 선택했다. [extension_decision01.json](extension_decision01.json), [selected.json](selected.json), [shared_ridge_selection.json](shared_ridge_selection.json)을 보존했다.

14개 모델·주 지표·8개 비교를 [evaluation_seal.json](evaluation_seal.json)에 고정하고 [test_exposure.json](test_exposure.json)에 최초 개봉을 기록했다. TEST 뒤 적합·집단/채널/기간/지표·선택 변경은 하지 않았다.

## 고정 방법과 기여 경계

X는 과거512×17, U는 새 TRAIN PCA17×5, E(X)=XU, D(Z)=ZUᵀ, r=X-D(E(X)), b=D(F0(E(X))). P_n은 마지막 과거값을 detach하고 n시점 반복한다.

```text
LEVEL_RES        b + P_H(r) + G(r - P_L(r))
LEVEL_RAW        b + P_H(r) + G_raw(X - P_L(X))
OLD_RES          b + G_old(r)
LEVEL_ONLY       b + P_H(r)
COMPRESS         b
LEVEL_LINEAR_RES D(T_tilde(E(X))) + P_H(r) + G(r - P_L(r))
```

E/D는 고정 PCA, F0는 고정 revision의 Chronos-Bolt-small이다. G는 공유·무편향·무활성512→32→48, 학습 파라미터17,920개, 출력층0 초기화다. LEVEL과 matched RAW는 같은 잔차 수준 복원항·초기 E/D/G tensor·동일 seed/epoch 표본 순서를 쓴다. 초기 함수는 둘 다 LEVEL_ONLY이며 OLD의 초기 COMPRESS와 다르다. 작은 선형 T는 문맥 정규화/역변환을 포함한 공유 affine512→48, G와 독립이며 배포 시 F0를 보관하지 않는다. LoRA는 미압축 r8/q,v/alpha16, native quantile loss다. 다른 신경망은 원채널 masked macro MSE라 학습 목적 차이를 실용 비교의 제한으로 남긴다.

잠재 E/D와 동결 단변량 TSFM 적응은 AdaPTS, 마지막 값 제거·복원은 NLinear의 기존 원리다. 새 normalization/PCA/LoRA 수학을 주장하지 않는다. 기여는 **제한된 잠재 채널에서 빠진 재구성 잔차에 수준 보존을 적용한 구체적 설계, 조건부 구조 성질, 가까운 대조와 정확도–비용 확인**이다. COMPRESS는 AdaPTS 전체 확률모형/최적 튜닝의 재현이 아니다. [선행과 읽기 깊이](CLAIMS_PRIOR_ART.md), [영문 원고](PAPER_DRAFT.md), [서지](references.bib)를 연결한다.

고정 직교 PCA와 공유 선형 G 조건에서 보정은 버린 채널 부분공간 안에 머물며, 그 공간의 상수 offset을 보존한다. LEVEL의 유효 시간 rank 상한은33, OLD의 G는32라 같은 학습 파라미터 수가 같은 함수 공간을 뜻하지 않는다. 초기 함수·최적화도 다르다. 이 대수식이나 별도 학습한 RAW 비교가 성능 원인의 순수 인과증명은 아니다. CURRENT E/D, 임의 비선형/채널별 G 또는 결측 mask가 있는 주 손실로 같은 성질을 일반화하지 않는다.

## 정확도: 전체·기간·seed를 함께 읽기

아래는 TRAIN 표준편차 정규화 좌표의 실제 관측 target에 대한 채널 동일 가중 손실이다. 전체는 채널별 A/B 오차합과 관측수를 먼저 합쳐 계산하며 기간 MSE의 단순 평균이 아니다. 신경망 행은 두 seed의 개별 손실 평균이며 예측 ensemble이 아니다.

```text
Method             Combined MSE  Combined MAE  TEST-A MSE  TEST-B MSE
LEVEL_RES              0.275196      0.324593    0.273677    0.276712
OLD_RES                0.538776      0.483495    0.563077    0.514472
LEVEL_RAW              0.346630      0.385883    0.358622    0.334635
LEVEL_ONLY             0.349041      0.386300    0.355840    0.342236
COMPRESS               0.977023      0.661870    1.123537    0.830506
F0                     0.229095      0.262074    0.231539    0.226649
LoRA                   0.223610      0.256347    0.230535    0.216683
LEVEL_LINEAR_RES       0.315646      0.354268    0.319826    0.311464
SHARED ridge           0.337277      0.376721    0.340548    0.334003
```

전체 LEVEL MSE의 상대 변화(각 참조를 분모로 사용)와 조건부95% 시간블록 구간:

```text
Reference          Relative change       Conditional block interval
OLD_RES                   -48.92%                   [-60.58, -31.35]%
LEVEL_RAW                 -20.61%                   [-31.58,  -9.44]%
LEVEL_ONLY                -21.16%                   [-32.06,  -9.16]%
F0                        +20.12%                   [ +9.29, +36.49]%
LoRA                      +23.07%                   [+10.09, +41.43]%
LEVEL_LINEAR_RES          -12.82%                   [-17.49,  -8.71]%
SHARED ridge              -18.41%                   [-25.63, -10.93]%
```

주 비교 LEVEL–OLD의 절대 MSE 차이는-0.263580이다. LEVEL의 두 seed 전체 MSE는0.275009/0.275383이다. LEVEL–RAW의 B기간 상대 구간 상한은 약-0.03%로 전체보다 근거가 약하다. 기간/채널 선택으로 이 불확실성을 지우지 않는다. 7원점 블록2,000회, A/B별 재표집 뒤 채널 오차합/관측수 합산이며 모든 채널과 방법을 함께 움직였다. 이 구간은 두 학습 seed를 고정한 시간 변동에 조건부이고, site/학습 seed 모집단의 일반적 신뢰구간이 아니다. p값·동등성·비열등성 판정은 만들지 않았다.

세 질문은 구분된다. LEVEL–OLD는 수준 보존 수정 전체 효과, LEVEL–RAW는 공통 수준복원 아래 입력 경로 차이, LEVEL–ONLY는 학습된 변화분의 추가 가치다. 모두 Robin에서 긍정적인 근거가 남았다. 두 선형 대조보다 낮은 오차는 포함한 대안 대비 TSFM 경로의 예측 가치를 지지하지만, 모든 선형모델의 상한이나 사전학습 지식만의 효과는 아니다. 반대로 F0/LoRA와의 남은 오차 손해는 두 기간·seed와 MAE에도 남는 실제 반례다.

모든 개별 seed·기간·채널 수치는 [evaluation01.json](evaluation01.json)에 있다. 평균으로 다른 자료를 합친 종합 점수는 사용하지 않았다.

## 비용: 처리량 우위는 확보되지 않았고 메모리 절충은 남았다

같은 RTX4070·FP32·TF32 off·CPU4thread·공통 VAL24원점에서 표준화 CPU입력→GPU→전체 원채널 CPU출력을 측정했다. 배치1 지연과 배치4 처리량은 다른 조건이다. 각 fit의 세 블록별 중앙값에서 다시 중앙값을 구하고, seed가 있는 모델은 그 fit 중심값을 두 seed에 걸쳐 평균한다. 결정론적 모델은 한 fit 중심값을 사용한다. 메모리는 모든 블록·seed의 최대 peak이며 원본에는 최솟값과 각 블록도 남았다.

```text
GPU method         batch4 origins/s  batch1 ms  allocated MiB  reserved MiB
LEVEL_RES                   162.32      23.68         214.49        216.00
OLD_RES                     163.55      23.66         214.49        216.00
LEVEL_RAW                   164.81      23.66         214.49        216.00
LEVEL_ONLY                  164.50      23.57         214.42        216.00
F0                          162.30      23.54         265.49        292.00
Merged LoRA                 161.39      23.59         265.49        294.00
LEVEL_LINEAR_RES           2124.94       1.39           8.89         22.00
```

LEVEL의 LoRA 대비 batch4 처리량 중심값 차이+0.58%는 안정적 우위로 해석하지 않는다. 가까운 모델·seed 쌍의 블록 순위 뒤집힘이65건이며 F0 전후6쌍의 IQR은 모두 겹쳤다. F0 전후 drift가 작다는 관찰도 모든 측정이 안정적이라는 증명은 아니다. 이전 Hog의 약2배 처리량은 이번 Robin에서 재확인되지 않았다. 과거와 이번 timing을 직접 나눈 개선율은 쓰지 않는다.

LEVEL의 batch4 peak allocated는224,908,288bytes, F0/LoRA는278,385,152bytes로19.21% 감소했다. 이는 실행 시 메모리 관찰이며 백본 파일 크기를 줄였다는 뜻이 아니다. LEVEL의 배포 tensor는190,944,460bytes, F0/병합 LoRA는190,872,100bytes다. 학습 파라미터17,920 대294,912와 전체 배포 파라미터47,736,106 대47,718,016을 구분한다. CURRENT 모델이나 C/K 비율의 속도 보장은 주장하지 않는다. 프로세스 RSS/peak working set은 runtime·자료까지 포함한 별도 원본 항목이며 GPU allocated/reserved와 합치지 않는다.

CPU 선형 대안의 batch4 처리량은 LEVEL_LINEAR_RES4,696.46원점/초, SHARED54,675.98원점/초, batch1 지연은0.546769/0.044217ms였다. SHARED batch4 블록값은111,967.15/49,120.53/54,675.98로 변동이 크다. CPU/GPU 배포 선택의 실제 대안이지 순수 GPU 커널 대결은 아니다. SHARED는 추론 시 coefficients24,624개를 buffer로 저장하므로 nn.Parameter가0이라고 무학습 모델로 세지 않는다. CPU 선형의 작은 비용과 더 큰 예측 오차를 함께 남긴다.

허용된 한 번의 [GPU상주 진단](cost_diagnostic01.json)에서 F0/LEVEL의 batch4 full-forward wall 중심값은6.358573/6.277819ms/원점으로 여전히 근접했다. CUDA event도 stream 구간을 측정하므로 CPU dispatch 대기까지 포함할 수 있고 순수 커널 시간/FLOPs가 아니다. 별도 구간의 시간을 빼서 전송 비용을 산출하거나 Windows·clock·온도를 확정 원인으로 지목하지 않는다. 원인 완전 규명 없이도 정확도 비교는 유효하게 남는다.

원본 [GPU78행](cost_gpu01.json), [CPU18행](cost_cpu01.json), [파생 수치와 학습 시간·메모리](report_values.json)를 연결한다. LoRA는 반환된 병합 모델의 출력 정합성을 확인했다. 학습 epoch가 다른 실행시간은 통제된 학습 속도비가 아니다.

## 검산·예산·산출물

실자료12/16attempt(신경망10+CPU ridge2), 합성 optimizer1/6세션 총10update, GPU작업점유1,536.831초/14,400초로 종료했다. 모델 실패·재시도·연장0, 새 다운로드0이다. CPU검사·분석 시간과 최종 저장크기는 [원장](ledger.json)과 최종 검산의 실제 snapshot을 따른다. 예산을 채우기 위한 추가 실행은 없다.

초기 AST3건과 문서 검사3건은 사전예약을 놓쳐 사후 계상한 절차 이탈이 있다. cp949 문서 읽기 실패와 UTF8 재시도도 보존했다. 모델/실자료 fit/optimizer 실패로 섞지 않으며, 과거 v2의22/20 위반을 소급 정당화하지 않는다. 보고 숫자의 표시 검산 및 GPU/CPU 역할 guard는 측정 전에, 그림 축 잘림은 최초 그림을 본 뒤 수리했다. 학습·평가·비용 원수치는 변경하지 않았다. [보고 수리](repair_reporting01.md), [그림 수리](figure_review.json)를 남긴다.

저장 checkpoint의 실제 갱신·동결·복원, step0 포함 VAL 선택, 동일 seed/epoch 표본 순서, 원점/mask, 기간 합산·seed 평균, 8개 대응 블록 구간, 비용 집계·merge·그림과 본문 수치를 검산한다. 실행팀의 검산이며 독립 재현 또는 LLM의 동의 자체를 과학적 근거로 세지 않는다. 원시 자료·가중치·예측 배열은 local cache에만 남기고 hash/경로로 연결한다. 기존644개 추적 파일과 무관한 미추적 항목을 보존한다.

- [최종 JSON](final_comparison.json), [전체·기간·seed CSV](final_comparison.csv), [전체 채널 CSV](final_channel_metrics.csv)
- [방법 그림](figures_layout02/method.png), [기간별 정확도](figures_layout02/robin_accuracy.png), [GPU 절충](figures_layout02/robin_gpu_accuracy_cost.png), [CPU 대안](figures_layout02/robin_cpu_accuracy_cost.png)
- [영문 논문 초안](PAPER_DRAFT.md), [선행·기여 경계](CLAIMS_PRIOR_ART.md), [재현 명령](commands.json), [최종 검산](final_checks.json)

## 최종 판단과 다음 결정 하나

**수준 보존 잔차 PEFT를 논문의 제한된 방법 기여로 유지한다.** 기존 원리의 결합이라는 경계를 유지하면서도, 새 Robin 집단에서 OLD/RAW/LEVEL_ONLY와 두 선형 대조보다 나은 예측 근거가 남았다. 이를 다시 아무 근거 없는 ‘주제 미확보’로 되돌리지 않는다. 실제 배포 주장은 **메모리–정확도 절충**으로 제한하며 안정적 처리량 이득·최고 정확도·전체 사전학습 비중복·논문 신규성/채택 완료는 주장하지 않는다.

새 집단 고정 비교와 실제 결과를 담은 원고 초안은 이번 회차의 산출물이다. 논문 제출 준비 완료는 별도 상태다. 다음 결정은 **이 고정 구성을 유지한 채 다른 한 도메인에서 적용 범위를 확인할지**다. 현재의 긍정·부정 결과를 보존한 상태에서 별도 자료 계약을 먼저 정하고, Robin을 다시 튜닝하거나 이번 원고에서 그 손해를 삭제하지 않을 것을 권고한다. 이번 승인 범위에서 그 추가 실험을 자동 시작하지 않는다.
