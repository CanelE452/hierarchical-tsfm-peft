> 사용자의 명시 요청에 따른 GitHub 게시용 링크 사본입니다. 수치와 본문은 보존했습니다. 원문 SHA256 `a9d2014680348984c933e474e8580e2c91ee5e1a87f5ab2e243ae43551f52492`. 원문에 있는 게시 금지/미게시 표현은 실험 당시 기록입니다.

# 교수님 미팅용 6장 발표 스토리

실제 PPT가 아닌 검산된 원수치 기반 구성안이다. [상세 한국어 보고서](adaptive_REPORT_KO.md)를 함께 사용한다.

## 1. 문제: 작은 K가 무엇을 보존해야 하는가

TSFM은 여러 시계열로 사전학습한 예측 모델이다. 현재 Chronos-Bolt wrapper는 C개 채널을 각각 같은 고정 모델에 넣는다. 압축을 연구하려면, 단순 자기 채널 예측보다 다른 채널을 함께 볼 실용적 이득이 있는지 먼저 확인해야 한다. 문제는 복원 오차를 줄이는 것만으로 미래 예측 정보를 보존했다는 보장이 없다는 점이다.

질문은 ‘같은 과거 특징·표본·목표·선택에서 모든 C를 보면 실제 미래 C를 더 잘 맞히는가, 그 정보의 작은-K 경로가 Raw 대비 비용을 줄일 여지가 있는가’다. 이번 단계는 첫 문장부터 검사한다.

## 2. 공정 비교 계약

Robin C17·Beijing C11, L512/H48, FIT256/TUNE64, 관측 channel-macro normalized MSE를 유지했다. SELF33은 자기 채널32×16 평균+현재값, JOINT33은 같은 모든 채널 특징, SELF512는 강한 자기 채널 전체 과거 기준이다. 세 경로 모두 현재값 anchor·actual-Y innovation·FIT 특징 정규화·λ=αNc·동일9pool·글로벌 macro TUNE α 선택을 쓴다. Raw는 같은 조건의 강한 실용 기준이다.

![같은 후보와 선택](../../reports/first_and_dense_v3/figures/02_same_pool_tune_selection.png)

계수·rank·특징 차원은 다르므로 채널 정보의 순수한 인과 효과나 독립 일반화로 과장하지 않는다.

## 3. 공통 정보의 TUNE 결과와 반례

- robin: SELF33→JOINT33 TUNE MSE 0.139778→0.262253, MAE 0.219167→0.294391; 탐색 gate 미통과.
- beijing_aotizhongxin: SELF33→JOINT33 TUNE MSE 0.843454→0.953966, MAE 0.554774→0.591202; 탐색 gate 미통과.

Beijing의 노출 EVAL에는 JOINT의 MSE가 개선되고 MAE는 악화하는 반례가 있다. Robin Raw의 TUNE에는 거의0인 채널 오차가 많지만 VAL에서 달라진다. 기간별 불안정성을 남기고, DEV를 보고 TUNE 분기를 뒤집지 않는다.

![Raw 기간별 채널 오차](../../reports/first_and_dense_v3/figures/08_raw_channel_period_uncertainty.png)

## 4. 작은 FIT의 과적합 설명을 한 번 검사

Robin self/joint를 동일 FIT 구간의512 origins로 함께 다시 학습했다. 원256과 교집합은 128개이고 중첩 창이며 포함 관계가 아니므로 독립 표본 수만의 효과로 주장하지 않는다. TUNE MSE SELF33 0.140339, JOINT33 0.260887; 탐색 gate 미통과다.

![한 번의 FIT512 진단](../../reports/first_and_dense_v3/figures/07_matched_FIT512_stop_diagnostic.png)

## 5. 실제 배포 비용을 정확도와 함께 읽기

B1/B8 whole에는 실제 원단위 과거의 causal forward-fill·FIT 정규화·전송/예측·동일 CPU inverse·원단위 CPU 반환을 포함했다. 각각warmup2·순서 균형6repeat다. 모델 로드와 저장은 제외했다. 메모리는 Bolt/계수 co-resident 조건이고 단독 방법 메모리가 아니다.

![실제 전체 추론 비용](../../reports/first_and_dense_v3/figures/06_whole_deployment_cost.png)

FIT512의4조건은 GPU가 없는 별도 CPU process이며 Raw를 새로 재지 않아 first24와 직접적인 속도 우위를 주장하지 않는다. 빠르기만 하거나 한 지표만 좋아지는 것은 방법 채택의 충분조건이 아니다.

## 6. 이번 기여와 남은 질문

이번에 확인한 것은 조건이 다른 기존 self/joint 비교를 동일 특징·글로벌 α·9pool로 맞추고, 단순 기준·실제 비용·선택 불안정성·한 번의 표본 진단을 연결한 근거다. 작은-K나 비선형 표현이 불가능하다고 증명한 연구 기여가 아니다.

TUNE 선행 gate를 통과하지 못한 이 자료/계약에서는 새로운 ForeCA·예측 E/D 학습을 진행하지 않고 조건부 보류한다. 미실행 방법의 실패로 쓰지 않는다. 재구성 목표의 불일치, 학습 최적화, 비선형/지연 정보의 가능성은 남았다. 적합한 사전 선정 자료와 강한 self 대조·독립 평가 설계가 다음 판단 조건이다. 기존 AdaPTS와의 직접 중복도 [선행연구 검토](../../LITERATURE_REVIEW_KO.md)에 남긴다.
