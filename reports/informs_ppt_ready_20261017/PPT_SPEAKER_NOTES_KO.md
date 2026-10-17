# 발표자 메모 — 결과가 답하는 질문의 범위

이 메모는 PPT 파일이나 실험 완료 선언이 아니다. 실제 표·그림은 [결과 지도](PPT_RESULT_MAP_KO.md), [주장 행렬](PPT_CLAIM_EVIDENCE_MATRIX.csv), [미완료 셀](PPT_PENDING_CELLS.csv)과 최종 패키지 검산을 함께 본다. 원 source의 전체 SHA는 결과 지도 S01–S17에 기록했다. Priority 1의 모델 재학습·새 추론은0이며 전체 MASTER는 paused·미완료다.

## 1. TSFM 배경에서 문제로 연결

“여기서 TSFM은 사전학습된 시계열 예측기입니다. Chronos-Bolt를 고정해 쓰면서, 여러 채널의 과거 정보를 어떤 입력 시계열로 구성해야 할지 묻습니다. 단순히 모델 파라미터를 적게 학습하는 것과, 원 C개 채널을 K개 잠재 시계열로 보내는 것은 다른 선택입니다.”

그다음 B2 전체 그림을 먼저 제시한다. P는 과거의 시간·채널 처리, E는 C→K, F는 동결 Bolt, D는 lag-aware 복원이며 마지막 원 C 현재값 anchor가 별도 경로로 남는다. P/E/D는 이미 미래 예측 MSE로 학습했다. 따라서 모든 미해결을 재구성 loss 탓이라고 설명하지 않는다. F 가중치가 고정돼도 앞단을 학습하려면 F 입력에 대한 역전파가 필요하므로 학습비용이 없다는 뜻도 아니다. 모든 원 정보가 K만 통과하거나 전체 메모리가 C/K로 줄었다고 말하지 않는다. 근거 S01–S03.

## 2. 먼저 Jena에서 관찰된 효과를 설명

“동일한 Jena C21/K6, 두 seed에서 B2의 MSE가 B0보다 낮았습니다. 다만 같은 개발 V를 반복 사용했고 B0와 B2는 모수 수와 선택 refit 길이도 다릅니다. 순수 시간혼합 효과를 분리한 결과는 아닙니다.”

MSE와 MAE를 함께 가리킨다. seed92602의 MAE 이득은 작다는 점을 생략하지 않는다. B0가 시간정보를 전혀 안 쓰는 모델이라고 설명하면 안 된다. B0도 F와 lag D로 전체 과거 창을 이용한다. 이 비교에서 새로 더한 것은 P라는 앞단 구조이며 실제 선택 학습량까지 포함한 관측이다. 근거 S04–S06.

## 3. Bolt 대조는 두 종류를 구분

“Jena R5a는 같은 최종 표현·디코더를 고정하고 전체 512K history에 선형 head를 학습했습니다. R5b는 같은 형태의 P/E/head/D를 no-Bolt 목적에 맞춰 새로 학습한 모델입니다. 두 선형 대조보다 B2의 MSE가 낮았습니다.”

R5a와 R5b를 같은 가중치 비교라고 묶지 않는다. PeMS C170에는 부모의 전체 CPU V148과 고정 head 비교가 있지만 joint no-Bolt는 없다. 부모는 제한1epoch, head는 추가3epoch다. 이 결과를 새 C32 matched 실험이나 PeMS joint 결과로 부르면 안 된다. “Bolt가 반드시 필요하다” 대신 “현재 이 선형 대조 안에서 조건부 근거가 있다”고 말한다. 근거 S03–S06.

## 4. 강한 기준선과 비용을 바로 이어 제시

“Jena에서는 Ridge·Direct G·선택한 새 LoRA가 B2보다 MSE가 낮습니다. B2는 LoRA보다 warm 추론이 빠르지만 정확도 손실이 있습니다. no-Bolt 선형 head는 더 저렴합니다. 따라서 정확도와 실제 비용의 교환관계를 보고 있습니다.”

LoRA는 `postT_cpu_jena_E1_LORA_R8_k0_s92601_family_v1`의 CPU search 최대3/선택 refit1 결과 하나로 본문을 통일한다. 옛 .25 CUDA LoRA와 섞지 않는다. 일부 Raw/PCA/PeMS baseline의 full-V 장치가 CUDA임을 표 각주에서 보인다. CPU6 검산이 full-V 장치를 바꾸지 않는다. Ridge는 닫힌해 예측기여서 optimizer 수렴 미확정이라고 말하지 않는다.

비용은 physical 입력부터 inverse·CPU 반환까지의 warm 한 batch 전체 시간이다. B8을8로 나눈 값이 아니다. warm10/timed20/별도 stage20이며 cold load·LoRA merge와 학습비용은 별도다. 과거 기록과 새 reference 비용의 quiet/RSS/측정시점 차이는 남긴다. PeMS C170 B0/G/LoRA 또는 C32 B0/B2/joint의 없는 CPU 비용을0·GPU 값·추정치로 채우지 않는다. Jena Raw는 parity 실패이므로 FAILED다. 근거 S03–S12.

## 5. 고정 예시를 보여줄 때

“예측오차를 보고 고른 그림이 아닙니다. 공통 V origin의 첫·중앙·끝과 채널 순서의 첫·중앙·끝을 미리 고정했습니다. 모든 H와 같은 관측 mask를 그렸고 결측은 비워 두었습니다.”

원단위가 다른 Jena 채널은 패널별로 읽는다. 그림의 cohort·origin·channel·단위·실제 포함 모델을 먼저 짚는다. PeMS 그림은 C170 기존 saved forecast이므로 C32 최종 성능 근거가 아니다. 고정 head와 joint head를 범례에서 명확히 나눈다. 사전 고정한 강한 기준선의 종류도 성능으로 골랐다고 설명하지 않는다. 그림은 오차 모양의 사례이지 평균 성능·통계적 유의성의 증명이 아니다. 실제 figure/검산이 없는 상태에서는 생성됐다고 말하지 않는다.

## 6. 새 PeMS matched 빈칸을 설명

“Raw/G C32는 완료됐지만 같은 C32/K8의 B0/B2는 search 중 예산으로 종료됐습니다. B0 847, B2 846 update와 저장점은 남아 있으나 fresh refit·final V가 없습니다. 따라서 새 matched 비교의 MSE/MAE는 NA입니다.”

완료 endpoint는0/.25/1이고 event의 목표3은3epoch 완료가 아니다. partial search의 F 입력 gradient 관측3386/3382는 구현 근거이며 최종 정확도 검증이 아니다. A_stop의 provisional 선택값을 Raw/G V와 비교하지 않는다. 독립 감사의 원 status는 rejected, 새 정확도·비용 행0이고 후처리8개는 미실행이다.

새 지시문 §7(5)는 학습 불충분으로 판단할 수 없으면 닫고 그대로 보고하도록 한다. 이 상태는 B2가 B0보다 나쁘다는 판정이 아니며 P1-2의 긍정 gate도 충족하지 않는다. 원 실행 마감이 끝났고 호환되는 C32 resume CLI는 검증되지 않았다. 실제 CP 존재와 원 optimizer/RNG 복원 성공을 구분한다. 새로운 자원·구조·학습 길이를 임의로 추가하지 않는다. 근거 S13–S14.

## 7. 질문에 답하고 끝내기

“현재는 예측 목적의 C→K 입력 적응을 동결 TSFM에 실제 연결한 후보를 설명할 수 있습니다. Jena에서의 조건부 이득과 일부 선형 no-Bolt 대조의 이득은 있지만, 모든 강한 기준선을 넘는 보편적 정확도–비용 우위는 없습니다. 새 PeMS matched 검증은 미완료이므로 Priority 2로 넘어가지 않습니다.”

Beijing 반례와 variable-K는 appendix에서 질문을 받으면 보여준다. 여러 K 출력6개를 독립 학습6개라고 세거나 B0 계열 보조 실험을 주 B2 방법으로 바꾸지 않는다. 과거 전용 입력 경계 검사 통과가 upstream/legacy 전체 무누출을 증명하지 않으며 단일 분할·두 seed·반복 개발 V·수렴 미확정을 마지막까지 유지한다. 근거 S15–S17.

## 본문에 쓰는 exact run ID

```text
Jena B0 seed92601   postT_cpu_jena_E3_B0_k6_s92601_v1
Jena B2 seed92601   postT_cpu_jena_E3_B2_k6_s92601_v1
Jena B0 seed92602   priority_seed2_jena_B0_k6_s92602_v1
Jena B2 seed92602   priority_seed2_jena_B2_k6_s92602_v1
Jena fixed no-Bolt  priority_jena_R5a_fixedB2_linear_k6_s92601_v1
Jena joint no-Bolt  priority_jena_R5b_joint_linear_k6_s92601_v1
Jena selected LoRA  postT_cpu_jena_E1_LORA_R8_k0_s92601_family_v1
PeMS C170 CPU B2    diagnostic_pems08_B2_parent_full_CPU_V148_k43_s92601_v1
PeMS C170 fixed    priority_pems08_fixedB2_linear_k43_s92601_v1
PeMS C32 partialB0 postT_cpu_pems08_E1_SUBSET_B0_k8_s92601_C32_priority_v1
PeMS C32 partialB2 postT_cpu_pems08_E1_SUBSET_B2_k8_s92601_C32_priority_v1
```
