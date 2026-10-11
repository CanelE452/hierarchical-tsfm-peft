> 게시용 사본입니다. 실험 완료 당시 원본 문서의 SHA256은 `b60e0708550699ac58a1be61f0d539c5cb7520b026e3ad38f5f480ce464cb646`입니다. 본문 수치·판단을 유지하고 링크만 GitHub 경로로 조정했습니다. 원문의 “이번 작업에서 push하지 않음”은 과학 실험 완료 당시 기록이며, 후속 사용자 승인으로 이 공개 사본을 게시합니다. 구 TEST 사고는 [범위·원본 SHA](PROTECTED_ACCESS_CONTEXT.json)만 공개하며 보호된 통계는 재게시하지 않습니다.

# 교수님 제안의 표현학습·복원 경로 직접 검증

현재 목표는 GTConvAE와 TS2Vec를 정상 구조로 학습하고, 학습한 표현을 동결 TSFM으로 실제 미래 예측한 뒤 대응 복원기로 원채널 미래를 복원할 수 있는지 판단하는 것이다. 이전 LEVEL 중심 A–H 요청은 대체됐다. LEVEL, 잔차 G, 마지막 잔차 carry, K5/K6, 시간 길이 보존, 공통 선형 D8, 잠재 자기예측 loss는 신규 방법의 조건이 아니다.

이전에도 E0·No-F·native TS2 대조가 있어 모든 경로에 LEVEL G가 붙었던 것은 아니다. 그러나 이전 GT 적용은 시간 길이를 보존하고 공통 선형 readout을 사용했으며 TS2도 작은 출력 차원에 맞췄다. 그 결과는 교수님이 기대한 정상 규모의 표현과 native reconstruction decoder를 직접 검증한 결과로 사용할 수 없다.

사용자가 대기 supervisor 중지를 승인한 뒤, 이전 학습 자체도 즉시 중지하라고 승인했다. [전환 기록](../QUEUE_TRANSITION.json)과 [중지 기록](../OLD_FIT_CANCELLED.json)을 보존했다. 완료한 과거 파일·중지 시 partial checkpoint·로그는 삭제하거나 새 결과로 덮지 않았다. 이전 미완료 비용·lifecycle·전수 검산은 목표 변경에 따라 실행하지 않는다. 완료로 처리하지 않는다.

## 판단 질문

1. 정상 목적의 학습으로 시공간 입력을 표현·복원하는가?
2. 같은 표현을 TSFM으로 미래 예측하면 native 복원기로 미래를 복원하는가?
3. 재구성 디코더를 그대로 쓰는 것과 동일 디코더를 실제 TRAIN F 출력에 calibration하는 것은 어떻게 다른가?
4. GT 공간 항을 사용했을 때 공간 항 제거 재학습보다 도움이 되는가?

## 실제 구현과 원문 대비 변경

- GT: [논문](https://proceedings.mlr.press/v198/sabbaqi22a.html)의 노드 유지, product graph, 양쪽 polynomial GT filters, max downsampling과 zero-insertion upsampling을 유지한 논문 기반 재구현이다. 공식 코드 미확보, Figure1B와 본문 연산 순서 차이, past→present temporal direction, 초기 product 계수, signed 입력용 linear 최종 출력은 [GT_METHOD_DELTA](GT_METHOD_DELTA.md)에 명시했다. L512→latent128×C×2→F12step→native D48×C이며 scalar 수는 절반, F의 행 수는 2C다. 공유 product 계수는 E와 함께 고정하고 calibration은 native D120개 필터만 적응한다.
- TS2Vec: [공식 pinned 소스](https://github.com/zhihanyue/ts2vec/tree/b0088e14a99706c05451316dc6db8d3da9351163)를 수정하지 않고 hidden64/depth10, 공식 crop/mask/contrast/AdamW/SWA와 기본200/600update 규칙을 사용한다. d16은 저차원 적용, d320은 공식 API 기본값의 확장 표현 대조다. 새 d→128→C 비선형 복원기는 원형에 없는 적용 모듈이다. EFD는 과거만 담은 offline window export를 D 학습과 추론에서 일치시키며, native ridge 대조는 공식 causal sliding1/padding200의 마지막 표현을 사용한다. [TS2_METHOD_DELTA](TS2_METHOD_DELTA.md).
- F: 실제 설치된 Chronos-Bolt2.3.2 small snapshot을 사용한다. embedding 좌표는 별도 batch 행이고 시간은 시간 축으로 유지한다. native64step 출력의 median을 실제 필요한12/48step으로 crop하며 공식 pipeline과 직접 model 경로의 synthetic parity를 확인한다. F를 처음 고정한 이유는 표현 연결의 효과를 분리하기 위해서다.

TS2의 정상 구조와 실제 학습 batch는 구분한다. Robin은 연속 TRAIN이3000보다 짧은 단일 instance여서 공식 fit의 실제 batch가1이고 instance contrastive 항은0이다. temporal hierarchical 항으로200 update를 수행했다. Jena canonical4조건은 메모리 상한 뒤 B4의 새 실행이며 정상 hidden/depth,600 update를 유지했지만 B8 대비 negative batch와 instance 노출은 달라졌다. 기본 batch의 정확한 재현이나 모든 자료에서 두 contrast 항이 모두 활성화된 학습으로 표시하지 않는다. 상세 근거는 [TS2 변경 기록](TS2_METHOD_DELTA.md)에 있다.

## 학습·비교·판정

Robin/Jena의 기존 실제 TRAIN/VAL과 관측 마스크만 사용한다. 정규화와 상관 graph는 TRAIN 관측값으로 적합한다. 재구성100epoch/patience15, decoder calibration50epoch/patience10은 초기 상한이며 최적 또는 수렴을 보장하지 않는다. 기존 개발 VAL에서 checkpoint와 ridge alpha를 선택하므로 독립 확증이라고 쓰지 않는다.

GT full4+공간 항 제거4, TS2Vec d16/d3208의 총16개 표현학습 조건을 실행한다. TS2의 별도 재구성 D와 모든 방법의 forecast calibration은 별도 fit으로 기록한다. 같은 origin·target·mask·스케일의 Raw F와 mean-time-compression R4, 동일 D의 latent persistence, TS2 native head를 비교한다. 나쁜 성능 때문에 다른 방법을 생략하거나 LEVEL로 복귀하지 않는다.

native ridge의 TRAIN은 stride1의1433/25648 origins이고 F-decoder calibration은 stride128의12/201 origins이다. 또한 causal last201 대 offline512 export, 선형 multi-horizon head 대 시점별 비선형 D가 다르다. 이것은 정상 사용 대조이며 native 우월성만으로 F가 불필요하다고 단정하는 동일 조건 ablation은 아니다. 수치 개선·MSE/MAE 교환관계·그대로 재사용과 미래 supervision 추가의 차이는 [최종16조건 진단](METHOD_FAILURE_DIAGNOSIS_KO.md)에 분리한다.

재구성·미래 예측·전체 추론비용을 분리한다. 입력/latent scalar 수와 F 행 수, horizon별·채널별 오류, 가능한 물리 단위 MAE, B1/B32의 실제 전체 경로 비용을 저장한다. 첫 실제 결과부터 중간 보고서를 만든다. 이후 모든 실제 예측은 별도 metric 코드로 검산한다.

## 범위와 보존

원본 첨부·과거 자료·원래 저장소 파일을 보존한다. 새 파일은 이 연구 폴더와 같은 이름 cache에 둔다. commit/push, 논문/PPT, 외부 전송은 하지 않는다. 과거 Electricity 보호 접근 사고는 [이전 기록](PROTECTED_ACCESS_CONTEXT.json)으로 남고, 새 loader의 TEST 접근0과 구분한다. 이 새 실행으로 이전 사고가 해소됐다고 주장하지 않는다.

8h GPU 작업과12h 전체 wall 운영 상한은 앞선 소비량을 보수적으로 이월했다. [예산](../RESOURCE_BUDGET.json)의 GPU/CPU/저장 이월값 일부는 보수적 예약이며 완전한 실측이 아니다. 상한에 도달하면 경로별 실제 상태·partial checkpoint·재개 조건·추가 예상 자원을 적으며 완료라고 쓰지 않는다.
