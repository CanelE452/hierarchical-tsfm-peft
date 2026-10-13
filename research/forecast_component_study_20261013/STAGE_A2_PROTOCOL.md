# 원래 FORECAST의 학습 origin 수 대조

2026-10-13 Stage A 실제 추론 후, Stage B의 새 구조와 표본 확대 효과를 혼동하지 않도록 추가한 소규모 원인 대조다. 현재 사용자의 FORECAST 실패 진단 요청 범위 안에서 수행한다.

- 기존 E5의 E_u0000 encoder/head를 그대로 읽어 시작한다. RF63, latent16, 원래 예측 head, Adam1e-3,600updates와 element-wise masked MSE를 유지한다.
- Jena batch8 / Robin batch2를 원래대로 유지하고 두 seed를 모두 수행한다.
- sparse 경로는 기록된 원래 학습 origins를 재생한다. dense 경로는 같은 FIT-only 공통256 origins를 사용하며 같은 seed로 배치를 만든다. 총8개 작은 CPU 신경망 학습이다.
- 다른 데이터·H·관측 범위·정규화는 바꾸지 않는다. 0/100/200/300/400/500/600 checkpoint 중 공통TUNE64 물리단위 macro MSE로 선택하고, 선택점 및 final600만 VAL 평가한다.
- 원인 변수는 학습 origin 구성이다. 중첩 origin256을 독립 표본256이라고 해석하지 않는다. sparse CPU 재생이 원래 GPU 결과와 완전 동일하지 않을 수 있으며 실제 차이를 기록한다.
- 새로운 stage_a2 폴더와 캐시에만 저장한다. TEST 접근 차단, 원본checkpoint/source SHA, 실제 비용, 유한gradient 및 저장 예측 재검산을 포함한다.
- 추가 CPU 작업 상한5분, GPU0. 새 dense 결과를 보고 구조·loss를 다시 바꾸지 않는다.
