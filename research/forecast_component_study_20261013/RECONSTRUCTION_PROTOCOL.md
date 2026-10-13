# 고정 과거 창 재구성 재평가 계약

2026-10-13. Stage B의 미니배치 loss 변화만으로 재구성 학습의 성공/실패를 추정하지 않기 위한 추가 진단이다. 새로운 학습·모델 선택은 하지 않는다.

- 저장된 REC/PRED codec의 update 0/100/200/400/600을 모두 평가한다. Jena/Robin, 두 seed, 두 objective로 총40 snapshots다.
- 원래 Stage B의 causal E와 linear D를 CPU에서 복원하고 과거512의 `D(E(X_past))`만 계산한다. Bolt를 로드하거나 실행하지 않는다.
- 기존 bounded TRAIN/VAL 원자료 파일의 SHA를 확인하고 같은 FIT-only 통계·fill을 사용한다. 평가 배치는 기존 공통 FIT256/TUNE64의 과거 창만 사용한다. VAL window/target/점수를 생성하지 않고 보호 TEST 경로는 audit hook으로 차단한다. 원자료는 기존 공유 TRAIN/VAL 파일 안에 저장되어 있음을 구분한다.
- 같은 입력·observed mask에서 정규화 및 물리 단위 channel-macro MSE/MAE와 채널별 수치를 평가한다. 겹치는 창의 위치는 중복 가중되며 독립 표본으로 해석하지 않는다.
- latent 채널별 분산/std, 최소 std, 최대 절대값, finite 여부를 기록한다. 소규모 미래-과거 변형 및 양성대조로 causal path를 확인한다. 잠재 분산이 0이 아니라는 사실만으로 모든 형태의 붕괴를 배제하지 않는다.
- 같은 checkpoint의 미래 TUNE 예측 수치는 기존 Stage B JSON에서 SHA로 연결해 재사용한다. 이번 진단의 새로운 과거 재구성 실행과 구별한다.
- 실제 재구성·latent 배열과 고정 입력/mask를 새 cache에 보존한다. 별도 NumPy 채널 루프를 통한 재채점으로 MSE/MAE 및 부호·단위 변환·형상·해시를 검사한다.
- CPU 최대180초, PyTorch threads2, GPU0, 추가 학습0. 새 `diagnose_reconstruction.py`, 본 계약, `reconstruction_audit_v1/`, 새 reconstruction cache만 기록한다. 기존 결과를 덮지 않는다.
- 판정: 고정 FIT/TUNE 재구성 개선 여부를 먼저 확인한다. 재구성 개선과 미래 예측 악화가 동시에 있으면 이 구성의 목표 불일치와 일치하는 증거다. 재구성도 나쁘면 학습/일반화 문제를 배제하지 않는다. 어느 결과도 전체 방법군의 불가능성이나 잠재 미래 예측성의 단독 인과 증거로 확대하지 않는다.
