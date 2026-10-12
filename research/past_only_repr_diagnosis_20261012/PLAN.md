# 과거 전용 표현·지연 그래프 진단

사용자가 2026-10-12 직접 실행을 승인한 지시문을 따른다. 기존 LEVEL의 개선·재개가 아니라 과거 정보로 계산 가능한 표현과 지연 관계의 예측 가치를 진단한다. 원문은 실행기 옆에 보존한다.

E1 정보 사용 검사를 통과한 경로에서 E2 지연 direct ridge → E3 동일 SWA 인코더 FULL/PREFIX → E4 고정 reconstruction decoder에 대한 잠재 예측기 비교를 실행한다. E3의 학습/추출 문맥 불일치 또는 E4의 학습 목적 불확실성이 남으면 작은 공통 인코더 E5를 실행한다. E2 noiseless 정렬 검사가 통과하면 관계 변화 합성 E6가 가능하며 실자료는 사전 LAG_SIGNAL gate로 결정한다.

단순 대안은 persistence, self/cross direct ridge, reconstruction이다. PREFIX는 동일 왼쪽 시작점의 정확한 단일 prefix 마지막 표현이다. padding 최적화를 검증 없이 쓰지 않는다. 같은 seed에서 decoder 초기값·batch 순서·600-update 예산을 맞춘다. reference latent는 진단 파일에만 보존한다. 기존 TRAIN 전체에 노출된 SWA 진단과 새 FIT-only 학습을 구분한다.

NEW_FIT은 기존 TRAIN 앞80% FIT, 뒤20% TUNE, VAL 개발 진단이다. 물리값은 저장 observed 값의 FP32 역변환으로 복원하며 초기 통계는 FIT-only이다. 가중치/전처리/graph는 평가 정답에서 적합하지 않는다. λ=[1e-4,1e-2,1,1e2,1e4]와 TUNE macro MSE 선택을 사전 고정한다. 서로 다른 단위의 자료를 합치지 않고 물리 채널별 값과 표준화 지표를 함께 낸다.

신규 계산 GPU E1–E4 90분, E5–E6/LSTM 90분, CPU-only 누적 wall60분이 중단 상한이다. 읽기·구현 시간과 분리하며 기존 실행 비용은 새 예산에 넣지 않는다. 이미 실행중인 기존 GPU 프로세스가 끝나고 단일 GPU lease를 확보한 뒤 실행한다. 기존 중단 경로를 재시작하지 않는다.

E3/E4 비용 preflight로 FULL 묶음 또는 DIAG_SUBSET(FIT64/TUNE32/VAL≤32)을 성능 확인 전에 결정한다. 이 묶음도 비싸면 Jena d16 두 seed의 균형 대조부터 완료하고 나머지는 BUDGET_HELD다. E5 Jena4조건×2seed, E6 각자료/합성조건4graph×2seed는 균형 묶음별 예측 비용을 확인한다. 나쁜 결과도 보존하고 임의로 seed/조건/지표를 바꾸지 않는다.

E6 목적의 사전 구체화: E5 Jena에서 TUNE direct ridge 원공간 MSE의 두 seed 평균이 CAUSAL_REC와 CAUSAL_FORECAST 중 어느 쪽이 작은지로 목적을 선택하고 동률은 REC로 한다. E5 incomplete이면 명시된 FORECAST 기본값이다. 대조학습 우위를 평가하지 않고 graph 목적을 추가 변경하지 않는다. 이 선택은 계산 분기이며 VAL 개선/확증 기준이 아니다.

누출·mask·split·교차채널 대응 오류·NaN·메모리/예산/동시GPU 위험은 해당 경로를 막고 원 attempt를 남긴다. 완료 검산은 효과·수렴·독립 일반화 증거와 구분한다. 목적/자료/비용의 실질 변경은 제안만 남긴다. 자동 commit/push/TEST/대형 모델 재학습/외부 전송은 하지 않는다.
