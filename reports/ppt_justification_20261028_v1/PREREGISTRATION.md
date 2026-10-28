# PPT 당위성 보완 실험 사전등록

기본20회: NO_F_LINEAR Jena5 → NO_F_LINEAR_NO_P Jena5 → NO_F_LINEAR Appliances5 → NO_ANCHOR ETTh1 5. 모든 seed는92601–92605다.
선택10회: A5_LAG0 ETTh1 5 → NO_ANCHOR Appliances5를 고정 순서로 예산이 허용할 때만 수행한다.
학습은10월28일17:30KST까지, 검증·발표자료 전달은20:00KST까지다. 승인된 새 캠페인만 수정한다.

F 제거: 기존 R5b 설계처럼 전체 P/E/D를 같은 초기화에서 새로 학습하고 F만 공유 Linear512→48으로 대체한다.
head는 마지막 latent를 반복하는 persistence 초기화이며24624개 파라미터가 추가된다. F 생성·호출은0이다.
NO_F_LINEAR_NO_P는 같은 fullB2 초기화 후P를Identity로 바꾸고 동일head를 추가한다. 기존Full/NoP Bolt와 함께P유무×예측기2×2 탐색을 구성한다.
기존NoP Jena 첫5는 cap3에서5/5 상한선택 후cap10으로 연장되어[3,6,6,6,6]을 선택했다. Full의cap3과 이 일정 차이를 공개한다.
Jena 사전 대비는 NO_F_LINEAR−OURS, NO_F_LINEAR_NO_P−A1, NO_F_LINEAR_NO_P−NO_F_LINEAR, A1−OURS다.
seed별 조건부 interaction=(NO_F_LINEAR_NO_P−NO_F_LINEAR)−(A1−OURS)의 전체 값과 중앙값을 기술한다. 일정/용량이 완전히 균형 잡힌 인과 상호작용 검정으로 부르지 않는다.
직접 수준 가산 제거: x_last 더하기만 없애며7lag D와 latent 차분·P/E/F·초기화를 유지한다. 새 bias는 없다.
입력 observed history와 z_last 및 lag별 관측 latent 차분은 여전히 남는다. 모든 과거 관측값을 제거한 실험으로 해석하지 않는다.
A5: x_last 가산은 유지하고 lag0만 사용한다. 세 질문을 서로 대체하거나 하나의 효과로 묶지 않는다.

원래 A_fit/A_stop, AdamW(.001,wd0), clip1, effective16/micro4, FP32, K(Jena6/App10/ETTh1 4), L512/H48을 유지한다.
checkpoint0/.25/1/3에서 A_stop 최소값으로 선택하고 동률은 이른 epoch, 선택 봉인 후 A60 fresh refit, 마지막 V 평가를 수행한다.
전체5개중3개이상 cap3 선택 시 해당5개 전체를0/.25/1/3/6/10으로 연장한다. 연장5개가 모두 끝나야 주 결과가 된다.
cap10도3/5이상 선택하면 학습 상한 미해결로 표시하고 더 반복하지 않는다. Full-P 첫5개는 모든 데이터 cap3선택0/5라 연장이 필요 없다.
우선순위는 기본20 → 그 연장 → 선택A5 ETTh1와 그 연장 → 선택NO_ANCHOR App와 그 연장이다.
각 묶음은 남은 시간에5개 모두 끝낼 수 있어야 시작한다. F유지 기본18분/연장30분으로 보수 추정한다.
NO_F는 첫 성공실행을 포함한 같은 데이터 완료시간 최댓값×1.35로 갱신하며 연장은10/3배, 하한60초다.
V의 좋고 나쁨으로 실행 범위·seed·선택·임계값을 바꾸지 않는다. 기술적 실패는 같은 설정1회 재시도, 과학 게이트 실패는 중단한다.

기존 Full-P 같은5seed의 예측 해시·cohort·정규화·FP64 지표를 먼저 검산해 재사용한다. 보호T 숫자는 읽지 않는다.
모든 seed, 중앙값/IQR, pair별 차이, 파라미터·시간·선택epoch를 공개한다. 중앙값/max-IQR 규칙은 기술적 비교이며 유의성·동등성 검정이 아니다.
n5·기존V 반복사용·새head 용량차이·no-anchor 매개화/최적화 차이·미완료/상한 상태를 숨기지 않는다.
결과가 방법론을 지지하지 않으면 주장 범위를 줄이고 기존 부정/불명확 결과를 보존한다.
