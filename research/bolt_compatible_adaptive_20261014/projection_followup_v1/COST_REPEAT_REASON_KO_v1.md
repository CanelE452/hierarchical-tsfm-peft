# 전체 비용 반복의 사유와 고정 조건

새 방향76셀에서 Beijing Raw F B8 batch 중앙값은27.5705ms였고, 후속 시간 보정72셀에서는18.7247ms였다. 두 실행의 원 physical/scaler/F model·median 계약은 같고 Raw callback의 핵심은 모두 `bolt(engine.tensor(history))`다. 부가 copy·검사·동시 상주 상태·경로 순서·측정 시각은 다르다. 이 차이를 압축이나 새 callback의 인과 효과로 설명할 근거는 없다.

따라서 시간 보정72셀 코드를 수정하지 않고 `horizon_decoder_cost_repeat_v1`이라는 새 출력 위치에서 전체72셀·360표본을 한 번 더 측정한다. 원 선정24계수·과학 결과·공차·원 물리 firstB TUNE·B1/B8·warmup2/repeat5·order seed와 측정 경계를 유지한다. 기존 결과·SOURCE·실패를 덮거나 더 좋은 실행만 선정하지 않는다. 새 실행은 이전 중단 학습의 재개가 아니다. 같은 목적의 지속 실행 안의 미해결 측정 변동 검산으로 별도 승인을 요구하지 않는다.

두 실행의 원시 표본과 paired 비율을 나란히 보며 실행 간 차이를 남긴다. 반복1회가 독립 입력 일반화나 안정된 최적 backend를 입증하지 않는다. 첫72셀 보고는 원 실행에 결속하고 반복 비용을 대입하지 않는다. 새 repeat의 원본 보존·점수 parity·정수 타이머 집계는 따로 검산한다. GPU process meter wall과 kernel 시간을 구분하며 TEST·설치·큰 모델 학습·자동 Git은 하지 않는다.
