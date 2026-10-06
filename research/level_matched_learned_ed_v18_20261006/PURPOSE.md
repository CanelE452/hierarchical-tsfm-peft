# Fixed PCA LEVEL vs matched learned E/D

같은 K에서 E/D를 forecast loss로 학습하는 것과 고정 PCA E/D에 외부 잔차 G를 붙이는 것 중 어떤 선택이 현재 세 자료의 오차와 gradient 경로 비용을 지지하는지 확인한다.

공식 AdaPTS의 Linear E/D 학습 경로를 참고한 구조 대조다. 전체 확률 모형의 재현이나 parameter-matched 비교로 주장하지 않는다. 이미 노출된 TEST를 쓰는 후속 대조이며, 기존 실험 횟수와 검산 PASS는 논문 신규성이나 독립 재현의 근거가 아니다.

원래 LEVEL과 내부 LoRA A·학습 U/Gamma·그룹 기저·비선형/시간축 후속 변형을 섞지 않는다. 기존 결과와 실패 기록을 보존한다.
