# 대응 비교 계약

실행 계약과 등록 cell은 [CAMPAIGN_CONTRACT.json](CAMPAIGN_CONTRACT.json), 데이터·채널 순서·분할은 [DATA_CONTRACT.json](DATA_CONTRACT.json)에 있다. 최초 선택점은 기존과 같은 0/.25/1/3 epoch다. epoch2를 새로 넣으면 endpoint의 작은 꼬리 batch 때문에 optimizer 업데이트 경계가 달라지므로 넣지 않았다.

창 내부 causal fill → A_fit 또는 fresh A60의 원채널 global z-score → 호출별 원채널 RevIN → 기존 B2 → anchor 포함 예측 전체의 RevIN inverse → global-normalized loss/metric → 물리 단위 출력의 순서다. ON은 2C affine 파라미터만 더하며 OFF는 원 모델을 그대로 반환한다. 결측 target은 observed mask로 제외하고 채널별 SSE/count의 평균을 MSE로 계산한다. 학습 선택에 V·M_cal·T를 사용하지 않는다.

3 epoch의 선택 구간 마지막 값이 이전 모든 checkpoint보다 엄격히 낮으면 해당 데이터의 OFF/ON과 두 seed를 모두 4/5/6 epoch까지 확인한다. source/data/F/scaler/PCA/optimizer/sampler/CPU·CUDA RNG 상태를 검증한 원 search checkpoint에서 별도 branch를 만들며 원 receipt를 고치지 않는다. contract 필드의 명시적 migration과 나머지 상태의 동일성은 각 확장 search의 MIGRATION_PROOF에 남긴다.

모든 등록 search, 공통 정규화 선택, Jena B0 선택을 먼저 완료·봉인한다. 이어 3 epoch 선택 모델과 조건부 6 epoch 선택 모델을 각각 fresh A60 refit하고 V 예측을 저장한다. 이는 cap3 결과와 cap6 결과를 모두 보존하면서 새 V를 선택에 사용할 여지를 막는 실행 순서다. 각 cap의 checkpoint schedule과 선택점은 해당 표에 표시한다.

정규화는 각 데이터의 두 seed에서 선택한 A_stop MSE 평균으로 OFF/ON을 정하고, 정확한 동률이면 OFF다. Jena 구성 비교는 그 공통 정규화와 최종 B2 cap에 대응한 B0 두 seed다. B0/B2는 P의 존재·용량이 다르므로 그 차이만으로 P의 순수 원리나 F의 고유 기여를 완전히 분리했다고 주장하지 않는다.

6 epoch에서도 마지막 checkpoint가 계속 개선하면 학습 상한/undertraining으로 표시한다. 다른 checkpoint가 선택돼도 수렴 증명은 없다. V는 반복 개발 평가다. 과거 T 노출, 사전학습 overlap UNKNOWN, 고정 K, 두 seed의 제한은 유지한다.

비용은 저장된 최종 scored bundle을 실제 reload한 GPU B1/B8의 물리 입력 전체 경로다. 창 구성은 기존 계약처럼 제외하고 causal fill·global normalize·RevIN·P/E/F/D·역정규화·CPU 출력은 포함한다. cache forecast는 timed 경로에 쓰지 않는다. batch elapsed/B를 ms per origin으로 표시하고 F rows와 실제 호출 수, 상주 F weights와 trainable adapter를 구분한다. Ridge 기존 CPU 비용은 별도 조건으로 표시한다.
