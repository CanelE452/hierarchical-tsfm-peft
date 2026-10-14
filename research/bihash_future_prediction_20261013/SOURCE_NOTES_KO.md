# ExoTimer 원형과 이번 adaptation

사용자 PDF `06237-AAAI26.WuL-ML (1).pdf`는 실제9페이지이며 SHA256 `5c5ac2b4b89c5b888ff8ca0c8bb5123bceed82d5f753b427cfb3784962a3789f`다. 본 계약은 식13–16을 읽고 만든 전이 연구다.

- [AAAI 공식 논문 PDF](https://ojs.aaai.org/index.php/AAAI/article/view/39908/43869)
- [공식 ExoTimer 저장소](https://github.com/graceLan1994/ExoTimer), 확인된 commit `f5ac33ecdef8cf62e1e20517751c7c36b21ec87d`. 해당 시점 tree는 README뿐이며 원 model/loss/trainer/hyperparameters를 확인할 수 없었다. 원 코드 재현을 주장하지 않는다.
- [UCI Household Electric Power Consumption](https://archive.ics.uci.edu/dataset/235/individual%2Bhousehold%2Belectric%2Bpower%2Bconsumption), 자료 계약·출처·license는 실제 ingest 기록에 연결한다.
- [UCI SML2010](https://archive.ics.uci.edu/dataset/274/sml2010), 두 원 파일·UTC·15분 계약을 실제 source bytes로 검사한다.

논문은 두 feature source마다 learnable projection+bias를 두고 Gumbel relaxation, hash similarity softmax, 원 continuous feature 조회, 양방향 GELU fusion, 두 예측의 학습 가능한 결합을 제시한다. 학습 가능한 partition은 단순 무작위 LSH나 고정 KMeans와 다르다. 다만 hash 자체가 InfoNCE와 동의어는 아니며 확인된 수식에 본 연구가 쓰는 symmetric InfoNCE 목적이 명시된 것은 아니다.

본 연구에서 변경한 것: 원 텍스트 feature를 FIT에서 이미 관측을 마친 미래 패턴 memory로 교체, 매 과거 시점의 정보 가용성 mask, continuous projected cosine/온도/.eps를 명시, 서로 다른 bank/window 수에서 A^T pooling의 column normalization, 미래 패턴 AE, forecast 목적의 공동 최적화, A의 training-only InfoNCE, tanh/ST/uniform/frozen maps 대조, C 상태 anchor와 frozen Chronos-Bolt 연결. 이 차이는 최종 결과의 일반화 범위를 제한한다.

원 query의 미래 정답을 두 번째 입력에 그대로 넣으면 배포할 수 없고 누출이다. FIT 미래 정답을 학습 loss에 쓰는 것과, 과거에 끝난 historical FIT target을 causal memory로 조회하는 것은 별개다. TUNE/EVAL target 또는 현재 미래 missingness mask는 prediction input이 아니다.

Gumbel 온도 감소는 임의 noise에 의한 sign flip을 없애지 않는다. soft/hard training과 deterministic sign 배포의 차이를 측정한다. 학습된 두 projection+bias에는 무작위 공유 Gaussian LSH의 정리나 거리 보존 보장이 자동으로 성립하지 않는다. Dense query-memory 계산은 N×M×r이며 실제 bit-packed/subquadratic 구현이 없으면 비용 절감으로 주장하지 않는다.

