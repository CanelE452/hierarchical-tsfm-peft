# 공개·비공개 경계

공개: 코드, aggregate SSE/SAE/count, full precision 지표, relative artifact bindings/SHA, recipe·receipt·ledger·resume queue, 그림 PNG/SVG와 plot source CSV. 공개 sufficient statistics만으로 정량 표·그림을 재생성할 수 있다.

비공개: 원자료 전체, full prediction/target/mask/context 배열, frozen-feature cache, model/optimizer/RNG/sampler checkpoints, 절대 자산·cache 경로, token/credential. 원 데이터 출처가 공개여도 임의 복제본 전체를 Git에 포함하지 않는다.

독립 예측 검산과 replay는 private binding이 필요하다. public-only 보고서 재생성은 모델·학습·대형 다운로드를 수행하지 않는다. F6 qualitative 원 곡선은 private replay 필요 범위이므로 이번 공개 그림에서 생략했다.
