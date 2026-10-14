# 선행연구: 작은 K·고정 Bolt와 겹치는 것과 남는 질문

상태: **Phase0 문헌 읽기 / 새 구현·실험 없음**. ForeCA/TRMF/DeepGLO/TS2Vec의 공식 본문에서 방법·실험 부분을 확인했다. AdaPTS는 arXiv v1 본문·부록과 공식 저장소를 읽었다. PMLR 최종 PDF는 fetch 실패로 **최종판 본문 미열람**이다. v1 확인을 최종판의 실험 확인으로 바꾸어 쓰지 않는다.

## ForeCA: 투영의 선택 기준을 바꾼다

[확인: 원문3–5쪽 §3–4, 식12/16/17] 정규화 스펙트럼 엔트로피를 줄이는 선형 성분을 whitening·직교 추출한다. 시간 길이를 줄이는 방법은 아니며 예측기·H·손실과 독립적인 예측 가능성 지표다. PCA/learned U와 겹치는 것은 선형 채널 투영이고, 차이는 variance 대신 spectral 기준이라는 점이다. 정상성·2차 통계가 중요하며 낮은 엔트로피가 원미래 C의 H48 MSE나 Bolt 이득을 보장하지 않는다. latent 분산·유효 rank와 원미래 오차를 함께 검사해야 한다. [공식 원문](https://proceedings.mlr.press/v28/goerg13.pdf).

## TRMF: 재구성에 잠재 AR 제약을 결합한다

[확인: 원문4–6쪽 §3–4, 식7/9/12] 관측 항목의 factorization 오차와 lag별 잠재 AR 제약을 교대로 최적화하고 latent AR 미래를 채널 계수로 복원한다. 원형은 frozen TSFM을 사용하지 않는다. 재구성과 잠재 예측 목적의 결합은 선행기술이다. 새 origin에서 과거만 쓰는 latent 추정·rolling update와 비용을 정의해야 현재 배포 계약과 맞는다. 원문의 signed graph는 시간 노드의 정규화 연결이어서 directed cross-channel lag graph의 검증 근거로 쓰지 않는다. [공식 원문](https://papers.neurips.cc/paper_files/paper/2016/file/85422afb467e9456013a2a51d4dff702-Paper.pdf).

## Think Globally, Act Locally / DeepGLO: 전역 예측과 지역 모델

[확인: 원문5–6쪽 §5.1–5.2, 식3/Algorithm2–4] low-rank factorization을 TCN으로 시간 정규화하고 전역 예측·원채널 과거·covariate를 지역 TCN에 넣는다. 전역 TCN은 학습되며 frozen Bolt 논문이 아니다. LEVEL/current-C와 global/local 발상이 겹친다. 지역 모델이 과거 sequence와 전역 future forecast를 함께 받는 차이는 있지만 strict K-only가 아니다. 미래에 알려지는 달력 공변량 계약을 실제 미래값 입력으로 확대하면 안 된다. 동등 보정·정보 사용·whole 비용을 맞춰 순이득을 확인해야 한다. [공식 원문](https://proceedings.neurips.cc/paper_files/paper/2019/file/3a0844cee4fcf57de0c71e9ad3035478-Paper.pdf).

## TS2Vec: 일반 시점 표현과 ridge 예측

[확인: 원문 Method/Forecasting] crop/mask 문맥의 같은 시점 positive에 temporal/instance contrast를 계층 시간 해상도로 적용한다. 작은 K 채널 압축·decoder·Bolt 호환을 목표로 하지 않는다. Forecasting은 마지막 표현→ridge→H×C이며 원형 native가 직접 대안이다. 공식 예측 코드는 `causal=True`, `sliding_length=1`, `sliding_padding=200`을 함께 쓴다. causal 플래그만으로 비sliding 분기가 과거 전용이 되지는 않는다. 기존 실험을 재사용하되 표현 사용·학습 origins·readout을 맞춰야 한다. [공식 원문](https://ojs.aaai.org/index.php/AAAI/article/view/20881/20640), [공식 예측 코드](https://github.com/yuezhihan/ts2vec/blob/main/tasks/forecasting.py).

## AdaPTS: 가장 직접적인 중복과 표준 대안

[확인: **v1**3–5쪽 §3.1/4.1/4.2, 부록C] encoder→frozen univariate FM→decoder를 실제 미래 오차로 학습한다. C→K 투영이며 시간 압축은 아니다. v1은 MOMENT-small/L512를 사용하고 먼저 forecast head를 학습한 뒤 동결한다. 완전 zero-shot Bolt와 baseline이 다르다. 공식 README는 MOMENT/Moirai/TTM/TimesFM을 지원하며 Bolt 직접 지원은 확인하지 못했다. Bolt 포팅을 원형 직접 재현으로 명명하지 않는다. 정사각 invertible 선형 adapter 이론을 K≪C/nonlinear Bolt의 보장으로 확대하면 안 된다. [v1 원문](https://arxiv.org/pdf/2502.10235v1), [공식 저장소](https://github.com/abenechehab/AdaPTS), [ICML 서지·최종판](https://proceedings.mlr.press/v267/benechehab25a.html).

현재 예측 목적 E/D 자체는 이 방법과 직접 겹친다. current-C/비용 차이가 기여가 되려면 Raw+동등 보정·작은 K·native 대안 비교 후에도 남아야 한다. 공식 API의 random window split이나 reconstruction/KL 기본 결합을 현재 명시적 FIT/TUNE 계약 대신 복사하면 안 된다. [공식 학습 소스](https://github.com/abenechehab/AdaPTS/blob/main/src/adapts/adapts.py).

## 후보 선택

[판단] 원리가 다른 세 계열은①ForeCA/TRMF형 시간 예측성②AdaPTS형 표준 predictive adapter③DeepGLO/LEVEL형 global/local 결합이다. 처음부터 모두 학습하지 않는다. 공통 정보 유용성과 기존 같은 E/D의 Bolt 가치가 첫 gate다. 필요한 경우 ForeCA 원리 기반 projection과 표준 linear forecast E/D를 최대2후보로 비교하며 후자를 신규법으로 주장하지 않는다. 구조보다 성립 조건과 동등 보정 후의 정확도/비용이 최상위 판단이다.
