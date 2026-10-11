# K별 비용–정확도와 교수님 제안 표현의 검증

사용자는 직전 권고의 목적 고정, K별 PCA/MLP·디코더 정규화 점검, GTConvAE/TS2Vec 중 한 방법 대조를 ‘그렇게 해줘’로 승인했다. routine hyperparameter·파일 구성은 이 승인 범위에서 정한다. 기존 sources/checkpoints/results와 TEST는 변경하지 않는다. 새 작업은 이 폴더와 같은 이름 cache에 기록하며 기존 history에는 이번 완료 기록만 추가한다.

## 목적 트리

뿌리: 고정 TSFM의 채널 처리 비용을 줄일 때 허용 가능한 정확도 손실과 실제 전체 추론 비용을 함께 판단한다. 소비처: 사용자·교수님이 왜 원채널 C 대신 K인지, 어떤 compressor가 필요한지를 판단한다.

- K sweep 목적 A: 작은 K가 얼마나 빨라지는지 확인. 목적 B: 같은 표현 계열에서 정확도 손실을 얼마나 감수하는지 확인. 필요한 조건: 같은 TRAIN/VAL/마스크/F/D fitting, 새K의 새MLP학습, K/C·변수 수·전체 E-F-D 비용 보고. 예상: 비용과 정확도가 모두 단조라는 보장은 없으며 어떤 K도 RAW보다 느리면 그 조건의 경량화 주장을 버린다.
- λ 확대 목적 A: 과거 grid상한 선택의 영향을 분리. 목적 B: 각 K/표현에 같은 규칙의 미래 readout을 제공. 필요한 조건: 모든 K/방법에 같은 TRAIN-only RMS·공통 grid·VAL 선택; 유리한 방법만 추가 탐색 금지. 예상: 순위가 바뀔 수 있으며 넓은grid에서도boundary가 남으면 최적성 주장을 유보한다.
- 교수님 방법 목적 A: TRAIN-only 자기지도 시간 표현이 PCA/MLP보다 좋은 FM 입력을 주는지 확인. 목적 B: 더 비싼 표현이 전체 비용 교환관계를 개선하는지 확인. 필요한 조건: 원형 알고리즘과 small-K 설정/외부D 변경의 구분, causal export 실제 경로 및 비용 포함. 예상: representation 품질이 좋아도 Bolt와 맞지 않거나 rolling encoding 비용 때문에 실용 이익이 없을 수 있다.

독자 질문 ‘그냥 C를 쓰면 되지 않나’에는 RAW 대비 MSE/MAE와 B1/B32 비용을 답으로 둔다. 허용 오차는 사용자가 고정하지 않았으므로 RAW 대비 MSE 증가5/10/20%를 모두 sensitivity로 표시하고 하나를 사용자 선호로 단정하지 않는다. 이는 반복 VAL development 분석이며 독립 TEST 결론이 아니다.

## 실행 계약

1. Robin K2/5/10(C17), Jena K3/6/12(C21), seed92601/92602, context512/horizon48, 공통 TRAIN/VAL origins·관측mask를 유지한다.
2. PCA 새K는 기존 standardized/imputed TRAIN전체행에 기존 centered covariance eigendecomposition을 적용한다. 기존middle K의basis는그대로재사용한다. PCA학습은closed form이며neuralepoch가없다.
3. MLP 새K8fits는기존구조의K출력으로fresh180epoch/2880updates를수행한다. 기존K middle4개encoder는봉인된180budget연구의selectedcheckpoint를재사용해중복학습하지않는다. 같은update수가같은FLOPs/parameter/wallbudget은아니다. 모든실제샘플origin stream SHA·optimizer/RNG·update수를이번에저장한다.
4. 주계약은FORECAST_RIDGE_D8만사용한다. 같은TRAIN512origins,TRAIN-onlyRMS,sumSSE ridge,lambda[1e-4,.01,1,10,100,1000,1e4]. E/F/예측latent가같은기존middle은저장trace를검증해재적합한다. 미래실제latent는D입력에넣지않는다.
5. 위7개grid의선택을진단용으로보존하고최종공통grid에는미리정한[1e-6,1e5,1e6]도모든조건에한번만추가한다. 즉최종grid10개를실행전에고정해교수방법까지같이적용한다. 초기7개선택과최종10개선택을모두보존한다. λ선택은같은VAL전체maskedMSE,같은tie규칙이다. 넓은grid에서boundary가남으면미확인으로보고하고무제한확장하지않는다.
6. 교수님제안은공식구현·출력계약·교체정도를확인해TS2Vec를선정한다. 별도method_protocol/source seal에서small-K/native contrastive·SWA/crop/mask·causal sliding1·외부D8을명시한다. default대규모논문설정재현이나GTConvAE실행으로부르지않는다. 중간K5/6에서두seed씩비교하고기존PCA/MLP최종grid와같은D를쓴다. update상한2880을맞추되objective/optimizer/parameter/dataorder차이를숨기지않는다.
7. RAW_F0도같은VAL입력에서재평가한다. 압축과D학습효과를구분하기위해원채널에동일D8을적합한RAW_C_D8(K=C,Eidentity,E학습0)도자료별하나포함한다. 실제TRAIN rawF예측으로같은10λ를적합하며PCA-fit으로부르지않는다. baselineD20개/200ridge,benchmark22경로×B1/B32=44cases다. 교수방법추가D4개/40ridge·benchmark8cases는별도stage다. 각최종E/D를B1/B32새process에서warmup5/timed20측정한다. 입력표준화·E/F/D·CPU↔GPU전송·sync포함;startup/디스크/채점제외. actualtimedinput/fullVALpredictionparity필수. trueCUDAallocated/reserved와sampledRSS를구분한다. 과학/문헌/검증과benchmarkGPU동시실행없음.
8. TRAIN-onlyfit/미래입력경계/F상태·no-grad/noTEST/source seal/이전파일보존을검증한다. K와MLP모델차원을결과id/cache명에포함해충돌을막는다. 실패·중단·재시도원장을보존하고예산이필요하면별도continuation으로기록한다.

## 판단과 완료 조건

자료별K/C·MSE/MAE·seed별값·선택λ·변수수·B1/B32시간·CUDA/RSS·학습비용을보고한다. MSE/latency Pareto와MSE5/10/20%조건의선택은모두development분석으로표기하고MAE손실도함께보인다. 교수님방법이개선되지않아도완료다. 완료는K별PCA/MLP전조건·λ공통추가규칙·교수방법선정/실행·전체비용·독립검증·보고서/그림/기여축소까지다. 이번G미포함으로LEVEL전체평가는아니다.

## 실행자 게이트

실제함수·shape·split을읽고확인한다. 원자료/중간checkpoint의C/K/source/origin/마스크일치가수치로성립해야reuse한다. 없는필드명을지시문에서가정하지않는다. 같은미래정답을입력으로넣거나이전폴더로쓰기·source seal불일치가생기면멈춰원인을확인한다. 새실험출력이이미존재하면덮어쓰지않고실제상태를검사한다. 유리한seed/K만재선택하거나failedrun비용을숨기지않는다.
