# 교수님 방법 직접 실험: 복원·예측·최적화 위치 진단

작성 범위: 2026-10-11, 현재 `protocol.json`의 canonical **16개 모두**에 완료 JSON이 있는 것을 확인한 진단이다. GT full/no-spatial은 두 자료·두 seed 총8개, TS2 d16/d320도 두 자료·두 seed 총8개다. Jena TS2 4개는 batch4 재실행이다. TS2의 최초 Robin pilot, 보호된 기존 결과, RSS 상한을 넘긴 Jena batch8 및 cache-only batch8 시도는 canonical 평균에 넣지 않는다. 이 파일은 실제 run·checkpoint·학습 소스를 변경하지 않는다. 전체 저장 예측의 독립 검산·비용 집계는 별도 보고서의 확인 범위를 따른다.

## 현재 관찰: 복원 가능성과 예측 가능성을 따로 본다

모든 값은 새 데이터 계약의 TRAIN-only 표준화 좌표, 기존 개발 VAL의 observed macro channel metric이다. 과거 512개 **입력 재구성**과 다음 48개 **미래 예측**의 대상 구간이 다르므로 두 MSE를 빼서 순수 TSFM 오차라고 해석하지 않는다. 반복 사용한 VAL이므로 독립 일반화 검증도 아니다.

```text
자료/조건                           입력 재구성 MSE/MAE       미래 TRANSFER_F MSE/MAE    미래 CALIBRATED_F MSE/MAE
Robin GT full s92601                 0.911652 / 0.691530       1.417153 / 0.916314         1.411559 / 0.914979
Robin GT full s92602                 0.921465 / 0.682103       1.449776 / 0.916689         1.448647 / 0.916714
Robin GT no-spatial s92601           0.921609 / 0.705377       1.426421 / 0.925944         1.426421 / 0.925944
Robin GT no-spatial s92602           0.899792 / 0.690123       1.415778 / 0.915085         1.413772 / 0.913551
Jena GT full s92601                  1.900379 / 0.921675       2.069021 / 1.015822         2.040766 / 1.005277
Jena GT full s92602                  0.257297 / 0.222071       0.777821 / 0.493367         0.774347 / 0.503067
Jena GT no-spatial s92601            2.208297 / 1.132361       2.193126 / 1.130374         2.193126 / 1.130374
Jena GT no-spatial s92602            1.976412 / 0.933571       2.119348 / 1.025946         2.103230 / 1.032503
Robin TS2 d16 s92601 epoch0_v2        0.400887 / 0.408745       0.980705 / 0.690139         0.980705 / 0.690139
Robin TS2 d16 s92602                 0.392856 / 0.428698       0.957489 / 0.728880         0.956809 / 0.726643
Robin TS2 d320 s92601                0.111495 / 0.209015       0.616574 / 0.543643         0.616504 / 0.544000
Robin TS2 d320 s92602                0.135573 / 0.237398       0.632825 / 0.564094         0.632825 / 0.564094
Jena TS2 d16 s92601 batch4            0.150933 / 0.261605       0.870604 / 0.627582         0.623235 / 0.491803
Jena TS2 d320 s92601 batch4           0.018147 / 0.092482       0.662064 / 0.468634         0.514303 / 0.406348
Jena TS2 d16 s92602 batch4            0.167356 / 0.273821       0.833465 / 0.601538         0.630853 / 0.492518
Jena TS2 d320 s92602 batch4           0.025448 / 0.106630       0.735020 / 0.503515         0.528330 / 0.416673

자료                  미래 RAW_F MSE/MAE         단순 mean-block R4→F→repeat MSE/MAE
Robin                 0.408587 / 0.339458         0.904856 / 0.609356
Jena                  0.554918 / 0.373891         0.721858 / 0.455264
```

[확인] canonical 16개 모두에서 **재구성 D를 그대로 쓴 TRANSFER_F**의 미래 MSE·MAE는 같은 자료의 Raw F보다 높다. 그러나 Jena TS2 d320의 **CALIBRATED_F**는 두 seed에서 Raw MSE0.554918보다 낮은0.514303·0.528330이고, MAE는 Raw0.373891보다 높은0.406348·0.416673이다. MSE 개선과 MAE 악화가 공존한다. 같은 표현의 native ridge는 두 seed에서 Raw의 두 지표를 모두 개선했다. “전부 실패했고 표현학습에 가능성이 없다”거나 “비지도 압축기 연결만으로 예측이 개선됐다”는 두 주장 모두 이 결과와 맞지 않는다. Raw 대비 개선이 나타난 경로에는 미래 정답으로 학습한 D 또는 head가 추가된다. 입력 재구성과 미래 예측의 구간도 다르다.

근거: 아래 수치·출처 추적의 canonical16개 JSON, [Robin 대조](baselines/robin/result.json), [Jena 대조](baselines/jena/result.json). 최초 [TS2 pilot](runs/robin_TS2_d16_s92601/result.json)은 구현 수정 전 기록으로만 보존하며 canonical 집계에서 제외한다.

## GT: 최적화·초기화 민감성과 공간 대조를 함께 본다

[확인] GT full의 Robin 두 seed는 TRAIN 창12개×100 epoch로 각각200 update, Jena 두 seed는201개×100 epoch로 각각2600 update를 실행했다. full4개의 최선 reconstruction checkpoint는 모두 epoch100이다. seed92601의 Robin VAL reconstruction MSE는 epoch1 1.132811→91 0.913943→100 0.911652, Jena는2.445968→1.927258→1.900379다. Jena seed92602는3.054094→0.263080→0.257297이다. **같은 자료·구조·학습 상한에서 Jena 재구성 MSE가1.900379와0.257297로 크게 갈렸다.** seed는 가중치 초기화와 학습 창 순서를 함께 바꾸므로 원인을 초기값 하나로 단정할 수 없지만, 초기화/최적화 경로 민감성을 무시하고 두 평균만으로 안정적인 성능이라고 보고해서도 안 된다. 두 seed는 모집단 안정성이나 유의성을 보장하지 않는다.

[확인] GT full seed92601의 두 자료 및 Robin seed92602는 calibration epoch50이 최선이다. Jena full seed92602는19 epoch·494 update 후 best epoch9를 선택했다. 미래 MSE0.777821→0.774347은 줄지만 MAE0.493367→0.503067은 커져 교환관계다. “calibration이 모두 개선했다”로 쓰지 않는다. 공간 제거 Jena seed92601은 reconstruction44 epoch 중29가 최선, seed92602는100 epoch 중98이 최선이다. 모든 GT가 상한까지 개선한 것도 아니며 모든 GT가 충분히 수렴한 것도 아니다.

[확인] 공간 대조의 원수치는 위 표에서 함께 보인다. Robin에서 full이 no-spatial보다 calibrated MSE/MAE 모두 좋은 것은 seed92601이고, seed92602는 no-spatial이 둘 다 좋다. Jena는 두 seed에서 full이 no-spatial보다 calibrated MSE/MAE 모두 낮지만, 그 차이는 seed마다 크게 다르다. 특히 잘 복원된 Jena full seed92602도 Raw F0.554918/0.373891 및 단순 시간 축소0.721858/0.455264보다 미래 오차가 높다. [판단] 이 자료·설정에서 Jena 공간 연산의 추가 가능성은 관찰됐으나 안정적인 효과 크기나 다른 자료로의 일반화는 미확인이다. graph의 독립적인 인과 효과를 초기화/최적화 경로와 분리해 입증한 결과도 아니다.

공유 product 계수와 E를 고정한 채 native D120개 filter만 적응시켰으므로 encoder/graph의 손실을 모두 교정할 자유도를 제공하지 않는다. 예정된 prototype 학습 완료와 최적화를 충분히 했다는 증거를 구분한다.

[추정] 가능한 원인은 다음과 같다. 현시점에서 인과적으로 확정한 원인은 아니다.

- 244개 전체 학습 계수와 노드에 공유되는 filter bank가 서로 다른 원채널의 패턴·수준을 복원하기에 제한적일 수 있다. 이는 원형 구조를 유지한 초기 적용 설정의 한계 가설이다.
- max pooling·hidden ReLU와 right-edge zero insertion이 자료의 신호를 보존하는 데 불리할 수 있다. scalar 절반의 시간 압축 자체도 손실을 만든다. 입력창 시작 3개 출력은 구조적으로 0이지만 전체 길이512의 약0.59%이므로 그 사실만으로 모든 reconstruction 실패를 설명하지 않는다.
- 원문 예시 rho0.14를 observed-entry **평균** loss에 적용했다. sum/expectation loss와 같은 상대 강도라고 보장하지 않는다. Jena selected product 계수는 `[s00≈0.000831,s01≈-0.065005,s10≈0.350296,s11≈-0.012126]`, Robin은 `[0.000773,0.484846,0.479951,0.000424]`다. 공간 coupling이 작아졌다는 관찰만으로 graph가 불필요하다고 결론내릴 수 없다. 필터 크기와 공동 작용하므로 product coefficient의 절대 크기는 단독 중요도 지표가 아니다.
- Jena 최종 curve의 TRAIN reconstruction 값 약0.638453와 VAL macro reconstruction MSE1.900379의 차이는 개발 구간의 변화·채널별 분포 차이 가능성을 보여 준다. TRAIN curve 값은 각 update 직전 **배치별 observed-entry MSE를 동일 가중치로 평균**한 값이다. 마지막 작은 배치에도 같은 가중치를 주며 epoch 끝의 고정 checkpoint로 전 TRAIN을 재평가한 metric이 아니다. VAL의 고정 checkpoint macro 평균과 직접 빼서 일반화 갭으로 정량화하지 않는다.

검증 가능한 다음 판단은 먼저 충분히 복원되는 E/D checkpoint를 확보할 수 있는가다. 더 오래 학습하거나 loss/구조를 변경하는 추가 실험은 별도 비용·프로토콜로 기록해야 한다. 현재 정해진 실험을 자동으로 변경하거나 기존 LEVEL로 돌아갈 이유로 삼지 않는다.

## GT의 time-delayed 경로와 directed lag graph 발견은 다르다

[확인·원문] GTConvAE의 Eq.(1)은 주어진 공간 shift S와 temporal shift S_T를 네 **scalar** `s00,s01,s10,s11`로 결합한다. Eq.(2)의 polynomial powers는 여러 spatiotemporal hop을 합친다. 원문은 이 product coupling을 학습하는 구조이며, 모든 채널쌍의 방향·지연 길이를 독립적으로 추정하는 파라미터화를 제시한 것으로 읽히지 않는다. [GTConvAE 원문, Section 2.1–2.2](https://proceedings.mlr.press/v198/sabbaqi22a/sabbaqi22a.pdf).

[확인·현재 구현] `data_io.py`의 graph는 TRAIN에서 두 채널이 모두 관측된 동시점 **절대 Pearson correlation**을 구한 뒤 diagonal0·spectral norm 정규화를 적용한 고정 대칭 행렬이다. `models_gt.py`는 graph를 buffer로 보관하며, 시간 방향은 past→present directed line이다. 따라서 이 구현은 다음을 충족한다.

- graph 이웃의 과거가 현재 다른 노드 표현에 영향을 주는 **시차가 있는 교차채널 연산 경로**를 갖는다.
- 공간 graph와 시간 line의 coupling scalar 및 GT polynomial filter를 학습한다.
- 노드쌍별 lag coefficient나 edge별 방향을 별도 학습하지 않는다. zero-lag abs correlation이 약한 지연 관계는 초기 graph에서도 약해질 수 있다.
- 입력 mask와 비선형 활성화 때문에 실제 영향이 입력별로 다르더라도 그것을 명시적 learned directed lag graph의 발견으로 부르지 않는다.

특히 `s11=0`만으로 모든 lagged cross-channel 경로가 사라지지 않는다. polynomial의 두 hop에는 `s01*s10`으로 공간 이동과 시간 이동이 결합될 수 있다. 이번 no-spatial 대조는 `s01=s11=0`을 동시에 고정하여 모든 공간 항을 제거하고 실제 재학습했다. 두 seed·두 자료의 대조가 완료됐으며 그 관찰은 위 표의 조건부 결과다. 이를 임의의 지연 그래프 학습의 증거나 모든 자료에서 graph가 유리하다는 주장으로 확장하지 않는다.

교수님 메시지는 지연된 채널 패턴을 활용할 수 있다는 초기 기대에는 부합한다. 그러나 **arbitrary directed, channel-specific lag graph를 데이터에서 발견했다**는 더 강한 주장까지 현재 원문·코드로 충족했다고 보고하면 과장이다. 원형 GTConvAE의 검증과 그런 별도 graph-learning 확장은 다른 질문이다.

## F의 추가 기여와 전체 Raw 예측 우위는 다르다

같은 E/D에서 F를 latent 마지막값 유지로 바꾸는 진단을 비교한다. latent 실제 미래와의 직접 오차가 아니라 **D로 복원한 원공간 미래 오차**다.

```text
자료/조건                         F transfer MSE/MAE          latent persistence transfer MSE/MAE
Robin GT full s92601              1.417153 / 0.916314          1.617472 / 0.954726
Robin GT full s92602              1.449776 / 0.916689          1.463597 / 0.915905
Jena GT full s92601               2.069021 / 1.015822          2.101736 / 1.029950
Jena GT full s92602               0.777821 / 0.493367          1.147355 / 0.607923
Jena GT no-spatial s92601         2.193126 / 1.130374          2.193127 / 1.130369
Robin TS2 d16 s92601 epoch0_v2     0.980705 / 0.690139          1.504484 / 0.858220
Robin TS2 d320 s92601              0.616574 / 0.543643          1.397943 / 0.783758
Jena TS2 d16 s92601 batch4         0.870604 / 0.627582          1.107237 / 0.694037
Jena TS2 d16 s92602 batch4         0.833465 / 0.601538          1.076047 / 0.684483
Jena TS2 d320 s92601 batch4        0.662064 / 0.468634          1.138117 / 0.610356
Jena TS2 d320 s92602 batch4        0.735020 / 0.503515          1.183484 / 0.623352
```

[확인] 완료16조건 모두에서 F transfer의 MSE가 단순 latent persistence보다 낮고, MAE도 낮은 것은14조건이다. Robin GT full seed92602는 MSE/MAE 교환관계이며, Jena no-spatial seed92601의 차이는 극히 작고 MAE는 약5e-6 높다. 모든 차이를 의미 있는 개선이라고 세지 않는다. 따라서 “F가 latent에 아무 도움도 주지 않았다”는 해석도 지나치지만, 모든 지표에서 항상 낫다는 주장도 틀리다. 학습된 no-F forecaster와의 최적 비교가 아니므로 TSFM만의 고유 효과나 전체 시스템 우위를 입증하지 않는다. 그대로 재사용한 D의 transfer가 Raw F보다 나쁘다는 결과와 양립한다.

[확인] 초기 seed92601에 한정해 사전 고정한3개 origin `[0,N//2,N-1]`의 GT latent temporal std는 Robin median0.142262, std0 좌표-origin1/102; Jena median0.001151, std0 좌표-origin41/126이었다. Robin TS2 d16 canonical median은0.611035, std0은0/48이다. [추정] Jena GT seed92601의 일부 latent가 매우 정적이거나 ReLU 비활성 상태일 가능성은 있다. 잘 복원된 seed92602까지 같은 상태로 일반화하지 않는다. 전체 VAL/모든 채널의 collapse 검증이 아니며 자료·encoder마다 척도가 다르므로 median std로 모델 품질을 순위화하지 않는다.

## TS2: 정상 encoder가 자동으로 reconstruction·forecast 목표를 풀지는 않는다

[확인] Robin d16/d320 두 seed 모두 hidden64/depth10, 공식200 update와 SWA를 수행했고 새 d→128→C decoder는100 epoch·200 update, best epoch100이다. 두 seed에서 d320의 재구성 MSE 범위0.111495–0.135573은 d16의0.392856–0.400887보다 낮고 F transfer도0.616574–0.632825 대0.957489–0.980705로 낮다. [판단] 이 적용에서 확장 표현이 도움이 된 관찰은 있으나 d가 바뀌면 E의 마지막 layer와 D의 parameter 수도 함께 늘기 때문에 순수 출력 차원 효과로 분리한 ablation은 아니다. Raw F보다 좋은 미래 예측도 아니다.

[확인] Robin의 연속 TRAIN 입력은 `[1,1992,17]`로 max_train_length3000보다 짧다. 공식 fit은 instance 분할을 하지 않고 `batch_size=min(8,1)=1`을 사용한다. 수정 없는 `models/losses.py`는 B1의 instance contrastive loss를0으로 반환하므로 **Robin에서는 hierarchical temporal contrast 항만 비영**이다. 이것은 정상 hidden/depth를 축소했다는 뜻이나 알고리즘 구현 오류가 아니지만, 두 contrast 항이 모두 실제 학습에 기여했다고 쓰면 부정확하다. 자료 구성에 따른 적용 한계로 기록한다. [공식 losses 소스](https://github.com/zhihanyue/ts2vec/blob/b0088e14a99706c05451316dc6db8d3da9351163/models/losses.py).

[확인] Jena TS2 4조건 모두 **batch4 메모리 재실행**이다. hidden64/depth10, 해당 d16/d320, max_train_length3000, 600 update와 SWA601을 유지했고 각각300 epoch였다. 원형의 contiguous-series split/crop/mask/hierarchical contrast/AdamW/SWA 코드를 유지하지만 batch8→4는 한 update의 negative batch와 instance 노출을 바꾼다. Jena의8개 split instance를 한 batch로 보는 공식B8 설정과 정확히 같은 학습 조건이라고 보고하지 않는다. 같은600 update에서 B4의 instance 노출 횟수는 B8의 절반이고 crop마다 노출되는 시간점 수는 별도로 달라질 수 있다. 기본 B8 및 cache-only B8이 RSS 상한을 넘긴 시도는 보존했고 canonical 집계에서 제외한다.

[추정] TS2의 contextual consistency/hierarchical contrast는 reconstruction을 직접 최소화하지 않는다. frozen E가 예측 및 원공간 복원에 필요한 정보·척도를 모두 보존한다고 보장하지 않는다. 또 main 경로는 과거512-window offline 표현이고 native ridge는 원형 causal export의 마지막201점이므로 서로 representation의 context 규칙도 다르다.

```text
TS2 조건                         calibrated F MSE/MAE        native ridge MSE/MAE       alpha
Robin d16 s92601 epoch0_v2        0.980705 / 0.690139          1.206158 / 0.784927         500
Robin d16 s92602                 0.956809 / 0.726643          1.267705 / 0.828112          20
Robin d320 s92601                0.616504 / 0.544000          1.274310 / 0.797779        1000
Robin d320 s92602                0.632825 / 0.564094          1.329633 / 0.829577        1000
Jena d16 s92601 batch4            0.623235 / 0.491803          0.616260 / 0.489764         0.1
Jena d16 s92602 batch4            0.630853 / 0.492518          0.686431 / 0.538860         0.1
Jena d320 s92601 batch4           0.514303 / 0.406348          0.439495 / 0.352308         0.5
Jena d320 s92602 batch4           0.528330 / 0.416673          0.443775 / 0.356460         0.2
```

[확인] Robin은 네 조건 모두 native ridge보다 F 연결의 오차가 낮다. Jena d16은 seed92601에서 native ridge가 calibrated F보다 조금 낫지만 seed92602에서는 calibrated F가 두 지표 모두 낫다. 둘 다 Raw F보다 나쁘다. 반면 **Jena d320 native ridge는 두 seed에서 MSE0.439495·0.443775, MAE0.352308·0.356460으로 Raw0.554918/0.373891을 두 지표 모두 개선한다.** 이것은 확장 TS2 표현과 직접 supervised future head의 가능성을 보여 주지만 TSFM 연결의 성공이나 F가 불필요하다는 증거는 아니다.

[확인] native head는 Robin/Jena 각각1433/25648개 TRAIN origins(stride1), F-decoder calibration은12/201개(stride128)를 써서 supervised sample 수와 label window 노출이 일치하지 않는다. 중첩 origin/window 수는 독립 표본 수가 아니다. native는 causal last201 표현과 선형 multi-horizon ridge, main은 offline512 표현의 실제 F 출력과 시점별 비선형 D다. 개발 VAL은 같지만 export·readout·학습 노출이 함께 달라진 정상 사용 대조다. native 우월성을 F 연결만의 실패 원인으로 단정하거나 동일 용량·동일 표본의 head ablation이라고 보고하지 않는다. 후속으로 export와 supervised origins를 맞추면 이 차이를 분리할 수 있으나 이번 실행에는 추가 실험하지 않았다.

[확인] Robin d16 seed92601과 d320 seed92602는 epoch0 A checkpoint가 최선이었다. d16 seed92601은 실제20 update 후 TRAIN forecast entry MSE0.211066→0.203575, VAL macro MSE0.982802→1.000358로 악화되어 초기 A를 복원했다. d16 seed92602의 best epoch11은 MSE/MAE 모두 조금 줄었지만 d320 seed92601의 best epoch1은 MSE0.616574→0.616504와 MAE0.543643→0.544000의 교환관계다. calibration이 모든 자료·seed에 유리하다고 일반화하지 않는다.

[확인] Jena TS2 4조건은 같은 D를 실제 F TRAIN predictions로 적응시켜 transfer 대비 MSE·MAE를 모두 줄였다. d16은 두 seed에서 MSE0.870604→0.623235와0.833465→0.630853, d320은0.662064→0.514303과0.735020→0.528330이다. d16 seed92601과 d320 두 seed의 best는 epoch47, d16 seed92602는 상한인50이다. D reconstruction은4조건 모두100 epoch·2600 update를 실행했고 best는 d16 seed92601의99, 나머지100이다. 이 상한은 충분한 수렴을 보장하지 않는다. representation을 변경하거나 기존 선형 D8로 바꾼 결과도 아니다. 개선에는 **원공간 미래 정답 supervision**이 추가되므로 순수 비지도 재구성 decoder의 그대로 재사용이 성공했다고 부르면 안 된다. d320의 Raw 대비 MSE 이득에는 MAE 악화가 남는다.

최초 Robin pilot은 초기 후보를 빠뜨려 epoch1이 선택되는 구현 누락이 있었다. 원본 보존 후 새 canonical run에서 수정했으며 pilot0.971811과 canonical0.980705의 차이를 epoch0 수정의 성능 효과로 해석하지 않는다. 재학습한 E/D 자체가 달라진 비교다.

[확인] d16은 Robin scalar8704→8192로 약5.88%, Jena10752→8192로 약23.81% 감소한다. **d320은 encoder 출력512×320=163840**으로 Robin 약18.82배/Jena 약15.24배의 표현 확장이다. 정상 hidden/depth, 실제 scalar 압축, 비용 이득은 서로 다른 특성이다. d320이 잘 복원된다고 경량화에 성공한 것으로 표시하지 않는다. GT의128×C×2는 scalar 절반이지만 F행수 C→2C가 늘어나는 점도 함께 보고한다.

## 최종16조건에서 남는 판단

[판단] **비지도 재구성 D의 그대로 재사용은16조건 모두 Raw보다 나빴다.** 그러나 “방법 전체가 실패했다”로 묶을 수는 없다. Jena 확장 TS2 d320에서는 두 seed 모두 F 출력에 맞춘 D 적응이 Raw MSE를 개선하고, 직접 native ridge는 Raw MSE·MAE를 모두 개선한다. 전자는 MAE 악화, 후자는 F 연결과 다른 export·readout·label exposure를 갖는다. 따라서 현재 남는 연구 질문은 “표현을 압축하면 항상 좋아지는가”보다 “재구성용 표현과 D가 실제 F의 미래 출력 분포에 얼마나 맞고, 미래 supervision을 추가한 개선이 어떤 조건에서 나타나는가”에 가깝다. d320은 압축이 아니라 표현 확장이라는 사실을 함께 유지한다.

[판단] GT는 Jena에서 공간 항의 조건부 이득과 큰 seed/최적화 민감성을 보였다. Robin TS2는 짧은 단일 TRAIN instance 때문에 instance contrast 항이0인 적용이며, Jena는 메모리 상한에 따른 B4 학습이다. 이를 임의 자료의 원형 방법 전체에 대한 실패 판정으로 확장하지 않는다. 두 seed·두 자료와 반복 개발 VAL이므로 안정적인 일반 성능 개선이나 논문 기여가 확증됐다는 결론도 아니다.

canonical16조건에는 미완료 결과가 없다. 추가 실험 판단은 reconstruction 최적화·F 연결·D 적응 중 어떤 공백에 답하는지 명시하며 기존 LEVEL로 자동 복귀하지 않는다. 원형 GT의 공유 coupling은 arbitrary lag graph 발견과 다르고, D calibration과 ridge에는 미래 supervision이 추가되며, native 대비 F의 차이는 export·readout·label exposure를 맞춘 후에야 분리할 수 있다. 전체 결과는 반복 사용한 개발 VAL에 한정된다.

## 수치·출처 추적

이 메모에서 읽은 protocol snapshot SHA256은 `c49f2064629779f425003ca129034ef77705cfc92a0c87907bd0c7d29414bc53`이며, 그16개 IDs를 모두 포함했다. 완료 JSON SHA256:

```text
robin_GT_full_s92601                        e1c66e72496cda94edb9cdd404ee32ecdacc321bd532662b1fc2953130a23b21
robin_GT_full_s92602                        81ea80de6e2f3e79ebea178e725fade0ce8a1aa1b9cad8fcfd1f87b31d384f05
robin_GT_no_spatial_s92601                  e6d6aa06024f1f733d89b1ff01a8e3d67d807ea06b11f933b7fec10a9b7282fe
robin_GT_no_spatial_s92602                  3eb1884d606bdfe2ec42deb561a6a177d8ece8e8feceea63ca1b0af8cf449eb1
jena_GT_full_s92601                         1aa6dd1296879962b60bc48d6295218300480f3d3c75d5e6c246daff405bc4f9
jena_GT_full_s92602                         9401d9768d290942491cac90215211e11ecc08ac1377052cb64ec860267f725a
jena_GT_no_spatial_s92601                   249658734ef98ce43d108352caaf6bf11374f8dbf29c28bf70c08fd34a435c34
jena_GT_no_spatial_s92602                   7b99d6421633cb7962866b0dd17abdd410e11de03c7445d35d9261140be29ba5
robin_TS2_d16_s92601_retry_epoch0_v2         1fde898d9878113f9175ad3e95fd6f6e7bf64ec12c30ff851d2a9e717ef020de
robin_TS2_d16_s92602                        57c22468d615bc48f2dde0fef1c4ff3d0f98ca10466cc70e54a7c84688477735
robin_TS2_d320_s92601                       549bf157042ee71c147b6881553197f0aed9d183c594c59be04a61abb0bf62d4
robin_TS2_d320_s92602                       58fe279fb33b2371c29585f6ef7ea8b3f892a5b4ee7bed2be7d1f9ca231a4267
jena_TS2_d16_s92601_retry_batch4_v4          9782b15adc66a91ccc187f961f8767caf5e3b03cd559cdb840848fc1bf49c86f
jena_TS2_d16_s92602_retry_batch4_v4          728bf2377fa290ac59dad71901a2fcf745ebf77ff1a999bcb1b66b92d3c466ba
jena_TS2_d320_s92601_retry_batch4_v4         a5f07966f4abd51ef2efa267037a8335810d3f3f24c443373018f9f384cbe781
jena_TS2_d320_s92602_retry_batch4_v4         12048472f99ec66aa0f47d247249f7cce114960e80446ff8f67a46efdf202f57
```

canonical 결과 연결: [Robin GT full seed1](runs/robin_GT_full_s92601/result.json), [seed2](runs/robin_GT_full_s92602/result.json), [Robin no-spatial seed1](runs/robin_GT_no_spatial_s92601/result.json), [seed2](runs/robin_GT_no_spatial_s92602/result.json), [Jena GT full seed1](runs/jena_GT_full_s92601/result.json), [seed2](runs/jena_GT_full_s92602/result.json), [Jena no-spatial seed1](runs/jena_GT_no_spatial_s92601/result.json), [seed2](runs/jena_GT_no_spatial_s92602/result.json), [Robin TS2 d16 seed1](runs/robin_TS2_d16_s92601_retry_epoch0_v2/result.json), [seed2](runs/robin_TS2_d16_s92602/result.json), [Robin d320 seed1](runs/robin_TS2_d320_s92601/result.json), [seed2](runs/robin_TS2_d320_s92602/result.json), [Jena TS2 d16 seed1 batch4](runs/jena_TS2_d16_s92601_retry_batch4_v4/result.json), [seed2 batch4](runs/jena_TS2_d16_s92602_retry_batch4_v4/result.json), [Jena d320 seed1 batch4](runs/jena_TS2_d320_s92601_retry_batch4_v4/result.json), [seed2 batch4](runs/jena_TS2_d320_s92602_retry_batch4_v4/result.json).

GT curve는 각 run의 reconstruction/calibration JSON, TS2 curve는 result의 `training.decoder_reconstruction.learning_curve`와 `calibration.learning_curve`가 지정한 cache를 읽었다. 초기 진단에서는 GT checkpoint를 CPU `torch.load(..., map_location='cpu', weights_only=True)`로 확인하고 latent는 사전에 고정한 origin의 diagnostic NPZ를 읽었다. 이번 갱신은 canonical16개 JSON과 이미 읽은 GT8개의 curve, TS2 학습 metadata 및 수정 없는 loss/fit 소스로 수치·batch 적용 범위를 대조했다. GPU 작업·보호된 TEST 읽기·실제 run/weight 수정은 모두0이다.

구현 근거: [GT 변경 기록](GT_METHOD_DELTA.md), [TS2 변경 기록](TS2_METHOD_DELTA.md), [GT model](models_gt.py), [TRAIN-only graph](data_io.py), [정상 TS2 adapter](ts2_adapter.py). GT 논문 기반 재구현과 TS2 공식 소스 재사용을 구분하며, 이번 결과를 원논문의 benchmark 재현으로 표시하지 않는다.
