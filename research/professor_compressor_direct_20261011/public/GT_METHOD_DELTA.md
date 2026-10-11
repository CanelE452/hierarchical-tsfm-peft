> 게시용 사본입니다. 실험 완료 당시 원본 문서의 SHA256은 `9d7fdce1100a27f7238f120ffc5661fc821ea80e504065ee5bc28413348cc1a8`입니다. 본문 수치·판단을 유지하고 링크만 GitHub 경로로 조정했습니다. 원문의 “이번 작업에서 push하지 않음”은 과학 실험 완료 당시 기록이며, 후속 사용자 승인으로 이 공개 사본을 게시합니다. 구 TEST 사고는 [범위·원본 SHA](PROTECTED_ACCESS_CONTEXT.json)만 공개하며 보호된 통계는 재게시하지 않습니다.

# GTConvAE: 원형의 시간 압축과 native decoder를 사용하는 재구현

[확인] 이번 구현은 **논문 기반 재구현**이다. 공식 구현이나 원논문 benchmark 수치 재현으로 표시하지 않는다. 원문은 [PMLR 논문](https://proceedings.mlr.press/v198/sabbaqi22a.html)과 [PDF](https://proceedings.mlr.press/v198/sabbaqi22a/sabbaqi22a.pdf), 사용자가 제공한 `sabbaqi22a.pdf`다. Figure 1, Section 2.1–2.4, Section 4.1–4.4, Appendix B/C를 확인했다.

PMLR의 Related Material은 PDF와 OpenReview를 연결한다. OpenReview 본문은 이번 조회에서 browser-verification 페이지로 전환됐으며 PDF는 확인됐다. 논문 제목 및 저자와 연결한 GitHub/OpenReview/PMLR 검색에서 GTConvAE 공식 소스는 확보하지 못했다. 이는 확인한 범위의 결과이며 코드가 존재하지 않는다는 단정이 아니다. 관련 GTCNN/STVNN 코드를 GTConvAE의 공식 소스로 대신 표시하지 않았다. 탐색은 요청한 20분 상한 안에서 마쳤다.

## 원문에서 유지한 연산

`models_gt.py`의 product shift는 다음 Eq. (1)을 직접 계산한다.

$$S_\diamond=s_{00}I_T\otimes I_C+s_{01}I_T\otimes S_C+s_{10}S_T\otimes I_C+s_{11}S_T\otimes S_C.$$

원문의 Eq. (2)는 $\sum_{k=0}^{K}h_kS_\diamond^k$다. 따라서 **order 4는 identity를 포함한 5 taps**다. encoder/decoder 양쪽에 Eq. (3)의 feature-pair filter bank를 사용하며, 공간 노드 C를 유지한다. 시간 그래프는 원문 실험과 같은 directed line 계열이며 여기서는 과거에서 현재로 전달하고 시작 경계를 zero-pad한다. circular wrap은 없다.

encoder feature는 `1→8→2`, decoder는 `2→8→1`이다. 각 encoder 단계는 `GT filter→max block pooling(r2)→ReLU`다. 물 자료 실험 Section 4.2의 feature 수, max summary, order 4, rho 0.14, Adam 학습률 5e-4를 출발점으로 삼았다. Robin/Jena 및 L512에 최적인 것으로 확인한 값은 아니다. 시간 길이는 `512→256→128`이고 scalar 수는 `512C→128C×2=256C`로 절반이다.

```text
부분                 filter shape             학습 계수 수
encoder 첫 단계      [5, 1, 8]                40
encoder 둘째 단계    [5, 8, 2]                80
decoder 첫 단계      [5, 2, 8]                80
decoder 둘째 단계    [5, 8, 1]                40
공유 product s       [s00,s01,s10,s11]         4
합계                                          244
```

모든 feature-bank filter에는 원문 수식에 없는 bias를 추가하지 않았다. TRAIN graph C×C는 고정 buffer이며 model parameter 수에 포함하지 않는다. 공간 제거 대조는 **s01과 s11 모두 literal zero buffer**이고 이 두 값에는 gradient/optimizer update가 없다. s00/s10은 학습한다. 따라서 공간 제거의 총 학습 계수는 242다. 전체 모델과 이 대조는 같은 feature·시간 압축·decoder 구조로 별도 재학습한다.

## 원문 공백과 이번 적용 선택

Figure 1(B)는 decoder의 upsampling 뒤 GT filter를 보여주지만, Section 2.3의 본문과 식 순서는 filter 뒤 upsampling이다. [추정·재현 해석] 요청 문서에 따라 **zero insertion→GT filter**를 사용했다. 원문에 유일한 해석으로 주장하지 않는다. 각 hidden decoder 단계는 ReLU, 마지막 출력은 **linear**로 바꿨다. 이는 음수를 포함하는 signed standardized Robin/Jena의 데이터 적용 변경이다.

Eq. (8)의 one-based `t=kr`를 따르므로 r2에서 latent 값을 zero-based `1,3,5,…`에 삽입한다. 이 위치와 past→present filter를 조합하면 decoder를 고립된 latent만으로 시작할 때 앞의 **3개 원시점은 구조적으로 0**이다. 이 startup 경계는 숨기거나 관측 정답을 0으로 clipping하지 않는다. 일반 재구성 지표에는 해당 관측 시점도 들어가며 모델의 한계다. 마지막 upsampling 뒤 filter가 실제로 존재하므로 내부 시점이나 마지막 출력의 모든 alternate phase가 고정 0으로 남는 구조는 아니다.

product 계수의 초기값은 원문에서 확인하지 못했다. [추정·적용 선택] `s=(0,.5,.5,0)`으로 시작해, spectral-normalized spatial graph와 temporal shift를 합친 초기 연산 크기를 안정적으로 둔다. 이 scale은 학습 제약이 아니며 product 전체를 forward마다 다시 정규화하지 않는다. 필터는 평균 0, 표준편차 $\sqrt{2/((K+1)F_{in})}$의 normal 초기화다. random seed는 runner가 모델 생성 직전에 설정한다.

재구성 학습의 squared error는 이번 자료의 실제 observed mask로 평균한다. 여기에 **raw product coefficient L1 sum**을 rho로 곱한다. loss를 관측 항목 평균으로 바꿨으므로 rho 0.14의 상대 강도는 원문 sum/expectation 표기와 같은 것으로 자동 간주하지 않는다. latent 자기예측 loss, PCA, D8, LEVEL G를 이 모델에 추가하지 않았다.

## TSFM 연결과 시간 대응

모델 API는 `GTConvAE(graph,no_spatial=False)` / `encode` / `decode` / `decode_future` / `sparsity_penalty`다. `encode([B,512,C])`가 `[B,128,C,2]`를 만든다. encoder의 latent tick j는 원시점 block `[4j,4j+3]`를 요약하며 representative timestamp는 오른쪽 끝 `4j+3`이다. filtered feature는 더 이전 시점의 정보를 포함할 수 있다.

```text
원시 과거 block       history latent tick     원시 미래 block     future latent tick
0..3                  0                       512..515            0
4..7                  1                       516..519            1
…                     …                       …                   …
508..511              127                     556..559            11
```

F 입력은 `[B*C*2,128]`, latent 미래는 `[B,12,C,2]`다. node×feature는 시계열 배치 행이고 시간 좌표를 feature 좌표와 이어 붙이지 않는다. raw C행보다 F행이 두 배가 되므로 비용 이득을 자동 주장하지 않는다.

`decode_future(z_history,z_predicted_future,horizon=48)`는 **관측 history latent와 F가 예측한 latent**를 연결한 후 같은 native GT decoder를 호출하고 physical indices `[512,560)`를 반환한다. 미래 실제 관측을 encoder에 넣어 inference feature로 사용하지 않는다. decoder가 causal이므로 추가적인 실제 미래 boundary state가 필요 없다. history를 이어서 시작부 3개 structural zeros가 미래 crop에 들어오지 않게 한다. 다만 right-edge phase 때문에 미래 block의 앞부분은 이전 latent tick의 decoder state 영향을 받는다. zero-phase interpolation 또는 lossless inverse로 표시하지 않는다.

H가 R로 나뉘지 않으면 `ceil(H/4)`개의 latent blocks를 F가 모두 예측한 뒤 H개만 crop한다. target padding은 없다. encoder는 complete four-point blocks를 요구하므로 이번 L512에서는 경계가 정확히 맞는다. 다른 L을 적용할 때는 관측 context의 trimming/phase 규칙을 먼저 선언해야 한다.

calibration A→B에서 원문처럼 E/D가 공유하는 product s는 encoder와 함께 고정한다. `encoder_parameters()`는 E의 120 filters와 공유 s를, `decoder_parameters()`는 native D의 **120 polynomial filters만** 반환한다. B는 바로 이 D 가중치를 적응시키고 기존 D8/ridge로 우회하지 않는다. A checkpoint 보존·실제 F TRAIN 예측·추가 감독·VAL 선택·비용은 runner가 별도 기록한다.

## CPU 구현 진단 결과

[확인] [GT_CPU_VERIFICATION_1791721824393702500.json](../GT_CPU_VERIFICATION_1791721824393702500.json)은 exit0 / 27 PASS다. degree4 dense Kronecker 출력 오차는 `2.22e-16`, input/filter/product gradient 오차 최대 `8.88e-16`이었다. spatial edge 영향과 graph 변경, 공간 제거 불변성, delta/constant/pooling/upsampling phase, `12→48`, nonmultiple-H crop, history boundary를 검사했다.

signed tiny synthetic batch 600 update에서는 full MSE `0.662626→0.063811`로 감소했다. 실제 encoder·native decoder·네 product 계수에 nonzero gradient와 weight update가 존재했다. 이 진단은 LR .01/rho 0이며 실자료의 예정 Adam5e-4/rho .14 학습과 구분한다. 같은 native decoder 1 update calibration은 E/shared-s byte 불변, D 양쪽 GT filter 호출·변경을 확인했고, save/load prediction은 bitwise 같았다. 총 synthetic optimizer steps는 601이다.

자원은 wall 8.267s, CPU 13.625s, peak RSS 601,272,320 bytes다. CUDA 초기화/실자료 읽기/보호 TEST 읽기/실제 science fit은 모두 0이다. 이 진단은 **실제 frozen F 미래 추론이나 Robin/Jena 성능의 증거가 아니다**. 원문 stability theorem의 spectral/Lipschitz 조건을 강제로 만족시켰다고 주장하지 않는다. 실제 학습·F 연결·예측 수치·곡선은 runner와 최종 보고서에서 확인한다.
