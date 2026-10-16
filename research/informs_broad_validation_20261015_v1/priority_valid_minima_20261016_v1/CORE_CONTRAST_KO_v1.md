# Beijing: 같은 K의 B2 추가 가치와 full-C 압축 경계

먼저 두 질문에 답한다. 아래 차이는 같은 반복 개발 V43/seed92601/CPU2에서 관찰한 원수치의 조건부 비교다. normalized channel-macro MSE가 주지표이며 MAE·전체 비용도 함께 판단한다. 성능에 따라 지표·gate·학습 길이를 바꾸지 않았다.

1. **B0 K6에 B2 앞단을 더할 근거가 있는가?** B2 K6−B0 K6의 MSE 차이는 `0.057459724032271398`, MAE 차이는 `0.016112403800580766`다. B1 whole median 차이는 `0.00022940011695027351`초, B8은 `-0.0070954998955130577`초다. 음수는 해당 오차/비용이 작다는 뜻이다. 이 값과 선택 refit·노출량을 함께 보아 이 개발 조건에서의 추가 가치를 판단하며, 독립적인 일반적 우위를 선언하지 않는다.
2. **B2를 C11에서 K6으로 줄일 때 정확도와 전체 비용이 어떻게 달라지는가?** K6−K11의 MSE 차이는 `0.075369477294669474`, MAE 차이는 `0.029108332832976758`이고 B1/B8 whole median 차이는 `-0.029959049774333835`/`-0.24647929985076189`초다. K6 PCA 초기화와 K11 identity 초기화·표현/앞단·파라미터 용량·선택 학습량의 차이가 함께 있으므로 순수 정보 압축 손실이나 encoder만의 효과로 확대하지 않는다.

[원정밀도 CSV](CORE_CONTRAST_RESULTS_v1.csv) · [입력 SHA와 실제 감사 결속](CORE_CONTRAST_BINDINGS_v1.json) · [기존 N4 보고](RESULTS_KO_v1.md)

```text
role     MSE                 MAE                 K reused
B0_K6    0.79796643709754878 0.59313931444669032 6 False
B2_K6    0.85542616112982017 0.60925171824727109 6 True
B2_K11   0.7800566838351507 0.58014338541429433 11 False
```

B0도 B2도 **frozen Bolt**를 실제 사용하는 경로다. 여기서는 noBolt를 새로 학습·비교하지 않았고, 같은 표현에 예측기만 교체한 실험도 아니다. 고정 PCA와 동일 D 효과를 분리한 비교로 부르지 않는다. 기존 N4 B2 K6의 전체 V·FP64·CPU6·두 비용은 exact audit/report/manifest SHA로 재사용했다. 새 B0 K6와 B2 K11만 실제 새 fit/fullV/독립채점/replay/4비용 증거를 연결했다.

```text
role     selected-refit search/refit-updates search/refit-exposures search/refit-seconds training-status
B0_K6    3.0 1868/2262 29880/36189 3575.1789815998636/4027.7962171998806 UNDERTRAINED_OR_BUDGET_LIMIT
B2_K6    1.0 1868/754 29880/12063 3372.8632249999791/1240.2778059002012 SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED
B2_K11   0.25 1868/189 29880/3016 6419.7077394998632/629.8730337000452 SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED
```

세 조건은 C11/L512/H48, A50[0,10519), A_stop[10519,12622), fresh A60[0,12622), V[14726,16830)의43 origins·20539 observed labels·원 scaler/target/mask/origin/channel 지문을 공유한다. 같은20539 labels를 반복 사용한 것이며 세 개의 독립 검증 표본이 아니다. 검색0/.25/1/3은 같아도 A_stop 선택 refit·실제 update/exposure·구조/모수·소요시간이 다를 수 있다. 한 seed·제한된3 검색·반복 개발 V이며 충분한 수렴과 독립 일반화는 미확정이다.

```text
role     B1 median/p90 seconds  B8 median/p90 seconds  CPU weights MiB B1/B8  observed-tree RSS bytes B1/B8
B0_K6    0.04792475001886487/0.049696689844131468 0.33349604997783899/0.3387223798315972 182.03187561035156/182.03187561035156 1004212224/1065009152
B2_K6    0.048154150135815144/0.050563150225207208 0.32640055008232594/0.34342451966367671 182.16180038452148/182.16180038452148 1009913856/1068437504
B2_K11   0.078113199910148978/0.079776540119200948 0.57287984993308783/0.58785454016178851 182.16347885131836/182.16347885131836 1028288512/1118081024
```

비용은 CPU2 물리 history/mask→채움/정규화→P/E/F/D→FP64 inverse/CPU 반환의 실제 한 batch 전체다. B8을8로 나눈 값이 아니고 K/C를 속도비로 쓰지 않는다. 각50trial은 warm10·ordinary timed20·별도 intrusive stage20이며 표는 ordinary whole median/p90다. stage 합=whole을 주장하지 않는다. F instances와 실제 stream/call은 CSV에 따로 기록했다. CPU weights는 F와 adapter의 합이며 F만의 resident bytes는 원 측정에서 분리하지 않아 NA/NOT_SEPARATELY_RECORDED다. CPU resident weights·관측된 tree RSS·GPU peak·로딩·학습 peak를 같은 메모리 범위로 섞지 않는다.

새2조건의 공개 충분통계2838행과 비용200trial/8summary/40지연시간 필드를 원 정책으로 감사했다. 기존 FP64 재채점·freshCPU6와 actual selected artifact/scaler/source 및 강한 observed process-tree closure/quiet 영수증을 결속했으며 이 builder가 다시 모델·NPZ·비용을 실행한 것은 아니다. 허용오차는 원 MSE/MAE atol1e-9/rtol1e-7, cost 요약은 기존 exact 정책이다. 검사 통과는 성능 개선·인과성·충분 수렴·일반화의 증거가 아니다.

**전체 MASTER 미완료**다. 다른 K/자료·C subset·seed·충분 학습·방법군·공유 multi-K 등 원 공백이 남는다. 보호 T를 새로 평가하거나 사용해 선택하지 않았으며 현재 비교의 실패도 그대로 보고한다. 다음 판단은 두 질문의 오차·전체비용·용량/학습량 혼재를 함께 검토하는 것이다. 이 보고에서 주방법을 자동 선택하거나 일반적인 Bolt 필요성을 확정하지 않는다.
