# MAIN 잔차 보정: 저장된 오차의 기여 분해

선택된 예측을 변경하지 않고 같은 실제 Y 관측 마스크로 오차 차이를 분해했다. 음수 기여는 해당 조건의 전체 정규화 채널 macro 오차 개선이다. 각 채널·구간·오차 크기별 값을 다시 정규화하지 않았으므로 각 분해의 합이 전체 점수 차이와 같다.

여러 겹치는 창과 두 seed를 독립 표본으로 간주하지 않았다. FIT/TUNE 및 반복 노출된 VAL/EVAL 진단이며, 신뢰구간·유의성·새 모델 선택·일반화 증거를 제공하지 않는다. MSE 개선과 MAE 악화가 함께 있으면 그 교환을 보존해 해석해야 한다.

윈도우 블록은 입력 순서의 `floor(4*i/N)`이다. 오차 구간은 모든 비교에서 원래 MAIN의 정규화 절대 오차 `[0,.25,.5,1,2,inf]`로 고정했다. E33K와 MEAN 비교도 같은 구간을 쓴다.

- [채널 기여](CHANNEL_CONTRIBUTIONS.csv)
- [윈도우 블록 기여](BLOCK_CONTRIBUTIONS.csv)
- [MAIN 오차 크기별 기여](ERROR_BINS.csv)
- [기존 TUNE 선택 여유](TUNE_MARGINS.csv)
- [입력 SHA·검산·자원 기록](MANIFEST.json)

```text
dataset seed split method/reference signed_MSE_delta signed_MAE_delta
robin 92601 val MEAN_CORRECTION/ORIGINAL_MAIN 0.000790385276218 -0.000757256240577
robin 92601 val RAW33C/ORIGINAL_MAIN 0 0
robin 92601 val E33K/ORIGINAL_MAIN 0 0
robin 92601 val E512K/ORIGINAL_MAIN 0 0
robin 92601 val E33K/MEAN_CORRECTION -0.000790385276218 0.000757256240577
robin 92602 val MEAN_CORRECTION/ORIGINAL_MAIN 0.00120288573456 3.40318430592e-05
robin 92602 val RAW33C/ORIGINAL_MAIN 0 0
robin 92602 val E33K/ORIGINAL_MAIN 0 0
robin 92602 val E512K/ORIGINAL_MAIN 0 0
robin 92602 val E33K/MEAN_CORRECTION -0.00120288573456 -3.40318430592e-05
beijing_aotizhongxin 92601 eval MEAN_CORRECTION/ORIGINAL_MAIN -0.00232595646223 0.00533132279016
beijing_aotizhongxin 92601 eval RAW33C/ORIGINAL_MAIN -0.00877628402006 0.00540095685903
beijing_aotizhongxin 92601 eval E33K/ORIGINAL_MAIN -0.00582466468431 0.0042610925384
beijing_aotizhongxin 92601 eval E512K/ORIGINAL_MAIN 0 0
beijing_aotizhongxin 92601 eval E33K/MEAN_CORRECTION -0.00349870822208 -0.00107023025176
beijing_aotizhongxin 92602 eval MEAN_CORRECTION/ORIGINAL_MAIN 0.00400065491472 0.00680566572538
beijing_aotizhongxin 92602 eval RAW33C/ORIGINAL_MAIN -0.00578796924141 0.00527958161195
beijing_aotizhongxin 92602 eval E33K/ORIGINAL_MAIN -0.00296839823388 0.00891520206749
beijing_aotizhongxin 92602 eval E512K/ORIGINAL_MAIN -0.00138924278504 0.00494233340944
beijing_aotizhongxin 92602 eval E33K/MEAN_CORRECTION -0.0069690531486 0.0021095363421
```

ZERO 선택은 기존 MAIN을 보존한 것이므로 새 잔차 학습의 개선으로 해석할 수 없다. E33K와 공유 FIT 평균 보정을 비교하면 과거 K 정보의 추가 이득과 단순 편향 보정 효과를 구분하는 데 도움이 되지만, 이 분해 자체가 인코더·예측기·디코더 중 단일 원인을 확정하지는 않는다.
