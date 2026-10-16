# 원단위 보조 오차와 LR 대응 조건

`physical_metrics.generate(public_root, cache_root, split="V_select")`는 기존의 검산된 예측 NPZ와 **이미 고정된 canonical A60 scaler**만으로 보조 점수를 계산한다. 원 Task·새 target window를 읽거나 scaler를 재적합하지 않는다. 주지표인 A60 정규화 관측 채널 macro MSE와 기존 점수표는 변경하지 않는다.

```powershell
python -X utf8 physical_metrics.py --public-root . --cache-root $env:INFORMS_CACHE_ROOT --split V_select
python -X utf8 -m unittest test_channel_patterns test_time_block_analysis test_physical_metrics -v
```

T는 선택 잠금과 승인된 route/artifact가 있고 **이미 저장된 T NPZ**가 생긴 뒤에만 `--split T_final`로 처리한다. 잠금 검사를 normalization·target cache 접근보다 먼저 수행한다. 새 T 추론이나 후보 선택은 이 API에 없다.

## 입력과 검산

공개 registry·DATA_CONTRACT, 예측 manifest·독립 reference·공개 충분통계, normalization artifact의 상대 경로와 SHA를 `INPUT_SNAPSHOT.json`에 먼저 봉인한다. 각 예측의 source/config/data/model/scaler와 고정 다섯 배열의 fingerprint, 정답·관측 mask·원점·채널을 검사한다. 기존 독립 채점기로 normalized 점수와 공개 SSE/SAE/count도 다시 대조한다. epoch endpoints·선택 CP/업데이트·LR·seed·ID·fit config와 prediction config를 따로 보존한다.

scaler는 동일 자료·fold·canonical A60 끝·data SHA·scaler SHA를 가진 기존 ridge NPZ 또는 작은 selected bundle의 mean/std에서만 가져온다. artifact SHA를 확인하고 공개 DATA_CONTRACT의 A_stop 끝과 대조한다. 작은 bundle을 CPU로 역직렬화할 수 있으나 F를 생성하거나 실행하지 않는다. 일치하는 scaler가 없으면 해당 경로만 제외하고 원 Task로 우회하지 않는다.

## 수식과 단위

관측 위치의 FP64 오차는 `e_phys=(pred_normalized-target_normalized)*std_c`이다. 평균은 상쇄되지만 mean/std 전체의 scaler SHA를 검증한다.

- 채널별 SSE·SAE·count와 MSE·MAE·RMSE를 보존한다. 관측수0 채널의 점수는 NA이다.
- macro MSE/MAE는 각 active 채널의 `SSE/count`, `SAE/count`를 균등 평균한다.
- `physical_macro_rmse=sqrt(physical_macro_mse)`이다. `physical_mean_channel_rmse`는 채널 RMSE의 평균으로 별도 보고한다.
- pooled MSE/MAE는 모든 관측 SSE/SAE의 합을 전체 관측수로 나눈다. pooled RMSE는 pooled MSE의 제곱근이다. 결측 때문에 macro와 pooled는 다를 수 있다.

PeMS는 원 feature0의 **flow counts, 5분 sample** 단위이며 시간당 교통량으로 변환하지 않는다. 타임스탬프가 없는 원 자료에 실제 달력 1970을 부여하지 않는다. Beijing/Jena/Robin의 서로 다른 물리 단위는 채널별로 우선 해석하며, 이들의 원단위 macro를 다른 자료나 normalized 주지표와 같은 의미로 비교하지 않는다. legacy FP32 정규화·역변환 계보는 기존 제한으로 남는다.

출력은 매번 새 `diagnostics/physical_metrics_<ns>/`에 생성한다. `PHYSICAL_SCORES.csv`, `PHYSICAL_CHANNEL_ERRORS.csv.gz`, 입력 봉인, source/output SHA·제외 사유·CPU 시간의 manifest를 제공한다. 공개되는 것은 집계 오차뿐이며 실제 정답·예측·원자료 배열은 게시하지 않는다.

## LR 대응 오류 수정 범위

이전 `channel_patterns.py`와 `time_block_analysis.py`는 같은 epoch/seed/K여도 서로 다른 LR를 대응시킬 수 있었다. 기존 두 소스와 테스트는 `source_snapshots/*_before_lr_pairing_<SHA>.py`에 그대로 보존했다. 기존 결과 snapshot도 수정하지 않는다.

현재 `training_identity`는 base/control의 `SOURCE_CONTRACT.spec.lr`와 config SHA, 조건부 H의 실제 `receipt.learning_rate`와 contract config SHA를 읽는다. 두 경로가 optimizer 학습 경로이면 LR가 정확히 같아야 paired에 포함한다. 학습 LR 또는 fit config가 미확인이면 paired 비교에서 제외하며, 그 경로의 개별 group 오차 자체를 지우지는 않는다. Raw·현재값·closed-form alpha/PCA 등의 비학습 기준선에는 LR를 강요하지 않는다. B0/B2가 CP0를 선택했더라도 학습 recipe를 탐색한 경로이므로 LR 대조 조건을 유지한다.

같은 frozen A 패턴 표로 다시 평가할 때는 반드시 `--pattern-snapshot`을 명시한다. 표를 새로 적합하지 않는다. 새 CSV는 left/right LR·fit config를 명시하며, 원 registry·학습 결과·예측 수치를 바꾸지 않는다. 이 수정은 같은 LR의 관측 비교를 보장하는 한 부분이며 최적화 예산·추가 입력 정보까지 자동으로 동등하게 만들지는 않는다.

CPU 테스트에서 잘못된 LR 쌍을 이전 코드가 허용하는 RED를 확인했다. 이후 같은 LR 허용, 다른 LR·미확인 학습 LR 거부, 실제 toy 예측 cache의 native-unit 계산, macro/pooled 차이, 잘못된 scale 거부, T 잠금 전 거부, Task/target 새 접근 금지를 확인했다.
