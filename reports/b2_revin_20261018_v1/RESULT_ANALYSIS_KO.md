# 결과 분석

V는 반복 개발 평가로 이미 사용한 자료이며 독립 테스트 결과가 아니다. 새 선택에 M_cal/T를 사용하지 않았으며, 기존 T 노출과 pretraining overlap UNKNOWN의 범위는 원 계약을 따른다. 두 seed·고정 K의 결과를 모든 데이터·압축률의 일반화로 확대하지 않는다.

주지표는 원채널 균등 macro MSE, 보조지표는 같은 공간의 macro MAE다. global A-fit/A60 z-score 단위이며 시계열을 채널 전체 pooled MSE로 바꾸지 않았다.

## 정확도와 실제 비용

```text
jena     B0_REVIN_OFF s92601 cap3/e3.0          MSE=0.7840299 MAE=0.52093171
jena     B0_REVIN_OFF s92602 cap3/e3.0          MSE=0.75509456 MAE=0.50258911
jena     B2_REVIN_OFF s92601 cap3/e1.0          MSE=0.68640186 MAE=0.49179861
jena     B2_REVIN_OFF s92602 cap3/e1.0          MSE=0.67821752 MAE=0.49886588
jena     B2_REVIN_ON s92601 cap3/e1.0           MSE=0.66223269 MAE=0.46705254
jena     B2_REVIN_ON s92602 cap3/e1.0           MSE=0.71455247 MAE=0.48076061
pems32   B2_REVIN_OFF s92601 cap6/e6.0          MSE=0.090347662 MAE=0.19895994
pems32   B2_REVIN_OFF s92602 cap6/e6.0          MSE=0.091469785 MAE=0.20069531
pems32   B2_REVIN_ON s92601 cap6/e5.0           MSE=0.090114126 MAE=0.1989611
pems32   B2_REVIN_ON s92602 cap6/e6.0           MSE=0.089050779 MAE=0.19682284
jena     LoRA_GPU_replica s92601 cap0/e1.0      MSE=0.56392295 MAE=0.39900994
jena     Raw s92601 cap0/eNone                  MSE=0.84919414 MAE=0.48138834
jena     Ridge s92601 cap0/eNone                MSE=0.55664634 MAE=0.4055218
pems32   DirectG s92601 cap0/e3.0               MSE=0.081465373 MAE=0.18577498
pems32   Raw s92601 cap0/eNone                  MSE=0.09051379 MAE=0.19521149
```

비용은 정확한 accuracy_run_id와 연결했다. CPU/GPU와 B1/B8은 따로 읽는다. 미측정·검증 실패 비용 0행은 숫자를 채우지 않았다. 원자료+mask→fill→global norm→RevIN/P/E/F/D→inverse→CPU 출력의 전체 경로이며 고정 특징 캐시의 읽기 속도를 추론 비용으로 쓰지 않는다. 같은 방법명이 다른 seed·부모·장치의 정확도/비용을 연결하는 근거가 되지 않는다.

## RevIN 대응 결과

jena cap3 seed92601: OFF MSE 0.68640186 → ON 0.66223269 (ON−OFF -0.02416917; -3.5211399%); MAE 0.49179861 → 0.46705254 (-5.0317486%). 선택 epoch OFF/ON은 1/1이다.

jena cap3 seed92602: OFF MSE 0.67821752 → ON 0.71455247 (ON−OFF 0.036334949; 5.3574182%); MAE 0.49886588 → 0.48076061 (-3.6292858%). 선택 epoch OFF/ON은 1/1이다.

pems32 cap3 seed92601: OFF MSE 0.090375036 → ON 0.090780967 (ON−OFF 0.00040593136; 0.44916315%); MAE 0.19860356 → 0.19801891 (-0.29437752%). 선택 epoch OFF/ON은 3/3이다.

pems32 cap3 seed92602: OFF MSE 0.09126615 → ON 0.089896028 (ON−OFF -0.0013701223; -1.5012382%); MAE 0.1995835 → 0.19716725 (-1.2106485%). 선택 epoch OFF/ON은 3/3이다.

pems32 cap6 seed92601: OFF MSE 0.090347662 → ON 0.090114126 (ON−OFF -0.00023353602; -0.25848596%); MAE 0.19895994 → 0.1989611 (0.00058411749%). 선택 epoch OFF/ON은 6/5이다.

pems32 cap6 seed92602: OFF MSE 0.091469785 → ON 0.089050779 (ON−OFF -0.0024190064; -2.6445962%); MAE 0.20069531 → 0.19682284 (-1.9295266%). 선택 epoch OFF/ON은 6/6이다.

cap3와 조건부 cap6를 각각 보존한다. ON만 더 학습한 비교가 아니며, 두 seed의 역방향 결과도 남긴다. 선택-refit 결과는 같은 선택 recipe의 비교이며 선택 epoch가 다르면 동일 업데이트 효과로 해석하지 않는다. origin별 대응 값은 PAIR_ORIGIN_DIFFS.csv에 있으며 IID 유의성이나 두 seed의 신뢰구간을 새로 주장하지 않는다.

## 구성 기여와 적용 한계

Jena seed92601 cap3: B2_jena_REVIN_OFF_s92601_cap3 − B0_jena_REVIN_OFF_s92601_cap3의 MSE 차이는 -0.097628034, MAE 차이는 -0.029133097다. 대응 상태는 MATCHED이며 구성 기여와 전체 구조의 우열을 구분한다.

Jena seed92602 cap3: B2_jena_REVIN_OFF_s92602_cap3 − B0_jena_REVIN_OFF_s92602_cap3의 MSE 차이는 -0.076877034, MAE 차이는 -0.0037232293다. 대응 상태는 MATCHED이며 구성 기여와 전체 구조의 우열을 구분한다.

SOFTS, FreTS, LIFT, N-HiTS, CycleNet/TimesNet, ForeCA, 전체 AdaPTS 비교와 추가 K·초기화 탐색은 제외했다. PeMS32 D는 이번 최종 발표 근거로 쓰지 않아 seed2 보완 조건이 발동하지 않는다.

Jena B0는 최종 B2의 정규화·학습 상한에 맞춘 구성 대조다. B0도 상한에서 개선 중이면 학습량 부족 가능성을 남기며, 이번 상한의 결과로 수렴이나 모든 학습 예산에서의 우위를 주장하지 않는다.

## 학습 상한

B0_REVIN_OFF s92601 cap3/e3.0: 상한에서도 개선 중; 수렴 여부 미확정
B0_REVIN_OFF s92602 cap3/e3.0: 상한에서도 개선 중; 수렴 여부 미확정
B2_REVIN_OFF s92601 cap3/e1.0: 등록한 선택 구간 완료; 수렴 여부 미확정
B2_REVIN_OFF s92602 cap3/e1.0: 등록한 선택 구간 완료; 수렴 여부 미확정
B2_REVIN_ON s92601 cap3/e1.0: 등록한 선택 구간 완료; 수렴 여부 미확정
B2_REVIN_ON s92602 cap3/e1.0: 등록한 선택 구간 완료; 수렴 여부 미확정
B2_REVIN_OFF s92601 cap6/e6.0: 상한에서도 개선 중; 수렴 여부 미확정
B2_REVIN_OFF s92602 cap6/e6.0: 상한에서도 개선 중; 수렴 여부 미확정
B2_REVIN_ON s92601 cap6/e5.0: 등록한 선택 구간 완료; 수렴 여부 미확정
B2_REVIN_ON s92602 cap6/e6.0: 상한에서도 개선 중; 수렴 여부 미확정
LoRA_GPU_replica s92601 cap0/e1.0: 등록한 선택 구간 완료; 수렴 여부 미확정
Raw s92601 cap0/eNone: 등록한 선택 구간 완료; 수렴 여부 미확정
Ridge s92601 cap0/eNone: 등록한 선택 구간 완료; 수렴 여부 미확정
DirectG s92601 cap0/e3.0: 등록한 선택 구간 완료; 수렴 여부 미확정
Raw s92601 cap0/eNone: 등록한 선택 구간 완료; 수렴 여부 미확정

상한 도달을 방법 실패나 수렴 완료로 바꿔 부르지 않는다. 선택과 데이터 단위 확장·공통 정규화는 A_stop만으로 고정하고, 각 선택 epoch에서 fresh A60 refit했다.
