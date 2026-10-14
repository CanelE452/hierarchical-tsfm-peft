# MAIN 예측 보존 보정 실험의 실행 전 고정 계약

이 문서는 [실행 계획](PLAN_MAIN_FORECAST_RESIDUAL_KO_v1.md)의 산술·비교·감사 계약을 고정한다. 사용자 응답 `call_c2e3607e77684a40bbd8e6a06612d9b9:0`의 “하라고 … main에 push하고 쭉하라고”를 해당 제안의 실행 및 완료 결과 게시 승인으로 해석한다. 실제 응답 원문과 제한은 새 AUTH에 기록한다. 반복 승인을 요구하지 않는다. 기존 종료된 시계·receipt는 재설정하지 않는다. 이 문서 자체는 실제 실행 결과가 아니다.

## 자료와 기준 예측

원 Robin C17/K4 및 Beijing Aotizhongxin C11/K2, seeds92601/92602, L512/H48, FIT256/TUNE64/반복VAL30 또는 노출EVAL146을 유지한다. 원 FIT 통계, causal fill된 과거, 원 origin·history/target observability를 바꾸지 않는다. 새 보호 TEST·미래 oracle·자료·seed·window sampling·대형 모델 학습은 없다.

입력은 완료된 matched-readout producer의 RAW_INPUT 및 CACHED_E_INPUT 배열이다. canonical report manifest SHA `1573f4ee628fc5598e41178457937c86137198bb4007b11f3790e2659386c6d2`와 독립 감사 PASS가 해당 producer SHA를 대상으로 했는지 묶는다. 원 D2/bridge/checkpoint/source SHA와 codec/F 원 상태의 계보를 검증한다. CPU fit은 E/F/D를 deserialize하거나 forward하지 않는다. 원 MAIN은 cached `original_prediction`의 finite FP32 `[N,48,C]`이며 과거에서 생성된 예측이다. actual Y·oracle teacher와 혼동하지 않는다. ROOT의 source review나 배열 shape만으로 과거 전용 생성이 입증되는 것은 아니다.

Normalized target는 FP32 `[N,48,C]`, 관측 마스크는 bool 같은 shape다. 미관측 target NaN은 유지한다. 관측 target/residual은 finite이며 통계·anchor·origin·C/K/p/shape와 입력 SHA를 원 자료와 대조한다. 모든 미래 Y 접근은 FIT 학습 또는 점수 계산이고 inference 인수로 전달하지 않는다.

## 특징과 목표

RAW33C는 원 C의 32개16-point FP32 mean+last를 시간→채널 순서로 펼친다. E33K도 같은 요약을 cached 과거 E에 적용하고 E512K는 전체512 latent를 그대로 펼친다. p는 Robin561/132/2048, Beijing363/66/1024다. 입력 공간과 readout 용량·규제 효과가 다르므로 encoder의 순수 인과 효과로 해석하지 않는다. RAW 보정은 원 과거 C를 추가로 사용하는 대조이며 strict K-only가 아니다. 원 MAIN에도 current-C anchor가 이미 포함된다.

Feature center/scale은 FIT raw FP32 특징을 FP64로 바꿔 mean/population RMS deviation을 계산하고 zero scale을1로 둔다. 이 상태는 TUNE/VAL/EVAL에서 갱신하지 않는다. 원 feature raw SHA는 모델 상태용 dtype+shape+bytes 규약, NPZ manifest array SHA는 raw contiguous byte 규약이며 둘을 섞지 않는다.

**FP32 actual Y−FP32 original MAIN**을 먼저 계산한 뒤 residual을 FP64 solver로 바꾼다. 추론은 transformed features FP64 @ coefficients FP64 + intercept → correction FP32 → original MAIN FP32 addition 순서다. 보정0은 MAIN bits를 그대로 복사해 signed zero까지 유지한다. 미래 정답·미래 mask·미래 oracle latent는 추론 함수의 인수가 아니다.

FIT 평균 보정은 같은 FP32 residual의 관측 값만 FP64로 합하고 각 H,C의 N_hc로 나눈 뒤 FP32로 cast한다. 관측0은0이다. 조건마다 한 번 계산하고 FIT residual/mask SHA를 묶어 세 경로가 동일 객체/값을 공유한다. 이 평균 계산의 N_hc는 ridge penalty의 N_c와 다른 역할이다.

## ridge와 선택

Ridge는 active C 균등 평균인 전체 관측 NH residual squared error/N_c + alpha 평균_C 합_h||W_scaled_hc||²를 최소화한다. 각(h,c)의 normal equation penalty는 **alpha*N_c**이며 horizon별 N_hc가 아니다. intercept는 penalize하지 않는다. 관측0의 coefficient/intercept는0이다. count/denominator/rank/condition 배열은 **H×C**, channel_total_count는 C다. 관측 count/rank는 int64, denominator/condition/coefficient는 FP64다.

동일 관측 mask별 centered thin SVD를 한 번 계산해7 alpha에 공유한다. alpha0은 minimum norm이고 cutoff는 float64 eps*max(n_observed,p)*smax다. p² Gram/inverse를 만들지 않는다. condition은 nullspace와 ridge를 포함한 원 정의를 따르며 NaN/Inf는 NPZ로 보존하고 JSON에는 finite summaries/null을 쓴다. rank는 계수 수나 독립 자유도와 같지 않다. 이 단계의 저장 배열 감사는 solver 최적성을 독립적으로 재학습/입증하지 않는다.

각 경로는 candidate0 ZERO_CORRECTION, candidate1 공유 MEAN_CORRECTION, 다음 alpha0,1e-4,.001,.01,.1,1,10 순서인 **9개**를 비교한다. alpha0은 baseline0과 다르다. 실제 Y의 normalized observed-channel macro **TUNE MSE** 최솟값을 고르고 exact tie는 가장 앞 후보다. 고정 slot과 실패를 보존하며 유한 COMPLETE 후보만 선택한다. baseline0/mean을 ridge factorization 전에 확보해 한 ridge 경로의 실패가 유효한 기준 예측을 취소하지 않게 한다. TUNE MSE가 같거나 나쁜 새 보정은 채택할 필요가 없다. 이 선택은 평가 성능을 보장하지 않는다.

총4조건×3경로=12 ridge preparations,7 alpha씩 **84 새 ridge 모델**, 공유 zero/mean을 세 경로에서 평가한 **108 candidate comparisons**다. 공유 평균은4개 FIT control이지12 새 fit이 아니다. 모든 후보의 FIT/TUNE 및 선택 모델의 세 split 실제 예측·원 관측 수/채널별 normalized·physical errors를 저장한다. 평가값으로 방법/alpha/특징을 변경하거나 사후 우승 조합을 고르지 않는다.

## 실제 배포 비용과 자원

새 CPU fit/감사/보고와 GPU cost는 서로 직렬·독점 lease로 실행한다. 승인 상한은 CPU process900초/GPU process300초/wall2400초/추가저장1GiB/RSS8GiB다. process creation부터 import/guard/IO/solver/forward를 charge한다. CPU job은 CUDA/F 접근을 금지하고 GPU cost만 원 E/F/D를 frozen eval/no_grad로 load/forward할 수 있다. 모델 상태/파일/소스를 전후 확인한다. 원 F를 업데이트하거나 stopped job을 재개하지 않는다. 각 ledger를 새 AUTH/고정 manifest SHA에 묶고 과거 receipt를 SHA 보호한다. outer supervision은 새로 만든 자기 job에만 적용한다.

과학 결과가 검산된 뒤 원 seed92601을 고정해2자료×B1/B8×6방법 RAW-F/MAIN/mean/RAW33C/E33K/E512K 보정의 **24 whole cells**를 측정한다. Latin6 위치 균형, warm2/repeat6, same-batch normalized/physical 반환과 원 관측 점수 계약을 지킨다. 원 MAIN 생성 E+F+D, CPU 보정, 특징/예측 전송을 모두 포함한다. 선택이 zero/mean이면 불필요한 특징을 건너뛰는 실제 경로를 기록한다. 원 cache generation/historical learning은 재사용이며 새 비용0으로 바꾸지 않는다. 원 bulk와 새 batch의 차이는 matched-batch parity와 구분한다. 저장 출력/일정/median 감사는 독립 timing/E/F replay가 아니다.

실패·자산 부족은 해당 경로에만 적용하고 독립 조건은 계속한다. 숫자를 좋게 만들려고 재학습·threshold 완화·alpha 변경·실패 제거를 하지 않는다. cap이나 실제 active 충돌이면 해당 실제 상태와 재개 조건을 남긴다. fixture/감사 통과는 미래 예측 개선·일반화·연구 완성을 뜻하지 않는다.
