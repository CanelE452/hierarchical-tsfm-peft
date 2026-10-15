# 고정 표현·Bolt에서 디코더 시간 구조와 학습 목적을 나누는 2×2 대조

상태: 사전 계획. 이 문서의 새 디코더 적합·추론 재생·비용 실행은 아직 0이다. 앞 raw subset 묶음의 보고서 독립 검산을 완료한 다음 실행한다. 기존 source/result/선정/자원 기록은 보존한다.

## 목적과 확인된 출발점

[확인] raw_subset_comparison_v2와 독립 감사 v3에서는 Robin의 PCA/FIT-target/raw-subset 세 고정 시점별 선형 D의 미래를 아는 실수 출력공간 minimum이 약 .393~.414로 같은 mask Raw .340보다 높았다. Beijing은 반대로 minimum의 여지가 있었지만 실제 예측이 그 수준에 도달하지 못했다. 실제 미래 E(Y)를 D에 넣는 것은 minimum과 같지 않았다. 원 채널을 섞지 않은 subset도 Raw를 넘지 못했다.

[판단] 다음 최소 질문은 `작은 K에서 정확도·전체 비용을 유지할 수 있는가 → 시점별 D 제약과 D 학습 목적을 분리하는가 → 같은 E·동결 Bolt·실제 미래 예측 캐시에서 causal lag와 FIT 예측 목적을 2×2 비교하는가`다. 표현을 새로 학습하거나 큰 모델을 추가하기 전에 이 두 설명을 직접 점검한다. 같은 목적의 최신 계속 실행 지시 안에서 진행하는 최소 대조라는 작업 판단이며, 이 개별 실험에 대한 별도의 인간 승인이 있었다고 주장하지 않는다.

## 고정 자산과 네 조건

Robin C17/K4, Beijing Aotizhongxin C11/K2, L512/H48, FIT256/TUNE64, 기존에 노출된 VAL30/EVAL146, 원 FIT mean/std·origin·관측 mask·causal fill을 유지한다. 현재 세 E(PCA/FIT_TARGET/RAW_SUBSET)와 NumPy Bolt backend의 실제 latent history/future 캐시 18개를 정확한 파일·배열 SHA와 수치 lineage로 재사용한다. FIT-target E는 이미 FIT 미래 감독을 사용한 공간이라는 사실을 유지한다. 보호 TEST·새 데이터·새 F 학습/추론은 이 CPU 재생 단계에 없다.

각 자료·E에서 다음 네 D를 모두 적합한다. 배포 계수는 FP32, 적합은 FP64 직접 least squares, singular 상대 cutoff는 1e-4다.

```text
조건              lag steps                  D 적합 목적
L0_RECON          [0]                        FIT 관측 과거 level 재구성
L7_RECON          [0,1,2,4,8,16,32]           FIT 관측 과거 level 재구성
L0_FORECAST       [0]                        FIT 실제 Bolt 미래로 actual-C innovation 예측
L7_FORECAST       [0,1,2,4,8,16,32]           FIT 실제 Bolt 미래로 actual-C innovation 예측
```

lag는 자료의 sampling step이며 분·시간 단위를 임의로 붙이지 않는다. 두 RECON은 공통 FIT 과거 `32:512`의 480시점만 쓰고, 동일 true-observed C response에서 predictor/response를 중심화해 LS와 분석적 절편을 적합한다. 원 512시점 D를 L0 대조군으로 재사용하지 않는다. 상수 response의 정상 zero slope와 무관측/관측1개/무분산 fallback을 구분한다. 절편은 아래 배포 innovation에서 상쇄된다.

두 FORECAST는 같은 FIT 창의 48 미래에서 아래의 실제 배포용 delta-Z 특징에 대해 `actualY-currentC`를 관측 C별로 직접 LS 적합한다. 절편을 추가하지 않는다. 미래 정답은 FIT 학습 target으로만 쓰며 predictor는 실제 동결 Bolt가 과거로 만든 예측이다. TUNE/DEV 값·실제 미래 E(Y)·oracle 잠재값은 적합에 넣지 않는다. 무관측이면 slope0, 관측1개이면 비중심 LS가 식별할 수 있는 부분을 기록한다. RECON과 FORECAST는 같은 FIT 창을 쓰지만 target 시점 수(480 대 48), 감독 정보와 분포가 다르다. 이것을 손실 함수만 바꾼 완전한 단일 인과 개입이라고 부르지 않는다. 같은 lag 내 배포 모델의 파라미터 수와 입력은 같다.

## 배포식과 정보 계약

현재 origin을 0, 미래 h를 1..48로 놓는다. `z[-ell]`의 구현 index는 `history[:, -1-ell]`이다. s<=0이면 알려진 과거 latent, s>0이면 저장된 실제 Bolt 예측을 쓰는 결합 시퀀스를 ztilde라 하면,

```text
deltaZ(h,ell) = ztilde(h-ell) - z(-ell)
prediction(h) = currentC + sum_ell deltaZ(h,ell) @ B_ell
```

현재 h0에서는 currentC와 일치한다. lagged known prefix 때문에 flat future latent를 넣어도 출력이 항상 currentC가 되는 것은 아니다. 이 flat-latent baseline과 실제 current-C persistence를 별도로 기록한다. 원 RF3의 paired-flat-reference 비선형 residual과 같지 않다.

입력은 K개 stream·K 과거 buffer·현재 C anchor다. origin마다 C 과거로 D를 새로 적합하거나 계수를 전송하지 않는다. strict K-only 비교는 아니다. L7은 L0보다 slope 수가 7배이므로 동등 용량 대조가 아니다. 원시 채널 간 인과 그래프를 학습하거나 지연의 인과성을 증명하는 실험도 아니다. 기존 native는 이전 D에 맞춘 모델이므로 이번 1차 주비교에서 제외한다. 결과가 유망하면 새 D에 맞춘 native를 별도 사전 조건으로 학습해야 한다.

L7의 회귀 계수는 많아지지만 H-step 전체에서 자유로운 미래 latent는 여전히 HK개다. 각 시점에 LK개 특징을 독립적으로 자유롭게 고르는 구조가 아니며 joint 출력 rank 상한은 HK를 넘지 않는다. 달라지는 것은 미래 전체의 출력공간 방향과 알려진 과거에 따른 offset이다. 이 차이를 floor의 rank/미지수 수와 함께 보고한다.

## 단계와 검산

1. 실제 저장소 HEAD·기존 변경·활성 작업·등록 source/result/PASS·자산 SHA를 확인한다. 과거 기록을 덮거나 작업을 종료하지 않는다.
2. pure 함수의 delayed-copy·lag index·known-prefix offset·flat baseline·mask·rank 결손·동일 시점 대조·미래 label 비사용을 합성 검증한다. 기존 폴더에서 versioned source를 만든다.
3. FIT 자산만 열어 자료2×E3×조건4=24 D를 적합한다. 계수·학습 입력/target binding·응답별 관측수/rank/singular/residual·실제 시간/자원을 기록한다. 모든 경로의 성공 또는 실패 잠금과 registry를 저장한 뒤 이번 TUNE/DEV 재생을 연다. 일부 실패는 해당 경로만 막는다.
4. 실제 F latent 캐시의 과거·origin·입력 E lineage를 검증해 24 D×3split=72 실제 D 추론 배열을 저장한다. 이것을 새로운 F 호출이나 backbone 학습으로 세지 않는다. 원수치 full-observed 정규화 macro MSE/MAE·per-C·원단위 진단과 complete-C 별도 비교를 기록한다.
5. 각 새 D/split의 공동 horizon 출력공간 진단을 새로 계산한다. 미래 잠재를 자유변수 v=(z1,...,zH)로 쌓으면 A는 HC×HK block lower-triangular이며 A[h,j]=B_(h-j)^T다(해당 lag가 있고 j<=h일 때). known-prefix와 기준 latent를 포함한 offset b를 직접 배포식과 독립 대조한다. **관측이 없는 horizon도 미래 잠재 미지수에서 삭제하지 않는다.** complete-C horizon에 대응하는 출력 행만 골라 모든 HK 좌표를 함께 LS 최적화한다. solver rcond1e-15·실수 수치 rank와 잔차 직교성·정상방정식 검산을 기록한다. 동일 complete-C 위치의 SSE 합/(위치수×C)로 점수를 합하며 창별 평균을 다시 평균하지 않는다. 원 시점별 K 하한이나 global rank-K 하한은 재사용하지 않는다.
6. actual F·flat latent·complete-C 실제 미래 E(Y) 각각의 FP32 prediction과 같은 수치 상태의 실수 trace를 분리해 floor 부등식·rounding allowance를 검산한다. E(Y)의 불완전 C 미래 위치는 currentC로 채우는 사전 진단 규칙을 사용한다. 이 위치도 lag로 뒤 출력에 영향을 줄 수 있음을 밝힌다. E(Y)는 최적 잠재값·배포 입력·선정 근거가 아니다. minimum은 해당 수치 실수 공간의 정규화 MSE에만 적용하며 MAE/물리 단위/모든 FP32 출력의 엄밀 하한이라고 부르지 않는다.
7. 별도 독립 auditor가 LS·모든 예측·mask/단위·joint map/offset/minimum·원파일 보존을 확인한다. accepted science 뒤 전체 배포 비용을 새 동일 cohort Raw와 측정하는 사전 계획/source를 만든다. 기존 다른 D/시행의 latency를 재사용하지 않는다. 보고서는 원수치·실패·불확실성·그림·상대 링크를 포함한다.

공동 solver의 1e-15 cutoff로 버려진 작은 singular 방향이 있을 수 있다. 이 경우 등록 minimum은 retained numeric 출력공간에 한정된다. 실제 real trace를 그 retained 공간에 투영한 값에만 필수 minimum 부등식을 검산하고, 버린 출력 성분과 원 real/FP32 trace의 차이·비교 boolean은 별도로 보존한다. 실제 예측이 그 numeric minimum보다 낮다는 이유로 공차를 늘리거나 원 예측을 실패로 없애지 않는다. 모든 실수 latent에 대한 하한이라고 확대하지 않는다. 모델 실행/실제 자료 접근 전 SOURCE 정교화이며 fit cutoff·배포식·목적·데이터를 바꾸는 수정은 아니다.

현재 네 조건은 모두 사전 지정한 탐색적 대조이고 DEV 최저 조건을 채택하지 않는다. 이후 비교는 TUNE 개선과 측정 비용을 근거로 판단하며 oracle만 낮아졌다는 이유로 성공이라고 쓰지 않는다. 모든 후속 설계는 이미 DEV 결과를 본 뒤의 탐색이며 독립 일반화가 아니다. RECON이 actual E(Y)에서도 실패하면 residual0-residualh 및 학습 목적 불일치가 남고, FORECAST가 실패해도 모든 예측 목적 표현이나 큰 모델의 불가능성을 뜻하지 않는다.

## 관련 방법과 실행 범위

잠재 성분의 지연값으로 선형 복원하고 성분별 예측을 복원식에 넣는 구조는 이미 [ODPC §2·§2.2](https://arxiv.org/html/1708.04705v1)에 있다. 이번 고정 TSFM/encoder/cache·현재값 anchor·두 D 목적의 대조를 ODPC 전체 구현이나 새 구조의 발견이라고 부르지 않는다. 기존 RF3·origin-reference source·direct-C ridge·actual-Y 학습도 보존한다.

[확인] 기존 `research/smallk_failure_followup_20261014/frozen_decoder_refit.py`와 `validation_v1/D2_ARRAY_AUDIT_v1.md`에는 실제 예측 목적 D2도 있었다. 고정 E/F/GELU hidden의 delta-z/delta-GELU에서 actualY-currentC를 RMS ridge로 적합하고 7 alpha를 TUNE으로 골랐으며 독립 검산은 28,365검사/실패0이었다. Robin 기존 D 평균 MSE TUNE .076528/VAL .569547 대 Y-refit .078294/.583851로 개선되지 않았다. 이번은 첫 predictive D가 아니라 현재 세 선형 E·실제 F 캐시·같은 lag별 배포 용량에서 두 적합 목적을 교차 대조하는 별도 조건이다. 과거 방식의 실패를 없애거나 새 성공으로 바꾸어 쓰지 않는다.

최신 상한 해제 지시에 따라 이 최소 CPU 실험을 계속하되 누적 영수증과 실패를 보존한다. TEST·설치/다운로드·큰 backbone 학습·기존 중단 실험 재개/종료·자동 commit/push·논문/PPT 작성은 수행하지 않는다. 별도 사용자 확인을 반복하지 않는다. 자산이나 구현 문제로 못 한 경로는 실제 상태·이유·재개 방법을 기록한다.
