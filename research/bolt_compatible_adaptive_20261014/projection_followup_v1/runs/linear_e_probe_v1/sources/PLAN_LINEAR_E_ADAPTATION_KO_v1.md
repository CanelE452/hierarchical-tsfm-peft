# 같은 초기값에서 선형 표현 적응과 lag 구조를 분리하는 후속

상태: 사전 계획. 새 Torch 실제 기술 검사·학습·추론·비용은 아직 실행하지 않았다. 직전 causal-lag 2×2 묶음의 보고서 검산과 그림 확인을 먼저 마친 뒤 실제 자산에 접근한다. 그 보고서와 독립적인 새 SOURCE·합성 fixture 준비는 병행할 수 있으며 보고서 검산 PASS를 실제 실행 gate로 유지한다.

## 질문과 출발점

[확인] 고정 E/F의 24개 D 대조와 전체 비용은 실제 실행됐다. Robin/PCA의 RECON에서 L0→L7 개발 MSE는 .5349284825→.4880930144로 낮아졌고, 같은 complete-C 수치 유지 공간 minimum은 .4009074045→.3068850408이었다. Raw .3404018870까지 실제 예측이 도달하지는 못했다. 예측 목적 D도 전체 문제를 해결하지 못했다. 공간 진단이 낮다는 사실은 과거로 그 좌표를 예측할 수 있다는 증거가 아니다.

[판단] 다음 최소 질문은 **동일한 초기 E/D와 예측 손실에서, E를 고정하는 제약을 풀면 개선되는가**다. lag0/lag7과 E 고정/학습의 2×2를 비교한다. 새로운 대조학습·그래프·큰 모델을 추가하지 않는다. 이전 joint-Y/RF3는 이미 frozen F 입력 gradient로 E/D를 학습했으므로 이번을 최초 end-to-end 학습이라고 부르지 않는다. 차이는 origin 기준의 선형 lag D와 선형 E를 같은 초기값에서 분리한다는 점이다.

## 고정 조건

Robin C17/K4, Beijing Aotizhongxin C11/K2, L512/H48, 기존 FIT256/TUNE64와 이미 노출된 VAL30/EVAL146, 원 scaler·관측 mask·causal fill을 유지한다. 새 TEST나 데이터는 사용하지 않는다. 초기 E는 기존 PCA 하나로 고정하고 center는 buffer로 유지한다. 이는 탐색을 줄이기 위한 초기값이며 PCA의 우수성이나 최선의 표현 선정을 주장하지 않는다. 설계 자체가 노출된 DEV를 본 뒤의 탐색이다.

자료2 × lag2([0], [0,1,2,4,8,16,32]) × E 고정/학습2 × minibatch seed2(92601,92602) = 16 fit이다. seed는 동일 초기값에서 표본 순서만 바꾸며 독립 초기화가 아니다. D 초기값은 직전 같은 PCA/lag FORECAST LS 계수로 고정한다. 모든 조건에 동일 Torch CP0/LS와 Raw를 함께 보존한다. D-only가 초기 LS에서 나빠진 것을 E 학습의 성공으로 해석하지 않는다. L7은 계수가 7배이므로 동일 용량의 시간 효과가 아니다.

모델은 `currentC + sum_l (z_combined(h-l)-z_past(-l)) @ D_l`이다. E는 시점별 선형 C→K이며 bias를 추가하지 않는다. F는 실제 frozen Chronos-Bolt module의 q.5, H64에서 앞48이다. 과거만 forward에 입력하고 FIT 미래 Y는 손실에만 쓴다. K streams·K 과거 buffer·현재 C anchor를 사용하므로 strict K-only가 아니다. 현재값 anchor와 알려진 과거 prefix에서 E로 가는 직접 gradient 경로도 존재한다.

## 기술 검사와 학습

1. 보고서 묶음 완료 후 실제 HEAD·기존 변경·작업·source/자산 SHA를 다시 확인한다. 기존 작업을 종료하거나 재개하지 않는다.
2. Torch pure 합성 검사에서 lag index·reference·flat prefix·mask·NumPy 식 parity와 gradient 경로를 확인한다. actual 검사 전 source를 동결한다.
3. 작은 실제 probe에서 자료별 L0/L7 CP0의 Z/F/lag 특징/예측/지표를 기존 NumPy 상태와 대조한다. CP0의 E 고정/학습 조건은 동일 backend에서 정확히 같아야 한다. 학습 batch4와 canonical 평가 batch32 차이는 별도 기록한다. parity 공차는 normalized 출력 atol1e-5/rtol2e-6을 먼저 적용하고, 실패하면 원 결과를 남기고 해당 경로를 멈춰 원인을 진단한다. 통과를 위해 공차만 늘리지 않는다.
4. F는 eval·파라미터 requires_grad=False·optimizer 제외·전후 model/weight hash 동일을 유지한다. 상수 분산의 기존 ZeroSafeInstanceNorm 의미를 보존한다. loss→F future→F input→E 경로만의 finite/nonzero gradient를 직접 과거 경로와 분리해 확인한다. 총 E gradient만으로 F 학습 경로를 증명하지 않는다. 실제 backward 시간·RSS/VRAM을 먼저 측정하며 smoke만 하고 끝내지 않는다.
5. AdamW lr.001/weight_decay0, global clip1, batch4, 600 update, CP0/100/200/400/600을 고정한다. 채널 균형 정규화 MSE를 사용한다. minibatch별 관측수로 나누는 대신 FIT 전체의 고정 채널별 관측수로 가중하고 N/B를 곱해 전체 FIT macro 목적의 표본 추정량을 사용한다. 결측 채널·관측0 fallback은 실제 계약으로 기록한다. 기존 per-batch macro 실험과 가중이 같다고 쓰지 않는다. E/D gradient norm·전체 clipping 비율·finite 상태·E singular value/latent variance/D gain을 기록하며 결과를 본 뒤 regularizer를 추가하지 않는다.
6. canonical32 TUNE 전체의 관측 채널 macro MSE가 가장 낮은 CP를 선택하고 exact tie는 빠른 CP를 고른다. 16개 성공/실패 잠금과 registry를 만든 후 DEV를 추론한다. CP0와 모든 TUNE CP 원수치·예측을 보존한다. 정상 종료가 아닌 경로를 완료라고 하지 않는다.
7. 독립 검산으로 저장 계수/CP/예측/손실·mask·선정·F 동결·기존 자산 보존을 확인한다. 새 전체 배포 비용은 같은 cohort Raw와 실제 E→F→lag D→물리 CPU 반환으로 측정하고 기존 fixed-D 비용을 재사용하지 않는다. 한국어 보고서·원수치·상대 링크·그림을 만든다.

기술 probe의 실제 추가 update와 본 학습 9,600 update를 구분한다. 분산0 gradient 처리는 기존 동작을 재사용하되 다른 폭주를 숨기지 않는다. canonical batch parity가 안 맞으면 원인을 확인한 별도 수정계획/version이 필요하다.

## 해석과 다음 gate

E 학습이 D-only와 CP0/Raw보다 나아져도 현재 구성의 공동 적응 효과라는 결론만 허용한다. F 입력 gradient가 성능에 필요했다거나 Bolt가 필요했다는 결론은 별도 stop-gradient-F/동일 native 대조 없이는 보류한다. 이득이 없으면 해당 선형 E/D·초기값·600 update 조건의 한계이며 모든 예측 목적 표현의 불가능성이 아니다. 일부 경로 실패는 독립 경로를 막지 않는다. DEV 선택·oracle 학습·좋은 결과만 선택은 하지 않는다.

사용자의 상한 해제·계속 실행 지시 안에서 같은 목적을 향하는 후속이라는 작업 판단이다. 개별 16 fit의 별도 인간 승인을 받았다고 주장하지 않는다. 반복 승인을 묻지 않으며 실제 자원 영수증을 계속 남긴다. TEST·설치/다운로드·큰 백본 파라미터 학습·기존 작업 종료/중단 실험 재개·자동 commit/push·논문/PPT는 포함하지 않는다.
