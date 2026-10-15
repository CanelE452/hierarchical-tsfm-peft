# 선택을 고정한 shrinkage 디코더의 출력공간 진단

상태: SOURCE와 합성 검산 준비. 실제 자산을 읽거나 실행하기 전 ROOT의 소스 검토와 실행 요청을 기다린다.

[판단] 이 작업은 진행 중인 과거 디코더의 안정성 원인 진단에 속한다. 별도의 개별 방법에 대한 정확한 인간 승인 발언이 있었다고 주장하지 않는다. 원 source·결과·실패·자원·선정 잠금을 보존한다. TEST, 데이터 추가, 새 rho 선택, DEV grid, encoder/forecaster/head 학습, F/head forward, 설치·Git 게시·기존 job 종료는 포함하지 않는다. 진단용 E(Y)의 선형 FP32 산술은 CPU에서 계산한다.

질문은 선택된 과거 적합 D[n] 자체의 출력공간이 제한되는지, 현재 실제 미래 잠재 예측에 문제가 남는지다. 원 shrinkage COMPLETE와 독립 PASS를 SHA로 결속하고 모든 네 개의 TUNE Bolt 선정 잠금을 검증한 뒤, 같은 selected rho의 FIT/TUNE/DEV 저장 상태만 사용한다. 현재 계약은 자료2×공간2×split3×numeric backend2=24개이며, 개발 구간은 그중8개다. Bolt NumPy와 native Torch CPU chunk32는 분리하며 native 두 seed는 동일 과거와 D를 공유한다. Bolt/native는 같은 rho와 lambda 규칙을 사용하고 실제 lambda가 두 numeric backend 사이에 byte-identical하다고 주장하지 않는다. 실제 상태가 이 계약과 다르면 추정하지 않고 미실행 이유를 남긴다.

각 origin에서 완전 관측된 C개 미래 위치만 사용하여 V=Y-currentC를 만든다. 저장된 D32[n]를 FP64로 해석한 후, SVD의 상대 cutoff1e-15로 수치 rowspace의 직교 projector를 계산한다. V를 그 공간에 투영한 residual energy/(완전 위치 수×C)가 같은 mask의 macro 정규화 MSE와 일치해야 한다. 별도의 직접 lstsq(D.T,V.T,rcond1e-15)로 투영·좌표·normal residual을 검산한다. projector의 대칭성·멱등성·rank와 에너지 분해도 확인한다. 미래 정답은 이 진단 좌표에만 쓰며 원 D·rho·branch·winner·예측기를 변경하지 않는다.

이 최소값은 해당 origin의 알려진 D32와 real arithmetic의 정규화 MSE에 한정된다. FP32 모든 표현의 엄밀한 하한, MAE·물리 MSE의 최소, 원 입력 정보의 근본 한계, 과거에서의 예측 가능성이나 독립 일반화 증명이 아니다. 이전 static D 또는 split 전체 공통 rank-K oracle의 하한을 새 D[n] 하한으로 재사용하지 않는다. Raw에는 하한 부등식을 적용하지 않는다.

같은 완전 C mask에서 Raw, 실제 Bolt/native 예측, 진단용 E(Y)를 비교한다. E(Y)는 고정 E32·center32와 원 cached latent last를 사용한다. Bolt는 NumPy 산술, native는 CPU Torch chunk32 산술을 사용한다. currentC+(E(Y)-cachedLast)D32를 FP32 einsum으로 계산하고 FP64 real trace도 별도로 계산한다. FP32의 MSE와 real trace 차이는 실제 측정된 rounding allowance로 기록한다. E(Y)는 엄밀한 최적 좌표가 아니며, 위 공간 최소보다 나쁜 경우를 decoder 문제가 전부 해결됐다는 뜻으로 바꾸지 않는다. 부분 관측 위치의 E(Y)는 보고·최소 비교에서 제외한다. CPU 산술의 finite 입력을 위해 제외된 위치 전체를0으로 치환한다. 고정 E가 시점별 선형 투영이므로 그 치환은 보고하는 완전 관측 위치에 영향을 주지 않는다.

새 감사·진단은 `audit_selected_shrinkage_output_space_v1.py`에서 수행한다. 원 predictor·ridge producer의 수학 helper를 import하지 않고 기존 독립 loader·mask scoring·FP32 anchor helper만 재사용한다. 저장된 미래 latent를 CPU 산술로 다시 복원할 뿐 새 F/head inference나 timer를 수행하지 않는다. source·gate·원 상태·캐시·selection lock의 SHA를 전후 확인하고, 보호된 TEST guard와 CPU Meter를 사용한다. 새 산출물은 exclusive 생성한다.

ROOT 검토 후 실행 형식:

```text
python audit_selected_shrinkage_output_space_v1.py --result runs/local_decoder_shrinkage_v1/RESULT.json --audit audits/LOCAL_DECODER_SHRINKAGE_AUDIT_v1.json --output runs/selected_shrinkage_output_space_v1
```

출력은 `RESULT.json`과 24개 선택 상태 진단 NPZ다. 실제 실행 및 수치 검산이 모두 끝난 경우에만 PASS로 표시한다. 합성 통과는 실제 진단 완료가 아니다. 이미 노출된 개발 구간의 탐색적 후속이며 independent holdout을 주장하지 않는다.
