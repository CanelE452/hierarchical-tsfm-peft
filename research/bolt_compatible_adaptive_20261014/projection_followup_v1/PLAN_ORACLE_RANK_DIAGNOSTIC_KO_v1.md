# 현재 디코더와 임의의 K차원 선형 출력공간을 분리하는 수학 진단

현재 E(Y) 진단은 실제 미래를 입력해도 E/D가 만드는 사영이 직교 최소제곱 사영이라는 보장이 없어 하한이 아니다. 고정 D의 출력공간 최소제곱 진단 뒤, 그 D의 문제와 같은 K의 어떤 공유 선형 출력공간에도 남는 오차를 구분한다. 새 예측 모델을 개발하거나 기존 방법을 다시 학습하는 작업이 아니다.

자료는 기존 승인된 Robin C17/K4와 Beijing C11/K2, FIT256/TUNE64/반복 VAL30·기노출 EVAL146, 동일 저장 FIT 정규화·origins·current-C anchor·실제 모든 C가 관측된 target 시점이다. 새 TEST·자료·모델·checkpoint 선택·배포 입력·예측 손실은 추가하지 않는다. 최신 사용자 상한 해제에 따라 기존 승인/manifest는 보존하고 CPU 진단 비용만 새 Meter receipt에 기록한다.

각 자료의 각 split에서 실제 알려진 미래 innovation `V=Y64-currentC64`의 complete 시점들을 M×C 행렬로 만든다. 미래 평균을 빼거나 별도의 미래 C intercept를 주지 않고 uncentered SVD를 계산한다. 상위 K 성분 `V_K`와 나머지 singular energy의 합을 저장한다. `sum(s[K:]²)/(M*C)`는 **이 이미 알려진 미래 행렬에 대해 임의의 공유 rank≤K 선형 innovation 출력공간**을 허용한 실수 연산 정규화 MSE의 수치적 최소다. 현재 고정 D floor 및 동일 mask의 RAW와 비교한다.

이 oracle는 **FIT뿐 아니라 TUNE·개발 split의 실제 미래로 각 split의 방향까지 맞추는 강한 미래 정보 사용 진단**이다. 하나의 D가 모든 split에서 작동한다는 의미가 아니며, 배포 불가능하다. 모델 순위·checkpoint·새 방법·분기 선택·일반화 주장에 사용하지 않는다. 작은 값도 과거 입력에 예측 정보가 충분함이나 현실적으로 달성 가능한 예측 성능을 뜻하지 않는다. 큰 값도 모든 비선형 디코더·representation·K 정보량의 근본 한계를 뜻하지 않는다. FP32 rounding은 실수 연산 최소의 범위 밖이고 MAE/원단위 지표는 이 정규화 MSE 해의 값일 뿐 별도의 최소가 아니다.

실제 payload를 열기 전에 완료된 v2 producer와 정확한 SHA를 묶은 PASS projection audit, 고정-D diagnostic과 정확한 SHA를 묶은 PASS 독립 span audit를 요구한다. 보호 TEST audit hook을 설치하고 모든 입력 NPZ를 선언한 파일/멤버 SHA·shape·dtype로 검산한다. 2자료×3split의 6개 미래 행렬에 대해 실제 V·복원·잔차·singular/eigen energy·Gram diagonal·좌표 사영·동일 mask·origin과 원수치를 fresh 디렉터리에 저장한다. 기존 선택 출력과 잠금·승인·소스의 SHA를 전후 확인한다.

검사는 회전한 rank1 해·잔차 에너지 보존·centered+intercept와의 차이·누락 target 제외·SVD 대 Gram PSD tail·고정 D와 저장 real-arithmetic 출력에 대한 포함관계·입력 불변성·미래 방향의 비배포 범위를 다룬다. ROOT의 별도 독립 검산을 통과한 숫자만 보고서에 넣는다.
