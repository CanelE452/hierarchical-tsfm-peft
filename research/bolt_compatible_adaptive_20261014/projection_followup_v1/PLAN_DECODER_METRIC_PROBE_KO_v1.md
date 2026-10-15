# 동일 latent에서 디코더만 바꾸는 원인 진단

현재 ForeCA의 과거 최소제곱 D는 D@E가 I에 가까워도 E@D가 정규화 C공간의 직교 사영이라는 보장이 없다. E(Y) 진단이 고정 D의 최소 오차보다 큰 경우도 확인했다. 그렇다고 비직교성 하나가 실제 예측 악화의 유일한 원인이라고 단정하지 않는다. 기존의 모든 E/F/선택 native head·K·L/H·FIT 통계·origins·현재 C anchor를 유지하고 캐시 latent에서 **D만 교체**하여 직접 확인한다.

대조는 원 `ORIGINAL_PAST_LS`와 `ENCODER_PINV`다. 대체 D는 기존 배포 E32를 FP64로 해석한 `pinv(E,rcond=1e-15)`를 FP32로 바꾼다. E의 full column rank를 먼저 확인한다. 미래 Y·hyperparameter 탐색·seed 선택·encoder/F/head 재학습은 사용하지 않는다. 기존 승인된 원인 진단의 결정적 디코더 비교이며 새 supervised E/D 학습 분기는 열지 않는다.

Robin C17/K4·Beijing C11/K2, FIT256/TUNE64/반복 VAL30·기노출 EVAL146의 정확한 완료 v2 producer와 PASS 독립 검산, 고정-D span 진단의 PASS 검산을 먼저 묶는다. NumPy FP32 전체 N 연산의 Bolt/E(Y), Torch CPU2 threads·chunk32 연산의 native를 각각 원본 backend로 복원한다. ORIGINAL D의 출력은 원 캐시와 byte-exact여야 한다. 같은 backend에 새 D를 넣고 `raw_last+(future-lastK)@D`로 실제 CPU 디코더 추론을 수행한다. 미래 latent와 last는 기존 저장값을 쓰며 E/F/head forward를 다시 실행하지 않는다.

2자료×2route×3split의12조건에서 두 D의 Bolt·기존 native 두 seed·E(Y)를 보존한다. native head는 원 D로 학습됐으므로 새 D 결과는 **고정 head의 디코더 개입**이며 새 D를 위해 최적화한 native 성능이 아니다. E(Y)와 고정-D 최소는 미래 정답을 진단/채점에만 사용하고 방법·checkpoint·분기를 선택하지 않는다. full observed macro 정규화 MSE/MAE·원단위 진단과 동일 complete-C mask 보조 지표를 모두 남긴다. 관측 과거 복원은 기존 저장 latent와 원 LS bias/원 FIT center를 사용하며 새 E forward나 미래 학습을 하지 않는다.

대체 D는 출력공간·scale·오차 증폭도 함께 바꾼다. 따라서 차이가 나도 그것을 비직교성만의 효과로 해석하지 않는다. Byte parity 실패·rank 부족은 해당 경로를 실패로 보존하고 독립 조건을 계속 확인한다. 선택 잠금·원 캐시·승인·소스 SHA를 전후 확인하고 fresh 디렉터리에 출력/계보를 저장한다.

이 단계는 캐시 재사용을 포함한 실제 CPU D 추론이다. **새 D의 전체 배포 비용은 측정하지 않는다.** E/F·전송·전처리를 포함한 새 배포 비용과 새 D용 head 최적화 여부는 미실행 상태로 명시한다. 별도 독립 최소제곱/캐시/출력 검산을 통과한 결과만 보고서에 사용한다.
