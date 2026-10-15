# 학습 후 남은 출력 공간과 잠재 예측 오차 분리

작성 시점의 계획이며 이 진단의 새 실제 실행은0이다. 선형 E 적응16조건은 과학·비용·보고서까지 검산을 마쳤다. F-input-gradient ablation4조건은 실제2400update·추론까지 끝났으며 독립 과학 감사와 비용·보고는 진행 중이다. 아래 SOURCE 준비는 병행하되 해당 입력 결과의 PASS를 결속한 뒤 실행한다. 현재 실행 중인 검산·보고를 대체하지 않는다.

[확인] full-gradient가 stop-gradient보다 TUNE·노출 DEV에서4/4 좋았지만 두 방식 모두 Raw 정확도에 미치지 못했다. [판단] 다음 질문은 **갱신한 E/D의 출력 공간 자체가 여전히 좁은가, 같은 공간에서 실제 F 미래를 잘못 예측하는가**다. 초기 PCA의 진단 minimum을 갱신된 D의 값으로 재사용하지 않는다. 새 인코더나 예측기를 무작정 추가하기 전 이 둘을 분리한다.

## 최소 비교

이미 학습한 full-gradient E갱신 L0/L7의8상태와 stop-gradient L0의4상태, 각각 TUNE/DEV의 총24진단을 계획한다. FIT 재학습·새 E/D 적합·새 F forward·optimizer update는0이다. 기존 결과와 선정 CP를 그대로 사용하며 이 진단으로 CP를 다시 고르지 않는다. L0 full/stop은 같은 초기값·order seed·600update budget의 비교이고, L7은 별도 공간 진단이다.

각 상태에서 저장된 actual-F 미래와 latent history를 사용한다. 새 D32·known past offset의 joint HC×HK 실수 맵을 만들고 원 rcond1e-15의 retained numeric left-singular 공간에 대한 complete-C normalized MSE minimum을 계산한다. 미래 자유변수는 HK이며 관측 없는 horizon의 미래 좌표도 유지한다. exact0열만 줄인 독립 LS 후 HK에 재삽입하는 기존 검산 경로와 SVD 출력 투영을 교차 확인한다. 원래 정의·공차·잘라낸 방향을 바꾸지 않는다. 이 수치 공간 minimum은 모든 FP32/모든 실수 예측에 대한 보편적 하한이 아니다. 원 실제 FP32, FP64 trace 및 retained trace의 점수와 버린 성분을 구분한다.

가능한 경우 같은 E에 미래 실제 C를 넣은 E(Y), latent persistence와 실제 F를 같은 D에 통과시킨 진단도 함께 계산한다. E(Y)는 미래 정답을 쓰므로 배포·선정 입력이 아니다. incomplete-C 미래 벡터는 전체를 current-C로 대체한 뒤 E에 넣고, 이 대체가 L7의 이후 관측 위치에도 영향을 준다는 한계를 명시한다. complete-C 위치에서만 이 진단을 비교한다. raw complete-C 참조도 실제 예측 파일·mask·원수치가 결속되는 경우에만 사용하고, 없는 자료를 정규화 역산·수치 추정으로 만들어 넣지 않는다.

## 판단과 산출물

각 조건의 원 actual/complete-C mask/count, 공간 minimum, actual-minus-minimum, E(Y)/persistence 진단, 원 계수·선정·예측 SHA를 보존한다. 실제 Torch 예측과 NumPy E/lag/D 재생의 등록 공차(atol1e-5/rtol2e-6) 및 독립 LS/SVD의 기존 FP64 공차를 확인한다. 큰 차이·rank 불일치·자산 부재는 해당 경로를 미완료로 기록한다. 독립 검사 PASS, 한국어 해석·표·그림과 재개 이유를 남긴다. 최소한 다음을 구분한다.

- 공간 minimum도 Raw 참조보다 높으면 현재 고정된 D32/offset 공간의 제약이 남는다.
- minimum은 낮고 actual F가 높으면 같은 표현·D 안의 잠재 미래 예측 또는 학습 적합 문제가 남는다. 그 minimum의 미래를 과거로 예측할 수 있다는 증거는 아니다.
- E(Y)도 낮지 않으면 재구성이나 자연스러운 E(Y)가 이 forecast-calibrated D의 미래 좌표와 정렬된다는 보장이 없다.

이미 노출된 두 자료의 적응적 원인 진단이며 독립 일반화·모든 작은 K 표현의 불가능성·Bolt/대조학습/그래프 필요성 검증이 아니다. 최신 계속 실행 지시 안의 동일 목적 후속 판단이며 새 개별 승인을 받았다고 주장하지 않는다. TEST·설치/다운로드·백본 파라미터 학습·기존 작업 종료/중단 학습 재개·자동 commit/push·논문/PPT는 포함하지 않는다.
