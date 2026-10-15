# L0에서 F 입력 gradient 기여를 분리하는 최소 대조

사전 계획이다. 직전 선형 E 적응 묶음의 수치·비용 검산은 끝났고 보고서 시각 보완·독립 검산은 진행 중이다. 새 SOURCE·합성 검사는 병행하되 실제 새 자료/model 실행은 그 보고서 검산 뒤 시작한다. 이 새 실험의 실제 실행은 아직0이다.

[확인] 직전 16조건에서 같은 lag/seed의8쌍 모두 E 갱신이 고정보다 TUNE·개발 MSE를 줄였지만16조건 모두 Raw보다 나빴다. 인코더는 실제 갱신됐고 수치 rank도 유지됐다. F 경유 gradient와 직접 encoded-past gradient가 함께 존재했다. 마지막 CP600 선정6/8만으로 공통 수렴 미달을 단정할 수 없으며 Robin seed92601은 CP400 뒤 TUNE가 악화됐다.

[판단] 바로 학습량을 늘리기 전에 **같은 F 예측값을 쓰면서 F 입력으로 돌아가는 gradient가 E 적응의 이득에 필요했는가**를 먼저 좁힌다. 전체 TSFM이나 F 예측값의 필요성을 검증하는 질문은 아니다.

## 고정 비교

새 조건은 L0만 Robin/Beijing×minibatch-order seed92601/92602의4 fit이다. 이전 같은 L0의 full-gradient E갱신4조건은 실제 결과3f0cbd9176f2e7882778452e537564aab7445e1b88c1534fd48b1bce7ef639a4와 독립 감사9796b4ed53f74186bc2513db56fe321c2d450cdd9bffbb7c5f341766e73ee36d로 결속해 재사용한다. baseline을 새로 학습했다고 세지 않는다. E고정·CP0/LS·Raw도 기존 원수치로 비교한다. L7이나 모든 지연 구조의 gradient 기여로 결론을 확대하지 않는다.

현재 PCA/FORECAST-LS 초기 E/D, 고정 center, K4/K2, L512/H48, FIT256/TUNE64, 원 scaler·origin·관측 mask·과거 전용 배포식, AdamW.001/WD0/clip1/batch4/600update와 CP0/100/200/400/600, 고정 FIT 관측수 가중 MSE를 그대로 유지한다. batch 순서도 seed별 이전600개와 같다. F의 **출력 tensor에만 detach**를 적용하고 E가 만드는 과거 기준값 경로를 유지한다. F forward는 갱신된 E로 매번 실제 실행하며 cached 미래나 실제 미래값으로 대체하지 않는다. F의 파라미터는 계속 동결한다.

## 실행·검산

1. 이전 source/result/보고서·자산 hash와 현재 작업을 확인하고 보존한다. 기존 실행 모듈을 수정하거나 숨은 monkeypatch로 바꾸지 않고 새 source에서 detach wrapper와 호출 경로를 명시한다.
2. 합성 검사 후 실제 CP0 probe에서 새 stop-gradient와 full-gradient의 Z/F/예측이 같고 원 canonical32 CP0와도 일치하는지 확인한다. 자료×seed의4개 첫 minibatch에서 clipping 전 D gradient가 같고, stop-gradient E gradient가 full-gradient의 직접 과거 경로 VJP와 일치해야 한다. FP32 gradient 비교는 atol1e-5/rtol2e-6, forward CP0는 같은 backend byte 일치와 기존 등록 공차를 유지한다. F 경로 VJP와 직접 경로 벡터·합산 residual·F grad None·optimizer 제외·전후 hash를 저장한다. 이 probe에서는 optimizer update를 하지 않는다.
3. 새4 fit을 초기값부터600update, 총2400update 실제 실행한다. 학습 중 F forward는 계속 실행하지만 F 입력 backward를 제거하므로 동일 update가 동일 GPU 계산량은 아니다. 전체 gradient clipping 빈도와 D의 유효 step 차이를 기록한다. probe 실패는 해당 자료/seed 경로에만 적용한다.
4. 전체 TUNE MSE/이른 tie로 고른4개의 성공/실패 잠금과 registry를 봉인한 뒤 새 DEV 배열을 추론한다. 고정 입력·정상 FIT 미래 감독과 기노출 DEV의 탐색적 조건을 그대로 밝힌다. 새 선택에 이전 DEV를 사용하지 않는다.
5. 저장 상태·예측·선정·로그·probe를 독립 검산하고 한국어 결과와 그림을 만든다. 새 계수의 배포 비용은 같은 경계의 Raw+full-gradient 두 seed+stop-gradient 두 seed를 같은 TUNE first-B cohort에서 측정한다. 2자료×5경로×B1/B8=20셀, warmup2·5완전순환(초기순서seed20261015)의100표본을 계획한다. source/hash·학습·warmup·저장은 타이머 밖이며 CPU 원단위 전처리→GPU E/F/D→물리 CPU 반환/sync를 포함한다. 이전 cohort 시간을 새 분모로 재사용하지 않는다.

결과가 비슷하면 이 조건에서 F 입력 gradient 없이도 E 적응 이득이 재현됐다고 말한다. 나빠지면 현재 최적화에서 그 gradient가 도움이 됐다는 조건부 결론이다. 어느 쪽도 F 예측값·Bolt 자체·대조학습·그래프의 필요성을 증명하지 않는다. global clipping 변화와 E/D 경로의 상호작용을 포함한 최적화 대조다. 후속 학습량 연장은 별도 판단이며 이번에 실행했다고 쓰지 않는다.

최신 상한 해제·계속 실행 지시 안에서 같은 목적을 향하는 최소 대조라는 작업 판단이다. 개별4조건을 별도로 인간 승인받았다고 주장하지 않으며 승인 질문을 반복하지 않는다. TEST·설치/다운로드·큰 백본 파라미터 학습·기존 작업 종료/중단 학습 재개·자동 commit/push·논문/PPT는 포함하지 않는다. 실패·사용 자원·미완료 및 재개 경로를 그대로 남긴다.
