# GTConvAE / TS2Vec 빠른 검증 완료 대사

과학 실행 **COMPLETE**, 최종 독립 검산 **PASS**, CPU 회귀 **805 passed / 0 failed / 0 skipped**. [한국어 보고서](README_KO.md), [독립 검산](VALIDATION_REPORT.json), [재현 명령](REPRODUCE.md)을 함께 제공한다.

## 두 질문의 결론

- 고정 GT에 L2048을 주면 L512 대비 MSE가 두 seed에서 각각 3.768676%, 25.740558% 줄었다. L1024는 두 seed 모두 나빴고, 고정 GT의 모든 길이는 같은 길이 Raw보다 나빴다. L2048 추가학습은 실행 범위에 없다.
- 같은 300 update 추가학습을 준 GT1024는 GT512보다 두 seed 모두 MSE/MAE가 나빴다. PRE 결합은 GT1024 대비 두 seed 평균 점수의 MSE가 8.782685%, MAE가 4.395972% 낮아졌다. RAND도 비슷하게 개선됐고 PRE 대 RAND MSE는 seed1 악화·seed2 개선이므로 사전학습 표현의 독립 이득이 일관되지는 않았다.
- 최종 8개 모델 모두 같은 L Raw보다 MSE/MAE가 높았다. GT512 seed2는 CP0보다 MSE가 낮아져도 MAE는 2.569572% 증가했다. 일부 개선을 전체 목표 달성으로 부르지 않는다.

## 실제 실행과 검증

기존 GT full 2개와 TS2Vec d320 2개를 재사용했다. 부모 재현4개, frozen13개, CP0 8개, 최종8개로 예측33개를 남겼다. 신규 예측 적응은 8개×300 updates이고 TRAIN smoke4개는 버렸다. 신규 GT reconstruction/TS2 대규모 사전학습은 0개다. 원점 pool197개를 같은 seed별 순서로 사용해 각 arm이2364개 창을 봤다. 최종 checkpoint는 VAL 점수로 고르지 않았다.

F 입력 gradient와 GT E/D/product/W의 실제 변화를 확인했고 F/TS2는 고정했다. 독립 FP64 검산은38개 비교, 채널/horizon2772행, 원점26341행, 비용42개와 반복1260행을 대조했다. CPU 회귀에서 기존 Labour 학습/과거 smoke 덮어쓰기를 일으키는 test_model_parity.py와 관련 Labour data를 읽는 test_baselines.py 두 통합 파일을 명시 제외했다. 제외 범위는 [TEST_VALIDATION](TEST_VALIDATION.json)에 있다.

private NPZ에서21개 산출물을 재생성했고, NPZ/cache 접근을 막은 별도 프로세스에서 공개 CSV로 PNG/SVG6개를 새로 그려 exact-byte 일치를 확인했다. [재현 증거](PUBLIC_REPLAY_VALIDATION.json)와 [실제 그림 검토](FIGURE_VISUAL_REVIEW.json)를 남겼다. 전역 Matplotlib 설정 상속 및 verifier의 Raw 비교8개 누락은 [렌더 수정 기록](REPORTING_AMENDMENT.json), [검산기 수정 기록](VERIFIER_AMENDMENT.json)에 이전 소스·이전 PASS proof·실패 로그 지문과 함께 보존했다. 과학 코어9개와 저장 예측·통계·그림은 검산기 수정으로 바뀌지 않았다.

## 비용·시간·한계

42개 fresh process에서 B1/B8을 측정했다. warmup420회, timed840회이며 online encoder·창별 결측 처리·scale·F/nativeD·역정규화·CPU forecast를 포함한다. GT는 시간축4배 축소, scalar 수2배 압축이며 C21과 F42 streams를 유지한다. TS2 native는320 streams다. allocated/reserved·CPU RSS·실제 F 호출/시계열 수를 [비용 표](COST.csv)에 구분했다. 일반 desktop WDDM 환경과 per-PID GPU memory N/A는 [측정 환경](COST_GPU_ISOLATION_CHECK.json)을 따른다.

누적 GPU 작업 wall은 **2943.328초 (49.06분)**이며 kernel busy 시간과 다르다. 이 완료 기록 생성까지 전체 wall은 4424.986초 (73.75분)다. 이후 Git 게시/원격 확인 시간은 최종 CLI 응답에 포함한다. cache는 이 시점 780673752 bytes로6GiB 한도 안이다.

Jena 한 데이터·두 seed·반복 개발 VAL·이 연결층·300 update의 결과다. 수렴·독립 일반화·zero-shot·통계적 유의성을 입증하지 않는다. historical raw_physical SHA 차이는 미해결이며 현재 loader/raw 및 기존6개 모델 좌표 지문 일치와 구분한다. 사전학습 overlap은 UNKNOWN이다. 이번 보호 TEST는 읽기·해시·요약하지 않았다. Fully observed history subset은 사후 보조 기술 통계이며 전체371개 주 평가 원점을 교체하지 않았다.

## 보존과 게시

[최종 보존](FINAL_PRESERVATION.json)은 protected tracked25077개, append-only 기존 history, 원 dirty checkout HEAD/status 및 발표자료2개를 실제 확인했다. 다른 실험·기존 예산은 재개하거나 초기화하지 않았다.

첫 payload는 main commit `51aaa27d3a4e8b0fa184a9c8d2172e7b8f3ba397`에 정상 게시했고, 원격181개 파일의 bytes/hash를 [실제 원격 proof](REMOTE_PROOF.json)로 확인했다. 이 완료 기록·최종 allowlist/TOP3는 이어지는 정상 closing push로 게시한다. 그 뒤 전체 allowlist를 fetched main blob과 대조한 최종 proof는 private에 남겨 자기 commit/hash의 순환을 피한다. 최종 응답에는 실제 확인한 closing commit URL을 제공한다.

계획된 과학 작업에 미완료·재개 항목이 없다. [재개 큐](RESUME_QUEUE.md)의 report/verify 명령은 재검증용이며 완료된 학습을 다시 실행하지 않는다.
