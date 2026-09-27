# v4 진행 상태

- 상태: 예정된 고정 확인·비용·최종 저장 증거 검산 완료. 새 학습/예측/TEST 재개봉 계획 없음. 관련 파일의 일반 main 게시 준비 중.
- 판정: 잔차 입력 설계 유지. 현 구성은 GPU 메모리와 정확도의 절충을 위한 조건부 선택지. 다음 판단은 미압축 F0/LoRA 대비 알려진 정확도 손해를 줄이는 개발 회차로 돌아갈지다. 자세한 범위와 수치는 TOPIC_DECISION.md에 있다.
- 기준: main/origin/main 시작 SHA 5eaeb55fda2d980055a33b3721b99cd3eab9e9d2. 기존 무관 미추적4항목 유지. 승인 PLAN/protocol 및 이전 v1/v2/v3 추적406개 hash 보존 확인.
- 승인: PLAN.md에 직전 최종 계획 응답 원문을 보존했다. 사용자 전체 승인으로 범위 내 구현·수리·실험·일반 commit/push가 허용됐다.
- 자료: Hog TRAIN 적격69/73 중 ID정렬 앞32, C32/K8. TRAIN1945/VAL30/TEST-A40/TEST-B40원점. 네 로컬 저장소 제한 검색에서 Hog 직접 예측·방법선택 흔적 미발견. 전체 이력 무사용 증명이나 백본 사전학습 비중복 보장은 아니다.
- 적합: 신경망16+CPU ridge3=19/24attempt 완료. 적합 실패·재시도0. 신경망16개 모두 정체 규칙 종료, 240epoch 연장 대상0. 합성 optimizer1/6세션, 세션 전체10update PASS. TRAIN 통계 준비는 별도 계상.
- 선택: VAL만 사용해 RES/RAW/LINEAR/COMPRESS 모두 FIXED_ED, ridge penalty0.001. RAW 두 seed는 실제768update 후step0 선택. selected.json과 evaluation_seal.json에 TEST 전에 고정.
- TEST: test_exposure.json에 최초 노출을 기록하고 v4_test_01에서19개 모델을 함께 평가했다. 전체 MSE RES3.220290, RAW/COMPRESS4.591135, LINEAR3.445439, F0 2.738044, LoRA2.683560. 이후 모델·선택·자료·주 지표 변경과 TEST 재평가 없음. Hog TEST는 이제 노출된 자료다.
- 비용: GPU66행/CPU18행 완료. RES batch4 중심357.14원점/초, peak allocated225.859MiB. 병합LoRA172.52원점/초·328.676MiB. 일부 블록 속도 순위 역전과 큰 초기 변동을 함께 보고한다. LINEAR/SHARED CPU 경로는 훨씬 저렴하다.
- 비용 수리 이력: GPU01은 LINEAR VAL24 재생의 엄격한 허용오차에서 중단. 원30원점 재생은 저장값과 완전 일치했고 재배치 미소차를 기록했다. GPU/CPU 재생 허용오차를 기존CPU/병합 기준으로 통일하고 원본·실패·진단·diff를 repairs에 보존. GPU02가 기존10행을 그대로 재사용해 나머지만 측정했다. block0는 중단된 블록이다. 최초 변동 원인은 미확정. 학습/TEST 재실행 없음.
- 추가 진단: 사전 조건을 충족해 GPU상주 대표3모델×batch1/4 진단1회 완료. 원인 확정을 위해 반복하지 않았다.
- 검산: final_checks.json PASS. 기존406개 파일, 승인/자료/백본/초기/선택 hash, 갱신·동결·복원, 저장 예측 점수와seed평균, 비용 산술·재사용행·상한을 확인. 실행팀 검산이며 독립 재현 아님. 이후 그림 글자·범례 배치만 수정하고 다시 생성·육안 확인했다. 초안 그림은 cache에 보존.
- 예산: GPU1,292.008초(21.533분)/14,400초, 게시 정적 검사 포함 CPU검사7.264초/900초, 자료통계0.942초, CPU집계·그림84.836초. 게시 검사 시 신규저장36,859,653bytes(약35.15MiB)/10GiB. publication_checks.json에 해당 시점 수치를 보존했다. 추가다운로드0, 자원충돌대기0, 활성job0. 원장은 ledger.json이며 상위GPUjob과하위fit를 중복 계상하지 않았다.
- 실행 주체: root만 자료·합성 검사·fit·추론·비용 실행. workers는 지정 코드와 읽기 검토만 수행했다. 재개 시 PLAN/STATUS·ledger/PID·완료 산출물을 먼저 확인하고 중복 적합/TEST 개봉을 피한다.
- 보존/경계: 이전 v1/v2/v3·봉인·실패·v2의22/20회 위반 보존. 새 계약으로 소급 승인하지 않았다. 다른 프로세스 종료·설치·서버·전역 환경 변경 없음. raw/가중치/예측 배열은 Git 게시 제외.
- 적용: 전역 AGENTS와 사용자 승인 PLAN을 따른다. 스킬의 개별 재승인·작업트리·프로세스 종료 예시는 이번 사용자 승인/제한으로 대체했다. 새 branch/PR/repo·force·hook 우회 없음.
