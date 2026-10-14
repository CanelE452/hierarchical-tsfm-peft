# 승인된 적응형 캠페인 종료 상태

상태: **실행 가능한 진단·실제 추론·비용·독립 검산·한국어 보고 완료, 사전 TUNE gate에 따른 후속 학습 분기 종료**. 연구 전체나 작은 K 방법의 성공을 선언하는 상태가 아니다. [최신 결과 보고서](reports/first_and_dense_v3/REPORT_KO.md), [교수님용 6장 스토리 MD](reports/first_and_dense_v3/PRESENTATION_BRIEF_KO.md), [선행 감사](PHASE0_AUDIT_KO.md)를 함께 읽는다. 이전 `RUN_STATUS.md`와 승인안의 대기 문장은 당시의 Phase 0 원본 기록으로 보존했다.

## 답과 실제 실행

SELF33와 JOINT33의 시간 요약·현재값 anchor·학습 목표·표본·관측 마스크·원 FIT 정규화·특징 정규화·7α·공유 두 통제·전체 macro TUNE 선택을 맞췄다. 강한 SELF512와 Raw Bolt도 같은 원수치 계약으로 비교했다. p33 대 p33C의 용량·규제 차이는 남으므로 채널 정보만의 인과 효과가 아니다.

- Robin FIT256: TUNE MSE SELF33 .139777591 / JOINT33 .262253305, joint MSE 87.62% 악화, MAE 34.32% 악화, 개선 블록0/4.
- Beijing FIT256: TUNE MSE SELF33 .843454174 / JOINT33 .953965639, joint MSE 13.10% 악화, MAE6.57% 악화, 개선 블록1/4.
- Robin의 한 번의 FIT512 진단: .140339388 / .260887098, joint MSE85.90% 악화, MAE33.59% 악화, 개선 블록0/4. 같은 FIT 시간 구간의 재표집이며 기존256과 교집합128개다. 독립 학습 표본을 두 배로 늘린 실험이 아니다.
- 반례: Beijing 기노출 EVAL의 joint MSE .745596667은 self .781774393보다 좋지만 MAE는 .521992208 대 .484819007로 나쁘다. 이 반례로 사전 TUNE 분기를 뒤집지 않았다.

현재 조건의 단순 다변량 readout 이득을 지지하지 못했다. 공통 신호의 부재, 모든 비선형/저차원 방법의 실패, Bolt의 불필요성을 증명하지 않았다. Raw Bolt는 두 개발 구간에서 더 정확한 기준이지만 이번 결과만으로 압축 경로에서의 Bolt 고유 기여나 독립 일반화를 입증하지 않는다.

유효 신규 ridge42묶음·재사용14묶음·72pool, 후보 FIT/TUNE 저장 예측112개·선택 예측24개를 독립 검산했다. 배열 순서 문제로 무효가 된 최초 dense14묶음을 포함한 물리 실행은 신규56·재사용14·90pool이다. 결정적 ridge의 중복을 신경망 seed나 독립 반복으로 세지 않았다. 이번 신규 신경망 학습은0이며, 이전 E/D·TS2Vec/GTConvAE·LEVEL·native 비교는 기존 결과로 구분해 재사용했다.

실제 비용은 첫24조건과 수정된 dense CPU4조건, 합28조건·168회 저장 출력이다. 원단위 과거의 causal forward-fill부터 CPU 원단위 반환까지 측정했다. 별도 CPU trial의4조건과 GPU/여러 계수가 상주하는24조건 사이의 메모리·속도 우위를 주장하지 않는다. [첫 수치 검산](audits/FIRST_READOUT_AUDIT_v2.json), [수정 dense 검산](audits/EXPANDED_READOUT_AUDIT_v2.json), [24비용 검산](audits/COST_AUDIT_v2.json), [4비용 검산](audits/EXPANDED_COST_AUDIT_v1.json)은 모두 PASS다. 검산은 타이머·GPU peak의 독립 재측정이나 새로운 평가 자료 검증이 아니다.

## 수정·보존과 미실행 이유

원 dense history의 값은 같았으나 저장 전후 메모리 배열 순서가 달라 FP32 블록 평균이 달라졌다. 무규제 joint의 저장 예측 재현 차이가4.37498e−5로 기존 기준을 넘었다. [실제 진단](audits/DENSE_LAYOUT_DIAGNOSTIC_v1.json) 후 C 배열로 통일하고, 같은 원수치·분할·목표·α·gate로 새 시도를 학습·검산했다. 허용 오차는 넓히지 않았다. 원 결과/소스/실패 receipt를 보존했으며 이 수치 오류가 기존 표현학습 성능 실패의 일반 원인이라고 주장하지 않는다.

새 ForeCA·linear E/D는 선행 공통 정보 gate 미통과로 미실행이다. 새 native head·동등 RAW33 잔차 결합은 추가 우선순위와 선행 공통 경로의 근거가 없어 이번 분기에서 실행하지 않았다. 원 MAIN84fit를 중복하지 않았다. 새 긴 Raw 입력도 이번 분기 종료 후 확대하지 않았다. Jena/Appliances·대조학습·동적 그래프·future-conditioned p49K·새 TEST는 이번 진단의 신규 실행이 아니다. 각 미실행 경로에 성공/실패 수치를 부여하지 않는다.

다음 판단은 공통 정보 가설을 지지하는 사전 선정 자료/변수 계약 또는 저차원·비선형 대조의 근거를 별도로 정하는 것이다. 원래 목적·자료·비용을 바꾸는 새 범위를 합의하기 전 자동 탐색을 늘리지 않는다. 현재 승인 범위 안에서 반복 승인을 요구하지 않고 끝까지 실행했으며, 종료 이유는 예산 소진이 아닌 사전 gate다.

## 자원과 보호 범위

[보고서 생성 후 실제 자원 snapshot](RESOURCE_SNAPSHOT_AFTER_REPORT_v3.json)의 2026-10-14 20:55:56 KST 기준 CPU process96.79초 / 승인7200초, GPU process40.01초 / 승인5400초, wall41.32분 / 승인240분, 새 저장 약233MiB / 승인6GiB다. 이후의 최종 파일 검산 시간이 별도 receipt에 추가되며 이 snapshot을 최종 합계로 바꾸어 말하지 않는다. 고정 시계는20:14:37부터 다음날00:14:37까지며 리셋하지 않았다. 당시 활성 lease와 연구 Python 프로세스는0이었다. 원본 source/이전 auth·manifest16개 SHA와 HEAD6454812를 확인했다.

행정 bootstrap 실패1건과 Python 실패 receipt5건을 모두 보존했다. 주요 원인은 Beijing origin metadata schema, 잘못 선택한 기본 Python의 chronos 부재, checked-file/array-manifest schema, dense 배열 순서, 보고서 capacity schema다. 보고서 source의 실행 후 후보도 보존하고 실제 사용 v1은 snapshot SHA로 복구했다. [실행 시도 CSV](reports/first_and_dense_v3/tables/EXECUTION_ATTEMPTS.csv)에 각 실패와 수선의 실제 상태가 있다.

보호 TEST·기존 job 종료·기존 중단 실험 재개·설치/다운로드·대형 backbone 학습·commit/push·실제 논문/PPT 작성은 하지 않았다. 배포 입력은 과거만 받는다. 합성 미래 target/mask 교란 검사와 실제 origin·FIT 통계·과거 입력 계약·저장 산술 검사를 구분하며, 실제 원자료 전체의 미래를 교란한 포괄적 누출 검증으로 확대하지 않는다.

시각자료8 PNG·8 SVG를 직접 확인했고 최종 v3 PNG8개는 확인한 v2와 byte SHA가 모두 동일하다. [최종 보고서 파일 검산](audits/FINAL_REPORT_ARTIFACT_AUDIT_v2.json)은 현재 보고서의 CSV·수치·상대 링크·그림·SHA를 확인하는 별도 결과다.
