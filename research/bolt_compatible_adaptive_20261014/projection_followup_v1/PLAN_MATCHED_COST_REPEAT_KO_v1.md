# 같은 전체 비용 프로토콜의 별도 반복

[확인] 첫 실제 `matched_shrunk_native_cost_v2`는 28조건·196samples를 완료했고, 동일 auditor source를 절대 --cost 경로로 호출한 `MATCHED_SHRUNK_NATIVE_COST_AUDIT_v2`가 PASS했다. 이전 감사 v1은 상대 literal JSON 경로를 absolute-path 필수 helper에 넘긴 호출 계약 오류였다. [실패 원인 기록](audits/MATCHED_COST_AUDIT_RELATIVE_PATH_FAILURE_IDENTITY_v1.json)을 보존하며 수치 공차·checksum·source·예측·타이머를 바꾸거나 학습을 다시 하지 않았다.

[확인] 첫 7-route cohort의 Beijing Raw B8은 median 10.4449625 ms/window이며 7개 sample 모두 약10.20~10.56 ms/window다. 앞서 별도 5-route cohort에서 관측한 Raw B8 median은5.4856375 ms/window였다. 두 cohort의 callback 구성/순서가 다르므로 이 차이의 원인은 미확정이다. 새 cohort 안에서 source·입력·실제 출력·타이머 검산이 통과했어도 특정 속도 비율의 재현성이 보장되지는 않는다.

[판단] 이미 실행·검산된 **같은** 7-route 프로토콜을 독립 프로세스에서 한 번 더 실행하여 비용 관측의 변동을 확인한다. 새 source나 방법을 만들지 않고 실행된 `run_matched_shrunk_native_cost_v2.py`와 `whole_matched_shrunk_native_cost_v1.py`를 그대로 사용한다. 새 tag는 `matched_shrunk_native_cost_repeat_v1`이고 기존 결과를 덮지 않는다. 같은 두 데이터·B1/B8·첫 origins·E·rho·head/CP·FIT stats·causal fill·mask·전처리·FP32 연산·physical 반환·공차를 사용한다. 모델·head 학습/선정0이다.

Warmup2/repeats7, seed20261015의 같은7 circular rotation, 28cells/196samples를 유지한다. 각 callback은 매번 관측 과거 ridge·구현 진단/input binding hash를 실제 계산한다. D cache나 seed 간 D 적합 공유를 하지 않는다. 공유 process 메모리와 CUDA counters, setup/파일 저장/SHA 제외 경계도 같다. 기존 작업을 종료하거나 benchmark를 위한 OS/GPU 설정을 변경하지 않는다. 새 heavy agent fixture 시작을 잠깐 미루는 협업 조정은 하되 단독 시스템을 보장한다고 쓰지 않는다.

두 원시 7-route cohort의 절대 시간·각 cohort Raw 기준 paired ratios·ratio of medians·모든 sample을 **각각** 보고한다. 빠른 cohort/방법만 선택하거나 첫 성공 비용을 새 반복으로 대체하지 않는다. 14번의 timing sample을 독립 데이터 generalization이나 14개 새 방법으로 세지 않는다. 기존 TUNE 정확도 및 채택 조건을 변경하지 않는다. 두 cohort의 차이가 남으면 기술적 관측 변동/미확정 원인을 보고하며 성능을 예측 오차에서 추정하지 않는다.

실제 반복이 끝나면 동일 독립 auditor source를 절대 --cost 경로와 새 `MATCHED_SHRUNK_NATIVE_COST_REPEAT_AUDIT_v1.json`으로 실행한다. 반복 SOURCE/fixture/실제 시작·완료를 혼동하지 않는다. TEST·설치·기존 job 종료/재개·큰 모델 학습·자동 Git 게시는 없다. 최신 상한 해제 지시 내의 비용 검증이며 새로운 exact-human 개별 승인을 주장하지 않는다.
