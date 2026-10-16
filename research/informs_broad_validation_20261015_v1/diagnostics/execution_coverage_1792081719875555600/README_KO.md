# 실행·최소 학습 커버리지 관찰

[확인] 현재 registry와 실제 완료 fit/stage/V manifest를 읽기 전용으로 대조했다. `COVERAGE.json`은 당시 실제 상태이며 학습·예측 재실행, checkpoint 검산, T 검증 또는 전체 완료 선언이 아니다. 초기 minimum .25 breadth, §7.3의 .25→1→3 최소 학습, 기존 51그룹 epoch1 계획, G/B0/B2만의 기존 epoch3 의도, 현재 Beijing 대표 심화, 시간 전이12, full-grid/lag/subset 확장을 별도 집계했다.

정확한 원래 51그룹/70 fit 목표와 미실행 ID·재개 명령은 COVERAGE.json, 전체 초기 핵심가족별 실제 lineage 최대 학습량과 미완료는 INITIAL_CORE_DEPTH.csv에 있다. `__e3`가 실제 1 epoch 검색도 포함해 수행했어도, 원 계획의 `__e1` artifact가 생성되었다고 바꿔 세지 않는다. 새 선택 모델의 학습량과 최대 검색 endpoint도 구분한다. A+M control은 A-only 부모 학습 완료로 대체하지 않는다.

등록되지 않은 E2 extended-parent 28 출력 계획과 아직 실물화하지 않은 G/B0/B2 epoch3 ID는 계획으로만 표시한다. 기본 E2 최소를 완료했더라도 그 새 부모 계획이 실행되었다고 세지 않는다. 모델 이름·alias·추가 seed·확장 K를 독립적 과학적 성공 횟수로 세지 않는다.

현재 GPU는 ROOT 소유이며 이 진단은 GPU/학습을 시작하지 않았다. 입력을 상대 경로+SHA로 봉인했고, 동시 실행 중 변경된 메타데이터는 별도 목록으로 남긴다. 미실행 조건을 성능 음성으로 삭제하지 않는다. 자원 추가 근거에는 ACTUAL_STAGE_DURATIONS.csv의 실제 세션 시간만 사용하고, 다른 C/K·최대3 refit·배포비용까지 보장하는 상한으로 외삽하지 않는다.

원 deepen3 의도는 소스197–202행의 4자료 G와 B0/B2 quarter/half/full C, 정확히 16그룹·28 fit이다. 처음 진단에서 epoch1 계획의 Robin/기존작은K까지 기계적으로 투영한 34개 수치는 이 스냅샷에서 교정했다. 기존 잘못된 스냅샷은 보존하되 최종 보고/게시는 이 교정본만 사용한다.
