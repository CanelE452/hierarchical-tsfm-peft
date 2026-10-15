# Raw subset 비용 감사의 문서 형식 처리 수정

[확인] 실제 비용 producer는 `runs/raw_subset_cost_v1/RESULT.json`에 COMPLETE 40조건·400샘플을 남겼다. SHA는 `60eb05ebdd1a7f24d82fa58d4dcab192c5c4684552882d1f61f175fc0585a23c`이며 실패 경로는 없다. 비용 실행 자체의 재측정이나 수치 수정은 하지 않는다.

[확인] 원 감사 `audit_raw_subset_cost_v1.py`(SHA `a91ee9a4ca1de36d53c0a01b353c0655e6d281944d1e8190789927502775f7b8`)의 실제 호출은 JSONDecodeError로 실패했다. `audits/RAW_SUBSET_COST_AUDIT_v1.json` SHA `42081d93049e78aaa8b313a00dd28f68e292129f9ac3b036832889137c6fb6c2`와 자원 영수증을 보존한다. ROOT가 같은 원 함수의 메타데이터 검사만 재호출해 traceback의 167행을 확인했다. Markdown인 numeric_repair_addendum을 다른 JSON parent들과 함께 verified_json에 넘긴 형식 처리 오류다. 이 재현은 모델·훈련·타이머를 실행하지 않았다.

[판단] 새 `audit_raw_subset_cost_v2.py`에서는 여섯 JSON parent를 기존 검증기로 읽고, Markdown parent는 절대 경로·확장자·존재·SHA·등록된 경우 파일 크기를 확인한다. Markdown을 JSON으로 해석하지 않는다. 원 감사의 다른 함수, 예측·타이머·공차·샘플·선정 조건은 바꾸지 않는다. 실제 형식 Markdown에서 원 오류를 재현하고 잘못된 JSON/Markdown SHA 및 크기 거부를 포함한 합성 검사 17개를 통과했다. ROOT는 원 전체 코드와 v2 차이를 읽었다.

v2의 실제 감사는 새 `audits/RAW_SUBSET_COST_AUDIT_v2.json`에 기록한다. 이 문서 작성 시점에는 해당 실제 감사가 아직 실행되지 않았다. 원 실패를 PASS로 덮거나 비용을 다시 재서 대체하지 않는다.
