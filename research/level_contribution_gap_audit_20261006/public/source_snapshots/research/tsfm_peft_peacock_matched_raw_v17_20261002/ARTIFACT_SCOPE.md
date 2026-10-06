> 공개 사본: 개인 경로·채팅 식별자와 링크를 정리했습니다. 원본 및 사본 해시는 감사 공개 폴더의 `publication_manifest.json`에 분리 기록합니다. 아래 상태·검산은 원래 감사/실행 시점의 기록이며, 이번 게시 검증이 아닙니다.

# TEST 노출 후 보충 대조 (v12 확인 실험 아님)

이 폴더의 모든 코드, JSON, CSV, 로그, checkpoint, NPZ와 receipt에 이 범위 표시가 적용된다.
JSON 객체/CSV 행/NPZ에는 scope를 직접 기록한다. 원본 알고리즘 보존을 위해 구조를 바꾸지 않은 curve 배열·초기 `.pt` 복사본은 이 파일과 최종 artifact manifest의 scope를 함께 읽는다.

`family='level'`은 **복사한 v12 고정 PCA 학습 경로를 고르는 내부 키**다. v17의 spec `role='matched_raw'` 및 새 checkpoint schema가 실제 모델을 식별한다. v17 적합을 원래 LEVEL 재학습이라고 해석하지 않는다.

## 고정 수식

`b = D(F0(E(X)))`, `r = X-D(E(X))`. E/D는 저장 TRAIN PCA로 고정한다.

- 기존 LEVEL: `b + repeat48(last(r)) + G(r-repeat512(last(r)))`
- 보충 RAW: `b + repeat48(last(r)) + G(X-repeat512(last(X)))`

G는 공유·무편향·무활성화 512→32→48, 학습 변수 17,920개. 수준 복원은 양쪽 모두 **잔차의 마지막 수준**이다. 미래값/미래 통계는 G 입력에 없다.

`train_confirmation_v12.py`의 Python AST는 원본과 동일하다. 모델의 수식 변경은 G 입력 인수 하나이고, 나머지 diff는 schema/설명/범위 표기다. `runtime_confirmation_v12.py`는 새 출력 경로·새 원장·이번 상한을 적용하며 원래 numeric 함수들을 그대로 사용한다. TEST 점수와 재표집도 v12의 순수 함수를 그대로 사용하고 LEVEL−RAW 역할만 연결한다. 전체 소스 diff가 한 줄이라고 주장하지 않는다.

## 승인된 실행 계약

두 seed 92601/92602, LR .001, AdamW wd0, clip1, batch4, 위상24·epoch당512 TRAIN 원점, 최대120epoch, strict VAL 비개선6회, ReduceLROnPlateau .5/patience2/relative threshold1e-4. step0 포함 전체 VAL 최소·동률 이른 checkpoint. 튜닝 없음.

v12 초기 G bytes를 복사했으며 PCA·표준화 통계를 새로 적합하지 않았다. 저장 LEVEL 예측 재채점 검증을 선행했다. OLD 원장의 예산/실험 계약을 확장하거나 봉인을 다시 연 것으로 해석하지 않는다.

모든 기존 자료는 노출된 평가다. 이번 봉인은 RAW 학습과 TEST 채점의 순서를 기록하며 독립 확인을 새로 만들지 않는다.

## 사전 결정

결합 TEST `100*(LEVEL/RAW-1)`의 조건부 95% 구간:
- 상한<0: Peacock 잔차 입력 제한 이득.
- 하한>0: Peacock 원입력 버전이 더 정확.
- 그 외: Peacock 차이를 가르지 못함.

TEST-A/B 방향이 다르면 반드시 덧붙인다. 기간·채널·seed를 재선택하지 않는다. 7원점/2,000회/seed9262026, unit_index0, 기간별 SeedSequence를 원본 그대로 사용한다. seed 손실 평균이며 예측 ensemble이 아니다.
