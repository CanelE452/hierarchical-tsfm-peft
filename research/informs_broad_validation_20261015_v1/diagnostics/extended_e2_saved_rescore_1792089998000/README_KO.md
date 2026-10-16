# 첫 E2 검산 시도의 형식 가정 오류

[확인] 첫 실행은 E2 wrapper가 만들지 않는 개별 `SCORES.csv`를 필수로 요구해 metadata preflight에서 중단됐다. FP64 재채점과 모델·GPU 실행은 0이다. 원 실행 driver와 실패 영수증을 보존했다. [공개용 실패 결과](RESULT_PUBLIC.json)는 오류 문자열의 개인 절대경로만 의미가 같은 설명으로 바꿨으며 [파생 증거](PUBLIC_PROJECTION_PROOF.json)에 원본·파생 SHA와 변경 필드를 남겼다. 원 `RESULT.json`은 공개 allowlist에서 제외한다.

후속 검산은 원 manifest 점수·public band 통계·combination receipt의 점수를 실제 대조했다. 이 첫 실패는 모델이나 기존 E2 예측 결과의 오류 판정이 아니다.
