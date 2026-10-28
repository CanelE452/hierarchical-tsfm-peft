# E 학습 기여를 분리하는 Jena 5-seed 대조

이 연구는 기존 RUN_PLAN의 확장이 아닌 별도 사전등록이다. Jena K6, seed92601–92605에서 원 B2를 같은 초기화로 만든 뒤 E.requires_grad만 False로 바꾼다. P/frozen F/lag7 D/anchor는 그대로 두고 학습한다. 같은 기존 Full5를 재사용한다.

A_fit0–50%/A_stop50–60%, checkpoint0/.25/1/3 최소 A_stop 선택(정확동률이면 이른 epoch), 선택봉인, fresh A60 refit, 개발 V70–80% 평가를 유지한다. Full5는 cap3선택0/5다. E-fixed5중3개이상 cap3이면 전5 cap10 연장, cap10도3개이상이면 학습상한 미해결로 표시한다. 보호T는 열지 않는다.

search와 refit 각각 PCA를 허용 학습구간에서 새로 계산하고 E 초기/최종/선택값이 정확히 같아야 한다. 두 단계의 PCA가 서로 같은 값이어야 하는 것은 아니다. 실제 optimizer parameter groups에서 E 제외와 gradient 부재를 검증한다.

시작 전 기본5×18분+가능연장5×30분+2분=4시간2분을 남겨야 한다. 기본 추정으로13:28KST 이후에는 새 study를 시작하지 않는다. 학습17:30, 전달20:00 마감이다. 시간/A_stop만 입장·연장을 결정한다. 예산으로 시작하지 못하면 NOT_RUN_BUDGET이며 근거가 확보됐다고 말하지 않는다. 기술적 동일설정 재시도는1회, 과학 게이트 실패는 중단한다.

주캠페인 종료기록과 실제 프로세스 부재, root gate, 공유GPU락, 살아있는 parent lease/ancestry를 모두 확인한다. parent는 기존controller.lock, 실제 GPU child는 기존worker.lock과 자기worker.lock을 보유해 parent가 종료돼도 GPU 중복을 막는다. 실행 전 nvidia-smi compute PID가 비어 있어야 한다.

비교는 같은 seed의 E_FIXED−OURS이며 양수이면 E 학습이 도움됐다는 뜻이다. 모든5점·paired차이·median/IQR·실제선택epoch/시간/파라미터를 보존한다. 새 결과는 반복 사용한 개발V의 탐색이며 n5로 동등성/독립일반화/모든설계의필연성을 주장하지 않는다. PCA나K의 최적성은 이 실험의 질문이 아니다. 불리하거나 불명확한 결과도 그대로 보고한다.
