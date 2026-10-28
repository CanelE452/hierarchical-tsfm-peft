# E 학습 기여를 분리하는 Jena 5-seed 대조

이 연구는 기존 RUN_PLAN의 확장이 아닌 별도 사전등록이다. Jena K6, seed92601–92605에서 원 B2를 같은 초기화로 만든 뒤 E.requires_grad만 False로 바꾼다. P/frozen F/lag7 D/anchor는 그대로 두고 학습한다. 같은 기존 Full5를 재사용한다.

A_fit0–50%/A_stop50–60%, checkpoint0/.25/1/3 최소 A_stop 선택(정확동률이면 이른 epoch), 선택봉인, fresh A60 refit, 개발 V70–80% 평가를 유지한다. Full5는 cap3선택0/5다. E-fixed5중3개이상 cap3이면 전5 cap10 연장, cap10도3개이상이면 학습상한 미해결로 표시한다. 보호T는 열지 않는다.

search와 refit 각각 PCA를 허용 학습구간에서 새로 계산하고 E 초기/최종/선택값이 정확히 같아야 한다. 두 단계의 PCA가 서로 같은 값이어야 하는 것은 아니다. 실제 optimizer parameter groups에서 E 제외와 gradient 부재를 검증한다.

시작 전 기본5×18분+가능연장5×30분+2분=4시간2분을 남겨야 한다. 기본 추정으로13:28KST 이후에는 새 study를 시작하지 않는다. 학습17:30, 전달20:00 마감이다. 시간/A_stop만 입장·연장을 결정한다. 예산으로 시작하지 못하면 NOT_RUN_BUDGET이며 근거가 확보됐다고 말하지 않는다. 기술적 동일설정 재시도는1회, 과학 게이트 실패는 중단한다.

주캠페인 종료기록과 실제 프로세스 부재, root gate, 공유GPU락, 살아있는 parent lease/ancestry를 모두 확인한다. parent는 기존controller.lock, 실제 GPU child는 기존worker.lock과 자기worker.lock을 보유해 parent가 종료돼도 GPU 중복을 막는다. CUDA 로딩 전 공통 host gate로 사전봉인한 desktop PID·생성시각·실행경로·실행파일 SHA만 허용한다. 새 프로세스·미확인·Python·WSL GPU context는 거부한다. 각 worker 시작 시 검사하며 영수증과 비공개 identity snapshot을 보존한다.

비교는 같은 seed의 E_FIXED−OURS이며 양수이면 E 학습이 도움됐다는 뜻이다. 모든5점·paired차이·median/IQR·실제선택epoch/시간/파라미터를 보존한다. 새 결과는 반복 사용한 개발V의 탐색이며 n5로 동등성/독립일반화/모든설계의필연성을 주장하지 않는다. PCA나K의 최적성은 이 실험의 질문이 아니다. 불리하거나 불명확한 결과도 그대로 보고한다.

## 운영 개정 AMENDMENT_001

원 등록과 소스는 revisions/preflight_001에 SHA와 함께 보존했다. WDDM 데스크톱 context 때문에 빈 GPU PID 조건이 실행 불가능해, 학습 전 정확한 호스트 식별정보 검사로 운영 조건만 개정한다. 모델·가설·seed·K·초기화·데이터 구간·학습·선택·연장 규칙은 바꾸지 않았다. 기본 전체 예약14520초와17:30 학습 마감도 같다.

이 실험은 속도 benchmark가 아니므로 비용 측정용60초 부하 상한을 적용하지 않는다. 정확히 등록한 desktop 동시 부하는 허용하고, 미확인 context나 학습/비용 프로세스 충돌은 계속 거부한다. 공유lock·live parent lease·root launch gate는 유지한다.

공통 helper SHA256: 944936b5a1b025de45d6cc470ea5301bc73c77f539794a56668aece58c10e898

공개 host 계약 SHA256: 831738eb0fe187af50f142b4d9b85a261f6337d1222131272c26bf165c97b844

기존 CPU 시험109개와 공통 helper 시험125개를 각 소스 SHA에 연결했다. 실제 E 학습은 아직0회이며 새 root 봉인·사전검증·launch gate 전에는 시작하지 않는다.
