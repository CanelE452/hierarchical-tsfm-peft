> 게시 사본입니다. 원문 작성 당시의 로컬·준비·미실행 상태는 역사로 보존합니다. 최신 완료/미해결 상태와 게시 범위는 [색인](../README_KO.md)을 확인하세요. 과학 원본은 바꾸지 않고 링크와 이 안내만 추가했습니다.

# 고정 upstream을 조건으로 한 보정의 시간 이전 진단

상태: 실행 전 확정 계획. [다음 판단 메모](PROFESSOR_METHOD_NEXT_DECISION_KO_v1.md)의 한정 프로토콜을 채택한다. 사용자의 같은 목적 지속 실행·추가 승인 질문 금지 지시 범위에서 로컬 구현·실행·검산·보고를 진행한다. 별도 승인 대기가 아니다. 원본·기존 작업 보존, 보호 TEST·설치·다운로드·큰 백본 학습·자동 Git 금지는 그대로다.

**질문:** 같은 고정 PCA K8 또는 Raw-C의 F 특징에서 앞 구간으로 적합하고 중간 구간에서 선택한 STATIC ΔD의 이득이 뒤 구간에도 남는가? 새 architecture·성공 기준·DEV winner를 만들지 않는다. 기존 노출 FIT 안에서 수행하는 적응적 조건부 진단이며 독립 holdout이나 전체 temporal validation이 아니다.

Robin FIT256만 쓴다. 기존 full-FIT scaler·E/center·RECON D·median F cache는 그대로 둔다. 이 upstream이 A 밖 구간까지 사용한 사실을 명시한다. 새 readout의 RMS와 계수만 A에서 적합한다. 원 TUNE/DEV/TEST 배열을 열거나 새 F/GPU/optimizer를 실행하지 않는다. 기존 hD·cross-moment science/audit 결과 및 source/hash gate에 결속한 FIT cache와 원 FIT target/mask/origin만 사용한다.

실제 origin은 timestamp 순서·유일성을 확인하고 앞128개=A 후보, 다음64개=B 후보, 마지막64개=C로 고정한다. 기존 데이터 계약의 origin o는 미래 첫 시점이며 response는 o,…,o+47이다. A는 o+47<첫 B origin, B는 o+47<첫 C origin 조건으로 purge한다. C 시작이나 K/함수/alpha는 결과를 보고 바꾸지 않는다. purge 전후 origin 수, 실제 timestamp 경계, 고유 response 시점·반복수와 채널별 관측수를 기록한다. 과거 history 공유는 허용하며 L+H 길이의 완전 분리라고 부르지 않는다.

PCA_K8/RAW_C 두 경로의 STATIC만, 기존 alpha `0,1e-4,.001,.01,.1,1,10,100,1000,10000`와 ZERO를 비교한다. A actual-F ungated φ의 uncentered RMS, observed channel-macro normalized MSE+ridge, 절편 없음, 실제 FP32 baseline+correction 산술과 동률 순서를 기존과 같게 유지한다. 두 경로 모두 B 점수로 잠근 뒤 C 출력을 생성한다. 계획량은2fit·22후보·44 A/B 평가·선정2 C 평가다. 일부 경로 실패는 보존하고 독립 경로를 계속한다.

주 결과는 C의 PCA_selected−PCA_ZERO, Raw_selected−Raw_ZERO, PCA_selected−Raw_selected MSE 차이다. B도 함께 보여 이전 여부를 읽는다. per-C·사전1–16/17–32/33–48 구간·actual endpoint interaction/energy는 보조 진단이다. 분할·horizon·채널을 독립 반복으로 세거나 p값을 만들지 않는다. 양성 결과도 기존 DEV 실패·Raw gap·독립성 부족을 없애지 않는다. 음성이면 더 복잡한 D를 자동으로 추가하지 않는다.

새 계수 비용은 이번 범위에서 미측정이다. 이전 상태 timer를 새 계수 정확도에 붙이지 않는다. 원 결과를 덮지 않는 새 run·선정 잠금·자원 영수증·source/input 전후 hash·검산·한국어 보고서/그림을 남긴다. 실제 자산이 계약을 만족하지 못하면 그 이유와 상태를 남기며 없는 origin/점수를 추정하지 않는다.
