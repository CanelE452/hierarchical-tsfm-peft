# T를 보기 전 확정하는 대표 비교 규칙

이 문서는 학습이 진행 중이고 `SELECTION_LOCK.json`이 없는 시점에 적은 최종 비교 규칙이다. 최종 ID와 정확한 V manifest·artifact SHA는 학습 종료 뒤 독립 감사·체크포인트 재추론을 통과한 실제 자산으로만 봉인한다. 이 문서 자체는 선택 잠금이나 T 실행 영수증이 아니다. [전체 실행 계약](CAMPAIGN_PLAN.md)과 [평가 API 계약](FINAL_EVALUATION_PROTOCOL_KO.md)을 따른다.

## 선택에 사용하는 정보

- 기준 지표는 A60 정규화 원채널의 observed channel-macro MSE다. MAE와 전체 파이프라인 비용도 함께 보인다. Bolt의 point 출력은 quantile 0.5이며 바꾸지 않는다.
- checkpoint는 각 학습 경로의 A_stop 규칙으로 고정된다. 가족·K·등록된 학습 endpoint 간 최종 후보 선택에는 V와 V 과거 입력으로 측정한 비용만 사용한다. T 결과·T 사례·T 패턴은 선택에 사용하지 않는다.
- 학습량·학습률·적합 구간이 다른 경로는 별도 조건이다. V에서 가장 낮은 경로를 선택해도 동일 튜닝 예산, 시간 앞단 P의 순수 효과 또는 충분 수렴이라고 해석하지 않는다.
- 동점은 정규화 MSE, MAE, experiment_id의 오름차순으로 결정한다. 실제 비용이 없으면 비용을 추정해 우위를 부여하지 않고 미측정으로 남긴다.

## 보존할 비교

접근 가능한 주 자료 Beijing·Jena·Electricity·PeMS08을 각각 평가한다. 서로 다른 단위·horizon의 점수를 단순 평균하지 않는다. M5의 인증 차단과 Robin의 짧은 스트레스 결과는 이 네 자료의 증거를 대신하지 않는다.

1. Raw-small/mini/tiny, DIRECT_G, LoRA, native ridge, CURRENT/SEASONAL을 유지한다. 별도 A+M 적합 G/LoRA는 더 많은 적합 정보를 가진 강한 대조로 표시한다. 그 점수를 A-only 부모의 점수와 같은 정보량이라고 부르지 않는다.
2. B0/B2의 quarter/half/C와 기존 작은 K, B4의 quarter/C, 고정 PCA 및 직접 B2/B4·AdaPTS 원리 포트를 보존한다. 같은 학습 endpoint의 B2−B0와, V로 고른 최선 경로의 비교를 따로 제시한다. 후자는 P만 바꾼 대조가 아니다.
3. E2에서는 전역 결합의 V 선택 경로와 B2/B0/Raw 조건부 H, 같은 용량의 과거 H, 기존 잔차 방식, Raw bias 및 A+M 대조를 남긴다. alpha0/1은 부모와 같은 계산 경로인 alias임을 표시하며 독립 학습으로 세지 않는다. 모든 V 계수·부모·기준선 점수는 보존한다.
4. E3의 Beijing·PeMS08 causal/bidirectional quarter/half/C를 유지한다. 둘 다 허용 과거 창 안의 시간 정보이며, directed graph 학습이나 인과관계 식별이라고 부르지 않는다.
5. E4의 Beijing·Electricity에서 shared multi-K, posthoc prefix, nested PCA 및 독립 K를 보존한다. 같은 checkpoint의 prefix들은 별도 학습이 아니다. 동일 총 update와 각 K의 동일 노출이 동시에 맞았다고 주장하지 않는다.

실제 잠금 ID 목록에는 중복 alias 정리, 실패 경로 및 자원 때문에 제외한 경로의 이유를 명시한다. 어떤 원리의 V 점수가 낮지 않다는 이유로 그 원리 전체를 누락하지 않는다. 선택된 일부 경로의 T 완료를 등록된 전체 grid·세 seed·두 번째 cutoff·채널 subset·장기 학습 완료로 바꾸어 쓰지 않는다.

모든 실제 V 점수·K 곡선·학습 상태와 강한 기준선의 통계는 보존한다. T의 정확한 후보 수는 남은 실제 추론 자원과 숫자를 보존하는 공개 통계의 250MiB 상한을 함께 확인해 잠금 전에 확정한다. T는 선택 경로의 등록된 전체 원점·전체 채널을 사용하며, 게시량을 맞추려고 T 원점이나 채널을 사후 표본추출하지 않는다. K별 V 분석과 선택된 K의 T 분석은 해당 split을 명시한다. 미선택 경로의 T는 미실행으로 남긴다.

## 실행 직전·직후 확인

실제 GPU 소유권을 인계받고 학습을 종료한 뒤, 선택 후보의 저장 V 독립 재채점과 고정 origin checkpoint 재추론 PASS를 확인한다. 살아 있는 training lease 또는 RUNNING 행이 있으면 잠그지 않는다. 정확한 감사 경로를 decision에 지정하고 portable route·부모·artifact·source를 봉인한 다음 공식 `execute_T` API로만 실행한다.

기존 CPU subset replay의 재사용과 새 GPU 재추론을 나눈다. 체크포인트 replay는 전체 optimizer 재실행이나 전체 V 재추론이 아니다. 하나의 실패는 해당 경로에 기록하고 독립 경로를 계속한다. 예산 경계는 DEFERRED_RESOURCE로 남긴다.

모든 V/T 산출이 끝날 때까지 root `SCORES.csv`를 만들지 않는다. 최종 report의 전체 점수표를 모은 뒤 root 점수표와 모든 manifest의 run/split/cohort 대응을 독립 감사한다. 비용은 저장 예측 읽기 시간이 아닌 실제 물리 입력의 P/E/F/D/G/H·역변환을 포함하며, fresh/shared 및 B1/B8을 구분한다.

E0의 옛 B는 역사적 개발 후속으로 별도 재추론·검산한다. E0 결과를 새로운 T의 선택에 되먹이지 않는다. 자원 내 미해결 학습, 마지막 checkpoint best, 자료 부족 및 보조 검사 과정의 보호 자료 접근 사고는 각각 숨기지 않고 보고한다. 최종 평가 통과는 독립 미관측 일반화나 방법의 신규성 증명이 아니다.
