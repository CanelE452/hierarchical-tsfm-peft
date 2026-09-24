# 난이도 가설 (격차 지도 Stage 2)

계약 §6 S2c. 격차가 가장 몰린 자리 최대 3개를 가설로 적는다. 방법 제안은 쓰지 않는다.

## 해당 없음

두 기간 모두 격차가 유지된 후보가 없어 위치 분석을 하지 않았다. STABILITY.csv 의 판정은 다음과 같다.

| id | 설정 | 역할 | 판정 | G_P1 | G_P2 |
| --- | --- | --- | --- | --- | --- |
| C1 | bitbrains_rnd/5T/medium | candidate | UNSTABLE_GAP | -0.04751080744949078 | 0.29586964185359377 |
| C2 | bitbrains_rnd/H/short | candidate | INFEASIBLE |  |  |
| C3 | bitbrains_fast_storage/H/short | candidate | INFEASIBLE |  |  |
| C4 | bizitobs_l2c/5T/medium | candidate | NO_GAP | 0.1143814538748599 | 0.1599418062289438 |
| N1 | bitbrains_rnd/5T/short | control | NO_GAP / CONTROL_OK | -0.03674242166127821 | 0.05001043965996253 |
| N2 | bizitobs_l2c/H/medium | control | INFEASIBLE |  |  |
| N3 | electricity/H/short | control | NO_GAP / CONTROL_OK | -0.0026295737966300424 | 0.0644319399036857 |

가설을 쓰려면 먼저 재현되는 격차가 있어야 한다. 이 단계에서는 없었다.
