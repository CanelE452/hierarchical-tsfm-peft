# Appliances·ETT P 분기 결과

개발 V의 탐색 결과이며 실행 범위는 과거 결과를 본 뒤 정했다. 보호 T는 평가하지 않았다.
기본 학습 60/60 완료. 상태 TRAINING_COMPLETE. 연장은 하지 않았다.
최종 검증에서 거절된 RESULT 0개는 원본 bytes와 해시·사유를 별도 보존하고 수치 분석에서 제외했다.
정확히10seed가 완성된 dataset×method만 주 비교로 썼다. 불완전 실행과 불리하거나 불명확한 결과도 보존했다.

# P branch numbers for slide16

Development-V exploratory results; signed effect vs Full P.

Jena unchanged: Time branch only +0.2% (UNCLEAR); Channel branch only +8.9% (FULL_P_LOWER).

```text
Method               Dataset             Effect   Decision                    cap3_selected n
P_TIME_ONLY          appliances                -0.4% UNCLEAR                      0/10 10
P_TIME_ONLY          ETTh1                     +0.5% UNCLEAR                      0/10 10
P_TIME_ONLY          ETTh2_safe17400           -3.5% UNCLEAR                      0/10 10
P_CHANNEL_ONLY       appliances                -1.3% UNCLEAR                      0/10 10
P_CHANNEL_ONLY       ETTh1                     -0.7% UNCLEAR                      1/10 10
P_CHANNEL_ONLY       ETTh2_safe17400           -3.6% UNCLEAR                      0/10 10
```

```text
Dataset             Time_only_params Channel_only_params
appliances                      35392                3802
ETTh1                           33536                 711
ETTh2_safe17400                 33536                 711
```

Positive effect means the retained-branch variant has higher MSE. Incomplete families are not ten-seed primary results.
PROVISIONAL propagates from either comparison arm; no cap10 extension was run. Unequal capacity and joint LayerNorm/E/D coupling remain.

시간/채널 분기의 용량은 다르며 공동 LayerNorm·E/D 결합이 남는다. 분기 제거의 조건부 효과이며 축 자체의 보편적 우월성·정보 보존 메커니즘을 입증하지 않는다.
IQR 기준은 기술적 판정이며 유의성·동등성 검정이 아니다. 기존 비교 방법의 최적화 공백이나 독립 테스트 일반화를 해결한 결과가 아니다.
실행 시작 2026-10-29T10:15:04.483555+00:00, 종료 2026-10-29T18:12:49.754723+00:00. 중단 사유: 없음.
새 학습 외 모델 선택·추가seed·cap10연장은 수행하지 않았다. 발표파일·대본은 생성하거나 수정하지 않았다.
