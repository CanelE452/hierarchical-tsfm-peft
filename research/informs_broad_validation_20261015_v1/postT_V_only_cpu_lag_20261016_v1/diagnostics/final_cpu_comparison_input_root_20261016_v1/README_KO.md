# Jena V43·CPU 배포 비교의 고정 집계 입력

[확인] 7개 명시 비교군의 저장된 전체 V43 점수, 해당 manifest에 결박된 FP64 검산, 실제 fresh CPU6 checkpoint 재추론, 독립 CPU 비용 산술 감사의 timed20 집계만 연결했다. 이번 작업은 메타데이터 조립과 렌더러의 stdlib `validate_rows` 검사이며 새 추론·학습·비용 계측·그림 생성이 아니다.

- [51열·7행 원정밀도 집계](AGGREGATE.csv)
- [입력 파일 SHA·조건별 증거·범위](SOURCE_BINDINGS.json)
- [고정 렌더러](../../render_final_cpu_comparison.py)
- [원 CPU 비용 산술 감사](../root_cpu_cost_arithmetic_20261016_v1/AUDIT_RESULT.json)
- [원 저장 예측 FP64 감사 결과](../../../audits/audit_1792101640838866500/PREDICTION_AUDITS.json)

원 precision을 가진 per-run SCORES.csv의 MSE/MAE 문자열을 그대로 사용했다. 7조건의 data/scaler/cohort 및 target·observed·origins·channel_ids 메타데이터가 동일함을 확인했다. 장치 `cpu`는 **이번 비용과 CPU6 재추론의 장치**다. Raw의 옛 전체 V 추론 장치는 manifest에 개별 기록되어 있지 않다. 원 ridge/G의 CPU 저장 V와 신규4조건의 CPU 저장 V도 출처를 나누었다. **7조건 모두의 새 CPU 전체 V43 재추론을 했다는 뜻이 아니다.**

```text
모델                  MSE                   MAE                  선택/refit epoch  refit update
RAW                  0.8491941389472357    0.4813883402485757   해당없음             0
RAW_RIDGE            0.5566463357048567    0.40552180175406594  해당없음             0
DIRECT_G             0.632217057436123     0.44152128982113786  3.0              2254
B0                   0.7841039310324822    0.5209622100413023   3.0              2254
B2                   0.6834428963980711    0.49388492449756366  1.0              752
LAG_CAUSAL           0.7432064184643359    0.5124572203253038   3.0              2254
LAG_BIDIR            0.7188903475533631    0.5033025985012579   1.0              752
```

G와 신규4 신경망은 모두 원 A50의 0.25→1→3epoch 검색을 완료했고 A_stop으로 선택한 가중치를 원 A60에서 새로 refit했다. 검색은 각 1,863 update/29,769 exposure이며 선택/refit 길이는 위 표와 SOURCE_BINDINGS의 원 영수증에 따로 적었다. 검색3epoch·선택3epoch·수렴은 같은 말이 아니다. Raw와 해석적 ridge에는 신경망 epoch를 붙이지 않았고 optimizer update는 0이다. Ridge의 원 alpha 선택은 A_stop을 사용했다. G는 명시된 `__cpu_fresh_e3`만 쓰며 다른 deep G 결과를 섞지 않았다.

비용 수치는 새 독립 AUDIT_RESULT의 `recomputed_summaries`에서 phase=`timed`, 20trial, stage_instrumentation=false인 값만 사용했다. CPU2·fresh process·whole physical pipeline·warm10/timed20/separate-stage20·row128·checkpoint inputs off 조건이다. 12개 유효 영수증의 600개 저장 trial을 검사한 기존 산술 감사 결과를 사용했으며 이번 조립에서 trial gzip이나 개별 오차를 다시 열지 않았다. stage_probe 시간은 섞지 않았다. min–max는 관측 범위이며 CI가 아니다.

Raw B1/B8는 원 row parity guard가 실패하여 비용 영수증이 없다. `FAILED`, parity=`FAIL`, `RAW_ROW_PARITY_GUARD_FAILED_NO_COST_RECEIPT`를 남기고 median/p90/min/max는 빈칸으로 보존했다. 어느 row/예외가 실패했는지 세부 원인은 아직 미확정이다. Raw의 CPU6 checkpoint replay PASS는 별도 검사로, 행 분할 parity PASS나 비용 완료로 승격하지 않는다. 배율 계산이나 다른 장치/배치/조건의 시간 대입도 하지 않았다.

동일 data/scaler/origins의 비교지만 구조·F 입력 폭·파라미터·학습 목표와 선택된 refit 연산량은 다르다. post-T V 재사용·단일 seed·단일 시간 분할의 탐색 결과이며 독립 일반화, 인과 그래프 증명, 충분 수렴, FrozenBolt의 필수성이나 원 연구 전체 완료를 주장하지 않는다. SOURCE_BINDINGS의 private 근거는 cache-relative 경로와 SHA만 기록했고 공개 본문에 개인 절대경로를 넣지 않았다. 아직 실제 그림을 만들지 않았으며 존재하지 않는 그림 링크도 추가하지 않았다.
