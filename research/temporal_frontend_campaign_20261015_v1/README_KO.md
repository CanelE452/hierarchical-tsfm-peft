# 시간 앞단 R1–R5: 실행 결과와 최종 자료

[확인] 승인된 핵심 학습·실제 추론·전체 비용 측정·수치 검산·한국어 보고서·그림 6종을 완료했다. 이것은 작은-K 방법의 성공이나 연구 문제 전체의 해결 선언이 아니다. 자료 부족으로 실행하지 못한 경로와 학습 미해결 상태를 아래에 구분한다.

먼저 [교수님 설명안](PROFESSOR_BRIEF_v3.md)을 읽고, 원수치와 반례는 [전체 보고서](REPORT_KO_v3.md), 선택 근거는 [판단 기록](DECISION_LOG_v3.md)에서 확인하면 된다. 문서의 v2/v3는 **표현·가독성 수정본**이며 새로운 학습이나 C 재선정을 뜻하지 않는다. 이전 문서·그림·실패 기록은 보존했다.

## 지금 답할 수 있는 것

1. **시간 입력 적응과 작은-K 압축은 분리해서 봐야 한다.** B로 선택한 r=C 후보의 C MSE는 Beijing B4 0.608003, Jena B2 0.996148로 Raw보다 작다. 같은 가족의 작은-K는 각각 0.808416/1.106624다. C/K는 E/D 용량·출력 공간도 함께 바꾸므로 정보 손실 하나의 인과 효과로 단정하지 않는다.
2. **작은-K+Bolt 결합의 독자적 추가 가치는 확보하지 못했다.** 두 자료의 B-selected temporal-K+G는 동일 입력·같은 클래스의 G-only보다 나빴다. LoRA도 강한 반례다. 작은-K는 F에 보내는 stream 수이며 현재 원 C anchor와 G의 원 C 과거 입력은 남는다.
3. **실패 지점은 자료마다 다르다.** Beijing B2/B4의 고정 D 진단 0.465465/0.478792는 Raw 0.655904보다 작다. 선정 D의 공간만으로 Raw 패배가 불가피하다고 할 수 없다. Robin B4 진단 0.107488은 Raw 0.074146보다 커서 해당 고정 D 수치 공간의 제약도 남는다. 실제 미래값을 사용하는 진단이므로 과거 표현의 예측 충분성·F의 도달 가능성을 증명하지 않는다.
4. **예측 목적 학습은 실제로 했다.** 이번 P/E/D는 재구성 손실이 아니라 관측된 원미래 Y의 normalized MSE로 학습했고 frozen F를 통해 입력 gradient를 흘렸다. 현재 실패를 단순히 ‘아직 예측 손실을 쓰지 않아서’라고 설명할 수 없다. 잠재 rank가 유지된 사실만으로 표현의 유용성을 증명할 수도 없다.

주 fold의 평가 C 원수치다. temporal-K는 Beijing B4/K2, Jena B2/K5를 각각 **B에서** 선택한 값이다.

```text
method                  Beijing_C_MSE  Jena_C_MSE
Raw L512                0.655904       1.223530
B0 smallK               0.833102       1.123924
B-selected temporal-K   0.808416       1.106624
same temporal-K + G     0.724497       1.084841
G-only                  0.612079       1.017580
LoRA q/v rank8          0.582379       0.914889
```

![같은 G에서 압축 경로의 추가 가치](figures/03_matched_residual_v2.png)

좁힌 연구 질문은 **원채널 미래 예측의 이득을 소수 F streams에서도 단순 직접 예측·LoRA보다 유지할 수 있는가?**다. 현재 조건에서 그 답은 미확보다. 시간 앞단의 조건부 효과는 남지만, 성공 방법론·학술적 신규성·새 독립 일반화는 확정하지 않는다.

## 실제 실행과 산출물

- R1: 42개 설정을 확인해 41개를 실제 추론했다. Robin L2048은 허용 FIT의 과거 길이가 부족해 BLOCKED다.
- R2: 두 주 자료 × C/K × B0–B4 × 2 seed, 40개 신경망을 새로 학습했다.
- R3/R5: 같은 G의 다섯 경로를 primary 10개·secondary 10개, 총 20개 학습했다. G seed 하나를 썼으며 두 G seed 확증으로 부르지 않는다.
- R4: direct 4개·LoRA 2개 신경망과 native/ridge 10개 alpha 적합, 총 16개 fitting configuration을 실행했다. ZERO는 추가 fit가 아니다.
- R5: 모듈 제거·중간 rank·전체 upstream 시간 재적합·Robin 스트레스 학습과 horizon/채널/과거 패턴/오차/잠재/고정 D 진단을 수행했다. 조건이 같은 B3 분해 제거 대조 두 건은 이번 캠페인 내부 결과를 검증해 재사용했다.

전체 신경망은 80개, 121,200 update다. ridge는 6개 logical route/10개 alpha 적합이다. 별도 probe는 200 update다. 신경망 update 구간 시간 합은 3,491.48초, checkpoint 평가·저장을 포함한 새 segment 시간 합은 3,699.90초다. ridge 적합 시간은 53.90초다. 이 값은 전체 캠페인 wall이나 GPU kernel 시간과 다르다.

[SCORES](SCORES.csv)는 B/C 296행이며 [현재 실행 목록](RUN_INDEX_v2.csv)은 862행이다. 독립 표본 수나 새 모델 수가 아니다. [COST](COST.csv)는 general 91개 경로·exclusive 47개 경로의 B1/B8 실제 CPU→CPU 경로다. 측정 6,994회와 warmup 1,380회를 보존했다. 속도는 balanced general 세션, 메모리는 별도 process scope를 기준으로 읽는다. 두 scope의 시간 차이 원인은 미확정이며 빠른 값만 고르지 않았다.

[최신 감사 연결](AUDIT_LATEST.json)이 지정한 [감사 v2](AUDIT_v2.json)는 PASS다. 저장 예측·mask·점수·선정·비용 산술을 재계산한 검증이며 학습이나 타이머의 독립 재실행은 아니다. 패턴 분석은 [ANALYSIS_COMPLETE](ANALYSIS_COMPLETE.json), 고정 D 진단은 [ORACLE_DIAGNOSTIC](ORACLE_DIAGNOSTIC.json)에 연결돼 있다.

## 남은 한계와 재개 조건

- 12개 신경망 경로는 최대 2,400 update에서도 미수렴·미해결이다. 정확한 ID·checkpoint·학습곡선·실제 update 시간은 전체 보고서와 fit receipt에 있다. 추가 학습의 필요성과 방법의 한계는 아직 분리되지 않았다. 이번 최대치를 넘어 계속 학습한 결과로 꾸미지 않았다.
- Robin L2048은 추가 GPU로 해결할 수 없는 자료 길이 문제다. 더 긴 허용 학습 자산과 새 경계를 먼저 정해야 재개할 수 있다. 로컬 tiny/mini 모델은 없어서 새 다운로드 없이 미실행으로 남겼다.
- 평가 C는 이번 선택에서 잠갔지만 과거 캠페인에 이미 노출된 FIT 내부다. `historically_unseen=False`이며 새 독립 holdout이 아니다. 두 cutoff와 두 seed도 확증 통계가 아니다.
- Jena/Robin의 이전 TRAIN 정규화 FP32 역변환 계보, safe NPZ member 압축 해제의 범위, A32 패턴 경계와 A32 학습 모니터 한계를 보고서에 유지했다.

Windows launcher 경쟁 검사 오탐과 감사 CSV reader 한도 오류를 수선했으며 원본 실패를 보존했다. [실행 수선 기록](EXECUTION_REPAIR_KO.md), [감사 manifest](AUDIT_LATEST.json), [최종 그림 provenance](FINAL_RENDER_PROVENANCE.json)를 확인할 수 있다. 그림 버전 추가는 과학 결과의 재실행·재선정이 아니다.

이번 작업에서 자동 commit/push, 보호 TEST 새 사용, 설치·다운로드, 기존 외부 작업 종료는 하지 않았다. 새로 만든 실패 비용 queue만 범위를 확인해 종료·수선했다. 학습 checkpoint와 예측 NPZ는 로컬 `.cache/temporal_frontend_campaign_20261015_v1/`에 보존했다. 보고서 이미지와 문서 링크는 campaign 내부 상대 경로다.
