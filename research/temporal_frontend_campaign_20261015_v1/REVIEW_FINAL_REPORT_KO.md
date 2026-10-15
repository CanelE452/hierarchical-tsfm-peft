# 최종 보고서 검토 — 2026-10-15

[확인] 원 3개 문서와 `REPORT_KO_v2.md`·`PROFESSOR_BRIEF_v2.md`·`DECISION_LOG_v2.md`를 읽고, 최종 v3 3개 문서까지 확인했다. 수정 필수 finding과 검토한 원수치의 불일치는 없다. 아래 검산은 보고서 생성 함수를 재사용하지 않고 CSV 원행을 직접 읽었다.

## 원수치 대조

[확인] 20개 수치가 표시 자릿수까지 일치한다. SCORE는 모두 primary fold이며 MSE는 소수 6자리다. 비용은 COST.csv의 해당 scope/batch 행이고 MiB는 bytes/1024²다. 보조 독립 검토도 35개 대조에서 불일치를 발견하지 않았다.

```text
SCORES id                            split  report=CSV
primary_beijing_A0                    C      0.655904
primary_beijing_A7                    C      0.726034
primary_beijing_B4_r2_s92601           C      0.808416
primary_jena_B2_r5_s92602              C      1.106624
primary_beijing_LoRA_qv8               C      0.582379
primary_jena_LoRA_qv8                  C      0.914889
primary_beijing_B0_r2_s92601           B      0.831239
primary_beijing_B2_r2_s92601           B      0.773044
primary_beijing_B4_r2_s92601           B      0.737272
primary_jena_B0_r5_s92601              B      0.731145
primary_jena_B2_r5_s92602              B      0.680315
primary_jena_B4_r5_s92602              B      0.692424

COST id                              scope      batch  field          report=CSV
primary_beijing_A0                    general    1      median_s       0.027095
primary_beijing_A0                    exclusive  1      median_s       0.012485
primary_beijing_A0                    general    8      median_s       0.023406
primary_beijing_A0                    exclusive  8      median_s       0.020326
primary_jena_A0                       general    8      median_s       0.032854
primary_jena_A0                       exclusive  8      median_s       0.032488
primary_beijing_Direct_B2              exclusive  1      peak_MiB       12.43
primary_jena_Direct_B2                 exclusive  8      peak_MiB       17.13
```

## 주장·scope·최신본

- [확인] 전체 보고서의 비용 scope 문단과 v3 판단 기록은 속도 판단에 balanced general latency, 메모리 판단에 exclusive-process peak를 사용한다고 구분한다. scope별 시간 차이의 원인은 미확정으로 남긴다.
- [확인] v1에서 명시성이 약했던 자료별 oracle 해석은 v2 및 v3 REPORT/DECISION에서 보완됐다. Beijing B2/B4 진단 0.465465/0.478792는 Raw 0.655904보다 낮으므로 고정 D 공간만으로 패배가 불가피하다고 하지 않는다. Robin B4 진단 0.107488은 Raw 0.074146보다 높지만 해당 선정 고정 D와 유지한 수치 공간의 제약으로 한정한다. 미래 Y 진단을 과거 표현의 충분성·F의 도달 가능성·모든 작은 K 방법의 불가능성으로 확대하지 않는다.
- [확인] G-only와 LoRA의 더 좋은 C 수치를 숨기지 않으며 v3 교수님 설명안에도 반례 표와 추가 가치 미확보 판단이 있다. 80개 neural logical fit 중 12개 undertrained/unresolved 상태를 실행 목록과 대조했다. 86 logical fits는 80 neural+6 deterministic ridge이며, 90 fitting configurations는 80+10 ridge alpha 적합이다.
- [확인] C는 이번 선택에서 잠갔지만 historically unseen 독립 holdout이 아니라는 점, 두 seed 범위를 신뢰구간으로 보지 않는 점, 두 번째 시간 경계를 조건부 반복으로 보는 점을 유지한다.
- [확인] `AUDIT_LATEST.json`이 지정한 `AUDIT_v2.json`·`RUN_INDEX_v2.csv`와 두 SHA가 실제 파일에 일치한다. REPORT는 최신 manifest와 v2 결과를 연결하고 원 attempt1은 보존 기록으로만 구분한다. DECISION의 RUN_INDEX도 v2다. 교수님 설명안은 전체 보고서 링크를 통해 감사 근거에 연결된다.
- [확인] v2→v3 세 문서는 Markdown 링크 target을 제외한 본문이 정확히 같다. 변경 target은 그림05 v3와 REPORT/PROFESSOR/DECISION 상호 링크뿐이며 로컬 링크 대상의 존재를 확인했다. DECISION v3는 v2와 내용 전체가 같다.

## 검토 한계

교수님 설명안 6절 자체는 general/exclusive 용어를 생략하므로 단독 구두 설명에서는 해당 scope를 함께 말하는 편이 명확하다. 연결된 비용 그림과 전체 보고서에는 구분이 있어 사실 오류나 차단 finding은 아니다. 독자 질문의 문제→대안→설정→결과→한계 연결은 세 문서에서 확인했다.

이번 최종 검토는 문서·CSV·JSON·링크 비교이며 모든 표의 전수 재검산이나 그림 픽셀 검토는 아니다. 6개 그림의 실제 시각 확인은 ROOT 담당이다. 원자료·F·GPU·새 fit·학습·비용 타이머·감사 재실행은 없었고, 이 검토 파일 외의 소스와 산출물을 수정하지 않았다.
