# 과거만으로 복원 공간을 바꾼 실제 결과

기록 시점: 2026-10-15 KST. **원 연구 목표는 미해결이며 다음 실험을 계속한다.** 아래36개 비교는 실제 CPU 실행과 독립 감사가 끝났고, 새 local D의 전체 배포 비용은 아직 미측정이다. 이전 시간·누적 자원 상한 때문에 멈추지 않는다.

[확인] 같은 E와 K, 원 cached Bolt/native 잠재 미래, current-C anchor를 고정하고 각 origin의 실제 관측 과거512에서만 affine D를 적합했다. cutoff1e-4는 centered slope의 상대 singular cutoff다. 미래 정답은 D 적합에 없으며 원 native head와 모델 선택은 바꾸지 않았다. TUNE/DEV의 알려진 과거는 inference-time fit에 쓰므로 그 자료가 어떤 fit에도 사용되지 않았다고 쓰지 않는다.

```text
관측 채널 macro 정규화 MSE       static D       local D       원 채널 Bolt
Robin / PCA / TUNE              0.104702758    0.693878798   0.056951091
Robin / FIT-target / TUNE       0.075569448    0.156193050   0.056951091
Robin / PCA / VAL               0.532063803    0.431921567   0.340401887
Robin / FIT-target / VAL        0.564240397    0.373466779   0.340401887
Beijing / PCA / TUNE            1.009348491    0.983708633   0.758435638
Beijing / FIT-target / TUNE     0.935872434    0.920843409   0.758435638
Beijing / PCA / EVAL            0.778177077    0.696380597   0.589575414
Beijing / FIT-target / EVAL     0.694519401    0.669978818   0.589575414
```

개발 구간의 일부 개선을 채택 판단으로 사용하지 않는다. 이미 노출된 개발 구간을 본 뒤 설계한 탐색적 후속이고, TUNE에서는 Robin이 악화했다. 원 static D로 학습된 frozen native는 Robin의 네 개발 조건에서 모두 악화했다. 새 local D에 최적화된 native를 학습한 비교가 아니므로 native 방법 전체의 불가능성을 뜻하지 않는다.

[확인] 새 D[n] 각각의 출력 공간에 미래 정답 변화량을 투영한 별도 진단은 다음과 같다. 실제 미래와 최적 잠재 좌표를 아는 **진단값**이며 배포 입력·학습·선정에 쓰지 않았다. 과거 표현의 예측 가능성이나 달성한 오차가 아니다.

```text
개발 complete-C 동일 mask        새 D[n] 실수 최소 MSE   실제 local Bolt   원 채널 Bolt
Robin / PCA                     0.203251731             0.431921567       0.340401887
Robin / FIT-target              0.200862892             0.373466779       0.340401887
Beijing / PCA                   0.354867871             0.703717251       0.598972295
Beijing / FIT-target            0.316059913             0.678325912       0.598972295
```

Beijing의 complete-C 원 채널 .598972295는 위 첫 표의 full-observed .589575414와 다른 mask다. 원 static-D나 공통 global rank-K oracle의 최소값을 conditional D[n]의 하한으로 재사용하지 않았다. 현재 진단도 normalized MSE의 지정한 D32-as-FP64 공간만 대상으로 하며 physical MSE·MAE의 최솟값이나 모든 조건부 디코더의 한계가 아니다.

[판단] 과거 적합으로 복원 가능한 공간은 달라졌다. 그러나 남은 실제 잠재 예측 오차와 그 오차를 전달하는 계수를 함께 봐야 한다. Robin TUNE의 실제 D 최대 절댓값은 PCA42.48·FIT-target33.59였다. 상대 조건수 규칙만으로 절대 계수나 미래 오차가 제한되지 않는다. 원 D를 새 einsum 순서로 계산한 rounding 대조의 최대 출력 차이는 이 묶음에서 대략 5e-7 이내였으며 큰 MSE 악화를 이 작은 연산 차이 하나로 설명할 수 없다. 계수 증가만으로 단일 인과 원인을 확정하지도 않는다.

이번 interface는 K future streams·current C에 **C의 과거/mask로 만든 origin별 K×C coefficients**를 추가한다. 표현학습 자체의 개선·strict K-only·같은 decoder-side information 비교라고 쓰지 않는다. 비용에는 원 과거 전처리와 E, local LS 및 구현된 진단 연산, F, D, 물리 단위 반환을 포함해야 한다.

검산 근거는 [실제 producer](runs/local_past_decoder_v1/RESULT.json) SHA `86f894c72ea091c3a7d2b068a901fa59ec9229868acd6437a53c8709203084dc`와 [독립 감사 PASS](audits/LOCAL_PAST_DECODER_PROBE_AUDIT_v1.json) SHA `96f37ff913ec4e8169b145a6bd67a3880b9e99f632f9d1fbbd96850183963bde`다. actual36쌍·24backend states·3264window fit·44304observed channel solver·2592score fields·원859파일SHA를 확인했다. no-observation fallback은 실제0이고 native 두 seed의 같은 past에서 D를 공유한 건수는1632다. 보호된 TEST·새 E/F/head 학습·모델 forward·Git 게시를 하지 않았다.

다음 [갱신 제한 계획](PLAN_LOCAL_DECODER_SHRINKAGE_KO_v1.md)과 [순수 수학 소스](local_decoder_shrinkage_v1.py)는 합성13검사만 통과한 준비 단계다. 실제 후보 적합·TUNE 선정·선택된 개발 추론은 아직0이다. 전체 비용·독립 검산·시각 보고서를 완료하기 전 완료나 개선을 주장하지 않는다. 이전 [보고서 모음](README_KO.md)의 SOURCE 상태 문장은 그 파일을 처음 만든 시점의 상태이며 이번 실제 실행 기록이 최신이다.
