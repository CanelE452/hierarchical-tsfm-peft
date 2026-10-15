# 과거 적합 디코더의 갱신을 제한하는 다음 대조

상태: SOURCE·사전 규칙 준비. 실제 적합·추론·비용 결과는 없다. 이전 local_past_decoder_v1의 실행 소스와 결과는 동결한다. 실제 새 실행 전 그36쌍의 독립 감사 PASS를 요구한다.

목적은 새 표현을 소개하는 것이 아니라, 관측 과거의 복원 오차를 낮추는 local D가 미래 오차를 증폭하는 부분을 분리하는 것이다. 현재 producer 원수치에서는 Robin FIT-target Bolt의 TUNE MSE가 .075569→.156193, PCA가 .104703→.693879로 증가했다. local D의 최대 계수는 각각33.59/42.48이었다. 개발 구간에서 오차가 줄었어도 이 값으로 원인을 확정하지 않는다. 독립 수치 감사·새 D별 공간 진단·전체 비용은 별도 단계다.

같은 E·K·causal 과거·관측 mask·cached Bolt/native 미래·원 selected head·current-C anchor를 유지한다. 각 origin/channel의 실제 관측 과거에서 평균을 제거한 Zc,yc로 G=ZcᵀZc/n, b=Zcᵀyc/n을 만든다. 고정된 원 inverse decoder D0를 prior로 사용하며,

```text
min_beta mean((Zc beta - yc)^2) + lambda ||beta - D0[:,c]||^2
lambda = rho * trace(G) / K
beta = D0[:,c] + solve(G + lambda I, b - G D0[:,c])
bias = mean(y) - mean(z) beta
```

rho는 사전 양수 목록 .001,.01,.1,1,10이다. 원 static D와 기존 cutoff1e-4 local D는 재사용 대조다. 양수 ridge는 cutoff를 쓰지 않는다. 기존 truncated minimum-norm local은 이 양수 ridge의 정확한 rho=0 한계라고 주장하지 않는다. trace(G)=0이면 centered Z가0이며 모든 slope가 같은 past prediction을 준다. 이때 prior slope를 그대로 유지하고 mean 기반 intercept를 적용하는 사전 규칙이다. 원 관측0이면 prior slope와 마지막 알려진 과거 C/Z를 통과하는 intercept를 사용한다. prior 선택/fallback도 명시하고 일반적인 최적 미래 복원이라고 쓰지 않는다.

중심화와 intercept는 원 관측 subset마다 수행한다. filled response를 target으로 세지 않는다. rho는 scale 상대 규칙이므로 절대 decoder gain·미래 오차의 상한이 아니다. 과거 관계가 미래에 바뀌거나 latent forecast가 틀리면 ridge를 써도 실패할 수 있다. 원 C 과거에서 만든 K×C coefficients가 추가되는 조건부 decoder이며 strict K-only 또는 동일 decoder-side information 비교가 아니다.

향후 실행은 자료별·공간별로 양수rho5개와 재사용static/local 대조의 **Bolt TUNE** 관측 macro 정규화 MSE를 비교한다. 총4개의 선정 잠금이며 native·개발 점수로 rho를 따로 고르지 않는다. 같은 값의 tie는 사전 순서 static,기존local,.001,.01,.1,1,10에서 먼저 온 조건을 선택한다. MAE는 보조이며 선택 목적을 바꾸지 않는다. 모든4선정 잠금을 저장한 뒤 선택한 조건만 개발 구간을 계산한다. 같은 공간의 frozen native 두 seed에는 Bolt가 고른 같은 rho 규칙을 적용하되 cached E의 원 backend 차이를 보존한다. native의 hyperparameter나 head에 최적화된 benchmark라고 쓰지 않는다. 미래 oracle·노출 개발 점수·전체 비용으로 rho를 선정하지 않는다. 선정은 탐색적 hyperparameter 선택이며 개발 구간은 이미 노출됐으므로 독립 holdout 확증이 아니다. 새 E/F/head 학습·큰 모델 재학습은 없다.

FIT/TUNE/DEV의 알려진 과거로 inference-time D를 적합하고 TUNE 미래 정답은 정상 rho 선택에 쓴다. 미래 정답을 전혀 쓰지 않았다고 말하지 않는다. 기존 native는 static D 목적의 원 head를 고정한 개입이며 새로운 D에 다시 최적화된 native benchmark가 아니다. 실제 CPU 실행·독립 검산 후에는 선택된 조건의 실제 F 연결 전체 비용을 따로 확인해야 한다. 계획의 solver 횟수·비용은 완료 수가 아니다.

[판단] 원 연구 질문의 복원 안정성 원인을 계속 확인하는 범위다. 사용자 지속 수행 지시와 상한 해제를 따른다. 별도의 exact-human 개별 승인이 있었다고 주장하지 않는다. 이전 승인/상한/실패/선정 기록은 보존하고 TEST·설치·기존 job 종료/재개·대형 backbone 학습·자동 Git 게시를 하지 않는다.
