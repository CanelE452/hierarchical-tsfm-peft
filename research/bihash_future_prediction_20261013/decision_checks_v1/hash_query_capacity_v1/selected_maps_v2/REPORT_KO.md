# Query 해시 정보 차원의 CPU 구조 진단

상태: COMPLETE_AVAILABLE_CASE_DIAGNOSTIC. 필수 DEV/SML 조건 18/18개, 실제 존재하는 Household 조건 3/6개를 읽었다. 전체 학습 완료 보고가 아니다.

`student_projection(K→32)`와 `student_hash(32→32)` 사이에는 비선형 함수가 없다. B의 deterministic query logit은 `Ws P z + Ws bP + bs`, A의 auxiliary logit은 `Ws P (z_last + mean(z)) + 2 Ws bP + bs`다. 저장된 실제 두 weight를 합성하여 shape/rank/singular values를 기록했다.

R32로 펼쳐도 affine 입력 정보 차원은 K를 넘지 않는다. K1의 32개 deterministic bit는 각 affine threshold에서 한 번만 바뀌므로 정확 산술의 query code 수는 최대 33개다. 이는 memory eligibility mask를 고정한 검색 점수 경로의 한계이며 Gumbel noise가 있는 학습에는 적용되지 않는다. 일반 K의 전체 code 수를 주장하지 않는다.

아래 logits·code 수는 모두 synthetic latent를 사용했다. 실제 데이터 query code 분포와 분리해야 한다. FP32 두 Linear와 FP64 합성식은 임계값 근처에서 sign이 다를 수 있다.

```text
dataset                    method               seed   K  rank  FP32↔FP64 max gap  FP64 reorder gap  synthetic codes32/64
jena                       B_GUMBEL             92601  5     5       1.2689907e-06     2.6645353e-15  238/238
jena                       A_BIHASH_GUMBEL      92601  5     5       1.6805015e-06     3.5527137e-15  245/245
jena                       B_GUMBEL             92602  5     5       1.0714409e-06     1.7763568e-15  236/236
jena                       A_BIHASH_GUMBEL      92602  5     5       1.6594274e-06     2.6645353e-15  246/246
jena                       B_GUMBEL             92603  5     5       9.4907234e-07     2.6645353e-15  246/246
jena                       A_BIHASH_GUMBEL      92603  5     5       1.5664643e-06     3.1086245e-15  234/234
robin                      B_GUMBEL             92601  4     4       1.0249019e-06     1.7763568e-15  228/228
robin                      A_BIHASH_GUMBEL      92601  4     4       2.4982179e-06     3.5527137e-15  191/191
robin                      B_GUMBEL             92602  4     4       6.0256991e-07     1.7763568e-15  231/231
robin                      A_BIHASH_GUMBEL      92602  4     4       1.5727927e-06     5.3290705e-15  231/231
robin                      B_GUMBEL             92603  4     4       7.1996656e-07     1.7763568e-15  236/236
robin                      A_BIHASH_GUMBEL      92603  4     4       8.6324734e-07     1.7763568e-15  215/215
uci274_sml2010             B_GUMBEL             92601  5     5       1.8448208e-06     1.7763568e-15  231/231
uci274_sml2010             A_BIHASH_GUMBEL      92601  5     5       9.6003832e-07     1.7763568e-15  230/230
uci274_sml2010             B_GUMBEL             92602  5     5       7.9781991e-07     2.6645353e-15  222/222
uci274_sml2010             A_BIHASH_GUMBEL      92602  5     5       1.7665999e-06     3.5527137e-15  220/220
uci274_sml2010             B_GUMBEL             92603  5     5       1.5210282e-06     1.7763568e-15  236/236
uci274_sml2010             A_BIHASH_GUMBEL      92603  5     5       1.6752309e-06     3.5527137e-15  227/227
uci235_household_hourly    B_GUMBEL             92601  1     1       6.4307773e-06     2.1316282e-14  33/33
uci235_household_hourly    A_BIHASH_GUMBEL      92601  1     1       1.1369638e-05     2.1316282e-14  33/33
uci235_household_hourly    B_GUMBEL             92602  1     1       2.1006756e-05     5.6843419e-14  33/33
```

전체 예측은 시간 인코더, 연속값 S/fusion, raw-last anchor, 시점에 따라 바뀌는 memory mask도 사용한다. 해시 query의 affine 경계가 전체 예측 능력의 붕괴나 성능 실패 원인이라는 결론은 아니다. A 해시 경로는 배포 때 제거된다. 실제 past-only 데이터에서 code 사용률과 조회 가중치의 다양성은 후속 검산으로 남아 있다.
