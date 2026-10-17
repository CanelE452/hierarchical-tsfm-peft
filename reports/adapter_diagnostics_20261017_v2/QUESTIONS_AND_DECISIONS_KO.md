# 질문별 판단

Q1 데이터·평가

[확인] pre-T 경계·mask·zero·scaler 지문과 독립 FP64/KKT 검사를 수행했다. 상류 보간·pretraining overlap UNKNOWN, 반복 V·과거 T46노출을 유지한다.

Q2 시간·표현·디코더·최적화

[확인] 기존 selected-refit B2−B0 관측:

[계약 동등성 확인] jena seed92601 (reference seed=92601.0, cpu→cpu): B2 MSE=0.6834428963980711, MAE=0.49388492449756366; 대 B0 ΔMSE=-12.83771584997208%, ΔMAE=-5.1975527249072275%

[계약 동등성 확인] jena seed92602 (reference seed=92602.0, cpu→cpu): B2 MSE=0.6788139247817864, MAE=0.5018712267137262; 대 B0 ΔMSE=-10.091916208827547%, ΔMAE=-0.12914550968850128%

[계약 동등성 확인] pems32 seed92601 (reference seed=92601, cuda→cuda): B2 MSE=0.09037503616821203, MAE=0.19860355836775184; 대 B0 ΔMSE=-9.964616493693665%, ΔMAE=-6.080660315325547%

[맥락 비교: MISSING_OR_DIFFERENT_device] pems08 seed92601 (reference seed=92601.0, cuda→cpu): B2 MSE=0.1042596968257376, MAE=0.2058417915000405; 대 B0 ΔMSE=-1.0241616962655866%, ΔMAE=-3.222843992351006%

jena cpu→cpu 계약 동등성이 확인된 완료 seed별 ΔMSE 평균=-11.464816029399813%, 범위=[-12.83771584997208,-10.091916208827547]; ΔMAE 평균=-2.6633491172978645%, 범위=[-5.1975527249072275,-0.12914550968850128]%. CI가 아니다.

동일 근접용량 CAP 대조:

[계약 동등성 확인] jena seed92601 (reference seed=92601, cpu→cpu): B2 MSE=0.6834428963980711, MAE=0.49388492449756366; 대 CAP ΔMSE=-10.356700841245745%, ΔMAE=-3.955626065401612%

[계약 동등성 확인] jena seed92602 (reference seed=92602, cpu→cpu): B2 MSE=0.6788139247817864, MAE=0.5018712267137262; 대 CAP ΔMSE=-6.059550220027131%, ΔMAE=-0.39422007647024543%

jena cpu→cpu 계약 동등성이 확인된 완료 seed별 ΔMSE 평균=-8.208125530636439%, 범위=[-10.356700841245745,-6.059550220027131]; ΔMAE 평균=-2.174923070935929%, 범위=[-3.955626065401612,-0.39422007647024543]%. CI가 아니다.

고정 PE full-history reader:

[계약 동등성 확인] jena seed92601 (reference seed=92601, cpu→cpu): B2_FULL_HISTORY_PROBE MSE=0.8442714731021048, MAE=0.5259053539510106; 대 B0_FULL_HISTORY_PROBE ΔMSE=2.2751504656630277%, ΔMAE=2.081045706128617%

[계약 동등성 확인] jena seed92602 (reference seed=92602, cpu→cpu): B2_FULL_HISTORY_PROBE MSE=0.8357508734218281, MAE=0.5218173653271749; 대 B0_FULL_HISTORY_PROBE ΔMSE=0.7074761791166884%, ΔMAE=0.9324235205081115%

jena cpu→cpu 계약 동등성이 확인된 완료 seed별 ΔMSE 평균=1.491313322389858%, 범위=[0.7074761791166884,2.2751504656630277]; ΔMAE 평균=1.5067346133183643%, 범위=[0.9324235205081115,2.081045706128617]%. CI가 아니다.

CAP 고정 PE reader 대조:

[계약 동등성 확인] jena seed92601 (reference seed=92601, cpu→cpu): B2_FULL_HISTORY_PROBE MSE=0.8442714731021048, MAE=0.5259053539510106; 대 B2_CAP_POINTWISE_FULL_HISTORY_PROBE ΔMSE=-1.6627346188394743%, ΔMAE=-1.86934414996573%

[계약 동등성 확인] jena seed92602 (reference seed=92602, cpu→cpu): B2_FULL_HISTORY_PROBE MSE=0.8357508734218281, MAE=0.5218173653271749; 대 B2_CAP_POINTWISE_FULL_HISTORY_PROBE ΔMSE=-3.3759009255899537%, ΔMAE=-2.8379222036145837%

jena cpu→cpu 계약 동등성이 확인된 완료 seed별 ΔMSE 평균=-2.519317772214714%, 범위=[-3.3759009255899537,-1.6627346188394743]; ΔMAE 평균=-2.353633176790157%, 범위=[-2.8379222036145837,-1.86934414996573]%. CI가 아니다.

[확인] 완료한 모든 seed에서 B2 probe의 MSE와 MAE가 B0보다 높다. 따라서 이 관측은 B2 표현의 고정 선형 접근성 개선을 뒷받침하지 않는다. λ1 고정·3,097,584 parameter의 full-history reader 조건에 한정된 반례이며, 표현의 모든 비선형 정보나 일반성을 판정하지 않는다.

채널 정보 own/cross Ridge 대조; cohort별 방향을 따로 읽는다:

[계약 동등성 확인] jena seednone (reference seed=None, cpu→cpu): ALL_CHANNEL_32BLOCK_RIDGE MSE=0.5822398391701266, MAE=0.45173869279248297; 대 OWN_CHANNEL_32BLOCK_RIDGE ΔMSE=4.59780320168561%, ΔMAE=11.396894267706555%

[계약 동등성 확인] pems32 seednone (reference seed=None, cpu→cpu): ALL_CHANNEL_32BLOCK_RIDGE MSE=0.09474669669927799, MAE=0.20353453670609112; 대 OWN_CHANNEL_32BLOCK_RIDGE ΔMSE=-4.106553113818976%, ΔMAE=-2.4696683969456057%

원 공동학습 D 대 재적합 shared D 비교:

[맥락 비교: MISSING_OR_DIFFERENT_dtype] jena seed92601 (reference seed=92601, cuda→cuda): B2_DECODER_SHARED MSE=0.6759618592190354, MAE=0.493656733668807; 대 B2_ORIGINAL_D ΔMSE=-1.0946098899294414%, ΔMAE=-0.046202261397425615%

[맥락 비교: MISSING_OR_DIFFERENT_dtype] jena seed92602 (reference seed=92602, cuda→cuda): B2_DECODER_SHARED MSE=0.6594456216178823, MAE=0.49062410051662286; 대 B2_ORIGINAL_D ΔMSE=-2.853253758027796%, ΔMAE=-2.2410358609091823%

jena cuda→cuda 맥락 비교 관측치의 완료 seed별 ΔMSE 평균=-1.9739318239786188%, 범위=[-2.853253758027796,-1.0946098899294414]; ΔMAE 평균=-1.143619061153304%, 범위=[-2.2410358609091823,-0.046202261397425615]%. CI가 아니다.

재적합 shared 대 Dh:

[계약 동등성 확인] jena seed92601 (reference seed=92601, cuda→cuda): B2_DECODER_HORIZON MSE=0.6892328558640555, MAE=0.4926257363132497; 대 B2_DECODER_SHARED ΔMSE=1.9632759547043914%, ΔMAE=-0.20884904129536955%

[계약 동등성 확인] jena seed92602 (reference seed=92602, cuda→cuda): B2_DECODER_HORIZON MSE=0.6622277282278117, MAE=0.48742010052896845; 대 B2_DECODER_SHARED ΔMSE=0.42188567468289156%, ΔMAE=-0.653045780727166%

jena cuda→cuda 계약 동등성이 확인된 완료 seed별 ΔMSE 평균=1.1925808146936414%, 범위=[0.42188567468289156,1.9632759547043914]; ΔMAE 평균=-0.43094741101126777%, 범위=[-0.653045780727166,-0.20884904129536955]%. CI가 아니다.

[확인] 공통 search endpoint의 개발 V 대조(B2−CAP)는 SAME_UPDATE_COMPARISONS.csv에 별도 기록한다. A_fit search 가중치의 평가이며 A_fit/A_stop 평가 행, A_stop 선택 및 selected fresh-A60 재적합 결과와 구분한다. 음수 Δ는 오차 개선이다.

jena seed=92601 search epoch=0.0 updates=0.0 exposures=0.0: ΔMSE=0.0%, ΔMAE=0.0%; EXACT_MATCHED

jena seed=92601 search epoch=0.25 updates=156.0 exposures=2481.0: ΔMSE=4.261571616518396%, ΔMAE=-0.18929908458562233%; EXACT_MATCHED

jena seed=92601 search epoch=1.0 updates=622.0 exposures=9923.0: ΔMSE=2.442386255409634%, ΔMAE=3.374768021195898%; EXACT_MATCHED

jena seed=92601 search epoch=3.0 updates=1863.0 exposures=29769.0: ΔMSE=-1.2655643855649037%, ΔMAE=2.7333843557640716%; EXACT_MATCHED

jena seed=92601: 공통 학습량에서 두 오차의 일관된 우위가 아니다. endpoint와 MSE/MAE에 따라 방향이 달라진다.

jena seed=92602 search epoch=0.0 updates=0.0 exposures=0.0: ΔMSE=0.0%, ΔMAE=0.0%; EXACT_MATCHED

jena seed=92602 search epoch=0.25 updates=156.0 exposures=2481.0: ΔMSE=-5.333163901301735%, ΔMAE=-0.6198122357217148%; EXACT_MATCHED

jena seed=92602 search epoch=1.0 updates=622.0 exposures=9923.0: ΔMSE=-2.6916773018727476%, ΔMAE=-1.837258232331596%; EXACT_MATCHED

jena seed=92602 search epoch=3.0 updates=1863.0 exposures=29769.0: ΔMSE=-2.048256851548677%, ΔMAE=-0.03801877353540541%; EXACT_MATCHED

jena seed=92602: 완료한 비영 endpoint 모두 MSE/MAE가 개선됐다.

선택 후 fresh-A60에서는 완료한 B2−CAP 대조 모두 MSE/MAE가 낮다. 이 selected-refit 관측을 공통 학습량의 일관된 시간·채널 혼합 효과로 바꾸어 읽지 않는다.

B2−B0와 B0−CAP의 같은 endpoint 대조 및 CP0의 실제 Δ도 같은 전용 CSV에 보존했다.

A_stop-selected refit의 학습량·용량 차이가 남는다. SAME_UPDATE_RESULTS의 동일 endpoint를 별도로 읽는다. 기존 H1~16 MAE 악화와 일부 채널 손해를 숨기지 않는다.

Q3 PCA 초기화

[확인] 실제 완료한 초기화 쌍:

[계약 동등성 확인] jena seed92601 (reference seed=92601, cuda→cuda): B2_QR MSE=0.7060527470310443, MAE=0.4996630839259965; 대 B2_PCA ΔMSE=2.86288345931501%, ΔMAE=1.5991247657182754%

[계약 동등성 확인] jena seed92602 (reference seed=92602, cuda→cuda): B2_QR MSE=0.7203017393205846, MAE=0.5072464169626417; 대 B2_PCA ΔMSE=6.205120519935174%, ΔMAE=1.6799184258679898%

jena cuda→cuda 계약 동등성이 확인된 완료 seed별 ΔMSE 평균=4.534001989625092%, 범위=[2.86288345931501,6.205120519935174]; ΔMAE 평균=1.6395215957931326%, 범위=[1.5991247657182754,1.6799184258679898]%. CI가 아니다.

두 arm의 common endpoint sampler/order 지문과 P/center 초기 지문은 원 job receipt에 결속했다. 해당 K·F·고정 학습 예산에서의 관측이며 PCA 일반적 필요성·설명분산과 미래정보 동일성을 입증하지 않는다.

Q4 단변량 backbone 교체

[확인] 완료한 두번째 단변량 backbone의 비교; 행별 계약 검증 상태를 따른다:

[계약 동등성 확인] jena seed92601 (reference seed=92601, cuda→cuda): Moirai_B2 MSE=0.6934476198994345, MAE=0.49004371238902666; 대 Moirai_B0 ΔMSE=-5.488042262928676%, ΔMAE=-3.37080174661546%

[계약 동등성 확인] jena seed92602 (reference seed=92602, cuda→cuda): Moirai_B2 MSE=0.6957503619615355, MAE=0.49693263658149955; 대 Moirai_B0 ΔMSE=-0.5123438009743896%, ΔMAE=2.197047578373424%

jena cuda→cuda 계약 동등성이 확인된 완료 seed별 ΔMSE 평균=-3.000193031951533%, 범위=[-5.488042262928676,-0.5123438009743896]; ΔMAE 평균=-0.586877084121018%, 범위=[-3.37080174661546,2.197047578373424]%. CI가 아니다.

[계약 동등성 확인] jena seed92601 (reference seed=92601, cuda→cuda): Moirai_B2 MSE=0.6934476198994345, MAE=0.49004371238902666; 대 Moirai_Raw ΔMSE=-8.746692105219859%, ΔMAE=10.52735982446867%

[계약 동등성 확인] jena seed92602 (reference seed=92602, cuda→cuda): Moirai_B2 MSE=0.6957503619615355, MAE=0.49693263658149955; 대 Moirai_Raw ΔMSE=-8.443665857288343%, ΔMAE=12.081128567491497%

jena cuda→cuda 계약 동등성이 확인된 완료 seed별 ΔMSE 평균=-8.5951789812541%, 범위=[-8.746692105219859,-8.443665857288343]; ΔMAE 평균=11.304244195980083%, 범위=[10.52735982446867,12.081128567491497]%. CI가 아니다.

선택 epoch·용량 차이와 반복 V 제한을 함께 유지한다. 두 백본 관측을 모든 TSFM 일반성으로 확대하지 않는다.

Q5 native multivariate 대안

[확인] 실제 native multivariate 개발 V 평가:

jena: MSE=0.6507532376921696, MAE=0.39933404413514123, device=cuda:0; 기존 Bolt Raw 대비 ΔMSE=-23.36814306102931%, ΔMAE=-17.045343489429737%; 맥락 비교 제한=MISSING_OR_DIFFERENT_device

pems08: MSE=0.08530764087177074, MAE=0.18167400062894662, device=cuda:0; 기존 Bolt Raw 대비 ΔMSE=-10.536730724300456%, ΔMAE=-7.466786890673074%; 맥락 비교 제한=MISSING_OR_DIFFERENT_device

backbone과 native channel grouping이 함께 바뀐 실용 비교다. B2 P/E의 단일요인 메커니즘 증거가 아니며, 정확도와 공정 E2E 비용을 함께 봐야 한다.

Q6 정확도·전체 비용

[확인] 실제 새 비용 영수증:

pems32 B0 cpu B1.0: warm ms/origin=59.44300000555813, status=COMPLETE_PROTOCOL

pems32 B0 cpu B8.0: warm ms/origin=52.93466875446029, status=COMPLETE_PROTOCOL

pems32 B2 cpu B1.0: warm ms/origin=62.44274997152388, status=COMPLETE_PROTOCOL

pems32 B2 cpu B8.0: warm ms/origin=53.306362533476204, status=COMPLETE_PROTOCOL

jena Raw cpu B1.0: warm ms/origin=139.44280007854104, status=COMPLETE

jena B0 cpu B1.0: warm ms/origin=48.62459981814027, status=COMPLETE

jena B2 cpu B1.0: warm ms/origin=48.65384986624122, status=COMPLETE

jena CAP cpu B1.0: warm ms/origin=49.16689987294376, status=COMPLETE

jena Ridge cpu B1.0: warm ms/origin=0.2991999499499798, status=COMPLETE

jena Raw cpu B8.0: warm ms/origin=144.57506872713566, status=COMPLETE

jena B0 cpu B8.0: warm ms/origin=42.69763748743571, status=COMPLETE

jena B2 cpu B8.0: warm ms/origin=42.97611248330213, status=COMPLETE

jena CAP cpu B8.0: warm ms/origin=43.490525014931336, status=COMPLETE

jena Ridge cpu B8.0: warm ms/origin=0.0726562284398824, status=COMPLETE

jena Raw cuda B1.0: warm ms/origin=12.843200005590916, status=COMPLETE

jena Raw cuda B8.0: warm ms/origin=5.876118753803894, status=COMPLETE

jena B0 cuda B1.0: warm ms/origin=18.541199853643775, status=COMPLETE

jena B0 cuda B8.0: warm ms/origin=2.4881124845705926, status=COMPLETE

jena B2 cuda B1.0: warm ms/origin=22.455150028690696, status=COMPLETE

jena B2 cuda B8.0: warm ms/origin=2.362831262871623, status=COMPLETE

jena Ridge cpu B1.0: warm ms/origin=0.39609987288713455, status=COMPLETE

jena Ridge cpu B8.0: warm ms/origin=0.0797125103417784, status=COMPLETE

jena Moirai_Raw cuda B1.0: warm ms/origin=28.29995029605925, status=COMPLETE

jena Moirai_Raw cuda B8.0: warm ms/origin=4.469312494620681, status=COMPLETE

jena Moirai_B0 cuda B1.0: warm ms/origin=16.52795006521046, status=COMPLETE

jena Moirai_B0 cuda B8.0: warm ms/origin=3.6285562673583627, status=COMPLETE

jena Moirai_B2 cuda B1.0: warm ms/origin=17.04204990528524, status=COMPLETE

jena Moirai_B2 cuda B8.0: warm ms/origin=3.6037875106558204, status=COMPLETE

jena Chronos2_nativeRaw cuda B1.0: warm ms/origin=29.620399931445718, status=COMPLETE

jena Chronos2_nativeRaw cuda B8.0: warm ms/origin=5.319481279002503, status=COMPLETE

[확인] Jena CPU B1 실용 대조: Ridge core=float64 MSE=0.5566463357048567, MAE=0.40552180175406594, warm ms/origin=0.2991999499499798; B2 core=float32 MSE=0.6834428963980711, MAE=0.49388492449756366, warm ms/origin=48.65384986624122. 같은 device/B/cohort/physical input/mask/scope/threads facet. Ridge가 두 오차와 지연 모두 낮다. 이 조건에서 B2의 실용적 약점이다. core 정밀도가 다르며 동일 정밀도 단일요인 비교가 아니다.

[확인] Jena CPU B8 실용 대조: Ridge core=float64 MSE=0.5566463357048567, MAE=0.40552180175406594, warm ms/origin=0.0726562284398824; B2 core=float32 MSE=0.6834428963980711, MAE=0.49388492449756366, warm ms/origin=42.97611248330213. 같은 device/B/cohort/physical input/mask/scope/threads facet. Ridge가 두 오차와 지연 모두 낮다. 이 조건에서 B2의 실용적 약점이다. core 정밀도가 다르며 동일 정밀도 단일요인 비교가 아니다.

[확인] Jena GPU Bolt Raw/B2 B1, dtype=FP32 core; FP64 inverse; CPU return: Raw=12.843200005590916, B2=22.455150028690696 warm ms/origin; Raw/B2 지연 비=0.5719489733616253x, B2가 느림. 같은 device/B/cohort/physical input/mask/scope/threads facet. 배치별 실측이며 C→K 변환이 항상 빠르다고 일반화하지 않는다.

[확인] Jena GPU Bolt Raw/B2 B8, dtype=FP32 core; FP64 inverse; CPU return: Raw=5.876118753803894, B2=2.362831262871623 warm ms/origin; Raw/B2 지연 비=2.4868973278533915x, B2가 빠름. 같은 device/B/cohort/physical input/mask/scope/threads facet. 배치별 실측이며 C→K 변환이 항상 빠르다고 일반화하지 않는다.

속도 비교는 장치·batch·dtype·physical input 지문이 같은 facet을 따른다. 위 CPU Ridge 실용 대조는 별도로 core 정밀도 차이를 명시한다. 미측정 Raw latency/speedup은 NA다. resident model params·trainable params·logical F rows·RSS sampling은 서로 다른 지표다.

결과에 따른 다음 변경은 NEXT_STEPS_KO.md에 적었다.
