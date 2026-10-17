# 남은 원 방법군: Jena AdaPTS port와 Beijing B4

[확인] 원 recipe 두 조건의 실제 학습·전체 V·FP64 재채점·fresh CPU6·B1/B8 검산이 완료된 감사에서만 이 보고를 만든다. 두 자료의 scaler/mask/origin/관측 라벨은 다르다. **자료 간 MSE 순위**를 만들지 않으며 각 자료 안에서만 원수치를 비교한다.

[원정밀도 CSV](REMAINING_FAMILY_RESULTS_v1.csv) · [보고 결속](REMAINING_FAMILY_BINDINGS_v1.json) · [실제 독립 감사](INDEPENDENT_REMAINING_FAMILY_AUDIT_ROOT_v1.json)

Jena C21 / V43 / observed labels43344

```text
role                        F-width MSE                 MAE                 B1sec/batch B8sec/batch selected-refit-epoch reused
B2_K6_REFERENCE              6 0.6834428963980711 0.49388492449756366 0.047770299948751926 0.32702095014974475 1.0 True
LORA_R8_C21_REFERENCE       21 0.56392294174335067 0.39900992921234152 0.13867014995776117 1.1420140000991523 1.0 True
B4_K6_REFERENCE              6 0.70435308018904963 0.49675056670884404 0.04856874980032444 0.34662930015474558 1.0 True
ADAPTS_BOLT_PORT_K6          6 0.73681898480041197 0.53338193003251122 0.046393000055104494 0.32289890013635159 3.0 False
```

Beijing C11 / V43 / observed labels20539

```text
role                        F-width MSE                 MAE                 B1sec/batch B8sec/batch selected-refit-epoch reused
B2_K11_REFERENCE            11 0.7800566838351507 0.58014338541429433 0.078113199910148978 0.57287984993308783 0.25 True
B4_K11                      11 0.76518717862032881 0.57774329234347521 0.077778900042176247 0.59488734998740256 0.25 False
```

새 결과는 Jena ADAPTS_BOLT_PORT K6와 Beijing B4 K11 두 행이다. Jena B2 K6/B4 K6/LoRA C21 및 Beijing B2 K11은 pinned 기존 결과 재사용이며 새 학습·추론·비용이나 독립 일반화로 세지 않는다. 참고의 상세 provenance는 [기존 Jena 초기 방법군](INITIAL_FAMILY_KO_v1.md), [B2 정확도/비용 검증](B2_REFERENCE_VALIDATION_KO_v1.md), [Beijing core 보고](../priority_valid_minima_20261016_v1/CORE_CONTRAST_KO_v1.md)에 남아 있다.

모든 행은 L512/H48·seed92601·CPU2·반복 개발 V43의 normalized channel-macro MSE/MAE다. Jena는43344, Beijing은20539 관측 라벨이며 같은 V를 반복 사용한 것이므로 독립 표본 수가 증가하지 않는다. 새 BASE의 actual selected refit0/.25/1/3, update/exposure와 stage wall을 CSV/결속에 그대로 보존한다. 선택을3으로 가정하지 않는다. search는 원 grid0/.25/1/3 endpoint이고 선택3이면 UNDERTRAINED_OR_BUDGET_LIMIT, 더 짧은 선택도 수렴을 입증하지 않는다. 제한 학습·한 seed·반복 개발 V에서 충분 학습/독립 일반화는 미확정이다.

AdaPTS는 저장소의 LinearAutoEncoder/MultichannelProjector 원리를 affine pointwise encoder/decoder·RevIN=False/patch1로 옮긴 **AdaPTSLinearPort**다. 이 저장소의 macro loss·시간 분할·미분 가능한 frozen Chronos-Bolt를 사용하므로 원 논문의 모든 설정·모델·재학습·수치를 완전 재현한 결과가 아니다. PCA 초기화나 lag7/current anchor가 없는 nn.Linear 경로이며 B2/B4의 학습 lag7 D/current anchor와 달라 encoder 하나의 효과 대조가 아니다. Beijing B4 K=C는 identity basis/center0에서 시작하는 등록 nonlinear 경로이고 B2 full-C도 identity basis이지만 모수·구조는 다르다. Jena B2/B4 K6 PCA 초기화, LoRA r8와effective C21/F 적응·lr1e-4, 새 B4/AdaPTS lr1e-3의 차이를 동등 compute/동일 표현 비교로 바꾸지 않는다.

비용은 physical history/mask→채움·정규화→실제 K개 F 입력→등록 decoder→원 C 미래·FP64 inverse/CPU 반환의 한 batch 전체다. warm10·ordinary timed20·별도 intrusive stage20을 분리하고 ordinary median/p90를 초/batch로 보존한다. B8은8개 입력 한 batch 전체이며8로 나누지 않는다. cold F pool/Route load·LoRA merge는 warm whole 밖이고 model_load는 원 기록 범위만이다. Route memory dict·guard tree RSS·별도 측정하지 않은 F-only resident NA를 구분한다. native family receipt가 실제 기록한 F counter/proofmap/float64 dtype는 그대로 보존하며 N7의 미기록 NA 계약과 혼동하지 않는다.

기존 Jena B2의 원 학습/정규화 ancestry와4GiB guard를 새3GiB 학습으로 소급 바꾸지 않는다. 새 B2 reference 비용은 별도 CPU2/RSS3GiB quiet 검증 결과다. 원 초기 POST mixed-slash PARTIAL·recovery 기록과 모든 기존 산출물을 그대로 둔다. 이 보고는 기존 결과 파일·숫자·MD를 고치지 않는다. 새로운 두 조건의 성능이 좋거나 나쁨은 완성 보고 gate가 아니며 partial/검산 실패는 ROOT가 성공·미실행·저장점·재개 조건을 별도 정직하게 보고해야 한다.

전체 E1/E4/56/MASTER 완료가 아니다. N7 실제 종료·보고/main 및 원 잔여 예산 확인 후 별도 입장한 경로만 보고할 수 있다. 보호 T 새 수치 파싱·평가·선택·기존 중단 실험 재개·예산 확대·새 방법 추가는 하지 않는다. 이 builder의 새 모델/Task/NPZ/CP·통계/trial 재집계·학습/추론/비용·프로세스 제어·Git/network는0이다.
