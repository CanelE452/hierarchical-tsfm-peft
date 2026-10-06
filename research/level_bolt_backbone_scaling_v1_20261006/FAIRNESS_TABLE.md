# Comparison fairness

[확인] 아래 값은 봉인된 자료·모델 manifest, 사전 실제 로드 receipt와 이번 비용 receipt에서 읽었습니다. 추론 중 gradient를 사용하는 변수 수는 모든 arm에서 0이며, target-data fit 변수 수는 기존 LEVEL의 G 적합 이력입니다. 이번 회차 새 fit은 0입니다.

Arm names: `BOLT_SMALL_LEVEL`, `BOLT_SMALL_F0`, `BOLT_MINI_F0`, `BOLT_TINY_F0`.

## Robin — C17/K5

| Attribute | Small + LEVEL | Small F0 | Mini F0 | Tiny F0 |
| --- | --- | --- | --- | --- |
| Family | Chronos-Bolt | Chronos-Bolt | Chronos-Bolt | Chronos-Bolt |
| Architecture | ChronosBoltModelForForecasting | ChronosBoltModelForForecasting | ChronosBoltModelForForecasting | ChronosBoltModelForForecasting |
| Pinned model revision | 772f3d25d38aec6d914c8949dab4462e2d46f5d8 | 772f3d25d38aec6d914c8949dab4462e2d46f5d8 | 251268337516a88e253628c43e1d26ec577b376b | a0e552de83495b5c28c14c71c374f3e33280b340 |
| Actual backbone params | 47718016 | 47718016 | 21236096 | 8652672 |
| Actual backbone parameter bytes | 190872064 | 190872064 | 84944384 | 34610688 |
| Total deployed params | 47736106 | 47718016 | 21236096 | 8652672 |
| Deployed parameter bytes | 190944424 | 190872064 | 84944384 | 34610688 |
| Deployed buffer bytes | 36 | 36 | 36 | 36 |
| Total deployed tensor bytes | 190944460 | 190872100 | 84944420 | 34610724 |
| Previously fitted target-data params | 17920 | 0 | 0 | 0 |
| Inference gradient-enabled params | 0 | 0 | 0 | 0 |
| Original input channels C | 17 | 17 | 17 | 17 |
| TSFM rows per origin | 5 | 17 | 17 | 17 |
| TRAIN PCA | Fixed C-to-K E/D | None | None | None |
| Residual G | Previously selected, frozen | None | None | None |
| Target-data fitting history | TRAIN PCA and selected G | None | None | None |
| New fits in this campaign | 0 | 0 | 0 | 0 |
| Context / horizon observations | 512 / 48 | 512 / 48 | 512 / 48 | 512 / 48 |
| Precision / TF32 | FP32 / off | FP32 / off | FP32 / off | FP32 / off |
| Same preprocessing | Yes: existing TRAIN statistics and input imputation | Yes: existing TRAIN statistics and input imputation | Yes: existing TRAIN statistics and input imputation | Yes: existing TRAIN statistics and input imputation |
| Same origins / target mask | Yes: sealed origin order and raw-finite target mask | Yes: sealed origin order and raw-finite target mask | Yes: sealed origin order and raw-finite target mask | Yes: sealed origin order and raw-finite target mask |
| Point forecast | Native q=0.5 -> fixed decode + G | Native q=0.5, first H48 | Native q=0.5, first H48 | Native q=0.5, first H48 |
| Loss aggregation | Mean losses of 2 selected seeds | One deterministic model | One deterministic model | One deterministic model |
| TEST exposure | Previously exposed; post-exposure practical control | Previously exposed; post-exposure practical control | Previously exposed; post-exposure practical control | Previously exposed; post-exposure practical control |
| Pretraining overlap | Unverified | Unverified | Unverified | Unverified |

## Peacock Education — C13/K4

| Attribute | Small + LEVEL | Small F0 | Mini F0 | Tiny F0 |
| --- | --- | --- | --- | --- |
| Family | Chronos-Bolt | Chronos-Bolt | Chronos-Bolt | Chronos-Bolt |
| Architecture | ChronosBoltModelForForecasting | ChronosBoltModelForForecasting | ChronosBoltModelForForecasting | ChronosBoltModelForForecasting |
| Pinned model revision | 772f3d25d38aec6d914c8949dab4462e2d46f5d8 | 772f3d25d38aec6d914c8949dab4462e2d46f5d8 | 251268337516a88e253628c43e1d26ec577b376b | a0e552de83495b5c28c14c71c374f3e33280b340 |
| Actual backbone params | 47718016 | 47718016 | 21236096 | 8652672 |
| Actual backbone parameter bytes | 190872064 | 190872064 | 84944384 | 34610688 |
| Total deployed params | 47736040 | 47718016 | 21236096 | 8652672 |
| Deployed parameter bytes | 190944160 | 190872064 | 84944384 | 34610688 |
| Deployed buffer bytes | 244 | 244 | 36 | 36 |
| Total deployed tensor bytes | 190944404 | 190872308 | 84944420 | 34610724 |
| Previously fitted target-data params | 17920 | 0 | 0 | 0 |
| Inference gradient-enabled params | 0 | 0 | 0 | 0 |
| Original input channels C | 13 | 13 | 13 | 13 |
| TSFM rows per origin | 4 | 13 | 13 | 13 |
| TRAIN PCA | Fixed C-to-K E/D | None | None | None |
| Residual G | Previously selected, frozen | None | None | None |
| Target-data fitting history | TRAIN PCA and selected G | None | None | None |
| New fits in this campaign | 0 | 0 | 0 | 0 |
| Context / horizon observations | 512 / 48 | 512 / 48 | 512 / 48 | 512 / 48 |
| Precision / TF32 | FP32 / off | FP32 / off | FP32 / off | FP32 / off |
| Same preprocessing | Yes: existing TRAIN statistics and input imputation | Yes: existing TRAIN statistics and input imputation | Yes: existing TRAIN statistics and input imputation | Yes: existing TRAIN statistics and input imputation |
| Same origins / target mask | Yes: sealed origin order and raw-finite target mask | Yes: sealed origin order and raw-finite target mask | Yes: sealed origin order and raw-finite target mask | Yes: sealed origin order and raw-finite target mask |
| Point forecast | Native q=0.5 -> fixed decode + G | Native q=0.5, first H48 | Native q=0.5, first H48 | Native q=0.5, first H48 |
| Loss aggregation | Mean losses of 2 selected seeds | One deterministic model | One deterministic model | One deterministic model |
| TEST exposure | Previously exposed; post-exposure practical control | Previously exposed; post-exposure practical control | Previously exposed; post-exposure practical control | Previously exposed; post-exposure practical control |
| Pretraining overlap | Unverified | Unverified | Unverified | Unverified |

## Jena — C21/K6

| Attribute | Small + LEVEL | Small F0 | Mini F0 | Tiny F0 |
| --- | --- | --- | --- | --- |
| Family | Chronos-Bolt | Chronos-Bolt | Chronos-Bolt | Chronos-Bolt |
| Architecture | ChronosBoltModelForForecasting | ChronosBoltModelForForecasting | ChronosBoltModelForForecasting | ChronosBoltModelForForecasting |
| Pinned model revision | 772f3d25d38aec6d914c8949dab4462e2d46f5d8 | 772f3d25d38aec6d914c8949dab4462e2d46f5d8 | 251268337516a88e253628c43e1d26ec577b376b | a0e552de83495b5c28c14c71c374f3e33280b340 |
| Actual backbone params | 47718016 | 47718016 | 21236096 | 8652672 |
| Actual backbone parameter bytes | 190872064 | 190872064 | 84944384 | 34610688 |
| Total deployed params | 47736188 | 47718016 | 21236096 | 8652672 |
| Deployed parameter bytes | 190944752 | 190872064 | 84944384 | 34610688 |
| Deployed buffer bytes | 36 | 36 | 36 | 36 |
| Total deployed tensor bytes | 190944788 | 190872100 | 84944420 | 34610724 |
| Previously fitted target-data params | 17920 | 0 | 0 | 0 |
| Inference gradient-enabled params | 0 | 0 | 0 | 0 |
| Original input channels C | 21 | 21 | 21 | 21 |
| TSFM rows per origin | 6 | 21 | 21 | 21 |
| TRAIN PCA | Fixed C-to-K E/D | None | None | None |
| Residual G | Previously selected, frozen | None | None | None |
| Target-data fitting history | TRAIN PCA and selected G | None | None | None |
| New fits in this campaign | 0 | 0 | 0 | 0 |
| Context / horizon observations | 512 / 48 | 512 / 48 | 512 / 48 | 512 / 48 |
| Precision / TF32 | FP32 / off | FP32 / off | FP32 / off | FP32 / off |
| Same preprocessing | Yes: existing TRAIN statistics and input imputation | Yes: existing TRAIN statistics and input imputation | Yes: existing TRAIN statistics and input imputation | Yes: existing TRAIN statistics and input imputation |
| Same origins / target mask | Yes: sealed origin order and raw-finite target mask | Yes: sealed origin order and raw-finite target mask | Yes: sealed origin order and raw-finite target mask | Yes: sealed origin order and raw-finite target mask |
| Point forecast | Native q=0.5 -> fixed decode + G | Native q=0.5, first H48 | Native q=0.5, first H48 | Native q=0.5, first H48 |
| Loss aggregation | Mean losses of 2 selected seeds | One deterministic model | One deterministic model | One deterministic model |
| TEST exposure | Previously exposed; post-exposure practical control | Previously exposed; post-exposure practical control | Previously exposed; post-exposure practical control | Previously exposed; post-exposure practical control |
| Pretraining overlap | Unverified | Unverified | Unverified | Unverified |

[확인] Mini/Tiny는 native multivariate 모델이 아니라 같은 Chronos-Bolt 계열의 더 작은 backbone입니다. 모든 F0는 `[B,512,C] -> [B*C,512]`로 원채널을 서로 독립적인 단변량 task로 처리합니다. LEVEL만 fixed TRAIN PCA 뒤 K개 task를 처리하고 원채널로 decode한 뒤 G를 더합니다. 부모 backbone과 배포 전체 크기를 혼동하지 않으며, 고정 E/D와 G도 배포 tensor bytes에 포함됩니다.

이 표는 `model_manifest.json`, `input_contract.json`, `preflight_checks.json`, `cost_comparison.csv` 및 `cost_cuda_summary.json`에 연결됩니다. 최종 검산·게시 상태는 `final_checks.json`과 `STATUS.md`에서 별도로 확인합니다.
