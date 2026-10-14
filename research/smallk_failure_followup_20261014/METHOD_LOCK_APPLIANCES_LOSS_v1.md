# Appliances loss warm-start method lock v1

이 문서는 Appliances 전용 loss 비교 준비의 source/API 계약이다. 실제 array/cache RESULT는 `prepare_appliances_inputs.py --execute`가 새 follow-up cache에 별도로 쓴다.

## 결정 문맥

- [확인] downstream 결정은 기존 old selected `S3_ANCHOR_NONLINEAR_D_LEARNED_REGIME_w0.1` codec에서 warm start한 뒤, 새 loss만 MSE/MAE/Huber δ1로 바꾸는 6개 fit의 비교다.
- [확인] primary selection metric은 계속 TUNE actual Y normalized observed channel-macro MSE다. MAE/Huber는 training loss 후보일 뿐 TUNE-MAE constraint나 MAE-primary selection으로 바꾸지 않는다.
- [확인] 새 후보는 각 seed/loss마다 600 updates, batch size 4, checkpoint `u0/u100/u200/u400/u600`을 평가한다.
- [확인] frozen F와 old codec architecture/state를 시작점으로 사용한다. q auxiliary/readout state는 old checkpoint provenance로 보존하지만, deployment와 새 loss 학습에는 사용하지 않는다.

## 권위 입력 계약

- Dataset: `appliances`
- Window/forecast: `L=512`, `H=48`
- Channels and basis: `C=26`, `K=6`
- Splits: FIT 256, TUNE 64, EVAL 82
- Seeds: `92601`, `92602`
- Leakage guard: TEST path read 금지, 새 `open_evaluation()` 호출 금지, old campaign restart 금지, GPU 실행 금지.

## old selected checkpoint 계약

- Seed `92601`: old selected update `600`, checkpoint `.cache/smallk_predictive_alignment_20261013/run_new_v1/appliances/S3_ANCHOR_NONLINEAR_D_LEARNED_REGIME_w0.1_s92601/u600.pt`, SHA `74dcda79543d5b82bbd46f007e26e43c7cb9bb17aa7c31c1644df243c469ec28`
- Seed `92602`: old selected update `100`, checkpoint `.cache/smallk_predictive_alignment_20261013/run_new_v1/appliances/S3_ANCHOR_NONLINEAR_D_LEARNED_REGIME_w0.1_s92602/u100.pt`, SHA `b858fea838fd23b65462b27bb7fe426457796807ac0e593abe410f4178b5ed77`

The CPU bridge verifies these hashes and payload metadata before exposing them to the runner.

## Runner API

`research.smallk_failure_followup_20261014.prepare_appliances_inputs.load_prepared_bundle(result_path)` returns:

- `manifest`: full `RESULT.json`
- `splits`: per split NPZ arrays
- `mean`, `std`, `columns`
- `checkpoint_cases`
- `raw_references`
- `method_plan`

Each split NPZ contains:

- `history`: normalized FP32 history for the original engine
- `history_physical`: raw physical past before causal fill/normalization
- `history_observed`: past observed mask
- `target`, `observed`, `origins`
- `raw_last`
- `raw_forecast`, `raw_plus_r_forecast`
- `learned_92601_forecast`, `learned_92602_forecast`
- `mean`, `std`

`load_codec_checkpoint_state(checkpoint_record)` verifies checkpoint SHA, loads the checkpoint on CPU, and returns `codec`, `q`, `update`, `seed`, `architecture`, `method`, `weight`, and `q_aux_deployment_use="preserved_not_used"`.

`raw_references["RAW_BOLT_RAW_R"]` points to the already exposed RAW+R prediction NPZ. Its schema is the old `engine.add_prediction` schema: `{split}_prediction`, `{split}_target`, `{split}_observed`, `{split}_origins`, `mean`, and `std`.
