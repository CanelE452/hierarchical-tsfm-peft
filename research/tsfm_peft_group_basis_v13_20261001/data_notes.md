# v13 group-basis data notes

This directory uses four already exposed development datasets: Robin, Jena,
Hog, and Peacock Education. No new raw dataset, download, period, target, or
column search is part of this preparation step.

The downstream decision is whether a fixed TRAIN-only disjoint positive group
basis is a better controlled representation than the prior PCA basis while
holding K, F0, G capacity, seeds, and learning policy fixed. The main leakage
risks are TEST-informed grouping, changing source periods/columns, and copying
trained residual weights from the old PCA basis. `prepare_v13.py` therefore
builds the grouping only from the saved TRAIN slice defined by each source
archive's `train_start`/`train_end`, preserves the original PCA basis as
`pca_basis`, and copies only the original untrained LEVEL
`residual.down.weight` and `residual.up.weight` tensors.

`reuse_manifest.json` is a read-only receipt manifest. It binds:

- `datasets[dataset].data.trainval/test`: existing v11/v12 source NPZ receipts.
- `datasets[dataset].initial_g_sources[str(seed)]`: existing untrained LEVEL
  G source checkpoint, tensor hashes, source result, and excluded selected
  checkpoint.
- `datasets[dataset].references`: 11 flat old reference rows per dataset:
  two `pca_level`, two `a`, one `f0`, two `full_mse`, two `native`, and two
  `direct` rows. Each row has canonical `family`, original `source_family`,
  `source_model_id`, `prediction`, `source_evaluation`, optional `checkpoint`,
  `restore`, and the original `source_row`.

Root should run preparation through the built-in ledger wrapper, for example:

```powershell
python research\tsfm_peft_group_basis_v13_20261001\prepare_v13.py prepare --job v13_group_basis_prepare_01 --reserve-s 150
```

The `prepare` action refuses to run without `--job` and `--reserve-s`; internally
it opens `runtime_v13.Job("cpu_analysis", ...)` before writing generated arrays.
When root runs that command, the generated runtime data contract is:

- `data_manifest.json`: `schema=v13_group_basis_data_manifest_1`;
  `datasets[dataset]` contains `trainval`, `test`, `initial_g`, `basis_sha256`,
  `groups`, `columns`, `C`, `K`, `block_origins`, and `phase_period`.
- `.cache/tsfm_peft_group_basis_v13_20261001/data/{dataset}/{trainval,test}.npz`:
  existing arrays copied with `basis` replaced by the new groupU and original
  `basis` retained as `pca_basis`.
- `.cache/tsfm_peft_group_basis_v13_20261001/initial_g/{dataset}_s{seed}_initial_g.pt`:
  `schema=v13_initial_g_1`, `state` with only the two residual G tensors,
  `tensor_hashes`, `source`, and `source_result`.
- `data_contract_group_basis_{dataset}.json`: per-dataset grouping details,
  including TRAIN bounds, groups, pairwise TRAIN correlation/distance table,
  fallback pairs, orthonormality error, source contracts, generated data
  receipts, and reference reuse rows.

Grouping rule: compute Pearson correlation for each pair using only common
finite observations in the actual TRAIN segment. If fewer than two common
observations or zero pair variance occurs, use rho=0 and distance sqrt(2), and
record the pair in `train_missing_pairs`. Distance is
`sqrt(max(0, 2 - 2 * clip(rho, -1, 1)))`; negative correlation is not converted
to absolute correlation. Average linkage uses mean original pair distances and
a deterministic lexicographic leaf-tuple tie rule. Final groups are sorted by
minimum original channel index. The stored group basis has
`U[c,j]=1/sqrt(group_size)` for members and zero otherwise.

Robin's source result ids are `v6_robin_level_linear_res_92601/92602`; this is
the original v6 residual LEVEL initialization source for the selected PCA LEVEL
run, not a same-seed reconstruction. Jena/Hog/Peacock use the corresponding
saved LEVEL result sources listed in `reuse_manifest.json`. For all four
datasets, execution verifies the saved untrained checkpoint sha256 and the
two residual G tensor hashes before copying them. Peacock v12 does not expose
the older `initial_parameter_sha256` field name, so its binding is the stored
`initial_source.sha256` plus per-tensor hashes.

As of this note, only read/hash receipt generation and static CLI checks have
been run here. New group bases, data archives, initial-G copy artifacts,
contracts, TEST predictions, scores, and model fits are not produced until
root runs the reserved preparation job.
