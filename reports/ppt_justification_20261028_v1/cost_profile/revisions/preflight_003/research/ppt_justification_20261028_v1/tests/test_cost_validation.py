"""Synthetic artifact tests only: never load a GPU, model, or real V outcomes."""
import json

import numpy as np
import pytest

from research.ppt_justification_20261028_v1 import verify_cost_results as verify


def fixture(index=0):
    raw = index == 0
    run_id = verify.FACETS[index]
    row = dict(run_id=run_id, family='RAW' if raw else 'B2', dataset='appliances',
               seed=92602 if index == 2 else 92601, K=26 if raw else 10, C=26, L=512, H=48,
               MSE=.0625, MAE=.25, prediction_path='synthetic-original.npz', prediction_sha256='a'*64,
               bundle_path=None if raw else 'synthetic.pt', bundle_sha256=None if raw else 'b'*64,
               model_state_sha='c'*64)
    reference = dict(pred_normalized=np.full((41, 48, 26), .25, np.float32),
                     target_normalized=np.zeros((41, 48, 26), np.float32), observed=np.ones((41, 48, 26), bool),
                     origins=np.arange(13814, 15741, 48, dtype=np.int64), channel_ids=np.asarray([str(i) for i in range(26)]),
                     mean=np.zeros(26, np.float32), std=np.ones(26, np.float32))
    regenerated = {key: value.copy() for key, value in reference.items()}
    regenerated['reference_pred_normalized'] = reference['pred_normalized'].copy()
    regenerated['pred_normalized'] = regenerated['pred_normalized'].astype(np.float64)
    source_metrics = verify.macro_metrics(reference)
    actual_metrics = verify.macro_metrics(regenerated)
    parity = dict(status='PASS', full_V_origins=41, last_batch_size=1, atol=1e-5, rtol=1e-4, max_abs=0.,
                  original_metrics=source_metrics, independent_metrics=actual_metrics, metric_deltas={'MSE': 0., 'MAE': 0.},
                  lora_merge_parity={'status': 'NOT_APPLICABLE'},
                  input_fingerprints={key: verify.fingerprint(reference[key]) for key in verify.COHORT_KEYS},
                  output_fingerprints=dict(original_normalized=verify.fingerprint(reference['pred_normalized']),
                                          regenerated_normalized=verify.fingerprint(regenerated['pred_normalized']),
                                          regenerated_physical=verify.fingerprint(regenerated['pred_normalized'])))
    trials = []
    for phase, count in (('warmup', 10), ('timed', 20)):
        for i in range(count):
            trials.append(dict(phase=phase, trial=i, index=i, whole_seconds=.01+i*.001,
                               route_internal_whole_seconds=.005, prediction_sha256='d'*64, pid=100+index,
                               CPU_RSS_MiB=300., process_GPU_MiB=None,
                               peak_allocated_MiB=220.+i if phase == 'timed' else None,
                               peak_reserved_MiB=250.+i if phase == 'timed' else None,
                               F_logical_streams=208 if raw else 80, F_physical_streams=208 if raw else 80, F_calls=2 if raw else 1))
    q1, median, q3, p95 = np.quantile([t['whole_seconds']*1000 for t in trials[10:]], [.25, .5, .75, .95])
    summary = dict(facet_id=run_id+'_DEFAULT128_B8', accuracy_run_id=run_id, stage='confirm', status='COMPLETE',
                   family=row['family'], dataset='appliances', seed=row['seed'], K=row['K'], C=26, L=512, H=48,
                   device='cuda', B=8, batch_size=8, requested_row_microbatch=128,
                   row_microbatch=128 if raw else 80, effective_row_microbatch=128 if raw else 80,
                   original=True, is_original=True, checkpoint_sha256=verify.F_SHA if raw else row['bundle_sha256'],
                   prediction_sha256=row['prediction_sha256'], loaded_F_state_sha256=verify.F_SHA, parity_status='PASS',
                   warmups=10, timed_trials=20, F_logical_streams=208 if raw else 80, F_calls=2 if raw else 1,
                   foundation_instances_created=1, core_dtype='float32', backbone_dtype='float32',
                   physical_output_dtype='float64', metric_accumulation_dtype='float64', empty_cache_once_before_loaded_baseline=True,
                   memory_layer='pytorch_allocated_tensor', cached_forecast_reads_in_timed_path=0, stage_probe_trials=0,
                   protocol_sha256='p'*64, p95_scope='20-sample descriptive quantile; no stable tail guarantee',
                   process_scope='fresh OS process, one facet, one serialized model load',
                   peak_reset_scope='after warmup and before every timed trial; reported peak is maximum across timed trials',
                   timing_scope='physical window and bool mask through fill/scaler/model/inverse/CPU output; window extraction and model load excluded',
                   process_pid=100+index, pid=100+index, process_create_time=1000.+index,
                   median_batch_ms=float(median), IQR_batch_ms=float(q3-q1), p95_batch_ms=float(p95),
                   median_ms_per_origin=float(median/8), origins_per_second=float(8000/median),
                   peak_allocated_MiB=239., peak_reserved_MiB=269., timed_peak_allocated_min_MiB=220.,
                   timed_peak_allocated_max_MiB=239., timed_peak_reserved_min_MiB=250., timed_peak_reserved_max_MiB=269.,
                   baseline_allocated_MiB=200., baseline_reserved_MiB=210., model_loaded_CPU_RSS_MiB=280.,
                   resident_weights_gpu_MiB=190., resident_weights_cpu_MiB=0., active_adapter_MiB=0. if raw else 1.,
                   model_load_seconds=1., model_load_with_state_hash_seconds=2., foundation_load_seconds=.5,
                   timing_window_extraction_seconds=.02, full_V_parity_seconds=.7, lora_merge_parity_seconds=0.,
                   process_GPU_sampled_peak_MiB=None, process_GPU_sampling_status='UNAVAILABLE',
                   process_GPU_sampling_reason='Synthetic unavailable per-PID measurement', CPU_RSS_sampled_peak_MiB=320.,
                   MSE=.0625, MAE=.25, recomputed_MSE=.0625, recomputed_MAE=.25)
    sampling = dict(samples=[dict(phase='warmup', elapsed_seconds=0., CPU_RSS_MiB=300., process_GPU_MiB=None),
                             dict(phase='timed', elapsed_seconds=1., CPU_RSS_MiB=320., process_GPU_MiB=None)],
                    sample_count=2, sampling_interval_seconds=.05, sampled_peaks_are_not_instantaneous_guarantees=True,
                    process_GPU_sampled_peak_MiB=None, CPU_RSS_sampled_peak_MiB=320., process_GPU_sampling_status='UNAVAILABLE',
                    process_GPU_sampling_reason=summary['process_GPU_sampling_reason'])
    states = {'F': verify.F_SHA}
    if not raw:
        states['adapter'] = row['model_state_sha']
    accuracy = {key: row[key] for key in ('run_id', 'family', 'dataset', 'seed', 'K', 'C', 'L', 'H', 'MSE', 'MAE')}
    accuracy['prediction_binding'] = dict(path=row['prediction_path'], sha256=row['prediction_sha256'])
    if not raw:
        accuracy['bundle_binding'] = dict(path=row['bundle_path'], sha256=row['bundle_sha256'])
    protocol = {'source_hashes': {'synthetic.py': 'e'*64}}
    receipt = dict(status='COMPLETE', training_executed=False, T_numeric_values_parsed=0, protocol_sha256='p'*64,
                   source_bindings={'runtime_source_sha256': protocol['source_hashes']}, accuracy=accuracy,
                   states_before=states, states_after=dict(states), summary=summary, actual_worker_pid=100+index,
                   environment=dict(device='cuda', sampling_interval_seconds=.05,
                                    CUDA_synchronization='before and after each timed whole path',
                                    autocast=False, quantization=False, compile=False, torch_version='SYNTHETIC',
                                    cuda_runtime='SYNTHETIC', allocator_backend='SYNTHETIC'),
                   CUBLAS_WORKSPACE_CONFIG=None, trials=trials, parity=parity, sampling=sampling)
    return receipt, row, protocol, 'p'*64, reference, regenerated


def check(args):
    return verify.validate_measurement(*args, adapter_tensor_sha='c'*64 if args[1]['family'] == 'B2' else None)


@pytest.mark.parametrize('index', [0, 1, 2])
def test_complete_synthetic_contract(index):
    result = check(fixture(index))
    assert result['run_id'] == verify.FACETS[index]
    assert result['recomputed_statistics']['Q1_batch_ms'] == pytest.approx(14.75)
    assert result['recomputed_statistics']['Q3_batch_ms'] == pytest.approx(24.25)
    assert result['recomputed_statistics']['peak_allocated_MiB'] == 239.


@pytest.mark.parametrize('mutation', [
    lambda r: r['trials'].pop(),
    lambda r: r['trials'][10].update(phase='warmup'),
    lambda r: r['trials'][10].update(whole_seconds=0.),
    lambda r: r['trials'][10].update(whole_seconds=float('nan')),
    lambda r: r['trials'][10].update(peak_allocated_MiB=None),
    lambda r: r['trials'][10].update(peak_allocated_MiB=0.),
    lambda r: r['trials'][10].update(peak_reserved_MiB=1.),
    lambda r: r['trials'][10].update(F_calls=9),
    lambda r: r['trials'][10].update(F_physical_streams=1),
    lambda r: r['trials'][10].update(prediction_sha256='invalid'),
    lambda r: r['summary'].update(median_batch_ms=9.),
    lambda r: r['summary'].update(IQR_batch_ms=1.),
    lambda r: r['summary'].update(p95_batch_ms=1.),
    lambda r: r['summary'].update(peak_allocated_MiB=100.),
    lambda r: r['summary'].update(warmups=1),
    lambda r: r['summary'].update(K=4),
    lambda r: r['summary'].update(requested_row_microbatch=80),
    lambda r: r['summary'].update(backbone_dtype='float16'),
    lambda r: r['summary'].update(resident_weights_gpu_MiB=.001),
    lambda r: r['summary'].update(process_GPU_sampled_peak_MiB=0.),
    lambda r: r['sampling']['samples'][0].update(process_GPU_MiB=0.),
    lambda r: r['states_after'].update(F='0'*64),
    lambda r: r['states_before'].update(adapter='0'*64),
    lambda r: r.update(protocol_sha256='0'*64),
    lambda r: r['source_bindings'].update(runtime_source_sha256={}),
    lambda r: r.update(T_numeric_values_parsed=1),
    lambda r: r.update(training_executed=True),
    lambda r: r['parity'].update(full_V_origins=40),
    lambda r: r['parity']['original_metrics'].update(MSE=.1),
    lambda r: r['environment'].update(autocast=True),
])
def test_bad_measurements_are_rejected(mutation):
    args = fixture(1)
    mutation(args[0])
    with pytest.raises(ValueError):
        check(args)


@pytest.mark.parametrize('field,change', [
    ('origins', lambda a: a.__setitem__(-1, 15788)),
    ('observed', lambda a: a.__setitem__((slice(None), slice(None), 0), False)),
    ('pred_normalized', lambda a: a.__setitem__((0, 0, 0), np.nan)),
    ('target_normalized', lambda a: a.__setitem__((0, 0, 0), 100.)),
    ('std', lambda a: a.__setitem__(0, 0.)),
    ('channel_ids', lambda a: a.__setitem__(0, a[1])),
])
def test_bad_reference_arrays_rejected(field, change):
    args = fixture()
    change(args[4][field])
    with pytest.raises(ValueError):
        check(args)


def test_saved_regeneration_drift_rejected_even_with_consistent_fingerprint():
    args = fixture()
    args[5]['pred_normalized'] += .5
    args[0]['parity']['output_fingerprints']['regenerated_normalized'] = verify.fingerprint(args[5]['pred_normalized'])
    with pytest.raises(ValueError, match='prediction parity'):
        check(args)


def test_masked_macro_weights_channels_equally():
    values = fixture()[4]
    values['observed'][1:, :, 0] = False
    values['pred_normalized'][:, :, 0] = 1.
    result = verify.macro_metrics(values)
    assert result['MSE'] == pytest.approx((1. + 25*.0625)/26)
    assert result['MAE'] == pytest.approx((1. + 25*.25)/26)


def test_table_uses_mean_seed_summaries_not_pooled_latency_or_best_seed():
    checks = [check(fixture(i)) for i in range(3)]
    checks[0]['recomputed_statistics'].update(median_batch_ms=100., peak_allocated_MiB=400.)
    checks[1]['recomputed_statistics'].update(median_batch_ms=20., peak_allocated_MiB=100.)
    checks[2]['recomputed_statistics'].update(median_batch_ms=30., peak_allocated_MiB=300.)
    table = verify.table_values(checks)
    assert table['ours_mean_seed_median_batch_ms'] == 25.
    assert table['ours_mean_seed_peak_allocated_MiB'] == 200.
    assert table['speedup_raw_over_ours'] == 4.
    assert table['peak_allocated_change_pct'] == -50.
    with pytest.raises(ValueError):
        verify.table_values(checks[:2])
    with pytest.raises(ValueError):
        verify.table_values([checks[0], checks[1], checks[1]])


def test_missing_real_measurements_never_produce_validation(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, 'REPORT', tmp_path)
    with pytest.raises(ValueError, match='Actual complete measurements required'):
        verify.validate(tmp_path)
    assert not (tmp_path/'COST_VALIDATION.json').exists()


def test_adapter_digest_algorithms_are_distinct_and_order_stable():
    state = {'z': np.asarray([1., 2.], dtype=np.float32), 'a': np.asarray([3.], dtype=np.float32)}
    sorted_sha, bundle_sha = verify.tensor_hashes(state)
    assert sorted_sha != bundle_sha
    assert verify.tensor_hashes(dict(reversed(list(state.items())))) == (sorted_sha, bundle_sha)
    state['a'][0] = 4.
    assert verify.tensor_hashes(state) != (sorted_sha, bundle_sha)


def test_adapter_requires_independent_original_tensor_digest():
    with pytest.raises(ValueError, match='original checkpoint'):
        verify.validate_measurement(*fixture(1), adapter_tensor_sha='0'*64)


@pytest.mark.parametrize('extra_row,wrong_scaler', [(False, False), (True, False), (False, True)])
def test_prefix_header_and_independent_A60_scaler(tmp_path, extra_row, wrong_scaler):
    receipt, _, _, _, reference, _ = fixture()
    physical = np.zeros((15788+int(extra_row), 26), np.float32)
    observed = np.ones(physical.shape, bool)
    prefix, config = tmp_path/'prefix.npz', tmp_path/'config.json'
    np.savez_compressed(prefix, physical=physical, observed=observed, columns=reference['channel_ids'])
    config.write_text(json.dumps(dict(prefix_only_approved=True, prefix_rows=15788,
                                     split_ranges=dict(V_select=[13814, 15788], T_final=[15788, 20000]))), encoding='utf8')
    receipt['parity']['input_fingerprints'].update(
        physical_windows=verify.fingerprint(np.zeros((41, 512, 26), np.float32)),
        physical_observed_windows=verify.fingerprint(np.ones((41, 512, 26), bool)))
    if wrong_scaler:
        reference['mean'][0] = 1.
    safe = dict(prefix_path=str(prefix), config_path=str(config))
    if extra_row or wrong_scaler:
        with pytest.raises(ValueError, match='header|scaler'):
            verify.validate_prefix(safe, reference, [receipt])
    else:
        verify.validate_prefix(safe, reference, [receipt])
