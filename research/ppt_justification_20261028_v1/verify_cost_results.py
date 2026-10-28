"""CPU-only verification of the three sealed Appliances cost measurements."""
import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import host_gpu_gate


REPO = Path(__file__).resolve().parents[2]
NAME = 'ppt_justification_20261028_v1'
REPORT = REPO / 'reports' / NAME / 'cost_profile'
FACETS = ('Raw_appliances', 'KTAU_main_appliances_K10_s92601',
          'KTAU_main_appliances_K10_s92602')
PREVIOUS_SEAL_SHA = '7c3e8880b563cf9ab5bc5bfe8993b7ec5bc359b219e2e6b13be549798bd8368e'
F_SHA = 'c2b4edf7db22d3f18e58ca79299a5c7e864e9b8f277e4e29e2f88d07dbb35874'
COHORT_KEYS = ('target_normalized', 'observed', 'origins', 'channel_ids', 'mean', 'std')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def fingerprint(value):
    value = np.ascontiguousarray(value)
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes()).hexdigest())


def digest_valid(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def number(value, label, *, positive=True):
    require(type(value) in (int, float) and math.isfinite(value), label + ': finite number required')
    require(value > 0 if positive else value >= 0, label + ': invalid sign/zero')
    return value


def close(actual, expected, label, *, atol=1e-10, rtol=1e-10):
    number(actual, label, positive=False)
    require(math.isclose(actual, expected, abs_tol=atol, rel_tol=rtol), label + ': recomputation differs')


def macro_metrics(values):
    pred, target, mask = (values[key] for key in ('pred_normalized', 'target_normalized', 'observed'))
    require(pred.shape == target.shape == mask.shape == (41, 48, 26), 'Incomplete full-V target shape')
    require(mask.dtype == np.bool_ and np.isfinite(pred).all() and np.isfinite(target[mask]).all(),
            'Invalid mask or nonfinite scored predictions/targets')
    counts, squared, absolute = [], [], []
    for channel in range(26):
        observed = mask[:, :, channel]
        error = pred[:, :, channel][observed].astype(np.float64) - target[:, :, channel][observed].astype(np.float64)
        require(error.size > 0, 'Every channel needs observed targets')
        counts.append(int(error.size))
        squared.append(math.fsum(float(x) * float(x) for x in error))
        absolute.append(math.fsum(abs(float(x)) for x in error))
    return dict(MSE=math.fsum(s / n for s, n in zip(squared, counts)) / 26,
                MAE=math.fsum(s / n for s, n in zip(absolute, counts)) / 26,
                observed=sum(counts), channel_sse=squared, channel_sae=absolute, channel_counts=counts,
                accumulation_dtype='float64', aggregation='equal-channel observed macro')


def validate_arrays(reference, regenerated, row, parity):
    expected_origins = np.arange(13814, 15788 - 48 + 1, 48, dtype=np.int64)
    require(reference['origins'].dtype == np.int64 and np.array_equal(reference['origins'], expected_origins),
            'Original V origins/order differ or approach protected T')
    require(int(reference['origins'][-1]) + 48 == 15782 < 15788, 'Protected T boundary exceeded')
    require(reference['channel_ids'].shape == (26,) and len(set(reference['channel_ids'].tolist())) == 26,
            'Channel identity/order invalid')
    for key in ('mean', 'std'):
        require(reference[key].shape == (26,) and np.isfinite(reference[key]).all(), 'Invalid scaler ' + key)
    require((reference['std'] > 0).all(), 'Nonpositive scaler std')
    for key in COHORT_KEYS:
        require(fingerprint(reference[key]) == fingerprint(regenerated[key]), 'Regenerated cohort differs: ' + key)
        require(parity['input_fingerprints'][key] == fingerprint(reference[key]), 'Input fingerprint differs: ' + key)
    require(fingerprint(regenerated['reference_pred_normalized']) == fingerprint(reference['pred_normalized']),
            'Regenerated archive is bound to another original prediction')
    for key, value in (('original_normalized', reference['pred_normalized']),
                       ('regenerated_normalized', regenerated['pred_normalized'])):
        require(parity['output_fingerprints'][key] == fingerprint(value), 'Output fingerprint differs: ' + key)
    source, actual = macro_metrics(reference), macro_metrics(regenerated)
    require(np.allclose(reference['pred_normalized'], regenerated['pred_normalized'], atol=1e-5, rtol=1e-4),
            'Full-V prediction parity failed')
    close(parity['max_abs'], float(np.max(np.abs(reference['pred_normalized'] - regenerated['pred_normalized']))), 'Parity max_abs')
    for key in ('MSE', 'MAE'):
        close(source[key], row[key], 'Registered source ' + key, atol=1e-12, rtol=0)
        close(actual[key], source[key], 'Regenerated ' + key, atol=1e-5, rtol=1e-4)
        delta = parity['metric_deltas'][key]
        require(type(delta) in (float, int) and math.isfinite(delta) and
                math.isclose(delta, actual[key] - source[key], abs_tol=1e-12, rel_tol=1e-10), 'Metric delta differs')
    for key, calculated in (('original_metrics', source), ('independent_metrics', actual)):
        recorded = parity[key]
        for field in ('MSE', 'MAE'):
            close(recorded[field], calculated[field], key + '.' + field, atol=1e-12)
        for field in ('observed', 'channel_counts', 'aggregation', 'accumulation_dtype'):
            require(recorded[field] == calculated[field], key + '.' + field + ': differs')
        for field in ('channel_sse', 'channel_sae'):
            require(np.shape(recorded[field]) == (26,) and np.allclose(recorded[field], calculated[field], atol=1e-10, rtol=1e-12),
                    key + '.' + field + ': differs')
    physical = parity['output_fingerprints']['regenerated_physical']
    require(physical['shape'] == [41, 48, 26] and physical['dtype'] == 'float64' and digest_valid(physical['sha256']),
            'Invalid physical-output receipt fingerprint')
    return dict(source_metrics=source, regenerated_metrics=actual,
                max_abs_prediction_difference=parity['max_abs'], target_end_exclusive=15782, T_start=15788)


def validate_measurement(receipt, row, protocol, protocol_sha, reference, regenerated, *, adapter_tensor_sha=None):
    run_id = row['run_id']
    require(run_id in FACETS, 'Unregistered cost facet')
    raw = run_id == FACETS[0]
    seed = 92602 if run_id == FACETS[2] else 92601
    expected = dict(family='RAW' if raw else 'B2', dataset='appliances', seed=seed,
                    K=26 if raw else 10, C=26, L=512, H=48)
    require(all(row[key] == value for key, value in expected.items()), 'Registered facet identity differs')
    require(receipt['status'] == 'COMPLETE' and receipt['training_executed'] is False and
            receipt['T_numeric_values_parsed'] == 0, 'Incomplete/trained/protected-T receipt')
    require(receipt['protocol_sha256'] == protocol_sha, 'Stale measurement protocol binding')
    require(receipt['source_bindings'] == {'runtime_source_sha256': protocol['source_hashes']}, 'Stale producer source bindings')
    candidate = {key: row[key] for key in ('run_id', 'family', 'dataset', 'seed', 'K', 'C', 'L', 'H', 'MSE', 'MAE')}
    candidate['prediction_binding'] = dict(path=row['prediction_path'], sha256=row['prediction_sha256'])
    if not raw:
        candidate['bundle_binding'] = dict(path=row['bundle_path'], sha256=row['bundle_sha256'])
    require(receipt['accuracy'] == candidate, 'Original accuracy/bundle binding differs')
    states = receipt['states_before']
    require(set(states) == ({'F'} if raw else {'F', 'adapter'}) and all(digest_valid(value) for value in states.values()) and
            states == receipt['states_after'], 'Adapter/F state changed or differs')
    if not raw:
        require(digest_valid(adapter_tensor_sha) and states['adapter'] == adapter_tensor_sha,
                'Loaded adapter sorted tensor hash differs from original checkpoint')
    summary = receipt['summary']
    fields = dict(expected, facet_id=run_id + '_DEFAULT128_B8', accuracy_run_id=run_id,
                  stage='confirm', status='COMPLETE', device='cuda', B=8, batch_size=8,
                  requested_row_microbatch=128, row_microbatch=128 if raw else 80,
                  effective_row_microbatch=128 if raw else 80, original=True, is_original=True,
                  checkpoint_sha256=F_SHA if raw else row['bundle_sha256'], prediction_sha256=row['prediction_sha256'],
                  loaded_F_state_sha256=F_SHA, parity_status='PASS', warmups=10, timed_trials=20,
                  F_logical_streams=208 if raw else 80, F_calls=2 if raw else 1,
                  foundation_instances_created=1, core_dtype='float32', backbone_dtype='float32',
                  physical_output_dtype='float64', metric_accumulation_dtype='float64',
                  empty_cache_once_before_loaded_baseline=True, memory_layer='pytorch_allocated_tensor',
                  cached_forecast_reads_in_timed_path=0, stage_probe_trials=0, protocol_sha256=protocol_sha,
                  p95_scope='20-sample descriptive quantile; no stable tail guarantee',
                  process_scope='fresh OS process, one facet, one serialized model load',
                  peak_reset_scope='after warmup and before every timed trial; reported peak is maximum across timed trials',
                  timing_scope='physical window and bool mask through fill/scaler/model/inverse/CPU output; window extraction and model load excluded')
    require(all(summary.get(key) == value for key, value in fields.items()), 'Summary execution contract differs')
    pid = receipt['actual_worker_pid']
    require(type(pid) is int and pid > 0 and summary['process_pid'] == summary['pid'] == pid, 'Worker PID mismatch')
    number(summary['process_create_time'], 'process_create_time')
    environment = receipt['environment']
    require(all(environment.get(key) == value for key, value in dict(device='cuda', sampling_interval_seconds=.05,
                CUDA_synchronization='before and after each timed whole path', autocast=False, quantization=False, compile=False).items()),
            'CUDA/timing environment differs')
    require(bool(environment['torch_version']) and bool(environment['cuda_runtime']) and bool(environment['allocator_backend']),
            'CUDA runtime evidence missing')
    require(receipt['CUBLAS_WORKSPACE_CONFIG'] is None or isinstance(receipt['CUBLAS_WORKSPACE_CONFIG'], str),
            'Invalid recorded CUBLAS environment')
    trials = receipt['trials']
    require(len(trials) == 30, 'Exactly 10 warmups and 20 timed trials required')
    for position, trial in enumerate(trials):
        warm = position < 10
        index = position if warm else position - 10
        require((trial['phase'], trial['index'], trial['trial']) == ('warmup' if warm else 'timed', index, index),
                'Trial order/index incomplete')
        require(trial['pid'] == pid and digest_valid(trial['prediction_sha256']), 'Invalid trial PID/output hash')
        require((trial['F_logical_streams'], trial['F_physical_streams'], trial['F_calls']) ==
                (208 if raw else 80, 208 if raw else 80, 2 if raw else 1), 'Actual F trial work differs')
        number(trial['whole_seconds'], 'Trial latency')
        if trial['route_internal_whole_seconds'] is not None:
            number(trial['route_internal_whole_seconds'], 'Internal latency')
        number(trial['CPU_RSS_MiB'], 'Trial CPU memory')
        if trial['process_GPU_MiB'] is not None:
            number(trial['process_GPU_MiB'], 'Trial process GPU memory')
        if warm:
            require(trial['peak_allocated_MiB'] is None and trial['peak_reserved_MiB'] is None, 'Warmup peak used as timed peak')
        else:
            number(trial['peak_allocated_MiB'], 'Timed allocated MiB')
            number(trial['peak_reserved_MiB'], 'Timed reserved MiB')
            require(trial['peak_reserved_MiB'] >= trial['peak_allocated_MiB'], 'Reserved CUDA memory below allocated')
    timed = trials[10:]
    q1, median, q3, p95 = np.quantile([trial['whole_seconds'] * 1000 for trial in timed], [.25, .5, .75, .95])
    computed = dict(Q1_batch_ms=float(q1), median_batch_ms=float(median), Q3_batch_ms=float(q3),
                    IQR_batch_ms=float(q3-q1), p95_batch_ms=float(p95), median_ms_per_origin=float(median/8),
                    origins_per_second=float(8000/median))
    for kind in ('allocated', 'reserved'):
        peaks = [trial['peak_' + kind + '_MiB'] for trial in timed]
        computed.update({f'peak_{kind}_MiB': max(peaks), f'timed_peak_{kind}_min_MiB': min(peaks),
                         f'timed_peak_{kind}_max_MiB': max(peaks)})
    for key, value in computed.items():
        if key not in ('Q1_batch_ms', 'Q3_batch_ms'):
            close(summary[key], value, key)
    for key in ('baseline_allocated_MiB', 'baseline_reserved_MiB', 'model_loaded_CPU_RSS_MiB',
                'resident_weights_gpu_MiB', 'model_load_seconds', 'model_load_with_state_hash_seconds',
                'foundation_load_seconds', 'timing_window_extraction_seconds', 'full_V_parity_seconds'):
        number(summary[key], key)
    for key in ('resident_weights_cpu_MiB', 'active_adapter_MiB', 'lora_merge_parity_seconds'):
        number(summary[key], key, positive=False)
    require(summary['resident_weights_gpu_MiB'] >= 47718016 * 4 / 1024**2 and
            summary['baseline_allocated_MiB'] >= summary['resident_weights_gpu_MiB'] and
            summary['baseline_reserved_MiB'] >= summary['baseline_allocated_MiB'] and
            min(trial['peak_allocated_MiB'] for trial in timed) >= summary['baseline_allocated_MiB'],
            'CUDA MiB/loaded-weight/peak relationship invalid')
    require(summary['model_load_with_state_hash_seconds'] >= summary['model_load_seconds'], 'Loader time exceeds wrapper time')
    validate_sampling(receipt)
    parity = receipt['parity']
    require(all(parity.get(key) == value for key, value in dict(status='PASS', full_V_origins=41,
                last_batch_size=1, atol=1e-5, rtol=1e-4, lora_merge_parity={'status': 'NOT_APPLICABLE'}).items()),
            'Full-V parity receipt incomplete')
    metrics = validate_arrays(reference, regenerated, row, parity)
    for key in ('MSE', 'MAE'):
        close(summary[key], metrics['source_metrics'][key], 'Summary source ' + key, atol=1e-12)
        close(summary['recomputed_' + key], metrics['regenerated_metrics'][key], 'Summary regenerated ' + key, atol=1e-12)
    return dict(run_id=run_id, seed=seed, family=expected['family'], recomputed_statistics=computed,
                full_V=metrics, worker_pid=pid, worker_create_time=summary['process_create_time'],
                invariant_sorted_tensor_hashes=states, loaded_foundation_native_order_sha256=F_SHA)


def validate_sampling(receipt):
    sampling, summary = receipt['sampling'], receipt['summary']
    samples = sampling['samples']
    require(samples and sampling['sample_count'] == len(samples) and sampling['sampling_interval_seconds'] == .05 and
            sampling['sampled_peaks_are_not_instantaneous_guarantees'] is True, 'Memory sampling incomplete')
    last_elapsed = -1
    for sample in samples:
        number(sample['elapsed_seconds'], 'Sample elapsed', positive=False)
        require(sample['elapsed_seconds'] >= last_elapsed and sample['phase'] in ('warmup', 'timed'), 'Sample order/phase invalid')
        last_elapsed = sample['elapsed_seconds']
        number(sample['CPU_RSS_MiB'], 'Sample CPU memory')
        if sample['process_GPU_MiB'] is not None:
            number(sample['process_GPU_MiB'], 'Sample GPU memory')
    cpu = max(sample['CPU_RSS_MiB'] for sample in samples)
    gpu = [sample['process_GPU_MiB'] for sample in samples if sample['process_GPU_MiB'] is not None]
    status = sampling['process_GPU_sampling_status']
    require(status in ('AVAILABLE', 'UNAVAILABLE'), 'Unknown GPU sampling status')
    if status == 'AVAILABLE':
        require(bool(gpu) and sampling['process_GPU_sampling_reason'] is None, 'AVAILABLE GPU samples missing')
    else:
        require(bool(sampling['process_GPU_sampling_reason']), 'Unavailable GPU metric needs a reason')
    expected_gpu = max(gpu) if gpu else None
    require(sampling['process_GPU_sampled_peak_MiB'] == expected_gpu and
            summary['process_GPU_sampled_peak_MiB'] == expected_gpu, 'GPU sampled peak inconsistent/unavailable encoded as zero')
    for key in ('process_GPU_sampling_status', 'process_GPU_sampling_reason'):
        require(sampling[key] == summary[key], 'GPU sampling metadata differs')
    close(sampling['CPU_RSS_sampled_peak_MiB'], cpu, 'Sampled CPU max')
    close(summary['CPU_RSS_sampled_peak_MiB'], cpu, 'Summary sampled CPU max')


def table_values(checks):
    require([row['run_id'] for row in checks] == list(FACETS), 'Exactly the three registered facets in order required')
    raw, first, second = [row['recomputed_statistics'] for row in checks]
    latencies = [row['median_batch_ms'] for row in (first, second)]
    peaks = [row['peak_allocated_MiB'] for row in (first, second)]
    latency, peak = math.fsum(latencies)/2, math.fsum(peaks)/2
    return dict(raw_median_batch_ms=raw['median_batch_ms'], raw_peak_allocated_MiB=raw['peak_allocated_MiB'],
                ours_seeds=[92601, 92602], ours_seed_median_batch_ms=latencies, ours_seed_peak_allocated_MiB=peaks,
                ours_mean_seed_median_batch_ms=latency, ours_mean_seed_peak_allocated_MiB=peak,
                speedup_raw_over_ours=raw['median_batch_ms']/latency,
                peak_allocated_change_pct=100*(peak/raw['peak_allocated_MiB']-1),
                aggregation='Arithmetic mean of two OURS seed medians and two seed maximum allocated peaks; deterministic Raw once')


def tensor_hashes(state):
    sorted_hash, per_tensor = hashlib.sha256(), {}
    for key in sorted(state):
        array = np.ascontiguousarray(state[key])
        require(array.dtype == np.float32 and np.isfinite(array).all(), 'Adapter tensor is not finite FP32')
        sorted_hash.update(key.encode())
        sorted_hash.update(array.tobytes())
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode())
        digest.update(json.dumps(list(array.shape)).encode())
        digest.update(array.tobytes())
        per_tensor[key] = digest.hexdigest()
    native_bundle = hashlib.sha256(json.dumps(per_tensor, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return sorted_hash.hexdigest(), native_bundle


def adapter_hash(row, reference):
    if row['family'] == 'RAW':
        return None
    import torch
    bundle = torch.load(row['bundle_path'], map_location='cpu', weights_only=False)
    state = {key: value.detach().numpy() for key, value in bundle['model'].items()}
    tensor_sha, bundle_sha = tensor_hashes(state)
    require(bundle_sha == bundle['model_state_sha'] == row['model_state_sha'], 'Original bundle model-state hash differs')
    require(bundle['spec'] == row['bundle_spec'] and bundle['contract'] == row['bundle_contract'] and
            bundle['normalization']['fit_end'] == 11841 and bundle['contract']['model_sha'] == F_SHA and
            bundle['selected_search_epoch'] == row['selected_epoch'], 'Original bundle contract differs')
    require(all(np.array_equal(bundle['normalization'][key], reference[key]) for key in ('mean', 'std')), 'Original bundle scaler differs')
    return tensor_sha


def validate_host_evidence(receipt, report):
    contract_path=report/'HOST_GPU_CONTRACT.json'
    host_gpu_gate.verify_identity_receipt(receipt['host_admission'],contract_path)
    result={}
    for field,suffix in (('host_preflight','HOST_PREFLIGHT'),('host_observation','HOST_OBSERVATION')):
        public_path=report/(receipt['summary']['accuracy_run_id']+'_'+suffix+'.json')
        require(read(public_path)==receipt[field], 'Public host evidence differs from measurement receipt')
        result[field]=host_gpu_gate.verify_observation_receipt(receipt[field],contract_path)
    require(receipt['host_preflight']['mode']=='performance_preflight' and
            receipt['host_observation']['mode']=='performance_observation' and
            receipt['host_preflight']['required_end_unix']<=receipt['host_observation']['required_start_unix'],
            'Host preflight/measurement phase ordering differs')
    return result


def validate(report=REPORT):
    report = Path(report).resolve()
    require(report == REPORT.resolve(), 'Validator is restricted to the registered campaign cost directory')
    required = ['COST_SEAL.json', 'COST_PROTOCOL.json', 'COST_INPUTS.json', 'ROOT_LAUNCH_GATE.json',
                'COST_LAUNCH.json', 'COST_SUMMARY.json', 'HOST_GPU_CONTRACT.json', 'AMENDMENT_002.json', 'HOST_GATE_CPU_TESTS.json'] + [
                    name+'_'+suffix+'.json' for name in FACETS for suffix in ('MEASUREMENT','HOST_PREFLIGHT','HOST_OBSERVATION')]
    require(all((report/name).is_file() for name in required), 'Actual complete measurements required; no validation file written')
    seal_sha=sha(report/'COST_SEAL.json')
    seal, protocol, rows = read(report/'COST_SEAL.json'), read(report/'COST_PROTOCOL.json'), read(report/'COST_INPUTS.json')['rows']
    require(seal['revision']==protocol['revision']==3 and seal['previous_seal_sha256']==PREVIOUS_SEAL_SHA,
            'Cost host amendment must replace and retain the original v2 seal')
    for name, expected in seal['files'].items():
        require(sha(report/name) == expected, 'Sealed file changed: ' + name)
    protocol_sha = sha(report/'COST_PROTOCOL.json')
    require(protocol['host_cpu_test_receipt_sha256']==sha(report/'HOST_GATE_CPU_TESTS.json'), 'Host CPU test binding changed')
    contract,_=host_gpu_gate.load_policy(report/'HOST_GPU_CONTRACT.json')
    require(protocol['host_gpu_gate']['contract_sha256']==sha(report/'HOST_GPU_CONTRACT.json') and
            protocol['host_gpu_gate']['policy_sha256']==contract['policy_sha256'], 'Host policy/contract registration binding differs')
    require(protocol['facet_order'] == list(FACETS) and [row['run_id'] for row in rows] == list(FACETS), 'Registered facet roster changed')
    require({p.name for p in report.glob('*_MEASUREMENT.json')} == {name+'_MEASUREMENT.json' for name in FACETS}, 'Unexpected or missing facet result')
    evidence = {name: sha(report/name) for name in required}
    for relative, expected in protocol['source_hashes'].items():
        require(sha(REPO/relative) == expected, 'Producer source changed: ' + relative)
    bindings = [protocol['reference_source'], dict(path=report/protocol['amendment']['path'], sha256=protocol['amendment']['sha256'])]
    safe = protocol['safe_data']
    bindings.extend(dict(path=safe[key+'_path'], sha256=safe[key+'_sha256']) for key in ('prefix', 'config'))
    for row in rows:
        bindings.append(dict(path=row['prediction_path'], sha256=row['prediction_sha256']))
        if row['bundle_path']:
            bindings.append(dict(path=row['bundle_path'], sha256=row['bundle_sha256']))
    for binding in bindings:
        require(sha(binding['path']) == binding['sha256'], 'Input bytes differ: ' + str(binding['path']))
    launch, gate, summary = (read(report/name) for name in ('COST_LAUNCH.json', 'ROOT_LAUNCH_GATE.json', 'COST_SUMMARY.json'))
    require(gate['status'] == 'PASS' and gate['allow_gpu'] is True and gate['registration_seal_sha256'] == seal_sha, 'Invalid root launch authorization')
    require(launch['facets'] == list(FACETS), 'Launch facets differ')
    require(summary['status'] == 'COMPLETE' and summary['training_executed'] is False and summary['T_numeric_values_parsed'] == 0,
            'Incomplete cost summary')
    require(len(summary['rows']) == 3 and [row['accuracy_run_id'] for row in summary['rows']] == list(FACETS), 'Summary facet roster incomplete')
    launched = datetime.fromisoformat(launch['started_utc']).timestamp()
    completed = datetime.fromisoformat(summary['completed_utc']).timestamp()
    require(completed >= launched, 'Completion predates launch')
    references, results, checks, host_checks = [], [], [], []
    for row, output_row in zip(rows, summary['rows']):
        receipt = read(report/(row['run_id']+'_MEASUREMENT.json'))
        host_checks.append(validate_host_evidence(receipt,report))
        lease = receipt['parent_lease_identity']
        require(lease['status'] == 'ACTIVE' and lease['pid'] == launch['pid'] and lease['active_facet'] == row['run_id'] and
                lease['registration_seal_sha256'] == seal_sha and lease['root_gate_sha256'] == evidence['ROOT_LAUNCH_GATE.json'] and
                '--run' in lease['command'] and 'research.'+NAME+'.cost_profile' in lease['command'], 'Facet parent lease differs')
        prediction_path = REPO/'.cache'/NAME/'cost_profile'/(row['run_id']+'_predictions.npz')
        require(Path(receipt['prediction_path']).resolve() == prediction_path.resolve(), 'Regenerated prediction outside registered output')
        require(sha(prediction_path) == receipt['prediction_sha256'], 'Regenerated array file hash differs')
        reference, regenerated = arrays(row['prediction_path']), arrays(prediction_path)
        checks.append(validate_measurement(receipt, row, protocol, protocol_sha, reference, regenerated,
                                           adapter_tensor_sha=adapter_hash(row, reference)))
        require(launched <= receipt['summary']['process_create_time'] <= completed, 'Worker belongs to a stale launch')
        references.append(reference)
        results.append(receipt)
        expected_row = dict(receipt['summary'], latency_ratio_to_raw=receipt['summary']['median_batch_ms']/summary['rows'][0]['median_batch_ms'],
                            peak_allocated_ratio_to_raw=receipt['summary']['peak_allocated_MiB']/summary['rows'][0]['peak_allocated_MiB'])
        require(output_row == expected_row, 'Published COST_SUMMARY differs from measured facet')
        bindings.append(dict(path=str(prediction_path), sha256=receipt['prediction_sha256']))
    require(len({(item['worker_pid'], item['worker_create_time']) for item in checks}) == 3, 'Facets did not use fresh process identities')
    require(len({item['invariant_sorted_tensor_hashes']['F'] for item in checks}) == 1, 'Loaded F tensor states differ across facets')
    for reference in references[1:]:
        require(all(fingerprint(reference[key]) == fingerprint(references[0][key]) for key in COHORT_KEYS), 'Cross-facet V cohort/scaler differs')
    validate_prefix(safe, references[0], results)
    # Recheck every public and source file after reading arrays; no stale PASS receipt.
    require(all(sha(report/name) == digest for name, digest in evidence.items()), 'Public evidence changed during validation')
    require(all(sha(REPO/name) == digest for name, digest in protocol['source_hashes'].items()), 'Producer changed during validation')
    require(all(sha(binding['path']) == binding['sha256'] for binding in bindings), 'Input arrays changed during validation')
    return dict(status='PASS', schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                registration_seal_sha256=seal_sha, previous_v2_seal_sha256=PREVIOUS_SEAL_SHA,
                cost_summary_sha256=evidence['COST_SUMMARY.json'], host_checks=host_checks,
                evidence_bindings=evidence, input_and_array_bindings=[dict(path=str(x['path']), sha256=x['sha256']) for x in bindings],
                producer_source_bindings=protocol['source_hashes'], verifier_sha256=sha(__file__),
                test_source_sha256=sha(Path(__file__).parent/'tests/test_cost_validation.py'),
                facets=checks, table_values=table_values(checks), training_executed=False, T_numeric_values_parsed=0,
                scope=dict(independent_cpu_checks='Artifact hashes, original/regenerated all-V arrays and observed macro FP64 scores, cohort/A60 scaler/pre-T bounds, trial work/statistics and table ratios',
                           pinned_execution_evidence='CUDA FP32 eval/one frozen F, per-call synchronization/reset, timing-trial output shape/finite checks and state hashes follow sealed producer receipts',
                           GPU_prediction_recomputed_by_verifier=False,
                           host_condition='Bounded registered Windows desktop concurrency; private identity and all-engine raw traces independently rechecked; not an idle/exclusive GPU claim',
                           unavailable_evidence='Individual timing-trial physical arrays and full-V physical outputs are not retained; their hashes are checked for valid receipt form, not independently replayed',
                           inference_only=True, p95_descriptive_only=True, process_GPU_sampled_peak_not_instantaneous_guarantee=True))


def validate_prefix(safe, reference, receipts):
    config = read(safe['config_path'])
    require(config['prefix_only_approved'] is True and config['prefix_rows'] == 15788 and
            config['split_ranges']['V_select'] == [13814, 15788] and config['split_ranges']['T_final'][0] == 15788,
            'Physical prefix approval/boundary differs')
    with np.load(safe['prefix_path'], allow_pickle=False) as archive:
        # Inspect NPY headers before numeric materialization, excluding any T rows.
        for key, shape, dtype in (('physical', (15788, 26), np.dtype('float32')), ('observed', (15788, 26), np.dtype('bool'))):
            with archive.zip.open(key+'.npy') as stream:
                version = np.lib.format.read_magic(stream)
                require(version in ((1, 0), (2, 0)), 'Unsupported physical numeric header')
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                header = reader(stream)
            require(header == (shape, False, dtype), 'Physical prefix numeric header exceeds registered range')
        physical, observed, columns = archive['physical'], archive['observed'], archive['columns']
    require(np.array_equal(columns, reference['channel_ids']) and np.isfinite(physical[observed]).all(), 'Physical cohort/observations invalid')
    fit, mask = physical[:11841].astype(np.float64), observed[:11841]
    counts = mask.sum(0)
    require((counts > 0).all(), 'A60 scaler has empty channel')
    mean = np.where(mask, fit, 0).sum(0)/counts
    std = np.sqrt((np.where(mask, fit-mean, 0)**2).sum(0)/counts)
    std[std == 0] = 1
    require(np.array_equal(mean.astype(np.float32), reference['mean']) and np.array_equal(std.astype(np.float32), reference['std']),
            'Original source scaler differs from A60 fit')
    indices = reference['origins'][:, None] + np.arange(48)[None, :]
    target = ((physical[indices]-reference['mean'])/reference['std']).astype(np.float32)
    require(np.array_equal(observed[indices], reference['observed']) and
            np.allclose(target[reference['observed']], reference['target_normalized'][reference['observed']], atol=1e-5, rtol=1e-4),
            'Bound source V targets differ from physical prefix')
    histories = np.stack([physical[int(origin)-512:int(origin)] for origin in reference['origins']]).astype(np.float32)
    masks = np.stack([observed[int(origin)-512:int(origin)] for origin in reference['origins']]).astype(bool)
    for receipt in receipts:
        require(receipt['parity']['input_fingerprints']['physical_windows'] == fingerprint(histories) and
                receipt['parity']['input_fingerprints']['physical_observed_windows'] == fingerprint(masks), 'Physical timing/parity input fingerprint differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Validate without writing COST_VALIDATION.json')
    args = parser.parse_args()
    value = validate()
    destination = REPORT/'COST_VALIDATION.json'
    if not args.check_only:
        if destination.exists():
            previous = read(destination)
            require(previous['status'] == 'PASS' and previous['evidence_bindings'] == value['evidence_bindings'] and
                    previous['verifier_sha256'] == value['verifier_sha256'] and previous['table_values'] == value['table_values'],
                    'Existing validation differs; preserve it and investigate')
        else:
            with destination.open('x', encoding='utf8', newline='\n') as stream:
                json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write('\n')
    print(json.dumps(dict(status='PASS', facets=3, written=not args.check_only, path=str(destination)), ensure_ascii=False))


if __name__ == '__main__':
    main()
