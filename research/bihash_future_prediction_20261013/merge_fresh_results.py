"""CPU-only strict union of preserved, disjoint fresh training attempts."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import math
import re
import time
import numpy as np
import torch
import runtime as rt
import run as ru

HOUSE, SML = 'uci235_household_hourly', 'uci274_sml2010'
DATASETS = (HOUSE, SML)
METHODS = tuple(ru.bm.METHODS)
SEEDS = tuple(ru.tr.SEEDS)
CHECKPOINTS = tuple(ru.tr.CHECKPOINTS)


def sha(path):
    return rt.sha(path)


def read(path):
    return rt.read(path)


def ref(path, root):
    path = Path(path).resolve()
    return {'path': path.relative_to(Path(root).resolve()).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size}


def tensor_hash(state):
    return rt.state_hash(state)


def resolve(root, value):
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (Path(root) / path).resolve()
    if not path.is_relative_to(Path(root).resolve()): raise ValueError('ARTIFACT_OUTSIDE_ROOT')
    return path


def verify(root, record):
    path = resolve(root, record['path'])
    if not path.is_file() or sha(path) != record['sha256']:
        raise ValueError('ARTIFACT_MISSING_OR_SHA_CHANGED: ' + str(path))
    if 'bytes' in record and path.stat().st_size != record['bytes']: raise ValueError('ARTIFACT_BYTE_COUNT_CHANGED')
    return path


def valid_tag(tag):
    if not re.fullmatch(r'[a-z][a-z0-9_]{0,63}', tag): raise ValueError('INVALID_RECOVERY_TAG')


def check_numeric(value):
    if isinstance(value, dict):
        for child in value.values(): check_numeric(child)
    elif isinstance(value, list):
        for child in value: check_numeric(child)
    elif isinstance(value, float) and not math.isfinite(value): raise ValueError('NONFINITE_SAVED_TRAINING_RECORD')


def optimizer_step(stored):
    states = stored.get('optimizer', {}).get('state', {})
    return max((float(value.get('step', 0)) for value in states.values()), default=0.)


def prediction_score(prediction, target, observed):
    if prediction.shape != target.shape or target.shape != observed.shape or not np.isfinite(prediction).all():
        raise ValueError('PREDICTION_SHAPE_OR_FINITE')
    error = np.where(observed, prediction.astype(np.float64) - target, 0.)
    count = observed.sum((0, 1)); valid = count > 0
    if not valid.all(): raise ValueError('TARGET_CHANNEL_WITHOUT_OBSERVATION')
    return float(((error*error).sum((0, 1)) / count)[valid].mean())


def check_reinference(current, original, target, observed):
    difference = np.abs(current.astype(np.float64)-original.astype(np.float64))
    if not np.all(difference <= 1e-5 + 2e-6*np.abs(original.astype(np.float64))):
        raise ValueError('SELECTED_REINFERENCE_ARRAY_GAP_EXCEEDS_FP32_BOUND')
    original_error = np.where(observed, original.astype(np.float64)-target, 0.)
    delta = np.where(observed, current.astype(np.float64)-original.astype(np.float64), 0.)
    count = observed.sum((0, 1)); valid = count > 0
    bound = float(((2*np.abs(original_error)*np.abs(delta)+delta*delta).sum((0, 1))/count)[valid].mean())
    gap = abs(prediction_score(current, target, observed)-prediction_score(original, target, observed))
    if gap > bound + 1e-12: raise ValueError('SELECTED_REINFERENCE_MSE_GAP_EXCEEDS_ACTUAL_DELTA_BOUND')
    return {'max_abs_array_gap': float(difference.max()), 'array_atol': 1e-5, 'array_rtol': 2e-6,
            'observed_macro_mse_abs_gap': gap, 'observed_macro_mse_delta_bound': bound,
            'bound': 'macro observed-channel mean of 2|CPpred-target||delta|+delta^2',
            'note': 'Different saved FP32 reinferences may differ; each NPZ is rescored against its own JSON strictly. Selection uses original CP TUNE score.'}


def reference(contract, root, lock_sha):
    dataset = contract['dataset']
    if dataset not in DATASETS or contract['method_lock']['sha256'] != lock_sha:
        raise ValueError('CONTRACT_METHOD_LOCK_MISMATCH')
    payload = verify(root, contract['payloads']['fit_tune'])
    contract_path = payload.parent / 'contract.json'
    if read(contract_path) != contract: raise ValueError('CONTRACT_JSON_PROVENANCE_MISMATCH')
    registry_path = Path(root) / '.cache/bihash_future_prediction_20261013/contract' / (dataset + '_' + lock_sha + '.json')
    registry = read(registry_path)
    if registry['dataset'] != dataset or registry['method_lock_sha256'] != lock_sha or registry['contract_sha256'] != sha(contract_path):
        raise ValueError('CONTRACT_REGISTRY_OR_SHA_MISMATCH')
    if resolve(root, registry['contract_path']) != contract_path.resolve(): raise ValueError('CONTRACT_REGISTRY_PATH_MISMATCH')
    for source in contract['source_records']: verify(root, source)
    unopened = {}
    for key in ('eval_past_input', 'eval_future_target'):
        record = contract['payloads'][key]; path = resolve(root, record['path'])
        if not path.is_file(): raise ValueError('DECLARED_EVAL_PAYLOAD_MISSING')
        if 'bytes' in record and path.stat().st_size != record['bytes']: raise ValueError('DECLARED_EVAL_PAYLOAD_BYTES_MISMATCH')
        unopened[key] = {**record, 'exists': True, 'content_opened_or_hashed_by_merge': False}
    with np.load(payload, allow_pickle=False) as z:
        physical, mask = z['physical'].copy(), z['mask'].astype(bool)
        mean, std = z['mean'].copy(), z['std'].copy()
        origins = {split: z['origins_' + split].copy() for split in ('fit', 'tune')}
        basis, pca_mean = z['basis'].copy(), z['pca_mean'].copy()
    if (contract['L'], contract['H'], contract['C'], contract['K']) != (512, 48, len(mean), basis.shape[1]):
        raise ValueError('CONTRACT_DIMENSION_MISMATCH')
    if contract['K'] != max(1, contract['C']//4) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError('FIT_STAT_OR_K_CONTRACT')
    targets, observed = {}, {}
    for split in ('fit', 'tune'):
        targets[split] = np.stack([np.where(mask[o:o+48], (physical[o:o+48]-mean)/std, 0.).astype(np.float32) for o in origins[split]])
        observed[split] = np.stack([mask[o:o+48] for o in origins[split]])
        if len(origins[split]) != (256 if split == 'fit' else 64): raise ValueError('FIT_TUNE_ORIGIN_COUNT')
        if not np.isfinite(targets[split][observed[split]]).all(): raise ValueError('NONFINITE_FIT_TUNE_TARGET')
    return {'mean': mean, 'std': std, 'origins': origins, 'target': targets, 'observed': observed,
            'basis': basis, 'pca_mean': pca_mean, 'contract_artifact': ref(contract_path, root),
            'registry_artifact': ref(registry_path, root), 'fit_tune_payload': contract['payloads']['fit_tune'], 'unopened_EVAL_payloads': unopened}


def audit_prediction(record, reference_data, splits, root):
    path = verify(root, record)
    scores, predictions = {}, {}
    with np.load(path, allow_pickle=False) as z:
        if not np.array_equal(z['mean'], reference_data['mean']) or not np.array_equal(z['std'], reference_data['std']):
            raise ValueError('PREDICTION_FIT_STAT_MISMATCH')
        for split in splits:
            for field in ('origins', 'target', 'observed'):
                expected = reference_data['origins' if field == 'origins' else field][split]
                if not np.array_equal(z[split + '_' + field], expected): raise ValueError('PREDICTION_SOURCE_CONTRACT_MISMATCH: ' + field)
            prediction = z[split + '_prediction'].copy()
            scores[split] = prediction_score(prediction, reference_data['target'][split], reference_data['observed'][split])
            predictions[split] = prediction
    return scores, predictions


def audit_case(case, result_path, producer, reference_data, lock, root, ae):
    dataset, seed, method, weight = case['dataset'], case['seed'], case['method'], case['weight']
    expected_label = ru.folder_name(method, weight, seed)
    if case['label'] != expected_label or weight != lock['weights'][method]: raise ValueError('CASE_IDENTITY_OR_WEIGHT')
    original_json = result_path.parent / dataset / (case['label'] + '.json')
    if not original_json.is_file() or read(original_json) != case: raise ValueError('CASE_JSON_PROVENANCE_MISMATCH')
    training = case['training']; check_numeric(training)
    if training['updates'] != 600 or training['batch_size'] != 8 or [c['update'] for c in training['curve']] != list(range(1, 601)):
        raise ValueError('CORE_NOT_ACTUAL_600_UPDATE_CURVE')
    if [c['update'] for c in training['checkpoints']] != list(CHECKPOINTS): raise ValueError('MISSING_5_CORE_CHECKPOINTS')
    if (case['L'], case['H'], case['K']) != (512, 48, reference_data['basis'].shape[1]): raise ValueError('CORE_L_H_K_MISMATCH')
    order = np.asarray(training['batch_indices'], np.int64)
    if order.shape != (600, 8) or hashlib.sha256(order.tobytes()).hexdigest() != training['order_sha256']:
        raise ValueError('COMMON_BATCH_ORDER_SHA_OR_SHAPE')
    if np.any(order < 0) or np.any(order >= len(reference_data['origins']['fit'])): raise ValueError('FIT_BATCH_INDEX_RANGE')
    batch_origins = reference_data['origins']['fit'][order]
    differences = np.abs(batch_origins[:, :, None]-batch_origins[:, None, :])
    if np.any(differences[:, ~np.eye(8, dtype=bool)] < 48): raise ValueError('BATCH_HORIZON_OVERLAP_OR_DUPLICATE')
    if case['future_pretrain']['path'] != ae['checkpoint']['path'] or case['future_pretrain']['sha256'] != ae['checkpoint']['sha256']:
        raise ValueError('COMMON_FUTURE_AE_REFERENCE_MISMATCH')
    expected_base = Path(root) / '.cache/bihash_future_prediction_20261013' / ('run_' + producer['tag']) / dataset / case['label']
    cp_scores, cp_predictions, selected_state = [], {}, None
    ae_state = torch.load(verify(root, ae['checkpoint']), map_location='cpu', weights_only=True)['future_autoencoder']
    for cp in training['checkpoints']:
        cp_path = verify(root, cp)
        if cp_path.parent.resolve() != expected_base.resolve(): raise ValueError('CHECKPOINT_PRODUCER_PATH_MISMATCH')
        stored = torch.load(cp_path, map_location='cpu', weights_only=True)
        if (stored['update'], stored['seed'], stored['method'], stored['weight']) != (cp['update'], seed, method, weight):
            raise ValueError('ACTUAL_CHECKPOINT_IDENTITY_MISMATCH')
        if optimizer_step(stored) != cp['update']: raise ValueError('ACTUAL_CHECKPOINT_OPTIMIZER_STEP_MISMATCH')
        if any(v.is_floating_point() and not bool(torch.isfinite(v).all()) for v in stored['model'].values()):
            raise ValueError('NONFINITE_CHECKPOINT_STATE')
        if cp['update'] == 0:
            for name, value in ae_state.items():
                if not torch.equal(stored['model']['future_autoencoder.' + name], value): raise ValueError('CP0_COMMON_AE_STATE_MISMATCH')
        scores, predictions = audit_prediction(cp['tune_prediction'], reference_data, ('tune',), root)
        reported = cp['tune']['normalized']['mse']
        if not math.isclose(scores['tune'], reported, rel_tol=1e-8, abs_tol=1e-9): raise ValueError('CHECKPOINT_TUNE_MSE_RESCORE_MISMATCH')
        cp_scores.append((reported, cp['update'], cp))
        if cp['update'] == training['selected_update']:
            selected_state, cp_predictions = stored['model'], predictions
        del stored
    selected = min(cp_scores, key=lambda value: (value[0], value[1]))[2]
    if selected['update'] != training['selected_update'] or selected['path'] != training['selected_checkpoint']['path'] or selected['sha256'] != training['selected_checkpoint']['sha256']:
        raise ValueError('SELECTED_CHECKPOINT_NOT_TUNE_ARGMIN')
    if tensor_hash(selected_state) != training['selected_hash']: raise ValueError('SELECTED_STATE_HASH_MISMATCH')
    scores, predictions = audit_prediction(case['predictions'], reference_data, ('fit', 'tune'), root)
    reinference = check_reinference(predictions['tune'], cp_predictions['tune'], reference_data['target']['tune'], reference_data['observed']['tune'])
    for split in ('fit', 'tune'):
        if not math.isclose(scores[split], case['scores'][split]['normalized']['mse'], rel_tol=1e-8, abs_tol=1e-9):
            raise ValueError('SELECTED_NPZ_OWN_JSON_MSE_MISMATCH')
    if not math.isclose(selected['tune']['normalized']['mse'], case['selected_tune_mse'], rel_tol=1e-8, abs_tol=1e-9):
        raise ValueError('SELECTED_TUNE_CP_SELECTION_SCORE_MISMATCH')
    copied = copy.deepcopy(case)
    copied['producer_lineage'] = {'producer_result': ref(result_path, root), 'original_case_json': ref(original_json, root),
                                  'original_status': producer['status'], 'producer_tag': producer['tag'],
                                  'filter': 'All cases of this producer dataset, no rank/performance filtering',
                                  'validated_actual_checkpoints': 5, 'selected_TUNE_mse_rescored': scores['tune'],
                                  'selected_CP_TUNE_selection_mse': selected['tune']['normalized']['mse'],
                                  'selected_vs_CP_FP32_reinference': reinference}
    return copied


def audit_snapshot(result, root, lock):
    if result['source_hashes'] != lock['source_hashes'] or len(result['source_hashes']) != 16:
        raise ValueError('SCIENTIFIC_SOURCE16_NOT_IDENTICAL')
    tag = result['tag']; valid_tag(tag)
    folder = Path(root) / '.cache/bihash_future_prediction_20261013/source_snapshots' / tag
    manifest_path = folder / 'MANIFEST.json'; manifest = read(manifest_path)
    if manifest['tag'] != tag or manifest['sha256'] != lock['source_hashes']: raise ValueError('SOURCE_SNAPSHOT_MANIFEST_MISMATCH')
    for relative, digest in lock['source_hashes'].items():
        if sha(resolve(root, relative)) != digest or sha(folder / relative) != digest: raise ValueError('SCIENTIFIC_SOURCE_CURRENT_OR_SNAPSHOT_SHA_CHANGED')
    if result['stage'] == 'fresh' and result.get('execution_subset'):
        script = result['producer']['script']; snapshot_script = result['producer']['snapshot_script']
        verify(root, script); verify(root, snapshot_script)
        if script['sha256'] != snapshot_script['sha256']: raise ValueError('SUBSET_PRODUCER_SNAPSHOT_SHA_CHANGED')
        if 'helper_script' in result['producer']:
            helper, saved_helper = result['producer']['helper_script'], result['producer']['snapshot_helper_script']
            verify(root, helper); verify(root, saved_helper)
            if helper['sha256'] != saved_helper['sha256']: raise ValueError('SUBSET_HELPER_SNAPSHOT_SHA_CHANGED')
    elif result['stage'] == 'fresh':
        script = ref(folder / 'research/bihash_future_prediction_20261013/run.py', root)
    else: raise ValueError('UNKNOWN_PRODUCER_STAGE')
    return {'source_snapshot_manifest': ref(manifest_path, root), 'producer_script': script}


def build_merged(producer_paths, method_lock, root=rt.ROOT):
    root = Path(root).resolve(); lock_path = Path(method_lock).resolve(); lock = read(lock_path); lock_sha = sha(lock_path)
    if not lock.get('locked'): raise ValueError('METHOD_LOCK_NOT_LOCKED')
    if len(producer_paths) != 2 or len({Path(p).resolve() for p in producer_paths}) != 2: raise ValueError('EXACTLY_TWO_DISJOINT_PRODUCERS_REQUIRED')
    merged = {'status': 'COMPLETE', 'stage': 'fresh', 'source_hashes': lock['source_hashes'], 'method_lock': ref(lock_path, root),
              'cases': [], 'future_pretrains': [], 'contracts': {}, 'data': {}, 'failed_paths': [], 'producers': [],
              'recovered_failed_paths': [], 'protected_TEST_opened': False, 'EVAL_future_target_opened': False,
              'new_training_updates_in_merge': 0, 'merge_provenance': {'original_failures_preserved': True, 'producer_result_count': 2},
              'coverage_definition': 'Union of actual complete, disjoint fresh dataset cases; original failed status remains unchanged'}
    claimed, resources, f_hashes = set(), set(), set()
    common = {}
    for supplied in producer_paths:
        path = Path(supplied).resolve(); result = read(path); status = result['status']
        if status not in ('COMPLETE', 'FAILED_PRESERVED') or any(key in result for key in ('error', 'error_type', 'traceback')):
            raise ValueError('PRODUCER_NOT_VALID_TERMINAL_OR_UNKNOWN_TOP_FAILURE')
        if result['method_lock']['sha256'] != lock_sha or resolve(root, result['method_lock']['path']) != lock_path:
            raise ValueError('PRODUCER_METHOD_LOCK_SHA_MISMATCH')
        snapshot = audit_snapshot(result, root, lock)
        failures = result.get('failed_paths', [])
        if status == 'FAILED_PRESERVED':
            if len(failures) != 1 or failures[0].get('stage') != 'data_prepare' or failures[0].get('dataset') != HOUSE or failures[0].get('seed') is not None or failures[0].get('label') != HOUSE:
                raise ValueError('UNEXPLAINED_PRODUCER_FAILURE_NOT_HOUSEHOLD_DATA_PREPARE_ONLY')
            failure = copy.deepcopy(failures[0]); failure['producer_result'] = ref(path, root)
            merged['recovered_failed_paths'].append(failure)
        elif failures: raise ValueError('COMPLETE_PRODUCER_HAS_UNRESOLVED_FAILURE')
        datasets = set(result['contracts'])
        expected_ds = {SML} if status == 'FAILED_PRESERVED' else {HOUSE}
        if datasets != expected_ds or set(result['data']) != datasets or claimed & datasets:
            raise ValueError('DISJOINT_EXACT_DATASET_COVERAGE_REQUIRED')
        if result.get('execution_subset') and (result.get('execution_subset') != [HOUSE] or result.get('selected_datasets') != [HOUSE]):
            raise ValueError('SUBSET_DATASET_DECLARATION_MISMATCH')
        claimed |= datasets
        expected = {(ds, seed, method, lock['weights'][method]) for ds in datasets for seed in SEEDS for method in METHODS}
        actual = [(c['dataset'], c['seed'], c['method'], c['weight']) for c in result['cases']]
        if len(actual) != 36 or len(set(actual)) != 36 or set(actual) != expected: raise ValueError('PRODUCER_CORE_COVERAGE_NOT_EXACT_36')
        ae_keys = [(a['dataset'], a['seed']) for a in result['future_pretrains']]
        if len(ae_keys) != 3 or len(set(ae_keys)) != 3 or set(ae_keys) != {(ds, seed) for ds in datasets for seed in SEEDS}:
            raise ValueError('PRODUCER_FUTURE_AE_COVERAGE_NOT_EXACT_3')
        references = {}
        for dataset in datasets:
            contract = result['contracts'][dataset]
            if contract['dataset'] != dataset or result['data'][dataset]['contract'] != contract: raise ValueError('PRODUCER_CONTRACT_DATA_MISMATCH')
            references[dataset] = reference(contract, root, lock_sha)
            merged['contracts'][dataset] = copy.deepcopy(contract); merged['data'][dataset] = copy.deepcopy(result['data'][dataset])
        ae_map = {}
        for ae in result['future_pretrains']:
            if ae['updates'] != 400 or ae['batch_size'] != 64 or [v['update'] for v in ae['curve']] != list(range(1, 401)):
                raise ValueError('FUTURE_AE_NOT_ACTUAL_400_UPDATE_CURVE')
            check_numeric(ae)
            stored = torch.load(verify(root, ae['checkpoint']), map_location='cpu', weights_only=True)
            if stored['update'] != 400 or stored['seed'] != ae['seed'] or optimizer_step(stored) != 400:
                raise ValueError('ACTUAL_FUTURE_AE_CHECKPOINT_OR_OPTIMIZER_STEP_MISMATCH')
            copied = copy.deepcopy(ae); copied['producer_lineage'] = {'producer_result': ref(path, root), 'original_status': status, 'producer_tag': result['tag']}
            merged['future_pretrains'].append(copied); ae_map[(ae['dataset'], ae['seed'])] = ae
            del stored
        for case in result['cases']:
            merged['cases'].append(audit_case(case, path, result, references[case['dataset']], lock, root, ae_map[(case['dataset'], case['seed'])]))
            group = (case['dataset'], case['seed'])
            signature = (case['training']['order_sha256'], case['initial_hash'], case['future_pretrain']['sha256'])
            if group in common and common[group] != signature: raise ValueError('METHOD_INITIALIZATION_OR_BATCH_NOT_MATCHED')
            common[group] = signature
        receipt_path = verify(root, result['resource_receipt']); receipt = read(receipt_path)
        if receipt['status'] != status or not receipt['gpu'] or receipt['tag'] != result['tag']:
            raise ValueError('ORIGINAL_GPU_RESOURCE_RECEIPT_NOT_MATCHING_TERMINAL')
        if receipt_path in resources: raise ValueError('DUPLICATE_GPU_RECEIPT_WOULD_DOUBLE_COUNT')
        resources.add(receipt_path)
        before, after = result['backbone_before_sha256'], result['backbone_after_sha256']
        if before != after: raise ValueError('BACKBONE_CHANGED_IN_PRODUCER')
        f_hashes.add(before)
        merged['producers'].append({'result': ref(path, root), 'original_status': status, 'tag': result['tag'], **snapshot,
                                   'resource_receipt': result['resource_receipt'], 'valid_case_count': len(result['cases']), 'valid_future_AE_count': len(result['future_pretrains']),
                                   'datasets': sorted(datasets), 'case_filter': 'All source cases after full scope validation, no metric-based filtering',
                                   'original_failed_paths': copy.deepcopy(failures), 'contracts_checked': {ds: {
                                       key: references[ds][key] for key in ('contract_artifact', 'registry_artifact', 'fit_tune_payload', 'unopened_EVAL_payloads')} for ds in datasets}})
    if claimed != set(DATASETS) or len(f_hashes) != 1 or len(merged['recovered_failed_paths']) != 1:
        raise ValueError('MERGED_SCOPE_BACKBONE_OR_RECOVERY_COVERAGE_INVALID')
    merged.update(backbone_before_sha256=next(iter(f_hashes)), backbone_after_sha256=next(iter(f_hashes)),
                  validated_checkpoint_count=360, validated_actual_TUNE_npz_count=360, validated_selected_FIT_TUNE_npz_count=72,
                  original_gpu_resource_receipts=[p['resource_receipt'] for p in merged['producers']],
                  original_gpu_wall_seconds_counted_once=sum(read(p)['wall_seconds'] for p in resources),
                  resource_accounting_note='These original two GPU receipts remain separately referenced; merge adds only its own CPU receipt. Never add their wall again as new GPU work.',
                  original_failed_attempt_changed=False, remaining_unresolved_failures=0)
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--producer-result', type=Path, action='append', required=True)
    parser.add_argument('--dev-result', type=Path, required=True)
    parser.add_argument('--method-lock', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args(); valid_tag(args.tag)
    folder = rt.HERE / ('run_' + args.tag)
    if folder.exists(): raise FileExistsError('MERGE_TAG_ALREADY_EXISTS')
    for path in args.producer_result + [args.dev_result, args.method_lock]:
        if not path.is_file(): raise FileNotFoundError('TERMINAL_INPUT_NOT_PRESENT: ' + str(path))
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    folder.mkdir(parents=True, exist_ok=False); limit = rt.Limit(args.tag, gpu=False)
    started = time.time(); status, error = 'COMPLETE', None
    try:
        output = build_merged(args.producer_result, args.method_lock)
        dev = read(args.dev_result)
        if dev['status'] != 'COMPLETE': raise ValueError('DEV_NOT_COMPLETE')
        ru.assert_full_training(dev, output)
        output['frozen_assert_full_training_DEV102_FRESH72_passed'] = True
        output['dev_result'] = ref(args.dev_result, rt.ROOT)
        output['producer_merge_script'] = ref(__file__, rt.ROOT)
        ru.source_check(output['source_hashes']); limit.check()
    except Exception as exc:
        status = 'BUDGET_HELD_PRESERVED' if isinstance(exc, rt.BudgetHeld) else 'FAILED_PRESERVED'
        error = str(exc); output = {'status': status, 'error': error, 'producer_results': [ref(p, rt.ROOT) for p in args.producer_result], 'new_training_updates_in_merge': 0}
    output.update(tag=args.tag, started_unix=started, finished_unix=time.time(),
                  resource_receipt=limit.close(status, error), merge_GPU_work=False)
    rt.write(folder / 'RESULT.json', output)
    print(rt.json_line({'fresh_merge_terminal': str(folder / 'RESULT.json'), 'status': output['status'], 'case_count': len(output.get('cases', []))}), flush=True)
    if output['status'] != 'COMPLETE': raise SystemExit(2)


if __name__ == '__main__': main()
