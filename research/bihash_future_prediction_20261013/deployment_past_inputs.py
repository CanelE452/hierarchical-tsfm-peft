"""Canonical first-window deployment inputs without constructing future targets."""
from pathlib import Path
import argparse
import numpy as np
import data as dt
import runtime as rt
import run as ru
import batch_science_reference as bs


LEGACY_MEMBERS = ('x', 'finite', 'mean', 'std', 'columns', 'times', 'val_origins')
PAST_MEMBERS = ('history', 'physical_history', 'history_observed', 'origins', 'mean', 'std')


def _result(value):
    if isinstance(value, dict):
        return value, None
    path = Path(value).resolve()
    return rt.read(path), rt._artifact(path)


def _checked_artifact(artifact):
    path = dt._check_not_test(rt.ROOT / artifact['path'])
    if 'target' in path.name.lower():
        raise ValueError('DEPLOYMENT_TARGET_PAYLOAD_FORBIDDEN')
    return bs.verify_artifact(artifact)


def _context(dev, fresh, evaluation, method_path):
    method_path = Path(method_path).resolve()
    lock = ru.check_method_sources(method_path)
    method_ref = rt._artifact(method_path)
    if not lock.get('locked') or lock.get('status') != 'LOCKED' or len(lock['source_hashes']) != 16:
        raise ValueError('DEPLOYMENT_METHOD_LOCK_OR_SOURCE16')
    for name, value in (('dev', dev), ('fresh', fresh), ('evaluation', evaluation)):
        if value.get('status') != 'COMPLETE' or value.get('source_hashes') != lock['source_hashes']:
            raise ValueError('DEPLOYMENT_RESULT_SOURCE16_OR_STATUS: ' + name)
    for value in (dev, fresh):
        ref = value['method_lock']
        if ref['sha256'] != method_ref['sha256'] or (rt.ROOT / ref['path']).resolve() != method_path:
            raise ValueError('DEPLOYMENT_RESULT_METHOD_LOCK')
    global_ref = evaluation['global_model_lock']
    global_lock = rt.read(_checked_artifact(global_ref))
    if (not global_lock.get('ALL_SELECTED') or global_lock.get('status') != 'ALL_SELECTED'
            or not global_lock.get('all_core_training_completed')
            or not global_lock.get('all_readouts_and_controls_selected')
            or global_lock.get('method_lock_sha256') != method_ref['sha256']
            or global_lock.get('source_hashes') != lock['source_hashes']):
        raise ValueError('DEPLOYMENT_GLOBAL_MODEL_LOCK')
    return lock, method_ref, global_ref


def _science_past_metadata(dataset, evaluation):
    records = [r for r in evaluation['core_evaluation'] if r['dataset'] == dataset]
    if not records:
        raise ValueError('DEPLOYMENT_SCIENCE_DATASET_MISSING')
    record = records[0]
    split = 'val' if dataset in ru.LEGACY else 'eval'
    artifacts = [r['prediction'] for r in record['entries'] if r['variant'] == 'regular']
    if record['split'] != split or len(artifacts) != 1:
        raise ValueError('DEPLOYMENT_SCIENCE_REGULAR_METADATA')
    artifact = artifacts[0]
    members = ('mean', 'std', split + '_origins')
    with np.load(_checked_artifact(artifact), allow_pickle=False) as z:
        values = {key: z[key].copy() for key in members}
    return values, {'artifact': artifact, 'accessed_members': list(members),
                    'prediction_arrays_opened': False, 'target_arrays_opened': False}


def _legacy_past(dataset, dev, windows):
    record = dev['data'][dataset]
    contracts = [r for r in record['source_records'] if r['role'] == 'legacy_contract']
    sources = [r for r in record['source_records'] if r['role'] == 'legacy_trainval_npz']
    if len(contracts) != 1 or len(sources) != 1:
        raise ValueError('DEPLOYMENT_LEGACY_SOURCE_RECORDS')
    contract_ref, source_ref = contracts[0], sources[0]
    saved = record['contract']
    if (saved['legacy_contract_sha256'] != contract_ref['sha256']
            or (rt.ROOT / saved['legacy_contract_path']).resolve() != (rt.ROOT / contract_ref['path']).resolve()):
        raise ValueError('DEPLOYMENT_LEGACY_CONTRACT_REFERENCE')
    contract = rt.read(_checked_artifact(contract_ref))
    if (contract['dataset'] != dataset or contract['trainval']['sha256'] != source_ref['sha256']
            or (rt.ROOT / contract['trainval']['path']).resolve() != (rt.ROOT / source_ref['path']).resolve()):
        raise ValueError('DEPLOYMENT_LEGACY_SOURCE_REFERENCE')
    with np.load(_checked_artifact(source_ref), allow_pickle=False) as z:
        values = {key: z[key].copy() for key in LEGACY_MEMBERS}
    x = values['x'].astype(np.float32)
    mask = values['finite'].astype(bool)
    source_mean = values['mean'].astype(np.float32)
    source_std = values['std'].astype(np.float32)
    source_std = np.where(source_std > 1e-6, source_std, 1.0).astype(np.float32)
    columns = [str(v) for v in values['columns'].tolist()]
    all_origins = values['val_origins'].astype(np.int64)
    if (x.ndim != 2 or mask.shape != x.shape or source_mean.shape != (x.shape[1],)
            or source_std.shape != source_mean.shape or len(columns) != x.shape[1]
            or all_origins.ndim != 1 or len(all_origins) < windows
            or np.any(all_origins < dt.L) or np.any(all_origins > len(x))
            or np.any(np.diff(all_origins) <= 0)):
        raise ValueError('DEPLOYMENT_LEGACY_SHAPE_OR_ORIGINS')
    physical = (x * source_std + source_mean).astype(np.float32)
    physical[~mask] = np.nan
    train_bounds = tuple(contract.get('bounds', {}).get('train', contract.get('train_bounds')))
    if len(train_bounds) != 2 or not 0 <= train_bounds[0] < train_bounds[1] <= len(x):
        raise ValueError('DEPLOYMENT_LEGACY_TRAIN_BOUNDS')
    fit_end = train_bounds[0] + int(np.floor(.8 * (train_bounds[1] - train_bounds[0])))
    fit_bounds = (train_bounds[0], fit_end)
    if list(fit_bounds) != saved['bounds']['fit']:
        raise ValueError('DEPLOYMENT_LEGACY_FIT_BOUNDS')
    mean, std = dt._fit_statistics(physical, mask, fit_bounds)
    if (dt.array_hash(mean) != record['mean_sha256'] or dt.array_hash(std) != record['std_sha256']
            or dt.array_hash(all_origins) != record['origin_sha256']['val']):
        raise ValueError('DEPLOYMENT_LEGACY_STAT_OR_ORIGIN_RECEIPT')
    history, raw, observed = dt._history_windows(physical, mask, all_origins[:windows], mean, std)
    pack = {'dataset': dataset, 'C': record['C'], 'K': record['K'], 'columns': columns,
            'mean': mean, 'std': std, 'history': {'val': history},
            'raw_history': {'val': raw}, 'raw_history_observed': {'val': observed},
            'origins': {'val': all_origins[:windows].copy()}}
    evidence = {'artifact': source_ref, 'contract_artifact': contract_ref,
                'accessed_members': list(LEGACY_MEMBERS), 'fit_bounds': list(fit_bounds),
                'reconstruction': '(x.float32 * guarded_source_std.float32 + source_mean.float32).float32; missing -> NaN',
                'statistics_API': 'data._fit_statistics(reconstructed_physical, finite_mask, FIT_bounds)',
                'history_API': 'data._history_windows(first_origins, FIT_mean, FIT_std)',
                'legacy_trainval_raw_bytes_opened': True,
                'legacy_x_contains_already_exposed_DEV_series': True,
                'selection_scope': 'FIT row statistics and first past windows only; no future windows assembled',
                'times_member_loaded_for_original_member_contract': True,
                'original_raw_source_bytes_opened': False, 'target_arrays_opened': False}
    return pack, all_origins, evidence


def load_past_pack(dataset, dev_result, fresh_result, evaluation_result, method_lock_path, windows=8):
    """Return only the canonical prefix of VAL/EVAL histories and access evidence."""
    if isinstance(windows, bool) or not isinstance(windows, int) or not 1 <= windows <= 8:
        raise ValueError('DEPLOYMENT_PAST_WINDOWS_MUST_BE_1_TO_8')
    dataset = dt.canonical_dataset(dataset)
    protected = dt.access_guard()
    dev, dev_ref = _result(dev_result)
    fresh, fresh_ref = _result(fresh_result)
    evaluation, evaluation_ref = _result(evaluation_result)
    lock, method_ref, global_ref = _context(dev, fresh, evaluation, method_lock_path)
    split = 'val' if dataset in ru.LEGACY else 'eval'
    if dataset in ru.LEGACY:
        pack, all_origins, source = _legacy_past(dataset, dev, windows)
    else:
        artifact = fresh['contracts'][dataset]['payloads']['eval_past_input']
        dt._check_not_test(rt.ROOT / artifact['path'])
        if Path(artifact['path']).name != 'EVAL_past_input_payload.npz':
            raise ValueError('DEPLOYMENT_EVAL_PAST_PAYLOAD_REQUIRED')
        pack, source = bs.load_past_only(dataset, fresh, evaluation, method_ref)
        all_origins = pack['origins']['eval'].copy()
        if len(all_origins) < windows:
            raise ValueError('DEPLOYMENT_PAST_PREFIX_TOO_SHORT')
        for field in ('history', 'raw_history', 'raw_history_observed', 'origins'):
            pack[field] = {'eval': pack[field]['eval'][:windows].copy()}
        source['past_payload_raw_bytes_opened'] = True
        source['selection_scope'] = 'first canonical EVAL past windows and stored FIT statistics'
    science, science_evidence = _science_past_metadata(dataset, evaluation)
    if (not np.array_equal(all_origins, science[split + '_origins'])
            or not np.array_equal(pack['mean'], science['mean'])
            or not np.array_equal(pack['std'], science['std'])):
        raise ValueError('DEPLOYMENT_SCIENCE_STATISTICS_OR_ORIGINS_NOT_EXACT')
    history, raw, mask = [pack[field][split] for field in ('history', 'raw_history', 'raw_history_observed')]
    if (history.shape != (windows, dt.L, pack['C']) or raw.shape != history.shape or mask.shape != history.shape
            or history.dtype != np.float32 or raw.dtype != np.float32 or mask.dtype != np.bool_
            or pack['origins'][split].dtype != np.int64 or not np.isfinite(history).all()
            or not np.isfinite(raw[mask]).all() or not np.isfinite(pack['mean']).all()
            or not np.isfinite(pack['std']).all() or np.any(pack['std'] <= 0)):
        raise ValueError('DEPLOYMENT_PAST_PACK_SHAPE_MASK_OR_STATS')
    evidence = {'dataset': dataset, 'split': split, 'selected_window_count': windows,
                'source_window_count': len(all_origins), 'exact_canonical_prefix': True,
                'selected_origins': pack['origins'][split].tolist(),
                'source16_exact': True, 'source16_hashes': lock['source_hashes'],
                'method_lock': method_ref, 'global_model_lock': global_ref,
                'global_model_lock_exact': True, 'source': source, 'science_metadata': science_evidence,
                'input_results': {'dev': dev_ref, 'fresh': fresh_ref, 'evaluation': evaluation_ref},
                'mean_sha256': dt.array_hash(pack['mean']), 'std_sha256': dt.array_hash(pack['std']),
                'full_origin_sha256': dt.array_hash(all_origins),
                'target_arrays_opened': False, 'protected_TEST_opened_or_hashed': False,
                'protected_TEST_guard_path_count': len(protected),
                'fresh_EVAL_past_allowed_by_existing_global_open_receipt': dataset in ru.FRESH}
    return pack, evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    limit = rt.Limit(args.tag, gpu=False)
    try:
        inputs = [rt.HERE / name / 'RESULT.json' for name in ('run_dev_v1', 'run_fresh_merged_v1', 'run_evaluate_v1')]
        rows = []
        for dataset in ru.LEGACY + ru.FRESH:
            limit.check()
            pack, evidence = load_past_pack(dataset, *inputs, rt.HERE / 'METHOD_LOCK.json')
            split = evidence['split']
            reproduced = np.stack([dt.window_forward_fill_normalize(raw, mask, pack['mean'], pack['std'])
                for raw, mask in zip(pack['raw_history'][split], pack['raw_history_observed'][split])])
            if not np.array_equal(reproduced, pack['history'][split]):
                raise ValueError('DEPLOYMENT_ACTUAL_PAST_PREPROCESS_NOT_EXACT')
            rows.append({'evidence': evidence, 'history_shape': list(pack['history'][split].shape),
                         'past_preprocess_exact': True, 'missing_past_cells': int((~pack['raw_history_observed'][split]).sum()),
                         'history_sha256': dt.array_hash(pack['history'][split]),
                         'raw_history_sha256': dt.array_hash(pack['raw_history'][split]),
                         'raw_history_observed_sha256': dt.array_hash(pack['raw_history_observed'][split])})
            print(rt.json_line({'dataset': dataset, 'shape': rows[-1]['history_shape'], 'source_windows': evidence['source_window_count']}), flush=True)
        resource = limit.close()
        rt.write(args.output, {'status': 'COMPLETE', 'CPU_only': True, 'rows': rows,
                              'loader': rt._artifact(Path(__file__)), 'resource_receipt': resource})
    except Exception as exc:
        limit.close('FAILED_PRESERVED', exc)
        raise


if __name__ == '__main__':
    main()
