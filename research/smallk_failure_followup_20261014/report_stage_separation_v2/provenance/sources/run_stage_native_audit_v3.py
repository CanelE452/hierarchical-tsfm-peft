"""Metered supplement to the preserved independent audit: all candidate arithmetic,
physical metrics, exact data contracts, teacher masks and simple controls.
No solver training, E/F forward, CUDA or protected TEST reads.
"""
from pathlib import Path
import argparse
import json
import sys

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def exact_conditions(keys):
    expected = {(d, s) for d in ('robin', 'beijing_aotizhongxin') for s in (92601, 92602)}
    if len(keys) != 4 or set(keys) != expected:
        raise ValueError('EXACT4_UNIQUE_ORIGINAL_CONDITIONS_REQUIRED')


def prefix_mask(mask, k):
    import numpy as np
    return np.repeat(np.logical_and.accumulate(mask.all(axis=2), axis=1)[:, :, None], k, axis=2)


def supplement(p, audit, full):
    import numpy as np
    splits, decoder, state = audit.load_source_bundle(p)
    failures = []
    checks = 0
    def verify(ok, label):
        nonlocal checks
        checks += 1
        if not ok:
            failures.append(label)
    def score(record, pred, values, label):
        nonlocal checks
        for units, std in (('normalized', None), ('physical_per_channel_diagnostic', values['std'])):
            actual = audit.observed_score(pred, values['target'], values['observed'], std)
            expected = record[units]
            audit.assert_score_close(expected, actual, label=label + ':' + units, failures=failures)
            for key in ('per_channel_mse', 'per_channel_mae'):
                verify(all(audit.approx_equal(a, b, atol=1e-8, rtol=1e-8) for a, b in zip(expected[key], actual[key]))
                       and len(expected[key]) == len(actual[key]), label + ':' + units + ':' + key)
            checks += 4
    c, k, evaluation, n_eval = audit.DATASETS[p['dataset']]
    for split, values in splits.items():
        full()
        n = {'fit': 256, 'tune': 64, evaluation: n_eval}[split]
        verify(values['latent_past'].shape == (n, 512, k), 'N512K:' + split)
        verify(values['target'].shape == values['observed'].shape == (n, 48, c), 'N48C:' + split)
        verify(values['observed'].dtype == np.bool_, 'BOOL_MASK:' + split)
        verify(np.array_equal(values['z_last'], values['latent_past'][:, -1:]), 'LAST_LATENT:' + split)
        verify(values['raw_last'].shape == (n, c), 'RAW_CURRENT_C:' + split)
        verify(np.isfinite(values['latent_past']).all() and np.isfinite(values['target'][values['observed']]).all(), 'FINITE_INPUT:' + split)
    fit = splits['fit']
    raw = audit.fixed_past_features(fit['latent_past'])
    center = raw.astype(np.float64).mean(axis=0)
    scale = np.sqrt(((raw.astype(np.float64) - center) ** 2).mean(axis=0)); scale[scale == 0] = 1.
    verify(np.array_equal(center, state['center']) and np.array_equal(scale, state['scale']), 'FIT_ONLY_FEATURE_STATS')
    verify(audit.array_sha(raw) == state['fit_raw_features_sha256'], 'FIT_RAW_FEATURE_SHA')
    teacher = prefix_mask(fit['observed'], k)
    teacher_input, _ = audit.load_npz_record(audit.find_artifact_by_stem(p, 'FIT_N1_TEACHER'), role='TEACHER')
    verify(np.array_equal(teacher_input['teacher_observed'], teacher), 'TEACHER_ACTUAL_COMPLETE_PREFIX_MASK')
    verify(p['teacher_contract']['fit_counts_by_horizon_K'] == teacher.sum(axis=0).tolist(), 'TEACHER_COUNTS')
    n1_coeff, _ = audit.load_npz_record(p['families']['N1']['selected_coefficient_artifact'], role='N1')
    designs = {s: audit.transform_features(state, v['latent_past']) for s, v in splits.items()}
    selected_latent = {s: (v['z_last'] + audit.predict_solution(designs[s], n1_coeff).astype(np.float32)).astype(np.float32)
                       for s, v in splits.items()}
    for family, row in p['families'].items():
        if row['status'] != 'COMPLETE':
            verify(False, 'INCOMPLETE_FAMILY:' + family); continue
        if family == 'N1':
            expected_counts = teacher.sum(axis=0)
        elif family == 'N2':
            expected_counts = fit['observed'].sum(axis=0)
        else:
            expected_counts = fit['observed'].sum(axis=(0, 1))
        verify(row['zero_observed_output_counts'] == int((expected_counts == 0).sum()), 'ZERO_OUTPUTS:' + family)
        for entry in row['alpha_grid']:
            full()
            arrays, _ = audit.load_npz_record(entry['coefficient_artifact'], role=family + '_CANDIDATE')
            verify(audit.solution_sha(entry['solution'], arrays) == entry['solution']['solution_sha256'], 'CANDIDATE_SOLUTION_SHA:' + family)
            verify(np.array_equal(arrays['observed_counts'], expected_counts), 'CANDIDATE_COUNTS:' + family)
            verify(np.isfinite(arrays['coefficients']).all() and np.isfinite(arrays['intercept']).all(), 'FINITE_COEFFICIENTS:' + family)
            if family in ('N1', 'N2'):
                verify(np.array_equal(arrays['channel_counts'], expected_counts.sum(axis=0)), 'CHANNEL_MACRO_DENOMINATOR:' + family)
            for split in ('fit', 'tune'):
                values = splits[split]
                if family == 'N1':
                    latent = (values['z_last'] + audit.predict_solution(designs[split], arrays).astype(np.float32)).astype(np.float32)
                    prediction = audit.original_anchored_prediction(latent, values['z_last'], values['raw_last'], decoder)
                elif family == 'N2':
                    prediction = values['raw_last'][:, None] + audit.predict_solution(designs[split], arrays).astype(np.float32)
                else:
                    if family == 'O1':
                        oracle, _ = audit.load_npz_record(audit.find_artifact_by_stem(p, 'ORACLE_INPUT_DIAGNOSTIC_' + split), role='ORACLE')
                        latent = oracle['oracle_z']
                    else:
                        latent = selected_latent[split]
                    prediction = audit.decoder_calibration_prediction(latent, values['z_last'], values['raw_last'], arrays)
                stored, _ = audit.load_npz_record(entry[split + '_artifact'], role='CANDIDATE_PREDICTION')
                verify(np.allclose(prediction, stored['prediction'], atol=audit.ATOL, rtol=audit.RTOL), 'ALL_CANDIDATE_ARITHMETIC:' + family + ':' + split)
                verify(np.array_equal(stored['origins'], values['origins']), 'CANDIDATE_ORIGINS:' + split)
                score(entry[split + '_scores'], stored['prediction'], values, family + ':' + split)
        for split, values in splits.items():
            stored, _ = audit.load_npz_record(row['selected_artifacts'][split], role='SELECTED_CONTRACT')
            for name in ('target', 'observed', 'origins', 'mean', 'std', 'raw_last'):
                verify(np.array_equal(stored[name], values[name]), 'SELECTED_INPUT_EXACT:' + family + ':' + name)
            score(row['splits'][split], stored['prediction'], values, family + ':selected:' + split)
    counts = fit['observed'].sum(axis=0)
    means = np.divide(np.where(fit['observed'], fit['target'].astype(np.float64) - fit['raw_last'][:, None], 0.).sum(axis=0),
                      counts, out=np.zeros(counts.shape, np.float64), where=counts > 0).astype(np.float32)
    for split, values in splits.items():
        full()
        for name, prediction in (('CURRENT_C_PERSISTENCE', np.repeat(values['raw_last'][:, None], 48, axis=1)),
                                 ('FIT_INTERCEPT_ONLY', values['raw_last'][:, None] + means)):
            score(p['controls'][name]['splits'][split], prediction, values, name + ':' + split)
        oracle, _ = audit.load_npz_record(audit.find_artifact_by_stem(p, 'ORACLE_INPUT_DIAGNOSTIC_' + split), role='O0_INPUT')
        prediction = audit.original_anchored_prediction(oracle['oracle_z'], values['z_last'], values['raw_last'], decoder)
        score(p['O0_ORACLE_OLD_D']['splits'][split], prediction, values, 'O0:' + split)
        for name, key in (('RAW', 'raw_forecast'), ('ORIGINAL_MAIN', 'original_prediction')):
            score(p['references'][split][name], values[key], values, name + ':' + split)
    return {'checks': checks, 'failures': failures, 'all_candidate_arithmetic': True, 'solver_optimality_replayed': False}


def reuse_raw_controls(p, audit, full):
    import numpy as np
    paths = {'robin': 'research/smallk_predictive_alignment_20261013/run_new_v1/robin/RAW_CONTROLS.json',
             'beijing_aotizhongxin': 'research/smallk_predictive_alignment_20261013/run_beijing_repair_v1/beijing_aotizhongxin/EVAL_RAW_CONTROLS.json'}
    result = {'dataset': p['dataset'], 'new_training': False, 'independent_old_training_replay': False,
              'matched_readout_capacity_or_anchor': False, 'methods': []}
    try:
        full()
        path = ROOT / paths[p['dataset']]
        source = audit.read_json(path)
        result['source'] = {'path': str(path), 'sha256': audit.sha_file(path)}
        if source['dataset'] != p['dataset']: raise ValueError('RAW_CONTROL_DATASET_MISMATCH')
        splits, _, _ = audit.load_source_bundle(p)
        for row in source['methods']:
            if row['method'] not in ('RAW_JOINT_RIDGE', 'RAW_SELF33', 'RAW_BOLT_RAW_R'): continue
            arrays, record = audit.load_npz_record(row['prediction'], role='OLD_RAW_CONTROL')
            scores = {}
            for split, values in splits.items():
                full()
                for name in ('target', 'observed', 'origins'):
                    if not np.array_equal(arrays[split + '_' + name], values[name]):
                        raise ValueError('OLD_RAW_CONTROL_NOT_COMPARABLE:' + split + ':' + name)
                if not np.array_equal(arrays['mean'], values['mean']) or not np.array_equal(arrays['std'], values['std']):
                    raise ValueError('OLD_RAW_CONTROL_DIFFERENT_STATS')
                prediction = arrays[split + '_prediction']
                scores[split] = {'normalized': audit.observed_score(prediction, values['target'], values['observed']),
                                'physical_per_channel_diagnostic': audit.observed_score(prediction, values['target'], values['observed'], values['std'])}
            result['methods'].append({'method': row['method'], 'prediction': record, 'splits': scores,
                                      'prior_training_metadata': {k: row[k] for k in ('correction', 'selection') if k in row}})
        result['status'] = 'COMPLETE' if len(result['methods']) == 3 else 'UNAVAILABLE'
    except Exception as error:
        import stage_separation_runtime_v1 as runtime
        if isinstance(error, runtime.BudgetHeld): raise
        result.update(status='UNAVAILABLE', methods=[], error={'type': type(error).__name__, 'message': str(error)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--producer-result', action='append', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    import stage_separation_runtime_v1 as runtime
    output = Path(args.out)
    if output.exists(): raise FileExistsError(output)
    limit = runtime.ContinuationLimit('stage_native_audit_v3', gpu=False)
    result = {'status': 'RUNNING', 'conditions': [], 'failed_count': 0,
              'authorization': limit.authorization_record, 'run_manifest': limit.run_manifest_record,
              'sources': [{'path': str(Path(__file__).resolve()), 'sha256': runtime.sha(__file__)}]}
    try:
        import audit_stage_native_core_v3 as audit
        import prepare_inputs
        import torch
        from threadpoolctl import threadpool_limits
        torch.set_num_threads(2)
        prepare_inputs.install_protected_test_guard()
        result['sources'].append({'path': str(Path(audit.__file__)), 'sha256': runtime.sha(audit.__file__)})
        producers = [audit.read_json(audit.resolve(p)) for p in args.producer_result]
        exact_conditions([(p['dataset'], p['seed']) for p in producers])
        for path, p in zip(args.producer_result, producers):
            limit.checkfull()
            with threadpool_limits(limits=2):
                base = audit.audit_one(audit.resolve(path))
                extra = supplement(p, audit, limit.checkfull)
            result['conditions'].append({'dataset': p['dataset'], 'seed': p['seed'],
                'producer_sha256': base['producer']['sha256'], 'producer_path': base['producer']['path'],
                'base': base, 'supplement': extra})
        result['failed_count'] = sum(len(r['base']['failures']) + len(r['supplement']['failures']) for r in result['conditions'])
        result['candidate_rows_checked'] = sum(r['base']['candidate_rows_checked'] for r in result['conditions'])
        result['selected_predictions_checked'] = sum(r['base']['selected_predictions_checked'] for r in result['conditions'])
        result['checks'] = sum(r['supplement']['checks'] for r in result['conditions'])
        result['reused_raw_controls'] = [reuse_raw_controls(next(p for p in producers if p['dataset'] == d), audit, limit.checkfull)
                                         for d in ('robin', 'beijing_aotizhongxin')]
        result['status'] = 'PASS' if result['failed_count'] == 0 else 'FAIL'
        limit.checkfull()
    except Exception as error:
        result.update(status='FAILED_PRESERVED', error={'type': type(error).__name__, 'message': str(error)})
    finally:
        result['resource_receipt'] = limit.close('COMPLETE' if result['status'] == 'PASS' else 'FAILED_PRESERVED', result.get('error'))
        if result['resource_receipt']['terminal_status'] != 'COMPLETE': result['status'] = result['resource_receipt']['terminal_status']
        runtime.write_new(output, result)
    print(json.dumps({k: result.get(k) for k in ('status', 'failed_count', 'checks', 'error')}))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
