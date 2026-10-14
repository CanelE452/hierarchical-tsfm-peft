"""CPU-only saved-array checks; no new F forward or independent timing trial."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[1]
sys.path.insert(0, str(STUDY))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def checked_path(record):
    path = Path(record['path'])
    if not path.is_absolute():
        path = ROOT / path
    if any(p.lower() == 'test' for p in path.parts):
        raise PermissionError('PROTECTED_TEST_PATH')
    if digest(path) != record['sha256']:
        raise ValueError('ARTIFACT_SHA_MISMATCH:' + str(path))
    return path


def audit(path, kind, output):
    from followup_runtime import FollowupLimit
    from prepare_inputs import install_protected_test_guard
    install_protected_test_guard()
    started = time.perf_counter()
    result = {'status': 'RUNNING', 'failed_count': 0, 'checks': 0,
              'scope': 'saved7 outputs, same-batch reference, timing arithmetic and metadata; no new F/timing replay',
              'GPU_execution': False, 'whole_goal_complete': False, 'failures': [], 'rows': []}
    limit = None

    def check(name, passed):
        result['checks'] += 1
        if not bool(passed):
            result['failed_count'] += 1
            result['failures'].append(name)

    try:
        limit = FollowupLimit('cost_array_audit_' + kind, gpu=False)
        path = Path(path).resolve()
        source_hash = digest(path)
        producer = json.loads(path.read_text(encoding='utf-8'))
        result['cost_producer'] = {'path': str(path), 'sha256': source_hash}
        expected_count = 28 if kind == 'joint' else 14
        check('producer_complete', producer['status'] == 'COMPLETE')
        check('coverage_count', len(producer['rows']) == expected_count == producer['actual_complete_rows'])
        check('F_preserved', producer['F_before'] == producer['F_after'])
        check('F_no_parameter_gradient', producer['F_has_no_parameter_gradients'] is True)
        keys = [(r['dataset'], r['seed'], r['objective'], r['batch']) for r in producer['rows']]
        if kind == 'joint':
            expected = {(d, s, o, b) for d in ('robin', 'beijing_aotizhongxin')
                        for s in (92601, 92602) for o in ('Y', 'RAW', 'MIXED') for b in (1, 8)}
            expected |= {(d, None, None, b) for d in ('robin', 'beijing_aotizhongxin') for b in (1, 8)}
        else:
            expected = {('appliances', s, o, b) for s in (92601, 92602)
                        for o in ('MSE', 'MAE', 'HUBER_DELTA1') for b in (1, 8)}
            expected |= {('appliances', None, None, b) for b in (1, 8)}
            check('RAW_plus_R_cost_not_measured', producer['RAW_PLUS_R_cost_measured'] is False)
        check('unique_exact_conditions', len(set(keys)) == expected_count and set(keys) == expected)
        for index, row in enumerate(producer['rows']):
            limit.shared.check()
            name = str(keys[index])
            timed_path = checked_path(row['timed_outputs'])
            reference_path = checked_path(row['matched_reference'])
            with np.load(timed_path, allow_pickle=False) as values:
                timed = {key: values[key].copy() for key in ('normalized', 'physical')}
            with np.load(reference_path, allow_pickle=False) as values:
                reference = {key: values[key].copy() for key in ('normalized', 'physical')}
            seconds = np.asarray(row['seconds'], dtype=np.float64)
            check(name + ':complete_warm2_repeat7', row['status'] == 'COMPLETE' and row['warmup'] == 2
                  and row['repeats'] == 7 and len(row['parity']) == 7)
            check(name + ':positive_finite7_seconds', seconds.shape == (7,) and np.isfinite(seconds).all() and (seconds > 0).all())
            check(name + ':median', np.isclose(np.median(seconds), row['median_seconds'], atol=1e-14, rtol=1e-12))
            check(name + ':per_window', np.isclose(np.median(seconds) / row['batch'], row['median_seconds_per_window'], atol=1e-14, rtol=1e-12))
            max_abs = {}
            for scale, atol, rtol in (('normalized', 1e-5, 2e-6), ('physical', 2e-3, 2e-5)):
                check(name + ':' + scale + ':shape7', timed[scale].shape == (7,) + reference[scale].shape
                      and reference[scale].shape[0] == row['batch'])
                check(name + ':' + scale + ':finite', np.isfinite(timed[scale]).all() and np.isfinite(reference[scale]).all())
                difference = timed[scale].astype(np.float64) - reference[scale].astype(np.float64)
                max_abs[scale] = float(np.max(np.abs(difference)))
                for repeat in range(7):
                    passed = bool(np.all(np.abs(difference[repeat]) <= atol + rtol * np.abs(reference[scale])))
                    recorded = row['parity'][repeat]['matched_batch_' + scale]['pass']
                    check(name + ':' + scale + ':actual_pass_' + str(repeat), passed)
                    check(name + ':' + scale + ':record_agrees_' + str(repeat), passed == recorded)
            result['rows'].append({'condition': list(keys[index]), 'median_seconds': float(np.median(seconds)),
                                   'max_abs_vs_same_batch': max_abs,
                                   'timed_outputs': row['timed_outputs'], 'matched_reference': row['matched_reference']})
        check('producer_unchanged', digest(path) == source_hash)
        result['status'] = 'PASS_WITHOUT_NEW_F_OR_TIMING_REPLAY' if result['failed_count'] == 0 else 'FAIL_PRESERVED'
        limit.check()
    except Exception as error:
        result.update(status='FAILED_PRESERVED', error={'type': type(error).__name__, 'message': str(error)})
    finally:
        if limit:
            receipt = limit.close('COMPLETE' if result['status'].startswith('PASS') else 'FAILED_PRESERVED', result.get('error'))
            result['shared_resource_receipt'] = receipt
            if receipt.get('terminal_status', 'COMPLETE') != 'COMPLETE':
                result['status'] = receipt['terminal_status']
        result['audit_source_sha256'] = digest(__file__)
        result['wall_seconds_before_write'] = time.perf_counter() - started
        with Path(output).open('x', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, allow_nan=False)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cost-result', required=True)
    parser.add_argument('--kind', choices=('joint', 'app'), required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    value = audit(args.cost_result, args.kind, args.out)
    print(json.dumps({'status': value['status'], 'checks': value['checks'], 'failed_count': value['failed_count']}))
    raise SystemExit(0 if value['status'].startswith('PASS') else 1)
