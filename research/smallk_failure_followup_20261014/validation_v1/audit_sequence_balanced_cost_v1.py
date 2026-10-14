"""Independent saved-output and schedule arithmetic; no F or timing replay."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[1]
sys.path.insert(0, str(STUDY))
ROUTES = ('RAW_BOLT', 'MAIN_s92601', 'MAIN_s92602', 'STATIC_RF1_s92601',
          'TEMPORAL_RF3_s92601', 'STATIC_RF1_s92602', 'TEMPORAL_RF3_s92602')
STAGES = {'fill', 'normalization', 'H2D', 'E', 'F', 'oldD', 'head', 'anchor_inverse', 'D2H'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_arrays(record):
    path = resolve(record['path'])
    if any(p.lower() == 'test' for p in path.parts):
        raise PermissionError('PROTECTED_TEST_PATH')
    if sha(path) != record['sha256']:
        raise ValueError('SOURCE_SHA_MISMATCH:' + str(path))
    with np.load(path, allow_pickle=False) as opened:
        arrays = {key: opened[key].copy() for key in opened.files}
    if sha(path) != record['sha256']:
        raise ValueError('SOURCE_CHANGED:' + str(path))
    return arrays


def verify_file(record, role):
    path = resolve(record['path'])
    if any(p.lower() == 'test' for p in path.parts):
        raise PermissionError('PROTECTED_TEST_PATH:' + role)
    before = sha(path)
    if before != record['sha256']:
        raise ValueError('SOURCE_SHA_MISMATCH:' + role + ':' + str(path))
    return {'path': str(path), 'sha256': before}


def close_arrays(left, right, scale):
    atol, rtol = (1e-5, 2e-6) if scale == 'normalized' else (2e-3, 2e-5)
    return (left.shape == right.shape and np.isfinite(left).all()
            and np.isfinite(right).all()
            and bool(np.all(np.abs(left.astype(np.float64)-right.astype(np.float64))
                            <= atol + rtol * np.abs(right.astype(np.float64)))))


def record_original_bulk_flag(result, check, name, value):
    ok = isinstance(value, dict) and isinstance(value.get('pass'), bool)
    check(name + ':original_bulk_flag_schema', ok)
    if ok and value['pass'] is False:
        result['original_bulk_declared_false_count'] += 1


def audit(path, output):
    from continuation_runtime_v2 import ContinuationLimit
    limit = None
    result = {'status': 'RUNNING', 'checks': 0, 'failed_count': 0, 'failures': [],
              'whole_rows_checked': 0, 'stage_rows_checked': 0,
              'scope': 'saved outputs, schedule, paired ratios and stage arithmetic; no new F/timing replay',
              'original_bulk_scope': 'file/hash and producer-declared diagnostic flags only; original bulk arrays are not independently numerically replayed here',
              'original_bulk_declared_false_count': 0,
              'F_forward_replayed': False, 'GPU_execution': False,
              'independent_generalization': False, 'whole_goal_complete': False}

    def check(name, passed):
        result['checks'] += 1
        if not bool(passed):
            result['failed_count'] += 1
            result['failures'].append(name)

    try:
        if Path(output).exists():
            raise FileExistsError(output)
        limit = ContinuationLimit('sequence_balanced_cost_array_audit_v1', gpu=False)
        source = resolve(path)
        before = sha(source)
        producer = json.loads(source.read_text(encoding='utf-8'))
        result['cost_producer'] = {'path': str(source), 'sha256': before}
        check('producer_COMPLETE', producer['status'] == 'COMPLETE')
        check('F_preserved', producer['F_before'] == producer['F_after'])
        check('no_F_parameter_gradients', producer['F_has_no_parameter_gradients'] is True)
        check('codec_head_preserved', producer['codec_head_hashes_before'] == producer['codec_head_hashes_after'])
        expected = {(d, b, r) for d in ('robin', 'beijing_aotizhongxin') for b in (1, 8) for r in ROUTES}
        whole = {(r['dataset'], r['batch'], r['route']): r for r in producer['rows']}
        stages = {(r['dataset'], r['batch'], r['route']): r for r in producer['stage_rows']}
        check('whole_exact28', len(producer['rows']) == 28 and set(whole) == expected)
        check('stage_exact28', len(producer['stage_rows']) == 28 and set(stages) == expected)
        check('reported_coverage28', producer['actual_complete_rows'] == 28 and producer['actual_complete_stage_rows'] == 28)
        cached = {}
        references = {}
        for key, row in whole.items():
            limit.shared.check()
            name = str(key)
            timed = load_arrays(row['timed_outputs'])
            ref = load_arrays(row['source_reference'])
            original_bulk_file = verify_file(row['original_bulk_source'], name + ':original_bulk_source')
            cached[key], references[key] = timed, ref
            result.setdefault('source_references', []).append({
                'dataset': key[0], 'batch': key[1], 'route': key[2],
                'source_reference': row['source_reference'],
                'original_bulk_source': original_bulk_file})
            seconds = np.asarray(row['seconds'], dtype=float)
            check(name+':warm2_repeat7', row['status'] == 'COMPLETE' and row['warmup_each_route'] == 2 and row['repeats'] == 7)
            check(name+':finite_positive7', seconds.shape == (7,) and np.isfinite(seconds).all() and (seconds > 0).all())
            check(name+':blocks', row['block'] == list(range(7)) and np.array_equal(timed['block'], row['block']))
            check(name+':positions', sorted(row['position']) == list(range(7)) and np.array_equal(timed['position'], row['position']))
            check(name+':source_reference_route', row['source_reference'].get('route') == key[2])
            check(name+':median', np.isclose(np.median(seconds), row['median_seconds'], atol=1e-14, rtol=1e-12))
            check(name+':perwindow', np.isclose(np.median(seconds)/row['batch'], row['median_seconds_per_window'], atol=1e-14, rtol=1e-12))
            raw = whole[key[0], key[1], 'RAW_BOLT']
            raw_seconds = dict(zip(raw['block'], raw['seconds']))
            paired = [raw_seconds[b] for b in row['block']]
            ratios = seconds / np.asarray(paired)
            check(name+':paired_RAW', np.array_equal(np.asarray(row['paired_RAW_seconds']), paired))
            check(name+':ratios', np.allclose(row['paired_RAW_ratios'], ratios, atol=1e-14, rtol=1e-12))
            check(name+':ratio_median', np.isclose(row['paired_RAW_median_ratio'], np.median(ratios), atol=1e-14, rtol=1e-12))
            for scale in ('normalized', 'physical'):
                check(name+':'+scale+':shape', timed[scale].shape == (7,)+ref[scale].shape and ref[scale].shape[0] == key[1])
                for i in range(7):
                    passed = close_arrays(timed[scale][i], ref[scale], scale)
                    check(name+':'+scale+':output'+str(i), passed)
                    check(name+':'+scale+':declared'+str(i), row['parity'][i]['matched_batch_'+scale]['pass'] == passed)
                    record_original_bulk_flag(result, check, name+':'+scale+':original_bulk_declared'+str(i), row['parity'][i]['original_bulk_'+scale])
            result['whole_rows_checked'] += 1
        for d in ('robin', 'beijing_aotizhongxin'):
            for b in (1, 8):
                for block in range(7):
                    positions = [whole[d,b,r]['position'][block] for r in ROUTES]
                    check(f'{d}:{b}:block{block}:unique_positions', sorted(positions) == list(range(7)))
        for key, row in stages.items():
            limit.shared.check()
            name = str(key)
            arrays = load_arrays(row['stage_outputs'])
            whole_output_file = verify_file(row['whole_output_source'], name + ':stage_whole_output_source')
            result.setdefault('stage_references', []).append({
                'dataset': key[0], 'batch': key[1], 'route': key[2],
                'stage_outputs': row['stage_outputs'], 'whole_output_source': whole_output_file})
            timings = row['stage_seconds']
            check(name+':stage_warm2_repeat7', row['status'] == 'COMPLETE' and row['warmup_completed'] == 2 and row['repeats'] == 7 and len(timings) == 7)
            check(name+':stage9_coverage', all(set(t) == STAGES for t in timings))
            for stage in STAGES:
                vals = np.asarray([t[stage] for t in timings])
                check(name+':stage:'+stage+':positive', np.isfinite(vals).all() and (vals > 0).all())
                check(name+':stage:'+stage+':median', np.isclose(np.median(vals), row['stage_median_seconds'][stage], atol=1e-14, rtol=1e-12))
            check(name+':stage_not_whole', row['stage_sum_is_whole_latency'] is False and row['measured_separately_after_all_balanced_whole_trials'] is True)
            for scale in ('normalized', 'physical'):
                check(name+':stage:'+scale+':shape', arrays[scale].shape == cached[key][scale].shape)
                for i in range(7):
                    check(name+':stage:'+scale+':reference'+str(i), close_arrays(arrays[scale][i], references[key][scale], scale))
                    check(name+':stage:'+scale+':whole'+str(i), close_arrays(arrays[scale][i], cached[key][scale][i], scale))
                    check(name+':stage:'+scale+':declared_reference'+str(i), row['parity'][i]['matched_batch_'+scale]['pass'] is True)
                    check(name+':stage:'+scale+':declared_whole'+str(i), row['parity'][i]['balanced_whole_same_block_'+scale]['pass'] is True)
            result['stage_rows_checked'] += 1
        check('producer_unchanged', sha(source) == before)
        result['status'] = 'PASS_WITHOUT_NEW_F_OR_TIMING_REPLAY' if result['failed_count'] == 0 else 'FAIL_PRESERVED'
        limit.check()
    except Exception as error:
        result.update(status='FAILED_PRESERVED', error={'type': type(error).__name__, 'message': str(error)})
    finally:
        if limit:
            try:
                result['shared_resource_receipt'] = limit.close('COMPLETE' if result['status'].startswith('PASS') else 'FAILED_PRESERVED', result.get('error'))
                if result['shared_resource_receipt']['terminal_status'] != 'COMPLETE':
                    result['status'] = result['shared_resource_receipt']['terminal_status']
                    result['terminal_error'] = result['shared_resource_receipt'].get('terminal_error')
            except Exception as close_error:
                result.update(status='FAILED_PRESERVED', resource_close_unavailable={
                    'type': type(close_error).__name__, 'message': str(close_error)})
        result['audit_source_sha256'] = sha(__file__)
        with Path(output).open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cost-result', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    value = audit(args.cost_result, args.out)
    print(json.dumps({k: value[k] for k in ('status', 'checks', 'failed_count', 'whole_rows_checked', 'stage_rows_checked')}))
    raise SystemExit(0 if value['status'].startswith('PASS') else 1)
