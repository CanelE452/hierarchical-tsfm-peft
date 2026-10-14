"""Independent CPU re-score/math audit of the two completed D1 producers."""
from pathlib import Path
import ast
import hashlib
import json
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[1]
sys.path.insert(0, str(STUDY))
from followup_runtime import FollowupLimit

RUNS = {'robin': 'd1_robin_v1', 'beijing_aotizhongxin': 'd1_beijing_v1'}
OUT = HERE / 'D1_ARRAY_AUDIT_v1.json'
MD = HERE / 'D1_ARRAY_AUDIT_v1.md'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1048576), b''):
            digest.update(part)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def score(prediction, target, observed, std=None):
    error = np.asarray(prediction, np.float64) - np.asarray(target, np.float64)
    if std is not None:
        error = error * std
    counts, mse, mae = [], [], []
    for channel in range(error.shape[-1]):
        values = error[:, :, channel][observed[:, :, channel]]
        counts.append(len(values))
        mse.append(float(np.mean(values ** 2)) if len(values) else None)
        mae.append(float(np.mean(np.abs(values))) if len(values) else None)
    return {'mse': float(np.mean([value for value in mse if value is not None])),
            'mae': float(np.mean([value for value in mae if value is not None])),
            'per_channel_mse': mse, 'per_channel_mae': mae, 'observed_counts': counts,
            'active_channels': sum(value is not None for value in mse)}


def difference(left, right):
    gap = np.asarray(left, np.float64) - np.asarray(right, np.float64)
    return {'mse': float(np.mean(gap ** 2)), 'mae': float(np.mean(np.abs(gap))),
            'max_abs': float(np.max(np.abs(gap)))}


def main():
    if OUT.exists() or MD.exists():
        raise FileExistsError('audit outputs already exist; preserve them')
    result = {'status': 'RUNNING', 'scientific_arrays_read': True, 'GPU_forward_replayed': False,
              'protected_TEST_accessed': False, 'independent_holdout': False,
              'claim_scope': 'saved actual arrays, CPU FP32 math replay and independent observed-score recomputation',
              'tolerances': {'saved_FP32_math': {'atol': 1e-5, 'rtol': 2e-6},
                             'score_metadata': {'atol': 1e-11, 'rtol': 1e-11},
                             'basis_orthogonality_atol': 1e-10, 'FIT_SVD_projector_atol': 1e-9},
              'checks': 0, 'failed_count': 0, 'failures': [], 'datasets': {}, 'artifacts': []}
    limit = None
    started = time.perf_counter()
    try:
        limit = FollowupLimit('d1_independent_audit_v1', gpu=False)
        import prepare_inputs
        prepare_inputs.install_protected_test_guard()
        def check(name, passed, details=None):
            result['checks'] += 1
            if not bool(passed):
                result['failed_count'] += 1
                result['failures'].append({'check': name, 'details': details})
        def same(name, actual, expected):
            check(name, actual.dtype == expected.dtype and actual.shape == expected.shape
                  and np.array_equal(actual, expected, equal_nan=True), difference(actual, expected))
        def close(name, actual, expected, atol=1e-5, rtol=2e-6):
            check(name, actual.shape == expected.shape and np.allclose(actual, expected, atol=atol, rtol=rtol),
                  difference(actual, expected))
        def checked_arrays(path, metadata=None):
            path = Path(path)
            before = sha(path)
            if metadata is not None:
                check(str(path) + ':file_sha', before == metadata['sha256'])
            with np.load(path, allow_pickle=False) as opened:
                arrays = {key: opened[key].copy() for key in opened.files}
            if metadata is not None and 'arrays' in metadata:
                check(str(path) + ':array_keys', set(arrays) == set(metadata['arrays']))
                for key, value in arrays.items():
                    declared = metadata['arrays'][key]
                    check(str(path) + ':' + key + ':array_sha', array_sha(value) == declared['sha256'])
                    check(str(path) + ':' + key + ':shape_dtype', list(value.shape) == declared['shape']
                          and str(value.dtype) == declared['dtype'])
            after = sha(path)
            check(str(path) + ':sha_before_after', before == after)
            result['artifacts'].append({'path': str(path.relative_to(ROOT)), 'sha256_before': before,
                                         'sha256_after': after, 'bytes': path.stat().st_size})
            limit.check()
            return arrays
        for dataset, tag in RUNS.items():
            run = ROOT / '.cache' / STUDY.name / 'condition_probes' / tag
            producer = read(run / 'RESULT.json')
            d1path = run / 'D1/RESULT.json'
            d1 = read(d1path)
            check(dataset + ':producer_complete', producer['status'] == d1['status'] == 'COMPLETE')
            check(dataset + ':F_dual_hash_frozen', producer['F_before'] == producer['F_after']
                  and producer['F_frozen_pass'] is True)
            check(dataset + ':D1_result_hash', sha(d1path) == producer['probe_result']['sha256'])
            for record in producer['source_snapshot']:
                check(dataset + ':snapshot:' + record['snapshot_path'], sha(resolve(record['snapshot_path']))
                      == record['snapshot_sha256'] == record['sha256'])
            snapshot = run / 'sources/projected_past_probe.py'
            source = ast.parse(snapshot.read_text(encoding='utf-8-sig'))
            fit_function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'fit_bases')
            check(dataset + ':basis_function_only_RAW_last_arguments',
                  [arg.arg for arg in fit_function.args.args] == ['fit_raw_forecast', 'fit_last'])
            fit_calls = [node for node in ast.walk(source) if isinstance(node, ast.Call)
                         and isinstance(node.func, ast.Name) and node.func.id == 'fit_bases']
            call_args = [ast.unparse(arg) for arg in fit_calls[0].args]
            check(dataset + ':basis_call_no_actual_Y_or_mask', len(fit_calls) == 1 and call_args == [
                "splits['fit']['raw_forecast']", "splits['fit']['history'][:, -1]"])
            bridgepath = resolve(d1['provenance']['input_bridge']['path'])
            bridge = read(bridgepath)
            check(dataset + ':bridge_sha', sha(bridgepath) == d1['provenance']['input_bridge']['sha256'])
            artifacts = {entry['path']: entry for entry in d1['artifacts']}
            inputs, bridge_inputs = {}, {}
            for split in d1['split_roles']:
                record = next(entry for entry in bridge['artifacts'] if entry['dataset'] == dataset and entry['split'] == split)
                current = checked_arrays(run / 'D1' / ('INPUT_' + split + '.npz'), artifacts['INPUT_' + split + '.npz'])
                original = checked_arrays(resolve(record['path']), record)
                for name, origin in [('history_normalized', 'history'), ('history_observed', 'history_observed'),
                                     ('raw_forecast', 'raw_forecast'), ('target', 'target'), ('observed', 'observed'),
                                     ('origins', 'origins'), ('raw_last', 'raw_last'), ('mean', 'mean'), ('std', 'std')]:
                    same(dataset + ':' + split + ':' + name + ':exact_bridge', current[name], original[origin])
                inputs[split], bridge_inputs[split] = current, original
            rawrecord = producer['original_RAW_reference']
            raw = checked_arrays(resolve(rawrecord['path']), rawrecord)
            for split, current in inputs.items():
                for suffix, field in [('prediction', 'raw_forecast'), ('target', 'target'),
                                      ('observed', 'observed'), ('origins', 'origins')]:
                    same(dataset + ':' + split + ':original_RAW:' + suffix, raw[split + '_' + suffix], current[field])
                same(dataset + ':' + split + ':RAW_mean', raw['mean'], current['mean'])
                same(dataset + ':' + split + ':RAW_std', raw['std'], current['std'])
            delta = (inputs['fit']['raw_forecast'].astype(np.float64)
                     - inputs['fit']['raw_last'].astype(np.float64)[:, None]).reshape(-1, len(raw['mean']))
            mu = delta.mean(0)
            eigenfamilies = {}
            for mode in ('through_origin', 'centered'):
                values = delta if mode == 'through_origin' else delta - mu
                _, singular, vh = np.linalg.svd(values, full_matrices=False)
                eigenfamilies[mode] = (singular ** 2 / len(values), vh.T)
            channels = len(raw['mean'])
            expected_names = ['identity_C'] + [mode + '_' + label for mode in ('through_origin', 'centered')
                for label in ('quarter', 'half', 'full_C_rotated')]
            check(dataset + ':family_coverage', [entry['name'] for entry in d1['candidates']] == expected_names)
            rows = []
            for candidate in d1['candidates']:
                name, mode, k = candidate['name'], candidate['mode'], candidate['K']
                basis_arrays = checked_arrays(run / 'D1' / candidate['basis_artifact']['path'], candidate['basis_artifact'])
                b64, b32 = basis_arrays['basis'], basis_arrays['basis_runtime_float32']
                check(dataset + ':' + name + ':basis_shape_rank', b64.shape == (channels, k)
                      and np.linalg.matrix_rank(b64) == k)
                close(dataset + ':' + name + ':orthogonality', b64.T @ b64, np.eye(k), 1e-10, 0.)
                same(dataset + ':' + name + ':runtime_basis_cast', b32, b64.astype(np.float32))
                if mode == 'identity':
                    same(dataset + ':identity_matrix', b64, np.eye(channels))
                    same(dataset + ':identity_mean', basis_arrays['delta_mean'], np.zeros(channels))
                else:
                    values, vectors = eigenfamilies[mode]
                    close(dataset + ':' + name + ':FIT_RAW_SVD_projector', b64 @ b64.T,
                          vectors[:, :k] @ vectors[:, :k].T, 1e-9, 0.)
                    close(dataset + ':' + name + ':eigenvalues', basis_arrays['eigenvalues'], values, 1e-10, 1e-10)
                    close(dataset + ':' + name + ':delta_mean', basis_arrays['delta_mean'], mu, 1e-12, 1e-12)
                mean64 = basis_arrays['delta_mean']
                perpendicular64 = mean64 - (mean64 @ b64) @ b64.T if mode == 'centered' else np.zeros(channels)
                close(dataset + ':' + name + ':mean_perpendicular', basis_arrays['mean_perpendicular'], perpendicular64, 1e-12, 0.)
                for split, current in inputs.items():
                    declared = candidate['splits'][split]
                    produced = checked_arrays(run / 'D1' / declared['artifact']['path'], declared['artifact'])
                    history = current['history_normalized']
                    last = current['raw_last'][:, None]
                    forecast = current['raw_forecast']
                    latent = history @ b32
                    same(dataset + ':' + name + ':' + split + ':latent_past', produced['latent_past'], latent)
                    same(dataset + ':' + name + ':' + split + ':latent_last', produced['latent_last'], latent[:, -1:])
                    same(dataset + ':' + name + ':' + split + ':raw_last', produced['raw_last'], current['raw_last'])
                    reconstruction = latent @ b32.T
                    same(dataset + ':' + name + ':' + split + ':history_reconstruction', produced['reconstructed_history'], reconstruction)
                    delta32 = forecast - last
                    if mode == 'centered':
                        m32 = mean64.astype(np.float32)
                        projected = last + m32 + ((delta32 - m32) @ b32) @ b32.T
                        perpendicular32 = m32 - (m32 @ b32) @ b32.T
                    else:
                        projected = last + (delta32 @ b32) @ b32.T
                        perpendicular32 = np.zeros(channels, np.float32)
                    decoded = last + (produced['latent_future'] - last @ b32) @ b32.T + perpendicular32
                    same(dataset + ':' + name + ':' + split + ':projected_RAW_FP32', produced['projected_raw_forecast'], projected)
                    same(dataset + ':' + name + ':' + split + ':decoded_actual_latent_future_FP32', produced['projected_past_forecast'], decoded)
                    last64, forecast64 = last.astype(np.float64), forecast.astype(np.float64)
                    delta64 = forecast64 - last64
                    projected64 = (last64 + mean64 + ((delta64 - mean64) @ b64) @ b64.T
                                   if mode == 'centered' else last64 + (delta64 @ b64) @ b64.T)
                    close(dataset + ':' + name + ':' + split + ':projected_RAW_FP64',
                          produced['projected_raw_reference_fp64'], projected64, 1e-12, 1e-12)
                    scores = {}
                    for method, prediction in [('raw', forecast), ('projected_raw', projected), ('projected_past', decoded)]:
                        norm = score(prediction, current['target'], current['observed'])
                        physical = score(prediction, current['target'], current['observed'], current['std'])
                        scores[method] = {**norm, 'physical': physical}
                        for scale, computed, expected in [('normalized', norm, declared['scores'][method]),
                                                         ('physical', physical, declared['scores'][method]['physical'])]:
                            for metric, value in computed.items():
                                if isinstance(value, list):
                                    for channel, (a, b) in enumerate(zip(value, expected[metric])):
                                        check(dataset + ':' + name + ':' + split + ':' + method + ':' + scale + ':' + metric + ':' + str(channel),
                                              a is None and b is None or a is not None and b is not None and np.isclose(a, b, atol=1e-11, rtol=1e-11))
                                else:
                                    check(dataset + ':' + name + ':' + split + ':' + method + ':' + scale + ':' + metric,
                                          np.isclose(value, expected[metric], atol=1e-11, rtol=1e-11))
                    if mode == 'identity':
                        close(dataset + ':' + split + ':identity_actual_F_vs_RAW', produced['latent_future'], forecast)
                    full_reconstruction64 = (history.astype(np.float64) @ b64) @ b64.T
                    rows.append({'candidate': name, 'mode': mode, 'K': k, 'C': channels, 'split': split,
                        'scores': scores,
                        'past_reconstruction_FP32': difference(reconstruction, history),
                        'past_reconstruction_FP64_basis': difference(full_reconstruction64, history),
                        'actual_F_projection_noncommutation': difference(decoded, projected),
                        'normalized_MSE_change_vs_RAW_percent': 100 * (scores['projected_past']['mse'] / scores['raw']['mse'] - 1),
                        'normalized_MAE_change_vs_RAW_percent': 100 * (scores['projected_past']['mae'] / scores['raw']['mae'] - 1)})
                    limit.check()
            result['datasets'][dataset] = {'producer': {'path': str((run / 'RESULT.json').relative_to(ROOT)), 'sha256': sha(run / 'RESULT.json')},
                                            'D1_result_sha256': sha(d1path), 'candidate_count': 7,
                                            'split_count': len(inputs), 'rows': rows}
        result['coverage'] = {'datasets': len(result['datasets']), 'candidates': 14,
                               'candidate_split_outputs': 42, 'input_splits': 6,
                               'scientific_NPZ_read_count': len(result['artifacts'])}
        result['status'] = 'PASS_WITHOUT_INDEPENDENT_GPU_REPLAY' if result['failed_count'] == 0 else 'FAIL_PRESERVED'
    except Exception as error:
        result['status'] = 'FAILED_PRESERVED'
        result['failure'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        result['wall_seconds_before_terminal'] = time.perf_counter() - started
        if limit is not None:
            receipt = limit.close('COMPLETE' if result['status'].startswith('PASS') else 'FAILED_PRESERVED', result.get('failure'))
            result['shared_resource_receipt'] = receipt
            if receipt.get('terminal_status', 'COMPLETE') != 'COMPLETE':
                result.update(status=receipt['terminal_status'], terminal_error=receipt.get('terminal_error'))
        result['audit_source_sha256'] = sha(__file__)
        with OUT.open('x', encoding='utf-8') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
        lines = ['# D1 저장 실제 배열의 독립 CPU 검산', '',
                 f"상태: {result['status']}; 검사 {result['checks']}개, 실패 {result['failed_count']}개.", '',
                 '원 F를 GPU에서 다시 실행하지 않았다. 실제 producer의 F 전후 두 state hash·동결 기록과 원 source snapshot SHA를 확인하고, 저장된 absolute latent future에서 FP32 decode를 재계산했다.',
                 'FIT RAW forecast−정확한 과거 last만으로 SVD basis subspace/mean을 독립 재계산했다. source snapshot AST에서 basis 적합의 입력에 Y/mask가 들어가지 않는 호출을 확인했다.',
                 '모든 실제 output/INPUT/bridge/RAW NPZ의 SHA·array dtype/shape/byte SHA 및 observed 채널 macro 정규화·원단위 MSE/MAE를 다시 계산했다.',
                 'Robin VAL과 이미 노출된 Beijing EVAL을 사용한 개발 증거다. 검산 통과는 예측 개선이나 독립 일반화의 검증이 아니다.', '',
                 '```text', 'dataset              candidate                    K    MSE        MAE         MSEΔ%       MAEΔ%']
        for dataset, item in result['datasets'].items():
            held = 'val' if dataset == 'robin' else 'eval'
            for row in item['rows']:
                if row['split'] == held:
                    metric = row['scores']['projected_past']
                    lines.append(f"{dataset:<21}{row['candidate']:<29}{row['K']:<5}{metric['mse']:<11.7f}{metric['mae']:<12.7f}{row['normalized_MSE_change_vs_RAW_percent']:<12.3f}{row['normalized_MAE_change_vs_RAW_percent']:.3f}")
        lines.extend(['```', '', 'full-C rotation은 정보량을 줄이지 않아도 F의 입력 좌표계를 바꾼다. 실제 FP32 history 재구성과 독립 FP64 basis 재구성 오차는 JSON에 별도로 보존했다. MSE 감소와 MAE 증가가 함께 나타나면 개선 지표를 선택해 숨기지 않는다.', '',
                      '[원수치·검사·출처](D1_ARRAY_AUDIT_v1.json) · [감사 코드](audit_D1_actual_v1.py)', ''])
        with MD.open('x', encoding='utf-8') as handle:
            handle.write('\n'.join(lines))
    print(json.dumps({'status': result['status'], 'checks': result['checks'], 'failed': result['failed_count'],
                       'output': str(OUT), 'failure': result.get('failure')}, ensure_ascii=False))
    return 0 if result['status'].startswith('PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
