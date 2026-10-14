"""Independent CPU audit of four completed frozen-decoder refit producers."""
from pathlib import Path
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
from audit_D1_actual_v1 import sha, read, resolve, array_sha, score, difference

OUT = HERE / 'D2_ARRAY_AUDIT_v1.json'
MD = HERE / 'D2_ARRAY_AUDIT_v1.md'
ALPHAS = (0., 1e-4, .001, .01, .1, 1., 10.)
DECODER_KEYS = ('codec.decode.weight', 'codec.decode.bias', 'residual.0.weight',
                'residual.0.bias', 'residual.2.weight', 'residual.2.bias')


def state_hash(state, bi=False):
    digest = hashlib.sha256()
    iterator = sorted(state.items()) if bi else state.items()
    for name, value in iterator:
        value = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        if bi:
            digest.update(str(value.dtype).encode())
            digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def cpu_decoder_features(values, decoder):
    import torch
    import torch.nn.functional as functional
    def linear(x, weight, bias):
        return functional.linear(torch.from_numpy(x.copy()), torch.from_numpy(weight.copy()), torch.from_numpy(bias.copy()))
    future, last = values['z_future'], values['z_last']
    hidden_future = functional.gelu(linear(future, decoder['residual.0.weight'], decoder['residual.0.bias']), approximate='none')
    hidden_last = functional.gelu(linear(last, decoder['residual.0.weight'], decoder['residual.0.bias']), approximate='none')
    features = np.concatenate((future - last, (hidden_future - hidden_last).numpy()), axis=-1)
    def source_decode(z, hidden):
        return linear(z, decoder['codec.decode.weight'], decoder['codec.decode.bias']) + functional.linear(
            hidden, torch.from_numpy(decoder['residual.2.weight'].copy()), torch.from_numpy(decoder['residual.2.bias'].copy()))
    original = (torch.from_numpy(values['raw_last'].copy())[:, None] + source_decode(future, hidden_future)
                - source_decode(last, hidden_last)).numpy()
    outgoing = np.concatenate((decoder['codec.decode.weight'].T, decoder['residual.2.weight'].T), axis=0)
    factored = values['raw_last'][:, None] + features @ outgoing
    return features, original, factored, outgoing


def fit_head(features, target, last, mask, alpha, fallback):
    # Independently reconstructed masked mean-square ridge equations.
    x64 = features.reshape(-1, features.shape[-1]).astype(np.float64)
    rms = np.sqrt(np.mean(x64 ** 2, axis=0))
    rms[rms == 0] = 1.
    x64 = x64 / rms
    residual = (target.astype(np.float64) - last.astype(np.float64)[:, None]).reshape(-1, target.shape[-1])
    observed = mask.reshape(-1, target.shape[-1])
    counts = observed.sum(0)
    scaled_weights = fallback.astype(np.float64) * rms[:, None]
    for channel in np.flatnonzero(counts):
        x, y = x64[observed[:, channel]], residual[observed[:, channel], channel]
        if alpha == 0:
            scaled_weights[:, channel] = np.linalg.lstsq(x, y, rcond=None)[0]
        else:
            scaled_weights[:, channel] = np.linalg.solve(x.T @ x / len(x) + alpha * np.eye(x.shape[1]), x.T @ y / len(x))
    outgoing = (scaled_weights / rms[:, None]).astype(np.float32)
    outgoing[:, counts == 0] = fallback[:, counts == 0]
    return outgoing, scaled_weights.astype(np.float32), rms, counts


def scores(prediction, values):
    normalized = score(prediction, values['target'], values['observed'])
    physical = score(prediction, values['target'], values['observed'], values['std'])
    teacher = score(prediction, values['raw_forecast'], values['observed'])
    return {'normalized': normalized, 'physical': physical, 'raw_teacher_normalized': teacher}


def main():
    if OUT.exists() or MD.exists():
        raise FileExistsError('new audit tag required; preserve original audit')
    started = time.perf_counter()
    result = {'status': 'RUNNING', 'checks': 0, 'failed_count': 0, 'failures': [], 'conditions': [], 'artifacts': [],
              'scientific_arrays_read': True, 'GPU_initialized': False, 'F_forward_replayed': False,
              'independent_holdout': False, 'protected_TEST_accessed': False,
              'physical_scores_interpretation': 'Channel-specific units preserved; no heterogeneous-unit macro superiority claim.',
              'tolerances': {'source_prediction': {'atol': 1e-5, 'rtol': 2e-6},
                             'score_metadata': {'atol': 1e-11, 'rtol': 1e-11}},
              'selection': 'Recompute all original seven TUNE actual-Y MSE scores and earliest exact tie; no VAL/EVAL selection'}
    limit = None
    try:
        limit = FollowupLimit('d2_independent_audit_v1', gpu=False)
        import prepare_inputs
        prepare_inputs.install_protected_test_guard()
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
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
        def checked_arrays(path, metadata):
            path = Path(path)
            before = sha(path)
            check(str(path) + ':file_sha', before == metadata['sha256'])
            with np.load(path, allow_pickle=False) as opened:
                arrays = {key: opened[key].copy() for key in opened.files}
            if 'arrays' in metadata:
                check(str(path) + ':keys', set(arrays) == set(metadata['arrays']))
                for key, value in arrays.items():
                    declared = metadata['arrays'][key]
                    check(str(path) + ':' + key + ':array_sha', array_sha(value) == declared['sha256'])
                    check(str(path) + ':' + key + ':dtype_shape', str(value.dtype) == declared['dtype']
                          and list(value.shape) == declared['shape'])
            after = sha(path)
            check(str(path) + ':file_unchanged', after == before)
            result['artifacts'].append({'path': str(path.relative_to(ROOT)), 'sha256_before': before,
                                         'sha256_after': after, 'bytes': path.stat().st_size})
            limit.check()
            return arrays
        def compare_scores(name, computed, expected):
            for scale in computed:
                for metric, value in computed[scale].items():
                    stored = expected[scale][metric]
                    if isinstance(value, list):
                        check(name + ':' + scale + ':' + metric + ':length', len(value) == len(stored))
                        for channel, (a, b) in enumerate(zip(value, stored)):
                            check(name + ':' + scale + ':' + metric + ':' + str(channel),
                                  a is None and b is None or a is not None and b is not None and np.isclose(a, b, atol=1e-11, rtol=1e-11))
                    else:
                        check(name + ':' + scale + ':' + metric, np.isclose(value, stored, atol=1e-11, rtol=1e-11))
        for dataset, short in (('robin', 'robin'), ('beijing_aotizhongxin', 'beijing')):
            for seed in (92601, 92602):
                tag = f'd2_{short}_s{seed}_v1'
                folder = ROOT / '.cache' / STUDY.name / 'condition_probes' / tag
                parent = read(folder / 'RESULT.json')
                d2folder = folder / 'D2'
                d2 = read(d2folder / 'RESULT.json')
                check(tag + ':COMPLETE', parent['status'] == d2['status'] == 'COMPLETE')
                check(tag + ':D2_result_hash', sha(d2folder / 'RESULT.json') == parent['probe_result']['sha256'])
                check(tag + ':F_frozen', parent['F_before'] == parent['F_after'] and parent['F_frozen_pass'])
                check(tag + ':MAIN_frozen', parent['MAIN_codec_frozen_pass'] and parent['MAIN_codec_hash_after']
                      == parent['MAIN_checkpoint']['selected_codec_hash_old_algorithm'])
                cp_record = parent['MAIN_checkpoint']['checkpoint']
                cp_path = resolve(cp_record['path'])
                before = sha(cp_path)
                check(tag + ':original_CP_sha', before == cp_record['sha256'])
                checkpoint = torch.load(cp_path, map_location='cpu', weights_only=True)
                state = checkpoint['codec']
                check(tag + ':old_state_hash', state_hash(state) == parent['MAIN_checkpoint']['selected_codec_hash_old_algorithm'])
                check(tag + ':Bi_state_hash', state_hash(state, bi=True) == parent['MAIN_checkpoint']['selected_codec_hash_bi_algorithm'])
                check(tag + ':CP_metadata', checkpoint['seed'] == seed and checkpoint['architecture'] == 'ANCHOR_NONLINEAR_D'
                      and checkpoint['update'] == parent['MAIN_checkpoint']['selected_update'] and checkpoint['method'] == 'MAIN'
                      and checkpoint['weight'] == 0.)
                for snapshot in parent['source_snapshot']:
                    check(tag + ':source_snapshot:' + snapshot['snapshot_path'], sha(resolve(snapshot['snapshot_path']))
                          == snapshot['snapshot_sha256'] == snapshot['sha256'])
                art = {entry['path']: entry for entry in d2['artifacts']}
                decoder = checked_arrays(d2folder / 'SOURCE_DECODER.npz', art['SOURCE_DECODER.npz'])
                check(tag + ':decoder_keys', tuple(decoder) == DECODER_KEYS)
                digest = hashlib.sha256()
                for key in DECODER_KEYS:
                    expected = state[key].detach().numpy()
                    same(tag + ':original_decoder:' + key, decoder[key], expected)
                    digest.update(key.encode())
                    digest.update(np.ascontiguousarray(decoder[key]).tobytes())
                check(tag + ':decoder_state_hash', digest.hexdigest() == d2['source_decoder_state_sha256']
                      == d2['source_record']['decoder_state_sha256'])
                bridgepath = resolve(parent['bridge_provenance']['input_bridge']['path'])
                bridge = read(bridgepath)
                check(tag + ':bridge_hash', sha(bridgepath) == parent['bridge_provenance']['input_bridge']['sha256'])
                original_sources = bridge['declared_asset_hashes']['predictions'][dataset]
                raw = checked_arrays(resolve(original_sources['raw']['path']), original_sources['raw'])
                main = checked_arrays(resolve(original_sources['main_' + str(seed)]['path']), original_sources['main_' + str(seed)])
                inputs, feature_arrays, baselines = {}, {}, {}
                outgoing_original = None
                for split in d2['split_roles']:
                    absolute_record = next(entry for entry in parent['artifacts'] if Path(entry['path']).name == 'ABSOLUTE_LATENTS_' + split + '.npz')
                    absolute = checked_arrays(resolve(absolute_record['path']), absolute_record)
                    values = checked_arrays(d2folder / ('INPUT_' + split + '.npz'), art['INPUT_' + split + '.npz'])
                    bridge_record = next(entry for entry in bridge['artifacts'] if entry['dataset'] == dataset and entry['split'] == split)
                    original = checked_arrays(resolve(bridge_record['path']), bridge_record)
                    for key in absolute:
                        same(tag + ':' + split + ':absolute_input:' + key, values[key], absolute[key])
                    same(tag + ':' + split + ':absolute_last_not_delta', values['z_last'], values['latent_past'][:, -1:])
                    for key in ('raw_last', 'raw_forecast', 'target', 'observed', 'origins', 'mean', 'std', 'history_observed'):
                        same(tag + ':' + split + ':bridge:' + key, values[key], original[key])
                    same(tag + ':' + split + ':rawlast_history_last', values['raw_last'], original['history'][:, -1])
                    for suffix, key in [('target', 'target'), ('observed', 'observed'), ('origins', 'origins'), ('prediction', 'original_prediction')]:
                        same(tag + ':' + split + ':stored_MAIN:' + suffix, main[split + '_' + suffix], values[key])
                    same(tag + ':' + split + ':stored_RAW', raw[split + '_prediction'], values['raw_forecast'])
                    for container in (raw, main):
                        same(tag + ':' + split + ':stored_stats_mean', container['mean'], values['mean'])
                        same(tag + ':' + split + ':stored_stats_std', container['std'], values['std'])
                    close(tag + ':' + split + ':source_GPU_vs_stored_MAIN', values['source_gpu_prediction'], values['original_prediction'])
                    features, source, factored, outgoing_original = cpu_decoder_features(values, decoder)
                    reference = checked_arrays(d2folder / ('REFERENCE_' + split + '.npz'), art['REFERENCE_' + split + '.npz'])
                    same(tag + ':' + split + ':frozen_hidden_feature_recompute', reference['features'], features)
                    for key, expected in [('ORIGINAL_D', values['original_prediction']), ('SOURCE_D_FP32', source),
                                          ('FACTORED_ORIGINAL_D', factored), ('RAW', values['raw_forecast'])]:
                        same(tag + ':' + split + ':reference:' + key, reference[key], expected)
                        compare_scores(tag + ':' + split + ':reference_scores:' + key,
                                       scores(expected, values), d2['reference_comparisons'][split]['scores'][key])
                    close(tag + ':' + split + ':CPU_source_D_vs_MAIN', source, values['original_prediction'])
                    close(tag + ':' + split + ':CPU_factored_D_vs_source', factored, source)
                    inputs[split], feature_arrays[split] = values, features
                    baselines[split] = {key: scores(expected, values) for key, expected in
                                       [('MAIN', values['original_prediction']), ('RAW', values['raw_forecast'])]}
                fit = inputs['fit']
                objectives = {'Y': fit['target'], 'RAW': fit['raw_forecast'], 'MIXED': .5 * fit['target'] + .5 * fit['raw_forecast']}
                refits = {}
                check(tag + ':all_objectives', set(d2['refits']) == {'Y', 'RAW', 'MIXED'})
                check(tag + ':no_intercept', d2['intercept'] is False and d2['strict_K_only_compression'] is False)
                for objective, target in objectives.items():
                    declared = d2['refits'][objective]
                    check(tag + ':' + objective + ':alpha_grid', [row['alpha'] for row in declared['alpha_grid']] == list(ALPHAS))
                    heads, tune_mse, tune_scores = [], [], []
                    for index, alpha in enumerate(ALPHAS):
                        filename = f'{objective}_alpha_{alpha:g}_HEAD.npz'
                        stored = checked_arrays(d2folder / filename, art[filename])
                        weight, scaled, rms, counts = fit_head(feature_arrays['fit'], target, fit['raw_last'], fit['observed'], alpha, outgoing_original)
                        same(tag + ':' + filename + ':FIT_RMS_only', stored['feature_rms'], rms)
                        same(tag + ':' + filename + ':FIT_observed_counts', stored['fit_observed_counts'], counts)
                        same(tag + ':' + filename + ':no_intercept_weight', stored['outgoing_weight'], weight)
                        same(tag + ':' + filename + ':masked_solution_scaled', stored['scaled_weight'], scaled)
                        check(tag + ':' + filename + ':alpha', float(stored['alpha']) == alpha)
                        prediction = inputs['tune']['raw_last'][:, None] + feature_arrays['tune'] @ weight
                        current_scores = scores(prediction, inputs['tune'])
                        compare_scores(tag + ':' + objective + ':alpha_' + str(alpha) + ':TUNE', current_scores,
                                       declared['alpha_grid'][index]['tune_scores'])
                        check(tag + ':' + objective + ':TUNE_selection_metric_' + str(alpha),
                              np.isclose(current_scores['normalized']['mse'], declared['alpha_grid'][index]['tune_mse'], atol=1e-11, rtol=1e-11))
                        heads.append(weight)
                        tune_mse.append(current_scores['normalized']['mse'])
                        tune_scores.append(current_scores)
                        limit.check()
                    selected = min(range(len(ALPHAS)), key=lambda index: tune_mse[index])
                    check(tag + ':' + objective + ':earliest_TUNE_selection', selected == declared['selected_index']
                          and ALPHAS[selected] == declared['selected_alpha'])
                    selected_weight = heads[selected]
                    selected_decoder = checked_arrays(d2folder / (objective + '_SELECTED_DECODER.npz'), art[objective + '_SELECTED_DECODER.npz'])
                    for key in DECODER_KEYS:
                        expected = selected_weight[:d2['K']].T if key == 'codec.decode.weight' else (
                            selected_weight[d2['K']:].T if key == 'residual.2.weight' else decoder[key])
                        same(tag + ':' + objective + ':selected_hidden_frozen:' + key, selected_decoder[key], expected)
                    split_scores = {}
                    for split, values in inputs.items():
                        prediction = values['raw_last'][:, None] + feature_arrays[split] @ selected_weight
                        name = objective + '_SELECTED_' + split + '.npz'
                        stored = checked_arrays(d2folder / name, art[name])
                        same(tag + ':' + name + ':selected_prediction', stored['prediction'], prediction)
                        same(tag + ':' + name + ':origins', stored['origins'], values['origins'])
                        split_scores[split] = scores(prediction, values)
                        compare_scores(tag + ':' + objective + ':selected:' + split, split_scores[split], declared['splits'][split])
                    refits[objective] = {'selected_alpha': ALPHAS[selected], 'selected_index': selected,
                                         'all_TUNE_normalized_mse': tune_mse, 'splits': split_scores}
                check(tag + ':CP_SHA_after', sha(cp_path) == before)
                check(tag + ':CP_state_hash_after_CPU_audit', state_hash(state) == parent['MAIN_checkpoint']['selected_codec_hash_old_algorithm'])
                result['conditions'].append({'dataset': dataset, 'seed': seed, 'tag': tag,
                    'producer': {'path': str((folder / 'RESULT.json').relative_to(ROOT)), 'sha256': sha(folder / 'RESULT.json')},
                    'D2_RESULT_sha256': sha(d2folder / 'RESULT.json'), 'baselines': baselines, 'refits': refits,
                    'original_MAIN_checkpoint': {'path': str(cp_path.relative_to(ROOT)), 'sha256_before': before, 'sha256_after': sha(cp_path)}})
                del state, checkpoint
                limit.check()
        result['GPU_initialized'] = torch.cuda.is_initialized()
        check('CPU_only_no_CUDA_initialized', not result['GPU_initialized'])
        result['coverage'] = {'conditions': len(result['conditions']), 'objectives': 12, 'alpha_TUNE_refits': 84,
                               'selected_split_outputs': 36, 'scientific_NPZ_reads': len(result['artifacts'])}
        result['status'] = 'PASS_WITHOUT_INDEPENDENT_F_FORWARD_REPLAY' if result['failed_count'] == 0 else 'FAIL_PRESERVED'
    except Exception as error:
        result.update(status='FAILED_PRESERVED', failure={'type': type(error).__name__, 'message': str(error)})
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
        lines = ['# D2 실제 디코더 재적합의 독립 CPU 검산', '',
                 f"상태: {result['status']}; 검사 {result['checks']}개, 실패 {result['failed_count']}개.", '',
                 '원 MAIN CP의 파일·codec state hash와 SOURCE_DECODER의 실제 hidden·출력 가중치를 대조했다. absolute latent·현재 last·정규화 통계·target/mask/origin의 원 bridge·MAIN·RAW 일치를 검사했다.',
                 '동결 GELU feature와 원/factored D, FIT-only RMS·동일 observed mask·절편 없는 ridge의 모든 일곱 alpha·TUNE 실제 Y MSE 선택·선택 head의 전 split 예측을 CPU에서 다시 계산했다.',
                 '원 F를 새로 실행하지 않았다. 저장 actual F 출력과 producer 전후 hash 기록을 확인한 검산이며 독립 holdout이 아니다. 다른 단위를 가진 원단위 채널을 합쳐 우수성을 주장하지 않는다.', '',
                 '```text', 'dataset              seed    method     alpha       MSE           MAE']
        for condition in result['conditions']:
            dataset, seed = condition['dataset'], condition['seed']
            split = 'val' if dataset == 'robin' else 'eval'
            for method in ('RAW', 'MAIN'):
                metric = condition['baselines'][split][method]['normalized']
                lines.append(f"{dataset:<21}{seed:<8}{method:<11}{'-':<12}{metric['mse']:<14.9f}{metric['mae']:.9f}")
            for objective, row in condition['refits'].items():
                metric = row['splits'][split]['normalized']
                lines.append(f"{dataset:<21}{seed:<8}{objective:<11}{row['selected_alpha']:<12g}{metric['mse']:<14.9f}{metric['mae']:.9f}")
        lines.extend(['```', '', '최저 VAL/EVAL 방법을 사후 선택하지 않았다. Y/RAW/MIXED 각각의 사전 TUNE 선택 결과를 모두 보존하며 원 head로 fallback하여 개선이라고 쓰지 않는다.', '',
                      '[원수치·검사·출처](D2_ARRAY_AUDIT_v1.json) · [감사 코드](audit_D2_actual_v1.py)', ''])
        with MD.open('x', encoding='utf-8') as handle:
            handle.write('\n'.join(lines))
    print(json.dumps({'status': result['status'], 'checks': result['checks'], 'failed_count': result['failed_count'],
                     'failure': result.get('failure'), 'output': str(OUT)}, ensure_ascii=False))
    return 0 if result['status'].startswith('PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
