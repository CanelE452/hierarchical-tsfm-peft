"""CPU structural audit of the selected query projection, using synthetic inputs.

Only saved case manifests/checkpoints and source metadata are read. No dataset
loader, Bolt inference, EVAL target, or training is executed.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import psutil
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METHODS = ('B_GUMBEL', 'A_BIHASH_GUMBEL')
SEEDS = (92601, 92602, 92603)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''): digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def artifact(path):
    return {'path': Path(path).resolve().relative_to(ROOT).as_posix(), 'sha256': sha(path), 'bytes': Path(path).stat().st_size}


def snapshot_tag(result_folder):
    return result_folder.removeprefix('run_')


def compose(projection, projection_bias, hash_weight, hash_bias, method):
    multiplicity = 2 if method == 'A_BIHASH_GUMBEL' else 1
    return hash_weight@projection, hash_weight@(multiplicity*projection_bias)+hash_bias


def rank_summary(value):
    matrix = np.asarray(value, np.float64)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    tolerance = float(max(matrix.shape)*np.finfo(np.float64).eps*singular_values[0]) if len(singular_values) else 0.
    return {'shape': list(matrix.shape), 'singular_values': singular_values.tolist(),
            'numerical_rank': int((singular_values > tolerance).sum()), 'rank_tolerance': tolerance,
            'rank_upper_bound_from_K': int(matrix.shape[1]), 'matrix_storage_dtype': 'float64 composition of actual saved FP32 weights'}


def k1_boundary_summary(effective, bias):
    if effective.shape[1] != 1: raise ValueError('K1_ONLY_THRESHOLD_BOUND')
    slope = effective[:, 0]; active = slope != 0
    thresholds = -bias[active]/slope[active]
    return {'nonconstant_bit_count': int(active.sum()), 'constant_bit_count': int((~active).sum()),
            'thresholds': thresholds.tolist(), 'distinct_finite_thresholds': int(len(np.unique(thresholds[np.isfinite(thresholds)]))),
            'exact_arithmetic_max_codes': int(active.sum())+1,
            'bound_scope': 'Deterministic sign of an affine scalar input. Every nonconstant bit changes once; coincident opposite-slope ties may add a boundary code but still consume at least two bit changes.',
            'retrieval_scope': 'At most this many query code/score patterns only with fixed trained memory keys and a fixed eligible-memory mask.',
            'applies_to_Gumbel_noise_training': False,
            'finite_precision_note': 'FP32 two-linear roundoff near thresholds is measured separately. The bound is an exact-arithmetic structural statement.'}


def query_parity(state, method, seed, samples):
    names = ('student_projection.weight', 'student_projection.bias', 'student_hash.weight', 'student_hash.bias')
    projection, projection_bias, hash_weight, hash_bias = [state[name].detach().cpu().float() for name in names]
    if projection.shape[0] != 32 or hash_weight.shape != (32, 32): raise ValueError('UNEXPECTED_QUERY_MAP_SHAPE')
    p64, pb64, w64, b64 = [value.double() for value in (projection, projection_bias, hash_weight, hash_bias)]
    effective, bias = compose(p64, pb64, w64, b64, method)
    k = projection.shape[1]
    generator_seed = seed*1009+13071
    rng = np.random.default_rng(generator_seed)
    scalar_extra = None
    if k == 1:
        boundary = k1_boundary_summary(effective.numpy(), bias.numpy())
        thresholds = np.asarray(boundary['thresholds'])
        finite = np.unique(thresholds[np.isfinite(thresholds) & (np.abs(thresholds) <= 10000)])
        intervals = [(finite[0]-max(1., abs(finite[0])))] if len(finite) else []
        intervals += ((finite[:-1]+finite[1:])/2).tolist()
        if len(finite): intervals.append(finite[-1]+max(1., abs(finite[-1])))
        neighborhoods = [t+offset*max(1., abs(t)) for t in finite for offset in (-1e-6, -1e-8, 0., 1e-8, 1e-6)]
        scalar_extra = np.asarray(np.linspace(-20, 20, 257).tolist()+intervals+neighborhoods, np.float32)[:, None]
    else:
        boundary = None
    if method == 'A_BIHASH_GUMBEL':
        history = rng.normal(0., 2., (samples, 512, k)).astype(np.float32)
        if scalar_extra is not None:
            constant_history = np.broadcast_to(scalar_extra[:, None]/2, (len(scalar_extra), 512, 1)).copy()
            history = np.concatenate((history, constant_history), axis=0)
        x32 = torch.from_numpy(history)
        x64 = x32.double()
        projected32 = F.linear(x32, projection, projection_bias)
        projected64 = F.linear(x64, p64, pb64)
        direct32 = F.linear(projected32[:, -1]+projected32.mean(1), hash_weight, hash_bias)
        direct64 = F.linear(projected64[:, -1]+projected64.mean(1), w64, b64)
        summary64 = x64[:, -1]+x64.mean(1)
        summary32 = x32[:, -1]+x32.mean(1)
        synthetic_input = history
        input_note = 'Synthetic full N x L512 x K history; actual A formula last projected row plus mean of projected rows. K1 boundary additions use constant history=q/2.'
    else:
        latent = rng.normal(0., 2., (samples, k)).astype(np.float32)
        if scalar_extra is not None: latent = np.concatenate((latent, scalar_extra), axis=0)
        x32 = torch.from_numpy(latent); summary32 = x32; summary64 = x32.double()
        direct32 = F.linear(F.linear(x32, projection, projection_bias), hash_weight, hash_bias)
        direct64 = F.linear(F.linear(summary64, p64, pb64), w64, b64)
        synthetic_input = latent
        input_note = 'Synthetic N x K past latent row; no actual encoder/raw data or future prediction was evaluated.'
    collapsed64 = F.linear(summary64, effective, bias)
    difference32 = (direct32.double()-collapsed64).abs()
    difference64 = (direct64-collapsed64).abs()
    code32 = torch.where(direct32 >= 0, 1, -1).to(torch.int8)
    code64 = torch.where(collapsed64 >= 0, 1, -1).to(torch.int8)
    disagreements = code32 != code64
    summary = {'synthetic_input_shape': list(synthetic_input.shape), 'synthetic_generator': 'numpy default_rng normal(0,2) cast float32',
               'synthetic_generator_seed': generator_seed, 'input_note': input_note,
               'sample_count': int(len(synthetic_input)), 'all_samples_are_synthetic': True,
               'FP32_two_linear_vs_FP64_composed_max_abs': float(difference32.max()),
               'FP32_two_linear_vs_FP64_composed_mean_abs': float(difference32.mean()),
               'FP64_two_linear_vs_FP64_composed_max_abs': float(difference64.max()),
               'FP64_reordering_atol': 1e-11,
               'FP64_algebraic_parity_pass': bool(difference64.max() <= 1e-11),
               'synthetic_FP32_unique_hard_query_codes': int(len(torch.unique(code32, dim=0))),
               'synthetic_FP64_composed_unique_hard_query_codes': int(len(torch.unique(code64, dim=0))),
               'FP32_vs_FP64_disagreeing_bits': int(disagreements.sum()),
               'FP32_vs_FP64_disagreeing_samples': int(disagreements.any(1).sum()),
               'max_FP64_abs_logit_among_disagreements': float(collapsed64[disagreements].abs().max()) if disagreements.any() else None,
               'not_actual_dataset_code_coverage': True, 'k1_exact_arithmetic_boundary': boundary}
    arrays = {'projection_weight_FP32': projection.numpy(), 'projection_bias_FP32': projection_bias.numpy(),
              'student_hash_weight_FP32': hash_weight.numpy(), 'student_hash_bias_FP32': hash_bias.numpy(),
              'effective_weight_FP64': effective.numpy(), 'effective_bias_FP64': bias.numpy(),
              'synthetic_input_FP32': synthetic_input, 'synthetic_summary_FP32': summary32.numpy(),
              'synthetic_summary_FP64': summary64.numpy(), 'query_logits_two_linears_FP32': direct32.numpy(),
              'query_logits_two_linears_FP64': direct64.numpy(), 'query_logits_composed_FP64': collapsed64.numpy(),
              'query_sign_FP32': code32.numpy(), 'query_sign_composed_FP64': code64.numpy()}
    return summary, arrays, {'projection': rank_summary(p64.numpy()), 'student_hash': rank_summary(w64.numpy()),
                             'effective_query_map': rank_summary(effective.numpy())}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--tag', default='selected_maps_v1')
    parser.add_argument('--samples', type=int, default=256)
    args = parser.parse_args()
    if args.samples <= 0 or args.samples > 512: raise ValueError('BOUNDED_SYNTHETIC_SAMPLE_COUNT_1_TO_512')
    if not args.tag.replace('_', '').isalnum() or '/' in args.tag or '\\' in args.tag: raise ValueError('INVALID_FRESH_DIAGNOSTIC_TAG')
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    started = time.perf_counter(); process = psutil.Process(); peak_rss = process.memory_info().rss
    folder = HERE / 'decision_checks_v1/hash_query_capacity_v1' / args.tag; folder.mkdir(parents=True, exist_ok=False)
    saved_script = folder / 'SOURCE_AT_RUN.py'
    with saved_script.open('xb') as handle: handle.write(Path(__file__).read_bytes())
    lock_path = HERE / 'METHOD_LOCK.json'; lock = read(lock_path)
    scientific_sources = lock['source_hashes']
    if any(sha(ROOT / path) != value for path, value in scientific_sources.items()): raise ValueError('SCIENTIFIC_SOURCE_CHANGED')
    output = {'status': 'INCOMPLETE', 'conditions': [], 'missing_cases': [], 'failed_cases': [],
              'method_lock': artifact(lock_path), 'diagnostic_script': artifact(saved_script),
              'current_script_at_start': artifact(__file__), 'scientific_source_hashes': scientific_sources,
              'expected_required_DEV_and_SML_conditions': 18, 'optional_Household_conditions': 6,
              'raw_dataset_files_opened': 0, 'data_loader_used': False, 'EVAL_target_opened': False, 'GPU_executed': False,
              'whole_training_campaign_complete_claimed': False, 'actual_dataset_query_code_coverage_measured': False}
    runs = (('jena', 'run_dev_v1', True), ('robin', 'run_dev_v1', True), ('uci274_sml2010', 'run_fresh_v1', True),
            ('uci235_household_hourly', 'run_household_recovery_v1', False))
    with torch.no_grad():
        for dataset, run_tag, required in runs:
            snapshot = ROOT / '.cache/bihash_future_prediction_20261013/source_snapshots' / snapshot_tag(run_tag) / 'MANIFEST.json'
            for seed in SEEDS:
                for method in METHODS:
                    weight = lock['weights'][method]; label = f'{method}_w{weight:g}_s{seed}'
                    path = HERE / run_tag / dataset / (label + '.json')
                    if not path.is_file():
                        output['missing_cases'].append({'dataset': dataset, 'seed': seed, 'method': method, 'required': required, 'expected_path': str(path.relative_to(ROOT))})
                        continue
                    try:
                        case = read(path)
                        if (case['dataset'], case['seed'], case['method'], case['weight']) != (dataset, seed, method, weight): raise ValueError('CASE_IDENTITY')
                        if case['training']['updates'] != 600 or len(case['training']['curve']) != 600: raise ValueError('CASE_NOT_COMPLETED_600')
                        manifest = read(snapshot)
                        if manifest['sha256'] != scientific_sources: raise ValueError('CASE_SCIENTIFIC_SNAPSHOT_MISMATCH')
                        checkpoint = case['training']['selected_checkpoint']; cp_path = ROOT / checkpoint['path']
                        if sha(cp_path) != checkpoint['sha256']: raise ValueError('ACTUAL_SELECTED_CHECKPOINT_SHA_MISMATCH')
                        stored = torch.load(cp_path, map_location='cpu', weights_only=True)
                        if (stored['method'], stored['seed'], stored['update']) != (method, seed, case['training']['selected_update']): raise ValueError('CHECKPOINT_IDENTITY')
                        summary, arrays, maps = query_parity(stored['model'], method, seed, args.samples)
                        if maps['effective_query_map']['shape'] != [32, case['K']]: raise ValueError('EFFECTIVE_QUERY_SHAPE_OR_K')
                        filename = dataset + '_' + label + '.npz'; npz_path = folder / filename
                        with npz_path.open('xb') as handle: np.savez_compressed(handle, **arrays)
                        record = {'dataset': dataset, 'seed': seed, 'method': method, 'weight': weight, 'K': case['K'],
                                  'selected_update': case['training']['selected_update'], 'source_case': artifact(path),
                                  'source_checkpoint': checkpoint, 'source_snapshot': artifact(snapshot), 'maps': maps,
                                  'synthetic_parity': summary, 'actual_arrays': artifact(npz_path),
                                  'query_role': 'Historical row retrieval code' if method == 'B_GUMBEL' else 'Training-only auxiliary code; the heads are removed from deployment'}
                        output['conditions'].append(record)
                        del stored, arrays
                        peak_rss = max(peak_rss, process.memory_info().rss)
                        if peak_rss > 2*1024**3 or time.perf_counter()-started > 90: raise RuntimeError('CPU_DIAGNOSTIC_RESOURCE_GUARD')
                    except Exception as exc:
                        output['failed_cases'].append({'dataset': dataset, 'seed': seed, 'method': method, 'error': str(exc), 'source_case_path': str(path.relative_to(ROOT))})
    required_missing = [c for c in output['missing_cases'] if c['required']]
    output['required_conditions_read'] = sum(c['dataset'] != 'uci235_household_hourly' for c in output['conditions'])
    output['optional_Household_conditions_read'] = sum(c['dataset'] == 'uci235_household_hourly' for c in output['conditions'])
    output['scientific_sources_unchanged'] = all(sha(ROOT / path) == value for path, value in scientific_sources.items())
    output['actual_case_and_checkpoint_sources_unchanged'] = all(sha(ROOT / c['source_case']['path']) == c['source_case']['sha256'] and sha(ROOT / c['source_checkpoint']['path']) == c['source_checkpoint']['sha256'] for c in output['conditions'])
    output['status'] = 'COMPLETE_AVAILABLE_CASE_DIAGNOSTIC' if not required_missing and not output['failed_cases'] and output['scientific_sources_unchanged'] and output['actual_case_and_checkpoint_sources_unchanged'] else 'PARTIAL_OR_FAILED_DIAGNOSTIC_PRESERVED'
    output['resource_receipt'] = {'gpu': False, 'main_wall_seconds': time.perf_counter()-started,
                                'process_wall_seconds_including_imports': time.time()-process.create_time(), 'sampled_peak_rss_bytes': peak_rss,
                                'torch_threads': torch.get_num_threads(), 'cuda_initialized': torch.cuda.is_initialized()}
    output['interpretation_limits'] = [
        'R32 after a linear K projection does not increase the affine input information dimension; effective WsP rank is at most K.',
        'For A, the exact affine input is z_last+mean(z), with effective bias 2Ws*b_projection+b_hash. A hash heads are training-only.',
        'K1 gives at most 33 deterministic query codes in exact arithmetic for 32 bits, and at most that many retrieval score patterns only for a fixed memory eligibility mask.',
        'Stochastic Gumbel training has independent noise per sample/bit; the deterministic K1 bound does not apply.',
        'For general K, no whole-space hyperplane code-count claim is made; only affine composition and rank<=K are established.',
        'Synthetic sample counts and code diversity are not observed dataset coverage or a generalization result.',
        'The full forecasting model also uses causal temporal encoding, continuous S, continuous fusion, raw-last anchoring and variable memory eligibility; this diagnostic does not establish full model capacity collapse or a cause of forecast failure.']
    with (folder / 'RESULT.json').open('x', encoding='utf-8') as handle: json.dump(output, handle, ensure_ascii=False, indent=2, allow_nan=False)
    lines = ['# Query 해시 정보 차원의 CPU 구조 진단\n\n', f'상태: {output["status"]}. 필수 DEV/SML 조건 {output["required_conditions_read"]}/18개, 실제 존재하는 Household 조건 {output["optional_Household_conditions_read"]}/6개를 읽었다. 전체 학습 완료 보고가 아니다.\n\n',
             '`student_projection(K→32)`와 `student_hash(32→32)` 사이에는 비선형 함수가 없다. B의 deterministic query logit은 `Ws P z + Ws bP + bs`, A의 auxiliary logit은 `Ws P (z_last + mean(z)) + 2 Ws bP + bs`다. 저장된 실제 두 weight를 합성하여 shape/rank/singular values를 기록했다.\n\n',
             'R32로 펼쳐도 affine 입력 정보 차원은 K를 넘지 않는다. K1의 32개 deterministic bit는 각 affine threshold에서 한 번만 바뀌므로 정확 산술의 query code 수는 최대 33개다. 이는 memory eligibility mask를 고정한 검색 점수 경로의 한계이며 Gumbel noise가 있는 학습에는 적용되지 않는다. 일반 K의 전체 code 수를 주장하지 않는다.\n\n',
             '아래 logits·code 수는 모두 synthetic latent를 사용했다. 실제 데이터 query code 분포와 분리해야 한다. FP32 두 Linear와 FP64 합성식은 임계값 근처에서 sign이 다를 수 있다.\n\n',
             '```text\ndataset                    method               seed   K  rank  FP32↔FP64 max gap  FP64 reorder gap  synthetic codes32/64\n']
    for condition in output['conditions']:
        value = condition['synthetic_parity']; rank = condition['maps']['effective_query_map']['numerical_rank']
        lines.append(f'{condition["dataset"]:26} {condition["method"]:20} {condition["seed"]}  {condition["K"]}  {rank:4}  {value["FP32_two_linear_vs_FP64_composed_max_abs"]:18.8g}  {value["FP64_two_linear_vs_FP64_composed_max_abs"]:16.8g}  {value["synthetic_FP32_unique_hard_query_codes"]}/{value["synthetic_FP64_composed_unique_hard_query_codes"]}\n')
    lines += ['```\n\n', '전체 예측은 시간 인코더, 연속값 S/fusion, raw-last anchor, 시점에 따라 바뀌는 memory mask도 사용한다. 해시 query의 affine 경계가 전체 예측 능력의 붕괴나 성능 실패 원인이라는 결론은 아니다. A 해시 경로는 배포 때 제거된다. 실제 past-only 데이터에서 code 사용률과 조회 가중치의 다양성은 후속 검산으로 남아 있다.\n']
    with (folder / 'REPORT_KO.md').open('x', encoding='utf-8') as handle: handle.write(''.join(lines))
    print(json.dumps({'status': output['status'], 'result': str(folder / 'RESULT.json'), 'conditions': len(output['conditions']), 'process_seconds': output['resource_receipt']['process_wall_seconds_including_imports']}, ensure_ascii=True), flush=True)
    if output['status'] != 'COMPLETE_AVAILABLE_CASE_DIAGNOSTIC': raise SystemExit(2)


if __name__ == '__main__': main()
