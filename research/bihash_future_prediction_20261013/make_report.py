"""Build an evidence-bound Korean report from terminal experiment artifacts.

No runner, data loader, torch model, or protected TEST is imported. Real report
generation is a post-evaluation operation; preparation can use --self-test.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import re
import tempfile
import time
from collections import defaultdict

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / '.cache' / HERE.name
DATASETS = ('jena', 'robin', 'uci235_household_hourly', 'uci274_sml2010')
SEEDS = (92601, 92602, 92603)
METHODS = ('NO_ALIGN', 'B_COSINE', 'B_TANH', 'B_GUMBEL', 'B_ST', 'B_FIXED_GUMBEL',
           'B_UNIFORM', 'A_CONTINUOUS', 'A_BIHASH_GUMBEL', 'A_BIHASH_SHUFFLE',
           'A_BIHASH_DETACH_E', 'A_BIHASH_FIXED_MAPS')
PROBES = ('NO_ALIGN', 'B_GUMBEL', 'B_COSINE', 'B_UNIFORM', 'A_BIHASH_GUMBEL', 'A_CONTINUOUS')
TITLES = {'jena': 'Jena · 반복 DEV_VAL', 'robin': 'Robin · 반복 DEV_VAL',
          'uci235_household_hourly': 'Household · 신규 EVAL', 'uci274_sml2010': 'SML2010 · 신규 EVAL'}
PALETTE = ('#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#333333')
CPU_RIDGE_IMPLEMENTATION = 'cpu_numpy_fp32_ridge_v1'
RIDGE_ROUTES = {'E_RAW_RIDGE', 'E_ENRICHED_RIDGE', 'RAW_SELF33', 'RAW_JOINT_RIDGE', 'RAW_BOLT_RAW_R'}
DECISION_FILES = {
    'decoder_bias': 'decision_checks_v1/decoder_bias_v1/reproduction_v1/RESULT.json',
    'quantile_projection': 'decision_checks_v1/quantile_projection_example_v1/RESULT.json',
    'preservation_current': 'decision_checks_v1/preservation_current_v1/RESULT.json',
    'preservation_final': 'decision_checks_v1/preservation_final_v1/RESULT.json',
    'hash_query_capacity': 'decision_checks_v1/hash_query_capacity_v1/selected_maps_v2/RESULT.json',
    'hash_query_capacity_failed_attempt': 'decision_checks_v1/hash_query_capacity_v1/selected_maps_v1/RESULT.json',
    'train_deploy_gap': 'train_deploy_gap_v1/actual_selected_v1/RESULT.json',
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    path = Path(path).resolve()
    if path.name.lower() in ('test.npz', 'jena_test.npz'):
        raise PermissionError('PROTECTED_TEST_NOT_AUTHORIZED')
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding='utf-8-sig'))


def fresh_text(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(value)


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'


def resolve_artifact(artifact):
    path = Path(artifact['path'])
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if not (path.is_relative_to(HERE) or path.is_relative_to(CACHE)):
        raise ValueError('REPORT_ARTIFACT_OUTSIDE_PRIMARY_STUDY: ' + str(path))
    if path.suffix.lower() != '.npz':
        raise ValueError('REPORT_FORECAST_ARTIFACT_NOT_NPZ')
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha(path) != artifact['sha256']:
        raise ValueError('REPORT_FORECAST_ARTIFACT_SHA_MISMATCH: ' + str(path))
    return path


def macro(prediction, target, observed):
    if prediction.ndim != 3 or prediction.shape != target.shape or prediction.shape != observed.shape:
        raise ValueError('REPORT_FORECAST_SHAPE_MISMATCH')
    if not np.isfinite(prediction).all() or not np.isfinite(target[observed]).all():
        raise ValueError('REPORT_NONFINITE_OBSERVED_VALUE')
    counts, mse, mae = [], [], []
    for channel in range(prediction.shape[-1]):
        mask = observed[..., channel]
        error = prediction[..., channel][mask].astype(np.float64) - target[..., channel][mask].astype(np.float64)
        counts.append(int(error.size))
        mse.append(float(np.mean(error * error)) if error.size else None)
        mae.append(float(np.mean(np.abs(error))) if error.size else None)
    valid = [i for i, count in enumerate(counts) if count]
    if not valid:
        raise ValueError('REPORT_NO_OBSERVED_TARGET')
    return {'mse': float(np.mean([mse[i] for i in valid])),
            'mae': float(np.mean([mae[i] for i in valid])),
            'per_channel_mse': mse, 'per_channel_mae': mae, 'observed_counts': counts}


class Predictions:
    def __init__(self):
        self.cache = {}
        self.artifacts = {}
        self.max_abs_vs_recorded = 0.

    def score(self, artifact, split, claimed):
        path = resolve_artifact(artifact)
        key = (str(path), split)
        if key not in self.cache:
            with np.load(path, allow_pickle=False) as arrays:
                needed = ('prediction', 'target', 'observed', 'origins')
                if any(split + '_' + name not in arrays.files for name in needed):
                    raise ValueError('REPORT_MISSING_SAVED_FORECAST_ARRAY')
                p, y = arrays[split + '_prediction'], arrays[split + '_target']
                mask = arrays[split + '_observed'].astype(bool)
                origins = arrays[split + '_origins']
                mean, std = arrays['mean'], arrays['std']
                if origins.shape != (len(p),) or len(set(origins.tolist())) != len(origins):
                    raise ValueError('REPORT_INVALID_ORIGINS')
                if mean.shape != (p.shape[-1],) or std.shape != mean.shape or np.any(std <= 0):
                    raise ValueError('REPORT_INVALID_FIT_STATISTICS')
                normalized = macro(p, y, mask)
                physical = macro(p * std + mean, y * std + mean, mask)
                result = {'norm_mse': normalized['mse'], 'norm_mae': normalized['mae'],
                          'physical_mse': physical['mse'], 'physical_mae': physical['mae'],
                          'windows': len(p), 'channels': p.shape[-1], 'horizon': p.shape[1],
                          'observed_targets': int(mask.sum()),
                          'per_channel_norm_mse': normalized['per_channel_mse'],
                          'per_channel_norm_mae': normalized['per_channel_mae'],
                          'observed_counts': normalized['observed_counts'],
                          'origins_sha256': hashlib.sha256(np.ascontiguousarray(origins).tobytes()).hexdigest(),
                          'prediction_path': path.relative_to(ROOT).as_posix(), 'prediction_sha256': artifact['sha256']}
            self.cache[key] = result
            self.artifacts[str(path)] = artifact['sha256']
        result = self.cache[key]
        if claimed:
            pairs = ((result['norm_mse'], claimed['normalized']['mse']),
                     (result['norm_mae'], claimed['normalized']['mae']),
                     (result['physical_mse'], claimed['mse']), (result['physical_mae'], claimed['mae']))
            for actual, recorded in pairs:
                self.max_abs_vs_recorded = max(self.max_abs_vs_recorded, abs(actual - recorded))
                if not math.isclose(actual, recorded, rel_tol=2e-6, abs_tol=1e-7):
                    raise ValueError('REPORT_RESCORE_DISAGREES_WITH_RECORDED_SCORE')
        return result.copy()


def read_terminal(path, stage):
    value = read(path)
    if value.get('stage') not in (stage, None):
        raise ValueError('REPORT_WRONG_STAGE_INPUT: ' + stage)
    if value.get('status') not in ('COMPLETE', 'FAILED_PRESERVED', 'BUDGET_HELD_PRESERVED', 'NOT_RUN', 'BLOCKED'):
        raise ValueError('REPORT_REQUIRES_ACTUAL_TERMINAL_STAGE_RECORD: ' + stage)
    return value


def method_lock(dev, fresh):
    references = [v['method_lock'] for v in (dev, fresh) if v.get('method_lock')]
    if not references:
        return None
    if len({r['sha256'] for r in references}) != 1:
        raise ValueError('REPORT_DIFFERENT_METHOD_LOCKS')
    artifact = references[0]
    path = ROOT / artifact['path']
    if sha(path) != artifact['sha256']:
        raise ValueError('REPORT_METHOD_LOCK_SHA_MISMATCH')
    lock = read(path)
    if not lock.get('locked'):
        raise ValueError('REPORT_METHOD_LOCK_NOT_LOCKED')
    return lock


def rows_from_results(dev, fresh, probes, evaluation, predictions):
    lock = method_lock(dev, fresh)
    cases = dev.get('cases', []) + fresh.get('cases', [])
    identities = [(c['dataset'], c['seed'], c['method'], c['weight']) for c in cases]
    if len(identities) != len(set(identities)):
        raise ValueError('REPORT_DUPLICATE_CORE_CASE')
    by_label = {(c['dataset'], c['label']): c for c in cases}
    core, checkpoint, curve, diagnostics, pretrain, readout_rows, controls = [], [], [], [], [], [], []
    for c in cases:
        base = {k: c[k] for k in ('dataset', 'seed', 'method', 'weight', 'label', 'K', 'L', 'H')}
        base.update(registered_parameters=c.get('metadata', {}).get('registered_parameters'),
                    training_trainable_parameters=c.get('initial_metadata', {}).get('trainable_parameters'),
                    deployment_active_parameters=c.get('metadata', {}).get('deployment_active_structural_parameters'),
                    deployment_memory_bytes=c.get('metadata', {}).get('deployment_memory_bytes'))
        chosen_weight = lock is not None and c['weight'] == lock['weights'][c['method']]
        base.update(chosen_weight=chosen_weight, selected_update=c['training']['selected_update'])
        for split, score in c['scores'].items():
            core.append({**base, 'split': split, 'variant': 'regular',
                         **predictions.score(c['predictions'], split, score)})
        for cp in c['training']['checkpoints']:
            checkpoint.append({**base, 'checkpoint_update': cp['update'],
                               **predictions.score(cp['tune_prediction'], 'tune', cp['tune']),
                               'checkpoint_path': cp['path'], 'checkpoint_sha256': cp['sha256']})
        curve.extend({**base, **step} for step in c['training']['curve'])
        for split, diag in c.get('selected_diagnostics', {}).items():
            diagnostics.append({**base, 'split': split, **diag,
                                'phi_centered_rank_bound': min(32, max(0, diag.get('sample_windows', 0) - 1)),
                                'rank_scope': 'selected diagnostic sample, not complete FIT population'})
        if c.get('evaluation'):
            e = c['evaluation']
            core.append({**base, 'split': e['split'], 'variant': 'regular',
                         **predictions.score(e['prediction'], e['split'], e['scores'])})
    # The post-joint-lock re-inference supersedes the earlier DEV_VAL artifact.
    for e in evaluation.get('core_evaluation', []):
        c = by_label[(e['dataset'], e['source_label'])]
        base = {k: c[k] for k in ('dataset', 'seed', 'method', 'weight', 'label', 'K', 'L', 'H')}
        base.update(registered_parameters=c.get('metadata', {}).get('registered_parameters'),
                    training_trainable_parameters=c.get('initial_metadata', {}).get('trainable_parameters'),
                    deployment_active_parameters=c.get('metadata', {}).get('deployment_active_structural_parameters'),
                    deployment_memory_bytes=c.get('metadata', {}).get('deployment_memory_bytes'))
        base.update(chosen_weight=lock is not None and c['weight'] == lock['weights'][c['method']],
                    selected_update=c['training']['selected_update'])
        core = [r for r in core if not (r['dataset'] == e['dataset'] and r['label'] == e['source_label'] and r['split'] == e['split'])]
        core.extend({**base, 'split': e['split'], 'variant': item['variant'],
                     **predictions.score(item['prediction'], e['split'], item['scores'])} for item in e['entries'])
    for p in dev.get('future_pretrains', []) + fresh.get('future_pretrains', []):
        for split, diag in p['diagnostics'].items():
            pretrain.append({'dataset': p['dataset'], 'seed': p['seed'], 'split': split, 'updates': p['updates'],
                             'rec_mse': diag['delta_reconstruction']['mse'], 'rec_mae': diag['delta_reconstruction']['mae'],
                             'phi_rank': diag['representation_rank'], 'phi_variance_mean': float(np.mean(diag['representation_variance'])),
                             'rank_scope': 'full saved FIT/TUNE pretrain diagnostic',
                             'first_training_loss': p['curve'][0]['loss'], 'last_training_loss': p['curve'][-1]['loss'],
                             'checkpoint_path': p['checkpoint']['path'], 'checkpoint_sha256': p['checkpoint']['sha256']})
    readout_lookup = {}
    for group in probes.get('readouts', []):
        for r in group['readouts']:
            identity = (group['dataset'], group['source_label'], r['method'])
            readout_lookup[identity] = (group, r)
            base = {'dataset': group['dataset'], 'seed': group['seed'], 'source_method': group['source_method'],
                    'source_label': group['source_label'], 'route': r['method'], 'duplicate_of': r.get('duplicate_of'),
                    'additional_fit': r.get('additional_fit', False), 'ridge_alpha': r['selection']['alpha']}
            readout_rows.extend({**base, 'split': split, **predictions.score(r['prediction'], split, score)}
                                for split, score in r['scores'].items())
    for e in evaluation.get('readout_evaluation', []):
        group, original = readout_lookup[(e['dataset'], e['source_label'], e['method'])]
        readout_rows.append({'dataset': e['dataset'], 'seed': e['seed'], 'source_method': group['source_method'],
                             'source_label': e['source_label'], 'route': e['method'], 'split': e['split'],
                             'duplicate_of': e.get('duplicate_of'), 'additional_fit': original.get('additional_fit', False),
                             'ridge_alpha': original['selection']['alpha'],
                             **predictions.score(e['prediction'], e['split'], e['scores'])})
    for group in probes.get('controls', []):
        for c in group['methods']:
            controls.extend({'dataset': group['dataset'], 'method': c['method'], 'split': split,
                             **predictions.score(c['prediction'], split, score)} for split, score in c['scores'].items())
    controls.extend({'dataset': e['dataset'], 'method': e['method'], 'split': e['split'],
                     **predictions.score(e['prediction'], e['split'], e['scores'])} for e in evaluation.get('control_evaluation', []))
    return {'lock': lock, 'cases': cases, 'core': core, 'checkpoints': checkpoint, 'curves': curve,
            'diagnostics': diagnostics, 'pretrain': pretrain, 'readouts': readout_rows, 'controls': controls}


def summary_rows(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    result = []
    for identity, group in grouped.items():
        row = dict(zip(keys, identity))
        row['N'] = len(group)
        row['seed_count'] = len({r.get('seed') for r in group if r.get('seed') is not None})
        for metric in ('norm_mse', 'norm_mae', 'physical_mse', 'physical_mae'):
            value = np.asarray([r[metric] for r in group], np.float64)
            row[metric + '_mean'] = float(value.mean())
            row[metric + '_seed_sd'] = float(value.std(ddof=1)) if len(value) > 1 else None
        row['windows_per_seed'] = sorted({r['windows'] for r in group})
        result.append(row)
    return result


def paired_ratios(rows, reference_method='NO_ALIGN', metric='norm_mse', method_key='method', route=None):
    reference = {(r['dataset'], r['split'], r.get('seed')): r for r in rows
                 if r[method_key] == reference_method and r.get('variant', 'regular') == 'regular'
                 and (route is None or r.get('route') == route)}
    output = []
    for row in rows:
        if row.get('variant', 'regular') != 'regular' or (route is not None and row.get('route') != route):
            continue
        key = (row['dataset'], row['split'], row.get('seed'))
        if key in reference and reference[key][metric] > 0:
            output.append({'dataset': row['dataset'], 'split': row['split'], 'seed': row.get('seed'),
                           'method': row[method_key], 'route': row.get('route'), 'metric': metric,
                           'value': row[metric], 'reference': reference[key][metric],
                           'change_percent': 100 * (row[metric] / reference[key][metric] - 1)})
    return output


def test_rows(rows):
    return [r for r in rows if r['split'] in ('val', 'eval')]


def csv_file(path, rows):
    if not rows:
        fresh_text(path, 'status\nNO_EXECUTED_ROWS\n')
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, allow_nan=False) if isinstance(v, (dict, list, tuple)) else v
                             for k, v in row.items()})


def group_change(rows):
    values = defaultdict(list)
    for row in rows:
        values[(row['dataset'], row['method'])].append(row['change_percent'])
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else None, len(v)) for k, v in values.items()}


def font_and_style():
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((name for name in ('Malgun Gothic', 'Noto Sans KR', 'Noto Serif KR', 'NanumGothic') if name in available), None)
    if chosen is None:
        raise RuntimeError('NO_VERIFIED_KOREAN_FONT')
    matplotlib.rcParams.update({'font.family': chosen, 'font.size': 10, 'axes.labelsize': 10,
                                'axes.titlesize': 12, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                                'axes.unicode_minus': False, 'svg.fonttype': 'path',
                                'axes.spines.top': False, 'axes.spines.right': False,
                                'savefig.facecolor': 'white'})
    import matplotlib.pyplot as plt
    return plt, chosen


def save_figure(plt, fig, folder, name):
    for suffix in ('png', 'svg'):
        path = folder / (name + '.' + suffix)
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return name


def no_data(ax, text='실행된 비교 없음'):
    ax.text(.5, .5, text, ha='center', va='center', transform=ax.transAxes)
    ax.set_axis_off()


def accuracy_cost_points(bundle, cost):
    selected_b = (bundle.get('lock') or {}).get('selected_B_method')
    methods = list(dict.fromkeys(['NO_ALIGN', 'B_GUMBEL'] + ([selected_b] if selected_b else [])
                                + ['RAW_BOLT', 'RAW_SELF33']))
    timings = {(r['dataset'], r['identity'], r['route']): r for r in cost.get('records', []) if r['batch'] == 1}
    core = test_rows([r for r in bundle['core'] if r['chosen_weight'] and r['variant'] == 'regular'])
    controls = test_rows(bundle['controls'])
    points = []
    for dataset in DATASETS:
        for method in methods:
            raw = method in ('RAW_BOLT', 'RAW_SELF33')
            scores = [r for r in (controls if raw else core) if r['dataset'] == dataset and r['method'] == method]
            pairs = [(r, timings[(dataset, method if raw else r['label'], method if raw else 'CORE')])
                     for r in scores if (dataset, method if raw else r['label'], method if raw else 'CORE') in timings]
            if not pairs:
                continue
            values = np.asarray([[1000 * timing['median_seconds_per_window'], score['norm_mse']]
                                 for score, timing in pairs], np.float64)
            means = values.mean(0); sd = values.std(0, ddof=1) if len(pairs) > 1 else (None, None)
            points.append({'dataset': dataset, 'method': method, 'N': len(pairs),
                           'median_ms_mean': float(means[0]), 'norm_mse_mean': float(means[1]),
                           'median_ms_seed_sd': float(sd[0]) if sd[0] is not None else None,
                           'norm_mse_seed_sd': float(sd[1]) if sd[1] is not None else None,
                           'scope': 'one raw control fit' if raw else 'selected regular core seed mean'})
    return points


def make_figures(out, bundle, core_summary, ratio, cost):
    plt, font = font_and_style()
    folder = out / 'figures'
    folder.mkdir()
    figures = []
    changes = group_change(ratio)
    fig, ax = plt.subplots(figsize=(10.8, 8.2))
    matrix = np.full((len(METHODS), len(DATASETS)), np.nan)
    for i, method in enumerate(METHODS):
        for j, ds in enumerate(DATASETS):
            if (ds, method) in changes:
                matrix[i, j] = changes[(ds, method)][0]
    if np.isfinite(matrix).any():
        bound = max(1., float(np.nanmax(np.abs(matrix))))
        image = ax.imshow(np.ma.masked_invalid(matrix), cmap='PuOr_r', vmin=-bound, vmax=bound, aspect='auto')
        ax.set_xticks(range(len(DATASETS)), [TITLES[d].replace(' · ', '\n') for d in DATASETS])
        ax.set_yticks(range(len(METHODS)), METHODS)
        for i, method in enumerate(METHODS):
            for j, ds in enumerate(DATASETS):
                if (ds, method) in changes:
                    mean, sd, n = changes[(ds, method)]
                    label = f'{mean:+.2f}%\nSD {sd:.2f}; N={n}' if sd is not None else f'{mean:+.2f}%\nN={n}'
                    ax.text(j, i, label, ha='center', va='center', fontsize=8,
                            color='white' if abs(mean) > .65 * bound else 'black')
        fig.colorbar(image, ax=ax, label='동일 seed NO_ALIGN 대비 정규화 MSE 변화 (%) · 낮을수록 좋음')
    else:
        no_data(ax)
    ax.set_title('예측 오차 변화 · CP/가중치는 TUNE에서 선택\nSD는 seed 간 표준편차, 창의 독립성·통계적 유의성 아님')
    figures.append(save_figure(plt, fig, folder, '01_core_mse_change'))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    selected = test_rows([r for r in bundle['core'] if r['chosen_weight'] and r['variant'] == 'regular'])
    displayed = ('NO_ALIGN', 'B_GUMBEL', 'B_COSINE', 'B_UNIFORM', 'B_FIXED_GUMBEL', 'A_BIHASH_GUMBEL')
    for ax, ds in zip(axes.flat, DATASETS):
        active = [r for r in selected if r['dataset'] == ds]
        if not active:
            no_data(ax)
            continue
        for i, method in enumerate(displayed):
            values = [r['norm_mse'] for r in active if r['method'] == method]
            if values:
                ax.scatter(np.arange(len(values)) * .035 + i - .035, values, color=PALETTE[i], s=25)
                ax.plot([i - .2, i + .2], [np.mean(values)] * 2, color=PALETTE[i], lw=2)
        raw = [r['norm_mse'] for r in test_rows(bundle['controls']) if r['dataset'] == ds and r['method'] == 'RAW_BOLT']
        if raw:
            ax.axhline(raw[0], color='black', ls='--', lw=1, label='RAW_BOLT (고정 경로 1개)')
        ax.set_xticks(range(len(displayed)), [v.replace('B_', '').replace('A_BIHASH_', 'A_') for v in displayed], rotation=22)
        ax.set_ylabel('정규화 MSE · 낮을수록 좋음')
        ax.set_title(TITLES[ds])
        ax.grid(axis='y', alpha=.2)
        if raw:
            ax.legend(fontsize=8)
    fig.suptitle('실제 추론 오차 · 점=각 seed, 선=평균; 목표 N=3\n모든 CP/보조 가중치 선택은 TUNE, RAW 경로는 seed 반복 없음')
    fig.tight_layout(rect=(0, 0, 1, .92))
    figures.append(save_figure(plt, fig, folder, '02_hash_controls_seed_errors'))

    readouts = test_rows(bundle['readouts'])
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, route in zip(axes, ('E_RAW_RIDGE', 'E_ENRICHED_RIDGE')):
        ratios = paired_ratios(readouts, method_key='source_method', route=route)
        changes = group_change(ratios)
        matrix = np.full((len(PROBES), len(DATASETS)), np.nan)
        for i, method in enumerate(PROBES):
            for j, ds in enumerate(DATASETS):
                if (ds, method) in changes:
                    matrix[i, j] = changes[(ds, method)][0]
        if not np.isfinite(matrix).any():
            no_data(ax)
            continue
        bound = max(1., float(np.nanmax(np.abs(matrix))))
        im = ax.imshow(np.ma.masked_invalid(matrix), cmap='PuOr_r', vmin=-bound, vmax=bound, aspect='auto')
        ax.set_xticks(range(4), ('Jena', 'Robin', 'Household', 'SML2010'), rotation=18)
        ax.set_yticks(range(len(PROBES)), PROBES)
        for i, method in enumerate(PROBES):
            for j, ds in enumerate(DATASETS):
                if (ds, method) in changes:
                    mean, sd, n = changes[(ds, method)]
                    ax.text(j, i, f'{mean:+.2f}%\nN={n}', ha='center', va='center', fontsize=8,
                            color='white' if abs(mean) > .65 * bound else 'black')
        fig.colorbar(im, ax=ax, label='NO_ALIGN의 동일 readout 대비 MSE 변화 (%)')
        ax.set_title(route + ' · 낮을수록 좋음')
    fig.suptitle('원 D를 제거한 frozen Ridge · 512K 표현 + 동일 last C anchor\nRidge alpha/원 CP는 TUNE 선택; A/NO_ALIGN Eaug=Eraw alias, 추가 학습 아님')
    fig.tight_layout(rect=(0, 0, 1, .91))
    figures.append(save_figure(plt, fig, folder, '03_frozen_readout_changes'))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    variants = ('regular', 'student', 'memory', 'uniform', 'reset_maps', 'tanh')
    lock = bundle['lock'] or {}
    methods = list(dict.fromkeys(('B_GUMBEL', lock.get('selected_B_method', 'B_GUMBEL'))))
    for ax, ds in zip(axes.flat, DATASETS):
        any_values = False
        for method_index, method in enumerate(methods):
            subset = [r for r in test_rows(bundle['core']) if r['dataset'] == ds and r['method'] == method]
            reference = {r['seed']: r['norm_mse'] for r in subset if r['variant'] == 'regular'}
            for i, variant in enumerate(variants):
                values = [100 * (r['norm_mse'] / reference[r['seed']] - 1) for r in subset
                          if r['variant'] == variant and r['seed'] in reference]
                if values:
                    any_values = True
                    offset = -.12 + .24 * method_index if len(methods) > 1 else 0
                    ax.scatter(np.full(len(values), i + offset), values, color=PALETTE[method_index], s=23)
                    ax.plot([i + offset - .09, i + offset + .09], [np.mean(values)] * 2, color=PALETTE[method_index], lw=2,
                            label=method if i == 0 else None)
        if not any_values:
            no_data(ax)
            continue
        ax.axhline(0, color='black', lw=.8)
        ax.set_xticks(range(len(variants)), variants, rotation=22)
        ax.set_ylabel('regular 대비 정규화 MSE 변화 (%)')
        ax.set_title(TITLES[ds])
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('고정 B_GUMBEL + 개발 TUNE 선택 B의 추론 전용 ablation\n점=seed, 목표 N=3; 재학습·EVAL 선택 없음, 낮을수록 좋음')
    fig.tight_layout(rect=(0, 0, 1, .92))
    figures.append(save_figure(plt, fig, folder, '04_inference_ablations'))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    curves = [r for r in bundle['curves'] if r['chosen_weight']]
    for ax, ds in zip(axes.flat, DATASETS):
        any_values = False
        for method, color in zip(('NO_ALIGN', 'B_GUMBEL', 'B_COSINE', 'B_UNIFORM', 'A_BIHASH_GUMBEL'), PALETTE):
            subset = [r for r in curves if r['dataset'] == ds and r['method'] == method]
            grouped = defaultdict(list)
            for row in subset:
                grouped[row['update']].append(row['main_loss'])
            if not grouped:
                continue
            any_values = True
            updates = sorted(grouped)
            value = np.array([np.mean(grouped[u]) for u in updates])
            # Nonoverlapping 20-update block means; raw unsmoothed values are CSV.
            ends, mean = [], []
            for start in range(0, len(value), 20):
                ends.append(updates[min(start + 19, len(value) - 1)])
                mean.append(float(value[start:start + 20].mean()))
            ax.plot(ends, mean, color=color, label=method)
        if not any_values:
            no_data(ax)
            continue
        ax.set_title(TITLES[ds])
        ax.set_xlabel('실제 optimization update')
        ax.set_ylabel('FIT batch 정규화 MSE · 낮을수록 좋음')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle('학습 안정성의 진단 · 동일 batch 순서, 목표 seed N=3\n20-update 블록 평균; 훈련 손실 감소는 EVAL 이득을 보장하지 않음')
    fig.tight_layout(rect=(0, 0, 1, .92))
    figures.append(save_figure(plt, fig, folder, '05_training_main_loss'))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    points = accuracy_cost_points(bundle, cost)
    selected_b = (bundle.get('lock') or {}).get('selected_B_method')
    methods = list(dict.fromkeys(['NO_ALIGN', 'B_GUMBEL'] + ([selected_b] if selected_b else [])
                                + ['RAW_BOLT', 'RAW_SELF33']))
    markers = ('o', 's', 'D', '^', 'v')
    for ax, ds in zip(axes.flat, DATASETS):
        dataset_points = [row for row in points if row['dataset'] == ds]
        if not dataset_points:
            no_data(ax)
            continue
        for row in dataset_points:
            index = methods.index(row['method'])
            ax.errorbar(row['median_ms_mean'], row['norm_mse_mean'],
                        xerr=row['median_ms_seed_sd'], yerr=row['norm_mse_seed_sd'],
                        color=PALETTE[index], marker=markers[index], markersize=7,
                        linestyle='none', capsize=3, elinewidth=1, markeredgecolor='white', markeredgewidth=.5)
        ax.set_title(TITLES[ds])
        ax.set_xscale('log')
        ax.set_xlabel('batch 1 · 전체 경로 median ms/window · 로그 축')
        ax.set_ylabel('정규화 MSE · 낮을수록 좋음')
        ax.grid(alpha=.2)
        ax.margins(x=.15, y=.18)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=PALETTE[index], marker=markers[index], linestyle='none', markersize=7,
                      label=method) for index, method in enumerate(methods)]
    fig.suptitle('실제 비용과 오차 · 지정 경로의 비교', y=.985)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .947), ncol=len(methods), frameon=False, fontsize=9)
    backend = ('CORE·RAW_BOLT: 원 CUDA | RAW_SELF33: NumPy CPU FP32 / BLAS 2'
               if cost.get('backend_contract', {}).get('CPU_implementation_id') == CPU_RIDGE_IMPLEMENTATION
               else 'backend 조건: 각 원 cost 기록 참조')
    fig.text(.5, .025,
             '표시: NO_ALIGN·사전 지정 B_GUMBEL·TUNE 선택 B·RAW_BOLT·RAW_SELF33; 전체 536행은 CSV에 보존\n'
             'CORE: 3 seed 평균 ± seed SD | RAW: 각각 1개 control fit, seed SD 없음 | warmup 2 / repeat 7 / sync\n'
             + backend + '\nCPU causal fill·정규화~출력 포함; 최초 load·memory cache 생성 제외',
             ha='center', va='bottom', fontsize=8)
    fig.tight_layout(rect=(0, .135, 1, .89))
    figures.append(save_figure(plt, fig, folder, '06_accuracy_cost'))
    return figures, font


def comparison_text(rows, method, reference, metric='norm_mse'):
    result = []
    for ds in DATASETS:
        a = {r['seed']: r[metric] for r in rows if r['dataset'] == ds and r['method'] == method and r['variant'] == 'regular'}
        b = {r['seed']: r[metric] for r in rows if r['dataset'] == ds and r['method'] == reference and r['variant'] == 'regular'}
        common = sorted(set(a) & set(b))
        if not common:
            result.append(f'{ds}: 미실행/대응 seed 없음')
            continue
        ratios = np.array([100 * (a[s] / b[s] - 1) for s in common])
        sd = float(ratios.std(ddof=1)) if len(ratios) > 1 else None
        deviation = f', seed SD {sd:.3f}%' if sd is not None else ', seed SD 계산 불가'
        result.append(f'{ds}: {np.mean([a[s] for s in common]):.6f} vs {np.mean([b[s] for s in common]):.6f}; '
                      f'대응 변화 {ratios.mean():+.3f}%{deviation}, N={len(common)}')
    return '\n'.join(result)


def audit_gate(audit, key):
    names = (key, 'deployment_parity') if key == 'deployment_output_parity' else (key,)
    nested = audit.get('gates', {})
    values = [container.get(name) for container in (audit, nested) if isinstance(container, dict) for name in names]
    verdicts = [value.get('passed') for value in values if isinstance(value, dict)]
    if any(value is False for value in verdicts):
        return 'FAIL'
    if any(value is True for value in verdicts):
        return 'PASS (선언된 검사 범위 안에서)'
    return '미확인: 전용 감사 판정 없음'


def descriptive_effect(rows, method, reference):
    """Describe finite paired observations; no post-hoc significance threshold."""
    comparisons = []
    for ds in DATASETS:
        a = {r['seed']: r['norm_mse'] for r in rows if r['dataset'] == ds and r['method'] == method and r['variant'] == 'regular'}
        b = {r['seed']: r['norm_mse'] for r in rows if r['dataset'] == ds and r['method'] == reference and r['variant'] == 'regular'}
        common = sorted(set(a) & set(b))
        if common:
            changes = [100 * (a[s] / b[s] - 1) for s in common]
            comparisons.append({'dataset': ds, 'mean': float(np.mean(changes)),
                                'improved_seeds': sum(v < 0 for v in changes), 'N': len(changes)})
    fresh = [r for r in comparisons if r['dataset'] not in ('jena', 'robin')]
    if len(fresh) < 2 or any(r['N'] != 3 for r in fresh):
        verdict = '두 신규 자료 × 3 seed의 대응 비교가 완성되지 않아 신규 자료에서의 유용성 판단은 미완료다.'
    elif all(r['mean'] < 0 for r in fresh):
        verdict = '두 신규 자료의 평균 MSE는 모두 감소했다. 이 두 분할에서의 긍정 증거이며, 전체 자료·목표 지표의 일관된 개선이나 보편적 가능성 증명은 아니다.'
    elif all(r['mean'] >= 0 for r in fresh):
        verdict = '두 신규 자료의 평균 MSE가 모두 개선되지 않았다. 이번 고정 용량·학습량·예측기 조건에서 추가 효용의 근거가 부족하며, 방법의 보편적 불가능성을 뜻하지 않는다.'
    else:
        verdict = '신규 자료 간 평균 MSE 변화 방향이 다르다. 효과가 자료에 의존하며 일관된 개선으로 결론내릴 수 없다.'
    evidence = '; '.join(f"{r['dataset']}: 대응 seed 개선 {r['improved_seeds']}/{r['N']}" for r in comparisons)
    return verdict + (' ' + evidence + '.' if evidence else '')


def predictor_comparisons(bundle):
    core = test_rows([r for r in bundle['core'] if r['chosen_weight'] and r['variant'] == 'regular'])
    readouts = test_rows(bundle['readouts'])
    controls = test_rows(bundle['controls'])
    lines = []
    for ds in DATASETS:
        values = []
        for name, subset in (
            ('B_GUMBEL Bolt+D', [r for r in core if r['dataset'] == ds and r['method'] == 'B_GUMBEL']),
            ('B_GUMBEL Eraw+Ridge', [r for r in readouts if r['dataset'] == ds and r['source_method'] == 'B_GUMBEL' and r['route'] == 'E_RAW_RIDGE']),
            ('B_GUMBEL Eaug+Ridge', [r for r in readouts if r['dataset'] == ds and r['source_method'] == 'B_GUMBEL' and r['route'] == 'E_ENRICHED_RIDGE']),
            ('RAW_BOLT', [r for r in controls if r['dataset'] == ds and r['method'] == 'RAW_BOLT']),
            ('RAW_JOINT_RIDGE', [r for r in controls if r['dataset'] == ds and r['method'] == 'RAW_JOINT_RIDGE']),
            ('RAW_BOLT_RAW_R', [r for r in controls if r['dataset'] == ds and r['method'] == 'RAW_BOLT_RAW_R']),
            ('RAW_SELF33', [r for r in controls if r['dataset'] == ds and r['method'] == 'RAW_SELF33']),
        ):
            if subset:
                values.append(f"{name}={np.mean([r['norm_mse'] for r in subset]):.6f} (N={len(subset)})")
        lines.append(ds + ':\n  ' + ('\n  '.join(values) if values else '미실행'))
    return '\n'.join(lines)


def coverage(dev, fresh, probes, evaluation, cost, bundle):
    return [
        {'item': 'DEV core fits', 'actual': len(dev.get('cases', [])), 'planned': 102},
        {'item': 'Fresh core fits', 'actual': len(fresh.get('cases', [])), 'planned': 72},
        {'item': 'Core fits at 600 updates', 'actual': sum(c['training']['updates'] == 600 and len(c['training']['curve']) == 600 for c in bundle['cases']), 'planned': 174},
        {'item': 'Future AE fits at 400 updates', 'actual': sum(p['updates'] == 400 and len(p['curve']) == 400 for p in dev.get('future_pretrains', []) + fresh.get('future_pretrains', [])), 'planned': 12},
        {'item': 'Saved core TUNE CP arrays', 'actual': len(bundle['checkpoints']), 'planned': 870},
        {'item': 'Readout conditions', 'actual': len(probes.get('readouts', [])), 'planned': 72},
        {'item': 'Unique Ridge readout fits', 'actual': sum(r.get('additional_fit', False) for g in probes.get('readouts', []) for r in g['readouts']), 'planned': 108},
        {'item': 'Scientific readout outputs', 'actual': sum(len(g['readouts']) for g in probes.get('readouts', [])), 'planned': 144},
        {'item': 'Evaluated selected core cases', 'actual': len(evaluation.get('core_evaluation', [])), 'planned': 144},
        {'item': 'Raw control paths', 'actual': sum(len(g['methods']) for g in probes.get('controls', [])), 'planned': 16},
        {'item': 'Cost rows', 'actual': len(cost.get('records', [])), 'planned': 536},
    ]


def code_table(rows, fields):
    string_rows = [[str(row.get(k, '')) for k in fields] for row in rows]
    widths = [max(len(k), max((len(row[i]) for row in string_rows), default=0)) for i, k in enumerate(fields)]
    lines = ['  '.join(k.ljust(width) for k, width in zip(fields, widths))]
    lines.extend('  '.join(value.ljust(width) for value, width in zip(row, widths)) for row in string_rows)
    return '```text\n' + '\n'.join(lines) + '\n```'


def decision_notes(args):
    """Optional, actually existing diagnostic files; never invent missing notes."""
    result = {}
    for name, relative in DECISION_FILES.items():
        explicit = getattr(args, name + '_result', None)
        path = Path(explicit).resolve() if explicit else (HERE / relative).resolve()
        if not path.is_file():
            if explicit:
                raise FileNotFoundError(path)
            continue
        if not path.is_relative_to(HERE):
            raise ValueError('DECISION_NOTE_OUTSIDE_PRIMARY_STUDY')
        if any(note['path'] == path for note in result.values()):
            continue
        item = {'path': path, 'value': read(path), 'files': [path],
                'result_link': '../' + path.relative_to(HERE).as_posix()}
        for filename, field in (('REPORT_KO.md', 'report_link'), ('ARRAY_READOUT_AUDIT.json', 'array_audit_link'),
                                ('SOURCE_PRESERVATION.json', 'source_preservation_link')):
            sibling = path.parent / filename
            if sibling.is_file():
                item['files'].append(sibling)
                item[field] = '../' + sibling.relative_to(HERE).as_posix()
                if filename.endswith('.json'):
                    item['source_preservation' if filename == 'SOURCE_PRESERVATION.json' else 'array_audit'] = read(sibling)
        if name == 'train_deploy_gap':
            for filename, field in (('ANALYSIS_KO.md', 'analysis_link'),
                                    ('CONDITION_METRICS_v1.csv', 'condition_metrics_link'),
                                    ('GROUP_SUMMARY_v1.csv', 'group_summary_link'),
                                    ('NUMERIC_RECHECK_v1.json', 'numeric_recheck_link')):
                sibling = path.parent / filename
                if sibling.is_file():
                    item['files'].append(sibling)
                    item[field] = '../' + sibling.relative_to(HERE).as_posix()
                    if filename.endswith('.json'):
                        item['numeric_recheck'] = read(sibling)
            source = HERE / 'measure_train_deploy_gap.py'
            if source.is_file():
                item['files'].append(source)
                item['source_link'] = '../measure_train_deploy_gap.py'
            reference = item['value'].get('resource_receipt')
            if isinstance(reference, dict) and reference.get('path'):
                receipt_path, receipt_value = recovery_json(reference)
                item['files'].append(receipt_path)
                item['gpu_receipt_link'] = '../' + receipt_path.relative_to(HERE).as_posix()
                item['gpu_receipt'] = receipt_value
        if name.startswith('preservation_'):
            identity = str(path).lower() + str(item['value'].get('tag', '')).lower()
            item['receipt_role'] = 'final' if 'preservation_final' in identity or '/final_v' in identity.replace('\\', '/') else 'current'
        result[name] = item
        # A legacy --preservation-current-result may point at a final receipt.
        # Keep the actual original current receipt as a distinct input as well.
        original = (HERE / relative).resolve()
        if name == 'preservation_current' and path != original and original.is_file():
            result['preservation_current_saved'] = {'path': original, 'value': read(original), 'files': [original],
                'result_link': '../' + original.relative_to(HERE).as_posix(), 'receipt_role': 'current'}
    return result


def train_deploy_gap_evidence(notes):
    note = notes.get('train_deploy_gap')
    empty = {'coverage_status': 'NOT_PROVIDED', 'rows': [], 'groups': [], 'actual_conditions': 0,
             'expected_conditions': 48, 'actual_draws': 0, 'expected_draws': 384}
    if note is None:
        return empty
    value = note['value']
    methods = ('B_GUMBEL', 'B_ST', 'B_FIXED_GUMBEL', 'B_TANH')
    expected = {(ds, seed, method) for ds in DATASETS for seed in SEEDS for method in methods}
    rows, identities = [], set()
    for condition in value.get('conditions', []):
        identity = (condition['dataset'], condition['seed'], condition['method'])
        if identity in identities or identity not in expected:
            raise ValueError('REPORT_DUPLICATE_OR_UNEXPECTED_TRAIN_DEPLOY_CONDITION')
        identities.add(identity)
        metrics, evidence = condition['metrics'], condition['evidence']
        deploy, training = metrics['deploy_fit'], metrics['train_draw_mean_mse']
        row = {'dataset': identity[0], 'seed': identity[1], 'method': identity[2],
               'split': 'FIT_DIAGNOSTIC', 'selected_update': evidence['primary_update'],
               'temperature': evidence['temperature'], 'draws': len(metrics['train_draw_fit']),
               'norm_mse_deploy_sign': deploy['mse'], 'norm_mae_deploy_sign': deploy['mae'],
               'norm_mse_training_draw_mean': training,
               'norm_mae_training_draw_mean': metrics['train_draw_mean_mae'],
               'training_vs_sign_mse_percent': 100 * (training / deploy['mse'] - 1) if deploy['mse'] else None,
               'forecast_gap_rms': metrics['forecast_gap_rms'],
               'query_flip_fraction_eligible_rows': metrics['query_code_sign_flip_fraction_eligible_rows'],
               'key_flip_fraction_used_keys': metrics['key_code_sign_flip_fraction_used_keys'],
               'cache_recompute_forecast_max_abs': metrics['cache_recompute_forecast_max_abs'],
               'checkpoint_path': condition['selected_checkpoint']['path'],
               'checkpoint_sha256': condition['selected_checkpoint']['sha256'],
               'scientific_scope': 'Fixed selected CP, prespecified last8 FIT windows, no heldout/generalization claim'}
        if any(not math.isfinite(float(row[key])) for key in (
                'norm_mse_deploy_sign', 'norm_mae_deploy_sign', 'norm_mse_training_draw_mean',
                'norm_mae_training_draw_mean', 'forecast_gap_rms', 'cache_recompute_forecast_max_abs')):
            raise ValueError('REPORT_NONFINITE_TRAIN_DEPLOY_METRIC')
        rows.append(row)
    groups = []
    for dataset in DATASETS:
        for method in methods:
            selected = [row for row in rows if row['dataset'] == dataset and row['method'] == method]
            if not selected:
                continue
            deploy = float(np.mean([row['norm_mse_deploy_sign'] for row in selected]))
            training = float(np.mean([row['norm_mse_training_draw_mean'] for row in selected]))
            flips = [row['query_flip_fraction_eligible_rows'] for row in selected
                     if row['query_flip_fraction_eligible_rows'] is not None]
            groups.append({'dataset': dataset, 'method': method, 'N_seeds': len(selected),
                           'split': 'FIT_DIAGNOSTIC', 'norm_mse_deploy_sign_mean': deploy,
                           'norm_mse_training_draw_mean': training,
                           'training_vs_sign_mse_percent_ratio_of_seed_means': 100 * (training / deploy - 1) if deploy else None,
                           'query_flip_fraction_eligible_rows_mean': float(np.mean(flips)) if flips else None})
    complete = (value.get('status') == 'COMPLETE' and identities == expected and
                all(row['draws'] == 8 for row in rows) and not value.get('failed_conditions') and
                value.get('actual_completed_conditions') == 48)
    numeric = note.get('numeric_recheck', {})
    numeric_same_result = numeric.get('source_result', {}).get('sha256') == sha(note['path'])
    return {**empty, 'coverage_status': 'COMPLETE_ACTUAL_48X8_FIT_DIAGNOSTIC' if complete else 'PARTIAL_OR_INCONSISTENT_DECLARED_COVERAGE',
            'declared_status': value.get('status', '미기록'), 'rows': rows, 'groups': groups,
            'actual_conditions': len(rows), 'actual_draws': sum(row['draws'] for row in rows),
            'missing_conditions': sorted(expected - identities),
            'failed_condition_count': len(value.get('failed_conditions', [])),
            'cache_recompute_forecast_max_abs': max((row['cache_recompute_forecast_max_abs'] for row in rows), default=None),
            'numeric_recheck_status': numeric.get('status', '미제공'),
            'numeric_recheck_checks': numeric.get('checks'), 'numeric_recheck_matches_result': numeric_same_result,
            'original_selected_fit_forecast_max_abs': numeric.get('max_primary_saved_FIT_forecast_gap') if numeric_same_result else None,
            'note': note}


def train_deploy_gap_text(evidence):
    if evidence['coverage_status'] == 'NOT_PROVIDED':
        return ['### 실제 학습 코드와 배포 sign 차이', '',
                '실제 Gumbel-noise train mode와 배포 sign의 대응 진단 결과가 제공되지 않았다. 위의 inference-only 고정 .25 tanh 민감도만으로 그 차이를 측정했다고 주장하지 않으며 실제 gap은 미확인으로 남는다.', '']
    note = evidence['note']
    display = [{'dataset': row['dataset'], 'method': row['method'], 'N': row['N_seeds'],
                'sign_MSE': number(row['norm_mse_deploy_sign_mean']),
                'train_MSE': number(row['norm_mse_training_draw_mean']),
                'train-sign(%)': number(row['training_vs_sign_mse_percent_ratio_of_seed_means']),
                'query_flip(%)': number(100 * row['query_flip_fraction_eligible_rows_mean'])
                    if row['query_flip_fraction_eligible_rows_mean'] is not None else '미기록'}
               for row in evidence['groups']]
    lines = ['### 실제 학습 코드와 배포 sign 차이: FIT 진단', '',
             f'별도 실제 status `{evidence["declared_status"]}`, coverage `{evidence["coverage_status"]}`: 완료 조건 `{evidence["actual_conditions"]}/48`, 실제 training-mode draw `{evidence["actual_draws"]}/384`, 실패 기록 `{evidence["failed_condition_count"]}`개다. 부분·불일치 coverage는 전체 완료로 읽지 않는다.', '',
             '같은 선택 CP, 사전 고정 마지막 8 FIT 창(248..255), 동일 target/mask에서 원 모델의 full train mode와 frozen eval Chronos-Bolt를 실제 연결해 예측했다. optimizer·가중치 갱신·새 CP 선택은 없다. primary 온도는 `temperature(selected_update)`로 u100=.875, u200=.75, u400=.5, u600=.25다. 선택된 고정 가중치와 CP noise counter를 이어 쓴 forward 진단이며, 과거 마지막 optimizer step(update-1)의 재연은 아니다. 고정 .25 inference-only tanh와 이 선택-update 학습 tanh를 구분한다.', '',
             code_table(display, ('dataset', 'method', 'N', 'sign_MSE', 'train_MSE', 'train-sign(%)', 'query_flip(%)')), '',
             '여기서 training-vs-sign(%)는 자료별 seed 평균 MSE의 비율이다. 실제 코드 flip·예측 차이는 train/deploy 불일치의 존재를 보여주지만, 이 차이가 기존 성능 실패의 원인임을 증명하지 않는다. noisy training보다 sign의 FIT MSE가 낮은 경우는 sign 전환이 오차를 악화시킨다는 주장에 대한 반례다. B_TANH에서는 sign flip이 0이어도 코드 크기·cosine 각도가 달라질 수 있다.', '',
             f'실제 train-mode의 memory 재계산과 cached sign 예측의 최대 차이 `{number(evidence["cache_recompute_forecast_max_abs"])}`. 원 저장 선택 FIT NPZ와 새 sign 예측의 최대 차이는 연결된 worker 수치 재검산에 기록된 `{number(evidence["original_selected_fit_forecast_max_abs"])}`이다. 미기록 수치를 0으로 대체하지 않는다. 각 draw 후 CP state/buffer/counter 복구와 backbone/source SHA 검사는 원 결과의 evidence를 따른다.', '',
             f'구현 worker의 CPU 원수치 재검산 status `{evidence["numeric_recheck_status"]}`, checks `{number(evidence["numeric_recheck_checks"])}`, 해당 RESULT SHA 연결 일치 `{evidence["numeric_recheck_matches_result"]}`. 이는 별도 agent의 최종 독립 validator를 대신하지 않으며 전체 감사 gate의 판정과 구분한다.', '',
             '두 범주의 logits `[I,-I]`와 독립 표준 Gumbel에서 sign flip의 이론 확률은 `sigmoid(-2|I|)`다. 공통 양의 온도 감소는 argmax noise flip을 제거하지 않는다. 원 구현의 uniform clamp·FP32 rounding 및 유한 draw 변동을 함께 기록한다.', '',
             '이 표는 FIT mechanism 진단이며 heldout/EVAL 예측 성능 검증이나 새 배포 방식 선택이 아니다. 8 noise draw는 독립 학습 seed가 아니고 FIT 창은 시간 중첩을 가질 수 있다. 기존 입력 재구성 실패·전체 방법의 가능/불가능·모든 데이터의 원인을 이 결과로 확정하지 않는다. EVAL target/TEST를 이 진단에서 열거나 hash하지 않았다. shared legacy TRAIN/DEVVAL 및 fresh FIT/TUNE 압축 컨테이너의 물리적 decompress 범위와 FIT 창 사용을 원 기록에서 구분했다.', '']
    if evidence['missing_conditions']:
        lines += [f'미완료 조건 `{len(evidence["missing_conditions"])}`개는 [원 RESULT]({note["result_link"]})의 조건/실패 목록과 대응하며, 가중치 재학습 없이 기존 CP로 해당 진단 경로만 재개할 수 있다. 자원·source 문제는 원 terminal 상태를 유지한다.', '']
    links = [('원 RESULT', 'result_link'), ('한국어 상세 분석', 'analysis_link'),
             ('조건별 원수치', 'condition_metrics_link'), ('자료·방법별 요약', 'group_summary_link'),
             ('worker CPU 수치 재검산', 'numeric_recheck_link'), ('실제 실행 소스', 'source_link'),
             ('별도 GPU process receipt', 'gpu_receipt_link')]
    lines += ['관련 원본: ' + ', '.join(f'[{label}]({note[key]})' for label, key in links if note.get(key)) + '.', '',
              '추가 GPU/CPU terminal receipts는 기존 자원 표에서 실제 경로당 한 번만 계산한다. 원 결과/수치 재검산의 인용이나 function-body wall을 해당 process receipt에 다시 더하지 않는다.', '']
    return lines


def decision_provenance(notes):
    rows = []
    for name, note in notes.items():
        for path in note['files']:
            rows.append({'diagnostic': name, 'role': 'actual_diagnostic_document',
                         'source_path': path.relative_to(ROOT).as_posix(), 'actual_sha256': sha(path),
                         'declared_sha256': None, 'current_sha_matches_declared': None})
        value = note['value']
        declared = {**value.get('source_hashes', {}), **value.get('scientific_source_hashes', {})}
        if name == 'train_deploy_gap' and value.get('script_sha256'):
            declared[(HERE / 'measure_train_deploy_gap.py').relative_to(ROOT).as_posix()] = value['script_sha256']
        for key in ('local_point_forecast_wrapper', 'installed_chronos_source'):
            source = value.get(key, {})
            if source.get('path') and source.get('sha256'):
                declared[source['path']] = source['sha256']
        for key in ('diagnostic_script', 'current_script_at_start'):
            source = value.get(key, {})
            if source.get('path') and source.get('sha256'):
                declared[source['path']] = source['sha256']
        for condition in value.get('conditions', []):
            for key in ('source_case', 'source_checkpoint', 'source_snapshot', 'actual_arrays', 'selected_checkpoint', 'source_result'):
                source = condition.get(key, {})
                if source.get('path') and source.get('sha256'):
                    declared[source['path']] = source['sha256']
        preservation = note.get('source_preservation', {})
        if preservation.get('preserved_copy') and preservation.get('sha256'):
            declared[preservation['preserved_copy']] = preservation['sha256']
        for relative, expected in declared.items():
            path = Path(relative)
            path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
            protected = path.name.lower() in ('test.npz', 'jena_test.npz', 'eval_future_target_payload.npz')
            actual = sha(path) if path.is_file() and not protected else None
            rows.append({'diagnostic': name, 'role': 'producer_declared_source_or_input',
                         'source_path': str(relative), 'declared_sha256': expected, 'actual_sha256': actual,
                         'current_sha_matches_declared': actual == expected if actual else None,
                         'provenance_scope': 'Current file equality; not a reconstruction of historical execution',
                         'not_read_reason': 'PROTECTED_TEST' if protected else ('SOURCE_FILE_ABSENT' if actual is None else None)})
        for key in ('validator_source_sha256', 'diagnostic_script_sha256'):
            if value.get(key):
                rows.append({'diagnostic': name, 'role': key, 'source_path': None,
                             'declared_sha256': value[key], 'actual_sha256': None,
                             'current_sha_matches_declared': None, 'provenance_scope': 'Declared SHA; source path absent in this result'})
        array_audit = note.get('array_audit', {})
        if array_audit.get('diagnostic_result_sha256'):
            actual = sha(note['path'])
            rows.append({'diagnostic': name, 'role': 'array_audit_result_link',
                         'source_path': note['path'].relative_to(ROOT).as_posix(),
                         'declared_sha256': array_audit['diagnostic_result_sha256'], 'actual_sha256': actual,
                         'current_sha_matches_declared': actual == array_audit['diagnostic_result_sha256']})
    return rows


def recovery_json(reference):
    path = Path(reference['path'])
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if not (path.is_relative_to(HERE) or path.is_relative_to(CACHE)) or path.suffix != '.json':
        raise ValueError('REPORT_RECOVERY_REFERENCE_OUTSIDE_PRIMARY_JSON_SCOPE')
    if not path.is_file() or sha(path) != reference['sha256']:
        raise ValueError('REPORT_RECOVERY_ARTIFACT_SHA_MISMATCH')
    return path, read(path)


def recovery_evidence(fresh):
    evidence = {'status': 'MERGED_LINEAGE_NOT_AVAILABLE', 'producer_rows': [], 'recovered_failures': [],
                'case_lineage': [], 'ae_lineage': [], 'data_repairs': [], 'files': [], 'data_repair_files': []}
    if not fresh.get('producers'):
        return evidence
    evidence['status'] = 'ACTUAL_MERGED_LINEAGE_AVAILABLE'
    producer_by_sha = {}
    for producer in fresh['producers']:
        path, original = recovery_json(producer['result'])
        if original['status'] != producer['original_status'] or original['tag'] != producer['tag']:
            raise ValueError('REPORT_RECOVERY_PRODUCER_TERMINAL_IDENTITY_MISMATCH')
        receipt_path, receipt = recovery_json(producer['resource_receipt'])
        if receipt.get('gpu') is not True or receipt['status'] != original['status']:
            raise ValueError('REPORT_RECOVERY_ORIGINAL_GPU_RECEIPT_MISMATCH')
        snapshot_path, _ = recovery_json(producer['source_snapshot_manifest'])
        evidence['files'] += [path, receipt_path, snapshot_path]
        producer_by_sha[producer['result']['sha256']] = (producer, original)
        evidence['producer_rows'].append({'tag': original['tag'], 'original_terminal_status': original['status'],
            'result_path': str(path), 'result_sha256': producer['result']['sha256'],
            'case_count': len(original.get('cases', [])), 'future_AE_count': len(original.get('future_pretrains', [])),
            'datasets': producer.get('datasets', []), 'original_failed_paths': original.get('failed_paths', []),
            'source_snapshot': producer['source_snapshot_manifest'], 'GPU_receipt': producer['resource_receipt'],
            'GPU_wall_seconds': receipt['wall_seconds'], 'new_training_by_merge': False})
        repair_ref = original.get('data_repair_result')
        if repair_ref:
            repair_path, repair = recovery_json(repair_ref)
            evidence['files'].append(repair_path); evidence['data_repair_files'].append(repair_path)
            evidence['data_repairs'].append({'path': str(repair_path), 'sha256': repair_ref['sha256'],
                'status': repair.get('status'), 'tag': repair.get('tag'), 'reported_wall_seconds': repair.get('wall_seconds'),
                'scope': 'Referenced subset ingest repair; not new model training or EVAL evaluation'})
    for kind, field, output_field in (('case', 'cases', 'case_lineage'), ('AE', 'future_pretrains', 'ae_lineage')):
        for item in fresh.get(field, []):
            lineage = item.get('producer_lineage')
            if not lineage or lineage['producer_result']['sha256'] not in producer_by_sha:
                raise ValueError('REPORT_RECOVERY_ITEM_MISSING_PRODUCER_LINEAGE')
            producer, original = producer_by_sha[lineage['producer_result']['sha256']]
            plain = {key: value for key, value in item.items() if key != 'producer_lineage'}
            if plain not in original.get(field, []): raise ValueError('REPORT_RECOVERY_ITEM_NOT_EQUAL_TO_ORIGINAL')
            record = {'kind': kind, 'dataset': item['dataset'], 'seed': item.get('seed'), 'label': item.get('label'),
                      'producer_tag': producer['tag'], 'original_status': producer['original_status'],
                      'producer_result': lineage['producer_result'], 'new_training_by_merge': False}
            if kind == 'case':
                case_path, source_case = recovery_json(lineage['original_case_json'])
                if source_case != plain: raise ValueError('REPORT_RECOVERY_CASE_JSON_MISMATCH')
                evidence['files'].append(case_path); record['original_case_json'] = lineage['original_case_json']
            evidence[output_field].append(record)
    for failure in fresh.get('recovered_failed_paths', []):
        producer, original = producer_by_sha[failure['producer_result']['sha256']]
        plain = {key: value for key, value in failure.items() if key != 'producer_result'}
        if plain not in original.get('failed_paths', []): raise ValueError('REPORT_RECOVERED_FAILURE_NOT_IN_ORIGINAL')
        evidence['recovered_failures'].append({**failure, 'recovery_state': 'RECOVERED_PRESERVED',
                                               'original_terminal_status': original['status'], 'producer_tag': producer['tag']})
    merge_path, merge_receipt = recovery_json(fresh['resource_receipt'])
    if merge_receipt.get('gpu') is not False: raise ValueError('REPORT_MERGE_MUST_HAVE_CPU_RECEIPT')
    evidence['files'].append(merge_path)
    evidence.update(merge_CPU_receipt=fresh['resource_receipt'], merge_CPU_wall_seconds=merge_receipt['wall_seconds'],
                    merged_status=fresh['status'], new_training_updates_in_merge=fresh.get('new_training_updates_in_merge'),
                    original_GPU_wall_seconds_summary=fresh.get('original_gpu_wall_seconds_counted_once'))
    return evidence


def recovery_text(evidence):
    if evidence.get('status') != 'ACTUAL_MERGED_LINEAGE_AVAILABLE':
        return ['## Fresh 복구 계보', '', '검산된 병합 producer 계보는 아직 미제공이다. 미완료 subset/merge나 없는 영수증을 완료로 추정하지 않는다.', '']
    lines = ['## Fresh 복구 계보: 원 실패와 새 완료를 함께 보존', '',
             '병합 결과의 COMPLETE는 검산된 과학 실행 범위의 합집합을 뜻한다. 원 실행의 FAILED_PRESERVED를 COMPLETE로 바꾸지 않았으며, 과거 실패를 미해결 실패와 구분한다.', '',
             code_table(evidence['producer_rows'], ('tag', 'original_terminal_status', 'case_count', 'future_AE_count', 'GPU_wall_seconds', 'result_sha256')), '',
             'SML의 원 실행 결과는 재학습 없이 재사용했다. Household는 동일 봉인 코드·METHOD_LOCK 조건에서 별도 실행한 결과다. 병합은 새 모델 학습이나 선택을 하지 않으며, source snapshot·각 원 RESULT·case JSON SHA를 계보 표에 남겼다.', '',
             f'병합의 새 학습 update `{number(evidence.get("new_training_updates_in_merge"))}`, 원 GPU wall 합계 요약 `{number(evidence.get("original_GPU_wall_seconds_summary"))}`초, 별도 CPU 병합 receipt `{number(evidence.get("merge_CPU_wall_seconds"))}`초. GPU 합계 요약은 새로운 자원 사용이 아니며 두 원 GPU 영수증은 실제 경로당 한 번만 계산하고 CPU 병합과 분리했다.', '',
             '[원 producer 계보](tables/recovery_producers.csv), [case 계보](tables/recovery_case_lineage.csv), [AE 계보](tables/recovery_AE_lineage.csv), [복구된 과거 실패](tables/recovered_failed_paths.csv).', '']
    for repair in evidence['data_repairs']:
        target = '../' + Path(repair['path']).relative_to(HERE).as_posix()
        lines += [f'추적 가능한 [원 자료 수리 receipt]({target}): status `{repair["status"]}`, tag `{repair["tag"]}`, SHA `{repair["sha256"]}`, 기록된 non-GPU phase wall `{number(repair["reported_wall_seconds"])}`초. 이는 자료 다운로드/메타데이터 수리이며 새 학습·EVAL 검사가 아니다.', '']
    for producer in evidence['producer_rows']:
        target = '../' + Path(producer['result_path']).relative_to(HERE).as_posix()
        receipt_path = Path(producer['GPU_receipt']['path'])
        receipt_path = receipt_path if receipt_path.is_absolute() else ROOT / receipt_path
        receipt_target = '../' + receipt_path.relative_to(HERE).as_posix()
        lines += [f'- `{producer["tag"]}`: [원 RESULT]({target}) status `{producer["original_terminal_status"]}`, SHA `{producer["result_sha256"]}` · [원 GPU receipt]({receipt_target})', '']
    return lines


def failed_path_rows(inputs):
    return [{'input_stage': stage, 'recovery_state': state, **failure}
            for stage, value in inputs.items()
            for field, state in (('failed_paths', 'UNRESOLVED'), ('recovered_failed_paths', 'RECOVERED_PRESERVED'))
            for failure in value.get(field, [])]


def number(value):
    return format(value, '.10g') if isinstance(value, (int, float)) else '미기록'


def decision_text(notes):
    if not notes:
        return ['## 추가 진단의 실제 상태', '', '추가 진단 파일은 제공되지 않았다. 합성 반례나 보존 결과를 추정해 채우지 않았다.', '']
    lines = ['## 추가 진단: 확인한 현상과 아직 확인하지 않은 원인', '']
    bias = notes.get('decoder_bias')
    if bias:
        value = bias['value']
        changes = value.get('actual_saved_bias_changes_u0_to_u600', {})
        changed = sum(v['changed_scalar_count'] for v in changes.values()) if changes else None
        drift = max((v['max_abs'] for v in changes.values()), default=None)
        rows = [r for r in value.get('rows', []) if r.get('update') == 600 and r.get('batch') == 8
                and r.get('mode') == 'saved_FIT_residual_Jacobian_replay' and r.get('form') == 'original']
        lines += ['### 디코더 bias: 수학적 취소와 실제 FP32 update는 다르다', '',
                  f'진단 status `{value.get("status", "미기록")}`. 실제 Jena NO_ALIGN CP0→600에서 두 최종 affine bias의 변경 scalar 합 `{number(changed)}`, 최대 절대 drift `{number(drift)}`를 기록했다. `x_last+D(future)-D(last)`의 정확한 실수 연산에서 최종 bias 미분은 0이지만, FP32의 flat(B×H) 합과 nested(H→B) 합 순서가 다른 경우 작은 gradient와 AdamW update가 남았다.', '']
        for row in rows:
            gradient = max((v['max_abs'] for v in row.get('bias_gradients', {}).values()), default=None)
            step = max((v['max_abs_parameter_change'] for v in row.get('adamw_steps', {}).get('fresh_zero_moments', {}).values()), default=None)
            lines.append(f'- CP600/B8/{row["dtype"]}/저장 FIT residual Jacobian replay: bias gradient 최대 `{number(gradient)}`, fresh AdamW 1 step 변화 최대 `{number(step)}`.')
        lines += ['', f'저장 tensor readout 감사 status `{bias.get("array_audit", {}).get("status", "미제공")}`, 검사 수 `{number(bias.get("array_audit", {}).get("actual_array_checks"))}`. 실제 decoder와 FIT residual을 사용했지만 K latent는 synthetic이며 실제 E/F 출력이 아니다. 과거 GPU gradient tensor/kernel replay가 아니고, 600-step drift 전체 원인·예측 실패 원인·bias 제거의 성능 이득을 확정하지 않는다. restored AdamW 조건은 기존 momentum 자체로 update를 만들 수도 있다.', '',
                  'structural active parameter 계산에서 bias를 수학적으로 빼는 것은 실제 FP32 parameter가 불변이거나 관련 연산 비용이 0이라는 뜻이 아니다. 이번 진단에서 과학 학습 소스·체크포인트를 수정하지 않았다.', '',
                  f'[상세 보고서]({bias.get("report_link", bias["result_link"])}) · [원수치/SHA]({bias["result_link"]})' +
                  (f' · [저장 배열 감사]({bias["array_audit_link"]})' if bias.get('array_audit_link') else ''), '']
    quantile = notes.get('quantile_projection')
    if quantile:
        value = quantile['value']
        checks = value.get('checks', {})
        raw = value.get('raw_median_error', {}); latent = value.get('decoded_latent_median_error', {})
        lines += ['### PCA와 median: 변환 순서만으로 MSE·MAE가 갈릴 수 있다', '',
                  f'합성 진단 status `{value.get("status", "미기록")}`, 실제 true checks `{sum(v is True for v in checks.values())}/{len(checks)}`. 모든 예측이 **같은 과거 조건 X를 이미 알고 있다**고 가정한 정확한 조건부 미래 분포의 반례다. 실제 과거 압축으로 인한 conditioning 정보 손실과 Chronos 모델 오차는 포함하지 않았다.', '',
                  f'원 채널별 median의 MSE/MAE `{number(raw.get("mse"))}/{number(raw.get("mae"))}`가 latent median 선형 복원에서는 `{number(latent.get("mse"))}/{number(latent.get("mae"))}`였다. RAW point forecast가 rank1로 정확히 표현돼도 투영된 분포의 median을 그대로 역투영하면 같아지지 않는다.', '',
                  '동일 조건부 분포에서는 선형 P에 대해 `E[P Y | X]=P E[Y | X]`가 성립한다. 이 사실은 실제 학습된 예측기에 대한 `F(PX)=P F(X)`를 보장하지 않는다. 평균과 분위수 목적 차이를 구분하는 반례이며, 실제 Robin/Beijing/Appliances 실패 원인을 확인한 것이 아니다. D 학습이나 calibration으로 보정할 가능성이 남으므로 구조적 복원 불가능성으로 결론내리지 않는다.', '',
                  f'[합성 반례 보고서]({quantile.get("report_link", quantile["result_link"])}) · [정확한 분포·원수치·소스 SHA]({quantile["result_link"]})', '']
    for _, preservation in [(key, value) for key, value in notes.items() if key.startswith('preservation_')]:
        value = preservation['value']
        final = preservation.get('receipt_role') == 'final'
        role = '최종 역할' if final else '중간 시점'
        limit = '최종 역할로 제공된 receipt가 검사한 파일 범위와 status에 한정한다.' if final else '이 중간 시점 검사는 최종 보고서 시점의 보존 재검사를 대신하지 않는다.'
        lines += [f'### 기존 자산 보존: {role} receipt', '',
                  f'보존 검사 status `{value.get("status", "미기록")}`, 실제 checked artifacts `{number(value.get("checked_artifacts"))}/{number(value.get("expected_artifacts"))}`, HEAD `{value.get("actual_HEAD", "미기록")}`. {limit} 전체 working tree가 깨끗하다는 의미나 연구 전체 완료로 확대하지 않는다.', '',
                  f'[{role} 보존 검사 원문]({preservation["result_link"]})', '']
    query = notes.get('hash_query_capacity')
    failed_query = notes.get('hash_query_capacity_failed_attempt')
    if query:
        value = query['value']; conditions = value.get('conditions', [])
        rows = [{'dataset': c['dataset'], 'method': c['method'], 'seed': c['seed'], 'K': c['K'],
                 'rank': c['maps']['effective_query_map']['numerical_rank'],
                 'FP32_gap': c['synthetic_parity']['FP32_two_linear_vs_FP64_composed_max_abs'],
                 'synthetic_codes_FP32': c['synthetic_parity']['synthetic_FP32_unique_hard_query_codes']} for c in conditions]
        lines += ['### Query 해시 차원: 연속 Linear의 affine 경계', '',
            f'실제 진단 status `{value.get("status")}`, 해당 실행 시점 필수 DEV/SML `{value.get("required_conditions_read", 0)}/18`, 당시 존재한 Household `{value.get("optional_Household_conditions_read", 0)}/6`조건만 읽었다. 나중에 생성된 조건으로 당시 범위를 채우지 않는다.', '',
            'K→R32 projection과 R32→32 hash 사이에 비선형 함수가 없으므로 effective map은 WsP이고 rank≤K다. B는 WsP z+Ws bP+bs, A의 last+mean auxiliary는 WsP(z_last+mean z)+2Ws bP+bs다. R32가 affine query 입력 정보 차원을 증가시키지 않는다는 구조적 확인이다.', '',
            'K1의 32-bit deterministic sign은 정확 산술에서 최대 33 query code다. 학습된 memory keys와 가용성 mask를 고정한 검색 점수 경로의 경계이며 Gumbel noise 학습에는 적용되지 않는다. 일반 K의 전체 hyperplane code 수를 주장하지 않는다.', '',
            '아래 code 수와 parity는 모두 synthetic latent이다. 실제 데이터 code coverage·독립 일반화·성능 실패 원인 검사가 아니다. 임계값 근처 FP32 rounding으로 sign이 달라질 수 있다. 전체 예측은 시간 encoder, continuous S/fusion, raw-last anchor와 시점별 memory mask도 사용하므로 전체 forecasting capacity 붕괴를 뜻하지 않는다. A hash는 학습 보조 경로이며 배포 때 제거된다.', '',
            code_table(rows, ('dataset', 'method', 'seed', 'K', 'rank', 'FP32_gap', 'synthetic_codes_FP32')), '',
            f'진단의 raw dataset load `{value.get("raw_dataset_files_opened", "미기록")}`, EVAL target open `{value.get("EVAL_target_opened", "미기록")}`, GPU 실행 `{value.get("GPU_executed", "미기록")}`. [상세 synthetic 진단]({query.get("report_link", query["result_link"])}) · [원수치/SHA]({query["result_link"]})', '']
    if failed_query:
        value = failed_query['value']
        lines += [f'첫 `selected_maps_v1` 진단은 source snapshot 경로의 run_ prefix 실수로 `{value.get("status")}`, 실제 조건 `{len(value.get("conditions", []))}`개였다. 이를 과학적 실패 결과로 쓰지 않고 실패 원문과 당시 source copy를 보존했다. 성공 v2와 실패 v1의 CPU 영수증은 각 실행을 한 번만 계산하며, 포함된 main wall을 process wall에 다시 더하지 않는다. [원 v1 기록]({failed_query["result_link"]})', '']
        if failed_query.get('source_preservation_link'):
            lines += [f'[실패 시점 source copy/SHA 보존]({failed_query["source_preservation_link"]})', '']
    lines += ['추가 진단들은 [추가 근거 provenance](tables/additional_evidence_provenance.csv)에 파일 SHA·선언된 소스 SHA·현재 파일 일치를 남겼다. 진단 상태와 적용 범위를 Bi-Hash의 실제 예측 유용성이나 독립 일반화 gate 대신 사용하지 않았다.', '']
    return lines


def report_text(inputs, bundle, tables, figures, audit, pilot, cover, report_hashes, font):
    stages = {stage: value['status'] for stage, value in inputs.items()}
    rows = test_rows([r for r in bundle['core'] if r['chosen_weight']])
    complete_counts = all(r['actual'] == r['planned'] for r in cover)
    stage_complete = all(v == 'COMPLETE' for v in stages.values())
    checked = audit.get('failed_count') == 0 and audit.get('status', '').startswith('PASS')
    joint = audit_gate(audit, 'joint_barrier')
    locked = bundle['lock']
    selected_b = locked.get('selected_B_method') if locked else None
    sml_windows = sorted({r['windows'] for r in rows if r['dataset'] == 'uci274_sml2010'})
    top_status = '실행 목록 충족, 독립 감사 판정 확인 필요'
    deployment_checked = (bundle.get('deployment_output_count') == 536
                          and bundle.get('deployment_output_receipt_terminal') is True
                          and bundle.get('deployment_output_recomputed_parity_pass') is True)
    canonical = bundle.get('canonical_cost', {})
    canonical_checked = canonical.get('coverage_status') in (None, 'CANONICAL_BACKEND_LINEAGE_NOT_PROVIDED',
                                                            'VERIFIED_CANONICAL_536_WITH_RECORDED_RECEIPTS')
    if complete_counts and stage_complete and checked and deployment_checked and canonical_checked:
        top_status = '실행 목록 충족, 감사의 선언된 범위에서 통과'
    elif not complete_counts or not stage_complete or not deployment_checked or not canonical_checked:
        top_status = '미완료: 아래 실제 상태와 재개 조건 참조'
    failures = []
    for stage, value in inputs.items():
        failures.extend({'stage': stage, **f} for f in value.get('failed_paths', []))
        if value.get('error'):
            failures.append({'stage': stage, 'dataset': 'stage', 'reason': value['error']})
        if value['status'] in ('NOT_RUN', 'BLOCKED'):
            failures.append({'stage': stage, 'dataset': 'stage', 'reason': value.get('reason', '실제 terminal 기록의 미실행 상태')})
    mse = comparison_text(rows, 'B_GUMBEL', 'NO_ALIGN')
    mae = comparison_text(rows, 'B_GUMBEL', 'NO_ALIGN', 'norm_mae')
    lines = [
        '# 과거 전용 Bi-Hash 미래 표현 실험 결과', '',
        f'**실제 상태: {top_status}.**', '',
        '이 실험은 ExoTimer 원형 전체 재현이 아니다. 논문 식 13–16의 두 학습 가능한 hash 경계, continuous feature 조회, 양방향 fusion, 두 예측 결합을 FIT의 historical future-pattern memory에 적용한 **adaptation**이다. 원 텍스트 입력·LLM·공식 학습 코드를 재현했다고 주장하지 않는다.', '',
        '이 문서는 Bi-Hash 하위 실험의 결과다. 기존 smallK의 모든 미해결 오류·지표 충돌·비용 문제나 전체 연구 목표의 완료를 뜻하지 않는다.', '',
        '이번 FIT 통계는 복원한 physical FP32에서 다시 계산했고 core는 3 seed/batch8이다. 이전 smallK의 통계·2 seed/batch4 조건과 달라 서로 다른 보고서의 정규화 점수 차이를 측정된 방법 개선으로 해석하지 않는다. 같은 연구 안의 새 NO_ALIGN 및 RAW controls와만 대응 비교한다.', '',
        '## 질문에 대한 답과 판단 범위', '',
        '### 학습 가능한 경계와 표현이 예측에 도움이 되었는가?', '',
        f'TUNE에서 고정한 주 B 방법은 `{selected_b or "미선택"}`이며, 사전 지정한 학습 가능 경계 `B_GUMBEL`과 구분한다. 선택 방법의 결과가 학습 가능한 maps의 효용을 대신 증명하지 않는다.', '',
        *(['선택 B 대 동일 NO_ALIGN:', '', '```text\n' + comparison_text(rows, selected_b, 'NO_ALIGN') + '\n```', '',
           '선택 B의 자료 안 효과: ' + descriptive_effect(rows, selected_b, 'NO_ALIGN'), ''] if selected_b else []),
        '사전 지정 B_GUMBEL 대 동일 NO_ALIGN의 실제 정규화 MSE 원수치 및 대응 seed 변화:', '',
        '```text\n' + mse + '\n```', '',
        '**자료 안의 효과 판단:** ' + descriptive_effect(rows, 'B_GUMBEL', 'NO_ALIGN'), '',
        '정규화 MAE도 별도로 비교한다. 주 MSE 개선을 모든 오차 지표 개선으로 바꾸어 말하지 않는다:', '',
        '```text\n' + mae + '\n```', '',
        f'양 hash 경계의 실제 gradient/parameter update gate: **{audit_gate(audit, "parameter_updates")}**. 구현·gradient 검사 통과와 예측 이득은 별개다. 작은 3 seed 실험은 가능/불가능의 보편 증명이 아니며, 동일 조건 안의 효과 크기·반례를 판단하는 증거다.', '',
        '### 실제 미래 누출이 있었는가?', '',
        f'독립 causality gate: **{audit_gate(audit, "causality")}**. 신규 두 자료의 joint 선택 barrier: **{joint}**. 현재 query의 미래 정답·미래 missingness mask는 prediction forward에 들어가지 않으며, FIT memory는 `s+48 <= tau+1`을 만족할 때만 조회하도록 구현했다. 검사 판정은 실제 아래 감사 파일의 범위에 한정한다. 전용 gate가 없으면 누출 부재를 확정하지 않는다.', '',
        '과거 입력 불변성 검사는 제공된 timestamp의 수치적 정보 가용성을 검사한다. 원 자료의 issued forecast 채널이 실제 언제 발행됐는지, 모든 채널의 의미적 가용성, Chronos-Bolt 사전학습 자료와의 중복까지 보증하지 않는다. 보호된 TEST를 새로 쓰지 않았다.', '',
        '### 표현·잠재 예측·복원 중 어디에 문제가 남았는가?', '',
        '동일 선택 CP의 Eraw 및 B의 Eaug를 고정하고, 원 D를 제거한 동일 Ridge로 미래 delta를 예측했다. 원 경로와 readout 경로를 나누어 아래 수치로 확인한다. Eraw+Ridge 이득은 frozen encoder의 이 readout에서의 예측 적합성, Eaug 대 Eraw 이득은 memory-enriched feature의 추가 유용성에 해당한다. 이 차이만으로 Bolt 단독 또는 D 단독을 원인으로 확정할 수 없다. E와 D가 이미 Bolt를 통해 최적화됐다는 조건도 남는다.', '',
        '```text\n' + predictor_comparisons(bundle) + '\n```', '',
        '![Frozen readout 비교](figures/03_frozen_readout_changes.png)', '',
        '미래 AE의 full FIT/TUNE reconstruction·variance·rank는 [pretrain 진단](tables/future_ae_diagnostics.csv)에, 선택 CP의 소표본 진단은 [checkpoint 진단](tables/selected_diagnostics.csv)에 분리했다. 후자는 최대 8창/R32이므로 중심화 rank 상한이 최대 7이다. rank 7을 표현 붕괴로 해석하지 않는다. reconstruction 감소는 미래 예측 적합성의 보장이 아니다.', '',
        '이번 REC는 FIT 미래 delta를 복원하는 phi 교사 AE의 손실이다. 과거 E를 입력 재구성만으로 학습한 실험이 아니며, E는 미래 forecast loss와 A의 정렬 목적 또는 B의 조회·fusion을 통해 학습된다. 따라서 이번 결과를 이전 입력 재구성 표현의 실패 원인에 대한 단독 인과 확인으로 바꾸어 말하지 않는다.', '',
        '### 이산 Bi-Hash·대조 정렬이 꼭 필요한가?', '',
        'B_COSINE, B_UNIFORM, B_FIXED_GUMBEL은 조회·fusion 용량을 맞춘 반례다. 고정 W_s/W_f에서도 E/phi/P가 학습되므로 전체 partition이 고정됐다는 뜻은 아니다. 경계의 gradient/update가 확인되어도 안정적인 예측 이득은 대응 결과로 별도 판단해야 한다. A_SHUFFLE과 DETACH_E는 미래 대응 관계와 encoder auxiliary gradient의 기여를 따로 확인한다. InfoNCE는 원 논문의 공식 loss가 아니라 이번에 추가한 목적이며, 비슷한 미래 패턴을 false negative로 만드는 문제가 남는다.', '',
        '조회는 512 query rows × 최대 128 memory entries × 32 code dimensions의 dense 계산이다. 실제 bit packing·subquadratic 검색을 구현한 것이 아니므로 hash라는 이름에서 비용 절감을 추론하지 않는다. 서로 다른 learnable projection+bias에 무작위 공유 Gaussian LSH의 거리 보존 보장을 그대로 적용하지 않는다.', '',
        *[part for reference in ('B_COSINE', 'B_UNIFORM', 'B_FIXED_GUMBEL')
          for part in ('B_GUMBEL 대 ' + reference + ':', '', '```text\n' + comparison_text(rows, 'B_GUMBEL', reference) + '\n```', '')],
        '연속 kernel 대비: ' + descriptive_effect(rows, 'B_GUMBEL', 'B_COSINE'), '',
        '균등 memory 대비: ' + descriptive_effect(rows, 'B_GUMBEL', 'B_UNIFORM'), '',
        '![동일 용량의 비교와 seed](figures/02_hash_controls_seed_errors.png)', '',
        'A 학습 정렬의 예측 변화는 [선택된 core 원수치](tables/core_seed_metrics.csv)와 [대응 변화](tables/paired_changes.csv)에서 true/shuffle/detach/fixed를 함께 읽어야 한다. A의 미래 교사/phi/hash heads는 prediction forward에서 미사용이며 활성 추론 parameter에서 제외했다. 물리적으로 checkpoint/GPU 객체에서 삭제한 것은 아니고 registered parameters/bank buffers가 코레지던트 상태로 측정 peak memory에 남는다. B의 historical bank·fusion은 실제 예측 경로에서 사용한다.', '',
        'A의 올바른 미래 대응 대 shuffle: ' + descriptive_effect(rows, 'A_BIHASH_GUMBEL', 'A_BIHASH_SHUFFLE'), '',
        'A의 encoder auxiliary gradient 대 DETACH_E: ' + descriptive_effect(rows, 'A_BIHASH_GUMBEL', 'A_BIHASH_DETACH_E'), '',
        '선택 A_CONTINUOUS weight .1과 A hash군 weight .03은 auxiliary 강도가 다르다. 선택된 두 결과의 차이를 이산화 하나의 순수 인과 효과로 해석하지 않는다.', '',
        '### Chronos-Bolt를 유지할 근거가 있는가?', '',
        'RAW_BOLT, RAW_BOLT_RAW_R, RAW_JOINT_RIDGE, RAW_SELF33과 각 encoder Ridge의 실측 오차·전체 추론 비용을 함께 비교한다. native Ridge는 별도 predictor를 FIT/TUNE으로 학습하므로 동일 함수의 Bolt 제거 ablation은 아니다. 단순 predictor가 더 좋으면 이 설정에서 백본 유지의 효용이 약해지는 근거가 되지만, 모든 TSFM의 일반적 무용성을 뜻하지 않는다.', '',
        '표현 간 readout 비교는 동일 512K+last C 폭을 사용한다. RAW_JOINT_RIDGE/RAW_SELF33은 32개 과거 block mean+last value를 쓰므로 표현 readout과 feature 폭·시간 해상도·head parameter 수가 같지 않다. 이 raw controls는 실용 대안이며, capacity를 완전히 맞춘 단일 원인 검사가 아니다.', '',
        '![비용과 정확도](figures/06_accuracy_cost.png)', '',
        '## 비교 조건과 선택 절차', '',
        '모든 core: L512/H48, K=max(1,floor(C/4)), 동일 trainable causal encoder·nonlinear decoder·last C anchor, frozen Chronos-Bolt-small, AdamW lr .001/wd0, batch8, 실제 600 update, 별도 codec/side clip1, 동일 비중첩 future batch order. future AE는 FIT만으로 400 update/batch64 pretrain하고 core REC weight .1을 공통 적용했다. PCA는 초기화이고 이후 E는 학습된다. strict K-only 배포를 주장하지 않는다.', '',
        'CP 0/100/200/400/600은 실제 TUNE 관측 채널 균등 정규화 MSE로 선택하고 tie는 이른 CP다. A weight .03/.1 및 주 B 방법도 Jena/Robin TUNE에서만 고정했다. DEV_VAL은 이미 반복 확인한 개발 결과다. 신규 Household/SML2010은 두 자료의 core/readout/control을 모두 고정한 joint barrier 이후 EVAL을 열어야 한다. joint 감사 없이는 신규 독립 검증 통과를 주장하지 않는다.', '',
        f'실제 TUNE 선택 B: `{selected_b or "미선택"}`. 실제 A weights: `{json.dumps(locked.get("weights", {}) if locked else {}, ensure_ascii=False)}`.', '',
        '주지표는 FIT 통계로 정규화한 관측 채널 균등 MSE다. 원단위 channel macro는 한 자료 안의 진단이며, 서로 다른 단위의 채널·자료를 원단위로 합산해 순위를 만들지 않는다. SD는 3개 seed의 기술 통계이고 겹치는 시계열 창이 독립 표본이라는 뜻이 아니다.', '',
        f'SML2010의 실제 제공 평가 origin 수는 seed당 `{sml_windows or "미제공"}`창이다. 3 seed가 같은 평가 창을 쓰므로 3배의 독립 시계열 표본으로 계산하지 않는다.', '',
        'L/H는 step 수다. 시간별 Household의 H48은 48시간, 15분 SML2010의 H48은 12시간이므로 동일한 실시간 예측 horizon이 아니다. Jena/Robin의 실제 빈도는 보존된 계약 범위를 확인하며 미확인 빈도를 추정하지 않는다. 자료 간 절대 수치보다 각 자료 안의 동일 조건 대응 비교를 해석한다.', '',
        '## 실제 결과와 진단', '',
        '![주지표 변화](figures/01_core_mse_change.png)', '',
        '![추론 ablation](figures/04_inference_ablations.png)', '',
        'hash군의 배포 regular는 deterministic sign이고, B_COSINE은 연속 projected cosine, B_UNIFORM은 eligible memory 균등 조회다. inference-only tanh는 동일 선택 모델의 sign 조회를 noise-free tanh(I/.25) 조회로 바꾼 민감도 진단이다. 이 tanh 변형 자체는 실제 Gumbel-noise train mode와 배포의 차이를 직접 측정한 것이 아니며, 별도 실제 gap의 제공 여부·측정 범위는 아래에서 구분한다. EVAL에서 재학습·최선 variant 선택을 하지 않는다. Tanh가 더 좋아도 새 deployment 선택의 독립 검증이 아니다. student/memory 단독은 branch 제거 효과를 보여주며 모든 차이를 hash 하나에 귀속시키지 않는다.', '',
        'B의 student-only 분기도 이미 memory를 조회해 보강한 Eaug를 Bolt에 넣는다. student와 memory 분기는 결합 손실로 공동 학습됐으므로, 단독 분기의 높은 오차와 fused의 낮은 오차는 공동 적응과 상보적 결합을 함께 반영한다. 각 분기의 독립적인 예측 적합성이나 bare E→Bolt 경로의 효과로 해석하지 않는다.', '',
        *train_deploy_gap_text(bundle.get('train_deploy_gap', train_deploy_gap_evidence({}))),
        '![학습 손실](figures/05_training_main_loss.png)', '',
        '실제 raw curve·gradient norm·main/aux gradient cosine·hash entropy/occupancy·retrieval weights·alpha는 아래 CSV에 남긴다. 진단 중 일부는 선택 CP의 첫 8창이므로 전체 분포의 추정으로 확대하지 않는다.', '',
        f'FIT memory-key 진단 실제 status `{bundle.get("hash_diagnostics_status", "미제공")}`, 제공된 core case `{bundle.get("hash_diagnostics_case_count", 0)}/174`.', '',
        '선택적 [전체 FIT memory-key 진단](tables/hash_memory_diagnostics.csv)은 미래 AE의 동일 초기 clone, bank 불변, W_s/W_f/phi 변화, positive fraction·bit entropy·collision·sign margin·CPU/GPU cache 일치를 구분한다. collision 감소나 경계 parameter update는 그 자체로 forecast utility가 아니다. NO_ALIGN/A/B_UNIFORM의 hard keys는 배포에서 사용하지 않으며, B_COSINE의 hard-code 값은 counterfactual 진단이다. 전체 coverage가 없으면 부분 진단으로만 읽는다.', '',
        *decision_text(bundle.get('decision_notes', {})),
        *recovery_text(bundle.get('recovery_evidence', {})),
        '## 실행 범위·감사·자원·미완료', '',
        code_table(cover, ('item', 'actual', 'planned')), '',
        code_table([{'stage': k, 'status': v} for k, v in stages.items()], ('stage', 'status')), '',
        f'독립 감사 status `{audit.get("status", "없음")}`, check_count `{audit.get("check_count", audit.get("checked_count", "없음"))}`, failed_count `{audit.get("failed_count", "없음")}`. 보고서 자체는 저장 NPZ를 다시 채점하고 SHA를 확인하지만, 이를 별도의 독립 일반화 검증이라고 부르지 않는다.', '',
        '[사용한 최종 감사 원문](AUDIT_RESULT.json)과 [보고서 생성 manifest](MANIFEST.json)에 실제 입력 SHA·출력 SHA·재채점 범위를 기록했다.', '',
        *audit_repair_text(bundle.get('audit_repair', {})),
        '비용은 batch1/8, warmup2/repeat7, CUDA sync의 median과 실제 repeat 값을 저장했다. CPU causal fill/FIT normalization/H2D/활성 경로/inverse normalization/D2H를 포함한다. 최초 disk/model load와 FIT memory-key precomputation은 측정 밖이므로 공짜라고 주장하지 않는다. VRAM/RSS는 백본과 전체 checkpoint가 함께 올라간 process 총량이며 경로의 전용 증가량이 아니다. parameter 수치는 adaptation module의 registered/학습 시작 시 trainable/deployment-active를 구분하며 frozen Bolt 본체 parameter 수를 포함하지 않는다. 추론 모드 requires_grad=False의 0을 학습 당시 trainable 수로 쓰지 않았다. deployment memory bytes도 별도로 남겼다.', '',
        *canonical_cost_text(canonical),
        f'별도 deployment 출력 보존 실행 status `{bundle.get("deployment_output_status", "미제공")}`, 실제 제공 경로 `{bundle.get("deployment_output_count", 0)}/536`; 독립 출력 parity gate: **{audit_gate(audit, "deployment_output_parity")}**.', '',
        f'보고서가 실제 저장 출력으로 다시 계산한 critical parity `{bundle.get("deployment_output_recomputed_parity_pass", "미제공")}`, 기존 bulk 대비 physical FAIL `{bundle.get("deployment_output_original_bulk_physical_failures", "미제공")}`행. 이 physical FAIL은 동일 batch 참조의 PASS와 별도로 보존하며, 새 참조 때문에 과거 bulk 판정이 바뀌었다고 쓰지 않는다. 실제 terminal GPU receipt 확인 `{bundle.get("deployment_output_receipt_terminal", False)}`.', '',
        *deployment_runtime_text(bundle.get('deployment_runtime', {})),
        '별도 deployment 검산은 동일한 과학 경로의 first 1/8창을 경로마다 한 번 더 추론해 실제 NPZ를 저장하고 독립 CPU로 대조하는 증거다. 이를 원 7회 timed output 전부의 재검산으로 부르지 않으며, 새 재추론 시간은 원 536개 cost rows/median에 섞지 않았다. 실제 참조가 동일 batch인 새 스키마와 기존 저장 prediction인 legacy 스키마를 CSV에서 구분한다. 별도 실행의 GPU 자원은 독립 receipt로 한 번 계산한다.', '',
        '선행 파일럿은 200-update 보조 정렬과 학습 없는 사후 retrieval뿐이다. 그 NPZ 재채점·joint fusion·분리 gradient·source seal·자원·독립 평가 gate에는 결함이 있으며, 이번 주실험의 완료나 강한 불가능성 근거로 사용하지 않는다. 선행 파일럿 GPU receipt 합 475.524856초와 미계측 fixture reserve60초는 **측정과 예산 reserve를 분리**한다. 이번 실제 terminal 자원 영수증은 [자원 표](tables/resource_receipts.csv)에 연결했다.', '',
        f'별도 [CPU 자원 기록](tables/cpu_accounting.csv)은 terminal CPU receipts와 ledger 밖의 partial/final audit·decoder bias 진단을 파일/receipt 단위로 중복 없이 기록했다. 명시된 wall 수치의 누적 하한 `{number(bundle.get("cpu_accounted_wall_lower_bound"))}`초, wall 지표 미기록/의미 미확정 행 `{number(bundle.get("cpu_unknown_wall_rows"))}`개다. 이는 CPU process time의 완전 계측이나 campaign wall이 아니다.', '',
        'v1 실패와 v2 재검사는 각각 실제 status·SHA·elapsed를 보존했다. 하나의 CPU receipt에 포함된 function-body elapsed는 다시 합산하지 않으며, GPU producer receipt를 인용한 CPU validator의 별도 body wall은 구분한다. `cpu_seconds`의 의미나 빠진 startup/write 시간은 임의로 wall에 합치지 않는다. 이번 report 생성 함수의 wall은 MANIFEST에 별도로 기록하며, CSV 작성 전의 누적 표에 소급해 추정하지 않는다.', '',
    ]
    if failures:
        lines += ['실패/미실행 경로:', '', code_table(failures, ('stage', 'dataset', 'label', 'reason', 'error')), '',
                  '완료되지 않은 항목은 완료로 쓰지 않는다. source/data 계약을 고정한 새 tag로 해당 stage를 재개하되 기존 실패·artifact를 보존해야 한다. GPU/CPU/storage/campaign 자원 상한을 넘긴 경우 기존 승인 범위 안의 잔여량과 추가 필요량을 실제 profile 및 terminal receipts로 다시 산정해야 한다.', '']
    else:
        recovered_count = len(bundle.get('recovery_evidence', {}).get('recovered_failures', []))
        lines += [f'현재 입력의 미해결 failed_paths는 0개이며, 복구된 원 실패 `{recovered_count}`개는 위 계보와 원 FAILED_PRESERVED 기록에 별도로 보존했다. 실행 횟수 충족만으로 누출·수치·일반화 gate 통과를 대신하지 않는다.', '']
    lines += [
        '## 재현 자료와 링크', '',
        'GitHub에서 읽을 수 있도록 PNG/SVG·원 CSV·이 MD를 같은 report 폴더 안의 상대 경로로 연결했다. 대형 `.cache`의 PT/NPZ 경로는 CSV에 기록한 로컬 재현용 자산이며 GitHub에 올라갔다고 가정하지 않는다. 현재 보고서 생성은 commit/push를 수행하지 않는다.', '',
        *[f'- [{name}](tables/{name})' for name in tables], '',
        *[f'- [{name}.svg](figures/{name}.svg)' for name in figures], '',
        '[실험 계약](../PLAN_KO.md), [원형과 adaptation 구분](../SOURCE_NOTES_KO.md), [선행 파일럿 감사](../PILOT_AUDIT_v1.json).', '',
        '[AAAI 공식 ExoTimer 논문](https://ojs.aaai.org/index.php/AAAI/article/view/39908/43869), '
        '[공식 저장소](https://github.com/graceLan1994/ExoTimer), '
        '[UCI Household](https://archive.ics.uci.edu/dataset/235/individual%2Bhousehold%2Belectric%2Bpower%2Bconsumption), '
        '[UCI SML2010](https://archive.ics.uci.edu/dataset/274/sml2010). 공식 저장소의 확인된 commit에서는 README/source-coming-soon만 확인됐으므로 원 코드 재현을 주장하지 않는다.', '',
        f'그림은 실제 설치를 확인한 `{font}` 폰트를 사용했다. scientific-visualization·matplotlib skill의 색각 안전 palette, PNG+SVG, 축 단위/낮을수록 좋음, seed 점·SD·선택 출처 표기 원칙을 적용했다. 최종 과학적 해석은 finite benchmark 결과와 미확인 provenance의 범위를 유지한다.', '',
    ]
    return '\n'.join(lines)


def validate_links(out, pending=()):
    text = (out / 'REPORT_KO.md').read_text(encoding='utf-8')
    links = re.findall(r'\]\(([^)]+)\)', text)
    for target in links:
        if target.startswith(('https://', 'http://')):
            continue
        if target in pending:
            continue
        if not (out / target).resolve().is_file():
            raise FileNotFoundError('REPORT_BROKEN_RELATIVE_LINK: ' + target)
    return len(links)


def resources_from_inputs(inputs):
    rows = []
    seen = set()
    entries = list(inputs.items())
    entries += [(stage + ':producer:' + p.get('tag', str(index)), {'resource_receipt': p['resource_receipt']})
                for stage, value in inputs.items() for index, p in enumerate(value.get('producers', []))
                if p.get('resource_receipt')]
    entries += [(stage + ':timing_process:' + str(index), {'resource_receipt': receipt})
                for stage, value in inputs.items()
                for index, receipt in enumerate(value.get('timing_resource_receipts_counted_once', []))]
    for stage, value in entries:
        receipt = value.get('resource_receipt')
        if isinstance(receipt, dict) and 'path' in receipt:
            path = Path(receipt['path'])
        elif isinstance(receipt, str):
            path = Path(receipt)
        elif isinstance(receipt, dict):
            rows.append({'stage': stage, 'scope': 'primary', **receipt})
            continue
        else:
            continue
        path = path if path.is_absolute() else ROOT / path
        path = path.resolve()
        if path not in seen:
            seen.add(path)
            if isinstance(receipt, dict) and sha(path) != receipt['sha256']:
                raise ValueError('REPORT_RESOURCE_RECEIPT_SHA_MISMATCH')
            rows.append({'stage': stage, 'scope': 'primary', 'receipt_path': str(path), 'receipt_sha256': sha(path), **read(path)})
    # Runtime only writes these once at process termination, never active state.
    for path in sorted((HERE / 'resources').glob('*.json')):
        path = path.resolve()
        if path not in seen:
            receipt = read(path)
            if 'finished_unix' not in receipt or 'wall_seconds' not in receipt:
                raise ValueError('REPORT_NONTERMINAL_RESOURCE_RECEIPT')
            seen.add(path)
            rows.append({'stage': receipt.get('tag'), 'scope': 'primary',
                         'receipt_path': str(path), 'receipt_sha256': sha(path), **receipt})
    return rows


def cpu_accounting(resource_rows, notes, audit_path, additional_files=()):
    """Count each recorded CPU receipt/body once; keep unknown wall scope open."""
    rows = []
    ledger_paths, ledger_signatures = {}, set()
    for resource in resource_rows:
        if resource.get('gpu') is not False:
            continue
        path = resource.get('receipt_path')
        if path:
            ledger_paths[str(Path(path).resolve())] = resource
        signature = tuple(resource.get(key) for key in ('pid', 'started_unix', 'finished_unix', 'wall_seconds'))
        if all(v is not None for v in signature):
            ledger_signatures.add(signature)
        rows.append({'category': 'terminal_CPU_ledger', 'status': resource.get('status'),
                     'source_path': path, 'source_sha256': resource.get('receipt_sha256'),
                     'metric_name': 'wall_seconds', 'reported_seconds': resource.get('wall_seconds'),
                     'counted_wall_seconds': resource.get('wall_seconds'), 'covered_by_CPU_receipt': False,
                     'scope': resource.get('wall_scope', resource.get('cpu_wall_scope', 'receipt-defined scope'))})
    candidates = set((HERE / 'validation_v1').glob('*.json'))
    candidates.add(Path(audit_path).resolve())
    candidates.update(Path(path).resolve() for path in additional_files)
    pilot = HERE / 'PILOT_AUDIT_v1.json'
    if pilot.is_file():
        candidates.add(pilot)
    for note in notes.values():
        candidates.update(p for p in note['files'] if p.suffix == '.json')
    seen = set()
    for source in sorted(candidates):
        source = source.resolve()
        if source in seen:
            continue
        seen.add(source)
        value = read(source)
        if value.get('gpu') is True:
            continue
        if value.get('status') is None and not value.get('resource_receipt'):
            continue
        receipt = value.get('resource_receipt')
        covered, extra_receipt = False, None
        if isinstance(receipt, dict) and receipt.get('path'):
            receipt_path = Path(receipt['path'])
            receipt_path = receipt_path.resolve() if receipt_path.is_absolute() else (ROOT / receipt_path).resolve()
            if str(receipt_path) in ledger_paths:
                covered = True
            else:
                referenced = read(receipt_path)
                if referenced.get('gpu') is True:
                    continue
                if referenced.get('gpu') is False:
                    extra_receipt = referenced
                    ledger_paths[str(receipt_path)] = referenced
        elif isinstance(receipt, dict) and receipt.get('gpu') is True:
            continue
        elif isinstance(receipt, dict) and receipt.get('gpu') is False:
            signature = tuple(receipt.get(key) for key in ('pid', 'started_unix', 'finished_unix', 'wall_seconds'))
            covered = all(v is not None for v in signature) and signature in ledger_signatures
            if not covered:
                extra_receipt = receipt
        metric = 'cpu_wall_seconds' if value.get('cpu_wall_seconds') is not None else ('cpu_seconds' if value.get('cpu_seconds') is not None else None)
        common = {'source_path': str(source), 'source_sha256': sha(source), 'status': value.get('status'),
                  'scope': value.get('cpu_wall_scope', 'Declared metric; Python startup/write scope not established')}
        if extra_receipt is not None:
            receipt_metric = next((key for key in ('process_wall_seconds_including_imports', 'wall_seconds', 'main_wall_seconds')
                                   if extra_receipt.get(key) is not None), None)
            wall = extra_receipt.get(receipt_metric) if receipt_metric else None
            rows.append({**common, 'category': 'CPU_receipt_outside_terminal_registry',
                         'metric_name': 'resource_receipt.' + receipt_metric if receipt_metric else None, 'reported_seconds': wall,
                         'counted_wall_seconds': wall, 'covered_by_CPU_receipt': False,
                         'scope': 'Recorded process wall includes imports; contained main wall not added again' if receipt_metric == 'process_wall_seconds_including_imports'
                         else extra_receipt.get('receipt_scope', extra_receipt.get('wall_scope', common['scope']))})
            signature = tuple(extra_receipt.get(key) for key in ('pid', 'started_unix', 'finished_unix', 'wall_seconds'))
            if all(v is not None for v in signature):
                ledger_signatures.add(signature)
            covered = True
        if metric is None and value.get('phase') == 'ingest_repair' and value.get('wall_seconds') is not None:
            metric = 'wall_seconds'
        if metric:
            rows.append({**common, 'category': 'CPU_audit_declared_metric', 'metric_name': metric,
                         'reported_seconds': value[metric], 'covered_by_CPU_receipt': covered,
                         'counted_wall_seconds': 0. if covered else (value[metric] if metric in ('cpu_wall_seconds', 'wall_seconds') else None),
                         'accounting_note': 'Receipt covers this invocation; body not counted twice' if covered else
                         ('Recorded non-GPU phase/body wall lower bound, may include I/O' if metric in ('cpu_wall_seconds', 'wall_seconds') else 'cpu_seconds meaning not inferred as wall')})
        elif not covered and extra_receipt is None:
            rows.append({**common, 'category': 'CPU_audit_wall_missing', 'metric_name': None,
                         'reported_seconds': None, 'counted_wall_seconds': None, 'covered_by_CPU_receipt': False,
                         'accounting_note': 'No recorded elapsed metric; missing duration is not replaced by zero'})
        elif covered and metric is None:
            rows.append({**common, 'category': 'CPU_audit_covered_by_terminal_receipt', 'metric_name': None,
                         'reported_seconds': None, 'counted_wall_seconds': 0., 'covered_by_CPU_receipt': True,
                         'accounting_note': 'Already counted through the terminal CPU receipt'})
    return rows


def physical_output_comparison(actual, prediction, mean, std):
    reference = prediction.astype(np.float32) * std.astype(np.float32) + mean.astype(np.float32)
    gap = np.abs(actual - reference)
    return {'max_abs': float(gap.max()),
            'normalized_max_abs': float(np.abs((actual - mean) / std - prediction).max()),
            'parity_pass': bool(np.all(gap <= .002 + 2e-5 * np.abs(reference)))}


def normalized_output_comparison(matched, bulk):
    gap = np.abs(matched.astype(np.float64) - bulk.astype(np.float64))
    return {'max_abs': float(gap.max()),
            'parity_pass': bool(np.all(gap <= 1e-5 + 2e-6 * np.abs(bulk.astype(np.float64))))}


def canonical_cost_evidence(cost):
    evidence = {'declared_status': cost.get('status'),
                'coverage_status': 'CANONICAL_BACKEND_LINEAGE_NOT_PROVIDED',
                'timing_rows': [], 'producer_rows': [], 'preserved_failures': [],
                'timed_output_rows': [], 'files': [], 'cpu_ridge_rows': 0, 'cuda_rows': 0,
                'cpu_rows_with_all_seven_outputs': 0, 'cuda_rows_with_all_seven_outputs': 0,
                'preserved_cuda_failed_rows': None}
    contract = cost.get('backend_contract', {})
    if contract.get('CPU_implementation_id') != CPU_RIDGE_IMPLEMENTATION:
        return evidence
    evidence.update(coverage_status='PARTIAL_OR_INCONSISTENT_CANONICAL_EVIDENCE', backend_contract=contract)
    records = cost.get('records', [])
    keys = [tuple(row[k] for k in ('dataset', 'identity', 'route', 'batch')) for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError('REPORT_DUPLICATE_CANONICAL_COST_ROUTE')
    mapping = cost.get('timing_producers', [])
    mapped_keys = [tuple(row['key']) for row in mapping]
    if len(mapped_keys) != len(set(mapped_keys)) or set(mapped_keys) != set(keys):
        raise ValueError('REPORT_CANONICAL_TIMING_PRODUCER_COVERAGE')
    sources = {}
    def source(reference):
        identity = (reference['path'], reference['sha256'])
        if identity not in sources:
            path, value = recovery_json(reference)
            evidence['files'].append(path)
            sources[identity] = value
        return sources[identity]
    for producer in cost.get('producers', []):
        original = source(producer['source_result'])
        if original['status'] != producer['original_status']:
            raise ValueError('REPORT_CANONICAL_PRODUCER_STATUS_MISMATCH')
        evidence['producer_rows'].append({**producer, 'source_record_count': len(original.get('records', []))})
    by_key = dict(zip(keys, records))
    for producer in mapping:
        key = tuple(producer['key']); row = by_key[key]
        ridge = row['route'] in RIDGE_ROUTES
        wanted = CPU_RIDGE_IMPLEMENTATION if ridge else 'original_cuda_fp32'
        if row['route'] not in RIDGE_ROUTES | {'CORE', 'RAW_BOLT'} or producer['deployment_implementation'] != wanted:
            raise ValueError('REPORT_CANONICAL_NONUNIFORM_IMPLEMENTATION')
        if ridge and row.get('deployment_implementation') != wanted:
            raise ValueError('REPORT_CANONICAL_ROW_IMPLEMENTATION_MISMATCH')
        reference = producer.get('canonical_selected_from_producer', producer.get('source_result'))
        original = source(reference)
        index = producer['source_row_index']
        if index < 0 or index >= len(original['records']) or row != original['records'][index]:
            raise ValueError('REPORT_CANONICAL_SOURCE_ROW_CHANGED')
        evidence['cpu_ridge_rows' if ridge else 'cuda_rows'] += 1
        evidence['timing_rows'].append({**dict(zip(('dataset', 'identity', 'route', 'batch'), key)),
            **producer, 'report_source_row_object_verified_unchanged': True,
            'report_latency_source': 'original source seven-repeat seconds/median; post-cost calls excluded'})
        saved = {'dataset': row['dataset'], 'identity': row['identity'], 'route': row['route'], 'batch': row['batch'],
                 'deployment_implementation': wanted, 'actual_output': row.get('actual_output'),
                 'all_seven_saved_outputs_available': False, 'timed_output_count': 0,
                 'scope': 'Seven timed arrays unavailable; recorded timings/asserts are retained',
                 'report_all_seven_matched_physical_parity_pass': None,
                 'report_all_seven_original_bulk_physical_parity_pass': None}
        if row.get('actual_output'):
            path = resolve_artifact(row['actual_output']); evidence['files'].append(path)
            with np.load(path, allow_pickle=False) as arrays:
                if 'all_7_physical_outputs' in arrays.files:
                    required = ('matched_science_normalized', 'original_bulk_science_normalized', 'mean', 'std')
                    if any(name not in arrays.files for name in required):
                        raise ValueError('REPORT_CANONICAL_TIMED_NPZ_MISSING_REFERENCE')
                    outputs = arrays['all_7_physical_outputs']
                    matched, bulk, mean, std = [arrays[name] for name in required]
                    if outputs.shape != (7, row['batch'], 48, len(mean)) or matched.shape != outputs.shape[1:] or bulk.shape != matched.shape:
                        raise ValueError('REPORT_CANONICAL_TIMED_OUTPUT_SHAPE')
                    if mean.shape != std.shape or not all(np.isfinite(a).all() for a in (outputs, matched, bulk, mean, std)) or np.any(std <= 0):
                        raise ValueError('REPORT_CANONICAL_TIMED_OUTPUT_INVALID_VALUE')
                    saved.update(all_seven_saved_outputs_available=True, timed_output_count=7,
                        scope='All seven actual timed outputs; separate post-cost one-call outputs excluded',
                        report_all_seven_matched_physical_parity_pass=all(physical_output_comparison(p, matched, mean, std)['parity_pass'] for p in outputs),
                        report_all_seven_original_bulk_physical_parity_pass=all(physical_output_comparison(p, bulk, mean, std)['parity_pass'] for p in outputs),
                        report_matched_to_original_bulk_normalized_parity_pass=normalized_output_comparison(matched, bulk)['parity_pass'])
                    evidence['cpu_rows_with_all_seven_outputs' if ridge else 'cuda_rows_with_all_seven_outputs'] += 1
        evidence['timed_output_rows'].append(saved)
    prior = cost.get('original_cuda_536_failure_preserved')
    if prior:
        original = source(prior['source_result'])
        failed_count = sum(not row.get('parity_pass', False) for row in original.get('records', []))
        if original['status'] != prior['status'] or failed_count != prior['failed_row_count'] or original.get('failed_paths', []) != prior['failed_paths']:
            raise ValueError('REPORT_CANONICAL_PRESERVED_CUDA_FAILURE_MISMATCH')
        evidence['preserved_cuda_failed_rows'] = failed_count
        evidence['preserved_failures'] = [{**failure, 'original_terminal_status': original['status'],
            'source_result': prior['source_result'], 'recovery_state': 'ORIGINAL_CUDA_FAILURE_PRESERVED'} for failure in prior['failed_paths']]
        earlier = prior.get('original_404_failed_cost', {})
        if earlier.get('source_result'):
            first = source(earlier['source_result'])
            if first['status'] != earlier['status']:
                raise ValueError('REPORT_CANONICAL_ORIGINAL_404_STATUS_MISMATCH')
            evidence.update(original_404_status=first['status'], original_404_rows=len(first.get('records', [])))
    receipts = cost.get('timing_resource_receipts_counted_once', [])
    receipt_paths = set()
    for reference in receipts + ([cost['resource_receipt']] if cost.get('resource_receipt') else []):
        path, receipt = recovery_json(reference); evidence['files'].append(path)
        if receipt.get('finished_unix') is None or not isinstance(receipt.get('wall_seconds'), (int, float)) or receipt['wall_seconds'] <= 0:
            raise ValueError('REPORT_CANONICAL_NONTERMINAL_RESOURCE_RECEIPT')
        if reference in receipts:
            if receipt.get('gpu') is not True:
                raise ValueError('REPORT_CANONICAL_TIMING_PROCESS_NOT_GPU_RECEIPT')
            receipt_paths.add(str(path))
        elif receipt.get('gpu') is not False:
            raise ValueError('REPORT_CANONICAL_MERGE_NOT_CPU_RECEIPT')
    evidence['timing_GPU_receipt_count'] = len(receipt_paths)
    cpu_outputs = [row for row in evidence['timed_output_rows'] if row['deployment_implementation'] == CPU_RIDGE_IMPLEMENTATION]
    evidence['saved_timed_original_bulk_physical_failure_rows'] = sum(
        row['report_all_seven_original_bulk_physical_parity_pass'] is False for row in evidence['timed_output_rows'])
    complete = (cost.get('status') == 'COMPLETE' and len(keys) == 536 and evidence['cuda_rows'] == 296
                and evidence['cpu_ridge_rows'] == 240 and evidence['cpu_rows_with_all_seven_outputs'] == 240
                and len(receipt_paths) == 3 and bool(cost.get('resource_receipt')) and len(evidence['producer_rows']) == 2
                and prior is not None and evidence['preserved_cuda_failed_rows'] == 40
                and evidence.get('original_404_rows') == 404 and evidence.get('original_404_status') == 'FAILED_PRESERVED'
                and contract.get('CPU_BLAS_threads') == 2 and contract.get('physical_atol') == .002 and contract.get('physical_rtol') == 2e-5
                and all(row.get('parity_pass') is True for row in records)
                and all(row['report_all_seven_matched_physical_parity_pass'] is True
                        and row.get('report_matched_to_original_bulk_normalized_parity_pass') is True for row in cpu_outputs))
    if complete:
        evidence['coverage_status'] = 'VERIFIED_CANONICAL_536_WITH_RECORDED_RECEIPTS'
    return evidence


def canonical_cost_text(evidence):
    if evidence.get('coverage_status', 'CANONICAL_BACKEND_LINEAGE_NOT_PROVIDED') == 'CANONICAL_BACKEND_LINEAGE_NOT_PROVIDED':
        return ['canonical CPU Ridge 비용 계보는 미제공이다. 기존 비용 기록에 CPU 구현이나 7회 출력 보존 완료를 소급해 주장하지 않는다.', '']
    return [f'canonical 비용의 선언 status `{evidence.get("declared_status")}`, 보고서 계보·저장 출력 검산 범위 `{evidence.get("coverage_status")}`.', '',
        f'현재 원 row 객체와 동일함을 검산한 CUDA CORE/RAW_BOLT `{evidence["cuda_rows"]}/296`행과 CPU Ridge `{evidence["cpu_ridge_rows"]}/240`행을 제공했다. 배포 계약은 모든 120 Ridge 경로 × batch1/8에 `{CPU_RIDGE_IMPLEMENTATION}` / NumPy CPU FP32 head / BLAS 2를 적용한다. 실패했던 일부 경로만 backend를 바꾸는 계약이 아니다. 백본·표현 경로의 CUDA 실행은 유지하며 backend와 thread 차이를 비용 비교 조건으로 공개한다.', '',
        f'원 404행 비용 실행 status `{evidence.get("original_404_status", "미제공")}`와 132행 보충 후 536행 CUDA 실행의 `{evidence.get("preserved_cuda_failed_rows", "미제공")}`개 Ridge physical FAIL / `FAILED_PRESERVED`는 원문·SHA와 함께 보존했다. 새 canonical COMPLETE는 새 배포 구현의 검산 범위이며 과거 FAIL을 PASS로 다시 표시하지 않는다.', '',
        'physical parity는 independent 동일 batch 과학 API 참조에 기존 atol .002 / rtol 2e-5를 그대로 적용한다. 동일 batch 참조와 기존 bulk 예측은 기존 normalized atol 1e-5 / rtol 2e-6로 연결한다. 기존 bulk에 대한 physical FAIL은 별도 결과로 남기며 accuracy NPZ·점수·모델 선택을 바꾸지 않는다.', '',
        f'canonical의 가용 7회 timed 출력에서 기존 bulk physical 비교가 실패한 행은 `{evidence.get("saved_timed_original_bulk_physical_failure_rows", "미제공")}`개다. 원 CUDA 536행의 40개 physical FAIL과 참조·구현·검사 모집단이 다르므로 같은 실패 개수로 합치거나 대체하지 않는다.', '',
        f'7회 실제 timed array 보존·재검산 가능 행: CPU Ridge `{evidence["cpu_rows_with_all_seven_outputs"]}/240`, 재사용 CUDA `{evidence["cuda_rows_with_all_seven_outputs"]}/296`. 첫 실행의 저장되지 않은 timed array를 별도 재추론으로 복원했다고 주장하지 않는다. [구현·timing producer 계보](tables/canonical_cost_timing_provenance.csv), [원 producer 상태](tables/canonical_cost_producers.csv), [원 CUDA 실패](tables/canonical_cost_preserved_failures.csv), [7회 출력 가용 범위](tables/cost_timed_output_availability.csv).', '',
        '원 404행 비용 실행은 7회 timing/assert를 남겼지만 timed array를 모두 저장하지 않았다. 원 132행 보충 실행과 새 240행 CPU Ridge 실행은 7회 timed arrays를 저장했으며 canonical에서는 실제 채택한 원 row의 가용 범위만 계산한다.', '',
        f'원 실패·보충·새 CPU-head 비용 GPU process receipt `{evidence.get("timing_GPU_receipt_count", 0)}/3`개와 별도 CPU merge receipt를 각각 한 번만 계산한다. timing mapping의 반복 receipt·원 wall 요약·median은 추가 process 비용이 아니다.', '']


def deployment_runtime_evidence(args, deployment_result):
    evidence = {'status': 'RUNTIME_PROFILE_NOT_PROVIDED', 'rows': [], 'files': [], 'partial_result': None}
    explicit = getattr(args, 'deployment_runtime_profile', None)
    reference = (deployment_result or {}).get('runtime_cap_adjustment')
    path = Path(explicit).resolve() if explicit else ((ROOT / reference['path']).resolve() if reference
            else HERE / 'DEPLOYMENT_PROOF_WALL_CAP_ADJUSTMENT_v1.json')
    if not path.is_file():
        if explicit or reference:
            raise FileNotFoundError(path)
        return evidence
    if not path.is_relative_to(HERE):
        raise ValueError('REPORT_RUNTIME_PROFILE_OUTSIDE_PRIMARY_STUDY')
    profile_ref = {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)}
    if reference and reference['sha256'] != profile_ref['sha256']:
        raise ValueError('REPORT_RUNTIME_PROFILE_SHA_MISMATCH')
    profile = read(path); prior = profile['preserved_partial']
    partial_path, partial = recovery_json(prior)
    receipt_path, receipt = recovery_json(prior['resource_receipt'])
    count = len(partial['records'])
    critical_pass = bool(count and all(row.get('critical_gates_pass') is True for row in partial['records']))
    after_recorded = bool(partial.get('backbone_after_sha256'))
    if (partial['status'] != prior['status'] or count != prior['actual_outputs']
            or count != partial['actual_saved_batch_outputs']
            or critical_pass != prior['all_402_critical_gates_pass']
            or after_recorded != prior['backbone_after_sha256_recorded']
            or partial['resource_receipt']['sha256'] != prior['resource_receipt']['sha256']):
        raise ValueError('REPORT_RUNTIME_PRESERVED_PARTIAL_METADATA_MISMATCH')
    if (receipt.get('gpu') is not True or receipt.get('status') != partial['status']
            or receipt.get('finished_unix') is None or receipt.get('wall_seconds', 0) <= 0):
        raise ValueError('REPORT_RUNTIME_PARTIAL_TERMINAL_GPU_RECEIPT_INVALID')
    if (profile.get('math_data_model_selection_or_tolerances_changed') is not False
            or profile['new_cap_seconds'] <= profile['original_cap_seconds']
            or profile.get('scientific_source16_unchanged') != partial.get('source_hashes')):
        raise ValueError('REPORT_RUNTIME_ADJUSTMENT_NOT_DECLARED_CAP_ONLY')
    source_ref = profile['source_producer']; source_path = (ROOT / source_ref['path']).resolve()
    newer_path = (ROOT / profile['new_producer_path']).resolve()
    if not source_path.is_relative_to(HERE) or not newer_path.is_relative_to(HERE):
        raise ValueError('REPORT_RUNTIME_PRODUCER_OUTSIDE_PRIMARY_STUDY')
    if sha(source_path) != source_ref['sha256'] or partial.get('script_sha256') != source_ref['sha256']:
        raise ValueError('REPORT_RUNTIME_ORIGINAL_PRODUCER_SHA_MISMATCH')
    newer_ref = {'path': newer_path.relative_to(ROOT).as_posix(), 'sha256': sha(newer_path)}
    fresh = deployment_result if reference else None
    if fresh and (fresh.get('runtime_wall_cap_seconds') != profile['new_cap_seconds']
                  or fresh.get('script_sha256') != newer_ref['sha256']
                  or fresh.get('source_hashes') != profile['scientific_source16_unchanged']):
        raise ValueError('REPORT_RUNTIME_FRESH_PRODUCER_OR_SOURCE_CONTRACT_MISMATCH')
    row = {'profile_status': profile['status'], 'profile_artifact': profile_ref,
           'original_local_wall_cap_seconds': profile['original_cap_seconds'],
           'new_local_wall_cap_seconds': profile['new_cap_seconds'],
           'shared_GPU_process_wall_cap_seconds': profile['shared_GPU_process_wall_cap_seconds'],
           'shared_campaign_wall_cap_seconds': profile['shared_campaign_wall_cap_seconds'],
           'math_data_model_selection_or_tolerances_changed': False,
           'recorded_scientific_source16_contract_identical': True,
           'partial_result': {k: prior[k] for k in ('path', 'sha256')},
           'partial_terminal_status': partial['status'], 'partial_actual_outputs': count,
           'partial_missing_outputs': partial['missing_batch_outputs'],
           'partial_all_saved_critical_gates_pass': critical_pass,
           'partial_failed_routes': len(partial.get('failed_routes', [])),
           'partial_backbone_after_sha256_recorded': after_recorded,
           'partial_backbone_hash_closure_established': bool(after_recorded and
                partial['backbone_before_sha256'] == partial['backbone_after_sha256']),
           'partial_GPU_receipt': prior['resource_receipt'], 'partial_GPU_process_wall_seconds': receipt['wall_seconds'],
           'original_producer': source_ref, 'new_producer': newer_ref,
           'fresh_result_terminal_status': fresh.get('status') if fresh else None,
           'fresh_actual_outputs': len(fresh.get('records', [])) if fresh else None,
           'partial_outputs_merged_into_fresh_proof': False,
           'resource_accounting_scope': 'Partial GPU process once plus fresh GPU process once; neither is cost median timing'}
    evidence.update(status='VERIFIED_RUNTIME_PROFILE_AND_PRESERVED_PARTIAL', rows=[row],
        files=[path, partial_path, receipt_path, source_path, newer_path], partial_result=partial,
        profile_link='../' + path.relative_to(HERE).as_posix(),
        partial_link='../' + partial_path.relative_to(HERE).as_posix(),
        receipt_link='../' + receipt_path.relative_to(HERE).as_posix())
    return evidence


def deployment_runtime_text(evidence):
    if evidence.get('status') != 'VERIFIED_RUNTIME_PROFILE_AND_PRESERVED_PARTIAL':
        return []
    row = evidence['rows'][0]
    return ['### 출력 검증의 runtime 제한 조정과 원 부분 실행', '',
        f'원 v2 출력 검증은 local `{row["original_local_wall_cap_seconds"]}`초 guard에서 `{row["partial_terminal_status"]}`로 종료했다. 저장 출력 `{row["partial_actual_outputs"]}/536`, 미완료 `{row["partial_missing_outputs"]}`개, 저장된 출력의 critical gate 모두 통과 `{row["partial_all_saved_critical_gates_pass"]}`, failed_routes `{row["partial_failed_routes"]}`개다. 이는 부분 출력의 상태이며 전체 경로 완료를 뜻하지 않는다.', '',
        f'원 부분 실행의 F-after SHA 존재 `{row["partial_backbone_after_sha256_recorded"]}`, 한 process의 백본 before/after closure 확인 `{row["partial_backbone_hash_closure_established"]}`. 누락된 F-after를 새 실행의 해시로 채우거나 이 원 기록을 COMPLETE로 승격하지 않는다.', '',
        f'조정은 새 단일 process의 local cap `{row["new_local_wall_cap_seconds"]}`초만 바꾸었다. 공유 GPU/campaign wall cap `{row["shared_GPU_process_wall_cap_seconds"]}/{row["shared_campaign_wall_cap_seconds"]}`초와 기록된 source16·수학·자료·모델 선택·허용오차 계약은 유지했다. 원 402개 출력과 새 출력을 합치지 않으며, 새 실행의 실제 terminal status `{row["fresh_result_terminal_status"] or "미제공"}`, 저장 출력 `{row["fresh_actual_outputs"] if row["fresh_actual_outputs"] is not None else "미제공"}/536`만 별도 판정한다.', '',
        f'원 부분 실행의 GPU process wall `{number(row["partial_GPU_process_wall_seconds"])}`초는 최종 실행의 성공 여부와 관계없이 영수증당 한 번 계산한다. 새 출력 검증 process의 비용도 별도 영수증으로 한 번 계산하며 536행 cost median에는 넣지 않는다. [cap 조정 원문/SHA]({evidence["profile_link"]}) · [원 부분 RESULT/SHA]({evidence["partial_link"]}) · [원 GPU receipt]({evidence["receipt_link"]}) · [runtime provenance](tables/deployment_runtime_provenance.csv).', '']


def audit_repair_evidence(audit_path):
    evidence = {'status': 'FULL_AUDIT_REPAIR_NOT_PROVIDED', 'rows': [], 'files': []}
    path = HERE / 'validation_v1/PRIMARY_FULL_AUDIT_REPAIR_AND_CPU_ACCOUNTING_v1.json'
    if not path.is_file():
        return evidence
    supplement = read(path)
    evidence['files'].append(path)
    for role, field in (('original_full_v1', 'original_failed_audit'), ('final_full_v2', 'final_audit')):
        reference = supplement[field]
        artifact_path, value = recovery_json(reference)
        if value['status'] != reference['status']:
            raise ValueError('REPORT_FULL_AUDIT_REPAIR_STATUS_MISMATCH')
        evidence['files'].append(artifact_path)
        evidence['rows'].append({'role': role, 'path': str(artifact_path), 'sha256': reference['sha256'],
            'status': value['status'], 'check_count': value.get('check_count', value.get('checked_count')),
            'failed_count': value.get('failed_count'), 'original_failures': reference.get('failures'),
            'report_link': '../' + artifact_path.relative_to(HERE).as_posix()})
    if (supplement.get('science_data_models_metric_tolerances_changed') is not False
            or supplement.get('numeric_improvement_or_generalization_claim') is not False):
        raise ValueError('REPORT_FULL_AUDIT_REPAIR_CHANGED_SCIENCE_CONTRACT')
    evidence['rows'].append({'role': 'repair_and_CPU_accounting_supplement', 'path': str(path), 'sha256': sha(path),
        'status': supplement['status'], 'check_count': supplement.get('check_count'), 'failed_count': supplement.get('failed_count'),
        'changes': supplement.get('changes'), 'science_data_models_metric_tolerances_changed': False,
        'numeric_improvement_or_generalization_claim': False, 'report_link': '../' + path.relative_to(HERE).as_posix()})
    final_ref = supplement['final_audit']
    evidence.update(status='VERIFIED_FULL_AUDIT_REPAIR_HISTORY',
        selected_audit_matches_final=sha(audit_path) == final_ref['sha256'],
        independent_CPU_declared_body_seconds=supplement.get('independent_CPU_declared_body_seconds'),
        supplement_cpu_wall_seconds=supplement.get('cpu_wall_seconds'),
        runtime_CPU_process_seconds=supplement.get('ledger_CPU_process_seconds_already_separate'))
    return evidence


def audit_repair_text(evidence):
    if evidence.get('status') != 'VERIFIED_FULL_AUDIT_REPAIR_HISTORY':
        return []
    original, final, supplement = evidence['rows']
    return ['### 전체 감사의 원 실패와 metadata 계약 수정', '',
        f'첫 전체 감사 v1은 감사 코드가 기존 자료에도 신규 평가 개봉 기록을 요구한 범위 오류와 진단 영수증의 상태 이름 불일치로 `{original["failed_count"]}`개 검사가 실패했다. 동일 범위 오류가 원 검사와 gate에 각각 기록된 것이므로 서로 다른 원인 3개로 해석하지 않으며, 원 `{original["status"]}` 영수증·로그·감사 소스를 보존했다.', '',
        f'전체 144개 모델의 사전 잠금은 유지하고 fresh 2자료의 최초 개봉 및 진단 입력범위 상태를 정확히 검사하도록 감사 코드만 수정했다. v2 전체 재검산 status `{final["status"]}`, 검사 `{final["check_count"]}`, 실패 `{final["failed_count"]}`이며 본 보고서 입력과 최종 감사 SHA 일치 `{evidence["selected_audit_matches_final"]}`다.', '',
        f'별도의 `{supplement["check_count"]}`개 검사 supplement status `{supplement["status"]}`, 실패 `{supplement["failed_count"]}`는 v1/v2의 원 결과·원수치·배포/비용 증거가 동일하고 학습·자료·허용오차가 바뀌지 않았음을 확인한다. 원 v1 자체를 PASS로 바꾸지 않으며 예측 개선이나 일반화의 추가 증거로 사용하지 않는다.', '',
        f'[원 v1 FAIL 감사/SHA]({original["report_link"]}) · [최종 v2 감사/SHA]({final["report_link"]}) · [수정 검산·CPU accounting supplement/SHA]({supplement["report_link"]}) · [감사 수정 계보](tables/full_audit_repair_provenance.csv).', '',
        f'기록된 독립 CPU audit/test body 합 `{number(evidence["independent_CPU_declared_body_seconds"])}`초, supplement 자체 body `{number(evidence["supplement_cpu_wall_seconds"])}`초, 별도 runtime CPU process 합 `{number(evidence["runtime_CPU_process_seconds"])}`초를 구분한다. 원 실패 실행도 한 번씩 계산하며 summary의 body 합을 각 원 영수증에 다시 더하지 않는다. 미계측 ad-hoc startup/test overhead는 추정하지 않는다.', '']


def deployment_rows(result):
    rows, identities = [], set()
    for record in result['records']:
        identity = (record['dataset'], record['identity'], record['route'], record['batch'])
        if identity in identities:
            raise ValueError('REPORT_DUPLICATE_DEPLOYMENT_OUTPUT_ROUTE')
        identities.add(identity)
        path = resolve_artifact(record['actual_output'])
        with np.load(path, allow_pickle=False) as arrays:
            required = ('actual_physical', 'raw_history', 'history_observed', 'origins',
                        'mean', 'std', 'source_history_normalized', 'independent_history_normalized')
            if any(key not in arrays.files for key in required):
                raise ValueError('REPORT_DEPLOYMENT_NPZ_MISSING_ARRAY')
            matched_schema = 'matched_science_normalized' in arrays.files
            if matched_schema and 'original_bulk_science_normalized' not in arrays.files:
                raise ValueError('REPORT_DEPLOYMENT_NPZ_MISSING_ORIGINAL_BULK')
            prediction_key = 'matched_science_normalized' if matched_schema else 'science_prediction'
            if prediction_key not in arrays.files:
                raise ValueError('REPORT_DEPLOYMENT_NPZ_MISSING_SCIENCE_REFERENCE')
            actual, prediction = arrays['actual_physical'], arrays[prediction_key]
            mean, std = arrays['mean'], arrays['std']
            batch = record['batch']
            if std.shape != mean.shape or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
                raise ValueError('REPORT_DEPLOYMENT_FIT_STATISTICS_INVALID')
            if actual.shape != prediction.shape or actual.shape != (batch, 48, len(mean)):
                raise ValueError('REPORT_DEPLOYMENT_FORECAST_SHAPE')
            if arrays['raw_history'].shape != (batch, 512, len(mean)) or arrays['origins'].shape != (batch,):
                raise ValueError('REPORT_DEPLOYMENT_INPUT_SHAPE')
            if not np.isfinite(actual).all() or not np.isfinite(prediction).all():
                raise ValueError('REPORT_NONFINITE_DEPLOYMENT_OUTPUT')
            matched_check = physical_output_comparison(actual, prediction, mean, std)
            bulk_check = normalized_check = None
            if matched_schema:
                bulk = arrays['original_bulk_science_normalized']
                if bulk.shape != prediction.shape or not np.isfinite(bulk).all():
                    raise ValueError('REPORT_DEPLOYMENT_ORIGINAL_BULK_SHAPE_OR_NONFINITE')
                bulk_check = physical_output_comparison(actual, bulk, mean, std)
                normalized_check = normalized_output_comparison(prediction, bulk)
            input_gap = np.abs(arrays['source_history_normalized'] - arrays['independent_history_normalized'])
            rows.append({**record, 'report_actual_physical_max_abs': matched_check['max_abs'],
                         'report_actual_normalized_max_abs': matched_check['normalized_max_abs'],
                         'report_actual_physical_parity_pass': matched_check['parity_pass'],
                         'report_reference_kind': 'independent_batch_matched_science' if matched_schema else 'legacy_saved_science_prediction',
                         'report_actual_to_matched_physical_parity_pass': matched_check['parity_pass'] if matched_schema else None,
                         'report_actual_to_original_bulk_physical_max_abs': bulk_check['max_abs'] if bulk_check else None,
                         'report_actual_to_original_bulk_physical_parity_pass': bulk_check['parity_pass'] if bulk_check else None,
                         'report_matched_to_original_bulk_normalized_max_abs': normalized_check['max_abs'] if normalized_check else None,
                         'report_matched_to_original_bulk_normalized_parity_pass': normalized_check['parity_pass'] if normalized_check else None,
                         'report_recomputed_critical_parity_pass': bool(matched_check['parity_pass'] and normalized_check['parity_pass']) if matched_schema else matched_check['parity_pass'],
                         'report_source_independent_preprocess_max_abs': float(input_gap.max()),
                         'report_comparison_scope': 'Separate one-call re-inference; not the seven timed outputs'})
    return rows


def generate(args):
    started = time.perf_counter()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.tag):
        raise ValueError('INVALID_REPORT_TAG')
    paths = {'dev': args.dev_result, 'fresh': args.fresh_result, 'probes': args.probe_result,
             'evaluate': args.evaluation_result, 'cost': args.cost_result}
    inputs = {stage: read_terminal(path, stage) for stage, path in paths.items()}
    audit = read(args.audit_result)
    pilot = read(HERE / 'PILOT_AUDIT_v1.json')
    # All scientific inputs must exist before creating a report directory.
    predictions = Predictions()
    bundle = rows_from_results(inputs['dev'], inputs['fresh'], inputs['probes'], inputs['evaluate'], predictions)
    diagnostic_path = getattr(args, 'diagnostic_result', None)
    hash_diagnostics = read(diagnostic_path) if diagnostic_path else None
    bundle['hash_diagnostics_status'] = hash_diagnostics.get('status', 'status 없음') if hash_diagnostics else '미제공'
    bundle['hash_diagnostics_case_count'] = len(hash_diagnostics.get('case_diagnostics', [])) if hash_diagnostics else 0
    deployment_path = getattr(args, 'deployment_output_result', None)
    deployment_result = read(deployment_path) if deployment_path else None
    if deployment_result and not isinstance(deployment_result.get('records'), list):
        raise ValueError('REPORT_DEPLOYMENT_RECORDS_SCHEMA_NOT_FIXED')
    deployment_records = deployment_rows(deployment_result) if deployment_result else []
    bundle['deployment_output_status'] = deployment_result.get('status', 'status 없음') if deployment_result else '미제공'
    bundle['deployment_output_count'] = len(deployment_result['records']) if deployment_result else 0
    bundle['deployment_output_recomputed_parity_pass'] = (all(row['report_recomputed_critical_parity_pass'] for row in deployment_records)
                                                         if deployment_records else None)
    bulk_checks = [row['report_actual_to_original_bulk_physical_parity_pass'] for row in deployment_records
                   if row['report_actual_to_original_bulk_physical_parity_pass'] is not None]
    bundle['deployment_output_original_bulk_physical_failures'] = sum(not passed for passed in bulk_checks) if bulk_checks else None
    bundle['deployment_output_receipt_terminal'] = False
    deployment_files = []
    if deployment_result and deployment_result.get('resource_receipt'):
        receipt_path, receipt = recovery_json(deployment_result['resource_receipt'])
        deployment_files.append(receipt_path)
        bundle['deployment_output_receipt_terminal'] = (deployment_result.get('status') == 'COMPLETE'
            and receipt.get('status') == 'COMPLETE' and receipt.get('gpu') is True
            and receipt.get('finished_unix') is not None and receipt.get('wall_seconds', 0) > 0)
    canonical = canonical_cost_evidence(inputs['cost'])
    runtime_profile = deployment_runtime_evidence(args, deployment_result)
    audit_repair = audit_repair_evidence(args.audit_result)
    notes = decision_notes(args)
    gap_evidence = train_deploy_gap_evidence(notes)
    recovered = recovery_evidence(inputs['fresh'])
    provenance = decision_provenance(notes)
    resource_inputs = {**inputs, **({'deployment_outputs': deployment_result} if deployment_result else {})}
    if runtime_profile['partial_result']:
        resource_inputs['preserved_partial_deployment_outputs'] = runtime_profile['partial_result']
    resource_records = resources_from_inputs(resource_inputs)
    cpu_records = cpu_accounting(resource_records, notes, args.audit_result, recovered['data_repair_files'])
    bundle['decision_notes'] = notes
    bundle['train_deploy_gap'] = gap_evidence
    bundle['recovery_evidence'] = recovered
    bundle['canonical_cost'] = canonical
    bundle['deployment_runtime'] = runtime_profile
    bundle['audit_repair'] = audit_repair
    bundle['cpu_accounted_wall_lower_bound'] = sum(r['counted_wall_seconds'] for r in cpu_records if r['counted_wall_seconds'] is not None)
    bundle['cpu_unknown_wall_rows'] = sum(r['counted_wall_seconds'] is None for r in cpu_records)
    selected = [r for r in bundle['core'] if r['chosen_weight']]
    summary = summary_rows(selected, ('dataset', 'split', 'method', 'variant'))
    ratios = paired_ratios(test_rows(selected)) + paired_ratios(test_rows(selected), metric='norm_mae')
    cover = coverage(inputs['dev'], inputs['fresh'], inputs['probes'], inputs['evaluate'], inputs['cost'], bundle)
    manifest_inputs = {stage: {'path': str(Path(path).resolve()), 'sha256': sha(path)} for stage, path in paths.items()}
    manifest_inputs['audit'] = {'path': str(Path(args.audit_result).resolve()), 'sha256': sha(args.audit_result)}
    if diagnostic_path:
        manifest_inputs['hash_diagnostics'] = {'path': str(Path(diagnostic_path).resolve()), 'sha256': sha(diagnostic_path)}
    if deployment_path:
        manifest_inputs['deployment_outputs'] = {'path': str(Path(deployment_path).resolve()), 'sha256': sha(deployment_path)}
    for name, note in notes.items():
        for path in note['files']:
            manifest_inputs[name + '_' + path.name] = {'path': str(path), 'sha256': sha(path)}
    for index, path in enumerate(dict.fromkeys(recovered['files'])):
        manifest_inputs['recovery_lineage_' + str(index)] = {'path': str(path), 'sha256': sha(path)}
    for index, path in enumerate(dict.fromkeys(canonical['files'] + deployment_files + runtime_profile['files'] + audit_repair['files'])):
        manifest_inputs['canonical_cost_or_deployment_evidence_' + str(index)] = {'path': str(path), 'sha256': sha(path)}
    out = HERE / ('report_' + args.tag)
    out.mkdir(exist_ok=False)
    table_dir = out / 'tables'; table_dir.mkdir()
    table_values = {
        'core_seed_metrics.csv': bundle['core'], 'core_summary.csv': summary,
        'paired_changes.csv': ratios, 'checkpoint_tune_scores.csv': bundle['checkpoints'],
        'training_curves.csv': bundle['curves'], 'selected_diagnostics.csv': bundle['diagnostics'],
        'future_ae_diagnostics.csv': bundle['pretrain'], 'representation_readout_metrics.csv': bundle['readouts'],
        'raw_control_metrics.csv': bundle['controls'], 'cost_measurements.csv': inputs['cost'].get('records', []),
        'resource_receipts.csv': resource_records, 'execution_coverage.csv': cover,
        'data_contracts.csv': [{'dataset': ds, **record} for value in (inputs['dev'], inputs['fresh']) for ds, record in value.get('data', {}).items()],
        'method_selection.csv': (bundle['lock'] or {}).get('selection_records', []),
        'inference_ablation_metrics.csv': [r for r in test_rows(selected) if r['variant'] != 'regular'],
        'audit_checks.csv': audit.get('checks', []),
        'failed_paths.csv': failed_path_rows(inputs),
        'recovered_failed_paths.csv': recovered['recovered_failures'],
        'recovery_producers.csv': recovered['producer_rows'],
        'recovery_case_lineage.csv': recovered['case_lineage'],
        'recovery_AE_lineage.csv': recovered['ae_lineage'],
        'data_repair_receipts.csv': recovered['data_repairs'],
        'hash_memory_diagnostics.csv': hash_diagnostics.get('case_diagnostics', []) if hash_diagnostics else [],
        'deployment_output_records.csv': deployment_records,
        'deployment_runtime_provenance.csv': runtime_profile['rows'],
        'full_audit_repair_provenance.csv': audit_repair['rows'],
        'canonical_cost_timing_provenance.csv': canonical['timing_rows'],
        'canonical_cost_producers.csv': canonical['producer_rows'],
        'canonical_cost_preserved_failures.csv': canonical['preserved_failures'],
        'cost_timed_output_availability.csv': canonical['timed_output_rows'],
        'additional_evidence_provenance.csv': provenance,
        'cpu_accounting.csv': cpu_records,
        'train_deploy_gap_conditions.csv': gap_evidence['rows'],
        'train_deploy_gap_summary.csv': gap_evidence['groups'],
    }
    for name, rows in table_values.items():
        csv_file(table_dir / name, rows)
    fresh_text(out / 'AUDIT_RESULT.json', json_text(audit))
    if hash_diagnostics:
        fresh_text(out / 'HASH_MEMORY_DIAGNOSTICS.json', json_text(hash_diagnostics))
    if deployment_result:
        fresh_text(out / 'DEPLOYMENT_OUTPUT_RESULT.json', json_text(deployment_result))
    figures, font = make_figures(out, bundle, summary, paired_ratios(test_rows(selected)), inputs['cost'])
    fresh_text(out / 'REPORT_KO.md', report_text(inputs, bundle, list(table_values), figures, audit, pilot, cover, manifest_inputs, font))
    link_count = validate_links(out, pending=('MANIFEST.json',))
    manifest = {'status': 'GENERATED_FROM_ACTUAL_TERMINAL_ARTIFACTS', 'created_unix': time.time(),
                'input_files': manifest_inputs, 'report_generator_sha256': sha(__file__),
                'prediction_npz_count': len(predictions.artifacts), 'prediction_split_count': len(predictions.cache),
                'prediction_artifacts': predictions.artifacts, 'max_abs_report_rescore_vs_recorded': predictions.max_abs_vs_recorded,
                'relative_links_checked': link_count, 'korean_font': font, 'execution_coverage': cover,
                'science_limit': 'Report rescore is not a new holdout evaluation or a proof of universal feasibility/impossibility',
                'files': {p.relative_to(out).as_posix(): sha(p) for p in sorted(out.rglob('*')) if p.is_file()},
                'cpu_wall_seconds': time.perf_counter() - started}
    fresh_text(out / 'MANIFEST.json', json_text(manifest))
    validate_links(out)
    print(json.dumps({'report': str(out / 'REPORT_KO.md'), 'prediction_npz_count': len(predictions.artifacts),
                      'csv_files': len(table_values), 'figure_pairs': len(figures), 'cpu_wall_seconds': manifest['cpu_wall_seconds']}, ensure_ascii=True))
    return out


def self_test():
    # Synthetic values never become study results or report outputs.
    p = np.array([[[1., 2.], [3., 4.]]], np.float32)
    y = np.array([[[0., 2.], [1., 99.]]], np.float32)
    m = np.array([[[1, 1], [1, 0]]], bool)
    result = macro(p, y, m)
    assert result['mse'] == 1.25 and result['mae'] == .75
    rows = [{'dataset': 'jena', 'split': 'val', 'seed': s, 'method': method, 'variant': 'regular',
             'norm_mse': value, 'norm_mae': value, 'physical_mse': value, 'physical_mae': value, 'windows': 64}
            for s in SEEDS for method, value in (('NO_ALIGN', 2.), ('B_GUMBEL', 1.))]
    ratios = paired_ratios(rows)
    assert all(r['change_percent'] == -50. for r in ratios if r['method'] == 'B_GUMBEL')
    summary = summary_rows(rows, ('dataset', 'split', 'method', 'variant'))
    assert all(r['N'] == 3 and r['norm_mse_seed_sd'] == 0. for r in summary)
    assert audit_gate({'status': 'PASS'}, 'joint_barrier').startswith('미확인')
    assert audit_gate({'joint_barrier': {'passed': False}}, 'joint_barrier') == 'FAIL'
    with tempfile.TemporaryDirectory(prefix='bihash_report_fixture_') as folder:
        path = Path(folder) / 'fixture.csv'
        csv_file(path, [{'metric': '관측 MSE', 'value': 1., 'array': [1, 2]}])
        assert path.read_text(encoding='utf-8-sig').startswith('metric,value,array')
        try:
            csv_file(path, [])
        except FileExistsError:
            pass
        else:
            raise AssertionError('Report must preserve existing artifacts')
    print(json.dumps({'status': 'SYNTHETIC_REPORT_HELPERS_PASS', 'tests': 7, 'real_data_opened': False}, ensure_ascii=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev-result', type=Path)
    parser.add_argument('--fresh-result', type=Path)
    parser.add_argument('--probe-result', type=Path)
    parser.add_argument('--evaluation-result', type=Path)
    parser.add_argument('--cost-result', type=Path)
    parser.add_argument('--audit-result', type=Path)
    parser.add_argument('--diagnostic-result', type=Path)
    parser.add_argument('--deployment-output-result', type=Path)
    parser.add_argument('--deployment-runtime-profile', type=Path)
    parser.add_argument('--decoder-bias-result', type=Path)
    parser.add_argument('--quantile-projection-result', type=Path)
    parser.add_argument('--preservation-current-result', type=Path)
    parser.add_argument('--preservation-final-result', type=Path)
    parser.add_argument('--hash-query-capacity-result', type=Path)
    parser.add_argument('--hash-query-capacity-failed-attempt-result', type=Path)
    parser.add_argument('--train-deploy-gap-result', type=Path)
    parser.add_argument('--tag')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if any(getattr(args, name) is None for name in ('dev_result', 'fresh_result', 'probe_result',
                                                  'evaluation_result', 'cost_result', 'audit_result', 'tag')):
        parser.error('All six actual result files and --tag are required. Missing artifacts are never invented.')
    generate(args)


if __name__ == '__main__':
    main()
