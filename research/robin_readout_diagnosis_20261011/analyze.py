"""Derive error mechanisms and figures from completed predictions; never fit."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NAMES = {'RAW_F': 'Raw F', 'ZERO_FULL_D8': 'Zero full D8', 'RES_FULL_D8': 'Residual full D8',
         'ZERO_DIAG_D8': 'Zero diagonal D8', 'RES_DIAG_D8': 'Residual diagonal D8',
         'RES_FULL_D0': 'Residual full D0', 'RES_DC_DIFF_D8': 'Residual DC-diff D8'}
COLORS = dict(zip(NAMES, ['#555555', '#c44e52', '#0072b2', '#cc79a7', '#009e73', '#e69f00', '#6b64a5']))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def load(receipt):
    path = ROOT / receipt['path']
    if sha(path) != receipt['sha256']:
        raise RuntimeError(f'Prediction archive changed: {path}')
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def stats(pred, target, mask):
    p, y = pred.astype(np.float64), target.astype(np.float64)
    w = mask.astype(bool)
    counts = w.sum(axis=(0, 1))
    def mean(value):
        return (np.where(w, value, 0).sum(axis=(0, 1)) / counts)
    error = p - y
    bias = mean(error)
    out = {'target_mean': mean(y), 'forecast_mean': mean(p), 'bias': bias,
           'mse': mean(error**2), 'mae': mean(np.abs(error)),
           'target_std': np.sqrt(mean((y - mean(y))**2)),
           'forecast_std': np.sqrt(mean((p - mean(p))**2)),
           'bias_squared': bias**2, 'centered_error_variance': mean((error - bias)**2)}
    return {key: value.tolist() for key, value in out.items()}


def macro(value, mask):
    count = mask.sum(axis=(0, 1))
    return float((np.where(mask, value, 0).sum(axis=(0, 1)) / count).mean())


def decompose(pred, raw, target, mask):
    delta = pred.astype(np.float64) - raw.astype(np.float64)
    raw_residual = target.astype(np.float64) - raw.astype(np.float64)
    energy = macro(delta**2, mask)
    alignment = 2 * macro(delta * raw_residual, mask)
    excess = macro((pred.astype(np.float64) - target)**2 - raw_residual**2, mask)
    if not np.isclose(energy - alignment, excess, atol=1e-12):
        raise RuntimeError('Error decomposition failed')
    raw_stats, pred_stats = stats(raw, target, mask), stats(pred, target, mask)
    return {'correction_energy': energy, 'twice_residual_alignment': alignment, 'excess_mse': excess,
            'delta_bias_squared': float(np.mean(pred_stats['bias_squared']) - np.mean(raw_stats['bias_squared'])),
            'delta_centered_error_variance': float(np.mean(pred_stats['centered_error_variance']) - np.mean(raw_stats['centered_error_variance']))}


def dc_summary(coeff, spec):
    channels = coeff.shape[1]
    if spec['dc_difference']:
        dc = np.eye(channels)
    else:
        dc = coeff.reshape(spec['lag'] + 1, channels, channels).sum(axis=0)
        if spec['residual']:
            dc = dc + np.eye(channels)
    return {'matrix': dc.tolist(), 'self_gains': np.diag(dc).tolist(),
            'distance_to_identity': float(np.linalg.norm(dc - np.eye(channels))),
            'cross_channel_norm': float(np.linalg.norm(dc - np.diag(np.diag(dc))))}


def coverage(origins, horizon):
    times, counts = np.unique(origins[:, None] + np.arange(horizon), return_counts=True)
    return {'origins': len(origins), 'horizon_rows': int(len(origins) * horizon),
            'unique_future_timestamps': len(times), 'mean_overlap': float(counts.mean()),
            'maximum_overlap': int(counts.max())}


def build_summary(result, data):
    if result['status'] != 'complete':
        raise RuntimeError('Study incomplete')
    summary = {'inputs': {'result.json': sha(HERE / 'result.json')},
               'scope': 'Repeated exploratory VAL and conditional TRAIN-only decoder CV; no independent TEST',
               'channels': data['channel_names'].tolist(),
               'coverage': {phase: coverage(data[phase + '_origins'], 48) for phase in ('train', 'val')},
               'raw': {phase: stats(data[phase + '_forecast'], data[phase + '_target'], data[phase + '_mask']) for phase in ('train', 'val')},
               'arms': {}}
    for arm, row in result['arms'].items():
        if arm == 'RAW_F':
            continue
        archive = load(row['full_train_archive'])
        selections = {}
        for selection, chosen in row['selections'].items():
            index = chosen['index']
            entry = {'lambda': chosen['lambda'], 'index': index,
                     'parameters': row['capacity']['trainable_parameters'],
                     'dc': dc_summary(archive['coefficients'][index], row['spec'])}
            for phase in ('train', 'val'):
                pred = archive['predictions_' + phase][index]
                target, mask, raw = (data[phase + '_' + key] for key in ('target', 'mask', 'forecast'))
                entry[phase] = {'stats': stats(pred, target, mask), 'decomposition': decompose(pred, raw, target, mask),
                                'per_origin_mse': [macro((p.astype(np.float64) - y)**2, m) for p, y, m in zip(pred[:, None], target[:, None], mask[:, None])],
                                'per_horizon_mse': [macro((pred[:, h:h+1].astype(np.float64) - target[:, h:h+1])**2, mask[:, h:h+1]) for h in range(48)]}
            entry['val']['alternate_nonoverlap_subsets'] = {}
            for parity in (0, 1):
                ix = np.arange(parity, len(data['val_origins']), 2)
                pred = archive['predictions_val'][index][ix]
                mask, target = data['val_mask'][ix], data['val_target'][ix]
                entry['val']['alternate_nonoverlap_subsets'][str(parity)] = {
                    'origins': len(ix), 'mse': macro((pred.astype(np.float64) - target)**2, mask),
                    'raw_mse': macro((data['val_forecast'][ix].astype(np.float64) - target)**2, mask),
                    'scope': 'Descriptive overlapping-cohort sensitivity; no refit or reselection'}
            selections[selection] = entry
        summary['arms'][arm] = selections
    return summary


def save(fig, name):
    folder = HERE / 'figures'
    folder.mkdir(exist_ok=True)
    for suffix in ('png', 'pdf'):
        fig.savefig(folder / (name + '.' + suffix), dpi=220, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def figures(result, data, summary):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    arms = list(result['arms'])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    positions = np.arange(len(arms))
    for ax, metric in zip(axes, ('mse', 'mae')):
        for offset, selection, label, hatch in ((-.16, 'VAL_EXPLORATORY', 'VAL-selected lambda', ''),
                                              (.16, 'TRAIN_ROLLING_CV', 'TRAIN-CV-selected lambda', '//')):
            values = [result['arms'][arm]['val']['all'][metric] if arm == 'RAW_F'
                      else result['arms'][arm]['selections'][selection]['val']['all'][metric] for arm in arms]
            ax.barh(positions + offset, values, height=.29, color=[COLORS[a] for a in arms],
                    alpha=.85, hatch=hatch, label=label)
        ax.axvline(result['arms']['RAW_F']['val']['all'][metric], color='#444444', linestyle='--', linewidth=1)
        ax.set_yticks(positions, [NAMES[a] for a in arms])
        ax.invert_yaxis()
        ax.set_xlabel('VAL channel-macro ' + metric.upper() + ' (normalized units)')
        ax.grid(axis='x', alpha=.15)
    axes[0].legend(loc='lower right', fontsize=9)
    fig.suptitle('Same Robin raw forecasts: seven readout arms, two fixed selection rules')
    save(fig, '01_comparison')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    lambdas = result['protocol']['ridge_lambdas']
    for arm in arms[1:]:
        row = result['arms'][arm]
        for ax, phase in zip(axes[:2], ('train', 'val')):
            ax.semilogx(lambdas, [c[phase]['all']['mse'] for c in row['choices']],
                        marker='.', color=COLORS[arm], label=NAMES[arm])
        axes[2].semilogx(lambdas, row['cv']['equal_fold_mean_mse'], marker='.', color=COLORS[arm], label=NAMES[arm])
    for ax, phase in zip(axes[:2], ('train', 'val')):
        ax.axhline(result['arms']['RAW_F'][phase]['all']['mse'], color=COLORS['RAW_F'], linestyle='--', label='Raw F')
    cv_raw = []
    for fold in result['folds']:
        ix = np.flatnonzero((data['train_origins'] >= fold['cut']) & (data['train_origins'] + 48 <= fold['evaluation_end']))
        cv_raw.append(macro((data['train_forecast'][ix].astype(np.float64) - data['train_target'][ix])**2, data['train_mask'][ix]))
    axes[2].axhline(np.mean(cv_raw), color=COLORS['RAW_F'], linestyle='--', label='Raw F')
    for ax, title in zip(axes, ('Full TRAIN fit', 'Repeated development VAL', 'Conditional TRAIN rolling CV')):
        ax.set_title(title)
        ax.set_xlabel('Ridge lambda (sum-SSE objective)')
        ax.set_ylabel('Channel-macro MSE')
        ax.grid(alpha=.15)
    axes[2].legend(fontsize=8)
    save(fig, '02_ridge_paths')

    legacy = summary['arms']['ZERO_FULL_D8']['VAL_EXPLORATORY']
    res = summary['arms']['RES_FULL_D8']['VAL_EXPLORATORY']
    raw = summary['raw']['val']
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    train_mean, val_mean = np.asarray(summary['raw']['train']['target_mean']), np.asarray(raw['target_mean'])
    axes[0, 0].scatter(train_mean, val_mean, c='#0072b2')
    for index in np.argsort(np.abs(val_mean - train_mean))[-6:]:
        axes[0, 0].annotate(summary['channels'][index].split('_')[-1], (train_mean[index], val_mean[index]), fontsize=9)
    axes[0, 0].set(xlabel='TRAIN target mean', ylabel='VAL target mean', title='Mean changes under fixed TRAIN normalization')
    zero_delta = np.asarray(legacy['val']['stats']['mse']) - np.asarray(raw['mse'])
    res_delta = np.asarray(res['val']['stats']['mse']) - np.asarray(raw['mse'])
    order = np.argsort(zero_delta)[::-1]
    x = np.arange(17)
    axes[0, 1].bar(x - .18, zero_delta[order], width=.35, label='Zero full D8', color=COLORS['ZERO_FULL_D8'])
    axes[0, 1].bar(x + .18, res_delta[order], width=.35, label='Residual full D8', color=COLORS['RES_FULL_D8'])
    axes[0, 1].set_xticks(x, [summary['channels'][i].split('_')[-1] for i in order], rotation=60, ha='right', fontsize=8)
    axes[0, 1].set(ylabel='VAL MSE minus Raw F', title='Channel damage at VAL-selected lambda')
    axes[0, 1].axhline(0, color='#555555', linewidth=1)
    axes[0, 1].legend(fontsize=8)
    categories = ['Raw F', 'Zero full D8', 'Residual full D8']
    states = [raw, legacy['val']['stats'], res['val']['stats']]
    bias2 = [np.mean(s['bias_squared']) for s in states]
    variance = [np.mean(s['centered_error_variance']) for s in states]
    axes[1, 0].bar(categories, bias2, color='#e69f00', label='Squared mean bias')
    axes[1, 0].bar(categories, variance, bottom=bias2, color='#56b4e9', label='Centered error variance')
    axes[1, 0].set(ylabel='VAL channel-macro MSE', title='Exact MSE decomposition; neither term proves causality alone')
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].plot(np.arange(1, 49), legacy['val']['per_horizon_mse'], label='Zero full D8', color=COLORS['ZERO_FULL_D8'])
    axes[1, 1].plot(np.arange(1, 49), res['val']['per_horizon_mse'], label='Residual full D8', color=COLORS['RES_FULL_D8'])
    axes[1, 1].plot(np.arange(1, 49), [macro((data['val_forecast'][:, h:h+1].astype(np.float64) - data['val_target'][:, h:h+1])**2, data['val_mask'][:, h:h+1]) for h in range(48)], label='Raw F', color=COLORS['RAW_F'], linestyle='--')
    axes[1, 1].set(xlabel='Forecast horizon (hours)', ylabel='VAL channel-macro MSE', title='Where readout damage occurs')
    axes[1, 1].legend(fontsize=8)
    save(fig, '03_mechanisms')

    fig = plt.figure(figsize=(14, 6.5), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 3.6])
    ax_legacy = fig.add_subplot(grid[0, 0])
    ax_restricted = fig.add_subplot(grid[1, 0])
    ax_trajectory = fig.add_subplot(grid[:, 1])
    fit_arms = arms[1:]
    matrix = np.asarray([np.asarray(summary['arms'][a]['TRAIN_ROLLING_CV']['val']['stats']['mse']) - np.asarray(raw['mse']) for a in fit_arms])
    for ax, part, labels, title in (
        (ax_legacy, matrix[:1], [NAMES[fit_arms[0]]], 'Legacy decoder: per-channel VAL MSE minus Raw F'),
        (ax_restricted, matrix[1:], [NAMES[a] for a in fit_arms[1:]], 'Other arms: separate color scale for small differences')):
        maxabs = max(float(np.abs(part).max()), 1e-8)
        im = ax.imshow(part, cmap='RdBu_r', vmin=-maxabs, vmax=maxabs, aspect='auto')
        ax.set_yticks(np.arange(len(labels)), labels, fontsize=9)
        ax.set_xticks(np.arange(17), [name.split('_')[-1] for name in summary['channels']], rotation=60, ha='right', fontsize=8)
        ax.set_title(title, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=.8, label='MSE difference')
    ax_legacy.tick_params(axis='x', labelbottom=False)
    for arm in ('ZERO_FULL_D8', 'RES_FULL_D8', 'RES_DIAG_D8', 'RES_DC_DIFF_D8'):
        ax_trajectory.plot(data['val_origins'], summary['arms'][arm]['TRAIN_ROLLING_CV']['val']['per_origin_mse'], color=COLORS[arm], label=NAMES[arm])
    ax_trajectory.plot(data['val_origins'], [macro((p.astype(np.float64) - y)**2, m) for p,y,m in zip(data['val_forecast'][:,None], data['val_target'][:,None], data['val_mask'][:,None])], color=COLORS['RAW_F'], linestyle='--', label='Raw F')
    ax_trajectory.set(xlabel='VAL origin (hour index; 24-hour stride)', ylabel='Origin channel-macro MSE', title='Correlated rolling observations; descriptive trajectories')
    ax_trajectory.legend(fontsize=8)
    fig.suptitle('TRAIN-CV-selected readouts: blue = improvement; red = damage')
    save(fig, '04_channel_origin_sensitivity')


if __name__ == '__main__':
    result = read(HERE / 'result.json')
    data = load(result['shared_inputs'])
    summary = build_summary(result, data)
    (HERE / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    figures(result, data, summary)
    print('Wrote summary.json and four PNG/PDF figure pairs in ' + str(HERE / 'figures'))
