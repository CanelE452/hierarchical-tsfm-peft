"""Refine two report figures while preserving the sealed v1 science and CSVs."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import time
import unicodedata

PROCESS_STARTED=time.perf_counter()
import numpy as np
import psutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATASETS = ('jena', 'robin', 'appliances', 'beijing_aotizhongxin')
ARCHITECTURES = ('DIRECT_PCA', 'ANCHOR_PCA', 'DIRECT_TEACHER', 'ANCHOR_TEACHER', 'ANCHOR_POINTWISE', 'ANCHOR_NONLINEAR_D')
AUXILIARIES = ('BASE_Q', 'LATENT_FUTURE_MSE', 'SUMMARY_MSE', 'SHUFFLED_SUMMARY', 'FIXED_REGIME', 'LEARNED_REGIME', 'SHUFFLED_REGIME')
DISPLAY = {'jena': 'Jena (development VAL)', 'robin': 'Robin (development VAL)',
           'appliances': 'Appliances (sealed EVAL)', 'beijing_aotizhongxin': 'Beijing Aotizhongxin (sealed EVAL)'}
SHORT = {'DIRECT_PCA': 'Direct PCA', 'ANCHOR_PCA': 'Anchor PCA', 'DIRECT_TEACHER': 'Direct teacher',
         'ANCHOR_TEACHER': 'Anchor teacher', 'ANCHOR_POINTWISE': 'Pointwise E', 'ANCHOR_NONLINEAR_D': 'Nonlinear D',
         'BASE_Q': 'Base', 'LATENT_FUTURE_MSE': 'Latent MSE', 'SUMMARY_MSE': 'Summary MSE',
         'SHUFFLED_SUMMARY': 'Shuffled summary', 'FIXED_REGIME': 'Random regime',
         'LEARNED_REGIME': 'Learned regime', 'SHUFFLED_REGIME': 'Shuffled regime'}
COLORS = ['#0072B2', '#009E73', '#E69F00', '#CC79A7', '#56B4E9', '#D55E00', '#6A51A3']
PEAK_SAMPLED_RSS=0


def sample_rss():
    global PEAK_SAMPLED_RSS
    PEAK_SAMPLED_RSS=max(PEAK_SAMPLED_RSS,psutil.Process().memory_info().rss)
    if PEAK_SAMPLED_RSS>8*1024**3:
        raise RuntimeError('REPORT_PROCESS_RSS_OVER_8GIB')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def relative(path, folder):
    import os
    return Path(os.path.relpath(Path(path).resolve(), folder.resolve())).as_posix()


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def table(headers, rows):
    if not rows:
        return '해당 결과는 아직 없다.'
    width = lambda text: sum(2 if unicodedata.east_asian_width(c) in 'FW' else 1 for c in str(text))
    widths = [max(width(h), *(width(row[i]) for row in rows)) for i, h in enumerate(headers)]
    line = lambda row: '  '.join(str(v) + ' ' * (widths[i] - width(v)) for i, v in enumerate(row)).rstrip()
    return '```text\n' + line(headers) + '\n' + '\n'.join(line(row) for row in rows) + '\n```'


def method(case, name='BOLT'):
    return next((m for m in case.get('methods', []) if m['method'] == name), None)


def value(record, split, metric='mse', scale='normalized'):
    score = record['scores'][split]
    return float(score['normalized'][metric] if scale == 'normalized' else score[metric])


def eval_split(case):
    return 'eval' if any('eval' in m['scores'] for m in case.get('methods', [])) else 'val'


def load_terminal(path):
    record = read(path)
    if str(record.get('status', '')).upper() in ('RUNNING', 'STARTED', 'IN_PROGRESS'):
        raise ValueError('ACTIVE_RECEIPT_NOT_ALLOWED: ' + str(path))
    return record


def combine(dev, new):
    cases, seen = [], set()
    for result in (dev, new or {}):
        for case in result.get('cases', []):
            identity = (case['dataset'], case['label'])
            if identity in seen:
                raise ValueError('DUPLICATE_CASE: ' + str(identity))
            seen.add(identity)
            cases.append(case)
    return cases


def score_rows(cases):
    rows = []
    for case in cases:
        for record in case.get('methods', []):
            for split in record['scores']:
                rows.append({'dataset': case['dataset'], 'stage': case['stage'], 'case': case['label'],
                    'architecture': case.get('architecture'), 'seed': case.get('seed'),
                    'auxiliary': case.get('aux_method'), 'weight': case.get('aux_weight'), 'K': case.get('K'),
                    'method': record['method'], 'split': split,
                    'selected_update': case.get('training', {}).get('selected_update'),
                    'normalized_mse': value(record, split), 'normalized_mae': value(record, split, 'mae'),
                    'physical_mse': value(record, split, scale='physical'),
                    'physical_mae': value(record, split, 'mae', 'physical'),
                    'prediction_path': record['prediction']['path'], 'prediction_sha256': record['prediction']['sha256']})
    return rows


def training_rows(cases):
    rows, checkpoint_rows, curve_rows = [], [], []
    for case in cases:
        train = case.get('training')
        if train is None:
            continue
        common = {'dataset': case['dataset'], 'stage': case['stage'], 'case': case['label'],
                  'architecture': case.get('architecture'), 'seed': case.get('seed'),
                  'auxiliary': case.get('aux_method'), 'weight': case.get('aux_weight')}
        rows.append({**common, 'updates': train['updates'], 'selected_update': train['selected_update'],
            'train_wall_seconds': train['train_wall_seconds'], 'initial_codec_hash': train['initial_codec_hash'],
            'order_sha256': train['order_sha256'], **case.get('capacity', {}),
            'q_registered_parameters': train.get('q_registered_parameters'),
            'q_active_loss_gradient': train.get('q_active_loss_gradient'),
            'auxiliary_fit_wall_seconds': case.get('teacher_auxiliary_fit_wall_seconds')})
        for cp in train['checkpoints']:
            for split, score in cp['forecast'].items():
                checkpoint_rows.append({**common, 'update': cp['update'], 'split': split,
                    'normalized_mse': score['normalized']['mse'], 'normalized_mae': score['normalized']['mae'],
                    'physical_mse': score['mse'], 'physical_mae': score['mae'], 'checkpoint_sha256': cp['sha256']})
        curve_rows.extend({**common, **point} for point in train['curve'])
    return rows, checkpoint_rows, curve_rows


def control_training_rows(cases):
    rows = []
    for case in cases:
        for record in case.get('methods', []):
            for key in ('native_training', 'correction', 'selection'):
                fitted = record.get(key)
                if not isinstance(fitted, dict):
                    continue
                rows.append({'dataset': case['dataset'], 'case': case['label'], 'seed': case.get('seed'),
                    'method': record['method'], 'fit_role': key, 'train_wall_seconds': fitted.get('train_wall_seconds'),
                    'updates': fitted.get('updates'), 'selected_update': fitted.get('selected_update'),
                    'selected_lambda': fitted.get('selected_lambda', fitted.get('lambda')),
                    'registered_parameters': fitted.get('registered_parameters', fitted.get('parameters')),
                    'E_D_optimized_for_native': fitted.get('E_D_optimized_for_native'),
                    'FIT_base_predictions_in_sample': fitted.get('FIT_base_predictions_in_sample'),
                    'selection_rule': fitted.get('selection_rule')})
    return rows


def diagnostics_rows(cases):
    diagnostics, geometry, oracles, channels = [], [], [], []
    for case in cases:
        common = {'dataset': case['dataset'], 'stage': case['stage'], 'case': case['label'],
                  'architecture': case.get('architecture'), 'seed': case.get('seed'),
                  'auxiliary': case.get('aux_method'), 'weight': case.get('aux_weight')}
        if 'alignment' in case:
            meta, diag = case['alignment']['metadata'], case['alignment']['FIT_diagnostics']
            row = {**common, **{k: v for k, v in diag.items() if not isinstance(v, (dict, list))},
                   'student_variance': json.dumps(diag.get('student_variance'))}
            for name in ('fixed', 'learned'):
                row.update({name + '_' + k: json.dumps(v) if isinstance(v, list) else v for k, v in diag.get(name, {}).items()})
            row.update({k: json.dumps(v) if isinstance(v, list) else v for k, v in meta.items()
                        if k in ('valid_horizons', 'total_horizons', 'complete_summary_origins', 'shuffle_available',
                                 'shuffle_failure_reason', 'shuffle_min_origin_distance_complete_subset',
                                 'observed_basis_condition_quantiles', 'target_latent_absolute_quantiles')})
            diagnostics.append(row)
            for name in ('fixed', 'learned'):
                distance = np.asarray(meta.get(name + '_pairwise_distances', []), float)
                nonzero = distance[np.triu_indices(len(distance), 1)] if len(distance) else np.array([])
                geometry.append({**common, 'prototype_kind': name, 'occupancy': json.dumps(meta.get(name + '_occupancy')),
                    'center_norms': json.dumps(meta.get(name + '_center_norms')),
                    'mean_pairwise_distance': float(nonzero.mean()) if len(nonzero) else None,
                    'minimum_pairwise_distance': float(nonzero.min()) if len(nonzero) else None,
                    'teacher_entropy': diag.get(name, {}).get('teacher_entropy'),
                    'student_entropy': diag.get(name, {}).get('student_entropy')})
        for point, key in (('u0', 'initial_oracle'), ('selected', 'oracle')):
            for split, record in case.get(key, {}).items():
                if not isinstance(record, dict) or 'same_D_direct_and_anchor_projection_residuals' not in record:
                    continue
                for mode, score in record['same_D_direct_and_anchor_projection_residuals'].items():
                    oracles.append({**common, 'checkpoint_role': point, 'split': split, 'projection_mode': mode,
                                    'normalized_mse': score['mse'], 'normalized_mae': score['mae'],
                                    'fixed_D_rank': record['fixed_D_rank'], 'diagnostic_uses_actual_future': True})
        for record in case.get('methods', []):
            for split, score in record['scores'].items():
                for index, (mse, mae, count) in enumerate(zip(score['normalized']['per_channel_mse'],
                        score['normalized']['per_channel_mae'], score['normalized']['observed_counts'])):
                    channels.append({**common, 'method': record['method'], 'split': split, 'channel_index': index,
                        'normalized_mse': mse, 'normalized_mae': mae, 'observed_count': count,
                        'physical_mse': score['per_channel_mse'][index], 'physical_mae': score['per_channel_mae'][index]})
    return diagnostics, geometry, oracles, channels


def selected_aux_cases(cases, dev, dataset):
    candidates = {c['name']: c['chosen_weight'] for c in dev.get('S2_selection', {}).get('candidates', [])}
    return [c for c in cases if c['dataset'] == dataset and c.get('aux_method') in AUXILIARIES
            and c['stage'] in ('S2', 'S3') and c.get('aux_weight') == candidates.get(c['aux_method'])]


def same_weight_contrasts(cases):
    rows=[]
    for aligned,control in (('SUMMARY_MSE','SHUFFLED_SUMMARY'),('LEARNED_REGIME','SHUFFLED_REGIME'),('LEARNED_REGIME','FIXED_REGIME')):
        for case in cases:
            if case['stage']!='S2' or case.get('aux_method')!=aligned:
                continue
            matches=[c for c in cases if c['stage']=='S2' and c['dataset']==case['dataset'] and c['seed']==case['seed']
                     and c.get('aux_method')==control and c.get('aux_weight')==case['aux_weight']]
            if len(matches)!=1:
                continue
            compared=matches[0]
            for split in ('fit','tune','val'):
                a,b=method(case),method(compared)
                row={'dataset':case['dataset'],'seed':case['seed'],'aux_weight':case['aux_weight'],
                     'aligned':aligned,'control':control,'split':split,'aligned_case':case['label'],'control_case':compared['label'],
                     'same_initial_codec':case['training']['initial_codec_hash']==compared['training']['initial_codec_hash'],
                     'same_batch_order':case['training']['order_sha256']==compared['training']['order_sha256'],
                     'aligned_selected_update':case['training']['selected_update'],'control_selected_update':compared['training']['selected_update']}
                for metric in ('mse','mae'):
                    for scale in ('normalized','physical'):
                        av,bv=value(a,split,metric,scale),value(b,split,metric,scale)
                        row['aligned_'+scale+'_'+metric]=av;row['control_'+scale+'_'+metric]=bv
                        row[scale+'_'+metric+'_change_percent']=100*(av/bv-1) if bv else None
                rows.append(row)
    return rows


def s1_cases(cases, dataset):
    return [c for c in cases if c['dataset'] == dataset and c['stage'] in ('S1', 'S3') and c.get('aux_method') == 'MAIN']


def average(cases, split, name='BOLT', metric='mse'):
    records = [method(c, name) for c in cases]
    scores = [value(m, split, metric) for m in records if m is not None and split in m['scores']]
    return float(np.mean(scores)) if scores else np.nan


def raw_record(cases, dataset):
    controls = [c for c in cases if c['dataset'] == dataset and c['label'] == 'RAW_CONTROLS']
    found = controls or [c for c in cases if c['dataset'] == dataset and c['stage'] == 'RAW']
    return method(found[0], 'RAW_BOLT') if found else None


def plt_style():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 11,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .18,
        'svg.fonttype': 'none', 'savefig.facecolor': 'white'})


def save_figure(fig, folder, name, caption):
    if name not in ('alignment_mechanism','accuracy_cost'):
        receipt=read(HERE/'report_v1/REPORT_MANIFEST.json')
        for extension in ('png','svg'):
            source=HERE/'report_v1'/(name+'.'+extension)
            assert sha(source)==receipt['outputs'][source.name]
            with (folder/source.name).open('xb') as handle:handle.write(source.read_bytes())
        plt.close(fig)
        return
    fig.suptitle(caption, fontsize=13, fontweight='bold')
    for extension in ('png', 'svg'):
        fig.savefig(folder / (name + '.' + extension), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def series(ax, groups, labels, split, raw=None, metric='mse'):
    for index, group in enumerate(groups):
        points = [value(method(c), split, metric) for c in group if method(c) and split in method(c)['scores']]
        if points:
            ax.bar(index, np.mean(points), color=COLORS[index % len(COLORS)], width=.64, alpha=.75)
            ax.scatter(np.full(len(points), index) + np.linspace(-.08, .08, len(points)), points, c='black', s=16, zorder=3)
    if raw is not None:
        ax.axhline(raw, color='black', linestyle='--', linewidth=1.1, label='RAW Bolt')
        ax.legend(fontsize=8)
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha='right')
    ax.set_ylabel('Normalized channel-macro ' + metric.upper())


def figures(cases, dev, costs, diagnostics, pairs, out):
    plt_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    for col, dataset in enumerate(DATASETS[:2]):
        all_s1 = s1_cases(cases, dataset)
        groups = [[c for c in all_s1 if c['architecture'] == name] for name in ARCHITECTURES]
        raw = raw_record(cases, dataset)
        series(axes[0, col], groups, [SHORT[n] for n in ARCHITECTURES], 'val', value(raw, 'val') if raw else None)
        axes[0, col].set_title(DISPLAY[dataset] + ' | selected by TUNE')
        for index, group in enumerate(groups):
            for c in group:
                cp0 = c['training']['checkpoints'][0]['forecast']['tune']['normalized']['mse']
                selected = c['training']['selected_checkpoint']['forecast']['tune']['normalized']['mse']
                x = index + (-.06 if c['seed'] == 92601 else .06)
                axes[1, col].plot([x, x], [cp0, selected], color=COLORS[index % 7], linewidth=2)
                axes[1, col].scatter([x], [cp0], facecolors='none', edgecolors=COLORS[index % 7], s=34)
                axes[1, col].scatter([x], [selected], color=COLORS[index % 7], s=25)
        axes[1, col].set_xticks(range(6), [SHORT[n] for n in ARCHITECTURES], rotation=35, ha='right')
        axes[1, col].set_ylabel('TUNE norm MSE: open u0 / filled selected')
    save_figure(fig, out, 's1_architecture', 'S1 | separate structure at u0 from added learning')

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for dataset, ax in zip(DATASETS, axes.flat):
        split = 'val' if dataset in DATASETS[:2] else 'eval'
        chosen = selected_aux_cases(cases, dev, dataset)
        baseline = {c['seed']: value(method(c), split) for c in chosen if c['aux_method'] == 'BASE_Q' and split in method(c)['scores']}
        for index, name in enumerate(AUXILIARIES):
            ratio = [100 * (value(method(c), split) / baseline[c['seed']] - 1) for c in chosen
                     if c['aux_method'] == name and c['seed'] in baseline and split in method(c)['scores']]
            if ratio:
                ax.bar(index, np.mean(ratio), color=COLORS[index % 7], alpha=.75)
                ax.scatter(np.full(len(ratio), index) + np.linspace(-.07, .07, len(ratio)), ratio, c='black', s=18)
        ax.axhline(0, color='black', linewidth=1)
        ax.set_title(DISPLAY[dataset]); ax.set_ylabel('Norm MSE change vs same-seed BASE_Q (%)')
        weights={c['name']:c['chosen_weight'] for c in dev.get('S2_selection',{}).get('candidates',[])}
        ax.set_xticks(range(7), [SHORT[n]+'\nweight '+str(weights.get(n,'?')) for n in AUXILIARIES], rotation=40, ha='right')
    save_figure(fig, out, 's2_alignment', 'Selected-weight pipelines | weights differ across methods; negative is better')

    fig, axes = plt.subplots(4, 3, figsize=(14.5, 15.5), constrained_layout=True)
    chosen_ids = {(c['dataset'], c['label']) for d in DATASETS for c in selected_aux_cases(cases, dev, d)}
    selected_diag = [r for r in diagnostics if (r['dataset'], r['case']) in chosen_ids]
    aliases=('Base','LatMSE','SumMSE','ShufSum','Fixed','Learned','ShufReg')
    dataset_short={'jena':'Jena','robin':'Robin','appliances':'Appliances','beijing_aotizhongxin':'Beijing Aotizhongxin'}
    for row_index,dataset in enumerate(DATASETS):
        for row in selected_diag:
            if row['dataset']!=dataset:continue
            index=AUXILIARIES.index(row['auxiliary'])
            x=index+(-.13 if row['seed']==92601 else .13)
            marker='o' if row['seed']==92601 else 's'
            color=COLORS[index]
            before=row.get('before_q_latent_future_normalized_mse')
            after=row.get('after_q_latent_future_normalized_mse')
            if before is not None and after is not None:
                axes[row_index,0].plot([x,x],[before,after],color=color,linewidth=1.6)
                axes[row_index,0].scatter(x,before,facecolor='white',edgecolor=color,marker=marker,s=45,zorder=3)
                axes[row_index,0].scatter(x,after,color=color,marker=marker,s=28,zorder=4)
            rank=row.get('student_effective_rank')
            if rank is not None:axes[row_index,1].scatter(x,rank,color=color,marker=marker,s=40)
            kl,constant=row.get('learned_student_vs_teacher_kl'),row.get('learned_constant_distribution_kl')
            if kl is not None and constant is not None and constant>0:
                axes[row_index,2].scatter(x,kl/constant,color=color,marker=marker,s=40)
        for ax in axes[row_index]:
            ax.set_xticks(range(7),aliases,rotation=32,ha='right',fontsize=10)
            ax.tick_params(axis='y',labelsize=10)
            ax.set_xlim(-.5,6.5)
        axes[row_index,0].set_title(dataset_short[dataset]+' | q before / after',fontsize=11)
        axes[row_index,0].set_ylabel('FIT latent norm MSE',fontsize=11)
        axes[row_index,1].set_title(dataset_short[dataset]+' | effective rank',fontsize=11)
        axes[row_index,1].set_ylabel('Student rank after q',fontsize=11)
        k=next((c['K'] for c in selected_aux_cases(cases,dev,dataset)),None)
        if k is not None:
            axes[row_index,1].axhline(k,color='#888888',linestyle=':',linewidth=1)
            axes[row_index,1].set_ylim(0,k+.35)
        axes[row_index,2].set_title(dataset_short[dataset]+' | prototype KL',fontsize=11)
        axes[row_index,2].set_ylabel('Student KL / constant KL',fontsize=11)
        axes[row_index,2].axhline(1,color='black',linestyle='--',linewidth=1)
    fig.legend(handles=[
        Line2D([],[],color='#555555',marker='o',linestyle='none',label='92601: left circle'),
        Line2D([],[],color='#555555',marker='s',linestyle='none',label='92602: right square'),
        Line2D([],[],color='#555555',marker='o',markerfacecolor='white',linestyle='none',label='Before q: open'),
        Line2D([],[],color='#555555',marker='o',linestyle='none',label='After q: filled')],
        loc='upper center',bbox_to_anchor=(.5,-.005),ncol=4,frameon=False,fontsize=11)
    save_figure(fig,out,'alignment_mechanism','FIT mechanism by dataset | per-method frozen weights; not deployment success')

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    selected_arch = dev.get('S1_selection', {}).get('selected')
    names = list(dict.fromkeys(('DIRECT_PCA', 'ANCHOR_PCA', selected_arch, *AUXILIARIES)))
    names = [n for n in names if n]
    for col, dataset in enumerate(DATASETS[2:]):
        groups = [[c for c in s1_cases(cases, dataset) + selected_aux_cases(cases, dev, dataset)
                   if (c['architecture'] if c.get('aux_method') == 'MAIN' else c['aux_method']) == name] for name in names]
        raw = raw_record(cases, dataset)
        for row, metric in enumerate(('mse', 'mae')):
            series(axes[row, col], groups, [SHORT.get(n, n) for n in names], 'eval',
                   value(raw, 'eval', metric) if raw and 'eval' in raw['scores'] else None, metric)
            axes[row, col].set_title(DISPLAY[dataset])
    save_figure(fig, out, 'new_data_accuracy', 'New-data EVAL | dataset locks; declared joint-gate deviation')

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), constrained_layout=True)
    for dataset, ax in zip(DATASETS, axes.flat):
        split = 'val' if dataset in DATASETS[:2] else 'eval'
        control = [c for c in cases if c['dataset'] == dataset and c['stage'] == 'CONTROLS']
        grouped = {}
        for c in control:
            for m in c['methods']:
                if split not in m['scores']: continue
                prefix = 'Raw' if c['label'] == 'RAW_CONTROLS' else ('Alignment' if 'LEARNED_REGIME' in c['source_case'] else 'Baseline')
                label=prefix + '/' + m['method'].replace('RAW_', '').replace('BOLT_', '')
                grouped.setdefault(label,[]).append(value(m,split))
        labels=list(grouped)
        points=[np.mean(grouped[label]) for label in labels]
        ax.bar(range(len(points)), points, color=[COLORS[i % 7] for i in range(len(points))], alpha=.8)
        for index,label in enumerate(labels):
            actual=grouped[label]
            ax.scatter(np.full(len(actual),index)+np.linspace(-.06,.06,len(actual)),actual,color='black',s=15,zorder=3)
        ax.set_xticks(range(len(labels)), labels, rotation=55, ha='right', fontsize=7)
        ax.set_ylabel('Normalized channel-macro MSE'); ax.set_title(DISPLAY[dataset])
    save_figure(fig, out, 'predictor_controls', 'Raw, native and residual controls | bars means, dots seeds; E/D Bolt-trained')

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    case_map = {(c['dataset'], c['label']): c for c in cases}
    selected_pipeline_ids={(c['dataset'],c['label']) for dataset in DATASETS for c in selected_aux_cases(cases,dev,dataset)}
    for dataset, ax in zip(DATASETS, axes.flat):
        split = 'val' if dataset in DATASETS[:2] else 'eval'; raw = raw_record(cases, dataset)
        for point in costs:
            if point['dataset'] != dataset: continue
            c = case_map.get((dataset, point['case'])); m = method(c, point['method']) if c else None
            if m is None or split not in m['scores']: continue
            if c['stage']=='S2' and (dataset,c['label']) not in selected_pipeline_ids:continue
            ratio = value(m, split) / value(raw, split) if raw else value(m, split)
            color = '#000000' if point['method'].startswith('RAW_') else ('#D55E00' if 'LEARNED_REGIME' in c['label'] else '#0072B2')
            marker = 's' if 'NATIVE' in point['method'] else ('^' if point['method'].endswith('_R') else 'o')
            ax.scatter(point['median_ms_per_window'], ratio, edgecolor=color,
                       facecolor=color if point['batch']==8 else 'none', marker=marker, s=35, alpha=.8)
            if point['batch']==8 and point['method'] in ('RAW_BOLT', 'RAW_SELF33', 'RAW_JOINT_RIDGE'):
                ax.annotate(point['method'], (point['median_ms_per_window'], ratio), xytext=(3, 3), textcoords='offset points', fontsize=7)
        ax.axhline(1, color='black', linestyle='--'); ax.set_title(DISPLAY[dataset])
        ax.set_xlabel('Median deployment ms / window: open B1, filled B8'); ax.set_ylabel('Normalized MSE / RAW Bolt')
    fig.legend(handles=[
        Patch(facecolor='#000000',label='Raw models'),
        Patch(facecolor='#0072B2',label='Other encoded paths'),
        Patch(facecolor='#D55E00',label='Learned-regime paths'),
        Line2D([],[],color='#777777',marker='o',linestyle='none',label='Circle: Bolt / raw standalone'),
        Line2D([],[],color='#777777',marker='s',linestyle='none',label='Square: native, including +R'),
        Line2D([],[],color='#777777',marker='^',linestyle='none',label='Triangle: Bolt + raw R'),
        Line2D([],[],color='#777777',marker='o',markerfacecolor='white',linestyle='none',label='Open: batch 1'),
        Line2D([],[],color='#777777',marker='o',linestyle='none',label='Filled: batch 8')],
        loc='upper center',bbox_to_anchor=(.5,-.005),ncol=4,frameon=False,fontsize=10)
    save_figure(fig, out, 'accuracy_cost', 'Selected-weight deployment paths | black raw, blue other codecs, red learned; q removed')

    fig, axes = plt.subplots(2, 4, figsize=(20, 8), constrained_layout=True)
    for col, dataset in enumerate(DATASETS):
        relevant = [r for r in pairs if r['dataset'] == dataset]
        for kind, ax in (('time_quarter', axes[0, col]), ('channel', axes[1, col])):
            selected = [r for r in relevant if r['partition'] == kind]
            names = sorted({r['partition_index'] for r in selected})
            for index in names:
                vals = [r['mse_change_percent'] for r in selected if r['partition_index'] == index]
                ax.scatter([index] * len(vals), vals, s=18, color='#D55E00', alpha=.8)
                if vals: ax.plot([index-.2, index+.2], [np.mean(vals)]*2, color='black', linewidth=1.3)
            ax.axhline(0, color='black', linestyle='--'); ax.set_title(DISPLAY[dataset])
            ax.set_xlabel('Chronological quarter (0..3)' if kind == 'time_quarter' else 'Channel index (CSV maps columns)')
            ax.set_ylabel('LEARNED_REGIME vs same-seed S1 MSE change (%)')
        if not relevant:
            axes[0, col].text(.5, .5, 'No permitted saved-prediction diagnostics yet', ha='center', transform=axes[0, col].transAxes)
    save_figure(fig, out, 'paired_time_channel', 'Paired diagnostics | overlapping histories and temporal dependence limit inference')


def macro(prediction, target, mask):
    residual = np.asarray(prediction, np.float64) - np.asarray(target, np.float64)
    count = mask.sum(axis=(0, 1)); valid = count > 0
    mse = (np.where(mask, residual * residual, 0).sum(axis=(0, 1)) / np.maximum(count, 1))[valid]
    mae = (np.where(mask, np.abs(residual), 0).sum(axis=(0, 1)) / np.maximum(count, 1))[valid]
    return float(mse.mean()), float(mae.mean())


def predictions_and_pairs(cases, dev, allowed):
    if not allowed:
        return [], []
    checks, pairs = [], []
    cache = {}
    def load(m):
        path = ROOT / m['prediction']['path']
        if sha(path) != m['prediction']['sha256']: raise ValueError('PREDICTION_SHA_CHANGED: ' + str(path))
        with np.load(path, allow_pickle=False) as z: arrays={k: z[k] for k in z.files}
        sample_rss()
        return arrays
    for case in cases:
        for m in case.get('methods', []):
            z = load(m)
            for split in m['scores']:
                p, y, mask = (z[split + '_' + key] for key in ('prediction', 'target', 'observed'))
                nmse, nmae = macro(p, y, mask)
                pmse, pmae = macro(p*z['std']+z['mean'], y*z['std']+z['mean'], mask)
                errors = {'normalized_mse': abs(nmse-value(m, split)), 'normalized_mae': abs(nmae-value(m, split, 'mae')),
                          'physical_mse': abs(pmse-value(m, split, scale='physical')), 'physical_mae': abs(pmae-value(m, split, 'mae', 'physical'))}
                if any(v > 2e-7*max(1., abs(value(m, split, k.split('_')[-1], 'physical' if k.startswith('physical') else 'normalized'))) for k, v in errors.items()):
                    raise ValueError('REPORT_RESCORE_DIFFERS: ' + case['label'] + '/' + m['method'] + '/' + split + str(errors))
                checks.append({'dataset': case['dataset'], 'case': case['label'], 'method': m['method'], 'split': split, **errors})
            if m['method'] == 'BOLT' and case.get('aux_method') in ('MAIN', 'LEARNED_REGIME'):
                cache[(case['dataset'], case['label'])] = z
    selected_arch = dev['S1_selection']['selected']
    for dataset in DATASETS:
        base = {c['seed']: c for c in s1_cases(cases, dataset) if c['architecture'] == selected_arch}
        aligned = [c for c in selected_aux_cases(cases, dev, dataset) if c['aux_method'] == 'LEARNED_REGIME']
        split = 'val' if dataset in DATASETS[:2] else 'eval'
        for c in aligned:
            b = base.get(c['seed'])
            if b is None: continue
            x, y = cache[(dataset, b['label'])], cache[(dataset, c['label'])]
            for key in ('target', 'observed', 'origins'):
                if not np.array_equal(x[split+'_'+key], y[split+'_'+key]): raise ValueError('PAIRED_ARRAYS_DIFFER: ' + key)
            n, _, channels = x[split+'_prediction'].shape
            for kind, partitions in (('time_quarter', np.array_split(np.arange(n), 4)), ('channel', range(channels))):
                for index, part in enumerate(partitions):
                    if kind == 'time_quarter':
                        if not len(part): continue
                        p0, p1, truth, observed = (x[split+'_prediction'][part], y[split+'_prediction'][part], x[split+'_target'][part], x[split+'_observed'][part])
                    else:
                        p0, p1, truth, observed = (x[split+'_prediction'][:, :, part:part+1], y[split+'_prediction'][:, :, part:part+1], x[split+'_target'][:, :, part:part+1], x[split+'_observed'][:, :, part:part+1])
                    if not observed.any(): continue
                    mse0, mae0 = macro(p0, truth, observed); mse1, mae1 = macro(p1, truth, observed)
                    pairs.append({'dataset': dataset, 'seed': c['seed'], 'split': split, 'baseline_case': b['label'], 'aligned_case': c['label'],
                        'partition': kind, 'partition_index': index, 'baseline_normalized_mse': mse0, 'aligned_normalized_mse': mse1,
                        'mse_change_percent': 100*(mse1/mse0-1) if mse0 else None, 'baseline_normalized_mae': mae0,
                        'aligned_normalized_mae': mae1, 'observed_positions': int(observed.sum()), 'forecast_origins': len(p0)})
    return checks, pairs


def cost_rows(cost):
    rows = []
    for record in (cost or {}).get('records', []):
        for latency in record['latency']:
            rows.append({'dataset': record['dataset'], 'case': record['label'], 'method': record['method'],
                'batch': latency['batch'], 'median_ms': latency['median_seconds']*1000,
                'median_ms_per_window': latency['median_seconds_per_window']*1000,
                'all_repeat_seconds': json.dumps(latency['seconds']), 'warmup': latency['warmup'], 'repeats': latency['repeats'],
                'physical_max_abs_vs_science': latency['physical_max_abs_vs_science'], 'parity_pass': latency['parity_pass'],
                'peak_allocated_vram_bytes': latency['peak_allocated_vram_bytes'], 'resident_rss_bytes': latency['resident_rss_bytes'],
                'includes': latency['includes'], 'co_resident_memory_note': latency['co_resident_memory_note']})
    return rows


def checkpoint_audit_rows(receipt):
    rows, alignment_rows = [], []
    for kind,key in (('CORE','records'),('NATIVE','native_records')):
        for record in (receipt or {}).get(key,[]):
            score=record['scores']
            rows.append({'dataset':record['dataset'],'case':record['label'],'predictor_role':kind,'update':record['update'],
                'normalized_mse':score['normalized']['mse'],'normalized_mae':score['normalized']['mae'],
                'physical_mse':score['mse'],'physical_mae':score['mae'],
                'normalized_metric_max_abs_vs_record':record['normalized_metric_max_abs_vs_record'],'pass':record['pass'],
                'absolute_tolerance':record['absolute_tolerance'],'relative_tolerance':record['relative_tolerance'],
                'checkpoint_path':record['checkpoint']['path'],'checkpoint_sha256':record['checkpoint']['sha256'],
                'prediction_path':record['prediction']['path'],'prediction_sha256':record['prediction']['sha256']})
    for record in (receipt or {}).get('alignment_records',[]):
        alignment_rows.append({'dataset':record['dataset'],'case':record['label'],'pass':record['pass'],
            'actual_FIT_reinference_wall_seconds':record['actual_FIT_reinference_wall_seconds'],**record['diagnostics'],
            **{'error_'+k:v for k,v in record['max_abs_vs_original'].items()},
            'alignment_source_path':record['alignment_source']['path'],'alignment_source_sha256':record['alignment_source']['sha256'],
            'prediction_path':record['prediction']['path'],'prediction_sha256':record['prediction']['sha256'],
            'selected_checkpoint_path':record['selected_checkpoint']['path'],'selected_checkpoint_sha256':record['selected_checkpoint']['sha256']})
    return rows,alignment_rows


def mean_table(cases, dev, stage):
    rows = []
    for dataset in DATASETS:
        group = s1_cases(cases, dataset) if stage == 'S1' else selected_aux_cases(cases, dev, dataset)
        names = ARCHITECTURES if stage == 'S1' else AUXILIARIES
        split = 'val' if dataset in DATASETS[:2] else 'eval'
        for name in names:
            matched = [c for c in group if (c['architecture'] if stage == 'S1' else c['aux_method']) == name]
            if not matched: continue
            rows.append([dataset, name, ','.join(str(c['seed'])[-1] for c in matched),
                ','.join(str(c['training']['selected_update']) for c in matched),
                f'{average(matched,"tune"):.6f}', f'{average(matched,split):.6f}',
                f'{average(matched,split,metric="mae"):.6f}',
                ','.join(f'{value(method(c),split):.6f}' for c in matched),
                '-' if stage == 'S1' else str(matched[0]['aux_weight'])])
    return table(['자료', '방법', 'seed끝자리', '선택u', 'TUNE normMSE', 'VAL/EVAL normMSE', 'VAL/EVAL normMAE', 'seed별normMSE', '보조weight'], rows)


def narrative(cases, dev, new, cost, audit, diagnostics, pairs, inputs, out, checkpoint_audit=None):
    selected = dev.get('S1_selection', {}).get('selected', '미선택')
    lines = ['# 작은 K의 상태 보존·미래 정렬·새 자료 검증 결과', '',
        '[확인] 정규화 관측 channel-macro MSE가 주지표다. 각 모델의 checkpoint와 보조 weight는 미래 TUNE로 선택했다. 두 seed는 최적화 변동의 제한된 반복이며 독립 자료 표본 두 개를 뜻하지 않는다.', '',
        '**[확인] 원 계획의 두 새자료 공동 모델 선택 barrier는 충족되지 않았다. Appliances EVAL을 먼저 개봉한 후 Beijing 원자료 로더를 수리했다. 아래는 동일 사전 방법과 자료별 lock을 유지한 결과이며 원 공동 평가 순서를 통과한 실험으로 쓰지 않는다.**', '',
        '## 질문에 대한 답', '', '### 작은 K의 문제를 해결했는가?', '']
    for dataset in DATASETS:
        split = 'val' if dataset in DATASETS[:2] else 'eval'; raw = raw_record(cases, dataset)
        group = [c for c in s1_cases(cases, dataset) if c['architecture'] == selected]
        if raw is None or not group or split not in raw['scores']:
            lines.append(f'- {dataset}: 필요한 평가 결과가 없어 이 질문은 미완료다.'); continue
        score = average(group, split); baseline = value(raw, split)
        lines.append(f'- [확인] {dataset}: 사전 TUNE 선택 구조 `{selected}`의 {split.upper()} normMSE 평균 {score:.9f}, RAW Bolt {baseline:.9f}, 변화 {100*(score/baseline-1):+.2f}%. normMAE 평균 {average(group,split,metric="mae"):.9f}, RAW {value(raw,split,"mae"):.9f}. seed별 normMSE는 ' + ', '.join(f'{c["seed"]}:{value(method(c),split):.9f}' for c in group) + '.')
    lines.extend(['','**[판단] 작은 K의 문제가 전반적으로 해결됐다는 결론은 지지되지 않는다. 개선된 압축 baseline과 원채널 RAW/RAW+R 기준을 구분해야 한다.**',''])
    for dataset in DATASETS[2:]:
        raw=raw_record(cases,dataset)
        group=[c for c in selected_aux_cases(cases,dev,dataset) if c['aux_method']=='LEARNED_REGIME']
        control=[c for c in cases if c['dataset']==dataset and c['stage']=='CONTROLS' and 'LEARNED_REGIME' in c.get('source_case','')]
        raw_case=next((c for c in cases if c['dataset']==dataset and c['label']=='RAW_CONTROLS'),None)
        raw_r=method(raw_case,'RAW_BOLT_RAW_R') if raw_case else None
        if not group or raw is None or 'eval' not in raw['scores']: continue
        learned=average(group,'eval');learned_mae=average(group,'eval',metric='mae')
        lines.append(f'- [확인] {dataset} LEARNED_REGIME의 EVAL normMSE {learned:.9f}는 RAW {value(raw,"eval"):.9f} 대비 {100*(learned/value(raw,"eval")-1):+.2f}%, normMAE {learned_mae:.9f}는 RAW {value(raw,"eval","mae"):.9f} 대비 {100*(learned_mae/value(raw,"eval","mae")-1):+.2f}%다.')
        if control and raw_r:
            adjusted=average(control,'eval',name='BOLT_RAW_R');adjusted_mae=average(control,'eval',name='BOLT_RAW_R',metric='mae')
            lines.append(f'  같은 raw R을 붙이면 LEARNED+R normMSE {adjusted:.9f}, RAW+R {value(raw_r,"eval"):.9f}로 {100*(adjusted/value(raw_r,"eval")-1):+.2f}%이며 normMAE는 {adjusted_mae:.9f} 대 {value(raw_r,"eval","mae"):.9f}다. 작은 MSE 차이가 MAE나 강한 기준군 우위까지 뜻하지 않는다.')
    lines.extend(['', '[판단] 이 수치가 작아졌다는 사실과 모든 채널·자료의 문제가 해결됐다는 주장은 다르다. anchor는 원채널 현재 상태 C개를 우회로 보존하고 K성분의 변화만 복원한다. TSFM 입력은 K채널이지만 strict K-only 압축은 아니다. 기존 VAL은 반복 개발 자료이고 새 EVAL만 방법 개발과 분리했다.', '',
        '### 미래 표현 정렬이 필요한가?', ''])
    for dataset in DATASETS:
        split = 'val' if dataset in DATASETS[:2] else 'eval'; chosen = selected_aux_cases(cases, dev, dataset)
        base = [c for c in chosen if c['aux_method'] == 'BASE_Q']
        if not base: continue
        rows = []
        for name in AUXILIARIES[1:]:
            group = [c for c in chosen if c['aux_method'] == name]
            if not group: continue
            ratio = 100*(average(group,split)/average(base,split)-1)
            rows.append([name, group[0]['aux_weight'], f'{average(group,split):.9f}', f'{ratio:+.2f}%', f'{average(group,split,metric="mae"):.9f}'])
        lines.extend([f'**{dataset} — {split.upper()}, BASE_Q normMSE={average(base,split):.9f}**', '',
            table(['방법', '개발봉인weight', 'normMSE평균', 'BASE대비', 'normMAE평균'], rows), ''])
    lines.extend(['[판단] 위 표와 S2 그림은 방법별로 TUNE 선택한 weight를 쓰므로 summary 대 shuffled summary, learned 대 shuffled regime의 weight가 다를 수 있다. 같은 선택 절차를 거친 전체 pipeline 비교이며 pairing 하나의 순수 인과 효과로 읽지 않는다. 기전 비교는 아래 개발 자료의 동일 weight 표와 전체 sweep 원수치에서 따로 본다. continuous latent/summary 회귀와 learned/fixed/shuffled 목적을 함께 비교했다. learned가 더 좋아도 Gaussian 고정 중심과 KMeans 중심의 norm·거리·teacher entropy·gradient scale이 다르므로 학습된 경계만의 인과 효과를 분리했다고 쓰지 않는다. SHUFFLED_SUMMARY와 SHUFFLED_REGIME은 유효 FIT origins subset에서 horizon 중첩 없는 bijection으로 정답을 바꾸는 대조다. q 이후의 정렬 loss 감소가 q 이전 잠재 예측이나 배포 예측 개선을 뜻하지 않는다.', '',
        '### 누출이 있었는가? 무엇이 독립적인 검증인가?', '',
        '[확인] 기존 보호 TEST를 새로 사용하지 않는 계약을 유지했다. 정렬 teacher의 미래 정답은 FIT 보조 목적에서 사용한다. 주 예측 학습은 FIT 미래 정답을 쓰고 checkpoint 선택은 TUNE 미래 정답으로 평가한다. 추론 입력은 원점 이전 실제 과거 창이다. 구현 fixture의 prefix/미래 perturbation 통과는 검사한 경로의 정보 경계를 확인하는 근거이며 성능 개선이나 모든 누출 부재의 일반 증명은 아니다.', '',
        '[확인] 새 UCI 자료는 METHOD_LOCK 뒤 다운로드·분리 저장하고 자료별 FIT/TUNE로 모델을 다시 학습했다. 원 CSV ingestion에서 EVAL 값도 split 저장을 위해 읽었다. 학습 loader는 FIT/TUNE payload만 반환하고 자료별 최종 model/source/contract lock 뒤 EVAL target을 개봉한다. 다만 Beijing 자료 준비가 실패한 상태에서 Appliances 평가를 먼저 진행해 공동 barrier는 실제 위반했다. 수리로 이 과거 사실을 없애지 않는다. 방법·weight·seed·자료 규칙을 바꾸지 않은 자료별 봉인 검증과 원 공동 protocol 준수 여부를 구분한다. 이 검증은 기존 가중치의 zero-shot transfer가 아니며 Chronos-Bolt 사전학습 자료와의 중복은 미확정이다.', '',
        '### Chronos-Bolt를 유지할 근거가 있는가?', ''])
    rows = []
    for c in cases:
        if c['stage'] != 'CONTROLS': continue
        split = eval_split(c)
        for m in c['methods']:
            if split in m['scores']:
                rows.append([c['dataset'], 'raw' if c['label']=='RAW_CONTROLS' else ('learned' if 'LEARNED_REGIME' in c['source_case'] else 'selected-S1'),
                    c.get('seed', '-'), m['method'], f'{value(m,split):.9f}', f'{value(m,split,"mae"):.9f}', f'{value(m,split,"mae","physical"):.9f}'])
    lines.extend([table(['자료', 'E/D조건', 'seed', '예측경로', 'normMSE', 'normMAE', '원단위MAE'], rows), '',
        '[판단] native는 Bolt로 학습된 E/D를 고정하고 shared linear 미래 예측기만 추가 600 updates로 적합했다. native에 최적화한 E/D를 쓰는 최선의 native 비교가 아니므로 Bolt의 보편적 우월성을 주장할 수 없다. 원자료 SELF33와 joint ridge 및 동일 raw residual R을 함께 확인해, 작은 K와 큰 백본의 비용을 지불할 실익이 실제 존재하는지 판단해야 한다.', '',
        '## 비교 계약과 실제 실행 범위', '',
        f'개발 S1 전 자료 공통 선택 구조는 `{selected}`다. 선택 규칙은 자료×두 seed의 TUNE normMSE/RAW TUNE normMSE 비율 log 평균이며 VAL/EVAL을 선택에 사용하지 않는다. S2 global TUNE 선택은 `{dev.get("S2_selection",{}).get("selected","미완료")}`이며 방법별 weight는 개발에서 한 개로 봉인했다. 새 자료에서는 이 winner 하나만 평가하지 않고 사전 지정한 전체 대조를 유지했다.', '',
        '공통 L512/H48, quarter K=max(1,floor(C/4)), FIT256/TUNE64, seeds92601/92602, batch4, AdamW lr0.001/decay0, gradient clip1, updates600, checkpoints0/100/200/400/600이다. 새 자료는 시간순60/20/20, EVAL origins stride48이다. 정답 horizon은 중첩되지 않지만 입력 과거 창은 겹치며 시간 자기상관도 남아 있다. Appliances는10분, Beijing은1시간이므로 L/H의 실제 시간 길이가 다르다.', '',
        table(['입력receipt', '상태', 'core학습fit', 'case수'], [[label, result.get('status','미실행'), sum('training' in c for c in result.get('cases',[])), len(result.get('cases',[]))] for label,result in [('개발',dev),('새자료/controls',new or {}),('최종비용',cost or {})]]), '',
        f'독립 검산 integrity 상태: `{(audit or {}).get("status", "미완료")}`, checks={(audit or {}).get("count","미기록")}, failed={(audit or {}).get("failed_count","미기록")}다. 별도 completion 상태는 `{(audit or {}).get("completion_status","미기록")}`이며 원 공동 barrier 미충족을 integrity PASS로 지우지 않는다. 부분 실행·실패 기록은 성공한 case와 합쳐 완료라고 부르지 않는다.', ''])
    fixture_rows=[]
    for dataset, fixture in dev.get('fixtures',{}).items():
        fixture_rows.append([dataset, fixture.get('identity_anchor_vs_RAW_BOLT_max_abs'),
            fixture.get('causal_active_temporal_prefix_max_abs'), fixture.get('existing_quarter_selected_checkpoint_forward_parity_max_abs'),
            fixture.get('missing_window_reset_all_missing_normalized_zero'), fixture.get('backbone_hash_unchanged')])
    oracle_rows=[]
    for case in cases:
        if case['stage']!='S1' or case.get('architecture') not in ('DIRECT_PCA','DIRECT_TEACHER'):
            continue
        diagnostic=case.get('initial_oracle',{}).get('val',{}).get('same_D_direct_and_anchor_projection_residuals',{})
        if diagnostic:
            oracle_rows.append([case['dataset'],case['architecture'],case['seed'],
                f'{diagnostic["DIRECT_AFFINE_Y"]["mse"]:.9f}',f'{diagnostic["ANCHOR_DELTA_Y_MINUS_LAST"]["mse"]:.9f}'])
    lines.extend([table(['자료','C항등anchor parity maxAbs','활성temporal 미래변경maxAbs','이전direct checkpoint parity','전결측reset','F hash보존'],fixture_rows),'',
        '고정 초기 D에서 같은 실제 미래를 두 방식으로 투영한 개발 VAL 진단은 다음과 같다. teacher output의 분산을 잘 설명하는 basis가 실제 미래 변화의 고정-D projection residual까지 줄인다고 보장되지 않는다. 원수치를 보고 두 명제를 분리한다.','',
        table(['자료','초기부분공간','seed','직접Y projection normMSE','현재값+Δ projection normMSE'],oracle_rows),'',
        '## S1: 구조 효과와 학습 효과', '', mean_table(cases,dev,'S1'), ''])
    anchor_rows=[]
    for dataset in DATASETS:
        split='val' if dataset in DATASETS[:2] else 'eval'
        direct=[c for c in s1_cases(cases,dataset) if c['architecture']=='DIRECT_PCA']
        anchored=[c for c in s1_cases(cases,dataset) if c['architecture']=='ANCHOR_PCA']
        if not direct or not anchored:continue
        d,a=average(direct,split),average(anchored,split)
        dm,am=average(direct,split,metric='mae'),average(anchored,split,metric='mae')
        anchor_rows.append([dataset,split,f'{d:.9f}',f'{a:.9f}',f'{100*(a/d-1):+.2f}%',
                            f'{dm:.9f}',f'{am:.9f}',f'{100*(am/dm-1):+.2f}%'])
    lines.extend(['anchor 단독 구조 대조는 같은 PCA 초기 E/D를 쓰는 DIRECT_PCA와 ANCHOR_PCA에서 따로 확인한다. DIRECT_PCA와 선택 teacher+nonlinear decoder anchor의 전체 차이를 anchor 하나의 효과로 부르지 않는다.','',
        table(['자료','평가','DIRECT_PCA MSE','ANCHOR_PCA MSE','anchor MSE변화','DIRECT MAE','ANCHOR MAE','anchor MAE변화'],anchor_rows),'',
        '[확인] Beijing의 anchor PCA는 직접 PCA보다 MSE가 커지고 MAE가 작아지는 방향이다. 네 자료에서 anchor 하나가 주오차를 일괄 해결했다고 말할 수 없다. 선택 baseline은 teacher basis와 nonlinear D도 함께 바뀌는 구조이므로 전체 pipeline의 결과로 해석한다.','',
        '![S1 구조와 학습 분리](s1_architecture.png)', '',
        'u0의 고정 projection/anchor 효과와 선택 checkpoint의 추가 학습 효과를 따로 확인한다. Pointwise E는 각 시점 RF1이지만 F는 여전히512시점 latent 과거를 본다. 전체 예측기가 현재값만 보는 비교가 아니다. TCN E와 linear pointwise E는 시간 결합뿐 아니라 비선형성·파라미터 용량도 달라 시간 정보 하나의 순수 인과 효과를 분리한 실험이라고 부르지 않는다. Nonlinear D는 K→32→C residual로 파라미터가 늘어난 대조다. affine decoder bias는 anchor에서 취소되므로 등록 수와 실제 구조적 활성 수를 구분한다.', '',
        '실제 미래를 고정 D의 열공간에 투영한 [oracle_projection.csv](oracle_projection.csv)는 배포할 수 없는 원인 진단이다. 미래마다 정답을 보고 최소제곱 좌표를 맞췄으며 배포 순위나 보편적 lower bound에 섞지 않는다.', '',
        '## S2: 보조 미래 정렬 전체 결과', '', mean_table(cases,dev,'S2'), '',
        '![미래 정렬과 shuffled 대조](s2_alignment.png)', '',
        '선택된 weight가 다른 결과를 pairing의 기전으로 해석하지 않기 위해 개발 자료 전체 sweep에서 weight0.03/0.1을 각각 고정해 비교했다. 같은 초기 E/D·배치 순서·600 updates·TUNE checkpoint 규칙이며 각 군이 선택한 checkpoint는 다를 수 있다.',''])
    contrast_rows=same_weight_contrasts(cases)
    contrast_table=[]
    for dataset in DATASETS[:2]:
        for aligned,control in (('SUMMARY_MSE','SHUFFLED_SUMMARY'),('LEARNED_REGIME','SHUFFLED_REGIME'),('LEARNED_REGIME','FIXED_REGIME')):
            for weight in (.03,.1):
                group=[r for r in contrast_rows if r['dataset']==dataset and r['aligned']==aligned and r['control']==control and r['aux_weight']==weight and r['split']=='val']
                if not group: continue
                av=np.mean([r['aligned_normalized_mse'] for r in group]);bv=np.mean([r['control_normalized_mse'] for r in group])
                contrast_table.append([dataset,aligned,control,weight,f'{av:.9f}',f'{bv:.9f}',f'{100*(av/bv-1):+.2f}%',
                    ','.join(f'{r["seed"]}:{r["normalized_mse_change_percent"]:+.2f}%' for r in group)])
    lines.extend([table(['자료','정렬군','대조군','동일weight','정렬VAL normMSE','대조VAL normMSE','정렬/대조변화','seed별변화'],contrast_table),'',
        '[same_weight_alignment_contrasts.csv](same_weight_alignment_contrasts.csv)에 FIT/TUNE/VAL·정규화/원단위 MSE/MAE·초기 hash·배치 순서·선택u를 모두 보존했다. 새 EVAL은 개발에서 선택한 weight만 실행했으므로 이 동일-weight 기전 대조를 새 자료에서 반복했다고 쓰지 않는다.','',
        '![q 전후·잠재 분산·constant 분포 대조](alignment_mechanism.png)', '',
        'FIT future coordinates는 관측 채널만으로 rank K 최소제곱을 적합했다. valid horizon과 complete3K summary 수, basis condition quantiles, target latent 극값을 [alignment_diagnostics.csv](alignment_diagnostics.csv)에 보존했다. rank 통과를 안정성 보장으로 해석하지 않는다. teacher/prototype/FIT 표준화는 고정하고 q는 학습에서만 사용한다. BASE_Q의 q는 등록되지만 보조 loss weight0으로 활성 학습 gradient를 받지 않는다.', '',
        'prototype 점유율·중심 norm·pairwise 거리·entropy는 [prototype_geometry.csv](prototype_geometry.csv), 주·보조 gradient norm/cosine은 [training_curves.csv](training_curves.csv)에 있다. 상수 분포 KL보다 나쁜 FIT 분할 정렬이나 effective rank 감소는 기전상의 경고이며 단독 성능 결론으로 바꾸지 않는다. 분산이 모두0인 경우도 구현의 entropy convention으로 effective rank1이 나올 수 있어 채널별 student variance를 함께 확인한다.', ''])
    mechanism_rows=[]
    for dataset in DATASETS:
        chosen=selected_aux_cases(cases,dev,dataset)
        for name in AUXILIARIES:
            group=[c for c in chosen if c['aux_method']==name and 'alignment' in c]
            if not group: continue
            summary=[]
            for metric in ('before_q_latent_future_normalized_mse','after_q_latent_future_normalized_mse',
                           'before_q_complete_summary_normalized_mse','after_q_complete_summary_normalized_mse','student_effective_rank'):
                numbers=[c['alignment']['FIT_diagnostics'].get(metric) for c in group]
                numbers=[v for v in numbers if v is not None]
                summary.append(f'{np.mean(numbers):.6f}' if numbers else '미확인')
            mechanism_rows.append([dataset,name,*summary])
    lines.extend([table(['자료','보조목적','q전 latent MSE','q후 latent MSE','q전 summary MSE','q후 summary MSE','q후 effective rank'],mechanism_rows),'',
        '[판단] q가 추론에서 제거되므로 q후 오차만 줄었다면 보조 head가 좌표 매핑을 흡수했을 가능성을 먼저 본다. q전 잠재 미래 오차·본 예측 오차·새 EVAL이 함께 좋아지는지 확인해야 한다. 반대로 q전 오차가 줄어도 원채널 decoder와의 결합 또는 채널 가중에 따라 본 오차는 나빠질 수 있다. 이 표는 FIT 정답을 사용한 기전 진단이다.','',
        '## 새 두 자료의 자료별 봉인 평가와 공동 순서 편차', '',
        '원 Beijing UCI ZIP SHA는 `b04da438b2f331ac0ffd45aebdfec0d20d2367feb5f6948c4b1f7ce1191e33c4`다. 정상 중첩 PRSA ZIP과 무관한 최상위 `data.csv`·`test.csv`가 함께 있었다. 초기 진단은 최상위 CSV만 보고 정상 PRSA가 없다고 잘못 판단했다. 정정 v2 진단은 중첩 ZIP의 사전 고정 Aotizhongxin member1개·35064행·시간당시각·내부중복0을 확인했다. 원 loader가 station 없는 두503행 CSV까지 붙여1006 NaT가 생겼고 duplicate1005로 실패했다. 정상 관측 행을 지울 문제로 해석하지 않았다.','',
        '별도 `newdata_repair.py`는 member별 필수 schema·station·시각으로 같은 원 ZIP의 Aotizhongxin PRSA만 선택한다. 원 source·실패 결과를 보존했으며 임의 중복 삭제·보간·다른 station·새 원자료·새 weight 탐색을 추가하지 않았다. 수리 전 봉인 note에서 동일11채널·60/20/20·K2·L512/H48·FIT/TUNE origin 규칙·두 seed·600 updates·METHOD_LOCK weight를 유지하도록 고정했다. 기존 legacy/Appliances 성공 fit은 재학습하지 않았다.','',
        '- [수리 전 봉인 실행 note](../BEIJING_REPAIR_EXECUTION_NOTE_KO.md)',
        '- [초기 진단 v1: 잘못된 결론 보존](../beijing_data_diagnosis/current_raw_metadata.json)',
        '- [정정 진단 v2: 중첩 member/schema/날짜 확인](../beijing_data_diagnosis/current_raw_metadata_v2.json)','',
        '최종 resolved manifest는 원 PARTIAL과 Beijing 수리 결과의 SHA·시간·추가 비용을 연결한다. 원 공동 barrier `joint_models_before_any_new_EVAL=false`는 완료 범위와 별개로 남긴다. Appliances EVAL 후에 원자료 로더 수리가 이루어졌다는 편차가 있으므로 “두 새자료가 어떤 EVAL도 보기 전에 공동으로 모두 봉인됐다”는 문장을 쓰지 않는다.','',
        '![독립 새자료 주지표와 MAE](new_data_accuracy.png)', '',
        '![raw/native/R 예측기 분리](predictor_controls.png)', '',
        '![시점·채널 paired 진단](paired_time_channel.png)', '',
        '이 paired 그림은 사전에 고정한 S1 선택 baseline과 LEARNED_REGIME을 같은 seed·정답·관측 mask·origin에서 비교한다. EVAL에서 좋아 보이는 다른 정렬 방법으로 대조를 바꾸지 않았다. 네 시간 구간과 채널 차이는 탐색 진단이며 유의성 검정이나 독립 표본 추론으로 쓰지 않는다.', '',
        '## 정확도·MAE·실제 비용', '',
        '![실제 배포 비용과 오차](accuracy_cost.png)', '',
        '비용은 모든 Python GPU 학습 이후 batch1/8, warmup2/repeat7/synchronize로 CPU512 causal FF/FIT 정규화/H2D/E/F/D/anchor/선택R/역정규화/D2H 전체 경로를 측정한다. 모델 로드와 disk read는 제외하고 로드 시간은 terminal receipt에 남긴다. alignment teacher/q/prototype은 추론에서 제거하지만 학습 및 적합 시간은 별도 포함한다. 공존 Bolt/model의 allocated VRAM·RSS를 exclusive 증분 메모리 절감이라고 주장하지 않는다.', '',
        '비용 측정의 실제 output 배열은 저장하지 않았다. timing-path parity는 producer 실제 실행·source·model binding·receipt의 기록 maxdiff를 검산한 것이며 timing output 배열을 독립적으로 다시 재채점한 것과 다르다. 저장 science prediction NPZ와 CP/q 재추론 NPZ는 별도로 독립 재채점했다.','',
        f'제공된 비용 receipt의 실제 방법 경로 수는 {len((cost or {}).get("records",[]))}, batch1/8 측정 row 수는 {sum(len(r.get("latency",[])) for r in (cost or {}).get("records",[]))}다. 개발 RAW stage의 Jena/Robin RAW_BOLT2개는 RAW_CONTROLS의 같은 배포 경로와 중복되어 비용은 canonical RAW_CONTROLS에 연결한다. 전체 science 방법 기록에서 이2개를 삭제하거나 별개 학습으로 세지 않는다. 짧은7회 벤치마크는 운영 환경의 장기 처리량·동시 부하 성능을 보장하지 않는다.','',
        '원단위 channel macro는 서로 다른 단위의 채널을 평균한 보조 수치다. 사용자가 선택한 정규화 MSE 우위가 정규화 MAE 또는 원단위 MAE 우위를 보장하지 않는다. [all_methods.csv](all_methods.csv), [training.csv](training.csv), [control_training.csv](control_training.csv), [batch_cost.csv](batch_cost.csv)에서 모두 확인한다.', ''])
    timing_summary=[]
    measured=cost_rows(cost)
    for dataset in DATASETS:
        groupings=[('RAW_BOLT',[c for c in cases if c['dataset']==dataset and c['label']=='RAW_CONTROLS']),
            ('selected S1',[c for c in s1_cases(cases,dataset) if c['architecture']==selected]),
            ('LEARNED_REGIME',[c for c in selected_aux_cases(cases,dev,dataset) if c['aux_method']=='LEARNED_REGIME'])]
        for label,group in groupings:
            names={c['label'] for c in group};method_name='RAW_BOLT' if label=='RAW_BOLT' else 'BOLT'
            fields=[]
            for batch in (1,8):
                values=[r['median_ms_per_window'] for r in measured if r['dataset']==dataset and r['case'] in names and r['method']==method_name and r['batch']==batch]
                fields.append(f'{np.mean(values):.6f}' if values else '미측정')
            timing_summary.append([dataset,label,*fields])
    lines.extend([table(['자료','고정비교경로','B1 median ms/window 평균','B8 median ms/window 평균'],timing_summary),'',
        '선택 S1과 LEARNED_REGIME은 두 seed 경로의 median 평균이며 RAW_BOLT는 canonical 한 개 경로의 median이다. 각 경로는7회 실제 wall 측정이다. 이 반복을 독립 자료 표본이나 통계적 유의성으로 해석하지 않는다. 정확도·MAE 손해와 batch별 지연 차이를 함께 판단한다.',''])
    lines.extend(['','## 모든 checkpoint의 실제 재추론 검산',''])
    if checkpoint_audit is None:
        lines.append('이 보고서에는 checkpoint 재추론 terminal receipt가 제공되지 않았다. 해당 검산을 실제 완료했다고 쓰지 않는다.')
    else:
        expected_core=sum(len(c['training']['checkpoints']) for c in cases if 'training' in c)
        expected_native=sum(len(m['native_training']['checkpoints']) for c in cases for m in c['methods'] if 'native_training' in m)
        expected_alignment=sum('alignment' in c for c in cases)
        lines.extend([f'[확인] 실제 재추론 receipt 상태는 `{checkpoint_audit.get("status")}`다. 새 학습 update 수는 `{checkpoint_audit.get("new_training_updates")}`다. 기존 saved checkpoint를 다시 읽고 미래 TUNE를 실제 F/native forward로 예측했으며, 정렬 q전·후는 선택 checkpoint의 FIT 잠재 미래를 다시 추론해 NPZ로 저장했다.','',
            table(['검산범위','실제record','scope기대','불통과record'],[[label,len(checkpoint_audit.get(key,[])),expected,sum(not r['pass'] for r in checkpoint_audit.get(key,[]))]
                  for label,key,expected in [('core TUNE checkpoint','records',expected_core),('native TUNE checkpoint','native_records',expected_native),('FIT q전후 잠재좌표','alignment_records',expected_alignment)]]),'',
            'core/native metric parity 허용은 absolute1e-6+relative2e-6, FIT q전후 진단은 absolute2e-5+relative2e-6이다. tolerance 내 통과를 bitwise 모델 등가, checkpoint 선택의 미래 성능 개선 또는 독립 일반화 증명으로 바꾸어 부르지 않는다. 이 pass는 saved artifact/source·실제 추론·기록 점수의 수치 일관성을 검사한 것이다. 실패한 재추론 경로는 PARTIAL로 존중하며 다시 학습한 성공 모델로 바꾸지 않는다.','',
            '- [core/native checkpoint 실제 재추론 원수치](checkpoint_reinference.csv)',
            '- [선택 FIT q전후 실제 재추론 원수치](alignment_reinference.csv)',''])
        for record in checkpoint_audit.get('source_results',[]):
            lines.append(f'- 재추론 입력 RESULT `{record["path"]}`, SHA256 `{record["sha256"]}`')
        for path,digest in checkpoint_audit.get('source_hashes',{}).items():
            lines.append(f'- 재추론 구현/source `{path}`, SHA256 `{digest}`')
        lines.append('')
    resource_rows=[]
    resource_records=[]
    new_resources=(new or {}).get('resources',{})
    source_resource_records=[]
    if not new_resources:
        for label,path in inputs:
            if path.parent.name in ('run_new_v1','run_beijing_repair_v1') and path.name=='RESULT.json':
                receipt=read(path)
                if receipt.get('resources'):source_resource_records.append((path.parent.name,receipt['resources']))
    stages=[('개발',dev),('새자료·controls resolved',new or {}),('checkpoint 재추론',checkpoint_audit or {}),('최종비용',cost or {})]
    stages.extend((label,{'resources':resource}) for label,resource in source_resource_records)
    for label,result in stages:
        resource=result.get('resources',{})
        if resource: resource_records.append(resource)
        resource_rows.append([label,resource.get('wall_seconds','미기록'),resource.get('peak_sampled_rss_bytes','미기록'),
            resource.get('peak_allocated_vram_bytes','미기록'),resource.get('new_bytes','미기록')])
    lines.extend([table(['terminal단계','GPU사용process wall초','peak sampled RSS bytes','peak allocated VRAM bytes','새출력 bytes'],resource_rows),'',
        f'제공된 위 terminal 단계의 GPU사용 process wall 합은 {sum(r.get("wall_seconds",0.) for r in resource_records):.3f}초다. 이전 preflight·보존 attempt도 포함하는 마지막 resource ledger의 누적값 최대는 {max((r.get("cumulative_gpu_process_wall_seconds",0.) for r in resource_records),default=0.):.3f}초다. checkpoint 재추론 추가 process wall은 {(checkpoint_audit or {}).get("resources",{}).get("wall_seconds","미실행")}초이며 새 학습을 더한 비용이 아니다.','',
        f'기록된 audit CPU wall 합(최종audit 포함)은 {(audit or {}).get("counted_audit_cpu_wall_seconds_including_current","미기록")}초다. 이 합은 기록된 receipts만 더한 값이며 짧은 test/read와 receipt 없는 중단 시도의 CPU wall은 포함하지 않는다. 보고 생성기의 별도 CPU-only wall/RSS·새파일 저장량은 REPORT_MANIFEST에 남긴다.','',
        '이 process wall은 GPU kernel busy time과 같지 않으며 inference·CPU 보조처리·checkpoint 저장도 포함한다. core 학습 wall에는 각 fit의 중간 FIT/TUNE 평가와 저장이 포함된다. teacher 추출·prototype 적합·native/ridge 추가 학습·재추론을 숨기지 않는다.', '',
        '## 실패·미완료 경로', ''])
    failures=[f for result in (dev,new or {},cost or {},checkpoint_audit or {}) for f in result.get('failed_paths',[])]
    if failures:
        lines.extend([table(['자료','경로','실제상태','이유'],[[f.get('dataset'),f.get('label'),f.get('status','FAILED_PATH_PRESERVED'),f.get('error')] for f in failures]),'',
            '실패한 artifact는 보존했다. 해당 경로의 입력·구현·수치 조건과 자원 잔여를 먼저 확인해야 재개할 수 있다. 실패한 후보를 정상 실행 결과로 대체하거나 전체 완료라고 쓰지 않는다.'])
    else:
        lines.append('제공된 terminal receipt에는 failed_paths가 없다. 별도 중단 attempt와 필수 scope 누락 여부는 독립 검산 receipt의 범위를 함께 확인한다.')
    historical=[]
    for label,path in inputs:
        if path.parent.name=='run_new_v1' and path.name=='RESULT.json':
            original=read(path)
            historical.extend([path.parent.name,f.get('dataset'),f.get('label'),f.get('status'),f.get('error')]
                              for f in original.get('failed_paths',[]))
    if historical:
        lines.extend(['','원 attempt의 실패는 아래처럼 별도 보존했다. 후속 수리 결과가 있어도 원 PARTIAL을 COMPLETE로 고치지 않았다.','',
            table(['원attempt','자료','경로','원상태','원실패이유'],historical),''])
    audit_rows=[]
    for label,path in inputs:
        if path.name=='audit.json':
            receipt=read(path)
            audit_rows.append([path.parent.name,receipt.get('status'),receipt.get('count',len(receipt.get('checks',[]))),
                receipt.get('failed_count'),receipt.get('cpu_wall_seconds')])
    if audit_rows:
        lines.extend(['',table(['검산attempt','실제상태','check수','실패수','CPU wall초'],audit_rows),'',
            '[확인] S1 v1의 FP32 관련 엄격한 수치 tolerance 검사 실패 기록과 S1 v2 통과 기록을 모두 보존했다. 두 결과의 차이 원인을 완전히 확정했다고 쓰지 않는다. 중간 S2 일부case 검사에서 전체52개 scope가 아직 없었던 실패와 완성 후 검산을 구분한다. 검산 통과는 저장 숫자·계약·선택 일관성의 근거이며 독립 일반화 성과를 뜻하지 않는다.'])
    lines.extend(['',
        '## 남은 원인과 다음 판단', '',
        '[판단] 현재값 bypass가 도움이 된다면 작은 K가 원채널 수준 자체까지 복원하던 부담이 실용적 병목이었다는 근거가 된다. 그러나 고정-D oracle·teacher basis·nonlinear D·pointwise E의 결과를 함께 보아야 표현, 잠재 미래 예측, decoder 중 단일 원인을 확정할 수 있다. 학습된 미래 정렬이 continuous regression와 shuffle 대조를 이기지 못하면 복잡한 분할을 추가할 근거는 약하다. 새 자료에서 방향이 뒤집히면 기존 개발 자료의 개선을 일반적 해결로 쓰지 않는다.', '',
        '문제 정의는 “작은 K에서도 TSFM이 사용할 미래 변화 성분을 만들고 현재 원채널 상태 보존과 결합했을 때 정확도·비용의 교환이 개선되는가”로 좁힐 수 있다. 향후 논문 기여는 실제 새자료 결과가 지지한 부분에 한정하며, ExoTimer Bi-Hash 재현이나 대조학습보다 본질적으로 우월하다는 주장을 넣지 않는다.', '',
        '## 자료·실행·검산 근거', '',
        '- [실험 목적·사전 계약](../PLAN_KO.md)', '- [실제 자산 preflight](../PREFLIGHT.json)',
        '- [선행 방법과 독립성 주장 범위](../SOURCE_NOTES_KO.md)',
        '- [보고서 파일 SHA manifest](REPORT_MANIFEST.json)', '- [전체 원수치](all_methods.csv)',
        '- [checkpoint 선택 경로](checkpoints.csv)', '- [학습·보조 gradient curves](training_curves.csv)',
        '- [채널 원수치](channel_metrics.csv)', '- [time/channel paired 비교](paired_diagnostics.csv)', ''])
    for label, path in inputs:
        lines.append(f'- [{label}]({relative(path,out)}), SHA256 `{sha(path)}`')
    lines.extend(['', '공식 자료 출처: [UCI Appliances374](https://archive.ics.uci.edu/dataset/374/appliances+energy+prediction), [UCI Beijing501](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data), 두 자료 CC BY4.0. 사전 고정 Appliances random rv1/rv2 제외, Beijing Aotizhongxin11수치 채널만 사용했다.', '',
        '방법 관점 출처: [NLinear 공식 마지막값 보존 코드](https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py), [ExoTimer 공식 논문](https://ojs.aaai.org/index.php/AAAI/article/view/39908/43869), [CPC 원저](https://arxiv.org/abs/1807.03748). 이번 방식은 ExoTimer에서 착안한 continuous future-regime 보조 정렬이며 Bi-Hash 재현이 아니다.', ''])
    return '\n'.join(lines)


def generate(args):
    start = time.perf_counter()
    sample_rss()
    paths = [('개발실행', Path(args.dev).resolve())]
    dev = load_terminal(paths[0][1]); new = cost = audit = checkpoint_audit = None
    for label, attr in [('새자료·control실행', 'new'), ('최종비용', 'cost'), ('독립검산', 'audit'),('checkpoint 실제재추론','checkpoint_audit')]:
        raw_path = getattr(args, attr)
        if raw_path:
            path = Path(raw_path).resolve(); paths.append((label,path))
            record = load_terminal(path)
            if attr == 'new': new = record
            elif attr == 'cost': cost = record
            elif attr == 'audit': audit = record
            else: checkpoint_audit=record
    recorded={path for _,path in paths}
    for folder in ('validation_preflight_v1','validation_S1_v1','validation_S1_v2',
                   'validation_S2_completed_partial_v1','validation_dev_v1','validation_dev_final'):
        path=(HERE/folder/'audit.json').resolve()
        if path.exists() and path not in recorded:
            load_terminal(path)
            paths.append(('보존된검산 '+folder,path));recorded.add(path)
    if new is not None:
        for label,subpath in [('원 새자료 PARTIAL 보존','run_new_v1/RESULT.json'),
                              ('Beijing 수리 실행','run_beijing_repair_v1/RESULT.json')]:
            path=(HERE/subpath).resolve()
            if path.exists() and path not in recorded:
                load_terminal(path);paths.append((label,path));recorded.add(path)
        for label,subpath in [('수리전 봉인 note','BEIJING_REPAIR_EXECUTION_NOTE_KO.md'),
                              ('Beijing 초기진단 v1','beijing_data_diagnosis/current_raw_metadata.json'),
                              ('Beijing 정정진단 v2','beijing_data_diagnosis/current_raw_metadata_v2.json')]:
            path=(HERE/subpath).resolve()
            if path.exists() and path not in recorded:
                paths.append((label,path));recorded.add(path)
    cases = combine(dev,new)
    if args.allow_evaluated_predictions and new is None:
        raise ValueError('FINAL_NEW_DATA_RECEIPT_REQUIRED_BEFORE_ANY_EVAL_NPZ_READ')
    out = HERE / args.report
    if out.resolve().parent != HERE.resolve() or not args.report.startswith('report_'):
        raise ValueError('REPORT_MUST_BE_A_NEW_REPORT_FOLDER_IN_THIS_STUDY')
    methods = score_rows(cases); trains, checkpoints, curves = training_rows(cases)
    control_trains=control_training_rows(cases);contrasts=same_weight_contrasts(cases)
    diagnostics, geometry, oracles, channels = diagnostics_rows(cases); costs = cost_rows(cost)
    cp_rows,alignment_reinference=checkpoint_audit_rows(checkpoint_audit)
    sample_rss()
    if args.check_only:
        print(json.dumps({'status':'SCHEMA_CHECK_ONLY_NO_REPORT_OR_NPZ_READ', 'cases':len(cases),'score_rows':len(methods),
            'training_fits':len(trains),'checkpoint_rows':len(checkpoints),'diagnostics':len(diagnostics)})); return
    checks, pairs = predictions_and_pairs(cases,dev,args.allow_evaluated_predictions)
    out.mkdir(exist_ok=False)
    for name, rows in [('all_methods',methods),('training',trains),('control_training',control_trains),('checkpoints',checkpoints),('training_curves',curves),
                       ('same_weight_alignment_contrasts',contrasts),
                       ('alignment_diagnostics',diagnostics),('prototype_geometry',geometry),('oracle_projection',oracles),
                       ('channel_metrics',channels),('paired_diagnostics',pairs),('batch_cost',costs),('report_rescore_checks',checks),
                       ('checkpoint_reinference',cp_rows),('alignment_reinference',alignment_reinference)]:
        write_csv(out/(name+'.csv'),rows)
    figures(cases,dev,costs,diagnostics,pairs,out)
    sample_rss()
    original=read(HERE/'report_v1/REPORT_MANIFEST.json')
    csv_equivalence=[]
    for path in sorted(out.glob('*.csv')):
        digest=sha(path)
        if original['outputs'].get(path.name)!=digest:raise AssertionError('V2_CSV_DIFFERS_FROM_SEALED_V1: '+path.name)
        csv_equivalence.append({'file':path.name,'sha256':digest,'same_as_sealed_v1':True})
    if len(csv_equivalence)!=15:raise AssertionError('V2_EXPECTED_15_IDENTICAL_CSVS')
    report = narrative(cases,dev,new,cost,audit,diagnostics,pairs,paths,out,checkpoint_audit)
    absent_links=[]
    for link in re.findall(r'\]\(([^)]+)\)',report):
        if link.startswith(('http:', 'https:')) or link=='REPORT_MANIFEST.json': continue
        if not (out/link).resolve().exists(): absent_links.append(link)
    if absent_links:
        for link in absent_links:
            report=report.replace(f']({link})', ']（미완료: 해당 산출물 없음）')
    with (out/'REPORT_KO.md').open('x',encoding='utf-8') as handle: handle.write(report)
    sources=[{'label':label,'path':str(path.relative_to(ROOT)),'sha256':sha(path)} for label,path in paths]
    allowed_complete=('COMPLETE','PASS','VERIFIED_COMPLETE','COMPLETE_RESOLVED_WITH_DECLARED_PROTOCOL_DEVIATION','PASS_WITH_DECLARED_PROTOCOL_DEVIATION')
    terminal_complete=all(r and str(r.get('status','')).upper() in allowed_complete for r in (dev,new,cost,audit))
    if checkpoint_audit is not None and checkpoint_audit.get('status')!='COMPLETE': terminal_complete=False
    actual={'cases':len(cases),'core_training_fits':len(trains),'score_rows':len(methods),'cost_rows':len(costs),
            'alignment_diagnostic_cases':len(diagnostics),'paired_diagnostic_rows':len(pairs),
            'checkpoint_reinference_rows':len(cp_rows),'alignment_reinference_rows':len(alignment_reinference)}
    required={'core_training_fits':120,'score_rows':606,'cost_rows':400,'alignment_diagnostic_cases':80,
              'checkpoint_reinference_rows':680,'alignment_reinference_rows':80}
    coverage_complete=all(actual[key]==expected for key,expected in required.items())
    deviation=(new or {}).get('protocol_deviation',{})
    declared_deviation=isinstance(deviation,dict) and deviation.get('joint_models_before_any_new_EVAL') is False
    if str((new or {}).get('status',''))=='COMPLETE_RESOLVED_WITH_DECLARED_PROTOCOL_DEVIATION' and not declared_deviation:
        terminal_complete=False
    complete=terminal_complete and coverage_complete and not absent_links
    report_status=('DERIVED_REPORT_COMPLETE_WITH_DECLARED_PROTOCOL_DEVIATION' if declared_deviation else 'DERIVED_REPORT_COMPLETE') if complete else 'PARTIAL_DERIVED_REPORT'
    output_bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    study_bytes=sum(p.stat().st_size for base in (HERE,ROOT/'.cache'/HERE.name) for p in base.rglob('*') if p.is_file())
    if study_bytes>12*1024**3:raise RuntimeError('STUDY_STORAGE_OVER_12GIB')
    manifest={'status':report_status,
        'source_receipts':sources,'report_generator_sha256':sha(__file__),
        'original_report_manifest':{'path':str((HERE/'report_v1/REPORT_MANIFEST.json').relative_to(ROOT)),
                                    'sha256':sha(HERE/'report_v1/REPORT_MANIFEST.json')},
        'original_generator_sha256':original['report_generator_sha256'],
        'csv_equivalence_to_sealed_v1':csv_equivalence,
        'figure_provenance':{'revised_layout_or_legend_only':['alignment_mechanism','accuracy_cost'],
                            'byte_identical_copied_from_sealed_v1':[p.name for p in sorted(out.glob('*'))
                                if p.suffix in ('.png','.svg') and p.stem not in ('alignment_mechanism','accuracy_cost')]},
        'actual_scope':actual,'required_full_scope':required,'full_scope_coverage':coverage_complete,
        'protocol_deviation':deviation,'original_joint_protocol_pass':False if declared_deviation else None,
        'checkpoint_audit_status':None if checkpoint_audit is None else checkpoint_audit.get('status'),
        'checkpoint_audit_new_training_updates':None if checkpoint_audit is None else checkpoint_audit.get('new_training_updates'),
        'saved_prediction_read_authorized':args.allow_evaluated_predictions,'prediction_rescore_checks':checks,
        'missing_outputs':absent_links,'no_training_or_original_data_open':True,'science_receipts_not_modified':True,
        'cpu_resources':{'cpu_only':True,'gpu_process':False,'process_wall_seconds':time.perf_counter()-PROCESS_STARTED,
                         'report_body_wall_seconds':time.perf_counter()-start,'peak_sampled_rss_bytes':PEAK_SAMPLED_RSS,
                         'sampling_note':'sampled at start, extraction, saved-array loads and completed figures; not continuous OS peak',
                         'output_bytes_before_manifest':output_bytes,'whole_new_study_bytes_before_manifest':study_bytes},
        'wall_seconds':time.perf_counter()-start,'outputs':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    with (out/'REPORT_MANIFEST.json').open('x',encoding='utf-8') as handle:
        json.dump(manifest,handle,ensure_ascii=False,indent=2,allow_nan=False); handle.write('\n')
    print(json.dumps({'status':manifest['status'],'report':str(out/'REPORT_KO.md'),'scope':manifest['actual_scope'],'wall_seconds':manifest['wall_seconds']},ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dev',required=True)
    parser.add_argument('--new')
    parser.add_argument('--cost')
    parser.add_argument('--audit')
    parser.add_argument('--checkpoint-audit')
    parser.add_argument('--report',default='report_v2')
    parser.add_argument('--allow-evaluated-predictions',action='store_true')
    parser.add_argument('--check-only',action='store_true')
    generate(parser.parse_args())
