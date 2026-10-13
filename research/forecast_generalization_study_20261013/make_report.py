"""Create read-only derived reports from the completed generalization study."""
from pathlib import Path
from functools import lru_cache
import argparse
import csv
import hashlib
import json
import re
import time
import unicodedata

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OLD = ROOT / 'research/forecast_component_study_20261013'
SEEDS = (92601, 92602)
COLORS = {'BASE': '#0072B2', 'LOW_LR': '#009E73', 'DECAY': '#D55E00',
          'BOLT': '#0072B2', 'BOLT_RAW_R': '#56B4E9', 'NATIVE': '#E69F00',
          'NATIVE_RAW_R': '#CC79A7', 'RAW_BOLT': '#000000', 'RAW_BOLT_RAW_R': '#666666',
          'RAW_R_ONLY': '#009E73', 'RAW_SELF_LAG': '#D55E00'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError('An expected CSV has no rows: ' + str(path))
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise ValueError('CSV rows have different fields: ' + str(path))
    with Path(path).open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def method(case, name):
    return next(m for m in case['methods'] if m['method'] == name)


def selected_update(case):
    return case.get('selected_update', case.get('reused_checkpoint', {}).get('selected_update'))


def selected_cp(case):
    return next(cp for cp in case['training']['checkpoints'] if cp['update'] == selected_update(case))


def score_value(record, split='val', scale='normalized', kind='mse'):
    value = record['scores'][split]
    return value[scale][kind] if scale == 'normalized' else value[kind]


def timings(record, original=False):
    return record['latency'] if original else record.get('clean_latency', record['latency'])


def key(case):
    return (case['dataset'], case['label'])


def variant(case):
    channels = 21 if case['dataset'] == 'jena' else 17
    if case['length'] == 128:
        return 'Quarter K | L128'
    if case['receptive_field'] == 31:
        return 'Quarter K | RF31'
    if case['k'] == channels:
        return 'Full identity' if case['initialization']['kind'] == 'IDENTITY' else 'Full PCA rotation'
    if case['k'] == channels // 2:
        return 'Half K'
    return 'Quarter K'


def full_group(cases, dataset, config):
    baseline = [c for c in cases if c['dataset'] == dataset and c['stage'] == 'G1'
                and c['objective'] == 'PRED' and c['config'] == config]
    return baseline + [c for c in cases if c['dataset'] == dataset and c['stage'] == 'G2']


def average(cases, name='BOLT', split='val', scale='normalized', kind='mse'):
    return float(np.mean([score_value(method(c, name), split, scale, kind) for c in cases]))


def format_rows(headers, rows):
    values = [[str(v) for v in row] for row in rows]
    width = lambda value: sum(2 if unicodedata.east_asian_width(ch) in 'FW' else 1 for ch in str(value))
    widths = [max(width(h), *(width(r[i]) for r in values)) for i, h in enumerate(headers)]
    line = lambda r: '  '.join(str(v)+' '*(widths[i]-width(v)) for i, v in enumerate(r)).rstrip()
    return '```text\n' + line(headers) + '\n' + '\n'.join(line(r) for r in values) + '\n```'


def flatten_methods(cases):
    rows = []
    for c in cases:
        for m in c['methods']:
            row = {'dataset': c['dataset'], 'stage': c['stage'], 'case': c['label'],
                   'objective': c.get('objective', 'RAW'), 'config': c.get('config', 'REUSED' if c['stage'] == 'G0' else 'RAW'),
                   'seed': c['seed'], 'selection_role': c.get('selection_role', 'RAW'),
                   'selected_update': selected_update(c), 'k': c['k'], 'component_length': c['length'],
                   'receptive_field': c.get('receptive_field'), 'initialization': c.get('initialization', {}).get('kind', 'REUSED' if c['stage'] == 'G0' else 'RAW'),
                   'method': m['method'], 'raw_correction_support': 512 if m['method'].endswith('RAW_R') or m['method'] == 'RAW_R_ONLY' else None,
                   'feature_support_self33': 33 if m['method'] == 'RAW_SELF_LAG' else None,
                   'preprocess_support_self33': 512 if m['method'] == 'RAW_SELF_LAG' else None,
                   'prediction_path': m['prediction']['path'], 'prediction_sha256': m['prediction']['sha256']}
            for split in ('fit', 'tune', 'val'):
                for scale in ('normalized', 'physical'):
                    for kind in ('mse', 'mae'):
                        row[f'{split}_{scale}_{kind}'] = score_value(m, split, scale, kind)
            for batch in (1, 8):
                latency = next(l for l in timings(m) if l['batch'] == batch)
                row[f'batch{batch}_median_ms'] = latency['median_seconds'] * 1000
                row[f'batch{batch}_median_ms_per_window'] = latency['median_seconds_per_window'] * 1000
            rows.append(row)
    return rows


def training_rows(cases):
    rows, seen = [], set()
    for c in cases:
        if 'training' not in c:
            continue
        train = c['training']
        identity = tuple(cp['sha256'] for cp in train['checkpoints'])
        if identity in seen:
            continue
        seen.add(identity)
        roles = [cc for cc in cases if cc.get('training', {}).get('checkpoints', [{}])[0].get('sha256') == identity[0]]
        row = {'dataset': c['dataset'], 'stage': c['stage'], 'case_without_selection_role': c['label'].rsplit('_', 1)[0],
               'objective': c['objective'], 'config': c['config'], 'seed': c['seed'], 'k': c['k'],
               'length': c['length'], 'receptive_field': c['receptive_field'], 'initialization': c['initialization']['kind'],
               'train_wall_seconds': train['train_wall_seconds'], 'updates': train['updates'], 'batch_size': train['batch_size'],
               'learning_rate': train['learning_rate'], 'weight_decay': train['weight_decay'],
               'trainable_parameters': train['trainable_parameters'], 'initial_codec_hash': train['initial_codec_hash'],
               'order_sha256': train['order_sha256'], 'evaluation_roles': '|'.join(cc['selection_role'] for cc in roles),
               'forecast_selected_update': train['selected_updates']['FORECAST'],
               'reconstruction_selected_update': train['selected_updates'].get('RECONSTRUCTION'),
               'same_checkpoint_for_both_roles': len(roles) > 1 and len({selected_update(cc) for cc in roles}) == 1,
               'post_fit_unique_timestamps': train['post_fit_reconstruction_diagnostic']['unique_post_fit_timestamps'],
               'post_fit_reconstruction_count_sum': sum(train['post_fit_reconstruction_diagnostic']['observed_counts']),
               'checkpoint_sha256': '|'.join(identity)}
        first, last = train['checkpoints'][0], train['checkpoints'][-1]
        for cp_name, cp in [('u0', first), ('u600', last), ('forecast_selected', selected_cp(next(cc for cc in roles if cc['selection_role'] == 'FORECAST')))]:
            for metric_name in ('mse', 'mae'):
                for split in ('fit', 'tune'):
                    row[f'{cp_name}_forecast_{split}_normalized_{metric_name}'] = cp['forecast'][split]['normalized'][metric_name]
                    row[f'{cp_name}_reconstruction_{split}_normalized_{metric_name}'] = cp['reconstruction'][split]['normalized'][metric_name]
                row[f'{cp_name}_reconstruction_tune_postfit_normalized_{metric_name}'] = cp['reconstruction_tune_post_fit_positions']['normalized'][metric_name]
        rows.append(row)
    return rows


def make_comparisons(cases, old):
    rows, pairs = [], []

    def add(stage, contrast, c, mname, reference_case, reference_name, reference_source='NEW'):
        a, b = method(c, mname), method(reference_case, reference_name)
        row = {'stage': stage, 'contrast': contrast, 'dataset': c['dataset'], 'seed': c['seed'],
               'case': c['label'], 'method': mname, 'reference_case': reference_case['label'],
               'reference_method': reference_name, 'reference_source': reference_source,
               'selected_update': selected_update(c), 'reference_selected_update': selected_update(reference_case),
               'condition_note': 'Same VAL target origins/masks; selection based on TUNE only; repeated VAL development'}
        for split in ('tune', 'val'):
            for scale in ('normalized', 'physical'):
                for kind in ('mse', 'mae'):
                    av, bv = score_value(a, split, scale, kind), score_value(b, split, scale, kind)
                    row[f'{split}_{scale}_{kind}_candidate'] = av
                    row[f'{split}_{scale}_{kind}_reference'] = bv
                    row[f'{split}_{scale}_{kind}_delta'] = av-bv
                    row[f'{split}_{scale}_{kind}_relative_percent'] = (av/bv-1)*100
        rows.append(row)
        pairs.append((row, a, b))

    for c in cases:
        if c['stage'] == 'G0':
            previous_case = next(cc for cc in old['cases'] if cc['dataset'] == c['dataset']
                                 and cc['label'] == c['objective'] and cc['seed'] == c['seed'])
            prior = dict(previous_case, selected_update=previous_case['training']['selected_update'])
            add('G0', 'normalized_selection_vs_old_physical_selection', c, 'BOLT', prior,
                c['objective'] + '_BOLT', 'OLD_SAVED_CHECKPOINT_PREDICTION')
            if c['objective'] == 'REC' and c['selection_role'] == 'FORECAST':
                ref = next(cc for cc in cases if cc['stage']=='G0' and cc['dataset']==c['dataset']
                           and cc.get('objective')=='REC' and cc['seed']==c['seed'] and cc['selection_role']=='RECONSTRUCTION')
                add('G0', 'REC_forecast_selection_vs_reconstruction_selection', c, 'BOLT', ref, 'BOLT')
        elif c['stage'] == 'G1':
            if c['config'] != 'BASE':
                ref = next(cc for cc in cases if cc['stage']=='G1' and cc['dataset']==c['dataset']
                           and cc['seed']==c['seed'] and cc['objective']==c['objective']
                           and cc['selection_role']==c['selection_role'] and cc['config']=='BASE')
                add('G1', c['config']+'_vs_BASE_same_objective_role', c, 'BOLT', ref, 'BOLT')
            if c['objective']=='PRED':
                ref = next(cc for cc in cases if cc['stage']=='G1' and cc['dataset']==c['dataset']
                           and cc['seed']==c['seed'] and cc['objective']=='REC'
                           and cc['selection_role']=='FORECAST' and cc['config']==c['config'])
                add('G1', 'PRED_vs_REC_same_forecast_selection', c, 'BOLT', ref, 'BOLT')
            elif c['selection_role']=='FORECAST':
                ref = next(cc for cc in cases if cc['stage']=='G1' and cc['dataset']==c['dataset']
                           and cc['seed']==c['seed'] and cc['objective']=='REC'
                           and cc['selection_role']=='RECONSTRUCTION' and cc['config']==c['config'])
                add('G1', 'REC_forecast_selection_vs_reconstruction_selection', c, 'BOLT', ref, 'BOLT')
        elif c['stage']=='G2':
            ref = next(cc for cc in cases if cc['stage']=='G1' and cc['dataset']==c['dataset']
                       and cc['seed']==c['seed'] and cc['objective']=='PRED' and cc['config']==c['config'])
            add('G2', variant(c)+'_vs_quarter_baseline', c, 'BOLT', ref, 'BOLT')
        if c.get('objective')=='PRED' and c['stage'] in ('G1','G2'):
            add('G3', 'Bolt_vs_native_same_encoder_decoder', c, 'BOLT', c, 'NATIVE')
            add('G3', 'Bolt_plusR_vs_native_plusR', c, 'BOLT_RAW_R', c, 'NATIVE_RAW_R')
            add('G3', 'selected_D_vs_common_reconstruction_D', c, 'BOLT', c, 'BOLT_COMMON_D')
            add('G3', 'correction_vs_Bolt_branch', c, 'BOLT_RAW_R', c, 'BOLT')
            raw = next(cc for cc in cases if cc['dataset']==c['dataset'] and cc['label']=='RAW')
            for name in ('RAW_BOLT','RAW_BOLT_RAW_R','RAW_R_ONLY','RAW_SELF_LAG'):
                add('G3', 'component_Bolt_vs_'+name, c, 'BOLT', raw, name)
                add('G3', 'component_Bolt_plusR_vs_'+name, c, 'BOLT_RAW_R', raw, name)
    return rows, pairs


@lru_cache(maxsize=8)
def prediction(path):
    with np.load(ROOT / path, allow_pickle=False) as z:
        return {name: z[name].copy() for name in ('val_prediction','val_target','val_observed','val_origins','mean','std')}


def macro(pred, target, observed):
    delta = pred.astype(np.float64)-target.astype(np.float64)
    count = observed.sum((0,1))
    valid = count > 0
    return float(np.mean((np.where(observed, delta*delta, 0).sum((0,1))/np.maximum(count,1))[valid])), \
           float(np.mean((np.where(observed, np.abs(delta), 0).sum((0,1))/np.maximum(count,1))[valid]))


def chronological_rows(pairs):
    rows = []
    for info, a, b in pairs:
        za, zb = prediction(a['prediction']['path']), prediction(b['prediction']['path'])
        if not np.array_equal(za['val_origins'], zb['val_origins']) or not np.array_equal(za['val_observed'], zb['val_observed']):
            raise AssertionError('Paired comparison has different origins or masks')
        for block, indices in enumerate(np.array_split(np.arange(len(za['val_origins'])), 4), 1):
            mask = za['val_observed'][indices]
            row = {'stage': info['stage'], 'contrast': info['contrast'], 'dataset': info['dataset'], 'seed': info['seed'],
                   'case': info['case'], 'method': info['method'], 'reference_case': info['reference_case'],
                   'reference_method': info['reference_method'], 'block': block, 'origins': len(indices),
                   'first_origin': int(za['val_origins'][indices[0]]), 'last_origin': int(za['val_origins'][indices[-1]]),
                   'observed_positions': int(mask.sum()), 'interpretation': 'Chronological descriptive block; overlapping targets, not independent replicate'}
            for scale in ('normalized','physical'):
                pa, pb, target = za['val_prediction'][indices], zb['val_prediction'][indices], za['val_target'][indices]
                if scale=='physical':
                    pa = pa*za['std']+za['mean']
                    pb = pb*zb['std']+zb['mean']
                    target = target*za['std']+za['mean']
                amse, amae = macro(pa,target,mask)
                bmse, bmae = macro(pb,target,mask)
                row.update({f'{scale}_mse_candidate':amse, f'{scale}_mse_reference':bmse,
                            f'{scale}_mse_delta':amse-bmse, f'{scale}_mae_candidate':amae,
                            f'{scale}_mae_reference':bmae, f'{scale}_mae_delta':amae-bmae})
            rows.append(row)
    return rows


def cost_rows(cases, original=False):
    rows = []
    for c in cases:
        for m in c['methods']:
            for l in timings(m, original):
                row = {'dataset': c['dataset'], 'stage':c['stage'], 'case':c['label'], 'method':m['method'],
                       'seed':c['seed'], 'k':c['k'], 'component_length':c['length'], 'rf':c.get('receptive_field'),
                       'timing_source':'ORIGINAL_DURING_TRAINING' if original else ('CLEAN_AFTER_TRAINING' if 'clean_latency' in m else 'ORIGINAL'),
                       'batch':l['batch'], 'warmup':l['warmup'], 'repeats':l['repeats'],
                       'median_batch_ms':l['median_seconds']*1000, 'median_window_ms':l['median_seconds_per_window']*1000,
                       'min_batch_ms':min(l['seconds'])*1000, 'max_batch_ms':max(l['seconds'])*1000,
                       'resident_rss_bytes':l['resident_rss_bytes'], 'co_resident_allocated_vram_bytes':l['peak_allocated_vram_bytes'],
                       'raw_correction_support':l['raw_correction_support'], 'pipeline_parity':l['pipeline_prediction_parity']['pass'],
                       'pipeline_physical_max_abs':l['pipeline_prediction_parity']['physical_max_abs'],
                       'includes':l['includes'], 'memory_note':l['co_resident_model_note']}
                row.update({f'repeat{i+1}_batch_ms':t*1000 for i,t in enumerate(l['seconds'])})
                rows.append(row)
    return rows


def channel_rows(cases, contracts):
    rows = []
    for c in cases:
        for m in c['methods']:
            for split in ('fit','tune','val'):
                scores=m['scores'][split]
                for i,name in enumerate(contracts[c['dataset']]['channels']):
                    row={'dataset':c['dataset'], 'stage':c['stage'], 'case':c['label'], 'method':m['method'],
                         'seed':c['seed'], 'split':split, 'channel_index':i, 'channel':name,
                         'observed_count':scores['observed_counts'][i], 'fit_mean':contracts[c['dataset']]['mean'][i],
                         'fit_std':contracts[c['dataset']]['std'][i]}
                    for scale in ('normalized','physical'):
                        s=scores['normalized'] if scale=='normalized' else scores
                        row.update({f'{scale}_mse':s['per_channel_mse'][i], f'{scale}_mae':s['per_channel_mae'][i]})
                    rows.append(row)
    return rows


def reconstruction_rows(cases):
    rows, seen=[],set()
    for c in cases:
        if 'training' not in c:
            continue
        cp0=c['training']['checkpoints'][0]
        if cp0['sha256'] in seen:
            continue
        seen.add(cp0['sha256'])
        for cp in c['training']['checkpoints']:
            row={'dataset':c['dataset'], 'stage':c['stage'], 'objective':c['objective'], 'config':c['config'],
                 'seed':c['seed'], 'k':c['k'], 'length':c['length'], 'rf':c['receptive_field'],
                 'initialization':c['initialization']['kind'], 'update':cp['update'], 'checkpoint_sha256':cp['sha256'],
                 'forecast_selected':cp['update']==c['training']['selected_updates']['FORECAST'],
                 'reconstruction_selected':cp['update']==c['training']['selected_updates'].get('RECONSTRUCTION')}
            for split in ('fit','tune'):
                for kind in ('mse','mae'):
                    row[f'forecast_{split}_normalized_{kind}']=cp['forecast'][split]['normalized'][kind]
                    row[f'reconstruction_{split}_normalized_{kind}']=cp['reconstruction'][split]['normalized'][kind]
            row['reconstruction_tune_postfit_normalized_mse']=cp['reconstruction_tune_post_fit_positions']['normalized']['mse']
            row['reconstruction_tune_postfit_normalized_mae']=cp['reconstruction_tune_post_fit_positions']['normalized']['mae']
            rows.append(row)
    return rows


def savefig(fig, folder, stem):
    fig.savefig(folder/(stem+'.png'), dpi=200, bbox_inches='tight')
    fig.savefig(folder/(stem+'.svg'), bbox_inches='tight')
    plt.close(fig)


def log_when_needed(ax, values):
    positive = [v for v in values if v > 0]
    if positive and max(positive)/min(positive) > 10:
        ax.set_yscale('log')
        ax.set_ylabel(ax.get_ylabel()+' (log)')


def draw(cases, selections, blocks, folder):
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11, 'axes.spines.top':False,
                         'axes.spines.right':False, 'svg.fonttype':'none', 'axes.titleweight':'bold'})
    fig, axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    for row,dataset in enumerate(('jena','robin')):
        for col,role in enumerate(('REC_RECONSTRUCTION','REC_FORECAST','PRED_FORECAST')):
            objective,selection_role=role.split('_',1)
            ax=axes[row,col]
            for i,config in enumerate(('BASE','LOW_LR','DECAY')):
                group=[c for c in cases if c['dataset']==dataset and c['stage']=='G1' and c['objective']==objective
                       and c['selection_role']==selection_role and c['config']==config]
                vals=[score_value(method(c,'BOLT')) for c in group]
                ax.plot([i-.07,i+.07], vals, color=COLORS[config],linewidth=1,alpha=.7)
                for j,c in enumerate(group):
                    ax.scatter(i+(-.07 if j==0 else .07), vals[j], marker='o' if c['seed']==SEEDS[0] else 's',s=55,color=COLORS[config])
                    ax.annotate('u'+str(selected_update(c)),(i+(-.07 if j==0 else .07),vals[j]),xytext=(3,6 if j==0 else -14),textcoords='offset points',fontsize=9)
                ax.plot([i-.13,i+.13],[np.mean(vals)]*2,color='black',linewidth=2)
            ax.set_xticks(range(3),['BASE','LOW LR','DECAY'])
            ax.set_ylabel('VAL normalized macro MSE')
            ax.set_title(f'{dataset.title()} | {objective} | {selection_role.title()} select')
            ax.grid(axis='y',alpha=.2)
    fig.suptitle('G1: Same initialization, batches and metric; dots/squares are the two seeds\nREC has separate reconstruction/forecast selection roles; horizontal mark = seed mean',fontsize=14)
    savefig(fig,folder,'g1_generalization')

    fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    for row,dataset in enumerate(('jena','robin')):
        group=full_group(cases,dataset,selections[dataset]['chosen_config'])
        raw=next(c for c in cases if c['dataset']==dataset and c['label']=='RAW')
        variants=['Quarter K','Half K','Full PCA rotation','Full identity','Quarter K | L128','Quarter K | RF31']
        for col,(split,scale,kind) in enumerate((('tune','normalized','mse'),('val','normalized','mse'),('val','physical','mae'))):
            ax=axes[row,col]
            all_values=[]
            for i,v in enumerate(variants):
                pair=sorted([c for c in group if variant(c)==v],key=lambda c:c['seed'])
                vals=[score_value(method(c,'BOLT'),split,scale,kind) for c in pair]
                all_values.extend(vals)
                for j,c in enumerate(pair):
                    ax.scatter(i+(-.06 if j==0 else .06),vals[j],marker='o' if j==0 else 's',s=48,color='#0072B2')
                ax.plot([i-.16,i+.16],[np.mean(vals)]*2,color='black',linewidth=2)
            ax.set_xticks(range(6),['Quarter','Half','Full\nPCA','Full\nidentity','L128','RF31'])
            ax.axhline(score_value(method(raw,'RAW_BOLT'),split,scale,kind),color='#666666',linestyle='--',linewidth=1.4,label='RAW Bolt')
            ax.set_ylabel(f'{split.upper()} {scale} macro {kind.upper()}')
            log_when_needed(ax,all_values)
            ax.set_title(f'{dataset.title()} | {selections[dataset]["chosen_config"]}')
            ax.grid(axis='y',alpha=.2)
            ax.legend(fontsize=9)
    fig.suptitle('G2: All planned component/history contrasts | Bolt branch only\nFull PCA is a rotation, identity is the uncompressed control; L128 resets missing-value fill',fontsize=14)
    fig.text(.5,-.015,'Circle = seed 92601; square = seed 92602; black horizontal mark = mean of the two seeds',ha='center',fontsize=10)
    savefig(fig,folder,'g2_components_history')

    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    for row,dataset in enumerate(('jena','robin')):
        baseline=[c for c in full_group(cases,dataset,selections[dataset]['chosen_config']) if variant(c)=='Quarter K']
        identity=[c for c in full_group(cases,dataset,selections[dataset]['chosen_config']) if variant(c)=='Full identity']
        raw=next(c for c in cases if c['dataset']==dataset and c['label']=='RAW')
        entries=[('RAW_BOLT',[raw],'RAW_BOLT'),('RAW_BOLT+R',[raw],'RAW_BOLT_RAW_R'),
                 ('RAW_R',[raw],'RAW_R_ONLY'),('SELF33',[raw],'RAW_SELF_LAG'),
                 ('Bolt',baseline,'BOLT'),('Bolt+R',baseline,'BOLT_RAW_R'),
                 ('Native',baseline,'NATIVE'),('Native+R',baseline,'NATIVE_RAW_R'),
                 ('Bolt/commonD',baseline,'BOLT_COMMON_D'),
                 ('Full-ID Bolt',identity,'BOLT'),('Full-ID Bolt+R',identity,'BOLT_RAW_R')]
        for col,(scale,kind) in enumerate((('normalized','mse'),('physical','mae'))):
            ax=axes[row,col]
            all_values=[]
            for i,(label,group,name) in enumerate(entries):
                vals=[score_value(method(c,name),'val',scale,kind) for c in group]
                all_values.extend(vals)
                color=COLORS.get(name,'#666666')
                for j,v in enumerate(vals):
                    ax.scatter(i+((-0.06 if j==0 else .06) if len(vals)==2 else 0),v,color=color,s=50,marker='o' if j==0 else 's')
                ax.plot([i-.15,i+.15],[np.mean(vals)]*2,color='black',linewidth=2)
            ax.set_xticks(range(len(entries)),[e[0] for e in entries],rotation=35,ha='right')
            ax.set_ylabel(f'VAL {scale} macro {kind.upper()}')
            log_when_needed(ax,all_values)
            ax.set_title(f'{dataset.title()} | quarter K + full identity (K=C)')
            ax.grid(axis='y',alpha=.2)
    fig.suptitle('G3: Quarter-K branch and full-identity control versus raw baselines\nFull identity has K=C, so any accuracy gain is not compression evidence; correction uses raw L512',fontsize=14)
    savefig(fig,folder,'g3_predictor_baselines')

    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    for row,dataset in enumerate(('jena','robin')):
        group=full_group(cases,dataset,selections[dataset]['chosen_config'])
        raw=next(c for c in cases if c['dataset']==dataset and c['label']=='RAW')
        for col,batch in enumerate((1,8)):
            ax=axes[row,col]
            for name in ('RAW_BOLT','RAW_BOLT_RAW_R','RAW_R_ONLY','RAW_SELF_LAG'):
                m=method(raw,name)
                t=next(l for l in timings(m) if l['batch']==batch)['median_seconds_per_window']*1000
                val=score_value(m)
                ax.scatter(t,val,color=COLORS[name],marker='X',s=80,label=name)
                raw_labels={'RAW_BOLT':'RAW','RAW_BOLT_RAW_R':'RAW+R','RAW_R_ONLY':'R-only','RAW_SELF_LAG':'SELF33'}
                offset={'RAW_BOLT':(-32,8),'RAW_BOLT_RAW_R':(-43,-11),'RAW_R_ONLY':(5,5),'RAW_SELF_LAG':(5,5)}[name]
                ax.annotate(raw_labels[name],(t,val),xytext=offset,textcoords='offset points',fontsize=9,
                            arrowprops={'arrowstyle':'-','linewidth':.6,'color':COLORS[name],'alpha':.7})
            for v in ['Quarter K','Half K','Full PCA rotation','Full identity','Quarter K | L128','Quarter K | RF31']:
                pair=[c for c in group if variant(c)==v]
                for name,marker in [('BOLT','o'),('BOLT_RAW_R','s'),('NATIVE_RAW_R','^')]:
                    for c in pair:
                        m=method(c,name)
                        t=next(l for l in timings(m) if l['batch']==batch)['median_seconds_per_window']*1000
                        ax.scatter(t,score_value(m),color=COLORS[name],marker=marker,s=26,alpha=.65)
                anchor=method(next(c for c in pair if c['seed']==SEEDS[0]),'BOLT')
                xx=next(l for l in timings(anchor) if l['batch']==batch)['median_seconds_per_window']*1000
                yy=score_value(anchor)
                labels={'Quarter K':'Q','Half K':'H','Full PCA rotation':'P','Full identity':'I','Quarter K | L128':'L','Quarter K | RF31':'R'}
                offsets={'Quarter K':(-22,15),'Half K':(5,8),'Full PCA rotation':(5,7),'Full identity':(5,-16),'Quarter K | L128':(5,0),'Quarter K | RF31':(5,-23)}
                ax.annotate(labels[v],(xx,yy),xytext=offsets[v],textcoords='offset points',fontsize=9,color=COLORS['BOLT'],
                            arrowprops={'arrowstyle':'-','linewidth':.6,'color':COLORS['BOLT'],'alpha':.7})
            ax.set_xscale('log')
            lo,hi=ax.get_xlim()
            ax.xaxis.set_major_locator(FixedLocator([v for v in (.1,.3,1,3,10,30) if lo<=v<=hi]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:g}'))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel('Median full-pipeline milliseconds per window (log)')
            ax.set_ylabel('VAL normalized macro MSE')
            ax.set_title(f'{dataset.title()} | batch {batch}')
            ax.grid(alpha=.2)
    handles=[plt.Line2D([],[],marker=marker,color=COLORS[name],linestyle='',label=name) for name,marker in [('BOLT','o'),('BOLT_RAW_R','s'),('NATIVE_RAW_R','^')]]
    fig.legend(handles=handles,loc='outside lower center',ncol=3)
    fig.suptitle('Accuracy–latency: all planned G2 variants plus their chosen G1 baseline\nClean timing after training; individual seeds; raw X = one deterministic fit; 2 warm-ups and 7 repeats\nBlue Bolt codes: Q = quarter K; H = half K; P = full PCA; I = full identity (K=C); L = actual L128; R = RF31',fontsize=13)
    savefig(fig,folder,'g3_accuracy_latency')

    fig,axes=plt.subplots(2,3,figsize=(17,8),layout='constrained')
    for row,dataset in enumerate(('jena','robin')):
        chosen=selections[dataset]['chosen_config']
        for col,contrast in enumerate(('PRED_vs_REC_same_forecast_selection','component_Bolt_plusR_vs_RAW_BOLT','component_Bolt_plusR_vs_RAW_BOLT_RAW_R')):
            labels={c['label'] for c in full_group(cases,dataset,chosen) if variant(c)==('Full identity' if col==2 else 'Quarter K')}
            ax=axes[row,col]
            eligible=[b for b in blocks if b['dataset']==dataset and b['contrast']==contrast and b['case'] in labels]
            for seed,marker in zip(SEEDS,('o','s')):
                points=sorted([b for b in eligible if b['seed']==seed],key=lambda b:b['block'])
                ax.plot([p['block'] for p in points],[p['normalized_mse_delta'] for p in points],marker=marker,label=str(seed))
            ax.axhline(0,color='black',linewidth=1)
            ax.set_xticks((1,2,3,4))
            ax.set_xlabel('Chronological quarter of VAL origins')
            ax.set_ylabel('Paired normalized MSE difference\n(candidate − reference; lower is better)')
            title=['quarter PRED − REC','quarter Bolt+R − raw Bolt','full-ID Bolt+R − raw Bolt+R'][col]
            ax.set_title(f'{dataset.title()} | '+title)
            ax.grid(axis='y',alpha=.2)
            ax.legend(title='Seed',fontsize=9)
    fig.suptitle('Time-block diagnostics: chosen-config quarter-K and full-identity controls\nFour descriptive blocks share overlapping forecast targets; no independence or significance claim',fontsize=14)
    savefig(fig,folder,'paired_time_blocks')


def narrative(result,cases,trains,contrasts,costs,validation_path,run_name):
    selections=result.get('configuration_selection', result.get('stage2_configuration_selection'))
    text=['# 정규화 평가 기준을 맞춘 학습 일반화·압축·예측기 대조 결과',
          '', '2026-10-13. **[확인] 주평가지표는 사용자가 선택한 FIT 통계 기반 정규화 channel-macro MSE다.** 학습 손실, 미래 TUNE checkpoint 선택, 보정 ridge lambda 선택을 이 기준에 맞췄다. REC 복원 선택은 같은 정규화 MSE를 과거 복원에 적용한 별도 역할이다. 아래는 반복 사용한 개발 VAL 진단이며 독립 일반화 검증이 아니다.',
          '', '## 먼저 답해야 할 질문', '', '### 선택 기준만 바꾸면 기존 결론이 달라지는가?', '']
    g0=[r for r in contrasts if r['stage']=='G0' and r['contrast']=='normalized_selection_vs_old_physical_selection']
    text.append(format_rows(['자료','목적/선택','seed','old→new u','VAL normMSE old→new','변화%'],[
        [r['dataset'],r['case'].split('_')[1]+'/'+r['case'].split('_')[2],r['seed'],f"{r['reference_selected_update']}→{r['selected_update']}",
         f"{r['val_normalized_mse_reference']:.6f}→{r['val_normalized_mse_candidate']:.6f}",f"{r['val_normalized_mse_relative_percent']:+.2f}"] for r in g0]))
    text += ['', '**[확인] G0는 저장된 모델의 학습을 재사용하고 선택한 state에서 미래 예측을 새로 실행했다.** 목표와 checkpoint 선택의 영향이며 새로운 학습 효과가 아니다. REC 복원 선택과 미래 선택을 합치지 않았다. 물리 단위 MSE 우위와 정규화 MSE 우위가 다를 수 있으므로 이전 결론을 새 주지표로 그대로 옮기지 않는다.', '',
             '### 낮은 학습률·weight decay가 일반화를 개선하는가?', '']
    rows=[]
    for dataset in ('jena','robin'):
        for objective,role in [('REC','RECONSTRUCTION'),('REC','FORECAST'),('PRED','FORECAST')]:
            for config in ('BASE','LOW_LR','DECAY'):
                group=sorted([c for c in cases if c['stage']=='G1' and c['dataset']==dataset and c['objective']==objective and c['selection_role']==role and c['config']==config],key=lambda c:c['seed'])
                rows.append([dataset,objective+'/'+role,config,','.join(str(selected_update(c)) for c in group),
                             f'{average(group,split="tune"):.6f}',f'{average(group):.6f}',
                             f'{average(group,scale="physical",kind="mae"):.6f}'])
    text.append(format_rows(['자료','목적/선택','설정','선택u(seed1,2)','미래 TUNE normMSE 평균','미래 VAL normMSE 평균','미래 VAL 원단위MAE 평균'],rows))
    for dataset in ('jena','robin'):
        selection=selections[dataset]
        text.append(f'\n**[확인] {dataset.title()} G2 설정은 {selection["chosen_config"]}**다. 후보별 두 seed 평균 미래 TUNE 정규화 MSE는 '+', '.join(f'{k}={v:.6f}' for k,v in selection['seed_mean_tune_normalized_forecast_mse'].items())+f'다. 최선 후보의 BASE 대비 TUNE 개선은 {100*selection["candidate_relative_improvement_vs_BASE"]:.2f}%였고, 사전 1% screen 규칙을 적용했다. 이 규칙은 유의성 검정이 아니다.')
        base=[c for c in cases if c['stage']=='G1' and c['dataset']==dataset and c['objective']=='PRED' and c['config']=='BASE']
        for config in ('LOW_LR','DECAY'):
            cc=[c for c in cases if c['stage']=='G1' and c['dataset']==dataset and c['objective']=='PRED' and c['config']==config]
            val_change=(average(cc)/average(base)-1)*100
            paired=[r for r in contrasts if r['stage']=='G1' and r['dataset']==dataset and r['contrast']==config+'_vs_BASE_same_objective_role' and r['case'].startswith('G1_PRED')]
            text.append(f'\n[확인] {dataset.title()} PRED {config}의 VAL 주오차 평균은 BASE 대비 {val_change:+.2f}%다. seed별 차이는 '+', '.join(f'{r["seed"]}: {r["val_normalized_mse_relative_percent"]:+.2f}%' for r in paired)+'다. 음수는 개선, 양수는 악화다.')
    text += ['', '[판단] 변화가 TUNE와 VAL에서 같은 방향인지, 두 seed와 네 시간 구간에서 반복되는지를 함께 본다. 낮은 TUNE 오차 하나만으로 일반화 해결을 선언하지 않는다. 모든 설정은 동일 초기 codec·배치 순서·600 update를 사용했고, 선택에 u0을 포함했다.', '',
             '![G1 학습·선택 비교](g1_generalization.png)', '',
             '### FIT 경계 이후의 과거 위치에서도 재구성이 개선되는가?', '']
    rows=[]
    for t in trains:
        if t['stage']=='G1' and t['objective']=='REC':
            rows.append([t['dataset'],t['seed'],t['config'],
                         f"{t['u0_reconstruction_fit_normalized_mse']:.6f}→{t['u600_reconstruction_fit_normalized_mse']:.6f}",
                         f"{t['u0_reconstruction_tune_normalized_mse']:.6f}→{t['u600_reconstruction_tune_normalized_mse']:.6f}",
                         f"{t['u0_reconstruction_tune_postfit_normalized_mse']:.6f}→{t['u600_reconstruction_tune_postfit_normalized_mse']:.6f}",
                         t['post_fit_unique_timestamps'],t['post_fit_reconstruction_count_sum']])
    text.append(format_rows(['자료','seed','설정','FIT normMSE u0→600','TUNE 창 전체 u0→600','TUNE post-FIT 위치 u0→600','고유시점','관측위치합'],rows))
    rec=[t for t in trains if t['stage']=='G1' and t['objective']=='REC']
    fit_better=sum(t['u600_reconstruction_fit_normalized_mse']<t['u0_reconstruction_fit_normalized_mse'] for t in rec)
    post_worse=sum(t['u600_reconstruction_tune_postfit_normalized_mse']>t['u0_reconstruction_tune_postfit_normalized_mse'] for t in rec)
    text.append(f'\n[확인] REC {len(rec)}개 fit 중 u600의 FIT 복원 주오차는 {fit_better}개에서 줄었고, FIT 이후 TUNE 위치의 복원 주오차는 {post_worse}개에서 커졌다. **복원 학습의 안정적 일반화가 확보됐다고 전제할 수 없다.** [판단] 이 조건의 일반화 한계와 미래 예측에 적합한 목적의 문제를 분리해야 하며, “복원은 성공했지만 미래만 실패”나 특정 방법의 본질적 한계로 단정하지 않는다.')
    text += ['', '[확인] post-FIT 점수는 TUNE 과거 창 중 FIT 끝 이후의 실제 관측 위치만 포함한다. TUNE 전체 과거 창에는 FIT 위치도 섞여 있어 이 보조 검사를 추가했다. 같은 시점이 여러 창에 반복되므로 관측위치합을 독립 표본수로 읽지 않는다. 이 점수는 checkpoint 선택에 쓰지 않았다. 전체 44 fit의 checkpoint 원수치는 [reconstruction_curves.csv](reconstruction_curves.csv)에 있다.', '',
             '### 성분 수·시간 범위 중 어디에서 손실이 생기는가?', '']
    rows=[]
    for dataset in ('jena','robin'):
        group=full_group(cases,dataset,selections[dataset]['chosen_config'])
        for v in ['Quarter K','Half K','Full PCA rotation','Full identity','Quarter K | L128','Quarter K | RF31']:
            cc=sorted([c for c in group if variant(c)==v],key=lambda c:c['seed'])
            rows.append([dataset,v,str(cc[0]['k']),str(cc[0]['length']),str(cc[0]['receptive_field']),
                         f'{average(cc,split="tune"):.6f}',f'{average(cc):.6f}',
                         ','.join(f'{score_value(method(c,"BOLT")):.6f}' for c in cc),
                         f'{average(cc,scale="physical",kind="mae"):.6f}'])
    text.append(format_rows(['자료','변형','K','L','RF','TUNE normMSE 평균','VAL normMSE 평균','VAL seed1,2','VAL 원단위MAE 평균'],rows))
    for dataset in ('jena','robin'):
        group=full_group(cases,dataset,selections[dataset]['chosen_config'])
        baseline=[c for c in group if variant(c)=='Quarter K']
        changes=[]
        for v in ['Half K','Full PCA rotation','Full identity','Quarter K | L128','Quarter K | RF31']:
            cc=[c for c in group if variant(c)==v]
            changes.append(f'{v}: {(average(cc)/average(baseline)-1)*100:+.2f}%')
        text.append(f'\n[확인] {dataset.title()} quarter-K 기본 대비 각 변형의 VAL 주오차 평균 변화는 '+', '.join(changes)+'다. 이 수치는 사전 대조의 진단이며 VAL에서 변형을 선택해 후속 실험을 바꾸지 않았다.')
        fullpca=[c for c in group if variant(c)=='Full PCA rotation']
        if all(selected_update(c)==0 for c in fullpca):
            raw=[next(c for c in cases if c['dataset']==dataset and c['label']=='RAW')]
            text.append(f'\n[확인] {dataset.title()} full PCA 두 seed는 모두 u0을 선택했다. BOLT의 VAL normMSE {average(fullpca):.6f} 대 RAW BOLT {average(raw,"RAW_BOLT"):.6f}는 **학습 전 고정 회전의 효과**다. 학습으로 좋아졌다는 증거로 읽지 않는다. 원단위 MAE는 full PCA {average(fullpca,scale="physical",kind="mae"):.3f}, RAW BOLT {average(raw,"RAW_BOLT",scale="physical",kind="mae"):.3f}로 함께 보고한다.')
    text += ['', '[확인] K=C의 PCA는 가역 회전이어도 Bolt가 보는 시계열이 원자료와 달라진다. 항등 초기화의 u0 실제 Bolt 출력은 RAW Bolt와 정확히 같았다. K 변화는 E/D 크기도 바꾼다. RF31/127은 같은 6층·같은 파라미터 개수·같은 초기 weight를 쓰지만 dilation 위치가 달라 순수 정보 범위만의 인과 효과로 부르지 않는다.', '',
             '[확인] L128은 실제 128 시점 입력을 창 시작의 FIT mean으로 다시 forward-fill했다. L512를 채운 뒤 잘라서 오래된 관측을 가져오지 않았다. 따라서 L과 결측 처리 경계가 함께 달라진다. **L128+R의 전체 지원 범위는 R 때문에 512다.** 이는 L128 성분만 쓰는 BOLT 경로와 구분한다.', '',
             '![성분 수·시간 범위](g2_components_history.png)', '',
             '### 동결 Bolt가 같은 표현의 단순 예측기·원자료 기준군보다 추가 가치를 갖는가?', '']
    rows=[]
    for dataset in ('jena','robin'):
        config=selections[dataset]['chosen_config']
        cc=[c for c in full_group(cases,dataset,config) if variant(c)=='Quarter K']
        raw=[next(c for c in cases if c['dataset']==dataset and c['label']=='RAW')]
        for name in ('RAW_BOLT','RAW_BOLT_RAW_R','RAW_R_ONLY','RAW_SELF_LAG','BOLT','BOLT_RAW_R','NATIVE','NATIVE_RAW_R','BOLT_COMMON_D'):
            group=raw if name.startswith('RAW_') else cc
            times={b:float(np.mean([next(l for l in timings(method(c,name)) if l['batch']==b)['median_seconds_per_window']*1000 for c in group])) for b in (1,8)}
            rows.append([dataset,name,f'{average(group,name):.6f}',f'{average(group,name,kind="mae"):.6f}',
                         f'{average(group,name,scale="physical"):.3f}',f'{average(group,name,scale="physical",kind="mae"):.3f}',
                         f'{times[1]:.3f}',f'{times[8]:.3f}'])
    text.append(format_rows(['자료','방법','VAL normMSE','VAL normMAE','VAL 원MSE','VAL 원MAE','b1 ms/창','b8 ms/창'],rows))
    for dataset in ('jena','robin'):
        cc=[c for c in full_group(cases,dataset,selections[dataset]['chosen_config']) if variant(c)=='Quarter K']
        raw=[next(c for c in cases if c['dataset']==dataset and c['label']=='RAW')]
        bolt_native=(average(cc,'BOLT')/average(cc,'NATIVE')-1)*100
        corrected_native=(average(cc,'BOLT_RAW_R')/average(cc,'NATIVE_RAW_R')-1)*100
        raw_change=(average(cc,'BOLT')/average(raw,'RAW_BOLT')-1)*100
        raw_r_change=(average(cc,'BOLT_RAW_R')/average(raw,'RAW_BOLT')-1)*100
        r_only_change=(average(cc,'BOLT_RAW_R')/average(raw,'RAW_R_ONLY')-1)*100
        d_change=(average(cc,'BOLT')/average(cc,'BOLT_COMMON_D')-1)*100
        text.append(f'\n[확인] {dataset.title()} quarter-K에서는 BOLT의 native 대비 VAL 주오차 변화가 {bolt_native:+.2f}%, BOLT+R의 native+R 대비 변화가 {corrected_native:+.2f}%다. RAW BOLT 대비 BOLT는 {raw_change:+.2f}%, BOLT+R는 {raw_r_change:+.2f}%다. BOLT+R의 RAW R-only 대비 변화는 {r_only_change:+.2f}%다. 학습된 D를 common reconstruction D로 바꾸기 전 BOLT 오차의 상대 차이는 {d_change:+.2f}%다. 두 seed 평균이며 반대 방향 지표는 위 표와 모든 짝 대조에 함께 남겼다.')
    rows=[]
    for dataset in ('jena','robin'):
        group=full_group(cases,dataset,selections[dataset]['chosen_config'])
        raw=[next(c for c in cases if c['dataset']==dataset and c['label']=='RAW')]
        for v in ['Quarter K','Half K','Full PCA rotation','Full identity','Quarter K | L128','Quarter K | RF31']:
            cc=sorted([c for c in group if variant(c)==v],key=lambda c:c['seed'])
            rows.append([dataset,v,cc[0]['k'],f'{average(cc,"BOLT"):.6f}',f'{average(cc,"BOLT_RAW_R"):.6f}',
                         f'{average(cc,"NATIVE"):.6f}',f'{average(cc,"NATIVE_RAW_R"):.6f}',
                         f'{(average(cc,"BOLT_RAW_R")/average(raw,"RAW_BOLT")-1)*100:+.2f}',
                         f'{average(cc,"BOLT_RAW_R",scale="physical",kind="mae"):.3f}'])
    text += ['', '같은 분석을 quarter-K에만 한정하지 않고 모든 사전 변형에서 수행했다. 아래는 두 seed 평균이다. 실험 조건 선택은 TUNE만 사용했고, 이 표에서 VAL로 선택을 변경하지 않았다.', '',
             format_rows(['자료','변형','K','BOLT normMSE','BOLT+R normMSE','native normMSE','native+R normMSE','BOLT+R/RAW BOLT %','BOLT+R 원MAE'],rows)]
    for dataset in ('jena','robin'):
        group=full_group(cases,dataset,selections[dataset]['chosen_config'])
        identity=sorted([c for c in group if variant(c)=='Full identity'],key=lambda c:c['seed'])
        half=[c for c in group if variant(c)=='Half K']
        raw=[next(c for c in cases if c['dataset']==dataset and c['label']=='RAW')]
        half_error=(average(half,'BOLT')/average(raw,'RAW_BOLT')-1)*100
        half_cost={b:float(np.mean([next(l for l in timings(method(c,'BOLT')) if l['batch']==b)['median_seconds_per_window']*1000 for c in half])) for b in (1,8)}
        raw_cost={b:next(l for l in timings(method(raw[0],'RAW_BOLT')) if l['batch']==b)['median_seconds_per_window']*1000 for b in (1,8)}
        text.append(f'\n[확인] {dataset.title()} half-K는 RAW BOLT 대비 주오차 평균이 {half_error:+.2f}%다. clean batch8 비용은 {half_cost[8]:.3f} 대 {raw_cost[8]:.3f} ms/창({(half_cost[8]/raw_cost[8]-1)*100:+.2f}%), batch1은 {half_cost[1]:.3f} 대 {raw_cost[1]:.3f} ms/창({(half_cost[1]/raw_cost[1]-1)*100:+.2f}%)다. **작은 K의 정확도 우위와 특정 batch의 비용 교환을 분리해서 판단해야 한다.**')
        change=(average(identity,'BOLT')/average(raw,'RAW_BOLT')-1)*100
        corrected=(average(identity,'BOLT_RAW_R')/average(raw,'RAW_BOLT_RAW_R')-1)*100
        if all(selected_update(c)==0 for c in identity) and abs(change)<1e-5 and abs(corrected)<1e-5:
            text.append(f'\n[확인] {dataset.title()} full identity 두 seed는 모두 u0을 선택했다. 미래 BOLT와 BOLT+R의 주오차는 각각 RAW BOLT와 RAW BOLT+R와 수치상 동일하다. **이 조건에서는 학습에 의한 개선이 없고, 원자료 기능과 같은 초기 state가 선택됐다.** 별도 E/D 경로의 추론 비용은 추가로 측정하되 동일한 기능을 새 성능 이득으로 부르지 않는다.')
        else:
            text.append(f'\n[확인] {dataset.title()} full identity의 RAW BOLT 대비 주오차 변화는 {change:+.2f}%, full identity BOLT+R의 RAW BOLT+R 대비 변화는 {corrected:+.2f}%다. **K=C인 양성 결과는 압축의 이득을 입증하지 않는다.** [판단] 이 조건이 나아지면 원 좌표에서 시작한 미래 목적의 E/D adaptation이 유효할 가능성으로 해석하고, K<C 조건의 정확도/비용 근거와 분리한다. 실제 수치가 나쁜 자료·seed도 함께 남긴다.')
        raw_bolt=method(raw[0],'RAW_BOLT')
        raw_corr=method(raw[0],'RAW_BOLT_RAW_R')
        text.append(f'\n반대 방향 지표도 같은 문단에서 확인한다. {dataset.title()} full identity BOLT의 seed별 정규화 MAE는 '+', '.join(f'{score_value(method(c,"BOLT"),kind="mae"):.6f}' for c in identity)+f', RAW BOLT는 {score_value(raw_bolt,kind="mae"):.6f}다. 원단위 MAE는 '+', '.join(f'{score_value(method(c,"BOLT"),scale="physical",kind="mae"):.3f}' for c in identity)+f', RAW BOLT는 {score_value(raw_bolt,scale="physical",kind="mae"):.3f}다. full identity BOLT+R의 정규화 MAE는 '+', '.join(f'{score_value(method(c,"BOLT_RAW_R"),kind="mae"):.6f}' for c in identity)+f', RAW BOLT+R는 {score_value(raw_corr,kind="mae"):.6f}다. 원단위 MAE는 '+', '.join(f'{score_value(method(c,"BOLT_RAW_R"),scale="physical",kind="mae"):.3f}' for c in identity)+f', RAW BOLT+R는 {score_value(raw_corr,scale="physical",kind="mae"):.3f}다. 주지표 MSE 개선을 모든 오차 지표의 개선으로 바꾸어 말하지 않는다.')
    text += ['', '[확인] 같은 selected E/D에 Bolt와 native shared linear를 연결했다. **E/D는 Bolt 미래 손실로 학습되어 native 전용으로 최적화된 표현이 아니다.** 따라서 Bolt가 native보다 좋아도 모든 native 전체 방법보다 우월하다는 비교는 아니다. native가 더 좋으면 이 조건에서 Bolt 필요성을 반박하는 사례다. Native는 추가 600 update를 학습하므로 총 학습 비용까지 동일하지는 않다. Bolt+R와 native+R는 같은 FIT/TUNE origins·R 특징·정규화 lambda 규칙을 사용한다. R은 FIT base의 in-sample 잔차로 학습했으며 cross-fitting은 하지 않았다. SELF33은 자신의 마지막 33 시점 특징만 사용하지만 결측 forward-fill의 지원 범위는 512다.', '',
             '[판단] 주어진 조건에서 Bolt 추가 가치는 동일 E/D의 native보다 낮은 주오차인지와, 전체 시스템이 RAW Bolt·R 단독·SELF33보다 유리한 정확도/비용 조합을 만드는지 두 단계로 판단한다. 압축 경로가 REC보다 나아지는 것만으로 전체 기여를 주장하지 않는다. 실제 양성·음성 대조의 차이는 [G3_paired_contrasts.csv](G3_paired_contrasts.csv)에 전부 남겼다.', '',
             '![같은 표현 예측기와 강한 원자료 기준군](g3_predictor_baselines.png)', '',
             '![정확도와 비용](g3_accuracy_latency.png)', '',
             '### 다음 방법이 필요한가?', '',
             '[판단] 대조학습·미래 표현 정렬·동적 그래프를 이번 결과의 설명으로 소급하지 않는다. 먼저 주지표가 일치한 학습 변경, K·시간 범위와 RAW/동일 E/D 대조가 어디까지 해결했는지 위 수치로 판단한다. 더 복잡한 학습 목적이 필요하다면 직접 Bolt 미래 손실보다 높여야 하는 성능과 기존 강한 원자료 기준군을 모두 명시해야 한다. 두 자료의 최선 경로가 다르면 하나의 범용 개선이라고 묶지 않는다.', '',
             '## 시간·채널·반대 방향 지표', '',
             '![시간 구간별 짝 비교](paired_time_blocks.png)', '',
             '각 seed의 동일 VAL origins를 시간 순서로 네 부분으로 나누어 관측 마스크 기반 channel-macro 오차 차이를 다시 계산했다. 미래 target이 겹치므로 네 구간을 독립 반복실험이나 유의성 증거로 쓰지 않는다. 같은 paired contrast의 정규화 MSE/MAE와 원단위 MSE/MAE는 [chronological_blocks.csv](chronological_blocks.csv), 모든 채널의 평균·표준편차·관측 수·오차는 [channel_metrics.csv](channel_metrics.csv)에 있다. 채널 균등 정규화는 단위를 제거하지만 표준편차가 작은 채널의 큰 표준화 오차도 주지표에 반영한다. 원단위 macro는 이질 단위를 평균한 보조 지표이므로 자료 간 합산하지 않는다.', '',
             '## 실제 실행·재사용·비용', '']
    native=[m['native_training'] for c in cases for m in c['methods'] if 'native_training' in m]
    correction=[m['correction'] for c in cases for m in c['methods'] if 'correction' in m]
    self_fit=[m['selection'] for c in cases for m in c['methods'] if m['method']=='RAW_SELF_LAG']
    rss=max(c['resident_rss_bytes'] for c in costs)
    vram=max(c['co_resident_allocated_vram_bytes'] for c in costs)
    resource=result.get('resources',{})
    attempts=result.get('attempt_accounting',{})
    text += [format_rows(['실행 항목','실제 수/시간'],[
        ['G0 저장된 fit 재사용, 새 BOLT 추론',str(sum(c['stage']=='G0' for c in cases))+' case; 추가 E/D 학습 0'],
        ['G1 신규 E/D fit',str(sum(t['stage']=='G1' for t in trains))+' × 600 update'],
        ['G2 신규 E/D fit',str(sum(t['stage']=='G2' for t in trains))+' × 600 update'],
        ['G1/G2 native 유효조건의 추가 fit',str(len(native))+' × 600 update'],
        ['수리 포함 core 실제 attempt / u600완료 / 유효조건',f"{attempts.get('core_fit_attempts_including_repairs','-')} / {attempts.get('completed_core600_fit_runs_including_duplicate_repair','-')} / {len(trains)}"],
        ['u600 미도달 core attempt',str(attempts.get('core_fit_attempts_without_u600_checkpoint','-'))],
        ['수리 포함 native 실제 u600완료 / 유효조건',f"{attempts.get('native600_fit_runs_including_duplicate_repair','-')} / {len(native)}"],
        ['평가된 method 행',str(sum(len(c['methods']) for c in cases))],
        ['batch 1/8 비용 행',str(len(costs))],
        ['중복 제거한 E/D training wall',f"{sum(t['train_wall_seconds'] for t in trains):.3f} s"],
        ['native training wall',f"{sum(t['train_wall_seconds'] for t in native):.3f} s"],
        ['ridge fitting wall',f"{sum(t['train_wall_seconds'] for t in correction+self_fit):.3f} s"],
        ['메인·수리·preflight·clean GPU process wall 합',f"{resource.get('own_gpu_process_wall_seconds',resource.get('wall_seconds',0)):.3f} s"],
        ['별도 metric_alignment 보충의 GPU process wall 합',f"{resource.get('supplement_gpu_process_wall_seconds',0):.3f} s"],
        ['동일 사용자 task 전체 GPU process wall 합',f"{resource.get('all_gpu_process_wall_seconds_summed_conservatively',resource.get('wall_seconds',0)):.3f} s"],
        ['모델 로드 포함 여부', '각 실행 body에 포함; source_runs 원기록의 bolt_load_wall_seconds 참조'],
        ['cost 측정 최대 resident RSS',f'{rss/1024**3:.3f} GiB'],
        ['cost 측정 공존 peak allocated VRAM',f'{vram/1024**2:.3f} MiB']]), '',
        '[확인] E/D fit 시간은 동일 REC fit의 두 selection 역할을 중복 합산하지 않았다. 같은 checkpoint를 두 역할로 추론하면 평가 행은 두 개지만 학습은 하나다. 유효 44 fit과 실제 수리 포함 48 attempt를 구분하며 중복·실패의 시간도 task process wall 합에는 포함했다. 측정된 train wall은 코드의 training 함수 범위이며 checkpoint 점수·저장을 포함한다. CUDA kernel 순수 실행시간이나 전체 캠페인 wall과 동일하지 않다. 보충 study의 학습·GPU process wall도 같은 사용자 task 자원 합에 넣었지만 236 method 과학적 비교 행에는 섞지 않았다. 동시에 실행된 시간이 있어 summed process wall은 실제 elapsed wall과 다르다.', '',
        '[확인] 표와 그림의 주 비용은 **모든 Python GPU 학습 종료 후 저장된 모델에서 다시 측정한 clean_latency**다. 최초 측정에는 별도 보충 학습과의 동시 경합 가능성이 있었으므로 그 472행은 [batch_cost_original.csv](batch_cost_original.csv)와 원 RESULT latency 필드로 보존했다. clean 236방법×batch1/8 재측정은 [run_clean_cost_v1/RESULT.json](../run_clean_cost_v1/RESULT.json)에 있고 추가 학습은0이다. 일반 데스크톱 GPU 앱은 그대로 두었으므로 완전히 전용인 장비의 benchmark는 아니다.', '',
        '[확인] 비용은 batch1/8, warmup2, repeat7의 median이다. CPU forward-fill·정규화, H2D, E/F/D, 선택적 R512 특징/추론, 역정규화, D2H, synchronize를 포함한다. 디스크·모델 로드는 제외했고 Bolt 로드는 각 실행 원기록에 별도 기록했다. 모든 측정에 Bolt와 현재 모듈이 함께 resident이므로 공존 allocated VRAM을 방법별 독립 절감량으로 해석하지 않는다. 7회 최소/최대·개별값·parity·RSS/VRAM은 [batch_cost.csv](batch_cost.csv)에 있다.', '',
        '## 실패 attempt와 수치 경로 수리', '',
        '[확인] 최초 [run_full_v1/RESULT.json](../run_full_v1/RESULT.json)은 212개 method를 완료하고 네 경로의 실패를 그대로 보존했다. 두 경로는 resource ledger 파일 교체 중 Windows 파일 잠금 오류였고, Jena full identity 두 seed는 상수 rain 입력의 Bolt 내부 정규화 backward에서 비유한 gradient가 생겼다. 낮은 정확도 때문에 중단된 경로가 아니다. [run_repair_v1/RESULT.json](../run_repair_v1/RESULT.json)에서 네 경로를 추가 실행했다. 기존 성공 결과·실패 파일·실행 source를 덮지 않았다.', '',
        '[확인] [gradient 진단](../run_gradient_diagnosis_v1/RESULT.json)은 두 seed 첫 batch에서 원 forward loss를 보존하면서 E/D gradient와 실제 일시 optimizer update가 유한한지 확인했다. InstanceNorm의 분산0에서 sqrt는 미분이 정의되지 않으므로, backward0은 유일한 수학적 gradient를 복구했다는 주장이 아니라 **forward를 보존하는 finite 수치 convention**이다. 검사 fixture에서 모든 forward 최대 차이0, 양분산 gradient 최대 차이0였고 동결 모델 weight SHA도 같았다. 이 범위와 두 seed 재실행으로 검증했으며 가능한 모든 입력의 동등성을 증명하지 않는다. 일시 diagnostic step은 원 state로 복원했다.', '',
        '## 검증과 적용 범위', '',
        f'- 실행 상태: `{result["status"]}`. 원 실행 결과는 [{run_name}/RESULT.json](../{run_name}/RESULT.json).',
        f'- Frozen Bolt 초기/최종 state SHA 일치: `{result["bolt_initial_hash"] == result["bolt_final_hash"]}`.',
        f'- 실행 source SHA 최종 보존 확인: `{result.get("source_hashes_unchanged_at_completion", "해결본의 개별 실행 provenance 확인")}`. 실행 파일의 source_snapshots 기록으로 원본 스냅샷을 확인한다.',
        '- 실제 E/D gradient 및 update, 공식 Bolt 추론 parity, causal counterfactual, full identity와 RAW parity, RF31 matched-capacity/support, Adam/AdamW zero-decay 5-step parity를 fixture로 확인했다. fixture 통과는 성능 개선의 증거가 아니다.',
        '- 신규 비교의 init/order·TUNE 선택·NPZ 재채점·source/data/origin SHA·실제 비용 경로를 별도 검산 결과와 연결했다.',
        '- FIT256, TUNE64, VAL Jena371/Robin30 origins, H48, 두 seed, 같은 기존 TRAIN/VAL 자료다. 새 TEST 접근 없이 수행했다. 여러 model을 이 VAL로 이미 진단했고 TUNE도 checkpoint/optimizer/ridge 선택에 반복 사용했으므로 일반화 주장은 제한된다.',
        '- Common D는 모든 E에서 같은 FIT 복원 적합 규칙이지만 수치 weight는 다르다. E와 D를 동시에 학습했기 때문에 표현 자체의 기여를 D로부터 완전히 분리한 인과실험은 아니다. K 변화는 압축량뿐 아니라 E/D/output 파라미터 수도 바꾼다.',
        '- BASE/LOW_LR/DECAY·K·history만의 제한된 screen이다. 더 긴 학습·학습 origin 확대·다른 백본·새 목적·새 자료는 수행하지 않았다. 낮은 성능으로 계획된 대조를 생략하지 않았다.', '']
    if validation_path:
        text.append('독립 검산: ['+validation_path.name+']('+Path('..',validation_path.relative_to(HERE)).as_posix()+'). 검사 버전별 오류·수정 기록은 원파일을 보존한다.')
    else:
        text.append('독립 검산 상태는 생성 당시 제공되지 않았으므로 이 보고서만으로 최종 완료를 선언하지 않는다. 후속 최종 검산 기록을 확인해야 한다.')
    source_audit=HERE/'validation_source_v1/audit.json'
    if source_audit.exists():
        source_validation=read(source_audit)
        if source_validation['status']!='PASS':
            raise AssertionError('Raw-source companion verification did not pass')
        text += ['', '[확인] 강화된 원자료 검산도 [validation_source_v1/audit.json](../validation_source_v1/audit.json)·[설명](../validation_source_v1/audit.md)에서 PASS다. 실제 TRAIN/VAL source NPZ 두 개만 열어 FIT mean/std, 전체 정답·관측 마스크·origins를 재구성했고 236 prediction artifact와 일치했다. 실제 L128/512 경계에서 forward-fill을 다시 계산했고 post-FIT 위치의 mask/count SHA 및 source/snapshot SHA를 검사했다. 이 검사는 기록된 데이터·정보 경계와 구현 계약의 일치를 확인하며, **가능한 모든 정보 누출의 부재를 증명하거나 독립적인 일반화 성능을 검증하지 않는다.**']
    text += ['', '## 원수치·재현 파일', '',
             '- [all_methods.csv](all_methods.csv): 모든 case/method의 FIT/TUNE/VAL 정규화·원단위 MSE/MAE, 선택·지원 범위·prediction SHA.',
             '- [unique_training.csv](unique_training.csv): 중복 selection 역할을 제거한 E/D fit·초기값·order SHA·학습 wall.',
             '- [G0_paired_contrasts.csv](G0_paired_contrasts.csv), [G1_paired_contrasts.csv](G1_paired_contrasts.csv), [G2_paired_contrasts.csv](G2_paired_contrasts.csv), [G3_paired_contrasts.csv](G3_paired_contrasts.csv): 모든 사전 대조의 원수치·짝 차이.',
             '- [reconstruction_curves.csv](reconstruction_curves.csv): FIT/TUNE/post-FIT 복원과 미래 예측의 모든 저장 checkpoint.',
             '- [chronological_blocks.csv](chronological_blocks.csv), [channel_metrics.csv](channel_metrics.csv), [batch_cost.csv](batch_cost.csv): 시간·채널·비용.',
             '- [REPORT_MANIFEST.json](REPORT_MANIFEST.json): 입력 결과/기존 재사용 결과/보고 소스/산출 파일 SHA 및 보고 재채점 검사.',
             '- [실행 계획](../PLAN_KO.md), [실험 코드](../experiment.py), [보고 생성 코드](../make_report.py).', '',
             '그림은 모두 저장된 JSON/NPZ 원수치로 생성했다. PNG와 동일 stem의 SVG를 함께 저장했다. SCIENCE 실험·기존 결과·cache를 변경하거나 새 모델을 학습하지 않고 보고서만 생성한다.', '']
    return '\n'.join(text)


def generate(run_name,report_name,validation,finish_existing=False):
    started=time.perf_counter()
    source=HERE/run_name/'RESULT.json'
    result=read(source)
    if result['status']!='COMPLETE' or result.get('failed_paths'):
        raise ValueError('A full completed result is required; no missing results are inferred')
    cases=result['cases']
    out=HERE/report_name
    if out.exists() and not finish_existing:
        raise FileExistsError(out)
    if finish_existing and (not out.exists() or (out/'REPORT_MANIFEST.json').exists()):
        raise ValueError('Finish-existing requires an existing report without a manifest')
    old_path=OLD/'stage_b_full_v1/RESULT.json'
    old=read(old_path)
    methods=flatten_methods(cases)
    trains=training_rows(cases)
    contrasts,pairs=make_comparisons(cases,old)
    blocks=chronological_rows(pairs)
    costs=cost_rows(cases)
    channels=channel_rows(cases,result['contracts'])
    curves=reconstruction_rows(cases)
    expected={'methods':236,'G0':12,'G1_unique_training':24,'G2_unique_training':20,'native_training':32,'latency_rows':472}
    actual={'methods':len(methods),'G0':sum(c['stage']=='G0' for c in cases),
            'G1_unique_training':sum(t['stage']=='G1' for t in trains),'G2_unique_training':sum(t['stage']=='G2' for t in trains),
            'native_training':sum('native_training' in m for c in cases for m in c['methods']), 'latency_rows':len(costs)}
    if actual!=expected:
        raise AssertionError(f'Expected full scope differs: {actual}, expected {expected}')
    metrics_checks=[]
    for c in cases:
        for m in c['methods']:
            z=prediction(m['prediction']['path'])
            nmse,nmae=macro(z['val_prediction'],z['val_target'],z['val_observed'])
            pmse,pmae=macro(z['val_prediction']*z['std']+z['mean'],z['val_target']*z['std']+z['mean'],z['val_observed'])
            errors={'normalized_mse':abs(nmse-score_value(m)), 'normalized_mae':abs(nmae-score_value(m,kind='mae')),
                    'physical_mse':abs(pmse-score_value(m,scale='physical')), 'physical_mae':abs(pmae-score_value(m,scale='physical',kind='mae'))}
            if any(v>1e-9 for v in errors.values()):
                raise AssertionError('Report NPZ rescore differs: '+c['label']+'/'+m['method']+str(errors))
            matching_hash=sha(ROOT/m['prediction']['path'])==m['prediction']['sha256']
            if not matching_hash:
                raise AssertionError('Prediction hash changed: '+m['prediction']['path'])
            metrics_checks.append({'case':c['label'],'method':m['method'],'errors':errors,'prediction_sha_match':matching_hash})
    if not finish_existing:
        out.mkdir(exist_ok=False)
        write_csv(out/'all_methods.csv',methods)
        write_csv(out/'unique_training.csv',trains)
        for stage in ('G0','G1','G2','G3'):
            write_csv(out/(stage+'_paired_contrasts.csv'),[r for r in contrasts if r['stage']==stage])
        write_csv(out/'chronological_blocks.csv',blocks)
        write_csv(out/'batch_cost.csv',costs)
        write_csv(out/'batch_cost_original.csv',cost_rows(cases,original=True))
        write_csv(out/'channel_metrics.csv',channels)
        write_csv(out/'reconstruction_curves.csv',curves)
        draw(cases,result.get('configuration_selection',result.get('stage2_configuration_selection')),blocks,out)
    validation_path=Path(validation).resolve() if validation else None
    if validation_path and not validation_path.exists():
        raise FileNotFoundError(validation_path)
    report=narrative(result,cases,trains,contrasts,costs,validation_path,run_name)
    if finish_existing:
        if (out/'REPORT_KO.md').read_text(encoding='utf-8') != report:
            raise AssertionError('Existing report differs from regenerated text')
    else:
        (out/'REPORT_KO.md').write_text(report,encoding='utf-8')
    links=re.findall(r'\]\(([^)]+)\)',report)
    link_checks=[]
    for link in links:
        if not link.startswith(('http:','https:')):
            target=(out/link).resolve()
            self_manifest=target==out/'REPORT_MANIFEST.json'
            if not target.exists() and not self_manifest:
                raise AssertionError('Broken report link: '+link)
            link_checks.append({'link':link,'exists':True,'checked_after_manifest_write':self_manifest})
    manifest={'status':'DERIVED_REPORT_COMPLETE', 'input_result':str(source.relative_to(ROOT)), 'input_result_sha256':sha(source),
              'old_reused_result':str(old_path.relative_to(ROOT)), 'old_reused_result_sha256':sha(old_path),
              'report_generator_sha256':sha(__file__), 'actual_scope':actual, 'expected_scope':expected,
              'source_runs':result.get('source_runs'), 'attempt_accounting':{k:v for k,v in result.get('attempt_accounting',{}).items() if k!='completion_evidence'},
              'cost_protocol':result.get('cost_protocol'),
              'primary_metric':result['primary_metric'], 'independent_validation_file':str(validation_path.relative_to(ROOT)) if validation_path else None,
              'independent_validation_sha256':sha(validation_path) if validation_path else None,
              'raw_source_validation':{'path':str((HERE/'validation_source_v1/audit.json').relative_to(ROOT)),
                                       'sha256':sha(HERE/'validation_source_v1/audit.json')} if (HERE/'validation_source_v1/audit.json').exists() else None,
              'prediction_recalculation_checks':metrics_checks, 'relative_link_checks':link_checks,
              'outputs':{p.name:sha(p) for p in out.iterdir() if p.is_file()},
              'report_wall_seconds':time.perf_counter()-started,
              'finished_existing_outputs':finish_existing,
              'no_training_in_report_generator':True, 'no_mutation_of_science_results':True}
    with (out/'REPORT_MANIFEST.json').open('x',encoding='utf-8') as f:
        json.dump(manifest,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')
    for item in link_checks:
        if not (out/item['link']).resolve().exists():
            raise AssertionError('Post-manifest link does not exist: '+item['link'])
    print(json.dumps({'status':manifest['status'],'scope':actual,'report':str(out/'REPORT_KO.md'),
                      'wall_seconds':manifest['report_wall_seconds']},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='run_full_v1')
    parser.add_argument('--report',default='report_v1')
    parser.add_argument('--validation')
    parser.add_argument('--finish-existing',action='store_true')
    args=parser.parse_args()
    generate(args.run,args.report,args.validation,args.finish_existing)
