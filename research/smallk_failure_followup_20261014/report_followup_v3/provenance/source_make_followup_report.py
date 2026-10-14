"""CPU report adapter for terminal D0/D1/D2/D3 evidence; no reads on import."""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import re
import shutil
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATASETS = ('robin', 'beijing_aotizhongxin')
SEEDS = (92601, 92602)
DISPLAY = {'robin': 'Robin C17/K4 · 반복 VAL', 'beijing_aotizhongxin': 'Beijing C11/K2 · 노출 EVAL'}
TERMINAL = {'COMPLETE', 'FAILED_PRESERVED', 'BUDGET_HELD_PRESERVED'}
COLORS = {'RAW': '#333333', 'MAIN': '#0072B2', 'MAIN_NATIVE': '#D55E00',
          'projected_raw': '#0072B2', 'projected_past': '#D55E00',
          'ORIGINAL_D': '#777777', 'Y': '#0072B2', 'RAW_teacher': '#009E73', 'MIXED': '#CC79A7'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def save_csv(path, rows):
    if not rows:
        raise ValueError('EMPTY_REPORT_TABLE: ' + str(path))
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def checked_json(path, expected=None):
    path = resolve(path)
    digest = sha(path)
    if expected is not None and digest != expected:
        raise ValueError('REPORT_INPUT_HASH_MISMATCH: ' + str(path))
    return read(path), {'path': str(path), 'sha256': digest, 'file_hash_rechecked': True}


def load_producer(path, block):
    producer, record = checked_json(path)
    if producer.get('status') not in TERMINAL:
        raise ValueError('TERMINAL_PRODUCER_REQUIRED: ' + str(path))
    if producer.get('dataset') not in DATASETS:
        raise ValueError('UNKNOWN_REPORT_DATASET')
    if block in ('D2', 'D3') and producer.get('seed') not in SEEDS:
        raise ValueError('UNKNOWN_REPORT_SEED')
    receipt = producer.get('shared_resource_receipt', {})
    if receipt.get('terminal_status', producer['status']) != producer['status']:
        raise ValueError('PRODUCER_RESOURCE_TERMINAL_STATUS_MISMATCH')
    item = {'block': block, 'producer': producer, 'record': record, 'folder': resolve(path).parent}
    if block in ('D0', 'D1', 'D2') and producer['status'] == 'COMPLETE':
        key = 'diagnostic_result' if block == 'D0' else 'probe_result'
        child = producer[key]
        scientific, child_record = checked_json(child['path'], child['sha256'])
        accepted = 'COMPLETE_SAVED_OUTPUT_DIAGNOSTICS' if block == 'D0' else 'COMPLETE'
        if scientific.get('status') != accepted:
            raise ValueError('COMPLETE_PRODUCER_CHILD_NOT_COMPLETE')
        if block != 'D0' and scientific.get('scientific_execution') is not True:
            raise ValueError('ACTUAL_SCIENTIFIC_EXECUTION_REQUIRED')
        item.update(scientific=scientific, child_record=child_record,
                    child_folder=resolve(child['path']).parent)
    return item


def assess_coverage(items):
    seen = set()
    complete = set()
    for item in items:
        p, block = item['producer'], item['block']
        key = (block, p['dataset'], p.get('seed') if block in ('D2', 'D3') else None)
        if key in seen:
            raise ValueError('DUPLICATE_REPORT_CONDITION: ' + str(key))
        seen.add(key)
        if p['status'] == 'COMPLETE':
            complete.add(key)
        if block == 'D0' and p['status'] == 'COMPLETE':
            lineage = p.get('native_same_MAIN_lineage', {})
            if set(lineage) != {str(seed) for seed in SEEDS} or not all(row.get('same_E_D_state_torch_equal') is True for row in lineage.values()):
                raise ValueError('SAME_MAIN_NATIVE_STATE_PROOF_REQUIRED')
        if block == 'D1' and p['status'] == 'COMPLETE':
            science = item['scientific']
            if len(science.get('candidates', [])) != 7 or p.get('F_frozen_pass') is not True:
                raise ValueError('D1_SEVEN_CANDIDATES_FROZEN_F_REQUIRED')
        if block == 'D2' and p['status'] == 'COMPLETE':
            if p.get('F_frozen_pass') is not True or p.get('MAIN_codec_frozen_pass') is not True:
                raise ValueError('D2_FROZEN_E_F_REQUIRED')
            if set(item['scientific'].get('refits', {})) != {'Y', 'RAW', 'MIXED'}:
                raise ValueError('D2_ALL_THREE_OBJECTIVES_REQUIRED')
            for values in item['scientific']['refits'].values():
                grid = values['alpha_grid']
                chosen = min(range(len(grid)), key=lambda index: float(grid[index]['tune_mse']))
                if chosen != values['selected_index'] or grid[chosen]['alpha'] != values['selected_alpha']:
                    raise ValueError('D2_TUNE_SELECTION_METADATA_MISMATCH')
        if block == 'D3' and p['status'] == 'COMPLETE':
            rows = p.get('rows', [])
            expected = {(method, batch, preprocessing) for method in ('RAW_BOLT', 'MAIN')
                        for batch in (1, 8) for preprocessing in ('old_loop', 'vectorized')}
            actual = {(r.get('method'), r.get('batch'), r.get('preprocessing')) for r in rows}
            if len(rows) != 8 or actual != expected or any(r.get('status') != 'COMPLETE' for r in rows):
                raise ValueError('D3_EIGHT_ROUTE_COVERAGE_REQUIRED')
            if p.get('F_before') != p.get('F_after') or p.get('codec_before') != p.get('codec_after'):
                raise ValueError('D3_F_CODEC_HASH_PRESERVATION_REQUIRED')
            for row in rows:
                if (row.get('warmup') != 2 or row.get('repeats') != 7 or len(row.get('whole_seconds', [])) != 7
                        or len(row.get('whole_parity', [])) != 7 or len(row.get('decomposed_parity', [])) != 7
                        or len(row.get('decomposed_seconds', [])) != 7):
                    raise ValueError('D3_WARM2_REPEAT7_REQUIRED')
                for check in row.get('whole_parity', []):
                    if not check['matched_batch_normalized']['pass'] or not check['matched_batch_physical']['pass']:
                        raise ValueError('D3_MATCHED_BATCH_GATE_FAILED')
                for check in row['decomposed_parity']:
                    if any(not value['pass'] for key, value in check.items() if key != 'repeat'):
                        raise ValueError('D3_DECOMPOSED_MATCHED_BATCH_GATE_FAILED')
    expected = {(block, dataset, seed if block in ('D2', 'D3') else None)
                for block in ('D0', 'D1', 'D2', 'D3') for dataset in DATASETS
                for seed in (SEEDS if block in ('D2', 'D3') else (None,))}
    if seen - expected:
        raise ValueError('UNEXPECTED_REPORT_CONDITION')
    return {'expected_conditions': len(expected), 'supplied_conditions': len(seen),
            'complete_conditions': len(complete), 'missing_conditions': [list(k) for k in sorted(expected - seen, key=str)],
            'noncomplete_conditions': [list(k) for k in sorted(seen - complete, key=str)],
            'all_declared_blocks_complete': seen == expected and complete == expected,
            'independent_holdout': False, 'improvement_established_by_coverage': False}


def aggregate_metrics(rows, keys, fields):
    groups = {}
    for row in rows:
        group = tuple(row[key] for key in keys)
        groups.setdefault(group, []).append(row)
    result = []
    for group, values in groups.items():
        row = dict(zip(keys, group))
        row['metric_rows_averaged'] = len(values)
        row['seed_aggregation'] = 'arithmetic_mean_of_metrics_not_prediction_ensemble'
        row.update({key: float(np.mean([float(value[key]) for value in values])) for key in fields})
        result.append(row)
    return result


def summarize_audit(audit, name):
    status = audit.get('status', 'UNKNOWN')
    checks = audit.get('checks')
    if isinstance(checks, list):
        count = len(checks)
        failed = sum(check.get('passed') is not True for check in checks)
        if audit.get('failed_count') != failed:
            raise ValueError('AUDIT_FAILURE_COUNT_MISMATCH')
    else:
        count, failed = checks, audit.get('failed_count')
    row = {'name': name, 'status': status, 'checks': count, 'failed_count': failed,
           'claim_scope': audit.get('claim_scope', audit.get('scope', 'scope declared in copied receipt')),
           'all_declared_scopes_passed': str(status).startswith('PASS') and failed == 0}
    if audit.get('schema') == 'smallk_d0_independent_array_audit_v1':
        datasets = audit.get('datasets', {})
        nested = {key: value.get('status') for key, value in datasets.items()}
        row.update(nested_dataset_statuses=nested,
            all_declared_scopes_passed=status == 'COMPLETE' and set(nested) == set(DATASETS)
                and all(value == 'PASS' for value in nested.values()),
            claim_scope='D0 independent saved-array arithmetic; per-dataset literal PASS; no model forward replay')
    return row


def collect_tables(items):
    tables = {key: [] for key in ('d0_summary', 'd0_channels', 'd0_tails', 'd0_rank',
                                 'd1', 'd2', 'd2_selection', 'd3', 'd3_stages', 'resources')}
    copies = []
    resource_paths = set()
    for item in items:
        p, block, dataset = item['producer'], item['block'], item['producer']['dataset']
        if p['status'] != 'COMPLETE':
            continue
        receipt = p.get('shared_resource_receipt')
        if receipt and 'path' in receipt:
            path = resolve(receipt['path'])
            if path not in resource_paths:
                resource, record = checked_json(path, receipt['sha256'])
                resource_paths.add(path)
                tables['resources'].append({'block': block, 'dataset': dataset, 'seed': p.get('seed', ''),
                    'tag': resource.get('tag', ''), 'gpu': resource.get('gpu'), 'status': resource.get('status'),
                    'wall_seconds': resource.get('wall_seconds'), 'wall_scope': resource.get('wall_scope'), **record})
        if block == 'D0':
            for path in item['child_folder'].glob('*.csv'):
                copies.append({'source': str(path), 'destination': f'D0_{dataset}_{path.name}', 'sha256': sha(path)})
            for key, name in (('d0_summary', 'summary.csv'), ('d0_channels', 'channels.csv'),
                              ('d0_tails', 'tails.csv'), ('d0_rank', 'rank_projection.csv')):
                tables[key].extend({'dataset': dataset, **row} for row in csv_rows(item['child_folder'] / name))
        elif block == 'D1':
            for candidate in item['scientific']['candidates']:
                for split, values in candidate['splits'].items():
                    for method, score in values['scores'].items():
                        tables['d1'].append({'dataset': dataset, 'candidate': candidate['name'], 'mode': candidate['mode'],
                            'K': candidate['K'], 'C': item['scientific']['current_state_bypass_C'], 'split': split, 'method': method,
                            'mse': score['mse'], 'mae': score['mae'], 'physical_mse': score['physical']['mse'],
                            'physical_mae': score['physical']['mae'],
                            'past_reconstruction_mse': values['past_reconstruction']['mse'],
                            'forecast_noncommutation_mse': values['forecast_noncommutation']['mse'],
                            'raw_projection_approximation_mse': values['raw_projection_vs_original_raw']['mse'],
                            'full_C_is_compression_claim': candidate['full_C_is_compression_claim'],
                            'requires_full_C_RAW_for_projected_RAW': method == 'projected_raw'})
        elif block == 'D2':
            source = item['child_folder'] / 'METRICS.csv'
            copies.append({'source': str(source), 'destination': f'D2_{dataset}_s{p["seed"]}_ALL_METRICS.csv', 'sha256': sha(source)})
            tables['d2'].extend({'dataset': dataset, 'seed': p['seed'], **row} for row in csv_rows(source))
            for objective, values in item['scientific']['refits'].items():
                tables['d2_selection'].append({'dataset': dataset, 'seed': p['seed'], 'objective': objective,
                    'selected_alpha': values['selected_alpha'], 'selected_index': values['selected_index'],
                    'tune_mse': values['alpha_grid'][values['selected_index']]['tune_mse'],
                    'fit_feature_RMS_hash': values['feature_rms_sha256'], 'weight_hash': values['selected_weight_sha256'],
                    'selection_rule': item['scientific']['selection_rule'],
                    'fallback_original_channels': json.dumps(values['missing_fit_channels_original_fallback'])})
        else:
            for row in p['rows']:
                base = {'dataset': dataset, 'seed': p['seed'], 'method': row['method'], 'batch': row['batch'],
                        'preprocessing': row['preprocessing']}
                checks = row['whole_parity']
                tables['d3'].append({**base, 'median_ms': row['whole_median_seconds'] * 1000,
                    'median_ms_per_window': row['whole_median_seconds_per_window'] * 1000,
                    'warmup': row['warmup'], 'repeats': row['repeats'],
                    'whole_seconds': json.dumps(row['whole_seconds']),
                    'matched_normalized_max_abs': max(c['matched_batch_normalized']['max_abs'] for c in checks),
                    'matched_physical_max_abs': max(c['matched_batch_physical']['max_abs'] for c in checks),
                    'original_bulk_normalized_failed_repeats': sum(not c['original_bulk_normalized']['pass'] for c in checks),
                    'original_bulk_physical_failed_repeats': sum(not c['original_bulk_physical']['pass'] for c in checks),
                    'whole_output_path': row['whole_outputs']['path'], 'whole_output_sha256': row['whole_outputs']['sha256']})
                for name, seconds in row['stage_median_seconds'].items():
                    tables['d3_stages'].append({**base, 'stage': name, 'median_ms_with_synchronization': seconds * 1000,
                        'stage_seconds': json.dumps([value[name] for value in row['decomposed_seconds']]),
                        'sum_is_whole_latency': False})
    tables['d0_summary_mean'] = aggregate_metrics(tables['d0_summary'], ('dataset', 'split', 'method'),
        ('normalized_mse', 'normalized_mae', 'mse_gap', 'mae_gap')) if tables['d0_summary'] else []
    tables['d0_channels_mean'] = aggregate_metrics(tables['d0_channels'], ('dataset', 'split', 'method', 'channel', 'channel_index'),
        ('normalized_mse', 'normalized_mae', 'target_normalized_mean', 'target_normalized_std',
         'prediction_normalized_mean', 'prediction_normalized_std')) if tables['d0_channels'] else []
    selected = [row for row in tables['d2'] if row['channel'] == 'macro' and row['metric_scale'] == 'normalized'
                and row['selected'] == 'True' and row['method'] in ('RAW', 'ORIGINAL_D', 'REFIT_Y', 'REFIT_RAW', 'REFIT_MIXED')]
    tables['d2_selected_macro'] = selected
    tables['d2_selected_mean'] = aggregate_metrics(selected, ('dataset', 'split', 'method'), ('mse', 'mae')) if selected else []
    return tables, copies


def configure_plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    available = {font.name for font in font_manager.fontManager.ttflist}
    font = 'Malgun Gothic' if 'Malgun Gothic' in available else 'DejaVu Sans'
    plt.rcParams.update({'font.family': font, 'font.size': 10, 'axes.unicode_minus': False,
        'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none',
        'savefig.dpi': 220, 'axes.titlesize': 11, 'axes.labelsize': 10})
    return plt, font


def finish_figure(fig, name, output, plt, note, *, bottom=.055, top=.95):
    fig.text(.5, .012, note, ha='center', va='bottom', fontsize=9)
    fig.tight_layout(rect=(0, bottom, 1, top))
    fig.savefig(output / (name + '.png'), bbox_inches='tight')
    fig.savefig(output / (name + '.svg'), bbox_inches='tight')
    plt.close(fig)


def figure_d0(tables, folder, plt):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for i, dataset in enumerate(DATASETS):
        splits = ('fit', 'tune', 'val' if dataset == 'robin' else 'eval')
        for j, metric in enumerate(('normalized_mse', 'normalized_mae')):
            ax = axes[i, j]
            for method in ('RAW', 'MAIN', 'MAIN_NATIVE'):
                values = [next((float(r[metric]) for r in tables['d0_summary_mean']
                    if r['dataset'] == dataset and r['method'] == method and r['split'] == split), np.nan) for split in splits]
                ax.plot(range(3), values, marker='o', color=COLORS[method], label=method)
                for row in tables['d0_summary']:
                    if row['dataset'] == dataset and row['method'] == method:
                        ax.scatter(splits.index(row['split']), float(row[metric]), color=COLORS[method], s=17, alpha=.7)
            ax.set_xticks(range(3), ['FIT', 'TUNE', '반복 VAL' if dataset == 'robin' else '노출 EVAL'])
            ax.set_ylabel('정규화 ' + ('MSE' if j == 0 else 'MAE'))
            ax.set_title(DISPLAY[dataset])
            ax.grid(axis='y', alpha=.2)
    axes[0, 0].legend(loc='upper left', fontsize=9)
    fig.suptitle('D0: 기존 동일 E/D에서 native 예측기로 바꿔도 해결되지 않음')
    finish_figure(fig, '01_D0_split_metrics', folder, plt, 'MAIN/native: seed92601·92602 metric 평균 + 실제 점 · RAW 단일 공유 기준 · 독립 holdout 아님')


def figure_channels(tables, folder, plt):
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    rows = tables['d0_channels_mean']
    for index, dataset in enumerate(DATASETS):
        split = 'val' if dataset == 'robin' else 'eval'
        base = sorted([r for r in rows if r['dataset'] == dataset and r['split'] == split and r['method'] == 'RAW'], key=lambda r: int(r['channel_index']))
        gaps = []
        for row in base:
            main = next(r for r in rows if r['dataset'] == dataset and r['split'] == split and r['method'] == 'MAIN' and r['channel_index'] == row['channel_index'])
            gaps.append(main['normalized_mse'] - row['normalized_mse'])
        ax = axes[0, index]
        ax.barh(range(len(base)), gaps, color=['#D55E00' if value > 0 else '#0072B2' for value in gaps])
        ax.set_yticks(range(len(base)), [r['channel'].replace('Robin_office_', '') for r in base])
        ax.invert_yaxis()
        ax.axvline(0, color='#333333', linewidth=.7)
        ax.set_xlabel('MAIN - RAW 채널 정규화 MSE (양수: 악화)')
        ax.set_title(DISPLAY[dataset] + ' · 모든 채널')
    zelma = [r for r in rows if r['dataset'] == 'robin' and r['channel'].endswith('Zelma')]
    for index, field in enumerate(('mean', 'std')):
        ax = axes[1, index]
        for method in ('RAW', 'MAIN', 'MAIN_NATIVE'):
            values = [next((r['prediction_normalized_' + field] for r in zelma if r['split'] == split and r['method'] == method), np.nan) for split in ('fit', 'tune', 'val')]
            ax.plot(range(3), values, marker='o', color=COLORS[method], label=method + ' 예측')
        target = [next((r['target_normalized_' + field] for r in zelma if r['split'] == split and r['method'] == 'RAW'), np.nan) for split in ('fit', 'tune', 'val')]
        ax.plot(range(3), target, 'k--', marker='s', label='관측 target')
        ax.set_xticks(range(3), ['FIT', 'TUNE', '반복 VAL'])
        ax.set_title('Robin Zelma: 정규화 ' + ('평균' if field == 'mean' else '표준편차'))
        ax.set_ylabel('정규화 값')
        ax.grid(axis='y', alpha=.2)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle('D0: 오차 집중 위치와 level/variance shift · 원인과는 구분')
    finish_figure(fig, '02_D0_channel_gap_Zelma', folder, plt, '채널 gap 평균은 seed metric 평균 · Zelma는 기존 VAL에서 관측한 집중 위치이며 새 채널 선택 기준이 아님')


def figure_d1(tables, folder, plt):
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    order = ('identity_C', 'through_origin_quarter', 'through_origin_half', 'through_origin_full_C_rotated',
             'centered_quarter', 'centered_half', 'centered_full_C_rotated')
    labels = ['Identity C', 'Origin ¼', 'Origin ½', 'Origin C 회전', 'Centered ¼', 'Centered ½', 'Centered C 회전']
    for i, dataset in enumerate(DATASETS):
        split = 'val' if dataset == 'robin' else 'eval'
        selected = [r for r in tables['d1'] if r['dataset'] == dataset and r['split'] == split]
        for j, metric in enumerate(('mse', 'mae')):
            ax = axes[i, j]
            for method, label in (('projected_raw', 'RAW forecast 투영 (진단)'), ('projected_past', 'F(투영한 과거) 실제 추론')):
                values = [next((float(r[metric]) for r in selected if r['candidate'] == name and r['method'] == method), np.nan) for name in order]
                ax.plot(range(7), values, marker='o', color=COLORS[method], label=label)
            raw = next((float(r[metric]) for r in selected if r['method'] == 'raw'), np.nan)
            ax.axhline(raw, color='#333333', linestyle='--', label='RAW_BOLT')
            for position in (0, 3, 6):
                ax.axvspan(position - .25, position + .25, color='#999999', alpha=.12)
            ax.set_xticks(range(7), labels, rotation=25, ha='right')
            ax.set_title(DISPLAY[dataset])
            ax.set_ylabel('정규화 ' + metric.upper())
            ax.grid(axis='y', alpha=.2)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='upper center', ncol=3, bbox_to_anchor=(.5, .94), fontsize=9)
    fig.suptitle('D1: 출력의 저차원 근사와 입력 투영 뒤 TSFM 예측은 다른 문제')
    finish_figure(fig, '03_D1_projection_order', folder, plt, 'FIT RAW 변화로만 basis 고정 · 7개 후보 전부 표시 · 회색 K=C identity/회전은 압축 아님 · projected RAW는 full-C 비용 필요', top=.90)


def figure_d2(tables, folder, plt):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    methods = ('RAW', 'ORIGINAL_D', 'REFIT_Y', 'REFIT_RAW', 'REFIT_MIXED')
    colors = ('#333333', '#777777', '#0072B2', '#009E73', '#CC79A7')
    labels = ('RAW_BOLT', '원 D', 'D: Y', 'D: RAW', 'D: .5 혼합')
    for i, dataset in enumerate(DATASETS):
        split = 'val' if dataset == 'robin' else 'eval'
        for j, metric in enumerate(('mse', 'mae')):
            ax = axes[i, j]
            for x, method in enumerate(methods):
                values = [float(r[metric]) for r in tables['d2_selected_macro'] if r['dataset'] == dataset and r['split'] == split and r['method'] == method]
                if method == 'RAW' and values:
                    values = values[:1]
                mean = float(np.mean(values)) if values else np.nan
                ax.bar(x, mean, color=colors[x], alpha=.75, width=.65)
                ax.scatter([x + offset for offset in np.linspace(-.08, .08, len(values))], values, color='#111111', s=20, zorder=3)
                if values:
                    ax.annotate(f'{mean:.3f}', (x, mean), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=9)
            ax.set_xticks(range(5), labels, rotation=15)
            ax.set_ylabel('정규화 ' + metric.upper())
            ax.set_title(DISPLAY[dataset] + ' · 동결 E/F/hidden')
            ax.margins(y=.15)
            ax.grid(axis='y', alpha=.2)
    fig.suptitle('D2: 같은 absolute latent에서 출력 가중치만 FIT 재적합')
    finish_figure(fig, '04_D2_frozen_decoder', folder, plt, 'MAIN/refit: 두 seed metric 평균 + 점 · RAW 공유 reference 1개 · α는 TUNE 실제 Y MSE로 선택 · 전체 α/원단위는 CSV')


def figure_latency(tables, folder, plt):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    order = (('RAW_BOLT', 'old_loop'), ('MAIN', 'old_loop'), ('RAW_BOLT', 'vectorized'), ('MAIN', 'vectorized'))
    labels = ('RAW 원 fill', 'MAIN 원 fill', 'RAW 벡터 fill', 'MAIN 벡터 fill')
    for i, dataset in enumerate(DATASETS):
        for j, batch in enumerate((1, 8)):
            ax = axes[i, j]
            for x, (method, kind) in enumerate(order):
                values = [float(r['median_ms_per_window']) for r in tables['d3'] if r['dataset'] == dataset and r['batch'] == batch and r['method'] == method and r['preprocessing'] == kind]
                value = float(np.mean(values)) if values else np.nan
                ax.bar(x, value, color='#333333' if method == 'RAW_BOLT' else '#0072B2', alpha=.85 if kind == 'old_loop' else .45)
                ax.scatter([x + offset for offset in np.linspace(-.08, .08, len(values))], values, color='#D55E00', s=20, zorder=3)
                if values:
                    ax.annotate(f'{value:.2f}', (x, value), xytext=(0, 4), textcoords='offset points', ha='center', fontsize=9)
            ax.set_xticks(range(4), labels, rotation=15)
            ax.set_ylabel('전체 경로 ms / 창')
            ax.set_title(DISPLAY[dataset] + f' · B{batch}')
            ax.margins(y=.15)
            ax.grid(axis='y', alpha=.2)
    fig.suptitle('D3: 동일 causal fill 최적화를 RAW와 MAIN 양쪽에 적용')
    finish_figure(fig, '05_D3_whole_latency', folder, plt, '최종 동일 protocol warm2/repeat7 median · 두 seed 프로세스 평균 + 점 · cold/warm·cadence 불확실성 있음 · load/reference/save 제외')


def figure_stages(tables, folder, plt):
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    order = ('fill', 'normalization', 'H2D', 'E', 'F', 'D', 'anchor_inverse', 'D2H')
    stage_colors = ('#E69F00', '#56B4E9', '#999999', '#009E73', '#0072B2', '#CC79A7', '#D55E00', '#333333')
    routes = (('RAW_BOLT', 'old_loop'), ('MAIN', 'old_loop'), ('RAW_BOLT', 'vectorized'), ('MAIN', 'vectorized'))
    for i, dataset in enumerate(DATASETS):
        for j, batch in enumerate((1, 8)):
            ax, left = axes[i, j], np.zeros(4)
            for stage, color in zip(order, stage_colors):
                values = []
                for method, kind in routes:
                    selected = [float(r['median_ms_with_synchronization']) / batch for r in tables['d3_stages']
                                if r['dataset'] == dataset and r['batch'] == batch and r['method'] == method and r['preprocessing'] == kind and r['stage'] == stage]
                    values.append(float(np.mean(selected)) if selected else np.nan)
                ax.barh(range(4), values, left=left, color=color, label=stage)
                left += np.asarray(values)
            ax.set_yticks(range(4), ['RAW 원 fill', 'MAIN 원 fill', 'RAW 벡터 fill', 'MAIN 벡터 fill'])
            ax.invert_yaxis()
            ax.set_xlabel('각 단계 별도 동기화 median 합 / 창 (ms)')
            ax.set_title(DISPLAY[dataset] + f' · B{batch}')
            ax.grid(axis='x', alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=8, bbox_to_anchor=(.5, .049), fontsize=9)
    fig.suptitle('D3: CPU fill와 E/F/D 비용 위치 · 합은 실제 전체 latency와 다름')
    finish_figure(fig, '06_D3_synchronized_stages', folder, plt, '각 stage synchronize overhead 포함 · D는 D(z_F),D(z_last) · RAW E/D는 host identity overhead · 두 seed metric 평균', bottom=.105)


def number(value):
    return f'{float(value):.9f}'


def report_markdown(tables, coverage, audits, figures, resource_rows):
    lines = ['# 작은 K 실패의 원인 분리: Robin·Beijing 후속 결과', '',
        '[확인] 이 보고서는 기존 조건의 저장 출력과 새 실제 추론·decoder 적합·비용 계측을 모은 개발 증거다. '
        'Robin VAL은 반복 사용했고 Beijing EVAL은 이미 노출됐다. 보호 TEST와 새 독립 holdout은 사용하지 않았다.', '',
        f"선언한 생산 조건: {coverage['complete_conditions']}/{coverage['expected_conditions']} COMPLETE."]
    lines += ['', '## 먼저 답할 질문', '',
        '**native로 바꾸면 해결되는가?** 같은 MAIN checkpoint의 E/D가 실제 CPU torch.equal로 일치하는 native control도 확인했다. '
        '아래 원수치는 Bolt용 E/D를 고정한 native 비교이며 native용 E/D를 새로 공동 최적화한 최선 성능은 아니다.', '',
        '```text', '자료                    RAW MSE/MAE               MAIN MSE/MAE              MAIN-native MSE/MAE']
    for dataset in DATASETS:
        split = 'val' if dataset == 'robin' else 'eval'
        values = []
        for method in ('RAW', 'MAIN', 'MAIN_NATIVE'):
            row = next((r for r in tables['d0_summary_mean'] if r['dataset'] == dataset and r['split'] == split and r['method'] == method), None)
            values.append('미완료' if row is None else f"{number(row['normalized_mse'])}/{number(row['normalized_mae'])}")
        lines.append(f"{dataset:<23} {'   '.join(values)}")
    lines += ['```', '', '**FIT의 96% 설명률은 평가 예측을 보장하는가?** 아니다. 아래 값은 FIT RAW forecast−현재 상태로 고정한 centered basis의 분산 설명률이다. '
        '실제 target의 예측 정보 설명률이나 VAL 설명률이 아니다. projected RAW 자체도 full-C RAW 추론을 먼저 필요로 하는 진단이다.', '', '```text']
    for dataset in DATASETS:
        values = [r for r in tables['d0_rank'] if r['dataset'] == dataset and r['projection_mode'] == 'mean_centered']
        for row in values:
            lines.append(f"{dataset} {row['split']}: K{row['k']} FIT retained={float(row['retained_fit_energy_fraction']):.9f}; projected RAW target MSE={number(row['normalized_mse'])}, MAE={number(row['normalized_mae'])}; RAW MSE={number(row['raw_actual_target_normalized_mse'])}")
    lines += ['```', '', '**예측기와 좌표 변환 중 어디가 남는가?** D1은 7개 고정 basis를 모두 보여준다. '
        'K=C 회전은 정보 복원이 거의 정확해도 channel-wise TSFM의 입력 좌표를 바꾸며 결과가 달라질 수 있다. '
        '회전을 압축 방법이나 예측에 유리한 독립 검증으로 부르지 않는다. MSE와 MAE의 변화가 반대면 함께 보고한다.', '', '```text']
    for dataset in DATASETS:
        split = 'val' if dataset == 'robin' else 'eval'
        for row in tables['d1']:
            if row['dataset'] == dataset and row['split'] == split and row['method'] == 'projected_past' and ('full_C_rotated' in row['candidate'] or row['candidate'] == 'identity_C'):
                lines.append(f"{dataset} {row['candidate']}: MSE={number(row['mse'])}, MAE={number(row['mae'])}; history reconstruction MSE={float(row['past_reconstruction_mse']):.9g}")
    lines += ['```', '', '**출력 D만 새로 맞추면 해결되는가?** D2는 원 E/F와 decoder GELU hidden을 고정하고 output weights만 재적합했다. '
        'Y, 과거 기반 RAW forecast, .5 혼합을 같은 FIT mask로 적합하고 각각 TUNE 실제 Y의 정규화 MSE로 α를 선택했다. '
        '전체 α·원 D·선택 출력·원단위·채널별 값을 모두 보존한다. 이 비교는 모든 nonlinear decoder의 보편적 하한이나 최선 성능을 뜻하지 않는다.', '',
        '```text', '자료                    방법             MSE            MAE (2 seed metric 평균)']
    for row in tables['d2_selected_mean']:
        if row['split'] == ('val' if row['dataset'] == 'robin' else 'eval'):
            lines.append(f"{row['dataset']:<23} {row['method']:<16} {number(row['mse'])}    {number(row['mae'])}")
    lines += ['```', '', '**batch1은 어디에서 느린가, 공통 전처리 개선은 도움이 되는가?** D3는 동일 raw physical past와 mask를 사용한다. '
        '벡터 causal fill은 stored history와 byte-exact이며 RAW와 MAIN 양쪽에 동일하게 적용한다. '
        '예측 정확도를 바꾸는 개선으로 세지 않는다. 아래 값은 seed별 프로세스의 warm2/repeat7 median 평균이며 독립 장비 반복이 아니다.', '',
        '```text', '자료                    B    방법       원 fill ms/창    벡터 fill ms/창    상대 변화']
    for dataset in DATASETS:
        for batch in (1, 8):
            for method in ('RAW_BOLT', 'MAIN'):
                values = {}
                for kind in ('old_loop', 'vectorized'):
                    selected = [float(r['median_ms_per_window']) for r in tables['d3'] if r['dataset'] == dataset and r['batch'] == batch and r['method'] == method and r['preprocessing'] == kind]
                    if selected:
                        values[kind] = float(np.mean(selected))
                if len(values) == 2:
                    before, after = values['old_loop'], values['vectorized']
                    lines.append(f"{dataset:<23} {batch:<4} {method:<10} {before:.6f}          {after:.6f}            {100*(after/before-1):+.2f}%")
    bulk_fails = sum(int(row['original_bulk_physical_failed_repeats']) for row in tables['d3'])
    lines += ['```', '', f'[확인] 원 bulk reference의 physical gate 실패 반복은 {bulk_fails}개이며 그대로 별도 보존했다. '
        'matching-batch의 실제 원 controls.deployment와 normalized engine.predict를 reference로 사용한 검사와 혼동하지 않는다. '
        '분해 계측은 각 단계 synchronize를 더하므로 단계 합을 실제 전체 경로 latency로 해석하지 않는다.', '',
        '**측정 protocol을 왜 다시 확인했는가?** 최초 D3 v1 결과와 resource guard 진단을 보존한다. '
        '각 trial/stage에서 수행한 전체 storage/path scan은 timer 밖이어도 추론 사이 cadence와 GPU idle 상태에 영향을 줄 수 있다. '
        '후속 burst v2는 각 route 전후의 전체 CPU/storage 확인과 매 trial/stage의 GPU·RSS·campaign 확인을 분리한다. '
        '원 numerical pipeline·비교 bound를 바꾼 결과로 읽지 않는다. 최초 측정과 protocol 진단은 별도 표와 원문 JSON에 남긴다.', '',
        '[CPU guard의 실제 측정값](tables/d3_guard_measurements.csv)은 cadence 문제가 가능한 위치를 보여준다. '
        '이 CPU-only 측정 자체가 GPU cooldown이나 전체 latency 변동의 원인을 입증한 것은 아니다.', '',
        '현재 warmup2/repeat7만으로 cold start와 안정된 GPU warm 상태를 엄밀히 분리했다고 할 수 없다. '
        'timing 순서·clock/power·짧은 반복의 변동 때문에 특정 수치의 감소를 장비 조건 전반의 무조건적인 speedup으로 주장하지 않는다. '
        '주 비교는 최종 동일 protocol에서 RAW/MAIN과 양쪽의 공통 fill을 비교한 개발 계측이다.', '',
        '**무엇을 해결했다고 말할 수 있는가?** 저장값/실제 경로의 조건을 맞추고 native·projection 순서·동결 D·공통 전처리의 기여를 구분하는 근거가 생겼다. '
        '단순 reconstruction 안정성, latent 예측 적합성, decoder 적합을 하나의 실패 원인으로 단정할 근거는 아직 없다. '
        '현재 결과만으로 대조학습·동적 그래프·Chronos-Bolt의 필요성이나 모든 작은 K 방법의 불가능성을 확정하지 않는다.', '',
        '## 비교 계약과 불확실성', '',
        '두 자료 모두 L512/H48, 저장 FIT FP32 mean/std를 그대로 재사용한다. Robin C17/K4와 Beijing C11/K2, '
        'seed92601·92602, 원 MAIN batch4 학습과 batch8 저장 추론 조건이다. '
        '현재 상태 C개를 우회 보존하므로 strict K-only 압축이 아니다. D1은 학습 seed 비교가 없는 FIT 고정 선형 진단이다.', '',
        '이전 Bi-Hash의 새 3-seed 학습 recipe·초기화·통계 조건과 이 old 2-seed 결과를 섞어 '
        'Bi-Hash나 후속 방법의 성능 개선으로 읽지 않는다. 채널 단위가 다른 physical MSE/MAE는 데이터셋끼리 합치지 않는다. '
        '평균은 seed별 metric의 산술 평균이며 prediction ensemble이 아니다. 겹치는 창과 작은 seed 수에 독립 표본 CI/유의성을 붙이지 않는다.', '',
        'Zelma의 gap/level/variance 차이는 관측된 오차 집중 위치다. VAL을 보고 선정한 채널, rank, α를 새 독립 선택으로 부르지 않는다. '
        'D2 α 선택은 TUNE이며 보고서가 EVAL 값으로 최적 목적을 다시 고르지 않는다.', '',
        '## 실제 시각자료', '']
    for title, name in figures:
        lines += [f'![{title}](figures/{name}.png)', '', f'[벡터 SVG](figures/{name}.svg)', '']
    lines += ['## 원수치와 전체 후보', '',
        '- [D0 전체 split/seed](tables/d0_summary.csv), [모든 채널](tables/d0_channels.csv), [tail 기여](tables/d0_tails.csv)',
        '- [D0 FIT rank 설명률과 투영](tables/d0_rank.csv), [D1 7개 후보·모든 split](tables/d1.csv)',
        '- [D2 전체 α/채널/정규화·원단위](tables/d2.csv), [선택 α](tables/d2_selection.csv), [선택 metric 평균](tables/d2_selected_mean.csv)',
        f"- [D3 실제 {len(tables['d3'])}개 조건](tables/d3.csv), [동기화 단계별 비용](tables/d3_stages.csv)",
        '- [최초 D3 v1 비용 보존](tables/d3_preserved.csv), [최초 v1 단계 비용](tables/d3_preserved_stages.csv)',
        '- [생산 입력 path·SHA](provenance/input_manifest.csv), [실제 공유 자원 영수증](tables/resources.csv)', '',
        '큰 NPZ·CP는 ignored cache의 literal path+SHA로 provenance CSV에 남긴다. 이 문서는 GitHub에서 존재하지 않는 cache 파일을 hyperlink하지 않는다. '
        '각 copied CSV와 작은 JSON/audit는 이 보고서 폴더의 상대경로로 연결한다.', '', '## 검산 상태', '', '```text']
    for row in audits:
        lines.append(f"{row['name']}: {row['status']}; checks={row.get('checks', 'unknown')}; failed={row.get('failed_count', 'unknown')}; scope={row.get('claim_scope', '')}; nested={row.get('nested_dataset_statuses', {})}")
    lines += ['```', '', '검사 통과는 해당 arithmetic/lineage/원 출력 재현 범위의 근거다. 독립적인 성능 개선이나 일반화의 증명으로 바꾸어 말하지 않는다.', '',
        '## 비용과 실행 상태', '',
        f"중복 제거한 입력 생산 자원 영수증은 {len(resource_rows)}개다. GPU process wall={sum(float(r['wall_seconds']) for r in resource_rows if r['gpu']):.6f}s; CPU process wall={sum(float(r['wall_seconds']) for r in resource_rows if not r['gpu']):.6f}s."]
    lines += ['이 합은 이 보고서에 포함한 D0/D1/D2/D3 및 보존한 최초 protocol producer 영수증만의 합이다. 이전 Bi 캠페인 전체 비용이나 독립 검산·보고서 비용을 포함한다고 하지 않는다. '
        '현재 보고서 생성은 FollowupLimit CPU 영수증에 별도로 기록하며 미계측 startup/write overhead는 unknown이다.', '',
        f"미완료 조건: {json.dumps(coverage['missing_conditions'] + coverage['noncomplete_conditions'], ensure_ascii=False)}", '',
        '실패·미완료 경로는 원 terminal RESULT의 상태·error·partial coverage와 함께 provenance에 보존한다. '
        '재개는 같은 완료 파일을 덮는 방식이 아니라 남은 경로에 새 tag를 사용하고 기존 공유 예산과 자산 SHA를 다시 확인하는 방식이다.', '']
    return '\n'.join(lines)


def local_links(markdown, output):
    pattern = re.compile(r'(!?)\[([^\]]+)\]\(([^)]+)\)')
    checked = []

    def replacement(match):
        prefix, label, target = match.groups()
        if '://' in target or target.startswith('#'):
            return match.group(0)
        path = (Path(output) / target).resolve()
        if not path.is_relative_to(Path(output).resolve()):
            raise ValueError('REPORT_LINK_OUTSIDE_ARTIFACT')
        if not path.is_file():
            if prefix:
                raise ValueError('REPORT_IMAGE_MISSING: ' + target)
            return label + ' (미생성)'
        checked.append(target)
        return match.group(0)

    return pattern.sub(replacement, markdown), checked


def describe():
    return {'status': 'PREPARATION_ONLY', 'actual_report_generated': False, 'scientific_reads_on_import': False,
            'planned_producers': {'D0': 2, 'D1': 2, 'D2': 4, 'D3': 4}, 'figures': 6,
            'protected_TEST': False, 'new_model_inference': False, 'GPU_execution': False,
            'generation_opt_in': '--generate with terminal producer paths and independent audit receipts',
            'output': 'research/smallk_failure_followup_20261014/report_<fresh-tag> (exclusive)'}


def generate(args):
    if not args.generate:
        raise ValueError('ACTUAL_REPORT_GENERATION_OPT_IN_REQUIRED')
    if not args.tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.tag):
        raise ValueError('SAFE_FRESH_REPORT_TAG_REQUIRED')
    if not args.preserved_d3 or not args.measurement_diagnostic:
        raise ValueError('ORIGINAL_D3_AND_GUARD_DIAGNOSTIC_REQUIRED')
    output = HERE / ('report_' + args.tag)
    output.mkdir(exist_ok=False)
    result = {'status': 'RUNNING', 'actual_report_generated': False, 'new_model_inference': False,
              'GPU_execution': False, 'protected_TEST_accessed': False, 'tag': args.tag}
    limit = None
    started = time.perf_counter()
    try:
        spec = importlib.util.spec_from_file_location('followup_report_runtime', HERE / 'followup_runtime.py')
        runtime = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = runtime
        spec.loader.exec_module(runtime)
        limit = runtime.FollowupLimit('report_' + args.tag, gpu=False)
        guard_spec = importlib.util.spec_from_file_location('followup_report_guard', HERE / 'prepare_inputs.py')
        guard = importlib.util.module_from_spec(guard_spec)
        sys.modules[guard_spec.name] = guard
        guard_spec.loader.exec_module(guard)
        guard.install_protected_test_guard()
        items = [load_producer(path, block) for block in ('D0', 'D1', 'D2', 'D3') for path in getattr(args, block.lower())]
        coverage = assess_coverage(items)
        tables, copies = collect_tables(items)
        preserved = [load_producer(path, 'D3') for path in args.preserved_d3]
        old_tables, _ = collect_tables(preserved)
        tables['d3_preserved'], tables['d3_preserved_stages'] = old_tables['d3'], old_tables['d3_stages']
        unique_resource_paths = {row['path'] for row in tables['resources']}
        tables['resources'].extend(row for row in old_tables['resources'] if row['path'] not in unique_resource_paths)
        diagnostic_records = []
        tables['d3_guard_measurements'] = []
        for path in args.measurement_diagnostic:
            diagnostic, record = checked_json(path)
            if diagnostic.get('status') == 'RUNNING':
                raise ValueError('TERMINAL_MEASUREMENT_DIAGNOSTIC_REQUIRED')
            diagnostic_records.append(record)
            for name, values in diagnostic.get('measurements', {}).items():
                tables['d3_guard_measurements'].append({'diagnostic': Path(path).name, 'guard_component': name,
                    'median_ms': float(values['median_seconds']) * 1000,
                    'samples_seconds': json.dumps(values['seconds']),
                    'scope': diagnostic.get('scope', ''), 'source_sha256': record['sha256']})
        audit_rows, audit_records = [], []
        for path in args.audit:
            audit, record = checked_json(path)
            if audit.get('status') == 'RUNNING':
                raise ValueError('TERMINAL_AUDIT_REQUIRED')
            audit_rows.append(summarize_audit(audit, Path(path).name))
            audit_records.append(record)
        result.update(coverage=coverage, audits=audit_rows,
                      producer_statuses=[{'block': item['block'], 'dataset': item['producer']['dataset'],
                          'seed': item['producer'].get('seed'), 'status': item['producer']['status'], 'error': item['producer'].get('error')} for item in items])
        for directory in ('tables', 'figures', 'provenance', 'provenance/producers', 'provenance/audits'):
            (output / directory).mkdir(exist_ok=False)
        inputs = []
        for source in (Path(__file__), HERE / 'followup_runtime.py', HERE / 'prepare_inputs.py'):
            record = {'role': 'report_executed_source', 'path': str(source.resolve()),
                      'sha256': sha(source), 'file_hash_rechecked': True}
            shutil.copyfile(source, output / 'provenance' / ('source_' + source.name))
            inputs.append(record)
        for index, item in enumerate(items):
            p = item['producer']
            label = f"{item['block']}_{p['dataset']}_s{p.get('seed', 'none')}"
            shutil.copyfile(item['record']['path'], output / 'provenance/producers' / (label + '.json'))
            inputs.append({'role': 'producer', 'block': item['block'], **item['record']})
            if 'child_record' in item:
                shutil.copyfile(item['child_record']['path'], output / 'provenance/producers' / (label + '_scientific.json'))
                inputs.append({'role': 'scientific_result', 'block': item['block'], **item['child_record']})
            for source in p.get('input_provenance', p.get('provenance', {})).get('input_artifacts', []):
                inputs.append({'role': 'declared_scientific_NPZ_not_rehashed_by_report', 'block': item['block'],
                               'path': source['path'], 'sha256': source['sha256'], 'file_hash_rechecked': False})
        for index, item in enumerate(preserved):
            destination = output / 'provenance/producers' / f'D3_original_preserved_{index + 1:02d}.json'
            shutil.copyfile(item['record']['path'], destination)
            inputs.append({'role': 'D3_original_preserved_producer', **item['record']})
        for index, record in enumerate(diagnostic_records):
            shutil.copyfile(record['path'], output / 'provenance' / (f'measurement_diagnostic_{index + 1:02d}_' + Path(record['path']).name))
            inputs.append({'role': 'D3_measurement_guard_diagnostic', **record})
        for index, record in enumerate(audit_records):
            name = f'{index + 1:02d}_' + Path(record['path']).name
            shutil.copyfile(record['path'], output / 'provenance/audits' / name)
            inputs.append({'role': 'independent_audit', **record})
        for copied in copies:
            shutil.copyfile(copied['source'], output / 'tables' / copied['destination'])
            if sha(output / 'tables' / copied['destination']) != copied['sha256']:
                raise ValueError('COPIED_CSV_HASH_MISMATCH')
            inputs.append({'role': 'original_CSV', 'path': copied['source'], 'sha256': copied['sha256'], 'file_hash_rechecked': True})
        for name, rows in tables.items():
            if rows:
                save_csv(output / 'tables' / (name + '.csv'), rows)
        save_csv(output / 'provenance/input_manifest.csv', inputs)
        write(output / 'provenance/COVERAGE.json', coverage)
        write(output / 'provenance/AUDIT_SUMMARY.json', audit_rows)
        limit.check()
        plt, font = configure_plotting()
        figures = [('D0 split MSE/MAE', '01_D0_split_metrics'), ('D0 모든 채널과 Zelma shift', '02_D0_channel_gap_Zelma'),
                   ('D1 모든 basis와 예측 순서', '03_D1_projection_order'), ('D2 목적별 동결 decoder', '04_D2_frozen_decoder'),
                   ('D3 실제 전체 latency', '05_D3_whole_latency'), ('D3 별도 동기화 단계', '06_D3_synchronized_stages')]
        for function in (figure_d0, figure_channels, figure_d1, figure_d2, figure_latency, figure_stages):
            function(tables, output / 'figures', plt)
            limit.check()
        markdown = report_markdown(tables, coverage, audit_rows, figures, tables['resources'])
        markdown += '\n검산 영수증 원문:\n\n' + '\n'.join(
            f"- [{Path(record['path']).name}](provenance/audits/{index + 1:02d}_{Path(record['path']).name})"
            for index, record in enumerate(audit_records)) + '\n'
        markdown += '\n측정 protocol 진단 원문:\n\n' + '\n'.join(
            f"- [{Path(record['path']).name}](provenance/measurement_diagnostic_{index + 1:02d}_{Path(record['path']).name})"
            for index, record in enumerate(diagnostic_records)) + '\n'
        markdown += '\n최초 D3 producer 보존:\n\n' + '\n'.join(
            f"- [원 v1 #{index + 1}](provenance/producers/D3_original_preserved_{index + 1:02d}.json)"
            for index, item in enumerate(preserved)) + '\n'
        markdown, link_paths = local_links(markdown, output)
        with (output / 'REPORT_KO.md').open('x', encoding='utf-8') as handle:
            handle.write(markdown)
        result.update(status='COMPLETE', actual_report_generated=True, font=font,
            input_count=len(inputs), figures_count=len(figures), tables_rows={key: len(rows) for key, rows in tables.items()},
            relative_links_checked=link_paths,
            scientific_scope_complete=coverage['all_declared_blocks_complete'], independent_holdout=False,
            report_body_wall_seconds=time.perf_counter() - started,
            audit_all_pass=bool(audit_rows) and all(row['all_declared_scopes_passed'] for row in audit_rows),
            improvement_established=False, source_sha256=sha(__file__))
        for record in inputs:
            if record.get('file_hash_rechecked') and sha(record['path']) != record['sha256']:
                raise ValueError('REPORT_INPUT_CHANGED_DURING_GENERATION')
        limit.check()
    except Exception as error:
        result.update(status='FAILED_PRESERVED', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        if limit is not None:
            receipt = limit.close(result['status'], result.get('error'))
            result['shared_resource_receipt'] = receipt
            if receipt.get('terminal_status', result['status']) != 'COMPLETE':
                result.update(status=receipt.get('terminal_status', result['status']), error=receipt.get('terminal_error', result.get('error')))
        write(output / 'RESULT.json', result)
        manifest = {'generator_result_status': result['status'],
                    'artifacts': [{'path': str(path.relative_to(output)).replace('\\', '/'), 'sha256': sha(path), 'bytes': path.stat().st_size}
                                  for path in sorted(output.rglob('*')) if path.is_file()]}
        write(output / 'MANIFEST.json', manifest)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--tag')
    for block in ('d0', 'd1', 'd2', 'd3'):
        parser.add_argument('--' + block, nargs='+', type=Path)
    parser.add_argument('--audit', nargs='+', type=Path)
    parser.add_argument('--preserved-d3', nargs='+', type=Path)
    parser.add_argument('--measurement-diagnostic', nargs='+', type=Path)
    args = parser.parse_args()
    if args.describe or not args.generate:
        print(json.dumps(describe(), ensure_ascii=False, indent=2))
        return
    if not args.tag or any(getattr(args, key) is None for key in ('d0', 'd1', 'd2', 'd3', 'audit', 'preserved_d3', 'measurement_diagnostic')):
        parser.error('--generate requires --tag --d0 --d1 --d2 --d3 --audit --preserved-d3 --measurement-diagnostic actual terminal paths.')
    result = generate(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result['status'] != 'COMPLETE':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
