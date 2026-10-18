"""Deterministic public-table reporting; no model, private array or data access."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


INPUTS = ('RUNS.csv', 'COST.csv', 'TRAINING_HISTORY.csv', 'COMPARISONS.csv',
          'ERROR_SUFFICIENT_STATS.csv', 'PAIR_ORIGIN_DIFFS.csv',
          'CAMPAIGN_CONTRACT.json', 'DECISIONS.json')
COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#000000')
LIMITATION = ('V는 반복 개발 평가로 이미 사용한 자료이며 독립 테스트 결과가 아니다. '
              '새 선택에 M_cal/T를 사용하지 않았으며, 기존 T 노출과 '
              'pretraining overlap UNKNOWN의 범위는 원 계약을 따른다. '
              '두 seed·고정 K의 결과를 모든 데이터·압축률의 일반화로 확대하지 않는다.')


def parse_bool(value):
    if value is None or value == '':
        return None
    if type(value) is bool:
        return value
    if str(value).strip().lower() in ('true', '1', 'on'):
        return True
    if str(value).strip().lower() in ('false', '0', 'off'):
        return False
    raise ValueError('Expected explicit boolean: ' + str(value))


def number(value):
    if value is None or value == '' or str(value).upper() in ('UNKNOWN', 'UNDEFINED', 'NOT_MEASURED'):
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Nonfinite public numeric field')
    return result


def integer(value):
    result = number(value)
    if result is not None and result != int(result):
        raise ValueError('Expected an integer identity field')
    return None if result is None else int(result)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def _write_csv(path, rows):
    fields = sorted(set().union(*(row.keys() for row in rows))) if rows else ['status']
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
                              if isinstance(value, (dict, list)) else value for key, value in row.items()})


def _run(row):
    value = dict(row)
    if 'observed' not in value and 'n_targets' in value:
        value['observed'] = value['n_targets']
    for key in ('MSE', 'MAE', 'selected_epoch', 'search_seconds', 'refit_seconds'):
        if key in value:
            value[key] = number(value[key])
    for key in ('seed', 'cap', 'C', 'K', 'L', 'H', 'observed', 'n_targets', 'search_updates', 'refit_updates'):
        if key in value:
            value[key] = integer(value[key])
    for key in ('revin', 'final'):
        value[key] = parse_bool(value.get(key))
    return value


def _index(runs):
    result = {}
    for row in map(_run, runs):
        key = (row['run_id'], row.get('cap'))
        if key in result:
            raise ValueError('Duplicate run/cap identity: ' + str(key))
        result[key] = row
    return result


def _lookup(index, identifier, cap):
    if (identifier, cap) in index:
        return index[identifier, cap]
    if cap is not None:
        raise ValueError('Referenced run does not have the declared cap: ' + identifier + f' cap{cap}')
    matches = [row for (name, _), row in index.items() if name == identifier]
    if len(matches) != 1:
        raise ValueError('Missing or ambiguous referenced run/cap: ' + identifier)
    return matches[0]


def accuracy_cost_rows(runs, costs):
    final = [row for row in map(_run, runs) if row['final'] is True]
    output = []
    for run in final:
        attached = []
        for raw_cost in costs:
            if raw_cost.get('accuracy_run_id') != run['run_id']:
                continue
            if raw_cost.get('cap') not in (None, '') and integer(raw_cost['cap']) != run.get('cap'):
                continue
            if raw_cost.get('dataset') not in (None, '', run['dataset']):
                raise ValueError('Cost accuracy dataset mismatch')
            cost = dict(raw_cost)
            batch = integer(cost['batch_size'])
            if batch not in (1, 8):
                raise ValueError('Final cost batch must be 1 or 8')
            valid = parse_bool(cost.get('cost_valid')) is True
            if cost.get('status') not in (None, '', 'COMPLETE', 'REUSED_VERIFIED'):
                valid = False
            latency = number(cost.get('median_ms_per_origin')) if valid else None
            if valid and (latency is None or latency <= 0):
                raise ValueError('Valid cost needs positive latency')
            row = dict(run, device=cost['device'], batch_size=batch, cost_id=cost.get('cost_id', ''),
                median_ms_per_origin=latency, cost_valid=valid,
                cost_status=cost.get('status', 'COMPLETE' if valid else 'NOT_MEASURED'),
                cost_source_file=cost.get('source_file', ''), cost_source_sha256=cost.get('source_sha256', ''),
                cost_accuracy_run_id=cost['accuracy_run_id'])
            for key, value in cost.items():
                if key not in ('MSE', 'MAE', 'seed', 'method', 'dataset', 'revin', 'cap', 'run_id',
                               'final', 'cost_valid', 'median_ms_per_origin', 'batch_size', 'status'):
                    row.setdefault(key, value)
            row['cost_protocol_scope'] = 'Exact accuracy_run_id; device/batch remain separate'
            attached.append(row)
        output.extend(attached)
        required_device = 'cpu' if run.get('family') == 'RAW_RIDGE' else 'cuda'
        for batch in (1, 8):
            if not any(row['device'] == required_device and row['batch_size'] == batch for row in attached):
                output.append(dict(run, device=required_device, batch_size=batch, median_ms_per_origin=None,
                    cost_valid=False, cost_status='NOT_MEASURED', cost_id='', cost_accuracy_run_id=run['run_id'],
                    cost_protocol_scope='No corresponding measured ' + required_device.upper() + ' route'))
    return output


def comparison_rows(runs, comparisons):
    index = _index(runs)
    result = []
    for raw in comparisons:
        row = dict(raw)
        row['kind'] = str(row['kind']).upper()
        row['cap'] = integer(row.get('cap'))
        row['seed'] = integer(row.get('seed'))
        row.setdefault('comparison_id', ':'.join(str(row.get(key,'')) for key in ('kind','dataset','seed','cap')))
        if row['kind'] in ('REVIN','COMPOSITION') and row.get('match_status') == 'MATCHED' and row['cap'] not in (3,6):
            raise ValueError('Matched comparison requires a declared registered cap3 or cap6')
        candidate = _lookup(index, row['candidate_run_id'], row['cap'])
        control = _lookup(index, row['control_run_id'], row['cap'])
        for key in ('dataset', 'seed', 'C', 'K', 'L', 'H'):
            if candidate.get(key) != control.get(key):
                raise ValueError('Comparison axis identity mismatch: ' + key)
        row['dataset'] = candidate['dataset']
        row['candidate_selected_epoch'] = candidate.get('selected_epoch')
        row['control_selected_epoch'] = control.get('selected_epoch')
        row['candidate_training_status'] = candidate.get('training_status')
        row['control_training_status'] = control.get('training_status')
        row['final'] = candidate['final'] is True and control['final'] is True
        if row['kind'] == 'REVIN' and row.get('match_status') == 'MATCHED':
            if candidate.get('revin') is not True or control.get('revin') is not False:
                raise ValueError('RevIN comparison must be ON minus OFF')
            if candidate.get('family') != 'B2' or control.get('family') != 'B2':
                raise ValueError('RevIN comparison must use the same B2 family')
        if row['kind'] == 'COMPOSITION' and row.get('match_status') == 'MATCHED':
            if candidate.get('family') != 'B2' or control.get('family') != 'B0' or candidate['dataset'] != 'jena':
                raise ValueError('Composition comparison must be Jena B2 minus B0')
            if candidate.get('revin') is None or candidate['revin'] is not control.get('revin'):
                raise ValueError('Composition normalization must be identical')
        for metric in ('MSE', 'MAE'):
            left, right = candidate[metric], control[metric]
            delta = None if left is None or right is None else left - right
            supplied = number(row.get('delta_' + metric))
            if supplied is not None and (delta is None or not math.isclose(supplied, delta, abs_tol=1e-12, rel_tol=1e-12)):
                raise ValueError('Supplied comparison differs from exact referenced metric')
            row.update({f'candidate_{metric}':left, f'control_{metric}':right,
                f'delta_{metric}':delta, f'delta_{metric}_pct':None if right in (None, 0) or delta is None else 100 * delta / right})
        row['interpretation'] = 'Selected recipe comparison; selected epochs may differ'
        result.append(row)
    return result


def aggregate_statistics(rows):
    sums = {}
    for row in rows:
        channel = str(row['channel_id'])
        sse, sae, count = number(row['SSE']), number(row['SAE']), integer(row['count'])
        if sse is None or sae is None or count is None or min(sse, sae, count) < 0:
            raise ValueError('Invalid public error statistic')
        totals = sums.setdefault(channel, [[], [], 0])
        totals[0].append(sse)
        totals[1].append(sae)
        totals[2] += count
    if not sums or any(totals[2] == 0 for totals in sums.values()):
        raise ValueError('Every registered channel needs observed targets')
    return dict(MSE=math.fsum(math.fsum(sse) / count for sse, _, count in sums.values()) / len(sums),
        MAE=math.fsum(math.fsum(sae) / count for _, sae, count in sums.values()) / len(sums),
        observed=sum(totals[2] for totals in sums.values()), channels=len(sums))


def error_change_rows(statistics, comparisons, runs):
    index = _index(runs)
    by_run = {}
    for row in statistics:
        key = (str(row['channel_id']), integer(row['horizon']))
        group = by_run.setdefault(row['run_id'], {})
        if key in group:
            raise ValueError('Duplicate channel/horizon statistic')
        group[key] = row
    result = []
    for pair in comparisons:
        if pair['kind'] not in ('REVIN','COMPOSITION') or pair.get('match_status') != 'MATCHED' or not pair['final']:
            continue
        left = by_run.get(pair['candidate_run_id'], {})
        right = by_run.get(pair['control_run_id'], {})
        if not left or left.keys() != right.keys():
            raise ValueError('Matched comparison error-statistic coordinates differ')
        for role, stats in (('candidate', left), ('control', right)):
            totals = aggregate_statistics(stats.values())
            run = _lookup(index, pair[role+'_run_id'], pair['cap'])
            if totals['channels'] != run['C']:
                raise ValueError('Error statistics omit a registered channel')
            if len({key[1] for key in stats}) != run['H'] or len(stats) != run['C'] * run['H']:
                raise ValueError('Error statistics omit a registered channel/horizon coordinate')
            if run.get('observed') is not None and totals['observed'] != run['observed']:
                raise ValueError('Error statistics have different observed target count')
            for metric in ('MSE', 'MAE'):
                if not math.isclose(totals[metric], run[metric], abs_tol=1e-12, rel_tol=1e-12):
                    raise ValueError('Error statistics do not reconstruct saved run macro metric')
        for key, a in left.items():
            b = right[key]
            n1, n0 = integer(a['count']), integer(b['count'])
            if n1 != n0:
                raise ValueError('Matched comparison target counts differ')
            candidate = _lookup(index,pair['candidate_run_id'],pair['cap'])
            control = _lookup(index,pair['control_run_id'],pair['cap'])
            row = dict(kind=pair['kind'], comparison_id=pair['comparison_id'],
                dataset=pair['dataset'], seed=pair['seed'],
                cap=pair['cap'], candidate_run_id=pair['candidate_run_id'], control_run_id=pair['control_run_id'],
                candidate_revin=candidate['revin'], control_revin=control['revin'],
                channel_id=key[0], horizon=key[1], count=n1,
                delta_MSE=None if not n1 else (number(a['SSE'])-number(b['SSE'])) / n1,
                delta_MAE=None if not n1 else (number(a['SAE'])-number(b['SAE'])) / n1,
                candidate_SSE=number(a['SSE']), control_SSE=number(b['SSE']),
                candidate_SAE=number(a['SAE']), control_SAE=number(b['SAE']))
            result.append(row)
    return result


def training_note(status):
    if status == 'TRAINING_CAP_STILL_IMPROVING':
        return '상한에서도 개선 중; 수렴 여부 미확정'
    if status in ('CAP_OR_UNDERTRAINING', 'UNDERTRAINED_OR_BUDGET_LIMIT'):
        return '학습 상한 또는 학습량 부족; 개선 지속 여부 미확정; 수렴 여부 미확정'
    if status == 'NO_TRAINING':
        return '무학습 기준선'
    return '등록한 선택 구간 완료; 수렴 여부 미확정'


def _plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def close_figure(fig):
    _plotting().close(fig)


def _label(row):
    seed = integer(row.get('seed'))
    text = str(row.get('method', row.get('family', 'Unknown')))
    if seed is not None:
        text += f' s{seed}'
    cap = integer(row.get('cap'))
    if cap is not None:
        text += f' cap{cap}/e{row.get("selected_epoch", "?")}'
    return text


def _tidy(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.15)


def accuracy_cost_figure(rows):
    plt = _plotting()
    valid = [row for row in rows if row['cost_valid'] is True and row['median_ms_per_origin'] is not None]
    facets = sorted({(row['dataset'], row['device'], row['batch_size']) for row in valid})
    panels = max(1, len(facets))
    fig, axes = plt.subplots(math.ceil(panels / 2), 2, figsize=(12, 3.8 * math.ceil(panels / 2)), squeeze=False,
                             layout='constrained')
    methods = sorted({str(row['method']) for row in valid})
    palette = {method:COLORS[i % len(COLORS)] for i, method in enumerate(methods)}
    for ax, facet in zip(axes.flat, facets):
        for row in valid:
            if (row['dataset'], row['device'], row['batch_size']) != facet:
                continue
            marker = 's' if row.get('seed') == 92602 else ('o' if row.get('seed') == 92601 else 'D')
            ax.scatter(row['median_ms_per_origin'], row['MSE'], marker=marker, color=palette[str(row['method'])],
                       s=48, label=_label(row), edgecolors='white', linewidth=.4)
        ax.set_title(f'{facet[0]} | {facet[1]} | B{facet[2]}')
        ax.set_xscale('log')
        ax.set_xlabel('Whole-path latency (ms/origin; log scale)')
        ax.set_ylabel('Macro MSE (normalized²)')
        ax.legend(fontsize=7, loc='best', framealpha=.85)
        _tidy(ax)
    for ax in list(axes.flat)[len(facets):]:
        ax.axis('off')
    if not facets:
        axes.flat[0].text(.5, .5, 'No valid measured final route', ha='center', transform=axes.flat[0].transAxes)
    fig.suptitle('Accuracy and cost — repeated development V; separate device and batch', fontsize=12)
    return fig


def revin_figure(rows):
    plt = _plotting()
    pairs = [row for row in rows if row['kind']=='REVIN' and row.get('match_status')=='MATCHED']
    datasets = sorted({row['dataset'] for row in pairs})
    fig, axes = plt.subplots(max(1,len(datasets)), 2, figsize=(11, 3.3 * max(1,len(datasets))),
                             squeeze=False, layout='constrained')
    for i, dataset in enumerate(datasets):
        for j, metric in enumerate(('MSE', 'MAE')):
            ax = axes[i,j]
            for seed, color, marker in ((92601,COLORS[0],'o'),(92602,COLORS[1],'s')):
                selected = sorted((row for row in pairs if row['dataset']==dataset and row['seed']==seed), key=lambda row:row['cap'])
                ax.plot([row['cap'] for row in selected], [row['delta_'+metric] for row in selected],
                        marker=marker, color=color, linestyle='--', label=f'seed {seed}')
            ax.axhline(0, color='#555555', linewidth=.8)
            ax.set_title(f'{dataset} | {metric}: ON minus OFF')
            ax.set_xticks(sorted({row['cap'] for row in pairs if row['dataset']==dataset}))
            ax.set_xlabel('Registered search cap (epochs)')
            ax.set_ylabel(f'Δ macro {metric} ('+('normalized²)' if metric=='MSE' else 'normalized units)'))
            ax.legend(frameon=False)
            _tidy(ax)
    fig.suptitle('RevIN paired recipe comparison — negative improves; both seeds retained', fontsize=12)
    return fig


def error_change_figure(rows):
    plt = _plotting()
    grouped, seen = {}, set()
    for row in rows:
        kind = str(row['kind']).upper()
        if kind not in ('REVIN','COMPOSITION'):
            raise ValueError('Unknown error-comparison kind')
        facet = (kind,row['comparison_id'],row['dataset'],row['seed'],row['cap'])
        cell = (kind,row['comparison_id'],row['channel_id'],row['horizon'])
        if cell in seen:
            raise ValueError('Duplicate comparison/channel/horizon cell')
        seen.add(cell)
        grouped.setdefault(facet,[]).append(row)
    facets = sorted(grouped)
    fig, axes = plt.subplots(max(1,math.ceil(len(facets)/2)), 2,
        figsize=(12, 4.6 * max(1,math.ceil(len(facets)/2))), squeeze=False, layout='constrained')
    limits = {}
    for row in rows:
        value = row['delta_MSE']
        if value is not None:
            limits[row['dataset']] = max(limits.get(row['dataset'],0.), abs(value))
    for ax, facet in zip(axes.flat,facets):
        kind,identifier,dataset,seed,cap = facet
        selected = grouped[facet]
        normalization = {(parse_bool(row['candidate_revin']),parse_bool(row['control_revin'])) for row in selected}
        if len(normalization) != 1:
            raise ValueError('Error-comparison normalization changed within a facet')
        candidate_norm,control_norm = normalization.pop()
        if kind=='REVIN':
            if (candidate_norm,control_norm) != (True,False):
                raise ValueError('RevIN heatmap must be ON minus OFF')
            comparison_label = 'REVIN ON − OFF'
        else:
            if candidate_norm is None or candidate_norm is not control_norm:
                raise ValueError('Composition heatmap requires the same normalization')
            comparison_label = 'COMPOSITION B2 − B0 | RevIN '+('ON' if candidate_norm else 'OFF')
        channels = list(dict.fromkeys(row['channel_id'] for row in selected))
        horizons = sorted({row['horizon'] for row in selected})
        cells = {(row['channel_id'],row['horizon']):row['delta_MSE'] for row in selected}
        matrix = np.array([[np.nan if cells.get((channel,horizon)) is None else cells[channel,horizon]
                            for horizon in horizons] for channel in channels])
        limit = limits.get(dataset,0.) or 1e-12
        image = ax.imshow(matrix, cmap='RdBu_r', vmin=-limit, vmax=limit, aspect='auto', interpolation='nearest')
        ax.set_yticks(range(len(channels)), channels, fontsize=7)
        step = max(1,len(horizons)//8)
        ax.set_xticks(range(0,len(horizons),step), horizons[::step])
        ax.set_xlabel('Forecast horizon index')
        ax.set_ylabel('Original channel (registered order)')
        ax.set_title(f'{comparison_label}\n{dataset} | seed {seed} | cap {cap}\n{identifier}',fontsize=9)
        fig.colorbar(image, ax=ax, label='Δ MSE (normalized²)', shrink=.85)
    for ax in list(axes.flat)[len(facets):]:
        ax.axis('off')
    if not facets:
        axes.flat[0].text(.5,.5,'No matched final error-statistic comparison',ha='center',transform=axes.flat[0].transAxes)
    fig.suptitle('Channel × horizon errors — negative improves the named candidate; zero-count cells masked', fontsize=12)
    return fig


def training_curve_rows(history, runs):
    indexed = {}
    for raw in history:
        if raw.get('stage') not in (None, '', 'search', 'A_fit'):
            continue
        value = number(raw.get('astop_mse', raw.get('A_stop_MSE')))
        if value is None:
            continue
        row = dict(raw, epoch=number(raw['epoch']), astop_mse=value,
                   seed=integer(raw['seed']), revin=parse_bool(raw['revin']), cap=integer(raw.get('cap')))
        key = (row['dataset'], row.get('family','B2'), row['seed'], row['revin'], row['epoch'])
        if key in indexed and not math.isclose(indexed[key]['astop_mse'],value,rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError('Duplicated search epoch has different A_stop score')
        if key not in indexed or (row['cap'] or 0) > (indexed[key]['cap'] or 0):
            indexed[key] = row
    final = [row for row in map(_run,runs) if row['final'] and row.get('family') in ('B2','B0')]
    for row in indexed.values():
        choices = [run for run in final if (run['dataset'],run['family'],run['seed'],run['revin']) ==
                   (row['dataset'],row.get('family','B2'),row['seed'],row['revin'])]
        if len(choices)==1:
            row.update(selected_epoch=choices[0]['selected_epoch'], final_cap=choices[0]['cap'],
                       training_status=choices[0]['training_status'])
    return list(indexed.values())


def training_figure(rows):
    plt = _plotting()
    groups = sorted({(row['dataset'],row.get('family','B2')) for row in rows})
    fig, axes = plt.subplots(max(1,len(groups)), 2, figsize=(11,3.4 * max(1,len(groups))),
                             squeeze=False, layout='constrained')
    for i, group in enumerate(groups):
        for j, seed in enumerate((92601,92602)):
            ax = axes[i,j]
            selected = [row for row in rows if (row['dataset'],row.get('family','B2'),row['seed'])==(*group,seed)]
            caps = {integer(row.get('final_cap')) for row in selected if row.get('final_cap') is not None}
            for enabled,color,marker in ((False,COLORS[0],'o'),(True,COLORS[1],'s')):
                line = sorted((row for row in selected if row['revin'] is enabled),key=lambda row:row['epoch'])
                if not line:
                    continue
                ax.plot([row['epoch'] for row in line],[row['astop_mse'] for row in line],
                        color=color,marker=marker,markersize=4,label='RevIN ON' if enabled else 'RevIN OFF')
                stars = [row for row in line if row.get('selected_epoch')==row['epoch']]
                if stars:
                    ax.scatter([row['epoch'] for row in stars],[row['astop_mse'] for row in stars],
                               marker='*',s=140,color=color,edgecolors='black',linewidth=.5,zorder=4)
            for cap in caps:
                ax.axvline(cap,color='#777777',linestyle=':',linewidth=.8)
            ax.set_title(f'{group[0]} | {group[1]} | seed {seed} | cap '+('/'.join(map(str,sorted(caps))) or 'unknown'))
            ax.set_xlabel('Search epochs; star = selected checkpoint')
            ax.set_ylabel('A_stop original-channel macro MSE')
            ax.legend(frameon=False,fontsize=8)
            _tidy(ax)
    fig.suptitle('Selection curves — A_fit training / A_stop selection; convergence unresolved at the cap',fontsize=12)
    return fig


def _format(value):
    return 'UNKNOWN' if value is None or value == '' else (f'{value:.8g}' if isinstance(value,float) else str(value))


def _documents(runs, table, comparisons, contract, decisions, input_hashes, historical=None):
    final = [row for row in map(_run,runs) if row['final']]
    findings = []
    for row in comparisons:
        if row['kind']=='REVIN' and row.get('match_status')=='MATCHED':
            findings.append(f"{row['dataset']} cap{row['cap']} seed{row['seed']}: "
                f"OFF MSE {_format(row['control_MSE'])} → ON {_format(row['candidate_MSE'])} "
                f"(ON−OFF {_format(row['delta_MSE'])}; {_format(row['delta_MSE_pct'])}%); "
                f"MAE {_format(row['control_MAE'])} → {_format(row['candidate_MAE'])} "
                f"({_format(row['delta_MAE_pct'])}%). 선택 epoch OFF/ON은 "
                f"{_format(row['control_selected_epoch'])}/{_format(row['candidate_selected_epoch'])}이다.")
    accuracy = '\n'.join(f"{row['dataset']:8s} {_label(row):38s} MSE={_format(row['MSE'])} MAE={_format(row['MAE'])}"
                         for row in final)
    limits = '\n'.join(f"{_label(row)}: {training_note(row.get('training_status'))}" for row in final)
    failed_costs = sum(row['cost_valid'] is not True for row in table)
    sources = '\n'.join(f'{name}  {digest}' for name,digest in input_hashes.items())
    composition = [row for row in comparisons if row['kind']=='COMPOSITION']
    composition_text = '\n\n'.join(f"Jena seed{row['seed']} cap{row['cap']}: "
        f"{row['candidate_run_id']} − {row['control_run_id']}의 MSE 차이는 "
        f"{_format(row['delta_MSE'])}, MAE 차이는 {_format(row['delta_MAE'])}다. "
        f"대응 상태는 {row.get('match_status','UNKNOWN')}이며 구성 기여와 전체 구조의 우열을 구분한다."
        for row in composition) or '대응 Jena B0/B2 결과는 현재 공개 비교 입력에서 확인되지 않았다.'
    metadata = json.dumps(decisions,ensure_ascii=False,indent=2,sort_keys=True)
    refs = contract.get('method_references', {})
    reference_text = json.dumps(refs,ensure_ascii=False,indent=2,sort_keys=True) if refs else '공식 원문·수정점·source pin은 CAMPAIGN_CONTRACT.json과 기존 METHOD_REFERENCES.md를 따른다.'
    evidence = historical if historical is not None else contract.get('reused_public_artifacts',[])
    reused_artifacts = json.dumps(evidence,ensure_ascii=False,indent=2,sort_keys=True)
    records = evidence.get('artifacts', []) if isinstance(evidence,dict) else evidence
    historical_links = []
    for row in records:
        reference = row.get('reference','')
        if reference.startswith('repo://'):
            relative = reference[len('repo://'):]
            if relative.startswith('/') or '..' in Path(relative).parts:
                raise ValueError('Historical public reference must be repository relative')
            historical_links.append(f"[{Path(relative).name}](../../{relative}) — "
                f"{row.get('scope','Historical evidence')}；SHA256 `{row.get('sha256','UNKNOWN')}`")
    historical_text = '\n\n'.join(historical_links)
    excluded = ('SOFTS, FreTS, LIFT, N-HiTS, CycleNet/TimesNet, ForeCA, 전체 AdaPTS 비교와 '
                '추가 K·초기화 탐색은 제외했다. PeMS32 D는 이번 최종 발표 근거로 쓰지 않아 seed2 보완 조건이 발동하지 않는다.')
    return {
        'README_KO.md': '# B2 + RevIN 최소 실행 결과\n\n'
            '정확도–비용, 대응 RevIN 비교, Jena 구성 기여·한계의 세 묶음이다. '
            '신규 개선 방법은 outer RevIN 하나이며 두 데이터의 seed92601/92602를 유지한다.\n\n'
            '[결과 분석](RESULT_ANALYSIS_KO.md) · [선택 결정](DECISIONS_KO.md) · [재현](REPRODUCE_KO.md)\n\n'
            '- [T1 정확도–비용](T1_ACCURACY_COST.csv)\n- [T2 RevIN cap3/6 대응](T2_REVIN_MATCHED.csv)\n'
            '- [T3 Jena 구성 기여](T3_COMPOSITION_LIMITS.csv)\n\n' + LIMITATION + '\n\n'
            '그림: [F1](figures/F1_accuracy_cost.png), [F2](figures/F2_revin_paired.png), '
            '[F3](figures/F3_channel_horizon.png), [F4](figures/F4_astop_curves.png). '
            'SVG와 source CSV도 같은 figures 폴더에 있다. 실제 검산과 원격 게시 증거는 별도 검증 JSON을 따른다.\n\n'
            +historical_text+'\n',
        'RESULT_ANALYSIS_KO.md': '# 결과 분석\n\n'+LIMITATION+'\n\n'
            '주지표는 원채널 균등 macro MSE, 보조지표는 같은 공간의 macro MAE다. '
            'global A-fit/A60 z-score 단위이며 시계열을 채널 전체 pooled MSE로 바꾸지 않았다.\n\n'
            '## 정확도와 실제 비용\n\n```text\n'+accuracy+'\n```\n\n'
            f'비용은 정확한 accuracy_run_id와 연결했다. CPU/GPU와 B1/B8은 따로 읽는다. '
            f'미측정·검증 실패 비용 {failed_costs}행은 숫자를 채우지 않았다. '
            '원자료+mask→fill→global norm→RevIN/P/E/F/D→inverse→CPU 출력의 전체 경로이며 '
            '고정 특징 캐시의 읽기 속도를 추론 비용으로 쓰지 않는다. '
            '같은 방법명이 다른 seed·부모·장치의 정확도/비용을 연결하는 근거가 되지 않는다.\n\n'
            '## RevIN 대응 결과\n\n'+'\n\n'.join(findings)+'\n\n'
            'cap3와 조건부 cap6를 각각 보존한다. ON만 더 학습한 비교가 아니며, '
            '두 seed의 역방향 결과도 남긴다. 선택-refit 결과는 같은 선택 recipe의 비교이며 '
            '선택 epoch가 다르면 동일 업데이트 효과로 해석하지 않는다. '
            'origin별 대응 값은 PAIR_ORIGIN_DIFFS.csv에 있으며 IID 유의성이나 두 seed의 신뢰구간을 새로 주장하지 않는다.\n\n'
            '## 구성 기여와 적용 한계\n\n'+composition_text+'\n\n'+excluded+'\n\n'
            'Jena B0는 최종 B2의 정규화·학습 상한에 맞춘 구성 대조다. '
            'B0도 상한에서 개선 중이면 학습량 부족 가능성을 남기며, '
            '이번 상한의 결과로 수렴이나 모든 학습 예산에서의 우위를 주장하지 않는다.\n\n'
            '## 학습 상한\n\n'+limits+'\n\n'
            '상한 도달을 방법 실패나 수렴 완료로 바꿔 부르지 않는다. '
            '선택과 데이터 단위 확장·공통 정규화는 A_stop만으로 고정하고, 각 선택 epoch에서 fresh A60 refit했다.\n',
        'DECISIONS_KO.md': '# 선택·확장 결정과 범위\n\n'
            '첫 비교는 기존과 같은 0/.25/1/3 epoch checkpoint다. '
            '데이터의 OFF/ON×두 seed 중 하나라도 마지막 checkpoint가 strict best이면 '
            '해당 네 arm 모두 4/5/6 epoch까지 확인한다. 원 optimizer/sampler/CPU·CUDA RNG의 '
            'immutable continuation을 사용하고 마지막 선택 뒤 fresh A60을 수행한다.\n\n'
            '공통 정규화는 데이터별 두 seed의 A_stop best MSE 평균으로 선택하며 동률 OFF다. '
            '같은 정규화·장치·선택 규칙의 기존 B0가 없으면 Jena B0 두 seed를 추가한다.\n\n'
            '실제 봉인 결정은 아래 입력과 DECISIONS.json의 source binding을 따른다.\n\n```json\n'+metadata+'\n```\n\n'
            +excluded+'\n\n'+LIMITATION+'\n',
        'REPRODUCE_KO.md': '# 공개 표·그림 재생성\n\n'
            '이 명령은 공개 CSV/JSON 입력과 선택적인 기존 그림 출처 메타데이터만 읽으며 '
            '모델·원자료·private NPZ·GPU를 사용하지 않는다. '
            '실험의 재학습과 공개 집계의 재생성은 다른 작업이다. 프로젝트 Python 환경에서 실행한다.\n\n'
            '```powershell\npython -m research.b2_revin_20261018_v1.reporting '
            '--report-root reports/b2_revin_20261018_v1 '
            '--output-root .cache/b2_revin_public_replay\n```\n\n'
            'matplotlib Agg backend, 고정 SVG hash salt, timestamp 없는 PNG/SVG metadata를 사용한다. '
            '동일 환경과 입력 bytes에서 표·문서·그림 bytes를 대사한다. '
            '보고 generator는 실험 상태·선택·검산 입력을 수정하지 않는다.\n\n'
            '입력 SHA256:\n\n```text\n'+sources+'\n```\n\n'+LIMITATION+'\n',
        'REPORT_METHOD_NOTES_KO.md': '# 보고 집계의 방법과 출처\n\n'
            '공식 RevIN 원문·코드 revision·라이선스와 포트 차이는 별도 '
            '[METHOD_REFERENCES.md](METHOD_REFERENCES.md)를 따른다.\n\n'
            '기존 B2와 데이터·학습 계약은 봉인된 broad/diagnostics/improvements source를 재사용한다. '
            '이번 변경은 원 C채널 창별 outer RevIN이다. 공식 전체 학습 recipe의 재현으로 부르지 않는다.\n\n'
            '```text\n'+reference_text+'\n```\n\n'
            'RevIN은 causal fill와 train z-score 뒤 biased variance, eps1e-5, detached mean/std, '
            'gamma1/beta0를 사용한다. 공식 affine inverse 분모 gamma+eps²를 포함한다. '
            'anchor까지 포함한 출력을 한 번 inverse하고 global-normalized target 공간에서 loss를 구한다. '
            '실제 source hash·수정점은 CAMPAIGN_CONTRACT.json 및 RUNS의 bindings를 따른다.\n'
            '\n기존 검증 결과·그림의 재사용 경로와 원 SHA는 아래 목록이다. '
            '과거 측정 조건은 그대로 유지하며 새 RevIN 결과로 재레이블하지 않는다.\n\n'
            '```json\n'+reused_artifacts+'\n```\n'
    }


def build(report_root, output_root=None):
    report = Path(report_root)
    output = report if output_root is None else Path(output_root)
    names = list(INPUTS)
    historical_path = report/'HISTORICAL_ARTIFACTS.json'
    if historical_path.is_file():
        names.append(historical_path.name)
    input_hashes = {name:_sha(report/name) for name in names}
    historical = json.loads(historical_path.read_text(encoding='utf-8-sig')) if historical_path.name in input_hashes else None
    tables = {name:_read_csv(report/name) for name in INPUTS if name.endswith('.csv')}
    contract = json.loads((report/'CAMPAIGN_CONTRACT.json').read_text(encoding='utf-8-sig'))
    decisions = json.loads((report/'DECISIONS.json').read_text(encoding='utf-8-sig'))
    runs = tables['RUNS.csv']
    cost = accuracy_cost_rows(runs,tables['COST.csv'])
    comparisons = comparison_rows(runs,tables['COMPARISONS.csv'])
    coverage = {}
    for row in comparisons:
        if row['kind']=='REVIN' and row.get('match_status')=='MATCHED':
            coverage.setdefault((row['dataset'],row['cap']),[]).append(row['seed'])
    if any(sorted(seeds) != [92601,92602] for seeds in coverage.values()):
        raise ValueError('Each matched RevIN dataset/cap requires both seeds exactly once')
    if not all((dataset,3) in coverage for dataset in ('jena','pems32')):
        raise ValueError('Both datasets require the initial cap3 RevIN comparison')
    pair_counts, pair_origins = {}, {}
    for row in tables['PAIR_ORIGIN_DIFFS.csv']:
        identifier = row.get('comparison_id') or ':'.join(str(value) for value in
            (str(row['kind']).upper(),row['dataset'],integer(row['seed']),integer(row['cap'])))
        pair_counts[identifier] = pair_counts.get(identifier,0)+1
        pair_origins.setdefault(identifier,set()).add(integer(row['origin']))
    for row in comparisons:
        row['paired_stat_rows'] = pair_counts.get(row['comparison_id'],0)
        row['paired_origin_rows'] = len(pair_origins.get(row['comparison_id'],set()))
    changes = error_change_rows(tables['ERROR_SUFFICIENT_STATS.csv'],comparisons,runs)
    curves = training_curve_rows(tables['TRAINING_HISTORY.csv'],runs)
    revin = [row for row in comparisons if row['kind']=='REVIN']
    composition = [dict(row, limitation='Same normalization/recipe; fixed K and two-seed development scope')
                   for row in comparisons if row['kind']=='COMPOSITION']
    output.mkdir(parents=True,exist_ok=True)
    figures = output/'figures'
    figures.mkdir(exist_ok=True)
    for name,rows in [('T1_ACCURACY_COST.csv',cost),('T2_REVIN_MATCHED.csv',revin),('T3_COMPOSITION_LIMITS.csv',composition)]:
        _write_csv(output/name,rows)
    for name,text in _documents(runs,cost,comparisons,contract,decisions,input_hashes,historical).items():
        (output/name).write_text(text,encoding='utf-8',newline='\n')
    plt = _plotting()
    import matplotlib
    artifacts = []
    with matplotlib.rc_context({'svg.hashsalt':'b2_revin_20261018_v1', 'font.family':'DejaVu Sans',
            'font.size':9, 'axes.titlesize':10, 'axes.labelsize':9, 'legend.fontsize':8,
            'svg.fonttype':'none', 'savefig.facecolor':'white'}):
        for name,rows,factory in [('F1_accuracy_cost',cost,accuracy_cost_figure),
                ('F2_revin_paired',revin,revin_figure),('F3_channel_horizon',changes,error_change_figure),
                ('F4_astop_curves',curves,training_figure)]:
            _write_csv(figures/(name+'.source.csv'),rows)
            fig = factory(rows)
            try:
                fig.savefig(figures/(name+'.png'),dpi=220,bbox_inches='tight',metadata={'Software':'b2_revin public reporting'})
                fig.savefig(figures/(name+'.svg'),bbox_inches='tight',metadata={'Date':None,'Creator':'b2_revin public reporting'})
            finally:
                plt.close(fig)
            artifacts.append(name)
    current = {name:_sha(report/name) for name in names}
    if current != input_hashes:
        raise ValueError('Public reporting inputs changed while generating artifacts')
    paths = ['T1_ACCURACY_COST.csv','T2_REVIN_MATCHED.csv','T3_COMPOSITION_LIMITS.csv',
             'README_KO.md','RESULT_ANALYSIS_KO.md','DECISIONS_KO.md','REPRODUCE_KO.md','REPORT_METHOD_NOTES_KO.md']
    paths += ['figures/'+name+suffix for name in artifacts for suffix in ('.png','.svg','.source.csv')]
    state = dict(evaluation_scope='REPEATED_DEVELOPMENT_V', convergence_claim='UNRESOLVED',
        publication_status='SEPARATE_REMOTE_PROOF', input_hashes=input_hashes,
        final_accuracy_rows=sum(_run(row)['final'] is True for row in runs),
        accuracy_cost_rows=len(cost), matched_revin_rows=len(revin), composition_rows=len(composition),
        error_change_rows=len(changes), figures=artifacts, generated_sha256={name:_sha(output/name) for name in paths})
    (output/'REPORT_OUTPUTS.json').write_text(json.dumps(state,indent=2,ensure_ascii=False,sort_keys=True)+'\n',encoding='utf-8',newline='\n')
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path)
    args = parser.parse_args(argv)
    state = build(args.report_root,args.output_root)
    print(json.dumps({key:state[key] for key in ('evaluation_scope','final_accuracy_rows','matched_revin_rows','figures')}))


if __name__ == '__main__':
    main()
