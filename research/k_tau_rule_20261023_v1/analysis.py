"""Public-table-only K_tau analysis; importing performs no campaign work."""
import csv
import json
import math
from pathlib import Path

import numpy as np


DATASETS = ('jena', 'ETTh1', 'ETTh2_safe17400', 'appliances')
ELIGIBLE_DATASETS = ('jena', 'ETTh1', 'appliances')
SEEDS = tuple(range(92601, 92611))


def _read_csv(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def _number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('Public metric must be finite')
    return result


def _two_seed(rows):
    selected = [row for row in rows if row['seed'] in SEEDS[:2]]
    if sorted(row['seed'] for row in selected) != list(SEEDS[:2]):
        return None
    values = [row['mse'] for row in selected]
    return dict(mean=math.fsum(values)/2, min=min(values), max=max(values))


def _inputs(root):
    tau = {}
    for row in _read_csv(root/'K_TAU.csv'):
        if not math.isclose(_number(row['tau']), .9, rel_tol=0., abs_tol=1e-12):
            continue
        dataset = row['dataset']
        if dataset not in DATASETS or dataset in tau:
            raise ValueError('Unknown or duplicate primary K_tau dataset')
        k, channels = int(row['K_A_fit']), int(row['C'])
        if not 1 <= k <= channels:
            raise ValueError('K_tau is outside original channel count')
        tau[dataset] = dict(K_main=k, K_A60=int(row['K_A60']), C=channels)
    runs = []
    identities = set()
    for row in _read_csv(root/'RUNS.csv'):
        dataset, role = row['dataset'], row['role']
        if dataset not in DATASETS or role not in ('main', 'comparison'):
            raise ValueError('Unregistered public run dataset or role')
        seed, k = int(row['seed']), int(row['K'])
        if seed not in (SEEDS if role == 'main' else SEEDS[:2]):
            raise ValueError('Public run contains an unregistered seed')
        if role == 'main' and dataset in tau and k != tau[dataset]['K_main']:
            raise ValueError('Main run K differs from preregistered A_fit K_tau')
        identity = (dataset, role, seed) if role == 'main' else (dataset, role, k, seed)
        if identity in identities:
            raise ValueError('Duplicate public run identity')
        identities.add(identity)
        runs.append(dict(row, K=k, seed=seed, mse=_number(row['mse']), mae=_number(row['mae'])))
    plan_path = root/'RUN_PLAN.json'
    plan = json.loads(plan_path.read_text(encoding='utf-8-sig')) if plan_path.exists() else {}
    g4 = plan.get('G4', {})
    baselines = {dataset: dict(raw=None, candidates={},
        comparison_required=g4.get(dataset, {}).get('reused_historical_results') is False)
        for dataset in DATASETS}
    for row in _read_csv(root/'BASELINE_RECALCULATION.csv'):
        dataset = row['dataset']
        if dataset not in baselines or row.get('row_policy', 'DEFAULT128') != 'DEFAULT128':
            continue
        target = baselines[dataset]
        if row['family'] == 'RAW':
            if target['raw'] is not None:
                raise ValueError('Duplicate deterministic Raw baseline')
            target['raw'] = _number(row['MSE_mean'])
        elif row['family'] == 'B2':
            k = int(row['K'])
            if k in target['candidates']:
                raise ValueError('Duplicate historical B2 K baseline')
            target['candidates'][k] = dict(mean=_number(row['MSE_mean']),
                min=_number(row['MSE_min']), max=_number(row['MSE_max']))
    return tau, runs, baselines


def _summarize(dataset, tau, runs, baseline):
    main = sorted([row for row in runs if row['dataset'] == dataset and row['role'] == 'main'],
                  key=lambda row: row['seed'])
    comparisons = [row for row in runs if row['dataset'] == dataset and row['role'] == 'comparison']
    comparison_required = baseline['comparison_required'] or bool(comparisons)
    ktau = tau.get(dataset, {})
    candidates = dict(baseline['candidates'])
    historical_best = min((value['mean'] for value in candidates.values()), default=None)
    comparison_complete = True
    if comparison_required:
        comparison_complete = len(candidates) == 3
        for k in candidates:
            value = _two_seed([row for row in comparisons if row['K'] == k])
            if value is None:
                comparison_complete = False
            else:
                candidates[k] = value
        if any(row['K'] not in baseline['candidates'] for row in comparisons):
            raise ValueError('Comparison run K is outside registered historical candidates')
    first_two = _two_seed(main)
    seed_ids = [row['seed'] for row in main]
    complete = (seed_ids == list(SEEDS) and bool(ktau) and baseline['raw'] is not None
                and len(baseline['candidates']) == 3 and comparison_complete)
    values = [row['mse'] for row in main]
    q1, median, q3 = (map(float, np.percentile(values, [25, 50, 75], method='linear'))
                      if values else (None, None, None))
    iqr = None if not values else q3-q1
    loss = None
    if first_two is not None and len(candidates) == 3 and comparison_complete:
        loss = first_two['mean']-min([first_two['mean']] + [v['mean'] for v in candidates.values()])
    success = median < baseline['raw'] if complete else None
    improved = (first_two['mean'] < historical_best
                if first_two is not None and historical_best is not None else None)
    summary = dict(dataset=dataset, K_main=ktau.get('K_main'), C=ktau.get('C'),
        K_A60=ktau.get('K_A60'), n=len(values), median=median,
        mean=math.fsum(values)/len(values) if values else None,
        Q1=q1, Q3=q3, IQR=iqr, min=min(values) if values else None,
        max=max(values) if values else None,
        MAE_mean=math.fsum(row['mae'] for row in main)/len(main) if main else None,
        Raw=baseline['raw'], first_two_mean=first_two['mean'] if first_two else None,
        loss=loss, limitation='LIMITATION' if loss is not None and iqr is not None and loss > iqr else '',
        main_improves_historical_best=improved, success=success,
        complete=complete, baseline_source='CURRENT_COMPARISON' if comparison_required else 'HISTORICAL',
        comparison_complete=comparison_complete,
        decision_scope='PRIMARY' if dataset in ELIGIBLE_DATASETS else 'DESCRIPTIVE_ONLY',
        status='SUCCESS' if success is True else 'FAIL' if success is False else 'PARTIAL')
    details = dict(summary, missing_seeds=[seed for seed in SEEDS if seed not in seed_ids],
                   seed_ids=seed_ids, comparison_seed_rows=len(comparisons))
    curve = dict(candidates)
    if comparison_required and not comparison_complete:
        curve = {k: value for k in candidates
                 if (value := _two_seed([r for r in comparisons if r['K'] == k])) is not None}
    if first_two is not None and ktau:
        curve[ktau['K_main']] = first_two
    return summary, details, main, curve


def _figures(root, summaries, mains, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with plt.rc_context({'font.family': 'sans-serif', 'font.size': 9, 'axes.labelsize': 9,
                         'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 300}):
        for kind in ('K_CURVE', 'BOXPLOT'):
            fig, axes = plt.subplots(2, 2, figsize=(9, 6.6), constrained_layout=True)
            for index, (axis, row) in enumerate(zip(axes.flat, summaries)):
                dataset, k = row['dataset'], row['K_main']
                axis.set_title(f'{chr(65+index)}  {dataset}', loc='left', fontweight='bold')
                axis.set_ylabel('V normalized-channel MSE')
                axis.grid(axis='y', color='#dddddd', linewidth=.6)
                if row['Raw'] is not None:
                    axis.axhline(row['Raw'], linestyle='--', color='#333333', linewidth=1.1, label='Raw')
                if kind == 'K_CURVE':
                    curve = curves[dataset]
                    for j, (width, value) in enumerate(sorted(curve.items())):
                        axis.errorbar(width, value['mean'],
                            yerr=[[value['mean']-value['min']], [value['max']-value['mean']]],
                            fmt='o', capsize=4, color='#0072B2',
                            label='2-seed mean / min–max' if j == 0 else None)
                        if width == k:
                            axis.scatter([width], [value['mean']], marker='*', s=155,
                                         color='#D55E00', edgecolors='#222222', linewidths=.5,
                                         zorder=5, label='Preselected K_tau')
                    axis.set_xlabel('K')
                    axis.set_xticks(sorted(set(curve) | ({k} if k is not None else set())))
                    axis.text(.02, .03, f'K_tau = {k if k is not None else "unavailable"}',
                              transform=axis.transAxes, fontsize=8)
                else:
                    values = [r['mse'] for r in mains[dataset]]
                    if values:
                        q1, median, q3 = np.percentile(values, [25, 50, 75], method='linear')
                        iqr = q3-q1
                        inside = [v for v in values if q1-1.5*iqr <= v <= q3+1.5*iqr]
                        axis.bxp([dict(med=median, q1=q1, q3=q3,
                            whislo=min(inside), whishi=max(inside), fliers=[])],
                            positions=[1], widths=.35, showfliers=False, patch_artist=True,
                            boxprops=dict(facecolor='#c9e2ef', edgecolor='#0072B2'),
                            medianprops=dict(color='#D55E00', linewidth=1.5))
                        axis.scatter(1+np.linspace(-.09, .09, len(values)), values,
                                     s=22, color='#222222', alpha=.8, zorder=4, label='Individual seeds')
                    axis.set_xticks([1], [f'K_tau={k}; n={len(values)}/10'])
                    axis.set_xlim(.6, 1.4)
                    if not row['complete']:
                        axis.text(.03, .96, 'PARTIAL', transform=axis.transAxes, va='top', color='#D55E00')
                handles, labels = axis.get_legend_handles_labels()
                if handles:
                    axis.legend(handles, labels, loc='best', fontsize=7, frameon=False)
            caption = ('Error bars: two fixed seeds, descriptive min–max; not confidence intervals.'
                       if kind == 'K_CURVE' else 'Linear quartiles; whiskers: 1.5 IQR; all available seeds shown.')
            fig.supxlabel(caption, fontsize=8)
            for extension in ('png', 'svg'):
                fig.savefig(root/f'FIG_{kind}.{extension}')
            plt.close(fig)


def build_report(report_root, *, stop_reason=None, elapsed_seconds=None):
    """Write outputs from the campaign's public CSV tables and optional run plan."""
    root = Path(report_root)
    if not root.is_dir():
        raise ValueError('Existing campaign report directory required')
    tau, runs, baseline = _inputs(root)
    summaries, details, mains, curves = [], {}, {}, {}
    for dataset in DATASETS:
        summary, detail, main, curve = _summarize(dataset, tau, runs, baseline[dataset])
        summaries.append(summary)
        details[dataset], mains[dataset], curves[dataset] = detail, main, curve
    complete = all(row['complete'] for row in summaries)
    failed = [d for d in ELIGIBLE_DATASETS if details[d]['success'] is False]
    verdict = ('ADOPT' if not failed else 'ADOPT_CONDITIONAL' if len(failed) == 1 else 'REPLAN')
    decision = dict(decision='STOPPED' if stop_reason else verdict if complete else 'PARTIAL',
        status='STOPPED' if stop_reason else 'COMPLETE' if complete else 'PARTIAL', complete=complete,
        eligible_datasets=list(ELIGIBLE_DATASETS), failed_datasets=failed, datasets=details,
        stop_reason=stop_reason, elapsed_seconds=elapsed_seconds,
        rules=dict(success='10 fixed main seeds; median V MSE < Raw',
                   quartiles='numpy.percentile(method=linear)', limitation='loss > IQR',
                   loss='first two main seed mean - min(existing candidate means union first two main seed mean)',
                   partial='No ADOPT verdict until all four datasets and required comparison rows are complete'),
        limitations=['V is repeatedly reused development evaluation, not an independent test.',
                     'Two-seed ranges are descriptive, not confidence intervals.',
                     'Decision thresholds are preregistered provisional agreement criteria.',
                     'Historical T exposure and unknown pretraining overlap remain limitations.'])
    with (root/'SUMMARY.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    with (root/'DECISION.json').open('w', encoding='utf-8') as handle:
        json.dump(decision, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    _figures(root, summaries, mains, curves)
    lines = [f'판정: {decision["decision"]}', '',
             'K는 A_fit의 사전 지정 K_tau(0.9)를 그대로 사용했다. A60 K와 V 결과로 다시 선택하지 않았다.', '',
             '```text', 'dataset                 K    n   median     mean       Q1         Q3        Raw       loss       status']
    def fmt(value):
        return 'NA' if value is None else f'{value:.6f}'
    for row in summaries:
        lines.append(f'{row["dataset"]:23} {str(row["K_main"]):>3} {row["n"]:4} '
            + ' '.join(f'{fmt(row[name]):>10}' for name in ('median','mean','Q1','Q3','Raw','loss'))
            + f' {row["status"]} {row["limitation"]}')
    lines.extend(['```', '', 'ETTh2_safe17400은 별도 기술 결과이며 세 데이터의 판정에는 포함하지 않는다.',
        '손해는 첫 두 seed 평균과 기존 세 후보 및 K_main 중 최솟값의 차이다. 현재 comparison 행이 있으면 그 후보 결과를 우선한다.',
        'LIMITATION은 손해가 K_main seed 분포의 IQR보다 엄격히 클 때 표시한다.', '',
        'V는 반복 개발 평가이며 독립 테스트가 아니다. 두 seed 최소–최대 막대는 신뢰구간이 아니다.',
        '상자그림은 linear 사분위수와 1.5 IQR 수염을 쓰며 관측한 seed를 모두 표시한다.',
        '과거 T 노출, 사전학습 중복 여부 미확인, 임시 합의 기준의 한계가 남는다. 부정적인 결과도 그대로 유지했다.', '',
        f'중단 이유: {stop_reason or "없음"}', f'누적 경과 시간(초): {elapsed_seconds if elapsed_seconds is not None else "미기록"}',
        '불완전한 결과에서는 ADOPT 판정을 내리지 않는다.' if not complete else '사전 지정한 네 데이터의 seed와 비교 결과가 완비됐다.', '',
        '그림: FIG_K_CURVE.png / FIG_K_CURVE.svg / FIG_BOXPLOT.png / FIG_BOXPLOT.svg'])
    (root/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return decision
