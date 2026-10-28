"""Verify saved V arrays and report the registered Jena branch contrasts."""
from datetime import datetime, timezone
import csv
import io
import json
from pathlib import Path

import numpy as np

from . import runtime, run_one


ORDER = ('A1', 'P_TIME_ONLY', 'P_CHANNEL_ONLY', 'OURS', 'A2')
LABELS = dict(A1='No P', P_TIME_ONLY='Time branch only', P_CHANNEL_ONLY='Channel branch only',
              OURS='Full P', A2='A2: large channel replacement')


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=keys)
    writer.writeheader()
    writer.writerows(rows)
    temporary = path.with_suffix('.csv.tmp')
    temporary.write_text(stream.getvalue(), encoding='utf-8-sig')
    temporary.replace(path)


def normalized_row(row):
    out = dict(row)
    for name in ('seed', 'K', 'trainable_parameters', 'normalization_fit_end'):
        out[name] = int(float(out[name]))
    for name in ('MSE', 'MAE', 'selected_epoch', 'training_cap'):
        out[name] = float(out[name])
    for name in ('extended', 'reused'):
        out[name] = str(out.get(name, False)).lower() == 'true'
    out['display_name'] = LABELS[out['method_id']]
    return out


def verify_arrays(row, cohort):
    path = Path(row['prediction_path'])
    if runtime.sha(path) != row['prediction_sha256']:
        raise ValueError('Prediction archive changed: ' + row['job_id'])
    with np.load(path, allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in ('pred_normalized', 'target_normalized', 'observed',
                                        'origins', 'channel_ids')}
        for key in ('mean', 'std'):
            if key in archive.files:
                arrays[key] = archive[key]
            elif row['job_id'] not in ('HISTORICAL_jena_K6_s92601', 'HISTORICAL_jena_K6_s92602'):
                raise ValueError('Missing normalization metadata outside two known legacy archives')
    p, y, mask = arrays['pred_normalized'], arrays['target_normalized'], arrays['observed'].astype(bool)
    if p.shape != (43, 48, 21) or y.shape != p.shape or mask.shape != p.shape:
        raise ValueError('Unexpected Jena V shape')
    if int(arrays['origins'].max()) + 48 >= 16772:
        raise ValueError('V target overlaps protected T')
    if cohort is not None:
        for key in ('origins', 'channel_ids', 'mean', 'std', 'target_normalized', 'observed'):
            if key not in arrays or key not in cohort:
                continue
            if not np.array_equal(arrays[key], cohort[key], equal_nan=arrays[key].dtype.kind in 'fc'):
                raise ValueError('Reference cohort/scaler differs: ' + key)
    difference = p.astype(np.float64) - y.astype(np.float64)
    if not np.isfinite(difference[mask]).all() or not mask.all():
        raise ValueError('Expected finite, fully observed registered Jena cohort')
    mse, mae = float(np.mean(difference[mask] ** 2)), float(np.mean(np.abs(difference[mask])))
    if max(abs(mse - row['MSE']), abs(mae - row['MAE'])) > 1e-9:
        raise ValueError('Independent saved-array metric mismatch')
    return arrays, dict(job_id=row['job_id'], sha256=row['prediction_sha256'],
                        shape=list(p.shape), MSE=mse, MAE=mae,
                        normalization_arrays_present='mean' in arrays and 'std' in arrays,
                        legacy_normalization_check='Exact normalized target/cohort equality; original provenance for historical seeds 92601/92602.', status='PASS')


def public_hashes(paths):
    return {path.resolve().relative_to(runtime.REPO.resolve()).as_posix(): runtime.sha(path)
            for path in paths}


def verify_job_evidence(job, registration, completion):
    job_id = job['job_id']
    root = runtime.REPORT / 'jobs' / job_id
    result_path = root / 'RESULT.json'
    statuses = completion['run_statuses']
    result = runtime.read(result_path)
    spec = run_one.build_spec(job)
    if (statuses.get(job_id) != 'COMPLETE' or result.get('status') != 'COMPLETE'
            or result.get('job') != job or result.get('registration') != registration
            or result.get('spec') != spec or result.get('selection_sealed_before_V') is not True):
        raise ValueError('Completed result differs from sealed job/controller: ' + job_id)
    attempt = result['attempt']
    if type(attempt) is not int or attempt not in (1, 2):
        raise ValueError('Result has an unregistered attempt: ' + job_id)
    public = root / f'attempt_{attempt:03d}'
    required = [result_path, root / f'WORKER_RECEIPT_attempt{attempt}.json']
    required += [public / name for name in ('START.json', 'SOURCE_CONTRACT.json', 'SEARCH_RESULT.json',
                                           'SELECTION.json', 'REFIT_SUMMARY.json', 'RECEIPT.json',
                                           'OPENED_FILES.json')]
    if any(not path.is_file() for path in required):
        raise ValueError('Completed job is missing public evidence: ' + job_id)
    start = runtime.read(public / 'START.json')
    receipt = runtime.read(public / 'RECEIPT.json')
    worker = runtime.read(required[1])
    if (start.get('job') != job or start.get('attempt') != attempt
            or start.get('registration') != registration or start.get('spec') != spec):
        raise ValueError('Start identity differs from completed job: ' + job_id)
    if (receipt.get('status') != 'COMPLETE' or receipt.get('job') != job
            or receipt.get('attempt') != attempt or receipt.get('registration') != registration
            or receipt.get('training_attempted') is not True or receipt.get('training_executed') is not True
            or receipt.get('T_numeric_values_parsed') != 0
            or any(receipt.get(key, {}).get('changed') != [] for key in ('preserved_before', 'preserved_after'))):
        raise ValueError('Successful attempt receipt is invalid: ' + job_id)
    if (worker.get('job_id') != job_id or worker.get('attempt') != attempt
            or worker.get('status') != 'COMPLETE' or worker.get('exit_code') != 0):
        raise ValueError('Successful worker receipt is invalid: ' + job_id)
    recorded_attempts = [r for r in completion.get('attempts', {}).get(job_id, [])
                         if r.get('attempt') == attempt and 'exit_code' in r]
    if any(r['exit_code'] != 0 or r.get('job_id') != job_id for r in recorded_attempts):
        raise ValueError('Controller attempt differs from successful worker: ' + job_id)
    if runtime.read(public / 'OPENED_FILES.json').get('T_numeric_values_parsed') != 0:
        raise ValueError('Opened-file audit does not confirm protected T exclusion: ' + job_id)
    runtime.assert_protected_T(result)
    source = runtime.read(public / 'SOURCE_CONTRACT.json')
    contract = result['contract']
    if (source.get('spec') != spec or not source.get('sources')
            or any(source.get(key) != contract.get(key) for key in ('source_sha', 'config_sha', 'data_sha', 'model_sha'))):
        raise ValueError('Source contract differs from completed job: ' + job_id)
    selection_path = public / 'SELECTION.json'
    if Path(result['selection_path']).resolve() != selection_path.resolve():
        raise ValueError('Result points outside its successful selection attempt: ' + job_id)
    run_one.check_selection(selection_path, result['selection_sha256'], job)
    selection = runtime.read(selection_path)
    search = runtime.read(public / 'SEARCH_RESULT.json')
    selected, selected_mse = run_one.choose_epoch(search, spec['epochs'])
    if (selection.get('spec') != spec or selection.get('contract') != contract
            or selection.get('selection_split') != 'A_stop' or selection.get('curves') != search['curves']
            or selection.get('selected_epoch') != selected or selection.get('selected_astop_mse') != selected_mse
            or result.get('search_summary') != search['summary']):
        raise ValueError('Selection evidence differs from registered A_stop search: ' + job_id)
    refit = runtime.read(public / 'REFIT_SUMMARY.json')
    if refit.get('status') != 'SEGMENT_COMPLETE' or result.get('refit_summary') != refit:
        raise ValueError('Fresh refit evidence is incomplete: ' + job_id)
    row = result['row']
    if (any(row.get(key) != value for key, value in job.items()) or row.get('reused') is not False
            or row.get('selected_epoch') != selected or row.get('training_cap') != spec['epochs'][-1]
            or row.get('cap_selected') != (selected == spec['epochs'][-1])):
        raise ValueError('Result row differs from job or sealed selection: ' + job_id)
    if runtime.sha(result['bundle_path']) != result['bundle_sha256']:
        raise ValueError('Selected adapter bundle changed: ' + job_id)
    if runtime.sha(row['prediction_path']) != row['prediction_sha256']:
        raise ValueError('Completed prediction archive changed: ' + job_id)
    return result, public_hashes(required)


def verify_completion_evidence(plan, completion):
    if completion.get('status') != 'TRAINING_COMPLETE':
        raise ValueError('Base training is not complete; final report cannot imply completion')
    for key, extended in (('jobs', False), ('extension_jobs', True)):
        jobs = plan[key]
        if len(jobs) != 20:
            raise ValueError('RUN_PLAN must contain exactly twenty ' + key)
        for job in jobs:
            run_one.validate_job(job)
            if job['extended'] is not extended:
                raise ValueError('RUN_PLAN extension flag differs')
        for method in run_one.METHOD_FAMILY:
            if sorted(j['seed'] for j in jobs if j['method_id'] == method) != list(range(92601, 92611)):
                raise ValueError('RUN_PLAN is missing or duplicates a registered seed: ' + method)
    statuses = completion['run_statuses']
    for job in plan['jobs']:
        path = runtime.REPORT / 'jobs' / job['job_id'] / 'RESULT.json'
        if not path.is_file() or statuses.get(job['job_id']) != 'COMPLETE':
            raise ValueError('Missing completed base job: ' + job['job_id'])

    registration = run_one.validate_registration(plan['jobs'][0])
    results, verified_files = {}, {}
    for job in plan['jobs'] + plan['extension_jobs']:
        job_id = job['job_id']
        root = runtime.REPORT / 'jobs' / job_id
        result_path = root / 'RESULT.json'
        if not result_path.is_file():
            if statuses.get(job_id) == 'COMPLETE':
                raise ValueError('Controller completion has no result: ' + job_id)
            continue
        result, hashes = verify_job_evidence(job, registration, completion)
        results[job_id] = result
        verified_files.update(hashes)
    for method in run_one.METHOD_FAMILY:
        completed = all(j['job_id'] in results for j in plan['extension_jobs'] if j['method_id'] == method)
        if (completion['extension_status'][method]['status'] == 'COMPLETE') != completed:
            raise ValueError('Extension family completion differs from verified results: ' + method)
    base_complete = sum(j['job_id'] in results for j in plan['jobs'])
    return results, base_complete, verified_files


def main():
    completion = runtime.read(runtime.REPORT / 'EXECUTION_COMPLETION.json')
    if completion['status'] != 'TRAINING_COMPLETE':
        raise ValueError('Base training is not complete; final report cannot imply completion')
    preservation = runtime.guard_preserved()
    plan = runtime.read(runtime.REPORT / 'RUN_PLAN.json')
    verified_results, base_complete, verified_files = verify_completion_evidence(plan, completion)
    references = runtime.read(runtime.REPORT / 'REFERENCE_RUNS.json')
    rows = [normalized_row(dict(row, reused=True)) for row in references['rows']]
    all_new, primary_new = [], []
    for method in ('P_TIME_ONLY', 'P_CHANNEL_ONLY'):
        basic = [j for j in plan['jobs'] if j['method_id'] == method]
        extended = [j for j in plan['extension_jobs'] if j['method_id'] == method]
        completed_extensions = all(j['job_id'] in verified_results for j in extended)
        selected_ids = {j['job_id'] for j in (extended if completed_extensions else basic)}
        for job in basic + extended:
            if job['job_id'] not in verified_results:
                continue
            result = verified_results[job['job_id']]
            row = normalized_row(result['row'])
            row['primary'] = job['job_id'] in selected_ids
            row['provisional'] = completion['extension_status'][method]['status'] in ('NOT_RUN_BUDGET', 'INCOMPLETE', 'BASE_INCOMPLETE') or bool(completion['extension_status'][method].get('undertrained_at_10'))
            all_new.append(row)
            if row['primary']:
                primary_new.append(row)
    rows += primary_new
    checks = []
    canonical = next(r for r in rows if r['method_id'] == 'OURS' and r['seed'] == 92603)
    cohort, _ = verify_arrays(canonical, None)
    for row in rows + [r for r in all_new if not r['primary']]:
        arrays, check = verify_arrays(row, cohort)
        if cohort is None:
            cohort = arrays
        checks.append(check)
    summaries = []
    for method in ORDER:
        group = sorted((r for r in rows if r['method_id'] == method), key=lambda r: r['seed'])
        if [r['seed'] for r in group] != list(range(92601, 92611)):
            raise ValueError('Primary group must contain exactly ten registered seeds: ' + method)
        mse = np.array([r['MSE'] for r in group])
        mae = np.array([r['MAE'] for r in group])
        q1, median, q3 = np.quantile(mse, [.25, .5, .75])
        summaries.append(dict(method_id=method, display_name=LABELS[method], n=10,
            MSE_median=float(median), MSE_Q1=float(q1), MSE_Q3=float(q3), MSE_IQR=float(q3-q1),
            MAE_median=float(np.median(mae)), parameters=group[0]['trainable_parameters'],
            cap=group[0]['training_cap'], extended=group[0]['extended'],
            selected_epochs=json.dumps([r['selected_epoch'] for r in group]),
            provisional=any(r.get('provisional', False) for r in group)))
    lookup = {s['method_id']: s for s in summaries}
    ours = {r['seed']: r for r in rows if r['method_id'] == 'OURS'}
    contrasts, paired = [], []
    for method in ('P_TIME_ONLY', 'P_CHANNEL_ONLY', 'A1', 'A2'):
        s, ref = lookup[method], lookup['OURS']
        gap = s['MSE_median'] - ref['MSE_median']
        spread = max(s['MSE_IQR'], ref['MSE_IQR'])
        decision = 'FULL_P_LOWER' if gap > spread else 'VARIANT_LOWER' if gap < -spread else 'UNCLEAR'
        if s['provisional']:
            decision += '_PROVISIONAL'
        differences = []
        for row in sorted((r for r in rows if r['method_id'] == method), key=lambda r: r['seed']):
            delta = row['MSE'] - ours[row['seed']]['MSE']
            differences.append(delta)
            paired.append(dict(method_id=method, seed=row['seed'], variant_MSE=row['MSE'],
                               full_P_MSE=ours[row['seed']]['MSE'], delta_MSE=delta))
        contrasts.append(dict(method_id=method, contrast=f'{LABELS[method]} minus Full P',
            delta_median_MSE=gap, effect_pct=100*gap/ref['MSE_median'], spread=spread,
            descriptive_decision=decision, paired_delta_median=float(np.median(differences)),
            seeds_full_P_lower=sum(d > 0 for d in differences), n=10,
            inference='Descriptive exploratory V comparison; no significance/equivalence claim.'))
    write_csv(runtime.REPORT / 'RUNS.csv', rows)
    write_csv(runtime.REPORT / 'ALL_NEW_RUNS.csv', all_new)
    write_csv(runtime.REPORT / 'SUMMARY.csv', summaries)
    write_csv(runtime.REPORT / 'CONTRASTS.csv', contrasts)
    write_csv(runtime.REPORT / 'PAIRED_DIFFERENCES.csv', paired)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={'width_ratios': [1.65, 1]})
    groups = [[r['MSE'] for r in sorted(rows, key=lambda x: x['seed']) if r['method_id'] == m] for m in ORDER]
    boxes = axes[0].boxplot(groups, positions=np.arange(5), widths=.48, patch_artist=True, showfliers=False)
    colors = ['#b7bdc5', '#63a6c8', '#94c5b1', '#9b6caa', '#d7b27b']
    for i, (values, patch) in enumerate(zip(groups, boxes['boxes'])):
        patch.set_facecolor(colors[i])
        axes[0].scatter(i + np.linspace(-.14, .14, len(values)), values, s=23, color='#333333', alpha=.75, zorder=3)
    short_names = ['No P', 'Time only', 'Channel only', 'Full P', 'A2']
    tick_labels = []
    for method, short_name in zip(ORDER, short_names):
        summary = lookup[method]
        label = f'{short_name}\ncap {summary["cap"]:g}'
        if summary['extended']:
            label += ' (extended)'
        if summary['provisional']:
            label += '\nPROVISIONAL'
        tick_labels.append(label)
    axes[0].set_xticks(range(5), tick_labels, fontsize=9)
    axes[0].set_ylabel('Normalized V MSE (lower is better)')
    axes[0].set_title('Jena: P branch ablation, 10 seeds')
    axes[0].grid(axis='y', alpha=.2)
    axes[1].barh(np.arange(5), [lookup[m]['parameters'] for m in ORDER], color=colors)
    axes[1].set_yticks(range(5), short_names)
    axes[1].invert_yaxis()
    axes[1].set_xlabel('Trainable adapter parameters')
    axes[1].set_xlim(0, 44000)
    for i, m in enumerate(ORDER):
        count = lookup[m]['parameters']
        axes[1].text(count+350, i, f'{count:,}', va='center', fontsize=9)
    axes[1].set_title('Unequal branch capacities')
    fig.suptitle('Exploratory development-V results; joint LayerNorm and E/D mixing retained', fontsize=10)
    fig.text(.5, .015, 'Cap = maximum searched epoch, not selected epoch. A2 provides parameter-count context, not full capacity/compute control.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, .96))
    for suffix in ('png', 'svg', 'pdf'):
        fig.savefig(runtime.REPORT / f'FIG_JENA_P_BRANCHES.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    lines = ['# Jena P 분기 결과', '', '개발 검증 V의 탐색 결과다. 기존 P 기여를 본 뒤 Jena를 선택했으며 보호 T는 평가하지 않았다.', '',
             f'기본 실행 {sum(not r["extended"] for r in all_new)}/20개, 추가 연장 {sum(r["extended"] for r in all_new)}개 완료. 연장 10개 전체가 끝난 경우에만 해당 방법의 주 결과로 사용했다.', '',
             '```text', 'Method                    n      median MSE      IQR       parameters']
    for s in summaries:
        lines.append(f'{s["display_name"]:<28} {s["n"]:>2}     {s["MSE_median"]:.6f}    {s["MSE_IQR"]:.6f}    {s["parameters"]:>6}')
    lines += ['```', '']
    for c in contrasts:
        lines.append(f'- {LABELS[c["method_id"]]}: Full P 대비 MSE {c["effect_pct"]:+.2f}%, 기술적 판정 {c["descriptive_decision"]}; seed별 차이 중앙값 {c["paired_delta_median"]:+.6f}.')
    lines += ['', '양수는 해당 변형의 오차가 더 크다는 뜻이다. 원 중앙값/max-IQR 규칙은 유의성 또는 동등성 검정이 아니다.', '',
              '해석 범위: 현재 구현에서 분기를 제거하는 효과다. 시간 분기와 채널 분기의 용량이 다르고 joint LayerNorm/E/D 결합이 남으므로 시간축 자체의 우월성, 완전한 채널 독립성, 다른 데이터에서의 효과를 주장하지 않는다.', '',
              '기존 C2/C3 비교의 잠정 상태는 이번 실험으로 해소되지 않는다. No-P는 기존 cap10 연장, Full-P/A2는 기존 cap3 결과이며, 새 변형의 일정과 상한 상태는 SUMMARY.csv와 EXECUTION_COMPLETION.json에 공개한다.', '',
              '그림: FIG_JENA_P_BRANCHES.png / .svg / .pdf. 모든 seed 점을 표시했다.', '']
    (runtime.REPORT / 'RESULTS_KO.md').write_text('\n'.join(lines), encoding='utf8')
    public_outputs = ['RUNS.csv', 'ALL_NEW_RUNS.csv', 'SUMMARY.csv', 'CONTRASTS.csv',
                      'PAIRED_DIFFERENCES.csv', 'RESULTS_KO.md',
                      'FIG_JENA_P_BRANCHES.png', 'FIG_JENA_P_BRANCHES.svg', 'FIG_JENA_P_BRANCHES.pdf']
    input_names = ['REGISTRATION_SEAL.json', 'PREREGISTRATION.json', 'PREREGISTRATION.md',
                   'RUN_PLAN.json', 'PRESERVATION_BASELINE.json', 'REFERENCE_RUNS.json',
                   'EXECUTION_COMPLETION.json']
    verified_files.update(public_hashes(runtime.REPORT / name for name in public_outputs + input_names))
    output = dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(),
                  base_complete=base_complete, new_result_count=len(all_new), primary_rows=len(rows),
                  verified_files=verified_files,
                  checked_prediction_archives=len(checks), independent_FP64_checks=checks,
                  preservation=preservation, T_numeric_values_parsed=0,
                  summary=summaries, contrasts=contrasts,
                  inputs={'registration_seal_sha256': runtime.sha(runtime.REPORT / 'REGISTRATION_SEAL.json'),
                          'completion_sha256': runtime.sha(runtime.REPORT / 'EXECUTION_COMPLETION.json')})
    runtime.atomic(runtime.REPORT / 'ANALYSIS_VALIDATION.json', output)
    print(json.dumps(dict(status='PASS', base_complete=base_complete, total_new=len(all_new), primary_rows=len(rows))))


if __name__ == '__main__':
    main()
