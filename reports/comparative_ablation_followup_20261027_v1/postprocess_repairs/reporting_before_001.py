"""Read-only followup analysis, fixed prediction checks, and sealed reports."""
import argparse
import csv
from decimal import Decimal
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

sys.dont_write_bytecode = True
from research.comparative_ablation_20261025_v1 import analysis
from . import common


METHODS = ('OURS', 'A1', 'A2', 'A4')
SEEDS = set(range(92601, 92611))


def _put(path, payload):
    path = Path(path)
    payload = payload.encode('utf8') if isinstance(payload, str) else payload
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError('Sealed reporting artifact differs: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload)


def _json(path, value):
    _put(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n')


def _csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    _put(path, stream.getvalue())


def _read_csv(path):
    with Path(path).open(encoding='utf8', newline='') as stream:
        return list(csv.DictReader(stream))


def _bool(value):
    return value is True or value == 'True'


def _number(value):
    return None if value is None or value == '' else float(value)


def _fmt(value, digits=6):
    return 'NA' if value is None else f'{float(value):.{digits}f}'


def _label(row):
    label = row.get('decision_label', row['decision'])
    if row.get('provisional') and label != 'INCOMPLETE' and not label.endswith('_PROVISIONAL'):
        label += '_PROVISIONAL'
    return label


def primary_figure_rows(rows):
    """Use only the primary flags produced by the frozen registered analysis."""
    selected = []
    for row in rows:
        if row['method_id'] in (*METHODS, 'RAW') and _bool(row['is_primary']):
            selected.append(dict(dataset=row['dataset'], method_id=row['method_id'],
                seed=int(row['seed']), job_id=row['job_id'], MSE=float(row['MSE']),
                extended=_bool(row['extended'])))
    for dataset in analysis.DATASETS:
        for method in (*METHODS, 'RAW'):
            group = [r for r in selected if (r['dataset'], r['method_id']) == (dataset, method)]
            if len(group) != len({r['seed'] for r in group}):
                raise ValueError('Duplicate seed in updated figure: ' + dataset + '/' + method)
    return sorted(selected, key=lambda r: (analysis.DATASETS.index(r['dataset']), r['method_id'], r['seed']))


def prediction_checks(summaries, floor_rows):
    """Apply the planner's fixed descriptive thresholds, never decision rules."""
    lookup = {(r['dataset'], r['method_id']): r for r in summaries}
    checks = []
    for dataset in analysis.DATASETS:
        ours, variant = lookup.get((dataset, 'OURS'), {}), lookup.get((dataset, 'A4'), {})
        n, n_ours = int(variant.get('n', 0)), int(ours.get('n', 0))
        base, value = _number(ours.get('MSE_median')), _number(variant.get('MSE_median'))
        complete = n == n_ours == 10 and _bool(ours.get('complete')) and _bool(variant.get('complete'))
        # Decimal receipt values preserve exact >=50 and strict <10 boundaries.
        effect_decimal = (Decimal(100) * (Decimal(str(value)) - Decimal(str(base))) /
                          Decimal(str(base))) if base and value is not None else None
        effect = float(effect_decimal) if effect_decimal is not None else None
        ett = dataset.startswith('ETTh')
        matches = (effect_decimal >= 50 if ett else abs(effect_decimal) < 10) if complete and effect_decimal is not None else None
        checks.append(dict(kind='A4_EFFECT', dataset=dataset, method_id='A4', n=n,
            n_ours=n_ours, expected_n=10, effect_pct=effect,
            fixed_prediction='effect_pct >= 50' if ett else 'abs(effect_pct) < 10',
            status='INCOMPLETE' if matches is None else 'MATCH' if matches else 'NOT_MATCH',
            provisional=_bool(ours.get('provisional')) or _bool(variant.get('provisional'))))
    for dataset in ('ETTh1', 'ETTh2_safe17400'):
        for method in ('A4', 'C2', 'C3'):
            group = [r for r in floor_rows if r['dataset'] == dataset and r['method_id'] == method
                     and _bool(r.get('is_primary', True)) and r['status'] == 'PASS']
            seeds = [int(r['seed']) for r in group]
            if len(seeds) != len(set(seeds)):
                raise ValueError('Duplicate primary seed in D1 prediction check')
            ratios = [float(r['floor_fraction']) for r in group]
            if not all(np.isfinite(ratios)):
                raise ValueError('Nonfinite D1 ratio')
            complete = set(seeds) == SEEDS
            median = float(np.median(ratios)) if ratios else None
            checks.append(dict(kind='D1_RATIO', dataset=dataset, method_id=method, n=len(group),
                expected_n=10, median_per_job_floor_fraction=median,
                fixed_prediction='median(per-job floor_MSE / actual_same_cohort_MSE) >= 0.5',
                aggregation='Median of primary per-job ratios; not ratio of the two medians.',
                status='INCOMPLETE' if not complete else 'MATCH' if median >= .5 else 'NOT_MATCH'))
    return checks


def _prepare_matplotlib():
    config = common.CACHE / 'matplotlib'
    config.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(config)
    import matplotlib
    matplotlib.use('Agg')
    # Frozen analysis resets the environment later; resolve both caches first.
    from matplotlib import font_manager
    font_manager.findfont('DejaVu Sans')
    if Path(matplotlib.get_cachedir()).resolve() != config.resolve():
        raise ValueError('Matplotlib was initialized outside the followup cache')
    return matplotlib


def _figure(output, rows):
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'svg.hashsalt': common.NAME, 'font.size': 11,
                                'font.family': 'DejaVu Sans'})
    selected = primary_figure_rows(rows)
    counts = {(d, m): len([r for r in selected if (r['dataset'], r['method_id']) == (d, m)])
              for d in analysis.DATASETS for m in (*METHODS, 'RAW')}
    exported = [dict(r, n=counts[r['dataset'], r['method_id']],
                     expected_n=1 if r['method_id'] == 'RAW' else 10,
                     missing_n=(1 if r['method_id'] == 'RAW' else 10) - counts[r['dataset'], r['method_id']])
                for r in selected]
    for dataset in analysis.DATASETS:
        for method in (*METHODS, 'RAW'):
            if counts[dataset, method] == 0:
                exported.append(dict(dataset=dataset, method_id=method, seed='', job_id='',
                    MSE='', extended='', n=0, expected_n=1 if method == 'RAW' else 10,
                    missing_n=1 if method == 'RAW' else 10))
    _csv(output / 'FIG_ABLATION_UPDATED.csv', exported)
    colors = ('#0072B2', '#E69F00', '#009E73', '#CC79A7')
    markers = ('o', 's', '^', 'D')
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6))
    for panel, ax, dataset in zip('ABCD', axes.flat, analysis.DATASETS):
        labels = []
        for position, (method, color, marker) in enumerate(zip(METHODS, colors, markers)):
            group = sorted([r for r in selected if (r['dataset'], r['method_id']) == (dataset, method)],
                           key=lambda r: r['seed'])
            values = [r['MSE'] for r in group]
            extended = ' (ext10)' if group and all(r['extended'] for r in group) else ''
            labels.append(f'{method}{extended}\nn={len(values)}/10' + (f'; missing={10-len(values)}' if len(values) < 10 else ''))
            if len(values) > 1:
                ax.boxplot([values], positions=[position], widths=.48, manage_ticks=False,
                    patch_artist=True, showfliers=False, boxprops=dict(facecolor=color, alpha=.28),
                    medianprops=dict(color='#111111', linewidth=1.6))
            if values:
                # Fixed seed positions make all ten observations visible and reproducible.
                offsets = [(r['seed'] - 92605.5) * .028 for r in group]
                ax.scatter(position + np.asarray(offsets), values, color=color, marker=marker,
                           edgecolors='#222222', linewidths=.4, s=30, zorder=3)
        raw = next((r['MSE'] for r in selected if r['dataset'] == dataset and r['method_id'] == 'RAW'), None)
        if raw is not None:
            ax.axhline(raw, color='#555555', linestyle='--', linewidth=1.3, label=f'Raw = {raw:.4f}')
            ax.legend(frameon=False, fontsize=9, loc='best')
        ax.set_xticks(range(len(METHODS)), labels)
        ax.set_title(f'{panel}  {dataset}', loc='left', fontweight='bold')
        ax.set_ylabel('V MSE (A60-normalized)')
        ax.set_xlim(-.6, len(METHODS)-.4)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.18)
        ax.tick_params(axis='x', labelsize=9)
    fig.subplots_adjust(left=.075, right=.985, bottom=.13, top=.95, hspace=.38, wspace=.24)
    fig.text(.5, .025, 'Box: Q1–Q3; line: median; whiskers: 1.5×IQR; points: individual fixed seeds.\n'
             'Primary extension is used only when all 10 seeds are complete. V is development validation; T remains protected.',
             ha='center', fontsize=10)
    for suffix in ('png', 'svg'):
        buffer = io.BytesIO()
        metadata = {'Date': None} if suffix == 'svg' else {'Software': common.NAME}
        fig.savefig(buffer, format=suffix, dpi=300, metadata=metadata)
        _put(output / f'FIG_ABLATION_UPDATED.{suffix}', buffer.getvalue())
    plt.close(fig)


def _prediction_document(checks):
    text = ['# Fixed planner prediction check', '',
        '[확인] 아래 기준은 후속 지시문 §5의 사전 고정 예측이며 등록 판정을 바꾸지 않는다.',
        'A4 effect(%) = 100 × (A4 median − OURS median) / OURS median.',
        'ETTh1·ETTh2: effect ≥ +50%. Jena·Appliances: |effect| < 10% (경계 10%는 불일치).',
        'D1: ETTh1·ETTh2에서 A4·C2·C3 각각의 **job별 하한/같은 대상점 실제 MSE 비율의 중앙값** ≥ 0.5.',
        'D1 집계는 두 중앙값의 비율이 아니다. 원문은 집계 방식을 명시하지 않아 데이터별 중앙값 보고 규칙에 맞췄다.',
        'n<10 또는 정해진 seed가 빠지면 INCOMPLETE이며 예측의 성공·실패를 주장하지 않는다.', '',
        '```text', 'kind       dataset              method n/10 value           prediction status']
    for row in checks:
        value = row.get('effect_pct') if row['kind'] == 'A4_EFFECT' else row.get('median_per_job_floor_fraction')
        text.append(f"{row['kind']:<11}{row['dataset']:<21}{row['method_id']:<7}{row['n']:<5}{_fmt(value):<16}{row['status']}")
    text.extend(['```', '', '[추정] D1의 큰 비율은 출력 공간 제약과 수준 이동 설명에 부합할 수 있으나 '
                 '마지막 관측값·lag의 개별 인과 효과를 식별하지 않는다.', ''])
    return '\n'.join(text)


def _documents(output, summaries, decision, checks, floor_data, status, results, stage):
    lookup = {(r['dataset'], r['method_id']): r for r in summaries}
    ablation = {(r['dataset'], r['method_id']): r for r in decision['ablation']}
    slide = ['# Slide numbers updated', '',
             'V is development validation; T was not evaluated. Registered median/IQR decisions are unchanged.', '',
             '## Slide 16 — P ablation', '']
    for dataset in analysis.DATASETS:
        variant, ours = lookup[dataset, 'A1'], lookup[dataset, 'OURS']
        effect = _number(ablation[dataset, 'A1']['effect'])
        pct = 100 * effect / float(ours['MSE_median']) if effect is not None else None
        slide.append(f"- {dataset}: OURS={_fmt(ours['MSE_median'])}, A1={_fmt(variant['MSE_median'])}, "
                     f"effect={_fmt(pct, 2)}%, n={variant['n']}/10, extended={variant['extended']}, "
                     f"undertrained_at_10={variant['undertrained_at_10']}. "
                     f"English verdict: P contribution verdict = {_label(ablation[dataset, 'A1'])}; A1 removes P.")
    slide.extend(['', '## Slide 18 — Whole D ablation and output-space floor', ''])
    for dataset in analysis.DATASETS:
        variant, ours = lookup[dataset, 'A4'], lookup[dataset, 'OURS']
        prediction = next(r for r in checks if r['kind'] == 'A4_EFFECT' and r['dataset'] == dataset)
        floor = next((r for r in floor_data['d1_summary'] if (r['dataset'], r['method_id']) == (dataset, 'A4')), {})
        slide.append(f"- {dataset}: OURS={_fmt(ours['MSE_median'])}, A4={_fmt(variant['MSE_median'])}, "
                     f"effect={_fmt(prediction['effect_pct'], 2)}%, n={variant['n']}/10, "
                     f"D1 median floor/actual={_fmt(floor.get('floor_fraction_median'), 3)} "
                     f"(n={floor.get('n', 0)}/10). English verdict: Whole-D contribution verdict = "
                     f"{_label(ablation[dataset, 'A4'])}; last-value and lag effects remain inseparable.")
    slide.extend(['', 'Primary aggregate: jena, ETTh1, appliances; ETTh2_safe17400 is reported separately.',
                  'Comparison verdicts remain provisional where required C2/C3 extensions are out of scope.',
                  'Followup execution scope was selected after inspecting the first campaign results.', ''])
    _put(output / 'SLIDE_NUMBERS_UPDATED.md', '\n'.join(slide))
    new_results = [r for r in results if r.get('job', {}).get('method_id') == 'A4' or
        (r.get('job', {}).get('method_id') == 'A1' and r['job'].get('dataset') == 'jena' and r['job'].get('extended'))]
    # Some public receipts identify jobs through row rather than job.
    if not new_results:
        new_results = [r for r in results if r.get('row', {}).get('method_id') == 'A4' or
            (r.get('row', {}).get('method_id') == 'A1' and r['row'].get('dataset') == 'jena' and r['row'].get('extended'))]
    complete_new = sum(r.get('status') == 'COMPLETE' for r in new_results)
    text = ['# 비교·절제 후속 실험 결과', '',
        f'[확인] 단계={stage}; 수집 RESULT.json={len(results)}개; 후속 완료={complete_new}/50개. '
        '원 RAW 4개·OURS 40개는 등록 G1_REUSE_AUDIT를 통해 별도로 재사용한다.',
        f"[확인] 실행 상태={status.get('status', 'UNSPECIFIED')}; 중단 이유={status.get('stop_reason') or '없음/미기록'}.",
        f"[확인] 첫 새 학습부터 경과={_fmt(_number(status.get('elapsed_seconds')), 1)}초 "
        f"({_fmt(_number(status.get('elapsed_seconds')) / 3600 if status.get('elapsed_seconds') is not None else None, 2)}시간); "
        f"시작 허용 예산 잔여={_fmt(_number(status.get('remaining_seconds')), 1)}초. "
        '한도는 실제 경과 13시간 또는 2026-10-28 00:00 KST이며 진행 중 job만 마친다.',
        '[확인] 입력 스냅샷은 REPORTING_INPUTS.json, 무수정 등록 분석은 registered_analysis/, '
        '판정은 DECISION.json·ABLATION.csv·COMPARATIVE.csv에 보존한다.', '',
        '```text', 'dataset              method n/10 median V MSE IQR        effect%   decision / primary / cap']
    for dataset in analysis.DATASETS:
        for method in ('A1', 'A4'):
            row, verdict = lookup[dataset, method], ablation[dataset, method]
            effect = _number(verdict['effect'])
            pct = 100 * effect / float(lookup[dataset, 'OURS']['MSE_median']) if effect is not None else None
            text.append(f"{dataset:<21}{method:<7}{str(row['n'])+'/10':<6}{_fmt(row['MSE_median']):<13}"
                        f"{_fmt(row['MSE_IQR']):<11}{_fmt(pct, 2):<10}{_label(verdict)}; "
                        f"extended={row['extended']}; cap={row['cap_selected_count']}; "
                        f"undertrained_at_10={row['undertrained_at_10']}")
    text.extend(['```', '', '주 집계 (jena·ETTh1·appliances, 잠정 판정도 포함하므로 확정 기여 수가 아님):', '', '```text',
                 'method  CONTRIBUTES HARMS UNCLEAR INCOMPLETE provisional'])
    for method in ('A1', 'A4'):
        counts = decision['ablation_primary_counts'][method]
        provisional = sum(r['method_id'] == method and r['in_primary_aggregate'] and r['provisional']
                          for r in decision['ablation'])
        text.append(f"{method:<8}{counts['CONTRIBUTES']:<12}{counts['HARMS']:<6}{counts['UNCLEAR']:<8}"
                    f"{counts['INCOMPLETE']:<11}{provisional}")
    text.extend(['```', '', '[확인] effect(%)는 variant와 OURS 중앙값 차이를 OURS 중앙값으로 나눈 값이다. '
                 '등록 절제 판정은 절대 effect와 두 IQR 중 큰 spread를 비교한다. '
                 '예측의 +50%·10% 기준으로 판정을 바꾸지 않는다.', '', 'D0 고정 PCA 하한:', '',
                 '```text', 'dataset              A60 floor  V floor    outside energy C1 same cohort excluded'])
    for row in floor_data['d0']:
        text.append(f"{row['dataset']:<21}{_fmt(row['A60_floor_MSE']):<11}{_fmt(row['V_floor_MSE']):<11}"
                    f"{_fmt(row['V_energy_outside_ratio']):<15}{_fmt(row['C1_actual_same_cohort_MSE']):<15}{_fmt(row['excluded_fraction'])}")
    if not floor_data['d0']:
        text.append('Pretraining D0 evidence: ../preflight/D0_SUBSPACE_FLOOR.csv (unchanged).')
    text.extend(['```', '', 'D1 학습 디코더 하한 (주 결과 seed별 비율의 중앙값):', '', '```text',
                 'dataset              method n/10 floor      actual     median ratio status'])
    for row in floor_data['d1_summary']:
        text.append(f"{row['dataset']:<21}{row['method_id']:<7}{str(row['n'])+'/10':<6}"
                    f"{_fmt(row.get('floor_MSE_median')):<11}{_fmt(row.get('actual_same_cohort_MSE_median')):<11}"
                    f"{_fmt(row.get('floor_fraction_median')):<13}{row['status']}")
    text.extend(['```', '', _prediction_document(checks), '', '남은 범위와 해석:', '',
        '- [확인] A4 대 OURS는 D 전체(마지막 관측값 + 변화량 + lag)의 효과다. '
        'A5 미실행으로 마지막 관측값과 lag를 분리하지 않는다.',
        '- [확인] 실행 범위는 첫 캠페인 결과를 본 뒤 발표상 공백을 메우기 위해 선택했다. '
        'job 정의·seed·판정 규칙은 첫 사전 등록을 그대로 사용했다.',
        '- [확인] A3(E의 효과), A5(lag만의 효과), C2·C3 연장, A1/ETTh2 연장, '
        'A4 연장은 NOT_RUN_SCOPE다. 예산 실패와 구분한다.',
        '- [확인] 필요한 A4 연장이 범위 밖이면 PROVISIONAL을 유지한다. '
        'A1/jena 연장도 10 seed 전부 있어야 주 결과로 채택한다. '
        '상한 10 epoch를 5개 이상 고르면 undertrained_at_10=True를 그대로 보고한다.',
        '- [확인] 4개 데이터를 모두 보고하며 주 집계는 jena·ETTh1·appliances 3개다. '
        'ETTh2_safe17400는 보조 보고다. n<10 그룹은 INCOMPLETE다.',
        '- [확인] Jena 비교의 TIE_PROVISIONAL을 포함하여 미완료 비교 연장에 따른 '
        '잠정 판정은 이번 P·D 결과로 해제하지 않는다.',
        '- [확인] V는 개발 검증 구간이며 보호 테스트 T가 아니다. '
        '학습·선택은 A_fit·A_stop/A60에 한정한다. T 숫자를 읽은 job은 통과할 수 없다.',
        '- [추정] 높은 D1 비율은 출력 공간이 오차의 큰 부분을 설명함을 뒷받침하지만 '
        '수준 이동 하나만을 원인으로 확정하지 않는다. D0는 고정 PCA C1에만 직접 적용한다.',
        '', '등록 비교 판정:', '', '```text'])
    for row in decision['comparative']:
        text.append(f"{row['dataset']:<21}{_label(row)}")
    text.extend(['```', '', '실행·미실행 목록, 실제 경과시간, 예산·마감·중단 근거:', '', '```json',
                 json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True), '```', '',
                 '[확인] 기존 파일 보존 검사: PRESERVATION_REPORTING.json. '
                 '완료 산출물과 입력의 SHA-256: REPORTING_COMPLETION.json·REPORTING_INPUTS.json.',
                 '그림: FIG_ABLATION_UPDATED.png/.svg; 각 seed와 미완료 수: FIG_ABLATION_UPDATED.csv.', ''])
    _put(output / 'RESULTS_KO.md', '\n'.join(text))


def _result_digest(results):
    payload = json.dumps(results, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf8')).hexdigest()


def verify_completion(output, results=None):
    receipt = common.read(Path(output) / 'REPORTING_COMPLETION.json')
    for name, expected in receipt['outputs'].items():
        path = Path(output) / name
        if not path.is_file() or common.sha(path) != expected:
            raise ValueError('Completed reporting output changed: ' + str(path))
    if results is not None and _result_digest(results) != receipt['input_results_sha256']:
        raise ValueError('Reporting input snapshot changed after completion; refuse to rewrite sealed output')
    common.guard_preserved()
    return receipt


def run(stage='final', status=None):
    if stage not in ('interim', 'final'):
        raise ValueError('stage must be interim or final')
    output = common.REPORT / 'interim_A4' if stage == 'interim' else common.REPORT
    output.mkdir(parents=True, exist_ok=True)
    phase = 'preservation'
    try:
        common.guard_preserved()
        common.install_write_guard()
        results = common.all_results()
        if stage == 'interim':
            results = [r for r in results if not (r['job']['method_id'] == 'A1' and
                       r['job']['dataset'] == 'jena' and r['job']['extended'])]
        if (output / 'REPORTING_COMPLETION.json').exists():
            # Interim is a frozen earlier snapshot while extension jobs may now exist.
            return verify_completion(output, results if stage == 'final' else None)
        input_path = output / 'REPORTING_INPUTS.json'
        if input_path.exists():
            saved = common.read(input_path)
            ids = {row['job_id'] for row in saved['jobs']}
            snapshot_results = [r for r in results if r.get('job', r.get('row', {})).get('job_id') in ids]
            if _result_digest(snapshot_results) != saved['input_results_sha256']:
                raise ValueError('Unfinished reporting input snapshot changed')
            if stage == 'final' and len(snapshot_results) != len(results):
                raise ValueError('Additional final inputs appeared after reporting snapshot was sealed')
            results, status = snapshot_results, saved['execution_status']
        if status is None:
            path = common.REPORT / ('EXECUTION_COMPLETION.json' if stage == 'final' else 'PROGRESS.json')
            status = common.read(path) if path.exists() else {'status': 'UNSPECIFIED'}
        snapshot = dict(stage=stage, input_result_count=len(results), input_results_sha256=_result_digest(results),
            jobs=[dict(job_id=r.get('job', r.get('row', {})).get('job_id'), status=r.get('status')) for r in results],
            execution_status=status, original_reuse_audit_sha256=common.sha(common.OLD_REPORT / 'G1_REUSE_AUDIT.json'))
        _json(output / 'REPORTING_INPUTS.json', snapshot)
        phase = 'matplotlib'
        _prepare_matplotlib()
        phase = 'registered_analysis'
        registered = output / 'registered_analysis'
        decision = analysis.report(registered, results, status)
        # The registered RESULTS_KO remains exact in its dedicated directory.
        for name in analysis.GENERATED_FILES:
            if name != 'RESULTS_KO.md':
                _put(output / name, (registered / name).read_bytes())
        rows = _read_csv(registered / 'RUNS.csv')
        summaries = _read_csv(registered / 'SUMMARY.csv')
        phase = 'floors'
        from . import floors
        floor_data = floors.run(output, methods=('A4',) if stage == 'interim' else ('C2', 'C3', 'A4'),
                                include_d0=stage == 'final', results=results)
        phase = 'prediction_checks'
        checks = prediction_checks(summaries, floor_data['d1'])
        _json(output / 'PREDICTION_CHECK.json', checks)
        _put(output / 'PREDICTION_CHECK.md', _prediction_document(checks))
        phase = 'updated_figure'
        os.environ['MPLCONFIGDIR'] = str(common.CACHE / 'matplotlib')
        _figure(output, rows)
        phase = 'documents'
        _documents(output, summaries, decision, checks, floor_data, status, results, stage)
        copied = []
        if stage == 'final':
            for name in ('D0_SUBSPACE_FLOOR.csv', 'D0_MEAN_SHIFT_CHANNELS.csv', 'D0_D1_VALIDATION.json'):
                source = common.REPORT / 'preflight' / name
                if source.exists():
                    target = output / (source.stem + '_PREFLIGHT' + source.suffix)
                    _put(target, source.read_bytes())
                    copied.append(target)
        phase = 'preservation'
        preservation = common.guard_preserved()
        _json(output / 'PRESERVATION_REPORTING.json', preservation)
        names = set(analysis.GENERATED_FILES) | {
            'REPORTING_INPUTS.json', 'PREDICTION_CHECK.json', 'PREDICTION_CHECK.md',
            'SLIDE_NUMBERS_UPDATED.md', 'PRESERVATION_REPORTING.json',
            'FIG_ABLATION_UPDATED.csv', 'FIG_ABLATION_UPDATED.png', 'FIG_ABLATION_UPDATED.svg',
            'D1_DECODER_FLOOR.csv', 'D1_DECODER_FLOOR_SUMMARY.csv', 'D0_D1_VALIDATION.json'}
        if stage == 'final':
            names.update(('D0_SUBSPACE_FLOOR.csv', 'D0_MEAN_SHIFT_CHANNELS.csv'))
        names.update(p.name for p in copied)
        names.update(p.relative_to(output).as_posix() for p in registered.rglob('*') if p.is_file())
        receipt = dict(stage=stage, status='COMPLETE', input_result_count=len(results),
            input_results_sha256=snapshot['input_results_sha256'],
            outputs={name: common.sha(output / name) for name in sorted(names)},
            comparative_overall=decision['comparative_overall'],
            T_numeric_values_read=0, preservation=preservation)
        _json(output / 'REPORTING_COMPLETION.json', receipt)
        return receipt
    except Exception as exc:
        scientific = (phase in ('preservation', 'floors', 'prediction_checks') or
                      isinstance(exc, FileExistsError) or
                      (phase == 'registered_analysis' and isinstance(exc, ValueError)))
        common.atomic(output / 'REPORTING_FAILURE.json', dict(stage=stage, phase=phase,
            classification='SCIENTIFIC_GATE' if scientific else 'POSTPROCESS_FAILED',
            error=f'{type(exc).__name__}: {exc}', traceback=traceback.format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('interim', 'final'))
    args = parser.parse_args()
    receipt = run(args.stage)
    print(json.dumps({key: value for key, value in receipt.items() if key != 'outputs'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
