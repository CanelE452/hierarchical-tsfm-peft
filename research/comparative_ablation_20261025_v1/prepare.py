"""Seal the supplied experiment protocol and enumerate every permitted job."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

from . import runtime


METHODS = ('C1', 'C2', 'C3', 'A1', 'A2', 'A3', 'A4', 'A5')
DOCUMENT = Path('C:/Users/User/Downloads/comparative_ablation_executor_instructions.md')


def jobs():
    base, extensions = [], []
    for method in METHODS:
        for seed in ([92601] if method == 'C1' else range(92601, 92611)):
            for dataset, k in runtime.KS.items():
                entry = dict(job_id=f'CA_{method}_{dataset}_K{k}_s{seed}',
                             method_id=method, dataset=dataset, K=k, seed=seed, extended=False)
                base.append(entry)
                if method != 'C1':
                    extensions.append(dict(entry, job_id=entry['job_id'] + '_ext10', extended=True))
    return base, extensions


def budget():
    with (runtime.REPO / 'reports/k_tau_rule_20261023_v1/RUNS.csv').open(newline='', encoding='utf8') as stream:
        prior = list(csv.DictReader(stream))
    profiles, expected, upper, extension = {}, {}, {}, {}
    for dataset in runtime.KS:
        rows = [r for r in prior if r['dataset'] == dataset and r['search_seconds']]
        search = max(float(r['search_seconds']) for r in rows)
        per_epoch = max(float(r['refit_seconds']) / float(r['selected_epoch'])
                        for r in rows if float(r['selected_epoch']))
        observed_refit = statistics.mean(float(r['refit_seconds']) for r in rows)
        profiles[dataset] = dict(search_cap3_max_seconds=search,
                                refit_per_epoch_max_seconds=per_epoch,
                                historical_mean_refit_seconds=observed_refit)
        for method in METHODS:
            # The allowance is provisional until each family's first real job.
            # C3 gets 180 s for ten-sample A_stop/V evaluation; others get 30 s.
            overhead = 180. if method == 'C3' else 30.
            cap = 6 if method in ('C2', 'C3') else 3
            key = f'{method}/{dataset}'
            if method == 'C1':
                expected[key] = upper[key] = 60.
            else:
                expected[key] = search * cap / 3 + (per_epoch * cap if cap == 6 else observed_refit) + overhead
                upper[key] = search * cap / 3 + per_epoch * cap + overhead
                extension[key] = search * 10 / 3 + per_epoch * 10 + overhead
    total = lambda values: sum(v * (1 if k.startswith('C1/') else 10) for k, v in values.items())
    comparison = sum(v * (1 if k.startswith('C1/') else 10)
                     for k, v in upper.items() if k.split('/')[0] in ('C1', 'C2', 'C3'))
    return dict(status='PASS_ESTIMATE' if comparison <= 40 * 3600 else 'STOP_BUDGET',
                profiles=profiles, per_job_expected_seconds=expected, per_job_upper_seconds=upper,
                expected_base_hours=total(expected)/3600, cap_selected_base_upper_hours=total(upper)/3600,
                comparison_upper_hours=comparison/3600, worst_extension_additional_hours=total(extension)/3600,
                assumptions=['Linear scaling of measured search time with cap.',
                             'A1-A5 expected refit uses historical OURS mean; C2/C3 assumes full-cap refit.',
                             'Per-job load/evaluation allowance 30s, C3 180s; these are unmeasured estimates.',
                             'No claimed concurrency speedup. Update after first real job per family.',
                             'If updated remaining estimate exceeds time available, drop A5 then earlier tail families.'],
                source='reports/k_tau_rule_20261023_v1/RUNS.csv')


def main():
    source = DOCUMENT.read_text(encoding='utf8')
    section3 = source.split('## 3.', 1)[1].split('## 4.', 1)[0]
    section5 = source.split('## 5.', 1)[1].split('## 6.', 1)[0]
    section6 = source.split('## 6.', 1)[1].split('## 7.', 1)[0]
    markdown = ('# 비교·절제 실험 사전등록\n\n'
                '캠페인: comparative_ablation_20261025_v1. 아래 고정값·결정 기준·중단 조건은 제공 문서에서 그대로 보존했다.\n'
                '공식 AdaPTS 전체 재현이 아니라 동일 TSFM 조건의 어댑터 비교다. T 수치는 사용하지 않는다.\n\n'
                '## 3.' + section3 + '## 5.' + section5 + '## 6.' + section6)
    registered = {}
    for name in ('models.py', 'runtime.py', 'run_one.py'):
        path = Path(__file__).parent / name
        registered[path.relative_to(runtime.REPO).as_posix()] = runtime.sha(path)
    protocol = dict(campaign=runtime.NAME, created_utc=datetime.now(timezone.utc).isoformat(),
                    markdown=markdown, fixed_conditions_verbatim='## 3.' + section3,
                    decision_criteria_verbatim='## 5.' + section5,
                    stop_conditions_verbatim='## 6.' + section6,
                    source_document_sha256=runtime.sha(DOCUMENT), registered_sources=registered,
                    datasets=runtime.KS, seeds=list(range(92601,92611)),
                    limit_seconds=40*3600, latest_start='2026-10-27T21:00:00+09:00',
                    foundation_sha256=runtime.F_SHA, current_training_executed=False,
                    approval_status='Local Git commit requires direct user approval under global instructions.')
    estimate = budget()
    base, ext = jobs()
    plan = dict(campaign=runtime.NAME, jobs=base, extension_jobs=ext,
                limit_seconds=40*3600, latest_start='2026-10-27T21:00:00+09:00',
                base_job_count=len(base), training_job_count=len(base)-4,
                budget=estimate, extension_trigger='At least 5 of 10 seeds choose current cap, using A_stop only.',
                extension_priority=['C2/jena','C3/jena','C2/appliances','C3/appliances',
                                    'C2/ETTh1','C3/ETTh1','C2/ETTh2_safe17400','C3/ETTh2_safe17400',
                                    'A1','A2','A3','A4','A5'])
    runtime.REPORT.mkdir(parents=True, exist_ok=True)
    with (runtime.REPORT/'PREREGISTRATION.md').open('x',encoding='utf8',newline='\n') as stream:
        stream.write(markdown)
    runtime.write_new(runtime.REPORT/'PREREGISTRATION.json', protocol)
    runtime.write_new(runtime.REPORT/'RUN_PLAN.json', plan)
    runtime.write_new(runtime.REPORT/'G6_BUDGET.json', estimate)
    runtime.write_new(runtime.REPORT/'GIT_PREFLIGHT.json', dict(
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=runtime.REPO,text=True).strip(),
        required_commit='769db577da01bb2a7ccbd179909d7e12fd13c7fe',
        original_checkout='E:/CODING/proj/hierarchical-tsfm-peft',
        original_head='c7a06e522e1cd8e100a3e2291a7f693dd3f7125f',
        resolution='Native managed isolated worktree at the required remote HEAD; prior worktrees unchanged.'))
    print(json.dumps(dict(base_jobs=len(base), trained_jobs=len(base)-4,
                         budget=estimate['status'], expected_base_hours=estimate['expected_base_hours'],
                         comparison_upper_hours=estimate['comparison_upper_hours'])))


if __name__ == '__main__':
    main()
