"""Close the declared scope only from terminal receipts and reviewed artifacts."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CACHE = ROOT / '.cache' / HERE.name


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', required=True)
    parser.add_argument('--report', default='report_v2')
    parser.add_argument('--generator', default='make_report_v2.py')
    parser.add_argument('--claim-review', default='REPORT_CLAIM_REVIEW_v2.json')
    parser.add_argument('--output', default='FINAL_AUDIT_V2.json')
    args = parser.parse_args()
    destination = HERE / args.output
    if destination.resolve().parent != HERE or destination.exists():
        raise ValueError('Fresh final audit file required')
    started = time.perf_counter()
    checks = []

    def check(name, passed, detail=None):
        checks.append({'name': name, 'pass': bool(passed), 'detail': detail})

    def artifact(record, name):
        path = Path(record['path'])
        if not path.is_absolute():
            path = ROOT / path
        check(name + ':exists', path.is_file(), str(path))
        if path.is_file():
            check(name + ':sha', sha(path) == record['sha256'])

    preflight = read(HERE / 'PREFLIGHT.json')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    check('HEAD_unchanged_no_automatic_commit', head == preflight['actual_head'], head)
    for index, item in enumerate(preflight['preserve_files']):
        artifact(item, f'preservation:{index}')
    artifact(preflight['plan'], 'frozen_PLAN')
    run_names = ('preflight_v1', 'dev_v1', 'new_v1', 'beijing_repair_v1', 'new_resolved_v1',
                 'checkpoint_audit_v1', 'clean_cost_v1')
    runs = {}
    sources = []
    for name in run_names:
        path = HERE / ('run_' + name) / 'RESULT.json'
        check(name + ':terminal_exists', path.is_file())
        if not path.is_file():
            continue
        result = read(path)
        runs[name] = result
        sources.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'status': result.get('status')})
        if name == 'new_v1':
            failures = result.get('failed_paths', [])
            check(name + ':original_PARTIAL_preserved', result.get('status') == 'PARTIAL_FAILED_PATHS')
            check(name + ':known_loader_failure_preserved', len(failures) == 1
                  and failures[0]['dataset'] == 'beijing_aotizhongxin'
                  and failures[0]['label'] == 'DATA_PREPARATION')
        elif name == 'new_resolved_v1':
            check(name + ':resolved_scope_declares_deviation', result.get('status') == 'COMPLETE_RESOLVED_WITH_DECLARED_PROTOCOL_DEVIATION')
            check(name + ':original_joint_protocol_not_claimed', result.get('protocol_deviation', {}).get('joint_models_before_any_new_EVAL') is False)
            check(name + ':no_post_Appliances_method_tuning', result.get('protocol_deviation', {}).get('Beijing_tuning_changed_after_Appliances_EVAL') is False)
            check(name + ':no_unresolved_failed_paths', not result.get('unresolved_failed_paths'))
            for index, parent in enumerate(result.get('source_results', [])):
                artifact(parent, name + ':parent:' + str(index))
        else:
            check(name + ':COMPLETE', result.get('status') == 'COMPLETE', result.get('status'))
            check(name + ':no_failed_paths', not result.get('failed_paths'), result.get('failed_paths'))
        if name != 'new_resolved_v1':
            check(name + ':backbone_hash', result.get('bolt_final_hash') == result.get('bolt_initial_hash') and bool(result.get('bolt_initial_hash')))
        for file_path, digest in result.get('source_hashes', {}).items():
            artifact({'path': file_path, 'sha256': digest}, name + ':source:' + file_path)
    dev, new = runs.get('dev_v1', {}), runs.get('new_resolved_v1', {})
    dev_fits = [c for c in dev.get('cases', []) if 'training' in c]
    new_fits = [c for c in new.get('cases', []) if 'training' in c]
    check('S1_24_actual_fits', len([c for c in dev_fits if c['stage'] == 'S1']) == 24)
    check('S2_52_actual_fits', len([c for c in dev_fits if c['stage'] == 'S2']) == 52)
    check('S3_44_actual_fits_locked_architecture', len(new_fits) == 44, len(new_fits))
    for case in dev_fits + new_fits:
        training = case['training']
        key = case['dataset'] + ':' + case['label']
        check(key + ':600_updates', training.get('updates') == 600 and len(training.get('curve', [])) == 600)
        check(key + ':checkpoints', [c['update'] for c in training.get('checkpoints', [])] == [0, 100, 200, 400, 600])
        artifact(training['selected_checkpoint'], key + ':selected_checkpoint')
    native = [m['native_training'] for c in new.get('cases', []) for m in c.get('methods', []) if 'native_training' in m]
    check('native_controls_16_actual_fits', len(native) == 16, len(native))
    check('native_all_600', all(t.get('updates') == 600 and len(t.get('curve', [])) == 600 for t in native))
    check('two_new_datasets_evaluated', set(new.get('evaluation_open_receipts', {})) == {'appliances', 'beijing_aotizhongxin'})
    for tag, dataset in (('new_v1', 'appliances'), ('beijing_repair_v1', 'beijing_aotizhongxin')):
        parent = runs.get(tag, {})
        global_lock_path = HERE / ('run_' + tag) / 'MODEL_SELECTION_LOCK.json'
        check(dataset + ':own_global_model_lock_exists', global_lock_path.is_file())
        if global_lock_path.is_file():
            lock = read(global_lock_path)
            check(dataset + ':own_global_lock_after_training', lock['created_unix'] == parent.get('all_training_finished_unix'))
            receipt = parent.get('evaluation_open_receipts', {}).get(dataset, {})
            check(dataset + ':first_EVAL_after_own_lock', receipt.get('first_and_only_open_started_unix', 0) > lock['created_unix'])
            check(dataset + ':first_EVAL_origin_count', receipt.get('origin_count', 0) > 0)
            artifact(parent['global_model_selection_lock'], dataset + ':own_model_lock')
    repair_note = HERE / 'BEIJING_REPAIR_EXECUTION_NOTE_KO.md'
    artifact({'path': str(repair_note), 'sha256': '293bed78826fbadf9730520229a77368a592e84107c9356f47c92d4cf5f09e49'}, 'pre_repair_note')
    check('original_two_dataset_joint_barrier_failed_and_reported', new.get('protocol_deviation', {}).get('joint_models_before_any_new_EVAL') is False)
    cp = runs.get('checkpoint_audit_v1', {})
    check('all_core_checkpoint_actual_reinference_600', len(cp.get('records', [])) == 600)
    check('all_native_checkpoint_actual_reinference_80', len(cp.get('native_records', [])) == 80)
    check('all_alignment_FIT_pre_post_q_80', len(cp.get('alignment_records', [])) == 80)
    check('checkpoint_audit_no_training', cp.get('new_training_updates') == 0)
    cost = runs.get('clean_cost_v1', {})
    cost_records = cost.get('records', [])
    check('canonical_cost_methods_200', len(cost_records) == 200, len(cost_records))
    check('whole_path_cost_B1_B8_400', sum(len(r.get('latency', [])) for r in cost_records) == 400)
    check('cost_no_training', cost.get('new_training_updates') == 0)
    for index, record in enumerate(cost_records):
        values = record['latency']
        check(f'cost:{index}:batches_and_actual_repeats', {v['batch'] for v in values} == {1, 8}
              and all(v['warmup'] == 2 and v['repeats'] == 7 and len(v['seconds']) == 7 and v['parity_pass'] for v in values))
    independent_path = Path(args.audit).resolve()
    independent = read(independent_path)
    check('independent_full_scope_audit_PASS', independent.get('status') == 'PASS' and independent.get('failed_count') == 0,
          {'path': str(independent_path), 'count': independent.get('count'), 'status': independent.get('status')})
    report_folder = HERE / args.report
    manifest_path = report_folder / 'REPORT_MANIFEST.json'
    manifest = read(manifest_path)
    check('report_complete_declaring_original_protocol_deviation',
          manifest.get('status') == 'DERIVED_REPORT_COMPLETE_WITH_DECLARED_PROTOCOL_DEVIATION', manifest.get('status'))
    check('report_full_scope_coverage', manifest.get('full_scope_coverage') is True
          and all(manifest.get('actual_scope', {}).get(key) == value
                  for key, value in manifest.get('required_full_scope', {}).items()))
    for index, receipt in enumerate(manifest.get('source_receipts', [])):
        artifact(receipt, 'report:source_receipt:' + str(index))
    check('report_uses_this_independent_audit', any(
        Path(receipt['path']).name == independent_path.name
        and receipt['sha256'] == sha(independent_path)
        for receipt in manifest.get('source_receipts', [])))
    for name, digest in manifest['outputs'].items():
        artifact({'path': str(report_folder / name), 'sha256': digest}, 'report:' + name)
    check('report_generator_frozen', sha(HERE / args.generator) == manifest['report_generator_sha256'])
    first_report = HERE / 'report_v1'
    first_manifest = read(first_report / 'REPORT_MANIFEST.json')
    for name, digest in first_manifest['outputs'].items():
        artifact({'path': str(first_report / name), 'sha256': digest}, 'preserved_report_v1:' + name)
    check('first_report_generator_preserved', sha(HERE / 'make_report.py') == first_manifest['report_generator_sha256'])
    claim_path = HERE / args.claim_review
    claim = read(claim_path)
    check('report_claim_review_PASS', claim.get('status') == 'PASS' and claim.get('failed_count') == 0)
    artifact(claim['report'], 'report_claim_review:report')
    check('report_claim_review_covers_active_report', claim['report']['sha256'] == sha(report_folder / 'REPORT_KO.md'))
    for markdown in (report_folder / 'REPORT_KO.md', HERE / 'README.md'):
        check(markdown.name + ':exists', markdown.is_file())
        if not markdown.is_file():
            continue
        for link in re.findall(r'\]\(([^)]+)\)', markdown.read_text(encoding='utf-8')):
            if link.startswith(('http:', 'https:', '#')):
                continue
            target = (markdown.parent / link.split('#', 1)[0]).resolve()
            if target == destination.resolve():
                check(markdown.name + ':local_link:planned_audit_self_reference', True, str(target))
            else:
                check(markdown.name + ':local_link:' + link, target.exists())
    visual = read(HERE / 'VISUAL_REVIEW.json')
    check('actual_visual_review_7_PNG', visual.get('status') == 'PASS' and len(visual.get('images', [])) == 7)
    for index, item in enumerate(visual.get('images', [])):
        artifact(item, 'visual:' + str(index))
    resources = [read(p) for p in sorted((HERE / 'resources').glob('*.json'))]
    gpu_wall = sum(r['wall_seconds'] for r in resources if r.get('gpu_process'))
    cpu_receipts = []
    for path in sorted(HERE.glob('validation_*/audit.json')):
        receipt = read(path)
        if 'cpu_wall_seconds' in receipt:
            cpu_receipts.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
                                 'wall_seconds': receipt['cpu_wall_seconds']})
    for path in sorted(HERE.glob('report_v*/REPORT_MANIFEST.json')):
        receipt = read(path)
        cpu_receipts.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
                             'wall_seconds': receipt.get('cpu_resources', {}).get('process_wall_seconds', 0)})
    for path in sorted(HERE.glob('REPORT_CLAIM_REVIEW_v*.json')):
        receipt = read(path)
        cpu_receipts.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
                             'wall_seconds': receipt.get('cpu_wall_seconds', receipt.get('cpu_body_wall_seconds', 0))})
    for path in sorted(HERE.glob('FINAL_AUDIT*.json')):
        receipt = read(path)
        cpu_receipts.append({'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
                             'wall_seconds': receipt.get('wall_seconds', 0)})
    cpu_report = manifest.get('cpu_resources', {})
    cpu_wall = sum(r['wall_seconds'] for r in cpu_receipts)
    storage = sum(p.stat().st_size for base in (HERE, CACHE) for p in base.rglob('*') if p.is_file())
    check('GPU_process_wall_under_7200', gpu_wall < 7200, gpu_wall)
    check('recorded_CPU_audit_and_report_wall_under_7200', cpu_wall < 7200, cpu_wall)
    check('report_CPU_only_and_sampled_RAM_under_8GiB', cpu_report.get('cpu_only') is True
          and cpu_report.get('peak_sampled_rss_bytes', 9*1024**3) <= 8*1024**3)
    check('all_RAM_and_allocated_VRAM_under_8GiB', all(max(r['peak_sampled_rss_bytes'], r['peak_allocated_vram_bytes']) <= 8*1024**3 for r in resources))
    check('new_storage_under_12GiB', storage < 12*1024**3, storage)
    check('campaign_wall_under_8h', time.time() - preflight['started_unix'] < 28800)
    report = {'status': 'PASS_WITH_DECLARED_PROTOCOL_DEVIATION' if all(c['pass'] for c in checks) else 'FAIL', 'created_unix': time.time(),
              'count': len(checks), 'failed_count': sum(not c['pass'] for c in checks), 'checks': checks,
              'terminal_sources': sources, 'independent_audit': {'path': str(independent_path.relative_to(ROOT)), 'sha256': sha(independent_path)},
              'report_claim_review': {'path': str(claim_path.relative_to(ROOT)), 'sha256': sha(claim_path)},
              'report_manifest_sha256': sha(manifest_path), 'visual_review_sha256': sha(HERE / 'VISUAL_REVIEW.json'),
              'resource_totals': {'gpu_process_wall_seconds': gpu_wall, 'new_storage_bytes': storage,
                                  'recorded_CPU_audit_and_report_wall_seconds': cpu_wall,
                                  'CPU_receipts': cpu_receipts,
                                  'CPU_measurement_limit': 'Each terminal audit attempt and report process counted once; short tests, source reads and an interrupted audit without a receipt are not fully measured.'},
              'wall_seconds': time.perf_counter() - started, 'source_sha256': sha(__file__),
              'protocol_deviation': new.get('protocol_deviation'),
              'limitation': 'Evidence closure with original joint two-dataset barrier unmet. No universal accuracy gains, exhaustive leakage absence, or foundation pretraining independence established.'}
    with destination.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({k: report[k] for k in ('status', 'count', 'failed_count', 'resource_totals', 'wall_seconds')}))
    if report['status'] != 'PASS_WITH_DECLARED_PROTOCOL_DEVIATION':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
