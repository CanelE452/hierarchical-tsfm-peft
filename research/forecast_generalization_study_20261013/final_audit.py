"""Verify the finished study, report bindings, costs and preserved prior artifacts."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import re
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
HEAD = 'e57d0f01f148dd4004c53531c7c24c45198636ab'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(data_validation, filename):
    started = time.perf_counter()
    checks = []

    def check(name, passed, detail=None):
        checks.append({'name': name, 'pass': bool(passed), 'detail': detail})

    source = HERE/'run_resolved_v1/RESULT.json'
    result = read(source)
    report = HERE/'report_v1'
    manifest = read(report/'REPORT_MANIFEST.json')
    check('resolved_complete', result['status'] == 'COMPLETE')
    check('coverage', len(result['cases']) == 70 and sum(len(c['methods']) for c in result['cases']) == 236)
    check('unique_core_native_conditions', result['new_training_runs'] == 44 and result['new_native_training_conditions'] == 32)
    check('frozen_backbone', result['bolt_initial_hash'] == result['bolt_final_hash'])
    check('head_unchanged', subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() == HEAD)
    for name, artifact in result['source_runs'].items():
        check('source_run_'+name, sha(ROOT/artifact['path']) == artifact['sha256'])
    for path, digest in result['source_hashes'].items():
        check('frozen_source_'+Path(path).name, sha(ROOT/path) == digest)
    full = read(ROOT/result['source_runs']['full']['path'])
    repair = read(ROOT/result['source_runs']['repair']['path'])
    clean = read(ROOT/result['source_runs']['clean_cost']['path'])
    check('failed_paths_preserved_and_repaired', len(full['failed_paths']) == 4 and repair['status'] == 'COMPLETE' and not repair.get('failed_paths'))
    for tag, run in [('full', full), ('repair', repair), ('clean', clean)]:
        check(tag+'_source_snapshot_unchanged', run['source_hashes_unchanged_at_completion'])
        check(tag+'_backbone_unchanged', run['bolt_initial_hash'] == run['bolt_final_hash'])
        for path, digest in run['source_hashes'].items():
            check(tag+'_current_source_'+Path(path).name, sha(ROOT/path) == digest)
    check('no_training_in_clean_cost', clean['new_training_updates'] == 0 and clean['status'] == 'COMPLETE')
    check('no_competing_python_before_clean_cost', clean['competing_python_compute_processes_before_load'] == [])
    count = 0
    for record in clean['records']:
        for timing in record['latency']:
            count += 1
            check('clean_timing_'+record['dataset']+'_'+record['label']+'_'+record['method']+'_b'+str(timing['batch']),
                  timing['warmup'] == 2 and timing['repeats'] == 7 and len(timing['seconds']) == 7
                  and all(x > 0 for x in timing['seconds'])
                  and np.isclose(np.median(timing['seconds']), timing['median_seconds'], rtol=0, atol=1e-15)
                  and timing['pipeline_prediction_parity']['pass'])
    check('472_clean_timings', count == 472)
    for path, fixture in result['fixtures'].items():
        check('frozen_causal_gradient_fixture_'+path, fixture['pass'] and fixture['trainable_bolt_parameters'] == 0
              and fixture['adam_vs_adamw_wd0_5step_max_abs'] == 0
              and fixture['full_identity_initial_actual_bolt_max_abs_vs_raw'] == 0)
    for fixture in result['gradient_cuda_fixtures']:
        check('identity_cuda_repair_'+str(fixture['seed']), not fixture['original_gradient_finite']
              and fixture['repaired_input_gradient_finite'] and fixture['repaired_encoder_decoder_gradient_finite']
              and fixture['forward_max_abs_vs_original'] == 0 and fixture['temporary_update_restored']
              and fixture['backbone_hash_unchanged'])
    resources = result['resources']
    receipts = resources['own_receipts']+resources['supplement_receipts']
    total = 0
    for artifact in receipts:
        receipt = read(ROOT/artifact['path'])
        check('terminal_resource_'+receipt.get('tag', str(receipt['pid'])), receipt['status'] != 'RUNNING' and sha(ROOT/artifact['path']) == artifact['sha256'])
        total += receipt['wall_seconds']
    check('gpu_process_wall_sum_and_cap', np.isclose(total, resources['all_gpu_process_wall_seconds_summed_conservatively'], atol=1e-8, rtol=0) and total < 2700, total)
    check('memory_cap', max(resources['peak_sampled_rss_bytes'], resources['peak_allocated_vram_bytes']) < 8*1024**3)
    storage = sum(p.stat().st_size for root in (HERE, ROOT/'.cache'/HERE.name,
                    ROOT/'research/forecast_metric_alignment_20261013', ROOT/'.cache/forecast_metric_alignment_20261013')
                  for p in root.rglob('*') if p.is_file())
    check('storage_cap_actual', storage < 4*1024**3, storage)
    for validation in (HERE/'validation_resolved_v1/audit.json', Path(data_validation)):
        verdict = read(validation)
        check('independent_validation_'+validation.parent.name, verdict['status'].startswith('PASS'), sha(validation))
    check('report_input_binding', manifest['input_result_sha256'] == sha(source) and manifest['report_generator_sha256'] == sha(HERE/'make_report.py'))
    for name, digest in manifest['outputs'].items():
        check('report_output_'+name, sha(report/name) == digest)
    for name, expected in [('all_methods.csv', 236), ('unique_training.csv', 44), ('batch_cost.csv', 472)]:
        with (report/name).open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        check('csv_rows_'+name, len(rows) == expected, len(rows))
    check('report_236_npz_recalculations', len(manifest['prediction_recalculation_checks']) == 236
          and all(q['prediction_sha_match'] and max(q['errors'].values()) < 1e-9 for q in manifest['prediction_recalculation_checks']))
    links = []
    for document in [HERE/'README.md', report/'REPORT_KO.md']:
        for raw in re.findall(r'\]\(([^)]+)\)', document.read_text(encoding='utf-8')):
            target = raw.strip('<>').split('#', 1)[0]
            if target.startswith(('https:', 'http:')) or not target:
                continue
            resolved = (document.parent/target).resolve()
            check('relative_link_'+document.name+'_'+target, resolved.exists())
            links.append({'document': str(document.relative_to(HERE)), 'target': target})
    figures = sorted(report.glob('*.png'))
    check('five_png_figures', len(figures) == 5)
    check('five_svg_figures', len(list(report.glob('*.svg'))) == 5)
    old = ROOT/'research/forecast_component_study_20261013'
    old_manifest = read(old/'FINAL_AUDIT.json')
    for name, artifact in old_manifest['artifacts'].items():
        digest = artifact['sha256'] if isinstance(artifact, dict) else artifact
        check('old_sealed_'+name, sha(old/name) == digest)
    check('scope_is_development_val', 'not independent' in result['scope'])
    output = {'status': 'PASS' if all(x['pass'] for x in checks) else 'FAIL',
              'created_unix': time.time(), 'head': HEAD, 'checks': checks,
              'result_sha256': sha(source), 'report_sha256': sha(report/'REPORT_KO.md'),
              'readme_sha256': sha(HERE/'README.md'), 'audit_source_sha256': sha(__file__),
              'report_manifest_sha256': sha(report/'REPORT_MANIFEST.json'),
              'relative_links': links, 'figures': {p.name: sha(p) for p in figures},
              'gpu_process_wall_seconds': total, 'storage_bytes': storage,
              'wall_seconds': time.perf_counter()-started,
              'note': 'Figure readability is inspected by the root agent separately. Passing checks establish the listed contracts, not independent generalization or universal leakage absence.'}
    if Path(filename).name != filename:
        raise ValueError('Audit filename must stay in this study directory')
    with (HERE/filename).open('x', encoding='utf-8') as stream:
        json.dump(output, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': output['status'], 'checks': len(checks), 'failed': [q['name'] for q in checks if not q['pass']], 'gpu_process_wall_seconds': total, 'storage_bytes': storage}))
    return output['status'] == 'PASS'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-validation', required=True)
    parser.add_argument('--output', default='FINAL_AUDIT.json')
    args = parser.parse_args()
    raise SystemExit(0 if audit(args.data_validation, args.output) else 1)
