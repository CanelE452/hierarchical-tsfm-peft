"""Guarded independent file/link and terminal-JSON-to-CSV report verification."""
from pathlib import Path
import argparse
import csv
import json
import math
import re
import sys
from urllib.parse import unquote

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    import matched_readout_runtime_v1 as runtime
    limit = runtime.ContinuationLimit('matched_report_file_csv_verification_v2', gpu=False)
    report = (ROOT / args.report).resolve()
    output = (ROOT / args.out).resolve()
    result = {'status': 'RUNNING', 'checks': 0, 'failures': [],
              'independent_E_F_or_solver_replay': False}
    try:
        if output.exists(): raise FileExistsError(output)
        limit.checkfull()
        manifest = runtime.read(report / 'MANIFEST.json')
        def check(passed, label):
            result['checks'] += 1
            if not passed: result['failures'].append(label)
            limit.shared.checkfast()
        check(manifest['status'] == 'COMPLETE' and manifest['audit_gate']['passed'], 'TERMINAL_REPORT_AUDIT_GATE')
        for index, record in enumerate(manifest['artifact_records']):
            if index % 16 == 0: limit.checkfull()
            path = (report / record['path']).resolve()
            check(path.is_relative_to(report) and path.is_file() and runtime.sha(path).lower() == record['sha256'].lower(),
                  'ARTIFACT_SHA:' + record['path'])
        for record in manifest['input_records']:
            limit.shared.checkfast()
            check(runtime.sha(record['path']).lower() == record['sha256'].lower()
                  and runtime.sha(report / record['copy_relative']).lower() == record['sha256'].lower(),
                  'INPUT_COPY_SHA:' + record['copy_relative'])
        for document in report.rglob('*.md'):
            for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', document.read_text(encoding='utf-8-sig')):
                if target.startswith(('https://', 'http://', '#')): continue
                path = (document.parent / unquote(target.strip('<>').split('#', 1)[0])).resolve()
                check(path.is_relative_to(ROOT) and path.exists() and '.cache' not in path.relative_to(ROOT).parts,
                      'LOCAL_MD_LINK:' + str(document.relative_to(report)) + ':' + target)
        pngs = sorted((report / 'figures').glob('*.png'))
        svgs = sorted((report / 'figures').glob('*.svg'))
        check(len(pngs) == 6 and {p.stem for p in pngs} == {p.stem for p in svgs}, 'SIX_PNG_SVG_PAIRS')
        check(runtime.read(report / 'RESULT.json')['status'] == 'COMPLETE', 'REPORT_RESULT_TERMINAL_COMPLETE')
        producers = {}
        for record in manifest['input_records']:
            if record.get('role') == 'producer':
                producer = runtime.read(report / record['copy_relative'])
                producers[producer['dataset'], producer['seed']] = producer
        check(set(producers) == {(d, s) for d in ('robin', 'beijing_aotizhongxin') for s in (92601, 92602)},
              'EXACT_FOUR_UNIQUE_PRODUCERS')
        with (report / 'METRICS.csv').open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        check(len(rows) == 120 and manifest['metrics_rows'] == 120, '120_METRIC_ROWS')
        keys = []
        for row in rows:
            key = (row['dataset'], int(row['seed']), row['method'], row['split'], row['metric_scale'])
            keys.append(key)
            producer = producers[key[:2]]
            source = (producer['references'][row['split']][row['method']]
                      if row['method'] in ('RAW_BOLT', 'ORIGINAL_MAIN')
                      else producer['families'][row['method']]['splits'][row['split']])[row['metric_scale']]
            for metric in ('mse', 'mae'):
                check(math.isclose(float(row[metric]), source[metric], abs_tol=1e-12, rel_tol=1e-12),
                      'CSV_METRIC:' + str(key) + ':' + metric)
            for field in ('observed_counts', 'per_channel_mse', 'per_channel_mae'):
                check(json.loads(row[field]) == source[field], 'CSV_CHANNEL_VALUES:' + str(key) + ':' + field)
        check(len(set(keys)) == 120, 'UNIQUE_METRIC_KEYS')
        for filename, count in (('ALPHA_GRID.csv', 56), ('CAPACITY.csv', 12), ('COST.csv', 12)):
            with (report / filename).open(encoding='utf-8-sig', newline='') as handle:
                check(len(list(csv.DictReader(handle))) == count, 'ROW_COUNT:' + filename)
        for dataset, summary in manifest['summary'].items():
            for method, pair in summary.items():
                selected = [r for r in rows if r['dataset'] == dataset and r['method'] == method
                            and r['split'] in ('val', 'eval') and r['metric_scale'] == 'normalized']
                check(len(selected) == 2, 'EXACT_TWO_SEED_SUMMARY:' + dataset + ':' + method)
                for metric in ('mse', 'mae'):
                    check(math.isclose(sum(float(r[metric]) for r in selected) / 2, pair[metric],
                                       abs_tol=1e-12, rel_tol=1e-12), 'SUMMARY:' + dataset + ':' + method + ':' + metric)
        result.update(status='PASS' if not result['failures'] else 'FAIL',
                      failed_count=len(result['failures']),
                      manifest_sha256=runtime.sha(report / 'MANIFEST.json'),
                      normalized_two_seed_summary=manifest['summary'],
                      images=[{'path': str(p), 'sha256': runtime.sha(p)} for p in pngs])
        limit.checkfull()
    except Exception as error:
        result.update(status='FAILED_PRESERVED', error={'type': type(error).__name__, 'message': str(error)})
    finally:
        receipt = limit.close('COMPLETE' if result['status'] == 'PASS' else 'FAILED_PRESERVED', result.get('error'))
        result['resource_receipt'] = receipt
        if receipt['terminal_status'] != 'COMPLETE': result['status'] = receipt['terminal_status']
        runtime.write_new(output, result)
    print(json.dumps({k: result.get(k) for k in ('status', 'checks', 'failed_count', 'error')}))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
