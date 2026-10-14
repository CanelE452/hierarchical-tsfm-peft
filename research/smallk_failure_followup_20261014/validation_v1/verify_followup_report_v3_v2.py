"""Independent report artifact checks; no model inference or target arrays."""
from pathlib import Path
import csv
import hashlib
import json
import re
import sys

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[1]
sys.path.insert(0, str(STUDY))
from followup_runtime import FollowupLimit


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    output = HERE / 'REPORT_FOLLOWUP_v3_ROOT_CHECK_v2.json'
    if output.exists():
        raise FileExistsError(output)
    limit = FollowupLimit('followup_report_v3_root_check_v2', gpu=False)
    result = {'status': 'RUNNING', 'scope': 'report hashes, links, coverage, displayed values and preservation; no numerical arrays or model replay',
              'independent_holdout': False, 'scientific_improvement_claim': False,
              'checks': []}
    def check(name, condition):
        result['checks'].append({'name': name, 'pass': bool(condition)})
    try:
        reports = [STUDY / ('report_followup_v' + str(i)) for i in (1, 2, 3)]
        for report in reports:
            manifest = read(report / 'MANIFEST.json')
            check(report.name + '/manifest_status', manifest['generator_result_status'] == 'COMPLETE')
            for entry in manifest['artifacts']:
                path = report / entry['path']
                check(report.name + '/hash/' + entry['path'], path.is_file() and sha(path) == entry['sha256'])
                check(report.name + '/bytes/' + entry['path'], path.stat().st_size == entry['bytes'])
        latest = reports[-1]
        producer = read(latest / 'RESULT.json')
        check('actual_report', producer['status'] == 'COMPLETE' and producer['actual_report_generated'])
        check('scope_12', producer['coverage']['expected_conditions'] == 12 and producer['coverage']['complete_conditions'] == 12)
        check('producer_terminal', len(producer['producer_statuses']) == 12 and all(v['status'] == 'COMPLETE' for v in producer['producer_statuses']))
        check('audits_four', len(producer['audits']) == 4 and all(v['all_declared_scopes_passed'] for v in producer['audits']))
        check('no_generalization_claim', producer['independent_holdout'] is False and producer['improvement_established'] is False)
        check('no_new_inference', producer['new_model_inference'] is False and producer['GPU_execution'] is False)
        check('coverage_d3', producer['tables_rows']['d3'] == 32 and producer['tables_rows']['d3_preserved'] == 8)
        check('six_png', len(list((latest / 'figures').glob('*.png'))) == 6)
        check('six_svg', len(list((latest / 'figures').glob('*.svg'))) == 6)
        text = (latest / 'REPORT_KO.md').read_text(encoding='utf-8')
        for token in ('0.960106701', '0.569546544', '0.340401887', '0.661957198', '0.589575414', 'REFIT_RAW', 'warm2/repeat7', '독립 holdout'):
            check('display/' + token, token in text)
        for link in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
            if link.startswith(('https://', 'http://', '#')):
                continue
            check('relative_link/' + link, not Path(link).is_absolute() and (latest / link.split('#')[0]).is_file())
        for path in (latest / 'tables').glob('*.csv'):
            if path.name == 'resources.csv':
                continue
            for older in reports[:-1]:
                check('table_unchanged/' + older.name + '/' + path.name, (older / 'tables' / path.name).is_file() and sha(older / 'tables' / path.name) == sha(path))
        for path in (latest / 'figures').glob('*.png'):
            check('png_v2_unchanged/' + path.name, sha(reports[1] / 'figures' / path.name) == sha(path))
        with (latest / 'provenance/input_manifest.csv').open(encoding='utf-8-sig', newline='') as handle:
            rows = list(csv.DictReader(handle))
        result['input_manifest_rows'] = len(rows)
        result['report_manifest_sha256'] = sha(latest / 'MANIFEST.json')
        result['report_sha256'] = sha(latest / 'REPORT_KO.md')
        result['root_visually_inspected_pngs'] = [p.name for p in sorted((latest / 'figures').glob('*.png'))]
        result['failed_count'] = sum(not v['pass'] for v in result['checks'])
        result['check_count'] = len(result['checks'])
        result['status'] = 'PASS' if result['failed_count'] == 0 else 'FAIL_PRESERVED'
        limit.check()
    except Exception as error:
        result.update(status='FAILED_PRESERVED', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        result['shared_resource_receipt'] = limit.close('COMPLETE' if result['status'] == 'PASS' else 'FAILED_PRESERVED')
        if result['shared_resource_receipt'].get('terminal_status') != 'COMPLETE':
            result['status'] = result['shared_resource_receipt']['terminal_status']
        with output.open('x', encoding='utf-8') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
    print(json.dumps({'status': result['status'], 'checks': result.get('check_count'), 'failed': result.get('failed_count'), 'path': str(output)}, ensure_ascii=True))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
