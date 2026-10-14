"""Fresh metered file/hash/link verification of a generated stage report."""
from pathlib import Path
import argparse
import json
import re
import sys
from urllib.parse import unquote

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))


def local_link_targets(text):
    return [unquote(s.strip('<>').split('#', 1)[0]) for s in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text)
            if not s.startswith(('https://', 'http://', '#'))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    import stage_separation_runtime_v1 as runtime
    report = (ROOT / args.report).resolve()
    output = ROOT / args.out
    if output.exists(): raise FileExistsError(output)
    limit = runtime.ContinuationLimit('stage_report_final_verification_v1', gpu=False)
    result = {'status': 'RUNNING', 'checks': 0, 'failures': [], 'independent_numerical_replay': False}
    try:
        manifest = runtime.read(report / 'MANIFEST.json')
        result['manifest'] = {'path': str(report / 'MANIFEST.json'), 'sha256': runtime.sha(report / 'MANIFEST.json')}
        if manifest.get('generator_result_status', manifest.get('status')) != 'COMPLETE':
            result['failures'].append('REPORT_GENERATOR_NOT_COMPLETE')
        for record in manifest['artifacts']:
            path = (report / record['path']).resolve()
            limit.checkfull()
            result['checks'] += 1
            if not path.is_relative_to(report) or not path.is_file() or runtime.sha(path) != record['sha256']:
                result['failures'].append('MANIFEST_FILE_SHA:' + record['path'])
        for document in report.rglob('*.md'):
            for target in local_link_targets(document.read_text(encoding='utf-8-sig')):
                result['checks'] += 1
                path = (document.parent / target).resolve()
                if not path.is_relative_to(ROOT) or not path.exists() or '.cache' in path.relative_to(ROOT).parts:
                    result['failures'].append('NONREVIEWABLE_LOCAL_LINK:' + str(document.relative_to(report)) + ':' + target)
        figures = sorted((report / 'figures').glob('*.png'))
        svgs = sorted((report / 'figures').glob('*.svg'))
        result['checks'] += 1
        if len(figures) != 6 or {p.stem for p in figures} != {p.stem for p in svgs}:
            result['failures'].append('SIX_PNG_SVG_PAIRS_REQUIRED')
        result['figures'] = [{'path': str(p), 'sha256': runtime.sha(p)} for p in figures]
        result['failed_count'] = len(result['failures'])
        result['status'] = 'PASS' if not result['failures'] else 'FAIL'
        limit.checkfull()
    except Exception as error:
        result.update(status='FAILED_PRESERVED', error={'type': type(error).__name__, 'message': str(error)})
    finally:
        result['resource_receipt'] = limit.close('COMPLETE' if result['status'] == 'PASS' else 'FAILED_PRESERVED', result.get('error'))
        if result['resource_receipt']['terminal_status'] != 'COMPLETE': result['status'] = result['resource_receipt']['terminal_status']
        runtime.write_new(output, result)
    print(json.dumps({k: result.get(k) for k in ('status', 'checks', 'failed_count', 'error')}))
    return 0 if result['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
