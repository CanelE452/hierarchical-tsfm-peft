"""Report file/hash/link checks, without scientific array or model reads."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(report, output, previous=None):
    from continuation_runtime_v2 import ContinuationLimit
    limit = None
    result = {'status': 'RUNNING', 'checks': 0, 'failed_count': 0, 'failures': [],
              'scope': 'report file hashes/links/table rows and unchanged earlier tables; no scientific inference',
              'new_F_forward': False, 'independent_generalization': False}

    def check(name, passed):
        result['checks'] += 1
        if not bool(passed):
            result['failed_count'] += 1
            result['failures'].append(name)

    try:
        if Path(output).exists():
            raise FileExistsError(output)
        limit = ContinuationLimit('report_files_' + report.name + '_v1', gpu=False)
        manifest_path = report / 'MANIFEST.json'
        manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
        result['report_manifest'] = {'path': str(manifest_path), 'sha256': sha(manifest_path)}
        check('generator_complete', manifest.get('generator_result_status') == 'COMPLETE')
        artifacts = manifest.get('artifacts', [])
        report_entries = [entry for entry in artifacts if entry.get('path') == 'REPORT_KO.md']
        check('manifest_report_md_once', len(report_entries) == 1)
        for entry in artifacts:
            path = report / entry['path']
            check('exists:' + entry['path'], path.is_file())
            check('hash:' + entry['path'], path.is_file() and sha(path) == entry['sha256'])
            check('bytes:' + entry['path'], path.is_file() and path.stat().st_size == entry['bytes'])
            limit.shared.check()
        text_path = report / (report_entries[0]['path'] if report_entries else 'REPORT_KO.md')
        text = text_path.read_text(encoding='utf-8')
        result['report_md'] = {'path': str(text_path), 'sha256': sha(text_path)}
        if report_entries:
            check('report_md_manifest_hash', result['report_md']['sha256'] == report_entries[0]['sha256'])
        links = re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text)
        relative = [link for link in links if not link.startswith(('https://', 'http://', '#'))]
        for link in relative:
            raw = link.strip('<>').split('#')[0]
            check('relative:' + link, not Path(raw).is_absolute() and (report / raw).is_file())
        result['relative_links_checked'] = len(relative)
        check('six_png', len(list((report/'figures').glob('*.png'))) == 6)
        check('six_svg', len(list((report/'figures').glob('*.svg'))) == 6)
        tables = {}
        for path in sorted((report/'tables').glob('*.csv')):
            with path.open(encoding='utf-8-sig', newline='') as stream:
                tables[path.name] = sum(1 for _ in csv.DictReader(stream))
            check('nonempty:' + path.name, tables[path.name] > 0)
            if previous is not None and path.name != 'resources.csv':
                old = previous/'tables'/path.name
                check('earlier_table_unchanged:' + path.name, old.is_file() and sha(old) == sha(path))
        result['table_rows'] = tables
        result['root_visual_inspection_required'] = True
        result['visual_inspection_claimed_by_checker'] = False
        check('manifest_unchanged', sha(manifest_path) == result['report_manifest']['sha256'])
        result['status'] = 'PASS_FILE_AND_LINK_CHECKS' if result['failed_count'] == 0 else 'FAIL_PRESERVED'
        limit.check()
    except Exception as error:
        result.update(status='FAILED_PRESERVED', error={'type': type(error).__name__, 'message': str(error)})
    finally:
        if limit:
            try:
                result['shared_resource_receipt'] = limit.close('COMPLETE' if result['status'].startswith('PASS') else 'FAILED_PRESERVED', result.get('error'))
                if result['shared_resource_receipt']['terminal_status'] != 'COMPLETE':
                    result['status'] = result['shared_resource_receipt']['terminal_status']
                    result['terminal_error'] = result['shared_resource_receipt'].get('terminal_error')
            except Exception as close_error:
                result.update(status='FAILED_PRESERVED', resource_close_unavailable={
                    'type': type(close_error).__name__, 'message': str(close_error)})
        result['verification_source_sha256'] = sha(__file__)
        with Path(output).open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--previous', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    value = verify(args.report.resolve(), args.out.resolve(), args.previous.resolve() if args.previous else None)
    print(json.dumps({k: value[k] for k in ('status', 'checks', 'failed_count')}))
    raise SystemExit(0 if value['status'].startswith('PASS') else 1)
