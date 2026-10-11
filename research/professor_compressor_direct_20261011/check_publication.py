"""Check the public allowlist, links, images and historical science hashes."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import json
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / 'publication_checks.json'
ROOT_SECTION = '## 교수님 제안 GTConvAE·정상 크기 TS2Vec 직접 검증 (2026-10-11)'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def tracked(key):
    return subprocess.run(['git', 'ls-files', '--error-unmatch', '--', key], cwd=ROOT,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true', help='Verify unchanged local sealed originals and backups')
    parser.add_argument('--staged', action='store_true', help='Require exact allowlist staging and bytes')
    parser.add_argument('--write-receipt', action='store_true', help='Write the initial publication receipt once')
    args = parser.parse_args()
    manifest = read(HERE / 'publication_files.json')
    before = read(HERE / 'publication_before.json')
    selected = set(manifest['files'])
    checks, links = [], []
    def check(name, passed, **details):
        checks.append({'name': name, 'passed': bool(passed), **details})
    for key in sorted(selected):
        path = ROOT / key
        pending = path == OUTPUT and args.write_receipt and not path.exists()
        check('file:' + key, path.is_file() or pending)
        check('allowlisted_asset:' + key, path.suffix in ('.py', '.json', '.csv', '.md', '.png')
              and not any(part in ('.cache', '__pycache__', 'summary_archive') for part in path.parts))
        if not path.is_file():
            continue
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8'), filename=key)
            check('python_syntax:' + key, True)
        elif path.suffix == '.json':
            read(path)
            check('json_parse:' + key, True)
        elif path.suffix == '.csv':
            rows = [row for row in csv.reader(path.open(encoding='utf-8-sig', newline='')) if row]
            check('csv_rectangular:' + key, bool(rows) and all(len(row) == len(rows[0]) for row in rows))
        elif path.suffix == '.png':
            check('PNG:' + key, path.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n')
        if path.suffix != '.md':
            continue
        content = path.read_text(encoding='utf-8')
        if key == 'README.md':
            check('root_index', ROOT_SECTION in content)
            content = content.split(ROOT_SECTION, 1)[1].split('\n## ', 1)[0] if ROOT_SECTION in content else ''
        if key.startswith('_docs/history/'):
            # History is append-only evidence; it includes preexisting local context.
            continue
        check('no_control_bytes:' + key, not any(c in content for c in ('\x0c', '\x08')))
        for flag, destination in re.findall(r'(!?)\[[^\]]*\]\(([^)]+)\)', content):
            if destination.startswith(('https://', 'http://', 'mailto:', '#')):
                continue
            value = destination.strip('<>').split('#', 1)[0]
            target = (path.parent / value).resolve()
            inside = target.is_relative_to(ROOT)
            target_key = target.relative_to(ROOT).as_posix() if inside else str(target)
            pending = target == OUTPUT and args.write_receipt and not OUTPUT.exists()
            covered = target_key in selected or tracked(target_key)
            if target.is_dir() and inside:
                covered = any(key.startswith(target_key.rstrip('/') + '/') for key in selected)
            check('link:' + key + ':' + destination,
                  inside and not re.match(r'^[A-Za-z]:[\\/]|^/|^file:', value)
                  and (target.exists() or pending) and covered,
                  target=target_key)
            links.append({'document': key, 'destination': destination, 'target': target_key, 'image': bool(flag)})
    for key in manifest['already_tracked_dependencies']:
        check('tracked_dependency:' + key, (ROOT / key).is_file() and tracked(key))
    check('original_audit_unchanged', sha(HERE / 'FINAL_EXECUTION_AUDIT.json') == before['original_audit_sha256'])
    audit = read(HERE / 'FINAL_EXECUTION_AUDIT.json')
    check('science_status', audit['status'] == 'PASS_DIRECT_SCOPE_REQUIRED_EXECUTION'
          and audit['required_current_runs_complete'] and not audit['uncompleted_required_direct_runs'])
    check('science_counts', all(audit['counts'].get(key) == expected for key, expected in {
        'canonical_representation_fits': 16, 'baseline_datasets': 2,
        'forecast_metric_rows': 76, 'reconstruction_metric_rows': 16,
        'canonical_cost_cells': 88, 'canonical_case_figures': 80, 'summary_figures': 1,
        'partial_failed_TS2_RAM': 8,
    }.items()))
    check('old_scope_incomplete', audit['old_scope']['status'] == 'SUPERSEDED_INCOMPLETE_USER_CANCELLED'
          and not audit['old_scope']['auto_resume'])
    strict = read(ROOT / audit['checks']['strict_CPU_receipt'])
    check('strict_completed_receipt', strict['summary']['campaign_complete']
          and strict['summary']['overall_saved_completed_artifacts_pass'] and strict['errors'] == [])
    rec = read(ROOT / audit['checks']['full_reconstruction_receipt'])
    check('full_reconstruction_receipt', rec['status'] == 'PASS_ALL_16_FULL_VAL_RECONSTRUCTIONS'
          and rec['expected_canonical_jobs'] == {'GT': 8, 'TS2': 8}
          and rec['verified_full_reconstructions'] == {'GT': 8, 'TS2': 8}
          and rec['new_scientific_fits'] == 0 and rec['F_forward_calls'] == 0
          and rec['protected_TEST_new_reads'] == 0)
    for key, expected in before['sealed_artifacts'].items():
        if key == '_docs/history/2026-10-11.md':
            continue
        if key in selected or args.local:
            check('sealed_science:' + key, (ROOT / key).is_file() and sha(ROOT / key) == expected)
    for row in before['public_document_copies']:
        check('public_document:' + row['public'], sha(ROOT / row['public']) == row['public_sha256'])
        check('document_original_hash:' + row['original'],
              before['sealed_artifacts'].get(row['original'], row['original_sha256']) == row['original_sha256'])
    context = read(HERE / 'public/PROTECTED_ACCESS_CONTEXT.json')
    check('old_incident_disclosure', context['original_local_sha256'] == before['excluded_original_incident_sha256']
          and not context['global_no_new_TEST_zero_claim_valid']
          and context['new_scope_does_not_reverse_old_incident']
          and context['new_Robin_Jena_direct_scope_TEST_reads'] == 0
          and not any('statistics' in key for key in context))
    check('original_incident_not_allowlisted', context['original_local_path'] not in selected)
    figures = [key for key in selected if key.endswith('.png')]
    check('canonical_81_figures', len(figures) == 81)
    original_figures = read(HERE / 'FIGURE_MANIFEST.json')
    for job in read(HERE / 'protocol.json')['jobs']:
        for relative, expected in original_figures[job['id']]['sha256'].items():
            key = (HERE / relative).relative_to(ROOT).as_posix()
            check('sealed_canonical_image:' + key, key in selected and sha(ROOT / key) == expected)
    if args.local:
        for row in before['preserved_originals']:
            check('original_backup:' + row['source'], sha(ROOT / row['backup']) == row['sha256'])
        check('local_old_incident_preserved', sha(ROOT / context['original_local_path']) == context['original_local_sha256'])
    if args.staged:
        staged = set(subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT,
                                            text=True, encoding='utf-8').splitlines())
        expected = selected - ({OUTPUT.relative_to(ROOT).as_posix()} if args.write_receipt and not OUTPUT.exists() else set())
        check('exact_staged_scope', staged == expected, missing=sorted(expected-staged), extra=sorted(staged-expected))
        for key in sorted(staged):
            blob = subprocess.check_output(['git', 'show', ':' + key], cwd=ROOT)
            check('staged_bytes:' + key, hashlib.sha256(blob).hexdigest() == sha(ROOT / key))
    failures = [row for row in checks if not row['passed']]
    receipt = {
        'status': 'FAIL' if failures else 'PASS', 'checked_UTC': datetime.now(timezone.utc).isoformat(),
        'scope': 'Publication structure and historical evidence hashes; no model/data/TEST loading',
        'base_commit': before['base_commit'], 'local_checked': args.local, 'staged_checked': args.staged,
        'checks_count': len(checks), 'failures_count': len(failures), 'failures': failures,
        'checks': checks, 'links': links,
        'files': [{'path': key, 'sha256': sha(ROOT / key), 'bytes': (ROOT / key).stat().st_size}
                  for key in sorted(selected) if (ROOT / key).is_file() and ROOT / key != OUTPUT],
        'self_exclusion': 'Receipt cannot hash itself',
    }
    if args.write_receipt:
        with OUTPUT.open('x', encoding='utf-8') as handle:
            json.dump(receipt, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
    print(f'{receipt["status"]}: {len(checks)} checks, {len(links)} links, {len(receipt["files"])} files')
    for failure in failures:
        print(failure)
    return int(bool(failures))


if __name__ == '__main__':
    raise SystemExit(main())
