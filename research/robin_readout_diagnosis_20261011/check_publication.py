"""Validate GitHub paths and sealed evidence, with optional local/staged checks."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import ast
import hashlib
import json
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / 'publication_checks.json'
ROOT_SECTION = '## Robin 원채널 디코더 진단과 개선 비교 (2026-10-11)'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def resolve(key):
    path = Path(key)
    return path if path.is_absolute() else ROOT / path


def tracked(key):
    return subprocess.run(['git', 'ls-files', '--error-unmatch', '--', key], cwd=ROOT,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true', help='Also verify local arrays and preserved originals')
    parser.add_argument('--staged', action='store_true')
    parser.add_argument('--write-receipt', action='store_true')
    args = parser.parse_args()
    manifest = read(HERE / 'publication_files.json')
    selected = set(manifest['files'])
    before = read(HERE / 'publication_before.json')
    checks, links = [], []
    def check(name, passed, **details):
        checks.append({'name': name, 'passed': bool(passed), **details})
    for key in sorted(selected):
        path = ROOT / key
        pending = path == OUTPUT and args.write_receipt and not path.exists()
        check('file:' + key, path.is_file() or pending)
        check('publication_asset:' + key, path.suffix not in
              ('.pt', '.pth', '.safetensors', '.ckpt', '.npz', '.npy', '.pkl', '.log', '.tmp')
              and '.cache' not in path.parts and '__pycache__' not in path.parts)
        if not path.is_file():
            continue
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8'), filename=key)
            check('python_syntax:' + key, True)
        if path.suffix == '.png':
            check('PNG:' + key, path.read_bytes()[:8] == b'\x89PNG\r\n\x1a\n')
        if path.suffix == '.pdf':
            check('PDF:' + key, path.read_bytes()[:5] == b'%PDF-')
        if path.suffix != '.md':
            continue
        content = path.read_text(encoding='utf-8')
        if key == 'README.md':
            content = content.split(ROOT_SECTION, 1)[1].split('\n## ', 1)[0]
        check('no_control_bytes:' + key, not any(c in content for c in ('\t', '\x0c', '\x08')))
        for image_flag, destination in re.findall(r'(!?)\[[^\]]*\]\(([^)]+)\)', content):
            if destination.startswith(('https://', 'http://', 'mailto:', '#')):
                continue
            value = destination.strip('<>').split('#', 1)[0]
            target = (path.parent / value).resolve()
            inside = target.is_relative_to(ROOT)
            target_key = target.relative_to(ROOT).as_posix() if inside else str(target)
            pending = target == OUTPUT and args.write_receipt and not target.exists()
            check('link:' + key + ':' + destination,
                  inside and not re.match(r'^[A-Za-z]:[\\/]|^/|^file:', value)
                  and (target.is_file() or pending) and (target_key in selected or tracked(target_key)))
            links.append({'document': key, 'destination': destination, 'target': target_key, 'image': bool(image_flag)})
    for key in manifest['already_tracked_dependencies']:
        check('tracked_dependency:' + key, (ROOT / key).is_file() and tracked(key))
    for key, expected in read(HERE / 'source_seal.json')['files'].items():
        check('science_source:' + key, sha(ROOT / key) == expected)
    for name, expected in read(HERE / 'summary.json')['inputs'].items():
        check('summary_input:' + name, sha(HERE / name) == expected)
    check('original_science_audit', sha(HERE / 'final_audit.json') == before['original_audit_sha256'])
    verification = read(HERE / 'verification.json')
    check('independent_science_verification', verification['status'] == 'PASS'
          and verification['checks_count'] == 21415 and verification['failures_count'] == 0
          and len(verification['warnings']) == 4)
    result = read(HERE / 'result.json')
    check('science_counts', result['status'] == 'complete' and result['counters'] == {
        'candidate_fits': 230, 'channel_solves': 3910, 'reused_full_train_candidates': 10,
        'new_backbone_calls': 0, 'new_test_predictions': 0, 'gpu_jobs': 0})
    changed = {row['source']: row for row in before['preserved_originals']}
    for row in before['original_artifacts']:
        key = row['path']
        if key not in changed:
            check('unchanged_science_artifact:' + key, sha(ROOT / key) == row['sha256'])
    if args.local:
        for key, row in changed.items():
            check('original_backup:' + key, sha(ROOT / row['backup']) == row['sha256'])
        for row in before['original_artifacts']:
            key = row['path']
            original = ROOT / changed[key]['backup'] if key in changed else ROOT / key
            check('preserved_original_science:' + key, sha(original) == row['sha256'])
        prior = read(HERE / 'preservation.before.json')
        for key, row in prior['files'].items():
            original = ROOT / changed[key]['backup'] if key in changed else resolve(key)
            check('preserved_parent:' + key, sha(original) == row['sha256'])
        history = changed['_docs/history/2026-10-11.md']
        check('history_append_only', hashlib.sha256((ROOT / history['source']).read_bytes()
              [:history['bytes']]).hexdigest() == history['sha256'])
        check('local_precommit_head', subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
              text=True).strip() == before['base_commit'])
    if args.staged:
        staged = set(subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=ROOT,
                                             text=True, encoding='utf-8').splitlines())
        expected = selected - ({OUTPUT.relative_to(ROOT).as_posix()} if args.write_receipt and not OUTPUT.exists() else set())
        check('exact_staged_scope', staged == expected,
              missing=sorted(expected - staged), extra=sorted(staged - expected))
        for key in sorted(staged):
            blob = subprocess.check_output(['git', 'show', ':' + key], cwd=ROOT)
            check('staged_bytes:' + key, hashlib.sha256(blob).hexdigest() == sha(ROOT / key))
    failures = [c for c in checks if not c['passed']]
    files = [{'path': key, 'sha256': sha(ROOT / key), 'bytes': (ROOT / key).stat().st_size}
             for key in sorted(selected) if (ROOT / key).is_file() and ROOT / key != OUTPUT]
    receipt = {'status': 'FAIL' if failures else 'PASS', 'checked_utc': datetime.now(timezone.utc).isoformat(),
               'scope': 'Publication only; original scientific audit is a historical snapshot',
               'base_commit': before['base_commit'], 'local_checked': args.local, 'staged_checked': args.staged,
               'checks_count': len(checks), 'failures_count': len(failures), 'failures': failures,
               'checks': checks, 'links': links, 'files': files, 'self_exclusion': 'Receipt cannot hash itself'}
    if args.write_receipt:
        with OUTPUT.open('x', encoding='utf-8') as handle:
            json.dump(receipt, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
    print(f'{receipt["status"]}: {len(checks)} checks, {len(links)} local links, {len(files)} files')
    for failure in failures:
        print(failure)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
