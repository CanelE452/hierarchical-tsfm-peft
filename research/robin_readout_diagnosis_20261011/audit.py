"""Capture immutable prior artifacts and audit this local study without fitting."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / '.cache' / HERE.name
PARENT = ROOT / 'research/level_k_tradeoff_20261011'
HISTORY = ROOT / '_docs/history/2026-10-11.md'


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT).decode('utf-8').strip()


def resolve(path):
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def capture():
    destination = HERE / 'preservation.before.json'
    if destination.exists():
        raise RuntimeError('Preservation baseline already exists')
    CACHE.mkdir(parents=True, exist_ok=True)
    paths = set(git('ls-files').splitlines())
    for name in ('parent_preservation.before.json', 'final_verification.json'):
        paths.update(row['path'] for row in read(PARENT / name)['artifacts'])
    paths.update(p.relative_to(ROOT).as_posix() for p in PARENT.rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.log')
    paths.update(p.relative_to(ROOT).as_posix() for p in (ROOT / '.cache' / PARENT.name).rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.log')
    paths.discard(HISTORY.relative_to(ROOT).as_posix())
    files = {}
    for key in sorted(paths):
        path = resolve(key)
        if not path.is_file():
            raise RuntimeError(f'Missing preserved artifact: {key}')
        files[key] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    history_bytes = HISTORY.read_bytes()
    (CACHE / 'history.before.txt').write_bytes(history_bytes)
    write(destination, {'created_utc': datetime.now(timezone.utc).isoformat(),
                        'git_head': git('rev-parse', 'HEAD'), 'files': files,
                        'history': {'path': HISTORY.relative_to(ROOT).as_posix(),
                                    'bytes': len(history_bytes), 'sha256': hashlib.sha256(history_bytes).hexdigest()}})
    print(f'PRESERVED {len(files)} files; history {len(history_bytes)} bytes; HEAD {git("rev-parse", "HEAD")}')


def seal():
    path = HERE / 'source_seal.json'
    if path.exists():
        raise RuntimeError('Science source seal already exists')
    files = [HERE / name for name in ('protocol.json', 'run.py', 'test_run.py')]
    files += [ROOT / name for name in (
        'research/level_k_tradeoff_20261011/sweep.py',
        'research/level_encoder_contract_20261010/runner.py',
        'research/level_encoder_contract_20261010/encoders.py',
        'research/level_lag_compressor_pilot_20261010/pilot.py',
        'research/level_matched_learned_ed_v18_20261006/runtime_v18.py')]
    write(path, {'created_utc': datetime.now(timezone.utc).isoformat(),
                 'scope': 'Before new scientific fits',
                 'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in files}})
    print(f'SEALED {len(files)} sources')


def check():
    checks = []
    def record(name, passed):
        checks.append({'name': name, 'passed': bool(passed)})
    baseline = read(HERE / 'preservation.before.json')
    for key, row in baseline['files'].items():
        path = resolve(key)
        record('preserved:' + key, path.is_file() and sha(path) == row['sha256'])
    record('git_head', git('rev-parse', 'HEAD') == baseline['git_head'])
    history = HISTORY.read_bytes()
    prefix = history[:baseline['history']['bytes']]
    record('history_prefix', hashlib.sha256(prefix).hexdigest() == baseline['history']['sha256'])
    for key, expected in read(HERE / 'source_seal.json')['files'].items():
        record('science_source:' + key, sha(resolve(key)) == expected)
    verification = read(HERE / 'verification.json')
    record('independent_verification', verification['status'] == 'PASS')
    result = read(HERE / 'result.json')
    record('science_complete', result['status'] == 'complete')
    for path in HERE.glob('*.md'):
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            target = target.strip('<>').split('#', 1)[0]
            if not target or re.match(r'https?://', target):
                continue
            destination = (path.parent / target).resolve()
            record('link:' + path.name + ':' + target,
                   not Path(target).is_absolute() and
                   (destination.is_file() or destination == HERE / 'final_audit.json'))
    artifacts = [{'path': p.relative_to(ROOT).as_posix(), 'sha256': sha(p), 'bytes': p.stat().st_size}
                 for p in sorted(HERE.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
                 and p.name != 'final_audit.json' and p.suffix != '.log']
    failed = [row for row in checks if not row['passed']]
    write(HERE / 'final_audit.json', {'status': 'FAIL' if failed else 'PASS', 'checks': checks,
                                    'check_count': len(checks), 'failures': failed, 'artifacts': artifacts,
                                    'artifact_count': len(artifacts)})
    print(f'{"FAIL" if failed else "PASS"}: {len(checks)} checks; {len(artifacts)} new artifacts')
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['capture', 'seal', 'check'])
    args = parser.parse_args()
    {'capture': capture, 'seal': seal, 'check': check}[args.mode]()
