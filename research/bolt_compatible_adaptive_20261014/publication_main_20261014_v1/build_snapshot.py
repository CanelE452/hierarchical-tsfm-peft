"""Publish the completed diagnostic evidence without running experiments."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import subprocess

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[2]
ALLOWED = {'.py', '.md', '.json', '.csv', '.png', '.svg', '.txt'}
LINK = re.compile(r'(!?)\[([^\]]*)\]\(([^)]+)\)')
EXPECTED_REPORT_SHA = '938ab3d7604d5bfb3f9657bfe8587407028fe2be2374d0fd0050e031d592ba5e'
REPORT = BASE / 'reports/first_and_dense_v3'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path):
    return path.relative_to(ROOT).as_posix()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')


def selected_files():
    selected = {p for p in BASE.iterdir() if p.is_file() and p.suffix in ALLOWED}
    for directory in ('runs', 'cost_runs', 'audits', 'resources_v1', 'reports/first_and_dense_v3'):
        selected.update(p for p in (BASE / directory).rglob('*')
                        if p.is_file() and p.suffix in ALLOWED
                        and '__pycache__' not in p.parts and p.name != 'ACTIVE.json')
    # Retain old report provenance, without promoting its obsolete Markdown or figures.
    for directory in ('reports/first_and_dense_v1', 'reports/first_and_dense_v2'):
        selected.update(p for p in (BASE / directory).rglob('*')
                        if p.is_file() and p.suffix in {'.json', '.py', '.csv'}
                        and '__pycache__' not in p.parts)
    for name in ('GT_METHOD_DELTA.md', 'TS2_METHOD_DELTA.md', 'METHOD_FAILURE_DIAGNOSIS_KO.md'):
        selected.add(ROOT / 'research/professor_compressor_direct_20261011' / name)
    selected.add(ROOT / 'RESEARCH_RESULTS_20261014_KO.md')
    selected.add(ROOT / 'README.md')
    selected.update(p for p in HERE.rglob('*') if p.is_file() and p.suffix in ALLOWED)
    return selected


def resolve_link(source, target):
    if target.startswith(('https:', 'http:', '#', 'mailto:', 'codex:')):
        return None
    return (source.parent / target.split('#', 1)[0]).resolve()


def make_documents():
    folder = HERE / 'reports'
    folder.mkdir(exist_ok=False)
    sources = [('adaptive_REPORT_KO.md', REPORT / 'REPORT_KO.md'),
               ('PRESENTATION_BRIEF_KO.md', REPORT / 'PRESENTATION_BRIEF_KO.md'),
               ('CAMPAIGN_CLOSURE_KO.md', BASE / 'CAMPAIGN_CLOSURE_KO.md'),
               ('DECISION_LEDGER_FINAL_KO.md', BASE / 'DECISION_LEDGER_FINAL_KO.md')]
    selected = selected_files()
    mappings = {source: folder / name for name, source in sources}
    originals = []
    for name, source in sources:
        destination = folder / name
        original_sha = sha(source)
        def rewrite(match):
            image, label, target = match.groups()
            path = resolve_link(source, target)
            if path is None:
                return match.group(0)
            fragment = '#' + target.split('#', 1)[1] if '#' in target else ''
            mapped = mappings.get(path, path)
            tracked = path.is_relative_to(ROOT) and subprocess.run(
                ['git', 'ls-files', '--error-unmatch', rel(path)], capture_output=True).returncode == 0
            if path in selected or path in mappings or tracked:
                return image + '[' + label + '](' + Path(os.path.relpath(mapped, destination.parent)).as_posix() + fragment + ')'
            return label + ' — 로컬 재현 자료 `' + (rel(path) if path.is_relative_to(ROOT) else str(path)) + '`'
        body = LINK.sub(rewrite, source.read_text(encoding='utf-8-sig'))
        note = ('> 사용자의 명시 요청에 따른 GitHub 게시용 링크 사본입니다. 수치와 본문은 보존했습니다. '
                '원문 SHA256 `' + original_sha + '`. 원문에 있는 게시 금지/미게시 표현은 실험 당시 기록입니다.\n\n')
        destination.write_text(note + body, encoding='utf-8', newline='\n')
        if sha(source) != original_sha:
            raise RuntimeError('ORIGINAL_CHANGED:' + rel(source))
        originals.append({'path': rel(source), 'sha256': original_sha, 'publication_copy': rel(destination)})
    write(HERE / 'ORIGINAL_REPORTS.json', originals)


def build(refresh=False):
    if sha(REPORT / 'MANIFEST.json') != EXPECTED_REPORT_SHA:
        raise RuntimeError('FROZEN_REPORT_CHANGED')
    for name in ('FIRST_READOUT_AUDIT_v2.json', 'EXPANDED_READOUT_AUDIT_v2.json',
                 'COST_AUDIT_v2.json', 'EXPANDED_COST_AUDIT_v1.json', 'FINAL_REPORT_ARTIFACT_AUDIT_v2.json'):
        value = json.loads((BASE / 'audits' / name).read_text(encoding='utf-8-sig'))
        if value.get('status') != 'PASS':
            raise RuntimeError('MISSING_ACTUAL_PASS:' + name)
    if not refresh:
        make_documents()
    write(HERE / 'LINK_AUDIT.json', {})
    write(HERE / 'FILES.json', {})
    selected = selected_files()
    tracked = {ROOT / p for p in subprocess.check_output(['git', 'ls-files'], text=True, encoding='utf-8').splitlines()}
    rows = []
    for source in sorted(p for p in selected if p.suffix == '.md'):
        for image, label, target in LINK.findall(source.read_text(encoding='utf-8-sig')):
            path = resolve_link(source, target)
            if path is None:
                continue
            okay = path.is_file() and (path in selected or path in tracked)
            if path.is_dir():
                okay = any(p.is_relative_to(path) for p in selected | tracked)
            rows.append({'source': rel(source), 'target': target, 'is_image': bool(image), 'pass': okay})
    failed = [row for row in rows if not row['pass']]
    audit = {'status': 'PASS' if not failed else 'FAIL', 'checks': len(rows),
             'image_links': sum(row['is_image'] for row in rows), 'failed_count': len(failed),
             'links': rows, 'failures': failed}
    write(HERE / 'LINK_AUDIT.json', audit)
    records = [{'path': rel(p), 'bytes': p.stat().st_size, 'sha256': sha(p)}
               for p in sorted(selected) if p != HERE / 'FILES.json']
    manifest = {'status': 'READY_FOR_EXPLICIT_STAGING' if not failed else 'FAIL',
                'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'explicit_user_requested_publication': True, 'scientific_runs_restarted': False,
                'files': records, 'file_count_excluding_manifest': len(records),
                'bytes': sum(row['bytes'] for row in records)}
    write(HERE / 'FILES.json', manifest)
    print(json.dumps({'status': manifest['status'], 'files': len(records) + 1,
                      'bytes': manifest['bytes'], 'links': len(rows), 'failed': len(failed)}))
    if failed:
        print(json.dumps(failed, ensure_ascii=False))
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh', action='store_true', help='Refresh file hashes after verification tools are added.')
    build(parser.parse_args().refresh)
