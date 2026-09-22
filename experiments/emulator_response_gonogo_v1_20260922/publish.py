#!/usr/bin/env python
"""Scoped commit/push of this experiment only. Never force-push, merge, rebase or stage unrelated files.

--code-only : before the run, commit experiments/<eid>/ (code, contract, protocol) and results/<eid>/PURPOSE.md.
default     : after a PASS verification, commit experiments/<eid>/ + results/<eid>/ small artifacts, push, record receipt.
"""
from __future__ import annotations
import argparse, re, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from emulator_screen.common import *

ALLOWED = {'.py', '.json', '.jsonl', '.csv', '.md', '.txt', '.png', '.svg', '.pdf'}
DENIED = {'.npz', '.npy', '.pt', '.pth', '.safetensors', '.bin', '.zip', '.whl', '.ttf', '.otf', '.woff', '.woff2'}
SECRET = re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|'
                    r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|https?://[^\s/]+:[^\s@]+@)', re.I)
ATTRIBUTION = ('\n\nCo-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>\n'
               'Claude-Session: https://claude.ai/code/session_01YVpSC4PQGjhw1nBvY7UxC7')


def git(*args):
    return run_command(['git', *args], cwd=REPO).stdout.strip()


def preconditions():
    if git('diff', '--cached', '--name-only'): raise Blocked('UNRELATED_STAGED_FILES')
    if not re.search(r'[:/]CanelE452/hierarchical-tsfm-peft(?:\.git)?$', git('remote', 'get-url', 'origin')): raise Blocked('WRONG_REMOTE')
    branch = git('branch', '--show-current')
    if not branch: raise Blocked('DETACHED_HEAD')
    git('fetch', 'origin', branch)
    if git('rev-list', 'FETCH_HEAD..HEAD'): raise Blocked('PREEXISTING_UNPUSHED_COMMITS')
    if git('merge-base', 'HEAD', 'FETCH_HEAD') != git('rev-parse', 'FETCH_HEAD'): raise Blocked('REMOTE_ADVANCED: no automatic merge/rebase')
    return branch


def collect(roots):
    files = []
    for root in roots:
        paths = [root] if root.is_file() else sorted(root.rglob('*'))
        for p in paths:
            if not p.is_file() or any(x in p.parts for x in ('__pycache__', '.pytest_cache')): continue
            if p.is_symlink(): raise Blocked('SYMLINK '+p.name)
            if p.suffix.lower() in DENIED: raise Blocked('PRIVATE_OR_LARGE_ARTIFACT_IN_SCOPE '+p.as_posix())
            if p.suffix.lower() not in ALLOWED: continue
            if p.stat().st_size > 20_000_000: raise Blocked('ARTIFACT_TOO_LARGE '+p.name)
            if p.suffix.lower() not in {'.png', '.pdf'} and SECRET.search(p.read_text(encoding='utf-8', errors='replace')):
                raise Blocked('POSSIBLE_SECRET '+p.name)
            files.append(p.relative_to(REPO).as_posix())
    return files


def commit_push(files, message, branch, scope):
    for i in range(0, len(files), 100): git('add', '-f', '--', *files[i:i+100])
    staged = git('diff', '--cached', '--name-only').splitlines()
    if not all(any(p == s or p.startswith(s+'/') for s in scope) for p in staged): raise Blocked('UNSCOPED_STAGING')
    if staged: git('commit', '-m', message+ATTRIBUTION)
    head = git('rev-parse', 'HEAD'); git('push', 'origin', f'HEAD:refs/heads/{branch}')
    remote = git('ls-remote', 'origin', f'refs/heads/{branch}').split()[0]
    if remote != head: raise Blocked('REMOTE_HASH_NOT_CONFIRMED')
    return head, len(staged)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--code-only', action='store_true'); a = p.parse_args()
    code = HERE.relative_to(REPO).as_posix(); out = OUT.relative_to(REPO).as_posix()
    branch = preconditions()
    if a.code_only:
        files = collect([HERE] + ([OUT/'PURPOSE.md'] if (OUT/'PURPOSE.md').exists() else []))
        head, n = commit_push(files, 'Add emulator response go/no-go runner, contract and protocol before training', branch,
                              [code, out+'/PURPOSE.md'])
        print(json.dumps(dict(code_commit=head, files=n), indent=2)); return
    from emulator_screen.verify import verify
    if verify()['status'] != 'PASS': raise Blocked('PUBLISH_REQUIRES_PASS')
    if read_json(OUT/'ENVIRONMENT.json').get('authenticity') != 'REAL_MODEL_RUN': raise Blocked('NOT_A_REAL_RUN')
    files = collect([HERE, OUT])
    status = read_json(OUT/'DECISION.json')['status']
    artifact, n = commit_push(files, f'Complete emulator response go/no-go on Jena ({status}) with reports and verification', branch, [code, out])
    url = f'https://github.com/CanelE452/hierarchical-tsfm-peft/blob/{artifact}/{out}/REPORT_KO.md'
    write_json(OUT/'PUSH_RECEIPT.json', dict(artifact_commit=artifact, remote_confirmed=True, branch=branch, report=url, status=status,
        utc=time.time(), note='This receipt references the artifact commit, not its own later receipt commit.'))
    final, _ = commit_push([f'{out}/PUSH_RECEIPT.json'], 'Record verified emulator go/no-go artifact commit', branch, [out])
    print(json.dumps(dict(report=url, artifact_commit=artifact, final_remote_commit=final, status=status, files=n), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try: main()
    except Blocked as e:
        print('PUSH_BLOCKED:', e, file=sys.stderr); raise SystemExit(2)
