import hashlib
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'reports/k_tau_rule_20261023_v1'
ORIGINAL = Path('E:/CODING/proj/hierarchical-tsfm-peft')


def digest(value):
    return hashlib.sha256(value).hexdigest()


before = json.loads((REPORT / 'PRE_HASHES.json').read_text(encoding='utf-8-sig'))
files = []
for relative, expected in before['files'].items():
    target = ROOT / relative
    actual = digest(target.read_bytes()) if target.is_file() else None
    files.append(dict(path=relative, expected_sha256=expected, actual_sha256=actual,
                      unchanged=actual == expected))
original_head = subprocess.check_output(['git', '-C', str(ORIGINAL), 'rev-parse', 'HEAD'], text=True).strip()
args = ['status', '--porcelain=v1', '-z']
raw = subprocess.check_output(['git', '-C', str(ORIGINAL), *args])
actual = digest(raw)
status_checks = [dict(command=args, serialization='raw_NUL_delimited_bytes', sha256=actual,
                      matches=actual == before['original_worktree_status_sha256'])]
allowed = ('research/k_tau_rule_20261023_v1/', 'reports/k_tau_rule_20261023_v1/',
           '.cache/k_tau_rule_20261023_v1/')
base_diff = subprocess.check_output(['git', 'diff', '--name-only', before['head']], cwd=ROOT, text=True).splitlines()
untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard'], cwd=ROOT, text=True).splitlines()
outside = sorted(path for path in set(base_diff + untracked) if not path.startswith(allowed))
passed = (all(row['unchanged'] for row in files) and original_head == before['original_worktree_head']
          and any(row['matches'] for row in status_checks) and not outside)
result = dict(status='PASS' if passed else 'FAIL', checked_unix=time.time(),
    checked_existing_files=len(files), all_selected_existing_file_hashes_unchanged=all(row['unchanged'] for row in files),
    original_worktree_head=original_head, original_worktree_head_unchanged=original_head == before['original_worktree_head'],
    original_worktree_status_unchanged=any(row['matches'] for row in status_checks),
    original_worktree_status_checks=status_checks, scope=before['scope'], new_namespace_only=not outside,
    outside_namespace_paths=outside, raw_data_or_T_files_hashed=0, files=files)
destination = REPORT / 'FINAL_HASH_VALIDATION.json'
destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k not in ('files','original_worktree_status_checks')}, ensure_ascii=False))
raise SystemExit(0 if passed else 1)
