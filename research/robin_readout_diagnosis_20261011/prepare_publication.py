"""Snapshot science receipts and define the authorized publication allowlist."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BACKUP = ROOT / '.cache' / HERE.name / 'publication_before'


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    destination = HERE / 'publication_before.json'
    if destination.exists() or BACKUP.exists():
        raise FileExistsError('Publication baseline exists; do not replace it')
    audit = json.loads((HERE / 'final_audit.json').read_text(encoding='utf-8'))
    if audit['status'] != 'PASS':
        raise RuntimeError('Original science audit must pass')
    for row in audit['artifacts']:
        if sha(ROOT / row['path']) != row['sha256']:
            raise RuntimeError('Science artifact changed: ' + row['path'])
    changed = [ROOT / 'README.md', ROOT / '_docs/history/2026-10-11.md']
    changed += [HERE / n for n in ('write_report.py', 'REPORT_KO.md', 'EXECUTION_NOTE_KO.md', 'README.md')]
    backups = []
    for source in changed:
        relative = source.relative_to(ROOT).as_posix()
        backup = BACKUP / relative
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(source.read_bytes())
        backups.append({'source': relative, 'backup': backup.relative_to(ROOT).as_posix(),
                        'sha256': sha(source), 'bytes': source.stat().st_size})
    write(destination, {'created_utc': datetime.now(timezone.utc).isoformat(),
                        'scope': 'Before authorized publication; original science evidence retained',
                        'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                        'original_audit_sha256': sha(HERE / 'final_audit.json'),
                        'original_artifacts': audit['artifacts'], 'preserved_originals': backups})
    selected = {r['path'] for r in audit['artifacts']}
    selected.add((HERE / 'final_audit.json').relative_to(ROOT).as_posix())
    selected.update((HERE / n).relative_to(ROOT).as_posix() for n in (
        'prepare_publication.py', 'check_publication.py', 'publication_before.json',
        'publication_files.json', 'publication_checks.json', 'PUBLICATION_KO.md'))
    selected.update(('README.md', '_docs/history/2026-10-11.md'))
    supporting = ['research/level_k_tradeoff_20261011/sweep.py',
                  'research/level_k_tradeoff_20261011/result.json',
                  'research/level_encoder_contract_20261010/runner.py',
                  'research/level_encoder_contract_20261010/encoders.py',
                  'research/level_lag_compressor_pilot_20261010/pilot.py',
                  'research/level_matched_learned_ed_v18_20261006/runtime_v18.py']
    write(HERE / 'publication_files.json', {'scope': 'Robin study, publication checks, README index and history only',
                                          'files': sorted(selected), 'already_tracked_dependencies': supporting,
                                          'excluded': ['raw data', 'model weights', 'prediction arrays',
                                                       'cache', 'logs', 'unrelated working-tree files']})
    print(f'Preserved {len(audit["artifacts"])} science artifacts and {len(backups)} originals; selected {len(selected)} files')


if __name__ == '__main__':
    main()
