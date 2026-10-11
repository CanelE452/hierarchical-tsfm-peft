"""Prepare public document copies without changing the sealed experiment."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BACKUP = ROOT / '.cache' / HERE.name / 'publication_before'
PUBLIC = HERE / 'public'
DOCUMENTS = (
    'REPORT_KO.md', 'INTERIM_RESULTS_KO.md', 'SOURCE_README.md',
    'GOAL_AND_METHOD_DELTA.md', 'GT_METHOD_DELTA.md', 'TS2_METHOD_DELTA.md',
    'DATA_NOTE_KO.md', 'METHOD_FAILURE_DIAGNOSIS_KO.md',
    'MEM_RETRY_NOTE_KO.md', 'NEW_GOAL_CODE_AUDIT_KO.md',
)
INCIDENT = 'research/level_compressor_fullscreen_20261011/PROTECTED_ACCESS_INCIDENT.json'
ROOT_SECTION = '## 교수님 제안 GTConvAE·정상 크기 TS2Vec 직접 검증 (2026-10-11)'


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    if (HERE / 'publication_before.json').exists() or BACKUP.exists() or PUBLIC.exists():
        raise FileExistsError('Publication preparation already exists; preserve it')
    audit_path = HERE / 'FINAL_EXECUTION_AUDIT.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    if audit['status'] != 'PASS_DIRECT_SCOPE_REQUIRED_EXECUTION':
        raise RuntimeError('The direct experiment audit must pass first')
    preserved = []
    for key, expected in {**audit['file_sha256'], **audit['current_code_sha256']}.items():
        if key == '_docs/history/2026-10-11.md':
            continue
        if sha(ROOT / key) != expected:
            raise RuntimeError('Sealed artifact changed: ' + key)
    for source in [audit_path, ROOT / 'README.md', *(HERE / name for name in DOCUMENTS)]:
        key = source.relative_to(ROOT).as_posix()
        backup = BACKUP / key
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_bytes(source.read_bytes())
        preserved.append({'source': key, 'backup': backup.relative_to(ROOT).as_posix(),
                          'sha256': sha(source), 'bytes': source.stat().st_size})
    PUBLIC.mkdir()
    incident = json.loads((ROOT / INCIDENT).read_text(encoding='utf-8'))
    write(PUBLIC / 'PROTECTED_ACCESS_CONTEXT.json', {
        'schema': 'public_protected_access_context_v1',
        'scope': 'Disclosure of an old campaign access violation; no target statistics redistributed',
        'original_local_path': INCIDENT,
        'original_local_sha256': sha(ROOT / INCIDENT),
        'incident_id': incident['incident_id'],
        'recorded_at_kst': incident['recorded_at_kst'],
        'status': incident['status'],
        'old_campaign_guard': 'FAILED: full Electricity GIFT target arrays were materialized twice',
        'model_training': False, 'model_inference': False, 'model_scoring': False,
        'method_or_hyperparameter_selection': False,
        'global_no_new_TEST_zero_claim_valid': False,
        'new_Robin_Jena_direct_scope_TEST_reads': 0,
        'new_scope_does_not_reverse_old_incident': True,
        'omitted': ['protected-target aggregate diagnostics', 'local diagnostic command payloads'],
        'original_record_retained_locally_unchanged': True,
    })
    copies = []
    for name in DOCUMENTS:
        original = HERE / name
        content = original.read_text(encoding='utf-8')
        def replace(match):
            flag, label, destination = match.groups()
            if destination.startswith(('https://', 'http://', 'mailto:', '#')):
                return match.group(0)
            value, separator, anchor = destination.strip('<>').partition('#')
            target = (HERE / value).resolve()
            if target == ROOT / INCIDENT:
                rewritten = 'PROTECTED_ACCESS_CONTEXT.json'
            elif target.parent == HERE and target.name in DOCUMENTS:
                rewritten = target.name
            else:
                rewritten = Path(os.path.relpath(target, PUBLIC)).as_posix()
            if separator:
                rewritten += '#' + anchor
            return f'{flag}[{label}]({rewritten})'
        public_content = re.sub(r'(!?)\[([^\]]*)\]\(([^)]+)\)', replace, content)
        notice = (
            '> 게시용 사본입니다. 실험 완료 당시 원본 문서의 SHA256은 '
            f'`{sha(original)}`입니다. 본문 수치·판단을 유지하고 링크만 GitHub 경로로 조정했습니다. '
            '원문의 “이번 작업에서 push하지 않음”은 과학 실험 완료 당시 기록이며, '
            '후속 사용자 승인으로 이 공개 사본을 게시합니다. 구 TEST 사고는 '
            '[범위·원본 SHA](PROTECTED_ACCESS_CONTEXT.json)만 공개하며 보호된 통계는 재게시하지 않습니다.\n\n'
        )
        destination = PUBLIC / name
        destination.write_text(notice + public_content, encoding='utf-8')
        copies.append({'original': original.relative_to(ROOT).as_posix(),
                       'original_sha256': sha(original),
                       'public': destination.relative_to(ROOT).as_posix(),
                       'public_sha256': sha(destination),
                       'transformation': 'Added publication context; local links relocated; old incident disclosure minimized'})
    selected = set()
    def include(path):
        selected.add(path.relative_to(ROOT).as_posix())
    for path in HERE.iterdir():
        if path.is_file() and path.suffix in ('.py', '.json', '.csv'):
            include(path)
    for folder in ('runs', 'baselines', 'resources', 'execution_environment', 'vendor_reference'):
        for path in (HERE / folder).rglob('*'):
            if path.is_file() and path.suffix in ('.json', '.py'):
                include(path)
    for path in (HERE / 'source_snapshots').rglob('*'):
        if path.is_file() and path.suffix in ('.json', '.py'):
            include(path)
    for path in (HERE / 'reconstruction_audit').rglob('*.json'):
        include(path)
    include(ROOT / audit['checks']['strict_CPU_receipt'])
    for path in PUBLIC.iterdir():
        include(path)
    protocol = json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))
    for job in protocol['jobs']:
        for suffix in ('forecast', 'reconstruction', 'latent_curves', 'latent_heatmap', 'horizon_errors'):
            include(HERE / 'figures' / f'{job["id"]}_{suffix}.png')
    include(HERE / 'figures' / 'METHOD_COMPARISON.png')
    selected.update((HERE / name).relative_to(ROOT).as_posix() for name in (
        'README.md', 'PUBLICATION_KO.md', 'publication_before.json',
        'publication_files.json', 'publication_checks.json'))
    selected.update(('README.md', '_docs/history/2026-10-11.md'))
    dependencies = [
        'research/level_matched_learned_ed_v18_20261006/runtime_v18.py',
        'research/level_matched_learned_ed_v18_20261006/data_contract_robin.json',
        'research/level_matched_learned_ed_v18_20261006/data_contract_jena.json',
        'research/level_k_tradeoff_20261011/vendor/ts2vec/ts2vec.py',
        'research/level_k_tradeoff_20261011/vendor/ts2vec/utils.py',
        'research/level_k_tradeoff_20261011/vendor/ts2vec/models/__init__.py',
        'research/level_k_tradeoff_20261011/vendor/ts2vec/models/encoder.py',
        'research/level_k_tradeoff_20261011/vendor/ts2vec/models/dilated_conv.py',
        'research/level_k_tradeoff_20261011/vendor/ts2vec/models/losses.py',
    ]
    write(HERE / 'publication_before.json', {
        'created_UTC': datetime.now(timezone.utc).isoformat(),
        'scope': 'Publication copies only; sealed science originals unchanged',
        'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'original_audit_sha256': sha(audit_path),
        'sealed_artifacts': {**audit['file_sha256'], **audit['current_code_sha256']},
        'preserved_originals': preserved, 'public_document_copies': copies,
        'excluded_original_incident_sha256': sha(ROOT / INCIDENT),
    })
    write(HERE / 'publication_files.json', {
        'scope': 'Completed direct GTConvAE/TS2Vec experiment; publication copies; README/history index',
        'files': sorted(selected), 'already_tracked_dependencies': dependencies,
        'canonical_figures': 81, 'original_documents_not_committed': list(DOCUMENTS),
        'excluded': ['raw data', 'weights/checkpoints', 'prediction/diagnostic arrays', '.cache',
                     'logs', 'image archives', 'old protected-target diagnostic statistics',
                     'unrelated working-tree files', 'new pending matched Jena experiment'],
    })
    print(f'Prepared {len(copies)} public document copies and {len(selected)} explicit files')


if __name__ == '__main__':
    main()
