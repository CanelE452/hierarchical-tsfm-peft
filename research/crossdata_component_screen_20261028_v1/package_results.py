"""Package reviewed scientific evidence; never create or include presentation files.

ROOT_RESULTS_GATE.json schema (all SHA256 values refer to current file bytes):
  status: PASS
  scope: EXPERIMENT_RESULTS_ONLY
  analysis_sha256: ANALYSIS_VALIDATION.json SHA256
  completion_sha256: EXECUTION_COMPLETION.json SHA256
  figures_visual_review:
    FIG_TIME_ONLY.png: {status: PASS, sha256: ...}
    FIG_LAG0.png: {status: PASS, sha256: ...}
    FIG_E_FIXED.png: {status: PASS, sha256: ...}
  reviewed_files: {repository/relative/path: SHA256, ...}
  allow_partial: true       # required unless all 24 planned jobs completed
  partial_reason: ...       # required with allow_partial

The gate's reviewed_files must bind all result tables, interpretation documents,
analysis/completion and three figures. An integrity-PASS analysis with a complete
matched primary cohort is required even for explicitly approved partial delivery.
Public pretraining archives are retained; archived .cache files are excluded.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

from . import runtime


NAME = 'crossdata_component_screen_20261028_v1'
REPORT = 'reports/' + NAME + '/'
SOURCE = 'research/' + NAME + '/'
FIGURES = ('FIG_TIME_ONLY.png', 'FIG_LAG0.png', 'FIG_E_FIXED.png')
TABLES = ('RUNS.csv', 'ALL_NEW_RUNS.csv', 'SUMMARY.csv', 'CONTRASTS.csv',
          'PAIRED_DIFFERENCES.csv', 'JOB_STATUS.csv', 'COVERAGE.csv')
REVIEWED = TABLES + ('RESULTS_KO.md', 'EVIDENCE_HANDOFF_KO.md',
                     'ANALYSIS_VALIDATION.json', 'EXECUTION_COMPLETION.json') + FIGURES
REQUIRED = REVIEWED + ('ROOT_RESULTS_GATE.json', 'RUN_PLAN.json', 'PREREGISTRATION.json',
    'PREREGISTRATION.md', 'REGISTRATION_SEAL.json', 'DATA_REGISTRY.json',
    'REFERENCE_RUNS.json', 'PRESERVATION_BASELINE.json', 'UNIT_TESTS.json')
MANIFEST = 'RESULTS_BUNDLE_MANIFEST.json'
RECEIPT = 'RESULTS_BUNDLE_RECEIPT.json'
ZIP_NAME = NAME + '_results_bundle.zip'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def safe_path(repo, relative):
    require(isinstance(relative, str) and '\\' not in relative and ':' not in relative,
            'Evidence path must be a repository-relative slash path')
    pure = PurePosixPath(relative)
    require(not pure.is_absolute() and pure.parts and '..' not in pure.parts,
            'Evidence path escapes the repository: ' + relative)
    path = repo / relative
    require(path.resolve().is_relative_to(repo.resolve()), 'Resolved evidence path escapes the repository')
    require(not path.is_symlink(), 'Symlink evidence is not packaged')
    return path


def exclusion_reason(relative):
    pure = PurePosixPath(relative)
    parts = {part.lower() for part in pure.parts}
    name = pure.name.lower()
    if parts & {'.cache', '__pycache__', '.pytest_cache', 'private', 'presentation',
                'predictions', 'checkpoints', 'raw_data', 'datasets'}:
        return 'Private/cache/raw-data/presentation directory'
    if (name in {'policy.json', 'identity.json', 'inventory.json'}
            or name.startswith(('private_', 'raw_counter', 'raw_host', 'raw_inventory',
                                'identity_snapshot', 'speaker_notes', 'base_notes'))):
        return 'Private host material or presentation notes'
    if relative.startswith(REPORT) and name.startswith('results_bundle_'):
        return 'Packaging output sidecar; excluded to avoid recursive bundles'
    allowed = {'.py'} if relative.startswith(SOURCE) else {'.json', '.md', '.csv', '.png', '.svg', '.py'}
    if pure.suffix.lower() not in allowed:
        return 'Unsupported public type (including ZIP, arrays, checkpoints, logs and presentations)'
    if pure.suffix.lower() == '.py' and relative.startswith(REPORT) and '/revisions/' not in relative:
        return 'Report-root Python is not a preserved source archive'
    return None


def prepare_results(repo):
    repo = Path(repo).resolve()
    bindings = {}

    def bind(relative, digest):
        path = safe_path(repo, relative)
        require(path.is_file() and sha(path) == digest, 'Missing or changed bound evidence: ' + relative)
        require(relative not in bindings or bindings[relative] == digest, 'Conflicting evidence SHA: ' + relative)
        bindings[relative] = digest
        return path

    for name in REQUIRED:
        require((repo / REPORT / name).is_file(), 'Required public result is missing: ' + name)
    gate_path = safe_path(repo, REPORT + 'ROOT_RESULTS_GATE.json')
    gate = read(gate_path)
    require(gate.get('status') == 'PASS' and gate.get('scope') == 'EXPERIMENT_RESULTS_ONLY',
            'Root scientific result review must pass')
    bind(REPORT + 'ROOT_RESULTS_GATE.json', sha(gate_path))
    analysis = read(bind(REPORT + 'ANALYSIS_VALIDATION.json', gate['analysis_sha256']))
    completion = read(bind(REPORT + 'EXECUTION_COMPLETION.json', gate['completion_sha256']))
    require(analysis.get('status') == 'PASS' and analysis.get('integrity_status') == 'PASS'
            and analysis.get('T_numeric_values_parsed') == 0 and analysis.get('primary_count', 0) > 0,
            'Integrity-PASS analysis with a complete primary cohort and no protected-T reads is required')
    require(completion.get('status') in ('TRAINING_COMPLETE', 'TRAINING_INCOMPLETE', 'NOT_RUN_BUDGET', 'STOPPED')
            and completion.get('active') is None, 'Execution is not terminal without an active worker')
    plan = read(repo / REPORT / 'RUN_PLAN.json')
    jobs = plan['jobs'] + plan['optional_jobs']
    require(len(plan['jobs']) == 16 and len(plan['optional_jobs']) == 8 and not plan.get('extension_jobs'),
            'The registered 16-base/8-optional/no-extension plan changed')
    ids = {job['job_id'] for job in jobs}
    require(len(ids) == 24, 'Duplicated registered jobs')
    statuses = completion['run_statuses']
    terminal = {'COMPLETE', 'FAILED', 'STOPPED_DEADLINE', 'NOT_RUN_BUDGET', 'BASE_INCOMPLETE'}
    require(set(statuses) == ids and all(status in terminal for status in statuses.values()),
            'Missing, unknown or nonterminal job disposition')
    complete = sum(status == 'COMPLETE' for status in statuses.values())
    base_complete = sum(statuses[job['job_id']] == 'COMPLETE' for job in plan['jobs'])
    optional_complete = complete - base_complete
    require(analysis.get('completed_count') == complete and completion.get('completed_count') == complete
            and completion.get('base_complete_count') == base_complete
            and completion.get('optional_complete_count') == optional_complete,
            'Analysis and execution completion counts disagree')
    require((completion['status'] == 'TRAINING_COMPLETE') == (complete == 24),
            'Top-level completion does not match all planned jobs')
    expected_primary = 0
    for dataset in ('pems32', 'beijing'):
        base = [job for job in plan['jobs'] if job['dataset'] == dataset]
        extra = [job for job in plan['optional_jobs'] if job['dataset'] == dataset]
        require(len(base) == 8 and len(extra) == 4, 'Dataset cohort size differs')
        base_done = all(statuses[job['job_id']] == 'COMPLETE' for job in base)
        n = 3 if base_done and all(statuses[job['job_id']] == 'COMPLETE' for job in extra) else 2 if base_done else 0
        coverage = [row for row in analysis['coverage'] if row['dataset'] == dataset]
        require(len(coverage) == 1 and coverage[0]['primary_n'] == n, 'Analysis primary cohort differs from completed four-arm groups')
        expected_primary += 4 * n
    require(complete > 0 and analysis['primary_count'] == expected_primary and expected_primary > 0,
            'Empty or inconsistent complete matched primary cohort')
    partial = complete != 24
    require(not partial or gate.get('allow_partial') is True and bool(str(gate.get('partial_reason', '')).strip()),
            'Partial or failed execution needs explicit root permission and a reason')

    visual = gate.get('figures_visual_review', {})
    require(set(visual) == set(FIGURES), 'All three scientific figures need visual review receipts')
    for name in FIGURES:
        require(visual[name].get('status') == 'PASS', 'Scientific figure review failed: ' + name)
        bind(REPORT + name, visual[name]['sha256'])
    reviewed = gate.get('reviewed_files', {})
    require({REPORT + name for name in REVIEWED}.issubset(reviewed), 'Root review does not bind all scientific outputs')
    for relative, digest in reviewed.items():
        require(relative.startswith((REPORT, SOURCE)) and exclusion_reason(relative) is None,
                'Root reviewed_files includes a nonpublic/out-of-scope member')
        bind(relative, digest)
    verified = analysis.get('verified_files', {})
    require(verified.get(REPORT + 'EXECUTION_COMPLETION.json') == gate['completion_sha256'],
            'Analysis does not bind the reviewed terminal execution')
    for relative, digest in verified.items():
        require(relative.startswith((REPORT, SOURCE)) and exclusion_reason(relative) is None,
                'Analysis includes nonpublic/out-of-scope evidence')
        bind(relative, digest)
    seal = read(repo / REPORT / 'REGISTRATION_SEAL.json')
    for name, digest in seal['files'].items():
        require('/' not in name and '\\' not in name, 'Seal member must be a report-root file')
        bind(REPORT + name, digest)
    registration = read(repo / REPORT / 'PREREGISTRATION.json')
    for relative, digest in registration['registered_sources'].items():
        require(relative.startswith('research/') and relative.endswith('.py'), 'Registered source is not Python research code')
        bind(relative, digest)
    if 'unit_test_receipt_sha256' in registration:
        bind(REPORT + 'UNIT_TESTS.json', registration['unit_test_receipt_sha256'])

    excluded = []
    for path in sorted((repo / REPORT).glob('TECHNICAL_AMENDMENT_*.json')):
        amendment = read(path)
        for entry in amendment.get('files', []):
            relative = entry['archive']
            require(relative.startswith(REPORT + 'revisions/'), 'Archive entry is outside preserved revisions')
            safe_path(repo, relative)
            reason = exclusion_reason(relative)
            if reason is None:
                bind(relative, entry['sha256'])
            else:
                excluded.append(dict(path=relative, reason=reason, archived_expected_sha256=entry['sha256'],
                                     private_bytes_read=False))

    files = {}
    for prefix in (SOURCE, REPORT):
        for path in sorted((repo / prefix).rglob('*')):
            if not path.is_file():
                continue
            relative = path.relative_to(repo).as_posix()
            safe_path(repo, relative)
            if exclusion_reason(relative) is not None:
                continue
            digest = sha(path)
            if relative in bindings:
                require(bindings[relative] == digest, 'Evidence changed during collection: ' + relative)
            bindings[relative] = digest
            files[relative] = dict(path=relative, sha256=digest, bytes=path.stat().st_size)
    own_relative = SOURCE + 'package_results.py'
    require(own_relative in files, 'The exact package implementation must be included')
    require(files[own_relative]['sha256'] == sha(Path(__file__)), 'Packaged source differs from the executing packager')
    metadata = dict(campaign=NAME, scope='EXPERIMENT_RESULTS_ONLY',
        completion_status=completion['status'], completed_count=complete, base_complete_count=base_complete,
        optional_complete_count=optional_complete, primary_count=analysis['primary_count'],
        partial=partial, partial_reason=gate.get('partial_reason') if partial else None,
        root_gate_sha256=sha(gate_path), analysis_sha256=gate['analysis_sha256'],
        completion_sha256=gate['completion_sha256'], figures_visual_review=visual,
        package_source_sha256=files[own_relative]['sha256'], archive_members_excluded=excluded,
        exclusions='Only campaign public source/reports; no cache, raw data, predictions, checkpoints, private host bytes or presentations.',
        protected_T_numeric_values_read=0, GPU_execution=False)
    return files, bindings, metadata


def verify_zip(path, manifest_bytes):
    manifest = json.loads(manifest_bytes)
    expected = {entry['path'] for entry in manifest['files']} | {REPORT + MANIFEST}
    with zipfile.ZipFile(path) as archive:
        members = archive.namelist()
        require(len(members) == len(expected) and set(members) == expected, 'ZIP exact-member verification failed')
        require(archive.testzip() is None, 'ZIP CRC verification failed')
        require(archive.read(REPORT + MANIFEST) == manifest_bytes, 'ZIP manifest bytes changed')
        for entry in manifest['files']:
            value = archive.read(entry['path'])
            require(len(value) == entry['bytes'] and hashlib.sha256(value).hexdigest() == entry['sha256'],
                    'ZIP member SHA/size differs: ' + entry['path'])
    return dict(status='PASS', checked_members=len(expected), checked_payload_files=len(manifest['files']),
                exact_members='PASS', crc='PASS', all_member_sha256_and_sizes='PASS')


def publish_new(path, data):
    require(not path.exists(), 'Existing output is preserved: ' + str(path))
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_results_bundle(repo, files, bindings, metadata):
    repo = Path(repo).resolve()
    report = repo / REPORT
    target = report / ZIP_NAME
    for name in (ZIP_NAME, MANIFEST, RECEIPT):
        require(not (report / name).exists(), 'Existing result output is preserved: ' + name)
    for relative, digest in bindings.items():
        require(sha(safe_path(repo, relative)) == digest, 'Evidence changed before packaging: ' + relative)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), **metadata,
                    files=[files[name] for name in sorted(files)])
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf8')
    fd, temporary = tempfile.mkstemp(prefix=NAME + '_results_', suffix='.zip.tmp', dir=report)
    os.close(fd)
    temporary = Path(temporary)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for relative in sorted(files):
                value = safe_path(repo, relative).read_bytes()
                require(len(value) == files[relative]['bytes'] and hashlib.sha256(value).hexdigest() == files[relative]['sha256'],
                        'Payload changed during ZIP creation: ' + relative)
                archive.writestr(relative, value)
            archive.writestr(REPORT + MANIFEST, manifest_bytes)
        verify_zip(temporary, manifest_bytes)
        for relative, digest in bindings.items():
            require(sha(safe_path(repo, relative)) == digest, 'Bound evidence changed during packaging: ' + relative)
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    readback = verify_zip(target, manifest_bytes)
    publish_new(report / MANIFEST, manifest_bytes)
    receipt = dict(status='PASS', created_utc=datetime.now(timezone.utc).isoformat(), **metadata,
        zip_path=target.relative_to(repo).as_posix(), zip_absolute_path=str(target), zip_sha256=sha(target),
        zip_bytes=target.stat().st_size, manifest_path=REPORT + MANIFEST,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(), production_readback=readback)
    publish_new(report / RECEIPT, (json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf8'))
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo', type=Path, default=runtime.REPO, help='Repository root (default: this checkout)')
    parser.add_argument('--check-only', action='store_true', help='Validate gates and membership without writing a ZIP or sidecar')
    args = parser.parse_args(argv)
    files, bindings, metadata = prepare_results(args.repo)
    if args.check_only:
        print(json.dumps(dict(status='READY', payload_files=len(files), **metadata), ensure_ascii=False))
        return 0
    receipt = write_results_bundle(args.repo, files, bindings, metadata)
    print(json.dumps(receipt, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
