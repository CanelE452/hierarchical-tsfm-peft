"""Archive public followup evidence without arrays, checkpoints, or nested ZIPs."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.dont_write_bytecode = True
from . import common


MANIFEST = 'BUNDLE_MANIFEST.json'
RECEIPT = 'BUNDLE_RECEIPT.json'
PUBLIC_SUFFIXES = {'.json', '.csv', '.md', '.txt', '.png', '.svg', '.py', '.log'}
MUTABLE_REPORT_NAMES = {'PROGRESS.json', 'FOLLOWUP_PROGRESS.json', 'POSTPROCESS_STATUS.json',
                        'CONTROLLER_STATE.json', 'REPORTING_FAILURE.json'}
REFERENCE_NAMES = ('RUN_PLAN.json', 'PREREGISTRATION.json', 'PREREGISTRATION.md',
    'PREREGISTRATION_COMMIT.json', 'PUBLICATION_PRECHECK.json', 'PRE_HASHES.json',
    'POST_HASHES.json', 'G1_REUSE_AUDIT.json', 'DEVIATIONS_FROM_ADAPTS.md',
    'RUNS.csv', 'SUMMARY.csv', 'COMPARATIVE.csv', 'ABLATION.csv', 'DECISION.json',
    'RESULTS_KO.md', 'SLIDE_NUMBERS.md', 'VMSE_DEFINITION.md', 'EXECUTION_COMPLETION.json')


def public_file(path):
    path = Path(path)
    parts = {part.lower() for part in path.parts}
    return (path.suffix.lower() in PUBLIC_SUFFIXES and
            not parts.intersection({'.cache', '__pycache__', 'checkpoints', 'checkpoint', 'raw_arrays'}) and
            not path.name.lower().startswith(('selected_bundle', 'prediction_array')) and
            path.name not in (MANIFEST, RECEIPT))


def collect_members():
    members = {}

    def add(path, archive_name=None):
        path = Path(path)
        if path.is_file() and public_file(path):
            name = archive_name or path.relative_to(common.REPO).as_posix()
            if name in members and members[name] != path:
                raise ValueError('Duplicate bundle member path: ' + name)
            members[name] = path

    for path in common.REPORT.rglob('*'):
        if path.name not in MUTABLE_REPORT_NAMES and path.suffix != '.log':
            add(path)
    for name in (common.NAME, common.OLD_NAME):
        for path in (common.REPO / 'research' / name).rglob('*'):
            add(path)
    # Old campaign receipts are reference evidence; new numeric reports are at root.
    for name in REFERENCE_NAMES:
        add(common.OLD_REPORT / name)
    for job in common.selected_jobs():
        directory = common.OLD_REPORT / 'jobs' / job['job_id']
        if directory.exists():
            for path in directory.rglob('*'):
                add(path)
    add(common.SOURCE_DOCUMENT, 'references/comparative_ablation_followup_executor_instructions.md')
    return dict(sorted(members.items()))


def reference_hashes():
    baseline = common.read(common.REPORT / 'PRESERVATION_BASELINE.json')
    source = common.REPO / 'research' / common.OLD_NAME
    rows = []
    for row in baseline['files']:
        path = Path(row['path'])
        old_result = path.name == 'RESULT.json' and path.is_relative_to(common.OLD_REPORT / 'jobs')
        reference = path.parent == common.OLD_REPORT and path.name in REFERENCE_NAMES
        original_source = path.parent == source and path.suffix == '.py'
        if old_result or reference or original_source:
            rows.append(dict(path=path.relative_to(common.REPO).as_posix(), sha256=row['sha256'],
                             bytes=row['bytes']))
    return sorted(rows, key=lambda row: row['path'])


def verify_archive(target, manifest_path):
    """Validate exact entry set, CRC, and SHA-256 of every entry including manifest."""
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_name = 'reports/' + common.NAME + '/' + MANIFEST
    expected = {row['path']: row for row in manifest['files']}
    expected[manifest_name] = dict(sha256=hashlib.sha256(manifest_bytes).hexdigest(), bytes=len(manifest_bytes))
    with zipfile.ZipFile(target) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError('Bundle entry set differs from sealed manifest')
        bad = archive.testzip()
        if bad is not None:
            raise ValueError('Bundle CRC mismatch: ' + bad)
        for name, row in expected.items():
            payload = archive.read(name)
            if len(payload) != row['bytes'] or hashlib.sha256(payload).hexdigest() != row['sha256']:
                raise ValueError('Bundle entry SHA/size mismatch: ' + name)
    return dict(entries=len(expected), zip_crc='PASS', all_entry_sha256='PASS',
                manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())


def run():
    from . import reporting
    common.guard_preserved()
    common.install_write_guard()
    reporting.verify_completion(common.REPORT)
    target = common.REPORT / (common.NAME + '_bundle.zip')
    manifest_path = common.REPORT / MANIFEST
    receipt_path = common.REPORT / RECEIPT
    if target.exists():
        verification = verify_archive(target, manifest_path)
        preservation = common.guard_preserved()
        receipt = dict(path=str(target), sha256=common.sha(target), bytes=target.stat().st_size,
                       preservation=preservation, **verification)
        if receipt_path.exists():
            if common.read(receipt_path) != receipt:
                raise ValueError('Existing bundle receipt differs')
        else:
            common.write_new(receipt_path, receipt)
        return receipt
    if receipt_path.exists():
        raise FileExistsError('Bundle receipt exists but archive is absent')
    members = collect_members()
    manifest = dict(kind='followup_public_reports_and_sources',
        files=[dict(path=name, bytes=path.stat().st_size, sha256=common.sha(path)) for name, path in members.items()],
        frozen_original_input_references=reference_hashes(),
        excluded=['prediction arrays', 'source datasets', 'model weights', 'checkpoints',
                  '.cache', '__pycache__', '.pyc', 'nested ZIP archives', 'mutable controller progress'],
        scope='New followup reports and source, all new-job public reports, frozen original campaign source '
              'and selected original reference reports. Old job RESULT.json files are hash-referenced; '
              'all their numerical analysis is included in the new RUNS/SUMMARY/ERROR_DISTRIBUTION reports.',
        reproducibility='Self-contained public result review. Recomputing predictions or saved-array metrics '
                        'requires local excluded datasets/model/checkpoint/prediction files.',
        T_numeric_values_read=0)
    if manifest_path.exists():
        if common.read(manifest_path) != manifest:
            raise ValueError('Packaging inputs differ from sealed manifest')
    else:
        common.write_new(manifest_path, manifest)
    expected = {row['path']: row for row in manifest['files']}
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=7) as archive:
        for name, path in members.items():
            payload = path.read_bytes()
            if (len(payload) != expected[name]['bytes'] or
                    hashlib.sha256(payload).hexdigest() != expected[name]['sha256']):
                raise ValueError('Bundle source changed while packaging: ' + str(path))
            archive.writestr(name, payload)
        archive.writestr('reports/' + common.NAME + '/' + MANIFEST, manifest_path.read_bytes())
    verification = verify_archive(target, manifest_path)
    preservation = common.guard_preserved()
    receipt = dict(path=str(target), sha256=common.sha(target), bytes=target.stat().st_size,
                   preservation=preservation, **verification)
    common.write_new(receipt_path, receipt)
    return receipt


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False))
