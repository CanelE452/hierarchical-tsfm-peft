"""Bundle verified presentation/report outputs, never arrays or checkpoints."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

from . import runtime


def main():
    report = runtime.REPORT
    analysis = runtime.read(report / 'ANALYSIS_VALIDATION.json')
    review = runtime.read(report / 'FINAL_REVIEW.json')
    if analysis.get('status') != 'PASS' or analysis.get('base_complete') != 20 or review.get('status') != 'PASS':
        raise ValueError('Final numeric and visual review must pass before delivery')
    presentation = report / 'presentation'
    required = ['Temporal_Channel_Input_Adaptation_revised.pptx',
                'Temporal_Channel_Input_Adaptation_revised.pdf', 'SPEAKER_NOTES_KO.md',
                'CHANGELOG_KO.md', 'README.md']
    for name in required:
        if not (presentation / name).is_file():
            raise FileNotFoundError('Missing presentation deliverable: ' + name)
    required_reviews = [presentation / name for name in required]
    required_reviews += [presentation / name for name in
                         ('PRESENTATION_VALIDATION.json', 'BUILD_FINAL.json', 'OFFICE_OPEN_FINAL.json')]
    required_reviews += [report / name for name in ('ANALYSIS_VALIDATION.json',
        'EXECUTION_COMPLETION.json', 'SUMMARY.csv', 'CONTRASTS.csv', 'RESULTS_KO.md',
        'FIG_JENA_P_BRANCHES.png')]
    missing_reviews = [p.relative_to(runtime.REPO).as_posix() for p in required_reviews
                       if p.relative_to(runtime.REPO).as_posix() not in review['reviewed_files']]
    if missing_reviews:
        raise ValueError('Final review is missing required deliverables: ' + ', '.join(missing_reviews))
    for path, digest in review['reviewed_files'].items():
        if runtime.sha(runtime.REPO / path) != digest:
            raise ValueError('Reviewed artifact has changed: ' + path)
    if not analysis.get('verified_files'):
        raise ValueError('Analysis must bind the verified execution evidence and report files')
    for path, digest in analysis['verified_files'].items():
        if runtime.sha(runtime.REPO / path) != digest:
            raise ValueError('Verified execution/report evidence has changed: ' + path)
    presentation_validation = runtime.read(presentation / 'PRESENTATION_VALIDATION.json')
    if (presentation_validation.get('status') != 'PASS'
            or presentation_validation.get('final_actual_data') is not True
            or presentation_validation.get('slide_count') != 19
            or presentation_validation.get('notes_count') != 19):
        raise ValueError('Final presentation validation must pass')
    for name, digest in presentation_validation['output_hashes'].items():
        if runtime.sha(presentation / name) != digest:
            raise ValueError('Validated presentation has changed: ' + name)
    office_check = runtime.read(presentation / 'OFFICE_OPEN_FINAL.json')
    if (office_check.get('status') != 'PASS' or office_check.get('final') is not True
            or office_check.get('normal_open') is not True or office_check.get('slide_count') != 19):
        raise ValueError('Final presentation must open normally in PowerPoint')
    preservation = runtime.guard_preserved()
    files = set()
    excluded = {'BUNDLE_MANIFEST.json', 'BUNDLE_RECEIPT.json'}
    for p in report.iterdir():
        if p.is_file() and p.name not in excluded and p.suffix.lower() in ('.json', '.csv', '.md', '.png', '.svg', '.pdf'):
            files.add(p)
    for p in (report / 'jobs').rglob('*'):
        if p.is_file() and p.suffix.lower() in ('.json', '.csv', '.md'):
            files.add(p)
    files.update(presentation / n for n in required)
    presentation_sources = ['PRESENTATION_VALIDATION.json', 'BUILD_FINAL.json',
        'CHANGE_MAP.json', 'SOURCE_RECOVERY.json', 'build_deck.cjs', 'validate_deck.py',
        'export_pdf.ps1', 'prepare_source.py', 'finalize.ps1', 'OFFICE_OPEN_FINAL.json',
        'assets/source_branding.png']
    presentation_sources += [f'assets/source_slide_{n:02d}.png' for n in range(1, 20)]
    for name in presentation_sources:
        p = presentation / name
        if not p.is_file():
            raise FileNotFoundError('Missing presentation build source: ' + name)
        files.add(p)
    optional_branch_results = presentation / 'BRANCH_RESULTS.json'
    if optional_branch_results.is_file():
        files.add(optional_branch_results)
    for p in (runtime.REPO / 'research' / runtime.NAME).rglob('*.py'):
        files.add(p)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), campaign=runtime.NAME,
                    scope='Presentation, report tables/figures, execution provenance, and new source. No raw data, predictions, or checkpoints.',
                    files=[dict(path=p.relative_to(runtime.REPO).as_posix(), bytes=p.stat().st_size,
                                sha256=runtime.sha(p)) for p in sorted(files)])
    runtime.atomic(report / 'BUNDLE_MANIFEST.json', manifest)
    target = report / f'{runtime.NAME}_bundle.zip'
    temporary = target.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(files):
            z.write(p, p.relative_to(runtime.REPO).as_posix())
        z.write(report / 'BUNDLE_MANIFEST.json', (report / 'BUNDLE_MANIFEST.json').relative_to(runtime.REPO).as_posix())
    with zipfile.ZipFile(temporary) as z:
        bad = z.testzip()
        if bad is not None:
            raise ValueError('ZIP CRC failed: ' + bad)
        names = z.namelist()
        if len(names) != len(set(names)) or len(names) != len(manifest['files']) + 1:
            raise ValueError('ZIP membership mismatch')
        for row in manifest['files']:
            data = z.read(row['path'])
            if len(data) != row['bytes'] or hashlib.sha256(data).hexdigest() != row['sha256']:
                raise ValueError('ZIP manifest mismatch: ' + row['path'])
        saved_manifest = z.read((report / 'BUNDLE_MANIFEST.json').relative_to(runtime.REPO).as_posix())
        if hashlib.sha256(saved_manifest).hexdigest() != runtime.sha(report / 'BUNDLE_MANIFEST.json'):
            raise ValueError('ZIP manifest file changed')
    temporary.replace(target)
    receipt = dict(status='PASS', completed_utc=datetime.now(timezone.utc).isoformat(),
                   zip_path=str(target), zip_sha256=runtime.sha(target), bytes=target.stat().st_size,
                   crc='PASS', manifest_hashes='PASS', checked_members=len(manifest['files']) + 1,
                   preservation=preservation, basic_runs=20, total_new_results=analysis['new_result_count'])
    runtime.atomic(report / 'BUNDLE_RECEIPT.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
