"""Bundle verified presentation/report outputs, never arrays or checkpoints."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

from . import runtime


def archived_public_files(repo, archive_relative, allowed_sources):
    """Include only receipt-bound public originals within this archive."""
    repo=repo.resolve()
    archive=repo/archive_relative
    if not archive.exists():
        return set()
    if not archive.resolve().is_relative_to(repo):
        raise ValueError('Public archive escapes repository')
    receipt=archive/'ARCHIVE_RECEIPT.json'
    if not receipt.is_file() or not receipt.resolve().is_relative_to(archive.resolve()):
        raise ValueError('Public archive requires its own ARCHIVE_RECEIPT.json')
    rows=json.loads(receipt.read_text(encoding='utf-8-sig'))['files']
    if not isinstance(rows,list) or not rows:
        raise ValueError('Public archive receipt has no files')
    files={receipt};seen_sources=set()
    for row in rows:
        source=row['source']
        archived=row['archived']
        if source not in allowed_sources or source in seen_sources:
            raise ValueError('Unexpected or duplicate archived public source: '+str(source))
        seen_sources.add(source)
        # Receipts use repository-relative slash paths; permit the two actual archive layouts.
        expected={archive_relative+'/'+source,archive_relative+'/'+Path(source).name}
        if archived not in expected:
            raise ValueError('Archived path differs from its public source: '+str(archived))
        p=repo/archived
        if (not p.resolve().is_relative_to(archive.resolve()) or p.suffix.lower() not in ('.json','.md','.py')
                or p.suffix!=Path(source).suffix or p in files):
            raise ValueError('Archived public path or suffix is invalid: '+archived)
        if not p.is_file() or p.stat().st_size!=row['bytes'] or runtime.sha(p)!=row['sha256']:
            raise ValueError('Archived bytes or SHA differs from receipt: '+archived)
        files.add(p)
    return files


def cost_public_files(repo):
    report=repo/'reports/ppt_justification_20261028_v1/cost_profile'
    names=('COST_INPUTS.json','COST_PROTOCOL.json','COST_SEAL.json','CPU_TESTS.json',
        'AMENDMENT_001.json','AMENDMENT_002.json','HOST_GPU_CONTRACT.json','HOST_GATE_CPU_TESTS.json','HOST_CODE_REVIEW.json',
        'ROOT_LAUNCH_GATE.json','COST_LAUNCH.json','COST_SUMMARY.json','COST_VALIDATION.json')
    facets=('Raw_appliances','KTAU_main_appliances_K10_s92601','KTAU_main_appliances_K10_s92602')
    names+=tuple(facet+'_'+suffix+'.json' for facet in facets
                 for suffix in ('MEASUREMENT','HOST_PREFLIGHT','HOST_OBSERVATION'))
    files={report/name for name in names if (report/name).is_file()}
    if any(not p.resolve().is_relative_to(report.resolve()) for p in files):
        raise ValueError('Cost public file escapes cost report directory')
    prefix='reports/ppt_justification_20261028_v1/cost_profile/'
    source='research/ppt_justification_20261028_v1/'
    common={prefix+n for n in ('COST_INPUTS.json','COST_PROTOCOL.json','COST_SEAL.json','CPU_TESTS.json')}
    first=common|{source+'cost_profile.py',source+'tests/test_cost_profile.py'}
    second=first|{prefix+'AMENDMENT_001.json',source+'verify_cost_results.py',source+'tests/test_cost_validation.py'}
    files.update(archived_public_files(repo,prefix+'revisions/preflight_001',first))
    files.update(archived_public_files(repo,prefix+'revisions/preflight_002',second))
    return files


def e_public_files(repo, fixed_e):
    report=repo/'reports/ppt_e_fixed_20261028_v1'
    source=repo/'research/ppt_e_fixed_20261028_v1'
    report_names=('PREREGISTRATION.md','PREREGISTRATION.json','RUN_PLAN.json','REGISTRATION_SEAL.json',
        'PRESERVATION_BASELINE.json','REFERENCE_RUNS.json','PREPARATION_READY.json','UNIT_TESTS.json',
        'ROOT_DESIGN_LOCK_BEFORE_PREVIEW.json','ROOT_LAUNCH_GATE.json','BUDGET_DECISION.json','LAUNCH.json','AMENDMENT_001.json',
        'PROGRESS.json','EXECUTION_COMPLETION.json','COMPLETION_VALIDATION.json','ANALYSIS_VALIDATION.json',
        'POSTPROCESS_TESTS.json','RUNS.csv','ALL_NEW_RUNS.csv','SUMMARY.csv','CONTRASTS.csv','PAIRED_DIFFERENCES.csv',
        'RESULTS_KO.md','FIG_E_FIXED.png','FIG_E_FIXED.svg','FIG_E_FIXED.pdf')
    source_names=('models.py','runtime.py','run_one.py','controller.py','prepare.py','postprocess.py',
                  'tests/test_contract.py','tests/test_postprocess.py')
    if not fixed_e.get('appendix'):
        report_names=tuple(n for n in report_names if n not in ('RUNS.csv','ALL_NEW_RUNS.csv','SUMMARY.csv','CONTRASTS.csv','PAIRED_DIFFERENCES.csv') and not n.startswith('FIG_E_FIXED.'))
    files={p for p in [*(report/n for n in report_names),*(source/n for n in source_names)] if p.is_file()}
    status_path=fixed_e.get('provenance',{}).get('status_evidence_path')
    if status_path:
        p=repo/status_path
        if not p.resolve().is_relative_to(report.resolve()) or p.suffix!='.json':
            raise ValueError('E disposition evidence is outside its public report scope')
        files.add(p)
    attempt_names=('START.json','SOURCE_CONTRACT.json','SEARCH_RESULT.json','SELECTION.json',
                   'REFIT_SUMMARY.json','RECEIPT.json','OPENED_FILES.json','HOST_GPU_GATE.json','E_INVARIANCE.json')
    for seed in range(92601,92606):
        for suffix in ('','_ext10'):
            job=report/'jobs'/f'PE_E_FIXED_jena_K6_s{seed}{suffix}'
            paths=[job/'RESULT.json']
            for attempt in (1,2):
                paths.append(job/f'WORKER_RECEIPT_attempt{attempt}.json')
                paths.extend(job/f'attempt_{attempt:03d}'/n for n in attempt_names)
            files.update(p for p in paths if p.is_file())
    archive_sources={'reports/ppt_e_fixed_20261028_v1/'+n for n in
        ('PREREGISTRATION.md','PREREGISTRATION.json','RUN_PLAN.json','REFERENCE_RUNS.json',
         'PRESERVATION_BASELINE.json','REGISTRATION_SEAL.json','UNIT_TESTS.json','PREPARATION_READY.json',
         'ROOT_DESIGN_LOCK_BEFORE_PREVIEW.json')}
    archive_sources.update('research/ppt_e_fixed_20261028_v1/'+n for n in
        ('models.py','runtime.py','run_one.py','controller.py','prepare.py','tests/test_contract.py'))
    files.update(archived_public_files(repo,'reports/ppt_e_fixed_20261028_v1/revisions/preflight_001',archive_sources))
    return files


def main():
    report = runtime.REPORT
    analysis = runtime.read(report / 'ANALYSIS_VALIDATION.json')
    review = runtime.read(report / 'FINAL_REVIEW.json')
    if analysis.get('status') != 'PASS' or analysis.get('base_complete') != 20 or review.get('status') != 'PASS':
        raise ValueError('Final numeric and visual review must pass before delivery')
    presentation = report / 'presentation'
    build=runtime.read(presentation/'BUILD_FINAL.json')
    fixed_e=build.get('e_fixed_results',{})
    expected_slides=22 if fixed_e.get('appendix') else 21
    required = ['Temporal_Channel_Input_Adaptation_justification.pptx',
                'Temporal_Channel_Input_Adaptation_justification.pdf', 'SPEAKER_NOTES_KO.md',
                'CHANGELOG_KO.md', 'README.md']
    for name in required:
        if not (presentation / name).is_file():
            raise FileNotFoundError('Missing presentation deliverable: ' + name)
    required_reviews = [presentation / name for name in required]
    required_reviews += [presentation / name for name in
                         ('PRESENTATION_VALIDATION.json', 'BUILD_FINAL.json', 'OFFICE_OPEN_FINAL.json')]
    required_reviews += [report / name for name in ('ANALYSIS_VALIDATION.json',
        'EXECUTION_COMPLETION.json', 'SUMMARY.csv', 'CONTRASTS.csv', 'RESULTS_KO.md',
        'FIG_F_REPLACEMENT.png','FIG_ANCHOR_LAG.png','FIG_PAIRED_CONTRASTS.png',
        'CLAIM_EVIDENCE_MAP.json','REMAINING_GAPS_KO.md','ROOT_PRESENTATION_GATE.json')]
    for name in ('COST_SUMMARY.json','COST_VALIDATION.json'):
        if (report/'cost_profile'/name).is_file():
            required_reviews.append(report/'cost_profile'/name)
    e_report=report.parent/'ppt_e_fixed_20261028_v1'
    if fixed_e.get('appendix'):
        required_reviews += [e_report/n for n in ('ANALYSIS_VALIDATION.json','COMPLETION_VALIDATION.json',
            'EXECUTION_COMPLETION.json','SUMMARY.csv','CONTRASTS.csv','RESULTS_KO.md','FIG_E_FIXED.png')]
    elif fixed_e.get('provenance',{}).get('status_evidence_path'):
        required_reviews.append(runtime.REPO/fixed_e['provenance']['status_evidence_path'])
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
            or presentation_validation.get('slide_count') != expected_slides
            or presentation_validation.get('notes_count') != expected_slides):
        raise ValueError('Final presentation validation must pass')
    for name, digest in presentation_validation['output_hashes'].items():
        if runtime.sha(presentation / name) != digest:
            raise ValueError('Validated presentation has changed: ' + name)
    gate=runtime.read(report/'ROOT_PRESENTATION_GATE.json')
    if gate.get('status')!='PASS' or gate.get('analysis_sha256')!=runtime.sha(report/'ANALYSIS_VALIDATION.json'):
        raise ValueError('Root independent analysis approval is stale')
    provenance=build['justification_results']['provenance']
    if provenance['analysis_sha256']!=runtime.sha(report/'ANALYSIS_VALIDATION.json') or provenance['gate_sha256']!=runtime.sha(report/'ROOT_PRESENTATION_GATE.json'):
        raise ValueError('Presentation is not bound to approved new evidence')
    if fixed_e.get('appendix'):
        if gate.get('e_fixed_status')!='INCLUDED' or fixed_e.get('status')!='COMPLETE':
            raise ValueError('E inclusion is not approved')
        for name,key in [('ANALYSIS_VALIDATION.json','analysis_sha256'),('EXECUTION_COMPLETION.json','completion_sha256'),('COMPLETION_VALIDATION.json','completion_validation_sha256')]:
            digest=runtime.sha(e_report/name)
            if fixed_e['provenance'][key]!=digest or gate['e_'+key]!=digest:
                raise ValueError('Approved E execution evidence changed')
        e_analysis=runtime.read(e_report/'ANALYSIS_VALIDATION.json')
        if e_analysis.get('status')!='PASS' or e_analysis.get('base_complete')!=5 or e_analysis.get('T_numeric_values_parsed')!=0:
            raise ValueError('E analysis is not verified')
        for name,digest in e_analysis['verified_files'].items():
            if runtime.sha(runtime.REPO/name)!=digest:raise ValueError('Verified E evidence changed: '+name)
    else:
        if fixed_e.get('status') not in ('NOT_RUN_BUDGET','INCOMPLETE') or fixed_e.get('status')!=gate.get('e_fixed_status'):
            raise ValueError('Final E non-completion must be explicit')
        p=fixed_e['provenance']
        if runtime.sha(runtime.REPO/p['status_evidence_path'])!=p['status_evidence_sha256'] or p['status_evidence_sha256']!=gate['e_status_evidence_sha256']:
            raise ValueError('E non-completion evidence changed')
    cost=build.get('cost_results',{})
    if cost.get('status')=='COMPLETE':
        for filename,key in [('COST_SUMMARY.json','summary_sha256'),('COST_VALIDATION.json','validation_sha256')]:
            digest=runtime.sha(report/'cost_profile'/filename)
            if cost['provenance'][key]!=digest or gate['cost_'+key]!=digest:
                raise ValueError('Presentation cost evidence changed after approval')
    legacy=report.parent/'jena_p_branch_20261027_v1'
    for filename,key in [('ANALYSIS_VALIDATION.json','analysis_validation_sha256'),('EXECUTION_COMPLETION.json','completion_sha256'),('SUMMARY.csv','summary_sha256'),('CONTRASTS.csv','contrasts_sha256')]:
        if runtime.sha(legacy/filename)!=build['branch_results']['provenance'][key]:
            raise ValueError('Legacy ten-seed presentation evidence changed')
    office_check = runtime.read(presentation / 'OFFICE_OPEN_FINAL.json')
    if (office_check.get('status') != 'PASS' or office_check.get('final') is not True
            or office_check.get('normal_open') is not True or office_check.get('slide_count') != expected_slides):
        raise ValueError('Final presentation must open normally in PowerPoint')
    preservation = runtime.guard_preserved()
    files = set()
    files.update(e_public_files(runtime.REPO,fixed_e))
    excluded = {'BUNDLE_MANIFEST.json', 'BUNDLE_RECEIPT.json'}
    for p in report.iterdir():
        if p.is_file() and p.name not in excluded and p.suffix.lower() in ('.json', '.csv', '.md', '.png', '.svg', '.pdf'):
            files.add(p)
    for p in (report / 'jobs').rglob('*'):
        if p.is_file() and p.suffix.lower() in ('.json', '.csv', '.md'):
            files.add(p)
    files.update(cost_public_files(runtime.REPO))
    files.update(presentation / n for n in required)
    presentation_sources = ['PRESENTATION_VALIDATION.json', 'BUILD_FINAL.json',
        'CHANGE_MAP.json', 'SOURCE_RECOVERY.json', 'BASE_NOTES_KO.json', 'assets/source_ablation_points.png', 'build_deck.cjs', 'justification_content.cjs', 'e_fixed_content.cjs', 'test_e_fixed_content.cjs', 'validate_deck.py',
        'export_pdf.ps1', 'prepare_source.py', 'finalize.ps1', 'OFFICE_OPEN_FINAL.json',
        'assets/source_branding.png']
    presentation_sources += [f'assets/source_slide_{n:02d}.png' for n in range(1, 20)]
    for name in presentation_sources:
        p = presentation / name
        if not p.is_file():
            raise FileNotFoundError('Missing presentation build source: ' + name)
        files.add(p)
    legacy=report.parent/'jena_p_branch_20261027_v1'
    files.update(legacy/name for name in ('ANALYSIS_VALIDATION.json','EXECUTION_COMPLETION.json','SUMMARY.csv','CONTRASTS.csv'))
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
