"""Package reviewed scientific results without reading or delivering a deck."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

from . import runtime
from .package import cost_public_files, e_public_files


NAME='ppt_justification_20261028_v1'
MAIN='reports/'+NAME+'/'
E='reports/ppt_e_fixed_20261028_v1/'
COST=MAIN+'cost_profile/'
FIGURES=('FIG_F_REPLACEMENT','FIG_ANCHOR_LAG','FIG_PAIRED_CONTRASTS')
OUTPUTS=('RUNS.csv','ALL_NEW_RUNS.csv','SUMMARY.csv','CONTRASTS.csv','PAIRED_DIFFERENCES.csv',
         'COVERAGE.csv','INTERACTION.json','CLAIM_EVIDENCE_MAP.json','HISTORICAL_EVIDENCE.json',
         'RESULTS_KO.md','REMAINING_GAPS_KO.md')
REQUIRED=OUTPUTS+('ANALYSIS_VALIDATION.json','EXECUTION_COMPLETION.json','DELIVERY_SCOPE.json',
    'ROOT_RESULTS_GATE.json','PREREGISTRATION.json','PREREGISTRATION.md','REGISTRATION_SEAL.json',
    'RUN_PLAN.json','PRESERVATION_BASELINE.json','REFERENCE_RUNS.json','PPT_CLAIM_EVIDENCE_AUDIT.json',
    'PPT_CLAIM_EVIDENCE_AUDIT_KO.md')+tuple(n+'.'+s for n in FIGURES for s in ('png','svg','pdf'))
OPTIONAL=('LAUNCH.json','PROGRESS.json','UNIT_TESTS.json','SCIENTIFIC_PREFLIGHT.json','ROOT_CPU_TESTS.json',
          'POSTPROCESS_TESTS.json','POSTPROCESS_STATUS.json','PACKAGE_TESTS.json','RESULTS_PACKAGE_TESTS.json')
SOURCE_NAMES=('models.py','runtime.py','run_one.py','controller.py','prepare.py','postprocess.py',
    'cost_profile.py','host_gpu_gate.py','verify_cost_results.py','package.py','package_results.py',
    'tests/test_scientific_contract.py','tests/test_postprocess.py','tests/test_cost_profile.py',
    'tests/test_cost_validation.py','tests/test_host_gpu_gate.py','tests/test_package.py','tests/test_package_results.py')
HISTORICAL=('research/informs_broad_validation_20261015_v1/priority_core_followup_20261016_v1/RESULTS_v2.csv',
    'reports/adapter_diagnostics_20261017_v2/QUESTIONS_AND_DECISIONS_KO.md',
    'reports/adapter_diagnostics_20261017_v2/METHOD_CONTRACT.md')
ATTEMPT_NAMES=('START.json','SOURCE_CONTRACT.json','SEARCH_RESULT.json','SELECTION.json',
               'REFIT_SUMMARY.json','RECEIPT.json','OPENED_FILES.json')
MANIFEST_NAME='RESULTS_BUNDLE_MANIFEST.json'
RECEIPT_NAME='RESULTS_BUNDLE_RECEIPT.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def public_path(repo,relative):
    if not isinstance(relative,str) or '\\' in relative:
        raise ValueError('Public member must use a relative slash path')
    pure=PurePosixPath(relative)
    if pure.is_absolute() or '..' in pure.parts or ':' in relative or not pure.parts:
        raise ValueError('Public member escapes repository: '+relative)
    parts=[p.lower() for p in pure.parts];name=pure.name.lower()
    if (set(parts)&{'presentation','.cache','private','__pycache__'} or 'draft' in name
            or name.startswith(('speaker_notes','base_notes'))
            or any(x in name for x in ('raw_counter','raw_host','raw_inventory','raw_log','private_inventory',
                                      'private_policy','private_identity','stderr','snapshot'))
            or name in ('policy.json','identity.json','inventory.json')):
        raise ValueError('Private or presentation member is excluded: '+relative)
    if pure.suffix.lower() not in ('.json','.csv','.md','.py','.png','.svg','.pdf'):
        raise ValueError('Unsupported public member type: '+relative)
    if pure.suffix.lower() in ('.png','.svg','.pdf') and not pure.name.startswith('FIG_'):
        raise ValueError('Only scientific FIG_* images/PDFs are public: '+relative)
    path=repo/relative
    if not path.resolve().is_relative_to(repo.resolve()):
        raise ValueError('Public member resolves outside repository: '+relative)
    return path


def prepare_results(repo):
    """Read terminal public evidence, bind hashes, and select an exact member set."""
    repo=Path(repo).resolve();files=set();bindings={}
    def bind(relative,digest):
        p=public_path(repo,relative)
        if not p.is_file() or sha(p)!=digest:
            raise ValueError('Bound result evidence changed or missing: '+relative)
        if relative in bindings and bindings[relative]!=digest:
            raise ValueError('Conflicting evidence bindings: '+relative)
        bindings[relative]=digest
        return p
    def bind_map(mapping,prefix=''):
        if not mapping:raise ValueError('Empty evidence hash binding')
        for relative,digest in mapping.items():bind(prefix+relative,digest)
    def add(relative):
        p=public_path(repo,relative)
        if not p.is_file():raise FileNotFoundError(relative)
        files.add(p);return p
    gate_path=add(MAIN+'ROOT_RESULTS_GATE.json');gate=read(gate_path)
    if (gate.get('status')!='PASS' or gate.get('scope')!='EXPERIMENT_RESULTS_ONLY'
            or gate.get('figures_visual_review')!='PASS'):
        raise ValueError('Independent result and scientific-figure review is required')
    bindings[MAIN+'ROOT_RESULTS_GATE.json']=sha(gate_path)
    scope=read(bind(MAIN+'DELIVERY_SCOPE.json',gate['delivery_scope_sha256']))
    if scope.get('status')!='USER_ASSIGNED_EXPERIMENTS_AND_EVIDENCE_ONLY':
        raise ValueError('Delivery scope does not authorize results-only packaging')
    analysis=read(bind(MAIN+'ANALYSIS_VALIDATION.json',gate['main']['analysis_sha256']))
    completion=read(bind(MAIN+'EXECUTION_COMPLETION.json',gate['main']['completion_sha256']))
    if (analysis.get('status')!='PASS' or analysis.get('base_complete')!=20
            or analysis.get('reference_rows')!=20 or analysis.get('T_numeric_values_parsed')!=0
            or completion.get('status')!='TRAINING_COMPLETE' or completion.get('active') is not None
            or completion.get('base_complete_count')!=20):
        raise ValueError('Terminal mandatory twenty-run scientific analysis is required')
    if analysis['inputs']['completion_sha256']!=gate['main']['completion_sha256']:
        raise ValueError('Analysis does not bind terminal execution')
    bind(MAIN+'REGISTRATION_SEAL.json',analysis['inputs']['registration_seal_sha256'])
    bind_map(read(repo/(MAIN+'REGISTRATION_SEAL.json'))['files'],MAIN)
    bind_map(read(repo/(MAIN+'PREREGISTRATION.json'))['registered_sources'])
    bind_map(analysis['verified_files'])
    bind_map(gate['reviewed_files'])
    required_review={MAIN+n for n in OUTPUTS+('ANALYSIS_VALIDATION.json','EXECUTION_COMPLETION.json','DELIVERY_SCOPE.json')}
    required_review.update(MAIN+n+'.png' for n in FIGURES)
    for name in REQUIRED:add(MAIN+name)
    for name in OPTIONAL:
        if (repo/(MAIN+name)).is_file():add(MAIN+name)
    plan=read(repo/(MAIN+'RUN_PLAN.json'))
    jobs=plan['jobs']+plan['optional_jobs']+plan['extension_jobs']
    ids=[j['job_id'] for j in jobs]
    if tuple(len(plan[k]) for k in ('jobs','optional_jobs','extension_jobs'))!=(20,10,30) or len(ids)!=len(set(ids)):
        raise ValueError('Registered job counts or identities differ')
    statuses=completion.get('run_statuses',{})
    terminal={'COMPLETE','FAILED','STOPPED_DEADLINE','NOT_RUN_BUDGET','NOT_REQUIRED','BASE_INCOMPLETE'}
    if (set(statuses)!=set(ids) or any(s not in terminal for s in statuses.values())
            or any(statuses[j['job_id']]!='COMPLETE' for j in plan['jobs'])
            or sum(s=='COMPLETE' for s in statuses.values())!=analysis['new_result_count']):
        raise ValueError('Execution contains a nonterminal, missing or inconsistent registered run')
    for job_id in ids:
        if not isinstance(job_id,str) or '/' in job_id or '\\' in job_id or '..' in job_id:
            raise ValueError('Invalid registered job ID')
        base=MAIN+'jobs/'+job_id+'/'
        names=['RESULT.json']
        for attempt in (1,2):
            names.append(f'WORKER_RECEIPT_attempt{attempt}.json')
            names.extend(f'attempt_{attempt:03d}/'+n for n in ATTEMPT_NAMES)
        for name in names:
            if (repo/(base+name)).is_file():add(base+name)
    for name in SOURCE_NAMES:add('research/'+NAME+'/'+name)
    for relative in HISTORICAL:
        if relative in analysis['verified_files']:add(relative)

    e=gate['e'];e_status=e.get('status');e_coverage=None
    if e_status=='VERIFIED_RESULT':
        ea=read(bind(E+'ANALYSIS_VALIDATION.json',e['analysis_sha256']))
        ec=read(bind(E+'EXECUTION_COMPLETION.json',e['completion_sha256']))
        ev=read(bind(E+'COMPLETION_VALIDATION.json',e['completion_validation_sha256']))
        e_coverage=ea.get('coverage',{})
        if (ea.get('status')!='PASS' or ea.get('base_complete')!=5 or ea.get('reference_rows')!=5
                or ea.get('T_numeric_values_parsed')!=0 or ev.get('status')!='PASS' or ev.get('T_numeric_values_parsed')!=0
                or ev.get('base_complete')!=5 or ev.get('primary_complete') is not True
                or ec.get('status') not in ('TRAINING_COMPLETE','TRAINING_INCOMPLETE') or ec.get('active') is not None
                or e_coverage.get('primary_n')!=5 or e_coverage.get('primary_complete') is not True):
            raise ValueError('E result lacks verified complete matched five-seed evidence')
        if (ea['inputs']['completion_sha256']!=e['completion_sha256']
                or ea['inputs']['completion_validation_sha256']!=e['completion_validation_sha256']
                or ev['inputs']['completion_sha256']!=e['completion_sha256']):
            raise ValueError('E analysis completion binding differs')
        if not e_coverage.get('scientific_complete') and not e_coverage.get('provisional'):
            raise ValueError('Incomplete E extension must remain provisional')
        rows=ea.get('summary',[])
        if (len(rows)!=2 or {r['method_id'] for r in rows}!={'OURS','E_FIXED'} or any(r['n']!=5 for r in rows)
                or any(json.loads(r['seed_ids'])!=list(range(92601,92606)) for r in rows)):
            raise ValueError('E analysis does not contain exactly two five-seed groups')
        if e_coverage.get('provisional') and not next(r for r in rows if r['method_id']=='E_FIXED').get('provisional'):
            raise ValueError('E provisional result was hidden in summary')
        bind_map(ea['verified_files']);bind_map(ev['verified_files'])
        e_names=('ANALYSIS_VALIDATION.json','COMPLETION_VALIDATION.json','EXECUTION_COMPLETION.json',
            'RUNS.csv','ALL_NEW_RUNS.csv','SUMMARY.csv','CONTRASTS.csv','PAIRED_DIFFERENCES.csv','RESULTS_KO.md',
            'FIG_E_FIXED.png','FIG_E_FIXED.svg','FIG_E_FIXED.pdf')
        for name in e_names:add(E+name)
        required_review.update(E+n for n in e_names if not n.endswith(('.svg','.pdf')))
        files.update(e_public_files(repo,{'appendix':True}))
    elif e_status in ('NOT_RUN_BUDGET','INCOMPLETE'):
        relative=e['disposition_path']
        if not relative.startswith(E) or '/' in relative[len(E):]:raise ValueError('E disposition must be in its public report root')
        disposition=read(bind(relative,e['disposition_sha256']))
        if disposition.get('status')!=e_status or not disposition.get('reason'):
            raise ValueError('E non-completion requires an explicit reason and matching state')
        if (repo/(E+'ANALYSIS_VALIDATION.json')).is_file():
            ea=read(repo/(E+'ANALYSIS_VALIDATION.json'))
            if ea.get('status')=='PASS' or ea.get('coverage',{}).get('primary_complete'):
                raise ValueError('A complete E result cannot be omitted as non-completion')
            if ea.get('integrity_status')!='PASS' or ea.get('status')!=e_status:
                raise ValueError('Existing E partial analysis is unverified or differs from disposition')
            bind_map(ea['verified_files'])
        files.update(e_public_files(repo,{'appendix':False,'provenance':{'status_evidence_path':relative}}))
        required_review.add(relative)
    else:raise ValueError('Explicit E verified-result or non-completion disposition required')

    cost=gate['cost'];cost_status=cost.get('status');cost_phase='POST_MEASUREMENT'
    archive_exclusions=[]
    cost_files=cost_public_files(repo,archive_exclusions=archive_exclusions)
    if cost_status=='VERIFIED':
        cs=read(bind(COST+'COST_SUMMARY.json',cost['summary_sha256']))
        cv=read(bind(COST+'COST_VALIDATION.json',cost['validation_sha256']))
        if (cs.get('status')!='COMPLETE' or cv.get('status')!='PASS' or cv.get('cost_summary_sha256')!=cost['summary_sha256']
                or cs.get('training_executed') is not False or cv.get('training_executed') is not False
                or cs.get('T_numeric_values_parsed')!=0 or cv.get('T_numeric_values_parsed')!=0):
            raise ValueError('Cost values require independently verified complete measurements')
        expected_facets={'Raw_appliances','KTAU_main_appliances_K10_s92601','KTAU_main_appliances_K10_s92602'}
        if len(cs.get('rows',[]))!=3 or {r.get('accuracy_run_id') for r in cs['rows']}!=expected_facets:
            raise ValueError('Verified cost must contain all three registered facets')
        bind_map(cv['evidence_bindings'],COST);bind_map(cv['producer_source_bindings'])
        for name in ('COST_SUMMARY.json','COST_VALIDATION.json'):
            add(COST+name);required_review.add(COST+name)
    elif cost_status in ('NOT_MEASURED','CONTAMINATED','INCOMPLETE'):
        relative=cost['disposition_path']
        if not relative.startswith(COST) or '/' in relative[len(COST):]:raise ValueError('Cost disposition must be in its public report root')
        disposition=read(bind(relative,cost['disposition_sha256']))
        if (disposition.get('status')!=cost_status or not disposition.get('reason')
                or disposition.get('phase') not in ('BUDGET','PREFLIGHT','MEASUREMENT','POST_MEASUREMENT')):
            raise ValueError('Cost non-result needs explicit phase, reason and matching state')
        cost_phase=disposition['phase']
        if (repo/(COST+'COST_VALIDATION.json')).is_file() and read(repo/(COST+'COST_VALIDATION.json')).get('status')=='PASS':
            raise ValueError('Verified cost result cannot be silently excluded')
        cost_files.discard(repo/(COST+'COST_SUMMARY.json'))
        add(relative);required_review.add(relative)
    else:raise ValueError('Explicit verified-cost or non-measurement disposition required')
    files.update(cost_files)
    missing=required_review-set(gate['reviewed_files'])
    if missing:raise ValueError('Independent review is missing required outputs: '+', '.join(sorted(missing)))
    # Additional reviewed interpretation or audit records may be included only within these report roots.
    for relative in gate['reviewed_files']:
        if relative.startswith((MAIN,E)) and relative not in (MAIN+MANIFEST_NAME,MAIN+RECEIPT_NAME):
            if cost_status!='VERIFIED' and relative==COST+'COST_SUMMARY.json':continue
            if e_status!='VERIFIED_RESULT' and relative in {E+n for n in
                ('RUNS.csv','ALL_NEW_RUNS.csv','SUMMARY.csv','CONTRASTS.csv','PAIRED_DIFFERENCES.csv',
                 'FIG_E_FIXED.png','FIG_E_FIXED.svg','FIG_E_FIXED.pdf')}:continue
            add(relative)
    for p in files:public_path(repo,p.relative_to(repo).as_posix())
    # Freeze every chosen member, including packager code and public archives not in the scientific analysis mapping.
    for p in files:
        relative=p.relative_to(repo).as_posix();digest=sha(p)
        if relative in bindings and bindings[relative]!=digest:raise ValueError('Evidence changed during member selection')
        bindings[relative]=digest
    metadata={'campaign':NAME,'scope':'EXPERIMENT_RESULTS_ONLY','base_complete':analysis['base_complete'],
        'optional_complete':analysis.get('optional_complete'),'total_new_results':analysis['new_result_count'],
        'coverage':analysis['coverage'],'e_status':e_status,'e_coverage':e_coverage,'cost_status':cost_status,'cost_phase':cost_phase,
        'archive_members_excluded':archive_exclusions,
        'gate_sha256':bindings[MAIN+'ROOT_RESULTS_GATE.json'],'delivery_scope_sha256':gate['delivery_scope_sha256'],
        'limitations':'Exploratory development V. No independent-test, equivalence, universal superiority or exclusive-idle-GPU claim.'}
    return files,bindings,metadata


def write_results_bundle(repo,files,bindings,metadata):
    """Atomically publish an exact, CRC-checked ZIP after all gates have passed."""
    repo=Path(repo).resolve();report=repo/MAIN
    target=report/(NAME+'_results_bundle.zip')
    if target.exists():raise FileExistsError('Existing results ZIP is preserved: '+str(target))
    for relative,digest in bindings.items():
        if sha(public_path(repo,relative))!=digest:raise ValueError('Bound evidence changed before packaging: '+relative)
    members=[]
    for p in sorted(files):
        relative=p.relative_to(repo).as_posix();public_path(repo,relative)
        if sha(p)!=bindings.get(relative):raise ValueError('Selected member changed before packaging: '+relative)
        members.append({'path':relative,'bytes':p.stat().st_size,'sha256':bindings[relative]})
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),**metadata,'files':members}
    manifest_bytes=(json.dumps(manifest,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8')
    manifest_member=MAIN+MANIFEST_NAME
    expected={row['path'] for row in members}|{manifest_member}
    fd,temp_name=tempfile.mkstemp(prefix=NAME+'_results_',suffix='.zip.tmp',dir=report)
    os.close(fd);temporary=Path(temp_name)
    try:
        with zipfile.ZipFile(temporary,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for p in sorted(files):archive.write(p,p.relative_to(repo).as_posix())
            archive.writestr(manifest_member,manifest_bytes)
        with zipfile.ZipFile(temporary) as archive:
            names=archive.namelist()
            if archive.testzip() is not None or set(names)!=expected or len(names)!=len(expected):
                raise ValueError('Results ZIP CRC or exact membership failed')
            if archive.read(manifest_member)!=manifest_bytes:raise ValueError('Results ZIP manifest changed')
            for row in members:
                data=archive.read(row['path'])
                if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
                    raise ValueError('Results ZIP member SHA or size mismatch: '+row['path'])
        if target.exists():raise FileExistsError('Results ZIP was created concurrently')
        temporary.replace(target)
    finally:
        if temporary.exists():temporary.unlink()
    atomic_bytes(report/MANIFEST_NAME,manifest_bytes)
    receipt={'status':'PASS','scope':'EXPERIMENT_RESULTS_ONLY','completed_utc':datetime.now(timezone.utc).isoformat(),
        'zip_path':str(target),'zip_sha256':sha(target),'bytes':target.stat().st_size,
        'manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),'checked_members':len(expected),
        'exact_members':'PASS','crc':'PASS','manifest_hashes':'PASS',**metadata}
    atomic_bytes(report/RECEIPT_NAME,(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf8'))
    return receipt


def atomic_bytes(path,data):
    fd,name=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(data)
        Path(name).replace(path)
    finally:
        if Path(name).exists():Path(name).unlink()


def main():
    runtime.guard_preserved()
    files,bindings,metadata=prepare_results(runtime.REPO)
    receipt=write_results_bundle(runtime.REPO,files,bindings,metadata)
    print(json.dumps(receipt,ensure_ascii=False))


if __name__=='__main__':main()
