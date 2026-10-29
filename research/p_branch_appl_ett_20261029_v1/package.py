"""Publish an integrity-checked result-only bundle, including honest partial runs.

ROOT_RESULTS_GATE.json binds status=PASS, scope=EXPERIMENT_RESULTS_ONLY,
analysis_sha256, completion_sha256, reviewed_files, figures_visual_review for the
PNG, and allow_partial/partial_reason when fewer than60 jobs completed.
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
from .postprocess import validate_plan

NAME = 'p_branch_appl_ett_20261029_v1'
REPORT = 'reports/' + NAME + '/'
SOURCE = 'research/' + NAME + '/'
FIGURE = 'FIG_P_BRANCHES_APPL_ETT.png'
REVIEWED = ('RUNS.csv','ALL_NEW_RUNS.csv','SUMMARY.csv','CONTRASTS.csv','PAIRED_DIFFERENCES.csv',
            'JOB_STATUS.csv','SLIDE_NUMBERS.md','PREDICTION_CHECK.md','RESULTS_KO.md',
            'ANALYSIS_VALIDATION.json','EXECUTION_COMPLETION.json',FIGURE,'FIG_P_BRANCHES_APPL_ETT.svg')
REQUIRED = REVIEWED + ('ROOT_RESULTS_GATE.json','RUN_PLAN.json','PREREGISTRATION.json','PREREGISTRATION.md',
                       'REGISTRATION_SEAL.json','REFERENCE_RUNS.json','PRESERVATION_BASELINE.json')
MANIFEST = 'BUNDLE_MANIFEST.json'
RECEIPT = 'BUNDLE_RECEIPT.json'
ZIP_NAME = NAME + '_bundle.zip'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def safe_path(repo, relative):
    require(isinstance(relative,str) and '\\' not in relative and ':' not in relative,'Evidence path must be repository-relative with slashes')
    pure=PurePosixPath(relative)
    require(not pure.is_absolute() and pure.parts and '..' not in pure.parts,'Evidence path escapes repository')
    path=repo/relative
    require(path.resolve().is_relative_to(repo.resolve()) and not path.is_symlink(),'Evidence is an escaping path or symlink')
    return path


def exclusion_reason(relative):
    pure=PurePosixPath(relative);parts={p.lower() for p in pure.parts};name=pure.name.lower()
    if parts & {'.cache','__pycache__','.pytest_cache','private','presentation','presentations','predictions','checkpoints','raw_data','datasets','draft','drafts','secrets','credentials'}:
        return 'Private, raw-data, cache or presentation directory'
    if name.startswith(('private_','raw_host','raw_counter','identity_snapshot','desktop_policy','speaker_notes','draft_','secret_','credential_')) or name in {'secret.json','secrets.json','credentials.json'}:
        return 'Private host bytes or presentation notes'
    if name.startswith('bundle_') or name == 'publication_precheck.json':
        return 'Packaging/publication sidecar excluded to avoid recursion'
    if pure.suffix.lower() not in {'.py','.ps1','.sh','.json','.csv','.md','.png','.svg','.patch'}:
        return 'Unsupported public type, including arrays/checkpoints/logs/presentations'
    return None


def prepare_results(repo=None):
    repo=Path(runtime.REPO if repo is None else repo).resolve()
    report=repo/REPORT
    require(all((report/name).is_file() for name in REQUIRED),'Required public result is missing')
    gate=read(report/'ROOT_RESULTS_GATE.json')
    require(gate.get('status')=='PASS' and gate.get('scope')=='EXPERIMENT_RESULTS_ONLY','Root result-only review must pass')
    bindings={}
    def bind(relative,digest):
        path=safe_path(repo,relative)
        require(path.is_file() and sha(path)==digest,'Bound evidence missing or changed: '+relative)
        require(relative not in bindings or bindings[relative]==digest,'Conflicting evidence hash')
        bindings[relative]=digest
        return path
    bind(REPORT+'ROOT_RESULTS_GATE.json',sha(report/'ROOT_RESULTS_GATE.json'))
    analysis=read(bind(REPORT+'ANALYSIS_VALIDATION.json',gate['analysis_sha256']))
    completion=read(bind(REPORT+'EXECUTION_COMPLETION.json',gate['completion_sha256']))
    require(analysis.get('status')==analysis.get('integrity_status')=='PASS' and analysis.get('T_numeric_values_parsed')==0,
            'Numeric integrity validation and zero protected-T reads are required')
    require(completion.get('status') in {'TRAINING_COMPLETE','TRAINING_INCOMPLETE','SCIENTIFIC_SUSPEND','UNKNOWN_SUSPEND','USER_CANCELLED','STOPPED'}
            and completion.get('active') is None,'Execution must be terminal without an active worker')
    jobs=validate_plan(read(report/'RUN_PLAN.json'));ids={j['job_id'] for j in jobs};statuses=completion['run_statuses']
    require(set(statuses)==ids,'Terminal disposition must cover all sixty jobs')
    complete=sum(v=='COMPLETE' for v in statuses.values())
    require(analysis.get('completed_count')==complete and analysis.get('base_complete')==complete,'Analysis/execution completion counts differ')
    require((completion['status']=='TRAINING_COMPLETE')==(complete==60),'TRAINING_COMPLETE does not match actual complete jobs')
    primary=0
    for dataset in runtime.DATASETS:
        for method in ('P_TIME_ONLY','P_CHANNEL_ONLY'):
            group=[j for j in jobs if (j['dataset'],j['method_id'])==(dataset,method)]
            if all(statuses[j['job_id']]=='COMPLETE' for j in group):primary+=10
    require(analysis.get('primary_count')==primary,'Primary complete-family count differs')
    partial=complete!=60
    require(not partial or gate.get('allow_partial') is True and bool(str(gate.get('partial_reason','')).strip()),'Partial/zero delivery requires an explicit truthful root review')
    visual=gate.get('figures_visual_review',{})
    require(visual.get(FIGURE,{}).get('status')=='PASS','Scientific PNG needs actual visual review')
    bind(REPORT+FIGURE,visual[FIGURE]['sha256'])
    reviewed=gate.get('reviewed_files',{})
    require({REPORT+n for n in REVIEWED}.issubset(reviewed),'Root review must bind all result tables/docs/figures')
    for relative,digest in reviewed.items():
        require(relative.startswith((REPORT,SOURCE)) and exclusion_reason(relative) is None,'Reviewed member is not public campaign evidence')
        bind(relative,digest)
    verified=analysis.get('verified_files',{})
    require(verified.get(REPORT+'EXECUTION_COMPLETION.json')==gate['completion_sha256'],'Analysis does not bind terminal execution')
    for relative,digest in verified.items():
        require(relative.startswith((REPORT,SOURCE)) and exclusion_reason(relative) is None,'Verified member is outside public campaign paths')
        bind(relative,digest)
    for name,digest in read(report/'REGISTRATION_SEAL.json')['files'].items():
        bind(REPORT+name,digest)
    for relative,digest in read(report/'PREREGISTRATION.json')['registered_sources'].items():
        require(relative.startswith('research/') and relative.endswith('.py'),'Registered source is not research Python')
        bind(relative,digest)
    files={}
    for prefix in (SOURCE,REPORT):
        for path in sorted((repo/prefix).rglob('*')):
            if not path.is_file():continue
            relative=path.relative_to(repo).as_posix();safe_path(repo,relative)
            if exclusion_reason(relative) is not None:continue
            digest=sha(path)
            require(relative not in bindings or bindings[relative]==digest,'Evidence changed during collection')
            files[relative]=dict(path=relative,sha256=digest,bytes=path.stat().st_size)
    own=SOURCE+Path(__file__).name
    require(own in files and files[own]['sha256']==sha(__file__),'Exact executing packager source must be included')
    metadata=dict(campaign=NAME,scope='EXPERIMENT_RESULTS_ONLY',completion_status=completion['status'],
        completed_count=complete,primary_count=primary,partial=partial,
        rejected_result_count=analysis.get('rejected_result_count',0),
        partial_reason=gate.get('partial_reason') if partial else None,root_gate_sha256=sha(report/'ROOT_RESULTS_GATE.json'),
        analysis_sha256=gate['analysis_sha256'],completion_sha256=gate['completion_sha256'],
        protected_T_numeric_values_read=0,GPU_execution=False,figures_visual_review=visual,
        exclusions='No cache, raw data, prediction arrays, checkpoints, private host bytes, credentials or presentations.')
    return files,metadata


def verify_zip(path,manifest_bytes):
    manifest=json.loads(manifest_bytes);expected={r['path'] for r in manifest['files']}|{REPORT+MANIFEST}
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist();require(len(names)==len(set(names)) and set(names)==expected,'ZIP exact-member set differs')
        require(archive.testzip() is None,'ZIP CRC failed')
        require(archive.read(REPORT+MANIFEST)==manifest_bytes,'Embedded manifest differs')
        for row in manifest['files']:
            value=archive.read(row['path'])
            require(len(value)==row['bytes'] and hashlib.sha256(value).hexdigest()==row['sha256'],'ZIP member hash/size differs')
    return dict(status='PASS',checked_members=len(expected),checked_payload_files=len(manifest['files']),
                exact_members='PASS',crc='PASS',all_member_sha256_and_sizes='PASS')


def matching_manifest(manifest,files,metadata):
    require({r['path']:r for r in manifest['files']}==files,'Existing manifest payload differs from current reviewed evidence')
    require(all(manifest.get(k)==v for k,v in metadata.items()),'Existing manifest metadata differs from current review')


def package_results(repo=None):
    repo=Path(runtime.REPO if repo is None else repo).resolve();report=repo/REPORT
    files,metadata=prepare_results(repo)
    target,manifest_path,receipt_path=(report/name for name in (ZIP_NAME,MANIFEST,RECEIPT))
    if target.exists():
        with zipfile.ZipFile(target) as archive:manifest_bytes=archive.read(REPORT+MANIFEST)
        matching_manifest(json.loads(manifest_bytes),files,metadata)
    else:
        require(not receipt_path.exists(),'Receipt exists without its ZIP; preserve and investigate')
        if manifest_path.exists():
            manifest_bytes=manifest_path.read_bytes();matching_manifest(json.loads(manifest_bytes),files,metadata)
        else:
            manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),**metadata,files=[files[k] for k in sorted(files)])
            manifest_bytes=(json.dumps(manifest,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf8')
        with tempfile.TemporaryDirectory(prefix='bundle_',dir=report) as temporary:
            temp=Path(temporary)/ZIP_NAME
            with zipfile.ZipFile(temp,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
                for relative in sorted(files):archive.write(repo/relative,relative)
                archive.writestr(REPORT+MANIFEST,manifest_bytes)
            verify_zip(temp,manifest_bytes)
            os.link(temp,target)
    check=verify_zip(target,manifest_bytes)
    if manifest_path.exists():require(manifest_path.read_bytes()==manifest_bytes,'Existing sidecar manifest differs')
    else:
        with manifest_path.open('xb') as handle:handle.write(manifest_bytes)
    receipt=dict(status='PASS',**metadata,zip_path=target.relative_to(repo).as_posix(),zip_absolute_path=str(target),
        zip_sha256=sha(target),zip_bytes=target.stat().st_size,manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        production_readback=check)
    if receipt_path.exists():
        require(read(receipt_path)==receipt,'Existing bundle receipt differs')
    else:
        with receipt_path.open('x',encoding='utf8',newline='\n') as handle:json.dump(receipt,handle,indent=2,ensure_ascii=False);handle.write('\n')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
    if args.check_only:
        files,metadata=prepare_results();value=dict(status='PASS',check_only=True,payload_files=len(files),**metadata)
    else:value=package_results()
    print(json.dumps(value,ensure_ascii=False))


if __name__=='__main__':main()
