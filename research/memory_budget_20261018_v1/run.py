"""Serial, resumable independent-process measurements; no implicit experiment."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from . import common as c


def _git(*args, cwd=c.REPO):
    return subprocess.check_output(['git', *args], cwd=cwd).decode().strip()


def _env():
    env = os.environ.copy()
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[name] = '2'
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONUNBUFFERED='1')
    return env


def _facets(rows):
    from .analysis import schedule
    facets = []
    for row in rows:
        family = row['family']
        width = None if family in ('DIRECT_G', 'RAW_RIDGE') else (
            row['K'] if family in ('B2', 'B0') else row['C'])
        for batch in (1, 8):
            for chunk in schedule(width, batch, 6 if row['dataset'] == 'jena' else 8):
                suffix = 'NA' if chunk is None else str(chunk)
                facets.append(dict(facet_id=row['run_id']+f'_B{batch}_R{suffix}',
                    accuracy_run_id=row['run_id'], dataset=row['dataset'], family=family,
                    seed=row['seed'], B=batch, batch_size=batch, C=row['C'], K=row['K'],
                    row_microbatch=chunk, device='cpu' if family == 'RAW_RIDGE' else 'cuda',
                    is_original=chunk is None or chunk == min(128, batch*width),
                    checkpoint_sha256=(row.get('bundle_binding') or {}).get('sha256', row.get('model_sha'))))
    return facets


def prepare(asset_root, original_root, attachment, presentation_paths):
    p = c.paths()
    p['cache'].mkdir(parents=True, exist_ok=True)
    p['report'].mkdir(parents=True, exist_ok=True)
    state_path = p['report']/'START_STATE.json'
    if state_path.exists():
        state = c.read(state_path)
        if state['protocol_sha256'] != c.sha(p['report']/'PROTOCOL.md'):
            raise ValueError('Preregistered protocol changed')
        _finish_prepare(p)
        return state
    dirty = subprocess.check_output(['git', 'status', '--porcelain=v1', '-z'], cwd=original_root)
    files = subprocess.check_output(['git', 'ls-files', '-z'], cwd=c.REPO).decode().split('\0')
    protected = {name: c.sha(c.REPO/name) for name in files if name and (c.REPO/name).is_file()}
    attachments = {Path(path).name: dict(path=str(Path(path).resolve()), sha256=c.sha(path))
                   for path in presentation_paths}
    guard = dict(original_root=str(Path(original_root).resolve()), original_HEAD=_git('rev-parse', 'HEAD', cwd=original_root),
        original_status_z_sha256=hashlib.sha256(dirty).hexdigest(), original_status_entries=dirty.count(b'\0'),
        protected_files=protected, presentations=attachments)
    c.sealed_json(p['cache']/'PRESERVATION_BASELINE.json', guard)
    parent_ctx = c.source_context(asset_root)
    rows = c.selected_rows(parent_ctx)
    c.sealed_json(p['cache']/'SOURCE_ROWS.json', rows)
    facets = _facets(rows)
    if len(facets) != 46 or sum(x['device'] == 'cuda' for x in facets) != 44:
        raise ValueError('Phase A preregistered coverage differs')
    c.sealed_json(p['cache']/'FACETS_A.json', facets)
    state = dict(campaign=c.NAME, started_unix=c.STARTED_UNIX, prepared_unix=time.time(),
        parent_commit=_git('rev-parse', 'HEAD'), origin_main=_git('ls-remote', 'origin', 'refs/heads/main').split()[0],
        protocol_sha256=c.sha(p['report']/'PROTOCOL.md'), attachment_sha256=c.sha(attachment),
        original_HEAD=guard['original_HEAD'], original_status_z_sha256=guard['original_status_z_sha256'],
        original_status_entries=guard['original_status_entries'], protected_tracked_files=len(protected),
        presentation_sha256={name:value['sha256'] for name,value in attachments.items()},
        operating_seconds=c.OPERATING_SECONDS, scientific_deadline_unix=c.SCIENTIFIC_DEADLINE,
        report_reserve_seconds=c.REPORT_RESERVE_SECONDS, gpu_facets=44, cpu_facets=2,
        T_reopened=False, V_independent=False, original_dirty_content_hash_claimed=False)
    c.sealed_json(state_path, state)
    c.sealed_json(p['cache']/'CAMPAIGN.json', dict(started_unix=c.STARTED_UNIX,
        asset_root=str(Path(asset_root).resolve()), state_sha256=c.sha(state_path),
        old_ledgers=[dict(reference='repo://.cache/'+name+'/CAMPAIGN.json', sha256=c.sha(c.REPO/'.cache'/name/'CAMPAIGN.json'))
            for name in ('adapter_diagnostics_20261017_v2', 'adapter_improvements_20261018_v1', 'b2_revin_20261018_v1')]))
    _finish_prepare(p)
    return state


def _finish_prepare(p):
    import numpy, torch, psutil
    from threadpoolctl import threadpool_info
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu', '--format=csv,noheader'], text=True).strip()
    if not (p['report']/'ENVIRONMENT.json').exists():
        c.sealed_json(p['report']/'ENVIRONMENT.json', dict(platform=platform.platform(), python=platform.python_version(),
        runtime='pinned project .venv; NumPy OpenBLAS', torch=torch.__version__, numpy=numpy.__version__,
        cuda_build=torch.version.cuda, gpu_start_snapshot=gpu, cpu_logical=psutil.cpu_count(),
        blas_threads=2, threadpool_libraries=[{k:v for k,v in item.items() if k!='filepath'} for item in threadpool_info()],
        dtype='float32', metric_dtype='float64', AMP=False, quantization=False, compile=False,
            model_downloads=0, data_downloads=0, measurement_scope='allocated tensors; PID samples if supported'))
    if not (p['report']/'EXECUTION_SOURCE_START.json').exists():
        c.sealed_json(p['report']/'EXECUTION_SOURCE_START.json', c.source_hashes())
    c.sealed_json(p['cache']/'PREPARE_COMPLETE.json',dict(start_state_sha256=c.sha(p['report']/'START_STATE.json'),
        environment_sha256=c.sha(p['report']/'ENVIRONMENT.json')))


def _resource_check(cache):
    """Block other live Python compute jobs; keep desktop GPU applications intact."""
    import psutil
    snapshots = []
    for prefix in ('adapter_diagnostics_20261017_v2', 'adapter_improvements_20261018_v1', 'b2_revin_20261018_v1'):
        active = c.REPO/'.cache'/prefix/'ACTIVE_GPU_LEASE.json'
        if active.exists():
            info = c.read(active)
            for identity in [info, *info.get('authorized_children', [])]:
                try:
                    proc = psutil.Process(int(identity['pid']))
                    if abs(proc.create_time()-float(identity['create_time'])) < .001:
                        raise RuntimeError('BLOCKED_RESOURCE: another campaign has a live GPU lease')
                except psutil.NoSuchProcess:
                    pass
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader'], text=True)
    gpu_snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,memory.total,utilization.gpu',
                                          '--format=csv,noheader,nounits'],text=True).strip()
    used,total,utilization=map(float,gpu_snapshot.split(','))
    if utilization>=80 or used/total>=.90:
        raise RuntimeError('BLOCKED_RESOURCE: high external GPU occupancy before facet')
    ancestors = {os.getpid(), *(proc.pid for proc in psutil.Process().parents())}
    for line in raw.splitlines():
        values = [value.strip() for value in line.split(',')]
        if len(values) < 3:
            continue
        pid = int(values[0]); name = Path(values[1]).name
        snapshots.append(dict(pid=pid, process=name, memory=values[-1]))
        if pid not in ancestors and name.lower() in ('python.exe', 'python', 'python3', 'pythonw.exe'):
            raise RuntimeError('BLOCKED_RESOURCE: an unrelated Python GPU process is active')
    value=dict(checked_unix=time.time(),processes=snapshots,device_used_MiB=used,
               device_total_MiB=total,utilization_percent=utilization,
               device_values_are_not_PID_memory=True)
    c.atomic_json(cache/'RESOURCE_CHECK.json',value)
    with (cache/'RESOURCE_CHECKS.jsonl').open('a',encoding='utf-8') as handle:
        handle.write(json.dumps(value,sort_keys=True)+'\n')


def receipts():
    return [c.read(path) for path in sorted((c.paths()['cache']/'facets').glob('*.receipt.json'))
            if c.read(path).get('status') in ('COMPLETE', 'INVALID')]


def execute(stage, phase='A', retry_failed=False):
    from research.informs_broad_validation_20261015_v1.resources import GPUReservation
    from .analysis import select_confirmations, export_receipts
    p = c.paths(); state = c.read(p['report']/'START_STATE.json'); ledger = c.read(p['cache']/'CAMPAIGN.json')
    if state['protocol_sha256'] != c.sha(p['report']/'PROTOCOL.md'):
        raise ValueError('Protocol SHA changed after registration')
    facets = c.read(p['cache']/f'FACETS_{phase}.json')
    parent_rows=c.read(p['cache']/('SOURCE_ROWS.json' if phase=='A' else 'B_PHASE_ROWS.json'))
    if phase=='A':
        current=c.selected_rows(c.source_context(ledger['asset_root']))
        if c.fingerprint(current)!=c.fingerprint(parent_rows):
            raise ValueError('Current parent accuracy identities differ from sealed SOURCE_ROWS')
    parent_by_id={value['run_id']:value for value in parent_rows}
    if stage == 'confirm':
        screens = [x['summary'] for x in receipts() if x['summary']['stage'] == 'screen'
                   and x['facet_id'] in {f['facet_id'] for f in facets}]
        if len(screens) != len(facets):
            raise ValueError('All terminal screen attempts required before confirm selection')
        selected = select_confirmations(screens)
        c.sealed_json(p['report']/f'CONFIRM_SELECTION_{phase}.json', selected)
        ids = {x['facet_id'] for x in selected if x['confirm_selected']}
        facets = [x for x in facets if x['facet_id'] in ids]
    _resource_check(p['cache'])
    p['cache'].joinpath('facets').mkdir(exist_ok=True)
    p['cache'].joinpath('logs').mkdir(exist_ok=True)
    # Phase A is mandatory: an operating-plan overrun is recorded, never silently skipped.
    deadline = max(c.SCIENTIFIC_DEADLINE, time.time()+3600) if phase == 'A' else c.SCIENTIFIC_DEADLINE
    with GPUReservation(p['cache'], total_gpu_seconds=1e12, reserve_seconds=0,
                        wall_deadline=deadline, purpose='audit', kind=f'{phase}_{stage}') as lease:
        for index, facet in enumerate(facets):
            lease.check()
            _resource_check(p['cache'])
            stem = facet['facet_id']+'_'+stage
            output = p['cache']/'facets'/(stem+'.receipt.json')
            request = dict(asset_root=ledger['asset_root'], resource_cache=str(p['cache']),
                accuracy_run_id=facet['accuracy_run_id'], batch_size=facet['batch_size'],
                row_microbatch=facet['row_microbatch'], stage=stage, facet_id=facet['facet_id'],
                protocol_sha256=state['protocol_sha256'], phase=phase,
                expected_candidate_sha256=c.fingerprint(parent_by_id[facet['accuracy_run_id']]))
            if phase == 'B':
                request['candidate_file'] = str(p['cache']/'B_PHASE_ROWS.json')
                request['candidate_file_sha256'] = c.sha(p['cache']/'B_PHASE_ROWS.json')
            request_path = p['cache']/'facets'/(stem+'.request.json')
            c.sealed_json(request_path, request)
            if output.exists():
                old = c.read(output)
                if old.get('status') == 'COMPLETE' and old.get('request_identity_sha256') == c.fingerprint(request):
                    print('REUSED', stem, flush=True)
                    continue
                if old.get('status') == 'INVALID' and not retry_failed:
                    print('RETAINED_INVALID',stem,flush=True)
                    continue
                if not retry_failed:
                    raise RuntimeError('Previous attempt preserved; retry with --retry-failed after repair: '+stem)
                import psutil
                owner = old.get('process', {})
                if owner.get('pid') and psutil.pid_exists(owner['pid']):
                    observed = psutil.Process(owner['pid']).create_time()
                    if abs(observed-float(owner.get('create_time', 0))) < .001:
                        raise RuntimeError('Cannot retry a live worker')
                archive=p['cache']/'attempts'; archive.mkdir(exist_ok=True)
                digest=c.sha(output)
                output.rename(archive/(stem+'.'+digest+'.receipt.json'))
                previous_log=p['cache']/'logs'/(stem+'.log')
                if previous_log.exists():
                    previous_log.rename(archive/(stem+'.'+digest+'.log'))
            log_path=p['cache']/'logs'/(stem+'.log')
            process_path=p['cache']/'facets'/(stem+'.process.json')
            if log_path.exists():
                if not retry_failed:
                    raise RuntimeError('Orphan log retained; inspect then use --retry-failed: '+stem)
                import psutil
                owner=c.read(process_path) if process_path.exists() else {}
                if owner.get('pid') and owner.get('create_time') and psutil.pid_exists(owner['pid']):
                    if abs(psutil.Process(owner['pid']).create_time()-owner['create_time'])<.001:
                        raise RuntimeError('Orphan worker is still alive')
                archive=p['cache']/'attempts'; archive.mkdir(exist_ok=True)
                log_path.rename(archive/(stem+'.orphan.'+c.sha(log_path)+'.log'))
            env = _env()
            if facet['device'] == 'cuda':
                env.update(lease.child_environment())
            print(f'RUN {index+1}/{len(facets)} {stem}', flush=True)
            started = time.time()
            with log_path.open('x', encoding='utf-8') as handle:
                import psutil
                child=subprocess.Popen([sys.executable, '-m', 'research.'+c.NAME+'.worker',
                    '--request', str(request_path), '--output', str(output)], cwd=c.REPO, env=env,
                    stdout=handle, stderr=subprocess.STDOUT)
                try:
                    created=psutil.Process(child.pid).create_time()
                except psutil.NoSuchProcess:
                    created=None
                process=dict(pid=child.pid,create_time=created,started_unix=started,
                             facet_id=facet['facet_id'],stage=stage)
                c.atomic_json(process_path,process)
                exit_code=child.wait()
                process.update(exit_code=exit_code,finished_unix=time.time())
                c.atomic_json(process_path,process)
            if not output.exists():
                c.sealed_json(output,dict(status='INVALID',facet_id=facet['facet_id'],
                    request_identity_sha256=c.fingerprint(request),
                    protocol_sha256=state['protocol_sha256'],
                    process=dict(process,wall_seconds=time.time()-started),
                    summary=dict(facet,stage=stage,status='INVALID',parity_status='FAIL_OR_NOT_COMPLETED'),
                    failure=dict(exception_type='WORKER_NO_RECEIPT',message='Worker exited before receipt; private log preserved')))
            receipt = c.read(output) if output.exists() else dict(status='WORKER_NO_RECEIPT')
            print('DONE', stem, receipt['status'], round(time.time()-started, 2), 'seconds', flush=True)
            c.atomic_json(p['report']/'STATUS.json', dict(status='RUNNING', phase=phase, stage=stage,
                completed=index+1, total=len(facets), last_facet=stem, last_status=receipt['status'], updated_unix=time.time()))
            if exit_code or receipt['status'] != 'COMPLETE':
                print('INVALID_EXCLUDED',stem,'evidence retained',flush=True)
    export_receipts(receipts(), p['report'], make_figures=False)


def portable_receipts():
    p=c.paths(); directory=p['report']/'receipts'; directory.mkdir(exist_ok=True)
    result=[]
    for source in sorted((p['cache']/'facets').glob('*.receipt.json')):
        item=c.read(source)
        if item['status'] not in ('COMPLETE','INVALID'):
            continue
        # Forecast arrays and full private parent records stay in the private cache.
        item.pop('accuracy',None)
        item['private_receipt_sha256']=c.sha(source)
        target=directory/source.name
        c.sealed_json(target,item)
        result.append(target)
    return result


def export(make_figures=True):
    from .analysis import export_receipts
    return export_receipts(portable_receipts(), c.paths()['report'], make_figures=make_figures)


def gate():
    from .analysis import _summary, phase_b_decision, select_confirmations
    p=c.paths(); existing=p['report']/'PHASE_B_DECISION.json'
    if existing.exists():
        return c.read(existing)
    values=receipts(); facets=c.read(p['cache']/'FACETS_A.json')
    ids={value['facet_id'] for value in facets}
    screen=[_summary(value) for value in values if value.get('summary',{}).get('stage')=='screen' and value['facet_id'] in ids]
    if len(screen)!=46:
        raise ValueError('Phase A decision requires all forty-six terminal screens')
    selection=select_confirmations(screen)
    expected={value['facet_id'] for value in selection if value['confirm_selected']}
    confirmed=[_summary(value) for value in values if value.get('summary',{}).get('stage')=='confirm' and value['facet_id'] in ids]
    if {value['facet_id'] for value in confirmed}!=expected or any(value['status']!='COMPLETE' or value['parity_status']!='PASS' for value in confirmed):
        raise ValueError('Phase A selected confirmations are incomplete')
    old_rows=c.read(p['cache']/'SOURCE_ROWS.json')
    controls=[value for value in old_rows if value['dataset']=='jena' and value['family']=='B2']
    old_seconds=sum(value['search_seconds']+value['refit_seconds'] for value in controls)
    training_estimate=old_seconds*(3+12)/6*1.20
    b2_screen=[value['process_wall_seconds'] for value in screen if value['dataset']=='jena' and value['family']=='B2']
    b2_confirm=[value['process_wall_seconds'] for value in confirmed if value['dataset']=='jena' and value['family']=='B2']
    # Up to20 new screens and20 confirms, with2x the slowest existing B2 facet wall.
    measurement_estimate=20*2*(max(b2_screen)+max(b2_confirm))
    decision=phase_b_decision(confirmed,remaining_seconds=max(0,c.STARTED_UNIX+c.OPERATING_SECONDS-time.time()),
        estimated_training_seconds=training_estimate, estimated_measurement_seconds=measurement_estimate,
        reserve_seconds=c.REPORT_RESERVE_SECONDS)
    invalid=[value['facet_id'] for value in screen if value['status']!='COMPLETE']
    if invalid:
        decision.update(status='SKIPPED_GATE',reason='INVALID_PHASE_A_COMPARISON',invalid_facets=invalid)
    context=dict(p,asset=Path(c.read(p['cache']/'CAMPAIGN.json')['asset_root']))
    evidence=[c.binding(path,context) for path in sorted((p['cache']/'facets').glob('*.receipt.json'))]
    decision.update(eligible=decision['status']=='PROCEED',phase_a_all_confirmed=not invalid,
        phase_a_valid_screens=46-len(invalid),phase_a_confirmed=len(confirmed),protocol_binding=c.binding(p['report']/'PROTOCOL.md',context),
        phase_a_evidence=evidence,decided_unix=time.time(),
        budget=dict(scientific_deadline_unix=c.SCIENTIFIC_DEADLINE,estimated_training_seconds=training_estimate,
                    estimated_measurement_seconds=measurement_estimate,old_two_K6_search_refit_seconds=old_seconds,
                    training_estimate_formula='old2seed_K6_seconds * (3+12)/6 *1.20',
                    measurement_estimate_formula='20*2*(max JenaB2 screen process wall + max JenaB2 confirm process wall)'),
        resume_command='python -m research.memory_budget_20261018_v1.run train-b')
    decision['decision_identity']=c.fingerprint(decision)
    c.sealed_json(existing,decision)
    return decision


def train_b():
    from research.informs_broad_validation_20261015_v1.resources import GPUReservation
    p=c.paths(); decision=gate()
    if decision['status']!='PROCEED':
        print('SKIPPED_GATE',decision['reason'],flush=True)
        return
    ledger=c.read(p['cache']/'CAMPAIGN.json')
    _resource_check(p['cache'])
    summary=p['report']/'B_TRAINING_SUMMARY.json'
    if not summary.exists():
        with GPUReservation(p['cache'],total_gpu_seconds=1e12,reserve_seconds=0,
                            wall_deadline=c.SCIENTIFIC_DEADLINE,purpose='training',kind='B_TRAINING') as lease:
            env=_env(); env.update(lease.child_environment())
            with (p['cache']/'B_training.log').open('a',encoding='utf-8') as handle:
                result=subprocess.run([sys.executable,'-m','research.'+c.NAME+'.training',
                    '--asset-root',ledger['asset_root'],'--gate',str(p['report']/'PHASE_B_DECISION.json'),
                    '--resource-cache',str(p['cache'])],cwd=c.REPO,env=env,stdout=handle,stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError('Phase B training interrupted; exact checkpoints and log preserved')
    if c.read(summary)['status']!='COMPLETE':
        raise ValueError('Phase B training summary is incomplete')
    facets=_facets(c.read(p['cache']/'B_PHASE_ROWS.json'))
    if len(facets)!=20:
        raise ValueError('New K row schedule must contain exactly20 facets')
    c.sealed_json(p['cache']/'FACETS_B.json',facets)
    print('PHASE_B_TRAINED',flush=True)


def preservation():
    p=c.paths(); old=c.read(p['cache']/'PRESERVATION_BASELINE.json')
    changed=[name for name,digest in old['protected_files'].items()
             if not (c.REPO/name).is_file() or c.sha(c.REPO/name) != digest]
    # An explicitly documented history append is allowed, never an old result overwrite.
    allowed=['_docs/history/2026-10-18.md']
    disallowed=[name for name in changed if name not in allowed]
    for name in changed:
        if name in allowed:
            base=c.read(p['report']/'START_STATE.json')['parent_commit']
            prefix=subprocess.check_output(['git','show',base+':'+name],cwd=c.REPO)
            if hashlib.sha256(prefix).hexdigest()!=old['protected_files'][name] or not (c.REPO/name).read_bytes().startswith(prefix):
                disallowed.append(name)
    raw=subprocess.check_output(['git','status','--porcelain=v1','-z'],cwd=old['original_root'])
    originals_ok=(_git('rev-parse','HEAD',cwd=old['original_root'])==old['original_HEAD']
                  and hashlib.sha256(raw).hexdigest()==old['original_status_z_sha256'])
    presentation_ok=all(c.sha(value['path'])==value['sha256'] for value in old['presentations'].values())
    proof=dict(status='PASS' if not disallowed and originals_ok and presentation_ok else 'FAIL',
        protected_files=len(old['protected_files']), changed_tracked_files=changed,
        allowed_history_append=allowed, original_HEAD_status_preserved=originals_ok,
        presentations_preserved=presentation_ok, checked_unix=time.time())
    c.atomic_json(p['report']/'PRESERVATION.json',proof)
    if proof['status']!='PASS':
        raise ValueError('Preservation check failed')
    return proof


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare','screen','confirm','export','preservation','gate','train-b'))
    parser.add_argument('--phase',choices=('A','B'),default='A')
    parser.add_argument('--asset-root')
    parser.add_argument('--original-root')
    parser.add_argument('--attachment')
    parser.add_argument('--presentation',action='append',default=[])
    parser.add_argument('--retry-failed',action='store_true')
    args=parser.parse_args()
    if args.action=='prepare':
        print(json.dumps(prepare(args.asset_root,args.original_root,args.attachment,args.presentation)))
    elif args.action in ('screen','confirm'):
        execute(args.action,args.phase,args.retry_failed)
    elif args.action=='export':
        export()
    elif args.action=='gate':
        print(json.dumps(gate()))
    elif args.action=='train-b':
        train_b()
    else:
        print(json.dumps(preservation()))


if __name__=='__main__':
    main()
