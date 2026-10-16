"""Metadata-only by default; explicit ROOT replay IDs only, never selection or T."""
import argparse
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def inside(root,relative):
    p=(root/relative).resolve()
    if Path(relative).is_absolute() or not p.is_relative_to(root):raise ValueError('UNSAFE_RELATIVE_PATH')
    return p
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)

def metadata(plan,public,cache):
    for name,digest in plan['source18'].items():
        if sha(public/name)!=digest:raise ValueError('SOURCE18_CHANGED')
    if sha(__file__)!=plan['wrapper_sha256']:raise ValueError('WRAPPER_CHANGED')
    with (public/'EXPERIMENT_REGISTRY.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    registry={r['experiment_id']:r for r in rows}
    if len(registry)!=len(rows):raise ValueError('DUPLICATE_REGISTRY_ID')
    ids=plan['run_ids']
    if len(set(ids))!=len(ids) or not ids:raise ValueError('EXPLICIT_UNIQUE_IDS_REQUIRED')
    records=[]
    for run in ids:
        entry=dict(run_id=run,status='READY_METADATA_ONLY',checkpoint_assets=[],existing_PASS=False)
        try:
            binding=plan['bindings'][run]
            actual=registry[run]
            if {k:actual.get(k,'') for k in binding['scientific_row']}!=binding['scientific_row']:raise ValueError('REGISTRY_SCIENCE_CHANGED')
            manifest=inside(public,binding['manifest_relative_path'])
            if sha(manifest)!=binding['manifest_sha256']:raise ValueError('V_MANIFEST_CHANGED')
            m=load(manifest)
            if m['run_id']!=run or m['split']!='V_select':raise ValueError('WRONG_RUN_OR_SPLIT')
            if not inside(cache,m['cache_relative_path']).is_file():raise ValueError('V_ARCHIVE_MISSING')
            for p in binding['checkpoint_assets']:
                exists=inside(cache,p['cache_relative_path']).is_file()
                entry['checkpoint_assets'].append(dict(p,exists=exists))
                if not exists:raise ValueError('CHECKPOINT_ASSET_MISSING')
            if binding.get('receipt_relative_path') and sha(inside(public,binding['receipt_relative_path']))!=binding['receipt_sha256']:
                raise ValueError('FIT_RECEIPT_CHANGED')
            previous=plan['existing_PASS29'].get(run)
            if previous:
                path=inside(public,previous['receipt_relative_path'])
                if sha(path)!=previous['receipt_sha256']:raise ValueError('OLD_REPLAY_RECEIPT_CHANGED')
                old=load(path)
                if old.get('status')!='PASS' or old['run_id']!=run or old['standard_manifest_sha256']!=binding['manifest_sha256']:
                    raise ValueError('OLD_PASS_NOT_EXACT_MANIFEST')
                entry['existing_PASS']=True
            entry['metadata_not_actual_route_rebuild']=True
        except Exception as error:entry.update(status='NOT_READY_PRESERVED',error_type=type(error).__name__)
        records.append(entry)
    return dict(status='READY_NOT_EXECUTED' if all(x['status']=='READY_METADATA_ONLY' for x in records) else 'PARTIAL_METADATA_NOT_READY',
        run_ids=ids,records=records,old_PASS_exact_id_intersection=sum(x['existing_PASS'] for x in records),
        planned_new_replays=sum(not x['existing_PASS'] for x in records),source18_unchanged=True,
        GPU_used=False,framework_imports=[n for n in ('torch','numpy','pandas') if n in sys.modules],
        selection_written=False,T_requested=False,registry_written=False)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan-json',type=Path,required=True);p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    public=Path(os.environ['INFORMS_PUBLIC_ROOT']).resolve();cache=Path(os.environ['INFORMS_CACHE_ROOT']).resolve()
    asset=Path(os.environ['INFORMS_ASSET_ROOT']).resolve();plan=load(args.plan_json)
    result=metadata(plan,public,cache)
    if args.execute:
        if any(os.environ.get(k) for k in ('INFORMS_GPU_TOKEN','INFORMS_GPU_LEASE_ID','INFORMS_GPU_OWNER_PID')):raise ValueError('NO_NESTED_GPU_OWNER')
        if (cache/'ACTIVE_GPU_LEASE.json').exists():raise ValueError('PREVIOUS_COST_OR_TRAINING_OWNER_STILL_PRESENT')
        if any(r['status']!='READY_METADATA_ONLY' for r in result['records']):raise ValueError('EXPLICIT_QUEUE_HAS_MISSING_ASSETS')
        if public==cache or public.is_relative_to(cache) or cache.is_relative_to(public):raise ValueError('ROOTS_NOT_DISJOINT')
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
        os.environ.setdefault('CUDA_VISIBLE_DEVICES','0')
        if os.environ['CUDA_VISIBLE_DEVICES']=='-1':raise ValueError('ROOT_GPU_HANDOFF_REQUIRED')
        sys.path.insert(0,str(public))
        resources=importlib.import_module('resources')
        stamp=str(time.time_ns());diagnostic=public/'diagnostics'/('ROOT_explicit_replay_'+stamp)
        started=dict(status='STARTED_REPLAY_ONLY',runtime_pid=os.getpid(),started_unix=time.time(),plan_sha256=sha(args.plan_json),run_ids=plan['run_ids'])
        save(diagnostic/'STARTED.json',started)
        lease=None
        try:
            with resources.GPUReservation(cache,total_gpu_seconds=43200,reserve_seconds=0,wall_deadline=1792223150,purpose='audit',kind='ROOT_final_explicit_V_replay') as lease:
                final=importlib.import_module('final_evaluation')
                result=final.execute_replays(plan['run_ids'],cache,public,lease,lease.deadline-5)
        finally:
            closure=dict(runtime_pid=os.getpid(),finished_unix=time.time(),selection_written_by_wrapper=False,T_requested=False)
            if lease is not None:
                receipt=cache/'gpu_receipts'/(lease.lease_id+'.json')
                if receipt.is_file():closure.update(lease_receipt_cache_relative_path=receipt.relative_to(cache).as_posix(),lease_receipt_sha256=sha(receipt),central_lease=load(receipt))
            closure['source18_unchanged']=all(sha(public/n)==v for n,v in plan['source18'].items())
            save(diagnostic/'CLOSURE.json',closure)
        save(diagnostic/'RESULT.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))

if __name__=='__main__':main()
