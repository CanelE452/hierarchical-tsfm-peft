"""Resolve the preserved first run and Beijing-only faithful loader repair."""
from pathlib import Path
import argparse
import time
import engine as en


def run(tag,firsttag,repairtag):
    output=en.HERE/('run_'+tag); output.mkdir(exist_ok=False)
    paths=[en.HERE/('run_'+firsttag)/'RESULT.json',en.HERE/('run_'+repairtag)/'RESULT.json']
    sources=[en.ex.read(p) for p in paths]
    if any(s['status'] not in ('COMPLETE','PARTIAL_FAILED_PATHS') for s in sources): raise ValueError('TERMINAL_SOURCES_REQUIRED')
    cases={}
    for source,path in zip(sources,paths):
        for case in source['cases']:
            key=(case['dataset'],case['label'])
            if key in cases: raise ValueError('UNAUTHORIZED_DUPLICATE_NEW_TRAINING_CASE '+str(key))
            cases[key]=case
    records=list(cases.values())
    main=sum('training' in c for c in records)
    native=sum('native_training' in m for c in records for m in c['methods'])
    method_count=sum(len(c['methods']) for c in records)
    datasets={c['dataset'] for c in records}
    expected_datasets={'jena','robin','appliances','beijing_aotizhongxin'}
    complete=main==44 and native==16 and method_count==124 and len(records)==64 and datasets==expected_datasets
    receipts={k:v for source in sources for k,v in source.get('evaluation_open_receipts',{}).items()}
    complete=complete and set(receipts)=={'appliances','beijing_aotizhongxin'}
    original_beijing_failures=[f for f in sources[0].get('failed_paths',[]) if f['dataset']=='beijing_aotizhongxin' and f['label']=='DATA_PREPARATION']
    other_failures=[f for source in sources for f in source.get('failed_paths',[]) if f not in original_beijing_failures]
    complete=complete and not other_failures
    hashes={}
    for source in sources:
        for path,digest in source['source_hashes'].items():
            if path in hashes and hashes[path]!=digest: raise ValueError('SOURCE_HASH_DISAGREEMENT '+path)
            hashes[path]=digest
    ownpath=Path(__file__)
    hashes[str(ownpath.relative_to(en.ROOT))]=en.ex.sha(ownpath)
    snapshotfolder=output/'sources';snapshotfolder.mkdir()
    with (snapshotfolder/ownpath.name).open('xb') as f:f.write(ownpath.read_bytes())
    parent_resources=[s['resources'] for s in sources]
    latest=max(parent_resources,key=lambda r:r['started_unix'])
    resources={'aggregated_not_additional_process':True,'source_resources':parent_resources,
        'wall_seconds':sum(r['wall_seconds'] for r in parent_resources),
        'cumulative_gpu_process_wall_seconds':latest['cumulative_gpu_process_wall_seconds'],
        'peak_sampled_rss_bytes':max(r['peak_sampled_rss_bytes'] for r in parent_resources),
        'peak_allocated_vram_bytes':max(r['peak_allocated_vram_bytes'] for r in parent_resources),
        'new_bytes':latest.get('new_bytes'),
        'note':'Reference aggregate of two already-recorded source processes; not registered as a resource ledger and never added again to GPU totals'}
    result={'status':'COMPLETE_RESOLVED_WITH_DECLARED_PROTOCOL_DEVIATION' if complete else 'PARTIAL_RESOLVED',
        'created_unix':time.time(),'cases':records,'training_fits':main,'native_fits':native,'method_count':method_count,
        'source_results':[{'path':str(p.relative_to(en.ROOT)),'sha256':en.ex.sha(p),'status':s['status']} for p,s in zip(paths,sources)],
        'source_hashes':hashes,'evaluation_open_receipts':receipts,'resources':resources,
        'resolver_provenance':{'path':str(ownpath.relative_to(en.ROOT)),'sha256':en.ex.sha(ownpath),'new_training_updates':0,
            'snapshot':str((snapshotfolder/ownpath.name).relative_to(en.ROOT))},
        'method_lock':sources[0]['method_lock'],'global_model_selection_locks':[s['global_model_selection_lock'] for s in sources],
        'original_failed_paths_preserved':original_beijing_failures,'unresolved_failed_paths':other_failures,
        'protocol_deviation':{'joint_models_before_any_new_EVAL':False,
            'actual_sequence':'Appliances FIT/TUNE models locked and first EVAL completed; Beijing raw loader failed before model fitting. Faithful time/parser repair only, unchanged METHOD_LOCK, then Beijing FIT/TUNE locks and its first EVAL.',
            'Beijing_tuning_changed_after_Appliances_EVAL':False,
            'claim_limit':'Two separate dataset-specific locked first evaluations under the original fixed development method policy; original joint-lock protocol was not met.'}}
    en.write(output/'RESULT.json',result)
    print(result['status'],len(records),main,native,method_count)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--tag',required=True); parser.add_argument('--firsttag',default='new_v1'); parser.add_argument('--repairtag',default='beijing_repair_v1')
    args=parser.parse_args()
    if not args.tag.replace('_','').isalnum(): raise ValueError('SAFE_TAG_REQUIRED')
    run(args.tag,args.firsttag,args.repairtag)
