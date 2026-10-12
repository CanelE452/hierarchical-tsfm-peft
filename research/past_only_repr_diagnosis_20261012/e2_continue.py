"""Preserve a failed writer attempt; verify complete records and run only missing ones."""
import json
import time
import numpy as np
from threadpoolctl import threadpool_limits
from common import *
from e2 import run_record, MODELS
from synthetic import make_record, CONDITIONS, SYNTHETIC_SEEDS, noiseless_alignment


def independent_score(pred,target,mask,std,mean):
    # Independently reproduce stored physical-space rounding before float64 errors.
    p=pred*std+mean
    y=target*std+mean
    e=np.where(mask,p.astype(np.float64)-y.astype(np.float64),0.)
    n=mask.sum((0,1));good=n>0
    return float(((e*e).sum((0,1))[good]/n[good]).mean()),float((np.abs(e).sum((0,1))[good]/n[good]).mean())


def verify_record(path):
    r=read(path)
    with np.load(r['shared_arrays']['path'],allow_pickle=False) as shared:
        target,mask=shared['val_target'],shared['val_mask']
        std,mean=shared['std'],shared['mean']
        origins=shared['val_origins']
        if ahash(target)!=r['target_hashes']['val'] or ahash(mask)!=r['mask_hashes']['val'] or ahash(origins)!=r['origin_hashes']['val']:
            raise ValueError('Saved shared targets/masks/origins changed')
        tt,mt=shared['tune_target'],shared['tune_mask']
    checks=[]
    for method,item in r['models'].items():
        if sha(item['arrays']['path'])!=item['arrays']['sha256']:
            raise ValueError('Prediction file bytes changed')
        with np.load(item['arrays']['path'],allow_pickle=False) as z:
            mse,mae=independent_score(z['prediction_val'],target,mask,std,mean)
            delta=max(abs(mse-item['val']['mse']),abs(mae-item['val']['mae']))
            if delta > 1e-8:
                raise ValueError('Independent physical metric mismatch')
            checks.append({'method':method,'mse':mse,'mae':mae,'max_abs_delta':delta})
            if method!='PERSISTENCE':
                candidates=[independent_score(z[f'lambda_{i}_prediction_tune'],tt,mt,std,mean)[0] for i in range(5)]
                if int(np.argmin(candidates))!=item['fit']['selected_grid_index']:
                    raise ValueError('TUNE selection mismatch')
    return {'status':'PASS','path':str(path),'result_sha':sha(path),'checks':checks,
            'source_snapshot_binding':{k:str(HERE/'source_snapshots/common_before_io_retry.py') if k=='common.py' and v==sha(HERE/'source_snapshots/common_before_io_retry.py') else str(HERE/k) for k,v in r['source_pins'].items()}}


def main():
    access_guard()
    if not read(HERE/'E1_GATE.json')['safety_pass']:
        raise ValueError('E1 not safe')
    sources={'common_before_io':sha(HERE/'source_snapshots/common_before_io_retry.py'),
             'common_after_io':sha(HERE/'common.py'),'e2':sha(HERE/'e2.py'),'synthetic':sha(HERE/'synthetic.py')}
    with threadpool_limits(2),Budget(1,False,'E2_missing_records_after_writer_failure') as budget:
        cases=[(d,None) for d in ('jena','robin')]+[(c,s) for c in CONDITIONS for s in SYNTHETIC_SEEDS]
        records=[];proofs=[]
        for ds,seed in cases:
            suffix=ds if seed is None else f'{ds}_{seed}'
            path=HERE/'e2'/f'run01_{suffix}.json'
            reused=path.exists()
            if not reused:
                data,arrays=(load_data(ds),None) if seed is None else make_record(ds,seed)
                r=run_record(data,f'run02_{suffix}',budget,arrays)
                path=HERE/'e2'/f'run02_{suffix}.json'
            else:
                r=read(path)
            proof=verify_record(path);proofs.append(proof)
            records.append({'tag':r['tag'],'path':str(path),'sha256':sha(path),'LAG_SIGNAL':r['LAG_SIGNAL'],
                            'reused_complete_same_campaign':reused,'new_science_execution':not reused})
            budget.check()
        result={'status':'COMPLETE_VERIFIED','experiment':'E2','records':records,'synthetic_alignment':noiseless_alignment()['status'],
                'independent_verification':proofs,'source_versions':sources,
                'preserved_partial_attempt':'run01_fixed_92619; failure WinError5 atomic resource replace',
                'numeric_algorithm_changed':False,'GPU_new_seconds':0,'TEST_new_access':0}
        write(HERE/'e2'/'SUMMARY_VERIFIED_run02.json',result)
        print(json.dumps({'status':result['status'],'records':len(records),'reused':sum(v['reused_complete_same_campaign'] for v in records),'new':sum(v['new_science_execution'] for v in records)}))


if __name__=='__main__':
    main()
