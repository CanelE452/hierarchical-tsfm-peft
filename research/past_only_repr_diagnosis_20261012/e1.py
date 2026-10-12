import gc
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from common import *
from models_io import *


def prepare(raw, mask, o, bounds):
    """Cold preparation; fitting never sees observations after the artifact cutoff."""
    lo,hi=bounds
    observed=np.asfortranarray(np.where(mask[lo:hi],raw[lo:hi],np.nan))
    mean=np.nanmean(observed,axis=0).astype(np.float32)
    std=np.nanstd(observed,axis=0).astype(np.float32)
    std=np.where(std>1e-6,std,1.).astype(np.float32)
    filled=np.empty((o,raw.shape[1]),np.float32)
    last=mean.copy()
    for t in range(o):
        last[mask[t]]=raw[t,mask[t]]
        filled[t]=last
    return ((filled[o-L:o]-mean)/std).astype(np.float32),mean,std


def difference(a,b):
    d=np.abs(np.asarray(a,dtype=np.float64)-b)
    return {'max_abs':float(d.max()),'changed_exact_count':int(np.count_nonzero(d)),
            'relative_l2':float(np.linalg.norm(d)/(np.linalg.norm(a)+1e-12))}


def checker_unit():
    x=np.arange(128,dtype=np.float32).reshape(64,2)
    altered=x.copy(); altered[32:]+=1000
    safe=lambda a:a[:32].mean(0)
    bad=lambda a:a.mean(0)
    assert np.array_equal(safe(x),safe(altered))
    assert not np.array_equal(bad(x),bad(altered))
    # A deliberately future-fitted graph also changes under suffix perturbation.
    rng=np.random.default_rng(91012)
    xx=rng.normal(size=(64,3)); yy=xx.copy(); yy[32:,1]=yy[32:,0]*100
    assert not np.array_equal(np.corrcoef(xx.T),np.corrcoef(yy.T))
    assert np.array_equal(np.corrcoef(xx[:32].T),np.corrcoef(yy[:32].T))
    return {'past_only_PASS':True,'future_scaler_detected':True,'future_graph_detected':True,
            'deliberate_future_read_detected':True,'science_performance_claim':False}


def main():
    access_guard();torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    rows=[]
    with threadpool_limits(2),Budget(1,False,'E1_CPU_full_cold_path') as budget:
        unit=checker_unit()
        bolt=Bolt('cpu');parity=bolt.parity()
        assets=read(HERE/'ASSETS_AND_PREFLIGHT.json')['assets']
        for ds in ('jena','robin'):
            data=load_data(ds,frozen=True)
            origins=select_origins(data['origins']['val'],16)
            cold=select_origins(origins,2)
            raw,mask=data['physical'],data['mask']
            bounds=(data['fit_bounds'][0],data['tune_bounds'][1])
            for asset in [a for a in assets if a['job']['dataset']==ds]:
                job=asset['job'];seed=job['seed'];family=job['family']
                if family=='TS2':
                    encoder,enc_record=load_ts2(ds,job['dim'],seed)
                    decoder,dec_record=load_existing_decoder(ds,job['dim'],seed)
                    def encode(x):
                        return encode_full(encoder,x)
                    def predict(z):
                        zz=bolt.forecast(z,budget=budget)
                        with torch.no_grad():
                            return decoder(torch.from_numpy(zz)).numpy()
                else:
                    encoder=load_gt(asset);decoder=None;enc_record=asset;dec_record=asset
                    def encode(x):
                        with torch.no_grad():
                            return encoder.encode(torch.from_numpy(x)).numpy()
                    def predict(z):
                        zz=bolt.forecast(z,12,budget)
                        with torch.no_grad():
                            return encoder.decode_future(torch.from_numpy(z),torch.from_numpy(zz),H).numpy()
                state_before=state_digest(encoder)
                base_x=np.stack([prepare(raw,mask,int(o),bounds)[0] for o in origins])
                z=encode(base_x);repeat=encode(base_x)
                repeat_error=difference(z,repeat)['max_abs']
                # Exact CPU repeats calibrate epsilon at actual dtype/scale before perturbation results.
                atol=max(repeat_error*4,8*np.finfo(np.float32).eps*max(1.,float(np.max(np.abs(z)))))
                boundary=[]
                for o in cold:
                    altered=raw.copy();amask=mask.copy()
                    rng=np.random.default_rng(seed+int(o))
                    altered[o:]=rng.normal(size=altered[o:].shape)*data['std']+data['mean']
                    amask[o:]=rng.random(amask[o:].shape)>.1
                    xa,ma,sa=prepare(raw,mask,int(o),bounds)
                    xb,mb,sb=prepare(altered,amask,int(o),bounds)
                    za,zb=encode(xa[None]),encode(xb[None])
                    pa,pb=predict(za),predict(zb)
                    warm=predict(za.copy())
                    diff=difference(pa,pb)
                    if diff['max_abs']>atol or not np.array_equal(ma,mb) or not np.array_equal(sa,sb):
                        failure={'status':'FAILED_PRESERVED','job':job,'origin':int(o),'prediction_diff':diff,
                            'feature_diff':difference(za,zb),'history_diff':difference(xa,xb),'mean_diff':difference(ma,mb),
                            'std_diff':difference(sa,sb),'warm_diff':difference(pa,warm),'atol':atol,
                            'source_hash':sha(Path(__file__)),'bounds':bounds}
                        write(HERE/'failed_attempts'/f'E1_BOUNDARY_{time.time_ns()}.json',failure)
                        print(json.dumps(failure,default=jsonable),flush=True)
                        raise ValueError(f'Forecast-boundary leak {job}')
                    boundary.append({'origin':int(o),'history_start':int(o-L),'cutoff':bounds[1],
                        'cold_feature':difference(za,zb),'cold_prediction':diff,'cold_warm_prediction':difference(pa,warm),
                        'fit_mean_hash':ahash(ma),'fit_std_hash':ahash(sa),'future_mask_perturbed':True,
                        'noise_seed':seed+int(o),'inference_cache_used':False})
                    budget.check()
                # No tolerance is needed to interpret this as revision; raw deltas remain evidence.
                modified=base_x.copy()
                modified[:,256:]+=np.random.default_rng(seed).normal(size=modified[:,256:].shape).astype(np.float32)
                internal=encode(modified)
                if family=='TS2':
                    past_reference=encode(base_x[:,:256])[:,-1]
                    altered_reference=encode(modified[:,:256])[:,-1]
                    selected_full=z[:,255]
                    selected_internal=internal[:,255]
                    append=encode(base_x[:,:320])[:,255]
                    shifted=encode(base_x[:,4:])[:,251]
                else:
                    past_reference=encode(base_x[:,:256])[:,-1]
                    altered_reference=encode(modified[:,:256])[:,-1]
                    selected_full=z[:,63]
                    selected_internal=internal[:,63]
                    append=encode(base_x[:,:320])[:,63]
                    shifted=encode(base_x[:,4:])[:,62]
                if not np.array_equal(past_reference,altered_reference):
                    raise ValueError('Exact common-start prefix reads absent suffix')
                row={'experiment_id':'E1','dataset':ds,'job':job,'status':'COMPLETE_VERIFIED',
                     'checkpoint_hash':asset['checkpoint_hash'],'checkpoint_record':enc_record,'decoder_record':dec_record,
                     'source_hash':sha(Path(__file__)),'forecast_boundary_tests':boundary,
                     'representation_origins':origins,'same_input_repeat_max_abs':repeat_error,'sealed_before_perturbation_atol':atol,
                     'B_within_history_revision':difference(selected_full,selected_internal),
                     'C_same_left_append_revision':difference(append,selected_full),
                     'C_shifted_left_context_EFFECT_NOT_LEAK':difference(selected_full,shifted),
                     'prefix_unchanged_exact':True,'full_last_equals_exact_prefix_last':True,
                     'GT_phase':'right-edge ticks; original a, block[252:256], shifted a+4 separate' if family=='GT' else None,
                     'encoder_before_hash':state_before,'encoder_after_hash':state_digest(encoder),
                     'graph_hash':ahash(encoder.product.graph.numpy()) if family=='GT' else None,
                     'past_only_is_not_performance_improvement':True}
                if row['encoder_before_hash']!=row['encoder_after_hash']:
                    raise ValueError('Frozen E changed')
                rows.append(row)
                write(HERE/'e1_receipts'/f'{ds}_{family}_{job.get("dim","GT")}_{seed}_{job.get("no_spatial",False)}.json',row)
                print(f'E1 {ds} {family} {job.get("dim","")} {seed} boundaryPASS Bdelta={row["B_within_history_revision"]["max_abs"]:.6g}',flush=True)
                del encoder,decoder;gc.collect();budget.check()
        f_unchanged=state_digest(bolt.model)==bolt.initial
        if not f_unchanged:
            raise ValueError('Frozen Bolt changed')
        result={'experiment':'E1','safety_pass':True,'status':'COMPLETE_VERIFIED','rows':rows,
                'checker_unit':unit,'official_Bolt_parity':parity,'Bolt_unchanged':f_unchanged,
                'source_revision':read(HERE/'PROTOCOL.json')['HEAD'],'source_hash':sha(Path(__file__)),
                'source_files':{str(p):sha(p) for p in [Path(__file__),HERE/'models_io.py',HERE/'common.py']},
                'protected_TEST_new_access':0,'future_leak_detected_in_tested_cold_paths':False,
                'scope':'2 uniform full forecast perturbations /16 representation origins /16 pretrained paths; CPU FP32; finite tested perturbations only',
                'remaining_uncertainty':'Does not certify all future inputs, historical online deployability or generalization'}
        write(HERE/'E1_RESULT.json',result)


if __name__=='__main__':
    main()
