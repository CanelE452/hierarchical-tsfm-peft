"""Exclusive Jena physical-input CPU2 accuracy-parity and whole-path latency."""
import csv
import importlib
import os
import threading
import time

import numpy as np
import torch
import psutil

from .core import atomic_json, sha
from . import existing_jobs as existing


def measure_jena(job, ctx):
    modules = existing._modules(ctx)
    task = existing._jena(ctx, modules)
    data = modules['data']
    ridge = importlib.import_module('ridge_baseline')
    pool, f = existing._foundation(ctx, modules, 48)
    models, norms, arrays, bindings = {}, {}, {}, {}
    root = ctx['asset']/'.cache'/existing.BROAD
    methods = ('RAW','B0','B2','B2_CAP_POINTWISE','RAW_RIDGE')
    loaded = time.perf_counter()
    mean,std,_ = data.fit_statistics(task,12579)
    for family in ('B0','B2','B2_CAP_POINTWISE'):
        model,norm,bundle,path = modules['legacy'].make_selected(task,92601,family,f,modules)
        models[family],norms[family] = model.eval(),norm
        bindings[family] = dict(artifact_sha256=sha(path), trainable_params=sum(p.numel() for p in model.parameters()),
                                selected_epoch=bundle.get('selected_search_epoch',bundle.get('selected_epoch')))
        if family=='B2_CAP_POINTWISE':
            p=existing._run_root(ctx,92601,family)/'selected_V.npz'
        else:
            relative=existing._old_run(92601,family)
            p=root/relative.parent.parent/'predictions'/relative.name/'V_select.npz'
        arrays[family]=existing._archive(p)
        bindings[family]['prediction_sha256']=sha(p)
    rid='primary_jena_E1_RAW_RIDGE_k0_s92601'
    artifact_path=root/'ridge'/rid/'artifact.npz'
    artifact=ridge.load_artifact(artifact_path)
    norms['RAW_RIDGE']=dict(mean=artifact['mean'],std=artifact['std'])
    norms['RAW']=dict(mean=mean,std=std)
    bindings['RAW_RIDGE']=dict(artifact_sha256=sha(artifact_path),trainable_params=int(artifact['coefficients'].size))
    bindings['RAW']=dict(artifact_sha256=f.initial_hash,trainable_params=0)
    for family,run in (('RAW','primary_jena_E1_RAW_BOLT_SMALL_k0_s92601'),('RAW_RIDGE',rid)):
        p=root/'predictions'/run/'V_select.npz'
        arrays[family]=existing._archive(p)
        bindings[family]['prediction_sha256']=sha(p)
    for value in arrays.values():
        existing.validate_cohort(value, arrays['B2'])
    load_seconds=time.perf_counter()-loaded+getattr(f,'load_seconds',0.)
    rss_values=[]
    stop=threading.Event()
    def sample():
        process=psutil.Process()
        while not stop.wait(.005):
            rss_values.append(process.memory_info().rss)
    watcher=threading.Thread(target=sample,daemon=True)
    watcher.start()
    trials,rows,parity=[],[],[]
    origins=arrays['B2']['origins']
    def predict(family,physical,mask):
        norm=norms[family]
        mean,std=norm['mean'],norm['std']
        filled=physical.copy() if mask.all() else np.stack([data.causal_forward_fill(x,m,mean) for x,m in zip(physical,mask)])
        normalized=((filled-mean)/std).astype(np.float32)
        if family=='RAW_RIDGE':
            prediction=ridge.predict_normalized(artifact,normalized)
        else:
            x=torch.as_tensor(normalized,dtype=torch.float32,device='cpu')
            with torch.no_grad():
                prediction=(f(x) if family=='RAW' else models[family](x,f)).double().cpu().numpy()
        return prediction.astype(np.float64)*std.astype(np.float64)+mean.astype(np.float64)
    try:
        # No competing training/inference is dispatched by the campaign during this job.
        other=[]
        for p in psutil.process_iter(['pid','name','cmdline']):
            if p.pid==os.getpid() or p.pid in {x.pid for x in psutil.Process().parents()}:
                continue
            argv=' '.join(p.info.get('cmdline') or [])
            if 'python' in (p.info['name'] or '').lower() and ('adapter_diagnostics_20261017_v2' in argv or 'mechanism_runner.py' in argv):
                other.append(p.pid)
        if other:
            raise RuntimeError('Exclusive cost phase has another campaign worker')
        for batch in (1,8):
            selected=origins[:batch]
            physical=np.stack([task['physical'][o-512:o] for o in selected]).astype(np.float32)
            mask=np.stack([task['observed'][o-512:o] for o in selected])
            physical_hash=data.array_sha256(physical)
            valid=[]
            for family in methods:
                result=predict(family,physical,mask)
                norm=norms[family]
                normalized=(result-norm['mean'].astype(np.float64))/norm['std'].astype(np.float64)
                expected=arrays[family]['pred_normalized'][:batch]
                ok=np.allclose(normalized,expected,rtol=1e-4,atol=1e-5)
                parity.append(dict(method=family,B=batch,passed=bool(ok),atol=1e-5,rtol=1e-4,
                                   max_absolute_difference=float(np.max(np.abs(normalized-expected)))))
                if ok:
                    valid.append(family)
                else:
                    rows.append(dict(method=family,B=batch,status='FAILED_PARITY',median_ms_batch=None,
                                     median_ms_origin=None,physical_input_sha256=physical_hash))
            for repetition in range(30):
                if time.time()>=ctx.get('job_deadline',ctx['deadline']):
                    raise TimeoutError('Cost deadline')
                for family in valid:
                    tick=time.perf_counter()
                    output=predict(family,physical,mask)
                    elapsed=time.perf_counter()-tick
                    if output.shape!=(batch,48,21) or not np.isfinite(output).all():
                        raise ValueError('Invalid whole path output')
                    if repetition>=10:
                        trials.append(dict(method=family,B=batch,trial=repetition-10,
                                           ms_batch=elapsed*1000,ms_origin=elapsed*1000/batch))
            for family in valid:
                values=np.asarray([r['ms_batch'] for r in trials if r['B']==batch and r['method']==family])
                q1,q3=np.percentile(values,[25,75])
                rows.append(dict(method=family,B=batch,status='COMPLETE',median_ms_batch=float(np.median(values)),
                    median_ms_origin=float(np.median(values)/batch),IQR_ms_batch=float(q3-q1),IQR_ms_origin=float((q3-q1)/batch),
                    warmups=10,repetitions=20,device='cpu',dtype='float32 core; FP64 inverse and CPU return',threads=2,
                    physical_input_sha256=physical_hash,mask_sha256=data.array_sha256(mask),origins=selected.tolist(),
                    parity='PASS',trainable_params=bindings[family]['trainable_params'],
                    logical_F_rows=batch*(21 if family=='RAW' else 0 if family=='RAW_RIDGE' else 6),
                    backbone_resident_parameters=sum(p.numel() for p in f.model.parameters()),
                    cold_shared_load_seconds=load_seconds,
                    scope='physical input/mask -> within-history causal fill -> FP32 normalize -> P/E/F/D or Ridge -> FP64 inverse -> CPU return'))
    finally:
        stop.set();watcher.join();pool.close()
    rss_values.append(psutil.Process().memory_info().rss)
    for row in rows:
        row['process_RSS_peak_sampled_bytes']=max(rss_values)
        row['RSS_scope']='5ms sampling across shared loaded comparison process; not isolated per-method activation memory'
    output=ctx['report']/'costs'/'JENA_MATCHED_COST.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    trial_path=output.with_name('JENA_MATCHED_TRIALS.csv')
    with trial_path.open('w',encoding='utf-8',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=['method','B','trial','ms_batch','ms_origin'])
        writer.writeheader();writer.writerows(trials)
    value=dict(status='COMPLETE',rows=rows,parity=parity,bindings=bindings,
               cpu_heavy_workers_during_cost=1,other_campaign_workers_observed=[],
               inference_only=True,cold_load_scope='shared F plus all four routes, not per method')
    atomic_json(output,value)
    return dict(status='COMPLETE',reason='Exclusive physical-input whole-path CPU2 measurements; failed parity stays NA',
                artifact_paths=[str(output),str(trial_path)],result=value)
