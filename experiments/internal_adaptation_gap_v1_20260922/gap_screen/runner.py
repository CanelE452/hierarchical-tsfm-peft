from __future__ import annotations
import gc, importlib.metadata, json, os, platform, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from .io import *
from .data import read_panel, acquire
from .cache import Features, build_cache
from .model import *
from .metrics import score, choose_checkpoint, best_under_budget

class Context:
    def __init__(self,repo: Path,code: Path,config: dict,device='cuda',model_dir=None):
        self.repo=repo.resolve(); self.code=code.resolve(); self.cfg=config; self.device=device; self.model_dir=model_dir
        self.out=self.repo/'results'/config['experiment_id']; self.cache=self.repo/'.cache'/config['experiment_id']
        self.out.mkdir(parents=True,exist_ok=True); self.cache.mkdir(parents=True,exist_ok=True)
    def paths(self,panel): return self.out/panel,self.cache/panel
    def panel(self,panel): return read_panel(self.paths(panel)[1]/'panel.npz')
    def assert_seal(self):
        seal=read_json(self.out/'SOURCE_SEAL.json')
        verify_code(self.code,seal)
        if seal['config_hash']!=digest_object(self.cfg):
            raise Blocked('CONFIG_CHANGED_AFTER_SEAL')


def environment(ctx):
    import psutil
    configure_runtime(ctx.device,ctx.cfg['precision'])
    if psutil.virtual_memory().available < ctx.cfg['min_free_ram_gib']*2**30: raise Blocked('INSUFFICIENT_FREE_RAM')
    if ctx.device.startswith('cuda'):
        smi=run_command(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],check=False)
        if smi.returncode==0:
            others=[r for r in smi.stdout.splitlines() if r.strip() and r.split(',')[0].strip()!=str(os.getpid())]
            # COMPAT r1 (Windows WDDM): compute-apps also lists desktop graphics clients (Type C+G, e.g. dwm.exe).
            # There, only pure compute processes (Type C in the nvidia-smi process table) count as another GPU job.
            if os.name=='nt':
                import re
                table=run_command(['nvidia-smi'],check=False).stdout
                compute={m.group(1) for m in re.finditer(r'^\|\s+\d+\s+\S+\s+\S+\s+(\d+)\s+C\s',table,re.M)}
                others=[r for r in others if r.split(',')[0].strip() in compute]
            if others: raise Blocked('GPU_BUSY: another compute process exists; do not terminate it or start a long wait')
    packages={}
    for name in ['torch','chronos-forecasting','transformers','peft','numpy','pandas','scipy','matplotlib','huggingface-hub']:
        try: packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: packages[name]=None
    if any(packages[k] is None for k in ('chronos-forecasting','transformers','peft')):
        raise Blocked('MISSING_MODEL_PACKAGES: install requirements in an isolated environment')
    githead=run_command(['git','rev-parse','HEAD'],cwd=ctx.repo,check=False).stdout.strip()
    e=dict(python=sys.version,platform=platform.platform(),packages=packages,torch_cuda=torch.version.cuda,
           device=ctx.device,gpu=torch.cuda.get_device_name() if ctx.device.startswith('cuda') else None,
           existing_head=githead,free_ram_bytes=psutil.virtual_memory().available,config=ctx.cfg,
           native_loss='official normalized native pinball; summed quantiles',
           evaluation='sorted raw predictions / TRAIN std; equal channels',
           authenticity='REAL_MODEL_RUN',created_utc=time.time())
    write_json(ctx.out/'ENVIRONMENT.json',e)
    freeze=run_command([sys.executable,'-m','pip','freeze'],check=False)
    (ctx.out/'requirements-lock.txt').write_text(freeze.stdout,encoding='utf-8')
    write_json(ctx.out/'SOURCE_SEAL.json',dict(code=code_manifest(ctx.code),config_hash=digest_object(ctx.cfg),created_utc=time.time()))


def tensor_batch(panel,origins,device,context=None):
    x,y,g=panel.batch(origins,context)
    return [torch.as_tensor(z,device=device) for z in (x,y,g)]


def predictions(model,panel,role,config,device,features=None,context=None):
    out=[]; horizon=panel.horizon
    with torch.no_grad():
        for o in panel.origins[role]:
            with autocast(device,config['precision']):
                if features is not None:
                    z=model(*features.batch([o],device),horizon)[1]
                else:
                    x,_,g=tensor_batch(panel,[o],device,context); z=model(x,g,horizon)[1]
            out.append(z.reshape(len(panel.columns),-1,horizon).float().cpu().numpy())
    return np.stack(out)


def evaluate_val(model,panel,config,device,path,features=None,levels=None):
    sync(device); start=time.perf_counter(); q=predictions(model,panel,'val',config,device,features)
    sync(device)
    y=panel.targets('val'); metrics,per=score(q,y,panel.sigma,levels)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,q=q,origins=panel.origins['val'])
    elapsed=time.perf_counter()-start
    return metrics,elapsed


def preflight_panel(ctx,name):
    cfg=ctx.cfg; result,cache=ctx.paths(name); result.mkdir(parents=True,exist_ok=True)
    panel=ctx.panel(name); ctx.assert_seal()
    if (result/'PREFLIGHT.json').exists():
        old=read_json(result/'PREFLIGHT.json')
        if old['status']!='PASS': raise Blocked('PREFLIGHT_PREVIOUS_FAILURE')
        return
    base,_=load_base(cfg,ctx.device,ctx.model_dir); model_fp=fingerprint(base)
    if panel.horizon%base.chronos_config.output_patch_size: raise Blocked('HORIZON_NOT_PATCH_ALIGNED')
    if panel.context>base.chronos_config.context_length: raise Blocked('CONTEXT_EXCEEDS_NATIVE: no silent truncation')
    write_json(result/'MODEL_RECEIPT.json',dict(backbone=cfg['backbone'],revision=cfg['model_revision'],
        actual_fingerprint=model_fp,model_config=base.config.to_dict(),levels=base.quantiles.cpu().tolist(),
        local_override=bool(ctx.model_dir),local_override_weights_verified_by_runtime_fingerprint=True))
    features=build_cache(base,panel,cfg,cache/'features_fit',['train','val'],ctx.device,model_fp)
    write_json(result/'FEATURE_CACHE_RECEIPT.json',features.receipt)
    x,y,g=tensor_batch(panel,panel.origins['train'][:1],ctx.device)
    f0=Predictor(base,'F0',cfg,cfg['seeds'][0]); levels=base.quantiles
    with torch.no_grad(),autocast(ctx.device,cfg['precision']):
        norm,raw,loc,scale=f0(x,g,panel.horizon)
        native=base(context=x,group_ids=g,num_output_patches=panel.horizon//base.chronos_config.output_patch_size,future_target=y)
        native_err=float((native.quantile_preds[...,:panel.horizon]-raw).abs().max())
        torch.testing.assert_close(raw,native.quantile_preds[...,:panel.horizon],rtol=cfg['prediction_rtol'],atol=cfg['prediction_atol'])
        ours=normalized_loss(norm,y,loc,scale,levels,base.chronos_config.use_arcsinh)
        torch.testing.assert_close(ours,native.loss,rtol=1e-5,atol=1e-5)
        # Changing future targets cannot change encoder features/predictions.
        other=base(context=x,group_ids=g,num_output_patches=panel.horizon//base.chronos_config.output_patch_size,future_target=y+987)
        torch.testing.assert_close(native.quantile_preds,other.quantile_preds,rtol=0,atol=0)
        # Two independent origins cannot exchange context.
        xx,_,gg=tensor_batch(panel,panel.origins['train'][:2],ctx.device)
        aa=f0(xx,gg,panel.horizon)[1]; xx2=xx.clone(); xx2[len(panel.columns):]+=1000
        bb=f0(xx2,gg,panel.horizon)[1]
        torch.testing.assert_close(aa[:len(panel.columns)],bb[:len(panel.columns)],rtol=cfg['prediction_rtol'],atol=cfg['prediction_atol'])
    expected_raw=raw.detach().clone(); del f0,base; gc.collect(); torch.cuda.empty_cache()
    checks=[]; ledger=Ledger(ctx.out/'SMOKE_LEDGER.jsonl',cfg['max_smoke_updates'])
    for arm in cfg['arms']:
        base,_=load_base(cfg,ctx.device,ctx.model_dir); seed=cfg['seeds'][0]
        if arm=='HEAD':
            model=CachedHead(base,cfg['head_kind'][name],cfg['head_mlp_width'],seed).to(ctx.device); del base
        else:
            model=Predictor(base,arm,cfg,seed)
            if len(model.target_names)!=cfg['model_shape_contract']['lora_modules']: raise Blocked('LORA_TARGET_COUNT_CHANGED')
            if sum(p.numel() for p in model.parameters() if p.requires_grad)!=cfg['model_shape_contract']['lora_parameters']: raise Blocked('LORA_PARAMETER_COUNT_CHANGED')
        levels=torch.tensor(read_json(result/'MODEL_RECEIPT.json')['levels'],device=ctx.device)
        initial=trainable_state(model); frozen=fingerprint(model,True)
        def forward():
            return model(*features.batch(panel.origins['train'][:1],ctx.device),panel.horizon) if arm=='HEAD' else model(x,g,panel.horizon)
        with torch.no_grad(),autocast(ctx.device,cfg['precision']): initial_pred=forward()[1]
        torch.testing.assert_close(initial_pred,expected_raw,rtol=cfg['prediction_rtol'],atol=cfg['prediction_atol'])
        official=check_loraplus_reference(model,cfg) if arm=='LORAPLUS' else None
        opt=build_optimizer(model,arm,cfg); grads=[]
        for step in (1,2):
            opt.zero_grad(set_to_none=True)
            with autocast(ctx.device,cfg['precision']):
                q,_,lo,sc=forward(); loss=normalized_loss(q,y,lo,sc,levels,model.arcsinh)
            loss.backward()
            named=[(n,p) for n,p in model.named_parameters() if p.requires_grad]
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for _,p in named)
            gn=float(torch.nn.utils.clip_grad_norm_([p for _,p in named],cfg['gradient_clip']))
            assert np.isfinite(gn) and gn>0; grads.append(gn)
            key=f'{name}_{arm}'; ledger.intent(key,step); opt.step(); sync(ctx.device); ledger.commit(key,step)
        assert frozen==fingerprint(model,True)
        after=trainable_state(model); assert any(not torch.equal(initial[n],v) for n,v in after.items())
        with torch.no_grad(),autocast(ctx.device,cfg['precision']): after_pred=forward()[1].detach().clone()
        restore(model,initial)
        with torch.no_grad(),autocast(ctx.device,cfg['precision']): torch.testing.assert_close(forward()[1],expected_raw,rtol=cfg['prediction_rtol'],atol=cfg['prediction_atol'])
        restore(model,after)
        with torch.no_grad(),autocast(ctx.device,cfg['precision']): torch.testing.assert_close(forward()[1],after_pred,rtol=0,atol=0)
        checks.append(dict(arm=arm,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            trainable_names=[n for n,p in model.named_parameters() if p.requires_grad],native_parity=True,
            frozen_unchanged=True,restore=True,gradient_norms=grads,official_loraplus=official))
        del model,opt; gc.collect(); torch.cuda.empty_cache()
    write_json(result/'PREFLIGHT.json',dict(status='PASS',checks=checks,native_max_abs=native_err,
        official_loss_parity=True,group_isolation=True,target_poison_invariant=True,main_updates=0,smoke_updates=6))


def fit(ctx,name,arm,seed):
    ctx.assert_seal(); cfg=ctx.cfg; result,cache=ctx.paths(name); panel=ctx.panel(name)
    key=f'{name}_{arm}_{seed}'; out=result/'fits'/f'{key}.json'; work=cache/'fits'/key
    if out.exists():
        if read_json(out).get('complete'): return
        raise Blocked('INCOMPLETE_FIT_RESULT')
    if work.exists(): raise Blocked(f'PARTIAL_FIT {key}: automatic replay is prohibited')
    work.mkdir(parents=True,exist_ok=False); start_all=time.perf_counter()
    base,load_s=load_base(cfg,ctx.device,ctx.model_dir)
    levels=base.quantiles.detach().clone(); features=None; cache_s=0.
    if arm=='HEAD':
        features=Features(cache/'features_fit'); cache_s=features.receipt['cache_build_s']
        model=CachedHead(base,cfg['head_kind'][name],cfg['head_mlp_width'],seed).to(ctx.device); del base; gc.collect(); torch.cuda.empty_cache()
    else: model=Predictor(base,arm,cfg,seed)
    opt=build_optimizer(model,arm,cfg); initial=trainable_state(model); frozen=fingerprint(model,True)
    initial_hash=digest_object({n:hashlib.sha256(v.numpy().tobytes()).hexdigest() for n,v in initial.items()})
    schedule=panel.schedule(seed,cfg['max_steps'],cfg['accumulate_origins']); np.save(work/'schedule.npy',schedule)
    ledger=Ledger(ctx.out/'UPDATE_LEDGER.jsonl',cfg['max_main_updates'])
    curves=[]; trace=[]; active=0.; validation=0.; checkpoint_s=0.; peak=0
    prep=load_s+cache_s+panel.metadata.get('data_prepare_s',0.)
    def checkpoint(step):
        nonlocal validation,checkpoint_s,peak
        if ctx.device.startswith('cuda'): peak=max(peak,int(torch.cuda.max_memory_allocated()))
        sync(ctx.device); tick=time.perf_counter()
        state_path=work/f'step_{step}.pt'
        torch.save(dict(trainable=trainable_state(model),optimizer=opt.state_dict(),step=step),state_path)
        checkpoint_s+=time.perf_counter()-tick
        metrics,seconds=evaluate_val(model,panel,cfg,ctx.device,work/f'val_{step}.npz',features,levels.cpu().numpy())
        validation+=seconds
        curves.append(dict(step=step,val_primary=metrics['primary'],val_nmae=metrics['nmae'],
            active_train_s=active,validation_s=validation,checkpoint_io_s=checkpoint_s,
            cold_ready_s=prep+active+validation+checkpoint_s,
            state_sha256=sha(state_path),state_path=state_path.relative_to(ctx.repo).as_posix(),
            val_prediction= (work/f'val_{step}.npz').relative_to(ctx.repo).as_posix()))
        write_json(work/'progress.json',dict(complete=False,key=key,curves=curves))
        if ctx.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
        print(f'{key} step={step} VAL={metrics["primary"]:.6g} training_s={active:.2f}',flush=True)
    checkpoint(0)
    for step,origins in enumerate(schedule,1):
        sync(ctx.device); t=time.perf_counter(); opt.zero_grad(set_to_none=True); total=0.
        for origin in origins:
            x,y,g=tensor_batch(panel,[origin],ctx.device)
            with autocast(ctx.device,cfg['precision']):
                q,_,lo,sc=model(*features.batch([origin],ctx.device),panel.horizon) if features else model(x,g,panel.horizon)
                loss=normalized_loss(q,y,lo,sc,levels,model.arcsinh)/len(origins)
            if not torch.isfinite(loss): raise FloatingPointError('NONFINITE_LOSS')
            loss.backward(); total+=float(loss.detach())
        params=[p for p in model.parameters() if p.requires_grad]
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params): raise FloatingPointError('BAD_GRADIENT')
        gn=float(torch.nn.utils.clip_grad_norm_(params,cfg['gradient_clip']))
        sync(ctx.device); before=time.perf_counter(); forwardback=before-t
        # Journal sync costs are deliberately excluded from compute but included in full wall time.
        ledger.intent(key,step); sync(ctx.device); tick=time.perf_counter(); opt.step(); sync(ctx.device)
        dt=forwardback+time.perf_counter()-tick; active+=dt; ledger.commit(key,step)
        trace.append(dict(step=step,loss=total,gradient_norm=gn,active_step_s=dt))
        if step in cfg['checkpoints']: checkpoint(step)
    assert frozen==fingerprint(model,True)
    final=trainable_state(model); changed=sum(not torch.equal(initial[n],v) for n,v in final.items())
    if changed==0: raise Blocked('TRAINABLES_DID_NOT_CHANGE')
    selected=choose_checkpoint(curves)
    info=dict(complete=True,key=key,panel=name,arm=arm,seed=seed,selected_step=selected['step'],
        curves=curves,trace=trace,initial_hash=initial_hash,frozen_unchanged=True,changed_trainable_tensors=changed,
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        head_kind=cfg['head_kind'][name] if arm=='HEAD' else None,lr_groups=[float(g['lr']) for g in opt.param_groups],
        train_schedule_sha256=sha(work/'schedule.npy'),cache_build_s_charged=cache_s,model_load_s=load_s,
        data_prepare_s_charged=panel.metadata.get('data_prepare_s',0.),
        full_job_wall_s=time.perf_counter()-start_all,training_peak_allocated_bytes=peak,
        selected_cold_ready_s=selected['cold_ready_s'],active_training_s=active,
        times_definition='cold-ready includes data preparation, model load, fit/VAL cache, updates, visited VAL calls and checkpoint I/O; excludes journal fsync; full job wall also disclosed',
        checkpoint_dir=work.relative_to(ctx.repo).as_posix())
    write_json(out,info)


def seal_selection(ctx,name):
    ctx.assert_seal(); cfg=ctx.cfg; result,cache=ctx.paths(name)
    path=result/'SELECTION_SEAL.json'
    if path.exists(): return
    fits=[read_json(result/'fits'/f'{name}_{arm}_{seed}.json') for seed in cfg['seeds'] for arm in cfg['arms']]
    common_budget=float(np.median([f['curves'][-1]['cold_ready_s'] for f in fits if f['arm']=='LORA']))
    choices={}
    for f in fits:
        budgets={}
        for fraction in (.5,1.):
            choice=best_under_budget(f['curves'],common_budget*fraction)
            budgets[str(fraction)]=None if choice is None else choice['step']
        steps=sorted({f['selected_step'],cfg['max_steps'],*[s for s in budgets.values() if s is not None]})
        choices[f['key']]=dict(selected_step=f['selected_step'],budget_steps=budgets,test_steps=steps)
    # Comparator identity is selected only from mean VAL, not retrospectively from TEST.
    meanvals={a:float(np.mean([choose_checkpoint(f['curves'])['val_primary'] for f in fits if f['arm']==a])) for a in cfg['arms']}
    internal=min(['LORA','LORAPLUS'],key=lambda a:meanvals[a])
    # Include cheap no-adaptation alternatives before opening TEST scores.
    panel=ctx.panel(name); base,loading=load_base(cfg,ctx.device,ctx.model_dir)
    f0=Predictor(base,'F0',cfg,cfg['seeds'][0]); levels=base.quantiles.cpu().numpy()
    sync(ctx.device); tick=time.perf_counter()
    qp=predictions(f0,panel,'val',cfg,ctx.device); sync(ctx.device)
    f0score=score(qp,panel.targets('val'),panel.sigma,levels)[0]['primary']
    np.savez_compressed(cache/'f0_val.npz',q=qp,origins=panel.origins['val'])
    f0seconds=time.perf_counter()-tick
    tick=time.perf_counter()
    sn=np.stack([panel.values[o-panel.horizon:o].T for o in panel.origins['val']])[:,:,None,:]
    sn=np.repeat(sn,len(levels),axis=2)
    naive_score=score(sn,panel.targets('val'),panel.sigma,levels)[0]['primary']
    naive_seconds=time.perf_counter()-tick
    cheap_values={'HEAD':meanvals['HEAD'],'F0_LONG':f0score,'SEASONAL_NAIVE':naive_score}
    cheap=min(cheap_values,key=cheap_values.get)
    write_json(result/'F0_REFERENCE.json',dict(val_primary=f0score,naive_val_primary=naive_score,
        f0_cold_ready_s=loading+panel.metadata.get('data_prepare_s',0.)+f0seconds,
        naive_cold_ready_s=panel.metadata.get('data_prepare_s',0.)+naive_seconds,
        full_reference_evaluation_s=f0seconds,model_load_s=loading))
    del f0,base;gc.collect();torch.cuda.empty_cache()
    selection=dict(choices=choices,common_cold_budget_s=common_budget,mean_selected_val=meanvals,
        internal_reference=internal,cheap_reference=cheap,cheap_validation=cheap_values,fit_hashes={f'{f["key"]}.json':sha(result/'fits'/f'{f["key"]}.json') for f in fits},
        created_utc=time.time(),test_scoring_started=False,config_hash=digest_object(cfg))
    write_json(path,selection)


def predict_test(ctx,name):
    ctx.assert_seal(); cfg=ctx.cfg; result,cache=ctx.paths(name); panel=ctx.panel(name)
    seal=read_json(result/'SELECTION_SEAL.json'); sealhash=sha(result/'SELECTION_SEAL.json')
    for fn,h in seal['fit_hashes'].items():
        if sha(result/'fits'/fn)!=h: raise Blocked('FIT_CHANGED_AFTER_SELECTION')
    pred_dir=cache/'test_predictions'; pred_dir.mkdir(parents=True,exist_ok=True)
    base,_=load_base(cfg,ctx.device,ctx.model_dir); levels=base.quantiles.cpu().numpy()
    receipt=read_json(result/'MODEL_RECEIPT.json')
    if fingerprint(base)!=receipt['actual_fingerprint']: raise Blocked('BACKBONE_CHANGED')
    features=build_cache(base,panel,cfg,cache/'features_test',['test'],ctx.device,receipt['actual_fingerprint'])
    f0=Predictor(base,'F0',cfg,cfg['seeds'][0])
    for label,context in [('F0_LONG',panel.context),('F0_SHORT',panel.short_context)]:
        p=pred_dir/(label+'.npz')
        if not p.exists(): np.savez_compressed(p,q=predictions(f0,panel,'test',cfg,ctx.device,context=context),origins=panel.origins['test'])
    # Seasonal naive: deterministic empirical distribution, not a calibrated probabilistic baseline.
    q=np.stack([panel.values[o-panel.horizon:o].T for o in panel.origins['test']])[:,:,None,:]
    q=np.repeat(q,len(levels),axis=2); np.savez_compressed(pred_dir/'SEASONAL_NAIVE.npz',q=q,origins=panel.origins['test'])
    del f0,base; gc.collect(); torch.cuda.empty_cache()
    for seed in cfg['seeds']:
        for arm in cfg['arms']:
            key=f'{name}_{arm}_{seed}'; f=read_json(result/'fits'/f'{key}.json'); base,_=load_base(cfg,ctx.device,ctx.model_dir)
            if arm=='HEAD':
                model=CachedHead(base,cfg['head_kind'][name],cfg['head_mlp_width'],seed).to(ctx.device); del base
            else: model=Predictor(base,arm,cfg,seed)
            for step in seal['choices'][key]['test_steps']:
                dest=pred_dir/f'{key}_{step}.npz'
                statepath=ctx.repo/f['checkpoint_dir']/f'step_{step}.pt'
                curve=next(v for v in f['curves'] if v['step']==step)
                if sha(statepath)!=curve['state_sha256']: raise Blocked('CHECKPOINT_CHANGED')
                restore(model,torch.load(statepath,map_location='cpu',weights_only=True)['trainable'])
                if not dest.exists():
                    pp=predictions(model,panel,'test',cfg,ctx.device,features=features if arm=='HEAD' else None)
                    np.savez_compressed(dest,q=pp,origins=panel.origins['test'])
            # Freshly restored prediction check on the first TEST origin, without reading its target.
            step=seal['choices'][key]['test_steps'][-1]; saved=np.load(pred_dir/f'{key}_{step}.npz')['q'][0]
            with torch.no_grad(),autocast(ctx.device,cfg['precision']):
                if arm=='HEAD': v=model(*features.batch(panel.origins['test'][:1],ctx.device),panel.horizon)[1]
                else:
                    xx,_,gg=tensor_batch(panel,panel.origins['test'][:1],ctx.device); v=model(xx,gg,panel.horizon)[1]
            np.testing.assert_allclose(v.float().cpu().numpy(),saved,rtol=cfg['prediction_rtol'],atol=cfg['prediction_atol'])
            del model; gc.collect(); torch.cuda.empty_cache()
    if sha(result/'SELECTION_SEAL.json')!=sealhash: raise Blocked('SELECTION_CHANGED_DURING_TEST')
    write_json(result/'PREDICTIONS_MANIFEST.json',dict(complete=True,selection_sha256=sealhash,
        predictions={p.relative_to(ctx.repo).as_posix():sha(p) for p in sorted(pred_dir.glob('*.npz'))},
        restore_first_origin_verified=True,all_saved_before_scoring=True,created_utc=time.time()))
