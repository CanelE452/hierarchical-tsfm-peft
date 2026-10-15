"""R1/R3/R4 and pre-C R5 refits; depends on completed, B-only R2."""
import argparse
import json
import time
import traceback
import numpy as np
import torch
import runtime as rt
import campaign as cp
import deployment as dp
from transforms import r1_metadata


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def core_specs():
    rows = read(rt.HERE/'R2_SELECTIONS.json')['results']
    return [dict(id=r['id'], type='ADAPTER', dataset=r['dataset'], fold=r['fold'], kind=r['kind'],
        r=r['r'], seed=r['seed'], updates=r['updates'], selected_update=r['selected_update'],
        selected_B=r['selected_B'], training_status=r['status'],checkpoint=f"selected/{r['id']}.pt") for r in rows]


def group_train(specs, win, bolt, initial=600, extend=True):
    group = []
    for spec in specs:
        try:
            model = dp.build({k:v for k,v in spec.items() if k != 'checkpoint'}, win, bolt)
            if spec.get('initial_checkpoint'):
                full=torch.load(rt.CACHE/spec['initial_checkpoint'],weights_only=True)
                subset={key:full[key] for key in model.state_dict()}
                model.load_state_dict(subset,strict=True)
            result, last, carry = cp.train(spec, model, bolt, win, initial,lr=spec.get('configured_lr',.001))
            group.append([spec, model, result, last, carry])
        except Exception as error:
            if 'model' in locals() and isinstance(model, dp.LoRA): model.close()
            rt.write(rt.HERE/'failures'/f"{spec['id']}.json", dict(**spec, status='BLOCKED' if 'out of memory' in str(error).lower() else 'INVALID',
                error=str(error), traceback=traceback.format_exc()))
            rt.event(stage='PATH_FAILURE', id=spec['id'], error=str(error))
    if extend:
        for endpoint in (1200, 2400):
            if endpoint <= initial: continue
            previous = endpoint//2
            if not any(v[2]['selected_update'] == previous or cp.still_decreasing(v[4]['checkpoints']) for v in group): break
            for item in group:
                spec, model, result, last, carry = item
                model.load_state_dict(last)
                result, last, carry = cp.train(spec, model, bolt, win, endpoint, carry,lr=spec.get('configured_lr',.001))
                item[2:] = [result, last, carry]
    results = []
    for spec, model, result, _, _ in group:
        rt.save(rt.CACHE/'selected'/f"{spec['id']}.pt", rt.clone(model))
        results.append(dict(**spec, checkpoint=f"selected/{spec['id']}.pt", updates=result['updates'],
            selected_update=result['selected_update'], selected_B=result['selected_B'], training_status=result['status']))
        if isinstance(model, dp.LoRA): model.close()
    del group; torch.cuda.empty_cache()
    return results


def r1(bolt):
    records, routes = [], []
    for dataset in ('beijing', 'jena', 'robin'):
        win = cp.dataset_windows(dataset)
        for index in range(14):
            variant = f'A{index}'; length = {12:1024, 13:2048}.get(index, 512)
            origins = win.origins['B'][win.origins['B'] >= length]
            spec = dict(id=f'primary_{dataset}_{variant}', type='RAW' if index in (0,12,13) else 'R1',
                variant=variant, dataset=dataset, fold='primary', length=length, seed=0, r=win.channels,
                updates=0, info='past_only', block='R1')
            if not len(origins):
                records.append(dict(**spec, status='BLOCKED', reason='NO_COMMON_ORIGIN_WITH_THIS_LOOKBACK'))
                continue
            try:
                model = dp.build(spec, win, bolt)
                pred = cp.infer(model, bolt, win, origins, length=length)
                y, mask = win.truth(origins); score = rt.score(pred, y, mask)
                meta = r1_metadata(variant, length, win.channels, period=win.data['period_from_existing_metadata'])
                records.append(dict(**spec, status='COMPLETE_VALID', selected_B=score,
                    B_origins=origins.tolist(), metadata=meta))
                spec['selected_B'] = score; routes.append(spec)
                if index in (12,13):
                    ref, _ = cp.assess(cp.Raw(), bolt, win, 'B', origins)
                    records[-1]['matched_Raw512_B'] = ref
                if index == 11:
                    effective = 512//win.data['period_from_existing_metadata']*win.data['period_from_existing_metadata']
                    ref = cp.infer(dp.CropRaw(effective), bolt, win, origins, length=512)
                    records[-1]['matched_effective_lookback_Raw_B'] = rt.score(ref, y, mask)
                rt.event(stage='R1_B', id=spec['id'], B_mse=score['mse'])
            except Exception as error:
                records.append(dict(**spec, status='INVALID', error=str(error), traceback=traceback.format_exc()))
                rt.event(stage='PATH_FAILURE', id=spec['id'], error=str(error))
        del win; torch.cuda.empty_cache()
    rt.write(rt.HERE/'R1_B.json', dict(records=records, C_scored=False))
    return routes


def r3(bolt, base_specs):
    results = []
    for dataset in ('beijing', 'jena'):
        win = cp.dataset_windows(dataset); k = win.data['small_k']
        # One fixed G seed leaves ten of the twenty G fits for full second-fold refits.
        for seed in rt.SEEDS[:1]:
            bases = [dict(id=f'primary_{dataset}_Raw', type='RAW', dataset=dataset, fold='primary'),
                     dict(id=f'primary_{dataset}_Current', type='CURRENT', dataset=dataset, fold='primary')]
            bases += [min([r for r in base_specs if r['dataset'] == dataset and r['kind'] == kind and r['r'] == k],
                          key=lambda r:r['selected_B']['mse'])
                      for kind in ('B0', 'B2', 'B4')]
            specs = [dict(id=f"{b['id']}_G_s{seed}", type='G', dataset=dataset, fold='primary',
                r=b.get('r', win.channels), seed=seed, block='R3', base=b) for b in bases]
            results += group_train(specs, win, bolt)
        del win; torch.cuda.empty_cache()
    rt.write(rt.HERE/'R3_SELECTIONS.json', dict(results=results, C_scored=False))
    return results


def ridge_fit(spec, win, bolt, base=None):
    import native_ridge as nr
    t = time.perf_counter(); origins = win.origins['A']; c = win.channels
    counts = win.counts.cpu().numpy()
    if base:
        model = dp.build(base, win, bolt).requires_grad_(False)
        gram = np.zeros((144,144), np.float64); rhs = np.zeros(144, np.float64)
    else:
        gram = np.zeros((c,48,33,33), np.float64); rhs = np.zeros((c,48,33), np.float64)
    for start in range(0,len(origins),32):
        with torch.no_grad():
            x,y,mask = win.get(origins[start:start+32])
            if base:
                z = model.encode(x); offset, design = nr.swap_design(z, model.D)
                nr.sufficient_update(gram, rhs, design, y-x[:, -1:]-offset, mask, counts)
            else: nr.direct_sufficient_update(gram, rhs, nr.direct_features(x), y-x[:, -1:], mask, counts)
    active_channels=int(np.sum(counts>0))
    gram/=active_channels;rhs/=active_channels
    candidates = []
    for alpha in ([0.,.01] if base else [.01]):
        if base: weights, metadata = nr.fit_from_sufficient(gram,rhs,alpha); weights=weights.reshape(3,48)
        else: weights,metadata = nr.fit_direct_from_sufficient(gram,rhs,alpha)
        weights = weights.astype(np.float32)
        model = dp.Swap(dp.build(base,win,bolt),weights) if base else dp.Ridge(weights)
        model = model.cuda().eval()
        b,_ = cp.assess(model,bolt,win,'B')
        candidates.append((b['mse'],weights,alpha,b,metadata))
        rt.event(stage='R4_RIDGE_CANDIDATE', id=spec['id'], alpha=alpha,B_mse=b['mse'])
    # Literal zero forecast-correction is available without an extra fit.
    weights0 = np.zeros_like(candidates[0][1])
    zero_model = (dp.Swap(dp.build(base,win,bolt),weights0) if base else dp.Ridge(weights0)).cuda()
    zero_b,_ = cp.assess(zero_model,bolt,win,'B')
    candidates.append((zero_b['mse'],weights0,None,zero_b,dict(type='ZERO_NO_FIT')))
    selected = min(candidates,key=lambda v:v[0])
    numeric=selected[4]
    unresolved=(numeric.get('numeric_status')=='NUMERIC_UNRESOLVED' or numeric.get('numeric_unresolved_equations',0)>0)
    relative = f"ridge/{spec['id']}.npy"; path=rt.CACHE/relative; path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f: np.save(f,selected[1],allow_pickle=False)
    result = dict(**spec, weights=relative, selected_alpha=selected[2], selected_B=selected[3],
        fit_candidates=len(candidates)-1, no_correction_candidate=True, fit_A_origins=len(origins),
        objective='Observed normalized channel-macro MSE + alpha * sum(weight**2)',
        candidate_diagnostics=[dict(alpha=v[2],B=v[3],numeric=v[4]) for v in candidates],
        fit_wall_seconds=time.perf_counter()-t, status='COMPLETE_VALID',
        training_status='UNDERTRAINED_OR_UNRESOLVED' if unresolved else 'COMPLETE_VALID',C_scored=False)
    rt.write(rt.HERE/'fits'/f"{spec['id']}_ridge.json",result)
    return result


def r4(bolt, core):
    results = []
    for dataset in ('beijing','jena'):
        win = cp.dataset_windows(dataset); k=win.data['small_k']
        selected = [min([r for r in core if r['dataset']==dataset and r['kind']==kind and r['r']==k],key=lambda v:v['selected_B']['mse'])
                    for kind in ('B2','B4')]
        for base in selected:
            spec=dict(id=base['id']+'_native',type='SWAP',dataset=dataset,fold='primary',r=k,seed=0,block='R4',base=base)
            results.append(ridge_fit(spec,win,bolt,base))
        specs=[dict(id=f'primary_{dataset}_Direct_{kind}',type='DIRECT',kind=kind,dataset=dataset,
                    fold='primary',r=win.channels,seed=rt.SEEDS[0],block='R4') for kind in ('B2','B4')]
        results+=group_train(specs,win,bolt)
        spec=dict(id=f'primary_{dataset}_Raw_Ridge',type='RIDGE',dataset=dataset,fold='primary',r=win.channels,seed=0,block='R4')
        results.append(ridge_fit(spec,win,bolt))
        # LoRA is attached only during its isolated fit and removed before any other route.
        torch.manual_seed(rt.SEEDS[0])
        spec=dict(id=f'primary_{dataset}_LoRA_qv8',type='LORA',dataset=dataset,fold='primary',r=win.channels,seed=rt.SEEDS[0],
            block='R4',configured_lr=.0001)
        raw_before,_=cp.assess(cp.Raw(),bolt,win,'B')
        hooks_before=sum(len(m._forward_hooks) for m in bolt.model.modules())
        results+=group_train([spec],win,bolt)
        hooks_after=sum(len(m._forward_hooks) for m in bolt.model.modules())
        raw_after,_=cp.assess(cp.Raw(),bolt,win,'B')
        assert hooks_before==hooks_after and abs(raw_before['mse']-raw_after['mse'])<1e-10
        rt.write(rt.HERE/f'LORA_CLEANUP_{dataset}.json',dict(status='PASS',hooks_before=hooks_before,hooks_after=hooks_after,
            raw_before=raw_before,raw_after=raw_after,F_base_weights_unchanged=rt.state_hash(bolt.model)==bolt.initial_hash))
        del win; torch.cuda.empty_cache()
    rt.write(rt.HERE/'R4_SELECTIONS.json',dict(results=results,C_scored=False,
        planned_fitting_configurations=16,
        actual_fitting_configurations=sum(r.get('fit_candidates',1) for r in results),
        direct_seeds=1,swap_objective='actual original-C Y',native_features='mean8/64/512 minus last, shared 144 coefficients'))
    return results


def r5_refits(bolt, core):
    results=[]; reuses=[]
    for dataset in ('beijing','jena'):
        primary=cp.dataset_windows(dataset); k=primary.data['small_k']
        chosen={kind:min([r for r in core if r['dataset']==dataset and r['kind']==kind and r['r']==k],key=lambda v:v['selected_B']['mse'])
                for kind in ('B0','B2','B4')}
        b4=chosen['B4']
        # Decomposition-off is the already matched and identically parameterized B3 refit.
        reuse=next(r for r in core if r['dataset']==dataset and r['kind']=='B3' and r['r']==k and r['seed']==b4['seed'])
        reuses.append(dict(question='B4_decomposition_removed', status='REUSED_VERIFIED', source=reuse['id'], comparator=b4['id'],
            basis='Same initialization/schedule/model class, B3 equals decomposition=False; CP budget may differ only at matched group endpoints'))
        ablations=[]
        for flag in ('seasonal_mix','trend_mix'):
            ablations.append(dict(id=f"{b4['id']}_without_{flag}",type='ADAPTER',dataset=dataset,fold='primary',
                kind='B4',r=k,seed=b4['seed'],block='R5_ABLATION',front_kwargs={flag:False},
                initial_checkpoint=f"checkpoints/{b4['id']}/cp0.pt"))
        results+=group_train(ablations,primary,bolt,initial=b4['updates'],extend=False)
        # One intermediate rank only if C helps on B but K loses relative to Raw.
        raw_b,_=cp.assess(cp.Raw(),bolt,primary,'B')
        for kind in ('B2','B4'):
            cr=min([r for r in core if r['dataset']==dataset and r['kind']==kind and r['r']==primary.channels],key=lambda v:v['selected_B']['mse'])
            if cr['selected_B']['mse']<raw_b['mse']<chosen[kind]['selected_B']['mse']:
                rank=min(2*k,primary.channels)
                specs=[dict(id=f'primary_{dataset}_{kind}_intermediate_r{rank}',type='ADAPTER',dataset=dataset,fold='primary',
                    kind=kind,r=rank,seed=chosen[kind]['seed'],block='R5_INTERMEDIATE')]
                results+=group_train(specs,primary,bolt)
        win=cp.dataset_windows(dataset,'secondary')
        specs=[dict(id=f'secondary_{dataset}_{kind}_r{k}',type='ADAPTER',dataset=dataset,fold='secondary',
            kind=kind,r=k,seed=rt.SEEDS[0],block='R5_TIME_REFIT') for kind in ('B0','B2','B4')]
        secondary=group_train(specs,win,bolt)
        results+=secondary
        bases=[dict(id=f'secondary_{dataset}_Raw',type='RAW',dataset=dataset,fold='secondary'),
               dict(id=f'secondary_{dataset}_Current',type='CURRENT',dataset=dataset,fold='secondary')]+secondary
        gspecs=[dict(id=b['id']+'_G',type='G',dataset=dataset,fold='secondary',r=b.get('r',win.channels),
            seed=rt.SEEDS[0],block='R5_TIME_REFIT',base=b) for b in bases]
        # Ten primary plus ten secondary G fits respect the campaign's twenty-fit maximum.
        results+=group_train(gspecs,win,bolt)
        del primary,win;torch.cuda.empty_cache()
    # Robin selected family is fixed using only primary B relative-to-Raw results.
    raw_mses={r['dataset']:r['selected_B']['mse'] for r in read(rt.HERE/'R1_B.json')['records'] if r.get('variant')=='A0' and r['status']=='COMPLETE_VALID'}
    family=min(('B2','B4'),key=lambda kind:np.mean([min(r['selected_B']['mse'] for r in core
        if r['dataset']==d and r['kind']==kind and r['r']==({'beijing':2,'jena':5}[d]))/raw_mses[d] for d in ('beijing','jena')]))
    win=cp.dataset_windows('robin'); k=win.data['small_k']
    specs=[dict(id=f'primary_robin_{kind}_r{k}',type='ADAPTER',dataset='robin',fold='primary',kind=kind,r=k,
        seed=rt.SEEDS[0],block='R5_STRESS') for kind in ('B0',family)]
    results+=group_train(specs,win,bolt)
    rt.write(rt.HERE/'R5_SELECTIONS.json',dict(results=results,reuses=reuses,Robin_family=family,C_scored=False))
    return results


def run():
    bolt=rt.Bolt(); core=core_specs()
    routes=r1(bolt)
    g=r3(bolt,core); native=r4(bolt,core); refits=r5_refits(bolt,core)
    routes+=core+g+native+refits
    for dataset in ('beijing','jena','robin'):
        for fold in ('primary','secondary'):
            if dataset=='robin' and fold=='secondary':continue
            for typ in ('RAW','CURRENT'):
                name=f'{fold}_{dataset}_{typ}'
                if any(r['id']==name for r in routes):continue
                routes.append(dict(id=name,type=typ,dataset=dataset,fold=fold,r={'beijing':11,'jena':21,'robin':17}[dataset],seed=0,updates=0))
    for dataset in ('beijing','jena','robin'):
        raw=next(r for r in routes if r['dataset']==dataset and r['fold']=='primary' and r['type']=='RAW' and r.get('length',512)==512)
        routes.append(dict(raw,id=raw['id']+'_micro4',row_microbatch=4,block='R5_RAW_MICROBATCH'))
    assert rt.state_hash(bolt.model)==bolt.initial_hash
    artifact_paths=set()
    def bind(spec):
        for key in ('checkpoint','weights','initial_checkpoint'):
            if spec.get(key):artifact_paths.add(spec[key])
        if spec.get('base'):bind(spec['base'])
    for spec in routes:bind(spec)
    scientific=('runtime.py','campaign.py','data.py','models.py','transforms.py','native_ridge.py',
                'deployment.py','followup.py','cost_and_score.py','cost_measure.py','patterns.py',
                'analysis_outputs.py','oracle_diagnostic.py')
    rt.write(rt.HERE/'SELECTION_LOCK.json',dict(status='ALL_SELECTIONS_LOCKED_BEFORE_C',time_unix=time.time(),
        C_scored=False,F_unchanged=True,F_model_hash=bolt.initial_hash,F_quantiles=list(bolt.model.chronos_config.quantiles),
        F_head_length=bolt.model.chronos_config.prediction_length,F_max_context=bolt.model.chronos_config.context_length,
        routes=routes,source_hashes={name:rt.sha(rt.HERE/name) for name in scientific},
        artifact_hashes={name:rt.sha(rt.CACHE/name) for name in sorted(artifact_paths)},
        statistics_hash=rt.sha(rt.HERE/'FOLD_STATISTICS.json'),followup_F_forward_calls=bolt.calls,
        followup_F_forecast_streams=bolt.streams,
        note='No C value or score used in model/checkpoint/alpha/branch selection; physical archive member reads disclosed'))


if __name__=='__main__':
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    rt.event(stage='PHASE_START',phase='followup')
    try:run();rt.event(stage='PHASE_COMPLETE',phase='followup')
    except Exception as error:
        rt.event(stage='PHASE_FAILURE',phase='followup',error=str(error),traceback=traceback.format_exc());raise
