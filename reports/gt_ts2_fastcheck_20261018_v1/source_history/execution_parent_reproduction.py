"""Actual frozen-parent, full-window feature and forecast experiment stages."""
from __future__ import annotations

import gc
import inspect
import platform
import time
from pathlib import Path

import numpy as np
import torch

from . import common as c
from . import data as d
from . import parents as p
from .model import ForecastRoute, FrozenFoundation


SEEDS = (92601, 92602)
ARMS = ('GT_FT_L512', 'GT_FT_L1024', 'GT_TS_PRE_L1024', 'GT_TS_RAND_L1024')


def candidate_specs():
    rows = [dict(run_id=f'A_RAW_L{length}', phase='A', mode='raw', seed=0, L=length,
                 arm='RAW_F', updates=0) for length in (512, 1024, 2048)]
    for seed in SEEDS:
        rows += [dict(run_id=f'A_GT_L{length}_s{seed}', phase='A', mode='gt', seed=seed,
                      L=length, arm='GT_FROZEN', updates=0) for length in (512, 1024, 2048)]
        rows += [dict(run_id=f'A_{mode.upper()}_L1024_s{seed}', phase='A', mode=mode, seed=seed,
                      L=1024, arm='TS2_NATIVE_D' if mode == 'ts2' else 'OUTPUT_ENSEMBLE_CONTEXT_ONLY',
                      updates=0) for mode in ('ts2', 'ensemble')]
        for arm in ARMS:
            mode = 'pre' if '_PRE_' in arm else 'rand' if '_RAND_' in arm else 'gt'
            rows.append(dict(run_id=f'B_{arm}_s{seed}', phase='B', mode=mode, seed=seed,
                             L=512 if arm == 'GT_FT_L512' else 1024, arm=arm, updates=None))
    return rows


def parity(actual, expected, atol=1e-5, rtol=1e-4):
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('Parent parity shape or finite-value failure')
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    result = dict(status='PASS' if np.allclose(actual, expected, atol=atol, rtol=rtol) else 'FAIL',
                  max_abs=float(difference.max(initial=0)),
                  max_relative=float((difference / np.maximum(np.abs(expected), 1e-12)).max(initial=0)),
                  atol=atol, rtol=rtol, shape=list(actual.shape),
                  actual_sha256=c.array_sha(actual), reference_sha256=c.array_sha(expected),
                  relative_diagnostic_denominator_floor=1e-12)
    if result['status'] != 'PASS':
        raise ValueError('Registered parent/forward parity tolerance failed: ' + str(result))
    return result


def source_identity():
    return {path.relative_to(c.ROOT).as_posix(): c.file_sha(path) for path in sorted(c.CODE.glob('*.py'))}


def context():
    paths = c.read_json(c.CACHE / 'PRIVATE_PATHS.json')
    data = d.load_data(paths['asset_root'])
    public, private = p.build_bindings(paths['asset_root'])
    if (c.REPORT / 'PARENT_BINDINGS.json').exists():
        if c.json_bytes(public) != (c.REPORT / 'PARENT_BINDINGS.json').read_bytes():
            raise ValueError('Canonical parent binding changed after registration')
    return dict(paths=paths, data=data, public_parents=public, parents=private)


def prepare():
    ctx = context()
    data = ctx['data']
    c.seal_json(c.REPORT / 'PARENT_BINDINGS.json', ctx['public_parents'])
    contract = dict(schema='gt_ts2_fastcheck_contract_v1', campaign=c.CODE.name,
        start=c.read_json(c.REPORT / 'START_STATE.json'), data=d.public_contract(data),
        candidates=candidate_specs(), seeds=list(SEEDS), H=48,
        training=dict(optimizer='Adam', lr=1e-4, weight_decay=0., grad_clip_norm=1.,
            effective_batch=8, default_updates=300, bounded_updates=100, microbatch=1,
            objective='normalized observed channel-macro MSE',
            schedule='numpy.default_rng(seed), complete permutations including final batch of 5, repeat',
            final_selection='fixed registered final update, never VAL-selected',
            checkpoint_updates=[0,100,300], TRAIN_diagnostics='fixed 16 linearly spaced legal TRAIN origins',
            feature_stats='PRE/RAND separately, all legal TRAIN window R4 ticks only, FP64 population',
            feature_cache='frozen unstandardized TS2 pooled features only; no GT/F output cache',
            bridge='bias-free 320->42, zero init, residual on [B,L/4,21,2]',
            frozen=['F parameters','TS2 parameters','GT adjacency buffer'],
            trainable=['GT E120','GT native D120','GT shared product4','fusion W13440 if present']),
        budget_choice_rule=dict(before='first Phase B training; rule fixed before new Phase A VAL',
            smoke='discarded 20 TRAIN updates for each of four arms, seed92601',
            step_estimate='1.25 times maximum mean update wall across the four TRAIN smoke arms',
            training_overhead_seconds=180.,
            decision='300 if eight-arm total fits both 80% wall and 80% GPU remainder; otherwise100 if full matched set fits; else NOT_RUN_BUDGET'),
        evaluation=dict(origins_sha256=c.array_sha(data['val_origins']), count=len(data['val_origins']),
            parent_reproduction='legacy global fill L512, separate A0 identity',
            new_inputs='independent physical window, TRAIN-mean initialization and within-window forward fill',
            metric='FP64 per-channel SSE/SAE/count then equal-channel average',
            exposure='repeated development VAL; historical parent selection preserved; no independent test claim'),
        cost=dict(batches=[1,8], warmups=10, repetitions=20,
            fixed_group_indices={str(b): [0,(len(data['val_origins'])-b)//2,len(data['val_origins'])-b] for b in (1,8)},
            groups_cycle=True, full_online_TS2=True,
            scope='physical history+mask -> fill/scale -> encoder(s)/bridge/F/native D -> inverse -> CPU forecast',
            excluded=['window slicing','model load','training feature cache creation'],
            memory_layers=['PyTorch peak allocated','PyTorch peak reserved'],
            PID_GPU_memory='not a required hard-cap certificate; device total never substituted'),
        limits=dict(new_TEST_access=0, pretraining_overlap='UNKNOWN',
            historical_raw_physical_hash='unresolved mismatch; current unchanged loader and six model-coordinate hashes match',
            frozen_versus_adapted='different questions; output ensemble is context only; two seeds not confidence interval'))
    c.seal_json(c.REPORT / 'EXPERIMENT_CONTRACT.json', contract)
    c.seal_json(c.CACHE / 'PREPARE_SOURCES.json', source_identity())
    return dict(status='PASS', parents=4, VAL_origins=len(data['val_origins']), TRAIN_origins=len(data['train_origins']),
                new_input_fingerprints={str(length):c.array_sha(d.windows(data,data['val_origins'],length)[0]) for length in (512,1024,2048)})


def cuda_setup():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')


def foundation(ctx):
    result = FrozenFoundation(ctx['paths']['model_directory'], device='cuda', row_microbatch=128)
    expected = c.read_json(c.ROOT / 'research/professor_compressor_direct_20261011/FOUNDATION_PROVENANCE.json')
    if c.state_sha(result.model) != expected['model_state_sha256']:
        raise ValueError('Loaded F tensor state differs from original professor parent')
    if c.file_sha(Path(inspect.getfile(type(result.model)))) != expected['installed_source_sha256']:
        raise ValueError('Installed F source differs from canonical parent')
    return result


def _feature_record(seed, mode):
    return c.read_json(c.CACHE / 'FEATURE_INDEX.json')[str(seed)][mode.upper()]


def _features(record, split):
    with np.load(record['features_paths'][split], allow_pickle=False) as archive:
        return archive['features'].copy()


def build_route(spec, ctx, f, *, adapted=False, online=True):
    mode, seed = spec['mode'], str(spec['seed'])
    route = dict(spec=spec)
    if mode != 'raw':
        binding = ctx['parents'][seed]
        if mode in ('gt','pre','rand','ensemble'):
            gt = p.load_gt(binding['GT'], 'cuda')
            stats = _feature_record(seed, mode) if mode in ('pre','rand') else None
            route['gt_route'] = ForecastRoute(gt, f,
                feature_mean=stats['mean'] if stats else None, feature_std=stats['std'] if stats else None).eval()
            if stats:
                route['feature_record'] = stats
            if adapted:
                updates = c.read_json(c.REPORT / 'UPDATE_DECISION.json')['updates']
                checkpoint = c.CACHE / 'training' / spec['run_id'] / f'checkpoint_{updates:06d}.pt'
                receipt = c.read_json(c.REPORT / 'training' / (spec['run_id'] + '.json'))
                if c.file_sha(checkpoint) != receipt['checkpoint_sha256']:
                    raise ValueError('Adapted checkpoint bytes changed')
                saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
                route['gt_route'].load_adapter_state(saved['adapter_state'])
        if mode in ('ts2','ensemble') or (mode in ('pre','rand') and online):
            route['ts2'] = p.load_ts2(binding['TS2'], 'cuda', random=mode=='rand')
        if mode in ('ts2','ensemble'):
            route['decoder'] = p.load_ts2_decoder(binding['TS2'], 'cuda')
    return route


def route_forward(route, f, x, features=None):
    mode = route['spec']['mode']
    if mode == 'raw':
        return f.forecast(x,48)
    if mode in ('pre','rand'):
        if features is None:
            features = ForecastRoute.pool_features(p.export_tensor(route['ts2'],x))
        return route['gt_route'].forecast(x, features)
    if mode == 'gt':
        return route['gt_route'].forecast(x)
    representations = p.export_tensor(route['ts2'],x)
    ts_prediction = route['decoder'](f.forecast(representations,48))
    if mode == 'ts2':
        return ts_prediction
    return .5 * (ts_prediction + route['gt_route'].forecast(x))


def predict(route, f, x, features=None, batch=8):
    result=[]
    with torch.no_grad():
        for start in range(0,len(x),batch):
            xt=torch.as_tensor(x[start:start+batch],device='cuda')
            ft=torch.as_tensor(features[start:start+batch],device='cuda') if features is not None else None
            result.append(route_forward(route,f,xt,ft).cpu().numpy())
            if start % 64 == 0:
                c.progress('predict', run_id=route['spec']['run_id'], current=start, total=len(x))
    return np.concatenate(result)


def prediction_artifact(spec,ctx,x,prediction,targets,masks,**metadata):
    from .analysis import score
    path=c.CACHE/'predictions'/(spec['run_id']+'.npz')
    data=ctx['data']
    c.seal_npz(path,prediction=prediction,target=targets,mask=masks,origins=data['val_origins'],
               mean=data['mean'],std=data['std'])
    metrics=score(prediction,targets,masks)
    receipt=dict(status='COMPLETE', **spec, **metadata, MSE=metrics['mse'],MAE=metrics['mae'],
        prediction_reference='cache://predictions/'+path.name,artifact_sha256=c.file_sha(path),
        input_sha256=c.array_sha(x),prediction_sha256=c.array_sha(prediction),target_sha256=c.array_sha(targets),
        mask_sha256=c.array_sha(masks),origins_sha256=c.array_sha(data['val_origins']),
        mean_sha256=c.array_sha(data['mean']),std_sha256=c.array_sha(data['std']),
        data_contract_sha256=data['contract_hash'],parent_data_contract_sha256=data['legacy_contract_hash'],
        source_identity=source_identity(), origin_count=len(prediction),H=48,C=21,
        new_training=spec['phase']=='B')
    c.seal_json(c.REPORT/'receipts'/(spec['run_id']+'.json'),receipt)
    return receipt


def validate():
    cuda_setup();ctx=context();f=foundation(ctx);data=ctx['data']
    from chronos.chronos_bolt import InstanceNorm
    checks={}; safe=f.model.instance_norm
    gt=p.load_gt(ctx['parents']['92601']['GT'],'cuda')
    x,_,_=d.windows(data,data['train_origins'][:2],1024)
    xt=torch.as_tensor(x,device='cuda')
    with torch.no_grad():real=gt.encode(xt)
    fixtures={'ramp':torch.linspace(-2,3,128,device='cuda').reshape(1,128,1),
              'constant':torch.full((1,128,2),2.,device='cuda'),'real_GT_latent':real}
    for name,z in fixtures.items():
        with torch.no_grad():
            f.model.instance_norm=InstanceNorm(eps=safe.eps,use_arcsinh=safe.use_arcsinh)
            legacy=f.forecast(z,12).cpu().numpy()
            f.model.instance_norm=safe
            current=f.forecast(z,12).cpu().numpy()
        checks['F_forward_'+name]=parity(current,legacy)
        differentiable=z.detach().clone().requires_grad_(True)
        output=f.forecast(differentiable,12)
        output.square().mean().backward()
        if differentiable.grad is None or not torch.isfinite(differentiable.grad).all():
            raise ValueError('F input gradient is disconnected or nonfinite: '+name)
        grad=float(differentiable.grad.abs().sum())
        if name != 'constant' and grad == 0:
            raise ValueError('F input gradient is zero: '+name)
        checks['F_gradient_'+name]=dict(finite=True,l1=grad)
    for length in (1024,2048):
        long_x,_,_=d.windows(data,data['train_origins'][16:18],length)
        # Compare encoder boundaries using the exact same filled physical suffix.
        long_x=torch.as_tensor(long_x,device='cuda')
        with torch.no_grad():
            long_z=gt.encode(long_x)[:,-128:]
            short_z=gt.encode(long_x[:,-512:])
        checks['GT_suffix_'+str(length)]=parity(long_z[:,3:].cpu().numpy(),short_z[:,3:].cpu().numpy())
    checks['GT_receptive_field']=dict(physical_ticks=16,latent_boundary_ticks=3,
        explanation='two degree4 past filters and two factor2 max pools; support [4j-12,4j+3]')
    del gt;gc.collect()
    for seed in SEEDS:
        binding=ctx['parents'][str(seed)]
        for mode,key in (('gt','GT'),('ts2','TS2')):
            spec=dict(run_id=f'A0_{key}_L512_s{seed}',phase='A0',mode=mode,seed=seed,L=512,
                      arm='HISTORICAL_CALIBRATED_F_REPRODUCTION',updates=0)
            history,target,mask=d.windows(data,data['val_origins'],512,legacy=True)
            parent_arrays=binding[key]['arrays']
            for name,value in (('targets',target),('mask',mask),('origins',data['val_origins']),('mean',data['mean']),('std',data['std'])):
                if not np.array_equal(parent_arrays[name],value):
                    raise ValueError('Historical parent target/mask/scale differs: '+name)
            route=build_route(spec,ctx,f)
            pred=predict(route,f,history,batch=4 if mode=='ts2' else 8)
            reproduction=parity(pred,parent_arrays['CALIBRATED_F'])
            prediction_artifact(spec,ctx,history,pred,target,mask,reproduction=reproduction,
                                checkpoint_sha256=binding[key]['checkpoint_sha256'])
            checks[spec['run_id']]=reproduction
            del route;gc.collect();torch.cuda.empty_cache()
    invariant=f.verify_unchanged()
    if not all(invariant.values()):raise ValueError('F changed or accumulated gradients')
    checks['F_invariant']=invariant
    checks['F_tensor_state_sha256']=c.state_sha(f.model)
    checks['F_context_maximum']=f.max_context
    c.seal_json(c.REPORT/'PARENT_AND_GRADIENT_VALIDATION.json',dict(status='PASS',checks=checks,source_identity=source_identity()))
    return dict(status='PASS',parent_reproductions=4,gradient_and_forward_checks=len(checks))


def features():
    from .feature_cache import build_feature_cache
    cuda_setup();ctx=context();index={}
    for seed in SEEDS:
        index[str(seed)]={}
        for kind in ('PRE','RAND'):
            c.progress('TS2_features',seed=seed,kind=kind)
            result=build_feature_cache(ctx['parents'][str(seed)]['TS2'],ctx['data'],kind,seed,
                c.CACHE/'features'/f'{kind}_s{seed}',
                callback=lambda stage,current,total:c.progress('TS2_features',seed=seed,kind=kind,substage=stage,current=current,total=total))
            index[str(seed)][kind]={key:result[key] for key in ('features_paths','mean','std','state_sha')}
            c.seal_json(c.REPORT/'features'/f'{kind}_s{seed}.json',result['receipt'])
            gc.collect();torch.cuda.empty_cache()
    c.seal_json(c.CACHE/'FEATURE_INDEX.json',index)
    return dict(status='PASS',feature_spaces=4,new_cache_bytes=c.current_budget()['new_cache_bytes'])
