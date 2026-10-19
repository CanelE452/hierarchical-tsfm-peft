"""CUDA next-update exact-resume validation; no evaluation or new checkpoint."""
import copy
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from research.adapter_diagnostics_20261017_v2.existing_jobs import _foundation
from research.informs_broad_validation_20261015_v1.networks import module_state_hash
from research.informs_broad_validation_20261015_v1.training import (
    PermutationStream, checkpoint_bundle, macro_loss, restore_bundle,
)
from . import common,data_contract,engine,models


def _equal(a,b):
    """Compare the full recursive state exactly, including optimizer tensors."""
    if isinstance(a,torch.Tensor) or isinstance(b,torch.Tensor):
        return (isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor) and a.dtype==b.dtype
                and a.shape==b.shape and torch.equal(a.detach().cpu(),b.detach().cpu()))
    if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
        return (isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and a.dtype==b.dtype
                and a.shape==b.shape and np.array_equal(a,b))
    if isinstance(a,dict) or isinstance(b,dict):
        return isinstance(a,dict) and isinstance(b,dict) and a.keys()==b.keys() and all(_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) or isinstance(b,(list,tuple)):
        return type(a) is type(b) and len(a)==len(b) and all(_equal(x,y) for x,y in zip(a,b))
    return type(a) is type(b) and a==b


def _cpu(value):
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value,dict):
        return {k:_cpu(v) for k,v in value.items()}
    if isinstance(value,list):
        return [_cpu(v) for v in value]
    if isinstance(value,tuple):
        return tuple(_cpu(v) for v in value)
    return copy.deepcopy(value)


def _deadline(limit):
    if time.time()>=limit:
        raise TimeoutError('Registered wall/GPU deadline reached during CUDA resume validation')


def _state_changes(initial,actual):
    if initial.keys()!=actual.keys():
        raise ValueError('Artifact model state keys differ from the declared constructor')
    if any(initial[k].shape!=actual[k].shape or initial[k].dtype!=actual[k].dtype for k in initial):
        raise ValueError('Artifact model state shapes/dtypes differ from the declared constructor')
    return [k for k in initial if not torch.equal(initial[k].detach().cpu(),actual[k].detach().cpu())]


def _read_checkpoint(path,bindings):
    path=Path(path)
    digest=common.sha(path)
    bindings[str(path.relative_to(common.REPO))]=digest
    return torch.load(path,map_location='cpu',weights_only=False)


def _b_state_audit(data,dataset,method,seed,bindings):
    model=models.make(data['s_model'],method,seed).cpu()
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    trainable={name for name,p in model.named_parameters() if p.requires_grad}
    frozen={name for name,p in model.named_parameters() if not p.requires_grad}
    buffers={name for name,_ in model.named_buffers()}
    if method=='READOUT_ONLY':
        if trainable!={'D'} or model.E.requires_grad or any(p.requires_grad for p in model.front.parameters()):
            raise AssertionError('READOUT_ONLY must train only D')
        if not _equal(model.front.state_dict(),data['s_model'].front.state_dict()) or not torch.equal(model.E,data['s_model'].E):
            raise AssertionError('READOUT_ONLY initial features differ from exact S')
    elif method in ('RESIDUAL_F','RESIDUAL_Q'):
        expected={'E','D'}|{'front.'+name for name,_ in model.front.named_parameters()}
        if method=='RESIDUAL_Q':expected|={'q.'+name for name,_ in model.q.named_parameters()}
        if trainable!=expected or frozen:
            raise AssertionError('Residual F/Q learnable P/E/D/Q flags differ')
    elif method=='RESIDUAL_RAW_MLP':
        if frozen or not trainable:
            raise AssertionError('Residual raw MLP parameters must be trainable')
    directory=common.CACHE/'runs'/dataset/f'{method}_s{seed}'
    cp0=_read_checkpoint(directory/'search'/engine._endpoint_name(0),bindings)
    if _state_changes(initial,cp0['model']):
        raise AssertionError('Actual search CP0 differs from exact fresh initialization')
    raw_zero=None
    if method=='RESIDUAL_RAW_MLP':
        model.load_state_dict(cp0['model'],strict=True)
        with torch.no_grad():
            correction=model(torch.as_tensor(data['x'][:4]))
        raw_zero=bool(torch.count_nonzero(correction)==0)
        if not raw_zero:raise AssertionError('Actual raw residual MLP CP0 output is not zero')
    elif bool(torch.count_nonzero(cp0['model']['D'])):
        raise AssertionError('Actual latent CP0 residual D must be exactly zero')
    rows={}
    for phase in ('search','refit'):
        saved=_read_checkpoint(directory/phase/'last.pt',bindings)
        ext=saved['contract']['external_contract']
        if (saved['extra']['phase']!=phase or saved['extra']['seed']!=seed
                or ext['dataset']!=dataset or ext['method']!=method
                or ext['parent_G']!=data['parent_g']['mirror']['sha256']
                or ext['parent_S']!=data['parent_s']['mirror']['sha256']):
            raise ValueError('B artifact phase/seed/parent contract differs')
        if saved['contract']['model_sha']!=engine._state_sha(models.make(data['s_model'],method,seed)):
            raise ValueError('B artifact initialization hash differs')
        changed=_state_changes(initial,saved['model'])
        nontrainable_changes=sorted(set(changed)&(frozen|buffers))
        undeclared=sorted(set(changed)-trainable)
        if nontrainable_changes or undeclared:
            raise AssertionError('B artifact changed a frozen parameter/buffer: '+str(undeclared))
        if 'center' in initial and not torch.equal(initial['center'],saved['model']['center']):
            raise AssertionError('Residual center changed in serialized artifact')
        if method=='READOUT_ONLY' and any(k!='D' for k in changed):
            raise AssertionError('READOUT_ONLY changed original S input features')
        exposures=int(saved['sampler']['exposures'])
        last_epoch=float(saved['extra']['curves'][-1]['epoch'])
        if phase=='refit' and exposures!=int(np.ceil(last_epoch*len(data['origins']))):
            raise AssertionError('Fresh refit selected epoch and dense M_cal exposures differ')
        zero_refit=(phase=='refit' and last_epoch==0)
        if zero_refit and (changed or int(saved['updates'])!=0):
            raise AssertionError('Epoch-zero fresh refit must retain every initialization tensor')
        rows[phase]=dict(changed_trainable_parameters=changed,nontrainable_or_buffer_changes=nontrainable_changes,
            updates=int(saved['updates']),target_origin_exposures=exposures,last_scored_epoch=last_epoch,
            zero_epoch_refit=zero_refit,zero_epoch_refit_unchanged=True if zero_refit else None,
            final_model_sha256=engine._state_sha(_loaded(model,saved['model'])))
    return dict(dataset=dataset,method=method,seed=seed,status='PASS',
        declared_trainable_parameters=sorted(trainable),declared_frozen_parameters=sorted(frozen),
        declared_buffers=sorted(buffers),actual_CP0_initialization_identical=True,
        actual_raw_MLP_CP0_output_zero=raw_zero,states=rows,
        training_flags_evidence='Current exact model constructor plus source-bound artifact contract; flags are not serialized historical runtime traces')


def _loaded(model,state):
    model.load_state_dict(state,strict=True)
    return model


def _c_state_audit(data,dataset,method,seed,directory,bindings):
    from .blocks import LatentBlocks,frozen_state_hash
    result=common.read(directory/'RESULT.json')
    if result['status']!='COMPLETE':
        raise ValueError('C artifact group is not complete')
    payload=_read_checkpoint(directory/'model.pt',bindings)
    if (payload['schema']!='incremental_C_final_model_v1' or payload['method']!=method
            or payload['dataset']!=dataset or payload['seed']!=seed
            or common.sha(directory/'model.pt')!=result['model_file_sha256']):
        raise ValueError('C final artifact metadata/bytes differ')
    if (payload['contract']['parent_G']!=data['parent_g']['mirror']['sha256']
            or payload['contract']['parent_S']!=data['parent_s']['mirror']['sha256']):
        raise ValueError('C artifact was initialized from different registered A60 parents')
    model=LatentBlocks(data['s_model'],method).cpu()
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    changed=_state_changes(initial,payload['model'])
    if any(k=='center' or k.startswith('front.') for k in changed):
        raise AssertionError('Final C front/center differs from original A60 S')
    if any(p.requires_grad for p in model.front.parameters()):
        raise AssertionError('C constructor front is not frozen')
    model.load_state_dict(payload['model'],strict=True)
    if engine._state_sha(model)!=result['model_sha256']:
        raise ValueError('C final serialized model hash differs')
    freeze_checks=copy.deepcopy(result.get('frozen_stage1_checks',{}))
    independent_freeze_checks={}
    if method=='SEQ_BLOCKS':
        required=('parameter_buffer_hash_unchanged','optimizer_checkpoint_hash_unchanged',
                  'refit_parameter_buffer_hash_unchanged','refit_optimizer_checkpoint_hash_unchanged',
                  'active_block2_parameters_disjoint')
        if not all(freeze_checks.get(k) is True for k in required):
            raise AssertionError('Recorded SEQ stage1 freeze checks missing/failing')
        first_path=Path(result['searches'][0]['selected_checkpoint'])
        first=_read_checkpoint(first_path,bindings)
        second=_read_checkpoint(Path(result['searches'][1]['selected_checkpoint']),bindings)
        if common.sha(first_path)!=freeze_checks['checkpoint_sha256']:
            raise AssertionError('SEQ selected stage1 optimizer checkpoint bytes changed')
        model.set_stage(1,correction_only=True)
        model.load_state_dict(first['model'],strict=True)
        if frozen_state_hash(model)!=freeze_checks['frozen_state_sha256']:
            raise AssertionError('SEQ recorded selected stage1 frozen hash differs')
        protected=[k for k in initial if k=='center' or k.startswith(('front.','E_blocks.0','D_blocks.0'))]
        if any(not torch.equal(first['model'][k],second['model'][k]) for k in protected):
            raise AssertionError('Selected SEQ stage2 artifact changed frozen stage1/front/center')
        refit1_path=Path(result['refits'][0]['checkpoint_path'])
        refit1=_read_checkpoint(refit1_path,bindings)
        refit2=_read_checkpoint(Path(result['refits'][1]['checkpoint_path']),bindings)
        model.load_state_dict(refit1['model'],strict=True)
        if frozen_state_hash(model)!=freeze_checks['refit_frozen_state_sha256']:
            raise AssertionError('SEQ fresh stage1 frozen hash differs')
        if common.sha(refit1_path)!=refit2['contract']['external_contract']['stage1_checkpoint_sha']:
            raise AssertionError('SEQ fresh stage1 optimizer checkpoint bytes changed')
        if any(not torch.equal(refit1['model'][k],refit2['model'][k])
               or not torch.equal(refit1['model'][k],payload['model'][k]) for k in protected):
            raise AssertionError('Fresh SEQ stage2/final artifact changed frozen stage1/front/center')
        independent_freeze_checks=dict(selected_stage2_protected_tensors_equal=True,
            fresh_refit_stage2_and_final_protected_tensors_equal=True,
            recorded_search_and_refit_frozen_hashes_reconstructed=True,
            selected_and_fresh_stage1_optimizer_checkpoint_bytes_preserved=True)
    raw_counts=result.get('foundation_counts_all_work',{})
    return dict(dataset=dataset,method=method,seed=seed,status='PASS',
        final_P_and_center_identical_to_A60=True,changed_block_parameters=changed,
        recorded_SEQ_freeze_checks=freeze_checks,independent_SEQ_artifact_freeze_checks=independent_freeze_checks,
        observations=[dict(field='foundation_counts_all_work.calls',raw_value=raw_counts.get('calls'),
                           interpreted_value=None,status='N/A_UNINSTRUMENTED_ATTRIBUTE',
                           reason='The foundation exposes physical_calls/row_calls, not calls; original private value preserved')],
        evidence_scope='Serialized final/checkpoint state comparison and existing recorded SEQ freeze checks; not a live parent-object before/after trace')


def audit_training_states(oldctx):
    """CPU artifact state audit of all mandatory B and completed admitted C models."""
    output=common.REPORT/'TRAINABLE_STATE_VALIDATION.json'
    if output.exists():
        raise FileExistsError('Preserve prior trainable-state validation attempt')
    started=time.perf_counter()
    limit=common.deadline(False)
    ctx=dict(oldctx,device='cpu')
    ctx.pop('_incremental_data',None)
    bindings={}
    result=dict(status='FAIL',scope='artifact_state_audit',B=[],C=[],
        historical_runtime_parent_P_before_after_claimed=False,
        new_V_numeric_reads=0,new_T_numeric_reads=0,GPU_operations=0,
        source_sha256=common.sha(Path(__file__)))
    try:
        admission=common.read(common.REPORT/'C_ADMISSION.json') if (common.REPORT/'C_ADMISSION.json').exists() else {'datasets':{}}
        for dataset in common.DATASETS:
            for seed in common.SEEDS:
                _deadline(limit)
                data=data_contract.load(ctx,dataset,seed)
                for role in ('parent_g','parent_s'):
                    parent=data[role]
                    path=ctx['cache']/parent['mirror']['reference'].removeprefix('cache://')
                    bindings[str(path.relative_to(common.REPO))]=parent['mirror']['sha256']
                for method in common.METHODS:
                    result['B'].append(_b_state_audit(data,dataset,method,seed,bindings))
                gate=admission['datasets'].get(dataset,{})
                if gate.get('status')=='ADMITTED':
                    for method in ('SEQ_BLOCKS','JOINT_FULL','JOINT_PREFIX'):
                        _deadline(limit)
                        directory=common.CACHE/'C'/dataset/f'{method}_s{seed}'
                        result['C'].append(_c_state_audit(data,dataset,method,seed,directory,bindings))
        unchanged=all(common.sha(common.REPO/relative)==digest for relative,digest in bindings.items())
        if not unchanged:raise AssertionError('Source parent/checkpoint artifact changed during CPU audit')
        if len(result['B'])!=16:raise AssertionError('All sixteen mandatory B configurations must be audited')
        result.update(status='PASS',B_configurations=16,B_search_states=16,B_refit_states=16,B_actual_CP0_states=16,
            C_completed_configurations=len(result['C']),source_checkpoint_and_parent_artifacts_unchanged=True,
            source_artifact_bindings=bindings,
            limitation='Artifact state equality establishes serialized final/frozen states; no blanket historical live-P immutability claim beyond existing recorded checks')
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error),source_artifact_bindings=bindings)
        raise
    finally:
        result['seconds']=time.perf_counter()-started
        common.atomic(output,result)
    return result


def _one_update(saved,data,foundation,device,limit):
    _deadline(limit)
    fresh=models.make(data['s_model'],'RESIDUAL_F',92601).to(device)
    optimizer=torch.optim.AdamW([p for p in fresh.parameters() if p.requires_grad],lr=.001,weight_decay=0.)
    fit=data['split']['fit_indices']
    sampler=PermutationStream(fit,92601)
    # Deepcopy also isolates non-capturable AdamW CPU step tensors from the source bundle.
    local=copy.deepcopy(saved)
    updates=restore_bundle(local,fresh,optimizer,sampler,local['contract'])
    if not _equal(sampler.state_dict(),saved['sampler']):
        raise AssertionError('Restored sampler differs before the next update')
    if not _equal(optimizer.state_dict(),saved['optimizer']):
        raise AssertionError('Full optimizer state differs before the next update')
    if not torch.equal(torch.get_rng_state(),saved['torch_rng']):
        raise AssertionError('CPU RNG restoration differs')
    if not _equal(torch.cuda.get_rng_state_all(),saved['cuda_rng']):
        raise AssertionError('CUDA RNG restoration differs')
    expected_model=engine._state_sha(fresh)
    if not _equal(fresh.state_dict(),saved['model']):
        raise AssertionError('Full model restoration differs')
    rows=sampler.draw(16)
    counts=torch.as_tensor(data['mask'][fit].sum((0,1)),device=device)
    optimizer.zero_grad(set_to_none=True)
    fresh.train()
    foundation.model.eval()
    before_rows=int(foundation.physical_streams)
    for lo in range(0,16,4):
        _deadline(limit)
        indices=rows[lo:lo+4]
        x=torch.as_tensor(data['x'][indices],device=device)
        g=torch.as_tensor(data['g'][indices],device=device)
        y=torch.as_tensor(np.where(data['mask'][indices],data['y'][indices],0.),device=device)
        mask=torch.as_tensor(data['mask'][indices],device=device)
        correction=fresh(x,foundation)
        loss=macro_loss(g+correction,y,mask,counts,len(fit),16)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite resume-smoke training loss')
        loss.backward()
    p_grad=any(p.grad is not None and bool(torch.count_nonzero(p.grad)) for p in fresh.front.parameters())
    e_grad=fresh.E.grad is not None and bool(torch.count_nonzero(fresh.E.grad))
    if not p_grad or not e_grad:
        raise AssertionError('The nonzero-D resumed update must have P/E gradients')
    norm=torch.nn.utils.clip_grad_norm_(fresh.parameters(),1.)
    if not torch.isfinite(norm):
        raise FloatingPointError('Nonfinite resume-smoke gradient norm')
    optimizer.step()
    torch.cuda.synchronize(device)
    _deadline(limit)
    bundle=checkpoint_bundle(fresh,optimizer,sampler,local['contract'],updates+1)
    receipt=dict(rows=rows.tolist(),origins=data['origins'][rows].tolist(),
        starting_updates=int(updates),updates_after=int(updates+1),
        starting_exposures=int(saved['sampler']['exposures']),exposures_after=int(sampler.exposures),
        restored_model_sha256=expected_model,next_model_sha256=engine._state_sha(fresh),
        full_optimizer_restored_before_update=True,
        P_gradient_nonzero=p_grad,E_gradient_nonzero=e_grad,
        F_physical_rows=int(foundation.physical_streams)-before_rows,
        CPU_rng_sha256=hashlib.sha256(bundle['torch_rng'].numpy().tobytes()).hexdigest(),
        CUDA_rng_sha256=[hashlib.sha256(t.numpy().tobytes()).hexdigest() for t in bundle['cuda_rng']])
    result=_cpu(bundle)
    del fresh,optimizer,local
    return result,receipt


def run(oldctx):
    """Execute two isolated restores/one-step continuations; never save weights."""
    if not str(oldctx['device']).startswith('cuda'):
        raise ValueError('This validation requires the real registered CUDA worker')
    path=common.REPORT/'CUDA_RESUME_VALIDATION.json'
    if path.exists():
        raise FileExistsError('Preserve prior CUDA validation attempt: '+str(path))
    limit=common.deadline(False)
    _deadline(limit)
    started=time.perf_counter()
    source=common.CACHE/'runs/jena/RESIDUAL_F_s92601/search/last.pt'
    checkpoint_sha=common.sha(source)
    saved=torch.load(source,map_location='cpu',weights_only=False)
    ext=saved['contract']['external_contract']
    if ext['dataset']!='jena' or ext['method']!='RESIDUAL_F' or saved['extra']['seed']!=92601 or saved['extra']['phase']!='search':
        raise ValueError('Wrong dataset/method/seed/phase in CUDA resume source')
    if not saved['cuda_rng']:
        raise ValueError('Real CUDA RNG state is required')
    data=data_contract.load(oldctx,'jena',92601)
    if ext['parent_G']!=data['parent_g']['mirror']['sha256'] or ext['parent_S']!=data['parent_s']['mirror']['sha256']:
        raise ValueError('Resume source differs from the registered A60 G/S parents')
    if ext['F_row_microbatch']!=128 or ext['checkpoint_inputs'] is not False:
        raise ValueError('Resume foundation execution recipe differs')
    initial=models.make(data['s_model'],'RESIDUAL_F',92601)
    arrays,_=engine._prepare(data)
    current_contract=engine._contract(initial,arrays,data['split']['fit_indices'],
                                      data['split']['select_indices'],92601,ext,'search')
    if not _equal(current_contract,saved['contract']):
        raise ValueError('Current model/data/engine contract differs from the real resume checkpoint')
    del initial,arrays
    g_before=module_state_hash(data['g_model'])
    s_before=module_state_hash(data['s_model'])
    pool=None
    result=dict(status='FAIL',dataset='jena',method='RESIDUAL_F',seed=92601,
        source_checkpoint_reference=source.relative_to(common.REPO).as_posix(),
        source_checkpoint_sha256=checkpoint_sha,source_sha256=common.sha(Path(__file__)),
        smoke_only=True,research_result=False,new_checkpoint_written=False,
        inference_evaluations=0,new_selection_opportunities=0,new_V_numeric_reads=0,new_T_numeric_reads=0)
    try:
        _deadline(limit)
        pool,foundation=_foundation(oldctx,oldctx['modules'],data['task']['H'],oldctx['device'])
        foundation.row_microbatch,foundation.checkpoint_inputs=128,False
        if ext['F_tensor_sha256']!=foundation.initial_hash:
            raise ValueError('Resume source foundation differs')
        first,one=_one_update(saved,data,foundation,oldctx['device'],limit)
        second,two=_one_update(saved,data,foundation,oldctx['device'],limit)
        checks={name:_equal(first[name],second[name]) for name in ('model','optimizer','sampler','torch_rng','cuda_rng','updates')}
        checks['next_batch']=one['rows']==two['rows']
        checks['checkpoint_file_unchanged']=common.sha(source)==checkpoint_sha
        checks['G_frozen_unchanged']=module_state_hash(data['g_model'])==g_before
        checks['S_frozen_unchanged']=module_state_hash(data['s_model'])==s_before
        foundation.assert_frozen_unchanged()
        checks['F_frozen_unchanged']=True
        checks['both_P_E_gradient_nonzero']=all(r['P_gradient_nonzero'] and r['E_gradient_nonzero'] for r in (one,two))
        if not all(checks.values()):
            raise AssertionError('CUDA exact resume validation failed: '+str(checks))
        result.update(status='PASS',bitwise_checks=checks,continuations=[one,two],
            isolated_next_update_count=2,effective_batch=16,microbatch=4,
            F_row_microbatch=128,checkpoint_inputs=False,F_tensor_sha256=foundation.initial_hash,
            optimizer='AdamW lr=.001 weight_decay=0 clip_norm=1',
            scope='Full checkpoint restored twice into independent fresh models; same next batch/update; no selection, forecast score or source checkpoint write',
            torch_version=torch.__version__,GPU=torch.cuda.get_device_name(oldctx['device']))
    except BaseException as error:
        result.update(error_type=type(error).__name__,error=str(error),
                      checkpoint_file_unchanged=common.sha(source)==checkpoint_sha)
        raise
    finally:
        try:
            if pool is not None:
                pool.close()
        except BaseException as error:
            result.update(status='FAIL',close_error_type=type(error).__name__,close_error=str(error))
            raise
        finally:
            result['seconds']=time.perf_counter()-started
            common.atomic(path,result)
    return result
