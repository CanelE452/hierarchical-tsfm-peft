"""Preloaded CPU FP32 Ridge heads, with GPU representations or Bolt as needed."""
import numpy as np
import torch
from threadpoolctl import threadpool_limits, threadpool_info
import runtime as runtime
import probes as pr
import benchmark as bench

IMPLEMENTATION_ID='cpu_numpy_fp32_ridge_v1'
RIDGE_ROUTES=('E_RAW_RIDGE','E_ENRICHED_RIDGE','RAW_JOINT_RIDGE','RAW_BOLT_RAW_R','RAW_SELF33')


def blas_context():
    return threadpool_limits(limits=2,user_api='blas')


def load_ridge(record,rt=runtime):
    artifact=record['model'];path=rt.ROOT/artifact['path']
    if rt.sha(path)!=artifact['sha256']:raise ValueError('CPU_RIDGE_MODEL_SHA')
    with np.load(path,allow_pickle=False) as z:
        ridge={k:z[k].copy() for k in z.files}
    if ridge['weight'].dtype!=np.float32 or ridge['bias'].dtype!=np.float32 or not np.isfinite(ridge['weight']).all() or not np.isfinite(ridge['bias']).all():
        raise ValueError('CPU_RIDGE_COEFFICIENT_DTYPE_OR_NONFINITE')
    ridge['horizon']=int(ridge.get('horizon',48));ridge['channels']=int(ridge.get('channels',len(ridge['weight'])))
    return ridge


def prepare_memory(model):
    if model is None:return {'model_present':False,'cache_recomputed':False}
    before=runtime.state_hash(model);parameters=runtime.state_hash(dict(model.named_parameters()))
    recomputed=not bool(model.memory_is_frozen)
    if recomputed:model.freeze_memory()
    after=runtime.state_hash(model)
    unchanged=parameters==runtime.state_hash(dict(model.named_parameters()))
    if not unchanged:raise ValueError('CPU_RIDGE_CACHE_PREPARATION_CHANGED_PARAMETERS')
    return {'model_present':True,'cache_recomputed':recomputed,'model_state_before_sha256':before,
        'model_state_after_sha256':after,'registered_parameters_unchanged':unchanged,
        'fixed_memory_is_frozen':bool(model.memory_is_frozen),
        'memory_buffers_sha256':runtime.state_hash({name:value for name,value in model.named_buffers() if name.startswith('memory_')}),
        'preparation_outside_timing':True}


def deployment(pack,physical,observed,origins,route,forecaster,model=None,ridge=None):
    if route not in RIDGE_ROUTES:raise ValueError('CPU_RIDGE_ROUTE_NOT_APPROVED')
    x=bench.preprocess(pack,physical,observed)
    if route in ('E_RAW_RIDGE','E_ENRICHED_RIDGE'):
        xt=torch.as_tensor(x,dtype=torch.float32,device='cuda')
        if route=='E_RAW_RIDGE':z=model.codec.encode(xt)
        else:
            if not bool(model.memory_is_frozen):raise ValueError('CPU_RIDGE_MEMORY_NOT_PREPARED_OUTSIDE_TIMING')
            z,_=model.encode_with_memory(xt,torch.as_tensor(origins,dtype=torch.long,device='cuda'))
        features=torch.cat((z.flatten(1),xt[:,-1]),dim=1).cpu().numpy()
        prediction=x[:,-1:]+pr.ridge_predict(features,ridge)
    elif route=='RAW_SELF33':
        prediction=np.empty((len(x),48,x.shape[-1]),np.float32)
        for channel in range(x.shape[-1]):
            one={'weight':ridge['weight'][channel],'bias':ridge['bias'][channel],'horizon':48,'channels':1}
            features=pr.block_features(x[:,:,channel:channel+1])
            prediction[:,:,channel:channel+1]=x[:,-1:,channel:channel+1]+pr.ridge_predict(features,one)
    else:
        base=forecaster(torch.as_tensor(x,dtype=torch.float32,device='cuda')).cpu().numpy() if route=='RAW_BOLT_RAW_R' else x[:,-1:]
        prediction=base+pr.ridge_predict(pr.block_features(x),ridge)
    return prediction*pack['std']+pack['mean']


def implementation_metadata(route,model=None,ridge=None):
    blas=[{k:p[k] for k in ('internal_api','num_threads','version','architecture') if k in p}
        for p in threadpool_info() if p['user_api']=='blas']
    return {'deployment_implementation':IMPLEMENTATION_ID,'route':route,'blas_threads_contract':2,
        'actual_BLAS_pools':blas,'all_BLAS_pools_have_two_threads':all(p['num_threads']==2 for p in blas),
        'Torch_threads':torch.get_num_threads(),'head_backend':'Original NumPy FP32 block_features/ridge_predict/channelwise self head',
        'inverse_backend':'CPU NumPy separate FP32 multiply/add','GPU_E_active':route.startswith('E_'),
        'GPU_Bolt_active':route=='RAW_BOLT_RAW_R','coefficients_preloaded_on_CPU':True,
        'coefficient_count':0 if ridge is None else ridge['weight'].size+ridge['bias'].size,
        'checkpoint_model_registered_parameters':0 if model is None else sum(p.numel() for p in model.parameters()),
        'full_checkpoint_co_resident':model is not None,'registered_and_active_are_different':True,
        'expected_prediction_cache_used':False,'disk_and_cache_preparation_excluded_from_latency':True,
        'CPU_feature_D2H_head_and_inverse_included_in_latency':True}
