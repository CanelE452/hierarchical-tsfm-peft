"""Approved Robin row-microbatch parity and deployment-cost grid."""
import argparse
from collections import defaultdict
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

from runtime_v7 import (HERE, ROOT, CACHE, V6, V6_CACHE, Job, digest, load_data,
                        save_json, score_arrays, source_receipt, environment_receipt)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def cleanup():
    torch.cuda.synchronize()
    gc.collect()
    torch._C._cuda_clearCublasWorkspaces()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def telemetry():
    fields = 'index,uuid,temperature.gpu,clocks.current.sm,utilization.gpu,power.draw,pstate'
    result = subprocess.run(['nvidia-smi', '--query-gpu='+fields,
                             '--format=csv,noheader,nounits'], capture_output=True, text=True)
    apps = subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory',
                           '--format=csv,noheader'],capture_output=True,text=True)
    return dict(utc=time.time(),fields=fields,gpu=result.stdout.strip().splitlines(),
                process_gpu_memory=apps.stdout.strip().splitlines(),meaning='N/A is unavailable, not zero')


def configs(family, batch):
    if family == 'direct_nlinear':
        return [('original', None)]
    if family == 'level_res':
        return [('original', None), ('generic', batch*5)] + ([('generic',5)] if batch==4 else [])
    return [('original',None),('generic',batch*17)] + [('generic',m) for m in ([5] if batch==1 else [20,5])]


def point(model, x, micro):
    b,l,c=x.shape
    flat=x.transpose(1,2).reshape(b*c,l)
    result=torch.empty((b*c,48),device=x.device,dtype=x.dtype)
    for start in range(0,b*c,micro):
        raw=model.backbone(context=flat[start:start+micro])
        result[start:start+micro].copy_(raw.quantile_preds[:,model.median,:48])
        del raw
    return result.reshape(b,c,48).transpose(1,2)


def generic(model,x,micro):
    if model.family in ('f0','lora'):
        return point(model,x,micro)
    if model.family != 'level_res':
        raise ValueError('Generic wrapper is restricted to approved LEVEL/F0/LoRA')
    z=model.encoder(x)
    residual=x-model.decoder(z)
    main=model.decoder(point(model,z,micro))
    last=residual[:,-1:,:].detach()
    return main+last.expand(-1,48,-1)+model.residual(residual-last)


def forward(model,x,variant,micro,device):
    x=x.to(device=device,dtype=torch.float32)
    y=model(x) if variant=='original' else generic(model,x,micro)
    return y.float().cpu()


def predict(model, inputs, batch, variant='original', micro=None, device='cuda'):
    return torch.cat([forward(model,x,variant,micro,device) for x in inputs.split(batch)])


def difference(actual,reference):
    delta=(actual-reference).abs()
    relative=delta/reference.abs().clamp_min(1e-12)
    violation=delta>1e-5+1e-4*reference.abs()
    return dict(pass_=not bool(violation.any()),atol=1e-5,rtol=1e-4,
                max_absolute_difference=float(delta.max()),
                max_relative_difference=float(relative.max()),
                relative_denominator_floor=1e-12,violating_elements=int(violation.sum()),
                elements=actual.numel())


def load_candidate(item, device):
    mod=module(item['module'],'_cost_v7_'+item['module'].stem)
    model=mod.restore_model(item['checkpoint'],device=device).eval()
    info=dict(trainable_parameters_before_merge=sum(p.numel() for p in model.parameters() if p.requires_grad),
              total_parameters_before_merge=sum(p.numel() for p in model.parameters()),
              checkpoint_path=str(item['checkpoint']),checkpoint_sha256=digest(item['checkpoint']))
    return model,info


def merge(model,inputs,device):
    before=predict(model,inputs,4,device=device)
    model.backbone=model.backbone.merge_and_unload(safe_merge=True)
    after=predict(model,inputs,4,device=device)
    parity=difference(after,before)
    if not parity['pass_']:
        raise ValueError('LoRA returned merged-model parity failed: '+str(parity))
    return parity


def model_bytes(model):
    pbytes=sum(p.numel()*p.element_size() for p in model.parameters())
    bbytes=sum(p.numel()*p.element_size() for p in model.buffers())
    return dict(deployed_parameters=sum(p.numel() for p in model.parameters()),
                parameter_bytes=pbytes,buffer_bytes=bbytes,deployment_tensor_bytes=pbytes+bbytes)


def items():
    result=[]
    for family,seeds in [('f0',[None]),('lora',[92601,92602]),('level_res',[92601,92602])]:
        for seed in seeds:
            identifier='v6_robin_'+family+('' if seed is None else '_'+str(seed))
            result.append(dict(id=identifier,family=family,seed=seed,
                               checkpoint=V6_CACHE/'runs'/identifier/'best.pt',module=V6/'model_v6.py'))
    selected=json.loads((HERE/'selected_robin.json').read_text(encoding='utf-8'))
    ids=selected['selected']['direct_nlinear']['run_ids']
    if len(ids)!=2:
        raise ValueError('Exactly two VAL-selected direct NLinear seeds required')
    for identifier in ids:
        result.append(dict(id=identifier,family='direct_nlinear',seed=int(identifier.rsplit('_',1)[1]),
                           checkpoint=CACHE/'runs'/identifier/'best.pt',module=HERE/'model_v7.py'))
    return result


def prepare():
    data=load_data('robin',include_test=False)
    origins=np.asarray(data['val_origins'])
    if len(origins)!=30:
        raise ValueError('Approved Robin VAL origin count changed')
    chosen=origins[np.linspace(0,29,24,dtype=int)]
    inputs=torch.from_numpy(np.stack([data['x'][o-512:o] for o in chosen]))
    test=load_data('robin',include_test=True)
    test_origins=np.concatenate([test['test_a_origins'],test['test_b_origins']])
    test_inputs=torch.from_numpy(np.stack([test['x'][o-512:o] for o in test_origins]))
    targets=np.stack([test['x'][o:o+48] for o in test_origins])
    mask=np.stack([test['finite'][o:o+48] for o in test_origins])
    return inputs,chosen,test_inputs,test_origins,targets,mask


def parity_all(candidates,prepared,job):
    inputs,origins,test_inputs,test_origins,targets,mask=prepared
    rows=[]
    for item in candidates:
        cleanup()
        model,info=load_candidate(item,'cuda')
        if item['family']=='lora':
            info['merge']=merge(model,inputs,'cuda')
        base_val=predict(model,inputs,4)
        base_test=predict(model,test_inputs,4)
        for batch in (1,4):
            for variant,micro in configs(item['family'],batch):
                job.heartbeat(f'parity {item["id"]} B{batch} {variant} M{micro}')
                actual=predict(model,inputs,batch,variant,micro)
                check=difference(actual,base_val)
                output=predict(model,test_inputs,batch,variant,micro)
                test_check=difference(output,base_test)
                if not check['pass_'] or not test_check['pass_']:
                    rows.append(dict(id=item['id'],batch=batch,variant=variant,micro=micro,
                                     val=check,test=test_check,status='FAILED'))
                    save_json(HERE/'microbatch_parity_incomplete.json',dict(rows=rows,source=source_receipt()))
                    raise ValueError('Full/chunk parity failed; no tolerance relaxation')
                score=score_arrays(output.numpy(),targets,mask)
                baseline=score_arrays(base_test.numpy(),targets,mask)
                rows.append(dict(id=item['id'],family=item['family'],seed=item['seed'],batch=batch,
                                 variant=variant,micro=micro,val=check,test=test_check,
                                 test_mse=score['mse'],test_mae=score['mae'],
                                 test_mse_difference=score['mse']-baseline['mse'],
                                 prediction_sha256=hashlib.sha256(output.numpy().tobytes()).hexdigest(),
                                 model=info))
                del actual,output
        if item['family']=='direct_nlinear':
            model.to('cpu')
            cpu=predict(model,inputs,4,device='cpu')
            cpu_check=difference(cpu,base_val)
            if not cpu_check['pass_']:
                raise ValueError('Direct CPU/GPU numerical parity failed')
            rows.append(dict(id=item['id'],kind='cpu_gpu_parity',val=cpu_check))
            del cpu
        del model,base_val,base_test
        cleanup()
    binding={str(path):digest(path) for path in [HERE/'cost_v7.py',HERE/'runtime_v7.py',HERE/'protocol.json',
             HERE/'selected_robin.json',HERE/'data_contract_robin.json']}
    for candidate in candidates:
        for path in (candidate['checkpoint'],candidate['module']):
            binding[str(path)]=digest(path)
    result=dict(status='PASS',rows=rows,val_origins=origins.tolist(),test_origins=test_origins.tolist(),binding=binding,
                test_information='Already exposed Robin; score differences verify same model, not choose models',
                source=source_receipt())
    save_json(HERE/'microbatch_parity.json',result)
    return result


def describe(values):
    return dict(median=float(np.median(values)),q25=float(np.quantile(values,.25)),
                q75=float(np.quantile(values,.75)),min=float(min(values)),max=float(max(values)))


def measure(model,inputs,batch,variant,micro,device,job):
    chunks=list(inputs.split(batch))
    if device=='cuda':
        cleanup()
        loaded=dict(allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())
    else:
        loaded=None
    for index in range(10):
        job.check_limits()
        forward(model,chunks[index%len(chunks)],variant,micro,device)
    if device=='cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    times=[]
    for repeat in range(20):
        job.check_limits()
        if device=='cuda':
            torch.cuda.synchronize()
        tick=time.perf_counter()
        for chunk in chunks:
            forward(model,chunk,variant,micro,device)
        if device=='cuda':
            torch.cuda.synchronize()
        times.append(time.perf_counter()-tick)
    mem=psutil.Process().memory_info()
    result=dict(seconds_per_24_origins=times,milliseconds_per_origin=describe([t*1000/24 for t in times]),
                origins_per_second=describe([24/t for t in times]),
                loaded_resident=loaded,process_cpu_memory={k:int(getattr(mem,k)) for k in ('rss','vms','peak_wset') if hasattr(mem,k)})
    if device=='cuda':
        result.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      final_allocated_bytes=torch.cuda.memory_allocated())
    return result


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:
        if row['sentinel'] is None:
            groups[(row['id'],row['batch'],row['variant'],row['micro'],row['device'])].append(row)
    fits=[]
    for key,values in groups.items():
        if len(values)!=3 or {r['block'] for r in values}!={0,1,2}:
            raise ValueError('Every cost configuration requires all three blocks')
        first=values[0]
        fits.append(dict(id=key[0],batch=key[1],variant=key[2],micro=key[3],device=key[4],
                         family=first['family'],seed=first['seed'],
                         milliseconds_per_origin=describe([r['milliseconds_per_origin']['median'] for r in values]),
                         origins_per_second=describe([r['origins_per_second']['median'] for r in values]),
                         peak_allocated_bytes=[r.get('peak_allocated_bytes') for r in values],
                         peak_reserved_bytes=[r.get('peak_reserved_bytes') for r in values],model=first['model']))
    byfamily=defaultdict(list)
    for row in fits:
        byfamily[(row['family'],row['batch'],row['variant'],row['micro'],row['device'])].append(row)
    centers=[]
    for key,values in byfamily.items():
        peaks=[np.median(r['peak_allocated_bytes']) for r in values if r['peak_allocated_bytes'][0] is not None]
        all_peaks=[p for r in values for p in r['peak_allocated_bytes'] if p is not None]
        centers.append(dict(family=key[0],batch=key[1],variant=key[2],micro=key[3],device=key[4],
                            mean_seed_median_ms=float(np.mean([r['milliseconds_per_origin']['median'] for r in values])),
                            mean_seed_median_throughput=float(np.mean([r['origins_per_second']['median'] for r in values])),
                            mean_seed_median_peak_allocated_bytes=float(np.mean(peaks)) if peaks else None,
                            observed_peak_allocated_range_bytes=[int(min(all_peaks)),int(max(all_peaks))] if all_peaks else None,
                            ids=[r['id'] for r in values]))
    drift=[]
    for block in range(3):
        for batch in (1,4):
            pair={r['sentinel']:r for r in rows if r['block']==block and r['batch']==batch and r['sentinel']}
            pre,post=(pair[k]['milliseconds_per_origin']['median'] for k in ('pre','post'))
            drift.append(dict(block=block,batch=batch,pre_ms=pre,post_ms=post,post_relative_change=(post/pre-1)))
    return dict(fits=fits,groups=centers,sentinels=drift,
                aggregation='median of three block medians per fit; arithmetic mean of seed medians; no iid confidence claim')


def cost_grid(candidates,prepared,job,device,rows):
    inputs=prepared[0]
    chosen=candidates if device=='cuda' else [r for r in candidates if r['family']=='direct_nlinear']
    def measure_item(item,block,sentinel=None):
        if device=='cuda':
            cleanup()
        model,info=load_candidate(item,device)
        if item['family']=='lora':
            info['merge']=merge(model,inputs,device)
        info.update(model_bytes(model))
        grid=[(b,v,m) for b in (1,4) for v,m in configs(item['family'],b)]
        if sentinel:
            grid=[(1,'original',None),(4,'original',None)]
        elif block==1:
            grid=grid[::-1]
        elif block==2:
            grid=grid[1:]+grid[:1]
        for batch,variant,micro in grid:
            job.heartbeat(f'cost {device} block{block} {item["id"]} B{batch} {variant} M{micro} {sentinel}')
            record=dict(id=item['id'],family=item['family'],seed=item['seed'],block=block,batch=batch,
                        variant=variant,micro=micro,device=device,sentinel=sentinel,
                        sequence=len(rows),model=info,telemetry_before=telemetry() if device=='cuda' else None)
            record.update(measure(model,inputs,batch,variant,micro,device,job))
            record['telemetry_after']=telemetry() if device=='cuda' else None
            rows.append(record)
            save_json(HERE/'cost_rows_partial.json',dict(rows=rows,source=source_receipt()))
        del model
        if device=='cuda':
            cleanup()
    for block in range(3):
        if device=='cuda':
            measure_item(candidates[0],block,'pre')
        order=chosen if block==0 else chosen[::-1] if block==1 else chosen[2:]+chosen[:2]
        for item in order:
            measure_item(item,block)
        if device=='cuda':
            measure_item(candidates[0],block,'post')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--name',required=True)
    parser.add_argument('--phase',choices=['parity','gpu','cpu'],required=True)
    args=parser.parse_args()
    category='cpu_analysis' if args.phase=='cpu' else 'gpu'
    with Job(args.name,category=category,reserve_seconds=3600 if args.phase=='gpu' else 600,
             metadata={'dataset':'robin','fit_count':0,'phase':args.phase}) as job:
        global np,torch,psutil
        import numpy as np
        import torch
        import psutil
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        candidates=items()
        prepared=prepare()
        with torch.inference_mode():
            if args.phase=='parity':
                parity_all(candidates,prepared,job)
            else:
                parity=json.loads((HERE/'microbatch_parity.json').read_text(encoding='utf-8'))
                if parity['status']!='PASS':
                    raise ValueError('Cost requires parity pass')
                if any(digest(path)!=expected for path,expected in parity['binding'].items()):
                    raise ValueError('Parity-tested code, checkpoint or data contract changed; revalidation required')
                path=HERE/f'cost_{args.phase}_rows.json'
                if path.exists():
                    raise RuntimeError('Cost phase already completed; no silent remeasurement')
                rows=[]
                cost_grid(candidates,prepared,job,'cuda' if args.phase=='gpu' else 'cpu',rows)
                expected=117 if args.phase=='gpu' else 12
                if len(rows)!=expected:
                    raise ValueError(f'Cost rows {len(rows)} != {expected}')
                save_json(path,dict(status='complete',rows=rows,environment=environment_receipt(),source=source_receipt()))
                if args.phase=='gpu':
                    save_json(HERE/'cost_summary_gpu.json',summarize(rows))
                else:
                    # CPU has no sentinels, use the same fit/group aggregation.
                    gpu=json.loads((HERE/'cost_gpu_rows.json').read_text(encoding='utf-8'))['rows']
                    save_json(HERE/'cost_summary.json',summarize(gpu+rows))


if __name__=='__main__':
    main()

