"""Stage A2: same initial FORECAST model, sparse replay vs common FIT256."""
from pathlib import Path
import argparse
import copy
import json
import os
import sys
import time
import traceback

os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import numpy as np
import torch
from diagnose_forecast import (ROOT, OLD, HERE, CAMPAIGNS, Limit, write_new,
                              save_prediction, measure_subset, data_subset)
from common import (access_guard,sha,ahash,read,load_data,windows,masks,physical_score,H,L)
from e5 import make_batches, masked_mse, objective_head, state_hash
from model_neural import TinyEncoder


class A2Limit(Limit):
    def check(self):
        v=self.status()
        if v['cpu_seconds']>=300 or v['wall_seconds']>=600:
            raise RuntimeError('STAGE_A2_RESOURCE_LIMIT_CPU300_WALL600')
        if v['peak_sampled_rss_bytes']>6*1024**3:
            raise RuntimeError('STAGE_A2_RSS_LIMIT_6GIB')

    def status(self):
        return dict(super().status(),cpu_limit_seconds=300,wall_limit_seconds=600)


@torch.inference_mode()
def predict(model,head,histories,limit):
    values=[]
    for start in range(0,len(histories),16):
        x=torch.from_numpy(histories[start:start+16])
        values.append(head(model(x)[:,-1]).reshape(len(x),H,model.channels).numpy())
        limit.check()
    return np.concatenate(values)


def save_checkpoint(path,model,head,step,seed,pool,initial_sha,batches_hash):
    with path.open('xb') as f:
        torch.save({'encoder':model.state_dict(),'head':head.state_dict(),
                    'step':step,'seed':seed,'pool':pool,
                    'initial_checkpoint_sha256':initial_sha,'batch_order_hash':batches_hash},f)
    return {'path':str(path.relative_to(ROOT)),'sha256':sha(path),'update':step}


def train_one(data,seed,pool,batches,initial_record,initial_state,old_training,folder,out,limit):
    started=time.perf_counter()
    torch.manual_seed(seed)
    model=TinyEncoder(data['x'].shape[1],causal=True)
    head=objective_head('CAUSAL_FORECAST',data['x'].shape[1])
    model.load_state_dict(initial_state['encoder']);head.load_state_dict(initial_state['head'])
    assert state_hash(model.state_dict())==state_hash(initial_state['encoder'])
    assert state_hash(head.state_dict())==state_hash(initial_state['head'])
    optimizer=torch.optim.Adam(list(model.parameters())+list(head.parameters()),lr=.001)
    unique=np.unique(batches)
    train_data=data_subset(data,unique)
    train_x=torch.from_numpy(windows(train_data,'subset'))
    train_y=torch.from_numpy(windows(train_data,'subset',True))
    train_m=torch.from_numpy(masks(train_data,'subset'))
    rows=np.searchsorted(unique,batches)
    histories={s:windows(data,s) for s in ('fit','tune','val')}
    curves=[];losses=[];chosen=None;chosen_state=None
    case={'dataset':data['dataset'],'seed':seed,'pool':pool,'unique_origins':unique.tolist(),
          'unique_origin_count':len(unique),'actual_sample_exposures':int(batches.size),
          'batch_size':batches.shape[1],'batch_order_hash':ahash(batches),
          'initial_checkpoint':initial_record,'encoder_initial_state_hash':state_hash(model.state_dict()),
          'head_initial_state_hash':state_hash(head.state_dict()),
          'gradient_finite_checks':0,'checkpoints':[],'curves':curves,'losses':losses,
          'shared_fit_scope':'FIT-only scaler and input fill, H48 and history512; same full-history CPU forward',
          'selection':'TUNE64 physical channel-macro MSE, earliest update tie',
          'objective':'Observed elementwise normalized MSE (same original E5), not channel-macro training loss'}
    with (folder/'batches.npz').open('xb') as f:
        np.savez_compressed(f,origins=batches)
    case['batch_file']={'path':str((folder/'batches.npz').relative_to(ROOT)),'sha256':sha(folder/'batches.npz')}
    previous_loss=None
    for step in range(601):
        if step%100==0:
            model.eval();head.eval()
            p=predict(model,head,histories['tune'],limit)
            tune=physical_score(data,p,'tune')
            with torch.inference_mode():
                first_rows=rows[0]
                first_pred=head(model(train_x[first_rows])[:,-1]).reshape(len(first_rows),H,model.channels)
                first_loss=float(masked_mse(first_pred,train_y[first_rows],train_m[first_rows]))
            cp=save_checkpoint(folder/f'E_u{step:04}.pt',model,head,step,seed,pool,initial_record['sha256'],ahash(batches))
            case['checkpoints'].append(cp)
            curves.append({'update':step,'tune':tune,'fixed_first_batch_objective':first_loss,
                           'preceding_training_batch_loss':previous_loss})
            if chosen is None or tune['mse']<chosen['tune']['mse']:
                chosen={'update':step,'tune':tune}
                chosen_state={'encoder':copy.deepcopy(model.state_dict()),'head':copy.deepcopy(head.state_dict())}
            print(json.dumps({'dataset':data['dataset'],'seed':seed,'pool':pool,'step':step,
                              'tune_mse':tune['mse'],'resource':limit.status()}),flush=True)
        if step==600:
            break
        model.train();head.train();optimizer.zero_grad(set_to_none=True)
        ids=rows[step]
        prediction=head(model(train_x[ids])[:,-1]).reshape(len(ids),H,model.channels)
        loss=masked_mse(prediction,train_y[ids],train_m[ids])
        if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite loss')
        loss.backward()
        if not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in list(model.parameters())+list(head.parameters())):
            raise ValueError('Nonfinite or disconnected gradient')
        case['gradient_finite_checks']+=1
        optimizer.step()
        previous_loss=float(loss.detach())
        losses.append(previous_loss)
        if step%10==0:limit.check()
    final_state={'encoder':copy.deepcopy(model.state_dict()),'head':copy.deepcopy(head.state_dict())}
    case['selected_update']=chosen['update']
    case['evaluations']=[]
    for update in sorted({chosen['update'],600}):
        state=final_state if update==600 else chosen_state
        model.load_state_dict(state['encoder']);head.load_state_dict(state['head'])
        model.eval();head.eval()
        predictions={s:predict(model,head,histories[s],limit) for s in ('fit','val')}
        rec={'update':update,'fit':physical_score(data,predictions['fit'],'fit'),
             'val':physical_score(data,predictions['val'],'val'),
             'tune':next(c['tune'] for c in curves if c['update']==update),
             'prediction':save_prediction(folder,f'VAL_u{update:04}',data,predictions['val'])}
        case['evaluations'].append(rec)
    if pool=='sparse_replay':
        previous_final=old_training['checkpoints'][-1]
        assert sha(previous_final['path'])==previous_final['sha256']
        historical=torch.load(previous_final['path'],map_location='cpu',weights_only=True)
        differences={part:max(float((final_state[part][name]-historical[part][name]).abs().max()) for name in final_state[part])
                     for part in ('encoder','head')}
        case['original_final_comparison']={'checkpoint_sha256':previous_final['sha256'],
                                           'parameter_max_abs_difference':differences,
                                           'original_device':'CPU' if data['dataset']=='jena' else 'CUDA',
                                           'note':'Different CPU threads or original GPU kernels may change floating-point training trajectory.'}
    case['wall_seconds']=time.perf_counter()-started
    case['status']='COMPLETE'
    write_new(out/f"{data['dataset']}_s{seed}_{pool}.json",case)
    return case


def main(tag):
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    limit=A2Limit()
    protected=access_guard()
    out=HERE/('stage_a2_'+tag);out.mkdir(exist_ok=False)
    cache=ROOT/'.cache/forecast_component_study_20261013/stage_a2'/tag;cache.mkdir(parents=True,exist_ok=False)
    r={'status':'RUNNING','device':'CPU','protected_test_paths':protected,'cases':[],'contracts':{},
       'objective':'Isolate sparse versus common FIT256 origin pool with original FORECAST architecture/head',
       'invariants':{'updates':600,'optimizer':'Adam','lr':.001,'H':48,'L':512,'RF':63,'latent_dimension':16,
                     'Jena_batch':8,'Robin_batch':2,'TUNE_selection':'Physical channel-macro MSE',
                     'sampler':'Original make_batches, same seed+31; sparse actual batches must match generated batches'},
       'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),HERE/'diagnose_forecast.py',
                        OLD/'e5.py',OLD/'model_neural.py',OLD/'common.py',HERE/'STAGE_A2_PROTOCOL.md']}}
    try:
        for dataset,campaign in CAMPAIGNS.items():
            data=load_data(dataset)
            r['contracts'][dataset]={'source':str(Path(data['source']).relative_to(ROOT)),
                'source_sha256':data['source_hash'],'mean':data['mean'].tolist(),'std':data['std'].tolist(),
                'origins':{s:v.tolist() for s,v in data['origins'].items()},
                'bounds':{'fit':list(data['fit_bounds']),'tune':list(data['tune_bounds']),'val':list(data['val_bounds'])},
                'new_dense_origin_pool':'Existing common FIT256, overlapping windows are not independent samples'}
            for seed in (92601,92602):
                olddir=OLD/'results'/campaign/f'CAUSAL_FORECAST_s{seed}'
                training=read(olddir/'E_training.json')
                sparse_batches=np.asarray([v['origins'] for v in training['training_exposures']],dtype=np.int64)
                sparse_origins=np.unique(sparse_batches)
                batch=sparse_batches.shape[1]
                assert np.array_equal(make_batches(sparse_origins,batch,600,seed+31),sparse_batches)
                assert ahash(sparse_batches)==training['batch_order_hash']
                initial=training['checkpoints'][0]
                assert initial['update']==0 and sha(initial['path'])==initial['sha256']
                initial_state=torch.load(initial['path'],map_location='cpu',weights_only=True)
                assert initial_state['step']==0 and initial_state['condition']=='CAUSAL_FORECAST' and initial_state['seed']==seed
                dense_batches=make_batches(data['origins']['fit'],batch,600,seed+31)
                for pool,batches in [('sparse_replay',sparse_batches),('dense256',dense_batches)]:
                    assert np.all(batches>=data['fit_bounds'][0]+512)
                    assert np.all(batches+48<=data['fit_bounds'][1])
                    folder=cache/f'{dataset}_s{seed}_{pool}';folder.mkdir()
                    case=train_one(data,seed,pool,batches,initial,initial_state,training,folder,out,limit)
                    case['source_training_record']={'path':str((olddir/'E_training.json').relative_to(ROOT)),
                                                    'sha256':sha(olddir/'E_training.json')}
                    r['cases'].append(case)
                    limit.check()
        r['status']='COMPLETE'
        r['resource']=limit.status()
        write_new(out/'RESULT.json',r)
        print(json.dumps({'status':'COMPLETE','result':str(out/'RESULT.json'),'resource':r['resource']}),flush=True)
    except BaseException as error:
        r['status']='FAILED_PRESERVED';r['error']=str(error);r['traceback']=traceback.format_exc();r['resource']=limit.status()
        write_new(out/'FAILURE.json',r)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);args=parser.parse_args()
    if not args.tag.replace('_','').isalnum():raise ValueError('Unsafe tag')
    main(args.tag)
