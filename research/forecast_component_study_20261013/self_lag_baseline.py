"""Strong channel-wise ridge reference on the same bounded study data."""
from pathlib import Path
import argparse
import json
import os
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'research/past_only_repr_diagnosis_20261012'))
import numpy as np
import psutil
from threadpoolctl import threadpool_limits
from common import (Ridge, LAMBDAS, L, H, access_guard, load_data, windows,
                    masks, physical_score, score, sha, ahash, jsonable)


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(jsonable(value), f, ensure_ascii=False, indent=2, allow_nan=False)


def feature(x):
    return x[:, -(np.arange(33)+1), :]


def fit(data, history, target, observed):
    channels = history['fit'].shape[-1]
    features = {s:feature(x) for s,x in history.items()}
    candidates = []
    for lam in LAMBDAS:
        weights = np.zeros((33,H,channels), np.float64)
        biases = np.zeros((H,channels), np.float64)
        for c in range(channels):
            groups = {}
            for h in range(H):
                groups.setdefault(observed['fit'][:,h,c].tobytes(), []).append(h)
            for hs in groups.values():
                rows = observed['fit'][:,hs[0],c]
                if rows.sum() < 2:
                    raise ValueError('Insufficient observed FIT rows')
                r = Ridge(lam).fit(features['fit'][rows,:,c],target['fit'][rows,:,c][:,hs])
                coef = r.coef if r.coef is not None else r.train.T@r.dual
                effective = coef/r.std[:,None]
                weights[:,hs,c] = effective
                biases[hs,c] = r.ym-r.mean@effective
        tune = np.einsum('nlc,lhc->nhc',features['tune'],weights)+biases
        value = physical_score(data,tune,'tune')
        candidates.append((value,lam,weights,biases))
    chosen = min(range(len(candidates)), key=lambda i:candidates[i][0]['mse'])
    metric,lam,weights,biases = candidates[chosen]
    pred = {s:np.einsum('nlc,lhc->nhc',f,weights)+biases for s,f in features.items()}
    return pred,weights,biases,{'selected_lambda':lam,'selected_index':chosen,
        'candidates':[{'lambda':v[1],'tune':v[0]} for v in candidates],
        'criterion':'TUNE physical channel macro MSE; earliest tie',
        'feature_width_per_channel':33,'channel_independent':True}


def preprocess(physical, observed, mean, std):
    last = np.broadcast_to(mean,(len(physical),len(mean))).copy()
    filled = np.empty_like(physical)
    for t in range(L):
        last = np.where(observed[:,t],physical[:,t],last)
        filled[:,t] = last
    return (filled-mean)/std


def benchmark(data, weights, biases, predictions):
    result=[]
    for batch in (1,8):
        oo=data['origins']['val'][:batch]
        raw=np.stack([data['physical'][o-L:o] for o in oo])
        valid=np.stack([data['mask'][o-L:o] for o in oo])
        def infer():
            x=preprocess(raw,valid,data['mean'],data['std'])
            y=np.einsum('nlc,lhc->nhc',feature(x),weights)+biases
            return y*data['std']+data['mean']
        for _ in range(2): infer()
        timing=[]
        for _ in range(7):
            start=time.perf_counter(); output=infer(); timing.append(time.perf_counter()-start)
        expected=predictions[:batch]*data['std']+data['mean']
        delta=float(np.max(np.abs(output-expected)))
        if not np.allclose(output,expected,rtol=1e-6,atol=1e-4):
            raise AssertionError('Timing/science pipeline mismatch')
        result.append({'batch':batch,'seconds':timing,'median_seconds':float(np.median(timing)),
            'median_seconds_per_window':float(np.median(timing))/batch,'warmup':2,'repeats':7,
            'device':'CPU, 2 BLAS threads; no GPU','timed_vs_saved_max_abs_physical':delta,
            'includes':'CPU full512 window forward-fill, FIT normalization, last33 self features, ridge, inverse normalization; excludes disk/model loading',
            'comparability':'CPU deployment reference, distinct from GPU component pipelines; no matched-GPU speedup claim'})
    return result


def run(tag):
    start=time.perf_counter();process=psutil.Process();cpu_start=sum(process.cpu_times()[:2])
    protected=access_guard()
    output=HERE/f'self_lag_{tag}';output.mkdir(exist_ok=False)
    cache=ROOT/f'.cache/forecast_component_study_20261013/self_lag_{tag}'
    cache.mkdir(parents=True,exist_ok=False)
    cases=[]
    for name in ('jena','robin'):
        data=load_data(name,sizes=(256,64,None))
        history={s:windows(data,s) for s in data['origins']}
        target={s:windows(data,s,True) for s in history}
        observed={s:masks(data,s) for s in history}
        train_start=time.perf_counter()
        pred,weights,biases,selection=fit(data,history,target,observed)
        train_seconds=time.perf_counter()-train_start
        prediction_path=cache/f'{name}_RAW_SELF_LAG.npz'
        with prediction_path.open('xb') as f:
            np.savez_compressed(f,**{s+'_prediction':p for s,p in pred.items()},
                val_target=target['val'],val_observed=observed['val'],
                **{s+'_origins':o for s,o in data['origins'].items()},
                mean=data['mean'],std=data['std'],weights=weights,biases=biases)
        method={'method':'RAW_SELF_LAG','scores':{s:physical_score(data,p,s) for s,p in pred.items()},
            'prediction':{'path':str(prediction_path.relative_to(ROOT)),'sha256':sha(prediction_path)},
            'selection':selection,'train_wall_seconds':train_seconds,
            'latency':benchmark(data,weights,biases,pred['val'])}
        case={'dataset':name,'label':'RAW_SELF_LAG','seed':None,'methods':[method],
            'source_path':str(Path(data['source']).relative_to(ROOT)),'source_sha256':data['source_hash'],
            'origin_hashes':{s:ahash(o) for s,o in data['origins'].items()},
            'bounds':{s:data[s+'_bounds'] for s in ('fit','tune','val')},
            'scope':'Additional simple reference; self-channel33 differs from full512 matched component comparisons'}
        write_new(output/f'{name}.json',case);cases.append(case)
        print(json.dumps({'dataset':name,'mse':method['scores']['val']['mse'],'mae':method['scores']['val']['mae']}),flush=True)
    summary={'status':'COMPLETE','cases':cases,'protected_test_guarded':protected,
        'source_sha256':sha(Path(__file__)),'resources':{'wall_seconds':time.perf_counter()-start,
        'cpu_seconds':sum(process.cpu_times()[:2])-cpu_start,'gpu_seconds':0,
        'rss_bytes':process.memory_info().rss}}
    write_new(output/'RESULT.json',summary)
    print(json.dumps(summary['resources']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=2): run(args.tag)
