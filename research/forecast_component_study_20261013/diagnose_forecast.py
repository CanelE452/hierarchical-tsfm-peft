"""Stage A: diagnose saved FORECAST checkpoints without neural retraining."""
from pathlib import Path
import argparse
import csv
import json
import os
import sys
import time
import traceback

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'research/past_only_repr_diagnosis_20261012'
sys.path.insert(0, str(OLD))
import numpy as np
import psutil
import torch
from common import access_guard, ahash, sha, read, load_data, windows, masks, physical_score, ridge_grouped, H, L
from model_neural import TinyEncoder
from e5 import objective_head, training_origins

HERE = Path(__file__).resolve().parent
CACHE = ROOT / '.cache/forecast_component_study_20261013/stage_a'
CAMPAIGNS = {'jena': 'E5_jena_balanced600_cpu_v1', 'robin': 'E5_robin_robin_gpu_balanced_v2'}


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


class Limit:
    def __init__(self):
        self.started = time.perf_counter()
        self.process = psutil.Process()
        self.start_cpu = sum(self.process.cpu_times()[:2])
        self.peak_rss = 0

    def status(self):
        self.peak_rss = max(self.peak_rss, self.process.memory_info().rss)
        return {'wall_seconds': time.perf_counter()-self.started,
                'cpu_seconds': sum(self.process.cpu_times()[:2])-self.start_cpu,
                'peak_sampled_rss_bytes': self.peak_rss,
                'gpu_seconds': 0, 'torch_threads': torch.get_num_threads(),
                'cpu_limit_seconds': 900, 'wall_limit_seconds': 1800}

    def check(self):
        v = self.status()
        if v['cpu_seconds'] >= 900 or v['wall_seconds'] >= 1800:
            raise RuntimeError('STAGE_A_RESOURCE_LIMIT')
        if v['peak_sampled_rss_bytes'] > 6*1024**3:
            raise RuntimeError('STAGE_A_RSS_LIMIT_6GIB')


def data_subset(data, origins):
    return dict(data, origins={'subset': np.asarray(origins, dtype=np.int64)})


def measure_subset(data, prediction, origins):
    return physical_score(data_subset(data, origins), prediction, 'subset')


@torch.inference_mode()
def encode(model, histories, limit, width=95):
    rows = []
    for start in range(0, len(histories), 64):
        z = model(torch.from_numpy(histories[start:start+64, -width:].copy()))
        rows.append(z[:, -33:].numpy())
        limit.check()
    return np.concatenate(rows)


@torch.inference_mode()
def head_predict(head, latent, channels):
    return head(torch.from_numpy(latent[:, -1].copy())).numpy().reshape(len(latent), H, channels)


@torch.inference_mode()
def causal_checks(model, histories):
    original = torch.from_numpy(histories[:2].copy())
    full = model(original)[:, -33:]
    crop = model(original[:, -95:])[:, -33:]
    altered = original.clone()
    altered[:, :-95] = altered[:, :-95].flip(1) + 777
    earlier_delta = float((full-model(altered)[:, -33:]).abs().max())
    crop_delta = float((full-crop).abs().max())
    final63 = model(original[:, -63:])[:, -1]
    last63_delta = float((full[:, -1]-final63).abs().max())
    boundary_input = original.clone()
    boundary_input[:, -1] += 1
    positive_delta = float((full-model(boundary_input)[:, -33:]).abs().max())
    if max(earlier_delta, crop_delta, last63_delta) > 2e-5 or positive_delta <= 0:
        raise AssertionError('Receptive-field counterfactual failed')
    return {'full512_vs_tail95_last33_max_abs': crop_delta,
            'change_before95_last33_max_abs': earlier_delta,
            'full512_vs_tail63_last1_max_abs': last63_delta,
            'change_latest_observation_positive_control_max_abs': positive_delta,
            'pass': True, 'tolerance': 2e-5}


def save_prediction(folder, name, data, prediction, extra=None):
    path = folder / (name+'.npz')
    payload = dict(prediction=prediction, target=windows(data, 'val', True),
                   observed=masks(data, 'val'), origins=data['origins']['val'])
    if extra:
        payload.update(extra)
    with path.open('xb') as f:
        np.savez_compressed(f, **payload)
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path)}


def metric_row(dataset, seed, update, method, scope, metric, tune=None):
    return {'dataset': dataset, 'seed': seed, 'update': update, 'method': method,
            'fit_scope': scope, 'val_mse': metric['mse'], 'val_mae': metric['mae'],
            'val_normalized_mse': metric['normalized']['mse'],
            'val_normalized_mae': metric['normalized']['mae'],
            'tune_mse': None if tune is None else tune['mse']}


def fit_ridge(data, features, limit, fit_split='fit'):
    split_data = dict(data, origins=dict(data['origins'], fit=data['origins'][fit_split]))
    result, selection, fitted = ridge_grouped(
        features[fit_split], windows(split_data, 'fit', True), masks(split_data, 'fit'),
        features['tune'], windows(data, 'tune', True), masks(data, 'tune'), features['val'],
        score_scale=data['std'], budget=limit)
    tune = np.zeros((len(features['tune']), H*data['x'].shape[1]))
    fit = np.zeros((len(features[fit_split]), H*data['x'].shape[1]))
    for columns, model in fitted:
        tune[:, columns] = model.predict(features['tune'])
        fit[:, columns] = model.predict(features[fit_split])
    return result, selection, measure_subset(data, tune.reshape(-1,H,data['x'].shape[1]), data['origins']['tune']), measure_subset(data,fit.reshape(-1,H,data['x'].shape[1]),data['origins'][fit_split])


def run(tag):
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    limit = Limit()
    protected = access_guard()
    folder = CACHE / tag
    folder.mkdir(parents=True, exist_ok=False)
    out = HERE / ('stage_a_'+tag)
    out.mkdir(exist_ok=False)
    result = {'status': 'RUNNING', 'purpose': 'Locate saved FORECAST failure; no neural optimizer updates',
              'protected_test_paths_guarded': protected,
              'new_neural_updates': 0, 'device': 'cpu', 'readout_fit': 'FIT only; no FIT+TUNE refit',
              'checkpoint_selection': 'Lowest TUNE physical channel-macro MSE of original head; earliest update tie',
              'ridge_selection': 'TUNE physical channel-macro MSE; original five-lambda order',
              'evaluation_scope': 'Repeated VAL development diagnosis, not independent generalization',
              'sources': {str(p.relative_to(ROOT)): sha(p) for p in
                          [Path(__file__), OLD/'common.py', OLD/'e5.py', OLD/'model_neural.py']},
              'cases': [], 'raw_baselines': [], 'rows': []}
    try:
        for dataset, campaign in CAMPAIGNS.items():
            data = load_data(dataset)
            original_origins = training_origins(data['fit_bounds'])
            data['origins']['original_train'] = original_origins
            histories = {s: windows(data, s) for s in data['origins']}
            channels = data['x'].shape[1]
            contract = {'source': str(Path(data['source']).relative_to(ROOT)), 'source_sha256': data['source_hash'],
                        'contract_path': f'research/level_matched_learned_ed_v18_20261006/data_contract_{dataset}.json',
                        'fit_bounds': list(data['fit_bounds']), 'tune_bounds': list(data['tune_bounds']),
                        'val_bounds': list(data['val_bounds']), 'mean': data['mean'].tolist(), 'std': data['std'].tolist(),
                        'normalization_scope': data['normalization_fit_scope'], 'H': H, 'L': L,
                        'origins': {s: oo.tolist() for s,oo in data['origins'].items()},
                        'origin_hashes': {s: ahash(oo) for s,oo in data['origins'].items()},
                        'source_contract_sha256': sha(ROOT/f'research/level_matched_learned_ed_v18_20261006/data_contract_{dataset}.json'),
                        'readout_tune_history_note': 'Input may extend into FIT; every target is inside TUNE, no future input.',
                        'input_fill_note': 'Existing windows() resets missing history at L512 boundary using FIT mean. RF63/95 denotes transformed input influence; forward fill may carry earlier observed values.'}
            write_new(out/(dataset+'_contract.json'), contract)
            for width in (63,95):
                method = f'RAW_JOINT_RIDGE_RF{width}'
                start = time.perf_counter()
                features = {s: x[:, -1-np.arange(width)].reshape(len(x),-1) for s,x in histories.items()}
                pred, chosen, tune, fitted = fit_ridge(data, features, limit)
                metric = physical_score(data,pred,'val')
                record = {'dataset':dataset,'method':method,'fit_scope':'COMMON_FIT_256',
                          'val':metric,'tune':tune,'fit':fitted,'selection':chosen,
                          'wall_seconds':time.perf_counter()-start,
                          'prediction':save_prediction(folder,dataset+'_'+method,data,pred)}
                result['raw_baselines'].append(record)
                result['rows'].append(metric_row(dataset,'NA','NA',method,'COMMON_FIT_256',metric,tune))
                print(json.dumps({'dataset':dataset,'completed':method,'val_mse':metric['mse']}),flush=True)
            pred=np.repeat(histories['val'][:,-1:, :],H,axis=1)
            metric=physical_score(data,pred,'val')
            result['raw_baselines'].append({'dataset':dataset,'method':'PERSISTENCE','val':metric,
                'prediction':save_prediction(folder,dataset+'_PERSISTENCE',data,pred)})
            result['rows'].append(metric_row(dataset,'NA','NA','PERSISTENCE','NONE',metric))
            for seed in (92601,92602):
                case_start=time.perf_counter()
                old_dir=OLD/'results'/campaign/f'CAUSAL_FORECAST_s{seed}'
                training=read(old_dir/'E_training.json')
                old_result=read(old_dir/'RESULT.json')
                actual_origins=sorted({o for ex in training['training_exposures'] for o in ex['origins']})
                if actual_origins != original_origins.tolist():
                    raise AssertionError('Observed training origins differ from reconstructed contract')
                case={'dataset':dataset,'seed':seed,'source_training_record':str((old_dir/'E_training.json').relative_to(ROOT)),
                      'source_training_sha256':sha(old_dir/'E_training.json'),
                      'unique_training_origins':actual_origins,'unique_training_origin_count':len(actual_origins),
                      'sample_exposures':training['actual_sample_exposures'],
                      'actual_future_target_timesteps':training['actual_future_target_timesteps'],
                      'checkpoints':[],'comparisons':[],'effective_history':{'original_head':63,'last_z_ridge':63,'last33_z_ridge':95},
                      'old_common_ridge':{'mode':'REUSED_RECORDED_RESULT', 'path':str((old_dir/'RESULT.json').relative_to(ROOT)),
                         'sha256':sha(old_dir/'RESULT.json'),'val':old_result['val'],'tune':old_result['tune'],
                         'prediction_path':old_result['prediction_path'],'prediction_sha256':old_result['prediction_sha256']}}
                model=TinyEncoder(channels,causal=True).eval()
                head=objective_head('CAUSAL_FORECAST',channels).eval()
                saved={}
                for cp in training['checkpoints']:
                    path=Path(cp['path'])
                    if sha(path)!=cp['sha256']:
                        raise AssertionError('Checkpoint SHA mismatch')
                    state=torch.load(path,map_location='cpu',weights_only=True)
                    if state['step']!=cp['update'] or state['condition']!='CAUSAL_FORECAST' or state['seed']!=seed:
                        raise AssertionError('Checkpoint metadata mismatch')
                    model.load_state_dict(state['encoder']);head.load_state_dict(state['head'])
                    before=time.perf_counter()
                    latents={s:encode(model,histories[s],limit) for s in ('fit','tune','original_train')}
                    metrics={s:measure_subset(data,head_predict(head,z,channels),data['origins'][s]) for s,z in latents.items()}
                    record={'update':cp['update'],'checkpoint_path':str(path.relative_to(ROOT)),
                            'checkpoint_sha256':cp['sha256'],'metrics':metrics,
                            'wall_seconds':time.perf_counter()-before}
                    case['checkpoints'].append(record)
                    saved[cp['update']]={'latents':latents,'state':state}
                selected=min(case['checkpoints'],key=lambda v:(v['metrics']['tune']['mse'],v['update']))['update']
                case['selected_update']=selected
                for update in sorted({selected,600}):
                    state=saved[update]['state']
                    model.load_state_dict(state['encoder']);head.load_state_dict(state['head'])
                    checks=causal_checks(model,histories['fit'])
                    latent=saved[update]['latents']
                    latent['val']=encode(model,histories['val'],limit)
                    pred=head_predict(head,latent['val'],channels)
                    metric=physical_score(data,pred,'val')
                    record={'update':update,'method':'ORIGINAL_FORECAST_HEAD','fit_scope':'ORIGINAL_SPARSE_ORIGINS',
                            'val':metric,'tune':next(v['metrics']['tune'] for v in case['checkpoints'] if v['update']==update),
                            'counterfactual_checks':checks,
                            'prediction':save_prediction(folder,f'{dataset}_s{seed}_u{update}_HEAD',data,pred)}
                    case['comparisons'].append(record)
                    result['rows'].append(metric_row(dataset,seed,update,record['method'],record['fit_scope'],metric,record['tune']))
                    for nlat,fit_split in ((1,'original_train'),(1,'fit'),(33,'fit')):
                        start=time.perf_counter()
                        features={s:z[:,-1-np.arange(nlat)].reshape(len(z),-1) for s,z in latent.items()}
                        pred,selection,tune,fitted=fit_ridge(data,features,limit,fit_split)
                        metric=physical_score(data,pred,'val')
                        method=f'LAST{nlat}_Z_RIDGE'
                        scope='ORIGINAL_SPARSE_ORIGINS' if fit_split=='original_train' else 'COMMON_FIT_256'
                        rec={'update':update,'method':method,'fit_scope':scope,'val':metric,'tune':tune,'fit':fitted,
                             'selection':selection,'wall_seconds':time.perf_counter()-start,
                             'prediction':save_prediction(folder,f'{dataset}_s{seed}_u{update}_{method}_{fit_split}',data,pred)}
                        if update==600 and nlat==33:
                            if sha(old_result['prediction_path'])!=old_result['prediction_sha256']:
                                raise AssertionError('Old prediction SHA mismatch')
                            with np.load(old_result['prediction_path'],allow_pickle=False) as archive:
                                difference=float(np.max(np.abs(pred-archive['prediction'])))
                            rec['old_prediction_reproduction']={'max_abs_normalized_prediction':difference,
                                'physical_mse_absolute_difference':abs(metric['mse']-old_result['val']['mse']),
                                'same_lambda':selection['lambda']==old_result['direct_ridge']['lambda'],
                                'note':'CPU cropped encoder vs original full encoder may have floating-point differences'}
                        case['comparisons'].append(rec)
                        result['rows'].append(metric_row(dataset,seed,update,method,scope,metric,tune))
                    limit.check()
                case['wall_seconds']=time.perf_counter()-case_start
                result['cases'].append(case)
                write_new(out/f'{dataset}_s{seed}.json',case)
                print(json.dumps({'dataset':dataset,'seed':seed,'selected_update':selected,
                                  'case_wall_seconds':case['wall_seconds']}),flush=True)
        result['status']='COMPLETE'
        result['resource']=limit.status()
        write_new(out/'RESULT.json',result)
        with (out/'metrics.csv').open('x',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(result['rows'][0]));writer.writeheader();writer.writerows(result['rows'])
        print(json.dumps({'status':result['status'],'output':str(out),'resource':result['resource']}),flush=True)
    except BaseException as error:
        result['status']='FAILED_PRESERVED'
        result['error']=str(error)
        result['traceback']=traceback.format_exc()
        result['resource']=limit.status()
        write_new(out/'FAILURE.json',result)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--tag',required=True)
    args=p.parse_args()
    if not args.tag.replace('_','').isalnum():
        raise ValueError('Only alphanumeric tag with underscores')
    run(args.tag)
