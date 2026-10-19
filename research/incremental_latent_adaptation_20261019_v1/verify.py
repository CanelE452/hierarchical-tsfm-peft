"""Independent FP64 verification of sealed predictions and public experiment evidence."""
import argparse
import csv
import gzip
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from . import common


ARRAY_KEYS = ('pred_normalized','target_normalized','observed','origins','channel_ids')
STAT_FIELDS = ('run_id','dataset','method','seed','prefix','split','origin','horizon',
               'channel_index','channel_id','sse','sae','count')
REQUIRED = ('README_KO.md','EXPERIMENT_PLAN.json','PRIOR_ART_AND_DELTA.md',
    'PARENT_AND_DATA_CONTRACT.json','EXECUTION_STATUS.csv','RESOURCE_LEDGER.json',
    'RESUME_QUEUE.md','RESIDUAL_PROBES.csv','RUNS.csv','COST.csv','COST_TRIALS.csv',
    'CLAIM_STATUS.md','RESULT_ANALYSIS_KO.md','FIGURE_POINTS.csv','FIGURE_MANIFEST.json','REPRODUCE.md')


def _na(value):
    return value is None or str(value).strip().lower() in ('','na','n/a','none','null','nan')


def _value(value):
    return 'NA' if _na(value) else value


def _identity(value):
    array = np.ascontiguousarray(value)
    return dict(shape=list(array.shape),dtype=str(array.dtype),sha256=hashlib.sha256(array.tobytes()).hexdigest())


def _close(left,right):
    return math.isclose(float(left),float(right),rel_tol=1e-10,abs_tol=1e-12)


def _csv(path):
    with Path(path).open(newline='',encoding='utf8') as stream:
        return list(csv.DictReader(stream))


def _receipt(path,value):
    """Keep canonical bytes stable and preserve superseded validation attempts."""
    path = Path(path)
    if path.is_file():
        previous = common.read(path)
        if previous==value:
            return
        number = len(list(path.parent.glob(path.stem+'_ATTEMPT_*.json')))+1
        common.atomic(path.with_name(path.stem+f'_ATTEMPT_{number:03}.json'),previous)
    common.atomic(path,value)


def _metadata(entry,key):
    aliases = dict(scaler=('scaler_sha256','scaler_sha'),
        g=('parent_G_sha256','G_parent_sha256','parent_G'),
        s=('parent_S_sha256','S_parent_sha256','parent_S','initial_parent_S'),
        data=('data_sha256','data_sha'))
    values = entry['metadata']
    for alias in aliases.get(key,(key,)):
        if not _na(values.get(alias)):
            return values[alias]
    return None


def _load(entry):
    path = Path(entry['prediction_path'])
    if not entry.get('prediction_sha256') or common.sha(path) != entry['prediction_sha256']:
        raise ValueError('Prediction file SHA256 differs: '+entry['run_id'])
    with np.load(path,allow_pickle=False) as archive:
        arrays = {key:archive[key].copy() for key in ARRAY_KEYS}
        metadata = json.loads(str(archive['metadata_json'].item()))
    if metadata != entry.get('metadata'):
        raise ValueError('Prediction metadata binding differs: '+entry['run_id'])
    if metadata.get('split') not in ('V','V_select') or metadata.get('T_access',False):
        raise ValueError('Final predictions must be the registered V cohort without T access')
    for key,value in arrays.items():
        if entry.get('arrays',{}).get(key) != _identity(value):
            raise ValueError('Prediction array SHA256/axes/dtype binding differs: '+entry['run_id']+' '+key)
    p,y,m,o,c = (arrays[key] for key in ARRAY_KEYS)
    if (p.ndim != 3 or p.shape != y.shape or p.shape != m.shape or m.dtype != np.bool_
            or o.shape != (len(p),) or o.dtype.kind not in 'iu' or np.any(np.diff(o)<=0)
            or c.shape != (p.shape[-1],) or len(np.unique(c)) != len(c)):
        raise ValueError('Invalid aligned original-channel prediction/target/bool-mask/origin/channel axes')
    if not np.isfinite(p).all() or not np.isfinite(y[m]).all():
        raise ValueError('Nonfinite prediction or observed target')
    difference = np.zeros(p.shape,np.float64)
    np.subtract(p.astype(np.float64),y.astype(np.float64),out=difference,where=m)
    sse,sae,count = np.square(difference),np.abs(difference),m.astype(np.int64)
    totals = count.sum((0,1),dtype=np.int64)
    active = np.flatnonzero(totals)
    if not len(active):
        raise ValueError('No observed final target')
    mse = math.fsum(float(sse[:,:,i].sum(dtype=np.float64))/int(totals[i]) for i in active)/len(active)
    mae = math.fsum(float(sae[:,:,i].sum(dtype=np.float64))/int(totals[i]) for i in active)/len(active)
    score = dict(mse=mse,mae=mae,n_observed=int(totals.sum()),n_origins=len(p))
    for key,value in score.items():
        if key not in entry.get('score',{}) or not _close(value,entry['score'][key]):
            raise ValueError('Independent FP64 score differs: '+entry['run_id']+' '+key)
    return dict(entry=entry,arrays=arrays,sse=sse,sae=sae,count=count,score=score)


def _same_cohort(left,right):
    a,b = left['arrays'],right['arrays']
    for key in ('observed','origins','channel_ids'):
        if not np.array_equal(a[key],b[key]):
            raise ValueError('Paired '+key+' differs: '+right['entry']['run_id'])
    if not np.array_equal(a['target_normalized'][a['observed']],b['target_normalized'][b['observed']]):
        raise ValueError('Paired observed targets differ: '+right['entry']['run_id'])
    for key in ('scaler','g','data'):
        av,bv = _metadata(left['entry'],key),_metadata(right['entry'],key)
        if av is None or bv is None or av != bv:
            raise ValueError('Paired '+key+' parent/data binding missing or different: '+right['entry']['run_id'])


def _statistics_file(path,loaded):
    temporary = Path(str(path)+'.tmp')
    count = 0
    with temporary.open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=6) as compressed:
            with io.TextIOWrapper(compressed,encoding='utf8',newline='') as stream:
                writer = csv.DictWriter(stream,fieldnames=STAT_FIELDS)
                writer.writeheader()
                for item in loaded:
                    entry,arrays = item['entry'],item['arrays']
                    shared = dict(run_id=entry['run_id'],dataset=entry['dataset'],method=entry['method'],
                        seed=_value(entry.get('seed')),prefix=_value(entry.get('prefix')),split='V_select')
                    for i,origin in enumerate(arrays['origins']):
                        for h in range(item['count'].shape[1]):
                            for c,channel in enumerate(arrays['channel_ids']):
                                writer.writerow(dict(shared,origin=int(origin),horizon=h,channel_index=c,
                                    channel_id=str(channel),sse=format(float(item['sse'][i,h,c]),'.17g'),
                                    sae=format(float(item['sae'][i,h,c]),'.17g'),count=int(item['count'][i,h,c])))
                                count += 1
    temporary.replace(path)
    return count


def _replay_statistics(path,loaded):
    expected = {item['entry']['run_id']:item['score'] for item in loaded}
    channels,seen,errors = {},set(),[]

    def finish(run_id):
        if run_id is None:
            return
        values = list(channels.values())
        active = [value for value in values if value['count']]
        if not active:
            errors.append('No observed sufficient statistics: '+run_id)
            return
        mse = math.fsum(math.fsum(v['sse'])/v['count'] for v in active)/len(active)
        mae = math.fsum(math.fsum(v['sae'])/v['count'] for v in active)/len(active)
        reference = expected[run_id]
        if (not _close(mse,reference['mse']) or not _close(mae,reference['mae'])
                or sum(v['count'] for v in values)!=reference['n_observed']):
            errors.append('Public FP64 sufficient-statistic replay differs: '+run_id)
        seen.add(run_id)

    current,rows = None,0
    with gzip.open(path,'rt',encoding='utf8',newline='') as stream:
        for row in csv.DictReader(stream):
            run_id = row['run_id']
            if run_id != current:
                finish(current)
                channels = {}
                current = run_id
            channel = channels.setdefault(int(row['channel_index']),dict(sse=[],sae=[],count=0))
            channel['sse'].append(float(row['sse']))
            channel['sae'].append(float(row['sae']))
            channel['count'] += int(row['count'])
            rows += 1
    finish(current)
    if seen!=set(expected):
        errors.append('Public sufficient statistics omit final runs')
    return dict(status='FAIL' if errors else 'PASS',errors=errors,run_count=len(seen),cell_count=rows,
                accumulation='Independent gzip/CSV readback with math.fsum per channel')


def _pairs(loaded):
    priority = {name:i for i,name in enumerate(('G','ZERO','ALPHA_fitted','MEAN','OWN','ALL',
        'S','RAW','READOUT_ONLY','RESIDUAL_Q','RESIDUAL_RAW_MLP','RESIDUAL_F','JOINT_FULL','JOINT_PREFIX','SEQ_BLOCKS'))}
    ordered = sorted(loaded,key=lambda item:(item['entry']['dataset'],priority.get(item['entry']['method'],99),item['entry']['run_id']))
    rows = []
    for left,right in itertools.combinations(ordered,2):
        a,b = left['entry'],right['entry']
        if a['dataset'] != b['dataset']:
            continue
        if not _na(a.get('seed')) and not _na(b.get('seed')) and str(a['seed']) != str(b['seed']):
            continue
        if a['method'] == b['method'] and a.get('prefix') == b.get('prefix'):
            continue
        _same_cohort(left,right)
        if not _na(a.get('seed')) and not _na(b.get('seed')):
            left_s,right_s = _metadata(a,'s'),_metadata(b,'s')
            if left_s is not None and right_s is not None and left_s!=right_s:
                raise ValueError('Paired S parent differs within the same seed')
        rmse,cmse = left['score']['mse'],right['score']['mse']
        seed = b.get('seed') if _na(a.get('seed')) else a.get('seed')
        rows.append(dict(dataset=a['dataset'],reference_run_id=a['run_id'],candidate_run_id=b['run_id'],
            reference_method=a['method'],candidate_method=b['method'],seed=_value(seed),aggregation='by_seed' if not _na(seed) else 'shared_parent',
            reference_seed=_value(a.get('seed')),candidate_seed=_value(b.get('seed')),
            reference_prefix=_value(a.get('prefix')),candidate_prefix=_value(b.get('prefix')),
            reference_MSE=rmse,candidate_MSE=cmse,delta_MSE=cmse-rmse,
            relative_MSE_gain_percent=100*(rmse-cmse)/rmse if rmse else 'NA',
            reference_MAE=left['score']['mae'],candidate_MAE=right['score']['mae'],
            delta_MAE=right['score']['mae']-left['score']['mae'],
            reference_alpha=_value(a.get('alpha',a['metadata'].get('alpha'))),
            candidate_alpha=_value(b.get('alpha',b['metadata'].get('alpha'))),
            reference_selected_epoch=_value(a.get('selected_epoch',a['metadata'].get('selected_epoch'))),
            candidate_selected_epoch=_value(b.get('selected_epoch',b['metadata'].get('selected_epoch'))),
            reference_parent_S_sha256=_value(_metadata(a,'s')),candidate_parent_S_sha256=_value(_metadata(b,'s')),
            parent_G_sha256=_metadata(a,'g'),scaler_sha256=_metadata(a,'scaler'),
            n_observed=left['score']['n_observed'],n_seed_pairs=1,
            interpretation='candidate minus reference; same observed targets and fixed parent G; no confidence interval'))
    grouped = {}
    for row in rows:
        if row['aggregation'] == 'by_seed':
            key = tuple(row[k] for k in ('dataset','reference_method','candidate_method','reference_prefix','candidate_prefix'))
            grouped.setdefault(key,[]).append(row)
    for group in grouped.values():
        if len(group) != 2 or {str(r['seed']) for r in group} != {'92601','92602'}:
            continue
        row = dict(group[0],seed='mean',aggregation='two_seed_mean',reference_run_id='NA; see by-seed',
            candidate_run_id='NA; see by-seed',reference_seed='NA; see by-seed',candidate_seed='NA; see by-seed',n_seed_pairs=2)
        for key in ('reference_MSE','candidate_MSE','delta_MSE','reference_MAE','candidate_MAE','delta_MAE'):
            row[key] = math.fsum(float(r[key]) for r in group)/2
        row['relative_MSE_gain_percent'] = 100*(row['reference_MSE']-row['candidate_MSE'])/row['reference_MSE'] if row['reference_MSE'] else 'NA'
        for key in ('reference_alpha','candidate_alpha','reference_selected_epoch','candidate_selected_epoch',
                    'reference_parent_S_sha256','candidate_parent_S_sha256'):
            row[key] = 'NA; see by-seed'
        rows.append(row)
    return rows


def _cost_key(row,use_id):
    if use_id:
        return row.get('cost_id')
    return tuple(str(_value(row.get(key))) for key in ('dataset','method','seed','prefix','batch_size','device'))


def verify_cost_tables(outdir):
    outdir = Path(outdir)
    errors = []
    if not (outdir/'COST.csv').is_file() or not (outdir/'COST_TRIALS.csv').is_file():
        return dict(status='MISSING',errors=['COST.csv/COST_TRIALS.csv missing'],rows=0)
    summaries,trials = _csv(outdir/'COST.csv'),_csv(outdir/'COST_TRIALS.csv')
    use_id = bool(summaries and trials and 'cost_id' in summaries[0] and 'cost_id' in trials[0])
    grouped = {}
    for trial in trials:
        grouped.setdefault(_cost_key(trial,use_id),[]).append(trial)
    seen = set()
    for summary in summaries:
        key = _cost_key(summary,use_id)
        if key in seen:
            errors.append('Duplicate cost summary key: '+str(key))
        seen.add(key)
        rows = grouped.get(key,[])
        timed = [r for r in rows if r.get('phase') == 'timed']
        warm = [r for r in rows if r.get('phase') == 'warmup']
        try:
            if (len(timed)!=20 or len(warm)!=10 or {int(r['trial']) for r in timed}!=set(range(20))
                    or {int(r['trial']) for r in warm}!=set(range(10))):
                raise ValueError('Expected ten distinct warmups and twenty distinct measured trials')
            batch = int(summary.get('batch_size',summary.get('B',0)))
            if batch not in (1,8) or any(int(r.get('batch_size',batch))!=batch for r in rows):
                raise ValueError('Cost batch identity differs')
            values = np.array([float(r['whole_seconds'])*1000 for r in timed],np.float64)
            if not np.isfinite(values).all() or values.min()<=0:
                raise ValueError('Positive finite whole-path times required')
            q25,median,q75 = np.quantile(values,[.25,.5,.75])
            expected = dict(median_batch_ms=median,IQR_batch_ms=q75-q25,
                median_origin_ms=median/batch,IQR_origin_ms=(q75-q25)/batch,
                median_ms_per_batch=median,median_ms_per_origin=median/batch)
            if not any(field in summary for field in ('median_batch_ms','median_ms_per_batch')):
                raise ValueError('Cost summary lacks a whole-path median')
            if 'IQR_batch_ms' not in summary:
                raise ValueError('Cost summary lacks a whole-path IQR')
            for field,value in expected.items():
                if field in summary and not _na(summary[field]) and not _close(summary[field],value):
                    raise ValueError('Independent cost median/IQR differs: '+field)
            for row in rows:
                if 'cached_forecast_or_features_reads' in row and int(row['cached_forecast_or_features_reads'])!=0:
                    raise ValueError('Cached forecast/features in measured deployment path')
                if 'F_instances_created' in row and int(row['F_instances_created'])>1:
                    raise ValueError('More than one F instance in cost trial')
                if 'F_physical_rows' in row and 'F_logical_rows' in row and int(row['F_physical_rows'])<int(row['F_logical_rows']):
                    raise ValueError('Physical F rows cannot be less than logical forward rows')
                if row.get('method') in ('G','ZERO','MEAN','OWN','ALL','RESIDUAL_Q','RESIDUAL_RAW_MLP') and int(row.get('F_logical_rows',0))!=0:
                    raise ValueError('No-F deployment method unexpectedly called F')
            if summary.get('parity_status','PASS')!='PASS':
                raise ValueError('Whole V deployment parity failed')
        except (ValueError,KeyError,TypeError) as error:
            errors.append(str(key)+': '+str(error))
    for key in grouped:
        if key not in seen:
            errors.append('Cost trials lack a matching summary: '+str(key))
    if not summaries:
        errors.append('No cost summaries')
    return dict(status='FAIL' if errors else 'PASS',errors=errors,rows=len(summaries),trial_count=len(trials),
                scope='FP64 median/IQR from raw whole-path trials; CPU/GPU scopes remain separate')


def _public_evidence(outdir):
    errors,missing,rows = [],[],[]
    for dataset,method,seed in itertools.product(common.DATASETS,common.METHODS,common.SEEDS):
        root = outdir/'runs'/dataset/f'{method}_s{seed}'
        candidates = sorted(root.glob('search_cap*.json'),key=lambda p:int(p.stem.rsplit('cap',1)[1]))
        if not candidates:
            missing.append(f'{dataset}/{method}/{seed}: search evidence missing')
            continue
        search_path = candidates[-1]
        search = common.read(search_path)
        cap = int(search_path.stem.rsplit('cap',1)[1])
        refit_path = root/f'refit_cap{cap}.json'
        if not refit_path.is_file():
            missing.append(f'{dataset}/{method}/{seed}: fresh refit evidence missing')
            continue
        refit = common.read(refit_path)
        identity = f'{dataset}/{method}/{seed}'
        if search.get('status')!='COMPLETE' or refit.get('status')!='COMPLETE':
            missing.append(identity+': partial search/refit')
        for phase,record in (('search',search),('refit',refit)):
            if record.get('new_V_used') is not False or record.get('F_frozen_unchanged') is not True or record.get('G_S_frozen_unchanged') is not True:
                errors.append(identity+': '+phase+' fixed-parent/V-use proof failed or absent')
            if record.get('G_shared_seed')!=92601:
                errors.append(identity+': fixed G seed identity differs')
        expected = min(search['curves'],key=lambda row:(row['M_select_mse'],row['epoch'])) if search.get('curves') else None
        if expected is None or search.get('selected_epoch')!=expected['epoch']:
            errors.append(identity+': M_select checkpoint choice differs')
        if refit.get('selected_epoch')!=search.get('selected_epoch') or refit.get('fresh_initialization') is not True:
            errors.append(identity+': fresh full-Mcal refit does not follow sealed selected epoch')
        if refit.get('initial_state_sha256')!=search.get('initial_state_sha256'):
            errors.append(identity+': search/refit initial state differs')
        if refit.get('fit_split')!='M_cal_dense_stride1' or refit.get('exposures')!=math.ceil(float(refit.get('selected_epoch',0))*refit.get('n_fit',0)):
            errors.append(identity+': full-Mcal exposure contract differs')
        rows.append(dict(dataset=dataset,method=method,seed=seed,cap=cap,
                         sampler=search.get('sampled_indices_sha256'),search_exposures=search.get('exposures')))
    for dataset,seed in itertools.product(common.DATASETS,common.SEEDS):
        matched = [r for r in rows if r['dataset']==dataset and r['seed']==seed]
        if len(matched)==4 and (len({r['sampler'] for r in matched})!=1 or len({r['cap'] for r in matched})!=1):
            errors.append(f'{dataset}/{seed}: method groups used different sampler orders or caps')
    return dict(status='FAIL' if errors else 'PARTIAL' if missing else 'PASS',errors=errors,missing=missing,
                mandatory_configurations_verified=len(rows),expected_configurations=16)


def _checkpoint_evidence(outdir):
    if outdir.resolve()!=common.REPORT.resolve():
        return dict(status='NOT_APPLICABLE_SYNTHETIC_OR_REPLAY')
    import torch
    errors,missing,checked = [],[],[]
    for dataset,method,seed,phase in itertools.product(common.DATASETS,common.METHODS,common.SEEDS,('search','refit')):
        directory = common.CACHE/'runs'/dataset/f'{method}_s{seed}'/phase
        checkpoint = directory/'last.pt'
        receipt_path = directory/'RESULT.json'
        label = f'{dataset}/{method}/{seed}/{phase}'
        if not checkpoint.is_file() or not receipt_path.is_file():
            missing.append(label+': complete checkpoint/receipt missing')
            continue
        try:
            bundle = torch.load(checkpoint,map_location='cpu',weights_only=False)
            receipt = common.read(receipt_path)
            if any(key not in bundle for key in ('model','optimizer','sampler','torch_rng','cuda_rng','contract','updates','extra')):
                raise ValueError('Complete restart state fields missing')
            if bundle['contract']!=receipt['contract']:
                raise ValueError('Checkpoint and completed receipt contracts differ')
            for key in ('torch_rng',):
                if not isinstance(bundle[key],torch.Tensor) or bundle[key].dtype!=torch.uint8 or not bundle[key].numel():
                    raise ValueError('CPU RNG state schema differs')
            if not isinstance(bundle['cuda_rng'],list) or not bundle['cuda_rng']:
                raise ValueError('CUDA-trained correction lacks CUDA RNG states')
            if any(not isinstance(state,torch.Tensor) or state.dtype!=torch.uint8 or not state.numel() for state in bundle['cuda_rng']):
                raise ValueError('CUDA RNG state schema differs')
            sampler = bundle['sampler']
            n = int(receipt['n_fit'])
            exposures = int(receipt['exposures'])
            if (sampler['exposures']!=exposures or bundle['extra']['exposures']!=exposures
                    or bundle['updates']!=receipt['updates'] or bundle['extra']['seed']!=seed):
                raise ValueError('Checkpoint updates/exposures/seed differ')
            if sorted(sampler['order'])!=list(range(n)) or sampler['position']!=exposures%n or sampler['epochs_completed']!=exposures//n:
                raise ValueError('Sampler permutation/position/epoch state differs')
            if sampler['origins_sha']!=hashlib.sha256(np.arange(n,dtype=np.int64).tobytes()).hexdigest():
                raise ValueError('Dense fitting-index sampler identity differs')
            sampled = np.asarray(bundle['extra']['sampled_indices'],np.int64)
            if len(sampled)!=exposures or hashlib.sha256(sampled.tobytes()).hexdigest()!=receipt['sampled_indices_sha256']:
                raise ValueError('Actual sampled training stream differs from receipt')
            for start in range(0,len(sampled),n):
                part = sampled[start:start+n]
                if len(np.unique(part))!=len(part) or np.any(part<0) or np.any(part>=n):
                    raise ValueError('Sampler stream is not permutation sampling')
            digest = hashlib.sha256()
            for name,value in bundle['model'].items():
                digest.update(name.encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
            if digest.hexdigest()!=receipt['final_state_sha256']:
                raise ValueError('Checkpoint model tensor hash differs from completed receipt')
            for group in bundle['optimizer']['param_groups']:
                if group['lr']!=.001 or group['weight_decay']!=0:
                    raise ValueError('AdamW learning-rate/weight-decay changed')
            if bundle['updates']>0 and not bundle['optimizer']['state']:
                raise ValueError('Trained checkpoint lacks optimizer moments')
            checked.append(dict(dataset=dataset,method=method,seed=seed,phase=phase,
                checkpoint_sha256=common.sha(checkpoint),updates=bundle['updates'],exposures=exposures))
        except (ValueError,KeyError,TypeError,OSError) as error:
            errors.append(label+': '+str(error))
    return dict(status='FAIL' if errors else 'PARTIAL' if missing else 'PASS',errors=errors,missing=missing,
        checkpoint_count=len(checked),checks=checked,
        CPU_resume_scope='Exact model/optimizer/sampler/CPU RNG continuation tested on synthetic CPU data',
        CUDA_resume_scope='Actual CUDA checkpoint fields and bindings checked; GPU next-update bitwise replay is not performed by this CPU verifier')


def _preservation(outdir):
    if outdir.resolve()!=common.REPORT.resolve():
        return dict(status='NOT_APPLICABLE_SYNTHETIC_OR_REPLAY')
    path = common.CACHE/'INITIAL_BASELINE.json'
    if not path.is_file():
        return dict(status='MISSING',errors=['Initial preservation baseline missing'])
    baseline = common.read(path)
    errors = []
    for relative,digest in baseline.get('protected_old_report_blobs',{}).items():
        target = common.REPO/relative
        if not target.is_file() or common.sha(target)!=digest:
            errors.append('Protected prior file changed/missing: '+relative)
    original = Path(r'E:\CODING\proj\hierarchical-tsfm-peft')
    status = subprocess.check_output(['git','status','--porcelain=v1','-z'],cwd=original)
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=original,text=True).strip()
    status_sha = hashlib.sha256(status).hexdigest()
    if status_sha!=baseline['original_status_default_sha256'] or head!=baseline['original_HEAD']:
        errors.append('Original dirty workspace HEAD/status differs from the initial baseline')
    return dict(status='FAIL' if errors else 'PASS',errors=errors,
        protected_blob_count=len(baseline.get('protected_old_report_blobs',{})),original_HEAD=head,original_status_sha256=status_sha)


def _C_evidence(outdir,decisions):
    if outdir.resolve()!=common.REPORT.resolve():
        return dict(status='NOT_APPLICABLE_SYNTHETIC_OR_REPLAY')
    errors,missing,checked = [],[],[]
    if decisions.get('all_B_and_C_selection_complete') is not True or decisions.get('new_V_predictions_or_scores_before_seal')!=0:
        errors.append('Global C seal does not prove all selection completed before V')
    registration = outdir/'C_SOURCE_AND_SELECTION_REGISTRATION.json'
    if registration.is_file():
        for name,digest in common.read(registration).get('sources',{}).items():
            path = common.REPO/'research'/common.NAME/name
            if not path.is_file() or common.sha(path)!=digest:
                errors.append('Registered C implementation changed: '+name)
    for dataset,decision in decisions.get('datasets',{}).items():
        if decision.get('status')!='COMPLETE_C_COMPARISON_GROUP':
            continue
        records = decision.get('models',[])
        expected = set(itertools.product(('SEQ_BLOCKS','JOINT_FULL','JOINT_PREFIX'),common.SEEDS))
        if {(r.get('method'),r.get('seed')) for r in records}!=expected or len(records)!=6:
            errors.append('Admitted C comparison group does not contain all six configurations: '+dataset)
        for record in records:
            method,seed = record['method'],record['seed']
            directory = common.CACHE/'C'/dataset/f'{method}_s{seed}'
            result_path,selection = directory/'RESULT.json',directory/'SELECTION_SEAL.json'
            label = f'{dataset}/{method}/{seed}'
            if not result_path.is_file() or not selection.is_file():
                missing.append('C result/selection seal missing: '+label)
                continue
            try:
                result,seal = common.read(result_path),common.read(selection)
                if result['status']!='COMPLETE' or result.get('new_V_predictions_or_scores')!=0:
                    raise ValueError('C comparison is partial or used V')
                if common.sha(result_path)!=record['result_sha256'] or common.sha(selection)!=record['selection_seal_sha256']:
                    raise ValueError('C result/selection evidence file SHA256 differs')
                best = min(result['prefix_selection_rows'],key=lambda row:(row['M_select_mse'],row['K']))
                if int(best['K'])!=int(result['selected_prefix']) or result['selected_prefix']!=record['selected_prefix']:
                    raise ValueError('C selected prefix does not follow M_select minimum/tie rule')
                if seal.get('V_used') is not False or seal.get('selection_split')!='M_select':
                    raise ValueError('C selection seal does not show M_select-only selection')
                if result.get('fresh_full_Mcal_refit') is not True or any(r.get('status')!='COMPLETE' for r in result['refits']):
                    raise ValueError('C fresh full-Mcal refits incomplete')
                if not Path(result['state_path']).is_file() or common.sha(result['state_path'])!=record['model_file_sha256']:
                    raise ValueError('C final artifact file SHA256 differs')
                if method=='SEQ_BLOCKS':
                    checks = result['frozen_stage1_checks']
                    required = ('parameter_buffer_hash_unchanged','optimizer_checkpoint_hash_unchanged',
                        'active_block2_parameters_disjoint','refit_parameter_buffer_hash_unchanged',
                        'refit_optimizer_checkpoint_hash_unchanged')
                    if any(checks.get(key) is not True for key in required) or checks.get('previous_optimizer_reused') is not False:
                        raise ValueError('Sequential stage1 freeze/cache/optimizer proof failed')
                checked.append(label)
            except (ValueError,KeyError,TypeError,OSError) as error:
                errors.append(label+': '+str(error))
    return dict(status='FAIL' if errors else 'PARTIAL' if missing else 'PASS',errors=errors,missing=missing,
                complete_conditional_configurations=len(checked),configurations=checked)


def _implementation_evidence(outdir,entries,decisions):
    if outdir.resolve()!=common.REPORT.resolve():
        return dict(status='NOT_APPLICABLE_SYNTHETIC_OR_REPLAY')
    errors,missing = [],[]
    path = outdir/'IMPLEMENTATION_VALIDATION.json'
    if not path.is_file():
        missing.append('IMPLEMENTATION_VALIDATION.json')
    else:
        value = common.read(path)
        if value.get('status')!='PASS' or value.get('tests',{}).get('exit_code')!=0:
            errors.append('Implementation focused tests did not pass')
        required = ('zero_correction_G','residual_loss_value_gradient','mask_channel_macro',
                    'causal_future_canary','normalization_inverse_once')
        for key in required:
            actual = value.get('checks',{}).get(key)
            if actual is not True and actual!='PASS' and not (isinstance(actual,dict) and actual.get('status')=='PASS'):
                missing.append('Implementation check proof: '+key)
    parity = []
    combined = outdir/'C_FULL_V_PARITY.json'
    if combined.is_file():
        value = common.read(combined)
        if value.get('status')!='PASS':
            errors.append('Actual full-V C call parity failed')
        parity = value.get('runs',[])
    else:
        for dataset,decision in decisions.get('datasets',{}).items():
            if decision.get('status')!='COMPLETE_C_COMPARISON_GROUP':
                continue
            path = outdir/f'C_FULL_V_CALL_PARITY_{dataset}.json'
            if not path.is_file():
                missing.append('Actual full-V concat/block F call parity: '+dataset)
                continue
            value = common.read(path)
            if value.get('status')!='PASS':
                errors.append('Actual full-V C call parity failed: '+dataset)
            parity.extend(dict(row,dataset=dataset) for row in value.get('rows',[]))
    for entry in entries:
        if entry['method'] not in ('SEQ_BLOCKS','JOINT_FULL','JOINT_PREFIX'):
            continue
        matched = [row for row in parity if row['dataset']==entry['dataset'] and row['method']==entry['method']
                   and str(row['seed'])==str(entry.get('seed')) and str(row['prefix'])==str(entry.get('prefix'))]
        if len(matched)!=1:
            missing.append('Actual full-V C call parity row: '+entry['run_id'])
            continue
        row = matched[0]
        if row.get('status')!='PASS' or row.get('n_origins')!=entry['score']['n_origins']:
            errors.append('Actual full-V C call parity cohort/status differs: '+entry['run_id'])
        difference = row.get('max_absolute_normalized_difference')
        if difference is None or not math.isfinite(float(difference)) or float(difference)<0:
            errors.append('Actual full-V C call parity numeric evidence missing: '+entry['run_id'])
        if int(entry['prefix'])==0 and row.get('zero_prefix_F_calls')!=0:
            errors.append('Prefix0 call parity invoked F: '+entry['run_id'])
    return dict(status='FAIL' if errors else 'PARTIAL' if missing else 'PASS',errors=errors,missing=missing,
                actual_C_full_V_parity_rows=len(parity))


def _figure_evidence(outdir):
    errors,missing = [],[]
    path = outdir/'FIGURE_MANIFEST.json'
    if not path.is_file():
        return dict(status='MISSING',errors=[],missing=['FIGURE_MANIFEST.json'])
    manifest = common.read(path)
    if manifest.get('status')!='COMPLETE':
        missing.extend(manifest.get('warnings') or ['Figure report is incomplete'])
    for record in manifest.get('figures',[]):
        target = outdir/record['file']
        if not target.is_file() or common.sha(target)!=record['sha256']:
            errors.append('Figure file SHA256 differs: '+record['file'])
    for name,digest in manifest.get('artifacts',{}).items():
        if not (outdir/name).is_file() or common.sha(outdir/name)!=digest:
            errors.append('Public report artifact SHA256 differs: '+name)
    for name,digest in manifest.get('input_files',{}).items():
        if name=='VALIDATION_REPORT.json':
            continue
        if not (outdir/name).is_file() or common.sha(outdir/name)!=digest:
            errors.append('Public figure source input changed: '+name)
    if outdir.resolve()==common.REPORT.resolve() and manifest.get('source_sha256')!=common.sha(Path(__file__).with_name('report.py')):
        errors.append('Figure producer source changed')
    proof = outdir/'REPLAY_VALIDATION.json'
    if not proof.is_file():
        missing.append('REPLAY_VALIDATION.json')
    else:
        replay = common.read(proof)
        if replay.get('status')!='PASS' or replay.get('values_and_text_identical') is not True or replay.get('figure_bytes_identical') is not True:
            errors.append('Independent-folder public figure/text replay failed')
        if replay.get('figure_manifest_sha256') is not None and replay['figure_manifest_sha256']!=common.sha(path):
            errors.append('Public figure replay evidence is stale')
    return dict(status='FAIL' if errors else 'PARTIAL' if missing else 'PASS',errors=errors,missing=missing,
                figure_file_count=len(manifest.get('figures',[])),validation_input_hash_excluded_to_avoid_self_reference=True)


def _requirements(outdir,entries,decisions):
    missing = [name for name in REQUIRED if not (outdir/name).is_file()]
    if not (outdir/'figures').is_dir() or not any((outdir/'figures').glob('*')):
        missing.append('figures/nonempty')
    present = {(e['dataset'],e['method'],str(e.get('seed'))) for e in entries}
    for dataset,method,seed in itertools.product(common.DATASETS,common.METHODS,common.SEEDS):
        if (dataset,method,str(seed)) not in present:
            missing.append(f'final V prediction: {dataset}/{method}/{seed}')
    errors = []
    plan_path = outdir/'EXPERIMENT_PLAN.json'
    if plan_path.is_file() and outdir.resolve()==common.REPORT.resolve():
        for relative,digest in common.read(plan_path).get('sources',{}).items():
            if not (common.REPO/relative).is_file() or common.sha(common.REPO/relative)!=digest:
                errors.append('Registered scientific source changed: '+relative)
    audit_path = outdir/'PARENT_AND_DATA_CONTRACT.json'
    if audit_path.is_file():
        audit = common.read(audit_path)
        if audit.get('status')!='PASS' or audit.get('numeric_T_values_read')!=0:
            errors.append('Parent/data audit or protected-data policy failed')
    decisions_by_data = decisions.get('decisions',decisions.get('datasets',{}))
    if isinstance(decisions_by_data,list):
        decisions_by_data = {r['dataset']:r for r in decisions_by_data}
    for dataset in common.DATASETS:
        record = decisions_by_data.get(dataset)
        if not isinstance(record,dict):
            missing.append('C admission/completion decision: '+dataset)
            continue
        status = record.get('status',record.get('completion_status'))
        if status=='NOT_ADMITTED_NO_INCREMENTAL_EVIDENCE':
            continue
        if status in ('NOT_RUN_BUDGET','PARTIAL_CHECKPOINTED'):
            missing.append('C group not completed: '+dataset+' '+status)
        elif status in ('COMPLETE','COMPLETED','COMPLETE_C_COMPARISON_GROUP'):
            if not (outdir/'PREFIX_RESULTS.csv').is_file():
                missing.append('PREFIX_RESULTS.csv')
            for method,seed in itertools.product(('SEQ_BLOCKS','JOINT_FULL','JOINT_PREFIX'),common.SEEDS):
                if (dataset,method,str(seed)) not in present:
                    missing.append(f'C final V prediction: {dataset}/{method}/{seed}')
                prefixes = {int(e['prefix']) for e in entries if e['dataset']==dataset and e['method']==method
                            and str(e.get('seed'))==str(seed) and not _na(e.get('prefix'))}
                required_prefixes = {0,6,12} if dataset=='jena' else {0,4,8}
                if prefixes!=required_prefixes:
                    missing.append(f'C all-prefix final predictions: {dataset}/{method}/{seed}')
        else:
            missing.append('Unresolved C admission/completion: '+dataset+' '+str(status))
    if (outdir/'EXECUTION_STATUS.csv').is_file():
        for row in _csv(outdir/'EXECUTION_STATUS.csv'):
            if row.get('canonical','true').lower()=='false' or row.get('attempt_role')=='historical' or not _na(row.get('superseded_by')):
                continue
            if row.get('status') in ('PARTIAL','PARTIAL_CHECKPOINTED','NOT_RUN_BUDGET','FAILED','BLOCKED','PENDING','RUNNING'):
                missing.append('Incomplete execution: '+str(row))
    if (outdir/'RUNS.csv').is_file():
        public = _csv(outdir/'RUNS.csv')
        for row in public:
            if str(row.get('seed','')).lower() in ('mean','average','two_seed_mean') or row.get('aggregation')=='two_seed_mean':
                for key in ('alpha','selected_epoch','parent_S_sha256','S_parent_sha256'):
                    if key in row and not (_na(row[key]) or str(row[key]).startswith('NA')):
                        errors.append('Mean-model auxiliary field borrowed from a seed: '+key)
        for entry in entries:
            matched = [r for r in public if r.get('run_id')==entry['run_id'] and r.get('split','V_select') in ('V','V_select')]
            if not matched:
                missing.append('Final V run absent from public RUNS.csv: '+entry['run_id'])
                continue
            for key in ('mse','mae'):
                actual = matched[0].get(key,matched[0].get(key.upper()))
                if actual is None or not _close(actual,entry['score'][key]):
                    errors.append('Public RUNS.csv score differs from saved prediction: '+entry['run_id']+' '+key)
    if (outdir/'COST.csv').is_file():
        costs = _csv(outdir/'COST.csv')
        required_methods = {'G','S','RAW','ALPHA_fitted',*common.METHODS,'MEAN','OWN','ALL','SEQ_BLOCKS','JOINT_FULL','JOINT_PREFIX'}
        for entry in entries:
            if entry['method'] not in required_methods:
                continue
            for batch in (1,8):
                matched = [r for r in costs if r.get('dataset')==entry['dataset'] and r.get('method')==entry['method']
                    and str(r.get('batch_size',r.get('B')))==str(batch)
                    and (_na(entry.get('seed')) or str(r.get('seed'))==str(entry.get('seed')))
                    and (_na(entry.get('prefix')) or str(r.get('prefix',r.get('executed_prefix',r.get('selected_prefix'))))==str(entry.get('prefix')))]
                if not matched:
                    missing.append(f'Actual full-path B{batch} cost missing: '+entry['run_id'])
    return dict(status='FAIL' if errors else 'PARTIAL' if missing else 'PASS',errors=errors,missing=missing)


def verify(manifest_path,*,outdir=common.REPORT):
    """Read final numeric arrays only after the public C-decision seal exists."""
    outdir = Path(outdir)
    seal_path = outdir/'C_DECISIONS.json'
    if not seal_path.is_file() or common.read(seal_path).get('status')!='SEALED_BEFORE_V':
        raise ValueError('All B/C decisions must be sealed before verification opens V arrays')
    decisions = common.read(seal_path)
    manifest = common.read(manifest_path)
    entries = manifest if isinstance(manifest,list) else manifest['entries']
    loaded,errors = [],[]
    seen = set()
    for entry in entries:
        try:
            if entry['run_id'] in seen:
                raise ValueError('Duplicate final run identity: '+entry['run_id'])
            seen.add(entry['run_id'])
            decision_binding = entry.get('metadata',{}).get('C_decision_sha256')
            if decision_binding is not None and decision_binding!=common.sha(seal_path):
                raise ValueError('Final forecast C-decision seal SHA256 differs')
            if (entry.get('metadata',{}).get('selection_used_V',False)
                    or entry.get('metadata',{}).get('deployment_inverse_scaler_count',entry.get('metadata',{}).get('inverse_scaler_count',1))!=1):
                raise ValueError('V selection or repeated inverse scaling in final metadata')
            loaded.append(_load(entry))
        except (ValueError,KeyError,OSError,TypeError) as error:
            errors.append(str(error))
    reference = {}
    for item in loaded:
        dataset = item['entry']['dataset']
        if dataset not in reference:
            reference[dataset] = item
        try:
            _same_cohort(reference[dataset],item)
        except ValueError as error:
            errors.append(str(error))
    if not entries:
        errors.append('Final manifest is empty')
    paired = []
    if not errors:
        try:
            paired = _pairs(loaded)
        except ValueError as error:
            errors.append(str(error))
    outdir.mkdir(parents=True,exist_ok=True)
    stat_rows = _statistics_file(outdir/'ERROR_SUFFICIENT_STATS.csv.gz',loaded)
    replay = _replay_statistics(outdir/'ERROR_SUFFICIENT_STATS.csv.gz',loaded)
    common.csv_write(outdir/'PAIRWISE_EFFECTS.csv',paired)
    cost = verify_cost_tables(outdir)
    execution = _public_evidence(outdir)
    checkpoints = _checkpoint_evidence(outdir)
    conditional = _C_evidence(outdir,decisions)
    implementation = _implementation_evidence(outdir,entries,decisions)
    figures = _figure_evidence(outdir)
    preservation = _preservation(outdir)
    requirements = _requirements(outdir,entries,decisions)
    all_errors = (errors+replay['errors']+cost.get('errors',[])+execution['errors']+checkpoints.get('errors',[])
        +conditional.get('errors',[])+implementation.get('errors',[])+figures.get('errors',[])+preservation.get('errors',[])+requirements['errors'])
    # Missing cost files are an incomplete run, rather than a numerical contradiction.
    if cost['status']=='MISSING':
        all_errors = [e for e in all_errors if e not in cost['errors']]
    prediction_validation = dict(status='FAIL' if errors else 'PASS',errors=errors,run_count=len(loaded),
        accumulation='independent FP64 per origin/horizon/channel; observed equal-channel macro',
        sufficient_statistics_rows=stat_rows,private_prediction_or_target_values_published=False)
    scores = [dict(run_id=r['entry']['run_id'],dataset=r['entry']['dataset'],method=r['entry']['method'],
        seed=r['entry'].get('seed'),prefix=r['entry'].get('prefix'),**r['score']) for r in loaded]
    numeric_errors = errors+replay['errors']+checkpoints.get('errors',[])+conditional.get('errors',[])+implementation.get('errors',[])
    if cost['status']!='MISSING':
        numeric_errors += cost.get('errors',[])
    numerical = dict(schema='INCREMENTAL_INDEPENDENT_NUMERICAL_VALIDATION_v1',
        status='FAIL' if numeric_errors else 'PASS',errors=numeric_errors,
        scope='Numerical consistency of available completed prediction, trial, checkpoint and fixed-parent evidence; completion/report/replay coverage is checked separately in VALIDATION_REPORT.json',
        prediction_validation=prediction_validation,scores=scores,paired_comparison_rows=len(paired),
        sufficient_statistics_replay=replay,
        cost_validation={k:v for k,v in cost.items() if k not in ('status','errors')},
        checkpoint_validation={k:v for k,v in checkpoints.items() if k not in ('status','missing','errors')},
        conditional_validation={k:v for k,v in conditional.items() if k not in ('status','missing','errors')},
        actual_C_full_V_parity_rows=implementation.get('actual_C_full_V_parity_rows',0),
        verifier_source_sha256=common.sha(Path(__file__)),manifest_sha256=common.sha(manifest_path),
        C_decisions_sha256=common.sha(seal_path),
        sufficient_statistics_sha256=common.sha(outdir/'ERROR_SUFFICIENT_STATS.csv.gz'),
        pairwise_effects_sha256=common.sha(outdir/'PAIRWISE_EFFECTS.csv'))
    _receipt(outdir/'NUMERICAL_VALIDATION.json',numerical)
    statuses = (cost['status'],execution['status'],checkpoints['status'],conditional['status'],implementation['status'],
                figures['status'],preservation['status'],requirements['status'])
    status = 'FAIL' if all_errors else 'PARTIAL' if any(s in ('PARTIAL','MISSING') for s in statuses) else 'PASS'
    result = dict(status=status,errors=all_errors,
        numerical_validation_sha256=common.sha(outdir/'NUMERICAL_VALIDATION.json'),
        numerical_status=numerical['status'],prediction_validation=prediction_validation,scores=scores,
        paired_comparison_rows=len(paired),cost_validation=cost,execution_validation=execution,
        checkpoint_validation=checkpoints,
        conditional_validation=conditional,
        implementation_validation=implementation,figure_validation=figures,
        sufficient_statistics_replay=replay,
        preservation_validation=preservation,requirement_audit=requirements,
        manifest_sha256=common.sha(manifest_path),C_decisions_sha256=common.sha(seal_path),
        sufficient_statistics_sha256=common.sha(outdir/'ERROR_SUFFICIENT_STATS.csv.gz'),
        pairwise_effects_sha256=common.sha(outdir/'PAIRWISE_EFFECTS.csv'),
        limitations=['V is repeatedly used development data','Two seeds do not establish confidence intervals',
            'M_select is reused for checkpoint and conditional-stage decisions','Prior protected-data exposure and pretraining overlap are not ruled out'])
    _receipt(outdir/'VALIDATION_REPORT.json',result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--outdir',default=str(common.REPORT))
    args = parser.parse_args(argv)
    result = verify(args.manifest,outdir=args.outdir)
    print(json.dumps(dict(status=result['status'],runs=result['prediction_validation']['run_count'],errors=result['errors']),ensure_ascii=False))
    return 0 if result['status']=='PASS' else 1


if __name__=='__main__':
    raise SystemExit(main())
