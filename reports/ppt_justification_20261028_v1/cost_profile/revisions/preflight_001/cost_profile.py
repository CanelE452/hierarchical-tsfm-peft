"""Frozen Appliances K10/Raw B8 inference costs; never trains a model."""
import argparse
from datetime import datetime, timezone
import csv
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from . import runtime


COST_REPORT = runtime.REPORT / 'cost_profile'
COST_CACHE = runtime.CACHE / 'cost_profile'
FACETS = ('Raw_appliances', 'KTAU_main_appliances_K10_s92601', 'KTAU_main_appliances_K10_s92602')
SOURCE_FILES = [
    f'research/{runtime.NAME}/cost_profile.py', f'research/{runtime.NAME}/runtime.py',
    'research/memory_budget_20261018_v1/worker.py',
    'research/b2_dataset_extension_20261018_v1/costs.py',
    *[f'research/informs_broad_validation_20261015_v1/{name}.py' for name in
      ('deployment', 'combination', 'data_contract', 'fit_base', 'training', 'networks', 'gpu_runtime', 'evaluation')]]


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in
                ('pred_normalized', 'target_normalized', 'observed', 'origins', 'channel_ids', 'mean', 'std')}


def verify_reference(row, saved, canonical=None):
    from research.memory_budget_20261018_v1.worker import independent_metrics
    if saved['pred_normalized'].shape != (41, 48, 26) or saved['target_normalized'].shape != (41, 48, 26):
        raise ValueError('Appliances full-V shape differs')
    expected = np.arange(13814, 15788-48+1, 48, dtype=np.int64)
    if not np.array_equal(saved['origins'], expected) or int(saved['origins'].max()) + 48 >= 15788:
        raise ValueError('Appliances V origin grid/last target differs')
    if canonical is not None:
        for key in ('target_normalized', 'observed', 'origins', 'channel_ids', 'mean', 'std'):
            if not np.array_equal(saved[key], canonical[key], equal_nan=saved[key].dtype.kind in 'fc'):
                raise ValueError('Cost models do not share exact cohort/scaler: ' + key)
    metrics = independent_metrics(saved)
    if max(abs(metrics[key] - float(row[key])) for key in ('MSE', 'MAE')) > 1e-12:
        raise ValueError('Cost accuracy binding differs from channel-macro FP64 metrics')
    return metrics


def prepare():
    if (COST_REPORT / 'COST_SEAL.json').exists():
        validate_seal()
        return dict(status='EXISTING_COST_SEAL_VERIFIED')
    source = runtime.REPO / 'reports/comparative_ablation_followup_20261027_v1/RUNS.csv'
    with source.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    references, canonical = [], None
    for run_id in FACETS:
        raw = run_id == 'Raw_appliances'
        seed = 92601 if raw else int(run_id[-5:])
        matching = [row for row in rows if row['dataset'] == 'appliances'
                    and row['method_id'] == ('RAW' if raw else 'OURS') and (raw or int(row['seed']) == seed)]
        if len(matching) != 1:
            raise ValueError('Cost source must bind one fixed accuracy row: ' + run_id)
        row = matching[0]
        if raw:
            prediction = runtime.DEFAULT_SOURCE / '.cache/b2_dataset_search_20261019_v1/datasets/appliances/costs/Raw_appliances_DEFAULT128_B8_predictions.npz'
            bundle = None
        else:
            prediction = Path(row['prediction_path'])
            bundle = runtime.DEFAULT_SOURCE / '.cache/k_tau_rule_20261023_v1/datasets/appliances/runs' / run_id / 'cap3/selected_bundle.pt'
        saved = arrays(prediction)
        metrics = verify_reference(row, saved, canonical)
        if canonical is None:
            canonical = saved
        if runtime.sha(prediction) != row['prediction_sha256']:
            raise ValueError('Source row prediction hash differs: ' + run_id)
        entry = dict(run_id=run_id, family='RAW' if raw else 'B2', dataset='appliances', seed=seed,
                     K=26 if raw else 10, C=26, L=512, H=48, MSE=float(row['MSE']), MAE=float(row['MAE']),
                     prediction_path=str(prediction), prediction_sha256=runtime.sha(prediction),
                     bundle_path=None if raw else str(bundle), bundle_sha256=None if raw else runtime.sha(bundle),
                     selected_epoch=None if raw else float(row['selected_epoch']), training_cap=0 if raw else 3,
                     validation=dict(status='PASS', shape=[41,48,26], T_start=15788, target_end=15782,
                                     metric_aggregation=metrics['aggregation'], T_numeric_values_parsed=0))
        if not raw:
            import torch
            artifact = torch.load(bundle, map_location='cpu', weights_only=False)
            if (artifact['spec']['family'] != 'B2' or artifact['spec']['latent_k'] != 10
                    or artifact['spec']['seed'] != seed or artifact['spec']['revin'] is not False
                    or artifact['normalization']['fit_end'] != 11841
                    or artifact['contract']['model_sha'] != runtime.F_SHA
                    or artifact['selected_search_epoch'] != entry['selected_epoch']):
                raise ValueError('Cost checkpoint is not the selected original Full-P K10 model')
            for key in ('mean', 'std'):
                if not np.array_equal(artifact['normalization'][key], saved[key]):
                    raise ValueError('Selected checkpoint scaler differs from scored prediction')
            entry.update(bundle_spec=artifact['spec'], bundle_contract=artifact['contract'],
                         model_state_sha=artifact['model_state_sha'])
        references.append(entry)
    prefix = runtime.DEFAULT_SOURCE / '.cache/b2_dataset_search_20261019_v1/data/appliances/PREFIX_PHYSICAL.npz'
    config = prefix.with_name('DATA_DIAGNOSTIC.json')
    protocol = dict(status='REGISTERED_BEFORE_MEASUREMENT', created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Existing selected Full-P Appliances K10 seeds92601/92602 and deterministic Raw; no training, refit, or model selection.',
        facet_order=list(FACETS), fresh_process_per_facet=True, concurrency=1,
        device='cuda', dtype='float32', threads=2, TF32=False, autocast=False, quantization=False, compile=False,
        batch_size=8, requested_F_row_microbatch=128, warmups=10, timed_trials=20,
        expected_work=dict(RAW=dict(streams=208,calls=2,row_sizes=[128,80]), B2=dict(streams=80,calls=1,row_sizes=[80])),
        core='Unmodified research.memory_budget_20261018_v1.worker.evaluate_facet(stage=confirm)',
        deployment='Unmodified broad-validation deployment.Route with selected exact original bundle, fixed A60 scaler, frozen Bolt.',
        timing_scope='physical history and mask -> local causal fill -> fixed A60 scale -> P/E/F/D or Raw -> inverse -> CPU output; excludes window slicing, load, full-V parity, hashes, metric scoring.',
        parity=dict(full_V_origins=41, complete_target_shape=[41,48,26], prediction_atol=1e-5,prediction_rtol=1e-4,
                    source_metric_atol=1e-12, regenerated_metric_atol=1e-5,regenerated_metric_rtol=1e-4,
                    metric='equal-channel observed macro FP64',state='Exact adapter/F tensor hashes before and after inference'),
        synchronization='Before and after every warmup/timed whole-path call; CUDA peak reset before each timed call.',
        memory_scope='Core empty_cache once after deployed model load and before loaded baseline/full-V parity; peak allocated=max20 timed trials, reserved reported separately, resident weights included.',
        summary='median/IQR/p95 from20 repeated first-B8 calls; descriptive p95, not service-tail guarantee.',
        restrictions='Never run while the scientific controller/worker is active. No GPU action in preparation. Root authorizes measurement after training.',
        estimated_minutes=[5,15], training_deadline='2026-10-28T17:30:00+09:00', delivery_deadline='2026-10-28T20:00:00+09:00',
        reference_source=dict(path=str(source),sha256=runtime.sha(source)),
        safe_data=dict(prefix_path=str(prefix),prefix_sha256=runtime.sha(prefix),config_path=str(config),config_sha256=runtime.sha(config),
                       A60_fit_end=11841,V=[13814,15788],T_start=15788,T_numeric_values_parsed=0),
        foundation=dict(state_sha256=runtime.F_SHA,parameters=47718016,model_path=str(runtime.DEFAULT_ASSET/runtime.MODEL_REL)),
        source_hashes={name:runtime.sha(runtime.REPO/name) for name in SOURCE_FILES})
    runtime.write_new(COST_REPORT/'COST_INPUTS.json',dict(rows=references))
    runtime.write_new(COST_REPORT/'COST_PROTOCOL.json',protocol)
    runtime.write_new(COST_REPORT/'COST_SEAL.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
                      files={name:runtime.sha(COST_REPORT/name) for name in ('COST_INPUTS.json','COST_PROTOCOL.json')}))
    return dict(status='SEALED_NOT_MEASURED',facets=3,CPU_reference_checks=3,estimated_minutes=[5,15])


def validate_seal():
    seal=runtime.read(COST_REPORT/'COST_SEAL.json')
    for name,digest in seal['files'].items():
        if runtime.sha(COST_REPORT/name)!=digest:
            raise ValueError('Cost input/protocol seal changed: '+name)
    protocol=runtime.read(COST_REPORT/'COST_PROTOCOL.json')
    for name,digest in protocol['source_hashes'].items():
        if runtime.sha(runtime.REPO/name)!=digest:
            raise ValueError('Cost producer source changed: '+name)
    for key in ('prefix','config'):
        if runtime.sha(protocol['safe_data'][key+'_path'])!=protocol['safe_data'][key+'_sha256']:
            raise ValueError('Bounded physical prefix changed')
    rows=runtime.read(COST_REPORT/'COST_INPUTS.json')['rows']
    for row in rows:
        for label in ('prediction','bundle'):
            if row[label+'_path'] is not None and runtime.sha(row[label+'_path'])!=row[label+'_sha256']:
                raise ValueError('Selected cost artifact changed: '+row['run_id'])
    return protocol, rows


def ensure_training_finished():
    path=runtime.REPORT/'EXECUTION_COMPLETION.json'
    if not path.is_file() or runtime.read(path).get('status') not in ('TRAINING_COMPLETE','TRAINING_INCOMPLETE'):
        raise RuntimeError('GPU cost measurement is forbidden until scientific training has ended')


class StateCheckedRoute:
    def __init__(self, route, inputs, device):
        from research.b2_dataset_extension_20261018_v1.costs import _states
        self.route,self.inputs,self.device=route,inputs,device
        self.request={'family':inputs['candidate']['family']}
        self.before=_states(route,inputs,self.request,device)
        self.after=None
        self.closed=False
        self.predictions=[]
        self.parity_batches=math.ceil(len(inputs['saved']['origins'])/8)

    def __getattr__(self,name):
        return getattr(self.route,name)

    def predict_physical(self,physical,observed):
        output=self.route.predict_physical(physical,observed)
        if len(self.predictions)<self.parity_batches:
            self.predictions.append(((output-self.route.mean)/self.route.std).copy())
        return output

    def close(self):
        if self.closed:
            return
        self.closed=True
        from research.b2_dataset_extension_20261018_v1.costs import _states
        try:
            self.after=_states(self.route,self.inputs,self.request,self.device)
        finally:
            self.route.close()


def measure(inputs, protocol_sha, *, device='cuda', sampler_factory=None):
    from research.memory_budget_20261018_v1 import worker as core
    wrapped=[]
    original_loader=inputs['loader']
    def loader():
        route=original_loader()
        try:
            value=StateCheckedRoute(route,inputs,device)
            wrapped.append(value)
            return value
        except BaseException:
            route.close()
            raise
    active=dict(inputs,loader=loader,registered_rows=(128,))
    candidate=inputs['candidate']
    request=dict(asset_root='REGISTERED_SELECTED_INPUTS',accuracy_run_id=candidate['run_id'],
                 batch_size=8,row_microbatch=128,stage='confirm',protocol_sha256=protocol_sha,
                 facet_id=candidate['run_id']+'_DEFAULT128_B8')
    kwargs={'device':device}
    if sampler_factory is not None:
        kwargs['sampler_factory']=sampler_factory
    result=core.evaluate_facet(request,active,**kwargs)
    checked=wrapped[0]
    if checked.before!=checked.after:
        raise ValueError('Selected adapter/foundation tensor states changed across inference')
    predictions=np.concatenate(checked.predictions)
    if core._fingerprint(predictions)!=result['parity']['output_fingerprints']['regenerated_normalized']:
        raise ValueError('Captured complete-V outputs differ from measurement parity')
    result.update(states_before=checked.before,states_after=checked.after,
                  T_numeric_values_parsed=0,training_executed=False)
    return result,dict(inputs['saved'],pred_normalized=predictions,
                       reference_pred_normalized=inputs['saved']['pred_normalized'])


def make_inputs(row, protocol):
    modules=runtime.load_modules()
    deployment=importlib.import_module('deployment')
    if Path(deployment.__file__).resolve()!=runtime.REPO/'research/informs_broad_validation_20261015_v1/deployment.py':
        raise ValueError('Wrong deployment source root')
    task=runtime.load_task('appliances')
    runtime.assert_protected_T(dict(job={'dataset':'appliances'},task_access=task['_campaign_access']))
    saved=arrays(row['prediction_path'])
    verify_reference(row,saved)
    mean,std,_=modules['data'].fit_statistics(task,11841)
    if not np.array_equal(mean,saved['mean']) or not np.array_equal(std,saved['std']):
        raise ValueError('Physical task normalization differs from cost accuracy')
    target_rows=saved['origins'][:,None]+np.arange(48)[None]
    observed=task['observed'][target_rows]
    targets=((task['physical'][target_rows]-mean)/std).astype(np.float32)
    if not np.array_equal(observed,saved['observed']) or not np.allclose(targets[observed],saved['target_normalized'][observed],atol=1e-5,rtol=1e-4):
        raise ValueError('Physical V targets/mask differ from bound score arrays')
    candidate={key:row[key] for key in ('run_id','family','dataset','seed','K','C','L','H','MSE','MAE')}
    candidate['prediction_binding']=dict(path=row['prediction_path'],sha256=row['prediction_sha256'])
    if row['bundle_path']:
        candidate['bundle_binding']=dict(path=row['bundle_path'],sha256=row['bundle_sha256'])
    def loader():
        if row['bundle_path']:
            bundle=Path(row['bundle_path'])
            spec=dict(row['bundle_spec'],artifact_cache_relative_path=bundle.name,artifact_sha256=row['bundle_sha256'])
            cache_root=bundle.parent
        else:
            spec=dict(family='RAW',run_id=row['run_id'],dataset='appliances',foundation_sha=runtime.F_SHA)
            cache_root=COST_CACHE
        spec.update(device='cuda',canonical_fit_end=11841,row_microbatch=128,F_row_microbatch=128,checkpoint_inputs=False)
        pool=deployment.FoundationPool(runtime.DEFAULT_ASSET/runtime.MODEL_REL,'cuda',
                                       factory=lambda p,h,m,d:runtime.load_foundation(device=d))
        try:
            route=deployment.Route(spec,task,cache_root,device='cuda',foundation_pool=pool)
            route.owns_pool=True
            return route
        except BaseException:
            pool.close()
            raise
    return dict(candidate=candidate,loader=loader,task=task,saved=saved,
                expected_f_sha=runtime.F_SHA,expected_f_params=47718016,
                source_ctx=dict(repo=str(runtime.REPO)),
                source_bindings=dict(runtime_source_sha256=protocol['source_hashes']))


def run_worker(run_id):
    ensure_training_finished()
    protocol,rows=validate_seal()
    row=next(r for r in rows if r['run_id']==run_id)
    receipt=COST_REPORT/(run_id+'_MEASUREMENT.json')
    prediction=COST_CACHE/(run_id+'_predictions.npz')
    if receipt.exists() or prediction.exists():
        raise FileExistsError('Preserve existing cost attempt; never overwrite a measured facet')
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    result,values=measure(make_inputs(row,protocol),runtime.sha(COST_REPORT/'COST_PROTOCOL.json'))
    validate_seal()
    prediction.parent.mkdir(parents=True,exist_ok=True)
    with prediction.open('xb') as stream:
        np.savez_compressed(stream,**values)
    result.update(prediction_path=str(prediction),prediction_sha256=runtime.sha(prediction),
                  protocol_sha256=runtime.sha(COST_REPORT/'COST_PROTOCOL.json'),
                  CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    runtime.write_new(receipt,result)
    return dict(status=result['status'],run_id=run_id,receipt=str(receipt))


def run_all():
    ensure_training_finished()
    validate_seal()
    with runtime.CampaignLock(runtime.CACHE/'controller.lock'), runtime.CampaignLock(runtime.CACHE/'worker.lock'), runtime.CampaignLock(COST_CACHE/'cost.lock'):
        runtime.write_new(COST_REPORT/'COST_LAUNCH.json',dict(started_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),facets=list(FACETS)))
        for run_id in FACETS:
            log=COST_CACHE/(run_id+'.log')
            log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('x',encoding='utf8') as stream:
                process=subprocess.run([sys.executable,'-B','-m',f'research.{runtime.NAME}.cost_profile','--worker',run_id],
                    cwd=runtime.REPO,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),stdout=stream,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if process.returncode!=0:
                raise RuntimeError('Cost facet failed; preserve log: '+str(log))
        rows=[runtime.read(COST_REPORT/(run_id+'_MEASUREMENT.json'))['summary'] for run_id in FACETS]
        raw=rows[0]
        for row in rows:
            row['latency_ratio_to_raw']=row['median_batch_ms']/raw['median_batch_ms']
            row['peak_allocated_ratio_to_raw']=row['peak_allocated_MiB']/raw['peak_allocated_MiB']
        runtime.write_new(COST_REPORT/'COST_SUMMARY.json',dict(status='COMPLETE',rows=rows,
                          completed_utc=datetime.now(timezone.utc).isoformat(),training_executed=False,T_numeric_values_parsed=0))
    return dict(status='COMPLETE',facets=3)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--prepare',action='store_true')
    group.add_argument('--run',action='store_true')
    group.add_argument('--worker',choices=FACETS)
    args=parser.parse_args()
    result=prepare() if args.prepare else run_worker(args.worker) if args.worker else run_all()
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
