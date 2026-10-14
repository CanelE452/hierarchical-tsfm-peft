"""CPU audit of the preserved auxiliary pilot, without importing its runners."""
from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PILOT = ROOT / 'research/bihash_future_alignment_20261013'
CACHE = ROOT / '.cache/bihash_future_alignment_20261013'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ahash(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def checked_path(value):
    path = Path(value)
    path = path if path.is_absolute() else ROOT / path
    if path.name.lower() in ('test.npz', 'jena_test.npz'):
        raise PermissionError('Protected TEST is excluded from this audit')
    return path


def main():
    start = time.perf_counter()
    destination = HERE / 'PILOT_AUDIT_v1.json'
    if destination.exists():
        raise FileExistsError(destination)
    checks, gaps = [], []

    def check(name, passed, detail=None):
        checks.append({'name': name, 'pass': bool(passed), 'detail': detail})

    def close(name, actual, expected, rtol=2e-9, atol=1e-9):
        a, b = np.asarray(actual), np.asarray(expected)
        passed = a.shape == b.shape and np.allclose(a, b, rtol=rtol, atol=atol)
        diff = float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size and a.shape == b.shape else None
        check(name, passed, {'max_abs': diff, 'rtol': rtol, 'atol': atol})

    helper_path = ROOT / 'research/smallk_predictive_alignment_20261013/validation_contracts.py'
    spec = importlib.util.spec_from_file_location('independent_previous_math', helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    preflight_path = ROOT / 'research/smallk_predictive_alignment_20261013/PREFLIGHT.json'
    preflight = read(preflight_path)
    dev_path, retrieval_path = PILOT/'run_dev_v1/RESULT.json', PILOT/'run_retrieval_v2/RESULT.json'
    fixture_path = PILOT/'run_fixture_v2/FIXTURE.json'
    dev, retrieval, fixture = read(dev_path), read(retrieval_path), read(fixture_path)
    source_paths = [PILOT/'bihash_pilot.py', PILOT/'retrieval_posthoc.py', PILOT/'make_report.py',
                    ROOT/'research/smallk_predictive_alignment_20261013/engine.py',
                    ROOT/'research/past_only_repr_diagnosis_20261012/common.py', helper_path, preflight_path,
                    dev_path, retrieval_path, fixture_path]
    original_hashes = {str(p.relative_to(ROOT)): sha(p) for p in source_paths}
    check('dev_terminal', dev['status']=='COMPLETE')
    check('retrieval_terminal', retrieval['status']=='COMPLETE')
    check('dev_scope', len(dev['cases'])==16 and dev['updates']==200)
    check('retrieval_scope', len(retrieval['cases'])==8)
    data = {}
    for dataset, contract in dev['data_contracts'].items():
        path = checked_path(contract['source'])
        check('raw_source_SHA:'+dataset, sha(path)==contract['source_hash']==preflight['legacy_data'][dataset]['source']['sha256'])
        data[dataset] = helper.legacy_data(preflight['legacy_data'][dataset])
        check('data_counts:'+dataset, len(data[dataset]['origins']['fit'])==256 and len(data[dataset]['origins']['tune'])==64)
        check('data_normalization:'+dataset, contract['normalization_fit_scope']=='FIT_ONLY')

    def score_internal(label, score, mask):
        counts = mask.sum((0, 1))
        close(label+':observed_counts', score['observed_counts'], counts, rtol=0, atol=0)
        for norm, value in (('physical', score), ('normalized', score['normalized'])):
            for metric in ('mse', 'mae'):
                v = np.asarray(value['per_channel_'+metric])
                check(label+':'+norm+':finite_'+metric, np.isfinite(v).all() and np.all(v>=0))
                close(label+':'+norm+':channel_macro_'+metric, value[metric], v[counts>0].mean())
            check(label+':'+norm+':MAE_MSE_bound', np.all(np.asarray(value['per_channel_mae'])**2 <= np.asarray(value['per_channel_mse'])+1e-9))
            close(label+':'+norm+':mask_counts', value['observed_counts'], counts, rtol=0, atol=0)
        if mask.shape[1]==48:
            for segment, lo, hi in (('1_8',0,8),('9_24',8,24),('25_48',24,48)):
                part = score['horizon_metrics'][segment]
                score_internal(label+':'+segment, {**part, 'normalized':score['normalized']['horizon_metrics'][segment]}, mask[:,lo:hi])

    parameter_changes, negative_support = [], []
    initial_hashes = {}
    cases = {(c['dataset'],c['seed'],c['method']):c for c in dev['cases']}
    for case in dev['cases']:
        label = case['dataset']+':'+str(case['seed'])+':'+case['method']
        d = data[case['dataset']]
        check(label+':200_updates', case['updates']==200 and len(case['curve'])==200 and [r['update'] for r in case['curve']]==list(range(1,201)))
        check(label+':CP_updates', [c['update'] for c in case['checkpoints']]==[0,50,100,200])
        close(label+':loss_composition', [c['loss'] for c in case['curve']], [c['main_loss']+case['weight']*c['aux_loss'] for c in case['curve']], rtol=3e-7,atol=2e-6)
        check(label+':finite_curve', np.isfinite([[r[k] for k in ('main_loss','aux_loss','loss','gradient_norm')] for r in case['curve']]).all())
        order = np.random.default_rng(case['seed']).integers(0,256,size=(200,4))
        check(label+':actual_batch_order_SHA', ahash(order)==case['order_sha256'])
        origins = d['origins']['fit'][order]
        distances = np.abs(origins[:,:,None]-origins[:,None,:])
        offdiag = ~np.eye(4,dtype=bool)[None]
        negatives = {'dataset':case['dataset'],'seed':case['seed'],'method':case['method'],
                     'directed_negative_pairs':2400,'same_origin_pairs':int(((distances==0)&offdiag).sum()),
                     'overlapping_future_pairs':int(((distances<48)&offdiag).sum()),
                     'query_rows_with_no_nonoverlap_negative':int((((distances>=48)&offdiag).sum(2)==0).sum())}
        negative_support.append(negatives)
        states = {}
        for cp in case['checkpoints']:
            path = checked_path(cp['path'])
            check(label+':CP_SHA:'+str(cp['update']),sha(path)==cp['sha256'])
            state = torch.load(path,map_location='cpu',weights_only=True)
            check(label+':CP_metadata:'+str(cp['update']),state['update']==cp['update'] and state['method']==case['method'] and state['seed']==case['seed'])
            if cp['update'] in (0,200): states[cp['update']]=state
            for branch in ('forecast','forecast_q'):
                if branch not in cp: continue
                check(label+':scored_splits:'+branch,set(cp[branch])=={'fit','tune'})
                for split,score in cp[branch].items():
                    mask = np.stack([d['mask'][o:o+48] for o in d['origins'][split]])
                    score_internal(label+':CP'+str(cp['update'])+':'+branch+':'+split,score,mask)
        expected = min((c['forecast']['tune']['normalized']['mse'],c['update']) for c in case['checkpoints'])[1]
        check(label+':TUNE_BOLT_choice',case['selected_update']==expected)
        if case['method']!='BASE':
            q_expected = min((c['forecast_q']['tune']['normalized']['mse'],c['update']) for c in case['checkpoints'])[1]
            check(label+':TUNE_Q_choice',case['selected_q_update']==q_expected)
        initial_hashes.setdefault((case['dataset'],case['seed']),[]).append(helper.state_hash(states[0]['codec']))
        record = {'dataset':case['dataset'],'seed':case['seed'],'method':case['method'],'groups':{}}
        for name,prefixes,branch in (('E',('codec.project.','codec.input.','codec.temporal.','codec.output.'),'codec'),
                                      ('D',('codec.decode.','residual.'),'codec'),('Q',('q.',),'alignment'),
                                      ('Wpast',('past_hash.',),'alignment'),('Wfuture',('future_hash.',),'alignment'),
                                      ('hybrid',('hybrid.',),'alignment')):
            if states[0][branch] is None:continue
            keys=[k for k in states[0][branch] if k.startswith(prefixes)]
            differences=[np.max(np.abs(states[200][branch][k].numpy()-states[0][branch][k].numpy())) for k in keys]
            record['groups'][name]={'max_abs_update':float(max(differences,default=0.)),
                                    'changed_parameter_tensors':int(sum(v>0 for v in differences)),'tensors':len(keys)}
        parameter_changes.append(record)
        if case['method'] in ('BIHASH','BIHASH_SHUFFLED'):
            for group in ('E','D','Q','Wpast','Wfuture','hybrid'):
                check(label+':actual_'+group+'_state_update',record['groups'][group]['max_abs_update']>0)
        if case['method']=='BASE':continue
        state = states[0]['alignment']; basis=states[0]['codec']['fixed_basis'].numpy()
        metadata=case['alignment_metadata']
        target=np.stack([d['x'][o:o+48] for o in d['origins']['fit']]); mask=np.stack([d['mask'][o:o+48] for o in d['origins']['fit']])
        histories=np.stack([d['physical'][o-512:o] for o in d['origins']['fit']]); hm=np.stack([d['mask'][o-512:o] for o in d['origins']['fit']])
        last=helper.independent_window_fill(histories,hm,d['mean'],d['std'])[:,-1]
        coordinate,valid=helper.independent_coordinates(target,mask,last,basis)
        close(label+':FIT_future_coordinates',state['target'].numpy(),coordinate,rtol=2e-6,atol=2e-6)
        close(label+':FIT_valid_mask',state['valid'].numpy(),valid,rtol=0,atol=0)
        check(label+':FIT_target_SHA',ahash(state['target'].numpy())==metadata['target_sha256'])
        check(label+':FIT_valid_SHA',ahash(valid)==metadata['valid_sha256'])
        summary,smask=helper.independent_summary(state['target'].numpy(),valid)
        close(label+':summary_mask',state['summary_valid'].numpy(),smask,rtol=0,atol=0)
        means=np.array([summary[smask[:,j],j].mean() for j in range(summary.shape[1])],np.float32)
        stds=np.array([summary[smask[:,j],j].std() for j in range(summary.shape[1])],np.float32)
        stds=np.where(stds>1e-6,stds,1.).astype(np.float32)
        close(label+':FIT_summary_mean',state['means'].numpy(),means,rtol=1e-5,atol=2e-6)
        close(label+':FIT_summary_std',state['stds'].numpy(),stds,rtol=1e-5,atol=2e-6)
        close(label+':FIT_summary_targets',state['summary_target'].numpy(),(summary-means)/stds,rtol=1e-5,atol=1e-5)
        permutation=state['shuffle'].numpy(); usable=smask.all(1)
        check(label+':shuffle_bijection',np.array_equal(np.sort(permutation),np.arange(len(permutation))))
        check(label+':shuffle_loss_mask',np.array_equal(usable,usable[permutation]))
        check(label+':shuffle_target_nonoverlap',np.all(np.abs(d['origins']['fit'][usable]-d['origins']['fit'][permutation[usable]])>=48))
    for key,values in initial_hashes.items():check('same_initial_codec:'+str(key),len(set(values))==1)

    retrieval_rows=[]
    for case in retrieval['cases']:
        d=data[case['dataset']]; source=cases[case['dataset'],case['seed'],case['method']]
        causal=d['origins']['fit'][None,:]+48<=d['origins']['tune'][:,None]
        check('retrieval_source_CP:'+str((case['dataset'],case['seed'],case['method'])),case['selected_update']==source['selected_update'])
        for result in case['retrieval']:
            prefix='retrieval:'+case['dataset']+':'+str(case['seed'])+':'+case['method']+':'+result['mode']
            check(prefix+':causal_min',result['causal_bank_min']==int(causal.sum(1).min()))
            check(prefix+':causal_max',result['causal_bank_max']==int(causal.sum(1).max()))
            check(prefix+':selected_memory_end_lag_nonnegative',result['nearest_fit_target_end_lag_min']>=0)
            check(prefix+':alpha_grid',set(result['candidate_scores'])=={'0.0','0.25','0.5','0.75','1.0'})
            mask=np.stack([d['mask'][o:o+48] for o in d['origins']['tune']])
            for alpha,payload in result['candidate_scores'].items():score_internal(prefix+':alpha'+alpha,payload['scores'],mask)
            alpha_one=result['candidate_scores']['1.0']['scores']['normalized']['mse']
            expected=source['selected_checkpoint']['forecast_q']['tune']['normalized']['mse']
            close(prefix+':alpha1_is_standard_CP_Q_path',alpha_one,expected,rtol=2e-6,atol=1e-6)
            chosen=min(result['candidate_scores'],key=lambda a:result['candidate_scores'][a]['scores']['normalized']['mse'])
            retrieval_rows.append({'dataset':case['dataset'],'seed':case['seed'],'method':case['method'],
                                   'mode':result['mode'],'best_alpha_on_same_TUNE':float(chosen),
                                   'normalized_mse':result['candidate_scores'][chosen]['scores']['normalized']['mse'],
                                   'memory_causal_counts':[int(causal.sum(1).min()),int(causal.sum(1).max())]})
    resources=[]
    for path in sorted((PILOT/'resources').glob('*.json')):
        r=read(path)
        if r['status']=='RUNNING':raise ValueError('Active resource ledger must not be audited')
        resources.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),**r})
    check('resource_receipts_five',len(resources)==5)
    gpu_wall=sum(r['wall_seconds'] for r in resources if r['gpu_process'])
    forecast_files=list(CACHE.rglob('*.npz'))+list(PILOT.rglob('*.npz'))
    gaps.extend([
        {'gate':'forecast_npz_raw_rescore','met':False,'reason':'No saved forecast NPZ exists; JSON aggregation/count/selection checks are not prediction rescore.'},
        {'gate':'retrieval_aware_joint_forecast_training','met':False,'reason':'Retrieval uses frozen auxiliary checkpoints, no_grad and a post-hoc alpha grid; no main-loss training of deployed memory fusion.'},
        {'gate':'all_GPU_resource_accounted','met':False,'reason':'Five terminal ledgers exclude --fixture GPU execution because fixture() does not instantiate Limit.'},
        {'gate':'separate_branch_gradient_fixture','met':False,'reason':'Recorded hash_grad sums all alignment parameters including Q and hybrid; codec aggregate from auxiliary delta does not establish decoder gradients.'},
        {'gate':'independent_evaluation','met':False,'reason':'Checkpoint and post-hoc alpha both selected/reported on repeatedly used TUNE; VAL rows loaded but no VAL scores emitted.'},
        {'gate':'nonoverlap_contrastive_negatives','met':False,'reason':'Auxiliary CE uses all offdiagonal in-batch targets without duplicate/horizon-overlap masking; actual counts retained.'},
        {'gate':'implementation_at_training_time_source_seal','met':False,'reason':'Pilot receipts record module paths but not execution-time source hashes/snapshots; current source hashes are provenance only.'},
    ])
    for p in source_paths:check('current_source_preserved:'+str(p.relative_to(ROOT)),sha(p)==original_hashes[str(p.relative_to(ROOT))])
    failed=[c for c in checks if not c['pass']]
    result={'status':'AUDITED_WITH_MATERIAL_GAPS' if not failed else 'AUDIT_CHECK_FAILURES_WITH_MATERIAL_GAPS',
            'artifact_checks_status':'PASS' if not failed else 'FAIL','created_unix':time.time(),
            'sources':original_hashes,'checks':checks,'checked_count':len(checks),'failed_count':len(failed),
            'scientific_gates':gaps,'primary_goal_complete':False,
            'parameter_state_updates':parameter_changes,'negative_pair_support':negative_support,
            'posthoc_retrieval':retrieval_rows,'resource_receipts':resources,
            'recorded_GPU_process_wall_seconds_lower_bound':gpu_wall,'fixture_resources_missing':True,
            'forecast_npz_files':[str(p.relative_to(ROOT)) for p in forecast_files],
            'fixture_receipt':{'path':str(fixture_path.relative_to(ROOT)),'sha256':sha(fixture_path),
                               'hash_gradient_aggregate':fixture['hash_parameters_receive_gradient_sum_abs'],
                               'codec_gradient_aggregate':fixture['codec_parameters_receive_gradient_sum_abs']},
            'semantic_review':{'optimizer':'AdamW(codec + all alignment parameters), lr .001, decay0; groups checked by actual CP0 to CP200 state changes',
                               'Gumbel':'Correct per-bit stack(I,-I), dim -1, hard=False, soft probability to +/-1; deterministic sign(0) maps -1',
                               'auxiliary':'One-way batch CE + .2 future-informed hybrid MSE + .02 bit balance; not symmetric contrastive or original loss',
                               'deployment':'BOLT omits alignment; BOLT_Q retains Q only; posthoc memory uses fixed FIT future coordinates',
                               'data':'Historical TRAIN/VAL source, FIT-only stats. Only FIT/TUNE forecast scores emitted; prior TUNE reused',
                               'limits':'No full fusion dual-branch predictor, frozen-vs-trainable projection control, matched trained cosine control, or independent evaluation'},
            'limitations':['No GPU, new training, TEST read or forecast reinference',
                           'Missing prediction arrays prevent independent forecast MSE/MAE rescore',
                           'Source tracing and CP changes do not replace a per-module live-gradient fixture',
                           'Short pilot failures cannot establish impossibility of Bi-Hash future prediction'],
            'cpu_wall_seconds':time.perf_counter()-start,'cpu_only':True,'audit_source_sha256':sha(Path(__file__))}
    serialized = json.dumps(result,ensure_ascii=False,indent=2)
    with destination.open('x',encoding='utf-8') as handle:
        handle.write(serialized+'\n')
    print(json.dumps({k:result[k] for k in ('status','artifact_checks_status','checked_count','failed_count','recorded_GPU_process_wall_seconds_lower_bound','cpu_wall_seconds')}))


if __name__=='__main__':
    main()
