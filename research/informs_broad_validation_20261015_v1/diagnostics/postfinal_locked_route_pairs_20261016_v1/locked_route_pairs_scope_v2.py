"""Public-only decomposition of fixed structural locked route comparisons."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


DATASETS = ('beijing', 'jena', 'electricity', 'pems08')
ROLES = ('G', 'Hpast', 'H_B0', 'H_B2', 'RawH', 'AlphaFit', 'RawSmall', 'LoRA')
HEADS = ('Hpast', 'H_B0', 'H_B2', 'RawH')
REFERENCE_KEYS = ('target_normalized', 'observed', 'origins', 'channel_ids')
PATTERNS = ('variability', 'seasonality', 'delayed_crosschannel', 'intermittency')
DEPENDENCIES = {
    'diagnostics/public_horizon_summary_20261016_v1/horizon_summary.py': 'ca7d4711fca27c1dcc0903945b3d915f6a016279824736a19c29d58beaa9d9d2',
    'time_block_analysis.py': '4cc9cb52c7339f9547d8bfa6f0c9f2455ccf0f0874d19d3cb2581c8bce6b9f89',
}
ROOT = Path(__file__).resolve().parents[2]
SOURCE_FREEZE_SHA256 = 'd5499db440871a928ef4d0275c26a222e657ac9b044d5eaa1295e0bdb2bed9fa'
_MODULES = {}


def byte_sha(value):
    return hashlib.sha256(value).hexdigest()


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):value.update(chunk)
    return value.hexdigest()


def _load_module(relative, name):
    path = ROOT / relative
    if file_sha(path) != DEPENDENCIES[relative]:raise ValueError('Pinned dependency changed: '+relative)
    if name not in _MODULES:
        if str(ROOT) not in sys.path:sys.path.insert(0, str(ROOT))
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); _MODULES[name] = value
    return _MODULES[name]


def horizon():
    return _load_module('diagnostics/public_horizon_summary_20261016_v1/horizon_summary.py', 'locked_pair_horizon')


def block_module():
    return _load_module('time_block_analysis.py', 'locked_pair_blocks')


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def comparison_roles():
    return [(role, 'G') for role in ('H_B0', 'H_B2', 'RawH', 'Hpast', 'AlphaFit')] + [
        ('H_B2', 'H_B0'), ('H_B0', 'Hpast'), ('H_B2', 'Hpast'),
        ('H_B0', 'RawH'), ('H_B2', 'RawH'), ('H_B2', 'AlphaFit'),
        ('RawH', 'RawSmall'), ('H_B2', 'LoRA')]


def _role(spec):
    family = spec.get('family')
    if family == 'DIRECT_G':return 'G'
    if family == 'RAW' and spec.get('model_size') == 'small':return 'RawSmall'
    if family == 'LORA':return 'LoRA'
    if family == 'ALPHA':return 'AlphaFit'
    if family == 'COMBINATION':
        if spec.get('variant') == 'past_control':return 'Hpast'
        if spec.get('variant') == 'conditional':return {'B0':'H_B0', 'B2':'H_B2', 'RAW':'RawH'}.get(spec.get('b_route', {}).get('family'))
    return None


def map_roles(bindings):
    found = defaultdict(list)
    for run_id, item in bindings.items():
        spec = item['route_spec']; role = _role(spec); dataset = spec.get('dataset')
        if dataset in DATASETS and role:found[dataset, role].append(dict(run_id=run_id, spec=spec))
    result, errors = {}, {}
    for dataset in DATASETS:
        result[dataset] = {}
        for role in ROLES:
            choices = found[dataset, role]
            result[dataset][role] = choices[0] if len(choices)==1 else None
            if len(choices)!=1:errors[dataset+':'+role] = 'Expected exactly one locked metadata role; found '+str(len(choices))
    return result, errors


def _guard(public):
    path = public/'diagnostics/whole_available_audit_1792092300000/SOURCE_FREEZE.json'
    if file_sha(path)!=SOURCE_FREEZE_SHA256:raise ValueError('Original source39 freeze manifest changed')
    expected = _json(path)['source39']
    values = {name:file_sha(public/name) for name in expected}
    if values != expected:raise ValueError('Frozen source39 differs')
    values.update({name:file_sha(public/name) for name in ('EXPERIMENT_REGISTRY.csv', 'SELECTION_LOCK.json')})
    for name, expected_sha in DEPENDENCIES.items():
        if file_sha(public/name)!=expected_sha:raise ValueError('Pinned public helper changed: '+name)
    return values


def _bound_json(public, relative, expected_sha, inputs):
    path = horizon().inside(public, relative)
    if file_sha(path)!=expected_sha:raise ValueError('Public metadata SHA differs: '+relative)
    inputs[relative] = dict(file=relative, sha256=expected_sha, bytes=path.stat().st_size)
    return _json(path)


def _parent_check(parent, route, manifest):
    for key, expected in (('run_id',route['run_id']), ('model_sha',route['expected_prediction_model_sha']),
                          ('data_sha',manifest['data_sha']), ('scaler_sha',manifest['scaler_sha'])):
        if parent.get(key)!=expected:raise ValueError('Calibration parent differs: '+key)
    artifact=route.get('artifact_sha256')
    if artifact is None and route['family']=='RAW':
        artifact=route['expected_prediction_model_sha']
        if route['standard_V_binding']['model_sha']!=artifact:raise ValueError('Raw pretrained prediction identity differs')
    if not artifact or parent.get('artifact_sha256')!=artifact:raise ValueError('Calibration parent artifact differs')


def check_shared_recipes(roles):
    errors = {}
    if not roles.get('G'):return {role:'Shared locked G missing' for role in ROLES if role not in {'G','RawSmall','LoRA'}}
    g = roles['G']['spec']; gm = None; recipe = None
    keys = ('head_parameter_count','head_input_width','seed','learning_rate','effective_batch','microbatch','registered_epoch_endpoints','actual_epochs','unique_M_origins','updates')
    for role in (*HEADS, 'AlphaFit'):
        item=roles.get(role)
        if not item:continue
        try:
            route=item['spec']; receipt=item['receipt']; manifest=item['manifest']
            if route.get('g_route')!=g:raise ValueError('Different frozen G route')
            for name in ('g_m','g_v'):_parent_check(receipt['parent_models'][name], g, manifest)
            cache={key:receipt['parent_cache_fingerprints'][key] for key in ('g_m','g_v')}
            if gm is None:gm=cache
            if cache!=gm:raise ValueError('Different fixed G M/V cache fingerprints')
            if role!='Hpast':
                for name in ('parent_m','parent_v'):_parent_check(receipt['parent_models'][name], route['b_route'], manifest)
            if role in HEADS:
                if receipt['selected_checkpoint_sha256']!=route['artifact_sha256']:raise ValueError('Selected head checkpoint differs')
                if receipt.get('parents_unchanged') is not True or receipt.get('parent_forecast_gradients') is not False:raise ValueError('Recorded parent freezing differs')
                current={key:receipt[key] for key in keys}
                if recipe is None:recipe=current
                if current!=recipe:raise ValueError('Head capacity or LR/seed/batch/epoch budget differs')
            else:
                if receipt['updates']!=0 or receipt['training_status']!='NO_TRAINING':raise ValueError('AlphaFit must remain closed-form, not matched optimizer training')
                if route.get('alpha')!=receipt['alpha_fit']['alpha_fit'] or not 0<=float(route['alpha'])<=1:raise ValueError('Alpha route differs from M-cal fitted clipped scalar')
            if role=='RawH' and route.get('b_route')!=roles['RawSmall']['spec']:raise ValueError('RawH does not use locked RawSmall')
        except (ValueError, KeyError, TypeError) as exc:errors[role]=str(exc)
    a,b=roles.get('AlphaFit'),roles.get('H_B2')
    if a and b:
        if a['spec'].get('b_route')!=b['spec'].get('b_route') or a['receipt']['parent_cache_fingerprints']!=b['receipt']['parent_cache_fingerprints']:
            errors['AlphaFit']='AlphaFit and H_B2 do not share both fixed parents and M/V caches'
    return errors


def metadata_plan(public, scores, input_manifest, pattern_evaluation, time_blocks):
    public=Path(public).resolve(); guard=_guard(public); h=horizon()
    official=h.metadata_plan(scores, public, input_manifest)
    lock=_json(public/'SELECTION_LOCK.json');roles,errors=map_roles(lock['route_bindings']);inputs={}
    t_scores={item['score']['run_id']:item for item in official['routes'] if item['score']['split']=='T_final'}
    for dataset in DATASETS:
        for role,item in roles[dataset].items():
            if item is None:continue
            try:
                run_id=item['run_id'];route=item['spec'];entry=t_scores[run_id]
                manifest=_json(h.inside(public,entry['manifest_relative_path']))
                if route.get('run_id')!=run_id or manifest['metadata']['route_spec']!=route:raise ValueError('Actual T route differs from frozen locked route')
                if manifest['model_sha']!=lock['allowed_route_sha'][run_id] or manifest['metadata']['route_sha']!=lock['allowed_route_sha'][run_id]:raise ValueError('Actual route identity differs from lock')
                if manifest['metadata']['artifact_bindings']!=lock['route_bindings'][run_id]['artifact_bindings']:raise ValueError('T artifact bindings differ from frozen lock')
                if route.get('artifact_sha256') and route['artifact_sha256']!=lock['allowed_artifact_sha'][run_id]:raise ValueError('Selected route artifact differs from frozen lock')
                item.update(entry=entry,manifest=manifest,receipt=None)
                if role in (*HEADS,'AlphaFit'):
                    receipt=_bound_json(public,route['calibration_receipt_public_relative_path'],route['calibration_receipt_sha256'],inputs)
                    if receipt['run_id']!=run_id or receipt['execution_status']!='COMPLETE_VALID':raise ValueError('Calibration receipt is not completed matching run')
                    if any(receipt['contract'][key]!=manifest[key] for key in ('data_sha','scaler_sha')):raise ValueError('Calibration data/scaler differs')
                    if receipt['fit_split']!='M_cal_dense_stride1' or receipt['checkpoint_split']!='V_select_strideH':raise ValueError('Calibration splits differ')
                    if role=='H_B0' and not any(route['b_route']==v['route_spec'] for v in lock['route_bindings'].values()):raise ValueError('H_B0 parent is not an exact locked standalone B0')
                    if role!='Hpast':
                        b=route['b_route'];binding=b['standard_V_binding']
                        original=_bound_json(public,binding['manifest_public_relative_path'],binding['manifest_sha256'],inputs)
                        if original['run_id']!=b['run_id'] or original['model_sha']!=b['expected_prediction_model_sha'] or original['archive_sha256']!=binding['archive_sha256']:raise ValueError('Frozen b V-parent metadata differs')
                    item['receipt']=receipt
            except (ValueError,KeyError,TypeError,OSError) as exc:errors[dataset+':'+role]='Role binding failed: '+str(exc)
        available={role:item for role,item in roles[dataset].items() if item and 'entry' in item and dataset+':'+role not in errors}
        for role,reason in check_shared_recipes(available).items():errors[dataset+':'+role]=reason
    pattern_evaluation=Path(pattern_evaluation).resolve();time_blocks=Path(time_blocks).resolve()
    for directory in (pattern_evaluation,time_blocks):
        if not directory.is_relative_to(public):raise ValueError('Auxiliary inputs must be public relative paths')
    eval_path=pattern_evaluation/'EVALUATION_MANIFEST.json';evaluation=_json(eval_path)
    if evaluation['split']!='T_final' or evaluation['registry_sha256']!=guard['EXPERIMENT_REGISTRY.csv']:raise ValueError('Pattern evaluation split/registry differs')
    if evaluation['pattern_fit_used_future_targets'] is not False or evaluation['source_sha256']!=file_sha(public/'channel_patterns.py'):raise ValueError('Frozen pattern provenance differs')
    if file_sha(pattern_evaluation.parent/'PATTERN_MANIFEST.json')!=evaluation['pattern_manifest_sha256'] or file_sha(pattern_evaluation.parent/'CHANNEL_PATTERNS.csv')!=evaluation['pattern_table_sha256']:raise ValueError('Frozen A-only pattern provenance differs')
    block_path=time_blocks/'TIME_BLOCK_MANIFEST.json';blocks=_json(block_path)
    if blocks['split']!='T_final' or blocks['registry_sha256']!=guard['EXPERIMENT_REGISTRY.csv']:raise ValueError('Time block metadata split/registry differs')
    if blocks['source_sha256']!=DEPENDENCIES['time_block_analysis.py']:raise ValueError('Original block source differs')
    for directory,manifest,names in ((pattern_evaluation,evaluation,('CHANNEL_ERRORS.csv','GROUP_ERRORS.csv')),(time_blocks,blocks,('BLOCK_RULES_PRE_SCORE.json','COHORT_BLOCKS.json'))):
        for name in names:
            p=directory/name;expected=manifest['outputs'][name]
            if file_sha(p)!=expected:raise ValueError('Auxiliary output SHA differs: '+name)
            inputs[p.relative_to(public).as_posix()]=dict(file=p.relative_to(public).as_posix(),sha256=expected,bytes=p.stat().st_size)
    for p in (eval_path,block_path,pattern_evaluation.parent/'PATTERN_MANIFEST.json',pattern_evaluation.parent/'CHANNEL_PATTERNS.csv'):
        inputs[p.relative_to(public).as_posix()]=dict(file=p.relative_to(public).as_posix(),sha256=file_sha(p),bytes=p.stat().st_size)
    rules=_json(time_blocks/'BLOCK_RULES_PRE_SCORE.json');plans=_json(time_blocks/'COHORT_BLOCKS.json')
    pairs=[]
    for dataset in DATASETS:
        for left,right in comparison_roles():
            a,b=roles[dataset][left],roles[dataset][right]
            reasons=[errors[dataset+':'+role] for role in (left,right) if dataset+':'+role in errors]
            if not reasons and a and b:
                try:compatible_descriptors(a['manifest'],b['manifest'])
                except ValueError as exc:reasons.append(str(exc))
            pairs.append(dict(dataset=dataset,left_role=left,right_role=right,left_run_id=a['run_id'] if a else '',right_run_id=b['run_id'] if b else '',status='BLOCKED_METADATA' if reasons else 'READY_METADATA_ONLY',reason='; '.join(reasons)))
    return dict(status='METADATA_ONLY_GZIP0',official=official,roles=roles,role_errors=errors,pairs=pairs,
        guard=guard,source_sha256=file_sha(__file__),dependencies=DEPENDENCIES,extra_input_bindings=list(inputs.values()),
        pattern_evaluation=pattern_evaluation.relative_to(public).as_posix(),time_blocks=time_blocks.relative_to(public).as_posix(),
        rules=rules,plans=plans,statistics_bytes_read=0,target_values_verified_here=False,private_arrays_models_GPU_access=False)


def compatible_descriptors(left,right):
    for key in ('split','cohort','data_sha','scaler_sha'):
        if left[key]!=right[key]:raise ValueError('Paired identity differs: '+key)
    for key in REFERENCE_KEYS:
        if left['arrays'][key]!=right['arrays'][key]:raise ValueError('Paired target/mask/channel/origin descriptor differs: '+key)


def statistics_record(rows,score,manifest):
    h=horizon();n,c,H=int(score['n_origins']),int(score['num_channels']),int(score['horizon'])
    origins=[];channels=[];oi={};ci={};sse=np.zeros((3,n,c),np.float64);sae=np.zeros_like(sse);counts=np.zeros((3,n,c),np.int64)
    band_index={(a,b):i for i,(_,a,b) in enumerate(h.bands(H)) if a<b}
    def collect():
        for row in rows:
            origin=h.integer(row['origin'],'origin');channel=row['channel']
            if origin not in oi:oi[origin]=len(origins);origins.append(origin)
            if channel not in ci:ci[channel]=len(channels);channels.append(channel)
            if len(origins)>n or len(channels)>c:raise ValueError('Public statistic dimensions exceed bound descriptor')
            key=(h.integer(row['horizon_start'],'start'),h.integer(row['horizon_end'],'end'))
            if key not in band_index:raise ValueError('Unknown public horizon band')
            band,i,j=band_index[key],oi[origin],ci[channel]
            sse[band,i,j]=float(row['sse']);sae[band,i,j]=float(row['sae']);counts[band,i,j]=h.integer(row['count'],'count')
            yield row
    h.aggregate_rows(collect(),score)
    order=np.argsort(origins);origins=np.asarray(origins,np.dtype(manifest['arrays']['origins']['dtype']))[order]
    for key,value in (('origins',origins),('channel_ids',np.asarray(channels,np.dtype(manifest['arrays']['channel_ids']['dtype'])))):
        actual=dict(shape=list(value.shape),dtype=str(value.dtype),sha256=byte_sha(value.tobytes()))
        if actual!=manifest['arrays'][key]:raise ValueError('Public statistic axis differs from original array descriptor: '+key)
    sse,sae,counts=sse[:,order,:],sae[:,order,:],counts[:,order,:]
    identity={key:score[key] for key in ('dataset','fold','cohort','data_sha','scaler_sha')}
    identity['run_id']=score['run_id']
    return dict(identity=identity,array_contract={key:manifest['arrays'][key] for key in REFERENCE_KEYS},
        sse=sse.sum(0),sae=sae.sum(0),counts=counts.sum(0),band_sse=sse,band_sae=sae,band_counts=counts,
        origins=origins,channel_ids=channels,horizon=H,channels=c)


def _summary_metadata(item):
    if not item:return {}
    receipt=item.get('receipt') or {};score=item['entry']['score']
    keys=('head_parameter_count','head_input_width','seed','learning_rate','effective_batch','microbatch','registered_epoch_endpoints','actual_epochs','updates','selected_updates','training_status','selected_checkpoint_sha256')
    result={key:receipt.get(key,score.get(key,'')) for key in keys}
    result.update(run_id=item['run_id'],max_epoch=score.get('max_epoch'),selected_epoch=score.get('selected_epoch'),
                  optimizer_comparison='closed_form_not_equal_optimizer_budget' if item['spec']['family']=='ALPHA' else 'recorded_recipe_only',
                  selected_updates_equal_not_assumed=True,calibration_spec_content_read=False,
                  calibration_spec_sha256=item['spec'].get('calibration_spec_sha256',''),
                  fit_receipt_public_relative_path=item['spec'].get('fit_receipt_public_relative_path',''),
                  fit_receipt_sha256=item['spec'].get('fit_receipt_sha256',''))
    result.update({key:score.get(key,'') for key in ('family','fold','latent_k','source_sha','config_sha','model_sha','data_sha','scaler_sha')})
    result.update(calibration_source_sha=receipt.get('contract',{}).get('source_sha',''),calibration_config_sha=receipt.get('contract',{}).get('config_sha',''))
    return result


def _write_csv(path,rows):
    if not rows:return
    with path.open('x',encoding='utf-8',newline='') as f:
        keys=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def _write_json(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def generate(public,scores,input_manifest,pattern_evaluation,time_blocks):
    public=Path(public).resolve();plan=metadata_plan(public,scores,input_manifest,pattern_evaluation,time_blocks)
    output=public/'diagnostics/postfinal_locked_route_pairs_20261016_v1'/('snapshot_'+str(time.time_ns()));output.mkdir(parents=True,exist_ok=False)
    ledger=[dict(r) for r in plan['pairs']];overall=[];horizons=[];time_rows=[];group_rows=[];failures=[]
    channel_path=public/plan['pattern_evaluation']/'CHANNEL_ERRORS.csv'
    with channel_path.open(encoding='utf-8',newline='') as f:channel_rows=list(csv.DictReader(f))
    channels_by_run=defaultdict(list)
    for r in channel_rows:channels_by_run[r['run_id']].append(r)
    for dataset in DATASETS:
        records={};role_failures={};roles=plan['roles'][dataset]
        for role,item in roles.items():
            if item is None or dataset+':'+role in plan['role_errors']:continue
            try:
                entry=item['entry'];path=horizon().inside(public,entry['public_stats_relative_path'])
                if file_sha(path)!=entry['public_stats_sha256']:raise ValueError('Public statistics SHA differs before aggregation')
                with gzip.open(path,'rt',encoding='utf-8',newline='') as f:
                    reader=csv.DictReader(f)
                    if tuple(reader.fieldnames or ())!=horizon().STAT_FIELDS:raise ValueError('Public statistic schema differs')
                    record=statistics_record(reader,entry['score'],item['manifest'])
                if file_sha(path)!=entry['public_stats_sha256']:raise ValueError('Public statistics changed during aggregation')
                pattern={r['channel']:r for r in channels_by_run[item['run_id']]}
                if len(pattern)!=record['channels']:raise ValueError('Channel pattern coverage differs')
                for j,name in enumerate(record['channel_ids']):
                    r=pattern[name]
                    if any(r[key]!=record['identity'][key] for key in ('cohort','data_sha','scaler_sha')):raise ValueError('Frozen channel error identity differs')
                    for key,field in (('sse','sse'),('sae','sae')):horizon().same_number(r[key],float(record[field][:,j].sum()),'channel '+key)
                    if int(r['count'])!=int(record['counts'][:,j].sum()):raise ValueError('Frozen channel observed counts differ')
                    if any(r[p+'_group'] not in {'low','high','NA'} for p in PATTERNS):raise ValueError('Unknown frozen A pattern group')
                record['patterns']=pattern;records[role]=record
            except (ValueError,OSError,KeyError,TypeError,EOFError,csv.Error) as exc:role_failures[role]=str(exc)
        for row in [r for r in ledger if r['dataset']==dataset]:
            if row['status']=='BLOCKED_METADATA':continue
            arole,brole=row['left_role'],row['right_role']
            if arole in role_failures or brole in role_failures:
                row.update(status='FAILED_PUBLIC_STATISTICS',reason='; '.join(role_failures[r] for r in (arole,brole) if r in role_failures));continue
            try:
                pair_overall=[];pair_time=[];pair_horizons=[];pair_groups=[]
                a,b=records[arole],records[brole]
                if not np.array_equal(a['band_counts'],b['band_counts']):raise ValueError('Paired observed counts differ within horizon bands')
                common={k:row[k] for k in ('dataset','left_role','right_role','left_run_id','right_run_id')}
                ar,br=_summary_metadata(roles[arole]),_summary_metadata(roles[brole])
                common.update(split='T_final',cohort=a['identity']['cohort'],data_sha=a['identity']['data_sha'],scaler_sha=a['identity']['scaler_sha'],
                    metric_scale='A_normalized_original_channels',strong_LoRA_control_not_equal_head=arole=='H_B2' and brole=='LoRA',
                    recipe_table='ROLE_RECIPES.csv',left_recipe_key=ar['run_id'],right_recipe_key=br['run_id'],
                    left_learning_rate=ar['learning_rate'],right_learning_rate=br['learning_rate'],
                    left_selected_updates=ar['selected_updates'],right_selected_updates=br['selected_updates'])
                full=block_module().paired_summary(a,b,np.arange(len(a['origins'])))
                origin_sha=a['array_contract']['origins']['sha256'];key=a['identity']['cohort']+':'+origin_sha;stored=plan['plans'][key]
                rule=plan['rules']['datasets'][dataset];actual=block_module().block_plan(a['origins'],a['horizon'],rule['period'])
                if stored!=actual or rule['block_origins']!=actual['block_origins']:raise ValueError('Actual public origins differ from frozen metadata block plan')
                pair_overall.append(dict(common,**full,overall_not_mean_of_blocks=True))
                for block in stored['blocks']:
                    pair_time.append(dict(common,**{k:v for k,v in block.items() if k not in {'indices','n_origins'}},
                        **block_module().paired_summary(a,b,block['indices']),block_origins=stored['block_origins']))
                for i,(label,start,end) in enumerate(horizon().bands(a['horizon'])):
                    aa=dict(a,sse=a['band_sse'][i],sae=a['band_sae'][i],counts=a['band_counts'][i],horizon=end-start)
                    bb=dict(b,sse=b['band_sse'][i],sae=b['band_sae'][i],counts=b['band_counts'][i],horizon=end-start)
                    pair_horizons.append(dict(common,band=label,horizon_start=start,horizon_end=end,**block_module().paired_summary(aa,bb,np.arange(len(a['origins'])))))
                for pattern in PATTERNS:
                    for group in ('low','high','NA'):
                        ids=[]
                        for j,name in enumerate(a['channel_ids']):
                            ga=a['patterns'][name][pattern+'_group'];gb=b['patterns'][name][pattern+'_group']
                            if ga!=gb:raise ValueError('Paired frozen A groups differ')
                            if ga==group:ids.append(j)
                        aa=dict(a,sse=a['sse'][:,ids],sae=a['sae'][:,ids],counts=a['counts'][:,ids],channels=len(ids))
                        bb=dict(b,sse=b['sse'][:,ids],sae=b['sae'][:,ids],counts=b['counts'][:,ids],channels=len(ids))
                        pair_groups.append(dict(common,pattern=pattern,group=group,**block_module().paired_summary(aa,bb,np.arange(len(a['origins'])))))
                overall.extend(pair_overall);time_rows.extend(pair_time);horizons.extend(pair_horizons);group_rows.extend(pair_groups)
                row.update(status='PASS_PUBLIC_SUFFICIENT_STATISTICS',reason='')
            except (ValueError,KeyError,TypeError) as exc:row.update(status='FAILED_PAIR_CONTRACT',reason=str(exc))
    for r in ledger:
        if r['status']!='PASS_PUBLIC_SUFFICIENT_STATISTICS':failures.append(r)
    if _guard(public)!=plan['guard']:raise ValueError('Frozen source/registry/lock changed during report')
    for item in plan['extra_input_bindings']:
        if file_sha(public/item['file'])!=item['sha256']:raise ValueError('Auxiliary input changed during report')
    for entry in plan['official']['routes']:
        if file_sha(public/entry['manifest_relative_path'])!=entry['manifest_sha256']:raise ValueError('Official prediction manifest changed during report')
    for values in plan['roles'].values():
        for item in values.values():
            if item and 'entry' in item:
                entry=item['entry']
                if file_sha(public/entry['public_stats_relative_path'])!=entry['public_stats_sha256']:raise ValueError('Selected public statistics changed after aggregation')
    if file_sha(scores)!=plan['official']['scores_sha256'] or file_sha(input_manifest)!=plan['official']['input_manifest_sha256']:raise ValueError('Final report inputs changed during report')
    for name,rows in (('PAIR_INDEX.csv',ledger),('OVERALL_PAIRED.csv',overall),('TIME_BLOCK_PAIRED.csv',time_rows),('HORIZON_PAIRED.csv',horizons),('PATTERN_PAIRED.csv',group_rows)):_write_csv(output/name,rows)
    recipes=[dict(dataset=ds,role=role,**_summary_metadata(item)) for ds,values in plan['roles'].items() for role,item in values.items() if item and 'entry' in item]
    _write_csv(output/'ROLE_RECIPES.csv',recipes)
    bound_plan={key:value for key,value in plan.items() if key not in {'roles','official'}}
    bound_plan['official']={key:value for key,value in plan['official'].items() if key!='routes'}
    bound_plan['role_bindings']=[dict(dataset=ds,role=role,run_id=item['run_id'],spec=item['spec'],
        manifest_relative_path=item['entry']['manifest_relative_path'],manifest_sha256=item['entry']['manifest_sha256'],
        public_stats_relative_path=item['entry']['public_stats_relative_path'],public_stats_sha256=item['entry']['public_stats_sha256'])
        for ds,values in plan['roles'].items() for role,item in values.items() if item and 'entry' in item]
    _write_json(output/'INPUT_BINDINGS.json',bound_plan)
    result=dict(status='PASS_PUBLIC_ONLY_LOCKED_ROUTE_DECOMPOSITION' if not failures else 'PARTIAL_FAILED_PAIRS',requested_pairs=52,verified_pairs=52-len(failures),failures=failures,
        table_rows=dict(overall=len(overall),time_blocks=len(time_rows),horizon=len(horizons),pattern=len(group_rows)),
        target_values_reverified=False,descriptor_and_observed_counts_verified=True,original_arrays_reaudited=False,
        private_arrays_models_GPU_access=False,new_inference_training_or_selection=False,confidence_intervals_or_independence_claim=False,
        same_head_recipe_does_not_mean_same_selected_updates=True,closed_form_alpha_not_equal_optimizer_training=True,
        prior_original_14_pairs_not_added_as_independent_samples=True,source_and_inputs_unchanged=True,
        output_sha256={p.name:file_sha(p) for p in output.iterdir() if p.is_file()})
    _write_json(output/'RESULT.json',result)
    (output/'README_KO.md').write_text('# 잠금된 T route의 공개 충분통계 비교\n\n52쌍은 T 관찰 뒤 수행한 보고용 구조 분해다. 방법과 역할은 기존 선택 잠금에 고정됐으며 이번 보조표에서 점수 순위로 비교를 빼거나 바꾸지 않았다. '
        '기존 target/mask/channel/origin descriptor와 공개 SSE/SAE/count를 확인했으며 원 target 값이나 NPZ를 다시 검산하지 않았다. '
        '시간 block은 원 metadata plan을 그대로 사용하고, 패턴 그룹은 frozen A-only 분류를 재적합하지 않았다. '
        '채널 macro를 사용하며 block 또는 band 평균을 전체 오차로 대체하지 않는다. '
        'Hpast는 b route가 없는 대조다. 동일 head 용량·요청 학습 예산과 선택 update가 같은 것은 다르다. '
        'AlphaFit은 closed-form이고 LoRA는 별도의 강한 대조이므로 동일 head 학습으로 주장하지 않는다. '
        '기존 14쌍과 겹칠 수 있어 독립 표본으로 합산하지 않는다. CI·유의성·추가 선택은 수행하지 않았다.\n',encoding='utf-8')
    return output


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('public-root','scores','input-manifest','pattern-evaluation','time-blocks'):p.add_argument('--'+name,required=True)
    p.add_argument('--execute-report',action='store_true');a=p.parse_args(argv)
    args=(a.public_root,a.scores,a.input_manifest,a.pattern_evaluation,a.time_blocks)
    if a.execute_report:print(generate(*args))
    else:
        value=metadata_plan(*args)
        print(json.dumps(dict(status=value['status'],requested_pairs=len(value['pairs']),ready_pairs=sum(r['status']=='READY_METADATA_ONLY' for r in value['pairs']),role_errors=value['role_errors'],statistics_bytes_read=0,actual_report_executed=False,source_sha256=value['source_sha256']),indent=2))


if __name__=='__main__':main()
