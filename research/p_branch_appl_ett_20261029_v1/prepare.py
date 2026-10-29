"""Audit existing bounded inputs and seal the approved sixty-run branch study."""
import argparse
import csv
from datetime import datetime, timezone
import difflib
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np

from . import runtime


REPO = runtime.REPO
REPORT = runtime.REPORT
CODE = Path(__file__).parent
METHODS = ('P_TIME_ONLY', 'P_CHANNEL_ONLY')
SEEDS = list(range(92601, 92611))
INSTRUCTIONS = Path('C:/Users/User/Downloads/p_branch_appl_ett_executor_instructions_resilient.md')
ORIGINAL = INSTRUCTIONS.with_name('p_branch_appl_ett_executor_instructions.md')
OLD_RUNS = REPO/'reports/comparative_ablation_followup_20261027_v1/RUNS.csv'
AXES = {
    'appliances': dict(C=26, K=10, fit_end=9867, refit_end=11841, V_start=13814, T_start=15788, shape=(41,48,26)),
    'ETTh1': dict(C=7, K=4, fit_end=8710, refit_end=10452, V_start=12194, T_start=13936, shape=(36,48,7)),
    'ETTh2_safe17400': dict(C=7, K=4, fit_end=8700, refit_end=10440, V_start=12180, T_start=13920, shape=(36,48,7)),
}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    text = value if isinstance(value, str) else json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding='utf8') != text:
            raise FileExistsError('Refusing to replace existing registration input: '+str(path))
        return
    with path.open('x', encoding='utf8', newline='\n') as stream:
        stream.write(text)


def header(archive, key):
    with archive.open(key+'.npy') as stream:
        version = np.lib.format.read_magic(stream)
        if version not in ((1,0),(2,0)):
            raise ValueError('Unsupported NPY header version')
        reader = np.lib.format.read_array_header_1_0 if version == (1,0) else np.lib.format.read_array_header_2_0
        return reader(stream)


def verified_arrays(row, canonical=None):
    axes = AXES[row['dataset']]
    path = Path(row['prediction_path'])
    if runtime.sha(path) != row['prediction_sha256']:
        raise ValueError('Prediction fingerprint changed: '+row['job_id'])
    with zipfile.ZipFile(path) as archive:
        for key in ('pred_normalized','target_normalized','observed'):
            shape, order, dtype = header(archive, key)
            if tuple(shape) != axes['shape'] or dtype.hasobject:
                raise ValueError('Bounded reference header differs: '+row['job_id'])
        for key in ('mean','std'):
            shape, order, dtype = header(archive, key)
            if tuple(shape) != (axes['C'],) or dtype.hasobject:
                raise ValueError('Required normalization metadata differs: '+row['job_id'])
    with np.load(path, allow_pickle=False) as archive:
        origins = archive['origins']
        expected = np.arange(axes['V_start'], axes['T_start']-48+1,48)
        if not np.array_equal(origins, expected) or int(origins.max())+48 >= axes['T_start']:
            raise ValueError('Reference origin grid overlaps protected T')
        arrays = {key: archive[key].copy() for key in ('pred_normalized','target_normalized','observed',
                  'origins','channel_ids','mean','std')}
    if canonical is not None:
        for key in ('target_normalized','observed','origins','channel_ids','mean','std'):
            x,y = arrays[key],canonical[key]
            if not np.array_equal(x,y,equal_nan=x.dtype.kind in 'fc'):
                raise ValueError('Reference cohort/scaler differs: '+row['job_id']+' '+key)
    p,y,mask = arrays['pred_normalized'],arrays['target_normalized'],arrays['observed']
    if mask.dtype != np.bool_ or not np.isfinite(p).all() or not np.isfinite(y[mask]).all():
        raise ValueError('Reference observations are not finite')
    errors = p.astype(np.float64)-y.astype(np.float64)
    mses,maes = [],[]
    for channel in range(axes['C']):
        e = errors[:,:,channel][mask[:,:,channel]].ravel()
        if not len(e):
            raise ValueError('Reference channel has no observations')
        mses.append(math.fsum(float(z)*float(z) for z in e)/len(e))
        maes.append(math.fsum(abs(float(z)) for z in e)/len(e))
    mse,mae = math.fsum(mses)/len(mses),math.fsum(maes)/len(maes)
    error = max(abs(mse-float(row['MSE'])),abs(mae-float(row['MAE'])))
    if error > 1e-9:
        raise ValueError('Reference FP64 metrics differ: '+row['job_id'])
    return arrays,dict(job_id=row['job_id'],prediction_sha256=row['prediction_sha256'],
          shape=list(p.shape),MSE=mse,MAE=mae,maximum_metric_error=error,
          normalization_arrays_present=True,fully_observed=bool(mask.all()),T_numeric_values_parsed=0)


def audit_inputs():
    if (REPORT/'INPUT_AUDIT.json').exists():
        audit = runtime.read(REPORT/'INPUT_AUDIT.json')
        for path,digest in audit['bindings'].items():
            if runtime.sha(path) != digest:
                raise ValueError('Previously audited input changed: '+path)
        return audit
    if subprocess.run(['git','merge-base','--is-ancestor','2d3ffa94','HEAD'],cwd=REPO).returncode:
        raise ValueError('Required base commit is absent')
    old_protocol = runtime.read(REPO/'reports/jena_p_branch_20261027_v1/PREREGISTRATION.json')
    for path,digest in old_protocol['registered_sources'].items():
        if runtime.sha(REPO/path) != digest:
            raise ValueError('Original Jena registered source changed')
    with OLD_RUNS.open(encoding='utf-8-sig',newline='') as stream:
        all_rows = list(csv.DictReader(stream))
    refs = [r for r in all_rows if r['dataset'] in AXES and r['method_id'] in ('OURS','A1','A2')]
    raw = [r for r in all_rows if r['dataset'] in AXES and r['method_id']=='RAW']
    if len(refs)!=90 or len(raw)!=3:
        raise ValueError('Expected ninety learned references and three raw rows')
    for dataset,axes in AXES.items():
        for method in ('OURS','A1','A2'):
            group=[r for r in refs if r['dataset']==dataset and r['method_id']==method]
            if sorted(int(r['seed']) for r in group)!=SEEDS:
                raise ValueError('Reference seed roster differs')
            if any(int(float(r['K']))!=axes['K'] or int(float(r['normalization_fit_end']))!=axes['refit_end'] for r in group):
                raise ValueError('Reference K/A60 range differs')
    bindings={str(OLD_RUNS.resolve()):runtime.sha(OLD_RUNS)}
    checks,canonical=[],{}
    for dataset in AXES:
        row=next(r for r in refs if r['dataset']==dataset and r['method_id']=='OURS' and int(r['seed'])==92603)
        canonical[dataset],_=verified_arrays(row)
    for row in refs:
        _,check=verified_arrays(row,canonical[row['dataset']])
        checks.append(check)
        bindings[str(Path(row['prediction_path']).resolve())]=row['prediction_sha256']
    modules=runtime.load_modules()
    tasks=[]
    for dataset,axes in AXES.items():
        task=runtime.load_task(dataset)
        runtime.assert_protected_T(dict(job={'dataset':dataset},task_access=task['_campaign_access']))
        mean,std,_=modules['data'].fit_statistics(task,axes['refit_end'])
        if not np.array_equal(mean,canonical[dataset]['mean']) or not np.array_equal(std,canonical[dataset]['std']):
            raise ValueError('A60 physical scaler differs from references')
        access=task['_campaign_access']
        for label in ('prefix','config'):
            key=label+'_path'
            bindings[str(Path(access[key]).resolve())]=runtime.sha(access[key])
        tasks.append(dict(dataset=dataset,axes=axes,prefix_rows=len(task['physical']),
               split_ranges=task['split_ranges'],data_sha256=task['data_sha256'],
               array_sha256=task['array_sha256'],access=access,
               A_fit_scaler_sha=modules['fit'].scaler_fingerprint(*modules['data'].fit_statistics(task,axes['fit_end'])[:2]),
               A60_scaler_sha=modules['fit'].scaler_fingerprint(mean,std),T_numeric_values_parsed=0))
    import torch
    gpu_csv=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total,driver_version','--format=csv,noheader'],text=True).strip()
    expected=runtime.read(REPO/'reports/comparative_ablation_followup_20261027_v1/ENVIRONMENT_VALIDATION.json')['environment']
    environment=dict(python=sys.version.split()[0],torch=torch.__version__,numpy=np.__version__,cuda_build=torch.version.cuda,
                     GPU=gpu_csv.split(',')[0].strip(),gpu_csv=gpu_csv)
    if any(environment[k]!=expected[k] for k in environment):
        raise ValueError('Execution environment differs from original references')
    foundation=modules['gpu'].FrozenBolt(runtime.DEFAULT_ASSET/runtime.MODEL_REL,48,
                   row_microbatch=128,checkpoint_inputs=False,device='cpu')
    if foundation.initial_hash!=runtime.F_SHA:
        raise ValueError('Actual CPU-inspected frozen foundation changed')
    foundation.assert_frozen_unchanged()
    foundation_check=dict(status='PASS',state_sha256=foundation.initial_hash,
        parameters=sum(p.numel() for p in foundation.model.parameters()),device='cpu',
        instance_norm_source_sha256=foundation.instance_norm_source_sha256)
    del foundation
    gc.collect()
    if torch.cuda.is_initialized():
        raise ValueError('Input audit unexpectedly initialized CUDA')
    write(REPORT/'REFERENCE_RUNS.json',dict(status='VERIFIED',source=str(OLD_RUNS),source_sha256=runtime.sha(OLD_RUNS),
          rows=refs,raw_rows=raw,checks=checks,reference_rows=90,protected_T_numeric_values_read=0))
    write(REPORT/'DATA_REGISTRY.json',dict(status='PASS',datasets=tasks,protected_T_numeric_values_read=0))
    write(REPORT/'ENVIRONMENT_VALIDATION.json',dict(status='PASS',environment=environment,
          all_environment_fields_equal=True,foundation=foundation_check,CUDA_initialized=False,training_executed=False))
    audit=dict(status='PASS',created_utc=stamp(),reference_count=90,dataset_count=3,
          maximum_FP64_metric_error=max(c['maximum_metric_error'] for c in checks),
          bindings=bindings,protected_T_numeric_values_read=0,CUDA_initialized=False,training_executed=False)
    write(REPORT/'INPUT_AUDIT.json',audit)
    return audit


def external_input_bindings(audit):
    paths = {Path(path).resolve() for path in audit['bindings']}
    paths.add(runtime.DEFAULT_SOURCE/'reports/k_tau_rule_20261023_v1/CURRENT_DATA_FINGERPRINTS.json')
    for dataset, axes in AXES.items():
        paths.add(runtime.DEFAULT_SOURCE/'reports/b2_dataset_search_20261019_v1/datasets'/dataset/'TRAIN_IDENTITY.json')
        paths.add(runtime.DEFAULT_SOURCE/'.cache/k_tau_rule_20261023_v1/results'/
                  f'KTAU_main_{dataset}_K{axes["K"]}_s92603.json')
    model = runtime.DEFAULT_ASSET/runtime.MODEL_REL
    model_files = [path for path in model.rglob('*') if path.is_file()]
    if not model_files:
        raise ValueError('Pinned local foundation snapshot is missing')
    paths.update(path.absolute() for path in model_files)
    paths.update(path.resolve() for path in model_files)
    return dict(status='BOUND_BEFORE_TRAINING', created_utc=stamp(),
                scope='Previously audited bounded prefixes, reference archives, historical identity/scaler provenance and local foundation snapshot; opaque SHA only.',
                files=[dict(path=str(path),sha256=runtime.sha(path),bytes=path.stat().st_size)
                       for path in sorted(paths)],protected_T_numeric_values_read=0)


def preserved_paths(references, external):
    paths=set()
    for folder in ('research','reports'):
        for path in (REPO/folder).rglob('*'):
            if not path.is_file() or path.is_relative_to(CODE) or path.is_relative_to(REPORT):
                continue
            if any(part in {'.validation_dependencies','__pycache__','.pytest_cache'} for part in path.parts):
                continue
            if path.name.startswith('~$') or path.suffix in {'.tmp','.pyc'}:
                continue
            paths.add(path.resolve())
    paths.update(Path(r['prediction_path']).resolve() for r in references)
    paths.update(Path(row['path']).absolute() for row in external['files'])
    return sorted(paths)


def seal():
    existing=REPORT/'REGISTRATION_SEAL.json'
    if existing.exists():
        value=runtime.read(existing)
        for name,digest in value['files'].items():
            if runtime.sha(REPORT/name)!=digest:
                raise ValueError('Existing seal input changed')
        return dict(status='EXISTING_SEAL_VERIFIED',seal_sha256=runtime.sha(existing))
    audit=audit_inputs()
    tests=runtime.read(REPORT/'CPU_TESTS.json')
    if tests.get('status')!='PASS' or tests.get('exit_code')!=0:
        raise ValueError('Successful current CPU fixture evidence required before sealing')
    sources={p.relative_to(REPO).as_posix():runtime.sha(p) for p in sorted(CODE.rglob('*.py'))}
    if tests.get('source_sha256')!=sources:
        raise ValueError('CPU tests do not bind the final source snapshot')
    external=external_input_bindings(audit)
    write(REPORT/'EXTERNAL_INPUT_BINDINGS.json',external)
    baseline=dict(created_utc=stamp(),scope='Existing research/report artifacts, reused predictions, bounded prefixes, historical input provenance and local foundation snapshot; excludes new campaign, mutable interpreter caches, validation tooling and Office lock files; opaque hashing only.',
          files=[dict(path=str(p),sha256=runtime.sha(p),bytes=p.stat().st_size) for p in preserved_paths(runtime.read(REPORT/'REFERENCE_RUNS.json')['rows'],external)])
    write(REPORT/'PRESERVATION_BASELINE.json',baseline)
    jobs=[dict(job_id=f'PBA_{method}_{dataset}_K{axes["K"]}_s{seed}',method_id=method,dataset=dataset,K=axes['K'],seed=seed,extended=False)
          for seed in SEEDS for dataset,axes in AXES.items() for method in METHODS]
    execution=dict(admission_deadline=None,original_admission_target='2026-10-30T09:00:00+09:00',
        deadline_override_authority='Latest direct user goal: 지시문 이행해줘 시간 초과됐다고 멈추지 말고 잘 안됬으면 다시하고',
        initial_estimates_seconds=dict(appliances=647.449,ETTh1=852.185,ETTh2_safe17400=873.753),
        hard_wall_seconds=dict(appliances=2700.,ETTh1=3408.740,ETTh2_safe17400=3495.012),
        admission_margin_seconds=60,retries=1,resource_wait_seconds=60,max_controller_restarts=3,heartbeat_seconds=10,
        reestimate_multiplier=1.35,concurrency=1,
        technical_recovery_waves='Attempts1/2 initially; further positive attempts require a prospective ROOT-approved technical recovery authorization binding original seal, exact job and source hashes after diagnosed infrastructure remediation. Never retry a valid COMPLETE result or retry because V is unfavorable.')
    write(REPORT/'RUN_PLAN.json',dict(campaign=runtime.NAME,jobs=jobs,extension_jobs=[],training_deadline=None,
          original_admission_target=execution['original_admission_target'],execution_policy=execution))
    documents=['PREREGISTRATION.json','PREREGISTRATION.md','RUN_PLAN.json','REFERENCE_RUNS.json',
               'PRESERVATION_BASELINE.json','DATA_REGISTRY.json','ENVIRONMENT_VALIDATION.json','INPUT_AUDIT.json',
               'CPU_TESTS.json','EXTERNAL_INPUT_BINDINGS.json','CODE_DIFF.patch','SOURCE_INSTRUCTIONS_ORIGINAL.md','SOURCE_INSTRUCTIONS_RESILIENT.md']
    write(REPORT/'SOURCE_INSTRUCTIONS_ORIGINAL.md',ORIGINAL.read_text(encoding='utf8'))
    write(REPORT/'SOURCE_INSTRUCTIONS_RESILIENT.md',INSTRUCTIONS.read_text(encoding='utf8'))
    patches=[]
    original_dir=REPO/'research/jena_p_branch_20261027_v1'
    for path in sorted(CODE.rglob('*.py')):
        relative=path.relative_to(CODE)
        old=original_dir/relative
        previous=old.read_text(encoding='utf8').splitlines(True) if old.exists() else []
        patches.extend(difflib.unified_diff(previous,path.read_text(encoding='utf8').splitlines(True),
                       fromfile=('research/jena_p_branch_20261027_v1/'+relative.as_posix()) if old.exists() else '/dev/null',
                       tofile=path.relative_to(REPO).as_posix()))
    write(REPORT/'CODE_DIFF.patch',''.join(patches))
    protocol=dict(campaign=runtime.NAME,status='PREREGISTERED_LOCAL_HASH_SEAL',created_utc=stamp(),
        authority=execution['deadline_override_authority'],base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=REPO).strip(),
        registered_sources=sources,registration_documents=documents,execution_policy=execution,controller_policy=execution,
        datasets=list(AXES),data_bounds=AXES,seeds=SEEDS,base_runs=60,extension_jobs=0,
        methods=dict(P_TIME_ONLY='Original temporal norm/residual MLP; feature branch removed.',
                     P_CHANNEL_ONLY='Original feature norm/residual MLP; temporal branch removed.'),
        trainable_parameters=runtime.EXPECTED_PARAMETERS,
        initialization='Same-seed original Full factory; keep retained branch tensors without reinitialization; shared LayerNorm and E/D remain.',
        backbone=dict(F_SHA=runtime.F_SHA,model_path=str(runtime.DEFAULT_ASSET/runtime.MODEL_REL)),
        training=dict(optimizer='AdamW',lr=.001,weight_decay=0,gradient_clip=1,effective_batch=16,microbatch=4,
                      F_row_microbatch=128,checkpoint_inputs=False,revin=False,dtype='float32',L=512,H=48,
                      base_epochs=[0,.25,1,3],extension_epochs=[],selection='Minimum A_stop MSE, earlier exact tie; sealed before fresh A60 refit and V.'),
        extension_policy=dict(status='NOT_RUN_SCOPE',trigger_cap_selected_count=5,extend=False,provisional='Any complete10-seed party selects cap3 in>=5seeds; incomplete primary groups excluded.'),
        analysis=dict(primary_metric='V normalized MSE',secondary_metric='V normalized MAE',
                      decision='Original median/max-IQR descriptive rule; no significance/equivalence inference.',
                      scope='Exploratory repeated development V; scope selected after existing results; no protected T generalization.',
                      primary_group_size=10,group_axes=['dataset','method_id'],paired_axis='seed'),
        expected_reference_effects=dict(appliances=dict(A1=.6,A2=-1.6),ETTh1=dict(A1=-.1,A2=-.8),ETTh2_safe17400=dict(A1=-5.,A2=.5)),
        preregistered_predictions=dict(P_TIME_ONLY='All three UNCLEAR',P_CHANNEL_ONLY='Appliances/ETTh1 UNCLEAR; ETTh2 may have lower median and select cap3; never used for selection.'),
        input_audit_sha256=runtime.sha(REPORT/'INPUT_AUDIT.json'),source_instructions_sha256=runtime.sha(INSTRUCTIONS),
        external_input_bindings_sha256=runtime.sha(REPORT/'EXTERNAL_INPUT_BINDINGS.json'),
        preservation_policy='Initial and final full SHA256 required. Every process first scan fully hashes baseline; routine repeated scans use only process-local validated SHA/stat cache, with critical scientific sources and references freshly hashed. A metadata change forces rehash; cache usage is reported.',
        publication=dict(commit_before_training=False,allowed_roots=[str(CODE.relative_to(REPO)),str(REPORT.relative_to(REPO))],
                         command='git push origin HEAD:main',force=False,include_partial=True,PPT_edited=False),
        deliverables=['60 job statuses and valid completed predictions privately retained','CSV summaries/paired contrasts/status',
                      '3panel PNG/SVG','SLIDE_NUMBERS.md','PREDICTION_CHECK.md','RESULTS_KO.md',
                      'independent numerical/preservation audit','results-only ZIP exact membership CRC SHA size','normal final main publication'])
    write(REPORT/'PREREGISTRATION.json',protocol)
    markdown='# P branch Appliances/ETT preregistration\n\n'+INSTRUCTIONS.read_text(encoding='utf8')
    markdown+='\n\n## Latest direct authorization overrides\n\nThe user explicitly requested execution and instructed not to stop because time elapsed. The original09:00 admission cutoff is disabled; there is no global campaign time cutoff. Fixed per-attempt watchdogs and original numerical recipe remain. Diagnosed infrastructure-only recovery waves must be authorized and recorded prospectively; valid completed or unfavorable runs are never repeated. No commit before training. Final publication is limited to the two new campaign folders.\n'
    write(REPORT/'PREREGISTRATION.md',markdown)
    write(existing,dict(created_utc=stamp(),method='Local immutable-before-training SHA256 seal; no Git commit claimed.',
                        files={name:runtime.sha(REPORT/name) for name in documents}))
    runtime.guard_preserved(full=True)
    return dict(status='SEALED',base_jobs=60,extension_jobs=0,preserved_files=len(baseline['files']),
                reference_rows=90,global_time_cutoff=None,seal_sha256=runtime.sha(existing))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args()
    value=audit_inputs() if args.audit_only else seal()
    print(json.dumps(value if not args.audit_only else {k:v for k,v in value.items() if k!='bindings'},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
