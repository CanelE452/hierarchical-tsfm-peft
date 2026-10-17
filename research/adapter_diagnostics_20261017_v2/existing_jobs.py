"""Explicit handlers for existing, hash-bound diagnostics. Import performs no I/O."""
import copy
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
from types import SimpleNamespace


OLD = 'informs_b2_why_where_20261017'
BROAD = 'informs_broad_validation_20261015_v1'
SEEDS = (92601, 92602)
FAMILIES = ('B0', 'B2', 'B2_CAP_POINTWISE')
EPOCHS = (0., .25, 1., 3.)
UPDATES = (0, 156, 622, 1863)
EXPOSURES = (0, 2481, 9923, 29769)
BOUNDS = dict(A_fit=(0, 10482), A_stop=(10482, 12579), V_select=(14675, 16772))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _job_id(job):
    value = job['job_id']
    if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError('Unsafe job_id')
    return value


def _identity(job):
    keys = ('job_id', 'question_id', 'family', 'seed', 'endpoint_index', 'split', 'batch',
            'K', 'device', 'scientific_identity')
    identity = {k: job.get(k) for k in keys}
    identity['handler_sha256'] = sha(__file__)
    return hashlib.sha256(json.dumps(identity, sort_keys=True,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _reference(path, ctx):
    path = Path(path).resolve()
    for root in ('report', 'cache', 'repo', 'asset'):
        if root in ctx and path.is_relative_to(Path(ctx[root]).resolve()):
            return root + '://' + path.relative_to(Path(ctx[root]).resolve()).as_posix()
    raise ValueError('Artifact outside registered roots')


def _resolve(reference, ctx):
    root, relative = reference.split('://', 1)
    path = (Path(ctx[root])/relative).resolve()
    if not path.is_relative_to(Path(ctx[root]).resolve()):
        raise ValueError('Artifact reference escapes root')
    return path


def _receipt_path(job, ctx):
    return Path(ctx['report'])/'jobs'/(_job_id(job)+'_handler.json')


def _deadline(ctx):
    return ctx.get('job_deadline', ctx['deadline'])


class RSSSampler:
    """Observed process RSS peak; 5ms sampling can miss shorter transients."""
    def __init__(self, read_rss, interval_seconds=.005):
        self.read_rss = read_rss
        self.interval = interval_seconds
        self.peak = self.samples = 0
        self.stop_event = threading.Event()
        self.thread = None
        self.started = None

    def sample(self):
        self.peak = max(self.peak, int(self.read_rss()))
        self.samples += 1

    def start(self):
        self.started = time.perf_counter()
        self.sample()
        def observe():
            while not self.stop_event.wait(self.interval):
                self.sample()
        self.thread = threading.Thread(target=observe, name='diagnostic-cost-rss', daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        self.sample()
        return dict(process_RSS_sampled_peak_bytes=self.peak, RSS_sample_count=self.samples,
                    RSS_sampling_interval_seconds=self.interval,
                    RSS_scope='physical parity + 10 warmup + 20 timed + 20 stage_probe; excludes cold model load',
                    RSS_scope_seconds=None if self.started is None else time.perf_counter()-self.started,
                    RSS_peak_is_sampled_not_continuous=True)


def save_receipt(job, ctx, value):
    paths = [Path(p) for p in value.get('artifact_paths', [])]
    if value['status'] in ('COMPLETE', 'REUSED_VERIFIED') and not paths:
        raise ValueError('Completed job needs an existing artifact')
    bindings = [dict(path=_reference(p, ctx), sha256=sha(p)) for p in paths]
    public = dict(value, artifact_paths=[v['path'] for v in bindings],
                  artifact_hashes=bindings, scientific_identity_sha256=_identity(job),
                  job_id=job['job_id'])
    atomic_json(_receipt_path(job, ctx), public)


def completed_receipt(job, ctx):
    path = _receipt_path(job, ctx)
    if not path.exists():
        return None
    saved = read(path)
    if saved['scientific_identity_sha256'] != _identity(job):
        raise ValueError('Completed cache scientific identity changed')
    if saved['status'] not in ('COMPLETE', 'REUSED_VERIFIED'):
        return None
    paths = []
    for entry in saved['artifact_hashes']:
        artifact = _resolve(entry['path'], ctx)
        if not artifact.exists() or sha(artifact) != entry['sha256']:
            raise ValueError('Completed artifact missing or changed')
        paths.append(str(artifact))
    if not paths:
        raise ValueError('Completed receipt lacks artifact')
    return dict(status='REUSED_VERIFIED', reason='Same scientific identity and all saved artifact hashes verified',
                artifact_paths=paths, result=saved.get('result'))


def score_arrays(prediction, target, mask):
    import numpy as np
    p, y, m = np.asarray(prediction), np.asarray(target), np.asarray(mask)
    if p.ndim != 3 or p.shape != y.shape or p.shape != m.shape:
        raise ValueError('Expected aligned origin,horizon,channel arrays')
    if m.dtype != np.bool_:
        raise ValueError('An explicit boolean observed mask is required')
    counts = m.sum((0, 1), dtype=np.int64)
    if not np.all(counts > 0):
        raise ValueError('Every registered channel must have observed targets')
    if not np.isfinite(p[m]).all() or not np.isfinite(y[m]).all():
        raise ValueError('Nonfinite observed prediction or target')
    difference = np.where(m, p.astype(np.float64)-y.astype(np.float64), 0.)
    sse = np.sum(difference*difference, (0, 1), dtype=np.float64)
    sae = np.sum(np.abs(difference), (0, 1), dtype=np.float64)
    return dict(MSE=math.fsum((sse/counts).tolist())/len(counts),
                MAE=math.fsum((sae/counts).tolist())/len(counts), observed=int(counts.sum()),
                channel_sse=sse.tolist(), channel_sae=sae.tolist(), channel_counts=counts.tolist(),
                horizon_sse=np.sum(difference*difference, 0, dtype=np.float64).tolist(),
                horizon_sae=np.sum(np.abs(difference), 0, dtype=np.float64).tolist(),
                horizon_counts=m.sum(0, dtype=np.int64).tolist(),
                origin_sse=np.sum(difference*difference, 1, dtype=np.float64).tolist(),
                origin_sae=np.sum(np.abs(difference), 1, dtype=np.float64).tolist(),
                origin_counts=m.sum(1, dtype=np.int64).tolist())


def _array_fingerprint(array):
    import numpy as np
    return dict(shape=list(array.shape), dtype=str(array.dtype),
                sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest())


def validate_cohort(values, reference):
    import numpy as np
    for key in ('target_normalized', 'observed', 'origins', 'channel_ids'):
        a, b = values[key], reference[key]
        if (a.shape != b.shape or a.dtype != b.dtype or
                not np.array_equal(a, b, equal_nan=a.dtype.kind in 'fc')):
            raise ValueError('Evaluation cohort mismatch: '+key)
    if values['pred_normalized'].shape != reference['target_normalized'].shape:
        raise ValueError('Prediction axes mismatch')


def _archive(path, expected_sha=None, fingerprints=None):
    import numpy as np
    if expected_sha and sha(path) != expected_sha:
        raise ValueError('Prediction archive SHA mismatch')
    with np.load(path, allow_pickle=False) as source:
        if set(source.files) != {'pred_normalized', 'target_normalized', 'observed', 'origins', 'channel_ids'}:
            raise ValueError('Unexpected prediction archive schema')
        values = {key: source[key] for key in source.files}
    for key, expected in (fingerprints or {}).items():
        if _array_fingerprint(values[key]) != expected:
            raise ValueError('Array fingerprint mismatch: '+key)
    return values


def _old_run(seed, family):
    if seed not in SEEDS or family not in FAMILIES:
        raise ValueError('Unregistered Jena condition')
    if family == 'B2_CAP_POINTWISE':
        return Path('runs')/f'cap_s{seed}'
    namespace = 'postT_V_only_cpu_lag_20261016_v1' if seed == 92601 else 'priority_core_followup_20261016_v1'
    run = f'postT_cpu_jena_E3_{family}_k6_s92601_v1' if seed == 92601 else f'priority_seed2_jena_{family}_k6_s92602_v1'
    return Path(namespace)/'runs'/run


def _run_root(ctx, seed, family, public=False):
    old = OLD if family == 'B2_CAP_POINTWISE' else BROAD
    root = Path(ctx['repo'])/('reports' if old == OLD else 'research')/old if public else Path(ctx['asset'])/'.cache'/old
    return root/_old_run(seed, family)


def _preserved_sha(path, ctx):
    index = read(Path(ctx['asset'])/'.cache'/OLD/'USER_STOP_PRESERVED_FILES_v1.json')
    resolved = Path(path).resolve()
    entry = next((r for r in index if Path(r['path']).resolve() == resolved), None)
    if entry is None or sha(path) != entry['sha256']:
        raise ValueError('Preserved asset identity mismatch')
    return entry['sha256']


def cap_rescore(job, ctx):
    seed = int(job['seed'])
    old_public = _run_root(ctx, seed, 'B2_CAP_POINTWISE', True)/'SELECTED_V_RESULT.json'
    native = read(old_public)
    if (native['seed'] != seed or native['model'] != 'B2_CAP_POINTWISE' or
            native['C'] != 21 or native['K'] != 6 or native['fit'] != 'fresh_A60'):
        raise ValueError('CAP selected scientific identity differs')
    archive = _run_root(ctx, seed, 'B2_CAP_POINTWISE')/'selected_V.npz'
    values = _archive(archive, native['private_prediction_archive_sha256'], native['array_fingerprints'])
    if values['pred_normalized'].shape != (43, 48, 21):
        raise ValueError('CAP selected axes differ from registered Jena V43')
    old = _old_run(seed, 'B2')
    reference_archive = Path(ctx['asset'])/'.cache'/BROAD/old.parent.parent/'predictions'/old.name/'V_select.npz'
    reference_manifest = Path(ctx['repo'])/'research'/BROAD/old.parent.parent/'predictions'/old.name/'V_select'/'PREDICTION_MANIFEST.json'
    manifest = read(reference_manifest)
    reference = _archive(reference_archive, manifest['archive_sha256'], manifest['arrays'])
    validate_cohort(values, reference)
    if native['scaler_sha256'] != manifest['scaler_sha']:
        raise ValueError('CAP selected scaler mismatch')
    score = score_arrays(values['pred_normalized'], values['target_normalized'], values['observed'])
    differences = {k: abs(score[k]-native[k]) for k in ('MSE', 'MAE')}
    if max(differences.values()) > 1e-12:
        raise ValueError('Independent FP64 versus native FP64 score mismatch')
    result = dict(score, seed=seed, family='B2_CAP_POINTWISE', dataset='jena', C=21, K=6,
                  eval='V_select', metric_scale='canonical_A60', native_absolute_difference=differences,
                  tolerance=1e-12, subtraction_dtype='float64', accumulation_dtype='float64',
                  stored_precision='float32', new_forward=False, scaler_sha256=manifest['scaler_sha'],
                  source_archive_sha256=sha(archive), source_metrics_sha256=sha(old_public),
                  reference_manifest_sha256=sha(reference_manifest), array_fingerprints=native['array_fingerprints'])
    output = Path(ctx['report'])/'scores'/(_job_id(job)+'.json')
    atomic_json(output, result)
    return dict(status='COMPLETE', reason='Independent stored-array FP64 rescore and exact B2 cohort reconciliation',
                artifact_paths=[str(output), str(archive), str(reference_archive), str(old_public), str(reference_manifest)], result=result)


def endpoint_spec(job):
    index, seed, family, split = job['endpoint_index'], int(job['seed']), job['family'], job['split']
    if type(index) is not int or index not in range(4) or seed not in SEEDS or family not in FAMILIES or split not in BOUNDS:
        raise ValueError('Unregistered common endpoint condition')
    return dict(epoch=EPOCHS[index], updates=UPDATES[index], exposures=EXPOSURES[index],
                metric_scale='canonical_A60' if split == 'V_select' else 'search_A50',
                stride=1 if split == 'A_fit' else 48)


def _modules(ctx):
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '2'
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['INFORMS_ASSET_ROOT'] = str(ctx['asset'])
    os.environ['INFORMS_CACHE_ROOT'] = str(Path(ctx['asset'])/'.cache'/BROAD)
    campaign = Path(ctx['repo'])/'research'/BROAD
    code = Path(ctx['repo'])/'reports'/OLD/'code'
    for path in (str(campaign), str(code)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import numpy as np
    import torch
    import psutil
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(2)
    result = dict(np=np, torch=torch, psutil=psutil, campaign=campaign,
                  private=Path(ctx['asset'])/'.cache'/OLD, public=Path(ctx['report']),
                  old_cache=Path(ctx['asset'])/'.cache'/BROAD, deadline=_deadline(ctx))
    result['blas_controller'] = threadpool_limits(limits=2, user_api='blas')
    for name, key in (('fit_base', 'fit'), ('training', 'training'), ('data_contract', 'data'),
                      ('deployment', 'deployment'), ('evaluation', 'evaluation'), ('mechanism_runner', 'legacy')):
        result[key] = importlib.import_module(name)
    for key in ('fit', 'training', 'data', 'deployment', 'evaluation'):
        if Path(result[key].__file__).resolve().parent != campaign.resolve():
            raise ValueError('Scientific module import root changed')
    return result


def _foundation(ctx, modules, horizon, device='cpu'):
    pool = modules['deployment'].FoundationPool(Path(ctx['asset'])/modules['legacy'].MODEL_REL, device)
    foundation = pool.get(horizon)
    if foundation.initial_hash != modules['legacy'].F_SHA:
        pool.close()
        raise ValueError('Foundation weights identity changed')
    return pool, foundation


def _jena(ctx, modules):
    return modules['legacy'].load_jena(SimpleNamespace(asset=Path(ctx['asset'])), modules)[0]


def _cp0_reuse(job, ctx, specification):
    if job['family'] != 'B2_CAP_POINTWISE' or job['endpoint_index'] != 0:
        return None
    seed = job['seed']
    audit_path = Path(ctx['asset'])/'.cache'/OLD/'CAP_INITIALIZATION_AUDIT_RESULT.json'
    audit = read(audit_path)
    row = next(r for r in audit['rows'] if r['seed'] == seed)
    if audit['status'] != 'PASS_CAP_INITIALIZATION_BOTH_SEEDS' or not all(row['checks'].values()):
        raise ValueError('CAP CP0 functional equivalence audit failed')
    cap_source = read(_run_root(ctx, seed, 'B2_CAP_POINTWISE', True)/'SOURCE_CONTRACT.json')
    if audit['cap_control_sha256'] != cap_source['cap_control_sha256']:
        raise ValueError('CAP CP0 architecture source differs from audit')
    for family in ('B0', 'B2'):
        checkpoint = _run_root(ctx, seed, family)/'search'/'checkpoint_000000.pt'
        if sha(checkpoint) != row['original_CP0'][family]['sha256']:
            raise ValueError('Audited original CP0 changed')
    cap_checkpoint = _run_root(ctx, seed, 'B2_CAP_POINTWISE')/'search'/'checkpoint_000000.pt'
    _preserved_sha(cap_checkpoint, ctx)
    score_path = Path(ctx['repo'])/'reports'/OLD/'same_update'/f's{seed}'/f'B0_e0_{job["split"]}.json'
    result = read(score_path)
    if result['metric_scale'] != specification['metric_scale'] or result['updates'] != 0:
        raise ValueError('CP0 reused scale/progress mismatch')
    result.update(model='B2_CAP_POINTWISE', params=35708, checkpoint_sha256=sha(cap_checkpoint),
                  source_score_sha256=sha(score_path), equivalence_audit_sha256=sha(audit_path),
                  reuse_reason='Audited CP0 identity front, same E/D/scaler and same frozen F',
                  new_forward=False, scientific_identity=job.get('scientific_identity'))
    return result


def common_endpoint(job, ctx):
    if job.get('device') == 'cuda':
        from .gpu_endpoints import common_endpoint_cuda
        return common_endpoint_cuda(job, ctx)
    specification = endpoint_spec(job)
    seed, family, index, split = job['seed'], job['family'], job['endpoint_index'], job['split']
    score_path = Path(ctx['repo'])/'reports'/OLD/'same_update'/f's{seed}'/f'{family}_e{index}_{split}.json'
    checkpoint = _run_root(ctx, seed, family)/'search'/f'checkpoint_{index:06d}.pt'
    if family != 'B2_CAP_POINTWISE' and score_path.exists():
        result = read(score_path)
        if (sha(checkpoint) != result['checkpoint_sha256'] or result['epoch'] != specification['epoch']
                or result['metric_scale'] != specification['metric_scale'] or result['eval'] != split):
            raise ValueError('Existing endpoint scientific identity mismatch')
        result.update(new_forward=False, source_score_sha256=sha(score_path))
        status, reason = 'REUSED_VERIFIED', 'Existing B0/B2 endpoint identity and checkpoint SHA verified'
    else:
        result = _cp0_reuse(job, ctx, specification)
        if result is not None:
            status, reason = 'REUSED_VERIFIED', 'CAP CP0 equals audited original B0 endpoint'
        else:
            modules = _modules(ctx)
            task = _jena(ctx, modules)
            pool, foundation = _foundation(ctx, modules, 48)
            try:
                torch, fit = modules['torch'], modules['fit']
                saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
                if saved['updates'] != specification['updates'] or saved['sampler']['exposures'] != specification['exposures']:
                    raise ValueError('Search endpoint update/exposure mismatch')
                if family == 'B2_CAP_POINTWISE':
                    cap = importlib.import_module('cap_control')
                    model, norm = cap.create_cap(modules['legacy'].spec_for(seed), task, 10482, foundation)
                else:
                    model, norm = fit.create_model(modules['legacy'].spec_for(seed, family), task, 10482, foundation)
                declared = read(_run_root(ctx, seed, family, True)/'SOURCE_CONTRACT.json')
                for name in ('source_sha', 'config_sha', 'data_sha', 'model_sha'):
                    if saved['contract'][name] != declared[name]:
                        raise ValueError('Search endpoint source contract changed: '+name)
                if (saved['contract']['source_sha'] != fit._source_contract()[0] or
                        saved['contract']['data_sha'] != task['data_sha256'] or
                        saved['contract']['scaler_sha'] != fit.scaler_fingerprint(norm['mean'], norm['std'])):
                    raise ValueError('Actual task/scaler/scientific source differs')
                model.load_state_dict(saved['model'], strict=True)
                mean, std, _ = modules['data'].fit_statistics(task, 12579)
                canonical = dict(mean=mean, std=std) if split == 'V_select' else None
                origins = modules['data'].origins(BOUNDS[split], 512, 48, specification['stride'])
                archive = Path(ctx['cache'])/'endpoints'/(_job_id(job)+'.npz')
                archive.parent.mkdir(parents=True, exist_ok=True)
                temporary = archive.with_suffix('.partial.npz')
                if temporary.exists():
                    temporary.unlink()
                native = modules['legacy'].score(model, norm, task, origins, foundation, modules,
                                                  metric_norm=canonical, save=temporary)
                values = _archive(temporary, native['private_prediction_archive_sha256'], native['array_fingerprints'])
                independent = score_arrays(values['pred_normalized'], values['target_normalized'], values['observed'])
                if max(abs(independent[k]-native[k]) for k in ('MSE', 'MAE')) > 1e-12:
                    raise ValueError('New endpoint independent FP64 reconciliation failed')
                with temporary.open('rb') as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary, archive)
                result = dict(native, **{k: v for k, v in independent.items() if k not in native},
                              seed=seed, model=family, epoch=specification['epoch'], updates=specification['updates'],
                              exposure=specification['exposures'], eval=split, metric_scale=specification['metric_scale'],
                              dataset='jena', C=21, K=6, same_update=True, new_forward=True,
                              checkpoint_sha256=sha(checkpoint), private_prediction_ref=_reference(archive, ctx),
                              source_contract_sha256=sha(_run_root(ctx, seed, family, True)/'SOURCE_CONTRACT.json'),
                              fit='A_fit_search', selection='NOT_SELECTED_COMMON_ENDPOINT', device='cpu',
                              independent_FP64=True, convergence_proven=False)
                status, reason = 'COMPLETE', 'One retained search endpoint scored without retraining'
            finally:
                pool.close()
    output = Path(ctx['report'])/'scores'/(_job_id(job)+'.json')
    atomic_json(output, result)
    paths = [str(output), str(checkpoint)]
    if score_path.exists():
        paths.append(str(score_path))
    if result.get('private_prediction_ref'):
        paths.append(str(_resolve(result['private_prediction_ref'], ctx)))
    return dict(status=status, reason=reason, artifact_paths=paths, result=result)


def full_history_probe(job, ctx):
    modules = _modules(ctx)
    task = _jena(ctx, modules)
    seed, family = int(job['seed']), job['family']
    model, norm, bundle, path = modules['legacy'].make_selected(task, seed, family, object(), modules)
    probe = importlib.import_module('linear_probe')
    fit_origins = modules['data'].origins((0, 12579), 512, 48, 1)
    eval_origins = modules['data'].origins(BOUNDS['V_select'], 512, 48, 48)
    preflight = probe.preflight_probe(task, model, fit_origins, eval_origins)
    preflight_path = Path(ctx['report'])/'preflight'/(_job_id(job)+'.json')
    atomic_json(preflight_path, preflight)
    if not preflight['admissible']:
        return dict(status='SKIPPED_BUDGET', reason='Exact full-history probe exceeds registered memory/FLOP preflight',
                    artifact_paths=[str(preflight_path)], result=preflight)
    private = Path(ctx['cache'])/'probes'/_job_id(job)
    public = Path(ctx['report'])/'probes'/_job_id(job)
    if private.exists() or public.exists():
        return dict(status='PARTIAL_CHECKPOINTED', reason='Incomplete exact probe cache retained; fresh attempt required',
                    artifact_paths=[str(preflight_path)], result=preflight)
    result = probe.run_probe(task, model, norm, fit_origins, eval_origins, private, public, batch=32,
              windows_factory=modules['training'].FastWindows, deadline=_deadline(ctx),
              run_metadata=dict(seed=seed, model=family, dataset='jena', C=21, K=6,
              fit='A60', eval='repeated_development_V43', selected_epoch=bundle.get('selected_epoch', bundle.get('selected_search_epoch')),
              checkpoint_sha256=sha(path), scaler_sha256=modules['fit'].scaler_fingerprint(norm['mean'], norm['std'])))
    return dict(status='COMPLETE', reason='Exact registered fixed-PE full-history lambda=1 probe; F/D not called',
                artifact_paths=[str(public/'RESULT.json'), str(private/'RIDGE_COEFFICIENTS_FP64.npz')], result=result)


def registered_cost_conditions():
    return [dict(question_id='E01', family=family, batch=batch, dataset_cohort='pems08_C32',
                 K=8, device='cpu', warmups=10, timed_trials=20,
                 dependency_family=family) for family in ('B0', 'B2') for batch in (1, 8)]


def profile_job(job, ctx):
    """Profile a disposable CPU clone; never update any retained experiment state."""
    if job['question_id'] == 'C01' and job.get('device') == 'cuda':
        return profile_pems_cuda(job, ctx)
    if job['question_id'] not in ('A03', 'C01'):
        raise ValueError('Short actual profiles are provided for endpoint evaluation and PeMS continuation')
    modules = _modules(ctx)
    torch, fit, training, data = (modules[k] for k in ('torch', 'fit', 'training', 'data'))
    rng = torch.get_rng_state().clone()
    pool = None
    started, cpu_started = time.perf_counter(), time.process_time()
    timings = []
    try:
        if job['question_id'] == 'C01':
            resume = importlib.import_module('pems_resume')
            task, _, _ = resume.load_task(modules['campaign'], modules['old_cache'])
            _, source, checkpoint = resume._original(modules['campaign'], modules['old_cache'], job['family'])
            pool, foundation = _foundation(ctx, modules, 12)
            spec = resume._contract_check(source, task, foundation, fit)
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            model, norm = fit.create_model(spec, task, 8928, foundation)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=0)
            origins = data.origins((0, 8928), 512, 12, 1)
            sampler = training.PermutationStream(origins, 92601)
            updates = training.restore_bundle(saved, model, optimizer, sampler, saved['contract'])
            windows = training.FastWindows(task, norm['mean'], norm['std'], 'cpu')
            predict = lambda x, k: fit.predict_model(model, x, foundation, spec['family'], k)
            # One warm-up followed by five measured real effective-batch updates.
            for index in range(6):
                if time.time() >= _deadline(ctx):
                    raise TimeoutError('Profile reached campaign deadline')
                tick = time.perf_counter()
                segment = training.train_segment(model, windows, origins, sampler, optimizer,
                    sampler.exposures+16, predict, effective_batch=16, microbatch=4,
                    starting_updates=updates, budget_deadline=_deadline(ctx))
                if segment['status'] != 'SEGMENT_COMPLETE' or segment['updates'] != updates+1:
                    raise TimeoutError('Disposable profile update did not finish')
                updates = segment['updates']
                if index:
                    timings.append(time.perf_counter()-tick)
            remaining = math.ceil((25215-saved['sampler']['exposures'])/16)
            worst_refit = math.ceil(math.ceil(3*10190)/16)
            estimate = 1.5*max(timings)*(remaining+worst_refit)
            return dict(status='PROFILED', question_id='C01', family=job['family'], device='cpu',
                        warmups=1, timed_updates=5, measured_seconds=timings,
                        original_remaining_search_updates=remaining, worst_case_refit_updates=worst_refit,
                        estimated_train_wall_seconds=estimate,
                        estimate_excludes_search_eval_refit_eval_and_final_V=True,
                        global_rng_restored=True, disposable_profile_only=True,
                        source_checkpoint_sha256=sha(checkpoint),
                        wall_seconds=time.perf_counter()-started, CPU_seconds=time.process_time()-cpu_started,
                        final_RSS_bytes=modules['psutil'].Process().memory_info().rss)
        specification = endpoint_spec(job)
        task = _jena(ctx, modules)
        pool, foundation = _foundation(ctx, modules, 48)
        saved = torch.load(_run_root(ctx, job['seed'], job['family'])/'search'/f'checkpoint_{job["endpoint_index"]:06d}.pt',
                           map_location='cpu', weights_only=False)
        if job['family'] == 'B2_CAP_POINTWISE':
            model, norm = importlib.import_module('cap_control').create_cap(modules['legacy'].spec_for(job['seed']), task, 10482, foundation)
        else:
            model, norm = fit.create_model(modules['legacy'].spec_for(job['seed'], job['family']), task, 10482, foundation)
        model.load_state_dict(saved['model'], strict=True)
        model.eval()
        windows = training.FastWindows(task, norm['mean'], norm['std'], 'cpu')
        origins = data.origins(BOUNDS[job['split']], 512, 48, specification['stride'])
        x, _, _ = windows.get(origins[:16])
        with torch.no_grad():
            fit.predict_model(model, x, foundation, 'B2', 6)
            for _ in range(5):
                if time.time() >= _deadline(ctx):
                    raise TimeoutError('Profile reached campaign deadline')
                tick = time.perf_counter()
                fit.predict_model(model, x, foundation, 'B2', 6)
                timings.append(time.perf_counter()-tick)
        return dict(status='PROFILED', question_id='A03', device='cpu', warmups=1,
                    timed_forwards=5, measured_seconds=timings, batch_origins=len(x),
                    total_eval_origins=len(origins), estimated_forward_wall_seconds=1.5*max(timings)*math.ceil(len(origins)/16),
                    estimate_excludes_array_serialization=True, global_rng_restored=True,
                    disposable_profile_only=True, wall_seconds=time.perf_counter()-started,
                    CPU_seconds=time.process_time()-cpu_started,
                    final_RSS_bytes=modules['psutil'].Process().memory_info().rss)
    finally:
        torch.set_rng_state(rng)
        if pool is not None:
            pool.close()


def cuda_branch_payload(original, spec, terminal_sha, original_reference, cuda_rng):
    """Explicitly convert a copied CPU bundle; retained source objects stay unchanged."""
    if spec['device'] != 'cuda' or spec['family'] not in ('B0', 'B2') or original['cuda_rng']:
        raise ValueError('Only a CPU-to-CUDA PeMS B0/B2 branch may be converted')
    lineage = dict(continuation_kind='CPU_TO_CUDA_NUMERICAL_BRANCH',
                   original_terminal_sha256=terminal_sha, original_checkpoint_ref=original_reference,
                   original_contract=copy.deepcopy(original['contract']),
                   original_model_ref=original_reference+'#/model',
                   original_optimizer_ref=original_reference+'#/optimizer',
                   original_sampler_ref=original_reference+'#/sampler',
                   original_CPU_RNG_ref=original_reference+'#/torch_rng',
                   CUDA_RNG_definition='New branch-only seeded CUDA generator; historical CPU had no CUDA RNG',
                   historical_bitwise_equivalence_claimed=False)
    identity = dict(spec=spec, source_contract=original['contract'], terminal_sha256=terminal_sha,
                    continuation_kind=lineage['continuation_kind'])
    converted = copy.deepcopy(original)
    converted['contract']['config_sha'] = hashlib.sha256(json.dumps(identity, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    converted['cuda_rng'] = copy.deepcopy(cuda_rng)
    converted['extra']['device_branch'] = lineage
    lineage['branch_contract'] = copy.deepcopy(converted['contract'])
    return converted, lineage


def _cuda_spec(source):
    return dict(source['spec'], device='cuda',
                continuation_kind='CPU_TO_CUDA_NUMERICAL_BRANCH',
                original_CPU_config_sha256=source['config_sha'], branch_source_sha256=sha(__file__))


def _branch_rng(torch, seed):
    return [torch.Generator(device=f'cuda:{i}').manual_seed(seed).get_state()
            for i in range(torch.cuda.device_count())]


def _optimizer_devices(optimizer):
    inspected = 0
    for group in optimizer.param_groups:
        for parameter in group['params']:
            for name, value in optimizer.state.get(parameter, {}).items():
                if hasattr(value, 'device') and name != 'step':
                    inspected += 1
                    if value.device != parameter.device:
                        raise ValueError('Restored AdamW moment device differs from its parameter')
    return dict(non_step_optimizer_tensors_checked=inspected,
                all_non_step_optimizer_tensors_match_parameter_device=True,
                step_scalar_may_remain_CPU_for_non_capturable_AdamW=True)


def resume_payload_fingerprint(value):
    import numpy as np
    if hasattr(value, 'detach'):
        return resume_payload_fingerprint(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        value = dict(array=_array_fingerprint(value))
    elif isinstance(value, dict):
        value = [(type(key).__name__, str(key), resume_payload_fingerprint(item))
                 for key, item in sorted(value.items(), key=lambda pair: (type(pair[0]).__name__, str(pair[0])))]
    elif isinstance(value, (list, tuple)):
        value = [resume_payload_fingerprint(item) for item in value]
    elif isinstance(value, np.generic):
        value = value.item()
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def disposable_resume_payload(original, device, **branch):
    if device == 'cpu':
        return copy.deepcopy(original)
    return cuda_branch_payload(original, **branch)[0]


def profile_pems_cuda(job, ctx):
    """CPU/CUDA next-step difference plus five disposable CUDA updates; no files changed."""
    modules = _modules(ctx)
    torch, fit, training, data = (modules[k] for k in ('torch', 'fit', 'training', 'data'))
    resume = importlib.import_module('pems_resume')
    task, _, _ = resume.load_task(modules['campaign'], modules['old_cache'])
    _, source, path = resume._original(modules['campaign'], modules['old_cache'], job['family'])
    original = torch.load(path, map_location='cpu', weights_only=False)
    original_fingerprint = resume_payload_fingerprint(original)
    profiles, pools = [], []
    cpu_after = None
    started = time.perf_counter()
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            for device in ('cpu', 'cuda'):
                pool, foundation = _foundation(ctx, modules, 12, device)
                pools.append(pool)
                spec = source['spec'] if device == 'cpu' else _cuda_spec(source)
                payload = disposable_resume_payload(original, device, spec=spec, terminal_sha=sha(path),
                    original_reference=_reference(path, ctx), cuda_rng=_branch_rng(torch, int(spec['seed'])))
                if resume_payload_fingerprint(original) != original_fingerprint:
                    raise ValueError('Disposable profile mutated its source bundle')
                restored_state_fingerprint = resume_payload_fingerprint(dict(model=payload['model'],
                    optimizer=payload['optimizer'], sampler=payload['sampler'], torch_rng=payload['torch_rng']))
                model, norm = fit.create_model(spec, task, 8928, foundation)
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=0)
                origins = data.origins((0, 8928), 512, 12, 1)
                sampler = training.PermutationStream(origins, 92601)
                updates = training.restore_bundle(payload, model, optimizer, sampler, payload['contract'])
                devices = _optimizer_devices(optimizer)
                windows = training.FastWindows(task, norm['mean'], norm['std'], device)
                predict = lambda x, k: fit.predict_model(model, x, foundation, spec['family'], k)
                timings = []
                for index in range(1 if device == 'cpu' else 6):
                    if time.time() >= _deadline(ctx):
                        raise TimeoutError('CUDA continuation profile deadline')
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    tick = time.perf_counter()
                    segment = training.train_segment(model, windows, origins, sampler, optimizer,
                        sampler.exposures+16, predict, effective_batch=16, microbatch=4,
                        starting_updates=updates, budget_deadline=_deadline(ctx))
                    if device == 'cuda':
                        torch.cuda.synchronize()
                    if segment['status'] != 'SEGMENT_COMPLETE' or segment['updates'] != updates+1:
                        raise TimeoutError('Disposable continuation update did not finish')
                    updates = segment['updates']
                    if index:
                        timings.append(time.perf_counter()-tick)
                    if index == 0:
                        state = training.cpu_state(model)
                        if device == 'cpu':
                            cpu_after = dict(model=state, sampler=sampler.state_dict(),
                                             initial_state_fingerprint=restored_state_fingerprint,
                                             loss=segment.get('mean_observed_training_batch_loss'))
                        else:
                            if cpu_after['initial_state_fingerprint'] != restored_state_fingerprint:
                                raise ValueError('CPU/CUDA profile started from different source states')
                            if training.fingerprint(cpu_after['sampler']) != training.fingerprint(sampler.state_dict()):
                                raise ValueError('CPU/CUDA continuation next-step sampler order differs')
                            differences = {key: float((state[key].double()-cpu_after['model'][key].double()).abs().max())
                                           for key in state}
                            branch_audit = dict(status='MEASURED_NUMERICAL_DEVICE_BRANCH',
                                identical_initial_model_optimizer_sampler_CPU_RNG=True,
                                initial_state_fingerprint=restored_state_fingerprint,
                                CPU_and_CUDA_sampler_after_next_update_equal=True,
                                max_absolute_parameter_difference=max(differences.values()),
                                per_parameter_max_absolute_difference=differences,
                                CPU_next_update_loss=cpu_after['loss'],
                                CUDA_next_update_loss=segment.get('mean_observed_training_batch_loss'),
                                optimizer_device_check=devices, historical_bitwise_equivalence_claimed=False,
                                original_CPU_RNG_restored=True, new_branch_CUDA_RNG_restored=True,
                                audit_updates_excluded_from_training=True)
                if device == 'cuda':
                    remaining = math.ceil((25215-original['sampler']['exposures'])/16)
                    refit = math.ceil(math.ceil(3*10190)/16)
                    result = dict(status='PROFILED', device='cuda', family=job['family'],
                        warmups=1, timed_updates=5, measured_seconds=timings,
                        original_remaining_search_updates=remaining, worst_case_refit_updates=refit,
                        estimated_train_wall_seconds=1.5*max(timings)*(remaining+refit),
                        estimate_excludes_evaluation=True, next_update_branch_audit=branch_audit,
                        original_checkpoint_sha256=sha(path), global_CPU_and_CUDA_RNG_restored=True,
                        disposable_profile_only=True, wall_seconds=time.perf_counter()-started)
                del windows, optimizer, model
                if resume_payload_fingerprint(original) != original_fingerprint:
                    raise ValueError('Disposable update mutated original checkpoint tensors')
                pool.close()
                pools.pop()
            result['source_bundle_fingerprint_before_and_after']=original_fingerprint
            result['source_bundle_unchanged_after_both_profiles']=True
            return result
    finally:
        for pool in pools:
            pool.close()


def profile_pems_cuda_pair(ctx):
    profiles = {family: profile_pems_cuda(dict(question_id='C01', family=family, device='cuda'), ctx)
                for family in ('B0', 'B2')}
    return dict(status='PROFILED', device='cuda', profiles=profiles,
                conservative_pair_seconds=sum(p['estimated_train_wall_seconds'] for p in profiles.values())+300,
                evaluation_and_load_margin_seconds=300,
                original_CPU_registered_jobs_preserved=True)


def _cuda_selected_score(model, norm, task, origins, foundation, modules, archive, ctx):
    np, torch = modules['np'], modules['torch']
    windows = modules['training'].FastWindows(task, norm['mean'], norm['std'], 'cuda')
    predictions, targets, masks = [], [], []
    model.eval()
    with torch.no_grad():
        for lo in range(0, len(origins), 16):
            if time.time() >= _deadline(ctx):
                raise TimeoutError('CUDA V evaluation deadline')
            if ctx.get('lease'):
                ctx['lease'].check()
            x, y, mask = windows.get(origins[lo:lo+16])
            prediction = modules['fit'].predict_model(model, x, foundation, 'B2', 8)
            predictions.append(prediction.cpu().numpy()); targets.append(y.cpu().numpy()); masks.append(mask.cpu().numpy())
    values = dict(pred_normalized=np.concatenate(predictions), target_normalized=np.concatenate(targets),
                  observed=np.concatenate(masks), origins=origins, channel_ids=np.asarray(task['columns'], str))
    temporary = archive.with_suffix('.partial.npz')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **values)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, archive)
    result = score_arrays(values['pred_normalized'], values['target_normalized'], values['observed'])
    result.update(private_prediction_archive_sha256=sha(archive),
                  array_fingerprints={k: _array_fingerprint(v) for k, v in values.items()},
                  scaler_sha256=modules['fit'].scaler_fingerprint(norm['mean'], norm['std']))
    return result


def resume_pems_cuda(job, ctx):
    modules = _modules(ctx)
    torch, fit = modules['torch'], modules['fit']
    resume = importlib.import_module('pems_resume')
    task, access, _ = resume.load_task(modules['campaign'], modules['old_cache'])
    family = job['family']
    _, source, original = resume._original(modules['campaign'], modules['old_cache'], family)
    original_sha = sha(original)
    audit_path = Path(ctx['asset'])/'.cache'/OLD/f'PEMS_RESTORE_AUDIT_{family}'/'RESULT.json'
    audit = read(audit_path)
    if audit['status'] != 'PASS_EXACT_NEXT_UPDATE' or audit['original_sha256'] != original_sha:
        raise ValueError('Original CPU restore audit is absent or mismatched')
    spec = _cuda_spec(source)
    private, public = Path(ctx['cache'])/'runs'/f'pems_{family}', Path(ctx['report'])/'runs'/f'pems_{family}'
    private.mkdir(parents=True, exist_ok=True); public.mkdir(parents=True, exist_ok=True)
    atomic_json(private/'PREFIX_ACCESS.json', access)
    (private/'search').mkdir(exist_ok=True)
    branch_rng = _branch_rng(torch, 92601)
    for path in sorted(original.parent.glob('checkpoint_*.pt')):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        converted, lineage = cuda_branch_payload(payload, spec, original_sha, _reference(path, ctx), branch_rng)
        destination = private/'search'/path.name
        if destination.exists():
            retained = torch.load(destination, map_location='cpu', weights_only=False)
            if retained['contract'] != converted['contract']:
                raise ValueError('Existing CUDA branch checkpoint contract differs')
        else:
            temporary = destination.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                torch.save(converted, stream); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, destination)
    lineage.update(original_CPU_restore_audit_sha256=sha(audit_path),
                   CPU_registered_skip_is_a_separate_job=True, new_spec=spec,
                   registered_dependency_ids=job.get('dependency_ids', []))
    atomic_json(public/'DEVICE_BRANCH_LINEAGE.json', lineage)
    pool, foundation = _foundation(ctx, modules, 12, 'cuda')
    original_restore = modules['training'].restore_bundle
    restore_checks = []
    def observed_restore(payload, model, optimizer, sampler, contract):
        updates = original_restore(payload, model, optimizer, sampler, contract)
        devices = _optimizer_devices(optimizer)
        if not torch.equal(torch.get_rng_state(), payload['torch_rng']):
            raise ValueError('CPU RNG was not restored from device branch checkpoint')
        actual_cuda = torch.cuda.get_rng_state_all()
        if len(actual_cuda) != len(payload['cuda_rng']) or any(not torch.equal(a,b) for a,b in zip(actual_cuda,payload['cuda_rng'])):
            raise ValueError('CUDA RNG was not restored from device branch checkpoint')
        restore_checks.append(dict(updates=updates, parameter_device=str(next(model.parameters()).device),
                                   CPU_RNG_restored=True, CUDA_RNG_restored=True, **devices))
        atomic_json(public/'ACTUAL_BRANCH_RESTORE_CHECKS.json', dict(checks=restore_checks,
                    exactness_scope='saved CUDA branch state only; historical CPU-to-CUDA is numerical'))
        return updates
    modules['training'].restore_bundle = observed_restore
    try:
        # Check all inherited scientific identities before any branch update.
        resume._contract_check(source, task, foundation, fit)
        contract = {key: converted['contract'][key] for key in ('source_sha', 'config_sha', 'data_sha', 'model_sha')}
        search = fit._stage(spec, task, 8928, resume.BOUNDS['A_stop'], [.25, 1., 3.],
                            private/'search', public/'search', foundation, _deadline(ctx), contract)
        atomic_json(public/'SEARCH_SUMMARY.json', search['summary'])
        atomic_json(public/'SEARCH_CURVES.json', search['curves'])
        if search['summary']['status'] != 'SEGMENT_COMPLETE':
            return dict(status='PARTIAL_CHECKPOINTED', reason='CUDA branch search checkpointed',
                        artifact_paths=[str(public/'SEARCH_SUMMARY.json')], result=search['summary'])
        epoch = search['summary']['selected_epoch']
        if search['summary']['completed_epochs'] != list(EPOCHS) or search['summary']['exposures'] != 25215:
            raise ValueError('CUDA branch did not finish original endpoint schedule')
        refit = fit._stage(spec, task, 10713, None, [epoch] if epoch else [], private/'refit', public/'refit',
                           foundation, _deadline(ctx), contract)
        atomic_json(public/'REFIT_SUMMARY.json', refit['summary'])
        if refit['summary']['status'] != 'SEGMENT_COMPLETE':
            return dict(status='PARTIAL_CHECKPOINTED', reason='CUDA branch fresh A60 refit checkpointed',
                        artifact_paths=[str(public/'REFIT_SUMMARY.json')], result=refit['summary'])
        bundle = dict(model=refit['model'], normalization=refit['normalization'], spec=spec,
                      contract=refit['summary']['contract'], selected_epoch=epoch, lineage=lineage,
                      search_summary=search['summary'], refit_summary=refit['summary'])
        bundle['model_state_sha'] = modules['training'].fingerprint({k: modules['data'].array_sha256(v.numpy()) for k,v in bundle['model'].items()})
        artifact = private/'selected_bundle.pt'; temporary = artifact.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            torch.save(bundle, stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, artifact)
        model, norm = fit.create_model(spec, task, 10713, foundation)
        model.load_state_dict(bundle['model'], strict=True)
        origins = modules['data'].origins(resume.BOUNDS['V_select'], 512, 12, 12)
        archive = private/'selected_V.npz'
        metrics = _cuda_selected_score(model, norm, task, origins, foundation, modules, archive, ctx)
        metrics.update(status='COMPLETE_SELECTED_V', model=family, seed=92601, dataset='pems08_C32', C=32, K=8,
                       selected_epoch=epoch, fit='fresh_A60_after_CPU_to_CUDA_search_branch', eval='V_select',
                       device='cuda', artifact_sha256=sha(artifact), updates=refit['summary']['updates'],
                       exposure=refit['summary']['exposures'], exact_CPU_continuation=False,
                       continuation_kind='CPU_TO_CUDA_NUMERICAL_BRANCH', historical_bitwise_equivalence_claimed=False,
                       original_partial_sha256=original_sha, params=refit['summary']['trainable_parameters'])
        output = public/'SELECTED_V_RESULT.json'; atomic_json(output, metrics)
        if sha(original) != original_sha:
            raise ValueError('Original CPU checkpoint changed during CUDA branch')
        return dict(status='COMPLETE', reason='Explicit numerical CPU-to-CUDA branch plus fresh A60 refit and V148',
                    artifact_paths=[str(output), str(public/'DEVICE_BRANCH_LINEAGE.json'),
                                    str(public/'ACTUAL_BRANCH_RESTORE_CHECKS.json'), str(artifact), str(archive)], result=metrics)
    finally:
        modules['training'].restore_bundle = original_restore
        pool.close()


def resume_pems(job, ctx):
    if job.get('device') == 'cuda':
        return resume_pems_cuda(job, ctx)
    if job.get('device', 'cpu') != 'cpu':
        raise ValueError('Unsupported continuation device')
    family = job['family']
    if family not in ('B0', 'B2'):
        raise ValueError('Unregistered PeMS family')
    modules = _modules(ctx)
    resume = importlib.import_module('pems_resume')
    old_cache = Path(ctx['asset'])/'.cache'/BROAD
    task, access, plan = resume.load_task(modules['campaign'], old_cache)
    identifier, source, original = resume._original(modules['campaign'], old_cache, family)
    audit = read(Path(ctx['asset'])/'.cache'/OLD/f'PEMS_RESTORE_AUDIT_{family}'/'RESULT.json')
    if audit['status'] != 'PASS_EXACT_NEXT_UPDATE' or audit['original_sha256'] != sha(original):
        raise ValueError('Original CPU exact-resume audit not verified')
    private, public = Path(ctx['cache'])/'runs'/f'pems_{family}', Path(ctx['report'])/'runs'/f'pems_{family}'
    private.mkdir(parents=True, exist_ok=True)
    public.mkdir(parents=True, exist_ok=True)
    atomic_json(private/'PREFIX_ACCESS.json', access)
    (private/'search').mkdir(exist_ok=True)
    for path in original.parent.glob('checkpoint_*.pt'):
        destination = private/'search'/path.name
        if not destination.exists():
            shutil.copyfile(path, destination)
        if sha(destination) != sha(path):
            raise ValueError('Retained historical checkpoint copy changed')
    pool, foundation = _foundation(ctx, modules, 12)
    try:
        spec = resume._contract_check(source, task, foundation, modules['fit'])
        contract = {key: source[key] for key in ('source_sha', 'config_sha', 'data_sha', 'model_sha')}
        search = modules['fit']._stage(spec, task, 8928, resume.BOUNDS['A_stop'], [.25, 1., 3.],
                                       private/'search', public/'search', foundation, _deadline(ctx), contract)
        atomic_json(public/'SEARCH_SUMMARY.json', search['summary'])
        atomic_json(public/'SEARCH_CURVES.json', search['curves'])
        if search['summary']['status'] != 'SEGMENT_COMPLETE':
            return dict(status='PARTIAL_CHECKPOINTED', reason='Original PeMS search checkpointed at new deadline',
                        artifact_paths=[str(public/'SEARCH_SUMMARY.json')], result=search['summary'])
        epoch = search['summary']['selected_epoch']
        if search['summary']['completed_epochs'] != list(EPOCHS) or search['summary']['exposures'] != 25215:
            raise ValueError('Original PeMS search endpoint completeness failed')
        refit = modules['fit']._stage(spec, task, 10713, None, [epoch] if epoch else [], private/'refit',
                                      public/'refit', foundation, _deadline(ctx), contract)
        atomic_json(public/'REFIT_SUMMARY.json', refit['summary'])
        if refit['summary']['status'] != 'SEGMENT_COMPLETE':
            return dict(status='PARTIAL_CHECKPOINTED', reason='Fresh A60 refit checkpointed at new deadline',
                        artifact_paths=[str(public/'REFIT_SUMMARY.json')], result=refit['summary'])
        bundle = dict(model=refit['model'], normalization=refit['normalization'], spec=spec,
                      contract=refit['summary']['contract'], selected_epoch=epoch,
                      search_summary=search['summary'], refit_summary=refit['summary'])
        bundle['model_state_sha'] = modules['training'].fingerprint({k: modules['data'].array_sha256(v.numpy()) for k, v in bundle['model'].items()})
        artifact = private/'selected_bundle.pt'
        temporary = artifact.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            modules['torch'].save(bundle, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, artifact)
        model, norm = modules['fit'].create_model(spec, task, 10713, foundation)
        model.load_state_dict(bundle['model'], strict=True)
        origins = modules['data'].origins(resume.BOUNDS['V_select'], 512, 12, 12)
        archive = private/'selected_V.npz'
        temporary = private/'selected_V.partial.npz'
        if temporary.exists():
            temporary.unlink()
        metrics = modules['legacy'].score(model, norm, task, origins, foundation, modules, save=temporary)
        with temporary.open('rb') as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, archive)
        metrics.update(status='COMPLETE_SELECTED_V', model=family, seed=92601, dataset='pems08_C32',
                       C=32, K=8, selected_epoch=epoch, fit='fresh_A60_after_resumed_A50', eval='V_select',
                       device='cpu', artifact_sha256=sha(artifact), updates=refit['summary']['updates'],
                       exposure=refit['summary']['exposures'], exact_CPU_continuation=True,
                       original_partial_sha256=sha(original), original_resume_audit_sha256=sha(Path(ctx['asset'])/'.cache'/OLD/f'PEMS_RESTORE_AUDIT_{family}'/'RESULT.json'))
        output = public/'SELECTED_V_RESULT.json'
        atomic_json(output, metrics)
        return dict(status='COMPLETE', reason='Exact copied CPU search continuation, selected fresh A60 refit and V148',
                    artifact_paths=[str(output), str(artifact), str(archive)], result=metrics)
    finally:
        pool.close()


def end_to_end_cost(job, ctx):
    private = Path(ctx['cache'])/'runs'/f'pems_{job["family"]}'
    required = [private/'selected_bundle.pt', private/'selected_V.npz',
                Path(ctx['report'])/'runs'/f'pems_{job["family"]}'/'SELECTED_V_RESULT.json']
    if not all(path.exists() for path in required):
        return dict(status='NOT_RUN_DEPENDENCY', reason='Registered PeMS C32 selected model and final V absent', artifact_paths=[])
    modules = _modules(ctx)
    cost = importlib.import_module('pems_cost')
    resume = importlib.import_module('pems_resume')
    measure = importlib.import_module('measure_costs')
    family, batch = job['family'], int(job['batch'])
    if family not in ('B0', 'B2') or batch not in (1, 8):
        raise ValueError('Unregistered PeMS cost condition')
    public = Path(ctx['report'])/'costs'/f'pems_{family}_b{batch}'
    public.mkdir(parents=True, exist_ok=True)
    pool = route = rss_sampler = None
    started = time.perf_counter()
    try:
        np, torch, psutil = modules['np'], modules['torch'], modules['psutil']
        protocol = cost.pin_protocol(Path(ctx['repo']))
        quiet = cost._quiet(psutil, Path(ctx['asset']))
        artifact, prediction, selected = required
        metrics = read(selected)
        if metrics['status'] != 'COMPLETE_SELECTED_V' or metrics['C'] != 32 or metrics['K'] != 8:
            raise ValueError('Cost requires completed original C32 selected result')
        bundle = torch.load(artifact, map_location='cpu', weights_only=False)
        if bundle['spec']['family'] != family or bundle['normalization']['fit_end'] != 10713:
            raise ValueError('Cost selected family/normalization mismatch')
        task, _, plan = resume.load_task(modules['campaign'], modules['old_cache'])
        values = _archive(prediction, metrics['private_prediction_archive_sha256'], metrics['array_fingerprints'])
        origins = modules['data'].origins(resume.BOUNDS['V_select'], 512, 12, 12)
        if not np.array_equal(values['origins'], origins) or len(origins) != 148:
            raise ValueError('Cost fixed V148 origin mismatch')
        if not np.array_equal(values['channel_ids'], np.asarray(task['columns'], str)):
            raise ValueError('Cost selected channel order differs')
        physical = np.stack([task['physical'][int(o)-512:int(o)] for o in origins[:batch]])
        observed = np.stack([task['observed'][int(o)-512:int(o)] for o in origins[:batch]])
        original = next(r for r in protocol['original_receipts'] if r['batch'] == batch)
        if (cost.sha_bytes(physical) != original['physical_input_sha256'] or
                cost.sha_bytes(observed) != original['observation_mask_sha256']):
            raise ValueError('Cost physical input/mask differs from original C32 Raw/G')
        class Lease:
            def check(self):
                if time.time() >= _deadline(ctx):
                    raise TimeoutError('Cost reached reserved job deadline')
                if psutil.Process().memory_info().rss > 3*1024**3:
                    raise MemoryError('Cost exceeded original 3GiB RSS cap')
                if torch.get_num_threads() != 2 or torch.cuda.is_initialized():
                    raise RuntimeError('Original CPU2/GPU0 cost contract changed')
                cost._quiet(psutil, Path(ctx['asset']))
        lease = Lease()
        lease.check()
        pool = modules['deployment'].FoundationPool(Path(ctx['asset'])/plan['model_asset_relative_path'], 'cpu')
        route_spec = dict(run_id='new_pems_'+family+'_b'+str(batch), family=family, latent_k=8,
                          canonical_fit_end=10713, device='cpu', row_microbatch=128,
                          artifact_cache_relative_path=artifact.relative_to(Path(ctx['cache'])).as_posix(),
                          artifact_sha256=sha(artifact))
        route = modules['deployment'].Route(route_spec, task, Path(ctx['cache']), pool.model_path, 'cpu', pool)
        rss_sampler = RSSSampler(lambda: psutil.Process().memory_info().rss)
        rss_sampler.start()
        value = route.predict_physical(physical, observed)
        normalized = ((value-route.mean.astype(np.float64))/route.std.astype(np.float64)).astype(np.float32)
        expected = values['pred_normalized'][:batch]
        if not np.allclose(normalized, expected, atol=cost.ATOL, rtol=cost.RTOL):
            raise ValueError('Physical route parity failed; latency remains unmeasured')
        rows = measure.measure_balanced([route], physical, observed, warmups=10, timed_trials=20, stage_trials=20, lease=lease)
        rss_measurement = rss_sampler.stop()
        rss_sampler = None
        geometry = (batch*8, batch*8, math.ceil(batch*8/128))
        if len(rows) != 50 or any((r['F_logical_streams'], r['F_physical_streams'], r['F_calls']) != geometry for r in rows):
            raise ValueError('Measured F geometry/trial count differs')
        timed = np.asarray([r['whole_seconds'] for r in rows if r['phase'] == 'timed'], np.float64)
        q25, median, q75 = np.quantile(timed, [.25, .5, .75])
        trials = public/'COST_TRIALS.csv.gz'
        temporary = trials.with_name('COST_TRIALS.partial.csv.gz')
        if temporary.exists():
            temporary.unlink()
        measure._write_trials(temporary, rows)
        os.replace(temporary, trials)
        result = dict(status='COMPLETE_PROTOCOL', family=family, batch_size=batch, dataset='pems08_C32',
                      C=32, K=8, timing_status='MEASURED', **cost.PROTOCOL,
                      median_seconds=float(median), IQR_seconds=float(q75-q25),
                      q25_seconds=float(q25), q75_seconds=float(q75),
                      ms_per_batch=float(median*1000), ms_per_origin=float(median*1000/batch),
                      individual_timed_seconds=timed.tolist(), summaries=measure.summarize(rows),
                      memory=route.memory_state(), cold_load_seconds=route.load_seconds,
                      source_protocol=protocol, physical_input_sha256=cost.sha_bytes(physical),
                      observation_mask_sha256=cost.sha_bytes(observed), artifact_sha256=sha(artifact),
                      selected_metrics_sha256=sha(selected), prediction_archive_sha256=sha(prediction),
                      trials_sha256=sha(trials), scope=protocol['full_pipeline'],
                      forward_parity=dict(status='PASS', atol=cost.ATOL, rtol=cost.RTOL),
                      quiet_before=quiet, quiet_after=cost._quiet(psutil, Path(ctx['asset'])),
                      elapsed_wall_seconds=time.perf_counter()-started,
                      process_RSS_bytes=psutil.Process().memory_info().rss,
                      exclusive_queue_owned_by_caller=True, cached_forecast_reads_for_timing=0,
                      RSS_sampler_active_during_latency=True, **rss_measurement)
    finally:
        if rss_sampler is not None:
            rss_sampler.stop()
        if route is not None:
            route.close()
        if pool is not None:
            pool.close()
    output = public/'RESULT.json'
    atomic_json(output, result)
    return dict(status='COMPLETE', reason='Original four-route cost contract with independent source/output roots',
                artifact_paths=[str(output), str(trials)], result=result)


def execute_job(job, ctx):
    """Run one explicitly dispatched job; no historic campaign bootstrap/admission."""
    previous = completed_receipt(job, ctx)
    if previous is not None:
        previous['artifact_paths'] = [p for p in previous['artifact_paths'] if Path(p).resolve().is_relative_to(Path(ctx['report']).resolve())]
        previous['artifact_paths'].append(str(_receipt_path(job, ctx)))
        return previous
    if time.time() >= _deadline(ctx):
        return dict(status='SKIPPED_BUDGET', reason='New campaign execution deadline reached', artifact_paths=[])
    handlers = dict(A02=cap_rescore, A03=common_endpoint, B01=full_history_probe,
                    C01=resume_pems, E01=end_to_end_cost)
    if job['question_id'] not in handlers:
        raise ValueError('Unknown existing-results question')
    try:
        result = handlers[job['question_id']](job, ctx)
    except FileNotFoundError:
        result = dict(status='BLOCKED_MISSING_ASSET', reason='One or more hash-bound original assets are absent', artifact_paths=[])
    save_receipt(job, ctx, result)
    result['artifact_paths'] = [p for p in result['artifact_paths'] if Path(p).resolve().is_relative_to(Path(ctx['report']).resolve())]
    result['artifact_paths'].append(str(_receipt_path(job, ctx)))
    return result
