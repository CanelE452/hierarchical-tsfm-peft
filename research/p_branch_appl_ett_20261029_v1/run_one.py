"""One registered immutable search -> selection seal -> fresh A60 -> V job."""
import argparse
from datetime import datetime, timezone
import gc
import os
import psutil
import math
from pathlib import Path
import subprocess
import time

from . import runtime


METHOD_FAMILY = dict(P_TIME_ONLY='B2', P_CHANNEL_ONLY='B2')
JOB_FIELDS = {'job_id', 'method_id', 'dataset', 'K', 'seed', 'extended'}
SOURCE_NAMES = ('models.py', 'runtime.py', 'run_one.py', 'controller.py')


def validate_job(job):
    if (set(job) != JOB_FIELDS or job['method_id'] not in METHOD_FAMILY
            or job['dataset'] not in runtime.DATASETS or type(job['K']) is not int
            or job['K'] != runtime.KS[job['dataset']]
            or type(job['seed']) is not int or job['seed'] not in range(92601, 92611)
            or job['extended'] is not False):
        raise ValueError('Job differs from three-dataset P-branch registration')
    expected = f"PBA_{job['method_id']}_{job['dataset']}_K{job['K']}_s{job['seed']}"
    if job['job_id'] != expected:
        raise ValueError('Job identifier differs from registered axes')
    return job


def source_hashes():
    return {f'research/{runtime.NAME}/{name}': runtime.sha(Path(__file__).with_name(name))
            for name in SOURCE_NAMES}


def schedule(job):
    validate_job(job)
    return [.25, 1., 3.]


def build_spec(job):
    validate_job(job)
    return dict(run_id=job['job_id'], dataset=job['dataset'], method_id=job['method_id'],
                family=METHOD_FAMILY[job['method_id']], latent_k=job['K'], seed=job['seed'],
                init_kind=('QR' if job['method_id'] == 'A3' else
                           'PYTORCH_DEFAULT' if job['method_id'] in ('C2', 'C3') else 'PCA'), revin=False,
                device='cuda', fold='primary', core_dtype='float32', lr=.001,
                effective_batch=16, microbatch=4, F_row_microbatch=128,
                checkpoint_inputs=False, epochs=schedule(job),
                custom_source_sha256=runtime.sha(Path(__file__).with_name('models.py')),
                new_scientific_source_hashes=source_hashes())


def validate_registration(job):
    validate_job(job)
    plan = runtime.read(runtime.REPORT / 'RUN_PLAN.json')
    entries = plan.get('jobs', []) + plan.get('extension_jobs', [])
    if sum(entry == job for entry in entries) != 1:
        raise ValueError('Job absent or duplicated in RUN_PLAN')
    protocol = runtime.read(runtime.REPORT / 'PREREGISTRATION.json')
    seal = runtime.read(runtime.REPORT / 'REGISTRATION_SEAL.json')
    required_names = {'PREREGISTRATION.json', 'PREREGISTRATION.md', 'RUN_PLAN.json',
                      'PRESERVATION_BASELINE.json', 'REFERENCE_RUNS.json'}
    expected_names = set(protocol.get('registration_documents', required_names))
    if not required_names.issubset(expected_names) or set(seal['files']) != expected_names:
        raise ValueError('Registration seal differs from the registered input document roster')
    if any(Path(name).is_absolute() or '..' in Path(name).parts for name in expected_names):
        raise ValueError('Registration document path is outside the campaign')
    for name, digest in seal['files'].items():
        if runtime.sha(runtime.REPORT / name) != digest:
            raise ValueError('Registration document changed after sealing: ' + name)
    sources = protocol['registered_sources']
    for path, digest in sources.items():
        if not (runtime.REPO / path).resolve().is_relative_to(runtime.REPO.resolve()):
            raise ValueError('Registered source path is outside repository')
        if runtime.sha(runtime.REPO / path) != digest:
            raise ValueError('Registered source changed: ' + path)
    if any(sources.get(path) != digest for path, digest in source_hashes().items()):
        raise ValueError('Campaign source missing or changed in registration')
    created = datetime.fromisoformat(seal['created_utc']).timestamp()
    if created > time.time():
        raise ValueError('Registration seal is in the future')
    runtime.guard_preserved()
    return dict(seal_sha256=runtime.sha(runtime.REPORT / 'REGISTRATION_SEAL.json'),
                registered_at_utc=seal['created_utc'], files=seal['files'])


def recovery_authorizations(job, *, started_unix=None):
    validate_job(job)
    cutoff = time.time() if started_unix is None else float(started_unix)
    records = []
    for path in sorted(runtime.REPORT.glob('RECOVERY_AUTHORIZATION_[0-9][0-9][0-9].json')):
        value = runtime.read(path)
        if value.get('job_id') != job['job_id']:
            continue
        attempts = value.get('allowed_attempts')
        if (value.get('status') != 'APPROVED_TECHNICAL_RECOVERY'
                or not isinstance(attempts, list) or len(attempts) != 2
                or any(type(a) is not int for a in attempts) or attempts[0] < 3
                or attempts[0] % 2 != 1 or attempts != [attempts[0], attempts[0] + 1]
                or value.get('registration_seal_sha256') != runtime.sha(runtime.REPORT / 'REGISTRATION_SEAL.json')
                or value.get('scientific_source_hashes') != source_hashes()):
            raise ValueError('Recovery authorization differs from immutable job or sources')
        try:
            created = datetime.fromisoformat(value['created_utc']).timestamp()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('Recovery authorization has no valid prior timestamp') from exc
        if created > cutoff:
            raise ValueError('Recovery authorization was not approved before attempt start')
        records.append(dict(authorization_path=str(path), authorization_sha256=runtime.sha(path),
                            allowed_attempts=attempts, created_utc=value['created_utc']))
    values = [a for record in records for a in record['allowed_attempts']]
    if len(values) != len(set(values)):
        raise ValueError('Recovery authorization duplicates a registered attempt')
    return records


def allowed_attempts(job):
    return sorted([1, 2] + [a for record in recovery_authorizations(job) for a in record['allowed_attempts']])


def validate_attempt(job, attempt, *, started_unix=None):
    validate_job(job)
    if type(attempt) is not int or attempt < 1:
        raise ValueError('Attempt must be a positive registered integer')
    if attempt in (1, 2):
        return dict(kind='BASE', attempt=attempt)
    matches = [record for record in recovery_authorizations(job)
               if attempt in record['allowed_attempts']]
    if len(matches) != 1:
        raise ValueError('Attempt has no exact technical recovery authorization')
    if started_unix is not None and datetime.fromisoformat(matches[0]['created_utc']).timestamp() > float(started_unix):
        raise ValueError('Recovery authorization was not approved before attempt start')
    return dict(kind='TECHNICAL_RECOVERY', attempt=attempt, **matches[0])


def choose_epoch(search, epochs):
    if search['summary']['status'] != 'SEGMENT_COMPLETE':
        raise RuntimeError('Search did not finish its registered schedule')
    curve = [(float(row['epoch']), float(row['astop_mse'])) for row in search['curves']]
    if [point[0] for point in curve] != [0., *epochs] or not all(math.isfinite(x[1]) for x in curve):
        raise ValueError('A_stop curve differs from the complete registered schedule')
    selected, mse = min(curve, key=lambda point: (point[1], point[0]))
    summary = search['summary']
    if summary['selected_epoch'] != selected or summary['selected_astop_mse'] != mse:
        raise ValueError('Selection is not minimum A_stop with the earlier exact tie')
    return selected, mse


def check_selection(path, digest, job):
    seal = runtime.read(path)
    if (runtime.sha(path) != digest or seal['job'] != job
            or seal['status'] != 'SEALED' or seal['V_used'] is not False):
        raise ValueError('Selection seal changed before fresh A60/V')


def predict_saved(model, norm, task, foundation, spec, path, modules, *, deterministic=False):
    import numpy as np
    import torch
    from .models import evaluation_scope
    from research.adapter_diagnostics_20261017_v2.existing_jobs import score_arrays
    origins = modules['data'].origins(task['split_ranges']['V_select'], task['L'], task['H'], task['H'])
    prefix_end = int(origins.max()) + task['H']
    if prefix_end > task['split_ranges']['T_final'][0]:
        raise ValueError('V prediction would cross into T')
    prefix = dict(task, physical=task['physical'][:prefix_end], observed=task['observed'][:prefix_end])
    windows = modules['training'].FastWindows(prefix, norm['mean'], norm['std'], spec['device'])
    pieces = {name: [] for name in ('pred_normalized', 'target_normalized', 'observed')}
    before, started = foundation.streams, time.perf_counter()
    with evaluation_scope(model, deterministic=deterministic):
        for start in range(0, len(origins), 8):
            x, y, mask = windows.get(origins[start:start + 8])
            p = modules['fit'].predict_model(model, x, foundation, spec['family'])
            if p.dtype != torch.float32 or p.shape != y.shape or not torch.isfinite(p[mask]).all():
                raise ValueError('V output violates the FP32 finite/shape contract')
            for key, value in zip(pieces, (p, y, mask)):
                pieces[key].append(value.cpu().numpy())
    values = {key: np.concatenate(value) for key, value in pieces.items()}
    values.update(origins=origins.astype(np.int64), channel_ids=np.asarray(task['columns']),
                  mean=np.asarray(norm['mean'], np.float32), std=np.asarray(norm['std'], np.float32))
    measured = score_arrays(values['pred_normalized'], values['target_normalized'], values['observed'])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **values)
    with np.load(path, allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    independent, _ = modules['evaluation'].error_statistics(saved['pred_normalized'], saved['target_normalized'],
        saved['observed'], saved['origins'], spec['run_id'], 'V_select', task['dataset'], saved['channel_ids'])
    if max(abs(measured['MSE'] - independent['mse']), abs(measured['MAE'] - independent['mae'])) > 1e-9:
        raise ValueError('Independent saved-array FP64 score mismatch')
    samples = 10 if spec['method_id'] == 'C3' and not deterministic else 1
    logical_rows = foundation.streams - before
    if logical_rows != len(origins) * spec['latent_k'] * samples:
        raise ValueError('Unexpected foundation row count in V prediction')
    return dict(MSE=measured['MSE'], MAE=measured['MAE'], prediction_path=str(path),
                prediction_sha256=runtime.sha(path), independent_FP64='PASS',
                logical_F_rows=logical_rows, evaluation_seconds=time.perf_counter() - started,
                metric_accumulation_dtype='float64', point_statistic='quantile_0.5',
                n_origins=len(origins), n_observed=measured['observed'], deterministic=deterministic)


def run_stages(job, task, foundation, modules, spec, cache, public, deadline):
    """The injectable orchestration seam used by synthetic ordering/bounds tests."""
    fit, training = modules['fit'], modules['training']
    fit_end, refit_end = runtime.FIT_ENDS[job['dataset']]
    bounds = task['split_ranges']
    if (bounds['A_fit'][1], bounds['A_stop'][1]) != (fit_end, refit_end):
        raise ValueError('Registered split endpoints changed')
    core_sha, core_files = fit._source_contract()
    axes = dict(L=task['L'], H=task['H'], C=task['physical'].shape[1], columns=task['columns'],
                split_ranges=bounds, source_array_fingerprints=task['array_sha256'])
    contract = dict(source_sha=core_sha, config_sha=training.fingerprint(dict(spec=spec, axes=axes)),
                    data_sha=task['data_sha256'], model_sha=foundation.initial_hash)
    runtime.write_new(public / 'SOURCE_CONTRACT.json', dict(contract, sources=core_files, spec=spec))
    if job['method_id'] == 'C1':
        selected, selected_mse, search = 0., None, None
    else:
        search = fit._stage(spec, task, fit_end, bounds['A_stop'], spec['epochs'],
                            cache / 'search', public / 'search', foundation, deadline, contract)
        selected, selected_mse = choose_epoch(search, spec['epochs'])
        runtime.write_new(public / 'SEARCH_RESULT.json', dict(summary=search['summary'], curves=search['curves']))
    seal_path = public / 'SELECTION.json'
    runtime.write_new(seal_path, dict(status='SEALED', job=job, V_used=False,
        selection_split='NONE_FIXED_PCA' if search is None else 'A_stop',
        selected_epoch=selected, selected_astop_mse=selected_mse, spec=spec, contract=contract,
        curves=[] if search is None else search['curves']))
    seal_sha = runtime.sha(seal_path)
    check_selection(seal_path, seal_sha, job)
    if search is None:
        model, norm = fit.create_model(spec, task, refit_end, foundation)
        refit = dict(model=training.cpu_state(model), normalization=norm,
                     summary=dict(status='SEGMENT_COMPLETE', seconds=0., exposures=0, updates=0,
                                  trainable_parameters=0))
    else:
        refit = fit._stage(spec, task, refit_end, None, [selected] if selected else [],
                          cache / 'refit', public / 'refit', foundation, deadline, contract)
        if refit['summary']['status'] != 'SEGMENT_COMPLETE':
            raise RuntimeError('Fresh A60 refit did not finish')
        model, norm = fit.create_model(spec, task, refit_end, foundation)
        model.load_state_dict(refit['model'], strict=True)
    bundle_path = cache / 'selected_bundle.pt'
    fit._torch_new(bundle_path, dict(model=refit['model'], normalization=norm, spec=spec, contract=contract,
        selected_search_epoch=selected, refit_summary=refit['summary'],
        search_summary=None if search is None else search['summary'], selection_sha256=seal_sha))
    runtime.write_new(public / 'REFIT_SUMMARY.json', refit['summary'])
    check_selection(seal_path, seal_sha, job)
    scored = predict_saved(model, norm, task, foundation, spec, cache / 'selected_A60_V_select.npz', modules)
    deterministic = None
    if job['method_id'] == 'C3':
        deterministic = predict_saved(model, norm, task, foundation, spec,
            cache / 'diagnostic_dropout_off_V_select.npz', modules, deterministic=True)
    row = dict(dataset=job['dataset'], method_id=job['method_id'], family=spec['family'], K=job['K'],
        seed=job['seed'], job_id=job['job_id'], reused=False, extended=job['extended'],
        selected_epoch=selected, training_cap=spec['epochs'][-1] if spec['epochs'] else 0,
        cap_selected=bool(spec['epochs']) and selected == spec['epochs'][-1],
        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        search_seconds=0. if search is None else search['summary']['seconds'],
        refit_seconds=refit['summary']['seconds'], normalization_fit_end=refit_end, **scored)
    return dict(status='COMPLETE', job=job, spec=spec, row=row, rows=[row],
        selection_path=str(seal_path), selection_sha256=seal_sha,
        selection_sealed_before_V=True, bundle_path=str(bundle_path), bundle_sha256=runtime.sha(bundle_path),
        search_summary=None if search is None else search['summary'], refit_summary=refit['summary'],
        deterministic_diagnostic=deterministic, task_access=task['_campaign_access'], contract=contract)


def run_job(job, *, attempt=1, deadline=None, source=runtime.DEFAULT_SOURCE, asset=runtime.DEFAULT_ASSET, worker_started_unix=None):
    registration = validate_registration(job)
    attempt_authorization = validate_attempt(job, attempt, started_unix=worker_started_unix)
    preserved_before = runtime.guard_preserved()
    result_path = runtime.REPORT / 'jobs' / job['job_id'] / 'RESULT.json'
    if result_path.exists():
        saved = runtime.read(result_path)
        if saved['job'] != job or saved['registration'] != registration:
            raise ValueError('Existing successful job registration differs')
        for field, digest in (('bundle_path', 'bundle_sha256'),):
            if runtime.sha(saved[field]) != saved[digest]:
                raise ValueError('Existing selected bundle changed')
        if runtime.sha(saved['row']['prediction_path']) != saved['row']['prediction_sha256']:
            raise ValueError('Existing successful V archive changed')
        return saved
    cache = runtime.CACHE / 'jobs' / job['job_id'] / f'attempt_{attempt:03d}'
    public = runtime.REPORT / 'jobs' / job['job_id'] / f'attempt_{attempt:03d}'
    if cache.exists() or public.exists():
        raise FileExistsError('Attempt already exists; controller must use the next registered attempt')
    cache.mkdir(parents=True)
    public.mkdir(parents=True)
    opened = runtime.record_opens()
    started, foundation, result = time.time(), None, None
    receipt = dict(job=job, attempt=attempt, started_utc=datetime.now(timezone.utc).isoformat(),
                   status='FAILED', registration=registration, preserved_before=preserved_before,
                   training_attempted=False, training_executed=False, T_numeric_values_parsed=0)
    try:
        set_phase('IMPORTS')
        modules = runtime.load_modules()
        from . import models
        spec = build_spec(job)
        set_phase('DATA')
        task = runtime.load_task(job['dataset'], source=source, asset=asset)
        runtime.assert_protected_T(dict(job=job, task_access=task['_campaign_access']))
        set_phase('FOUNDATION')
        foundation = runtime.load_foundation(asset=asset)
        runtime.write_new(public / 'START.json', dict(receipt, spec=spec, device=str(foundation.device)))
        receipt['training_attempted'] = job['method_id'] != 'C1'
        receipt['training_executed'] = None if receipt['training_attempted'] else False
        set_phase('TRAINING')
        with models.patch_core(modules['fit']):
            result = run_stages(job, task, foundation, modules, spec, cache, public,
                                time.time() + 7200 if deadline is None else deadline)
        set_phase('FINAL_VALIDATION')
        foundation.assert_frozen_unchanged()
        if source_hashes() != spec['new_scientific_source_hashes']:
            raise ValueError('Campaign implementation changed during the job')
        receipt['preserved_after'] = runtime.guard_preserved()
        receipt['status'] = 'COMPLETE'
        receipt['training_executed'] = bool((result['search_summary'] or {}).get('updates', 0)
                                            + result['refit_summary']['updates'])
        result.update(registration=registration, attempt=attempt, attempt_authorization=attempt_authorization, elapsed_seconds=time.time() - started,
                      source_roots=dict(source=str(source), asset=str(asset)),
                      peak_allocated_bytes=modules['gpu'].torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=modules['gpu'].torch.cuda.max_memory_reserved())
    except BaseException as exc:
        receipt.update(error=f'{type(exc).__name__}: {exc}', failure_kind=classify_failure(exc, _PHASE), phase=_PHASE)
        raise
    finally:
        receipt['elapsed_seconds'] = time.time() - started
        frozen_error = None
        if foundation is not None:
            try:
                foundation.assert_frozen_unchanged()
            except BaseException as exc:
                frozen_error = exc
                receipt['status'] = 'FAILED'
                receipt['frozen_foundation_error'] = f'{type(exc).__name__}: {exc}'
        runtime.write_new(public / 'RECEIPT.json', receipt)
        runtime.write_new(public / 'OPENED_FILES.json', dict(
            scope='Python audit open events; native model loading may not emit these events',
            T_numeric_values_parsed=0, opens=[dict(path=p, mode=m, flags=f) for p, m, f in sorted(opened)]))
        foundation = None
        gc.collect()
        if frozen_error is not None:
            raise frozen_error
    runtime.assert_protected_T(result)
    set_phase('WRITING_RESULTS')
    runtime.write_new(result_path, result)
    return result


_PHASE = 'BOOTSTRAP'


def set_phase(phase):
    global _PHASE
    _PHASE = phase
    path = runtime.CACHE / 'LIVE_WORKER.json'
    if path.exists():
        record = runtime.read(path)
        if record.get('pid') == os.getpid():
            runtime.atomic(path, dict(record, phase=phase, phase_updated_utc=datetime.now(timezone.utc).isoformat()))


SCIENTIFIC_MESSAGES = (
    'Job differs from', 'Job identifier differs', 'Job absent or duplicated',
    'Registration ', 'Registered source ', 'Campaign source missing',
    'A_stop curve differs', 'Selection is not minimum A_stop', 'Selection seal changed',
    'V prediction would cross', 'V output violates', 'Independent saved-array metric mismatch',
    'Independent saved-array FP64 score mismatch',
    'Unexpected foundation row count', 'Registered split endpoints changed',
    'Existing successful ', 'Existing selected bundle changed',
    'Existing protected file changed', 'Cannot change protected existing file',
    'Campaign implementation changed', 'Foundation state changed',
    'Frozen backbone weights changed', 'Frozen backbone acquired parameter gradients',
    'Foundation row/horizon output does not match contract', 'H exceeds actual local Bolt output',
    'Historical prefix ', 'Historical pre-T ', 'Historical A_fit/A60 ',
    'Bounded prefix ', 'Unsupported bounded prefix ', 'Protected T proof ',
    'Jena bounded loader ', 'Dataset is outside', 'Registered real campaign device',
    'Only registered P branches', 'Unknown P branch', 'Only the first attempt',
    'Attempt already exists', 'Attempt ', 'Recovery authorization ', 'Scientific source changed', 'Run source/config/data/model contract changed',
    'Invalid A_stop forecast', 'A_stop contains no observed target', 'No legal training origins',
    'No legal A_stop origins', 'Selected artifact bytes changed',
)


def classify_failure(exc, phase):
    message = str(exc)
    if any(message.startswith(prefix) for prefix in SCIENTIFIC_MESSAGES):
        return 'SCIENTIFIC'
    if isinstance(exc, FileNotFoundError) and any(name in message for name in
            ('PREREGISTRATION', 'REGISTRATION_SEAL', 'RUN_PLAN', 'PRESERVATION_BASELINE', 'REFERENCE_RUNS',
             'Pinned local Bolt snapshot is absent')):
        return 'SCIENTIFIC'
    if isinstance(exc, TimeoutError):
        return 'TECHNICAL'
    if isinstance(exc, (ImportError, ModuleNotFoundError)) and phase == 'IMPORTS':
        return 'TECHNICAL'
    if isinstance(exc, PermissionError) and getattr(exc, 'winerror', None) in (5, 32, 33):
        filename = getattr(exc, 'filename', None)
        if filename and any(Path(filename).resolve().is_relative_to(p.resolve()) for p in (runtime.REPORT, runtime.CACHE)):
            return 'TECHNICAL'
    if isinstance(exc, RuntimeError) and any(token in message.lower() for token in
            ('cuda out of memory', 'cuda error: out of memory', 'cuda driver error',
             'driver shutting down', 'cuda context', 'cublas_status_alloc_failed',
             'another live process holds')):
        return 'TECHNICAL'
    return 'UNKNOWN'


WORKER_RESERVATIONS = []


def retain_worker_reservations():
    own = runtime.CampaignLock(runtime.CACHE / 'worker.lock')
    shared = runtime.CampaignLock(runtime.SHARED_CACHE / 'worker.lock')
    own.__enter__()
    try:
        shared.__enter__()
    except BaseException:
        own.__exit__(None, None, None)
        raise
    WORKER_RESERVATIONS.extend((own, shared))


def main(argv=None):
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--attempt', type=int, default=1)
    parser.add_argument('--deadline', type=float, help='Optional per-attempt limit, never a campaign admission deadline')
    args = parser.parse_args(argv)
    started = time.time()
    identity = psutil.Process(os.getpid())
    receipt = dict(job_id=None, attempt=args.attempt, pid=os.getpid(),
                   process_creation_time=identity.create_time(), started_unix=started,
                   status='FAILED', failure_kind=None, phase='REGISTRATION', exit_code=1)
    job = None
    try:
        set_phase('REGISTRATION')
        job = runtime.read(args.job)
        receipt['job_id'] = job['job_id']
        runtime.install_write_guard()
        retain_worker_reservations()
        runtime.atomic(runtime.CACHE / 'LIVE_WORKER.json', dict(receipt, status='RUNNING', job=job))
        result = run_job(job, attempt=args.attempt, deadline=args.deadline, worker_started_unix=started)
        runtime.assert_protected_T(result)
        runtime.guard_preserved()
        receipt.update(status='COMPLETE', phase='COMPLETE', exit_code=0)
    except BaseException as exc:
        kind = classify_failure(exc, _PHASE)
        receipt.update(error=f'{type(exc).__name__}: {exc}', failure_kind=kind, phase=_PHASE)
        if isinstance(exc, KeyboardInterrupt):
            receipt.update(status='USER_CANCELLED', failure_kind='USER_CANCELLED', exit_code=130)
        elif kind in ('SCIENTIFIC', 'UNKNOWN'):
            receipt.update(status='SCIENTIFIC_SUSPEND', exit_code=20)
        import traceback
        traceback.print_exc()
    finally:
        receipt.update(finished_unix=time.time(), wall_seconds=time.time() - started)
        if job is not None:
            runtime.write_new(runtime.REPORT / 'jobs' / job['job_id'] /
                              f'WORKER_RECEIPT_attempt{args.attempt}.json', receipt)
            live = runtime.CACHE / 'LIVE_WORKER.json'
            if live.exists() and runtime.read(live).get('pid') == os.getpid():
                runtime.atomic(live, dict(receipt, job=job))
    print(json.dumps(receipt, ensure_ascii=False, allow_nan=False), flush=True)
    return receipt['exit_code']


if __name__ == '__main__':
    raise SystemExit(main())
