"""A_stop search, explicit checkpoint branches and fresh A60 refit."""
import copy
import hashlib
import math
from pathlib import Path
import re
import time

import numpy as np
import torch

from research.adapter_improvements_20261018_v1 import training_jobs as previous
from research.adapter_diagnostics_20261017_v2 import existing_jobs as existing
from . import contracts, revin


RECIPE = contracts.RECIPE


def _modules(ctx):
    return ctx['modules'] if 'modules' in ctx else existing._modules(ctx)


def _scientific_sources(ctx):
    root = Path(ctx['repo']).resolve()
    path = Path(__file__).resolve()
    files = dict(ctx.get('scientific_source_hashes', {}))
    own = path.relative_to(root).as_posix()
    if own in files and files[own] != contracts.sha(path):
        raise ValueError('Registered scientific source hash mismatch: ' + own)
    files[own] = contracts.sha(path)
    for name, expected in files.items():
        if contracts.sha(root / name) != expected:
            raise ValueError('Registered scientific source hash mismatch: ' + name)
    return files


def _spec(job, ctx):
    for name, expected in RECIPE.items():
        if name in job and job[name] != expected:
            raise ValueError('Unregistered recipe override: ' + name)
    if type(job.get('revin')) is not bool:
        raise ValueError('RevIN policy must be bool')
    if job['family'] not in ('B2', 'B0') or job.get('init_kind', 'PCA') != 'PCA':
        raise ValueError('Only B2/B0 PCA is registered')
    if int(job['seed']) not in contracts.SEEDS:
        raise ValueError('Unregistered training seed')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', str(job['job_id'])):
        raise ValueError('Unsafe registered job ID')
    sources = _scientific_sources(ctx)
    for name, expected in job.get('scientific_source_hashes', {}).items():
        if sources.get(name) != expected:
            raise ValueError('Job scientific source hash mismatch: ' + name)
    return dict(RECIPE, run_id=job['job_id'], dataset=job['cohort'], family=job['family'],
                init_kind='PCA', latent_k=int(job['K']), seed=int(job['seed']),
                device=job.get('device', 'cuda'), fold='primary', core_dtype='float32',
                revin=job['revin'], new_scientific_source_hashes=sources,
                scientific_identity=job.get('scientific_identity', {}))


def _resources(job, ctx):
    return previous._resources(job, ctx)


def _axes(task, spec):
    return dict(L=int(task['L']), H=int(task['H']), C=int(task['physical'].shape[1]),
                columns=list(task['columns']), fold=spec.get('fold', 'primary'),
                source_array_fingerprints=task.get('array_sha256', {}))


def _base_contract(spec, task, foundation, modules):
    source_sha, sources = modules['fit']._source_contract()
    return dict(source_sha=source_sha,
                config_sha=modules['training'].fingerprint(dict(spec=spec, axes=_axes(task, spec))),
                data_sha=task['data_sha256'], model_sha=modules['fit']._model_sha(foundation)), sources


def _stage_contract(base, norm, fit_end, modules):
    fit, data, training = modules['fit'], modules['data'], modules['training']
    result = dict(base, scaler_sha=fit.scaler_fingerprint(norm['mean'], norm['std']))
    result['initialization_sha'] = training.fingerprint(dict(scaler_sha=result['scaler_sha'], fit_end=fit_end,
        basis_sha=None if norm['basis'] is None else data.array_sha256(norm['basis']),
        center_sha=None if norm['pca_center'] is None else data.array_sha256(norm['pca_center'])))
    return result


def _state_fingerprint(value):
    def encode(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            return dict(dtype=str(tensor.dtype), shape=list(tensor.shape), sha256=hashlib.sha256(raw).hexdigest())
        if isinstance(item, np.ndarray):
            return dict(dtype=str(item.dtype), shape=list(item.shape), sha256=hashlib.sha256(item.tobytes()).hexdigest())
        if isinstance(item, dict):
            return {str(key): encode(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [encode(child) for child in item]
        return item
    return contracts.fingerprint(encode(value))


def _state_without_contract(checkpoint):
    return {key: value for key, value in checkpoint.items() if key != 'contract'}


def _source_path(name, ctx, modules):
    return Path(modules['fit'].__file__).resolve().parent / name if '/' not in name else Path(ctx['repo']) / name


def _import_parent(parent, active, spec, task, norm, stage_dir, public_dir, ctx, modules):
    if not isinstance(parent, dict) or not all(key in parent for key in ('path', 'sha256', 'contract', 'source_files', 'spec')):
        raise ValueError('Bound parent checkpoint descriptor is required')
    path = Path(parent['path'])
    if contracts.sha(path) != parent['sha256']:
        raise ValueError('Parent checkpoint SHA changed')
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if checkpoint['contract'] != parent['contract']:
        raise ValueError('Bound parent contract differs from checkpoint')
    for key in ('model', 'optimizer', 'sampler', 'torch_rng', 'cuda_rng', 'updates', 'extra'):
        if key not in checkpoint:
            raise ValueError('Parent checkpoint lacks complete ' + key)
    if not checkpoint['optimizer'].get('state'):
        raise ValueError('Parent checkpoint has no optimizer state')
    for name, expected in parent['source_files'].items():
        if contracts.sha(_source_path(name, ctx, modules)) != expected:
            raise ValueError('Parent scientific source changed: ' + name)
    if modules['training'].fingerprint(parent['source_files']) != checkpoint['contract']['source_sha']:
        raise ValueError('Parent source contract fingerprint mismatch')
    parent_spec = parent['spec']
    for key in ('family', 'latent_k', 'seed', 'dataset', 'device'):
        if parent_spec.get(key) != spec.get(key):
            raise ValueError('Parent recipe mismatch: ' + key)
    for key, value in RECIPE.items():
        if parent_spec.get(key) != value:
            raise ValueError('Parent recipe mismatch: ' + key)
    if parent_spec.get('init_kind', 'PCA') != 'PCA' or parent_spec.get('revin', False) is not spec['revin']:
        raise ValueError('Parent normalization or initialization changed')
    if parent_spec.get('epochs') != list(contracts.INITIAL_EPOCHS):
        raise ValueError('Parent must be the completed three-epoch stage')
    if modules['training'].fingerprint(dict(spec=parent_spec, axes=_axes(task, parent_spec))) != checkpoint['contract']['config_sha']:
        raise ValueError('Parent configuration or data axes changed')
    for key in ('data_sha', 'model_sha', 'scaler_sha', 'initialization_sha'):
        if checkpoint['contract'].get(key) != active.get(key):
            raise ValueError('Parent active binding mismatch: ' + key)
    legal = modules['data'].origins((0, norm['fit_end']), task['L'], task['H'], 1)
    sampler = modules['training'].PermutationStream(legal, spec['seed'])
    expected_exposures = 3 * len(legal)
    sampler.draw(expected_exposures)
    if checkpoint['sampler'] != sampler.state_dict() or checkpoint['sampler']['exposures'] != expected_exposures:
        raise ValueError('Parent sampler, order, RNG or exposures changed')
    expected_updates = sum(math.ceil((right-left)/16) for left, right in zip(
        [0, math.ceil(.25*len(legal)), len(legal)], [math.ceil(.25*len(legal)), len(legal), expected_exposures]))
    if checkpoint['updates'] != expected_updates or checkpoint['extra']['completed_epochs'] != [0., .25, 1., 3.]:
        raise ValueError('Parent updates or epoch history changed')
    stage_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)
    original = stage_dir / 'parent_original.pt'
    if original.exists():
        if contracts.sha(original) != parent['sha256']:
            raise ValueError('Immutable original parent copy changed')
    else:
        with original.open('xb') as handle:
            handle.write(path.read_bytes())
    migrated = copy.deepcopy(checkpoint)
    migrated['contract'] = active
    before = _state_fingerprint(_state_without_contract(checkpoint))
    after = _state_fingerprint(_state_without_contract(migrated))
    if before != after:
        raise ValueError('Checkpoint branch altered training state')
    branch = stage_dir / 'checkpoint_000000.pt'
    if not branch.exists():
        modules['fit']._torch_new(branch, migrated)
    imported = torch.load(branch, map_location='cpu', weights_only=False)
    if imported['contract'] != active or _state_fingerprint(_state_without_contract(imported)) != before:
        raise ValueError('Imported checkpoint altered state or branch contract')
    proof = dict(parent_checkpoint=contracts.binding(path, ctx), parent_checkpoint_sha256=parent['sha256'],
                 imported_checkpoint=contracts.binding(branch, ctx), original_parent_copy=contracts.binding(original, ctx),
                 parent_contract=checkpoint['contract'], branch_contract=active,
                 parent_source_files=parent['source_files'], parent_spec=parent_spec,
                 state_preserved=True, preserved_state_sha256=before,
                 imported_original_sha256=contracts.sha(original),
                 changed_fields=['contract'], reason='Explicit new immutable three-to-six epoch continuation',
                 sampler_exposures=expected_exposures, optimizer_updates=expected_updates,
                 initialization_sha256=active['initialization_sha'], source_migration_explicit=True)
    proof_path = public_dir / 'MIGRATION_PROOF.json'
    contracts.sealed_json(proof_path, proof)
    return proof_path


def search(job, ctx, epochs=contracts.INITIAL_EPOCHS, parent_checkpoint=None):
    """Train/select exclusively on A_fit/A_stop; never refit or materialize V."""
    schedule = list(map(float, epochs))
    if schedule not in (list(contracts.INITIAL_EPOCHS), list(contracts.EXTENDED_EPOCHS)):
        raise ValueError('Only registered three/six epoch schedules are allowed')
    if parent_checkpoint is not None and schedule != list(contracts.EXTENDED_EPOCHS):
        raise ValueError('Checkpoint migration is only for the six-epoch branch')
    spec = dict(_spec(job, ctx), epochs=schedule)
    modules, task, foundation = _resources(job, ctx)
    bounds = previous._registered_splits(task)
    cap = int(schedule[-1])
    cache_run = Path(ctx['cache']) / 'runs' / job['job_id'] / f'cap{cap}'
    public_run = Path(ctx['report']) / 'runs' / job['job_id'] / f'cap{cap}'
    cache_run.mkdir(parents=True, exist_ok=True)
    public_run.mkdir(parents=True, exist_ok=True)
    deadline = min(ctx['deadline'], ctx.get('job_deadline', ctx['deadline']))
    factory_ctx = dict(ctx, modules=modules, scientific_source_hashes=_scientific_sources(ctx))
    with revin.model_factory(factory_ctx) as create:
        base, source_files = _base_contract(spec, task, foundation, modules)
        contracts.sealed_json(cache_run / 'RUN_CONTRACT.json', base)
        contracts.sealed_json(public_run / 'SOURCE_CONTRACT.json', dict(base, files=source_files, spec=spec))
        completed = cache_run / 'SEARCH_RESULT.json'
        if completed.exists():
            result = contracts.read(completed)
            if result['contract'] != base or contracts.sha(result['checkpoint_path']) != result['parent_checkpoint']['sha256']:
                raise ValueError('Completed search bytes or contract changed')
            if ('migration_proof_binding' in result) != (parent_checkpoint is not None):
                raise ValueError('Completed search migration parent identity changed')
            if parent_checkpoint is not None:
                proof = contracts.read(contracts.resolve_binding(result['migration_proof_binding'], ctx))
                contracts.resolve_binding(proof['imported_checkpoint'], ctx)
                contracts.resolve_binding(proof['original_parent_copy'], ctx)
                if (proof['parent_checkpoint_sha256'] != parent_checkpoint.get('sha256')
                        or proof['parent_contract'] != parent_checkpoint.get('contract')
                        or proof['parent_spec'] != parent_checkpoint.get('spec')
                        or proof['parent_source_files'] != parent_checkpoint.get('source_files')
                        or contracts.sha(parent_checkpoint['path']) != proof['parent_checkpoint_sha256']):
                    raise ValueError('Completed search migration parent changed')
            return result
        model, norm = create(spec, task, bounds['A_fit'][1], foundation)
        active = _stage_contract(base, norm, bounds['A_fit'][1], modules)
        initial_state = _state_fingerprint(model.base.state_dict() if spec['revin'] else model.state_dict())
        migration = None
        if parent_checkpoint is not None:
            migration = _import_parent(parent_checkpoint, active, spec, task, norm,
                cache_run / 'search', public_run / 'search', ctx, modules)
        stage = modules['fit']._stage_body(spec, task, bounds['A_fit'][1], bounds['A_stop'], schedule,
                    cache_run / 'search', public_run / 'search', foundation, deadline, base, model, norm)
        summary = stage['summary']
        result = dict(status='COMPLETE' if summary['status'] == 'SEGMENT_COMPLETE' else 'PARTIAL_CHECKPOINTED',
                      spec=spec, contract=base, source_files=source_files, curves=stage['curves'],
                      selected_epoch=summary['selected_epoch'], summary=summary, cap=cap,
                      initial_base_state_sha256=initial_state, V_used=False, selection_split='A_stop')
        latest = sorted((cache_run / 'search').glob('checkpoint_*.pt'))[-1]
        checkpoint = torch.load(latest, map_location='cpu', weights_only=False)
        result.update(checkpoint_path=str(latest), checkpoint_binding=contracts.binding(latest, ctx),
                      parent_checkpoint=dict(path=str(latest), sha256=contracts.sha(latest),
                          contract=checkpoint['contract'], source_files=source_files, spec=spec))
        if migration is not None:
            result['migration_proof_path'] = str(migration)
            result['migration_proof_binding'] = contracts.binding(migration, ctx)
        public = {key: value for key, value in result.items() if key not in ('checkpoint_path', 'parent_checkpoint', 'migration_proof_path')}
        if result['status'] == 'COMPLETE':
            contracts.sealed_json(public_run / 'search' / 'RECEIPT.json', summary)
            contracts.sealed_json(public_run / 'SEARCH_RESULT.json', public)
            contracts.sealed_json(completed, result)
        return result


def _selection_decision(job, ctx, cap):
    value = ctx.get('selection_decision')
    if not isinstance(value, dict) or 'path' not in value or 'sha256' not in value:
        raise ValueError('Byte-bound group selection decision required before V')
    path = Path(value['path'])
    if contracts.sha(path) != value['sha256']:
        raise ValueError('Group selection decision SHA changed')
    decision = contracts.read(path)
    if (decision.get('cohort') != job['cohort'] or decision.get('cap') != cap
            or decision.get('selection_split') != 'A_stop' or decision.get('V_used') is not False
            or decision.get('status') != 'SEALED'):
        raise ValueError('Group selection decision identity or information boundary changed')
    if len(decision.get('matched_arms', [])) != 4 or len(set(decision['matched_arms'])) != 4:
        raise ValueError('Whole four-arm matched selection decision required before V')
    if job['family'] == 'B2' and job['job_id'] not in decision.get('matched_arms', []):
        raise ValueError('Model is absent from matched group selection decision')
    return contracts.binding(path, ctx)


def _refit_predict(job, ctx, search_result, cap):
    """Fresh A60 fit at the sealed A_stop epoch, then save and verify V arrays."""
    decision = _selection_decision(job, ctx, cap)
    spec = dict(_spec(job, ctx), epochs=list(contracts.INITIAL_EPOCHS if cap == 3 else contracts.EXTENDED_EPOCHS))
    if cap not in (3, 6) or search_result.get('status') != 'COMPLETE' or search_result.get('spec') != spec:
        raise ValueError('Complete matching search stage is required for refit')
    modules, task, foundation = _resources(job, ctx)
    bounds = previous._registered_splits(task)
    cache_run = Path(ctx['cache']) / 'runs' / job['job_id'] / f'cap{cap}'
    public_run = Path(ctx['report']) / 'runs' / job['job_id'] / f'cap{cap}'
    output = public_run / 'REFIT_RESULT.json'
    stored_search = contracts.read(cache_run / 'SEARCH_RESULT.json')
    if any(stored_search.get(key) != search_result.get(key)
           for key in ('spec', 'contract', 'curves', 'selected_epoch', 'summary', 'cap')):
        raise ValueError('Sealed search selection or scientific state changed')
    deadline = min(ctx['deadline'], ctx.get('job_deadline', ctx['deadline']))
    factory_ctx = dict(ctx, modules=modules, scientific_source_hashes=_scientific_sources(ctx))
    with revin.model_factory(factory_ctx) as create:
        base, source_files = _base_contract(spec, task, foundation, modules)
        if base != search_result['contract']:
            raise ValueError('Search/refit scientific contract changed')
        if output.exists():
            result = contracts.read(output)
            if result['selection_decision_binding'] != decision:
                raise ValueError('Refit selection decision changed')
            contracts.resolve_binding(result['bundle_binding'], ctx)
            contracts.resolve_binding(result['rows'][0]['prediction_binding'], ctx)
            return result
        selected = float(search_result['selected_epoch'])
        refit = modules['fit']._stage(spec, task, bounds['A_stop'][1], None, [selected] if selected else [],
            cache_run / 'refit', public_run / 'refit', foundation, deadline, base)
        if refit['summary']['status'] != 'SEGMENT_COMPLETE':
            return dict(status='PARTIAL_CHECKPOINTED', phase='fresh_A60_refit', refit_summary=refit['summary'])
        model_sha = modules['training'].fingerprint({key: modules['data'].array_sha256(value.numpy())
                                                   for key, value in refit['model'].items()})
        bundle = dict(model=refit['model'], normalization=refit['normalization'], spec=spec,
                      contract=refit['summary']['contract'], selected_search_epoch=selected,
                      search_summary=search_result['summary'], refit_summary=refit['summary'], model_state_sha=model_sha)
        artifact = cache_run / 'selected_bundle.pt'
        if artifact.exists():
            stored = torch.load(artifact, map_location='cpu', weights_only=False)
            keys = ('model', 'normalization', 'spec', 'contract', 'selected_search_epoch', 'model_state_sha', 'search_summary')
            if _state_fingerprint({key: stored[key] for key in keys}) != _state_fingerprint({key: bundle[key] for key in keys}):
                raise ValueError('Existing selected fresh-A60 bundle changed')
            bundle = stored
            refit['summary'] = stored['refit_summary']
        else:
            modules['fit']._torch_new(artifact, bundle)
        receipt = dict(status='COMPLETE_VALID', run_id=job['job_id'], family=job['family'], revin=job['revin'],
            seed=job['seed'], epochs=spec['epochs'], training_cap=cap, selected_epoch=selected,
            selected_astop_mse=search_result['summary']['selected_astop_mse'], optimizer='AdamW', weight_decay=0.,
            search_fit_end=bounds['A_fit'][1], refit_fit_end=bounds['A_stop'][1], normalization_fit_end=bounds['A_stop'][1],
            search_train_origins=search_result['summary']['train_origins'], search_stop_origins=search_result['summary']['stop_origins'],
            search_exposures=search_result['summary']['exposures'], search_updates=search_result['summary']['updates'],
            refit_exposures=refit['summary']['exposures'], refit_updates=refit['summary']['updates'],
            artifact_cache_relative_path=artifact.relative_to(Path(ctx['cache'])).as_posix(),
            artifact_sha256=contracts.sha(artifact), model_state_sha=model_sha,
            source_sha=base['source_sha'], refit_scaler_sha=refit['summary']['contract']['scaler_sha'],
            selection_decision_binding=decision, fresh_A60=True,
            training_status='CAP_OR_UNDERTRAINING' if selected == cap else 'SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED')
        contracts.sealed_json(public_run / 'refit' / 'RECEIPT.json', refit['summary'])
        contracts.sealed_json(public_run / 'BASE_FIT_RECEIPT.json', receipt)
        receipt_binding = contracts.binding(public_run / 'BASE_FIT_RECEIPT.json', ctx)
        model, norm = create(spec, task, bounds['A_stop'][1], foundation)
        model.load_state_dict(bundle['model'], strict=True)
        norm['family'] = job['family']
        origins = modules['data'].origins(bounds['V_select'], task['L'], task['H'], task['H'])
        archive = Path(ctx['cache']) / 'predictions' / job['job_id'] / f'cap{cap}' / 'selected_A60_V_select.npz'
        scored = previous._predict_saved(model, norm, task, origins, foundation, modules, archive, ctx)
        row = dict(scored, run_id=job['job_id'], method=job['family'] + '_REVIN_' + ('ON' if job['revin'] else 'OFF'),
            family=job['family'], revin=job['revin'], seed=job['seed'], dataset=job['cohort'],
            C=task['physical'].shape[1], K=job['K'], L=task['L'], H=task['H'], device=spec['device'],
            cap=cap, stage='selected_refit', evaluation='V_select', fit='fresh_A60', selected_epoch=selected,
            updates=refit['summary']['updates'], exposures=refit['summary']['exposures'],
            target_trained_params=sum(p.numel() for p in model.parameters() if p.requires_grad),
            P_params=sum(p.numel() for p in model.front.parameters()), E_params=model.E.numel(), D_params=model.D.numel(),
            RevIN_params=2*task['physical'].shape[1] if spec['revin'] else 0,
            source_sha=base['source_sha'], config_sha=base['config_sha'], data_sha=base['data_sha'], model_sha=base['model_sha'],
            selection_split='A_stop', V_exposure='previously_reused_development',
            search_seconds=search_result['summary']['seconds'], refit_seconds=refit['summary']['seconds'])
        result = dict(status='COMPLETE', run_id=job['job_id'], cap=cap, rows=[row],
            bundle_binding=contracts.binding(artifact, ctx), fit_receipt=receipt, source_files=source_files,
            source_sha=base['source_sha'], config_sha=base['config_sha'], data_sha=base['data_sha'], model_sha=base['model_sha'],
            selection_sealed_before_V=True, selection_decision_binding=decision, selection_receipt_binding=receipt_binding,
            recipe=dict(RECIPE, optimizer='AdamW', weight_decay=0., gradient_clip=1., dtype='float32',
                        sampler_initialization_rng_separate=True), search_summary=search_result['summary'], refit_summary=refit['summary'])
        contracts.sealed_json(output, result)
        if callable(getattr(model, 'close', None)):
            model.close()
        if callable(getattr(foundation, 'assert_frozen_unchanged', None)):
            foundation.assert_frozen_unchanged()
        return result
