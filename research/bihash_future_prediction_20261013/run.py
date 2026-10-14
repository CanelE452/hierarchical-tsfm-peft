"""Actual bounded BiHash experiment phases; preserves every terminal attempt."""
from pathlib import Path
import argparse
import copy
import json
import time
import traceback
import numpy as np
import torch
import model as bm
import train as tr
import probes as pr
import benchmark as bench
import runtime as rt
import data as dt

LEGACY = ('jena', 'robin')
FRESH = ('uci235_household_hourly', 'uci274_sml2010')


def pack_tuple(value):
    data, history, target, observed, history_observed = value
    bank = data['bank']
    return {**data, 'history': history, 'target': target, 'observed': observed, 'history_observed': history_observed,
        'bank': {'delta': bank['bank_delta'], 'observed': bank['bank_observed'], 'origins': bank['bank_origins']}}


def load_pack(dataset, method_lock=None):
    return pack_tuple(dt.load_legacy(dataset) if dataset in LEGACY else dt.load_training_new(dataset, method_lock))


def add_evaluation(pack, payload):
    for field in ('history', 'target', 'observed', 'origins'):
        pack[field]['eval'] = payload[field]
    pack['raw_history']['eval'] = payload['physical_history']
    pack['raw_history_observed']['eval'] = payload['history_observed']
    pack['history_observed']['eval'] = payload['history_observed']
    return pack


def source_check(sources):
    for path, expected in sources.items():
        if rt.sha(rt.ROOT / path) != expected: raise RuntimeError('ACTIVE_SOURCE_CHANGED: ' + path)


def check_method_sources(method_lock):
    lock = rt.read(method_lock)
    source_check(lock['source_hashes'])
    return lock


def assert_selected_coverage(cases, readouts, controls, lock):
    expected_cases = {(ds, seed, method, lock['weights'][method]) for ds in LEGACY + FRESH for seed in tr.SEEDS for method in bm.METHODS}
    actual = [(c['dataset'], c['seed'], c['method'], c['weight']) for c in cases]
    if len(actual) != len(expected_cases) or set(actual) != expected_cases:
        raise ValueError('INCOMPLETE_EXACT_SELECTED_CORE_COVERAGE')
    expected_probes = {(ds, seed, method) for ds in LEGACY + FRESH for seed in tr.SEEDS for method in tr.PROBE_METHODS}
    actual_probes = [(r['dataset'], r['seed'], r['source_method']) for r in readouts]
    if len(actual_probes) != len(expected_probes) or set(actual_probes) != expected_probes:
        raise ValueError('INCOMPLETE_EXACT_READOUT_COVERAGE')
    expected_controls = {(ds, name) for ds in LEGACY + FRESH for name in ('RAW_BOLT', 'RAW_JOINT_RIDGE', 'RAW_BOLT_RAW_R', 'RAW_SELF33')}
    actual_controls = [(r['dataset'], m['method']) for r in controls for m in r['methods']]
    if len(actual_controls) != len(expected_controls) or set(actual_controls) != expected_controls:
        raise ValueError('INCOMPLETE_EXACT_RAW_CONTROL_COVERAGE')


def assert_full_training(dev_result, fresh_result):
    expected = {(ds, seed, method, weight) for ds in LEGACY for seed in tr.SEEDS for method in bm.METHODS
        for weight in (tr.WEIGHTS if method.startswith('A_') else (0.,))}
    actual = [(c['dataset'], c['seed'], c['method'], c['weight']) for c in dev_result['cases']]
    if len(actual) != len(expected) or set(actual) != expected: raise ValueError('INCOMPLETE_EXACT_102_DEV_FITS')
    if len(fresh_result['cases']) != 72: raise ValueError('INCOMPLETE_72_FRESH_FITS')
    for case in dev_result['cases'] + fresh_result['cases']:
        training = case['training']
        if training['updates'] != 600 or len(training['curve']) != 600:
            raise ValueError('CORE_NOT_600_UPDATES')
        if [c['update'] for c in training['checkpoints']] != list(tr.CHECKPOINTS): raise ValueError('MISSING_CORE_CHECKPOINT')
    ae = dev_result['future_pretrains'] + fresh_result['future_pretrains']
    expected_ae = {(ds, seed) for ds in LEGACY + FRESH for seed in tr.SEEDS}
    if len(ae) != 12 or {(a['dataset'], a['seed']) for a in ae} != expected_ae:
        raise ValueError('INCOMPLETE_EXACT_12_FUTURE_AE_FITS')
    if any(a['updates'] != 400 or len(a['curve']) != 400 for a in ae): raise ValueError('FUTURE_AE_NOT_400_UPDATES')


def pack_record(pack):
    return {'dataset': pack['dataset'], 'C': pack['C'], 'K': pack['K'], 'columns': pack['columns'],
        'source_records': pack['source_records'], 'contract': pack['contract'], 'pca_metadata': pack['pca_metadata'],
        'mean_sha256': dt.array_hash(pack['mean']), 'std_sha256': dt.array_hash(pack['std']),
        'basis_sha256': dt.array_hash(pack['basis']), 'pca_mean_sha256': dt.array_hash(pack['pca_mean']),
        'origin_counts': {s: len(o) for s, o in pack['origins'].items()},
        'origin_sha256': {s: dt.array_hash(o) for s, o in pack['origins'].items()},
        'bank_origin_count': len(pack['bank']['origins']), 'bank_origin_sha256': dt.array_hash(pack['bank']['origins']),
        'bank_delta_sha256': dt.array_hash(pack['bank']['delta']), 'bank_observed_sha256': dt.array_hash(pack['bank']['observed'])}


def folder_name(method, weight, seed):
    return f'{method}_w{weight:g}_s{seed}'


def _path_failure(output, stage, dataset, seed, label, exc):
    failure = {'stage': stage, 'dataset': dataset, 'seed': seed, 'label': label,
        'error_type': type(exc).__name__, 'error': str(exc), 'traceback': traceback.format_exc(),
        'status': 'FAILED_PATH_PRESERVED', 'resume': 'Fix the evidenced cause and run a fresh tag; preserve this attempt'}
    output.setdefault('failed_paths', []).append(failure)
    print(rt.json_line({'path_failed': failure}), flush=True)
    return failure


def _fatal(exc):
    return isinstance(exc, rt.BudgetHeld) or str(exc).startswith(('ACTIVE_SOURCE_CHANGED', 'BACKBONE_CHANGED', 'PROTECTED_TEST'))


def _split_codec(model):
    return list(model.codec.parameters()), [p for n, p in model.named_parameters() if not n.startswith('codec.') and p.requires_grad]


def real_fixture(pack, forecaster, limit, folder):
    before_f = rt.state_hash(forecaster.model)
    indices = tr.batch_order(pack['origins']['fit'], tr.SEEDS[0], updates=1)[0]
    x = rt.tensor(pack['history']['fit'][indices]); origins = torch.as_tensor(pack['origins']['fit'][indices], device='cuda')
    target = rt.tensor(pack['target']['fit'][indices]); observed = torch.as_tensor(pack['observed']['fit'][indices], dtype=torch.bool, device='cuda')
    latent_input = x[:, :, :pack['K']].detach().clone().requires_grad_(True)
    latent_forecast = forecaster(latent_input)
    input_gradient = torch.autograd.grad(latent_forecast.square().mean(), latent_input)[0]
    if not torch.isfinite(input_gradient).all() or float(input_gradient.norm()) <= 0:
        raise ValueError('FROZEN_BOLT_INPUT_GRADIENT_NOT_FINITE_NONZERO')
    identity = bm.en.Codec(np.zeros(pack['C'], np.float32), np.eye(pack['C'], dtype=np.float32), tr.SEEDS[0], 'ANCHOR_PCA').to('cuda')
    with torch.no_grad(): identity_error = float((identity.forward_path(x, forecaster) - forecaster(x)).abs().max())
    if identity_error > 2e-6: raise ValueError('IDENTITY_ANCHOR_RAW_PARITY')
    groups = {'E': ('codec.codec.project.', 'codec.codec.input.', 'codec.codec.temporal.', 'codec.codec.output.'),
        'D': ('codec.codec.decode.', 'codec.residual.'), 'S_projection': ('student_projection.',),
        'Ws': ('student_hash.',), 'Wf': ('future_hash.',), 'phi': ('future_autoencoder.encoder.',)}
    records = []
    for method in ('B_GUMBEL', 'A_BIHASH_GUMBEL'):
        model = tr.make_model(pack, tr.SEEDS[0], method)
        initial = rt.clone_state(model); initial_hash = rt.state_hash(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0.)
        main_only = None
        if method == 'B_GUMBEL':
            model.train(); optimizer.zero_grad(set_to_none=True)
            losses = model.losses(x, origins, target, observed, forecaster)
            losses['main'].backward()
            norms = {name: tr._parameter_norm(model, prefixes) for name, prefixes in groups.items()}
            for name in ('E', 'Ws', 'Wf', 'phi'):
                if not np.isfinite(norms[name]) or norms[name] <= 0: raise ValueError('B_MAIN_ONLY_GRADIENT: ' + name)
            codec, side = _split_codec(model)
            torch.nn.utils.clip_grad_norm_(codec, 1.); torch.nn.utils.clip_grad_norm_(side, 1.)
            optimizer.step()
            changes = {name: max(float((p.detach().cpu() - initial[n]).abs().max()) for n, p in model.named_parameters() if n.startswith(prefixes)) for name, prefixes in groups.items()}
            main_only = {'actual_main_only_optimizer_step': True, 'gradient_norms': norms, 'parameter_max_abs_from_initial': changes}
            model.load_state_dict(initial)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0.)
        steps = []
        for step in range(3):
            model.train(); optimizer.zero_grad(set_to_none=True)
            losses = model.losses(x, origins, target, observed, forecaster, update=step)
            total = losses['main'] + .1 * losses['aux'] + .1 * losses['rec']
            total.backward()
            gradient_norms = {name: tr._parameter_norm(model, prefixes) for name, prefixes in groups.items()}
            if not all(np.isfinite(v) for v in gradient_norms.values()): raise ValueError('REAL_FIXTURE_NONFINITE_GROUP_GRADIENT')
            if any(p.grad is not None or p.requires_grad for p in forecaster.model.parameters()): raise ValueError('BACKBONE_GRADIENT_OR_TRAINABLE')
            codec, side = _split_codec(model)
            torch.nn.utils.clip_grad_norm_(codec, 1.); torch.nn.utils.clip_grad_norm_(side, 1.)
            optimizer.step()
            changes = {name: max(float((p.detach().cpu() - initial[n]).abs().max()) for n, p in model.named_parameters() if n.startswith(prefixes))
                for name, prefixes in groups.items()}
            steps.append({'step': step + 1, 'main': float(losses['main'].detach()), 'aux': float(losses['aux'].detach()),
                'rec': float(losses['rec'].detach()), 'gradient_norms': gradient_norms, 'parameter_max_abs_from_initial': changes})
            limit.check()
        for name in ('E', 'Ws', 'Wf', 'phi'):
            if steps[-1]['parameter_max_abs_from_initial'][name] <= 0: raise ValueError('REAL_FIXTURE_NO_STEP_CHANGE: ' + name)
        model.load_state_dict(initial)
        if rt.state_hash(model) != initial_hash: raise ValueError('REAL_FIXTURE_RESTORE_HASH')
        model.eval()
        with torch.no_grad():
            baseline = model(x, origins, forecaster)
            changed_current_future = model.losses(x, origins, target + 11, observed, forecaster)['prediction']
            if not torch.equal(changed_current_future, baseline): raise ValueError('CURRENT_FUTURE_ENTERS_PREDICTION')
            single = model(x[:1], origins[:1], forecaster)
            batch_max = float((baseline[:1] - single).abs().max())
            if batch_max > 2e-5: raise ValueError('REAL_QUERY_BATCH_DEPENDENCY')
            if method.startswith('B_'):
                model.codec.codec.output.weight.fill_(.03125)
                original, details = model.encode_with_memory(x, origins)
                changed = x.clone(); changed[:, 256:] += 33
                raw_changed, _ = model.encode_with_memory(changed, origins)
                raw_prefix_delta = float((raw_changed[:, :256] - original[:, :256]).abs().max())
                # For one query prefix, change only future records unavailable by that prefix's final row.
                cutoff = int(origins[0]) - 512 + 255 + 1
                unavailable = model.bank_end_exclusive > cutoff
                old_memory = model.bank_delta.clone(); model.bank_delta[unavailable] += 77
                memory_changed, _ = model.encode_with_memory(x[:1], origins[:1])
                memory_prefix_delta = float((memory_changed[:, :256] - original[:1, :256]).abs().max())
                model.bank_delta.copy_(old_memory)
                if raw_prefix_delta != 0 or memory_prefix_delta > 2e-6: raise ValueError('REAL_CAUSAL_MEMORY_PREFIX')
                fallback = model(x, torch.full_like(origins, -100000), forecaster)
                ordinary = model.codec.forward_path(x, forecaster)
                fallback_max = float((fallback - ordinary).abs().max())
                if fallback_max != 0: raise ValueError('REAL_MEMORY_FALLBACK')
            else:
                raw_prefix_delta = memory_prefix_delta = fallback_max = None
        model.load_state_dict(initial)
        model.train()
        _ = model(x, origins, forecaster)
        continuation_state = rt.clone_state(model)
        next_prediction = model(x, origins, forecaster).detach()
        model.load_state_dict(continuation_state)
        repeat_prediction = model(x, origins, forecaster).detach()
        noise_continuation_error = float((next_prediction - repeat_prediction).abs().max())
        if noise_continuation_error != 0: raise ValueError('GPU_NOISE_CHECKPOINT_CONTINUATION')
        model.load_state_dict(initial)
        model.eval()
        with torch.no_grad(): saved_prediction = model(x, origins, forecaster).cpu().numpy()
        forecast_artifact = rt.save_npz(folder / (method + '_FIT_FORECAST.npz'), fit_prediction=saved_prediction,
            fit_target=pack['target']['fit'][indices], fit_observed=pack['observed']['fit'][indices],
            fit_origins=pack['origins']['fit'][indices], mean=pack['mean'], std=pack['std'])
        cloned = tr.make_model(pack, tr.SEEDS[0], method)
        cloned.load_future_pretrain(rt.clone_state(model.future_autoencoder))
        ae_clone_pass = rt.state_hash(cloned.future_autoencoder) == rt.state_hash(model.future_autoencoder)
        if not ae_clone_pass: raise ValueError('FUTURE_AE_COMMON_STATE_CLONE')
        records.append({'method': method, 'initial_hash': initial_hash, 'restored_hash': rt.state_hash(model),
            'actual_three_optimizer_steps': steps, 'main_only': main_only, 'batch_vs_single_max_abs': batch_max,
            'active_temporal_raw_prefix_max_abs': raw_prefix_delta, 'unavailable_future_bank_prefix_max_abs': memory_prefix_delta,
            'empty_eligible_memory_fallback_max_abs': fallback_max, 'current_future_prediction_invariance': True,
            'noise_calls_checkpoint_continuation_max_abs': noise_continuation_error, 'future_AE_state_clone_pass': ae_clone_pass,
            'forecast_artifact': forecast_artifact, 'forecast_score': bm.en.metrics(pack, saved_prediction,
                pack['target']['fit'][indices], pack['observed']['fit'][indices])})
    if rt.state_hash(forecaster.model) != before_f: raise ValueError('BACKBONE_CHANGED')
    return {'status': 'PASS', 'dataset': pack['dataset'], 'fit_batch_indices': indices.tolist(),
        'fit_batch_origins': pack['origins']['fit'][indices].tolist(), 'backbone_before_sha256': before_f,
        'backbone_after_sha256': rt.state_hash(forecaster.model), 'fixtures': records,
        'normalizer_fixture': bm.en.normalizer_fixture(), 'frozen_F_input_gradient_norm': float(input_gradient.norm()),
        'identity_anchor_vs_raw_max_abs': identity_error}


def profile(pack, forecaster, folder, limit, updates):
    model = tr.make_model(pack, tr.SEEDS[0], 'NO_ALIGN')
    pretrain_state = rt.clone_state(model.future_autoencoder)
    pretrain_record = {'checkpoint': rt.save(folder / 'PROFILE_ONLY_FUTURE_AE_U0.pt', {'future_autoencoder': pretrain_state}),
        'profile_only_untrained_future_AE': True}
    cases = []
    for method, weight in (('NO_ALIGN', 0.), ('B_GUMBEL', 0.), ('A_BIHASH_GUMBEL', .1)):
        cache = rt.CACHE / folder.name / folder_name(method, weight, tr.SEEDS[0]); cache.mkdir(parents=True, exist_ok=False)
        _, case = tr.train_case(pack, tr.SEEDS[0], method, weight, pretrain_state, pretrain_record, forecaster, cache, rt, limit, updates=updates)
        cases.append(case)
    seconds_per_step = max(float(np.median(c['training']['training_step_seconds'][min(3, updates - 1):])) for c in cases)
    checkpoint_seconds = max(float(np.median(c['training']['checkpoint_seconds'])) for c in cases)
    return {'cases': cases, 'profile_updates': updates, 'profile_only': True,
        'measured_median_optimization_step_seconds_max_method': seconds_per_step,
        'measured_median_checkpoint_seconds_max_method': checkpoint_seconds,
        '174_core_optimization_extrapolation_seconds': seconds_per_step * 600 * 174,
        '870_checkpoint_extrapolation_seconds': checkpoint_seconds * 870,
        'conservative_174_core_fit_wall_extrapolation_seconds': seconds_per_step * 600 * 174 + checkpoint_seconds * 870,
        'extrapolation_limit': 'Jena only, maximum of measured methods; excludes AE/probe/cost and startup. Short-run CP overhead separated from training steps.'}


def dev(output, forecaster, limit, sources, run_folder):
    packs = {ds: load_pack(ds) for ds in LEGACY}
    output['data'] = {ds: pack_record(p) for ds, p in packs.items()}
    output['future_pretrains'], output['cases'] = [], []
    for ds, pack in packs.items():
        for seed in tr.SEEDS:
            source_check(sources)
            pretrain_folder = rt.CACHE / run_folder.name / ds / f'future_AE_s{seed}'; pretrain_folder.mkdir(parents=True, exist_ok=False)
            try:
                state, pretrain_record = tr.pretrain_future(pack, seed, pretrain_folder, rt, limit)
                output['future_pretrains'].append(pretrain_record)
            except Exception as exc:
                if _fatal(exc): raise
                _path_failure(output, 'future_AE', ds, seed, pretrain_folder.name, exc); continue
            for method in bm.METHODS:
                for weight in tr.WEIGHTS if method.startswith('A_') else (0.,):
                    source_check(sources)
                    folder = rt.CACHE / run_folder.name / ds / folder_name(method, weight, seed); folder.mkdir(parents=True, exist_ok=False)
                    try:
                        _, case = tr.train_case(pack, seed, method, weight, state, pretrain_record, forecaster, folder, rt, limit)
                        output['cases'].append(case); rt.write(run_folder / ds / (folder.name + '.json'), case)
                    except Exception as exc:
                        if _fatal(exc): raise
                        _path_failure(output, 'core', ds, seed, folder.name, exc)
    if len(output['cases']) == 102:
        selection = tr.select_methods(output['cases'])
        fresh_source_plan = rt.read(rt.HERE / 'FRESH_SOURCE_PLAN.json')
        lock = {**selection, 'status': 'LOCKED', 'locked': True, 'created_unix': time.time(), 'source_hashes': sources,
            'newdata_sources': fresh_source_plan['newdata_sources'], 'seeds': list(tr.SEEDS), 'primary_metric': 'normalized channel macro MSE',
            'future_AE': 'FIT only, fixed 400 updates; all methods same pretrained state per dataset/seed'}
        rt.write(rt.HERE / 'METHOD_LOCK.json', lock); output['method_lock'] = {'path': str((rt.HERE / 'METHOD_LOCK.json').relative_to(rt.ROOT)), 'sha256': rt.sha(rt.HERE / 'METHOD_LOCK.json')}
        # Existing DEVVAL is evaluated only after the method/weight decision is locked.
        for case in output['cases']:
            if case['weight'] != selection['weights'][case['method']]: continue
            pack = packs[case['dataset']]; model = tr.load_model(pack, case, rt)
            pred = tr.predict(model, pack['history']['val'], pack['origins']['val'], forecaster, limit)
            artifact = tr.prediction_artifact(rt, rt.CACHE / run_folder.name / case['dataset'] / case['label'] / 'SELECTED_DEVVAL.npz', pack, {'val': pred})
            case['evaluation'] = {'split': 'val', 'repeated_development': True, 'prediction': artifact, 'scores': tr.scores(pack, pred, 'val')}
            rt.write(run_folder / case['dataset'] / (case['label'] + '_DEVVAL.json'), case['evaluation'])
    return output


def fresh(output, forecaster, limit, sources, run_folder, method_lock):
    lock = check_method_sources(method_lock)
    packs = {}
    output['future_pretrains'], output['cases'], output['contracts'] = [], [], {}
    for ds in FRESH:
        try:
            output['contracts'][ds] = dt.prepare_new(ds, method_lock)
            packs[ds] = load_pack(ds, method_lock)
        except Exception as exc:
            if _fatal(exc): raise
            _path_failure(output, 'data_prepare', ds, None, ds, exc)
    output['data'] = {ds: pack_record(p) for ds, p in packs.items()}
    for ds, pack in packs.items():
        for seed in tr.SEEDS:
            source_check(sources)
            pretrain_folder = rt.CACHE / run_folder.name / ds / f'future_AE_s{seed}'; pretrain_folder.mkdir(parents=True, exist_ok=False)
            try:
                state, pretrain_record = tr.pretrain_future(pack, seed, pretrain_folder, rt, limit)
                output['future_pretrains'].append(pretrain_record)
            except Exception as exc:
                if _fatal(exc): raise
                _path_failure(output, 'future_AE', ds, seed, pretrain_folder.name, exc); continue
            for method in bm.METHODS:
                weight = lock['weights'][method]
                folder = rt.CACHE / run_folder.name / ds / folder_name(method, weight, seed); folder.mkdir(parents=True, exist_ok=False)
                try:
                    _, case = tr.train_case(pack, seed, method, weight, state, pretrain_record, forecaster, folder, rt, limit)
                    output['cases'].append(case); rt.write(run_folder / ds / (folder.name + '.json'), case)
                except Exception as exc:
                    if _fatal(exc): raise
                    _path_failure(output, 'core', ds, seed, folder.name, exc)
    output['method_lock'] = {'path': str(Path(method_lock).relative_to(rt.ROOT)), 'sha256': rt.sha(method_lock)}
    return output


def selected_cases(dev_result, fresh_result, lock):
    return [c for c in dev_result['cases'] if c['weight'] == lock['weights'][c['method']]] + fresh_result['cases']


def probe_phase(output, forecaster, limit, run_folder, dev_result, fresh_result, method_lock):
    lock = check_method_sources(method_lock)
    cases = selected_cases(dev_result, fresh_result, lock)
    packs = {ds: load_pack(ds, method_lock) for ds in LEGACY + FRESH}
    output.update(readouts=[], controls=[], data={ds: pack_record(p) for ds, p in packs.items()})
    for ds, pack in packs.items():
        folder = rt.CACHE / run_folder.name / ds / 'RAW_CONTROLS'; folder.mkdir(parents=True, exist_ok=False)
        try:
            control = pr.fit_raw_controls(pack, forecaster, folder, rt, limit)
            output['controls'].append(control); rt.write(run_folder / ds / 'RAW_CONTROLS.json', control)
        except Exception as exc:
            if _fatal(exc): raise
            _path_failure(output, 'raw_controls', ds, None, 'RAW_CONTROLS', exc)
        for case in (c for c in cases if c['dataset'] == ds and c['method'] in tr.PROBE_METHODS):
            folder = rt.CACHE / run_folder.name / ds / case['label']; folder.mkdir(parents=True, exist_ok=False)
            try:
                model = tr.load_model(pack, case, rt)
                readout = pr.fit_representation_readout(pack, case, model, folder, rt, limit)
                readout['source_label'] = case['label']; output['readouts'].append(readout)
                rt.write(run_folder / ds / (case['label'] + '_READOUT.json'), readout)
            except Exception as exc:
                if _fatal(exc): raise
                _path_failure(output, 'representation_probe', ds, case['seed'], case['label'], exc)
    return output


def evaluation_phase(output, forecaster, limit, sources, run_folder, dev_result, fresh_result, probe_result, method_lock):
    lock = check_method_sources(method_lock)
    cases = selected_cases(dev_result, fresh_result, lock)
    assert_full_training(dev_result, fresh_result)
    assert_selected_coverage(cases, probe_result['readouts'], probe_result['controls'], lock)
    selected_models = [{**c['training']['selected_checkpoint'], 'dataset': c['dataset'], 'seed': c['seed'], 'method': c['method'],
        'weight': c['weight'], 'selected_update': c['training']['selected_update'], 'artifact_role': 'core'} for c in cases]
    control_models = [{**r['model'], 'dataset': c['dataset'], 'seed': None, 'method': r['method'], 'artifact_role': 'raw_control'}
        for c in probe_result['controls'] for r in c['methods'] if 'model' in r]
    readout_models = [{**r['model'], 'dataset': c['dataset'], 'seed': c['seed'], 'source_method': c['source_method'],
        'method': r['method'], 'source_label': c['source_label'], 'artifact_role': 'representation_readout'}
        for c in probe_result['readouts'] for r in c['readouts'] if r.get('additional_fit', False)]
    selected_models += control_models + readout_models
    global_lock = {'status': 'ALL_SELECTED', 'ALL_SELECTED': True, 'created_unix': time.time(),
        'method_lock_sha256': rt.sha(method_lock), 'source_hashes': sources,
        'newdata_contracts': {ds: dt.required_model_selection_fields(ds, method_lock) for ds in FRESH},
        'selected_models': selected_models, 'all_core_training_completed': True, 'all_readouts_and_controls_selected': True}
    global_lock.update(core_model_identities=selected_models[:144], control_model_identities=control_models,
        readout_model_identities=readout_models, future_AE_checkpoints=[p['checkpoint'] for p in dev_result['future_pretrains'] + fresh_result['future_pretrains']])
    global_path = run_folder / 'GLOBAL_MODEL_LOCK.json'; rt.write(global_path, global_lock)
    output.update(core_evaluation=[], readout_evaluation=[], control_evaluation=[], evaluation_open_receipts=[])
    packs = {ds: load_pack(ds, method_lock) for ds in LEGACY + FRESH}
    for ds in FRESH:
        source_check(sources)
        before = time.time(); payload = dt.open_new_evaluation(ds, global_path, method_lock)
        packs[ds] = add_evaluation(packs[ds], payload)
        receipt = {'dataset': ds, 'opened_unix': before, 'completed_open_unix': time.time(),
            'global_model_lock': {'path': str(global_path.relative_to(rt.ROOT)), 'sha256': rt.sha(global_path)},
            'method_lock_sha256': rt.sha(method_lock), 'payload_hashes': payload['payload_hashes'],
            'opened_only_after_all_selected': True}
        rt.write(run_folder / (ds + '_EVAL_FIRST_OPEN.json'), receipt); output['evaluation_open_receipts'].append(receipt)
    for ds, pack in packs.items():
        split = 'val' if ds in LEGACY else 'eval'
        for case in (c for c in cases if c['dataset'] == ds):
            folder = rt.CACHE / run_folder.name / ds / case['label']; folder.mkdir(parents=True, exist_ok=False)
            try:
                model = tr.load_model(pack, case, rt)
                predictions = {'regular': tr.predict(model, pack['history'][split], pack['origins'][split], forecaster, limit)}
                if case['method'] in {'B_GUMBEL', lock['selected_B_method']}:
                    for variant in ('student', 'memory', 'uniform', 'tanh'):
                        predictions[variant] = tr.predict(model, pack['history'][split], pack['origins'][split], forecaster, limit, variant=variant)
                    initial_cp = case['training']['checkpoints'][0]
                    initial = torch.load(rt.ROOT / initial_cp['path'], map_location='cpu', weights_only=True)['model']
                    reset = copy.deepcopy(model)
                    reset.student_hash.load_state_dict({key.removeprefix('student_hash.'): value for key, value in initial.items() if key.startswith('student_hash.')})
                    reset.future_hash.load_state_dict({key.removeprefix('future_hash.'): value for key, value in initial.items() if key.startswith('future_hash.')})
                    reset.freeze_memory()
                    predictions['reset_maps'] = tr.predict(reset, pack['history'][split], pack['origins'][split], forecaster, limit)
                entries = []
                for variant, prediction in predictions.items():
                    artifact = tr.prediction_artifact(rt, folder / (variant + '_' + split.upper() + '.npz'), pack, {split: prediction})
                    entries.append({'variant': variant, 'prediction': artifact, 'scores': tr.scores(pack, prediction, split)})
                evaluation = {'dataset': ds, 'seed': case['seed'], 'source_label': case['label'], 'method': case['method'],
                    'split': split, 'entries': entries, 'no_retraining_or_new_selection': True}
                output['core_evaluation'].append(evaluation); rt.write(run_folder / ds / (case['label'] + '_EVALUATION.json'), evaluation)
            except Exception as exc:
                if _fatal(exc): raise
                _path_failure(output, 'core_inference', ds, case['seed'], case['label'], exc)
        for control in (c for c in probe_result['controls'] if c['dataset'] == ds):
            for method in control['methods']:
                prediction = pr.evaluate_raw_control(pack, split, method, forecaster, rt, limit)
                artifact = tr.prediction_artifact(rt, rt.CACHE / run_folder.name / ds / (method['method'] + '_' + split.upper() + '.npz'), pack, {split: prediction})
                output['control_evaluation'].append({'dataset': ds, 'method': method['method'], 'split': split,
                    'prediction': artifact, 'scores': tr.scores(pack, prediction, split)})
        for readout in (r for r in probe_result['readouts'] if r['dataset'] == ds):
            case = next(c for c in cases if c['dataset'] == ds and c['label'] == readout['source_label'])
            model = tr.load_model(pack, case, rt)
            for method in readout['readouts']:
                prediction = pr.evaluate_readout(pack, split, case, model, method, rt, limit)
                artifact = tr.prediction_artifact(rt, rt.CACHE / run_folder.name / ds / (case['label'] + '_' + method['method'] + '_' + split.upper() + '.npz'), pack, {split: prediction})
                output['readout_evaluation'].append({'dataset': ds, 'seed': case['seed'], 'source_label': case['label'],
                    'method': method['method'], 'split': split, 'duplicate_of': method.get('duplicate_of'),
                    'prediction': artifact, 'scores': tr.scores(pack, prediction, split)})
    output['global_model_lock'] = {'path': str(global_path.relative_to(rt.ROOT)), 'sha256': rt.sha(global_path)}
    return output


def _artifact_array(artifact, split):
    path = rt.ROOT / artifact['path']
    if rt.sha(path) != artifact['sha256']: raise ValueError('SCIENCE_ARTIFACT_SHA')
    with np.load(path, allow_pickle=False) as z: return z[split + '_prediction'].copy()


def cost_phase(output, forecaster, limit, dev_result, fresh_result, probe_result, evaluation_result, method_lock):
    lock = check_method_sources(method_lock); cases = selected_cases(dev_result, fresh_result, lock)
    assert_full_training(dev_result, fresh_result)
    assert_selected_coverage(cases, probe_result['readouts'], probe_result['controls'], lock)
    packs = {ds: load_pack(ds, method_lock) for ds in LEGACY + FRESH}
    global_path = rt.ROOT / evaluation_result['global_model_lock']['path']
    for ds in FRESH: packs[ds] = add_evaluation(packs[ds], dt.open_new_evaluation(ds, global_path, method_lock))
    output.update(records=[], aliases=[], canonical_paths=268, expected_rows=536)
    for ds, pack in packs.items():
        split = 'val' if ds in LEGACY else 'eval'
        for case in (c for c in cases if c['dataset'] == ds):
            item = next(r for r in evaluation_result['core_evaluation'] if r['dataset'] == ds and r['source_label'] == case['label'])
            expected = _artifact_array(next(r['prediction'] for r in item['entries'] if r['variant'] == 'regular'), split)
            model = tr.load_model(pack, case, rt)
            output['records'] += bench.cost(pack, 'CORE', expected, split, forecaster, limit, model=model, identity=case['label'])
        for control in (c for c in probe_result['controls'] if c['dataset'] == ds):
            for method in control['methods']:
                item = next(r for r in evaluation_result['control_evaluation'] if r['dataset'] == ds and r['method'] == method['method'])
                expected = _artifact_array(item['prediction'], split)
                ridge = None if method['method'] == 'RAW_BOLT' else bench.load_ridge(method, rt)
                output['records'] += bench.cost(pack, method['method'], expected, split, forecaster, limit, ridge=ridge, identity=method['method'])
        for readout in (r for r in probe_result['readouts'] if r['dataset'] == ds):
            case = next(c for c in cases if c['dataset'] == ds and c['label'] == readout['source_label'])
            model = tr.load_model(pack, case, rt)
            for method in readout['readouts']:
                if method.get('duplicate_of'):
                    output['aliases'].append({'dataset': ds, 'source_label': case['label'], 'route': method['method'], 'canonical_route': method['duplicate_of']}); continue
                item = next(r for r in evaluation_result['readout_evaluation'] if r['dataset'] == ds and r['source_label'] == case['label'] and r['method'] == method['method'])
                expected = _artifact_array(item['prediction'], split); ridge = bench.load_ridge(method, rt)
                output['records'] += bench.cost(pack, method['method'], expected, split, forecaster, limit, model=model, ridge=ridge,
                    identity=case['label'] + '_' + method['method'])
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', required=True, choices=('fixture', 'profile', 'dev', 'fresh', 'probes', 'evaluate', 'cost'))
    parser.add_argument('--tag', required=True)
    parser.add_argument('--dev-result', type=Path)
    parser.add_argument('--fresh-result', type=Path)
    parser.add_argument('--probe-result', type=Path)
    parser.add_argument('--evaluation-result', type=Path)
    parser.add_argument('--method-lock', type=Path, default=rt.HERE / 'METHOD_LOCK.json')
    parser.add_argument('--profile-updates', type=int, default=30)
    args = parser.parse_args()
    if not (rt.HERE / 'PREFLIGHT.json').exists(): raise RuntimeError('PREFLIGHT_NOT_SEALED_NO_GPU_START')
    run_folder = rt.HERE / ('run_' + args.tag); run_folder.mkdir(parents=True, exist_ok=False)
    sources = rt.source_snapshot(args.tag)
    output = {'tag': args.tag, 'stage': args.stage, 'started_unix': time.time(), 'source_hashes': sources, 'failed_paths': []}
    limit = rt.Limit(args.tag, gpu=True)
    status, error = 'COMPLETE', None
    try:
        bm.en.configure()
        forecaster = bm.en.ex.previous.FrozenBolt(); bm.en.replace_normalizer(forecaster)
        backbone_before = rt.state_hash(forecaster.model)
        output['backbone_before_sha256'] = backbone_before
        if args.stage == 'fixture': output['fixture'] = real_fixture(load_pack('jena'), forecaster, limit, run_folder)
        elif args.stage == 'profile': output.update(profile(load_pack('jena'), forecaster, run_folder, limit, args.profile_updates))
        elif args.stage == 'dev': output = dev(output, forecaster, limit, sources, run_folder)
        elif args.stage == 'fresh': output = fresh(output, forecaster, limit, sources, run_folder, args.method_lock)
        elif args.stage == 'probes': output = probe_phase(output, forecaster, limit, run_folder, rt.read(args.dev_result), rt.read(args.fresh_result), args.method_lock)
        elif args.stage == 'evaluate': output = evaluation_phase(output, forecaster, limit, sources, run_folder, rt.read(args.dev_result), rt.read(args.fresh_result), rt.read(args.probe_result), args.method_lock)
        else: output = cost_phase(output, forecaster, limit, rt.read(args.dev_result), rt.read(args.fresh_result), rt.read(args.probe_result), rt.read(args.evaluation_result), args.method_lock)
        source_check(sources)
        output['backbone_after_sha256'] = rt.state_hash(forecaster.model)
        if output['backbone_after_sha256'] != backbone_before: raise ValueError('BACKBONE_CHANGED')
        if output['failed_paths']: status = 'FAILED_PRESERVED'
    except Exception as exc:
        status = 'BUDGET_HELD_PRESERVED' if isinstance(exc, rt.BudgetHeld) else 'FAILED_PRESERVED'
        error = str(exc); output.update(error=error, error_type=type(exc).__name__, traceback=traceback.format_exc())
        print(rt.json_line({'stage_terminal_error': args.stage, 'error': error}), flush=True)
    output.update(status=status, finished_unix=time.time(), resource_receipt=limit.close(status, error))
    rt.write(run_folder / 'RESULT.json', output)
    print(rt.json_line({'terminal_result': str(run_folder / 'RESULT.json'), 'status': status,
        'case_count': len(output.get('cases', [])), 'failed_paths': len(output.get('failed_paths', []))}), flush=True)
    if status != 'COMPLETE': raise SystemExit(2)


if __name__ == '__main__': main()
