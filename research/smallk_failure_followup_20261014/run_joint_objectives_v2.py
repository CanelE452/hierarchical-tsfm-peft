"""New joint E/D objective comparison from identical original selected MAIN CPs.

No old asset/model/GPU read on import. One explicitly executed fit per fresh
process; original campaign code and results remain read-only.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / '.cache' / HERE.name / 'joint_objectives'
OBJECTIVES = ('Y', 'RAW', 'MIXED')
UPDATES = 600
CHECKPOINTS = (0, 100, 200, 400, 600)
BATCH_SIZE = 4
LEARNING_RATE = .001
WEIGHT_DECAY = 0.
CLIP_NORM = 1.
LOCAL_SECONDS = 150
DIAGNOSTIC_UPDATES = (0, 99, 199, 399, 599)


def training_order(seed, count):
    return np.random.default_rng(seed).integers(0, count, size=(UPDATES, BATCH_SIZE))


def observed_macro_loss(prediction, target, observed):
    import torch
    counts = observed.sum((0, 1))
    active = counts > 0
    if not bool(active.any()):
        raise ValueError('TRAIN_BATCH_HAS_NO_OBSERVED_TARGET')
    error = torch.where(observed, prediction - target, 0.)
    squared = error.square().sum((0, 1))
    return (squared / counts.clamp_min(1))[active].mean()


def objective_losses(prediction, actual_y, teacher, observed, objective):
    if objective not in OBJECTIVES:
        raise ValueError('UNKNOWN_OBJECTIVE')
    y = observed_macro_loss(prediction, actual_y, observed)
    raw = observed_macro_loss(prediction, teacher, observed)
    training = y if objective == 'Y' else raw if objective == 'RAW' else .5 * y + .5 * raw
    return training, y, raw


def choose_checkpoint(rows):
    if not rows or any(not np.isfinite(row['tune_actual_Y_mse']) for row in rows):
        raise ValueError('FINITE_ACTUAL_Y_TUNE_CHECKPOINTS_REQUIRED')
    return min(range(len(rows)), key=lambda index: rows[index]['tune_actual_Y_mse'])


def forward_trace(codec, bolt, x):
    z = codec.encode(x)
    future = bolt(z)
    return x[:, -1:] + codec.decode(future) - codec.decode(z[:, -1:]), z, future


def parameter_group(name):
    if name.startswith('codec.decode.') or name.startswith('residual.'):
        return 'D'
    return 'E'


def parameter_gradient_proof(codec, loss, z, future):
    import torch
    upstream = torch.autograd.grad(loss, future, retain_graph=True)[0]
    input_gradient = torch.autograd.grad(future, z, grad_outputs=upstream, retain_graph=True)[0]
    parameters = list(codec.named_parameters())
    gradients = torch.autograd.grad(loss, [value for _, value in parameters], retain_graph=True, allow_unused=True)
    groups = {}
    for group in ('E', 'D'):
        selected = [(name, gradient) for (name, _), gradient in zip(parameters, gradients) if parameter_group(name) == group]
        groups[group] = {'norm': float(torch.sqrt(sum(gradient.detach().square().sum() for _, gradient in selected if gradient is not None))),
                         'finite': all(gradient is None or bool(torch.isfinite(gradient).all()) for _, gradient in selected),
                         'active_names': [name for name, gradient in selected if gradient is not None and bool((gradient != 0).any())]}
    return {'frozen_F_input_gradient_norm': float(input_gradient.detach().norm()),
            'frozen_F_input_gradient_finite': bool(torch.isfinite(input_gradient).all()),
            'future_upstream_gradient_norm': float(upstream.detach().norm()), 'E_D_gradients': groups}


def parameter_change_proof(initial, current):
    groups = {}
    for group in ('E', 'D'):
        rows = []
        for name, value in current.items():
            if name == 'fixed_basis' or parameter_group(name) != group:
                continue
            left = initial[name].detach().cpu().numpy()
            right = value.detach().cpu().numpy()
            delta = right.astype(np.float64) - left.astype(np.float64)
            rows.append({'name': name, 'changed': bool(np.any(delta != 0)),
                         'max_abs': float(np.max(np.abs(delta))), 'l2': float(np.linalg.norm(delta))})
        groups[group] = {'changed_parameter_tensors': sum(row['changed'] for row in rows),
                         'registered_parameter_tensors': len(rows), 'parameters': rows}
    return groups


def array_score(prediction, target, observed, std=None):
    error = np.asarray(prediction, np.float64) - np.asarray(target, np.float64)
    if std is not None:
        error *= std
    counts = observed.sum((0, 1))
    mse, mae = [], []
    for channel in range(error.shape[-1]):
        values = error[:, :, channel][observed[:, :, channel]]
        mse.append(float(np.mean(values ** 2)) if len(values) else None)
        mae.append(float(np.mean(np.abs(values))) if len(values) else None)
    return {'mse': float(np.mean([value for value in mse if value is not None])),
            'mae': float(np.mean([value for value in mae if value is not None])),
            'per_channel_mse': mse, 'per_channel_mae': mae,
            'observed_counts': counts.tolist(), 'active_channels': int((counts > 0).sum())}


def prediction_scores(prediction, values, std):
    return {'normalized_actual_Y': array_score(prediction, values['target'], values['observed']),
            'physical_actual_Y': array_score(prediction, values['target'], values['observed'], std),
            'normalized_RAW_teacher': array_score(prediction, values['raw_forecast'], values['observed'])}


def snapshot(output, probes, original_sources):
    folder = output / 'sources'
    folder.mkdir()
    paths = [Path(__file__), HERE / 'METHOD_LOCK_JOINT_v2.md', HERE / 'run_condition_probes.py',
             HERE / 'prepare_inputs.py', HERE / 'followup_runtime.py']
    paths += [Path(record['path']) for record in original_sources]
    records = []
    for path in {str(path.resolve()): path.resolve() for path in paths}.values():
        relative = path.relative_to(ROOT)
        destination = folder / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        before = probes.sha(path)
        with path.open('rb') as source, destination.open('xb') as copied:
            shutil.copyfileobj(source, copied)
        if probes.sha(destination) != before or probes.sha(path) != before:
            raise ValueError('SOURCE_CHANGED_DURING_SNAPSHOT')
        records.append({'path': str(path), 'sha256': before,
                        'snapshot_path': str(destination), 'snapshot_sha256': before})
    probes.write(folder / 'MANIFEST.json', {'sources': records})
    return records


def save_checkpoint(path, torch, codec, update, seed, objective, probes):
    state = {name: value.detach().cpu().clone() for name, value in codec.state_dict().items()}
    with path.open('xb') as handle:
        torch.save({'codec': state, 'new_update': update, 'seed': seed, 'objective': objective,
                    'architecture': 'ANCHOR_NONLINEAR_D', 'new_joint_comparison': True}, handle)
    return {'path': str(path), 'sha256': probes.sha(path), 'bytes': path.stat().st_size,
            'state_sha_old_algorithm': probes.old_state_hash(state),
            'state_sha_Bi_algorithm': probes.bi_state_hash(state)}


def predict(engine, codec, bolt, history, check):
    output = []
    with engine.torch.no_grad():
        for begin in range(0, len(history), 8):
            output.append(codec.forward_path(engine.tensor(history[begin:begin + 8]), bolt).cpu().numpy())
            check()
    return np.concatenate(output)


class GuardCadence:
    """Full shared CPU/storage checks at boundaries; base caps every batch."""
    def __init__(self, limit, local_check):
        self.limit, self.local_check = limit, local_check
        self.calls = {'full': 0, 'fast_base': 0}
        self.seconds = {'full': 0., 'fast_base': 0.}

    def _call(self, mode, callback):
        self.local_check()
        started = time.perf_counter()
        try:
            return callback()
        finally:
            self.calls[mode] += 1
            self.seconds[mode] += time.perf_counter() - started

    def full(self):
        return self._call('full', self.limit.check)

    def fast(self):
        return self._call('fast_base', self.limit.shared.check)

    def summary(self):
        return {'explicit_calls': dict(self.calls), 'explicit_wall_seconds': dict(self.seconds),
                'excluded': 'FollowupLimit constructor/terminal-close guards; included in whole process receipt',
                'fast_scope': 'unchanged base GPU/CPU process-wall,campaign,RSS/VRAM,Bi storage guard',
                'full_scope': 'all above plus supplementary CPU and combined followup storage/disk guard',
                'full_boundaries': '10 optimizer updates; checkpoints; before/after every CP or NPZ write; final source verification',
                'local_process_cap_seconds': LOCAL_SECONDS}


def execute(args):
    import psutil
    import prepare_inputs as inputs
    import run_condition_probes as probes
    import followup_runtime as runtime
    output = CACHE / args.tag
    output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'RUNNING', 'new_joint_comparison': True, 'objective': args.objective,
              'dataset': args.dataset, 'seed': args.seed, 'tag': args.tag, 'checkpoints': [],
              'curve': [], 'gradient_diagnostics': [], 'selected_inference': {},
              'independent_holdout': False, 'protected_TEST_accessed': False,
              'old_campaign_resumed': False, 'local_process_wall_limit_seconds': LOCAL_SECONDS,
              'recipe': {'new_updates': UPDATES, 'batch_size': BATCH_SIZE, 'optimizer': 'AdamW',
                         'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY, 'clip_norm': CLIP_NORM,
                         'checkpoints': list(CHECKPOINTS), 'loss_scale': 'normalized observed channel-macro MSE',
                         'checkpoint_selection': 'TUNE actual-Y normalized MSE; earliest exact tie',
                         'teacher_scope': 'cached original RAW F(past) FIT forecasts; same actual-Y observed mask',
                         'MIXED_definition': '.5*actual_Y_MSE+.5*cached_RAW_teacher_MSE',
                         'initialization': 'original already-Y-TUNE-selected MAIN CP; new optimizer state',
                         'strict_K_only_compression': False, 'current_raw_state_anchor_C': True}}
    started = time.perf_counter()
    process_birth = psutil.Process().create_time()
    limit, codec, bolt, guards = None, None, None, None
    def local_check():
        if time.time() - process_birth >= LOCAL_SECONDS:
            raise limit.rt.BudgetHeld('JOINT_FIT_PROCESS_WALL_150S')
        if psutil.Process().memory_info().rss > 8 * 1024 ** 3:
            raise limit.rt.BudgetHeld('JOINT_FIT_PROCESS_RSS_8GIB')
    def check():
        return guards.full()
    def fast_check():
        return guards.fast()
    try:
        limit = runtime.FollowupLimit('joint_' + args.tag, gpu=True)
        guards = GuardCadence(limit, local_check)
        check()
        result['guard_repair_lineage'] = {
            'v1_source_sha256': '13449a7e6e3f3210cdd013bb185939d8d5180fb25375c308dd402e32199e5cae',
            'preserved_v1_result': probes.verified_file({'path': '.cache/smallk_failure_followup_20261014/joint_objectives/joint_robin_y_s92601_v1/RESULT.json', 'sha256': '0c123056900e4e2c4f3bc1cc6f853106145afe0b04a5a29ec250f6e6da7da6f5'}),
            'measured_guard_diagnostic': probes.verified_file({'path': 'research/smallk_failure_followup_20261014/validation_v1/JOINT_GUARD_TIMING_DIAGNOSTIC_v1.json', 'sha256': 'd6eace339b2e762440e5ea8b1f4937e7c476eee284bd1ece022f4a57bfd55370'}),
            'change_scope': 'resource check cadence only; no cap/numerical/data/objective/optimization changes',
            'historical_failure_cost_charged_separately': True, 'u400_restart': False}
        inputs.install_protected_test_guard()
        bridge, splits, mean, std, provenance = probes.load_bridge(args.input_bridge, args.dataset, inputs.verify_phase_barrier)
        original_sources = probes.verify_source_records(bridge)
        teacher_load_started = time.perf_counter()
        raw_source = probes.check_original_raw(bridge, args.dataset, splits, mean, std)
        teacher_load_wall = time.perf_counter() - teacher_load_started
        check()
        result.update(bridge_provenance=provenance, original_RAW_teacher_source=raw_source,
                      source_snapshots=snapshot(output, probes, original_sources))
        result['cached_teacher_cost'] = {
            'reuse': 'existing original full-C RAW F(past) forecasts; no new teacher inference',
            'current_RAW_file_hash_load_compare_wall_seconds': teacher_load_wall,
            'historical_teacher_generation_compute_wall_seconds': None,
            'historical_teacher_generation_GPU_process_wall_seconds': None,
            'historical_scope': 'per-file FIT forecast generation timer unavailable in bridge; unknown rather than zero',
            'new_fit_resource_receipt_excludes_historical_teacher_generation': True}
        engine = probes.import_old_engine()
        torch = engine.torch
        codec, initial_state, cp_record = probes.load_main_codec(engine, bridge, args.dataset, args.seed)
        result['original_MAIN_checkpoint'] = cp_record
        result['initial_codec_hash'] = probes.old_state_hash(codec)
        result['initial_codec_Bi_hash'] = probes.bi_state_hash(codec)
        result['same_initialization_across_objectives_key'] = {'dataset': args.dataset, 'seed': args.seed,
            'original_MAIN_CP_sha256': cp_record['checkpoint']['sha256'], 'initial_codec_hash': result['initial_codec_hash']}
        bolt = engine.ex.previous.FrozenBolt()
        engine.replace_normalizer(bolt)
        result['F_before'] = probes.model_hashes(bolt)
        if result['F_before'] != {'old_algorithm': probes.OLD_F_HASH, 'bi_algorithm': probes.BI_F_HASH}:
            raise ValueError('ORIGINAL_F_DUAL_HASH_MISMATCH')
        if any(parameter.requires_grad for parameter in bolt.model.parameters()):
            raise ValueError('F_PARAMETER_GRADIENT_MUST_BE_FROZEN')
        main_record = bridge['declared_asset_hashes']['predictions'][args.dataset]['main_' + str(args.seed)]
        original, original_file = probes.load_checked_npz(main_record)
        main_predictions = probes.compare_reference(original, splits, mean, std)
        result['original_MAIN_prediction_source'] = original_file
        for split, values in splits.items():
            result.setdefault('original_MAIN_scores', {})[split] = prediction_scores(main_predictions[split], values, std)
            result.setdefault('original_RAW_scores', {})[split] = prediction_scores(values['raw_forecast'], values, std)
        order = training_order(args.seed, len(splits['fit']['history']))
        check()
        order_record = probes.save_npz(output / 'TRAINING_ORDER.npz', {'indices': order})
        check()
        result.update(training_order=order_record, training_order_sha256=probes.array_sha(order),
                      batch_order_scope='same seed/order/count across all Y,RAW,MIXED fits')
        check()
        result['FIT_teacher_arrays'] = probes.save_npz(output / 'FIT_TEACHER_CONTRACT.npz', {
            'cached_RAW_forecast': splits['fit']['raw_forecast'], 'actual_Y': splits['fit']['target'],
            'actual_Y_observed': splits['fit']['observed'], 'origins': splits['fit']['origins'], 'mean': mean, 'std': std})
        check()
        fit = splits['fit']
        observed = fit['observed']
        result['FIT_target_finite_contract'] = {
            'actual_Y_observed_all_finite': bool(np.isfinite(fit['target'][observed]).all()),
            'actual_Y_unobserved_all_finite': bool(np.isfinite(fit['target'][~observed]).all()),
            'actual_Y_unobserved_nonfinite_count': int((~np.isfinite(fit['target'][~observed])).sum()),
            'cached_RAW_teacher_all_finite': bool(np.isfinite(fit['raw_forecast']).all()),
            'fit_per_channel_observed_counts': observed.sum((0, 1)).tolist(),
            'zero_observed_fit_channels': np.flatnonzero(observed.sum((0, 1)) == 0).tolist(),
            'loss_rule': 'mask residual before square to keep missing-target backward finite; same active-channel counts'}
        if not result['FIT_target_finite_contract']['actual_Y_observed_all_finite'] or not result['FIT_target_finite_contract']['cached_RAW_teacher_all_finite']:
            raise ValueError('FIT_OBSERVED_Y_OR_RAW_TEACHER_NONFINITE')
        codec.train().requires_grad_(True)
        optimizer = torch.optim.AdamW(codec.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        check()
        training_started = time.perf_counter()
        for update in range(UPDATES + 1):
            local_check()
            if update in CHECKPOINTS:
                check()
                codec.eval()
                forecasts = {split: predict(engine, codec, bolt, splits[split]['history'], fast_check) for split in ('fit', 'tune')}
                if update == 0:
                    parity = {split: probes.compare_prediction(forecasts[split], main_predictions[split]) for split in forecasts}
                    result['initial_original_MAIN_parity'] = parity
                    if not all(row['pass'] for row in parity.values()):
                        raise ValueError('INITIAL_MAIN_FP32_NORMALIZED_PARITY')
                check()
                cp = save_checkpoint(output / f'u{update}.pt', torch, codec, update, args.seed, args.objective, probes)
                check()
                arrays = {split + '_prediction': forecasts[split] for split in forecasts}
                for split in forecasts:
                    arrays.update({split + '_target': splits[split]['target'], split + '_observed': splits[split]['observed'],
                                   split + '_origins': splits[split]['origins'], split + '_RAW_teacher': splits[split]['raw_forecast']})
                arrays.update(mean=mean, std=std)
                check()
                predictions = probes.save_npz(output / f'u{update}_FIT_TUNE.npz', arrays)
                check()
                checkpoint_scores = {split: prediction_scores(forecasts[split], splits[split], std) for split in forecasts}
                result['checkpoints'].append({'new_update': update, 'checkpoint': cp, 'FIT_TUNE_predictions': predictions,
                    'scores': checkpoint_scores, 'tune_actual_Y_mse': checkpoint_scores['tune']['normalized_actual_Y']['mse']})
                print(json.dumps({'event': 'JOINT_CHECKPOINT', 'dataset': args.dataset, 'seed': args.seed,
                    'objective': args.objective, 'new_update': update, 'TUNE_actual_Y_MSE': result['checkpoints'][-1]['tune_actual_Y_mse'],
                    'elapsed_seconds': time.perf_counter() - started}), flush=True)
                codec.train()
            if update == UPDATES:
                break
            indices = order[update]
            optimizer.zero_grad(set_to_none=True)
            x = engine.tensor(fit['history'][indices])
            prediction, z, future = forward_trace(codec, bolt, x)
            training_loss, y_loss, teacher_loss = objective_losses(prediction, engine.tensor(fit['target'][indices]),
                engine.tensor(fit['raw_forecast'][indices]), engine.ex.boolean(fit['observed'][indices]), args.objective)
            if not bool(torch.isfinite(training_loss)):
                raise ValueError('JOINT_OBJECTIVE_NONFINITE')
            if update in DIAGNOSTIC_UPDATES:
                proof = parameter_gradient_proof(codec, training_loss, z, future)
                proof.update(new_update=update, next_optimizer_update=update + 1)
                result['gradient_diagnostics'].append(proof)
                if not proof['frozen_F_input_gradient_finite'] or not all(row['finite'] for row in proof['E_D_gradients'].values()):
                    raise ValueError('JOINT_FROZEN_F_INPUT_OR_CODEC_GRADIENT_NONFINITE')
            training_loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(codec.parameters(), CLIP_NORM)
            if not bool(torch.isfinite(norm)):
                raise ValueError('JOINT_TOTAL_GRADIENT_NONFINITE')
            optimizer.step()
            result['curve'].append({'new_update': update + 1, 'objective_loss': float(training_loss.detach()),
                'actual_Y_loss': float(y_loss.detach()), 'cached_RAW_teacher_loss': float(teacher_loss.detach()),
                'gradient_norm_before_clip': float(norm)})
            if update % 10 == 0:
                check()
        result['training_wall_seconds'] = time.perf_counter() - training_started
        final_state = {name: value.detach().cpu().clone() for name, value in codec.state_dict().items()}
        result['final_u600_codec_hash'] = probes.old_state_hash(codec)
        result['new_update_parameter_proof'] = parameter_change_proof(initial_state, final_state)
        result['initial_fixed_basis_unchanged'] = torch.equal(initial_state['fixed_basis'], final_state['fixed_basis'])
        if not result['initial_fixed_basis_unchanged']:
            raise ValueError('FIXED_BASIS_BUFFER_CHANGED')
        result['learning_executed'] = {'optimizer_updates': len(result['curve']),
            'E_updated': result['new_update_parameter_proof']['E']['changed_parameter_tensors'] > 0,
            'D_updated': result['new_update_parameter_proof']['D']['changed_parameter_tensors'] > 0,
            'frozen_F_input_gradient_nonzero_diagnostic_count': sum(row['frozen_F_input_gradient_norm'] > 0 for row in result['gradient_diagnostics'])}
        result['learning_executed']['full_learning_proof_pass'] = (
            result['learning_executed']['optimizer_updates'] == UPDATES and result['learning_executed']['E_updated']
            and result['learning_executed']['D_updated']
            and result['learning_executed']['frozen_F_input_gradient_nonzero_diagnostic_count'] > 0)
        if not result['learning_executed']['full_learning_proof_pass']:
            raise ValueError('JOINT_600_UPDATES_E_D_CHANGE_OR_F_INPUT_GRADIENT_PROOF_MISSING')
        selected = choose_checkpoint(result['checkpoints'])
        selected_cp = result['checkpoints'][selected]['checkpoint']
        if probes.sha(selected_cp['path']) != selected_cp['sha256']:
            raise ValueError('SELECTED_NEW_CP_FILE_HASH')
        stored = torch.load(selected_cp['path'], map_location='cpu', weights_only=True)
        codec.load_state_dict(stored['codec'], strict=True)
        codec.eval().requires_grad_(False)
        result.update(selected_index=selected, selected_new_update=result['checkpoints'][selected]['new_update'],
                      selected_checkpoint=selected_cp, selected_codec_hash=probes.old_state_hash(codec),
                      selected_is_initial_original_MAIN=selected == 0)
        if result['selected_codec_hash'] != selected_cp['state_sha_old_algorithm']:
            raise ValueError('SELECTED_NEW_CODEC_HASH')
        for split, values in splits.items():
            prediction = predict(engine, codec, bolt, values['history'], fast_check)
            check()
            record = probes.save_npz(output / ('SELECTED_' + split + '.npz'), {
                'prediction': prediction, 'target': values['target'], 'observed': values['observed'],
                'origins': values['origins'], 'mean': mean, 'std': std, 'original_RAW_forecast': values['raw_forecast'],
                'original_MAIN_forecast': main_predictions[split]})
            check()
            result['selected_inference'][split] = {'artifact': record, 'scores': prediction_scores(prediction, values, std),
                                                  'role': provenance['split_roles'][split]}
        result['source_after'] = [probes.verified_file(record) for record in result['source_snapshots']]
        result['original_CP_after'] = probes.verified_file(cp_record['checkpoint'])
        result['original_MAIN_prediction_after'] = probes.verified_file(original_file)
        result['original_RAW_teacher_after'] = probes.verified_file(raw_source)
        check()
        result['status'] = 'COMPLETE'
    except Exception as error:
        budget_type = getattr(getattr(limit, 'rt', None), 'BudgetHeld', ())
        result['status'] = 'BUDGET_HELD_PRESERVED' if budget_type and isinstance(error, budget_type) else 'FAILED_PRESERVED'
        result['failure'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        if bolt is not None:
            try:
                result['F_after'] = probes.model_hashes(bolt)
                result['F_frozen_pass'] = result.get('F_before') == result['F_after']
                result['F_has_no_parameter_gradients'] = all(parameter.grad is None for parameter in bolt.model.parameters())
                if not result['F_frozen_pass'] or not result['F_has_no_parameter_gradients']:
                    result.update(status='FAILED_PRESERVED', failure={'type': 'ValueError', 'message': 'F_WEIGHT_OR_GRADIENT_CHANGED'})
            except Exception as error:
                result.update(status='FAILED_PRESERVED', F_after_unavailable={'type': type(error).__name__, 'message': str(error)})
        result['runner_wall_seconds'] = time.perf_counter() - started
        if guards is not None:
            result['guard_cadence'] = guards.summary()
        if limit is not None:
            receipt = limit.close(result['status'], None if result['status'] == 'COMPLETE' else result.get('failure'))
            result['shared_resource_receipt'] = receipt
            probes.apply_resource_terminal(result, receipt)
        elif getattr(runtime, 'LAST_CONSTRUCTOR_TERMINAL', None) is not None:
            result['shared_resource_receipt'] = runtime.LAST_CONSTRUCTOR_TERMINAL
            probes.apply_resource_terminal(result, runtime.LAST_CONSTRUCTOR_TERMINAL)
        probes.write(output / 'RESULT.json', result)
    return result


def describe():
    return {'status': 'PREPARATION_ONLY', 'actual_data_opened': False, 'GPU_executed': False,
            'conditions': 'Robin/Beijing × original seeds92601/92602 × Y/RAW/MIXED =12 fits',
            'new_updates': UPDATES, 'initialization': 'same original selected MAIN CP within each dataset/seed',
            'checkpoint_selection': 'actual-Y TUNE MSE including initial checkpoint0',
            'local_process_cap_seconds': LOCAL_SECONDS, 'same_remaining_shared_Bi_budget': True,
            'expected_GPU_process_wall_seconds_all12': [500, 900], 'hard_local_cap_sum_seconds': 1800}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--describe', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--dataset', choices=('robin', 'beijing_aotizhongxin'))
    parser.add_argument('--seed', type=int, choices=(92601, 92602))
    parser.add_argument('--objective', choices=OBJECTIVES)
    parser.add_argument('--input-bridge')
    parser.add_argument('--tag')
    args = parser.parse_args(argv)
    if args.execute and not all((args.dataset, args.seed, args.objective, args.input_bridge, args.tag)):
        parser.error('--execute requires dataset,seed,objective,input-bridge,tag')
    if args.tag is not None and (not args.tag or len(args.tag) > 64
            or any(ch not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for ch in args.tag)):
        parser.error('fresh tag must be lowercase ASCII letters/digits/underscore/hyphen up to64chars')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.describe:
        print(json.dumps(describe(), ensure_ascii=False, indent=2))
        return 0
    result = execute(args)
    print(json.dumps({'status': result['status'], 'output': str(CACHE / args.tag),
                     'selected_new_update': result.get('selected_new_update')}, ensure_ascii=False))
    return 0 if result['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
