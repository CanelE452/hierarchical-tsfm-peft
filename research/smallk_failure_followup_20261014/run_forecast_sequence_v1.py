"""Fresh approved RF1/RF3 fits from original MAIN; execution is explicitly gated."""
from pathlib import Path
import argparse
import json
import shutil
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / '.cache' / HERE.name / 'forecast_sequence_v1'
DATASETS = {'robin': {'C': 17, 'K': 4}, 'beijing_aotizhongxin': {'C': 11, 'K': 2}}
SEEDS = (92601, 92602)
HEADS = ('STATIC_RF1', 'TEMPORAL_RF3')
UPDATES = 600
CHECKPOINTS = (0, 100, 200, 400, 600)
BATCH_SIZE = 4
LEARNING_RATE = .001
WEIGHT_DECAY = 0.
CLIP_NORM = 1.
LOCAL_SECONDS = 150
DIAGNOSTIC_UPDATES = (0, 99, 199, 399, 599)
LOCKED_HELPER_SOURCES = {
    'forecast_sequence_decoder.py': 'f61bbeb1bdb104c420bbf46cd41d3b8b74c1d9931dc326739191b8303108853d',
    'run_joint_objectives_v2.py': '352cb68c499a84ce9e5e32b95f4adfb2af96c89b226da28eec44630a3b26d21f',
    'run_condition_probes.py': '94874b92486a86f56eac46dccd3ec9fe59b161aeda109f98d2f84ff0fd7c7648',
}


def training_order(seed, count):
    return np.random.default_rng(seed).integers(0, count, size=(UPDATES, BATCH_SIZE))


def observed_macro_loss(prediction, target, observed):
    import run_joint_objectives_v2 as joint
    return joint.observed_macro_loss(prediction, target, observed)


def choose_checkpoint(rows):
    import run_joint_objectives_v2 as joint
    return joint.choose_checkpoint(rows)


def forward_trace(codec, bolt, head, history):
    import forecast_sequence_decoder as sequence
    return sequence.forecast_sequence_trace(codec, bolt, head, history)


def prediction_scores(prediction, values, std):
    import run_joint_objectives_v2 as joint
    return joint.prediction_scores(prediction, values, std)


def predict_trace(engine, codec, bolt, head, history, check):
    arrays = {key: [] for key in ('prediction', 'past_z_last_two', 'absolute_future_z',
                                  'base_MAIN_style_prediction', 'head_residual', 'raw_last')}
    with engine.torch.no_grad():
        for begin in range(0, len(history), 8):
            check()
            x = engine.tensor(history[begin:begin + 8])
            prediction, past, future, residual = forward_trace(codec, bolt, head, x)
            base = x[:, -1:] + codec.decode(future) - codec.decode(past[:, -1:])
            batch = {'prediction': prediction, 'past_z_last_two': past[:, -2:], 'absolute_future_z': future,
                     'base_MAIN_style_prediction': base, 'head_residual': residual, 'raw_last': x[:, -1:]}
            for name, value in batch.items():
                arrays[name].append(value.detach().cpu().numpy())
            check()
    return {name: np.concatenate(values) for name, values in arrays.items()}


def parameter_gradient_proof(codec, head, loss, past, future):
    import torch
    import run_joint_objectives_v2 as joint
    proof = joint.parameter_gradient_proof(codec, loss, past, future)
    named = list(head.named_parameters())
    gradients = torch.autograd.grad(loss, [value for _, value in named], retain_graph=True, allow_unused=True)
    active = [gradient for gradient in gradients if gradient is not None]
    proof['Head_gradients'] = {
        'norm': float(torch.sqrt(sum(gradient.detach().square().sum() for gradient in active))) if active else 0.,
        'finite': all(gradient is None or bool(torch.isfinite(gradient).all()) for gradient in gradients),
        'active_names': [name for (name, _), gradient in zip(named, gradients)
                         if gradient is not None and bool((gradient != 0).any())],
        'zero_hidden_gradient_at_initial_update_is_expected': True,
    }
    return proof


def head_change_proof(initial, current):
    rows = []
    for name, value in current.items():
        delta = value.detach().cpu().numpy().astype(np.float64) - initial[name].detach().cpu().numpy().astype(np.float64)
        rows.append({'name': name, 'changed': bool(np.any(delta != 0)),
                     'max_abs': float(np.max(np.abs(delta))), 'l2': float(np.linalg.norm(delta))})
    return {'changed_parameter_tensors': sum(row['changed'] for row in rows),
            'registered_parameter_tensors': len(rows), 'parameters': rows,
            'output_bias_structurally_cancelled': True}


def apply_local_finalization_cap(result, elapsed_seconds, phase):
    held = elapsed_seconds >= LOCAL_SECONDS
    result.setdefault('local_finalization_checks', {})[phase] = {
        'process_elapsed_seconds': elapsed_seconds, 'local_fit_cap_seconds': LOCAL_SECONDS, 'held': held}
    if held:
        previous = result.get('failure')
        if previous is not None and previous.get('message') != 'FORECAST_SEQUENCE_FIT_PROCESS_WALL_150S':
            result.setdefault('failure_before_local_finalization', previous)
        result['status'] = 'BUDGET_HELD_PRESERVED'
        result['failure'] = {'type': 'BudgetHeld', 'message': 'FORECAST_SEQUENCE_FIT_PROCESS_WALL_150S',
                             'phase': phase, 'process_elapsed_seconds': elapsed_seconds,
                             'local_fit_cap_seconds': LOCAL_SECONDS}


def state_hashes(codec_state, head_state, probes):
    combined = {'codec.' + name: value for name, value in codec_state.items()}
    combined.update({'head.' + name: value for name, value in head_state.items()})
    return {'state_sha_old_algorithm': probes.old_state_hash(codec_state),
            'state_sha_Bi_algorithm': probes.bi_state_hash(codec_state),
            'head_state_sha_old_algorithm': probes.old_state_hash(head_state),
            'head_state_sha_Bi_algorithm': probes.bi_state_hash(head_state),
            'combined_state_sha_Bi_algorithm': probes.bi_state_hash(combined)}


def save_checkpoint(path, torch, codec, head, update, dataset, seed, head_type, probes):
    codec_state = {name: value.detach().cpu().clone() for name, value in codec.state_dict().items()}
    head_state = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
    with path.open('xb') as handle:
        torch.save({'codec': codec_state, 'head': head_state, 'new_update': update,
                    'dataset': dataset, 'seed': seed, 'head_type': head_type, 'objective': 'Y',
                    'architecture': 'ANCHOR_NONLINEAR_D', 'new_forecast_sequence_comparison': True}, handle)
    return {'path': str(path), 'sha256': probes.sha(path), 'bytes': path.stat().st_size,
            **state_hashes(codec_state, head_state, probes)}


def restore_selected_state(torch, codec, head, selected_record, dataset, seed, head_type, probes):
    checked = probes.verified_file(selected_record)
    stored = torch.load(checked['path'], map_location='cpu', weights_only=True)
    if (stored.get('dataset') != dataset or stored.get('seed') != seed or stored.get('head_type') != head_type
            or stored.get('objective') != 'Y' or stored.get('architecture') != 'ANCHOR_NONLINEAR_D'
            or stored.get('new_forecast_sequence_comparison') is not True
            or stored.get('new_update') not in CHECKPOINTS):
        raise ValueError('SELECTED_SEQUENCE_CP_IDENTITY')
    hashes = state_hashes(stored['codec'], stored['head'], probes)
    if any(selected_record[name] != value for name, value in hashes.items()):
        raise ValueError('SELECTED_SEQUENCE_CP_STATE_HASH')
    codec.load_state_dict(stored['codec'], strict=True)
    head.load_state_dict(stored['head'], strict=True)
    restored = state_hashes(codec.state_dict(), head.state_dict(), probes)
    if restored != hashes:
        raise ValueError('RESTORED_SEQUENCE_CP_STATE_HASH')
    return {'new_update': stored['new_update'], **restored}


def snapshot(output, probes, original_sources, runtime_source):
    folder = output / 'sources'
    folder.mkdir()
    paths = [Path(__file__), HERE / 'METHOD_LOCK_FORECAST_SEQUENCE_v1.md', runtime_source,
             HERE / 'continuation_runtime_v1.py', HERE / 'fast_storage_scan_v1.py',
             HERE / 'PLAN_FORECAST_SEQUENCE_DECODER_KO.md', HERE / 'prepare_inputs.py']
    paths += [HERE / name for name in LOCKED_HELPER_SOURCES]
    paths += [Path(record['path']) for record in original_sources]
    records = []
    for path in {str(path.resolve()): path.resolve() for path in paths}.values():
        destination = folder / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        before = probes.sha(path)
        with path.open('rb') as source, destination.open('xb') as copied:
            shutil.copyfileobj(source, copied)
        if probes.sha(destination) != before or probes.sha(path) != before:
            raise ValueError('SOURCE_CHANGED_DURING_SEQUENCE_SNAPSHOT')
        records.append({'path': str(path), 'sha256': before,
                        'snapshot_path': str(destination), 'snapshot_sha256': before})
    probes.write(folder / 'MANIFEST.json', {'sources': records})
    return records


class GuardCadence:
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
                'excluded': 'runtime constructor/close checks; included in whole process receipt',
                'fast_scope': 'root-approved continuation base process/campaign/device/resource caps',
                'full_scope': 'all root-approved continuation CPU/storage/disk/resource caps',
                'full_boundaries': '10 updates; checkpoint; before/after every CP/NPZ; source checks',
                'local_process_cap_seconds': LOCAL_SECONDS}


def execute(args):
    import psutil
    import run_condition_probes as probes
    output = CACHE / args.tag
    output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'RUNNING', 'new_forecast_sequence_comparison': True,
              'dataset': args.dataset, 'seed': args.seed, 'head': args.head, 'objective': 'Y', 'tag': args.tag,
              'checkpoints': [], 'curve': [], 'gradient_diagnostics': [], 'selected_inference': {},
              'protected_TEST_accessed': False, 'independent_holdout': False,
              'old_campaign_resumed': False, 'new_extra_raw_C_side_information': False,
              'local_process_wall_limit_seconds': LOCAL_SECONDS,
              'runtime_gate': 'root-owned ContinuationLimit requires explicit human approval/run manifest before asset reads',
              'recipe': {'new_updates': UPDATES, 'batch_size': BATCH_SIZE, 'optimizer': 'AdamW',
                         'lr': LEARNING_RATE, 'weight_decay': WEIGHT_DECAY, 'clip_norm': CLIP_NORM,
                         'checkpoints': list(CHECKPOINTS), 'loss_scale': 'normalized observed channel-macro actual-Y MSE',
                         'checkpoint_selection': 'actual-Y TUNE MSE, earliest exact tie, includes update0',
                         'initialization': 'same original already-Y-selected MAIN E/D CP per dataset/seed; fresh optimizer',
                         'new_head_output_initialization': 'zero weight and bias; exact zero residual',
                         'current_raw_state_anchor_C': True, 'strict_K_only_compression': False,
                         'head_inputs': 'existing E past last2 and frozen F predicted future; no actual future inputs'}}
    started = time.perf_counter()
    process_birth = psutil.Process().create_time()
    runtime = limit = guards = codec = head = bolt = None
    def local_check():
        if time.time() - process_birth >= LOCAL_SECONDS:
            raise limit.rt.BudgetHeld('FORECAST_SEQUENCE_FIT_PROCESS_WALL_150S')
        if psutil.Process().memory_info().rss > 8 * 1024 ** 3:
            raise limit.rt.BudgetHeld('FORECAST_SEQUENCE_FIT_RSS_8GIB')
    def check():
        return guards.full()
    def fast_check():
        return guards.fast()
    try:
        # This root-owned constructor is the mandatory approval/campaign gate.
        import continuation_runtime_v2 as runtime
        limit = runtime.ContinuationLimit('sequence_' + args.tag, gpu=True)
        guards = GuardCadence(limit, local_check)
        check()
        result['continuation_authorization'] = limit.authorization_record
        result['continuation_run_manifest'] = limit.run_manifest_record
        result['continuation_guard_adapter'] = limit.guard_adapter_record
        import prepare_inputs as inputs
        import forecast_sequence_decoder as sequence
        import run_joint_objectives_v2 as joint
        inputs.install_protected_test_guard()
        for name, expected in LOCKED_HELPER_SOURCES.items():
            probes.verified_file({'path': str(HERE / name), 'sha256': expected})
        load_started = time.perf_counter()
        bridge, splits, mean, std, provenance = probes.load_bridge(args.input_bridge, args.dataset, inputs.verify_phase_barrier)
        original_sources = probes.verify_source_records(bridge)
        raw_source = probes.check_original_raw(bridge, args.dataset, splits, mean, std)
        check()
        result.update(bridge_provenance=provenance, original_RAW_prediction_source=raw_source,
                      source_snapshots=snapshot(output, probes, original_sources, Path(runtime.__file__)),
                      head_capacity=sequence.capacity_summary(args.dataset))
        check()
        engine = probes.import_old_engine()
        torch = engine.torch
        codec, initial_codec, cp_record = probes.load_main_codec(engine, bridge, args.dataset, args.seed)
        if cp_record['K'] != DATASETS[args.dataset]['K'] or initial_codec['fixed_basis'].shape != (
                DATASETS[args.dataset]['C'], DATASETS[args.dataset]['K']):
            raise ValueError('LOCKED_MAIN_C_K_MISMATCH')
        head = sequence.make_head(args.dataset, args.head, args.seed).to('cuda')
        registered_head_parameters = sum(parameter.numel() for parameter in head.parameters())
        if registered_head_parameters != result['head_capacity']['registered'][args.head]:
            raise ValueError('LOCKED_HEAD_PARAMETER_COUNT')
        result['registered_head_parameters'] = registered_head_parameters
        initial_head = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
        result['original_MAIN_checkpoint'] = cp_record
        result['initial_state_hashes'] = state_hashes(initial_codec, initial_head, probes)
        result['same_initialization_across_heads_key'] = {'dataset': args.dataset, 'seed': args.seed,
            'original_MAIN_CP_sha256': cp_record['checkpoint']['sha256'],
            'initial_codec_hash': result['initial_state_hashes']['state_sha_old_algorithm']}
        f_started = time.perf_counter()
        bolt = engine.ex.previous.FrozenBolt()
        engine.replace_normalizer(bolt)
        result['F_load_wall_seconds'] = time.perf_counter() - f_started
        result['F_before'] = probes.model_hashes(bolt)
        if result['F_before'] != {'old_algorithm': probes.OLD_F_HASH, 'bi_algorithm': probes.BI_F_HASH}:
            raise ValueError('ORIGINAL_F_DUAL_HASH_MISMATCH')
        if any(parameter.requires_grad for parameter in bolt.model.parameters()):
            raise ValueError('F_PARAMETER_GRADIENT_MUST_BE_FROZEN')
        original, original_file = probes.load_checked_npz(
            bridge['declared_asset_hashes']['predictions'][args.dataset]['main_' + str(args.seed)])
        main_predictions = probes.compare_reference(original, splits, mean, std)
        result['original_MAIN_prediction_source'] = original_file
        for split, values in splits.items():
            result.setdefault('original_MAIN_scores', {})[split] = prediction_scores(main_predictions[split], values, std)
            result.setdefault('original_RAW_scores', {})[split] = prediction_scores(values['raw_forecast'], values, std)
        result['preparation_model_source_load_wall_seconds'] = time.perf_counter() - load_started
        order = training_order(args.seed, len(splits['fit']['history']))
        check()
        result['training_order'] = probes.save_npz(output / 'TRAINING_ORDER.npz', {'indices': order})
        check()
        result['training_order_sha256'] = probes.array_sha(order)
        result['batch_order_scope'] = 'identical original seed FIT count and 600x4 order across RF1/RF3'
        fit = splits['fit']
        observed = fit['observed']
        result['FIT_target_finite_contract'] = {
            'actual_Y_observed_all_finite': bool(np.isfinite(fit['target'][observed]).all()),
            'actual_Y_unobserved_all_finite': bool(np.isfinite(fit['target'][~observed]).all()),
            'actual_Y_unobserved_nonfinite_count': int((~np.isfinite(fit['target'][~observed])).sum()),
            'fit_per_channel_observed_counts': observed.sum((0, 1)).tolist(),
            'zero_observed_fit_channels': np.flatnonzero(observed.sum((0, 1)) == 0).tolist(),
            'mask_rule': 'mask residual before square; actual-Y labels enter loss only'}
        if not result['FIT_target_finite_contract']['actual_Y_observed_all_finite']:
            raise ValueError('FIT_OBSERVED_Y_NONFINITE')
        codec.eval()
        head.eval()
        warm_started = time.perf_counter()
        warm = {split: predict_trace(engine, codec, bolt, head, values['history'], fast_check)
                for split, values in splits.items()}
        result['initial_all_split_original_MAIN_parity'] = {
            split: probes.compare_prediction(values['prediction'], main_predictions[split]) for split, values in warm.items()}
        result['initial_head_residual_exact_zero'] = all(np.count_nonzero(values['head_residual']) == 0 for values in warm.values())
        if (not all(row['pass'] for row in result['initial_all_split_original_MAIN_parity'].values())
                or not result['initial_head_residual_exact_zero']):
            raise ValueError('INITIAL_ALL_SPLIT_MAIN_PARITY_OR_HEAD_ZERO')
        warm_arrays = {'mean': mean, 'std': std}
        for split, values in warm.items():
            warm_arrays.update({split + '_' + name: array for name, array in values.items()})
            warm_arrays.update({split + '_target': splits[split]['target'], split + '_observed': splits[split]['observed'],
                                split + '_origins': splits[split]['origins'], split + '_original_MAIN': main_predictions[split]})
        check()
        result['warmstart_all_splits'] = probes.save_npz(output / 'WARMSTART_ALL_SPLITS.npz', warm_arrays)
        check()
        result['warmstart_inference_save_wall_seconds'] = time.perf_counter() - warm_started
        codec.train().requires_grad_(True)
        head.train().requires_grad_(True)
        parameters = list(codec.parameters()) + list(head.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        training_started = time.perf_counter()
        for update in range(UPDATES + 1):
            fast_check()
            if update in CHECKPOINTS:
                check()
                codec.eval()
                head.eval()
                forecasts = {split: (warm[split] if update == 0 else predict_trace(
                    engine, codec, bolt, head, splits[split]['history'], fast_check)) for split in ('fit', 'tune')}
                check()
                cp = save_checkpoint(output / f'u{update}.pt', torch, codec, head, update,
                                     args.dataset, args.seed, args.head, probes)
                check()
                arrays = {'mean': mean, 'std': std}
                for split, values in forecasts.items():
                    arrays.update({split + '_' + name: array for name, array in values.items()})
                    arrays.update({split + '_target': splits[split]['target'], split + '_observed': splits[split]['observed'],
                                   split + '_origins': splits[split]['origins']})
                check()
                predictions = probes.save_npz(output / f'u{update}_FIT_TUNE.npz', arrays)
                check()
                scores = {split: prediction_scores(values['prediction'], splits[split], std)
                          for split, values in forecasts.items()}
                result['checkpoints'].append({'new_update': update, 'checkpoint': cp,
                    'FIT_TUNE_predictions': predictions, 'scores': scores,
                    'tune_actual_Y_mse': scores['tune']['normalized_actual_Y']['mse']})
                print(json.dumps({'event': 'FORECAST_SEQUENCE_CHECKPOINT', 'dataset': args.dataset, 'seed': args.seed,
                    'head': args.head, 'new_update': update, 'TUNE_actual_Y_MSE': result['checkpoints'][-1]['tune_actual_Y_mse'],
                    'elapsed_seconds': time.perf_counter() - started}), flush=True)
                codec.train()
                head.train()
            if update == UPDATES:
                break
            indices = order[update]
            optimizer.zero_grad(set_to_none=True)
            prediction, past, future, _ = forward_trace(codec, bolt, head, engine.tensor(fit['history'][indices]))
            loss = observed_macro_loss(prediction, engine.tensor(fit['target'][indices]), engine.ex.boolean(observed[indices]))
            if not bool(torch.isfinite(loss)):
                raise ValueError('SEQUENCE_ACTUAL_Y_LOSS_NONFINITE')
            if update in DIAGNOSTIC_UPDATES:
                proof = parameter_gradient_proof(codec, head, loss, past, future)
                proof.update(new_update=update, next_optimizer_update=update + 1)
                result['gradient_diagnostics'].append(proof)
                if (not proof['frozen_F_input_gradient_finite'] or not proof['Head_gradients']['finite']
                        or not all(row['finite'] for row in proof['E_D_gradients'].values())):
                    raise ValueError('SEQUENCE_F_INPUT_OR_E_D_HEAD_GRADIENT_NONFINITE')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, CLIP_NORM)
            if not bool(torch.isfinite(norm)):
                raise ValueError('SEQUENCE_TOTAL_GRADIENT_NONFINITE')
            optimizer.step()
            result['curve'].append({'new_update': update + 1, 'objective_loss': float(loss.detach()),
                                    'actual_Y_loss': float(loss.detach()), 'gradient_norm_before_clip': float(norm)})
            if (update + 1) % 10 == 0:
                check()
        result['training_checkpoint_wall_seconds'] = time.perf_counter() - training_started
        final_codec = {name: value.detach().cpu().clone() for name, value in codec.state_dict().items()}
        final_head = {name: value.detach().cpu().clone() for name, value in head.state_dict().items()}
        result['final_u600_state_hashes'] = state_hashes(final_codec, final_head, probes)
        result['new_update_parameter_proof'] = joint.parameter_change_proof(initial_codec, final_codec)
        result['new_update_parameter_proof']['Head'] = head_change_proof(initial_head, final_head)
        result['initial_fixed_basis_unchanged'] = torch.equal(initial_codec['fixed_basis'], final_codec['fixed_basis'])
        if not result['initial_fixed_basis_unchanged']:
            raise ValueError('FIXED_BASIS_BUFFER_CHANGED')
        learning = {'optimizer_updates': len(result['curve']),
                    **{group + '_updated': result['new_update_parameter_proof'][group]['changed_parameter_tensors'] > 0
                       for group in ('E', 'D', 'Head')},
                    'frozen_F_input_gradient_nonzero_diagnostic_count': sum(
                        row['frozen_F_input_gradient_norm'] > 0 for row in result['gradient_diagnostics'])}
        learning['full_learning_proof_pass'] = (learning['optimizer_updates'] == UPDATES
            and all(learning[group + '_updated'] for group in ('E', 'D', 'Head'))
            and learning['frozen_F_input_gradient_nonzero_diagnostic_count'] > 0)
        result['learning_executed'] = learning
        if not learning['full_learning_proof_pass']:
            raise ValueError('SEQUENCE_600_E_D_HEAD_UPDATE_OR_F_INPUT_GRADIENT_PROOF_MISSING')
        selected_started = time.perf_counter()
        selected = choose_checkpoint(result['checkpoints'])
        selected_cp = result['checkpoints'][selected]['checkpoint']
        restored = restore_selected_state(torch, codec, head, selected_cp, args.dataset, args.seed, args.head, probes)
        codec.eval().requires_grad_(False)
        head.eval().requires_grad_(False)
        result.update(selected_index=selected, selected_new_update=restored['new_update'], selected_checkpoint=selected_cp,
                      selected_state_hashes=restored, selected_is_initial_original_MAIN=selected == 0)
        for split, values in splits.items():
            traces = predict_trace(engine, codec, bolt, head, values['history'], fast_check)
            if selected == 0 and not probes.compare_prediction(traces['prediction'], main_predictions[split])['pass']:
                raise ValueError('SELECTED_ZERO_MAIN_PARITY')
            check()
            record = probes.save_npz(output / ('SELECTED_' + split + '.npz'), {
                **traces, 'target': values['target'], 'observed': values['observed'], 'origins': values['origins'],
                'mean': mean, 'std': std, 'original_RAW_forecast': values['raw_forecast'],
                'original_MAIN_forecast': main_predictions[split]})
            check()
            result['selected_inference'][split] = {'artifact': record,
                'scores': prediction_scores(traces['prediction'], values, std), 'role': provenance['split_roles'][split]}
        result['selected_restore_inference_save_wall_seconds'] = time.perf_counter() - selected_started
        check()
        result['source_after'] = [probes.verified_file(record) for record in result['source_snapshots']]
        result['original_CP_after'] = probes.verified_file(cp_record['checkpoint'])
        result['original_MAIN_prediction_after'] = probes.verified_file(original_file)
        result['original_RAW_prediction_after'] = probes.verified_file(raw_source)
        check()
        result['status'] = 'COMPLETE'
    except Exception as error:
        budget_type = getattr(getattr(limit, 'rt', None), 'BudgetHeld', ())
        result['status'] = 'BUDGET_HELD_PRESERVED' if budget_type and isinstance(error, budget_type) else 'FAILED_PRESERVED'
        result['failure'] = {'type': type(error).__name__, 'message': str(error)}
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
        apply_local_finalization_cap(result, time.time() - process_birth, 'pre_resource_close')
        if guards is not None:
            result['guard_cadence'] = guards.summary()
        if limit is not None:
            try:
                receipt = limit.close(result['status'], None if result['status'] == 'COMPLETE' else result.get('failure'))
                result['shared_resource_receipt'] = receipt
                probes.apply_resource_terminal(result, receipt)
            except Exception as error:
                result.update(status='FAILED_PRESERVED', resource_close_unavailable={
                    'type': type(error).__name__, 'message': str(error)})
        elif getattr(runtime, 'LAST_CONSTRUCTOR_TERMINAL', None) is not None:
            result['shared_resource_receipt'] = runtime.LAST_CONSTRUCTOR_TERMINAL
            probes.apply_resource_terminal(result, runtime.LAST_CONSTRUCTOR_TERMINAL)
        # Preserve the immutable global receipt if cooperative close itself crosses
        # the local fit cap, but never call the scientific fit COMPLETE afterward.
        apply_local_finalization_cap(result, time.time() - process_birth, 'post_resource_close')
        result['runner_wall_seconds'] = time.perf_counter() - started
        result['process_elapsed_before_RESULT_write_seconds'] = time.time() - process_birth
        probes.write(output / 'RESULT.json', result)
    return result


def describe():
    return {'status': 'SOURCE_ONLY_NOT_EXECUTED', 'actual_data_opened': False, 'GPU_executed': False,
            'conditions': [{'dataset': dataset, 'seed': seed, 'head': head} for dataset in DATASETS
                           for seed in SEEDS for head in HEADS],
            'initialization': 'original selected MAIN E/D CP; zero final head; no partial restart',
            'new_updates': UPDATES, 'batch_size': BATCH_SIZE, 'checkpoints': list(CHECKPOINTS),
            'loss': 'normalized observed channel-macro actual-Y MSE',
            'checkpoint_selection': 'actual-Y TUNE MSE, earliest tie, includes update0',
            'approval_gate': 'lazy root-owned ContinuationLimit constructor before all asset/model reads',
            'local_fit_cap_seconds': LOCAL_SECONDS, 'fit_GPU_process_cap_sum_seconds': 8 * LOCAL_SECONDS,
            'tests_for_this_runner': 'SOURCE_ONLY_NOT_RUN', 'original_sources_modified': False}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--describe', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--dataset', choices=tuple(DATASETS))
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--head', choices=HEADS)
    parser.add_argument('--input-bridge')
    parser.add_argument('--tag')
    args = parser.parse_args(argv)
    if args.execute and not all((args.dataset, args.seed, args.head, args.input_bridge, args.tag)):
        parser.error('--execute requires dataset,seed,head,input-bridge,fresh-tag')
    if args.tag is not None and (not args.tag or len(args.tag) > 64
            or any(ch not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for ch in args.tag)):
        parser.error('fresh tag must use lowercase ASCII letters/digits/underscore/hyphen up to64chars')
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
