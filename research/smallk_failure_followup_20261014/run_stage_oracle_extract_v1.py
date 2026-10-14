"""SOURCE-only default. Fresh approval gates real old-CP oracle extraction.

Uses cached D2 F outputs and a frozen original E/D. No new F load/forward,
optimizer, old campaign resume, or deployment using future observations.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT/'.cache'/HERE.name/'stage_separation_v1/oracle'
DATASETS = {'robin': (17, 4), 'beijing_aotizhongxin': (11, 2)}
SEEDS = (92601, 92602)
sys.path.insert(0, str(HERE))


def safe_tag(tag):
    if not tag or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in tag):
        raise ValueError('FRESH_SAFE_TAG_REQUIRED')
    return tag


def extract(args):
    import stage_separation_runtime_v1 as runtime
    limit = runtime.ContinuationLimit('stage_oracle_' + safe_tag(args.tag), gpu=True)
    output = CACHE/args.tag
    result = {'status': 'RUNNING', 'dataset': args.dataset, 'seed': args.seed,
              'diagnostic_only': True, 'future_observations_used': True,
              'past_only_forecasting_result': False, 'F_loaded': False,
              'F_forward_replayed': False, 'new_optimizer_updates': 0,
              'protected_TEST_accessed': False, 'whole_goal_complete': False,
              'artifacts': {}, 'source_records': [], 'checks': {},
              'permission_gate': limit.authorization_record, 'run_manifest': limit.run_manifest_record}
    codec = None
    initialized = False
    started = time.perf_counter()
    try:
        output.mkdir(parents=True, exist_ok=False)
        initialized = True
        import numpy as np
        import torch
        import prepare_inputs as inputs
        import run_condition_probes as probes
        import stage_separation_oracle_v1 as oracle
        inputs.install_protected_test_guard()
        def fast():
            return limit.shared.check()
        def full():
            return limit.check()
        bridge, splits, mean, std, provenance = probes.load_bridge(args.input_bridge, args.dataset, inputs.verify_phase_barrier)
        result['bridge_provenance'] = provenance
        result['source_records'].extend(probes.verify_source_records(bridge))
        result['source_records'].extend(provenance['input_artifacts'])
        result['source_records'].append(provenance['input_bridge'])
        result['source_records'].append(probes.check_original_raw(bridge, args.dataset, splits, mean, std))
        d2_path = probes.resolve(args.d2_result)
        oracle.assert_no_protected_test_path(d2_path, role='old_D2_producer')
        d2_record = {'path': str(d2_path), 'sha256': probes.sha(d2_path)}
        d2 = probes.read(d2_path)
        if (d2.get('status') != 'COMPLETE' or d2.get('stage') != 'D2'
                or d2.get('dataset') != args.dataset or d2.get('seed') != args.seed
                or d2.get('F_frozen_pass') is not True or d2.get('MAIN_codec_frozen_pass') is not True
                or d2.get('F_before') != {'old_algorithm': probes.OLD_F_HASH, 'bi_algorithm': probes.BI_F_HASH}
                or d2.get('F_before') != d2.get('F_after')):
            raise ValueError('ORIGINAL_D2_COMPLETE_CP_AND_F_LINEAGE_REQUIRED')
        result['D2_source'] = d2_record
        result['source_records'].append(d2_record)
        for item in d2.get('source_snapshot', []):
            result['source_records'].append(probes.verified_file(item))
            if probes.sha(item['snapshot_path']) != item['sha256']:
                raise ValueError('OLD_D2_SOURCE_SNAPSHOT_CHANGED')
        full()
        engine = probes.import_old_engine()
        codec, initial_state, cp_record = probes.load_main_codec(engine, bridge, args.dataset, args.seed)
        if cp_record['checkpoint']['sha256'] != d2['MAIN_checkpoint']['checkpoint']['sha256']:
            raise ValueError('SAME_ORIGINAL_MAIN_CHECKPOINT_REQUIRED')
        c, k = DATASETS[args.dataset]
        if tuple(initial_state['fixed_basis'].shape) != (c, k):
            raise ValueError('LOCKED_C_K_REQUIRED')
        result['MAIN_checkpoint'] = cp_record
        result['source_records'].extend((cp_record['checkpoint'], cp_record['case']))
        result['codec_before'] = {'old_algorithm': probes.old_state_hash(codec), 'bi_algorithm': probes.bi_state_hash(codec)}
        source_folder = output/'sources'
        source_folder.mkdir()
        snapshots = []
        for path in (Path(__file__).resolve(), HERE/'stage_separation_runtime_v1.py',
                     HERE/'stage_separation_oracle_v1.py', HERE/'METHOD_LOCK_STAGE_SEPARATION_v1.md',
                     HERE/'PLAN_STAGE_SEPARATION_KO_v1.md', HERE/'run_condition_probes.py',
                     HERE/'prepare_inputs.py', HERE/'fast_storage_scan_v1.py'):
            record = {'path': str(path), 'sha256': probes.sha(path)}
            destination = source_folder/path.name
            shutil.copyfile(path, destination)
            if probes.sha(destination) != record['sha256']:
                raise ValueError('SOURCE_SNAPSHOT_MISMATCH')
            snapshots.append({**record, 'snapshot_path': str(destination)})
        result['source_snapshots'] = snapshots
        result['D2_latent_reuse'] = {}
        result['coverage'] = {}
        for split, values in splits.items():
            full()
            old = next(r for r in d2['artifacts'] if Path(r['path']).name == 'ABSOLUTE_LATENTS_' + split + '.npz')
            cached, checked = probes.load_checked_npz(old)
            oracle.validate_d2_cached_latent_payload(cached)
            for name in ('target', 'observed', 'origins', 'raw_last', 'history_observed', 'raw_forecast'):
                if not np.array_equal(cached[name], values[name], equal_nan=True):
                    raise ValueError('OLD_D2_SAME_INPUT_ARRAY_REQUIRED:' + split + ':' + name)
            if not np.array_equal(cached['mean'], mean) or not np.array_equal(cached['std'], std):
                raise ValueError('OLD_D2_SAME_FIT_STATS_REQUIRED')
            result['D2_latent_reuse'][split] = old
            result['source_records'].append(checked)
            extracted = oracle.extract_normalized_oracle(codec, values['history'], values['target'], values['observed'],
                history_observed=values['history_observed'], tensor=engine.tensor,
                no_grad=torch.no_grad, check=fast, synchronize=torch.cuda.synchronize, batch_size=8)
            comparisons = {
                'past512_vs_old_D2': oracle.compare_arrays(extracted['current_past_latent'], cached['latent_past']),
                'past_z_last_vs_old_D2': oracle.compare_arrays(extracted['current_z_last'], cached['z_last']),
                'past_vs560_prefix': extracted['prefix_check'],
                'future_perturb_prefix': extracted['first_sample_forward_invariance']}
            result['checks'][split] = comparisons
            if not all(v.get('pass') is True for v in comparisons.values()):
                raise ValueError('ACTUAL_ENCODER_ORACLE_PREFIX_OR_CACHED_PARITY_NOT_PASSED:' + split)
            if not np.array_equal(extracted['future_filled'][values['observed']], values['target'][values['observed']]):
                raise ValueError('ORACLE_OBSERVED_NORMALIZED_TARGET_CHANGED')
            if extracted['oracle_z'].shape != (len(values['history']), 48, k):
                raise ValueError('ORACLE_LATENT_SHAPE')
            arrays = {'oracle_z': extracted['oracle_z'],
                      'past_prefix_latent': extracted['oracle_past_prefix_latent'],
                      'current_past_latent': extracted['current_past_latent'],
                      'current_z_last': extracted['current_z_last'],
                      'future_filled_normalized': extracted['future_filled'],
                      'future_fill_source': extracted['future_fill_source'],
                      'complete_future_prefix_mask': extracted['complete_future_prefix_mask'],
                      'target': values['target'], 'observed': values['observed'],
                      'origins': values['origins'], 'raw_last': values['raw_last'], 'mean': mean, 'std': std}
            result['artifacts'][split] = probes.save_npz(output/('ORACLE_DIAGNOSTIC_' + split + '.npz'), arrays)
            result['coverage'][split] = {
                'windows': len(values['history']), 'teacher_prefix_counts_by_horizon': extracted['complete_future_prefix_counts'],
                'original_Y_channel_counts': values['observed'].sum((0, 1)).tolist(),
                'normalization_contract': extracted['normalization_contract']}
            full()
        result['codec_after'] = {'old_algorithm': probes.old_state_hash(codec), 'bi_algorithm': probes.bi_state_hash(codec)}
        if result['codec_after'] != result['codec_before'] or any(p.grad is not None for p in codec.parameters()):
            raise ValueError('ORIGINAL_CODEC_CHANGED_OR_PARAMETER_GRADIENT')
        for record in result['source_records']:
            probes.verified_file(record)
        for record in snapshots:
            if probes.sha(record['snapshot_path']) != record['sha256'] or probes.sha(record['path']) != record['sha256']:
                raise ValueError('NEW_SOURCE_CHANGED_DURING_EXTRACTION')
        result.update(status='COMPLETE', encoder_extraction_wall_seconds=time.perf_counter()-started,
                      original_codec_frozen_pass=True, F_reference_scope='cached old D2 forward; F not loaded/replayed in this process')
        full()
    except Exception as error:
        result.update(status='BUDGET_HELD_PRESERVED' if isinstance(error, runtime.BudgetHeld) else 'FAILED_PRESERVED',
                      error={'type': type(error).__name__, 'message': str(error)})
    finally:
        if codec is not None:
            import run_condition_probes as probes
            result['codec_terminal_hash'] = {'old_algorithm': probes.old_state_hash(codec), 'bi_algorithm': probes.bi_state_hash(codec)}
            if result.get('codec_before') != result['codec_terminal_hash']:
                result.update(status='FAILED_PRESERVED', error={'message': 'ORIGINAL_CODEC_CHANGED'})
        result['resource_receipt'] = limit.close(result['status'], result.get('error'))
        if result['resource_receipt']['terminal_status'] != 'COMPLETE':
            result['status'] = result['resource_receipt']['terminal_status']
        if initialized:
            runtime.write_new(output/'RESULT.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--describe', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--dataset', choices=tuple(DATASETS))
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--input-bridge')
    parser.add_argument('--d2-result')
    parser.add_argument('--tag')
    args = parser.parse_args()
    if args.describe:
        print(json.dumps({'status': 'SOURCE_ONLY_NOT_EXECUTED', 'actual_assets_opened': False,
                          'scope': 'frozen E future-oracle diagnostic only; cached F reused',
                          'permission': 'fresh explicit stage-separation approval and fixed manifest required'}))
        return
    if not all((args.dataset, args.seed, args.input_bridge, args.d2_result, args.tag)):
        parser.error('execute needs exact original condition/actual bridge/D2 producer/fresh tag')
    result = extract(args)
    print(json.dumps({'status': result['status'], 'diagnostic_only': True, 'F_forward_replayed': False}))
    raise SystemExit(0 if result['status'] == 'COMPLETE' else 1)


if __name__ == '__main__':
    main()
