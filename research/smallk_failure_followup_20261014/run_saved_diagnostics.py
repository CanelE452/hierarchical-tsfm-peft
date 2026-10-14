"""Run D0 and stored-RAW projection with the audited old input bridge."""
from pathlib import Path
import argparse
import json
import re
import shutil

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / '.cache' / HERE.name / 'saved_diagnostics'
NATIVE_CONTROL_JSONS = {
    ('robin', 92601): (
        'research/smallk_predictive_alignment_20261013/run_new_v1/robin/S1_ANCHOR_NONLINEAR_D_MAIN_w0_s92601_CONTROLS.json',
        'b0c8dd13df0a01048a8a17e82e3f89bbf98e7dfdf40e5fa4dfb4fa2664c9ec49'),
    ('robin', 92602): (
        'research/smallk_predictive_alignment_20261013/run_new_v1/robin/S1_ANCHOR_NONLINEAR_D_MAIN_w0_s92602_CONTROLS.json',
        '6948217af61baa48653e9f506f20be1fe62d7d720d3945da88bd54a970420248'),
    ('beijing_aotizhongxin', 92601): (
        'research/smallk_predictive_alignment_20261013/run_beijing_repair_v1/beijing_aotizhongxin/EVAL_S3_ANCHOR_NONLINEAR_D_MAIN_w0_s92601_CONTROLS.json',
        'c4cfb0890955afff8ee4ce15d4b1e4079e0197921c56152b660c0ed08e6f840e'),
    ('beijing_aotizhongxin', 92602): (
        'research/smallk_predictive_alignment_20261013/run_beijing_repair_v1/beijing_aotizhongxin/EVAL_S3_ANCHOR_NONLINEAR_D_MAIN_w0_s92602_CONTROLS.json',
        '3f27e560c08fbb7e56ac5c95a7e28003d6d2aad86e93d59f629ff4ce29abe786'),
}


def verify_native_lineage(probes, bridge, dataset, seed):
    path, expected = NATIVE_CONTROL_JSONS[dataset, seed]
    json_source = probes.verified_file({'path': path, 'sha256': expected}, role='MAIN_NATIVE_CONTROL_JSON')
    control = probes.read(json_source['path'])
    cp_key = ('robin' if dataset == 'robin' else 'beijing') + '_' + str(seed)
    checkpoint = probes.verified_file(bridge['declared_asset_hashes']['checkpoints_file_hash_only'][cp_key])
    native = probes.verified_file(bridge['declared_asset_hashes']['predictions'][dataset]['native_' + str(seed)])
    declared = control['source_checkpoint']
    if (control['dataset'] != dataset or control['seed'] != seed
            or control['architecture'] != 'ANCHOR_NONLINEAR_D'
            or probes.resolve(declared['path']) != Path(checkpoint['path'])
            or declared['sha256'] != checkpoint['sha256']
            or control['source_case'] != Path(checkpoint['path']).parent.name):
        raise ValueError('MAIN_NATIVE_SOURCE_CHECKPOINT_LINEAGE')
    methods = [entry for entry in control['methods'] if entry['method'] == 'NATIVE']
    if (len(methods) != 1
            or probes.resolve(methods[0]['prediction']['path']) != Path(native['path'])
            or methods[0]['prediction']['sha256'] != native['sha256']):
        raise ValueError('MAIN_NATIVE_PREDICTION_LINEAGE')
    models_record = probes.verified_file(control['selected_models'], role='MAIN_NATIVE_MODELS')
    import torch
    main_state = torch.load(checkpoint['path'], map_location='cpu', weights_only=True)['codec']
    native_state = torch.load(models_record['path'], map_location='cpu', weights_only=True)['codec']
    if (set(main_state) != set(native_state)
            or any(not torch.equal(main_state[key], native_state[key]) for key in main_state)):
        raise ValueError('MAIN_NATIVE_CODEC_STATE_NOT_EXACT_MAIN')
    return {'control_json': json_source, 'MAIN_checkpoint': checkpoint,
            'native_prediction': native, 'native_models': models_record,
            'same_E_D_state_torch_equal': True,
            'codec_hash_old_algorithm': probes.old_state_hash(main_state),
            'scope': 'actual CPU checkpoint state equality and JSON/prediction lineage; no new model inference'}


def execute(args):
    import prepare_inputs as inputs
    import run_condition_probes as probes
    import diagnose_saved_outputs as diagnostics
    from followup_runtime import FollowupLimit

    output = CACHE / args.tag
    output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'RUNNING', 'dataset': args.dataset, 'new_model_inference': False,
              'protected_test_access': False, 'independent_holdout': False}
    limit = None
    try:
        limit = FollowupLimit('D0_' + args.tag, gpu=False)
        inputs.install_protected_test_guard()
        bridge, splits, mean, std, provenance = probes.load_bridge(
            args.input_bridge, args.dataset, inputs.verify_phase_barrier)
        result['input_provenance'] = provenance
        source_records = probes.verify_source_records(bridge)
        raw = probes.check_original_raw(bridge, args.dataset, splits, mean, std)
        predictions = bridge['declared_asset_hashes']['predictions'][args.dataset]
        result['native_same_MAIN_lineage'] = {
            str(seed): verify_native_lineage(probes, bridge, args.dataset, seed)
            for seed in (92601, 92602)}
        candidates = []
        for prefix, name in (('main', 'MAIN'), ('native', 'MAIN_NATIVE')):
            for seed in (92601, 92602):
                record = probes.verified_file(predictions[prefix + '_' + str(seed)])
                candidates.append({'name': name, 'seed': seed, 'path': record['path'],
                                   'sha256': record['sha256']})
        last_arrays = {'mean': mean, 'std': std}
        for split, values in splits.items():
            last_arrays[split + '_last'] = values['raw_last']
            last_arrays[split + '_origins'] = values['origins']
        last = probes.save_npz(output / 'PAST_LAST.npz', last_arrays)
        first_input = next(record for record in bridge['artifacts']
                           if record['dataset'] == args.dataset and record['split'] == 'fit')
        values, _ = probes.load_checked_npz(first_input)
        config = {'schema': 'smallk_saved_diagnostics_v1', 'dataset': args.dataset,
                  'storage_scale': 'normalized', 'normalization_origin_split': 'fit',
                  'splits': list(splits), 'split_roles': provenance['split_roles'],
                  'raw': {'path': raw['path'], 'sha256': raw['sha256']},
                  'candidates': candidates, 'channels': values['columns'].tolist(),
                  'time_bins': 4, 'tail_quantiles': [.9, .95, .99],
                  'k': 4 if args.dataset == 'robin' else 2,
                  'last': {'path': last['path'], 'sha256': last['sha256']}}
        config_path = output / 'SOURCE_MANIFEST.json'
        probes.write(config_path, config)
        config['source_manifest'] = {'path': str(config_path), 'sha256': probes.sha(config_path)}
        snapshot = output / 'sources'
        snapshot.mkdir()
        for source in (Path(__file__), HERE / 'diagnose_saved_outputs.py',
                       HERE / 'run_condition_probes.py', HERE / 'prepare_inputs.py',
                       HERE / 'followup_runtime.py'):
            with source.open('rb') as original, (snapshot / source.name).open('xb') as copy:
                shutil.copyfileobj(original, copy)
        probes.write(snapshot / 'MANIFEST.json', {
            'files': {path.name: probes.sha(path) for path in snapshot.iterdir()},
            'frozen_original_sources': source_records})
        limit.check()
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=2):
            diagnostic = diagnostics.execute(config, output / 'D0')
        result['diagnostic_result'] = {'path': str(output / 'D0/RESULT.json'),
                                       'sha256': probes.sha(output / 'D0/RESULT.json'),
                                       'status': diagnostic['status']}
        result['source_manifest'] = {'path': str(config_path), 'sha256': probes.sha(config_path)}
        result['last_source'] = last
        limit.check()
        result['status'] = 'COMPLETE'
    except Exception as error:
        result.update(status='FAILED_PRESERVED', failure={'type': type(error).__name__,
                                                         'message': str(error)})
        raise
    finally:
        if limit is not None:
            result['shared_resource_receipt'] = limit.close(
                result['status'], None if result['status'] == 'COMPLETE' else result.get('failure'))
            terminal = result['shared_resource_receipt']
            if terminal.get('terminal_status', result['status']) != 'COMPLETE':
                result['status'] = terminal.get('terminal_status', result['status'])
                result['resource_terminal_error'] = terminal.get('terminal_error')
                if not result.get('failure'):
                    result['failure'] = terminal.get('terminal_error')
        with (output / 'RESULT.json').open('x', encoding='utf-8') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', required=True)
    parser.add_argument('--input-bridge', required=True)
    parser.add_argument('--dataset', choices=('robin', 'beijing_aotizhongxin'), required=True)
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', args.tag):
        parser.error('tag must be a simple fresh directory name')
    result = execute(args)
    print(json.dumps({'status': result['status'], 'output': str(CACHE / args.tag)}, ensure_ascii=True))
    return 0 if result['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
