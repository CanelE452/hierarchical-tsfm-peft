"""One old-condition D1 or D2 probe per fresh process; no assets on import.

--execute requires an actual audited input bridge and completed Bi provenance.
All scientific reads/model loads occur after that gate. No old campaign run or
old Limit is called. Resources use followup_runtime.FollowupLimit exclusively.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / '.cache' / HERE.name / 'condition_probes'
ATOL, RTOL = 1e-5, 2e-6
OLD_F_HASH = 'c2b4edf7db22d3f18e58ca79299a5c7e864e9b8f277e4e29e2f88d07dbb35874'
BI_F_HASH = 'e0e0ab8b3e15a2f965a6c1f6692c49122921253da24cc6cceb457d8a038afd7f'
EXTRA_SOURCES = {
    'research/forecast_component_study_20261013/component_experiment.py':
        'ff4cf8c9ceeaec280a6af69e8c4e586685457030feece790faba464afd282d03',
    'research/forecast_generalization_study_20261013/diagnose_constant_gradient.py':
        'cc813ba2a6044f7d309cab9b30f36360ac41e2f93d163091c831e580359bba0a',
    'research/past_only_repr_diagnosis_20261012/common.py':
        '1bb0dc92a37d3bb3af525c2e72b62ab52430237ac4653086da6bf361a8c353c2',
}
CASE_PATHS = {
    'robin': 'research/smallk_predictive_alignment_20261013/run_dev_v1/robin/S1_ANCHOR_NONLINEAR_D_MAIN_w0_s{seed}.json',
    'beijing_aotizhongxin': 'research/smallk_predictive_alignment_20261013/run_beijing_repair_v1/beijing_aotizhongxin/EVAL_S3_ANCHOR_NONLINEAR_D_MAIN_w0_s{seed}.json',
}
EXPECTED_CODECS = {
    ('robin', 92601): (100, 'e453cda947c8812721ba2c19594132ea1c93764865528871584d94b88ffc85c3'),
    ('robin', 92602): (100, 'cc936accf917d95e959bd7bb54f2de872b1c7d6e82a4c9b3c7cf08fe6b8df50a'),
    ('beijing_aotizhongxin', 92601): (400, '5fbe188ed8b1a9b33be3c12b1ace1ce2c7f51ed7b769edc906417ba959c483cf'),
    ('beijing_aotizhongxin', 92602): (600, '758cb683a2e54ab31b438d51be7b81add038998c8d5dcf7005cc1ced69ac51e5'),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(1048576), b''):
            digest.update(part)
    return digest.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def resolve(path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def verified_file(record, *, role=None):
    path = resolve(record['path'])
    value = sha(path)
    if value != record['sha256']:
        raise ValueError('SOURCE_FILE_HASH_MISMATCH: ' + str(path))
    return {'path': str(path), 'sha256': value, 'verified_sha256': value,
            'role': role or record.get('role', '')}


def save_npz(path, arrays):
    with Path(path).open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    return {'path': str(Path(path).resolve()), 'sha256': sha(path),
            'bytes': Path(path).stat().st_size,
            'arrays': {key: {'shape': list(value.shape), 'dtype': str(value.dtype),
                             'sha256': array_sha(value)} for key, value in arrays.items()}}


def load_checked_npz(record):
    checked = verified_file(record)
    with np.load(checked['path'], allow_pickle=False) as opened:
        arrays = {key: opened[key].copy() for key in opened.files}
    declared = record.get('arrays')
    if declared is not None:
        if set(arrays) != set(declared):
            raise ValueError('BRIDGE_ARRAY_KEY_MISMATCH')
        for key, value in arrays.items():
            entry = declared[key]
            if (list(value.shape) != entry['shape'] or str(value.dtype) != entry['dtype']
                    or array_sha(value) != entry['sha256']):
                raise ValueError('BRIDGE_ARRAY_CONTENT_MISMATCH: ' + key)
    return arrays, checked


def compare_reference(reference, bridge_splits, mean, std):
    for key, exact in (('mean', mean), ('std', std)):
        if not np.array_equal(reference[key], exact):
            raise ValueError('REFERENCE_STATS_MISMATCH: ' + key)
    predictions = {}
    for split, values in bridge_splits.items():
        for suffix, exact in (('target', values['target']), ('observed', values['observed']),
                              ('origins', values['origins'])):
            value = reference[split + '_' + suffix]
            if value.dtype != exact.dtype or not np.array_equal(value, exact, equal_nan=True):
                raise ValueError('REFERENCE_CONDITION_MISMATCH: ' + split + '/' + suffix)
        prediction = reference[split + '_prediction']
        if prediction.dtype != np.float32 or prediction.shape != values['target'].shape:
            raise ValueError('REFERENCE_PREDICTION_SHAPE_DTYPE: ' + split)
        if not np.isfinite(prediction).all():
            raise ValueError('REFERENCE_PREDICTION_NONFINITE: ' + split)
        predictions[split] = prediction
    return predictions


def load_bridge(path, dataset, barrier_verifier):
    path = Path(path).resolve()
    manifest = read(path)
    if manifest.get('status') != 'COMPLETE' or dataset not in manifest.get('datasets', {}):
        raise ValueError('ACTUAL_BRIDGE_COMPLETE_DATASET_REQUIRED')
    actual = manifest.get('actual_status', {})
    if actual.get('old_npz_opened') is not True or actual.get('old_checkpoint_file_hash_done') is not True:
        raise ValueError('ACTUAL_BRIDGE_ASSET_CHECK_REQUIRED')
    phase = manifest['phase_barrier']
    for name in ('audit', 'report', 'manifest'):
        verified_file(phase[name], role='completed_bi_' + name)
    barrier = barrier_verifier(*(Path(phase[name]['path']) for name in ('audit', 'report', 'manifest')))
    records = [record for record in manifest['artifacts'] if record['dataset'] == dataset]
    expected = ('fit', 'tune', 'val' if dataset == 'robin' else 'eval')
    if {record['split'] for record in records} != set(expected) or len(records) != len(expected):
        raise ValueError('BRIDGE_SPLIT_COVERAGE')
    splits, sources, mean, std = {}, [], None, None
    for split in expected:
        record = next(record for record in records if record['split'] == split)
        arrays, checked = load_checked_npz(record)
        if mean is None:
            mean, std = arrays['mean'], arrays['std']
        if not np.array_equal(mean, arrays['mean']) or not np.array_equal(std, arrays['std']):
            raise ValueError('BRIDGE_SPLIT_STATS_DIFFER')
        if not np.array_equal(arrays['raw_last'], arrays['history'][:, -1]):
            raise ValueError('BRIDGE_LAST_NOT_EXACT_HISTORY_LAST')
        splits[split] = {key: arrays[key] for key in ('history', 'history_observed', 'raw_forecast',
                          'target', 'observed', 'origins', 'raw_last')}
        sources.append(checked)
    return manifest, splits, mean, std, {
        'input_bridge': {'path': str(path), 'sha256': sha(path), 'verified_sha256': sha(path)},
        'phase_barrier': barrier, 'input_artifacts': sources,
        'split_roles': manifest['datasets'][dataset]['split_roles'],
    }


def verify_source_records(bridge):
    records = [verified_file(record) for record in bridge['declared_asset_hashes']['source'].values()]
    for path, expected in EXTRA_SOURCES.items():
        records.append(verified_file({'path': path, 'sha256': expected}, role='frozen_old_source'))
    return records


def check_original_raw(bridge, dataset, splits, mean, std):
    record = bridge['declared_asset_hashes']['predictions'][dataset]['raw']
    original, source = load_checked_npz(record)
    predictions = compare_reference(original, splits, mean, std)
    for split, prediction in predictions.items():
        if not np.array_equal(prediction, splits[split]['raw_forecast']):
            raise ValueError('BRIDGE_RAW_FORECAST_NOT_ORIGINAL: ' + split)
    return source


def import_old_engine():
    original_common = ROOT / 'research/past_only_repr_diagnosis_20261012'
    old = ROOT / 'research/smallk_predictive_alignment_20261013'
    sys.path.insert(0, str(original_common))
    sys.path.insert(0, str(old))
    source = old / 'engine.py'
    specification = importlib.util.spec_from_file_location('followup_original_engine', source)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    if Path(module.ex.__file__).resolve() != ROOT / 'research/forecast_generalization_study_20261013/experiment.py':
        raise ValueError('ORIGINAL_EX_MODULE_PATH')
    if Path(module.ex.previous.__file__).resolve() != ROOT / 'research/forecast_component_study_20261013/component_experiment.py':
        raise ValueError('ORIGINAL_PREVIOUS_MODULE_PATH')
    module.configure()
    return module


def old_state_hash(module):
    digest = hashlib.sha256()
    state = module.state_dict() if hasattr(module, 'state_dict') else module
    for key, value in state.items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def bi_state_hash(module):
    digest = hashlib.sha256()
    state = module.state_dict() if hasattr(module, 'state_dict') else module
    for key, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def model_hashes(bolt):
    return {'old_algorithm': old_state_hash(bolt.model), 'bi_algorithm': bi_state_hash(bolt.model)}


def compare_prediction(left, right):
    difference = np.asarray(left, np.float64) - np.asarray(right, np.float64)
    return {'max_abs': float(np.abs(difference).max()),
            'rmse': float(np.sqrt(np.mean(difference ** 2))),
            'pass': bool(np.allclose(left, right, atol=ATOL, rtol=RTOL)),
            'normalized_tolerance': {'atol': ATOL, 'rtol': RTOL}}


def load_main_codec(engine, bridge, dataset, seed):
    import torch
    checkpoint_key = ('robin' if dataset == 'robin' else 'beijing') + '_' + str(seed)
    checkpoint_record = bridge['declared_asset_hashes']['checkpoints_file_hash_only'][checkpoint_key]
    checked = verified_file(checkpoint_record, role='MAIN_CP')
    case_path = resolve(CASE_PATHS[dataset].format(seed=seed))
    case = read(case_path)
    declared = case['training']['selected_checkpoint']
    if resolve(declared['path']) != Path(checked['path']) or declared['sha256'] != checked['sha256']:
        raise ValueError('MAIN_SELECTED_CP_LINEAGE')
    if case['dataset'] != dataset or case['seed'] != seed or case['architecture'] != 'ANCHOR_NONLINEAR_D':
        raise ValueError('MAIN_CASE_IDENTITY')
    checkpoint = torch.load(checked['path'], map_location='cpu', weights_only=True)
    expected_update, expected_hash = EXPECTED_CODECS[dataset, seed]
    if (case['training']['selected_update'] != expected_update
            or case['training']['selected_codec_hash'] != expected_hash):
        raise ValueError('MAIN_PREDECLARED_SELECTION_MISMATCH')
    if (checkpoint['seed'] != seed or checkpoint['architecture'] != case['architecture']
            or checkpoint['update'] != case['training']['selected_update']
            or checkpoint['method'] != 'MAIN' or checkpoint['weight'] != 0.):
        raise ValueError('MAIN_CP_METADATA')
    state = checkpoint['codec']
    if old_state_hash(state) != case['training']['selected_codec_hash']:
        raise ValueError('MAIN_CP_CODEC_STATE_SHA')
    basis = state['fixed_basis'].detach().cpu().numpy().copy()
    codec = engine.Codec(np.zeros(basis.shape[0], np.float32), basis, seed, case['architecture'])
    codec.load_state_dict(state, strict=True)
    codec.eval().requires_grad_(False).to('cuda')
    if old_state_hash(codec) != case['training']['selected_codec_hash']:
        raise ValueError('LOADED_MAIN_CODEC_STATE_SHA')
    record = {'checkpoint': checked, 'case': {'path': str(case_path), 'sha256': sha(case_path)},
              'selected_update': case['training']['selected_update'],
              'selected_codec_hash_old_algorithm': case['training']['selected_codec_hash'],
              'selected_codec_hash_bi_algorithm': bi_state_hash(codec),
              'architecture': case['architecture'], 'K': case['K'],
              'training_batch_size': case['training']['batch_size'], 'original_prediction_batch_size': 8}
    return codec, state, record


def extract_absolute_latents(splits, codec, bolt, *, tensor, no_grad, synchronize, check, batch_size=8):
    extracted, timings, parity = {}, {}, {}
    for split, values in splits.items():
        started = time.perf_counter()
        chunks = {key: [] for key in ('latent_past', 'z_future', 'z_last', 'source_gpu_prediction')}
        with no_grad():
            for begin in range(0, len(values['history']), batch_size):
                check()
                history = tensor(values['history'][begin:begin + batch_size])
                z = codec.encode(history)
                future = bolt(z)
                last = z[:, -1:]
                prediction = history[:, -1:] + codec.decode(future) - codec.decode(last)
                for key, value in (('latent_past', z), ('z_future', future), ('z_last', last),
                                   ('source_gpu_prediction', prediction)):
                    chunks[key].append(value.detach().cpu().numpy())
                check()
        synchronize()
        arrays = {key: np.concatenate(value).astype(np.float32, copy=False) for key, value in chunks.items()}
        parity[split] = compare_prediction(arrays['source_gpu_prediction'], values['original_prediction'])
        extracted[split] = {**arrays, 'raw_last': values['raw_last'], 'original_prediction': values['original_prediction'],
                            'raw_forecast': values['raw_forecast'], 'target': values['target'],
                            'observed': values['observed'], 'origins': values['origins'],
                            'history_observed': values['history_observed']}
        timings[split] = {'full_E_F_D_device_copy_wall_seconds': time.perf_counter() - started,
                          'windows': len(values['history']), 'batch_size': batch_size}
    return extracted, timings, parity


def decoder_source_record(dataset, seed, decoder, provenance, cp_record, prediction_source,
                          raw_source, sources, latent_artifacts, parity):
    from frozen_decoder_refit import decoder_state_sha256
    aliases = [('MAIN_CP', cp_record['checkpoint']), ('MAIN_PREDICTION', prediction_source),
               ('RAW_PREDICTION', raw_source)] + [('ORIGINAL_SOURCE', source) for source in sources]
    return {
        'dataset': dataset, 'seed': seed, 'architecture': 'ANCHOR_NONLINEAR_D',
        'input_length': 512, 'bi_complete': True,
        'decoder_state_sha256': decoder_state_sha256(decoder),
        'verification': {key: True for key in ('source_checkpoint', 'source_snapshots',
            'same_original_main_E_F', 'absolute_latents', 'original_stats_origins_masks', 'split_roles')},
        'verification_evidence': {'bridge': provenance, 'checkpoint': cp_record,
                                  'absolute_latents': latent_artifacts, 'source_D_parity': parity},
        'source_artifacts': [{**source, 'alias': alias} for alias, source in aliases],
    }


def source_snapshot(output, records):
    folder = output / 'sources'
    folder.mkdir()
    copied = []
    for path in (Path(__file__), HERE / 'projected_past_probe.py', HERE / 'frozen_decoder_refit.py',
                 HERE / 'prepare_inputs.py', HERE / 'followup_runtime.py'):
        target = folder / path.name
        with path.open('rb') as source, target.open('xb') as destination:
            shutil.copyfileobj(source, destination)
        copied.append({'path': str(path), 'sha256': sha(path), 'snapshot_path': str(target),
                       'snapshot_sha256': sha(target)})
    for record in {entry['path']: entry for entry in records}.values():
        path = Path(record['path'])
        target = folder / 'frozen_original' / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        with path.open('rb') as source, target.open('xb') as destination:
            shutil.copyfileobj(source, destination)
        if sha(target) != record['verified_sha256']:
            raise ValueError('FROZEN_SOURCE_SNAPSHOT_MISMATCH')
        copied.append({'path': str(path), 'sha256': record['sha256'],
                       'snapshot_path': str(target), 'snapshot_sha256': sha(target)})
    write(folder / 'MANIFEST.json', {'new_sources': copied, 'frozen_old_sources': records})
    return copied


def apply_resource_terminal(result, receipt):
    terminal_status = receipt.get('terminal_status', result['status'])
    if terminal_status != 'COMPLETE':
        result['status'] = terminal_status
        terminal_error = receipt.get('terminal_error')
        if terminal_error is not None:
            result['resource_terminal_error'] = terminal_error
            if 'failure' not in result:
                result['failure'] = terminal_error if isinstance(terminal_error, dict) else {
                    'type': 'ResourceTerminalFailure', 'message': str(terminal_error)}


def execute(args):
    import prepare_inputs as inputs
    from followup_runtime import FollowupLimit
    from projected_past_probe import run_probe, make_bolt_predictor
    from frozen_decoder_refit import run_decoder_refit, DecoderState
    import psutil
    if args.stage == 'D2' and args.seed is None:
        raise ValueError('D2_SINGLE_ORIGINAL_SEED_REQUIRED')
    if args.stage == 'D1' and args.seed is not None:
        raise ValueError('D1_HAS_NO_TRAINING_SEED')
    output = CACHE / args.tag
    output.mkdir(parents=True, exist_ok=False)
    result = {'status': 'RUNNING', 'scientific_execution': True, 'stage': args.stage,
              'dataset': args.dataset, 'seed': args.seed, 'tag': args.tag,
              'protected_test_access': False, 'large_backbone_training': False,
              'old_campaign_restart': False, 'independent_holdout': False,
              'tolerance': {'atol': ATOL, 'rtol': RTOL}, 'artifacts': [], 'timings': {}}
    limit, bolt, codec = None, None, None
    started = time.perf_counter()
    process_birth = psutil.Process().create_time()
    def check():
        if time.time() - process_birth >= 300:
            raise RuntimeError('CONDITION_PROBE_PROCESS_WALL_300S')
        return limit.check()
    try:
        limit = FollowupLimit(args.tag, gpu=True)
        inputs.install_protected_test_guard()
        t = time.perf_counter()
        bridge, splits, mean, std, provenance = load_bridge(args.input_bridge, args.dataset, inputs.verify_phase_barrier)
        result['bridge_provenance'] = provenance
        sources = verify_source_records(bridge)
        raw_source = check_original_raw(bridge, args.dataset, splits, mean, std)
        result['original_RAW_reference'] = raw_source
        result['source_snapshot'] = source_snapshot(output, sources)
        result['timings']['input_hash_load_validate_wall_seconds'] = time.perf_counter() - t
        check()
        t = time.perf_counter()
        engine = import_old_engine()
        bolt = engine.ex.previous.FrozenBolt()
        engine.replace_normalizer(bolt)
        engine.torch.cuda.synchronize()
        before = model_hashes(bolt)
        result['F_before'] = before
        if before != {'old_algorithm': OLD_F_HASH, 'bi_algorithm': BI_F_HASH}:
            raise ValueError('ORIGINAL_F_DUAL_STATE_HASH_MISMATCH')
        result['timings']['import_configure_model_load_hash_wall_seconds'] = time.perf_counter() - t
        check()
        if args.stage == 'D1':
            t = time.perf_counter()
            probe = run_probe(splits, make_bolt_predictor(bolt), mean, std, output / 'D1',
                              provenance={**provenance, 'frozen_old_sources': sources, 'F_before': before},
                              actual=True, shared_limit=limit, batch_size=8,
                              identity_atol=ATOL, identity_rtol=RTOL)
            result['probe_result'] = {'path': str(output / 'D1/RESULT.json'),
                                      'sha256': sha(output / 'D1/RESULT.json'), 'status': probe['status']}
            result['timings']['D1_full_probe_save_wall_seconds'] = time.perf_counter() - t
        else:
            t = time.perf_counter()
            codec, state, cp_record = load_main_codec(engine, bridge, args.dataset, args.seed)
            result['MAIN_checkpoint'] = cp_record
            source_prediction_record = bridge['declared_asset_hashes']['predictions'][args.dataset]['main_' + str(args.seed)]
            original, prediction_source = load_checked_npz(source_prediction_record)
            original_predictions = compare_reference(original, splits, mean, std)
            for split in splits:
                splits[split]['original_prediction'] = original_predictions[split]
            result['timings']['MAIN_checkpoint_reference_load_wall_seconds'] = time.perf_counter() - t
            extracted, timings, parity = extract_absolute_latents(
                splits, codec, bolt, tensor=engine.tensor, no_grad=engine.torch.no_grad,
                synchronize=engine.torch.cuda.synchronize, check=check, batch_size=8)
            result['timings']['D2_absolute_latent_extraction'] = timings
            result['source_GPU_D_vs_stored_MAIN'] = parity
            for split, arrays in extracted.items():
                result['artifacts'].append(save_npz(output / ('ABSOLUTE_LATENTS_' + split + '.npz'),
                                                   {**arrays, 'mean': mean, 'std': std}))
            if not all(row['pass'] for row in parity.values()):
                raise ValueError('ORIGINAL_SOURCE_GPU_D_VS_STORED_MAIN_PARITY')
            decoder = DecoderState.from_codec_state(state)
            source_record = decoder_source_record(args.dataset, args.seed, decoder, provenance, cp_record,
                                                   prediction_source, raw_source, sources, result['artifacts'], parity)
            t = time.perf_counter()
            refit = run_decoder_refit(output / 'D2', splits=extracted, codec_state=state, mean=mean, std=std,
                                     source_record=source_record, shared_limit=limit, execution_authorized=True,
                                     parity_atol=ATOL, parity_rtol=RTOL)
            result['probe_result'] = {'path': str(output / 'D2/RESULT.json'),
                                      'sha256': sha(output / 'D2/RESULT.json'), 'status': refit['status']}
            result['timings']['D2_CPU_refit_save_wall_seconds'] = time.perf_counter() - t
        check()
        result['status'] = 'COMPLETE'
    except Exception as error:
        result.update(status='FAILED_PRESERVED', failure={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        if bolt is not None:
            result['F_after'] = model_hashes(bolt)
            result['F_frozen_pass'] = result.get('F_before') == result['F_after']
            if result.get('F_frozen_pass') is False:
                result.update(status='FAILED_PRESERVED', failure={'type': 'ValueError', 'message': 'F_CHANGED'})
        if codec is not None:
            result['MAIN_codec_hash_after'] = old_state_hash(codec)
            result['MAIN_codec_frozen_pass'] = result['MAIN_codec_hash_after'] == result['MAIN_checkpoint']['selected_codec_hash_old_algorithm']
            if not result['MAIN_codec_frozen_pass']:
                result.update(status='FAILED_PRESERVED', failure={'type': 'ValueError', 'message': 'MAIN_CODEC_CHANGED'})
        result['timings']['runner_wall_seconds'] = time.perf_counter() - started
        if limit is not None:
            receipt = limit.close(result['status'], None if result['status'] == 'COMPLETE' else result.get('failure'))
            result['shared_resource_receipt'] = receipt
            apply_resource_terminal(result, receipt)
        write(output / 'RESULT.json', result)
    return result


def describe():
    return {'status': 'PREPARATION_ONLY', 'actual_data_opened': False, 'gpu_executed': False,
            'process_scope': 'one D1 dataset or one D2 dataset+seed; helper process-birth cap300s',
            'shared_limit': 'followup_runtime.FollowupLimit; existing Bi caps, no budget reset',
            'D1': 'seven fixed identity/quarter/half/full-C-rotated projection candidates; FIT RAW delta only',
            'D2': 'absolute original E(X),F(E(X)); original D replay, then frozen hidden output head Y/RAW/MIXED refit',
            'prediction_batch_size': 8, 'training_batch_size_of_original_MAIN': 4,
            'reinference_tolerance': {'atol': ATOL, 'rtol': RTOL},
            'old_F_hash': OLD_F_HASH, 'bi_F_hash': BI_F_HASH}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--describe', action='store_true')
    mode.add_argument('--execute', action='store_true')
    parser.add_argument('--stage', choices=('D1', 'D2'))
    parser.add_argument('--dataset', choices=tuple(CASE_PATHS))
    parser.add_argument('--seed', type=int, choices=(92601, 92602))
    parser.add_argument('--input-bridge')
    parser.add_argument('--tag')
    args = parser.parse_args(argv)
    if args.execute and not all((args.stage, args.dataset, args.input_bridge, args.tag)):
        parser.error('--execute requires --stage --dataset --input-bridge --tag')
    if args.tag is not None and (not args.tag or any(ch not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for ch in args.tag)):
        parser.error('--tag must use lowercase ASCII letters/digits/underscore/hyphen')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.describe:
        print(json.dumps(describe(), ensure_ascii=False, indent=2))
        return 0
    result = execute(args)
    print(json.dumps({'status': result['status'], 'output': str(CACHE / args.tag)}, ensure_ascii=False))
    return 0 if result['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
