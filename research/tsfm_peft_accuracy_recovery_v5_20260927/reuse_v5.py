"""Verify existing v1/v2/v4 artifacts without model loads or metric computation."""
import ast
import hashlib
import json
import struct
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
V1 = ROOT / 'research/tsfm_peft_development_20260926'
V2 = ROOT / 'research/tsfm_peft_followup_v2_20260926'
V4 = ROOT / 'research/tsfm_peft_fresh_group_v4_20260927'
C1 = ROOT / '.cache/tsfm_peft_development_20260926'
C2 = ROOT / '.cache/tsfm_peft_followup_v2_20260926'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def relative(path):
    return absolute(path).relative_to(ROOT).as_posix()


def digest(path):
    value = hashlib.sha256()
    with absolute(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def archive_schema(path):
    result = {}
    with zipfile.ZipFile(absolute(path)) as archive:
        for name in archive.namelist():
            with archive.open(name) as stream:
                if stream.read(6) != b'\x93NUMPY':
                    raise ValueError(f'Unexpected archive entry: {path}/{name}')
                major, minor = stream.read(2)
                size_bytes = 2 if major == 1 else 4
                size = int.from_bytes(stream.read(size_bytes), 'little')
                header = ast.literal_eval(stream.read(size).decode('utf-8' if major == 3 else 'latin1'))
                result[name.removesuffix('.npy')] = {
                    'shape': list(header['shape']), 'dtype': header['descr'],
                    'fortran_order': header['fortran_order'],
                }
    return result


def receipt(path, expected=None, with_schema=False):
    path = absolute(path)
    actual = digest(path)
    if expected is not None and actual != expected:
        raise ValueError(f'Existing artifact receipt mismatch: {path}')
    result = {'path': relative(path), 'sha256': actual, 'bytes': path.stat().st_size,
              'exists': True, 'recorded_sha256_match': expected is None or actual == expected,
              'recorded_sha256_available': expected is not None}
    if with_schema:
        result['arrays'] = archive_schema(path)
    return result


def prediction(path, expected=None, period=None, rows=None):
    entry = receipt(path, expected, with_schema=True)
    if set(entry['arrays']) != {'prediction', 'origins'}:
        raise ValueError(f'Unexpected prediction schema: {path}')
    entry.update(prediction_key='prediction', origins_key='origins', period=period,
                 row_slice=rows, alignment='Match saved integer origins exactly and preserve data columns order; never align by row count alone.')
    return entry


def old_model(run, dataset, evaluation, selection):
    v2 = run.startswith('v2_')
    public = V2 if v2 else V1
    cache = C2 if v2 else C1
    path = public / 'runs' / run / 'result.json'
    result = read(path)
    entry = {
        'id': run, 'family': {'residual': 'tsfm_res', 'raw_bypass': 'tsfm_raw'}.get(result['arm'], result['arm']),
        'seed': result['seed'], 'ed_mode': result.get('ed_mode', 'current'), 'latent': result['latent'],
        'checkpoint': receipt(result['checkpoint'], result['checkpoint_sha256']),
        'result': receipt(path, selection['result_hashes'][run]),
        'selected_epoch': result['selected_epoch'], 'epochs_completed': result['epochs_completed'],
        'recipe': {k: result.get(k) for k in ['lr', 'loss', 'epochs', 'epoch_origins', 'microbatch_origins', 'weight_decay', 'residual_rank']},
        'registered_trainable_parameters': result['trainable_parameters'],
        'initial_parameter_sha256': result.get('initial_parameter_sha256'),
        'evaluation_predictions': {},
    }
    if result.get('initial_checkpoint'):
        entry['initial_checkpoint'] = receipt(result['initial_checkpoint'], result['initial_checkpoint_sha256'])
    else:
        entry['initial_checkpoint'] = None
        entry['initialization_limit'] = 'No initial.pt was saved for this v1 run. Same seed alone is not evidence of identical G initialization.'
    val = selection['validation_receipts'][run]
    entry['validation_prediction'] = prediction(val['path'], val['sha256'], period='val')
    for row in evaluation['rows']:
        if row['dataset'] == dataset and row['fit'] == run:
            p = row['prediction']
            entry['evaluation_predictions'][row['period']] = prediction(p['path'], p['sha256'], period=row['period'])
    expected_periods = {'dev'} if dataset == 'electricity' else {'e1', 'e2'}
    if set(entry['evaluation_predictions']) != expected_periods:
        raise ValueError(f'Incomplete existing evaluation mapping: {run}')
    return entry


def build_manifest():
    old_evaluation = read(V2 / 'final_development.json')
    old_selection = read(V2 / 'final_selection.json')
    old_contract = read(V1 / 'data_contract.json')
    hog_evaluation = read(V4 / 'v4_test_01.json')
    hog_contract = read(V4 / 'data_contract.json')
    hog_seal = read(V4 / 'evaluation_seal.json')
    model_source = read(V4 / 'model_source.json')
    backbone = {'revision': model_source['revision'],
                'files': {name: receipt(meta['path'], meta['sha256']) for name, meta in model_source['files'].items()}}
    manifest = {
        'schema_version': 1,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Existing artifact identity and baseline alignment only. No model load, prediction, scoring, optimization, or benchmark.',
        'path_root': 'repository root; all paths below are repository-relative',
        'backbone': backbone,
        'sources': [receipt(p) for p in [V2 / 'final_development.json', V2 / 'final_selection.json',
                                        V1 / 'data_contract.json', V4 / 'v4_test_01.json',
                                        V4 / 'data_contract.json', V4 / 'evaluation_seal.json',
                                        V4 / 'model_source.json']],
        'datasets': {},
        'evaluation_contract': {
            'prediction_units': 'TRAIN-standardized original-channel units; do not invert standardization for normalized macro MSE/MAE.',
            'target': 'x[o:o+48] with original finite/targetmask, never treat imputed future values as observed targets.',
            'input': 'x[o-512:o]',
            'axis_order': 'prediction[origin,horizon,channel]',
            'aggregation': 'Per channel sum errors and observed counts over requested origins; macro mean across channels; then mean individual seed losses, not prediction ensemble.',
            'combined_periods': 'Combine per-channel error sums and counts across periods, not an unweighted mean of period MSEs.',
            'scope': 'Hog TEST-A/B, Bull E1/E2 and Electricity DEV are exposed development data for v5.',
        },
    }
    roles = {
        'electricity': {
            'original_res': [f'revision1_rank32_residual_{s}_0.001' for s in [92601, 92602]],
            'same_variant_raw': [f'revision1_rank32_raw_bypass_{s}_0.001' for s in [92601, 92602]],
            'selected_raw': [f'revision1_rank32_raw_bypass_{s}_0.001' for s in [92601, 92602]],
            'lora': ['extended120_lora_92601_0.0001', 'initial_lora_92602_0.0001'],
            'compress_current': [f'initial_compress_{s}_0.001' for s in [92601, 92602]], 'f0': ['f0'],
        },
        'bull': {
            'original_res': [f'v2_bull_residual_fixed_ed_{s}' for s in [92601, 92602]],
            'same_variant_raw': [f'v2_bull_raw_bypass_fixed_ed_{s}' for s in [92601, 92602]],
            'selected_raw': [f'bull_final_raw_bypass_{s}_0.001' for s in [92601, 92602]],
            'lora': [f'bull_final_lora_{s}_0.0001' for s in [92601, 92602]],
            'compress_current': [f'bull_final_compress_{s}_0.001' for s in [92601, 92602]],
            'existing_k8_res': [f'v2_bull_k8_residual_fixed_ed_{s}' for s in [92601, 92602]],
            'existing_k8_raw': [f'v2_bull_k8_raw_bypass_fixed_ed_{s}' for s in [92601, 92602]], 'f0': ['f0'],
        },
    }
    for dataset, contract_key in [('electricity', 'electricity_first32'), ('bull', 'bdg2_bull_office')]:
        contract = old_contract['datasets'][contract_key]
        data_path = ROOT / contract['npz']
        data_receipt = receipt(data_path, old_evaluation['data_sha256'][dataset], with_schema=True)
        dataset_entry = {'channels': 32 if dataset == 'electricity' else 16,
                         'selected_mode': 'current' if dataset == 'electricity' else 'fixed_ed',
                         'selected_K': 8 if dataset == 'electricity' else 4,
                         'data': {'trainval': data_receipt, 'evaluation': data_receipt,
                                  'x_key': 'x', 'mask_key': 'targetmask', 'times_key': 'times', 'columns_key': 'columns',
                                  'train_start': 0 if dataset == 'electricity' else 528,
                                  'train_end': 18412 if dataset == 'electricity' else 2520,
                                  'train_bounds_source': 'split_bounds[:2]' if dataset == 'electricity' else 'searchsorted(times, TRAIN dates [2016-01-23, 2016-04-15))',
                                  'train_origins_key': 'train_origins', 'val_origins_key': 'val_origins',
                                  'evaluation_origin_keys': {'dev': 'dev_origins'} if dataset == 'electricity' else {'e1': 'eval_e1_origins', 'e2': 'eval_e2_origins'},
                                  'origin_counts': contract['origin_counts'],
                                  'columns': contract['columns'] if dataset == 'electricity' else contract['selected_columns'],
                                  'basis': {'saved_data_key': None, 'checkpoint_key': 'basis',
                                            'construction': 'train=x[train_start:train_end]; centered=train-train.mean(axis=0,keepdims=True); eigh(centered.T@centered/len(centered)); select descending last K eigenvectors; copy float32.',
                                            'source': receipt(V1 / 'model.py')}},
                         'roles': roles[dataset], 'models': {}}
        if dataset == 'bull':
            dataset_entry['data']['do_not_use_as_selected_channels'] = ['raw_all', 'targetmask_all', 'sealed_columns']
        for run in dict.fromkeys(run for ids in roles[dataset].values() for run in ids if run != 'f0'):
            dataset_entry['models'][run] = old_model(run, dataset, old_evaluation, old_selection)
        f0 = {'id': 'f0', 'family': 'f0', 'seed': None, 'ed_mode': None, 'latent': None,
              'deterministic': True, 'checkpoint': backbone['files']['model.safetensors'],
              'checkpoint_format': 'Official full backbone safetensors, not an adapter best.pt.',
              'registered_trainable_parameters': 0, 'evaluation_predictions': {}}
        for row in old_evaluation['rows']:
            if row['dataset'] == dataset and row['arm'] == 'f0':
                p = row['prediction']
                f0['evaluation_predictions'][row['period']] = prediction(p['path'], p['sha256'], period=row['period'])
        f0_val = C1 / 'revision1_rank32_evaluation/f0_val.npz'
        f0['validation_prediction'] = prediction(f0_val, period='val') if dataset == 'electricity' else None
        if dataset == 'bull':
            f0['validation_limitation'] = 'No Bull F0 VAL prediction was identified in the bounded prior evaluation receipts; development E1/E2 predictions are available.'
        dataset_entry['models']['f0'] = f0
        manifest['datasets'][dataset] = dataset_entry
    hog_data = {
        'trainval': receipt(hog_contract['trainval']['path'], hog_contract['trainval']['sha256'], with_schema=True),
        'evaluation': receipt(hog_contract['test']['path'], hog_contract['test']['sha256'], with_schema=True),
        'x_key': 'x', 'mask_key': 'finite', 'times_key': 'times', 'columns_key': 'columns',
        'train_start': 528, 'train_end': 2520, 'train_bounds_source': 'Stored train_start/train_end scalars, matching TRAIN [2016-01-23,2016-04-15).',
        'train_origins_key': 'train_origins', 'val_origins_key': 'val_origins',
        'evaluation_origin_keys': {'test_a': 'test_a_origins', 'test_b': 'test_b_origins'},
        'origin_counts': hog_contract['origins']['counts'], 'columns': hog_contract['dataset']['selected_columns'],
        'basis': {'saved_data_key': 'basis', 'checkpoint_key': 'basis', 'rank': 8,
                  'construction': 'Use stored TRAIN-only basis for the original K8 path. For a separately approved K, use the same TRAIN-only covariance-eigh algorithm.',
                  'source': receipt(V1 / 'linear_fulltrain.py')},
    }
    hog_roles = {'original_res': hog_evaluation['roles']['tsfm_res'],
                 'same_variant_raw': hog_evaluation['roles']['tsfm_raw'],
                 'selected_raw': hog_evaluation['roles']['tsfm_raw'],
                 'f0': hog_evaluation['roles']['f0'], 'lora': hog_evaluation['roles']['lora'],
                 'compress_fixed': hog_evaluation['roles']['compress']}
    hog = {'channels': 32, 'selected_mode': 'fixed_ed', 'selected_K': 8,
           'data': hog_data, 'roles': hog_roles, 'models': {}}
    sealed = {entry['id']: entry for entry in hog_seal['models']}
    for run in dict.fromkeys(run for ids in hog_roles.values() for run in ids):
        saved = sealed[run]
        result_path = absolute(saved['result_path'])
        result = read(result_path)
        spec = result['spec']
        pred = hog_evaluation['models'][run]['prediction']
        entry = {'id': run, 'family': spec['family'], 'seed': spec.get('seed'),
                 'ed_mode': spec.get('ed_mode'), 'latent': spec.get('latent'),
                 'deterministic': saved.get('deterministic', False),
                 'checkpoint': receipt(saved['checkpoint'], saved['checkpoint_sha256']),
                 'result': receipt(result_path, saved['result_sha256']),
                 'selected_epoch': result['selected_epoch'],
                 'epochs_completed': result.get('epochs_completed', 0),
                 'recipe': {k: spec.get(k) for k in ['lr', 'epochs', 'residual_rank']},
                 'registered_trainable_parameters': result['trainable_parameters'],
                 'initial_parameter_sha256': result.get('initial_parameter_sha256'),
                 'validation_prediction': prediction(saved['best_val_prediction']['path'], saved['best_val_prediction']['sha256'], period='val'),
                 'evaluation_predictions': {period: prediction(pred['path'], pred['sha256'], period, bounds)
                                            for period, bounds in [('test_a', [0, 40]), ('test_b', [40, 80])]},
                 'combined_prediction': prediction(pred['path'], pred['sha256'], 'combined', [0, 80])}
        entry['initial_checkpoint'] = receipt(result['initial_checkpoint'], result['initial_checkpoint_sha256']) if result.get('initial_checkpoint') else None
        hog['models'][run] = entry
    manifest['datasets']['hog'] = hog
    bull = manifest['datasets']['bull']
    comparisons = []
    for seed in [92601, 92602]:
        k4 = bull['models'][f'v2_bull_residual_fixed_ed_{seed}']['initial_parameter_sha256']
        k8 = bull['models'][f'v2_bull_k8_residual_fixed_ed_{seed}']['initial_parameter_sha256']
        raw = bull['models'][f'v2_bull_k8_raw_bypass_fixed_ed_{seed}']['initial_parameter_sha256']
        comparisons.append({'seed': seed, 'k8_res_raw_all_initial_hashes_equal': k8 == raw,
                            'k4_k8_g_down_equal': k4['residual.down.weight'] == k8['residual.down.weight'],
                            'k4_k8_g_up_equal': k4['residual.up.weight'] == k8['residual.up.weight']})
    manifest['existing_bull_k8_limit'] = {
        'new_fit_attempts_for_reuse': 0, 'initialization_checks': comparisons,
        'interpretation': 'Existing K8 RES versus K8 RAW is initialization-matched. K4 versus K8 does not hold random G.down initialization fixed; do not claim an isolated K-only causal comparison.',
        'historical_res_val': {k: old_selection['selection']['bull']['residual']['variant_scores'][k] for k in ['fixed_ed:k4', 'fixed_ed:k8']},
        'past_result_policy': 'Retain the prior K8 unfavorable selection result; reuse is not a new trial or retroactive approval of v2 attempt overrun.'}
    manifest['missing_optional'] = ['Bull F0 VAL prediction not established by bounded receipt lookup.',
                                    'v1 initial.pt absent for Electricity selected RES/RAW/LoRA and Bull v1 CURRENT paths.',
                                    'All-epoch checkpoints are not assumed available.']
    return manifest


if __name__ == '__main__':
    output = HERE / 'reuse_manifest.json'
    if output.exists():
        raise FileExistsError('Preserve the existing manifest; inspect before replacing it.')
    result = build_manifest()
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'path': relative(output), 'sha256': digest(output),
                      'datasets': {name: len(value['models']) for name, value in result['datasets'].items()},
                      'bytes': output.stat().st_size, 'scope': result['scope']}))
