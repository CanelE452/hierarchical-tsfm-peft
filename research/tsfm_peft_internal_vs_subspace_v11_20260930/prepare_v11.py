"""Create pinned v11 manifests and reservations without model execution."""
import argparse
import ast
import copy
from pathlib import Path
import struct
import subprocess
import time
import zipfile

from runtime_v11 import (HERE, ROOT, CACHE, DATASETS, SEEDS, LIMITS, Job, artifact,
                         digest, environment_receipt, read_json, resolve_path, save_json)

V4 = ROOT / 'research/tsfm_peft_fresh_group_v4_20260927'
V5 = ROOT / 'research/tsfm_peft_accuracy_recovery_v5_20260927'
V6 = ROOT / 'research/tsfm_peft_level_confirmation_v6_20260928'
V7 = ROOT / 'research/tsfm_peft_practical_controls_v7_20260928'
V9 = ROOT / 'research/tsfm_peft_staged_p_v9_20260929'
CONTRACTS = {'robin': V9 / 'data_contract_robin.json', 'jena': V9 / 'data_contract_jena.json',
             'hog': V4 / 'data_contract.json'}
PARENT_DIRS = {'robin': V6, 'jena': V7, 'hog': V5}
BASELINE_DIRS = {'robin': V6, 'jena': V7, 'hog': V4}
PREFIX = {'robin': 'v6_robin', 'jena': 'v7_jena', 'hog': 'v4_hog'}


def protocol():
    neural = []
    def add(dataset, family, seed, lr):
        channels, latent, phase = DATASETS[dataset]
        run_id = f'v11_{dataset}_{family}_lr{lr:g}_{seed}'
        neural.append({'id': run_id, 'fit_id': run_id, 'dataset': dataset, 'family': family, 'seed': seed,
                       'lr': lr, 'channels': channels, 'latent': latent, 'epochs': 120,
                       'phase_period': phase, 'batch_size': 4, 'context': 512, 'horizon': 48})
    for seed in SEEDS:
        for lr in (1e-4, 1e-3):
            add('hog', 'direct_nlinear', seed, lr)
    for seed in SEEDS:
        for lr in (1e-4, 1e-3):
            for dataset in DATASETS:
                for family in ('a_p_lora_qfixed', 'full_lora_mse', 'b_learned_u'):
                    add(dataset, family, seed, lr)
    coefficient = [{'id': f'v11_{d}_{kind}_gamma_{s}', 'dataset': d, 'u_kind': kind, 'seed': s}
                   for d in DATASETS for kind in ('fixed', 'learned') for s in SEEDS]
    return {'schema_version': 1, 'stage': 'v11_internal_vs_subspace', 'datasets': DATASETS,
            'scope': 'All three are exposed development evaluations, not independent confirmation',
            'neural_fits': neural, 'fits': neural, 'coefficient_fits': coefficient, 'limits': LIMITS,
            'neural_recipe': {'optimizer': 'AdamW', 'weight_decay': 0, 'clip_norm': 1,
                'train_origins_per_epoch': 512, 'max_epochs': 120, 'strict_nonimprovement': 6,
                'scheduler': {'type': 'ReduceLROnPlateau', 'factor': .5, 'patience': 2,
                              'threshold': 1e-4, 'threshold_mode': 'rel'},
                'selection': 'step0 included; minimum full VAL macro MSE; earliest exact epoch tie',
                'lr_selection': 'minimum two-seed mean best VAL; exact tie prefers 1e-4',
                'automatic_epoch_extension': False, 'train_probe_origins': {'robin': 96, 'jena': 144, 'hog': 96}},
            'lora': {'r': 8, 'alpha': 16, 'dropout': 0, 'targets': ['q', 'v'], 'bias': 'none',
                     'expected_registered_parameters': 294912},
            'orthogonal': {'map': 'householder', 'use_trivialization': True,
                           'tied_encode_decode': True, 'gamma_during_u_training': 'identity'},
            'gamma': {'solver': 'scipy.optimize.lsq_linear', 'method': 'bvls', 'lsq_solver': 'exact',
                      'bounds': [0, 1], 'tol': 1e-10, 'max_iter': 100, 'fit_dtype': 'float64',
                      'export_dtype': 'float32', 'weights': '1/(C*N_c) over observed VAL rows',
                      'new_penalty': None, 'prune_only_exact_zero': True},
            'evaluation': {'main_metric': 'observed_channel_macro_MSE', 'secondary': ['MAE', 'signed_mean_error'],
                           'seed_aggregation': 'mean seed losses, not ensemble', 'all_selections_before_test': True,
                           'bootstrap': {'resamples': 2000, 'seed': 9262026, 'block_origins': {'robin': 7, 'jena': 42, 'hog': 7}}},
            'parity': {'atol': 1e-5, 'rtol': 1e-4, 'replay_atol': 1e-6, 'replay_rtol': 0, 'gram_maxabs': 1e-5},
            'cost': {'val_origins': 'chronological first 24 existing VAL origins', 'precision': 'float32',
                     'tf32': False, 'cpu_threads': 4, 'warmup_wrapper_calls': 10, 'passes': 10, 'blocks': 3,
                     'batch': [1, 4], 'uncompressed_chunk_grid': {'1': ['C', 'K'], '4': ['4C', '4K', 'K']},
                     'compressed_chunk_grid': {'1': ['K'], '4': ['4K', 'K']},
                     'gpu_rows_max_with_sentinels': 621, 'direct_cpu_rows': 36,
                     'boundary': 'standardized CPU input through complete online forecast to all-channel CPU output',
                     'profiler_max_gpu_seconds': 600, 'profiler_max_sessions': 1},
            'excluded': ['A+B', 'B LoRA', 'A Q/ED retraining', 'new rank/K/data/loss/gate',
                         'QLoRA', 'DoRA', 'compile', 'quantization', 'distillation', 'new backbone']}


def npz_headers(path):
    headers = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith('.npy'):
                raise ValueError('Unexpected NPZ member: ' + name)
            with archive.open(name) as handle:
                if handle.read(6) != b'\x93NUMPY':
                    raise ValueError('Invalid NPY magic')
                major, minor = handle.read(2)
                length_bytes = 2 if major == 1 else 4
                length = struct.unpack('<H' if length_bytes == 2 else '<I', handle.read(length_bytes))[0]
                header = ast.literal_eval(handle.read(length).decode('utf-8' if major == 3 else 'latin1'))
                headers[name[:-4]] = {'shape': list(header['shape']), 'dtype': header['descr'],
                                      'fortran_order': header['fortran_order']}
    return headers


def prior_prediction(evaluation, model_id):
    payload = read_json(evaluation)
    if 'entries' in payload:
        row = next(row for row in payload['entries'] if row['id'] == model_id)
        entry = artifact(row['predictions']['combined'], row['prediction_sha256'])
    else:
        row = payload['models'][model_id]
        entry = artifact(row['prediction']['path'], row['prediction']['sha256'])
    entry['source_evaluation'] = artifact(evaluation)
    entry['array_headers'] = npz_headers(entry['path'])
    return entry


def run_receipt(folder, run_id, prediction=None):
    result_path = folder / 'runs' / run_id / 'result.json'
    result = read_json(result_path)
    if result['status'] != 'complete':
        raise RuntimeError('Reused run is not complete: ' + run_id)
    record = {'id': run_id, 'spec': result['spec'], 'result': artifact(result_path),
              'checkpoint': artifact(result['checkpoint'], result['checkpoint_sha256']),
              'best_val_mse': result.get('best_val_mse'), 'selected_epoch': result.get('selected_epoch'),
              'executed_source_hashes': result.get('source', {}),
              'data_sha256': result.get('data_sha256')}
    if result.get('best_val_prediction'):
        val = result['best_val_prediction']
        record['best_val_prediction'] = artifact(val['path'], val['sha256'])
    if result.get('initial_source'):
        initial = copy.deepcopy(result['initial_source'])
        initial.update(artifact(initial['path'], initial['sha256']))
        record['initial_source'] = initial
    if result.get('initial_checkpoint'):
        record['initial_checkpoint'] = artifact(result['initial_checkpoint'], result['initial_checkpoint_sha256'])
    if prediction:
        record['test_prediction'] = prior_prediction(prediction, run_id)
    return record


def build_reuse(job):
    manifest = {'schema_version': 1, 'scope': 'bytes/schema read only; no model execution or new scoring',
                'created_utc': time.time(), 'datasets': {}, 'parents': {}, 'direct': {}, 'baselines': {}, 'sources': {},
                'download_bytes': 0, 'new_predictive_fits': 0, 'environment': environment_receipt()}
    for dataset, source in CONTRACTS.items():
        contract = read_json(source)
        destination = HERE / f'data_contract_{dataset}.json'
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise RuntimeError('Refusing to replace nonidentical existing data contract')
        destination.write_bytes(source.read_bytes())
        data = {'contract': artifact(destination), 'source_contract': artifact(source), 'splits': contract.get('splits'),
                'dimensions': {'channels': DATASETS[dataset][0], 'latent': DATASETS[dataset][1], 'phase': DATASETS[dataset][2]}}
        for kind in ('trainval', 'test'):
            entry = contract[kind]
            data[kind] = artifact(entry['path'], entry['sha256'])
            data[kind]['array_headers'] = npz_headers(data[kind]['path'])
        manifest['datasets'][dataset] = data
        manifest['parents'][dataset], manifest['direct'][dataset], manifest['sources'][dataset] = {}, {}, {}
        manifest['baselines'][dataset] = {'roles': {'base_level': [], 'direct_nlinear': [], 'f0': [], 'lora_native': []},
                                          'predictions': {}, 'runs': {}}
        baseline = manifest['baselines'][dataset]
        parent_prefix = {'robin': 'v6_robin', 'jena': 'v7_jena', 'hog': 'v5_hog'}[dataset]
        parent_eval = {'robin': V6 / 'evaluation01.json', 'jena': V7 / 'jena_eval01.json',
                       'hog': V5 / 'hog_level_01_predictions.json'}[dataset]
        native_eval = {'robin': V6 / 'evaluation01.json', 'jena': V7 / 'jena_eval01.json',
                       'hog': V4 / 'v4_test_01.json'}[dataset]
        parent_source = PARENT_DIRS[dataset] / {'robin': 'model_v6.py', 'jena': 'model_v7.py', 'hog': 'model_v5.py'}[dataset]
        for seed in SEEDS:
            parent_id = f'{parent_prefix}_level_res_{seed}'
            parent = run_receipt(PARENT_DIRS[dataset], parent_id, parent_eval)
            if parent['spec'].get('ed_mode') != 'fixed_ed':
                raise RuntimeError('Parent is not fixed PCA LEVEL')
            if dataset == 'hog' and (parent['spec'].get('method'), parent['spec'].get('variant')) != ('level', 'res'):
                raise RuntimeError('Hog parent is not learned LEVEL residual')
            if parent['data_sha256'] != data['trainval']['sha256']:
                raise RuntimeError('Parent training data differs from approved arrays')
            parent['current_restore_source'] = artifact(parent_source)
            manifest['parents'][dataset][str(seed)] = parent
            native_id = f'{PREFIX[dataset]}_lora_{seed}'
            native = run_receipt(BASELINE_DIRS[dataset], native_id, native_eval)
            sources = {'level_checkpoint': parent['checkpoint']['path'],
                       'level_checkpoint_sha256': parent['checkpoint']['sha256'],
                       'native_checkpoint': native['checkpoint']['path'],
                       'native_checkpoint_sha256': native['checkpoint']['sha256']}
            for role, record in (('base_level', parent), ('lora_native', native)):
                baseline['roles'][role].append(record['id'])
                baseline['runs'][record['id']] = record
                baseline['predictions'][record['id']] = {'role': role, 'seed': seed, 'prediction': record['test_prediction']}
            if dataset != 'hog':
                selection_path = V7 / f'selected_{dataset}.json'
                selected = read_json(selection_path)['selected']['direct_nlinear']
                direct_id = next(run_id for run_id in selected['run_ids'] if run_id.endswith('_' + str(seed)))
                direct_eval = V7 / f'{dataset}_eval01.json'
                direct = run_receipt(V7, direct_id, direct_eval)
                if direct['spec']['family'] != 'direct_nlinear' or direct['spec']['dataset'] != dataset:
                    raise RuntimeError('S source is not raw-channel direct NLinear')
                direct['selection_record'] = artifact(selection_path)
                direct['selection'] = selected
                direct['current_restore_source'] = artifact(V7 / 'model_v7.py')
                manifest['direct'][dataset][str(seed)] = direct
                sources.update(direct_checkpoint=direct['checkpoint']['path'], direct_checkpoint_sha256=direct['checkpoint']['sha256'])
                baseline['roles']['direct_nlinear'].append(direct_id)
                baseline['runs'][direct_id] = direct
                baseline['predictions'][direct_id] = {'role': 'direct_nlinear', 'seed': seed, 'prediction': direct['test_prediction']}
            manifest['sources'][dataset][str(seed)] = sources
        f0_id = f'{PREFIX[dataset]}_f0'
        f0 = run_receipt(BASELINE_DIRS[dataset], f0_id, native_eval)
        baseline['roles']['f0'].append(f0_id)
        baseline['runs'][f0_id] = f0
        baseline['predictions'][f0_id] = {'role': 'f0', 'seed': None, 'prediction': f0['test_prediction']}
        job.heartbeat({'stage': 'reuse', 'dataset': dataset, 'model_execution': False})
    model_source = V6 / 'model_source.json'
    backbone = read_json(model_source)
    manifest['backbone'] = {'revision': backbone['revision'], 'snapshot': backbone['snapshot'],
                            'source_receipt': artifact(model_source),
                            'files': {name: artifact(row['path'], row['sha256']) for name, row in backbone['files'].items()}}
    manifest['direct']['hog'] = {'status': 'new_four_fit_required', 'reason': 'No equivalent stored raw-channel DIRECT_NLINEAR in approved reuse records'}
    return manifest


def initialize():
    for name in ('PLAN.md', 'APPROVAL.txt', 'REQUEST_PLAN.txt'):
        if not (HERE / name).is_file():
            raise FileNotFoundError('Root must preserve approved verbatim text first: ' + name)
    for name in ('protocol.json', 'provenance.json', 'ledger.json'):
        if (HERE / name).exists():
            raise RuntimeError('Already initialized; preserve existing state and inspect: ' + name)
    CACHE.mkdir(parents=True, exist_ok=True)
    save_json(HERE / 'protocol.json', protocol())
    save_json(HERE / 'provenance.json', {'plan_sha256': digest(HERE / 'PLAN.md'),
        'approval_sha256': digest(HERE / 'APPROVAL.txt'), 'request_sha256': digest(HERE / 'REQUEST_PLAN.txt'),
        'protocol_sha256': digest(HERE / 'protocol.json'), 'created_utc': time.time(),
        'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'planned_base_commit': 'efc029f8aae619b499071967d6b4764cb55fb802',
        'base_tree': subprocess.check_output(['git', 'rev-parse', 'HEAD^{tree}'], cwd=ROOT, text=True).strip(),
        'environment': environment_receipt()})
    save_json(HERE / 'ledger.json', {'schema_version': 1, 'limits': LIMITS, 'jobs': [],
                                    'created_utc': time.time(), 'wait_events': [],
                                    'accounting': 'Parent resource jobs count time; nested fits count attempts only'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--initialize', action='store_true')
    parser.add_argument('--job-label', default='prepare_reuse01')
    args = parser.parse_args()
    if args.initialize:
        initialize()
    if (HERE / 'reuse_manifest.json').exists():
        raise RuntimeError('Reuse manifest already exists; do not silently overwrite')
    with Job(args.job_label, category='cpu_check', reserve_seconds=120,
             metadata={'purpose': 'verify existing bytes and write fixed reuse/protocol contracts; no forward/score'} ) as job:
        parsed = {}
        for path in sorted(HERE.glob('*.py')):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            parsed[path.name] = digest(path)
        save_json(HERE / 'prepare_static.json', {'status': 'pass', 'job_id': job.id,
                  'ast_parsed': parsed, 'model_forward': False, 'optimizer_updates': 0})
        manifest = build_reuse(job)
        save_json(HERE / 'reuse_manifest.json', manifest)
        save_json(HERE / 'model_source.json', manifest['backbone'])
        provenance = read_json(HERE / 'provenance.json')
        provenance['reuse_manifest_sha256'] = digest(HERE / 'reuse_manifest.json')
        save_json(HERE / 'provenance.json', provenance)
    print('V11 prepared: 40 neural + 12 coefficient basic fits; no model execution or new scoring', flush=True)


if __name__ == '__main__':
    main()
