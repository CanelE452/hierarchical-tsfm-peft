"""Bind adopted request, parent contracts and protected state without fitting."""
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from runtime_v18 import (CACHE, DATASETS, HERE, LIMITS, ROOT, SEEDS, array_hash,
                         artifact, digest, read_json, resolve, save_json)


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True, encoding='utf-8').strip()


def main():
    if (HERE / 'initial_manifest.json').exists():
        raise FileExistsError('Preparation already recorded; verify and reuse')
    old = ROOT / 'research/level_bolt_backbone_scaling_v1_20261006'
    inputs = read_json(old / 'input_contract.json')
    reuse = read_json(old / 'reuse_manifest.json')
    protected = {}
    candidates = set(git('ls-files').splitlines()) | set(git('ls-files', '--others', '--exclude-standard').splitlines())
    for label in sorted(candidates):
        path = ROOT / label
        if path.is_file() and not path.resolve().is_relative_to(HERE):
            protected[label] = {'sha256': digest(path), 'bytes': path.stat().st_size}
    save_json(CACHE / 'protected_workspace_before.json', protected)
    manifest = {'schema': 'matched_learned_ed_parent_reuse_v18', 'units': {},
                'parent_contract': artifact(old / 'input_contract.json'),
                'parent_baselines': artifact(old / 'reuse_manifest.json'), 'created_utc': time.time()}
    for dataset in DATASETS:
        unit = inputs['units'][dataset]
        for key in ('trainval', 'test', 'source_contract'):
            artifact(unit[key]['path'], unit[key]['sha256'])
        with np.load(resolve(unit['trainval']['path']), allow_pickle=False) as archive:
            array_receipts = {key: {'shape': list(archive[key].shape), 'dtype': str(archive[key].dtype),
                                  'sha256': array_hash(archive[key])} for key in archive.files}
        contract = {**unit, 'reuse': 'Literal parent bytes; no new PCA, split, standardization, imputation or channel order',
                    'trainval_array_receipts': array_receipts, 'phase_period': 144 if dataset=='jena' else 24,
                    'parent_generation': {'robin': 'v6', 'peacock_education': 'v12', 'jena': 'v7'}[dataset]}
        save_json(HERE / ('data_contract_' + dataset + '.json'), contract)
        levels = [row for row in reuse['units'][dataset]['rows'] if row['method']=='LEVEL']
        if [row['seed'] for row in levels] != list(SEEDS):
            raise ValueError('Original selected LEVEL seed order differs')
        for row in levels:
            for key in ('checkpoint', 'test_prediction', 'restore_module_artifact', 'source_result'):
                artifact(row[key]['path'], row[key]['sha256'])
        manifest['units'][dataset] = {'levels': levels, 'basis': array_receipts['basis'],
                                     'source_data_contract': artifact(HERE / ('data_contract_' + dataset + '.json'))}
    save_json(HERE / 'parent_reuse_manifest.json', manifest)
    protocol = {
        'schema': 'matched_learned_ed_protocol_v18', 'created_utc': time.time(),
        'request': artifact(HERE / 'REQUEST.txt'), 'method': 'MATCHED_LEARNED_ED',
        'full_AdaPTS_reproduction': False, 'datasets': list(DATASETS), 'seeds': list(SEEDS),
        'learning_rates': [1e-3, 1e-4], 'max_epochs': 120, 'patience': 6,
        'epoch_origins': 512, 'effective_batch_origins': 4, 'initial_microbatch': 4,
        'oom_microbatch_sequence': [4,2,1], 'optimizer': 'AdamW', 'weight_decay': 0,
        'clip_grad_norm': 1., 'scheduler': {'class': 'ReduceLROnPlateau', 'factor': .5, 'patience': 2,
                                        'threshold': 1e-4, 'threshold_mode': 'rel', 'step_after_epoch_val': True},
        'loss': 'Observed-mask channel-macro forecast MSE; no auxiliary reconstruction',
        'checkpoint_selection': 'Minimum full canonical VAL MSE including step0; strict improvement, earliest exact tie',
        'lr_selection': 'Minimum arithmetic mean of two selected seed VAL losses before any new TEST',
        'lr_exact_tie': {'robin': 1e-3, 'peacock_education': 1e-4, 'jena': 1e-3},
        'lr_tie_source': 'Peacock v12 min(mean,lr); Jena v7 first-listed; Robin no historical LR search uses first-listed',
        'sampling': 'Literal parent phase_sample SeedSequence([seed,epoch]); dataset phase periods retained',
        'model': {'encoder_bias': False, 'decoder_bias': False, 'E_initial': 'literal U.T', 'D_initial': 'literal U',
                  'backbone': 'Parent frozen Bolt-small revision; eval; autograd through operations retained',
                  'trainable': ['encoder.weight','decoder.weight'], 'residual_G': False, 'level_persistence': False},
        'fp32': True, 'tf32': False, 'step0_parity': {'atol': 1e-5, 'rtol': 1e-4,
                        'source': 'Existing Bolt preflight BATCH tolerance, all canonical VAL; no tolerance retuning'},
        'new_test_predictions': 6, 'joint_selection_seal_before_test': True, 'prediction_ensemble': False,
        'aggregation': 'A/B channel error sums/counts pooled before macro; mean selected seed losses; no cross-dataset total',
        'bootstrap': 'Exact prior paired moving-block draws/seed/blocks from data_contract files; A/B stratified',
        'training_cost_probe': {'batch_origins': 4, 'fixed_origins': 'First four chronological canonical TRAIN origins',
                      'warmup': 10, 'repetitions': 10, 'rotated_blocks': 3, 'optimizer_updates': 0,
                      'members': 'Both selected ED and original LEVEL seeds; same bytes; forward+MSE+backward'},
        'inference_cost': {'origins': 'First 24 chronological canonical VAL', 'batches': [1,4],
                          'warmup': 10, 'passes': 10, 'rotated_blocks': 3, 'output': 'Standardized CPU input to B48C CPU output'},
        'cost_interpretation': {'similar_operational_max_ratio_both_time_and_allocated': 1.25,
                               'substantially_higher_operational_either_ratio': 2.,
                               'intermediate_or_overlapping_direction': 'Conservative mixed result; show continuous ratios/raw ranges',
                               'not_equivalence_margin': True, 'not_convergence_time': True},
        'diagnostics_selection_use': False, 'limits': LIMITS,
        'preflight_estimate': 'Fixed forward/backward steps with zero optimizer updates; full 12x120epoch upper bound plus evaluation/cost',
        'test_exposure': 'Already exposed follow-up controls; not independent confirmation',
        'unplanned_followups': False,
    }
    save_json(HERE / 'protocol.json', protocol)
    save_json(HERE / 'ledger.json', {'schema': 'matched_learned_ed_ledger_v18', 'limits': LIMITS,
                 'jobs': [], 'fits': [], 'test_predictions': [], 'created_utc': time.time()})
    save_json(HERE / 'initial_manifest.json', {'branch': git('branch','--show-current'),
                 'head': git('rev-parse','HEAD'), 'live_origin_main': git('ls-remote','origin','refs/heads/main').split()[0],
                 'dirty_before': git('status','--short'), 'request': artifact(HERE / 'REQUEST.txt'),
                 'protected_file_count': len(protected), 'protected_receipt': artifact(CACHE / 'protected_workspace_before.json'),
                 'preparation_fit_count': 0, 'preparation_gpu_count': 0, 'created_utc': time.time()})
    print(json.dumps({'prepared': list(DATASETS), 'protected_files': len(protected), 'new_fits': 0}))


if __name__=='__main__':
    main()
