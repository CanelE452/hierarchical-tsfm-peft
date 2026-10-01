"""Create prospective text contracts only; no numerical/model execution."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def write_new(name, value):
    with (HERE / name).open('x', encoding='utf-8') as out:
        json.dump(value, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.write('\n')


def main():
    units = {
        'robin': {'C': 17, 'K': 5, 'phase_period': 24, 'block_origins': 7, 'frequency_minutes': 60},
        'jena': {'C': 21, 'K': 6, 'phase_period': 144, 'block_origins': 42, 'frequency_minutes': 10},
        'hog': {'C': 32, 'K': 8, 'phase_period': 24, 'block_origins': 7, 'frequency_minutes': 60},
        'peacock_education': {'C': 13, 'K': 4, 'phase_period': 24, 'block_origins': 7, 'frequency_minutes': 60}}
    fits = [{'id': f'{ds}_temporal_lora_lr{token}_s{seed}', 'dataset': ds,
             'family': 'temporal_lora', 'seed': seed, 'lr': lr, 'channels': u['C'],
             'latent': u['K'], 'context': 512, 'horizon': 48, 'stride': 4}
            for seed in (92601, 92602) for ds, u in units.items()
            for token, lr in (('1e-4', 1e-4), ('1e-3', 1e-3))]
    value = {'schema': 'v16_temporal_allocation_1',
        'base_commit': 'e706fb8e1f269dfeb301912b40e480b44a578aa0',
        'scope': 'Four exposed development datasets; prospective assistant decision under human delegation',
        'units': units, 'seeds': [92601, 92602], 'families': ['temporal_lora'], 'fits': fits,
        'model': {'stride': 4, 'outer_context': 512, 'outer_horizon': 48,
                  'inner_context': 128, 'inner_horizon_crop': 12, 'native_output_length': 64,
                  'formula': 'repeat4(F_phi(avg4(X))[:12]) + S(X) - repeat4(avg4(S(X)))',
                  'trainable': 'standard r8 alpha16 q,v LoRA only', 'direct_parent_frozen': True},
        'training': {'batch_size': 4, 'origins_per_epoch': 512, 'max_epochs': 120,
                     'early_stopping_patience': 6, 'weight_decay': 0, 'clip_norm': 1,
                     'scheduler_factor': .5, 'scheduler_patience': 2,
                     'scheduler_relative_threshold': 1e-4},
        'selection': {'metric': 'whole VAL observed channel-macro MSE, mean two seed losses',
                      'lr_candidates': [1e-4, 1e-3], 'step_zero': True,
                      'lr_exact_tie': 1e-4, 'epoch_exact_tie': 'earliest', 'test_use': False},
        'parity': {'atol': 1e-5, 'rtol': 1e-4, 'replay_atol': 1e-6, 'replay_rtol': 0},
        'evaluation': {'instances_per_dataset': 15, 'block_resamples': 2000, 'block_seed': 9262026},
        'cost': {'blocks': 3, 'warmups': 10, 'passes': 3, 'origins': 24,
                 'reserve_seconds': 3600, 'cpu_threads': 4, 'tf32': False,
                 'maximum_gpu_rows': 588, 'maximum_cpu_rows': 48,
                 'scope': 'standardized CPU input to complete CPU output; online forecasts; same-session'},
        'limits': {'real_fit': 20, 'basic_neural_fits': 16, 'technical_reserve': 4,
                   'gpu_seconds': 14400, 'cpu_analysis_seconds': 1800, 'cpu_check_seconds': 900,
                   'storage_bytes': 5 * 1024**3, 'downloads': 0,
                   'synthetic_optimizer_sessions': 4, 'synthetic_updates_per_session': 10}}
    write_new('protocol.json', value)
    write_new('provenance.json', {'files': {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                         for name in ('PLAN.md', 'AUTHORIZATION.txt', 'protocol.json')}})
    write_new('ledger.json', {'schema': 'v16_job_ledger_1', 'jobs': []})
    print('Prospective v16 contracts created; numerical work has not started.')


if __name__ == '__main__':
    main()
