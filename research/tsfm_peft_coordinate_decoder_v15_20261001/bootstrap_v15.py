"""Prospective text contracts and an empty ledger, without numerical execution."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def write_new(name, value):
    path = HERE / name
    with path.open('x', encoding='utf-8') as out:
        json.dump(value, out, indent=2, ensure_ascii=False, allow_nan=False)
        out.write('\n')


def main():
    units = {
        'robin': {'C': 17, 'K': 5, 'phase_period': 24, 'block_origins': 7, 'frequency_minutes': 60},
        'jena': {'C': 21, 'K': 6, 'phase_period': 144, 'block_origins': 42, 'frequency_minutes': 10},
        'hog': {'C': 32, 'K': 8, 'phase_period': 24, 'block_origins': 7, 'frequency_minutes': 60},
        'peacock_education': {'C': 13, 'K': 4, 'phase_period': 24, 'block_origins': 7, 'frequency_minutes': 60}}
    fits = [{'id': f'{ds}_{family}_lam{token}_s{seed}', 'dataset': ds, 'family': family,
             'seed': seed, 'lambda': penalty, 'channels': unit['C'], 'latent': unit['K']}
            for ds, unit in units.items() for family in ('raw_free', 'pca_free')
            for seed in (92601, 92602) for token, penalty in (('0', 0.), ('0p001', .001), ('0p1', .1))]
    value = {'schema': 'v15_coordinate_decoder_1',
        'base_commit': '886cda62e94070b6f3ec2bf07d1092f4d5c9c15f',
        'scope': 'Four exposed development datasets; prospective assistant decision under human delegation',
        'units': units, 'seeds': [92601, 92602], 'families': ['raw_free', 'pca_free'], 'fits': fits,
        'selection': {'metric': 'observed channel-macro MSE, mean seed losses',
                      'candidates': ['parent', 0., .001, .1], 'exact_ties': 'candidate order',
                      'test_use': False},
        'parity': {'atol': 1e-5, 'rtol': 1e-4, 'replay_atol': 1e-6, 'replay_rtol': 0},
        'cost': {'blocks': 3, 'warmups': 10, 'passes': 3, 'origins': 24,
                 'reserve_seconds': 2400, 'cpu_threads': 4, 'tf32': False,
                 'maximum_gpu_rows': 612, 'maximum_cpu_rows': 48,
                 'scope': 'standardized CPU input to complete CPU output; online forecasts; same-session'},
        'limits': {'real_fit': 52, 'basic_coefficient_fits': 48, 'technical_reserve': 4,
                   'gpu_seconds': 3600, 'cpu_analysis_seconds': 1800, 'cpu_check_seconds': 600,
                   'storage_bytes': 2147483648, 'downloads': 0, 'optimizer_sessions': 0}}
    write_new('protocol.json', value)
    write_new('provenance.json', {'files': {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
                                         for name in ('PLAN.md', 'AUTHORIZATION.txt', 'protocol.json')}})
    write_new('ledger.json', {'schema': 'v15_job_ledger_1', 'jobs': []})
    print('Prospective protocol/provenance/empty ledger created; computation not started.')


if __name__ == '__main__':
    main()
