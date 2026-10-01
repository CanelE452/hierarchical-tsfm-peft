"""Verify and bind existing arrays/parents; do not refit or infer."""
import argparse
import copy
from runtime_v16 import (HERE, ROOT, DATASETS, SEEDS, Job, artifact, load_data,
                         read_json, save_json, environment_receipt)

V15 = ROOT / 'research/tsfm_peft_coordinate_decoder_v15_20261001'
V11 = ROOT / 'research/tsfm_peft_internal_vs_subspace_v11_20260930'
V12 = ROOT / 'research/tsfm_peft_a_confirmation_v12_20261001'


def prepare(job):
    for name in ('data_manifest.json', 'reuse_manifest.json', 'initial_manifest.json'):
        if (HERE / name).exists():
            raise FileExistsError('Preparation already exists: ' + name)
    data = copy.deepcopy(read_json(V15 / 'data_manifest.json'))
    reuse = copy.deepcopy(read_json(V15 / 'reuse_manifest.json'))
    data.update(schema='v16_data_manifest_1', source_v15=artifact(V15 / 'data_manifest.json'))
    reuse.update(schema='v16_reuse_manifest_1', source_v15=artifact(V15 / 'reuse_manifest.json'))
    refs = ('direct', 'a', 'f0', 'full_mse', 'native')
    reuse['reference_families'] = list(refs)
    initial = {'schema': 'v16_historical_initial_lora_1', 'datasets': {}}
    for ds in DATASETS:
        entry = data['datasets'][ds]
        for split in ('trainval', 'test'):
            artifact(entry[split]['path'], entry[split]['sha256'])
        rows = [row for row in reuse['datasets'][ds]['references'] if row['family'] in refs]
        if len(rows) != 9:
            raise ValueError('Every dataset requires nine preserved references')
        for row in rows:
            for key in ('checkpoint', 'prediction', 'source_evaluation'):
                if row.get(key):
                    artifact(row[key]['path'], row[key]['sha256'])
        reuse['datasets'][ds]['references'] = rows
        reuse['datasets'][ds]['reference_family_counts'] = {f: sum(r['family'] == f for r in rows) for f in refs}
        initial['datasets'][ds] = {}
        for seed in SEEDS:
            folder = V12 / 'confirmation_initial' if ds == 'peacock_education' else V11 / 'initial'
            path = folder / ds / f'lora_{seed}.json'
            row = read_json(path)
            initial['datasets'][ds][str(seed)] = {'receipt': artifact(path), 'source': row}
        job.heartbeat('Verified existing data, references and initial records: ' + ds)
    save_json(HERE / 'data_manifest.json', data)
    save_json(HERE / 'reuse_manifest.json', reuse)
    save_json(HERE / 'initial_manifest.json', initial)
    shapes = {}
    for ds in DATASETS:
        arrays = load_data(ds)
        shapes[ds] = {'C': int(arrays['x'].shape[1]), 'train_origins': len(arrays['train_origins']),
                      'val_origins': len(arrays['val_origins']), 'phase_period': DATASETS[ds][2],
                      'same_origin_contract': True, 'new_statistics_fitted': False}
    save_json(HERE / 'prepare_receipt.json', {'data': artifact(HERE / 'data_manifest.json'),
        'reuse': artifact(HERE / 'reuse_manifest.json'), 'initial': artifact(HERE / 'initial_manifest.json'),
        'shapes': shapes, 'environment': environment_receipt(),
        'new_model_fits': 0, 'new_predictions': 0, 'test_statistics_examined': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--label', required=True)
    p.add_argument('--reserve-s', type=float, default=90)
    a = p.parse_args()
    with Job('cpu_analysis', a.label, reserve_s=a.reserve_s,
             metadata={'purpose': 'Verify existing bytes and TRAIN/VAL shapes only'}) as job:
        prepare(job)
    print('V16 existing data/parent/initial references prepared; no prediction or fitting.')
