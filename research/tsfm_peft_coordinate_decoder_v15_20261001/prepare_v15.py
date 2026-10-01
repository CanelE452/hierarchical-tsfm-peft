"""Bind existing data/parents and the TRAIN-PCA-only coordinate selector."""
import argparse
import copy
from runtime_v15 import HERE, ROOT, DATASETS, SEEDS, Job, artifact, load_data, read_json, save_json, environment_receipt

V14 = ROOT / 'research/tsfm_peft_nonlinear_correction_v14_20261001'


def prepare(job):
    import numpy as np
    from scipy.linalg import qr
    for name in ('data_manifest.json', 'reuse_manifest.json', 'selector_manifest.json'):
        if (HERE / name).exists():
            raise FileExistsError('Preparation already exists: ' + name)
    data = copy.deepcopy(read_json(V14 / 'data_manifest.json'))
    reuse = copy.deepcopy(read_json(V14 / 'reuse_manifest.json'))
    data.update(schema='v15_data_manifest_1', source_v14=artifact(V14 / 'data_manifest.json'))
    reuse.update(schema='v15_reuse_manifest_1', source_v14=artifact(V14 / 'reuse_manifest.json'))
    refs = ('direct', 'pca_level', 'a', 'f0', 'full_mse', 'native')
    reuse['reference_families'] = list(refs)
    parents = read_json(V14 / 'parent_cache.json')
    for ds in DATASETS:
        entry = data['datasets'][ds]
        for split in ('trainval', 'test'):
            artifact(entry[split]['path'], entry[split]['sha256'])
        entry['parent_cache'] = {'source': artifact(V14 / 'parent_cache.json'), 'role': 'direct',
                                 'scope': 'Existing full TRAIN/VAL DIRECT predictions reused by checked origin/hash'}
        entry.pop('initial_g_recipe_ref', None)
        rows = [row for row in reuse['datasets'][ds]['references'] if row['family'] in refs]
        if len(rows) != 11:
            raise ValueError('Every dataset requires eleven preserved references')
        for row in rows:
            for key in ('checkpoint', 'prediction', 'source_evaluation'):
                if row.get(key):
                    artifact(row[key]['path'], row[key]['sha256'])
        reuse['datasets'][ds]['references'] = rows
        reuse['datasets'][ds]['reference_family_counts'] = {f: sum(r['family'] == f for r in rows) for f in refs}
        job.heartbeat('Verified old data/reference files: ' + ds)
    save_json(HERE / 'data_manifest.json', data)
    save_json(HERE / 'reuse_manifest.json', reuse)
    out = {'schema': 'v15_train_pca_pivot_selection_1', 'datasets': {},
           'selection': 'FP64 scipy.linalg.qr(U0.T, mode=economic, pivoting=True), first K pivots',
           'source_parent_cache': artifact(V14 / 'parent_cache.json'), 'environment': environment_receipt()}
    for ds, (c, k, period) in DATASETS.items():
        arrays = load_data(ds)
        u = arrays['pca_basis'].astype(np.float64)
        _, r, pivots = qr(u.T, mode='economic', pivoting=True)
        indices = pivots[:k].astype(int)
        j = np.eye(c, dtype=np.float64)[:, indices]
        a = u.T @ j
        singular = np.linalg.svd(a, compute_uv=False)
        rank = int(np.linalg.matrix_rank(a))
        w = np.linalg.solve(a, u.T) if rank == k else None
        out['datasets'][ds] = {'C': c, 'K': k, 'indices': indices.tolist(),
            'columns': [data['datasets'][ds]['columns'][i] for i in indices],
            'E_raw': j.astype(np.float32).tolist(), 'E_pca': u.astype(np.float32).tolist(),
            'W_interp': None if w is None else w.astype(np.float32).tolist(),
            'interpolation_rank': rank, 'singular_values': singular.tolist(),
            'condition': float(np.linalg.cond(a)), 'WJ_identity_maxabs_fp64': None if w is None else float(np.max(np.abs(w @ j - np.eye(k)))),
            'pca_gram_maxabs': float(np.max(np.abs(u.T @ u - np.eye(k)))),
            'source': data['datasets'][ds]['trainval'],
            'train_origins': len(arrays['train_origins']), 'val_origins': len(arrays['val_origins']),
            'target_filling': False, 'performance_used_for_selection': False}
        job.heartbeat('TRAIN-only selector fixed: ' + ds)
    save_json(HERE / 'selector_manifest.json', out)
    save_json(HERE / 'prepare_receipt.json', {'data': artifact(HERE / 'data_manifest.json'),
        'reuse': artifact(HERE / 'reuse_manifest.json'), 'selector': artifact(HERE / 'selector_manifest.json'),
        'new_model_fits': 0, 'new_predictions': 0, 'test_statistics_examined': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--job', required=True)
    p.add_argument('--reserve-s', type=float, default=90)
    a = p.parse_args()
    with Job('cpu_analysis', a.job, reserve_s=a.reserve_s, metadata={'purpose': 'TRAIN-only selector and existing-byte provenance'}) as job:
        prepare(job)
    print('V15 reuse/data/selector prepared; fitting and predictions remain pending.')
