"""Pin existing artifacts and cache frozen-parent forecasts; never fit."""
import argparse
import gc
import json
from pathlib import Path
import subprocess
import time

from runtime_ridge import (HERE, ROOT, CACHE, V9, Job, digest, save_json, read_json,
                           load_data, load_parent, tensors, score_arrays, budget_snapshot)


def prepare_reuse(name):
    with Job(name, 'cpu_check', 60):
        assert not (HERE / 'reuse_manifest.json').exists()
        old = read_json(V9 / 'reuse_manifest.json')
        parents = old['parents']
        for dataset in ('robin', 'jena'):
            for seed in ('92601', '92602'):
                for key in ('checkpoint', 'model_source', 'result', 'best_val_prediction'):
                    item = parents[dataset][seed][key]
                    assert digest(item['path']) == item['sha256']
        contracts = {}
        for dataset in ('robin', 'jena'):
            path = V9 / f'data_contract_{dataset}.json'
            contract = read_json(path)
            for split in ('trainval', 'test'):
                item = contract[split]
                source = Path(item['path'])
                if not source.is_absolute():
                    source = ROOT / source
                assert digest(source) == item['sha256']
            contracts[dataset] = {'path': str(path), 'sha256': digest(path), 'contract': contract}
        preserved = {}
        paths = subprocess.check_output(['git', 'ls-files', 'research'], cwd=ROOT, text=True).splitlines()
        for path in paths:
            if (ROOT / path).is_file():
                preserved[path] = digest(ROOT / path)
        save_json(HERE / 'preservation.json', {'files': preserved, 'count': len(preserved)})
        save_json(HERE / 'reuse_manifest.json', {'parents': parents, 'contracts': contracts,
                  'v9_manifest': {'path': str(V9 / 'reuse_manifest.json'), 'sha256': digest(V9 / 'reuse_manifest.json')},
                  'created_utc': time.time(), 'test_values_read': False, 'test_file_hash_only': True,
                  'scope': 'Robin/Jena exposed development; unchanged source/normalization/targets/splits',
                  'backbone': old['backbone']})


def cache_parents(name, reserve_seconds):
    with Job(name, 'gpu', reserve_seconds, metadata={'purpose': 'parent_trainval_cache'}) as job:
        import numpy as np
        import torch
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        manifest_path = HERE / 'parent_cache.json'
        assert not manifest_path.exists()
        manifest = {'datasets': {}, 'uses_test': False, 'scope': 'fitting only; never deployment timing'}
        old = read_json(HERE / 'reuse_manifest.json')
        for dataset in ('robin', 'jena'):
            data = load_data(dataset)
            manifest['datasets'][dataset] = {}
            for seed in (92601, 92602):
                job.check_limits()
                parent = load_parent(dataset, seed, 'cuda')
                destination = CACHE / 'parents' / dataset / str(seed)
                destination.mkdir(parents=True, exist_ok=True)
                records = {}
                for split in ('train', 'val'):
                    origins = data[split + '_origins']
                    path = destination / (split + '.npy')
                    if path.exists():
                        raise RuntimeError('Preserve partial cache before a recorded retry')
                    prediction = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                                          shape=(len(origins), 48, data['x'].shape[1]))
                    tick = time.perf_counter()
                    with torch.no_grad():
                        for start in range(0, len(origins), 4):
                            if start % 512 == 0:
                                job.check_limits()
                            x, _, _ = tensors(data, origins[start:start + 4])
                            prediction[start:start + 4] = parent(x).cpu().numpy()
                            if start and start % 4096 == 0:
                                job.heartbeat(f'{dataset}/{seed}/{split} {start}/{len(origins)}')
                    prediction.flush()
                    check = None
                    if split == 'val':
                        receipt = old['parents'][dataset][str(seed)]['best_val_prediction']
                        assert digest(receipt['path']) == receipt['sha256']
                        with np.load(receipt['path'], allow_pickle=False) as saved:
                            assert np.array_equal(saved['origins'], origins)
                            difference = np.abs(prediction.astype(np.float64) - saved['prediction'])
                            violations = difference > 1e-5 + 1e-4 * np.abs(saved['prediction'])
                        check = {'max_abs': float(difference.max()), 'violations': int(violations.sum())}
                        assert check['violations'] == 0
                    target = np.stack([data['x'][o:o + 48] for o in origins])
                    mask = np.stack([data['finite'][o:o + 48] for o in origins])
                    records[split] = {'path': str(path), 'sha256': digest(path), 'origins': origins.tolist(),
                                      'score': score_arrays(prediction, target, mask), 'replay': check,
                                      'elapsed_s': time.perf_counter() - tick}
                    del prediction, target, mask
                    job.heartbeat(f'{dataset}/{seed}/{split} complete')
                records['parent_receipt'] = old['parents'][dataset][str(seed)]
                manifest['datasets'][dataset][str(seed)] = records
                save_json(HERE / 'parent_cache_partial.json', manifest)
                del parent, x
                gc.collect()
                torch.cuda.empty_cache()
        save_json(manifest_path, manifest)
        job.heartbeat('all four parent TRAIN/VAL caches verified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['reuse', 'parents'])
    parser.add_argument('--job', required=True)
    parser.add_argument('--reserve-seconds', type=float, default=900)
    args = parser.parse_args()
    if args.stage == 'reuse':
        prepare_reuse(args.job)
    else:
        cache_parents(args.job, args.reserve_seconds)
    print(json.dumps(budget_snapshot()), flush=True)
