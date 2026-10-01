"""Accounted frozen forecasts; latent caches are shared across parent seeds."""
import argparse
import copy
import gc

import numpy as np
import torch

from runtime_v15 import (HERE, CACHE, ROOT, DATASETS, SEEDS, Job, artifact, digest, resolve,
                        load_data, read_json, save_json, tensors, configure, require_storage, import_file)
from model_v15 import load_f0, reference_row, parity, selector


def origins_for(data, split):
    if split == 'test':
        values = np.concatenate([data['test_a_origins'], data['test_b_origins']])
    else:
        values = np.asarray(data[split + '_origins'])
    if not len(values) or np.any(np.diff(values) <= 0):
        raise ValueError('Expected distinct chronologically ordered origins')
    return values


def load_archive(entry, origins=None):
    receipt = artifact(entry['path'], entry['sha256'])
    with np.load(receipt['path'], allow_pickle=False) as archive:
        stored = archive['origins'].copy()
        prediction = archive['prediction'].copy()
    if (prediction.dtype != np.float32 or prediction.ndim != 3 or prediction.shape[:2] != (len(stored), 48)
            or np.any(np.diff(stored) <= 0) or not np.isfinite(prediction).all()):
        raise ValueError('Invalid immutable forecast cache')
    if origins is not None:
        origins = np.asarray(origins)
        indices = np.searchsorted(stored, origins)
        if np.any(indices >= len(stored)) or not np.array_equal(stored[indices], origins):
            raise ValueError('Requested origins are absent or misordered')
        stored, prediction = origins.copy(), prediction[indices]
    return {'origins': stored, 'prediction': prediction, 'receipt': receipt}


def latent_binding(dataset, space, split, data):
    return {'dataset': dataset, 'space': space, 'split': split,
            'data': artifact(data['_path']), 'selector_sha256': digest(HERE / 'selector_manifest.json'),
            'f0': reference_row(dataset, 'f0'), 'precision': 'float32', 'batch_size': 4,
            'point': 'Pinned native normalization and inverse-scaled median, first 48 observations',
            'files': {name: digest(HERE / name) for name in ('model_v15.py', 'cache_v15.py', 'runtime_v15.py')}}


def load_latent_cache(dataset, space, split, origins=None):
    if space not in ('raw', 'pca') or split not in ('train', 'val', 'test'):
        raise ValueError('Unknown fixed cache scope')
    data = load_data(dataset, include_test=split == 'test')
    entry = read_json(HERE / 'cache_manifest.json')['datasets'][dataset][space][split]
    if entry['binding'] != latent_binding(dataset, space, split, data):
        raise ValueError('Frozen latent cache binding changed')
    result = load_archive(entry, origins)
    expected = origins_for(data, split) if origins is None else np.asarray(origins)
    if not np.array_equal(result['origins'], expected) or result['prediction'].shape[-1] != DATASETS[dataset][1]:
        raise ValueError('Latent cache differs from the fixed origins/shape')
    return result


def bind_direct_caches(job):
    path = HERE / 'direct_cache_refs.json'
    if path.exists():
        for ds in DATASETS:
            for seed in SEEDS:
                for split in ('train', 'val'):
                    load_direct_cache(ds, seed, split)
        return
    manifest = read_json(HERE / 'data_manifest.json')
    output = {'schema': 'v15_direct_cache_reuse_1', 'datasets': {}, 'job_id': job.id}
    for ds in DATASETS:
        data = load_data(ds)
        entry = manifest['datasets'][ds]
        source = entry['parent_cache']['source']
        artifact(source['path'], source['sha256'])
        previous = read_json(resolve(source['path']))
        output['datasets'][ds] = {}
        for seed in SEEDS:
            old = previous['datasets'][ds]['direct'][str(seed)]
            binding = old['binding']
            if (binding['dataset'] != ds or binding['parent_role'] != 'direct' or binding['seed'] != seed
                    or old['parent'] != entry['parents']['direct'][str(seed)]
                    or binding['parent'] != old['parent'] or binding['precision'] != 'float32'):
                raise ValueError('Existing DIRECT cache has the wrong frozen parent')
            old_folder = resolve(source['path']).parent
            for name, sha in binding['files'].items():
                artifact(old_folder / name, sha)
            if old['data']['sha256'] != artifact(data['_path'])['sha256']:
                raise ValueError('DIRECT cache data differ from the current unchanged contract')
            record = {'source_manifest': copy.deepcopy(source), 'source_binding': copy.deepcopy(binding),
                      'parent': copy.deepcopy(old['parent']), 'data': copy.deepcopy(old['data'])}
            for split in ('train', 'val'):
                saved = load_archive(old[split])
                if (not np.array_equal(saved['origins'], origins_for(data, split))
                        or saved['prediction'].shape[-1] != DATASETS[ds][0]):
                    raise ValueError('Old DIRECT origins/shape differ')
                record[split] = saved['receipt']
            output['datasets'][ds][str(seed)] = record
            job.heartbeat({'direct_reuse': ds, 'seed': seed})
    save_json(path, output)


def load_direct_cache(dataset, seed, split, origins=None):
    if split not in ('train', 'val'):
        raise ValueError('DIRECT TEST predictions are old reference outputs, not TRAIN/VAL caches')
    data = load_data(dataset)
    record = read_json(HERE / 'direct_cache_refs.json')['datasets'][dataset][str(seed)]
    source = record['source_manifest']
    artifact(source['path'], source['sha256'])
    old = read_json(resolve(source['path']))['datasets'][dataset]['direct'][str(seed)]
    parent = read_json(HERE / 'data_manifest.json')['datasets'][dataset]['parents']['direct'][str(seed)]
    if record['source_binding'] != old['binding'] or record['parent'] != parent or record[split] != old[split]:
        raise ValueError('DIRECT reuse source binding changed')
    result = load_archive(record[split], origins)
    expected = origins_for(data, split) if origins is None else np.asarray(origins)
    if not np.array_equal(result['origins'], expected) or result['prediction'].shape[-1] != DATASETS[dataset][0]:
        raise ValueError('DIRECT cache order or original-channel dimensions differ')
    return result


def cache_forecasts(phase, job):
    if phase not in ('trainval', 'test'):
        raise ValueError('Phase must be trainval or test')
    if phase == 'test':
        from evaluate_v15 import ensure_exposure
        ensure_exposure()
    else:
        bind_direct_caches(job)
    manifest_path = HERE / 'cache_manifest.json'
    manifest = read_json(manifest_path) if manifest_path.exists() else {'schema': 'v15_latent_cache_1', 'datasets': {}}
    splits = ('train', 'val') if phase == 'trainval' else ('test',)
    for ds in DATASETS:
        data = load_data(ds, include_test=phase == 'test')
        slot = manifest['datasets'].setdefault(ds, {})
        pending = []
        for space in ('raw', 'pca'):
            group = slot.setdefault(space, {})
            for split in splits:
                if split in group:
                    load_latent_cache(ds, space, split)
                else:
                    pending.append((space, split))
        if not pending:
            continue
        model, source = load_f0(ds)
        model.eval()
        old = import_file(ROOT / 'research/tsfm_peft_internal_vs_subspace_v11_20260930/model_v11.py',
                          'v15_pinned_state_digest')
        frozen_before = old.state_digest(model.state_dict())
        try:
            with torch.no_grad():
                checks = manifest.setdefault('raw_coordinate_checks', {})
                if ds not in checks:
                    values = read_json(HERE / 'selector_manifest.json')['datasets'][ds]
                    raw_e = torch.as_tensor(values['E_raw'], dtype=torch.float32, device='cuda')
                    x, _, _ = tensors(data, data['val_origins'][:4])
                    selected_forecast = model.backbone_point(x @ raw_e)
                    full_forecast = model(x)[:, :, values['indices']]
                    check = parity(selected_forecast, full_forecast)
                    checks[ds] = {'parity': check, 'origins': data['val_origins'][:4].tolist(),
                                  'indices': values['indices'], 'job_id': job.id,
                                  'f0_source': source, 'selector_sha256': digest(HERE / 'selector_manifest.json'),
                                  'scope': 'Same frozen F0; selected original channels versus full original-channel output'}
                    save_json(manifest_path, manifest)
                    if not check['passed']:
                        raise ValueError('Selected raw-channel F0 differs from the corresponding full F0 output')
                    del x, raw_e, selected_forecast, full_forecast
                elif (not checks[ds]['parity']['passed'] or checks[ds]['f0_source'] != source
                      or checks[ds]['selector_sha256'] != digest(HERE / 'selector_manifest.json')):
                    raise ValueError('Previously recorded raw-coordinate parity is invalid or changed')
                for space, split in pending:
                    e, _ = selector(ds, space + '_free')
                    e = torch.as_tensor(e, device='cuda')
                    origins = origins_for(data, split)
                    require_storage(len(origins) * 48 * DATASETS[ds][1] * 4 + 1024 * 1024)
                    prediction = np.empty((len(origins), 48, DATASETS[ds][1]), dtype=np.float32)
                    for start in range(0, len(origins), 4):
                        job.check_limits()
                        group = origins[start:start + 4]
                        x, _, _ = tensors(data, group)
                        prediction[start:start + len(group)] = model.backbone_point(x @ e).cpu().numpy()
                        if start % 256 == 0:
                            job.heartbeat({'dataset': ds, 'space': space, 'split': split,
                                           'origins_done': start + len(group), 'total': len(origins)})
                    if not np.isfinite(prediction).all():
                        raise ValueError('Nonfinite frozen latent forecast')
                    x, _, _ = tensors(data, origins[:4])
                    online = model.backbone_point(x @ e)
                    replay = parity(online, prediction[:4], atol=1e-6, rtol=0)
                    model.chunk_rows = DATASETS[ds][1]
                    chunk = parity(model.backbone_point(x @ e), online)
                    model.chunk_rows = None
                    if not replay['passed'] or not chunk['passed']:
                        raise ValueError('Latent cache replay/chunk parity failed')
                    folder = CACHE / 'latent_forecasts' / job.label / ds / space
                    folder.mkdir(parents=True, exist_ok=True)
                    path = folder / (split + '.npz')
                    if path.exists():
                        raise FileExistsError('Preserve existing forecast attempt: ' + str(path))
                    np.savez_compressed(path, origins=origins, prediction=prediction)
                    slot[space][split] = {**artifact(path), 'job_id': job.id, 'job_label': job.label,
                                         'binding': latent_binding(ds, space, split, data),
                                         'replay_parity': replay, 'chunk_parity': chunk,
                                         'f0_source': source, 'seed_independent': True}
                    save_json(manifest_path, manifest)
                    del x, online, prediction, e
            frozen_after = old.state_digest(model.state_dict())
            frozen = {'before': frozen_before, 'after': frozen_after,
                      'passed': frozen_before == frozen_after,
                      'all_weights_frozen_and_no_grad': all(not p.requires_grad and p.grad is None for p in model.parameters()),
                      'phase': phase, 'job_id': job.id}
            manifest.setdefault('frozen_state_checks', {}).setdefault(ds, {})[job.label] = frozen
            save_json(manifest_path, manifest)
            if not frozen['passed'] or not frozen['all_weights_frozen_and_no_grad']:
                raise ValueError('Frozen F0 state changed during cache generation')
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
    return artifact(manifest_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('trainval', 'test'), required=True)
    parser.add_argument('--job', required=True)
    parser.add_argument('--reserve-s', type=float, default=900)
    args = parser.parse_args()
    with Job('gpu', args.job, reserve_s=args.reserve_s,
             metadata={'phase': args.phase, 'purpose': 'Frozen latent forecasts; zero fitting'}) as job:
        configure()
        cache_forecasts(args.phase, job)
    print('V15 frozen forecast cache complete; no fitting performed.')
