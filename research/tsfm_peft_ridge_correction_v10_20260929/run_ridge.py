"""Finite full-TRAIN ridge fitting and joint pre-evaluation VAL selection."""
import argparse
import gc
import json
import time

from runtime_ridge import (HERE, ROOT, CACHE, V9, Job, digest, save_json, read_json,
                           load_data, score_arrays, parent_receipt_for, budget_snapshot)


def configure():
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def cpu_prediction(data, origins, parent, weight, family, job):
    import numpy as np
    import torch
    from torch.nn import functional as F
    out = np.empty_like(parent, dtype=np.float32)
    basis = torch.as_tensor(data['basis'], dtype=torch.float32)
    weight = torch.as_tensor(weight, dtype=torch.float32)
    with torch.no_grad():
        for start in range(0, len(origins), 64):
            if start % 1024 == 0:
                job.check_limits()
            selected = origins[start:start + 64]
            x = torch.as_tensor(np.stack([data['x'][o - 512:o] for o in selected]), dtype=torch.float32)
            delta = x - x[:, -1:, :]
            z = F.linear(F.linear(delta, basis.T), basis) if family == 'ridge_p' else delta
            correction = F.linear(z.transpose(1, 2), weight).transpose(1, 2)
            out[start:start + len(selected)] = (torch.as_tensor(np.asarray(parent[start:start + len(selected)]).copy()) + correction).numpy()
    return out


def checks(name):
    with Job(name, 'cpu_check', 60):
        configure()
        from threadpoolctl import threadpool_limits
        from check_solver import run_checks
        with threadpool_limits(limits=4):
            result = run_checks()
        save_json(HERE / (name + '.json'), result)
        print(result['status'], flush=True)


def fit_all(name, reserve_seconds):
    assert not (HERE / 'selected.json').exists() and not (HERE / 'test_exposure.json').exists()
    with Job(name, 'cpu_analysis', reserve_seconds, metadata={'purpose': 'full_train_ridge'}) as job:
        import numpy as np
        import torch
        from threadpoolctl import threadpool_limits
        from ridge_solver import build_statistics, solve_ridge
        configure()
        cache_manifest = read_json(HERE / 'parent_cache.json')
        all_results = []
        with threadpool_limits(limits=4):
            for dataset in ('robin', 'jena'):
                data = load_data(dataset)
                origins = data['train_origins']
                target = np.stack([data['x'][o:o + 48] for o in origins])
                mask = np.stack([data['finite'][o:o + 48] for o in origins])
                val_origins = data['val_origins']
                val_target = np.stack([data['x'][o:o + 48] for o in val_origins])
                val_mask = np.stack([data['finite'][o:o + 48] for o in val_origins])
                parents, val_parents = {}, {}
                for seed in (92601, 92602):
                    records = cache_manifest['datasets'][dataset][str(seed)]
                    for split in ('train', 'val'):
                        assert digest(records[split]['path']) == records[split]['sha256']
                        assert np.array_equal(records[split]['origins'], data[split + '_origins'])
                    parents[seed] = np.load(records['train']['path'], mmap_mode='r', allow_pickle=False)
                    val_parents[seed] = np.load(records['val']['path'], mmap_mode='r', allow_pickle=False)

                def blocks():
                    for start in range(0, len(origins), 128):
                        job.check_limits()
                        selected = origins[start:start + 128]
                        yield np.arange(start, start + len(selected)), np.stack([data['x'][o - 512:o] for o in selected])

                for family in ('ridge_p', 'ridge_raw'):
                    begun = time.perf_counter()
                    stats = build_statistics(blocks, target, mask, parents, family=family, basis=data['basis'])
                    stat_receipt = {'dataset': dataset, 'family': family, 'seconds': time.perf_counter() - begun,
                                    'mask_groups': len(stats.horizon_groups), 'origins': len(origins),
                                    'scope': 'shared sufficient statistics; no coefficients fitted here'}
                    save_json(HERE / f'statistics_{dataset}_{family}.json', stat_receipt)
                    job.heartbeat(f'{dataset}/{family} statistics ready')
                    for penalty in (.001, .1):
                        for seed in (92601, 92602):
                            suffix = '001' if penalty == .001 else '100'
                            identifier = f'v10_{dataset}_{family}_{seed}_lambda{suffix}'
                            spec = {'id': identifier, 'dataset': dataset, 'seed': seed, 'family': family, 'penalty': penalty}
                            local = CACHE / 'runs' / identifier
                            public = HERE / 'runs' / identifier
                            if local.exists() or public.exists():
                                raise RuntimeError('Existing attempt output: preserve it before a retry')
                            with Job(identifier, 'fit', 0, metadata=spec) as fit_job:
                                local.mkdir(parents=True)
                                public.mkdir(parents=True)
                                save_json(public / 'attempt.json', {'status': 'running', 'spec': spec, 'ledger_id': fit_job.id})
                                weight, receipt = solve_ridge(stats, seed, penalty)
                                job.check_limits()
                                checkpoint = local / 'best.pt'
                                payload = dict(spec, weight=torch.from_numpy(weight.copy()),
                                               basis=torch.as_tensor(data['basis']), parent_receipt=parent_receipt_for(dataset, seed))
                                torch.save(payload, checkpoint)
                                saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
                                assert torch.equal(saved['weight'], payload['weight'])
                                assert saved['parent_receipt'] == payload['parent_receipt']
                                prediction = cpu_prediction(data, val_origins, val_parents[seed], saved['weight'], family, job)
                                val_score = score_arrays(prediction, val_target, val_mask)
                                val_path = local / 'best_val.npz'
                                np.savez_compressed(val_path, prediction=prediction, origins=val_origins)
                                train_prediction = cpu_prediction(data, origins, parents[seed], saved['weight'], family, job)
                                train_score = score_arrays(train_prediction, target, mask)
                                fp_difference = abs(train_score['mse'] - receipt['masked_macro_mse'])
                                assert fp_difference <= 1e-5 * max(1., abs(receipt['masked_macro_mse']))
                                result = {'id': identifier, 'spec': spec, 'status': 'complete', 'ledger_id': fit_job.id,
                                          'checkpoint': str(checkpoint), 'checkpoint_sha256': digest(checkpoint),
                                          'best_val_mse': val_score['mse'], 'val_score': val_score, 'train_score': train_score,
                                          'best_val_prediction': {'path': str(val_path), 'sha256': digest(val_path)},
                                          'solver': receipt, 'statistics': stat_receipt,
                                          'fp32_fp64_train_mse_difference': fp_difference,
                                          'checkpoint_roundtrip': 'exact FP64 weights and parent receipt',
                                          'parent_receipt': payload['parent_receipt'], 'new_fitted_coefficients': 24576,
                                          'cumulative_fitted_adapter_coefficients': 42496,
                                          'optimizer_updates': 0, 'fit_device': 'cpu', 'test_used': False,
                                          'data_sha256': digest(data['_path']), 'elapsed_s': time.perf_counter() - fit_job.started}
                                save_json(public / 'result.json', result)
                                save_json(public / 'attempt.json', {'status': 'complete', 'spec': spec, 'ledger_id': fit_job.id,
                                                                  'optimizer_updates': 0, 'horizon_solves': 48})
                                all_results.append(result)
                                del train_prediction, prediction, saved, payload
                                fit_job.heartbeat(f'{identifier}: TRAIN={train_score["mse"]:.6f}, VAL={val_score["mse"]:.6f}')
                    del stats
                    gc.collect()
                del data, target, mask, parents, val_parents
                gc.collect()
        selected = {'datasets': {}, 'selection_rule': 'per dataset/arm minimum two-parent mean whole-VAL MSE; exact ties .001 before .1',
                    'test_used_for_selection': False, 'seed_semantics': 'two different frozen parents, deterministic ridge fitting',
                    'parent_cache_sha256': digest(HERE / 'parent_cache.json'), 'created_utc': time.time()}
        for dataset in ('robin', 'jena'):
            selected['datasets'][dataset] = {}
            for family in ('ridge_p', 'ridge_raw'):
                choices = []
                for penalty in (.001, .1):
                    runs = [r for r in all_results if r['spec']['dataset'] == dataset and r['spec']['family'] == family and r['spec']['penalty'] == penalty]
                    assert len(runs) == 2
                    choices.append({'penalty': penalty, 'mean_val_mse': sum(r['best_val_mse'] for r in runs) / 2, 'runs': runs})
                chosen = min(choices, key=lambda r: r['mean_val_mse'])
                parent_score = sum(cache_manifest['datasets'][dataset][str(seed)]['val']['score']['mse'] for seed in (92601, 92602)) / 2
                chosen.update(parent_val_mse=parent_score,
                              policy_choice='parent' if chosen['mean_val_mse'] >= parent_score else family,
                              all_candidates=[{'penalty': r['penalty'], 'mean_val_mse': r['mean_val_mse']} for r in choices])
                selected['datasets'][dataset][family] = chosen
        save_json(HERE / 'selected.json', selected)
        job.heartbeat('all16 fits and both datasets VAL selections complete; TEST remains closed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['check', 'fit'])
    parser.add_argument('--job', required=True)
    parser.add_argument('--reserve-seconds', type=float, default=1200)
    args = parser.parse_args()
    if args.stage == 'check':
        checks(args.job)
    else:
        fit_all(args.job, args.reserve_seconds)
    print(json.dumps(budget_snapshot()), flush=True)
