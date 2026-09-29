"""Restore selected models online, then bind both selections before evaluation."""
import argparse
import ast
import gc
import time

from runtime_ridge import (HERE, CACHE, Job, read_json, save_json, digest, load_data,
                           tensors, score_arrays, budget_snapshot)


def static_checks(name):
    with Job(name, 'cpu_check', 30):
        for path in HERE.glob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'))
        save_json(HERE / (name + '.json'), {'status': 'PASS', 'syntax_files': [p.name for p in HERE.glob('*.py')],
                                           'optimizer_updates': 0, 'model_execution': False})


def online_checks(name):
    with Job(name, 'gpu', 120, metadata={'purpose': 'selected_online_validation'}) as job:
        import numpy as np
        import torch
        from model_ridge import restore_model
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        selection = read_json(HERE / 'selected.json')
        binding = digest(HERE / 'selected.json')
        rows = []
        for dataset, families in selection['datasets'].items():
            data = load_data(dataset)
            origins = data['val_origins']
            target = np.stack([data['x'][o:o + 48] for o in origins])
            mask = np.stack([data['finite'][o:o + 48] for o in origins])
            for family, choice in families.items():
                for run in choice['runs']:
                    assert digest(run['checkpoint']) == run['checkpoint_sha256']
                    model = restore_model(run['checkpoint'], device='cuda')
                    assert not any(p.requires_grad for p in model.parameters()) and not model.parent.training
                    prediction = []
                    q_max = 0.
                    with torch.no_grad():
                        for start in range(0, len(origins), 4):
                            job.check_limits()
                            x, _, _ = tensors(data, origins[start:start + 4])
                            parts = model.parts(x)
                            prediction.append((parts['parent_output'] + parts['correction']).cpu().numpy())
                            if family == 'ridge_p':
                                q = parts['correction'] - model.parent.decoder(model.parent.encoder(parts['correction']))
                                q_max = max(q_max, float(q.abs().max()))
                    prediction = np.concatenate(prediction)
                    receipt = run['best_val_prediction']
                    assert digest(receipt['path']) == receipt['sha256']
                    with np.load(receipt['path'], allow_pickle=False) as saved:
                        assert np.array_equal(saved['origins'], origins)
                        difference = np.abs(prediction.astype(np.float64) - saved['prediction'])
                        violations = difference > 1e-5 + 1e-4 * np.abs(saved['prediction'])
                    assert not violations.any(), 'Do not relax output tolerance'
                    if family == 'ridge_p':
                        assert q_max < 1e-5
                    scores = score_arrays(prediction, target, mask)
                    rows.append({'id': run['id'], 'dataset': dataset, 'family': family,
                                 'checkpoint_sha256': run['checkpoint_sha256'], 'val_max_absolute_difference': float(difference.max()),
                                 'violating_elements': int(violations.sum()), 'q_correction_max_abs': q_max if family == 'ridge_p' else None,
                                 'online_val_mse': scores['mse'], 'stored_val_mse': run['best_val_mse'],
                                 'fitted_coefficients': 24576, 'deployment_requires_grad': 0,
                                 'total_parameters': sum(p.numel() for p in model.parameters()),
                                 'parameter_bytes': sum(p.numel() * p.element_size() for p in model.parameters())})
                    save_json(HERE / (name + '_partial.json'), rows)
                    job.heartbeat('online VAL replay ' + run['id'])
                    del model, x, parts, prediction
                    gc.collect()
                    torch.cuda.empty_cache()
        assert digest(HERE / 'selected.json') == binding
        save_json(HERE / 'online_validation.json', {'status': 'PASS', 'selected_sha256': binding, 'rows': rows,
                                                    'atol': 1e-5, 'rtol': 1e-4, 'test_used': False})


def expose(name):
    with Job(name, 'cpu_check', 30):
        assert not (HERE / 'selection_seal.json').exists() and not (HERE / 'test_exposure.json').exists()
        selected = read_json(HERE / 'selected.json')
        check = read_json(HERE / 'online_validation.json')
        assert check['status'] == 'PASS' and check['selected_sha256'] == digest(HERE / 'selected.json')
        assert set(selected['datasets']) == {'robin', 'jena'}
        results = list((HERE / 'runs').glob('*/result.json'))
        assert len(results) == 16 and all(read_json(p)['status'] == 'complete' for p in results)
        save_json(HERE / 'selection_seal.json', {'selected_sha256': digest(HERE / 'selected.json'),
                  'reuse_manifest_sha256': digest(HERE / 'reuse_manifest.json'), 'created_utc': time.time(),
                  'protocol_sha256': digest(HERE / 'protocol.json'), 'online_validation_sha256': digest(HERE / 'online_validation.json'),
                  'complete_result_sha256': {str(p): digest(p) for p in results}})
        save_json(HERE / 'test_exposure.json', {'seal_sha256': digest(HERE / 'selection_seal.json'),
                  'first_opened_utc': time.time(), 'datasets': ['robin', 'jena'], 'scope': 'exposed development',
                  'trigger': 'all16 fits, both VAL selections and online validation complete; open all planned comparisons'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['static', 'online', 'expose'])
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    {'static': static_checks, 'online': online_checks, 'expose': expose}[args.stage](args.job)
    print(budget_snapshot(), flush=True)
