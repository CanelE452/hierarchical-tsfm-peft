"""One bounded VAL-only diagnosis of saved versus regrouped FP32 predictions."""
import json

from runtime_v4 import HERE, Job, load_data, save_json, source_receipt


def main():
    target = HERE / 'cost_replay_diagnostic_01.json'
    if target.exists():
        raise RuntimeError('Preserve original diagnostic')
    with Job('v4_cost_replay_diagnostic_01', category='gpu', reserve_seconds=120) as job:
        import numpy as np
        import torch
        from model_v4 import restore_model
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        data = load_data(include_test=False)
        all_origins = data['val_origins']
        indices = np.linspace(0, len(all_origins)-1, 24, dtype=int)
        inputs = torch.from_numpy(np.stack([data['x'][o-512:o] for o in all_origins]))
        selection = json.loads((HERE / 'selected.json').read_text(encoding='utf-8'))
        records = []
        for row in selection['all_neural_runs']:
            if row['id'] not in selection['selected']['linear_res']['run_ids']:
                continue
            model = restore_model(row['checkpoint'], device='cuda')
            with np.load(row['best_val_prediction']['path'], allow_pickle=False) as stored:
                reference = torch.from_numpy(stored['prediction'].copy())
            def predict(x, batch, inference):
                with torch.inference_mode(inference), torch.no_grad():
                    return torch.cat([model(chunk.to('cuda')).cpu() for chunk in x.split(batch)])
            full = predict(inputs, 4, False)
            subset = inputs[indices].contiguous()
            normal = predict(subset, 4, False)
            infer = predict(subset, 4, True)
            singleton = predict(subset, 1, True)
            def difference(a, b):
                error = (a-b).abs()
                return {'max_abs': float(error.max()), 'old_tolerance_failures': int((error > 1e-6+1e-5*b.abs()).sum()),
                        'existing_cpu_merge_tolerance_failures': int((error > 1e-5+1e-4*b.abs()).sum()),
                        'elements': error.numel()}
            records.append({'id': row['id'], 'full_original_grouping_vs_saved': difference(full, reference),
                            'subset_no_grad_vs_saved': difference(normal, reference[indices]),
                            'subset_inference_vs_no_grad': difference(infer, normal),
                            'subset_inference_vs_saved': difference(infer, reference[indices]),
                            'batch1_inference_vs_saved': difference(singleton, reference[indices])})
            del model
            torch.cuda.synchronize()
            torch._C._cuda_clearCublasWorkspaces()
            torch.cuda.empty_cache()
            job.heartbeat('VAL regrouping and execution-mode comparisons: ' + row['id'])
        save_json(target, {'status': 'complete', 'source': source_receipt(), 'records': records,
                          'new_fits': 0, 'test_read': False, 'target_scoring': False,
                          'scope': 'VAL-only output parity diagnosis; no timing performance claim.'})
        print(json.dumps(records))


if __name__ == '__main__':
    main()
