"""Bounded VAL-only replay diagnostic following cost parity failure; no fitting."""
from runtime_v9 import HERE, Job, digest, save_json, score_arrays, source_receipt

with Job('cost_replay_diagnostic01', category='gpu', reserve_seconds=180,
         metadata={'fit_count': 0, 'scope': 'failed Jena cost VAL replay; no TEST'}) as job:
    import numpy as np
    import torch
    import cost_v9 as cost
    cost.np, cost.torch = np, torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cost.decision_guard('jena')
    item = next(r for r in cost.candidates('jena')
                if r['id'] == 'v9_jena_staged_p_92602__basic')
    data = cost.prepare('jena')
    with np.load(cost.checked_path(item['val_prediction']), allow_pickle=False) as saved:
        assert np.array_equal(saved['origins'], data['origins'])
        reference = torch.from_numpy(saved['prediction'].copy())
    rows = []
    for context_name in ('no_grad', 'inference_mode'):
        context = getattr(torch, context_name)
        with context():
            model, _ = cost.load_candidate(item)
            replay = cost.predict(model, data['inputs'], 4, job=job)
            score = score_arrays(replay.numpy(), data['target'], data['mask'])
            rows.append({'context': context_name, 'difference': cost.difference(replay, reference, 1e-6, 0),
                         'mse': score['mse'], 'reference_mse': item['val_mse'],
                         'mse_difference': score['mse'] - item['val_mse']})
            del model, replay
        cost.cleanup()
    save_json(HERE / 'cost_replay_diagnostic01.json', {'rows': rows, 'item': item,
              'cost_source_sha256': digest(HERE / 'cost_v9.py'), 'source': source_receipt()})
    print(rows, flush=True)
