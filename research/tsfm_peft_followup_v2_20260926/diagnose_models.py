"""Minimal initial/selected CURRENT probes; no optimizer or fit."""
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from run import CACHE, HERE, ROOT, GPUJob, digest, load_data, phase_sample, save_json, score_arrays, source_receipt, tensors
from model import make_model, pca_basis

V1 = ROOT / 'research/tsfm_peft_development_20260926'


def current_id(dataset, seed):
    return f'revision1_rank32_residual_{seed}_0.001' if dataset == 'electricity' else f'bull_final_residual_{seed}_0.001'


@torch.no_grad()
def probe(model, data, origins, job, label):
    predictions, main_energy, residual_energy = [], 0.0, 0.0
    input_residual_energy, input_energy, count, input_count = 0.0, 0.0, 0, 0
    model.eval()
    for start in range(0, len(origins), 4):
        x, _, _ = tensors(data, origins[start:start + 4])
        z = model.encoder(x)
        remainder = x - model.decoder(z)
        main = model.decoder(model.backbone_point(z))
        correction = model.residual(remainder)
        predictions.append((main + correction).float().cpu().numpy())
        main_energy += main.double().square().sum().item()
        residual_energy += correction.double().square().sum().item()
        input_energy += x.double().square().sum().item()
        input_residual_energy += remainder.double().square().sum().item()
        count += main.numel()
        input_count += x.numel()
        if start % 40 == 0:
            job.heartbeat(f'{label}: {start}/{len(origins)}')
    prediction = np.concatenate(predictions)
    target = np.stack([data['x'][o:o + 48] for o in origins])
    mask = np.stack([data['finite'][o:o + 48] for o in origins])
    scores = score_arrays(prediction, target, mask)
    return prediction, dict(scores=scores, main_rms=(main_energy / count) ** .5,
                            correction_rms=(residual_energy / count) ** .5,
                            input_rms=(input_energy / input_count) ** .5,
                            reconstruction_remainder_rms=(input_residual_energy / input_count) ** .5)


def main():
    output = HERE / 'model_diagnostics.json'
    if output.exists():
        raise RuntimeError('Never overwrite completed diagnostics')
    local = CACHE / 'model_diagnostics'
    local.mkdir(parents=True, exist_ok=False)
    rows = []
    started = time.perf_counter()
    with GPUJob('minimal_current_initial_selected_diagnostics') as job:
        for dataset in ('electricity', 'bull'):
            data = load_data(dataset)
            origins_by_split = {'train_probe': np.sort(phase_sample(data['train_origins'], 9262026, 0, 96)),
                                'val': data['val_origins']}
            assert all(np.sum(origins_by_split['train_probe'] % 24 == p) == 4 for p in range(24))
            assert np.isin(origins_by_split['train_probe'], data['train_origins']).all()
            for seed in (92601, 92602):
                fit_id = current_id(dataset, seed)
                result_path = V1 / 'runs' / fit_id / 'result.json'
                result = json.loads(result_path.read_text(encoding='utf-8'))
                checkpoint_path = Path(result['checkpoint'])
                assert digest(checkpoint_path) == result['checkpoint_sha256']
                assert digest(data['_path']) == result['data_sha256']
                saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                assert saved['spec']['residual_rank'] == 32
                torch.manual_seed(seed)
                np.random.seed(seed)
                basis = pca_basis(data['x'][int(data['train_start']):int(data['train_end'])], saved['basis'].shape[1])
                assert np.array_equal(basis, saved['basis'])
                model = make_model('residual', basis, residual_rank=32, ed_mode='current')
                for state in ('initial', 'selected'):
                    if state == 'selected':
                        model.restore_adapter(saved['state'])
                    row = {'dataset': dataset, 'seed': seed, 'fit': fit_id, 'state': state,
                           'checkpoint_sha256': digest(checkpoint_path), 'splits': {}}
                    for split, origins in origins_by_split.items():
                        pred, measures = probe(model, data, origins, job, f'{fit_id} {state} {split}')
                        path = local / f'{fit_id}_{state}_{split}.npz'
                        np.savez_compressed(path, prediction=pred, origins=origins)
                        measures.update(prediction_sha256=digest(path), cache=str(path.relative_to(ROOT)))
                        if state == 'selected' and split == 'val':
                            with np.load(checkpoint_path.with_name('best_val.npz')) as previous:
                                assert np.array_equal(previous['origins'], origins)
                                error = float(np.max(np.abs(previous['prediction'] - pred)))
                            assert error <= 1e-6, error
                            measures['v1_validation_replay_max_abs'] = error
                        row['splits'][split] = measures
                    rows.append(row)
                    print(json.dumps({'dataset': dataset, 'seed': seed, 'state': state,
                                      'mse': {s: m['scores']['mse'] for s, m in row['splits'].items()}}), flush=True)
                del model, saved
                gc.collect()
                torch.cuda.empty_cache()
    save_json(output, {'status': 'complete', 'rows': rows, 'source': source_receipt(),
                       'elapsed_s': time.perf_counter() - started, 'optimizer_updates': 0,
                       'scope': 'Fixed TRAIN probe and full VAL; initial and selected only, not all-epoch curves. Branch magnitudes are descriptive, not causal attribution.'})


if __name__ == '__main__':
    main()
