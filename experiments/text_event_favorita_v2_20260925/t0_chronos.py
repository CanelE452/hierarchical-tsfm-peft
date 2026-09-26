"""T0d only: fixed synthetic covariate test; no Favorita scoring or fitting."""
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/text_event_favorita_v2_20260925'
os.environ['HF_HOME'] = str(Path(__file__).resolve().parent / '.cache/huggingface')


def main():
    gate = json.loads((OUT / 'C0_INDEPENDENT.json').read_text(encoding='utf-8'))
    if not (gate.get('pass_all') is True):
        raise RuntimeError('C0 explicit pass_all=true required before model import')
    import numpy as np
    import torch
    from chronos import Chronos2Pipeline

    start = time.monotonic()
    rng = np.random.default_rng(20260925)
    cov = rng.uniform(0, 1, 512).astype(np.float32)
    target = 10 * cov + rng.normal(0, .02, 512).astype(np.float32)
    future = np.array([.9, .1, .8, .2, .7, .3, .6], dtype=np.float32)
    model_path = Path.home() / '.cache/huggingface/hub/models--amazon--chronos-2/snapshots/29ec3766d36d6f73f0696f85560a422f50e8498c'
    assert model_path.is_dir()
    pipe = Chronos2Pipeline.from_pretrained(
        str(model_path), local_files_only=True, device_map='cuda', dtype=torch.bfloat16)
    levels = list(pipe.quantiles)
    assert len(levels) == 21
    inputs = [{'target': target, 'past_covariates': {'cov': cov},
               'future_covariates': {'cov': future}}, {'target': target}]
    q, median = pipe.predict_quantiles(inputs, prediction_length=7,
        quantile_levels=levels, context_length=512, batch_size=4, cross_learning=False)
    point = []
    for pred in q:
        a = pred.float().cpu().numpy()[0].clip(0)
        a = np.concatenate([a[:, :1], a, a[:, -1:]], axis=1)
        point.append(np.trapezoid(a, [0., *levels, 1.], axis=1))
    truth = 10 * future
    cov_mae = float(np.abs(point[0]-truth).mean())
    no_cov_mae = float(np.abs(point[1]-truth).mean())
    corr = float(np.corrcoef(point[0], truth)[0, 1])
    # Operational following check; not an experimental performance threshold.
    passed = bool(np.isfinite(point).all() and corr > 0 and cov_mae < no_cov_mae)
    report = {'test': 'T0d', 'status': 'PASS' if passed else 'STOP_DEBUG',
        'criterion_fixed_before_run': 'finite points; positive forecast/covariate correlation; MAE below identical target-only forecast',
        'seed': 20260925, 'model_revision': '29ec3766d36d6f73f0696f85560a422f50e8498c',
        'quantiles': levels, 'point_prediction': 'QMEAN21, nonnegative, constant endpoint extension',
        'future_covariate': future.tolist(), 'truth': truth.tolist(),
        'with_cov': point[0].tolist(), 'without_cov': point[1].tolist(),
        'cov_mae': cov_mae, 'no_cov_mae': no_cov_mae, 'correlation': corr,
        'fit_steps': 0, 'favorita_predictions': 0, 'wall_seconds': time.monotonic()-start,
        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated()}
    with (OUT/'T0_CHRONOS.json').open('x', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
