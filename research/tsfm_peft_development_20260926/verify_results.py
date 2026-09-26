"""CPU audit of completed fits; no model inference or protected scoring."""
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
import torch

from run import HERE, digest, load_data, phase_sample, save_json


def verify(path, data):
    row = json.loads(path.read_text())
    assert row['status'] == 'complete'
    assert digest(data['_path']) == row['data_sha256']
    checkpoint = Path(row['checkpoint'])
    assert digest(checkpoint) == row['checkpoint_sha256']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert saved['epoch'] == row['selected_epoch']
    assert saved['steps'] == row['selected_steps']
    with np.load(checkpoint.with_name('best_val.npz')) as archive:
        prediction, origins = archive['prediction'], archive['origins']
    assert np.array_equal(origins, data['val_origins'])
    assert np.isfinite(prediction).all()
    channel_scores = []
    for channel in range(prediction.shape[-1]):
        errors = [
            (float(prediction[i, h, channel]) - float(data['x'][o + h, channel])) ** 2
            for i, o in enumerate(origins)
            for h in range(prediction.shape[1])
            if data['finite'][o + h, channel]
        ]
        assert errors, 'No targets for a selected channel'
        channel_scores.append(math.fsum(errors) / len(errors))
    metric = math.fsum(channel_scores) / len(channel_scores)
    metric_error = abs(metric - row['best_val_mse'])
    assert metric_error < 1e-10, metric_error
    curve = json.loads(path.with_name('curve.json').read_text())
    selected = min(curve, key=lambda item: (item['val_mse'], item['epoch']))
    assert selected['epoch'] == row['selected_epoch']
    assert selected['val_mse'] == row['best_val_mse']
    assert curve[-1]['steps'] == row['total_steps']
    assert curve[-1]['epoch'] == row['epochs_completed']
    assert len(curve) == row['epochs_completed'] + 1
    schedule = hashlib.sha256()
    for epoch in range(1, row['epochs_completed'] + 1):
        origins = phase_sample(data['train_origins'], row['seed'], epoch, row.get('epoch_origins', 512))
        schedule.update(origins.astype('<i8').tobytes())
    assert schedule.hexdigest() == row['schedule_sha256']
    unchanged = [name for name, update in row['parameter_max_updates'].items() if update == 0]
    # Bolt predicts from one decoder token: its self-attention softmax has one
    # element and cannot depend on q. Cross-attention q remains trainable.
    for name in unchanged:
        assert row['arm'] == 'lora' and re.search(r'decoder\.block\.\d+\.layer\.0\.SelfAttention\.q\.lora_[AB]\.', name), name
    registered = sum(value.numel() for value in saved['state'].values())
    updated = sum(value.numel() for name, value in saved['state'].items() if row['parameter_max_updates'][name] > 0)
    assert registered == row['trainable_parameters']
    assert row['replay_error'] <= 1e-6
    return {'id': row['id'], 'metric_error': metric_error, 'selected_epoch': row['selected_epoch'], 'registered_parameters': registered, 'parameters_with_recorded_update': updated, 'unchanged_parameter_names': unchanged, 'checkpoint_sha256': row['checkpoint_sha256'], 'result_sha256': digest(path)}


def main():
    rows, datasets = [], {}
    for path in sorted((HERE / 'runs').glob('*/result.json')):
        record = json.loads(path.read_text())
        dataset = record['dataset']
        if dataset not in datasets:
            datasets[dataset] = load_data(dataset)
        rows.append(verify(path, datasets[dataset]))
    assert rows, 'No completed fits to verify'
    save_json(HERE / 'fit_verification.json', {'status': 'pass', 'script_sha256': digest(Path(__file__)), 'completed_fit_count': len(rows), 'scope': 'saved validation predictions, scalar metric recomputation, checkpoint hash, minimum-validation selection, paired sampling hash, recorded updates and replay errors; not independent reproduction or protected evaluation', 'fits': rows})
    print(json.dumps({'verified_fits': len(rows), 'max_metric_error': max(row['metric_error'] for row in rows)}))


if __name__ == '__main__':
    main()
