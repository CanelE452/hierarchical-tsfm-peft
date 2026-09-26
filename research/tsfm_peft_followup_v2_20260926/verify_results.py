"""CPU checks of saved fits, selections, schedules, and v1 preservation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from run import HERE, ROOT, digest, load_data, phase_sample, save_json


def scalar_channel_scores(pred, target, mask):
    errors, absolute = [], []
    for channel in range(pred.shape[-1]):
        valid = mask[..., channel].reshape(-1)
        difference = pred[..., channel].reshape(-1)[valid].astype(np.float64) - target[..., channel].reshape(-1)[valid].astype(np.float64)
        assert len(difference)
        errors.append(float(np.dot(difference, difference) / len(difference)))
        absolute.append(float(np.abs(difference).sum() / len(difference)))
    return {'mse': float(np.mean(errors)), 'mae': float(np.mean(absolute)), 'channel_mse': errors}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--cohort', required=True)
    args = parser.parse_args()
    output = HERE / f'{args.name}_verification.json'
    assert not output.exists(), 'Keep previous verification snapshots'
    provenance = json.loads((HERE / 'provenance.json').read_text(encoding='utf-8'))
    assert digest(HERE / 'PLAN.md') == provenance['plan_sha256']
    for name, expected in provenance['v1_tracked_sha256'].items():
        assert digest(ROOT / name) == expected, name
    rows, failed = [], []
    data_by_dataset = {}
    initial_groups = {}
    specs = json.loads(Path(args.cohort).read_text(encoding='utf-8'))
    expected = {spec['id']: spec for spec in specs}
    assert len(expected) == len(specs) and expected, 'Nonempty unique expected cohort required'
    for fit_id in expected:
        path = HERE / 'runs' / fit_id / 'result.json'
        assert path.exists(), f'Required fit missing: {fit_id}'
        assert json.loads(path.read_text(encoding='utf-8'))['status'] == 'complete'
    for attempt_path in sorted((HERE / 'runs').glob('*/attempt.json')):
        attempt = json.loads(attempt_path.read_text(encoding='utf-8'))
        if attempt['status'] != 'complete':
            failed.append({'id': attempt['id'], 'status': attempt['status']})
            continue
        result = json.loads(attempt_path.with_name('result.json').read_text(encoding='utf-8'))
        assert result['id'] == attempt['id'] and result['status'] == 'complete'
        if result['id'] in expected:
            assert all(result.get(k) == v and attempt.get(k) == v for k, v in expected[result['id']].items())
        dataset = result['dataset']
        if dataset not in data_by_dataset:
            data_by_dataset[dataset] = load_data(dataset)
        data = data_by_dataset[dataset]
        assert digest(data['_path']) == result['data_sha256']
        curve = json.loads(attempt_path.with_name('curve.json').read_text(encoding='utf-8'))
        assert [row['epoch'] for row in curve] == list(range(result['epochs_completed'] + 1))
        for row in curve:
            assert row['steps'] == row['epoch'] * 128
            if result.get('ed_mode') == 'slow_ed':
                assert row['learning_rates']['ed'] == row['learning_rates']['g'] * 0.1
        selected = min(curve, key=lambda row: row['val_mse'])
        assert selected['epoch'] == result['selected_epoch']
        assert selected['val_mse'] == result['best_val_mse']
        assert result['total_steps'] == result['epochs_completed'] * 128
        schedule_hash = hashlib.sha256()
        for epoch in range(1, result['epochs_completed'] + 1):
            schedule = phase_sample(data['train_origins'], result['seed'], epoch, 512)
            assert np.isin(schedule, data['train_origins']).all()
            schedule_hash.update(schedule.astype('<i8').tobytes())
        assert schedule_hash.hexdigest() == result['schedule_sha256']
        checkpoint = Path(result['checkpoint'])
        assert digest(checkpoint) == result['checkpoint_sha256']
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert saved['epoch'] == result['selected_epoch']
        assert saved['spec']['id'] == result['id']
        assert all(result.get(k) == v and attempt.get(k) == v for k, v in saved['spec'].items())
        assert all(torch.isfinite(p).all() for p in saved['state'].values())
        initial_path = Path(result['initial_checkpoint'])
        assert digest(initial_path) == result['initial_checkpoint_sha256']
        initial = torch.load(initial_path, map_location='cpu', weights_only=False)
        assert initial['spec'] == saved['spec'] and initial['epoch'] == 0 and initial['steps'] == 0
        assert np.array_equal(initial['basis'], saved['basis'])
        hashes = {k: hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k, v in initial['state'].items()}
        assert hashes == result['initial_parameter_sha256']
        group = (dataset, result['latent'], result['seed'])
        if group in initial_groups:
            assert initial_groups[group] == hashes, 'Paired modes/arms must use the same initial adapter state'
        initial_groups[group] = hashes
        selected_updates = {k: float((saved['state'][k] - initial['state'][k]).abs().max()) for k in initial['state']}
        assert selected_updates == result['selected_parameter_max_updates']
        if result.get('ed_mode') == 'fixed_ed':
            assert torch.equal(saved['state']['encoder.weight'], torch.as_tensor(saved['basis'].T))
            assert torch.equal(saved['state']['decoder.weight'], torch.as_tensor(saved['basis']))
            assert result['parameter_max_updates']['encoder.weight'] == 0
            assert result['parameter_max_updates']['decoder.weight'] == 0
        with np.load(checkpoint.with_name('best_val.npz'), allow_pickle=False) as archive:
            origins, prediction = archive['origins'], archive['prediction']
        assert np.array_equal(origins, data['val_origins'])
        target = np.stack([data['x'][o:o + 48] for o in origins])
        mask = np.stack([data['finite'][o:o + 48] for o in origins])
        scores = scalar_channel_scores(prediction, target, mask)
        error = abs(scores['mse'] - result['best_val_mse'])
        assert error < 1e-10, (result['id'], error)
        assert result['replay_error'] <= 1e-6
        rows.append({'id': result['id'], 'scalar_mse': scores['mse'], 'mse_abs_error': error,
                     'checkpoint_sha256': digest(checkpoint), 'selected_epoch': result['selected_epoch'],
                     'schedule_verified': True, 'restore_error': result['replay_error']})
    save_json(output, {'status': 'pass', 'complete_fits': len(rows), 'other_attempts': failed,
                       'required_cohort': str(Path(args.cohort)), 'required_cohort_sha256': digest(Path(args.cohort)),
                       'required_completed_ids': sorted(expected),
                       'v1_preserved_files': len(provenance['v1_tracked_sha256']), 'rows': rows,
                       'scope': 'Execution-agent CPU recomputation and artifact checks; not independent replication.'})
    print(json.dumps({'status': 'pass', 'complete_fits': len(rows), 'other_attempts': failed}))


if __name__ == '__main__':
    main()
