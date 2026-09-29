"""Read-only model replay on the original TRAIN probe and VAL; no fitting."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V9 = ROOT / 'research/tsfm_peft_staged_p_v9_20260929'
CACHE = ROOT / '.cache/tsfm_peft_followup_diagnosis_20260929'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


class DiagnosticJob:
    def __init__(self, name):
        self.path = HERE / (name + '_receipt.json')
        if self.path.exists():
            raise RuntimeError('Preserve previous attempt; use a new label')
        if (HERE / 'diagnostic_partial.json').exists() or (HERE / 'diagnostic.json').exists():
            raise RuntimeError('Preserve existing outputs; a new label alone is not a new output directory')
        old = [json.loads(p.read_text(encoding='utf-8')) for p in HERE.glob('*_receipt.json')]
        if any(r['status'] == 'running' for r in old):
            raise RuntimeError('Inspect active PID before any retry')
        self.remaining = 300 - sum(r.get('elapsed_s', 0) for r in old)
        if self.remaining <= 0:
            raise RuntimeError('Diagnostic GPU budget exhausted')
        processes = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                    '--format=csv,noheader'], capture_output=True, text=True, check=True)
        if any('python' in line.lower() for line in processes.stdout.splitlines()):
            raise RuntimeError('Another GPU Python process is active; leave it alone')
        self.started = time.perf_counter()
        self.record = {'status': 'running', 'label': name, 'pid': os.getpid(), 'category': 'gpu_inference_diagnostic',
                       'fit_attempts': 0, 'optimizer_updates': 0, 'test_access': False,
                       'reserved_seconds': self.remaining, 'started_utc': time.time(),
                       'source_sha256': digest(__file__), 'plan_sha256': digest(HERE / 'PLAN.md'),
                       'command': sys.argv, 'python': sys.executable}
        save(self.path, self.record)

    def check_limits(self):
        if time.perf_counter() - self.started >= self.remaining:
            raise RuntimeError('Diagnostic GPU time limit')
        used = sum(p.stat().st_size for base in (HERE, CACHE) if base.exists() for p in base.rglob('*') if p.is_file())
        if used > 100 * 1024**2:
            raise RuntimeError('Diagnostic storage limit')

    def heartbeat(self, label):
        self.check_limits()
        self.record.update(progress=label, elapsed_s=time.perf_counter() - self.started)
        save(self.path, self.record)
        print(label, flush=True)

    def finish(self, error=None):
        self.record.update(status='failed' if error else 'complete', elapsed_s=time.perf_counter() - self.started,
                           ended_utc=time.time(), error=error)
        save(self.path, self.record)


def run(job):
    sys.path.insert(0, str(V9))
    import numpy as np
    import torch
    from model_v9 import restore_model
    from runtime_v9 import load_data, phase_sample
    from train_v9 import configure, evaluate, array_parity
    configure()
    CACHE.mkdir(parents=True, exist_ok=True)
    data = load_data('robin', include_test=False)
    untouched = {p: digest(p) for p in V9.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    rows = []
    for family in ('staged_p', 'staged_raw'):
        for seed in (92601, 92602):
            run_id = f'v9_robin_{family}_{seed}'
            result_path = V9 / 'runs' / run_id / 'result.json'
            result = json.loads(result_path.read_text(encoding='utf-8'))
            local = Path(result['checkpoint']).parent
            origins_path = local / 'origins.npz'
            with np.load(origins_path, allow_pickle=False) as saved:
                probe = saved['train_probe'].copy()
                assert np.array_equal(saved['val'], data['val_origins'])
            assert np.array_equal(probe, phase_sample(data['train_origins'], seed, 0, 96, period=24))
            trained = result['best_trained']
            entries = [('initial', result['initial_checkpoint'], result['initial_checkpoint_sha256']),
                       ('best_trained', trained['checkpoint'], trained['checkpoint_sha256']),
                       ('final', result['restart_checkpoint'], result['restart_sha256'])]
            for name, path, expected_hash in entries:
                job.check_limits()
                assert digest(path) == expected_hash
                model = restore_model(path, device='cuda')
                model.set_alpha(1.)
                scores = {}
                predictions = {}
                for split, origins in [('train_probe', probe), ('val', data['val_origins'])]:
                    score, prediction, _ = evaluate(model, data, origins, job, f'{run_id} {name} {split}')
                    scores[split] = score
                    predictions[split] = prediction
                checks = {}
                if name == 'initial':
                    checks['train_replay_abs'] = abs(scores['train_probe']['mse'] - result['initial_train_probe']['mse'])
                    checks['val_replay_abs'] = abs(scores['val']['mse'] - result['initial_val_mse'])
                    assert max(checks.values()) <= 1e-8
                if name == 'best_trained':
                    receipt = trained['best_val_prediction']
                    assert digest(receipt['path']) == receipt['sha256']
                    with np.load(receipt['path'], allow_pickle=False) as saved:
                        assert np.array_equal(saved['origins'], data['val_origins'])
                        checks['prediction_replay'] = array_parity(predictions['val'], saved['prediction'], atol=1e-6, rtol=0)
                    assert checks['prediction_replay']['pass']
                    assert abs(scores['val']['mse'] - trained['val_mse']) <= 1e-8
                archive = CACHE / (run_id + '_' + name + '.npz')
                np.savez_compressed(archive, **predictions, train_probe_origins=probe, val_origins=data['val_origins'])
                saved_state = torch.load(path, map_location='cpu', weights_only=False)
                rows.append({'run': run_id, 'family': family, 'seed': seed, 'checkpoint_role': name,
                             'checkpoint': str(path), 'checkpoint_sha256': expected_hash, 'epoch': saved_state['epoch'],
                             'alpha': 1., 'scores': scores, 'checks': checks,
                             'probe_source': str(origins_path), 'probe_sha256': digest(origins_path),
                             'prediction': {'path': str(archive), 'sha256': digest(archive)},
                             'source_result_sha256': digest(result_path)})
                assert digest(path) == expected_hash
                del model, saved_state, predictions
                gc.collect()
                torch.cuda.empty_cache()
                save(HERE / 'diagnostic_partial.json', rows)
    assert all(p.exists() and digest(p) == sha for p, sha in untouched.items())
    summaries = []
    for family in ('staged_p', 'staged_raw'):
        for seed in (92601, 92602):
            group = {r['checkpoint_role']: r for r in rows if r['family'] == family and r['seed'] == seed}
            for name in ('best_trained', 'final'):
                change = {split: group[name]['scores'][split]['mse'] - group['initial']['scores'][split]['mse']
                          for split in ('train_probe', 'val')}
                summaries.append({'family': family, 'seed': seed, 'checkpoint_role': name, 'mse_change': change})
    save(HERE / 'diagnostic.json', {'status': 'complete', 'data_path': str(data['_path']),
                                 'data_sha256': digest(data['_path']), 'old_public_files_unchanged': len(untouched),
                                 'fit_attempts': 0, 'optimizer_updates': 0, 'test_access': False,
                                 'rows': rows, 'summary': summaries})
    print(json.dumps(summaries), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True)
    args = parser.parse_args()
    job = DiagnosticJob(args.job)
    try:
        run(job)
    except BaseException:
        job.finish(traceback.format_exc())
        raise
    else:
        job.finish()
