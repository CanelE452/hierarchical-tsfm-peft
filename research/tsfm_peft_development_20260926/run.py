import argparse
import contextlib
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from model import SNAPSHOT, make_model, macro_loss, pca_basis
from protected_protocol import prepare_protected_adaptation

CACHE = ROOT / '.cache/tsfm_peft_development_20260926'
PROTOCOL = json.loads((HERE / 'protocol.json').read_text(encoding='utf-8'))
torch.set_num_threads(4)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    for retry in range(6):
        try:
            tmp.replace(path)
            break
        except PermissionError:
            if retry == 5:
                raise
            if retry == 0:
                print(f'Atomic JSON replacement temporarily denied; bounded retry: {path.name}', file=sys.stderr, flush=True)
            time.sleep(0.05 * 2 ** retry)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def source_receipt():
    return {p.name: digest(p) for p in HERE.glob('*.py')} | {'protocol.json': digest(HERE / 'protocol.json'), 'PLAN.md': digest(HERE / 'PLAN.md')}


class GPUJob:
    def __init__(self, label):
        self.path = HERE / 'gpu_jobs.json'
        self.label = label

    def __enter__(self):
        import psutil
        CACHE.mkdir(parents=True, exist_ok=True)
        self.lock = CACHE / 'gpu.lock'
        if self.lock.exists():
            old = json.loads(self.lock.read_text(encoding='utf-8'))
            if psutil.pid_exists(old['pid']):
                raise RuntimeError(f'GPU job already live: {old}')
            raise RuntimeError('Stale GPU lock: inspect original job before explicit repair')
        jobs = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else []
        if any(j['status'] == 'running' for j in jobs):
            raise RuntimeError('Unclosed GPU ledger entry; inspect process before repair')
        self.used = sum(j['elapsed_s'] for j in jobs)
        if self.used >= PROTOCOL['gpu_budget_seconds']:
            raise RuntimeError('GPU budget exhausted')
        running = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout
        others = [line for line in running.splitlines() if 'python' in line.lower() and line.split(',')[0].strip() != str(os.getpid())]
        if others:
            raise RuntimeError(f'Other GPU Python process; do not terminate: {others}')
        with self.lock.open('x') as f:
            json.dump({'pid': os.getpid(), 'label': self.label}, f)
        self.started = time.perf_counter()
        self.jobs = jobs
        self.record = {'label': self.label, 'pid': os.getpid(), 'started_utc': time.time(), 'status': 'running', 'elapsed_s': 0.0, 'source': source_receipt()}
        jobs.append(self.record)
        self.heartbeat('starting')
        return self

    def heartbeat(self, progress):
        self.record.update(elapsed_s=time.perf_counter() - self.started, heartbeat_utc=time.time(), progress=progress)
        if torch.cuda.is_initialized():
            self.record['peak_allocated_bytes'] = max(self.record.get('peak_allocated_bytes', 0), torch.cuda.max_memory_allocated())
            self.record['peak_scope'] = 'maximum observed allocator high-water across segment resets; excludes reserved memory'
        save_json(self.path, self.jobs)
        if self.used + self.record['elapsed_s'] >= PROTOCOL['gpu_budget_seconds']:
            raise RuntimeError('GPU_TOTAL_BUDGET_REACHED')

    def __exit__(self, kind, exc, tb):
        self.record.update(status='failed' if kind else 'complete', elapsed_s=time.perf_counter() - self.started, ended_utc=time.time())
        if exc:
            self.record['error'] = ''.join(traceback.format_exception(kind, exc, tb))
        if torch.cuda.is_initialized():
            self.record['peak_allocated_bytes'] = max(self.record.get('peak_allocated_bytes', 0), torch.cuda.max_memory_allocated())
        save_json(self.path, self.jobs)
        self.lock.unlink()


def load_data(dataset):
    contract = json.loads((HERE / 'data_contract.json').read_text(encoding='utf-8'))
    key = {'electricity': 'electricity_first32', 'bull': 'bdg2_bull_office'}[dataset]
    path = ROOT / contract['datasets'][key]['npz']
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files}
    data['_path'] = path
    data['finite'] = data['targetmask']
    if dataset == 'electricity':
        data['train_start'], data['train_end'] = data['split_bounds'][:2]
    else:
        periods = contract['datasets'][key]['split_periods']['train']
        data['train_start'], data['train_end'] = [np.searchsorted(data['times'], np.datetime64(t, 'ns').astype(np.int64)) for t in periods]
    return data


def tensors(data, origins):
    x = np.stack([data['x'][o - 512:o] for o in origins])
    y = np.stack([data['x'][o:o + 48] for o in origins])
    mask = np.stack([data['finite'][o:o + 48] for o in origins])
    return [torch.as_tensor(a, device='cuda') for a in (x, y, mask)]


def phase_sample(origins, seed, epoch, n=512):
    rng = np.random.default_rng(np.random.SeedSequence([seed, epoch]))
    groups = [origins[origins % 24 == phase] for phase in range(24)]
    selected = []
    for phase, group in enumerate(groups):
        count = n // 24 + int(phase < n % 24)
        if not len(group):
            raise ValueError(f'no TRAIN origins for phase {phase}')
        selected.extend(rng.choice(group, count, replace=len(group) < count))
    return rng.permutation(selected)


def score_arrays(pred, target, mask):
    n = mask.sum(axis=(0, 1))
    if not np.all(n > 0):
        raise ValueError('no evaluation target for a selected channel')
    squared = (pred.astype(np.float64) - target) ** 2 * mask
    absolute = np.abs(pred.astype(np.float64) - target) * mask
    return {'mse': float(np.mean(squared.sum(axis=(0, 1)) / n)), 'mae': float(np.mean(absolute.sum(axis=(0, 1)) / n)), 'channel_mse': (squared.sum(axis=(0, 1)) / n).tolist(), 'target_counts': n.tolist(), 'origins': len(pred)}


@torch.no_grad()
def evaluate(model, data, origins, job, label):
    preds = []
    model.eval()
    for i in range(0, len(origins), PROTOCOL['eval_batch_origins']):
        x, _, _ = tensors(data, origins[i:i + PROTOCOL['eval_batch_origins']])
        preds.append(model(x).float().cpu().numpy())
        if i % 40 == 0:
            job.heartbeat(f'{label} {i}/{len(origins)}')
    pred = np.concatenate(preds)
    target = np.stack([data['x'][o:o + 48] for o in origins])
    mask = np.stack([data['finite'][o:o + 48] for o in origins])
    return score_arrays(pred, target, mask), pred


def frozen_hash(model):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if not p.requires_grad:
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def backward_batch(model, x, target, mask, loss_kind, microbatch_origins=1):
    count = None
    if loss_kind == 'mse' and microbatch_origins < len(x) and not bool(mask.all()):
        count = mask.sum(dim=(0, 1))
    batch_loss = 0.0
    for start in range(0, len(x), microbatch_origins):
        part = slice(start, start + microbatch_origins)
        if loss_kind == 'native':
            loss = model.native_loss(x[part], target[part], mask[part])
        else:
            loss = macro_loss(model(x[part]), target[part], mask[part], channel_count=count)
        if count is None:
            loss = loss / (len(x) / len(x[part]))
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite training loss')
        loss.backward()
        batch_loss += loss.detach().item()
    return batch_loss


def fit(data, spec, job, protected_receipt=None):
    if spec['dataset'] == 'bull' and protected_receipt is None:
        raise RuntimeError('Bull fit requires a validated and pinned final seal receipt')
    fit_id = spec['id']
    out = HERE / 'runs' / fit_id
    if out.exists():
        raise RuntimeError(f'Never overwrite an attempt: {out}')
    existing = list((HERE / 'runs').glob('*/attempt.json'))
    if len(existing) >= PROTOCOL['fit_attempt_budget']:
        raise RuntimeError('FIT_ATTEMPT_BUDGET_REACHED')
    out.mkdir(parents=True)
    local = CACHE / 'runs' / fit_id
    local.mkdir(parents=True)
    started = time.perf_counter()
    attempt = dict(spec, status='running', started_utc=time.time(), pid=os.getpid(), source=source_receipt(), data_sha256=digest(data['_path']))
    if protected_receipt is not None:
        attempt['protected_receipt'] = protected_receipt
    save_json(out / 'attempt.json', attempt)
    try:
        torch.manual_seed(spec['seed'])
        np.random.seed(spec['seed'])
        rank = spec.get('latent', 8)
        train_rows = data['x'][int(data['train_start']):int(data['train_end'])]
        basis = pca_basis(train_rows, rank)
        model = make_model(spec['arm'], basis, residual_rank=spec.get('residual_rank', 8))
        torch.cuda.reset_peak_memory_stats()
        frozen_before = frozen_hash(model)
        initial = model.adapter_state()
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=spec['lr'], weight_decay=spec.get('weight_decay', 0.0))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
        history = []
        best, best_epoch, stale, steps = float('inf'), 0, 0, 0
        schedule_hash = hashlib.sha256()
        epochs = spec.get('epochs', PROTOCOL['epochs'])
        for epoch in range(epochs + 1):
            train_losses = []
            if epoch:
                schedule = phase_sample(data['train_origins'], spec['seed'], epoch, spec.get('epoch_origins', 512))
                schedule_hash.update(schedule.astype('<i8').tobytes())
                model.train()
                for start in range(0, len(schedule), PROTOCOL['effective_batch_origins']):
                    batch_origins = schedule[start:start + PROTOCOL['effective_batch_origins']]
                    optimizer.zero_grad(set_to_none=True)
                    x, y, mask = tensors(data, batch_origins)
                    loss_kind = 'native' if spec['arm'] == 'lora' and spec.get('loss', 'native') == 'native' else 'mse'
                    batch_loss = backward_batch(model, x, y, mask, loss_kind, spec.get('microbatch_origins', PROTOCOL['microbatch_origins']))
                    norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                    optimizer.step()
                    train_losses.append(batch_loss)
                    steps += 1
                    if steps % 10 == 0:
                        job.heartbeat(f'{fit_id}: epoch {epoch} step {steps}')
            val, pred = evaluate(model, data, data['val_origins'], job, fit_id + ' validation')
            history.append({'epoch': epoch, 'steps': steps, 'train_loss': float(np.mean(train_losses)) if train_losses else None, 'val_mse': val['mse'], 'val_mae': val['mae'], 'lr': optimizer.param_groups[0]['lr'], 'elapsed_s': time.perf_counter() - started})
            if val['mse'] < best:
                best, best_epoch, stale = val['mse'], epoch, 0
                torch.save({'state': model.adapter_state(), 'spec': spec, 'epoch': epoch, 'steps': steps, 'basis': basis, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, local / 'best.pt')
                np.savez_compressed(local / 'best_val.npz', prediction=pred, origins=data['val_origins'])
            else:
                stale += 1
            save_json(out / 'curve.json', history)
            job.heartbeat(f'{fit_id}: epoch {epoch} val_mse {val["mse"]:.6f}')
            print(json.dumps({'fit': fit_id, 'epoch': epoch, 'val_mse': val['mse'], 'best_epoch': best_epoch, 'elapsed_s': round(time.perf_counter() - started, 1)}), flush=True)
            if epoch:
                scheduler.step(val['mse'])
            if epoch and stale >= PROTOCOL['patience']:
                break
        frozen_after = frozen_hash(model)
        assert frozen_before == frozen_after, 'frozen backbone mutated'
        trained = model.adapter_state()
        changes = {k: float((trained[k] - initial[k]).abs().max()) for k in initial}
        saved = torch.load(local / 'best.pt', map_location='cpu', weights_only=False)
        del optimizer, scheduler, params, model
        gc.collect()
        torch.cuda.empty_cache()
        restored = make_model(spec['arm'], basis, residual_rank=spec.get('residual_rank', 8))
        restored.restore_adapter(saved['state'])
        replay_score, replay = evaluate(restored, data, data['val_origins'], job, fit_id + ' checkpoint replay')
        with np.load(local / 'best_val.npz') as archive:
            replay_error = float(np.max(np.abs(replay - archive['prediction'])))
        assert replay_error <= 1e-6, replay_error
        result = dict(attempt, status='complete', epochs_completed=epoch, selected_epoch=best_epoch, selected_steps=saved['steps'], total_steps=steps, best_val_mse=best, replay_error=replay_error, frozen_sha256=frozen_before, schedule_sha256=schedule_hash.hexdigest(), trainable_parameters=sum(v.numel() for v in initial.values()), parameter_max_updates=changes, checkpoint=str(local / 'best.pt'), checkpoint_sha256=digest(local / 'best.pt'), at_improving_boundary=(epoch == epochs and best_epoch == epochs), elapsed_s=time.perf_counter() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated())
        save_json(out / 'result.json', result)
        attempt.update(status='complete', elapsed_s=result['elapsed_s'])
        save_json(out / 'attempt.json', attempt)
        del restored
        gc.collect()
        torch.cuda.empty_cache()
    except BaseException:
        attempt.update(status='failed', elapsed_s=time.perf_counter() - started, error=traceback.format_exc())
        save_json(out / 'attempt.json', attempt)
        raise


def initial_specs():
    result = []
    for seed in PROTOCOL['seeds']:
        for arm in PROTOCOL['arms']:
            for lr in PROTOCOL['lora_lrs' if arm == 'lora' else 'adapter_lrs']:
                result.append({'id': f'initial_{arm}_{seed}_{lr:g}', 'dataset': 'electricity', 'arm': arm, 'seed': seed, 'lr': lr, 'latent': 8, 'loss': 'native' if arm == 'lora' else 'mse'})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['initial', 'fits', 'evaluate'])
    parser.add_argument('--specs')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--dataset', default='electricity')
    args = parser.parse_args()
    if args.command == 'initial':
        specs = initial_specs()
        path = HERE / 'initial_specs.json'
        if path.exists():
            assert json.loads(path.read_text(encoding='utf-8')) == specs
        else:
            save_json(path, specs)
    elif args.command == 'fits':
        specs = json.loads(Path(args.specs).read_text(encoding='utf-8'))
    else:
        raise ValueError('Use a sealed evaluation script; no generic protected-scoring command')
    receipts = prepare_protected_adaptation(specs, HERE, ROOT)
    pending = []
    for spec in specs:
        result = HERE / 'runs' / spec['id'] / 'result.json'
        if result.exists():
            assert json.loads(result.read_text(encoding='utf-8'))['status'] == 'complete'
        else:
            pending.append(spec)
    if args.limit:
        pending = pending[:args.limit]
    if not pending:
        print('No pending fits; no GPU job started.')
        return
    datasets = {spec['dataset']: load_data(spec['dataset']) for spec in pending}
    with GPUJob(args.command) as job:
        for spec in pending:
            fit(datasets[spec['dataset']], spec, job, protected_receipt=receipts.get(spec['id']))


if __name__ == '__main__':
    main()
