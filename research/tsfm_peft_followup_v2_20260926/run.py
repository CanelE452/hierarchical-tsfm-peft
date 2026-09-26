import argparse
import contextlib
import gc
import hashlib
import json
import importlib.metadata
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

V1 = ROOT / 'research/tsfm_peft_development_20260926'
CACHE = ROOT / '.cache/tsfm_peft_followup_v2_20260926'
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
    files = list(HERE.glob('*.py')) + list(V1.glob('*.py'))
    files += [HERE / 'protocol.json', HERE / 'PLAN.md', V1 / 'data_contract.json',
              ROOT / 'src/hier_peft/lora.py']
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in files}


def environment_receipt():
    names = ['torch', 'chronos-forecasting', 'peft', 'transformers', 'numpy']
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'python': sys.version, 'executable': sys.executable, 'command': sys.argv,
            'cwd': str(Path.cwd()), 'versions': versions}


def storage_bytes():
    return sum(p.stat().st_size for p in CACHE.rglob('*') if p.is_file()) if CACHE.exists() else 0


def require_storage(reserve=0):
    used = storage_bytes()
    if used + reserve > PROTOCOL['storage_budget_bytes']:
        raise RuntimeError('V2_STORAGE_BUDGET_REACHED')
    return used


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
        require_storage()
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
        try:
            self.heartbeat('starting')
        except BaseException:
            self.__exit__(*sys.exc_info())
            raise
        return self

    def check_limits(self):
        if self.used + time.perf_counter() - self.started >= PROTOCOL['gpu_budget_seconds']:
            raise RuntimeError('GPU_TOTAL_BUDGET_REACHED')

    def heartbeat(self, progress):
        self.record.update(elapsed_s=time.perf_counter() - self.started, heartbeat_utc=time.time(), progress=progress)
        if torch.cuda.is_initialized():
            self.record['peak_allocated_bytes'] = max(self.record.get('peak_allocated_bytes', 0), torch.cuda.max_memory_allocated())
            self.record['peak_scope'] = 'maximum observed allocator high-water across segment resets; excludes reserved memory'
        save_json(self.path, self.jobs)
        require_storage()
        self.check_limits()

    def __exit__(self, kind, exc, tb):
        self.record.update(status='failed' if kind else 'complete', elapsed_s=time.perf_counter() - self.started, ended_utc=time.time())
        if exc:
            self.record['error'] = ''.join(traceback.format_exception(kind, exc, tb))
        if torch.cuda.is_initialized():
            self.record['peak_allocated_bytes'] = max(self.record.get('peak_allocated_bytes', 0), torch.cuda.max_memory_allocated())
        save_json(self.path, self.jobs)
        if self.lock.exists():
            owner = json.loads(self.lock.read_text(encoding='utf-8'))
            if owner['pid'] == os.getpid():
                self.lock.unlink()


def load_data(dataset):
    contract = json.loads((V1 / 'data_contract.json').read_text(encoding='utf-8'))
    key = {'electricity': 'electricity_first32', 'bull': 'bdg2_bull_office'}[dataset]
    path = ROOT / contract['datasets'][key]['npz']
    expected = PROTOCOL['data_sha256'][dataset]
    if digest(path) != expected:
        raise ValueError('v1 data identity changed; preserve the approved data contract')
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


def tensors(data, origins, device='cuda'):
    x = np.stack([data['x'][o - 512:o] for o in origins])
    y = np.stack([data['x'][o:o + 48] for o in origins])
    mask = np.stack([data['finite'][o:o + 48] for o in origins])
    return [torch.as_tensor(a, device=device) for a in (x, y, mask)]


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
        job.check_limits()
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


def optimizer_for(model, spec):
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if spec.get('ed_mode', 'current') == 'slow_ed':
        ed = [p for name, p in trainable if name.startswith(('encoder.', 'decoder.'))]
        rest = [p for name, p in trainable if not name.startswith(('encoder.', 'decoder.'))]
        if not ed or not rest or spec['ed_lr'] != spec['lr'] * 0.1:
            raise ValueError('SLOW_ED needs encoder/decoder and G groups at a 0.1 LR ratio')
        groups = [{'params': rest, 'lr': spec['lr'], 'name': 'g'},
                  {'params': ed, 'lr': spec['ed_lr'], 'name': 'ed'}]
    else:
        groups = [{'params': [p for _, p in trainable], 'lr': spec['lr'], 'name': 'trainable'}]
    optimizer = torch.optim.AdamW(groups, lr=spec['lr'], weight_decay=spec.get('weight_decay', 0.0))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    return optimizer, scheduler


def step_scheduler(scheduler, value, ed_mode):
    scheduler.step(value)
    if ed_mode == 'slow_ed':
        groups = scheduler.optimizer.param_groups
        groups[1]['lr'] = groups[0]['lr'] * 0.1
        scheduler._last_lr = [group['lr'] for group in groups]


def initial_path_check(model, x, y, mask):
    model.eval()
    main, correction = model.components(x)
    result = {'initial_correction_max': float(correction.detach().abs().max())}
    if result['initial_correction_max'] != 0:
        raise RuntimeError('G output must start at zero')
    direct = model.decoder(model.backbone_point(model.encoder(x)))
    difference = float((main.detach() - direct.detach()).abs().max())
    if difference > 1e-6:
        raise RuntimeError('initial compressed functions differ')
    result['initial_compress_error'] = difference
    if model.ed_mode != 'fixed_ed':
        grad = torch.autograd.grad(main.square().mean(), model.encoder.weight)[0]
        magnitude = float(grad.detach().abs().max())
        if not torch.isfinite(grad).all() or magnitude == 0:
            raise RuntimeError('encoder gradient through the frozen backbone is missing')
        result['encoder_backbone_gradient_max'] = magnitude
    else:
        result['frozen_ed'] = all(not p.requires_grad for layer in [model.encoder, model.decoder] for p in layer.parameters())
        normal = model.components(x, fixed_shortcut=False)
        a = macro_loss(main + correction, y, mask)
        b = macro_loss(normal[0] + normal[1], y, mask)
        params = list(model.residual.parameters())
        ga, gb = torch.autograd.grad(a, params), torch.autograd.grad(b, params)
        parity = max(float((u - v).abs().max()) for u, v in zip(ga, gb))
        if not torch.allclose(a, b, atol=1e-6, rtol=1e-6) or parity > 1e-6:
            raise RuntimeError('FIXED_ED shortcut changes loss or G gradient')
        result['fixed_shortcut_gradient_error'] = parity
    model.zero_grad(set_to_none=True)
    return result


def validate_spec(spec):
    if spec['dataset'] not in ('electricity', 'bull') or spec['arm'] not in ('residual', 'raw_bypass'):
        raise ValueError('this runner supports approved residual and matched RAW development fits')
    if spec.get('ed_mode') not in ('current', 'fixed_ed', 'slow_ed'):
        raise ValueError('invalid E/D mode')
    if spec['seed'] not in PROTOCOL['seeds'] or spec.get('residual_rank') != 32:
        raise ValueError('unapproved seed or temporal rank')
    if spec.get('epochs') not in (120, 240) or (spec['epochs'] == 240 and not spec.get('decision_record')):
        raise ValueError('240-epoch extension requires a recorded conditional decision')
    allowed_latents = {'electricity': (8,), 'bull': (4, 8)}
    if spec['latent'] not in allowed_latents[spec['dataset']]:
        raise ValueError('unapproved latent count')
    if spec['latent'] == 8 and spec['dataset'] == 'bull' and not spec.get('decision_record'):
        raise ValueError('Bull K8 requires a recorded conditional decision')
    if spec.get('lr') != 0.001 or spec.get('loss') != 'mse' or spec.get('microbatch_origins') != 4:
        raise ValueError('first-round recipe must preserve G LR, loss and microbatch')
    if not spec['id'] or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in spec['id']):
        raise ValueError('unsafe attempt id')


def fit(data, spec, job):
    validate_spec(spec)
    fit_id = spec['id']
    out = HERE / 'runs' / fit_id
    local = CACHE / 'runs' / fit_id
    if out.exists() or local.exists():
        raise RuntimeError(f'Never overwrite an attempt: {out}')
    existing = list((HERE / 'runs').glob('*/attempt.json'))
    if len(existing) >= PROTOCOL['fit_attempt_budget']:
        raise RuntimeError('FIT_ATTEMPT_BUDGET_REACHED')
    out.mkdir(parents=True)
    local.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    attempt = dict(spec, status='running', started_utc=time.time(), pid=os.getpid(), source=source_receipt(), data_sha256=digest(data['_path']), environment=environment_receipt(), scope='v2 exposed development; TRAIN updates, VAL selection only')
    save_json(out / 'attempt.json', attempt)
    try:
        torch.manual_seed(spec['seed'])
        np.random.seed(spec['seed'])
        rank = spec.get('latent', 8)
        train_rows = data['x'][int(data['train_start']):int(data['train_end'])]
        basis = pca_basis(train_rows, rank)
        model = make_model(spec['arm'], basis, residual_rank=spec['residual_rank'], ed_mode=spec['ed_mode'])
        torch.cuda.reset_peak_memory_stats()
        frozen_before = frozen_hash(model)
        initial = model.adapter_state()
        initial_hashes = {k: hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k, v in initial.items()}
        paired_initial_ids = []
        for previous_path in (HERE / 'runs').glob('*/result.json'):
            previous = json.loads(previous_path.read_text(encoding='utf-8'))
            keys = ('dataset', 'seed', 'arm', 'latent', 'residual_rank')
            if all(previous.get(key) == spec.get(key) for key in keys):
                if previous['initial_parameter_sha256'] != initial_hashes:
                    raise RuntimeError('paired initial adapter weights differ across E/D modes')
                paired_initial_ids.append(previous['id'])
        require_storage(32 * 1024 * 1024)
        torch.save({'state': initial, 'spec': spec, 'basis': basis, 'epoch': 0, 'steps': 0}, local / 'initial.pt')
        probe_origins = phase_sample(data['train_origins'], spec['seed'], 0, 96)
        x, y, mask = tensors(data, probe_origins[:4])
        path_check = initial_path_check(model, x, y, mask)
        del x, y, mask
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer, scheduler = optimizer_for(model, spec)
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
                    job.check_limits()
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
            history.append({'epoch': epoch, 'steps': steps, 'train_loss': float(np.mean(train_losses)) if train_losses else None, 'val_mse': val['mse'], 'val_mae': val['mae'], 'lr': optimizer.param_groups[0]['lr'], 'learning_rates': {g['name']: g['lr'] for g in optimizer.param_groups}, 'elapsed_s': time.perf_counter() - started})
            if val['mse'] < best:
                best, best_epoch, stale = val['mse'], epoch, 0
                require_storage(32 * 1024 * 1024)
                torch.save({'state': model.adapter_state(), 'spec': spec, 'epoch': epoch, 'steps': steps, 'basis': basis, 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict()}, local / 'best.pt')
                np.savez_compressed(local / 'best_val.npz', prediction=pred, origins=data['val_origins'])
            else:
                stale += 1
            save_json(out / 'curve.json', history)
            job.heartbeat(f'{fit_id}: epoch {epoch} val_mse {val["mse"]:.6f}')
            print(json.dumps({'fit': fit_id, 'epoch': epoch, 'val_mse': val['mse'], 'best_epoch': best_epoch, 'elapsed_s': round(time.perf_counter() - started, 1)}), flush=True)
            if epoch:
                step_scheduler(scheduler, val['mse'], spec['ed_mode'])
            if epoch and stale >= PROTOCOL['patience']:
                break
        frozen_after = frozen_hash(model)
        assert frozen_before == frozen_after, 'frozen backbone mutated'
        trained = model.adapter_state()
        changes = {k: float((trained[k] - initial[k]).abs().max()) for k in initial}
        if spec['ed_mode'] == 'fixed_ed' and any(changes[k] != 0 for k in ('encoder.weight', 'decoder.weight')):
            raise RuntimeError('FIXED_ED weights changed')
        if not any(changes[k] > 0 for k in changes if k.startswith('residual.')):
            raise RuntimeError('G parameters did not update during this fit')
        if spec['ed_mode'] != 'fixed_ed' and any(changes[k] == 0 for k in ('encoder.weight', 'decoder.weight')):
            raise RuntimeError('intended encoder or decoder parameters did not update')
        saved = torch.load(local / 'best.pt', map_location='cpu', weights_only=False)
        del optimizer, scheduler, params, model
        gc.collect()
        torch.cuda.empty_cache()
        restored = make_model(spec['arm'], basis, residual_rank=spec['residual_rank'], ed_mode=spec['ed_mode'])
        restored.restore_adapter(saved['state'])
        replay_score, replay = evaluate(restored, data, data['val_origins'], job, fit_id + ' checkpoint replay')
        with np.load(local / 'best_val.npz') as archive:
            replay_error = float(np.max(np.abs(replay - archive['prediction'])))
        assert replay_error <= 1e-6, replay_error
        result = dict(attempt, status='complete', epochs_completed=epoch, selected_epoch=best_epoch, selected_steps=saved['steps'], total_steps=steps, best_val_mse=best, replay_error=replay_error, frozen_sha256=frozen_before, schedule_sha256=schedule_hash.hexdigest(), trainable_parameters=sum(p.numel() for p in restored.parameters() if p.requires_grad), adapter_state_parameters=sum(v.numel() for v in initial.values()), parameter_max_updates=changes, selected_parameter_max_updates={k: float((saved['state'][k] - initial[k]).abs().max()) for k in initial}, initial_parameter_sha256=initial_hashes, paired_initial_ids=paired_initial_ids, initial_checkpoint=str(local / 'initial.pt'), initial_checkpoint_sha256=digest(local / 'initial.pt'), initial_path_check=path_check, checkpoint=str(local / 'best.pt'), checkpoint_sha256=digest(local / 'best.pt'), at_improving_boundary=(epoch == epochs and best_epoch == epochs), elapsed_s=time.perf_counter() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated())
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['fits'])
    parser.add_argument('--specs', required=True)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    specs = json.loads(Path(args.specs).read_text(encoding='utf-8'))
    if len({s['id'] for s in specs}) != len(specs):
        raise ValueError('duplicate run ids')
    for spec in specs:
        validate_spec(spec)
    pending = []
    for spec in specs:
        result = HERE / 'runs' / spec['id'] / 'result.json'
        if result.exists():
            completed = json.loads(result.read_text(encoding='utf-8'))
            if completed['status'] != 'complete' or any(completed.get(k) != v for k, v in spec.items()):
                raise ValueError('existing result does not match requested run')
        else:
            pending.append(spec)
    if args.limit:
        pending = pending[:args.limit]
    if not pending:
        print('No pending fits; no GPU job started.')
        return
    attempted = len(list((HERE / 'runs').glob('*/attempt.json')))
    if attempted + len(pending) > PROTOCOL['fit_attempt_budget']:
        raise RuntimeError('pending cohort exceeds the remaining attempt budget')
    datasets = {spec['dataset']: load_data(spec['dataset']) for spec in pending}
    with GPUJob(args.command) as job:
        for spec in pending:
            fit(datasets[spec['dataset']], spec, job)


if __name__ == '__main__':
    main()
