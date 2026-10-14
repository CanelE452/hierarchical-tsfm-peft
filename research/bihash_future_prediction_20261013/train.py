"""Matched FIT-only optimization and TUNE-only model selection."""
import math
import time
import numpy as np
import torch
import model as bm

SEEDS = (92601, 92602, 92603)
CHECKPOINTS = (0, 100, 200, 400, 600)
WEIGHTS = (.03, .1)
PROBE_METHODS = ('NO_ALIGN', 'B_GUMBEL', 'B_COSINE', 'B_UNIFORM', 'A_BIHASH_GUMBEL', 'A_CONTINUOUS')


def batch_order(origins, seed, updates=600, batch_size=8, horizon=48):
    origins = np.asarray(origins, dtype=np.int64)
    sorted_origins = np.sort(np.unique(origins))
    possible = []
    for origin in sorted_origins:
        if not possible or origin - possible[-1] >= horizon:
            possible.append(origin)
    if len(possible) < batch_size:
        raise ValueError('INSUFFICIENT_NONOVERLAPPING_FIT_ORIGINS')
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(updates):
        chosen = []
        for _ in range(64):
            chosen = []
            for index in rng.permutation(len(origins)):
                if all(abs(int(origins[index]) - int(origins[other])) >= horizon for other in chosen):
                    chosen.append(int(index))
                    if len(chosen) == batch_size:
                        break
            if len(chosen) == batch_size:
                break
        if len(chosen) != batch_size:
            # The maximum-cardinality interval schedule is deterministic and exists.
            selected = rng.choice(np.asarray(possible), size=batch_size, replace=False)
            chosen = [int(np.flatnonzero(origins == origin)[0]) for origin in selected]
        result.append(chosen)
    return np.asarray(result, dtype=np.int64)


def select_methods(cases, expected_groups=6):
    baseline = {(c['dataset'], c['seed']): c['selected_tune_mse'] for c in cases if c['method'] == 'NO_ALIGN'}
    if len(baseline) != expected_groups:
        raise ValueError('INCOMPLETE_BASELINE_SELECTION_GROUPS')
    choices = {}
    for case in cases:
        method, weight = case['method'], case['weight']
        if method == 'NO_ALIGN':
            continue
        choices.setdefault((method, weight), {})[(case['dataset'], case['seed'])] = case['selected_tune_mse']
    records = []
    for (method, weight), values in choices.items():
        if set(values) != set(baseline):
            raise ValueError('INCOMPLETE_METHOD_SELECTION_GROUPS: ' + method)
        ratios = [{'dataset': ds, 'seed': seed, 'tune_mse': values[(ds, seed)],
            'baseline_tune_mse': baseline[(ds, seed)], 'ratio': values[(ds, seed)] / baseline[(ds, seed)]}
            for ds, seed in sorted(baseline)]
        score = float(np.mean([np.log(r['ratio']) for r in ratios]))
        records.append({'method': method, 'weight': weight, 'mean_log_tune_ratio': score, 'groups': ratios})
    weights = {'NO_ALIGN': 0.}
    for method in sorted({m for m, _ in choices}):
        selected = min((r for r in records if r['method'] == method), key=lambda r: (r['mean_log_tune_ratio'], r['weight']))
        weights[method] = selected['weight']
    b_choices = [r for r in records if r['method'].startswith('B_')]
    if not b_choices:
        raise ValueError('INCOMPLETE_B_SELECTION')
    winner = min(b_choices, key=lambda r: (r['mean_log_tune_ratio'], bm.METHODS.index(r['method'])))
    return {'weights': weights, 'selected_B_method': winner['method'], 'selection_records': records,
        'rule': 'Equal dataset-seed groups, mean log TUNE MSE ratio to NO_ALIGN; no DEV_VAL/EVAL selection',
        'expected_groups': expected_groups}


def _tensor(value, device):
    return torch.as_tensor(value, dtype=torch.float32, device=device)


def _boolean(value, device):
    return torch.as_tensor(value, dtype=torch.bool, device=device)


def make_model(pack, seed, method, pretrain_state=None, device='cuda'):
    bank = pack['bank']
    model = bm.FutureBiHash(pack['pca_mean'], pack['basis'], seed, method,
        bank['delta'], bank['observed'], bank['origins'], horizon=48, representation=32, bits=32).to(device)
    if pretrain_state is not None:
        model.load_future_pretrain(pretrain_state)
    return model


def predict(model, history, origins, forecaster, limit=None, batch=8, variant=None):
    result = []
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for begin in range(0, len(history), batch):
            x = _tensor(history[begin:begin + batch], device)
            o = torch.as_tensor(origins[begin:begin + batch], dtype=torch.long, device=device)
            prediction = model(x, o, forecaster, variant=variant)
            result.append(prediction.cpu().numpy())
            if limit is not None: limit.check()
    return np.concatenate(result)


def scores(pack, prediction, split):
    return bm.en.metrics(pack, prediction, pack['target'][split], pack['observed'][split])


def prediction_artifact(rt, path, pack, predictions):
    arrays = {'mean': pack['mean'], 'std': pack['std']}
    for split, prediction in predictions.items():
        arrays.update({split + '_prediction': prediction, split + '_target': pack['target'][split],
            split + '_observed': pack['observed'][split], split + '_origins': pack['origins'][split]})
    return rt.save_npz(path, **arrays)


def pretrain_future(pack, seed, folder, rt, limit, updates=400, device='cuda'):
    started = time.perf_counter()
    channels = len(pack['mean'])
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        model = bm.FutureAutoencoder(channels).to(device)
    x = pack['history']['fit']
    delta = np.where(pack['observed']['fit'], pack['target']['fit'] - x[:, -1:], 0.).astype(np.float32)
    order = np.random.default_rng(seed + 10000).integers(0, len(x), size=(updates, 64))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0.)
    initial_hash = rt.state_hash(model)
    curve = []
    for update, indices in enumerate(order, 1):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.reconstruction_loss(_tensor(delta[indices], device), _boolean(pack['observed']['fit'][indices], device))
        if not torch.isfinite(loss): raise ValueError('FUTURE_AE_NONFINITE_LOSS')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm): raise ValueError('FUTURE_AE_NONFINITE_GRADIENT')
        optimizer.step()
        curve.append({'update': update, 'loss': float(loss.detach()), 'gradient_norm': float(norm)})
        if update % 20 == 0: limit.check()
    model.eval()
    diagnostic = {}
    with torch.no_grad():
        for split in ('fit', 'tune'):
            current_delta = np.where(pack['observed'][split], pack['target'][split] - pack['history'][split][:, -1:], 0.).astype(np.float32)
            reconstructed = []
            representations = []
            for begin in range(0, len(current_delta), 64):
                pred, latent = model(_tensor(current_delta[begin:begin + 64], device), _boolean(pack['observed'][split][begin:begin + 64], device))
                reconstructed.append(pred.cpu().numpy()); representations.append(latent.cpu().numpy())
            reconstructed, representations = np.concatenate(reconstructed), np.concatenate(representations)
            diagnostic[split] = {'delta_reconstruction': bm.en.ex.score(reconstructed, current_delta, pack['observed'][split]),
                'representation_variance': representations.var(0).tolist(), 'representation_rank': int(np.linalg.matrix_rank(representations - representations.mean(0)))}
    artifact = rt.save(folder / f'future_AE_u{updates}.pt', {'future_autoencoder': rt.clone_state(model), 'optimizer': optimizer.state_dict(),
        'seed': seed, 'update': updates, 'initial_hash': initial_hash})
    record = {'dataset': pack['dataset'], 'seed': seed, 'updates': updates, 'batch_size': 64, 'optimizer': 'AdamW',
        'lr': .001, 'weight_decay': 0., 'clip': 1., 'selection': f'fixed final {updates}, TUNE diagnostic only',
        'target': 'observed FIT future normalized Y minus causally filled last state; values and missingness mask',
        'initial_hash': initial_hash, 'final_hash': rt.state_hash(model), 'checkpoint': artifact,
        'order_sha256': bm.en.ex.ahash(order), 'curve': curve, 'diagnostics': diagnostic, 'wall_seconds': time.perf_counter() - started}
    return rt.clone_state(model), record


def _parameter_norm(model, prefix):
    squared = [p.grad.detach().square().sum() for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
    return float(torch.sqrt(sum(squared))) if squared else 0.


def diagnostics(model, pack, split, forecaster, limit=None):
    model.eval()
    device = next(model.parameters()).device
    sample = min(8, len(pack['history'][split]))
    x = _tensor(pack['history'][split][:sample], device)
    origins = torch.as_tensor(pack['origins'][split][:sample], device=device)
    with torch.no_grad():
        prediction, details = model(x, origins, forecaster, return_details=True)
        delta = _tensor(np.where(pack['observed'][split], pack['target'][split] - pack['history'][split][:, -1:], 0.)[:sample], device)
        mask = _boolean(pack['observed'][split][:sample], device)
        reconstruction, future = model.future_autoencoder(delta, mask)
        z = details['z'].reshape(-1, model.k).cpu().numpy()
        latent = future.cpu().numpy()
        result = {'origins': pack['origins'][split][:sample].tolist(), 'sample_windows': sample,
            'future_rec_mse': float(bm.masked_macro_loss(reconstruction, delta, mask)),
            'E_variance': z.var(0).tolist(), 'E_rank': int(np.linalg.matrix_rank(z - z.mean(0))),
            'future_phi_variance': latent.var(0).tolist(), 'future_phi_rank': int(np.linalg.matrix_rank(latent - latent.mean(0)))}
        if model.method.startswith('B_'):
            a = details['attention']; eligible = details['eligible']
            positive = a > 0
            result.update({'eligible_counts_first_last': torch.stack((eligible[:, 0].sum(-1), eligible[:, -1].sum(-1)), -1).cpu().tolist(),
                'forbidden_attention_max_abs': float(a.masked_select(~eligible).abs().max()) if (~eligible).any() else 0.,
                'retrieval_entropy_last': (-torch.where(a[:, -1] > 0, a[:, -1] * a[:, -1].clamp_min(1e-30).log(), 0.).sum(-1)).cpu().tolist(),
                'alpha': float(details['alpha']), 'student_correction_rms': float(details['student_correction'].square().mean().sqrt()),
                'retrieval_top5_memory_origins': model.bank_origins[a[:, -1].topk(min(5, a.shape[-1]), dim=-1).indices].cpu().tolist(),
                'retrieval_top5_weights': a[:, -1].topk(min(5, a.shape[-1]), dim=-1).values.cpu().tolist()})
            memory_phi = details['memory'].cpu().numpy()
            result['memory_phi_variance'] = memory_phi.var(0).tolist()
            result['memory_phi_rank'] = int(np.linalg.matrix_rank(memory_phi - memory_phi.mean(0)))
            codes = details['query_code'].reshape(-1, details['query_code'].shape[-1])
            result['query_code_variance'] = codes.var(0, unbiased=False).cpu().tolist()
            if model.method not in ('B_COSINE', 'B_UNIFORM'):
                balance = (codes > 0).float().mean(0)
                result['binary_bit_positive_fraction'] = balance.cpu().tolist()
                result['binary_bit_entropy'] = (-balance * balance.clamp_min(1e-30).log() - (1 - balance) * (1 - balance).clamp_min(1e-30).log()).cpu().tolist()
                result['fully_constant_binary_bits'] = int(((balance == 0) | (balance == 1)).sum())
        if model.method.startswith('A_'):
            rows = model.student_projection(details['z'])
            student = rows[:, -1] + rows.mean(1)
            q, p = model._codes(student, future, 600)
            logits = torch.nn.functional.normalize(q, dim=-1, eps=1e-6) @ torch.nn.functional.normalize(p, dim=-1, eps=1e-6).T / .2
            labels = torch.arange(len(x), device=device)
            result['paired_current_future_CE_eval'] = float((torch.nn.functional.cross_entropy(logits, labels) + torch.nn.functional.cross_entropy(logits.T, labels)) / (2 * math.log(len(x))))
            result['student_code_variance'] = q.var(0, unbiased=False).cpu().tolist()
    if limit is not None: limit.check()
    return result


def train_case(pack, seed, method, weight, pretrain_state, pretrain_record, forecaster, folder, rt, limit, updates=600, device='cuda'):
    started = time.perf_counter()
    model = make_model(pack, seed, method, pretrain_state, device)
    initial_hash, initial_metadata = rt.state_hash(model), model.metadata()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0.)
    order = batch_order(pack['origins']['fit'], seed, updates=updates)
    codec_parameters = list(model.codec.parameters())
    side_parameters = [p for name, p in model.named_parameters() if not name.startswith('codec.') and p.requires_grad]
    checkpoints, curve = [], []
    step_seconds, checkpoint_seconds = [], []
    for update in range(updates + 1):
        if update in CHECKPOINTS or update == updates:
            checkpoint_started = time.perf_counter()
            model.eval(); model.freeze_memory()
            prediction = predict(model, pack['history']['tune'], pack['origins']['tune'], forecaster, limit)
            checkpoint = rt.save(folder / f'u{update}.pt', {'model': rt.clone_state(model), 'optimizer': optimizer.state_dict(),
                'seed': seed, 'method': method, 'weight': weight, 'update': update})
            tune_artifact = prediction_artifact(rt, folder / f'u{update}_TUNE.npz', pack, {'tune': prediction})
            checkpoints.append({'update': update, **checkpoint, 'tune_prediction': tune_artifact,
                'tune': scores(pack, prediction, 'tune'), 'diagnostics': diagnostics(model, pack, 'tune', forecaster, limit)})
            print(rt.json_line({'case': folder.name, 'dataset': pack['dataset'], 'seed': seed, 'method': method, 'weight': weight,
                'update': update, 'tune_norm_mse': checkpoints[-1]['tune']['normalized']['mse'], 'elapsed_seconds': time.perf_counter() - started}), flush=True)
            checkpoint_seconds.append(time.perf_counter() - checkpoint_started)
        if update == updates: break
        step_started = time.perf_counter()
        model.train()
        indices = order[update]
        optimizer.zero_grad(set_to_none=True)
        losses = model.losses(_tensor(pack['history']['fit'][indices], device), torch.as_tensor(pack['origins']['fit'][indices], device=device),
            _tensor(pack['target']['fit'][indices], device), _boolean(pack['observed']['fit'][indices], device), forecaster, update=update,
            pair_permutation=np.roll(np.arange(8), 1) if method == 'A_BIHASH_SHUFFLE' else None)
        total = losses['main'] + weight * losses['aux'] + .1 * losses['rec']
        if not torch.isfinite(total): raise ValueError('NONFINITE_CORE_LOSS')
        gradient_diagnostic = {}
        if update in (0, 99, 199, 399, 599):
            vectors = {}
            for name in ('main', 'aux'):
                gradients = torch.autograd.grad(losses[name], codec_parameters, retain_graph=True, allow_unused=True)
                vectors[name] = torch.cat([torch.zeros_like(p).flatten() if g is None else g.detach().flatten() for p, g in zip(codec_parameters, gradients)])
                gradient_diagnostic[name + '_codec_unclipped_gradient_norm'] = float(vectors[name].norm())
            product = vectors['main'].norm() * vectors['aux'].norm()
            gradient_diagnostic['main_aux_codec_gradient_cosine'] = float(torch.dot(vectors['main'], vectors['aux']) / product) if product > 0 else None
        total.backward()
        if update in (0, 99, 199, 399, 599):
            gradient_diagnostic.update({name + '_unclipped_gradient_norm': _parameter_norm(model, prefix) for name, prefix in
                (('E', ('codec.codec.project.', 'codec.codec.input.', 'codec.codec.temporal.', 'codec.codec.output.')),
                 ('D', ('codec.codec.decode.', 'codec.residual.')), ('S_projection', 'student_projection.'),
                 ('Ws', 'student_hash.'), ('Wf', 'future_hash.'), ('phi', 'future_autoencoder.encoder.'))})
        codec_norm = torch.nn.utils.clip_grad_norm_(codec_parameters, 1.)
        side_norm = torch.nn.utils.clip_grad_norm_(side_parameters, 1.)
        if not torch.isfinite(codec_norm) or not torch.isfinite(side_norm): raise ValueError('NONFINITE_CORE_GRADIENT')
        optimizer.step()
        curve.append({'update': update + 1, 'main_loss': float(losses['main'].detach()), 'aux_loss': float(losses['aux'].detach()),
            'future_rec_loss': float(losses['rec'].detach()), 'total_loss': float(total.detach()),
            'codec_unclipped_gradient_norm': float(codec_norm), 'side_unclipped_gradient_norm': float(side_norm),
            'temperature': model.temperature(update), **gradient_diagnostic})
        if update % 10 == 0: limit.check()
        step_seconds.append(time.perf_counter() - step_started)
    selected = min(checkpoints, key=lambda cp: (cp['tune']['normalized']['mse'], cp['update']))
    state = torch.load(rt.ROOT / selected['path'], map_location='cpu', weights_only=True)
    model.load_state_dict(state['model']); model.eval().requires_grad_(False)
    predictions = {s: predict(model, pack['history'][s], pack['origins'][s], forecaster, limit) for s in ('fit', 'tune')}
    artifact = prediction_artifact(rt, folder / 'SELECTED_FIT_TUNE.npz', pack, predictions)
    record = {'dataset': pack['dataset'], 'seed': seed, 'method': method, 'weight': weight, 'K': model.k, 'L': 512, 'H': 48,
        'label': folder.name, 'initial_hash': initial_hash, 'initial_metadata': initial_metadata, 'metadata': model.metadata(),
        'future_pretrain': pretrain_record['checkpoint'], 'selected_tune_mse': selected['tune']['normalized']['mse'],
        'training': {'updates': updates, 'batch_size': 8, 'optimizer': 'AdamW', 'lr': .001, 'weight_decay': 0.,
            'codec_clip': 1., 'side_clip': 1., 'future_rec_weight': .1, 'aux_weight': weight,
            'order_sha256': bm.en.ex.ahash(order), 'batch_indices': order.tolist(), 'future_targets_pairwise_nonoverlap': True,
            'checkpoints': checkpoints, 'curve': curve, 'selected_update': selected['update'], 'selected_checkpoint': selected,
            'selection': 'minimum TUNE observed channel-macro normalized MSE, earliest exact tie',
            'wall_seconds': time.perf_counter() - started, 'selected_hash': rt.state_hash(model)},
        'predictions': artifact, 'scores': {s: scores(pack, p, s) for s, p in predictions.items()},
        'selected_diagnostics': {s: diagnostics(model, pack, s, forecaster, limit) for s in ('fit', 'tune')}}
    record['training'].update(training_step_seconds=step_seconds, checkpoint_seconds=checkpoint_seconds,
        total_optimization_step_seconds=sum(step_seconds), total_checkpoint_seconds=sum(checkpoint_seconds))
    return model, record


def load_model(pack, case, rt, device='cuda'):
    checkpoint = case['training']['selected_checkpoint']
    if rt.sha(rt.ROOT / checkpoint['path']) != checkpoint['sha256']: raise ValueError('CHECKPOINT_SHA_MISMATCH')
    saved = torch.load(rt.ROOT / checkpoint['path'], map_location='cpu', weights_only=True)
    model = make_model(pack, case['seed'], case['method'], device=device)
    model.load_state_dict(saved['model']); model.eval().requires_grad_(False)
    if rt.state_hash(model) != case['training']['selected_hash']: raise ValueError('SELECTED_MODEL_HASH_MISMATCH')
    return model
