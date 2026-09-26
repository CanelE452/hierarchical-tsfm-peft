import argparse
import copy
import gc
import importlib.metadata
import json
import platform
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from model import ForecastAdapter, SNAPSHOT, make_model, macro_loss, pca_basis
from run import CACHE, HERE, GPUJob, backward_batch, digest, frozen_hash, load_data, phase_sample, save_json, score_arrays, source_receipt, tensors


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
        self.chronos_config = SimpleNamespace(quantiles=[0.5])

    def forward(self, context):
        point = context[:, -48:] * self.scale
        return SimpleNamespace(quantile_preds=point[:, None, :])


def cpu_checks():
    torch.manual_seed(7)
    rng = np.random.default_rng(7)
    basis = pca_basis(rng.normal(size=(120, 8)), 3)
    assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-6)
    x = torch.randn(2, 512, 8)
    compress = ForecastAdapter(ToyBackbone(), 'compress', basis)
    residual = ForecastAdapter(ToyBackbone(), 'residual', basis)
    raw = ForecastAdapter(ToyBackbone(), 'raw_bypass', basis)
    assert torch.equal(compress(x), residual(x))
    assert torch.equal(compress(x), raw(x))
    y, mask = torch.randn(2, 48, 8), torch.ones(2, 48, 8, dtype=torch.bool)
    optimizer = torch.optim.AdamW([p for p in residual.parameters() if p.requires_grad], lr=0.001)
    before = residual.adapter_state()
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        macro_loss(residual(x), y, mask).backward()
        assert residual.encoder.weight.grad.abs().sum() > 0
        assert residual.backbone.scale.grad is None
        optimizer.step()
    after = residual.adapter_state()
    assert all((after[k] - before[k]).abs().max() > 0 for k in before)
    restored = ForecastAdapter(ToyBackbone(), 'residual', basis)
    restored.restore_adapter(after)
    assert torch.equal(restored(x), residual(x))
    origins = np.arange(512, 13000)
    first = phase_sample(origins, 92601, 1)
    assert np.array_equal(first, phase_sample(origins, 92601, 1))
    counts = np.bincount(first % 24, minlength=24)
    assert counts.max() - counts.min() == 1
    pred, target = rng.normal(size=(3, 48, 8)), rng.normal(size=(3, 48, 8))
    mask_n = rng.random((3, 48, 8)) > 0.2
    expected = np.mean([np.mean([(pred[o, t, c] - target[o, t, c]) ** 2 for o in range(3) for t in range(48) if mask_n[o, t, c]]) for c in range(8)])
    actual = score_arrays(pred, target, mask_n)['mse']
    assert abs(expected - actual) < 1e-12
    save_json(HERE / 'cpu_checks.json', {'passed': True, 'source': source_receipt(), 'checks': ['PCA orthonormal initialization', 'residual/raw zero-output reduction to compression', 'input gradient through frozen backbone', 'all intended adapter parameters update within two updates', 'frozen parameters untouched', 'adapter checkpoint restoration', 'paired phase-balanced sampling', 'masked channel-macro MSE scalar recompute'], 'metric_error': abs(expected - actual)})
    print('CPU checks PASS')


def gpu_checks():
    from chronos import ChronosBoltPipeline
    data = load_data('electricity')
    basis = pca_basis(data['x'][:int(data['train_end'])], 8)
    with GPUJob('real_checkpoint_integrity') as job:
        torch.manual_seed(92601)
        model = make_model('residual', basis)
        x, target, mask = tensors(data, data['train_origins'][:1])
        initial = model.adapter_state()
        frozen = frozen_hash(model)
        model.eval()
        with torch.no_grad():
            z = model.encoder(x)
            compressed = model.decoder(model.backbone_point(z))
            first = model(x)
            reduction_error = (first - compressed).abs().max().item()
        assert reduction_error == 0
        gradients = []
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001, weight_decay=0)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = macro_loss(model(x), target, mask)
            loss.backward()
            gradients.append({name: None if p.grad is None else p.grad.norm().item() for name, p in model.named_parameters() if p.requires_grad})
            assert model.encoder.weight.grad.norm() > 0
            assert all(p.grad is None for p in model.backbone.parameters())
            optimizer.step()
            job.heartbeat(f'residual smoke update {step + 1}/2')
        state = model.adapter_state()
        updates = {k: (state[k] - initial[k]).abs().max().item() for k in state}
        assert all(v > 0 for v in updates.values())
        assert frozen == frozen_hash(model)
        with torch.no_grad():
            expected = model(x).cpu()
        torch.save(state, CACHE / 'smoke_adapter.pt')
        del optimizer, model
        torch.cuda.empty_cache()
        restored = make_model('residual', basis)
        restored.restore_adapter(torch.load(CACHE / 'smoke_adapter.pt', weights_only=True))
        with torch.no_grad():
            replay = restored(x).cpu()
        replay_error = (expected - replay).abs().max().item()
        assert replay_error == 0
        del restored
        torch.cuda.empty_cache()
        pipe = ChronosBoltPipeline.from_pretrained(str(SNAPSHOT), device_map='cuda', torch_dtype=torch.float32)
        pipe.model.eval()
        flat = x.transpose(1, 2).reshape(-1, 512)
        with torch.no_grad():
            direct = pipe.model(context=flat).quantile_preds[:, :, :48].cpu()
            official = pipe.predict(flat, prediction_length=48)
        parity = (direct - official).abs().max().item()
        assert parity == 0
        base_point = direct[:, 4, :].transpose(0, 1)
        del pipe
        torch.cuda.empty_cache()
        lora = make_model('lora', basis)
        with torch.no_grad():
            lora_initial = lora(x).cpu()[0]
        lora_zero_error = (base_point - lora_initial).abs().max().item()
        assert lora_zero_error == 0
        before = frozen_hash(lora)
        loss = lora.native_loss(x, target, mask)
        loss.backward()
        assert all(p.grad is None for p in lora.parameters() if not p.requires_grad)
        opt = torch.optim.AdamW([p for p in lora.parameters() if p.requires_grad], lr=1e-4)
        opt.step()
        assert before == frozen_hash(lora)
        save_json(HERE / 'gpu_checks.json', {'passed': True, 'source': source_receipt(), 'smoke_updates': 3, 'reduction_error': reduction_error, 'replay_error': replay_error, 'official_output_error': parity, 'lora_zero_error': lora_zero_error, 'input_path_gradient_norms': gradients, 'updates': updates, 'frozen_unchanged': True, 'gpu_name': torch.cuda.get_device_name(), 'native_lora_loss': loss.item(), 'environment': {key: importlib.metadata.version(key) for key in ['torch', 'chronos-forecasting', 'peft', 'transformers', 'numpy']}, 'python': platform.python_version()})
        print('GPU real-checkpoint checks PASS')


def lifetime_checks():
    output = HERE / 'object_lifetime_audit.json'
    if output.exists():
        raise RuntimeError('Preserve the completed lifetime audit')
    records = []
    with GPUJob('model_object_lifetime_audit') as job:
        for arm in ('lora', 'raw_bypass', 'residual'):
            gc.collect()
            torch.cuda.empty_cache()
            before = torch.cuda.memory_allocated()
            model = make_model(arm, np.eye(32, 8, dtype=np.float32))
            torch.cuda.synchronize()
            loaded = torch.cuda.memory_allocated()
            del model
            torch.cuda.empty_cache()
            after_del = torch.cuda.memory_allocated()
            collected = gc.collect()
            torch.cuda.empty_cache()
            after_gc = torch.cuda.memory_allocated()
            assert after_gc == before, (arm, before, after_gc)
            records.append({'arm': arm, 'before_bytes': before, 'loaded_bytes': loaded, 'after_del_bytes': after_del, 'after_gc_bytes': after_gc, 'collected_objects': collected})
            job.heartbeat(f'{arm} lifetime audited')
    save_json(output, {'source': source_receipt(), 'scope': 'real model load/release only, no predictions or learning; artificial identity basis only sets C32/r8 adapter dimensions', 'records': records})
    print(records, flush=True)


def batching_checks():
    output = HERE / 'batching_checks.json'
    if output.exists():
        raise RuntimeError('Preserve the completed batching audit')
    data = load_data('electricity')
    selection = json.loads((HERE / 'initial_selection.json').read_text())
    records = []
    with GPUJob('effective_batch_gradient_parity') as job:
        origins = phase_sample(data['train_origins'], 92601, 1)[:4]
        x, target, observed = tensors(data, origins)
        missing = observed.clone()
        missing[1:, :, 0] = False
        missing[:, ::3, 1] = False
        missing[:, :, 2] = False
        for arm in ('lora', 'compress', 'residual', 'raw_bypass'):
            fit_id = selection[arm]['fits'][0]
            result = json.loads((HERE / 'runs' / fit_id / 'result.json').read_text())
            saved = torch.load(result['checkpoint'], map_location='cpu', weights_only=False)
            model = make_model(arm, saved['basis'])
            model.restore_adapter(saved['state'])
            model.train()
            loss_kind = 'native' if arm == 'lora' else 'mse'
            for mask_name, mask in (('complete', observed), ('synthetic_missing', missing)):
                measurements, gradients = {}, {}
                for microbatch in (1, 4):
                    model.zero_grad(set_to_none=True)
                    backward_batch(model, x, target, mask, loss_kind, microbatch)
                    model.zero_grad(set_to_none=True)
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    for _ in range(4):
                        model.zero_grad(set_to_none=True)
                        loss = backward_batch(model, x, target, mask, loss_kind, microbatch)
                    torch.cuda.synchronize()
                    measurements[microbatch] = {'loss': loss, 'seconds_per_backward_batch': (time.perf_counter() - started) / 4, 'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
                    gradients[microbatch] = torch.cat([p.grad.detach().cpu().reshape(-1).double() for p in model.parameters() if p.requires_grad])
                    job.heartbeat(f'{arm}/{mask_name}/micro{microbatch}')
                error = gradients[4] - gradients[1]
                relative_error = error.norm().item() / max(gradients[1].norm().item(), 1e-12)
                assert relative_error < 2e-5, (arm, mask_name, relative_error)
                torch.testing.assert_close(gradients[4], gradients[1], rtol=1e-4, atol=1e-6)
                assert abs(measurements[4]['loss'] - measurements[1]['loss']) < 1e-5 * max(1, abs(measurements[1]['loss']))
                records.append({'arm': arm, 'checkpoint_sha256': result['checkpoint_sha256'], 'mask': mask_name, 'relative_gradient_l2_error': relative_error, 'max_gradient_abs_error': error.abs().max().item(), 'measurements': measurements})
            model.zero_grad(set_to_none=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    save_json(output, {'passed': True, 'source': source_receipt(), 'data_sha256': digest(data['_path']), 'origins': origins.tolist(), 'scope': 'same effective batch4 and trained checkpoints; gradient-only checks, zero optimizer updates, complete and artificial missing masks; short warmed timing informs execution choice, not a paper latency benchmark', 'records': records})
    print(json.dumps(records), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--lifetime', action='store_true')
    parser.add_argument('--batching', action='store_true')
    args = parser.parse_args()
    if args.lifetime:
        lifetime_checks()
    elif args.batching:
        batching_checks()
    else:
        cpu_checks()
        if args.gpu:
            gpu_checks()
