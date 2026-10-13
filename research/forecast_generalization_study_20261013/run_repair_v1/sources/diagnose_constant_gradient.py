"""Test constant-window Bolt derivatives without modifying any prior run/source."""
from pathlib import Path
import argparse
import json
import time
import experiment as ex
from chronos.chronos_bolt import InstanceNorm


class ZeroSafeSqrt(ex.torch.autograd.Function):
    @staticmethod
    def forward(ctx, variance):
        scale = variance.sqrt()
        ctx.save_for_backward(variance, scale)
        return scale

    @staticmethod
    def backward(ctx, upstream):
        variance, scale = ctx.saved_tensors
        positive = variance > 0
        denominator = ex.torch.where(positive, scale, ex.torch.ones_like(scale))
        return ex.torch.where(positive, upstream/(2*denominator), ex.torch.zeros_like(upstream))


class ZeroSafeInstanceNorm(InstanceNorm):
    def forward(self, x, loc_scale=None):
        original_dtype = x.dtype
        x = x.to(ex.torch.float32)
        if loc_scale is None:
            loc = ex.torch.nan_to_num(ex.torch.nanmean(x, dim=-1, keepdim=True), nan=0.)
            variance = (x-loc).square().nanmean(dim=-1, keepdim=True)
            scale = ex.torch.nan_to_num(ZeroSafeSqrt.apply(variance), nan=1.)
            scale = ex.torch.where(scale == 0, self.eps, scale)
        else:
            loc, scale = loc_scale
        standardized = (x-loc)/scale
        if self.use_arcsinh:
            standardized = ex.torch.arcsinh(standardized)
        return standardized.to(original_dtype), (loc, scale)


def replace_normalizer(bolt):
    old = bolt.model.instance_norm
    bolt.model.instance_norm = ZeroSafeInstanceNorm(eps=old.eps, use_arcsinh=old.use_arcsinh)


def normalizer_fixture():
    value = ex.torch.ones(2, 16)
    value[1] = ex.torch.arange(16)/16
    results = []
    for normalizer in (InstanceNorm(), ZeroSafeInstanceNorm()):
        x = value.clone().requires_grad_(True)
        y, (loc, scale) = normalizer(x)
        (y.sum()+loc.sum()+scale.sum()).backward()
        results.append({'output': y.detach(), 'loc': loc.detach(), 'scale': scale.detach(), 'gradient': x.grad.detach()})
    before, after = results
    forward = max(float((before[k]-after[k]).abs().max()) for k in ('output', 'loc', 'scale'))
    positive_gradient = float((before['gradient'][1]-after['gradient'][1]).abs().max())
    assert forward == 0 and positive_gradient == 0 and ex.torch.isfinite(after['gradient']).all()
    assert not ex.torch.isfinite(before['gradient'][0]).all()
    return {'constant_original_input_gradient_finite': bool(ex.torch.isfinite(before['gradient'][0]).all()),
        'constant_repaired_input_gradient_finite': bool(ex.torch.isfinite(after['gradient'][0]).all()),
        'all_forward_max_abs': forward, 'positive_variance_gradient_max_abs': positive_gradient, 'pass': True}


def run(tag):
    ex.DEVICE = 'cpu'
    ex.previous.DEVICE = 'cpu'
    ex.torch.set_num_threads(2)
    ex.torch.set_num_interop_threads(1)
    protected = ex.access_guard()
    output = ex.HERE/('run_'+tag)
    output.mkdir(exist_ok=False)
    start = time.perf_counter()
    process = ex.psutil.Process()
    cpu_start = sum(process.cpu_times()[:2])
    fixture = normalizer_fixture()
    bolt = ex.previous.FrozenBolt()
    result = {'status': 'RUNNING', 'normalizer_fixture': fixture, 'cases': [], 'device': 'CPU', 'new_training_updates': 0,
        'protected_test_guarded': protected, 'source_hashes': {str(p.relative_to(ex.ROOT)): ex.sha(p) for p in
            (Path(__file__), ex.HERE/'experiment.py')}, 'bolt_initial_hash': bolt.initial_hash}
    data = ex.load_data('jena', sizes=(256, 64, None))
    history, target, observed, _ = ex.arrays(data, 512)
    for seed in ex.SEEDS:
        ids = ex.np.random.default_rng(seed).integers(0, 256, size=(600, 4))[0]
        mean, basis, _ = ex.initialization(data, len(data['columns']), 'IDENTITY')
        codec = ex.Codec(mean, basis, seed).eval()
        x = ex.tensor(history['fit'][ids]).requires_grad_(True)
        original = bolt(x)
        loss = ex.previous.macro_loss(codec.decode(original), ex.tensor(target['fit'][ids]), ex.boolean(observed['fit'][ids]))
        gradient = ex.torch.autograd.grad(loss, x)[0]
        nonfinite_channels = ~ex.torch.isfinite(gradient).all(1)
        original_record = {'loss': float(loss.detach()), 'input_gradient_finite': bool(ex.torch.isfinite(gradient).all()),
            'nonfinite_channels': sorted({data['columns'][c] for b, c in ex.torch.argwhere(nonfinite_channels).tolist()})}
        original_prediction = original.detach().clone()
        old_normalizer = bolt.model.instance_norm
        replace_normalizer(bolt)
        codec.zero_grad(set_to_none=True)
        x = ex.tensor(history['fit'][ids]).requires_grad_(True)
        future = codec.decode(bolt(codec.encode(x)))
        loss = ex.previous.macro_loss(future, ex.tensor(target['fit'][ids]), ex.boolean(observed['fit'][ids]))
        loss.backward()
        encoder = [p.grad for n, p in codec.named_parameters() if not n.startswith('decode.')]
        decoder = [p.grad for p in codec.decode.parameters()]
        repaired_record = {'loss': float(loss.detach()), 'input_gradient_finite': bool(ex.torch.isfinite(x.grad).all()),
            'encoder_gradient_finite': all(p is not None and bool(ex.torch.isfinite(p).all()) for p in encoder),
            'decoder_gradient_finite': all(p is not None and bool(ex.torch.isfinite(p).all()) for p in decoder),
            'forward_max_abs_vs_original': float((future.detach()-original_prediction).abs().max())}
        assert not original_record['input_gradient_finite']
        assert repaired_record['input_gradient_finite'] and repaired_record['encoder_gradient_finite'] and repaired_record['decoder_gradient_finite']
        assert repaired_record['forward_max_abs_vs_original'] == 0
        initial = ex.previous.clone_state(codec)
        ex.torch.optim.AdamW(codec.parameters(), lr=1e-3, weight_decay=0.).step()
        repaired_record['actual_encoder_update_max_abs'] = max(float((p.detach()-initial[n]).abs().max()) for n, p in codec.named_parameters() if not n.startswith('decode.'))
        repaired_record['actual_decoder_update_max_abs'] = max(float((p.detach()-initial[n]).abs().max()) for n, p in codec.named_parameters() if n.startswith('decode.'))
        codec.load_state_dict(initial)
        bolt.model.instance_norm = old_normalizer
        result['cases'].append({'dataset': 'jena', 'seed': seed, 'first_batch_indices': ids.tolist(), 'original': original_record,
            'repaired': repaired_record, 'temporary_optimizer_step_restored': True})
    assert ex.previous.state_hash(bolt.model) == bolt.initial_hash
    result['bolt_final_hash'] = ex.previous.state_hash(bolt.model)
    result['status'] = 'COMPLETE'
    result['resources'] = {'wall_seconds': time.perf_counter()-start, 'cpu_seconds': sum(process.cpu_times()[:2])-cpu_start,
                           'rss_bytes': process.memory_info().rss, 'gpu_seconds': 0}
    ex.write_new(output/'RESULT.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    args = parser.parse_args()
    run(args.tag)
