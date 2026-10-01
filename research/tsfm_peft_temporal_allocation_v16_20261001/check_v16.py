"""Reserved CPU-only synthetic temporal algebra; no real model or optimizer."""
import argparse
import ast
from types import SimpleNamespace

from runtime_v16 import HERE, DATASETS, Job, artifact, configure, save_json


def check(job):
    import torch
    from model_v16 import TemporalAllocationModel, average4, repeat4, fine_component

    destination = HERE / 'implementation_checks.json'
    if destination.exists():
        raise FileExistsError('Preserve the existing check record before an explicit repair run')
    sources = []
    for path in sorted(HERE.glob('*.py')):
        ast.parse(path.read_text(encoding='utf-8'))
        sources.append(artifact(path))
    generator = torch.Generator().manual_seed(16001)
    c = DATASETS['robin'][0]
    fine = torch.randn(2, 48, c, generator=generator, dtype=torch.float64)
    coarse = torch.randn(2, 12, c, generator=generator, dtype=torch.float64)
    p, q = repeat4(average4(fine)), fine_component(fine)
    torch.testing.assert_close(average4(repeat4(coarse)), coarse, atol=0, rtol=0)
    torch.testing.assert_close(p + q, fine, atol=1e-12, rtol=0)
    torch.testing.assert_close(average4(q), torch.zeros_like(coarse), atol=1e-12, rtol=0)
    torch.testing.assert_close(repeat4(average4(p)), p, atol=0, rtol=0)
    assert abs(float((p * q).sum())) < 1e-10
    constant = torch.full((2, 48, c), 3., dtype=torch.float64)
    assert torch.count_nonzero(fine_component(constant)) == 0
    ramp = torch.arange(48, dtype=torch.float64).reshape(1, 48, 1)
    expected = torch.tensor([-1.5, -.5, .5, 1.5], dtype=torch.float64).repeat(12).reshape(1, 48, 1)
    torch.testing.assert_close(fine_component(ramp), expected, atol=0, rtol=0)
    missing_mask = torch.tensor([1., 0., 0., 0.], dtype=torch.float64)
    counterexample_cross_term = float((torch.ones(4) * expected.flatten()[:4] * missing_mask).sum())
    assert counterexample_cross_term != 0

    class Direct(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fixed_scale = torch.nn.Parameter(torch.tensor(1.))

        def forward(self, x):
            h = torch.arange(48, dtype=x.dtype, device=x.device).reshape(1, 48, 1)
            return x[:, -1:, :] * self.fixed_scale + h * .125

    class Backbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fixed_scale = torch.nn.Parameter(torch.tensor(1.))
            self.chronos_config = SimpleNamespace(prediction_length=64, input_patch_size=16,
                input_patch_stride=16, use_reg_token=True, quantiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9])
            self.seen_shapes = []

        def forward(self, context):
            assert context.shape[1] == 128
            self.seen_shapes.append(list(context.shape))
            h = torch.arange(64, dtype=context.dtype, device=context.device).reshape(1, 1, 64)
            quantile_offset = torch.arange(9, dtype=context.dtype, device=context.device).reshape(1, 9, 1) - 4
            value = context.mean(dim=1).reshape(-1, 1, 1) * self.fixed_scale + h * .25 + quantile_offset
            return SimpleNamespace(quantile_preds=value)

    model = TemporalAllocationModel('robin', 92601, Direct(), Backbone(), adapt=False).eval()
    x = torch.randn(2, 512, c, generator=generator)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        main, detail = model.components(x)
        direct = model.direct(x)
        point = average4(x).mean(dim=1, keepdim=True) + torch.arange(12).reshape(1, 12, 1) * .25
        torch.testing.assert_close(main, repeat4(point), atol=1e-6, rtol=0)
        torch.testing.assert_close(detail, fine_component(direct), atol=0, rtol=0)
        prediction = model(x)
        alternative = direct + repeat4(point - average4(direct))
        torch.testing.assert_close(prediction, alternative, atol=1e-5, rtol=1e-4)
        assert tuple(prediction.shape) == (2, 48, c)
        model.chunk_rows = 5
        chunked = model(x)
        torch.testing.assert_close(chunked, prediction, atol=1e-6, rtol=0)
        model.chunk_rows = None
        clone = TemporalAllocationModel('robin', 92601, Direct(), Backbone(), adapt=False).eval()
        clone.load_state_dict(model.state_dict())
        torch.testing.assert_close(clone(x), prediction, atol=0, rtol=0)
        model.train()
        assert not model.backbone.training and not model.direct.training
        assert all(not p.requires_grad and p.grad is None for p in model.parameters())
        assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
        constant_x = torch.full((2, 512, c), 7.)
        assert torch.isfinite(model(constant_x)).all()
    row = {'status': 'passed', 'job_id': job.id, 'sources': sources, 'synthetic_optimizer_updates': 0,
           'real_fits': 0, 'real_backbone_forwards': 0, 'future_target_inputs': False,
           'checks': ['A R identity', 'P/Q complement and zero block means', 'complete-coordinate orthogonality',
                      'constant and ramp', 'within-block missing-mask counterexample', '128-input native-style 12-of-64 crop',
                      'all-channel row ordering', 'two formula parity', 'full/chunk', 'synthetic state restore',
                      'frozen mock parents and eval mode', 'finite constant-input toy output'],
           'masked_counterexample_cross_term': counterexample_cross_term,
           'actual_initial_adapter_and_gradient_check': 'Reserved for the planned real fit/VAL path; not tested by these mock modules',
           'scope': 'Synthetic implementation algebra only; no forecast-improvement evidence'}
    save_json(destination, row)
    job.heartbeat('CPU synthetic temporal checks complete; zero optimizer updates', updates=0)
    return row


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    parser.add_argument('--reserve-s', type=float, default=45)
    args = parser.parse_args()
    with Job('cpu_check', args.label, reserve_s=args.reserve_s, synthetic=True,
             metadata={'purpose': 'CPU synthetic temporal algebra', 'optimizer_updates': 0}) as job:
        configure()
        check(job)
    print('V16 synthetic algebra checks complete; no real backbone or fitting.')
