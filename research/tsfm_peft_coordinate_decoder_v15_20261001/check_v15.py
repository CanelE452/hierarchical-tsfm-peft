"""Reserved syntax and synthetic algebra checks, without optimizer or real fits."""
import argparse
import ast
from runtime_v15 import HERE, Job, artifact, save_json, budget_snapshot


def check(job):
    import numpy as np
    import torch
    files = []
    for path in sorted(HERE.glob('*.py')):
        ast.parse(path.read_text(encoding='utf-8'))
        files.append(artifact(path))
    from model_v15 import SelectorDecoder
    torch.set_num_threads(4)
    class Direct(torch.nn.Module):
        def forward(self, x):
            return x[:, -1:, :].repeat(1, 48, 1) + .2
    class F0(torch.nn.Module):
        chunk_rows = None
        def backbone_point(self, x):
            return x[:, -1:, :].repeat(1, 48, 1) ** 2
    generator = torch.Generator().manual_seed(15001)
    x = torch.randn(2, 512, 7, generator=generator)
    e = torch.eye(7)[:, [5, 1, 3]]
    w = torch.randn(3, 7, generator=generator)
    direct, f0 = Direct(), F0()
    model = SelectorDecoder(direct, f0, e, w)
    expected = direct(x) + (f0.backbone_point(x @ e) - direct(x) @ e) @ w
    actual = model(x)
    assert torch.equal(actual, expected)
    zero = SelectorDecoder(direct, None, e, torch.zeros_like(w))
    assert torch.equal(zero(x), direct(x))
    assert zero.backbone is None
    model.chunk_rows = 2
    assert torch.equal(model(x), expected)
    assert f0.chunk_rows == 2
    # Restoring state includes fixed E and learned W; zero-fallback construction is
    # intentionally not used as a mutable deployment checkpoint.
    nonzero = SelectorDecoder(Direct(), F0(), e, w + 1)
    nonzero.load_state_dict(model.state_dict())
    assert torch.equal(nonzero(x), expected)
    assert all(p.grad is None for p in model.parameters())
    from fit_v15 import fit_coefficients
    rng = np.random.default_rng(15102)
    z = rng.normal(size=(3, 48, 3))
    truth = rng.normal(size=(3, 7))
    parent = rng.normal(size=(3, 48, 7))
    target = parent + z @ truth
    mask = rng.uniform(size=target.shape) > .2
    target[~mask] = 1e20
    fitted, diag = fit_coefficients(z, parent, target, mask, 0., job)
    assert np.allclose(fitted, truth, atol=1e-12, rtol=0)
    assert diag['target_counts_by_channel'] == mask.sum((0, 1)).tolist()
    penalized, diag_r = fit_coefficients(z, parent, target, mask, .1, job)
    assert diag_r['normal_equation_residual_maxabs'] < 1e-12
    assert np.isfinite(penalized).all()
    record = {'status': 'pass', 'syntax_files': files, 'synthetic_optimizer_updates': 0,
              'real_fits': 0, 'checks': ['exact toy formula', 'zero fallback', 'chunk propagation', 'state restoration',
                         'masked synthetic coefficient recovery', 'positive-penalty stationarity'],
              'purpose': 'Implementation algebra only, not evidence of forecast improvement'}
    if (HERE / 'implementation_checks.json').exists():
        from runtime_v15 import read_json
        save_json(HERE / 'implementation_checks_initial.json', read_json(HERE / 'implementation_checks.json'))
    save_json(HERE / 'implementation_checks.json', record)
    job.heartbeat('Syntax and synthetic no-optimizer algebra checks passed')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--job', required=True)
    p.add_argument('--reserve-s', type=float, default=30)
    a = p.parse_args()
    with Job('cpu_check', a.job, reserve_s=a.reserve_s, metadata={'optimizer_updates': 0}) as job:
        check(job)
    print('CPU syntax/algebra checks passed.')
