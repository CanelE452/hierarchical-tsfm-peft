"""Synthetic structural gates, without real datasets or foundation weights."""
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'research/informs_broad_validation_20261015_v1'))
import fit_base
from research.comparative_ablation_20261025_v1 import models

torch.set_num_threads(2)


CASES = [('jena', 21, 6, 35717, 774), ('ETTh1', 7, 4, 34023, 2220),
         ('ETTh2_safe17400', 7, 4, 34023, 2220), ('appliances', 26, 10, 37114, 628)]


def fixture(channels, k, method, seed=92601):
    generator = np.random.default_rng(17)
    physical = generator.normal(size=(640, channels)).astype(np.float32)
    task = dict(physical=physical, observed=np.ones_like(physical, dtype=bool), L=512, H=48)
    spec = dict(method_id=method, family='B2', latent_k=k, seed=seed, device='cpu')
    return models.create_model(spec, task, 580, lambda z: z[:, -48:]), task, spec


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize('dataset,c,k,ours_count,hidden', CASES)
def test_counts_initialization_and_untouched_state(dataset, c, k, ours_count, hidden):
    (ours, norm), _, _ = fixture(c, k, 'OURS')
    assert count(ours) == ours_count
    expected = dict(OURS=ours_count, C1=0, C2=2*c*k+k+c, C3=2*c*k+k+c,
                    A1=8*c*k, A2=ours_count-33312+(2*c+1)*hidden+c,
                    A3=ours_count, A4=ours_count-6*c*k+c, A5=ours_count-6*c*k)
    for method, parameters in expected.items():
        (model, variant_norm), _, _ = fixture(c, k, method)
        assert count(model) == parameters, (dataset, method)
        np.testing.assert_array_equal(norm['mean'], variant_norm['mean'])
        np.testing.assert_array_equal(norm['std'], variant_norm['std'])
        if method in ('A2', 'A3', 'A4', 'A5'):
            omitted = ('front.mixer.temporal.',) if method == 'A2' else ()
            base_front = {n:v for n,v in ours.state_dict().items()
                          if n.startswith('front.') and not n.startswith(omitted)}
            variant_front = {n:v for n,v in model.state_dict().items()
                             if n.startswith('front.') and not n.startswith(omitted)}
            assert state_hash(base_front) == state_hash(variant_front)
            assert torch.equal(ours.center, model.center)
            if method != 'A3':
                assert torch.equal(ours.E, model.E)
        if method == 'A2':
            assert model.front.mixer.hidden == hidden
            assert abs(count(model) - count(ours)) <= 10
            assert torch.equal(model.D, ours.D)
        if method == 'A3':
            torch.testing.assert_close(model.E.T @ model.E, torch.eye(k), atol=1e-5, rtol=0)
            torch.testing.assert_close(model.D[0], torch.linalg.pinv(model.E))
            assert torch.count_nonzero(model.D[1:]) == 0
            np.testing.assert_array_equal(model.E.detach().numpy(), variant_norm['basis'])
        if method in ('A4', 'A5'):
            assert 'D' not in dict(model.named_parameters())


@pytest.mark.parametrize('dataset,c,k,ours_count,hidden', CASES)
def test_decoder_initial_algebra(dataset, c, k, ours_count, hidden):
    (ours, _), _, _ = fixture(c, k, 'OURS')
    (linear, _), _, _ = fixture(c, k, 'A4')
    (lag0, _), _, _ = fixture(c, k, 'A5')
    x = torch.randn(2, 512, c)
    z = ours.encode(x)
    u = torch.randn(2, 48, k)
    torch.testing.assert_close(linear.decode(x, z, u), u @ torch.linalg.pinv(ours.E) + ours.center)
    expected = x[:, -1:] + (u - z[:, -1:]) @ ours.D[0]
    torch.testing.assert_close(lag0.decode(x, z, u), expected)
    torch.testing.assert_close(ours.decode(x, z, u), expected)
    torch.testing.assert_close(linear(x, lambda _: u), linear.decode(x, z, u))
    torch.testing.assert_close(lag0(x, lambda _: u), expected)


def test_capacity_control_preserves_identity_but_has_only_pointwise_learned_branch():
    (model, _), _, _ = fixture(21, 6, 'A2')
    x = torch.randn(2, 512, 21)
    assert torch.equal(model.front(x), x)
    branch = model.front.mixer.temporal
    assert (branch.input.in_features, branch.input.out_features) == (21, 774)
    assert (branch.output.in_features, branch.output.out_features) == (774, 21)
    # Once nonzero, the learned branch itself still maps each time independently.
    nn.init.normal_(branch.output.weight)
    change = x.clone()
    change[:, 100] += 1
    delta = branch(change) - branch(x)
    assert torch.count_nonzero(delta[:, :100]) == 0
    assert torch.count_nonzero(delta[:, 101:]) == 0


def test_dropout_initial_weights_and_one_sample_training():
    (base, _), _, _ = fixture(21, 6, 'C2')
    (model, _), _, _ = fixture(21, 6, 'C3')
    assert state_hash(base.state_dict()) == state_hash(model.state_dict())
    assert [m.p for m in model.modules() if isinstance(m, nn.Dropout)] == [.1]
    x = torch.randn(2, 512, 21)
    seen = []
    def foundation(z):
        seen.append(z)
        return z[:, -48:].square()
    state = torch.random.get_rng_state()
    y = model(x, foundation)
    assert len(seen) == 1
    torch.random.set_rng_state(state)
    expected_z = F.dropout(model.encoder(x), p=.1, training=True)
    assert torch.equal(seen[0], expected_z)
    torch.testing.assert_close(y, model.decoder(expected_z[:, -48:].square()))
    y.sum().backward()
    assert model.encoder.weight.grad is not None and model.decoder.weight.grad is not None


def test_dropout_mc_whole_evaluation_rng_restore_and_deterministic_diagnostic():
    (model, _), _, _ = fixture(21, 6, 'C3')
    x = torch.randn(3, 512, 21)
    seen = []
    def foundation(z):
        value = z[:, -48:].square()
        seen.append(value)
        return value
    state = torch.random.get_rng_state().clone()
    with models.evaluation_scope(model):
        model.eval()  # Core evaluation does this after entry.
        first = model(x[:2], foundation)
        second = model(x[2:], foundation)
    assert model.training
    assert model._evaluation is None
    assert torch.equal(state, torch.random.get_rng_state())
    assert len(seen) == 20
    torch.testing.assert_close(first, model.decoder(torch.stack(seen[:10]).mean(0)))
    torch.testing.assert_close(second, model.decoder(torch.stack(seen[10:]).mean(0)))
    with models.evaluation_scope(model):
        assert torch.equal(first, model(x[:2], foundation))
        assert torch.equal(second, model(x[2:], foundation))
    with models.evaluation_scope(model, deterministic=True):
        diagnostic = model(x, foundation)
    torch.testing.assert_close(diagnostic, model.decoder(model.encoder(x)[:, -48:].square()))
    assert torch.equal(state, torch.random.get_rng_state())
    model.eval()
    with pytest.raises(RuntimeError, match='evaluation_scope'):
        model(x, foundation)


def test_mc_seed_matches_registered_offset_and_restores_on_error():
    (model, _), _, _ = fixture(7, 4, 'C3', seed=92608)
    x = torch.randn(1, 512, 7)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7092608)
        latent = torch.stack([F.dropout(model.encoder(x), p=.1, training=True)[:, -48:]
                              for _ in range(10)]).mean(0)
    with models.evaluation_scope(model):
        torch.testing.assert_close(model(x, lambda z: z[:, -48:]), model.decoder(latent))
    before = torch.random.get_rng_state()
    with pytest.raises(ValueError):
        with models.evaluation_scope(model):
            raise ValueError('synthetic evaluation failure')
    assert torch.equal(before, torch.random.get_rng_state())
    assert model.training and model._evaluation is None


def test_core_hooks_are_scoped_and_evaluate_c3_with_ten_samples():
    (_, _), task, spec = fixture(7, 4, 'C3')
    original = fit_base.create_model, fit_base.predict_model, fit_base._evaluate
    calls = []
    def foundation(z):
        calls.append(z)
        return z[:, -48:]
    class Windows:
        channels, device = 7, 'cpu'
        def get(self, indices):
            return torch.ones(len(indices), 512, 7), torch.zeros(len(indices), 48, 7), torch.ones(len(indices), 48, 7, dtype=torch.bool)
    with models.patch_core(fit_base):
        model, _ = fit_base.create_model(spec, task, 580, foundation)
        predict = lambda x, k: fit_base.predict_model(model, x, foundation, 'ADAPTS', k)
        value, _ = fit_base._evaluate(model, Windows(), [0, 1, 2], predict, None, 2)
        assert np.isfinite(value) and len(calls) == 20
    assert (fit_base.create_model, fit_base.predict_model, fit_base._evaluate) == original
