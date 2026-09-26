import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import model
import run


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.7))
        self.chronos_config = SimpleNamespace(quantiles=[0.5])

    def forward(self, context):
        point = self.scale * (context.mean(-1, keepdim=True) + context[:, -1:].square())
        return SimpleNamespace(quantile_preds=point[:, None, :].expand(-1, 1, 48))


def example(ed_mode='current', arm='residual'):
    torch.manual_seed(92601)
    basis = np.eye(4, 2, dtype=np.float32)
    return model.ForecastAdapter(FakeBackbone(), arm, basis, ed_mode=ed_mode)


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.x = torch.randn(4, 512, 4)
        self.y = torch.randn(4, 48, 4)
        self.mask = torch.rand(4, 48, 4) > 0.2

    def test_modes_preserve_paired_initial_state_and_function(self):
        variants = [example(mode) for mode in ['current', 'fixed_ed', 'slow_ed']]
        states = [m.adapter_state() for m in variants]
        for state in states[1:]:
            for key in states[0]:
                self.assertTrue(torch.equal(states[0][key], state[key]))
        predictions = [m(self.x) for m in variants]
        self.assertTrue(all(torch.equal(predictions[0], p) for p in predictions[1:]))

    def test_residual_linear_identity_is_not_raw_bypass(self):
        m = example()
        with torch.no_grad():
            m.residual.up.weight.normal_(std=0.1)
            m.encoder.weight.add_(0.05)
        z = m.encoder(self.x)
        rearranged = m.residual(self.x) + m.decoder(m.backbone_point(z) - m.residual(z))
        torch.testing.assert_close(m(self.x), rearranged, atol=2e-6, rtol=2e-6)

    def test_fixed_state_serializes_and_restores_frozen_ed(self):
        m = example('fixed_ed')
        state = m.adapter_state()
        self.assertIn('encoder.weight', state)
        self.assertIn('decoder.weight', state)
        with torch.no_grad():
            m.encoder.weight.add_(1)
        m.restore_adapter(state)
        self.assertTrue(torch.equal(m.encoder.weight, state['encoder.weight']))
        self.assertFalse(m.encoder.weight.requires_grad)
        self.assertFalse(m.decoder.weight.requires_grad)

    def test_encoder_gradient_crosses_frozen_backbone(self):
        for mode in ['current', 'slow_ed']:
            m = example(mode)
            main, _ = m.components(self.x)
            main.square().mean().backward()
            self.assertGreater(float(m.encoder.weight.grad.abs().max()), 0)
            self.assertIsNone(m.backbone.scale.grad)
            self.assertFalse(m.backbone.scale.requires_grad)

    def test_fixed_shortcut_preserves_nonzero_g_gradient(self):
        m = example('fixed_ed')
        with torch.no_grad():
            m.residual.up.weight.normal_(std=0.1)
        grads = []
        outputs = []
        for shortcut in [True, False]:
            main, correction = m.components(self.x, fixed_shortcut=shortcut)
            outputs.append(main + correction)
            loss = model.macro_loss(outputs[-1], self.y, self.mask)
            grads.append(torch.autograd.grad(loss, tuple(m.residual.parameters())))
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        for a, b in zip(*grads):
            self.assertGreater(float(a.abs().max()), 0)
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_masked_microbatch_gradient_matches_effective_batch(self):
        full = example('slow_ed')
        split = copy.deepcopy(full)
        self.mask[:3, :, 1] = False
        loss = model.macro_loss(full(self.x), self.y, self.mask)
        loss.backward()
        reported = run.backward_batch(split, self.x, self.y, self.mask, 'mse', 1)
        self.assertAlmostEqual(float(loss.detach()), reported, places=5)
        for a, b in zip(full.parameters(), split.parameters()):
            if a.requires_grad:
                torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=2e-6)

    def test_optimizer_updates_g_and_keeps_fixed_ed(self):
        m = example('fixed_ed')
        before = m.adapter_state()
        optimizer, _ = run.optimizer_for(m, {'lr': 0.001, 'ed_mode': 'fixed_ed'})
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            model.macro_loss(m(self.x), self.y, self.mask).backward()
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], 1)
            optimizer.step()
        after = m.adapter_state()
        for key in ['encoder.weight', 'decoder.weight']:
            self.assertTrue(torch.equal(before[key], after[key]))
        for key in ['residual.down.weight', 'residual.up.weight']:
            self.assertFalse(torch.equal(before[key], after[key]))

    def test_slow_scheduler_preserves_ratio_even_at_epsilon_boundary(self):
        m = example('slow_ed')
        optimizer, scheduler = run.optimizer_for(m, {'lr': 0.001, 'ed_lr': 0.0001, 'ed_mode': 'slow_ed'})
        optimizer.param_groups[0]['lr'] = 1e-7
        optimizer.param_groups[1]['lr'] = 1e-8
        for _ in range(4):
            run.step_scheduler(scheduler, 1.0, 'slow_ed')
        self.assertEqual(optimizer.param_groups[0]['lr'], 5e-8)
        self.assertEqual(optimizer.param_groups[1]['lr'], 5e-9)

    def test_integrated_path_checks_and_probe_schedule(self):
        for mode in ['fixed_ed', 'slow_ed']:
            audit = run.initial_path_check(example(mode), self.x, self.y, self.mask)
            self.assertEqual(audit['initial_correction_max'], 0)
        origins = np.arange(512, 2500)
        a = run.phase_sample(origins, 92601, 0, 96)
        b = run.phase_sample(origins, 92601, 0, 96)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(np.bincount(a % 24), np.full(24, 4))

    def test_approved_specs_and_independent_budget_guards(self):
        specs = json.loads((run.HERE / 'first_specs.json').read_text(encoding='utf-8'))
        self.assertEqual(len(specs), 8)
        for spec in specs:
            run.validate_spec(spec)
        self.assertEqual(run.PROTOCOL['fit_attempt_budget'], 20)
        self.assertEqual(run.PROTOCOL['gpu_budget_seconds'], 14400)
        job = run.GPUJob('synthetic')
        job.used = 14399
        job.started = time.perf_counter() - 2
        with self.assertRaisesRegex(RuntimeError, 'GPU_TOTAL_BUDGET_REACHED'):
            job.check_limits()
        with tempfile.TemporaryDirectory() as temp, patch.object(run, 'CACHE', Path(temp)):
            with self.assertRaisesRegex(RuntimeError, 'V2_STORAGE_BUDGET_REACHED'):
                run.require_storage(run.PROTOCOL['storage_budget_bytes'] + 1)


if __name__ == '__main__':
    unittest.main()
