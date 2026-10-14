"""Meaningful CPU contracts for causal learned binary future-pattern models."""
import copy
import unittest
import numpy as np
import torch
from torch import nn
import model as bm


def fixture(method='B_GUMBEL', seed=92601):
    rng = np.random.default_rng(31)
    basis = np.linalg.qr(rng.normal(size=(4, 2)))[0].astype(np.float32)
    delta = rng.normal(size=(5, 4, 4)).astype(np.float32)
    mask = np.ones_like(delta, dtype=bool)
    origins = np.array([0, 6, 12, 18, 40], dtype=np.int64)
    return bm.FutureBiHash(np.zeros(4, np.float32), basis, seed, method,
        delta, mask, origins, horizon=4, representation=8, bits=7)


class TinyFrozenForecast(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('scale', torch.tensor(.07))
    def forward(self, z):
        return z[:, -1:] + self.scale * z.mean(1, keepdim=True).expand(-1, 4, -1)


class BinaryTests(unittest.TestCase):
    def test_each_bit_has_two_classes_and_sign_tie_is_positive(self):
        logits = torch.tensor([[0., 2., -3., .5]])
        classes = bm.binary_class_logits(logits)
        self.assertEqual(tuple(classes.shape), (1, 4, 2))
        self.assertTrue(torch.equal(classes[..., 0], logits))
        self.assertTrue(torch.equal(classes[..., 1], -logits))
        code = bm.binary_code(logits, training=False, temperature=.25, mode='gumbel')
        self.assertTrue(torch.equal(code, torch.tensor([[1., 1., -1., 1.]])))
        self.assertNotEqual(float(code.sum()), 1.)

    def test_soft_and_st_share_noise_but_st_has_binary_forward_and_gradient(self):
        logits = torch.linspace(-1, 1, 28).reshape(4, 7).requires_grad_(True)
        soft = bm.binary_code(logits, True, .7, 'gumbel', torch.Generator().manual_seed(3))
        hard = bm.binary_code(logits, True, .7, 'st', torch.Generator().manual_seed(3))
        self.assertTrue(((soft > -1) & (soft < 1)).all())
        self.assertTrue((hard.abs() == 1).all())
        soft.square().sum().backward(retain_graph=True)
        soft_grad = logits.grad.clone(); logits.grad.zero_()
        # Same upstream gradient makes the ST derivative equal to the soft derivative.
        (hard * soft.detach()).sum().backward()
        self.assertTrue(torch.allclose(logits.grad, soft_grad / 2, atol=2e-7))
        self.assertGreater(float(logits.grad.norm()), 0.)

    def test_all_masked_attention_is_finite_zero(self):
        q = torch.randn(2, 6, 7, requires_grad=True)
        p = torch.randn(5, 7, requires_grad=True)
        eligible = torch.zeros(2, 6, 5, dtype=torch.bool)
        attention = bm.masked_attention(q, p, eligible)
        self.assertTrue(torch.equal(attention, torch.zeros_like(attention)))
        attention.sum().backward()
        self.assertTrue(torch.isfinite(q.grad).all())
        eligible[:, 3:, :2] = True
        attention = bm.masked_attention(q, p, eligible)
        self.assertTrue(torch.allclose(attention[:, 3:].sum(-1), torch.ones(2, 3)))
        self.assertTrue((attention[:, :, 2:] == 0).all())


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.x = torch.arange(2 * 32 * 4, dtype=torch.float32).reshape(2, 32, 4) / 99
        self.origins = torch.tensor([32, 38])
        self.forecaster = TinyFrozenForecast()

    def test_all_methods_have_common_initial_states_and_do_not_consume_global_rng(self):
        torch.manual_seed(51); before = torch.get_rng_state().clone()
        first = fixture('NO_ALIGN')
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for method in bm.METHODS:
            other = fixture(method)
            for name, value in first.state_dict().items():
                self.assertTrue(torch.equal(value, other.state_dict()[name]), (method, name))

    def test_prefix_survives_later_raw_data_and_ineligible_memory_future_changes(self):
        model = fixture().eval()
        with torch.no_grad(): model.codec.codec.output.weight.fill_(.03)
        base, details = model.encode_with_memory(self.x[:1], self.origins[:1])
        changed = self.x[:1].clone(); changed[:, 16:] += 50
        actual, _ = model.encode_with_memory(changed, self.origins[:1])
        self.assertTrue(torch.equal(base[:, :16], actual[:, :16]))
        self.assertTrue((details['eligible'][:, :3] == 0).all())
        self.assertTrue(torch.equal(base[:, :3], details['z'][:, :3]))
        model.bank_delta[2:] += 100
        actual, _ = model.encode_with_memory(self.x[:1], self.origins[:1])
        self.assertTrue(torch.equal(base[:, :15], actual[:, :15]))

    def test_empty_memory_falls_back_to_student_forecast_exactly(self):
        model = fixture().eval()
        origins = torch.tensor([-100, -90])
        prediction, details = model(self.x, origins, self.forecaster, return_details=True)
        baseline = model.codec.forward_path(self.x, self.forecaster)
        self.assertTrue(torch.equal(prediction, baseline))
        self.assertTrue(torch.equal(details['attention'], torch.zeros_like(details['attention'])))

    def test_inference_ablations_use_same_state_and_no_current_future(self):
        model = fixture().eval()
        baseline_state = copy.deepcopy(model.state_dict())
        regular, details = model(self.x, self.origins, self.forecaster, return_details=True)
        student = model(self.x, self.origins, self.forecaster, variant='student')
        memory = model(self.x, self.origins, self.forecaster, variant='memory')
        uniform = model(self.x, self.origins, self.forecaster, variant='uniform')
        relaxed = model(self.x, self.origins, self.forecaster, variant='tanh')
        self.assertTrue(torch.equal(student, details['student_prediction']))
        self.assertTrue(torch.equal(memory, details['memory_prediction']))
        self.assertTrue(torch.isfinite(uniform).all()); self.assertTrue(torch.isfinite(relaxed).all())
        self.assertFalse(torch.equal(relaxed, regular))
        for key, value in baseline_state.items(): self.assertTrue(torch.equal(value, model.state_dict()[key]))

    def test_prediction_does_not_depend_on_other_batch_members(self):
        model = fixture().eval()
        batch = model(self.x, self.origins, self.forecaster)
        single = model(self.x[:1], self.origins[:1], self.forecaster)
        self.assertTrue(torch.allclose(batch[:1], single, atol=3e-7, rtol=2e-6))

    def test_current_target_only_changes_losses_not_prediction(self):
        for method in ('B_GUMBEL', 'A_BIHASH_GUMBEL'):
            model = fixture(method).eval()
            y = torch.randn(2, 4, 4); mask = torch.ones_like(y, dtype=torch.bool)
            before = model(self.x, self.origins, self.forecaster)
            first = model.losses(self.x, self.origins, y, mask, self.forecaster)
            second = model.losses(self.x, self.origins, y + 20, mask, self.forecaster)
            self.assertTrue(torch.equal(first['prediction'], second['prediction']))
            self.assertTrue(torch.equal(before, first['prediction']))
            self.assertNotEqual(float(first['rec'].detach()), float(second['rec'].detach()))

    def test_detach_encoder_auxiliary_gradient_zero_but_main_unchanged(self):
        regular = fixture('A_BIHASH_GUMBEL').train()
        detached = fixture('A_BIHASH_DETACH_E').train()
        y = torch.randn(2, 4, 4); mask = torch.ones_like(y, dtype=torch.bool)
        for model in (regular, detached):
            loss = model.losses(self.x, self.origins, y, mask, self.forecaster)
            gradients = torch.autograd.grad(loss['aux'], tuple(model.codec.parameters()), allow_unused=True)
            norm = sum(float(g.abs().sum()) for g in gradients if g is not None)
            if model is detached: self.assertEqual(norm, 0.)
            else: self.assertGreater(norm, 0.)
        a = regular.losses(self.x, self.origins, y, mask, self.forecaster)['main']
        b = detached.losses(self.x, self.origins, y, mask, self.forecaster)['main']
        ga = torch.autograd.grad(a, tuple(regular.codec.parameters()), allow_unused=True)
        gb = torch.autograd.grad(b, tuple(detached.codec.parameters()), allow_unused=True)
        for one, two in zip(ga, gb):
            self.assertEqual(one is None, two is None)
            if one is not None: self.assertTrue(torch.equal(one, two))

    def test_detach_with_separate_codec_clipping_preserves_actual_main_update(self):
        baseline = fixture('NO_ALIGN').train()
        detached = fixture('A_BIHASH_DETACH_E').train()
        y = torch.randn(2, 4, 4); mask = torch.ones_like(y, dtype=torch.bool)
        for model in (baseline, detached):
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0.)
            losses = model.losses(self.x, self.origins, y, mask, self.forecaster)
            (losses['main'] + .1 * losses['aux'] + .1 * losses['rec']).backward()
            codec_parameters = list(model.codec.parameters())
            side_parameters = [p for name, p in model.named_parameters() if not name.startswith('codec.')]
            torch.nn.utils.clip_grad_norm_(codec_parameters, 1.)
            torch.nn.utils.clip_grad_norm_(side_parameters, 1.)
            optimizer.step()
        for name, value in baseline.codec.state_dict().items():
            self.assertTrue(torch.equal(value, detached.codec.state_dict()[name]), name)

    def test_dedicated_gumbel_randomness_and_checkpoint_continuation(self):
        one = fixture().train()
        before = torch.get_rng_state().clone()
        one(self.x, self.origins, self.forecaster)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        two = fixture().train(); two.load_state_dict(copy.deepcopy(one.state_dict()))
        expected = one(self.x, self.origins, self.forecaster)
        self.assertTrue(torch.equal(expected, two(self.x, self.origins, self.forecaster)))

    def test_shuffle_requires_bijective_nonoverlapping_future_windows(self):
        model = fixture('A_BIHASH_SHUFFLE').train()
        y = torch.randn(2, 4, 4); mask = torch.ones_like(y, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, 'EXPLICIT_NONOVERLAP'):
            model.losses(self.x, self.origins, y, mask, self.forecaster)
        with self.assertRaisesRegex(ValueError, 'NOT_BIJECTION'):
            model.losses(self.x, self.origins, y, mask, self.forecaster, pair_permutation=[0, 0])
        with self.assertRaisesRegex(ValueError, 'OVERLAP'):
            model.losses(self.x, self.origins, y, mask, self.forecaster, pair_permutation=[0, 1])
        losses = model.losses(self.x, self.origins, y, mask, self.forecaster, pair_permutation=[1, 0])
        self.assertTrue(torch.isfinite(losses['aux']))
        self.assertEqual(losses['details']['valid_pair_count'], 2)

    def test_missing_target_values_do_not_enter_future_encoder_or_reconstruction(self):
        model = fixture().eval()
        delta = torch.randn(2, 4, 4); observed = torch.ones_like(delta, dtype=torch.bool)
        observed[:, :, 2:] = False
        changed = delta.clone(); changed[:, :, 2:] = torch.nan
        a, ar = model.future_autoencoder(delta, observed)
        b, br = model.future_autoencoder(changed, observed)
        self.assertTrue(torch.equal(a, b)); self.assertTrue(torch.equal(ar, br))
        loss, _ = model.future_autoencoder.reconstruction_loss(changed, observed)
        self.assertTrue(torch.isfinite(loss))

    def test_auxiliary_heads_are_not_executed_during_A_or_baseline_prediction(self):
        for method in ('NO_ALIGN', 'A_BIHASH_GUMBEL'):
            model = fixture(method).eval()
            for parameter in model.student_projection.parameters():
                parameter.data.fill_(torch.nan)
            for parameter in model.future_autoencoder.parameters():
                parameter.data.fill_(torch.nan)
            prediction = model(self.x, self.origins, self.forecaster)
            self.assertTrue(torch.isfinite(prediction).all())

    def test_three_real_optimizer_steps_change_both_maps_encoder_and_future_encoder(self):
        for method in ('B_GUMBEL', 'A_BIHASH_GUMBEL'):
            model = fixture(method).train()
            initial = {k: v.detach().clone() for k, v in model.state_dict().items()}
            fixed = self.forecaster.state_dict()['scale'].clone()
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            y = torch.randn(2, 4, 4); mask = torch.ones_like(y, dtype=torch.bool)
            for step in range(3):
                optimizer.zero_grad(set_to_none=True)
                losses = model.losses(self.x, self.origins, y, mask, self.forecaster, update=step)
                total = losses['main'] + .1 * losses['aux'] + .1 * losses['rec']
                self.assertTrue(torch.isfinite(total))
                total.backward()
                for p in model.parameters():
                    if p.grad is not None: self.assertTrue(torch.isfinite(p.grad).all())
                optimizer.step()
            for prefix in ('student_hash.', 'future_hash.', 'codec.codec.project.', 'future_autoencoder.encoder.'):
                self.assertTrue(any(not torch.equal(v, initial[k]) for k, v in model.state_dict().items() if k.startswith(prefix)), (method, prefix))
            self.assertTrue(torch.equal(self.forecaster.state_dict()['scale'], fixed))

    def test_frozen_maps_are_registered_but_unchanged(self):
        for method in ('B_FIXED_GUMBEL', 'A_BIHASH_FIXED_MAPS'):
            model = fixture(method)
            self.assertTrue(all(not p.requires_grad for p in model.student_hash.parameters()))
            self.assertTrue(all(not p.requires_grad for p in model.future_hash.parameters()))

    def test_continuous_controls_use_the_same_trainable_projection_maps(self):
        for method in ('B_COSINE', 'A_CONTINUOUS'):
            model = fixture(method).train()
            y = torch.randn(2, 4, 4); mask = torch.ones_like(y, dtype=torch.bool)
            losses = model.losses(self.x, self.origins, y, mask, self.forecaster)
            (losses['main'] + .1 * losses['aux'] + .1 * losses['rec']).backward()
            for projection in (model.student_hash, model.future_hash):
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in projection.parameters() if p.grad is not None), 0.)

    def test_checkpoint_roundtrip_and_precomputed_deployment_memory(self):
        model = fixture().eval()
        expected = model(self.x, self.origins, self.forecaster)
        other = fixture().eval(); other.load_state_dict(copy.deepcopy(model.state_dict()))
        self.assertTrue(torch.equal(expected, other(self.x, self.origins, self.forecaster)))
        model.freeze_memory()
        self.assertTrue(torch.equal(expected, model(self.x, self.origins, self.forecaster)))
        other.load_state_dict(copy.deepcopy(model.state_dict()))
        self.assertTrue(torch.equal(expected, other(self.x, self.origins, self.forecaster)))


if __name__ == '__main__': unittest.main()
