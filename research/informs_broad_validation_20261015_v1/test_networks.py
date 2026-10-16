"""CPU synthetic tests; no data, foundation weights, downloads, or CUDA calls."""
import importlib
import unittest

import torch
from torch import nn

try:
    networks = importlib.import_module("networks")
except ModuleNotFoundError:
    networks = None


class ToyFoundation:
    def __init__(self, horizon):
        self.horizon, self.shapes = horizon, []

    def __call__(self, z):
        self.shapes.append(tuple(z.shape))
        return (z[:, -1:] + .2 * (z[:, -1:] - z.mean(1, keepdim=True))).expand(-1, self.horizon, -1)


class ToyQv(nn.Module):
    def __init__(self):
        super().__init__()
        self.q, self.v = nn.Linear(5, 5), nn.Linear(5, 5)

    def forward(self, x):
        return self.q(x) + self.v(x)


class NetworksTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def api(self, name):
        self.assertIsNotNone(networks, "networks implementation is missing")
        self.assertTrue(hasattr(networks, name), name + " is not implemented")
        return getattr(networks, name)

    def adapter(self, kind="B0", length=64, horizon=12, channels=5, k=5, **kwargs):
        return self.api("ForecastAdapter")(kind, length, horizon, channels, k,
            torch.zeros(channels), torch.eye(channels)[:, :k], **kwargs)

    def test_variable_length_horizon_and_cp0_identity(self):
        for length, horizon in ((512, 48), (512, 12), (364, 28)):
            x = torch.randn(2, length, 5)
            for kind in ("B0", "B2", "B4"):
                with self.subTest(length=length, horizon=horizon, kind=kind):
                    model = self.adapter(kind, length, horizon)
                    foundation = ToyFoundation(horizon)
                    torch.testing.assert_close(model(x, foundation), ToyFoundation(horizon)(x), atol=1e-6, rtol=1e-5)
                    self.assertEqual(foundation.shapes, [(2, length, 5)])
                    self.assertEqual(tuple(model.D.shape), (7, 5, 5))

    def test_prefix_registry_and_actual_foundation_streams(self):
        model = self.adapter(channels=7, k=7, k_grid=(2, 4, 7))
        x, foundation = torch.randn(2, 64, 7), ToyFoundation(12)
        for k in (2, 4, 7):
            self.assertEqual(tuple(model(x, foundation, latent_k=k).shape), (2, 12, 7))
        self.assertEqual([s[-1] for s in foundation.shapes], [2, 4, 7])
        with self.assertRaises(ValueError): model(x, foundation, latent_k=3)

    def test_lag_major_D_prefix_matches_explicit_sum(self):
        model = self.adapter(channels=4, k=4, k_grid=(2, 4))
        x, foundation = torch.randn(2, 64, 4), ToyFoundation(12)
        with torch.no_grad(): model.D.copy_(torch.randn_like(model.D))
        z = model.encode(x, latent_k=2); f = foundation(z)
        joined = torch.cat((z, f), 1)
        expected = x[:, -1:].expand(-1, 12, -1).clone()
        for i, lag in enumerate((0, 1, 2, 4, 8, 16, 32)):
            delta = joined[:, 64-lag:64-lag+12] - z[:, 63-lag:64-lag]
            for k in range(2): expected += delta[:, :, k:k+1] * model.D[i, k]
        torch.testing.assert_close(model(x, foundation, latent_k=2), expected)

    def test_causal7_prefix_and_bidirectional_control(self):
        x, changed = torch.randn(2, 64, 5), None
        changed = x.clone(); changed[:, 41:] += 20
        for encoder in ("causal7", "bidirectional7"):
            model = self.adapter(encoder_kind=encoder)
            with torch.no_grad(): model.lag_weights.fill_(.1)
            a, b = model.encode(x), model.encode(changed)
            if encoder == "causal7": torch.testing.assert_close(a[:, 32:41], b[:, 32:41], atol=0, rtol=0)
            else: self.assertGreater(float((a[:, 32:41]-b[:, 32:41]).detach().abs().max()), 0)
        with self.assertRaises(ValueError): self.adapter("B2", encoder_kind="causal7")

    def test_first_and_second_front_gradient_flow(self):
        for kind in ("B2", "B4"):
            model = self.adapter(kind, channels=4, k=2)
            optimizer = torch.optim.SGD(model.parameters(), lr=.01)
            x, y = torch.randn(2, 64, 4), torch.randn(2, 12, 4)
            (model(x, ToyFoundation(12))-y).square().mean().backward()
            self.assertGreater(sum(float(p.grad.abs().sum()) for n,p in model.front.named_parameters() if "output" in n and p.grad is not None), 0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
            (model(x, ToyFoundation(12))-y).square().mean().backward()
            self.assertGreater(sum(float(p.grad.abs().sum()) for n,p in model.front.named_parameters() if "input" in n and p.grad is not None), 0)

    def test_large_channel_shape_uses_bounded_feature_width(self):
        model = self.adapter("B2", length=364, horizon=28, channels=321, k=81)
        out = model(torch.randn(1, 364, 321), ToyFoundation(28))
        self.assertEqual(tuple(out.shape), (1, 28, 321))
        self.assertEqual(model.front.mixer.feature.input.out_features, 32)

    def test_G_and_conditional_initial_predictions(self):
        x = torch.randn(2, 64, 5)
        direct = self.api("DirectG")(64, 12)
        torch.testing.assert_close(direct(x), x[:, -1:].expand(-1, 12, -1), atol=0, rtol=0)
        head = self.api("ConditionalHead")(64, 12)
        g, b = torch.randn(2, 12, 5), torch.randn(2, 12, 5)
        for control in (False, True):
            torch.testing.assert_close(head.forecastforward(x, g, b, control=control), g, atol=0, rtol=0)
        with torch.no_grad(): head.output.weight.fill_(.1)
        torch.testing.assert_close(head(x, g, b, control=True), head(x, g, b+100, control=True), atol=0, rtol=0)
        self.assertGreater(float((head(x,g,b)-head(x,g,b+100)).detach().abs().max()), 0)

    def test_adapts_affine_port_is_distinct_from_anchored_lag_decoder(self):
        model = self.api("AdaPTSLinearPort")(64, 12, 5, 2)
        x, foundation = torch.randn(2, 64, 5), ToyFoundation(12)
        expected = model.decoder(foundation(model.encoder(x)))
        torch.testing.assert_close(model(x, ToyFoundation(12)), expected)
        self.assertIsNotNone(model.encoder.bias); self.assertIsNotNone(model.decoder.bias)
        self.assertFalse(hasattr(model, "D"))
        with torch.no_grad():
            model.decoder.weight.zero_(); model.decoder.bias.fill_(3)
        torch.testing.assert_close(model(x, foundation), torch.full((2,12,5),3.))

    def test_lora_adapter_only_gradients_merge_and_exact_restore(self):
        base = ToyQv().eval(); original = {k:v.clone() for k,v in base.state_dict().items()}
        adapter = self.api("LoRAQvAdapter")(base, rank=2, alpha=4., dropout=0.)
        x = torch.randn(3, 5); baseline = base(x).detach().clone()
        self.assertTrue(all(not p.requires_grad for p in base.parameters()))
        self.assertEqual(sum(p.numel() for p in adapter.parameters()), 40)
        base(x).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in base.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in adapter.parameters() if p.grad is not None),0)
        with torch.no_grad():
            for name,p in adapter.named_parameters():
                if name.endswith("B"): p.normal_(std=.1)
        hooked = base(x).detach().clone(); self.assertGreater(float((hooked-baseline).abs().max()),0)
        state = adapter.adapter_state_dict(); self.assertTrue(all("base" not in k for k in state))
        adapter.merge(); torch.testing.assert_close(base(x),hooked,atol=1e-6,rtol=1e-5)
        self.assertEqual(sum(len(m._forward_hooks) for m in base.modules()),0)
        adapter.unmerge(); torch.testing.assert_close(base(x),hooked,atol=1e-6,rtol=1e-5)
        adapter.load_adapter_state_dict(state); adapter.close(); adapter.close()
        for k,v in base.state_dict().items(): torch.testing.assert_close(v,original[k],atol=0,rtol=0)
        self.assertEqual(sum(len(m._forward_hooks) for m in base.modules()),0)

    def test_lora_exception_context_closes(self):
        base = ToyQv(); original = {k:v.clone() for k,v in base.state_dict().items()}
        with self.assertRaisesRegex(RuntimeError,"injected"):
            with self.api("LoRAQvAdapter")(base,rank=2,alpha=4.):
                raise RuntimeError("injected")
        for k,v in base.state_dict().items(): torch.testing.assert_close(v,original[k],atol=0,rtol=0)
        self.assertEqual(sum(len(m._forward_hooks) for m in base.modules()),0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
