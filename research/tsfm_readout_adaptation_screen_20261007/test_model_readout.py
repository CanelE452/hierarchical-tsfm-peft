"""CPU synthetic checks; no model weights or dataset archives are loaded."""
import copy
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from model_readout import Readout, compute_loss


class SyntheticHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Linear(512, 8)
        self.output_layer = nn.Linear(8, 576)
        self.residual_layer = nn.Linear(512, 576)
        self.dropout = nn.Dropout(.1)

    def forward(self, x):
        return self.dropout(self.output_layer(torch.relu(self.hidden_layer(x)))) + self.residual_layer(x)


class SyntheticNorm(nn.Module):
    def inverse(self, x, loc_scale):
        loc, scale = loc_scale
        return x * scale + loc


class SyntheticBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.prefix = nn.Linear(512, 512, bias=False)
        self.output_patch_embedding = SyntheticHead()
        self.instance_norm = SyntheticNorm()
        self.chronos_config = SimpleNamespace(quantiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9], prediction_length=64)
        self.num_quantiles = 9
        self.register_buffer('quantiles', torch.tensor(self.chronos_config.quantiles), persistent=False)

    def encode(self, context):
        loc = context.mean(-1, keepdim=True)
        scale = context.std(-1, correction=0, keepdim=True).clamp_min(1e-5)
        encoded = self.prefix((context - loc) / scale).unsqueeze(1)
        return encoded, (loc, scale), encoded, torch.ones(context.shape[0], 1, device=context.device)

    def decode(self, input_embeds, attention_mask, encoded):
        return encoded


def features(channels=3, batch=4):
    generator = torch.Generator().manual_seed(92)
    return {'h': torch.randn(batch, channels, 512, generator=generator),
            'loc': torch.randn(batch, channels, 1, generator=generator),
            'scale': torch.rand(batch, channels, 1, generator=generator) + .5,
            'point': torch.randn(batch, 48, channels, generator=generator)}


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91)

    def test_affine_identity_and_channel_mixing(self):
        model = Readout(SyntheticBackbone(), 3, 'OUTPUT_AFFINE')
        value = features()
        self.assertTrue(torch.equal(model.from_features(value), value['point']))
        with torch.no_grad():
            model.weight.zero_()
            model.weight[1, 0] = 2.
            model.bias.copy_(torch.tensor([1., 2., 3.]))
        result = model.from_features(value)
        torch.testing.assert_close(result[..., 0], 2 * value['point'][..., 1] + 1)
        torch.testing.assert_close(result[..., 1], torch.full_like(result[..., 1], 2))
        torch.testing.assert_close(result[..., 2], torch.full_like(result[..., 2], 3))

    def test_native_head_inverse_coordinates_and_gradient(self):
        model = Readout(SyntheticBackbone(), 3, 'NATIVE_HEAD')
        value = features()
        raw = model.backbone.output_patch_embedding(value['h'].reshape(12, 1, 512)).reshape(4, 3, 9, 64)
        expected = (raw[:, :, 4, :48] * value['scale'] + value['loc']).transpose(1, 2)
        pred = model.from_features(value)
        torch.testing.assert_close(pred, expected, rtol=0, atol=0)
        target = torch.zeros_like(pred)
        loss = compute_loss(pred, target, torch.ones_like(pred, dtype=torch.bool))
        explicit = expected.square().sum((0, 1)).div(4 * 48).mean()
        torch.testing.assert_close(loss, explicit)
        loss.backward()
        for name, tensor in model.named_parameters():
            if name in model.trainable_names:
                self.assertIsNotNone(tensor.grad)
                self.assertTrue(torch.isfinite(tensor.grad).all())
                self.assertGreater(float(tensor.grad.abs().sum()), 0.)
            else:
                self.assertIsNone(tensor.grad)

    def test_shared_denominator_missing_accumulation(self):
        value = features()
        target = torch.randn(4, 48, 3)
        mask = torch.ones_like(target, dtype=torch.bool)
        mask[0] = False
        mask[1, :27, 0] = False
        mask[2, :39, 1] = False
        mask[..., 2] = False
        target[~mask] = float('nan')
        count = mask.sum((0, 1))
        reference_model = Readout(SyntheticBackbone(), 3, 'OUTPUT_AFFINE')
        initial = reference_model.trainable_state()
        reference = compute_loss(reference_model.from_features(value), target, mask)
        reference.backward()
        reference_grad = {name: tensor.grad.clone() for name, tensor in reference_model.named_parameters() if tensor.requires_grad}
        for microbatch in (4, 2, 1):
            model = Readout(SyntheticBackbone(), 3, 'OUTPUT_AFFINE')
            model.load_trainable_state(initial)
            total = torch.zeros(())
            for start in range(0, 4, microbatch):
                batch = {key: tensor[start:start + microbatch] for key, tensor in value.items()}
                part = compute_loss(model.from_features(batch), target[start:start + microbatch], mask[start:start + microbatch], count)
                total += part.detach()
                part.backward()
            torch.testing.assert_close(total, reference.detach(), rtol=1e-6, atol=1e-7)
            for name, tensor in model.named_parameters():
                if tensor.requires_grad:
                    torch.testing.assert_close(tensor.grad, reference_grad[name], rtol=1e-6, atol=1e-7)
        with self.assertRaises(ValueError):
            compute_loss(value['point'], target, torch.zeros_like(mask))

    def test_prefix_detached_eval_state_and_nonpersistent_buffer_hash(self):
        model = Readout(SyntheticBackbone(), 3, 'NATIVE_HEAD')
        model.train()
        self.assertTrue(all(not module.training for module in model.modules()))
        x = torch.randn(4, 512, 3, requires_grad=True)
        value = model.features(x)
        self.assertEqual({key: tuple(tensor.shape) for key, tensor in value.items()},
                         {'h': (4, 3, 512), 'loc': (4, 3, 1), 'scale': (4, 3, 1), 'point': (4, 48, 3)})
        self.assertTrue(all(not tensor.requires_grad and not torch.is_inference(tensor) for tensor in value.values()))
        with torch.inference_mode():
            inference_called = model.features(x)
        self.assertTrue(all(not torch.is_inference(tensor) for tensor in inference_called.values()))
        frozen_before = model.frozen_hash()
        state = model.trainable_state()
        with torch.no_grad():
            next(model.backbone.output_patch_embedding.parameters()).add_(1)
        self.assertEqual(model.frozen_hash(), frozen_before)
        model.load_trainable_state(state)
        for name, tensor in model.trainable_state().items():
            self.assertTrue(torch.equal(tensor, state[name]))
        with torch.no_grad():
            model.backbone.quantiles.add_(1)
        self.assertNotEqual(model.frozen_hash(), frozen_before)

    def test_state_allowlist_rejects_prefix_and_wrong_shape(self):
        model = Readout(SyntheticBackbone(), 3, 'NATIVE_HEAD')
        state = model.trainable_state()
        altered = copy.deepcopy(state)
        altered['backbone.prefix.weight'] = model.backbone.prefix.weight.detach().clone()
        with self.assertRaises(ValueError):
            model.load_trainable_state(altered)
        altered = copy.deepcopy(state)
        first = next(iter(altered))
        altered[first] = altered[first].flatten()
        with self.assertRaises(ValueError):
            model.load_trainable_state(altered)


if __name__ == '__main__':
    torch.set_num_threads(4)
    unittest.main()
