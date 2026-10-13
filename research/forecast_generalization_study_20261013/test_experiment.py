"""Information, numerical and deployment contracts for the new campaign."""
import unittest
import numpy as np
import torch
import experiment as ex


class Contracts(unittest.TestCase):
    def setUp(self):
        ex.DEVICE = 'cpu'
        ex.previous.DEVICE = 'cpu'
        torch.set_num_threads(2)

    def test_variable_window_boundary_ignores_missing_payload(self):
        for length in (128, 512):
            raw = np.ones((1, length, 2), np.float32)*8
            mask = np.ones_like(raw, dtype=bool)
            mask[:, 0, 0] = False
            mask[:, 1, 1] = False
            raw[:, 0, 0] = np.nan
            raw[:, 1, 1] = 1e8
            actual = ex.preprocess(raw, mask, np.array([2, 4], np.float32), np.array([2, 2], np.float32))
            np.testing.assert_array_equal(actual[0, :2], [[0, 2], [3, 2]])
            self.assertEqual(actual.shape[1], length)

    def test_same_capacity_and_initial_weights_both_receptive_fields(self):
        a = ex.Codec(np.zeros(8, np.float32), np.eye(8, 2, dtype=np.float32), 92601, 127)
        b = ex.Codec(np.zeros(8, np.float32), np.eye(8, 2, dtype=np.float32), 92601, 31)
        self.assertEqual(ex.previous.state_hash(a), ex.previous.state_hash(b))
        self.assertEqual(sum(p.numel() for p in a.parameters()), sum(p.numel() for p in b.parameters()))
        for model, receptive_field in ((a, 127), (b, 31)):
            self.assertEqual(1+2*sum(layer.dilation[0] for layer in model.temporal), receptive_field)
            with torch.no_grad():
                model.output.weight.fill_(.1)
            x = torch.randn(2, 512, 8)
            altered = x.clone()
            altered[:, :-receptive_field] += 100
            torch.testing.assert_close(model.encode(x)[:, -1], model.encode(altered)[:, -1], atol=2e-6, rtol=0)
            torch.testing.assert_close(model.encode(x)[:, -1], model.encode(x[:, -receptive_field:])[:, -1], atol=2e-6, rtol=0)

    def test_old_codec_exact_initial_state_compatibility(self):
        mean = np.arange(8, dtype=np.float32)*.1
        basis = np.eye(8, 2, dtype=np.float32)
        new = ex.Codec(mean, basis, 92601)
        old = ex.previous.Codec(mean, basis, 92601)
        self.assertEqual(ex.previous.state_hash(new), ex.previous.state_hash(old))
        x = torch.randn(2, 512, 8)
        torch.testing.assert_close(new.encode(x), old.encode(x), rtol=0, atol=0)

    def test_full_rank_identity_reconstructs_without_training(self):
        codec = ex.Codec(np.zeros(4, np.float32), np.eye(4, dtype=np.float32), 92601)
        x = torch.randn(2, 128, 4)
        torch.testing.assert_close(codec.decode(codec.encode(x)), x, rtol=0, atol=0)

    def test_block_features_require_and_use_512_positions(self):
        x = np.arange(2*512*3, dtype=np.float32).reshape(2, 512, 3)
        np.testing.assert_array_equal(ex.block_features(x), ex.block_features(torch.from_numpy(x)).numpy())
        self.assertEqual(ex.block_features(x).shape, (2, 99))
        with self.assertRaises(AssertionError):
            ex.block_features(x[:, -128:])

    def test_self_lag_ridge_numpy_torch_parity(self):
        rng = np.random.default_rng(4)
        weights = rng.normal(size=(33, 48, 3))
        biases = rng.normal(size=(48, 3))
        x = rng.normal(size=(8, 512, 3)).astype(np.float32)
        model = ex.SelfRidge(weights, biases)
        expected = np.einsum('nlc,lhc->nhc', x[:, -(np.arange(33)+1), :], weights)+biases
        actual = model(torch.from_numpy(x)).numpy()
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=5e-6)


if __name__ == '__main__':
    unittest.main()
