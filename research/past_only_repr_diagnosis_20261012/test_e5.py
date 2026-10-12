"""CPU-only information-contract tests, not scientific training results."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import Budget, access_guard, write


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((HERE / 'model_neural.py').exists(), 'E5 encoder not implemented')
        self.assertTrue((HERE / 'e5.py').exists(), 'E5 runner not implemented')
        from model_neural import TinyEncoder
        from e5 import contrastive_loss, make_batches, training_origins, run_objective, objective_head
        self.Encoder = TinyEncoder
        self.contrastive_loss = contrastive_loss
        self.make_batches = make_batches
        self.training_origins = training_origins
        self.run_objective = run_objective
        self.objective_head = objective_head

    def test_causal_append_and_suffix_invariance(self):
        torch.manual_seed(31)
        model = self.Encoder(3, causal=True).eval()
        x = torch.randn(2, 160, 3)
        altered = x.clone()
        altered[:, 95:] = torch.randn_like(altered[:, 95:]) * 30
        with torch.no_grad():
            z, za, zp = model(x), model(altered), model(x[:, :95])
        self.assertEqual(z.shape, (2, 160, 16))
        torch.testing.assert_close(z[:, :95], za[:, :95], rtol=0, atol=0)
        # Different sequence lengths may select different floating-point kernels.
        torch.testing.assert_close(z[:, :95], zp, rtol=1e-5, atol=1e-6)

    def test_padding_only_changes_information_direction(self):
        torch.manual_seed(3)
        full = self.Encoder(3, causal=False).eval()
        causal = self.Encoder(3, causal=True).eval()
        causal.load_state_dict(full.state_dict())
        self.assertEqual(full.receptive_field, 63)
        self.assertEqual(causal.receptive_field, 63)
        self.assertEqual(full.parameter_count, causal.parameter_count)
        x = torch.randn(1, 150, 3, requires_grad=True)
        full(x)[0, 75].sum().backward()
        self.assertGreater(float(x.grad[:, 76:107].abs().sum()), 0)
        self.assertEqual(float(x.grad[:, :44].abs().sum()), 0)
        self.assertEqual(float(x.grad[:, 107:].abs().sum()), 0)
        x.grad = None
        causal(x)[0, 75].sum().backward()
        self.assertEqual(float(x.grad[:, 76:].abs().sum()), 0)
        self.assertEqual(float(x.grad[:, :13].abs().sum()), 0)

    def test_symmetric_negatives_exclude_near_and_duplicate_times(self):
        torch.manual_seed(9)
        p = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
        q = torch.nn.functional.normalize(torch.randn(6, 8), dim=-1)
        times = torch.tensor([10, 10, 20, 100, 200, 300])
        loss, audit = self.contrastive_loss(p, q, times, receptive_field=63)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(audit['same_timestamp_negative_pairs'], 0)
        self.assertEqual(audit['near_timestamp_negative_pairs'], 0)
        self.assertGreater(audit['negative_pairs'], 0)
        reverse, _ = self.contrastive_loss(q, p, times, receptive_field=63)
        torch.testing.assert_close(loss, reverse)

    def test_unusable_contrastive_batch_is_rejected(self):
        p = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
        with self.assertRaises(ValueError):
            self.contrastive_loss(p, p, torch.tensor([10, 11, 12]), 63)

    def test_training_windows_and_common_order(self):
        origins = self.training_origins((0, 9000))
        self.assertTrue(np.all(origins - 512 >= 0))
        self.assertTrue(np.all(origins + 48 <= 9000))
        self.assertTrue(np.all(np.diff(origins) >= 560))
        a = self.make_batches(origins, 8, 600, 92601)
        b = self.make_batches(origins, 8, 600, 92601)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(a.shape, (600, 8))
        self.assertTrue(all(len(np.unique(row)) == 8 for row in a))

    def test_all_objectives_keep_targets_out_of_encoder_input(self):
        rng = np.random.default_rng(8)
        data = {'x':rng.normal(size=(1200,3)).astype(np.float32),
                'mask':np.ones((1200,3),dtype=bool),'normalization_fit_scope':'FIT_ONLY'}
        origins = np.array([512,1072])
        for condition in ('FULL_REC','CAUSAL_REC','CAUSAL_CONTRAST','CAUSAL_FORECAST'):
            torch.manual_seed(6)
            model = self.Encoder(3,causal=condition!='FULL_REC')
            head = self.objective_head(condition,3)
            seen = []
            handle = model.register_forward_pre_hook(lambda module,args: seen.append(args[0].detach().clone()))
            loss,audit = self.run_objective(model,head,condition,data,origins,'cpu',123)
            loss.backward()
            handle.remove()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(v.shape == (2,512,3) for v in seen))
            if condition == 'CAUSAL_FORECAST':
                self.assertEqual(audit['future_target_values'],2*48*3)
                baseline = np.stack([data['x'][o-512:o] for o in origins])
                np.testing.assert_array_equal(seen[0].numpy(),baseline)
            else:
                self.assertEqual(audit['future_target_values'],0)
            if condition == 'CAUSAL_CONTRAST':
                self.assertEqual(audit['positive_timestamps'],audit['positive_information_cutoffs'])
                self.assertEqual(audit['same_timestamp_negative_pairs'],0)
                self.assertEqual(audit['near_timestamp_negative_pairs'],0)


if __name__ == '__main__':
    access_guard()
    torch.set_num_threads(2)
    with Budget(2, gpu=False, label='E5 CPU information-contract tests') as budget:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests))
        budget.check()
        write(HERE / 'receipts' / f'E5_CPU_TEST_{budget.path.stem}.json',
              {'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
               'science_training_updates': 0, 'gpu_calls': 0, 'budget_ledger': str(budget.path)})
    raise SystemExit(0 if result.wasSuccessful() else 1)
