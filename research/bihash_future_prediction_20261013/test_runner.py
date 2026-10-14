"""Training comparability and TUNE-only selection contracts."""
import unittest
import numpy as np
import train as tr
import probes as pr
import tempfile
from pathlib import Path
import torch
import runtime as rt


class RunnerTests(unittest.TestCase):
    def test_batches_are_fixed_unique_and_have_disjoint_future_windows(self):
        origins = np.arange(256) * 8 + 512
        a = tr.batch_order(origins, 92601)
        b = tr.batch_order(origins, 92601)
        self.assertTrue(np.array_equal(a, b))
        self.assertEqual(a.shape, (600, 8))
        for row in a:
            gaps = np.abs(origins[row, None] - origins[row][None, :])
            self.assertTrue((gaps[~np.eye(8, dtype=bool)] >= 48).all())
            self.assertEqual(len(np.unique(row)), 8)

    def test_impossible_nonoverlap_batch_fails_before_learning(self):
        with self.assertRaisesRegex(ValueError, 'INSUFFICIENT'):
            tr.batch_order(np.arange(12), 92601)

    def test_auxiliary_weights_selected_from_equal_group_TUNE_ratios_only(self):
        cases = []
        for dataset, seed in (('a', 1), ('a', 2), ('b', 1), ('b', 2)):
            cases.append({'dataset': dataset, 'seed': seed, 'method': 'NO_ALIGN', 'weight': 0., 'selected_tune_mse': 2.})
            for method in ('A_CONTINUOUS', 'A_BIHASH_GUMBEL'):
                cases.append({'dataset': dataset, 'seed': seed, 'method': method, 'weight': .03,
                    'selected_tune_mse': 1., 'devval_mse': 100.})
                cases.append({'dataset': dataset, 'seed': seed, 'method': method, 'weight': .1,
                    'selected_tune_mse': 1.2, 'devval_mse': 0.})
            for method in ('B_GUMBEL', 'B_COSINE'):
                cases.append({'dataset': dataset, 'seed': seed, 'method': method, 'weight': 0.,
                    'selected_tune_mse': .8 if method == 'B_COSINE' else 1.3})
        selected = tr.select_methods(cases, expected_groups=4)
        self.assertEqual(selected['weights']['A_CONTINUOUS'], .03)
        self.assertEqual(selected['weights']['A_BIHASH_GUMBEL'], .03)
        self.assertEqual(selected['selected_B_method'], 'B_COSINE')
        # Repeated VAL may be arbitrarily changed without changing the lock.
        for case in cases: case['devval_mse'] = 1e99
        self.assertEqual(selected, tr.select_methods(cases, expected_groups=4))

    def test_incomplete_selection_groups_cannot_silently_select_method(self):
        cases = [{'dataset': 'a', 'seed': 1, 'method': 'NO_ALIGN', 'weight': 0., 'selected_tune_mse': 2.},
            {'dataset': 'a', 'seed': 1, 'method': 'A_CONTINUOUS', 'weight': .03, 'selected_tune_mse': 1.}]
        with self.assertRaisesRegex(ValueError, 'INCOMPLETE'):
            tr.select_methods(cases, expected_groups=6)

    def test_readout_uses_observed_targets_and_recovers_future_delta(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(40, 4)).astype(np.float32)
        v = rng.normal(size=(12, 4)).astype(np.float32)
        w = rng.normal(size=(4, 6)).astype(np.float32)
        y = (x @ w).reshape(40, 3, 2); ty = (v @ w).reshape(12, 3, 2)
        mask = np.ones_like(y, bool); mask[:10, 1, 1] = False
        altered = y.copy(); altered[~mask] = np.nan
        one, record = pr.fit_grouped_ridge(x, y, mask, v, ty, np.ones_like(ty, bool))
        two, other = pr.fit_grouped_ridge(x, altered, mask, v, ty, np.ones_like(ty, bool))
        self.assertTrue(np.array_equal(one['weight'], two['weight']))
        self.assertTrue(np.array_equal(one['bias'], two['bias']))
        self.assertEqual(record['alpha'], .1)
        self.assertLess(np.mean((pr.ridge_predict(v, one) - ty) ** 2), 1e-3)

    def test_cpu_optimizer_checkpoint_npz_selection_roundtrip(self):
        class Limit:
            def check(self): pass
        class Forecaster(torch.nn.Module):
            def forward(self, z): return z[:, -1:].expand(-1, 48, -1) + .03 * z.mean(1, keepdim=True)
        torch.set_num_threads(2)
        rng = np.random.default_rng(9)
        history = {s: rng.normal(size=(n, 512, 3)).astype(np.float32) for s, n in (('fit', 16), ('tune', 8))}
        target = {s: rng.normal(size=(n, 48, 3)).astype(np.float32) for s, n in (('fit', 16), ('tune', 8))}
        observed = {s: np.ones_like(y, bool) for s, y in target.items()}
        origins = {s: np.arange(len(x)) * 48 + 512 for s, x in history.items()}
        bank_indices = np.arange(0, 16, 4)
        pack = {'dataset': 'SYNTHETIC_UNIT_FIXTURE', 'history': history, 'target': target, 'observed': observed,
            'origins': origins, 'mean': np.zeros(3, np.float32), 'std': np.ones(3, np.float32),
            'pca_mean': np.zeros(3, np.float32), 'basis': np.eye(3, dtype=np.float32)[:, :1],
            'bank': {'delta': target['fit'][bank_indices] - history['fit'][bank_indices, -1:],
                'observed': observed['fit'][bank_indices], 'origins': origins['fit'][bank_indices]}}
        rt.CACHE.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='unit_runner_', dir=rt.CACHE) as directory:
            folder = Path(directory)
            self.assertTrue(folder.resolve().is_relative_to(rt.CACHE.resolve()))
            state, pretrain = tr.pretrain_future(pack, 92601, folder / 'AE', rt, Limit(), updates=2, device='cpu')
            model, case = tr.train_case(pack, 92601, 'B_GUMBEL', 0., state, pretrain, Forecaster(), folder / 'B', rt, Limit(), updates=2, device='cpu')
            self.assertEqual(case['training']['updates'], 2)
            self.assertEqual(len(case['training']['curve']), 2)
            self.assertEqual([c['update'] for c in case['training']['checkpoints']], [0, 2])
            with np.load(rt.ROOT / case['predictions']['path'], allow_pickle=False) as z:
                actual = tr.predict(model, history['tune'], origins['tune'], Forecaster())
                self.assertTrue(np.array_equal(actual, z['tune_prediction']))
            restored = tr.load_model(pack, case, rt, device='cpu')
            self.assertEqual(rt.state_hash(restored), case['training']['selected_hash'])


if __name__ == '__main__': unittest.main()
