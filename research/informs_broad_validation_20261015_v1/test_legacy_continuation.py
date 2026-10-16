"""CPU-only tests. No legacy arrays, foundation model, CUDA, or fitting jobs."""
import copy
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('legacy_continuation', HERE/'legacy_continuation.py')
lc = importlib.util.module_from_spec(spec) if spec and spec.loader and (HERE/'legacy_continuation.py').exists() else None
if lc is not None:
    sys.modules[spec.name] = lc
    spec.loader.exec_module(lc)


class LegacyContinuationTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(lc, 'legacy continuation implementation is required')

    def test_original_prefix_and_nested_endpoints(self):
        for n in (31, 12063):
            rng = np.random.default_rng(92601)
            count = 2400*4
            if count <= n:
                expected = np.floor((np.arange(count)+.5)*n/count).astype(np.int64)
                rng.shuffle(expected)
            else:
                expected = np.concatenate([rng.permutation(n) for _ in range((count+n-1)//n)])[:count]
            small = lc.extended_schedule(n, 4800, 92601)
            large = lc.extended_schedule(n, 9600, 92601)
            np.testing.assert_array_equal(small[:2400].ravel(), expected)
            np.testing.assert_array_equal(small, large[:4800])
            np.testing.assert_array_equal(np.sort(large[2400:].ravel()[:n]), np.arange(n))
            self.assertFalse(np.array_equal(small, lc.extended_schedule(n, 4800, 92602)))

    def test_path_binding_preserves_guard_and_rejects_old_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            asset = Path(directory)/'assets'; campaign = Path(directory)/'worktree/research/new_campaign'
            cache = asset/'.cache/new_campaign'
            rt, dio = types.SimpleNamespace(), types.SimpleNamespace(_GUARD_INSTALLED=False)
            paths = lc.bind_legacy_paths(rt, dio, asset, campaign, cache)
            self.assertEqual(rt.ROOT, asset.resolve())
            self.assertEqual(dio.ROOT, asset.resolve())
            self.assertEqual(dio.HERE, campaign.resolve())
            self.assertEqual(rt.HERE, campaign.resolve()/'legacy_e0')
            self.assertEqual(rt.CACHE, cache.resolve()/'legacy_e0')
            self.assertTrue(rt.MODEL.is_relative_to(asset.resolve()))
            self.assertEqual(paths['data_guard_cache_root'], str(cache.resolve()))
            with self.assertRaises(ValueError):
                lc.bind_legacy_paths(rt, dio, asset, campaign, asset/'.cache/temporal_frontend_campaign_20261015_v1')
            with self.assertRaises(ValueError):
                lc.bind_legacy_paths(rt, dio, asset, asset/'research/temporal_frontend_campaign_20261015_v1', cache)

    def test_spec_keeps_ablation_and_frozen_parent_without_checkpoint_loading(self):
        value = {'id':'g','type':'G','dataset':'jena','seed':92601,'checkpoint':'selected/g.pt',
                 'base':{'id':'b','kind':'B4','r':5,'seed':92601,'dataset':'jena',
                         'front_kwargs':{'trend_mix':False},'checkpoint':'selected/b.pt'}}
        result = lc.model_spec(value)
        self.assertNotIn('checkpoint', result)
        self.assertNotIn('checkpoint', result['base'])
        self.assertEqual(result['base']['front_kwargs'], {'trend_mix':False})
        self.assertEqual(result['base']['type'], 'ADAPTER')
        self.assertIn('checkpoint', value)

    def test_optimizer_and_rng_replay_cpu_toy(self):
        import random
        torch.manual_seed(4)
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0.)
        x = torch.tensor([[1., 2., 3.]])
        optimizer.zero_grad(); model(x).square().sum().backward(); optimizer.step()
        state = copy.deepcopy(model.state_dict()); opt_state = copy.deepcopy(optimizer.state_dict())
        saved_rng = lc.capture_rng(include_cuda=False)
        schedule = lc.extended_schedule(17, 4800, 92601)
        tiny_checkpoint = io.BytesIO()
        torch.save({'state':state, 'optimizer':opt_state, 'rng':saved_rng,
                    'sampler':{'next_update':2401,'prefix_sha256':lc.ahash(schedule[:2401])}}, tiny_checkpoint)
        tiny_checkpoint.seek(0)
        reloaded = torch.load(tiny_checkpoint, map_location='cpu', weights_only=True)
        expected_random = (random.random(), float(np.random.random()), torch.rand(3))
        optimizer.zero_grad(); model(x).square().sum().backward(); optimizer.step()
        expected_state = copy.deepcopy(model.state_dict())
        resumed = torch.nn.Linear(3, 1); resumed.load_state_dict(reloaded['state'])
        resumed_optimizer = lc.restore_optimizer(resumed, reloaded['optimizer'], .001, expected_step=1)
        lc.restore_rng(reloaded['rng'], include_cuda=False)
        next_step = reloaded['sampler']['next_update']
        resumed_schedule = lc.extended_schedule(17, 4800, 92601)
        self.assertEqual(lc.ahash(resumed_schedule[:next_step]), reloaded['sampler']['prefix_sha256'])
        np.testing.assert_array_equal(schedule[next_step], resumed_schedule[next_step])
        actual_random = (random.random(), float(np.random.random()), torch.rand(3))
        self.assertEqual(expected_random[:2], actual_random[:2])
        torch.testing.assert_close(expected_random[2], actual_random[2], rtol=0, atol=0)
        resumed_optimizer.zero_grad(); resumed(x).square().sum().backward(); resumed_optimizer.step()
        for key in expected_state:
            torch.testing.assert_close(resumed.state_dict()[key], expected_state[key], rtol=0, atol=0)
        self.assertEqual(lc.optimizer_steps(resumed_optimizer.state_dict()), {2})
        with self.assertRaises(ValueError):
            lc.restore_optimizer(resumed, opt_state, .001, expected_step=2400)

    def test_checkpoint_validation_and_best_tie(self):
        carry = {'updates':2400, 'curve':[{}]*2400, 'checkpoints':[],
                 'optimizer':{'state':{0:{'step':torch.tensor(2400.)}},'param_groups':[]}}
        self.assertEqual(lc.validate_carry(carry, 2400)['optimizer_step'], 2400)
        self.assertEqual(lc.best_checkpoint([{'update':400,'B':{'mse':.2}},
                                             {'update':200,'B':{'mse':.2}}])['update'], 200)
        bad = copy.deepcopy(carry); bad['curve'].pop()
        with self.assertRaises(ValueError): lc.validate_carry(bad, 2400)


if __name__ == '__main__':
    unittest.main(verbosity=2)
