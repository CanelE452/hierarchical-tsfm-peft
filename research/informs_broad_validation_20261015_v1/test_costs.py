"""CPU-only deployment/cost tests with actual toy forward calls."""
import tempfile
import json
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import deployment as d
import measure_costs as m
from data_contract import array_sha256, fit_statistics, split_ranges
import fit_base
from networks import module_state_hash
from training import fingerprint
from networks import ConditionalHead
from combination import scaler_fingerprint


class ToyFoundation(torch.nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon
        self.row_microbatch = 8
        self.streams = self.row_calls = self.physical_streams = self.physical_calls = 0
        self.q, self.v = torch.nn.Linear(1, 1), torch.nn.Linear(1, 1)
        with torch.no_grad():
            self.q.weight.fill_(1); self.q.bias.zero_(); self.v.weight.fill_(1); self.v.bias.zero_()
        object.__setattr__(self, "model", self)
        self.initial_hash = module_state_hash(self)

    def forward(self, x):
        self.streams += x.shape[0] * x.shape[2]
        self.physical_streams += x.shape[0] * x.shape[2]
        self.row_calls += 1
        self.physical_calls += 1
        return self.v(self.q(x[:, -1:, :, None])).squeeze(-1).repeat(1, self.horizon, 1)

    def assert_frozen_unchanged(self):
        if module_state_hash(self) != self.initial_hash:
            raise AssertionError("Toy base changed")


class CostTests(unittest.TestCase):
    def task(self):
        values = np.arange(800 * 3, dtype=np.float32).reshape(800, 3)
        return {"name": "toy", "physical": values, "observed": np.ones_like(values, bool), "L": 40, "H": 4,
                "period": 12, "columns": ["a", "b", "c"], "data_sha256": "toy", "split_ranges": split_ranges(800)}

    def fixture(self, root, task, family, foundation=None):
        spec = {"run_id": family.lower(), "family": family, "seed": 9, "device": "cpu", "latent_k": 2}
        model, norm = fit_base.create_model(spec, task, task["split_ranges"]["A_stop"][1], foundation)
        if family == "LORA":
            with torch.no_grad():
                for adapter in model.adapters.values(): adapter.B.fill_(.02)
        state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        bundle = {"spec": spec, "normalization": norm, "model": state,
                  "contract": {"data_sha": task["data_sha256"], "model_sha": fit_base._model_sha(foundation), "scaler_sha": fit_base.scaler_fingerprint(norm["mean"], norm["std"])},
                  "model_state_sha": fingerprint({key: array_sha256(value.numpy()) for key, value in state.items()})}
        if callable(getattr(model, "close", None)): model.close()
        path = root / f"{family}.pt"
        torch.save(bundle, path)
        return {"run_id": family, "family": family, "artifact_cache_relative_path": path.name}

    def test_current_has_no_foundation_and_physical_fill_reset(self):
        task = self.task()
        with tempfile.TemporaryDirectory() as temp:
            route = d.Route({"run_id": "current", "family": "CURRENT"}, task, temp, device="cpu")
            x = task["physical"][500:540][None].copy()
            mask = np.ones(x.shape, bool)
            mask[:, -2:, 0] = False
            x[:, -2:, 0] = np.nan
            out = route.predict_physical(x, mask, measure_stages=True)
            self.assertEqual(out.shape, (1, 4, 3))
            np.testing.assert_allclose(out[:, :, 0], np.repeat(x[:, -3:-2, 0], 4, 1), rtol=1e-6)
            self.assertEqual(route.pool.instances_created, 0)
            self.assertEqual(route.last_receipt["F_logical_streams"], 0)
            self.assertIn("CPU_preprocess", route.last_receipt["stage_seconds"])
            route.close()

    def test_actual_raw_forward_streams_and_row_variants_parity(self):
        task = self.task()
        calls = []
        def factory(path, horizon, row_microbatch, device):
            calls.append(horizon)
            return ToyFoundation(horizon)
        pool = d.FoundationPool(None, "cpu", factory=factory)
        with tempfile.TemporaryDirectory() as temp:
            route = d.Route({"run_id": "raw", "family": "RAW"}, task, temp, device="cpu", foundation_pool=pool)
            x = np.stack([task["physical"][500:540]] * 2)
            result = route.predict_physical(x, np.ones_like(x, bool))
            self.assertEqual(route.last_receipt["F_logical_streams"], 6)
            parity = m.raw_row_parity(route, x, np.ones_like(x, bool))
            self.assertTrue(parity["all_passed"])
            self.assertEqual(calls, [4])
            self.assertEqual(result.shape, (2, 4, 3))
            route.close(); pool.close()

    def test_raw_rejects_wrong_foundation_pin_before_forward(self):
        task=self.task()
        pool=d.FoundationPool(None,'cpu',factory=lambda p,h,r,dev:ToyFoundation(h))
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                d.Route(dict(run_id='raw',family='RAW',foundation_sha='different F'),task,temp,device='cpu',foundation_pool=pool)
        pool.close()

    def test_raw_official_asset_files_are_verified_even_with_shared_pool(self):
        task=self.task()
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'model';path.mkdir();(path/'config.json').write_bytes(b'original')
            asset=dict(files={'config.json':dict(sha256=d.sha256_file(path/'config.json'))})
            pool=d.FoundationPool(path,'cpu',factory=lambda p,h,r,dev:ToyFoundation(h))
            (path/'config.json').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                d.Route(dict(run_id='raw',family='RAW',foundation_asset=asset),task,temp,device='cpu',foundation_pool=pool)
            pool.close()

    def test_unpinned_real_raw_is_rejected_before_model_loading(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(d.FoundationPool,'get',side_effect=AssertionError('unbound F loaded')):
            with self.assertRaises(ValueError):
                d.Route(dict(run_id='raw',family='RAW'),self.task(),temp,device='cpu')

    def test_balanced_order_and_real_trials(self):
        task = self.task()
        with tempfile.TemporaryDirectory() as temp:
            routes = [d.Route({"run_id": label, "family": "CURRENT"}, task, temp, device="cpu") for label in ("a", "b", "c")]
            x = task["physical"][500:540][None]
            rows = m.measure_balanced(routes, x, np.ones_like(x, bool), warmups=1, timed_trials=3)
            timed = [r for r in rows if r["phase"] == "timed"]
            self.assertEqual(len(timed), 9)
            for label in ("a", "b", "c"):
                self.assertEqual(sorted(r["order_position"] for r in timed if r["run_id"] == label), [0, 1, 2])
            self.assertTrue(all(r["whole_seconds"] > 0 for r in rows))
            self.assertTrue(all(r["F_logical_streams"] == 0 for r in rows))
            for route in routes: route.close()

    def test_worker_gpu_requires_valid_child_lease(self):
        with patch.object(m, "validate_gpu_child", side_effect=RuntimeError("missing lease")):
            with self.assertRaises(RuntimeError):
                m.worker_lease_check(Path("unused"), "cuda")

    def test_trained_G_does_not_instantiate_F(self):
        task = self.task()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = self.fixture(root, task, "DIRECT_G")
            route = d.Route(spec, task, root, device="cpu")
            x = task["physical"][500:540][None]
            result = route.predict_physical(x, np.ones_like(x, bool))
            self.assertEqual(result.shape, (1, 4, 3))
            self.assertEqual(route.pool.instances_created, 0)
            route.close()

    def test_lora_actual_merge_parity_and_exact_restore_shared_F(self):
        task = self.task()
        pool = d.FoundationPool(None, "cpu", factory=lambda p, h, r, dev: ToyFoundation(h))
        foundation = pool.get(task["H"])
        original = module_state_hash(foundation)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = self.fixture(root, task, "LORA", foundation)
            route = d.Route(spec, task, root, device="cpu", foundation_pool=pool)
            x = task["physical"][500:540][None]
            parity = route.check_lora_parity(x, np.ones_like(x, bool))
            self.assertEqual(parity["status"], "PASS")
            changed = route.predict_physical(x, np.ones_like(x, bool))
            self.assertNotEqual(module_state_hash(foundation), original)
            route.deactivate()
            self.assertEqual(module_state_hash(foundation), original)
            route.activate()
            np.testing.assert_array_equal(changed, route.predict_physical(x, np.ones_like(x, bool)))
            route.close()
            self.assertEqual(module_state_hash(foundation), original)
            pool.close()

    def test_requested_latent_prefix_is_actual_F_stream_count(self):
        task = self.task()
        pool = d.FoundationPool(None, "cpu", factory=lambda p, h, r, dev: ToyFoundation(h))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec = dict(self.fixture(root, task, "B0", pool.get(task["H"])), latent_k=1)
            route = d.Route(spec, task, root, device="cpu", foundation_pool=pool)
            x = np.stack([task["physical"][500:540]] * 2)
            route.predict_physical(x, np.ones_like(x, bool))
            self.assertEqual(route.last_receipt["F_logical_streams"], 2)
            route.close(); pool.close()

    def test_past_control_never_loads_unneeded_Raw_parent(self):
        task = self.task()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            g = self.fixture(root, task, "DIRECT_G")
            mean, std, _ = fit_statistics(task, task["split_ranges"]["A_stop"][1])
            head = ConditionalHead(task["L"], task["H"])
            bundle = {"model": head.state_dict(), "head_config": {"length": task["L"], "horizon": task["H"], "hidden": 32, "kind": "past_control"},
                      "contract": {"scaler_sha": scaler_fingerprint(mean, std), "data_sha": task["data_sha256"]}, "control": True}
            torch.save(bundle, root / "head.pt")
            spec = {"run_id": "past", "family": "COMBINATION", "variant": "past_control", "artifact_cache_relative_path": "head.pt",
                    "g_route": g, "b_route": {"run_id": "unused_raw", "family": "RAW"}}
            route = d.Route(spec, task, root, device="cpu")
            x = task["physical"][500:540][None]
            route.predict_physical(x, np.ones_like(x, bool))
            self.assertIsNone(route.b_route)
            self.assertEqual(route.pool.instances_created, 0)
            route.close()

    def test_native_ridge_uses_actual_CPU_coefficients_and_zero_F(self):
        task = self.task()
        mean, std, _ = fit_statistics(task, task["split_ranges"]["A_stop"][1])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            coefficients = np.zeros((3, 33, 4), np.float64)
            coefficients[:, -1] = .5
            np.savez(root / "ridge.npz", coefficients=coefficients, mean=mean, std=std, data_sha=np.asarray(task["data_sha256"]))
            spec = {"run_id": "ridge", "family": "RAW_RIDGE", "artifact_cache_relative_path": "ridge.npz"}
            route = d.Route(spec, task, root, device="cpu")
            x = task["physical"][500:540][None]
            result = route.predict_physical(x, np.ones_like(x, bool))
            expected = x[:, -1:] + .5 * std
            np.testing.assert_allclose(result, np.repeat(expected, 4, axis=1), rtol=1e-6)
            self.assertEqual(route.pool.instances_created, 0)
            self.assertEqual(result.dtype, np.float64)
            route.close()

    def test_worker_and_balanced_driver_actual_toy_outputs(self):
        task = self.task()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            single = root / "single.json"
            single.write_text(json.dumps({"run_id": "current", "family": "CURRENT", "dataset": "toy"}))
            args = SimpleNamespace(cache_root=str(root), device="cpu", route_spec=str(single), batch=1, model=None,
                                   warmups=1, timed_trials=2, stage_trials=1, output=str(root / "worker.json"))
            with patch.object(m, "load_task", return_value=task):
                receipt = m.worker(args)
            self.assertEqual(receipt["loading"]["foundation_instances_created"], 0)
            self.assertEqual(receipt["backbone_close_status"], "NO_FOUNDATION_INSTANTIATED")
            self.assertTrue((root / receipt["trials_file"]).is_file())
            multiple = root / "multiple.json"
            multiple.write_text(json.dumps([{"run_id": label, "family": "CURRENT", "dataset": "toy"} for label in ("one", "two")]))
            args.route_spec, args.output = str(multiple), str(root / "balanced.json")
            with patch.object(m, "load_task", return_value=task):
                receipt = m.balanced_driver(args)
            self.assertEqual(receipt["scope"], "general_shared_F_balanced")
            self.assertEqual(receipt["run_ids"], ["one", "two"])
            self.assertTrue((root / receipt["trials_file"]).is_file())


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
