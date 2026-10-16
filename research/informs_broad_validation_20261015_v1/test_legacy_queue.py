"""E0 lease/budget tests: synthetic CPU tensors and short authenticated children."""
import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


class LegacyResourceTests(unittest.TestCase):
    def setUp(self):
        import legacy_continuation
        self.lc = legacy_continuation
        self.temp = tempfile.TemporaryDirectory(prefix="informs_legacy_queue_")
        self.root = Path(self.temp.name)
        def cleanup():
            assert self.root.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()) and self.root.name.startswith("informs_legacy_queue_")
            self.temp.cleanup()
        self.addCleanup(cleanup)

    def test_guard_rejects_budget_before_work_without_cuda(self):
        self.assertTrue(callable(getattr(self.lc, "check_work_boundary", None)), "E0 lease boundary guard missing")
        import resources
        calls = []
        lease = types.SimpleNamespace(check=lambda: calls.append("checked"), remaining=20)
        self.lc.check_work_boundary(lease)
        self.assertEqual(calls, ["checked"])
        lease.remaining = 4
        with self.assertRaises(resources.BudgetExceeded):
            self.lc.check_work_boundary(lease)
        with self.assertRaises(resources.ResourceBusy):
            self.lc.check_work_boundary(None)

    def test_guarded_F_checks_each_validation_call_and_pending_endpoint_is_detected(self):
        self.assertTrue(callable(getattr(self.lc, "LeaseBoundFoundation", None)), "Assessment F calls lack lease checks")
        import resources
        calls = []
        lease = types.SimpleNamespace(check=lambda: None, remaining=20)
        wrapped = self.lc.LeaseBoundFoundation(lambda x: calls.append(x) or x + 1, lease)
        self.assertEqual(wrapped(2), 3)
        lease.remaining = 4
        with self.assertRaises(resources.BudgetExceeded):
            wrapped(5)
        self.assertEqual(calls, [2])
        self.assertEqual(self.lc.pending_checkpoint_update(4800, []), 4800)
        self.assertIsNone(self.lc.pending_checkpoint_update(4800, [dict(update=4800)]))
        self.assertIsNone(self.lc.pending_checkpoint_update(4801, []))

    def test_budget_checkpoint_replays_next_step_rng_and_sampler_and_rejects_tamper(self):
        self.assertTrue(callable(getattr(self.lc, "save_budget_checkpoint", None)), "Full E0 budget carry missing")
        torch.manual_seed(71)
        model = torch.nn.Linear(3, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0)
        x = torch.ones(2, 3)
        optimizer.zero_grad(); model(x).square().sum().backward(); optimizer.step()
        curve = [dict(update=1, loss=1., clipped=False)]
        sampler = dict(next_update=1, schedule_sha256="tiny-schedule", next_batch=[2, 4, 3, 1])
        contract = dict(legacy_id="toy", endpoint=4800, runner_source="toy-source")
        path = self.root / "budget.pt"
        self.lc.save_budget_checkpoint(path, model, optimizer, curve, [], sampler, contract, include_cuda=False)
        expected_rng = torch.rand(4)
        optimizer.zero_grad(); model(x).square().sum().backward(); optimizer.step()
        expected = copy.deepcopy(model.state_dict())
        saved = self.lc.load_budget_checkpoint(path, contract)
        resumed = torch.nn.Linear(3, 1); resumed.load_state_dict(saved["state"])
        opt = self.lc.restore_optimizer(resumed, saved["carry"]["optimizer"], .001, 1)
        self.lc.restore_rng(saved["carry"]["rng"], include_cuda=False)
        torch.testing.assert_close(torch.rand(4), expected_rng, atol=0, rtol=0)
        self.assertEqual(saved["carry"]["sampler"], sampler)
        opt.zero_grad(); resumed(x).square().sum().backward(); opt.step()
        for name in expected:
            torch.testing.assert_close(resumed.state_dict()[name], expected[name], atol=0, rtol=0)
        with self.assertRaises(ValueError):
            self.lc.load_budget_checkpoint(path, dict(contract, endpoint=9600))
        with path.open("ab") as handle:
            handle.write(b"tampered")
        with self.assertRaises(ValueError):
            self.lc.load_budget_checkpoint(path, contract)

    def test_parent_holds_lease_over_real_cpu_child_and_charges_full_wall(self):
        path = HERE / "legacy_queue.py"
        self.assertTrue(path.is_file(), "Parent E0 resource queue missing")
        spec = importlib.util.spec_from_file_location("legacy_queue", path)
        queue = importlib.util.module_from_spec(spec); spec.loader.exec_module(queue)
        plan = dict(routes=[dict(id="toy", dataset="toy", role="UNRESOLVED")], route_count=1)
        def command(**kwargs):
            code = ("import sys,time,json;sys.path.insert(0," + repr(str(HERE)) + ");"
                    "from resources import validate_gpu_child;"
                    "lease=validate_gpu_child(" + repr(str(self.root)) + ");lease.check();time.sleep(.05);lease.check();"
                    "open(" + repr(str(kwargs["result_path"])) + ",'x').write(json.dumps({'id':'toy','endpoint':4800,'status':'CONTINUED_ENDPOINT_COMPLETE'}))")
            return [sys.executable, "-c", code]
        with patch.object(queue.lc, "queue_plan", return_value=plan), patch.object(queue, "worker_command", side_effect=command):
            result = queue.run_queue(self.root / "assets", self.root, ids=["toy"], include_controls=False,
                                     endpoints=[4800], total_gpu_seconds=30, reserve_seconds=1,
                                     wall_deadline=time.time() + 30, execute=True)
        self.assertEqual(result["status"], "QUEUE_FINISHED")
        import json
        rows = [json.loads(line) for line in (self.root / "GPU_USAGE.jsonl").read_text().splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0]["charged_process_seconds"], .05)
        self.assertEqual(rows[0]["accounting_scope"], "exclusive_reserved_process_wall_including_children")
        self.assertFalse((self.root / "ACTIVE_GPU_LEASE.json").exists())

    def test_run_one_CPU_mock_stops_after_update_preserves_and_explicitly_resumes(self):
        import json
        import resources
        lc = self.lc
        asset = self.root / "assets"
        campaign = self.root / "new_campaign"
        cache = asset / ".cache" / "new_campaign"
        rt_cache = cache / "legacy_e0"
        legacy_id = "toy_legacy"
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.weight = torch.nn.Parameter(torch.tensor(.8))
            def cuda(self):
                return self
            def forward(self, x, foundation):
                return foundation(x) * self.weight
        class ToyBolt:
            def __init__(self):
                self.model = torch.nn.Linear(1, 1).requires_grad_(False)
                self.model.load_state_dict(foundation_state)
                self.initial_hash = lc.tensor_state_hash(self.model.state_dict())
            def __call__(self, x):
                return x[:, -48:]
        class Lease:
            remaining = 100
            def __init__(self):
                self.calls = 0
            def check(self):
                self.calls += 1
                if self.calls >= 7:
                    raise resources.BudgetExceeded("Toy next-step budget")
        source = ToyModel()
        optimizer = torch.optim.AdamW(source.parameters(), lr=.001, weight_decay=0)
        optimizer.zero_grad(); source.weight.square().backward(); optimizer.step()
        for state in optimizer.state.values():
            state["step"] = torch.tensor(2400.)
        state = copy.deepcopy(source.state_dict())
        curve = [dict(update=i+1, clipped=False) for i in range(2400)]
        checkpoints = [dict(update=2400, B=dict(mse=.1), state_sha256=lc.tensor_state_hash(state))]
        old_last = asset / "last2400.pt"
        old_last.parent.mkdir(parents=True)
        torch.save(dict(state=state, carry=dict(updates=2400, optimizer=optimizer.state_dict(), curve=curve, checkpoints=checkpoints)), old_last)
        rec_path = asset / "receipt.json"
        rec_path.write_text("{}", encoding="utf-8")
        cp_path = asset / ".cache" / lc.OLD_NAME / "checkpoints" / legacy_id / "cp2400.pt"
        cp_path.parent.mkdir(parents=True)
        torch.save(state, cp_path)
        stats_path = asset / "research" / lc.OLD_NAME / "FOLD_STATISTICS.json"
        stats_path.parent.mkdir(parents=True)
        stats_path.write_text(json.dumps(dict(toy=dict(primary=dict(A_mean=[0.], A_std=[1.])))), encoding="utf-8")
        rec = dict(id=legacy_id, dataset="toy", kind="B0", r=1, seed=92601, updates=2400,
                   fold="primary", learning_rate=.001, legal_A_origins=1, learning_curve=curve, checkpoints=checkpoints,
                   origin_schedule_sha256=lc.ahash(lc.extended_schedule(1, 2400, 92601)))
        foundation_state = copy.deepcopy(torch.nn.Linear(1, 1).state_dict())
        frozen_hash = lc.tensor_state_hash(foundation_state)
        win = types.SimpleNamespace(origins=dict(A=np.array([512]), B=np.array([550])), mean=np.array([0.], np.float32),
                                   std=np.array([1.], np.float32), counts=torch.tensor([192]), eval_A=np.array([512]),
                                   data=dict(observed=np.ones((600, 1), bool)))
        win.get = lambda origins: (torch.ones(4, 512, 1), torch.zeros(4, 48, 1), torch.ones(4, 48, 1, dtype=torch.bool))
        def save(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                torch.save(value, handle)
        rt = types.SimpleNamespace(HERE=campaign / "legacy_e0", CACHE=rt_cache, Bolt=ToyBolt,
                state_hash=lambda model: lc.tensor_state_hash(model.state_dict()),
                clone=lambda model:{key:value.detach().cpu().clone() for key,value in model.state_dict().items()},
                save=save, write=lc.write_json, loss=lambda p,y,m,c,n:(p-y).square().mean(), event=lambda **kwargs:None)
        cp = types.SimpleNamespace(dataset_windows=lambda *args:win,
                                   assess=lambda *args: (dict(mse=.2, mae=.3), None), still_decreasing=lambda rows:False)
        old_lock = dict(F_model_hash=frozen_hash, statistics_hash="toy-statistics")
        sources = dict(runtime="toy-source")
        def build(spec, model_state, modules):
            model = ToyModel(); model.load_state_dict(model_state); return model
        with patch.object(lc, "HERE", campaign), patch.object(lc, "receipt_for", return_value=(rec, rec_path)), \
             patch.object(lc, "checkpoint_path", return_value=old_last), \
             patch.object(lc, "load_legacy", return_value=(dict(runtime=rt, campaign=cp), old_lock, sources, {})), \
             patch.object(lc, "build_legacy_model", side_effect=build), patch.object(lc, "update_ledger"), \
             patch.object(torch.cuda, "synchronize"), patch.object(torch.cuda, "manual_seed_all"), \
             patch.object(torch.cuda, "get_rng_state_all", return_value=[]), patch.object(torch.cuda, "set_rng_state_all"):
            first = lc.run_one(asset, cache, legacy_id, 4800, {}, child_lease=Lease())
            self.assertEqual(first["status"], "BUDGET_LIMIT")
            self.assertEqual(first["current_update"], 2401)
            self.assertFalse(first["endpoint_complete"])
            self.assertFalse((rt_cache / "training" / legacy_id / "last_4800.pt").exists())
            second = lc.run_one(asset, cache, legacy_id, 4800, {}, child_lease=Lease(),
                                resume_budget_checkpoint=first["budget_checkpoint_cache_relative_path"])
            self.assertEqual(second["current_update"], 2402)
            self.assertEqual(second["budget_resume_fidelity"], "EXACT_WITHIN_NEW_RUN")
            self.assertEqual(second["continuation_fidelity"], "RESTARTED_NOT_EXACT_RESUME")
            # Endpoint weights can be complete while final A/B assessment is
            # pending. Resume must finish that assessment without a new step.
            first_path = cache / first["budget_checkpoint_cache_relative_path"]
            contract = lc.read_json(first_path.with_suffix(".pt.json"))["budget_contract"]
            endpoint_model = ToyModel(); endpoint_model.load_state_dict(state)
            endpoint_optimizer = lc.restore_optimizer(endpoint_model, optimizer.state_dict(), .001, 2400)
            for opt_state in endpoint_optimizer.state.values():
                opt_state["step"] = torch.tensor(4800.)
            endpoint_curve = [dict(update=i+1, clipped=False) for i in range(4800)]
            order = lc.extended_schedule(1, 4800, 92601)
            sampler = dict(next_update=4800, pending_checkpoint_update=4800, schedule_sha256=lc.ahash(order))
            endpoint_path = rt_cache / "training" / legacy_id / "budget_final_assessment.pt"
            lc.save_budget_checkpoint(endpoint_path, endpoint_model, endpoint_optimizer, endpoint_curve, checkpoints,
                                      sampler, contract, include_cuda=True)
            complete = lc.run_one(asset, cache, legacy_id, 4800, {},
                         child_lease=types.SimpleNamespace(check=lambda:None, remaining=100),
                         resume_budget_checkpoint=endpoint_path.relative_to(cache).as_posix())
            self.assertEqual(complete["status"], "CONTINUED_ENDPOINT_COMPLETE")
            final = lc.read_json(rt.HERE / "fits" / f"{legacy_id}_4800.json")
            self.assertEqual(final["incremental_updates"], 0)
            self.assertIn(4800, [row["update"] for row in final["checkpoints"]])


if __name__ == "__main__":
    unittest.main()
