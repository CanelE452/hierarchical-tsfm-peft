"""Synthetic CPU Torch checks in the active .venv; no campaign assets or models."""
from pathlib import Path
from unittest.mock import patch
import copy
import unittest

import numpy as np
import torch

import run_fixed_recon_d_e_adaptation_v1 as runner
from linear_e_adaptation_v1 import LinearLagAdapter


class ToyBolt:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1, bias=False).eval().requires_grad_(False)
        with torch.no_grad(): self.model.weight.fill_(.7)
    def __call__(self, z):
        return self.model(z[:, -1:]) + .1*z.mean(1, keepdim=True) + torch.linspace(-.2, .2, 48)[None, :, None]


def fixture():
    rng = np.random.default_rng(317)
    history = rng.normal(size=(6, 512, 3)).astype(np.float32); observed = np.ones((6, 48, 3), bool); observed[0, :, 1] = False
    target = (np.broadcast_to(history[:, -1:], (6, 48, 3)) + rng.normal(size=(6, 48, 3))*.1).astype(np.float32); target[~observed] = np.nan
    raw = {"history": history, "target": target, "observed": observed}
    model = LinearLagAdapter(np.zeros(3, np.float32), np.array([[.5], [.2], [-.1]], np.float32), rng.normal(scale=.1, size=(7, 3)).astype(np.float32), runner.LAGS)
    return model, ToyBolt(), raw


class FixedReconDEOnlyTests(unittest.TestCase):
    def test_actual_toy_old_trainer_updates_only_E_and_restores_fixed_flags_and_D_bytes(self):
        model, bolt, raw = fixture(); initial = runner.old.state_arrays(model); events, checkpoints = [], {}
        def checkpoint(update, state, score, outputs):
            checkpoints[update] = {k: v.detach().numpy().copy() for k, v in state.items()}; return {"synthetic": True}
        result = runner.train_E_only(model, bolt, raw, raw, 92601, lambda event: events.append(event), checkpoint, updates=4)
        self.assertEqual(result["training_updates_completed"], 4); self.assertEqual(result["D_frozen_after_optimizer_step_checks"], 4); self.assertEqual(result["D_and_center_checkpoint_byte_checks"], 2)
        for value in checkpoints.values():
            self.assertEqual(value["D"].tobytes(), initial["D"].tobytes()); self.assertEqual(value["center"].tobytes(), initial["center"].tobytes())
        self.assertNotEqual(checkpoints[4]["E"].tobytes(), initial["E"].tobytes()); self.assertFalse(model.D.requires_grad); self.assertIsNone(model.D.grad)
        from native_head_v1 import state_hash
        self.assertEqual(state_hash(model.state_dict()), result["selected_state_sha256"])
        logged = [e for e in events if e and e.get("stage") == "TRAINING_UPDATE"]
        self.assertEqual(len(logged), 4); self.assertTrue(all(e["D_gradient_norm_before_clip"] == 0. and e["D_grad_is_None"] and not e["D_requires_grad"] and e["gradient_clipping_parameter_scope"] == "E_ONLY" for e in logged))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in bolt.model.parameters())); self.assertFalse(result["independent_per_update_D_bytes_replayed"])

    def test_active_Torch_default_load_state_dict_preserves_destination_requires_grad_flags(self):
        model, _, _ = fixture(); runner.freeze_fresh_decoder(model)
        incoming = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}; incoming["D"] = (incoming["D"] + .3).requires_grad_(True)
        model.load_state_dict(incoming)
        self.assertFalse(model.D.requires_grad); self.assertTrue(model.E.requires_grad); self.assertIsNone(model.D.grad)
        self.assertTrue(torch.equal(model.D.detach(), incoming["D"].detach())); self.assertIsInstance(torch.__version__, str)

    def test_stale_gradient_and_accidentally_unfrozen_D_are_rejected(self):
        model, _, _ = fixture(); model.D.grad = torch.ones_like(model.D)
        with self.assertRaisesRegex(ValueError, "STALE"): runner.freeze_fresh_decoder(model)
        model, bolt, raw = fixture()
        def tamper(event):
            if event and event.get("stage") == "OPTIMIZER_STEP_EXECUTED": model.D.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "FIXED_D_GRAD_NONE"): runner.train_E_only(model, bolt, raw, raw, 92601, tamper, lambda *a: None, updates=2)

    def test_checkpoint_bytes_detect_mutation_without_claiming_per_step_byte_replay(self):
        model, bolt, raw = fixture()
        def tamper(event):
            if event and event.get("stage") == "OPTIMIZER_STEP_EXECUTED" and event["update"] == 1:
                with torch.no_grad(): model.D.add_(.1)
        with self.assertRaisesRegex(ValueError, "EVERY_CP_FIXED_INITIAL_D"): runner.train_E_only(model, bolt, raw, raw, 92601, tamper, lambda *a: None, updates=2)

    def test_failed_seed_does_not_skip_other_seed_or_two_resolution_locks(self):
        report = {"datasets": {"robin": {"conditions": {}}}, "failed_paths": [], "fixed_D_E_only_training_fits": 0, "training_updates_completed": 0, "checkpoint_states_saved": 0}; calls = []
        def run(seed):
            calls.append(("fit", seed))
            if seed == 92601: raise ValueError("synthetic failure")
            return {"status": "COMPLETE", "training_updates_completed": 600, "checkpoint_count": 5}
        def seal(seed, row): calls.append(("seal", seed, row["status"])); return {"path": str(seed), "sha256": "a"*64}
        def registry(rows): calls.append(("registry", len(rows))); return {"path": "registry", "sha256": "a"*64}
        result = runner.resolve_two(report, run, seal, registry)
        self.assertEqual(list(result), [92602]); self.assertEqual(calls[-1], ("registry", 2)); self.assertTrue(report["all_2_locks_before_DEV"]); self.assertEqual(report["fixed_D_E_only_training_fits"], 1)

    def test_exact_unchanged_training_prediction_functions_and_closed_acceptance_gate(self):
        from run_linear_e_adaptation_v1 import train_condition, predict_components
        self.assertIs(runner.train_condition, train_condition); self.assertIs(runner.predict_components, predict_components)
        self.assertEqual(runner.SEEDS, (92601, 92602)); self.assertEqual(runner.LAGS, (0, 1, 2, 4, 8, 16, 32))
        with patch.object(runner, "JOINT_AUDIT_SOURCE_SHA", None), patch.object(runner.old, "record", side_effect=AssertionError("closed asset read")), self.assertRaisesRegex(ValueError, "NOT_YET_REGISTERED"): runner.load_gates("a", "b", "c", "d", "e")

    def test_metadata_acceptance_tolerates_optional_bytes_and_keeps_all_same_lineage(self):
        keys = ("joint", "audit", "report", "manifest", "report_audit"); paths = {key: key + ".json" for key in keys}; hashes = {key: str(index + 1)*64 for index, key in enumerate(keys)}
        records = {key: {"path": str(runner.old.resolve(path)), "sha256": hashes[key]} for key, path in paths.items()}; extra = lambda value: {**value, "bytes": 31}
        f = {"signature": "same"}; source = {"path": "source.py", "sha256": runner.JOINT_SOURCE_SHA}; parent_ref = {"path": "parent.json", "sha256": "b"*64}
        joint = {"status": "COMPLETE", "controller_source": source, "full_gradient_parent": parent_ref, "full_gradient_parent_audit": parent_ref, "F_before": f, "F_after": f, "all_2_locks_before_DEV": True, "recon_initialization_training_fits": 2, "failed_paths": [], "selection_registry": parent_ref, "causal_lag_diagnostic": parent_ref, "causal_lag_audit": parent_ref}
        audit = {"status": "PASS", "training_verification_status": "PASS", "new_space_diagnostic_status": "COMPLETE", "partial_paths": [], "new_output_space_diagnostic_failures": [], "diagnostic": extra(records["joint"]), "audit_source": {"path": "auditor.py", "sha256": "c"*64}, "totals": runner.AUDIT_COUNTS}
        manifest = {"status": "COMPLETE", "files": [], "inputs": {"train": extra(records["joint"]), "audit": records["audit"]}}
        report_audit = {"status": "PASS", "report": extra(records["report"]), "manifest": records["manifest"]}; objects = dict(zip(paths.values(), (joint, audit, {}, manifest, report_audit)))
        def rec(path): return records[next(key for key, value in paths.items() if value == path)]
        def bound(actual, expected=None):
            if expected: self.assertEqual((actual["path"], actual["sha256"]), (expected["path"], expected["sha256"]))
            return Path(actual["path"])
        import run_linear_e_cost_v1 as parent
        with patch.object(runner, "FIXED_SHA", hashes), patch.object(runner, "JOINT_AUDIT_SOURCE_SHA", "c"*64), patch.object(runner.old, "record", side_effect=rec), patch.object(runner.old, "read", side_effect=lambda p: objects[p]), patch.object(runner.old, "bound", side_effect=bound), patch.object(parent, "load_gates", return_value=({"F_before": f, "F_after": f}, {}, {}, {}, {}, {}, {})):
            loaded = runner.load_gates(*(paths[key] for key in keys)); self.assertEqual(loaded[7]["joint_recon_diagnostic"], records["joint"])


if __name__ == "__main__": unittest.main()
