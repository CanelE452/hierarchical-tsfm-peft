"""Tiny CPU synthetic Q learning and metadata gates; no real model, data, F or GPU."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy
import inspect
import json
import subprocess
import sys
import unittest

import numpy as np
import torch

import run_orthogonal_q_learning_v1 as run
from orthogonal_latent_basis_v1 import OrthogonalLatentBasisAdapter
from linear_e_adaptation_v1 import global_fit_macro_mse


torch.set_num_threads(2)


class ToyBolt:
    def __init__(self, horizon=5, linear=False):
        self.model = torch.nn.Linear(1, 1).eval().requires_grad_(False); self.horizon, self.linear = horizon, linear; self.calls = 0
    def __call__(self, z):
        self.calls += 1
        value = 1.2*z[:, -1:]+.1*z.mean(1, keepdim=True) if self.linear else z[:, -1:].square()+.3*z[:, -2:-1].sin()
        return value.expand(-1, self.horizon, -1)


def fixture(*, zero_decoder=False, linear=False):
    rng = np.random.default_rng(773)
    frozen = {"center": rng.normal(scale=.1, size=6).astype(np.float32), "E": rng.normal(scale=.2, size=(6, 4)).astype(np.float32), "D": rng.normal(scale=.1, size=(28, 6)).astype(np.float32)}
    if zero_decoder: frozen["D"].fill(0)
    model = OrthogonalLatentBasisAdapter(frozen["center"], frozen["E"], frozen["D"], run.LAGS, learn_Q=True).eval(); bolt = ToyBolt(linear=linear)
    history = rng.normal(size=(8, 35, 6)).astype(np.float32)
    with torch.no_grad(): prediction, past, future = model(torch.from_numpy(history), bolt, return_components=True)
    target = prediction.numpy().copy()+rng.normal(scale=.15, size=prediction.shape).astype(np.float32)
    observed = np.ones_like(target, dtype=bool); observed[::2, :, 0] = False; observed[:, :, 5] = False; target[~observed] = np.nan
    raw = {"history": history, "target": target, "observed": observed, "raw_last": history[:, -1].copy(), "mean": np.arange(6, dtype=np.float32), "std": np.arange(1, 7, dtype=np.float32), "origins": np.arange(8, dtype=np.int64)+512, "raw_forecast": prediction.numpy().copy()+np.float32(.1)}
    cached = {"prediction": prediction.numpy().copy(), "latent_history": past.numpy().copy(), "latent_future": future.numpy().copy()}
    return model, bolt, raw, cached, frozen


def graph(model, bolt, raw):
    x, target, observed = (torch.from_numpy(raw[key]) for key in ("history", "target", "observed")); capture = {}
    def F(rotated): capture["rotated"] = rotated; capture["F_future"] = bolt(rotated); return capture["F_future"]
    prediction, _, future = model(x, F, return_components=True)
    loss = global_fit_macro_mse(prediction, target, observed, observed.sum((0, 1)), len(x))
    return loss, capture, future


def result_fixture():
    return {"states": {s: {"learning": {}} for s in run.STATES}, "failed_paths": [], "Q_training_fits": 0, "training_updates_completed": 0, "checkpoint_states_saved": 0, "DEV_evaluations_completed": 0, "resolved_selection_locks": 0, "all_6_locks_before_DEV": False}


class QLearningRunnerTests(unittest.TestCase):
    def test_score_reduction_link_tolerates_roundoff_without_changing_selection_score(self):
        from native_head_v1 import score_predictions
        from decoder_metric_probe_v1 import masked_scores
        rng = np.random.default_rng(1927); prediction = rng.normal(size=(7, 11, 6)).astype(np.float32); target = rng.normal(size=prediction.shape).astype(np.float32)
        for partial in (False, True):
            observed = rng.uniform(size=target.shape) > .25 if partial else np.ones_like(target, dtype=bool)
            source = score_predictions(prediction, target, observed); entry = {"scores_all_observed": masked_scores(prediction, target, observed, np.arange(1, 7, dtype=np.float32))}
            original = copy.deepcopy(source); run.check_score_link(source, entry); self.assertEqual(source, original)
            close = copy.deepcopy(entry); close["scores_all_observed"]["normalized"]["mse"] += 1e-13; run.check_score_link(source, close)
            wrong = copy.deepcopy(entry); wrong["scores_all_observed"]["normalized"]["mae"] += 1e-5
            with self.subTest(partial=partial), self.assertRaisesRegex(ValueError, "OWN_TUNE_SCORE"): run.check_score_link(source, wrong)

    def test_full_gradient_equals_input_chain_plus_independent_inverse_and_finite_difference(self):
        model, bolt, raw, _, _ = fixture()
        with torch.no_grad(): model.theta.copy_(torch.tensor([.04, -.02, .01, .03, -.05, .06]))
        loss, captured, future = graph(model, bolt, raw)
        proof, arrays = run.first_batch_gradients(model, loss, captured["rotated"], captured["F_future"], future)
        self.assertEqual(proof["status"], "PASS"); self.assertGreater(proof["theta_F_input_chain_norm"], 1e-5); self.assertGreater(proof["theta_inverse_Q_contribution_norm"], 1e-5)
        theta = model.theta.detach().clone(); numerical = []
        for i in range(6):
            values = []
            for sign in (1., -1.):
                with torch.no_grad(): model.theta.copy_(theta); model.theta[i] += sign*.001
                value, _, _ = graph(model, bolt, raw); values.append(float(value.detach()))
            numerical.append((values[0]-values[1])/.002)
        np.testing.assert_allclose(arrays["theta_gradient_full"], numerical, atol=2e-5, rtol=.01)
        self.assertTrue(all(p.grad is None for p in bolt.model.parameters())); json.dumps(proof, allow_nan=False)

    def test_linear_shared_F_has_nonzero_paths_that_cancel_and_zero_total_is_valid(self):
        model, bolt, raw, _, _ = fixture(linear=True); loss, capture, future = graph(model, bolt, raw)
        proof, arrays = run.first_batch_gradients(model, loss, capture["rotated"], capture["F_future"], future)
        self.assertEqual(proof["status"], "PASS"); self.assertGreater(proof["theta_F_input_chain_norm"], 1e-5); self.assertLess(proof["full_theta_gradient_norm"], 1e-6)
        np.testing.assert_allclose(arrays["theta_gradient_via_F_input"], -arrays["theta_gradient_via_inverse_Q"], atol=1e-6)

    def test_tiny_training_only_theta_changes_fixed_global_counts_logs_and_immutable_CP(self):
        model, bolt, raw, cached, frozen = fixture(); saved, events, probes = [], [], []
        def checkpoint(update, state, score, outputs): saved.append((update, copy.deepcopy(state), score, copy.deepcopy(outputs))); return {"status": "COMPLETE"}
        def gradient_callback(proof, arrays): probes.append((copy.deepcopy(proof), copy.deepcopy(arrays))); return {"synthetic": True}
        result = run.train_Q(model, bolt, raw, raw, cached, 92601, events.append, checkpoint, gradient_callback, updates=6)
        self.assertEqual([c[0] for c in saved], [0, 6]); self.assertEqual(len(probes), 1); self.assertEqual(len(result["learning_curve"]), 6)
        self.assertEqual(result["FIT_channel_observed_counts"], raw["observed"].sum((0, 1)).tolist()); self.assertEqual(result["training_updates_completed"], 6)
        self.assertTrue(torch.equal(saved[0][1]["theta"], torch.zeros(6))); self.assertGreater(float(saved[1][1]["theta"].norm()), 0.)
        for key in frozen: self.assertTrue(run.old.byte_equal(run.old.state_arrays(model)[key], frozen[key]))
        expected_order = np.random.default_rng(92601).integers(0, 8, size=(6, 4))
        import hashlib
        self.assertEqual(result["order_sha256"], hashlib.sha256(expected_order.tobytes()).hexdigest())
        for event in result["learning_curve"]: self.assertEqual(event["trainable_parameter_names"], ["theta"]); self.assertTrue(event["F_grad_is_None"]); self.assertEqual(len(event["theta_gradient_before_clip"]), 6)
        self.assertEqual(sum(e.get("stage") == "OPTIMIZER_STEP_EXECUTED" for e in events), 6); self.assertFalse(model.identity)

    def test_unobserved_labels_do_not_change_training_and_loss_uses_entire_FIT_counts(self):
        model, bolt, raw, cached, _ = fixture(); changed = copy.deepcopy(raw); changed["target"][~changed["observed"]] = np.inf
        other, other_bolt, _, _, _ = fixture(); callback = lambda *a: {"status": "COMPLETE"}; gradient = lambda *a: None
        a = run.train_Q(model, bolt, raw, raw, cached, 92602, lambda *a: None, callback, gradient, updates=2)
        b = run.train_Q(other, other_bolt, changed, changed, cached, 92602, lambda *a: None, callback, gradient, updates=2)
        self.assertEqual(a["learning_curve"], b["learning_curve"])
        prediction = torch.ones((4, 5, 6)); ids = [0, 1, 1, 4]; observed = torch.from_numpy(raw["observed"]); target = torch.from_numpy(raw["target"])
        counts = observed.sum((0, 1)); actual = global_fit_macro_mse(prediction, target[ids], observed[ids], counts, 8)
        error = np.where(raw["observed"][ids], prediction.numpy()-raw["target"][ids], 0.); active = counts.numpy() > 0
        expected = (8/4)*(np.square(error).sum((0, 1))[active]/counts.numpy()[active]).mean()
        self.assertAlmostEqual(float(actual), float(expected), places=6)

    def test_zero_decoder_exact_TUNE_tie_keeps_CP0_without_nonzero_gradient_requirement(self):
        model, bolt, raw, cached, _ = fixture(zero_decoder=True)
        trained = run.train_Q(model, bolt, raw, raw, cached, 92601, lambda *a: None, lambda *a: {"status": "COMPLETE"}, lambda *a: None, updates=2)
        self.assertEqual(trained["selected_update"], 0); self.assertFalse(trained["first_batch_gradient_probe"]["F_input_path_nonzero_observed"])
        self.assertEqual(trained["first_batch_gradient_probe"]["status"], "PASS")

    def test_CP0_and_nonzero_Q_canonical_arrays_independent_NumPy_inverse_and_old_D(self):
        model, bolt, raw, cached, frozen = fixture()
        for cp0 in (True, False):
            if not cp0:
                with torch.no_grad(): model.theta.copy_(torch.tensor([.04, -.02, .01, .03, -.05, .06]))
            outputs, calls = run.fixed.infer_control(model, bolt, raw["history"], cached, lambda *a: None, batch_size=3)
            self.assertEqual(calls, 3); Q = model.basis().detach().numpy()
            row, independent = run.evaluate_arrays(outputs, raw, cached, frozen, Q, cp0=cp0)
            self.assertEqual(row["status"], "COMPLETE"); self.assertEqual(row["CP0_parity_required"], cp0); self.assertEqual(independent["independent_NumPy_prediction"].shape, raw["target"].shape)
        outputs["unrotated_future"][0, 0, 0] += .1
        row, _ = run.evaluate_arrays(outputs, raw, cached, frozen, Q)
        self.assertEqual(row["status"], "FAILED_PARITY_PRESERVED"); self.assertFalse(row["checks"]["NumPy_inverse_Q_future"]["pass"])
        self.assertEqual(tuple(inspect.signature(run.fixed.infer_control).parameters), ("model", "bolt", "history", "cached", "check", "batch_size"))

    def test_six_locks_include_failure_and_precede_any_DEV(self):
        result = result_fixture(); events = []
        def run_one(state, seed):
            events.append(("run", state, seed))
            if (state, seed) == run.CONDITIONS[2]: raise ValueError("synthetic fit failed")
            return {"status": "COMPLETE", "training_updates_completed": 600, "checkpoint_count": 5}
        def seal(state, seed, row): events.append(("seal", state, seed)); return {"path": "synthetic"}
        def registry(rows): self.assertEqual(len(rows), 6); events.append(("registry",)); return {"path": "synthetic registry"}
        with self.assertRaisesRegex(ValueError, "SIX_LOCKS"): run.infer_development(result, {}, lambda *a: None)
        resolved = run.resolve_all(result, run_one, seal, registry)
        def infer(state, seed, row): self.assertEqual(events[-1], ("registry",)); return {"status": "COMPLETE"}
        run.infer_development(result, resolved, infer)
        self.assertEqual((result["resolved_selection_locks"], result["Q_training_fits"], result["DEV_evaluations_completed"]), (6, 5, 5)); self.assertEqual(len(result["failed_paths"]), 1)

    def test_F_gradients_and_fixed_parameters_cannot_enter_optimizer_scope(self):
        model, bolt, raw, cached, _ = fixture(); bolt.model.weight.requires_grad_(True)
        with self.assertRaisesRegex(ValueError, "F_FROZEN"): run.training_status(model, bolt)
        bolt.model.weight.requires_grad_(False); model.D.requires_grad_(True)
        with self.assertRaises(ValueError): run.training_status(model, bolt)

    def test_failed_CP_deployment_is_preserved_before_any_optimizer_step(self):
        model, bolt, raw, cached, _ = fixture(); saved, events = [], []
        def checkpoint(update, state, score, outputs): saved.append((update, copy.deepcopy(state), copy.deepcopy(outputs))); return {"status": "FAILED_PARITY_PRESERVED"}
        with self.assertRaisesRegex(ValueError, "VALID_DEPLOYMENT"): run.train_Q(model, bolt, raw, raw, cached, 92601, events.append, checkpoint, lambda *a: None, updates=2)
        self.assertEqual(len(saved), 1); self.assertEqual(saved[0][0], 0); self.assertFalse(any(e.get("stage") == "OPTIMIZER_STEP_EXECUTED" for e in events))

    def test_registered_metadata_gate_accepts_complete_parent_and_rejects_wrong_coverage(self):
        args = SimpleNamespace(fixed_controls_result="fixed_result.json", fixed_controls_audit="fixed_audit.json")
        result_binding = {"path": args.fixed_controls_result, "sha256": run.FIXED_RESULT_SHA}; audit_binding = {"path": args.fixed_controls_audit, "sha256": run.FIXED_AUDIT_SHA}
        parent = {"parent_report": {"path": "report.md", "sha256": "e"*64}}
        calibration = {"F_parent_signatures_reused": {"before": "same", "after": "same"}}
        controls = {"status": "COMPLETE", "protected_TEST_accessed": False, "controller_source": {"path": run.fixed.__file__, "sha256": run.FIXED_SOURCE_SHA}, "module_source": {"path": str(run.HERE/"orthogonal_latent_basis_v1.py"), "sha256": run.fixed.MODULE_SHA}, "controls_completed": 15, "resolved_control_locks": 15, "FIT_TUNE_evaluations_completed": 30, "DEV_evaluations_completed": 15, "all_15_locks_before_DEV": True, "failed_paths": [], "F_before": "same", "F_after": "same", "control_registry": {"path": "registry.json", "sha256": "f"*64}, **parent}
        audited = {"status": "PASS", "protected_TEST_accessed": False, "diagnostic": result_binding, "audit_source": {"path": "auditor.py", "sha256": run.FIXED_AUDIT_SOURCE_SHA}, "totals": run.FIXED_AUDIT_COUNTS.copy()}
        records = {run.fixed.__file__: controls["controller_source"], args.fixed_controls_result: result_binding, args.fixed_controls_audit: audit_binding}
        def previous(_): return calibration, {}, {}, {}, copy.deepcopy(parent)
        def read(path): return controls if path == args.fixed_controls_result else audited
        with patch.object(run.fixed, "load_gates", side_effect=previous), patch.object(run.old, "record", side_effect=lambda p: records[str(p)]), patch.object(run.old, "read", side_effect=read), patch.object(run.old, "bound", return_value=Path("synthetic")):
            self.assertEqual(run.load_gates(args)[4]["status"], "COMPLETE")
            audited["totals"]["score_fields_checked"] = 4319
            with self.assertRaisesRegex(ValueError, "COVERAGE"): run.load_gates(args)

    def test_closed_parent_audit_gate_before_metadata_reads_and_help_has_no_heavy_imports(self):
        with patch.object(run, "FIXED_AUDIT_SHA", None), patch.object(run.old, "record", side_effect=AssertionError("metadata opened")), self.assertRaisesRegex(ValueError, "NOT_YET_REGISTERED"): run.load_gates(SimpleNamespace())
        script = Path(run.__file__).resolve()
        code = "import runpy,sys;sys.argv=['runner','--help'];\ntry: runpy.run_path("+repr(str(script))+",run_name='__main__')\nexcept SystemExit as e: assert e.code==0\nassert 'torch' not in sys.modules and 'numpy' not in sys.modules\n"
        completed = subprocess.run([sys.executable, "-c", "import sys;sys.path.insert(0,"+repr(str(script.parent))+");"+code], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr); self.assertIn("--fixed-controls-audit", completed.stdout)


if __name__ == "__main__": unittest.main()
