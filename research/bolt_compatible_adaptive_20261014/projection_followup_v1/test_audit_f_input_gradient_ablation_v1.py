"""Generated tensors/files only; no campaign assets or real F/training replay."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy
import hashlib
import json
import tempfile
import unittest

import numpy as np
import torch

import audit_f_input_gradient_ablation_v1 as audit
from linear_e_adaptation_v1 import LinearLagAdapter, global_fit_macro_mse

audit.initialize_numpy()


def toy_future(z):
    return .3 * z[:, -1:] + .2 * z.mean(1, keepdim=True) + torch.linspace(-.2, .2, 48)[None, :, None]


def raw_fixture(n=6):
    rng = np.random.default_rng(13 + n); history = rng.normal(size=(n, 512, 3)).astype(np.float32)
    observed = np.ones((n, 48, 3), bool); observed[0, :23, 0] = False
    if n == 6: observed[:, :, 2] = False
    target = rng.normal(size=observed.shape).astype(np.float32); target[~observed] = np.nan
    raw = {"history": history, "history_observed": np.ones(history.shape, bool), "target": target, "observed": observed, "raw_last": history[:, -1].copy(), "origins": np.arange(n, dtype=np.int64), "mean": np.array([1., 2., 3.], np.float32), "std": np.array([2., 3., 4.], np.float32)}
    state = {"E": rng.normal(size=(3, 2)).astype(np.float32), "D": rng.normal(size=(2, 3)).astype(np.float32), "center": np.array([.1, -.2, .3], np.float32)}
    return raw, state


def prediction(raw, state):
    model = LinearLagAdapter(state["center"], state["E"], state["D"], (0,))
    with torch.no_grad():
        p, z, f = model(torch.from_numpy(raw["history"]), toy_future, return_components=True); features = model.lag_features(z, f)
    return {"prediction": p.numpy(), "latent_history": z.numpy(), "latent_future": f.numpy(), "lag_features": features.numpy(), **{key: raw[key] for key in ("target", "observed", "raw_last", "origins", "mean", "std")}}


def probe_fixture(fit=None, initial=None, seed=92601):
    if fit is None: fit, initial = raw_fixture()
    full = LinearLagAdapter(initial["center"], initial["E"], initial["D"], (0,)); stop = LinearLagAdapter(initial["center"], initial["E"], initial["D"], (0,))
    ids = np.random.default_rng(seed).integers(0, len(fit["history"]), size=(600, 4))[0]; x, y, mask = (torch.from_numpy(fit[key][ids]) for key in ("history", "target", "observed")); counts = fit["observed"].sum((0, 1))
    p, z, f = full(x, toy_future, return_components=True); q, zs, fs = stop(x, lambda value: toy_future(value).detach(), return_components=True)
    loss = global_fit_macro_mse(p, y, mask, torch.from_numpy(counts), len(fit["history"])); sloss = global_fit_macro_mse(q, y, mask, torch.from_numpy(counts), len(fit["history"]))
    eg, dg = torch.autograd.grad(loss, (full.E, full.D), retain_graph=True); fg = torch.autograd.grad(loss, f, retain_graph=True)[0]; ig = torch.autograd.grad(f, z, fg, retain_graph=True)[0]; via = torch.autograd.grad(z, full.E, ig, retain_graph=True)[0]; se, sd = torch.autograd.grad(sloss, (stop.E, stop.D))
    values = dict(full_E_gradient=eg, full_D_gradient=dg, F_future_gradient=fg, F_input_gradient=ig, F_path_E_gradient=via, direct_E_gradient=eg - via, stop_E_gradient=se, stop_D_gradient=sd, full_prediction=p, stop_prediction=q, full_latent_history=z, stop_latent_history=zs, full_latent_future=f, stop_latent_future=fs)
    arrays = {key: value.detach().numpy().copy() for key, value in values.items()}; arrays.update({"initial_" + key: value.copy() for key, value in initial.items()}); arrays.update(batch_indices=ids, FIT_channel_observed_counts=counts)
    checks = {}
    for name, a, b in (("stop_E_equals_full_minus_F_path", se, eg - via), ("stop_D_equals_full_D", sd, dg)):
        error = a.detach().numpy().astype(np.float64) - b.detach().numpy().astype(np.float64); checks[name] = {"pass": True, "atol": 1e-5, "rtol": 2e-6, "max_abs": float(np.abs(error).max()), "rmse": float(np.sqrt(np.mean(error * error)))}
    entry = {"status": "PASS", "optimizer_updates": 0, "forward_byte_exact": True, "finite_gradient_vectors": True, "F_parameters_frozen_and_grad_none": True, "parameter_grad_fields_none": True, "initial_state_sha256": audit.base.state_hash(initial), "loss": float(loss.detach()), "gradient_checks": checks, "gradient_norms": {key: float(np.linalg.norm(arrays[key].astype(np.float64))) for key in audit.GRADIENT_FIELDS}}
    return arrays, entry, fit, initial


class StopGradientAuditTests(unittest.TestCase):
    def test_numpy_derivatives_match_toy_autograd_with_nan_masks_and_inactive_channel(self):
        arrays, entry, fit, initial = probe_fixture(); result = audit.analytic_probe(arrays, entry, fit, initial, 92601)
        self.assertEqual(result["analytic_comparisons"], 7); self.assertEqual(result["gradient_vector_fields"], 8); self.assertFalse(result["saved_F_input_VJP_independently_replayed"])
        self.assertTrue(np.isnan(fit["target"][~fit["observed"]]).all()); self.assertTrue(np.array_equal(arrays["full_D_gradient"][:, 2], np.zeros(2)))

    def test_wrong_F_input_vjp_or_direct_or_D_gradient_rejected_even_with_updated_norm(self):
        original = probe_fixture()
        for key in ("F_input_gradient", "direct_E_gradient", "stop_D_gradient"):
            arrays, entry, fit, initial = copy.deepcopy(original); arrays[key].flat[0] += np.float32(.05); entry["gradient_norms"][key] = float(np.linalg.norm(arrays[key].astype(np.float64)))
            with self.subTest(key=key), self.assertRaises(ValueError): audit.analytic_probe(arrays, entry, fit, initial, 92601)

    def test_full_FIT_denominator_first_batch_loss_and_probe_update_contract(self):
        original = probe_fixture()
        for key in ("FIT_channel_observed_counts", "batch_indices", "loss", "optimizer_updates"):
            arrays, entry, fit, initial = copy.deepcopy(original)
            if key in arrays: arrays[key].flat[0] += 1
            else: entry[key] += .2 if key == "loss" else 1
            with self.subTest(key=key), self.assertRaises(ValueError): audit.analytic_probe(arrays, entry, fit, initial, 92601)

    def test_nonfinite_gradient_and_incorrect_reported_pair_error_rejected(self):
        arrays, entry, fit, initial = probe_fixture(); arrays["F_input_gradient"][0, 0, 0] = np.nan
        with self.assertRaises(ValueError): audit.analytic_probe(arrays, entry, fit, initial, 92601)
        arrays, entry, fit, initial = probe_fixture(); entry["gradient_checks"]["stop_D_equals_full_D"]["rmse"] = .01
        with self.assertRaises(ValueError): audit.analytic_probe(arrays, entry, fit, initial, 92601)

    def test_executed_source_detach_only_and_closed_gate_before_file_reads(self):
        text = (audit.HERE / "run_f_input_gradient_ablation_v1.py").read_text(encoding="utf-8")
        self.assertTrue(audit.source_contract(text)["detach_only_F_output"])
        with self.assertRaises(ValueError): audit.source_contract(text.replace("return self.bolt(z).detach()", "return self.bolt(z.detach())"))
        with patch.object(audit, "RESULT_SHA", None), patch.object(audit.base, "file_record", side_effect=AssertionError("unexpected file read")), self.assertRaisesRegex(ValueError, "NOT_YET_REGISTERED"): audit.metadata("absent", lambda: None)

    def test_600_log_clip_formula_and_optimizer_step_mislabel_rejected(self):
        fit, _ = raw_fixture(); order = np.random.default_rng(92601).integers(0, 6, size=(600, 4)); scale = 1 / (2 + 1e-6)
        logs = [{"stage": "TRAINING_UPDATE", "update": i + 1, "loss": .1, "E_gradient_norm_before_clip": 1., "D_gradient_norm_before_clip": 1.5, "global_gradient_norm_before_clip": 2., "clipped": True, "global_clip_scale_from_logged_norm": scale, "D_gradient_norm_after_clip_formula": scale * 1.5, "D_gradient_scaling_is_AdamW_step_norm": False} for i in range(600)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "updates.jsonl"
            def write(): path.write_text("\n".join(json.dumps(row) for row in logs), encoding="utf-8")
            write(); row = {"updates_log": audit.base.file_record(path), "training_updates_completed": 600, "learn_E": True, "global_clipping_fraction": 1., "FIT_channel_observed_counts": fit["observed"].sum((0, 1)).tolist(), "seed": 92601, "order_sha256": hashlib.sha256(order.tobytes()).hexdigest()}
            self.assertEqual(audit.check_logs(row, fit), 600)
            logs[12]["D_gradient_scaling_is_AdamW_step_norm"] = True; write(); row["updates_log"] = audit.base.file_record(path)
            with self.assertRaisesRegex(ValueError, "NOT_OPTIMIZER_STEP"): audit.check_logs(row, fit)

    def test_four_condition_synthetic_json_npz_bundle_all_CP_ties_locks_and_gradients(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); bundle, patches = synthetic_bundle(root)
            with patch.multiple(audit, **patches), patch.object(audit.base, "TRAIN_SHA", bundle["parent_sha"]), patch.object(audit.base, "FIT_N", 6), patch.object(audit.base, "TUNE_N", 4):
                report = audit.audit_result(bundle["path"], SimpleNamespace(check=lambda: None))
                self.assertEqual(report["status"], "PASS"); total = report["totals"]
                for key, value in (("stop_gradient_fits_checked", 4), ("checkpoint_states_checked", 20), ("selected_FIT_DEV_predictions_checked", 8), ("logged_updates_checked", 2400), ("analytic_gradient_comparisons_checked", 28), ("CP0_full_stop_parent_component_byte_comparisons_checked", 32)):
                    self.assertEqual(total[key], value)
                self.assertTrue(all(row["selected_update"] == 0 for data in report["datasets"].values() for row in data.values()))
                self.assertFalse(report["independent_F_input_VJP_replayed"]); json.dumps(report, allow_nan=False)
                value = json.loads(bundle["path"].read_text()); value["datasets"]["a"]["conditions"]["92601"]["selected_update"] = 100
                bundle["path"].write_text(json.dumps(value), encoding="utf-8")
                with patch.object(audit, "RESULT_SHA", audit.prior.file_sha(bundle["path"])), self.assertRaisesRegex(ValueError, "TUNE_ONLY"): audit.audit_result(bundle["path"], SimpleNamespace(check=lambda: None))


def synthetic_bundle(root):
    def json_file(name, value):
        path = root / (name + ".json"); path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8"); return audit.base.file_record(path)
    def npz(name, values):
        path = root / (name + ".npz"); np.savez(path, **values)
        return {**audit.base.file_record(path), "arrays": {k: {"dtype": str(v.dtype), "shape": list(v.shape), "sha256": hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()} for k, v in values.items()}}
    def forecast(name, raw, state):
        saved = prediction(raw, state); complete = np.broadcast_to(raw["observed"].all(2, keepdims=True), raw["observed"].shape)
        return {"artifact": npz(name, saved), "scores_all_observed": audit.prior.scores(saved["prediction"], raw["target"], raw["observed"], raw["std"]), "scores_complete_mask": audit.prior.scores(saved["prediction"], raw["target"], complete, raw["std"]) if complete.any() else {"status": "NO_COMPLETE_C"}}
    science, parent, datasets, registry_rows = {"status": "COMPLETE", "datasets": {}}, {"status": "COMPLETE", "datasets": {}, "controller_source": audit.base.file_record(audit.HERE / "run_linear_e_adaptation_v1.py"), "pure_source": audit.base.file_record(audit.HERE / "linear_e_adaptation_v1.py"), "F_before": {"f": "same"}, "F_after": {"f": "same"}}, {}, []
    for data in ("a", "b"):
        raws = {split: raw_fixture(n)[0] for split, n in (("fit", 6), ("tune", 4), ("dev", 3))}; _, initial = raw_fixture(); state = npz(data + "_state", initial)
        science["datasets"][data] = {"spaces": {"PCA": {"conditions": {"L0_FORECAST": {"splits": {split: {"input_source": npz(data + split + "_raw", raw)} for split, raw in raws.items()}}}}}}
        parent["datasets"][data], datasets[data] = {"conditions": {}}, {"conditions": {}}
        for seed in audit.SEEDS:
            name = data + str(seed); array, probe, _, _ = probe_fixture(raws["fit"], initial, seed)
            cp = {"update": 0, "state": state, **forecast(name + "CP0", raws["tune"], initial)}; cp["tune"] = cp["scores_all_observed"]["normalized"]
            previous = {"CP0_TUNE": cp, "selected_update": 0, "selected_state": state, "initial_state_sha256": audit.base.state_hash(initial), "order_sha256": hashlib.sha256(np.random.default_rng(seed).integers(0, 6, size=(600, 4)).tobytes()).hexdigest()}
            parent_name = data + "_L0_E_LEARN_s" + str(seed); parent["datasets"][data]["conditions"][parent_name] = previous
            saved = prediction(raws["tune"], initial); parity = npz(name + "parity", {**{prefix + key: saved[key] for prefix in ("full_", "stop_") for key in audit.COMPONENTS}, "origins": raws["tune"]["origins"]})
            probe.update(artifact=npz(name + "probe", array), CP0_TUNE_parity_artifact=parity, reference_CP0=cp["artifact"])
            logs = [{"stage": "TRAINING_UPDATE", "update": i + 1, "loss": .1, "E_gradient_norm_before_clip": .3, "D_gradient_norm_before_clip": .4, "global_gradient_norm_before_clip": .5, "clipped": False, "global_clip_scale_from_logged_norm": 1., "D_gradient_norm_after_clip_formula": .4, "D_gradient_scaling_is_AdamW_step_norm": False} for i in range(600)]
            log = root / (name + ".jsonl"); log.write_text("\n".join(json.dumps(r) for r in logs), encoding="utf-8")
            row = {"status": "COMPLETE", "dataset": data, "seed": seed, "lag": "L0", "learn_E": True, "training_updates_completed": 600, "checkpoint_count": 5, "initial_state_sha256": previous["initial_state_sha256"], "order_sha256": previous["order_sha256"], "FIT_channel_observed_counts": raws["fit"]["observed"].sum((0, 1)).tolist(), "global_clipping_fraction": 0., "updates_log": audit.base.file_record(log), "CP0_TUNE_parity_artifact": parity, "reference_CP0": cp["artifact"], "CP0_TUNE_byte_checks": {key: {"full_stop_byte_exact": True, "full_parent_byte_exact": True} for key in audit.COMPONENTS}, "probe": probe, "checkpoints": [{**cp, "update": update} for update in audit.base.CHECKPOINTS], "selected_update": 0, "selected_tune": cp["tune"], "selected_state": state, "selected_state_sha256": previous["initial_state_sha256"], "full_gradient_reference": {"condition": parent_name, "selected_update": 0, "selected_state": state}, "FIT": forecast(name + "FIT", raws["fit"], initial)}
            lock = json_file(name + "lock", {**row, "sealed_unix": 0., "DEV_opened_before_lock": False}); row["selection_lock"] = lock; row["DEV"] = forecast(name + "DEV", raws["dev"], initial)
            datasets[data]["conditions"][str(seed)] = row; registry_rows.append({"dataset": data, "seed": seed, "status": "COMPLETE", "selection_lock": lock})
    science_ref = json_file("science", science); science_audit = json_file("science_audit", {"status": "PASS", "diagnostic": science_ref}); parent.update(causal_lag_diagnostic=science_ref, causal_lag_audit=science_audit)
    parent_ref = json_file("parent", parent); parent_audit = json_file("parent_audit", {"status": "PASS", "diagnostic": parent_ref})
    report = root / "REPORT.md"; report.write_text("Synthetic report", encoding="utf-8"); report_ref = audit.base.file_record(report); manifest = json_file("manifest", {"files": []}); report_audit = json_file("report_audit", {"status": "PASS", "report": {**report_ref, "bytes": report.stat().st_size}, "manifest": {**manifest, "bytes": Path(manifest["path"]).stat().st_size}})
    dummy = json_file("dummy", {}); registry = json_file("registry", {"status": "ALL_FOUR_RESOLVED_BEFORE_DEV", "DEV_opened_before_seal": False, "sealed_unix": 0., "conditions": registry_rows})
    result = {"status": "COMPLETE", "controller_source": audit.base.file_record(audit.HERE / "run_f_input_gradient_ablation_v1.py"), "full_gradient_parent": parent_ref, "full_gradient_parent_audit": parent_audit, "parent_report": report_ref, "parent_report_manifest": manifest, "parent_report_audit": report_audit, "causal_lag_diagnostic": science_ref, "causal_lag_audit": science_audit, "plan": dummy, "source_records": [], "protected_prior_records": [], "input_records": [], "model_binary_before": dummy, "model_config_before": dummy, "F_before": {"f": "same"}, "F_after": {"f": "same"}, "selection_registry": registry, "datasets": datasets, "new_backbone_training": False, "protected_TEST_accessed": False, "independent_holdout": False, "strict_K_only": False, "whole_deployment_cost_measured": False, "new_full_gradient_fits": 0, "probe_optimizer_updates": 0, "full_gradient_conditions_reused": 4, "all_4_locks_before_DEV": True, "resolved_selection_locks": 4, "stop_gradient_training_fits": 4, "checkpoint_states_saved": 20, "training_updates_completed": 2400, "failed_paths": []}
    binding = json_file("RESULT", result)
    return {"path": Path(binding["path"]), "parent_sha": parent_ref["sha256"]}, {"RESULT_SHA": binding["sha256"], "PARENT_AUDIT_SHA": parent_audit["sha256"], "REPORT_SHA": report_ref["sha256"], "MANIFEST_SHA": manifest["sha256"], "REPORT_AUDIT_SHA": report_audit["sha256"], "DATASETS": {"a": (3, 2, "dev", 3), "b": (3, 2, "dev", 3)}}


if __name__ == "__main__": unittest.main()
