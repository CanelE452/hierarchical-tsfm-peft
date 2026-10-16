"""CPU toy tests for pre-output truth seals and campaign audit coverage."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import time
import unittest

import numpy as np

import audit_campaign as a
import data_contract as d
import evaluation as e
from combination import scaler_fingerprint
import prediction_cache
import raw_calibration


class CampaignAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.private, self.public = self.root / "private", self.root / "public"
        self.private.mkdir(); self.public.mkdir()
        x = np.arange(2400, dtype=np.float32).reshape(800, 3)
        self.task = {"name": "toy", "L": 40, "H": 4, "columns": ["x", "y", "z"], "physical": x,
                     "observed": np.ones_like(x, bool), "data_sha256": "toy_data", "split_ranges": d.split_ranges(800)}
        self.mean, self.std, _ = d.fit_statistics(self.task, self.task["split_ranges"]["A_stop"][1])
        oo = d.origins(self.task["split_ranges"]["V_select"], 40, 4, 4)
        targets, masks = zip(*(d.target_window(self.task, int(o), self.mean, self.std) for o in oo))
        self.arrays = {"target_normalized": np.stack(targets), "observed": np.stack(masks), "origins": oo,
                       "channel_ids": np.asarray(self.task["columns"])}
        self.parents = []
        for name in ("parent", "g"):
            parent = prediction_cache.predict_parent(name, self.task, "V_select", self.private, self.public, None, baseline="CURRENT")
            self.parents.append(parent)

    def registry(self, names, partial=False):
        rows = [{"experiment_id": name, "execution_status": "PARTIAL_SCREEN025" if partial else "COMPLETE_VALID",
                 "reason": "limited screen" if partial else "", "requirement": "MANDATORY_MINIMUM"} for name in names]
        rows += [{"experiment_id": name, "execution_status": "COMPLETE_VALID", "reason": "", "requirement": "MANDATORY_MINIMUM"} for name in ("parent", "g")]
        rows.append({"experiment_id": "blocked", "execution_status": "BLOCKED_ACCESS_OR_LICENSE", "reason": "official HTTP401", "requirement": "MANDATORY_MINIMUM"})
        for filename in ("EXPERIMENT_REGISTRY.csv", "EXPERIMENT_REGISTRY_INITIAL.csv"):
            with (self.public / filename).open("w", newline="", encoding="utf8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    def output(self, spec, refs):
        run_id = spec["run_id"]
        private = self.private / "combinations" / run_id
        private.mkdir(parents=True)
        (private / "selected.pt").write_bytes(b"toy selected parameter bytes")
        contract = dict(next(iter(refs.values()))["expected_receipt_contract"])
        receipt = {"run_id": run_id, "variant": spec["variant"], "execution_status": "COMPLETE_VALID", "contract": contract,
                   "selected_checkpoint_sha256": d.sha256_file(private / "selected.pt"), "prediction_artifacts": {},
                   "parent_models": {"parent_v": self.parents[0]["metadata"], "g_v": self.parents[1]["metadata"]}}
        labels = ["alpha0", "alpha05", "alpha1", "fitted"] if spec["variant"] == "alpha_family" else ["head"]
        for i, label in enumerate(labels):
            arrays = dict(self.arrays, pred_normalized=np.full_like(self.arrays["target_normalized"], i * .1))
            filename = f"{label}_V_select.npz" if spec["variant"] == "alpha_family" else "V_select.npz"
            path = private / filename
            np.savez_compressed(path, **arrays)
            receipt["prediction_artifacts"][label] = {"V_select": {"file": filename, "sha256": d.sha256_file(path)}}
        if spec["variant"] == "alpha_family": receipt["alpha_fit"] = {"alpha_fit": .4}
        return receipt

    def prepare(self, run_id="toy_CONDITIONAL_run", variant="conditional"):
        spec = {"run_id": run_id, "variant": variant, "seed": 9, "device": "cpu"}
        refs = a.prepare_combination_reference(run_id, self.task, *self.parents, self.mean, self.std,
                                               self.private, self.public, spec=spec)
        return spec, refs

    def test_alpha_aliases_are_all_registered_and_rescored_F0(self):
        spec, refs = self.prepare("toy_ALPHA_FIT_run", "alpha_family")
        self.assertEqual({r["run_id"] for r in refs.values()}, {"toy_ALPHA0_run", "toy_ALPHA05_run", "toy_ALPHA1_run", "toy_ALPHA_FIT_run"})
        receipt = self.output(spec, refs)
        wrapped = a.wrap_combination_output(receipt, refs, self.private, self.public)
        self.registry([r["run_id"] for r in refs.values()])
        report = a.audit_all(self.public, self.private)
        self.assertEqual(report["prediction_rescoring_passed"], 6)
        self.assertEqual(report["checkpoint_replay_count"], 0)
        self.assertEqual(report["training_rerun_count"], 0)
        for item in wrapped:
            self.assertEqual(item["cost_record"]["actual_F_streams"], 0)
            self.assertTrue(item["metadata"]["target_cohort_sealed_before_fit"])
            self.assertTrue(item["metadata"]["selected_model_identity_bound_after_fit"])

    def test_wrong_origins_rejected_even_with_rehashed_output(self):
        spec, refs = self.prepare()
        receipt = self.output(spec, refs)
        path = self.private / "combinations" / spec["run_id"] / "V_select.npz"
        with np.load(path) as f: arrays = {k: f[k] for k in f.files}
        arrays["origins"] = arrays["origins"] + 1
        np.savez_compressed(path, **arrays)
        receipt["prediction_artifacts"]["head"]["V_select"]["sha256"] = d.sha256_file(path)
        with self.assertRaises(ValueError): a.wrap_combination_output(receipt, refs, self.private, self.public)

    def test_reference_cannot_be_created_after_predictions(self):
        output = self.private / "combinations" / "late"
        output.mkdir(parents=True); (output / "V_select.npz").write_bytes(b"already predicted")
        with self.assertRaises(ValueError): self.prepare("late")

    def test_exact_selected_checkpoint_hash_is_required(self):
        spec, refs = self.prepare()
        receipt = self.output(spec, refs)
        (self.private / "combinations" / spec["run_id"] / "selected.pt").write_bytes(b"changed")
        with self.assertRaises(ValueError): a.wrap_combination_output(receipt, refs, self.private, self.public)

    def test_partial_scope_and_missing_blocked_registry_detected(self):
        spec, refs = self.prepare()
        a.wrap_combination_output(self.output(spec, refs), refs, self.private, self.public)
        self.registry([spec["run_id"]], partial=True)
        report = a.audit_all(self.public, self.private)
        self.assertEqual(report["prediction_rescoring_passed"], 3)
        self.assertFalse(report["mandatory_coverage_complete"])
        path = self.public / "EXPERIMENT_REGISTRY.csv"
        lines = path.read_text().splitlines(); path.write_text("\n".join(lines[:-1]) + "\n")
        report2 = a.audit_all(self.public, self.private)
        self.assertTrue(report2["errors"])
        self.assertNotEqual(report["audit_batch"], report2["audit_batch"])

    def test_cached_combination_cannot_claim_new_F_streams(self):
        spec, refs = self.prepare()
        a.wrap_combination_output(self.output(spec, refs), refs, self.private, self.public)
        self.registry([spec["run_id"]])
        path = self.public / "predictions" / spec["run_id"] / "V_select/PREDICTION_MANIFEST.json"
        manifest = json.loads(path.read_text())
        manifest["cost_record"]["actual_F_streams"] = len(self.arrays["origins"])
        path.write_text(json.dumps(manifest))
        report = a.audit_all(self.public, self.private)
        self.assertEqual(report["prediction_rescoring_passed"], 2)
        self.assertTrue(report["errors"])

    def test_raw_bias_exact_artifact_and_dense_parent_rescoring(self):
        run_id = "toy_RAW_CALIBRATED_run"
        raw_v = self.parents[0]
        raw_m = prediction_cache.predict_parent("parent", self.task, "M_cal", self.private, self.public, None, baseline="CURRENT", dense=True)
        spec = {"run_id": run_id, "variant": "raw_calibration"}
        refs = a.prepare_combination_reference(run_id, self.task, raw_v, raw_v, self.mean, self.std, self.private, self.public, spec=spec, kind="raw_calibration")
        receipt = raw_calibration.fit_raw_calibration(run_id, self.task, raw_m, raw_v, self.mean, self.std, self.private, self.public)
        wrapped = a.wrap_combination_output(receipt, refs, self.private, self.public, kind="raw_calibration")
        self.assertIsNotNone(wrapped[0]["metadata"]["realized_model_binding"]["selected_artifact_sha256"])
        self.registry([run_id])
        report = a.audit_all(self.public, self.private)
        self.assertFalse(report["errors"])
        self.assertEqual(report["dense_private_rescoring_passed"], 1)

    def test_registered_extension_allowed_but_unregistered_extra_refused(self):
        self.registry([])
        current = self.public / "EXPERIMENT_REGISTRY.csv"
        rows = a._rows(current)
        extra = dict(rows[0], experiment_id="new_shared_prefix", execution_status="PENDING_QUEUE", reason="registered next route")
        extension = self.public / "EXPERIMENT_REGISTRY_EXTENSION_toy.csv"
        with extension.open("w", newline="", encoding="utf8") as f:
            writer = csv.DictWriter(f, fieldnames=list(extra)); writer.writeheader(); writer.writerow(extra)
        extension.with_suffix(".json").write_text(json.dumps({"registered_unix": time.time(), "csv_sha256": d.sha256_file(extension),
            "before_execution": True, "before_T_selection": True, "run_ids": [extra["experiment_id"]]}))
        with current.open("a", newline="", encoding="utf8") as f: csv.DictWriter(f, fieldnames=list(extra)).writerow(extra)
        report = a.audit_all(self.public, self.private)
        self.assertFalse(report["errors"])
        self.assertEqual(report["registered_extension_rows"], 1)
        with current.open("a", newline="", encoding="utf8") as f: csv.DictWriter(f, fieldnames=list(extra)).writerow(dict(extra, experiment_id="unregistered_extra"))
        self.assertTrue(a.audit_all(self.public, self.private)["errors"])

    def test_extension_cannot_change_previously_registered_contract(self):
        self.registry([])
        current = self.public / "EXPERIMENT_REGISTRY.csv"
        rows = a._rows(current); rows[0]["requirement"] = "OPTIONAL_AFTER_NEGATIVE_RESULT"
        with current.open("w", newline="", encoding="utf8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        self.assertTrue(a.audit_all(self.public, self.private)["errors"])

    def test_malformed_manifest_is_reported_without_losing_other_audits(self):
        self.registry([])
        broken = self.public / "predictions/broken/V_select/PREDICTION_MANIFEST.json"
        broken.parent.mkdir(parents=True)
        broken.write_text("{bad json")
        report = a.audit_all(self.public, self.private)
        self.assertEqual(report["prediction_rescoring_passed"], 2)
        self.assertTrue(report["errors"])

    def legacy_output(self, expected=None):
        import legacy_evaluation as legacy
        from test_legacy_evaluation import ToyFoundation, ToyModel, ToyWindow
        relative = Path("legacy_e0/evaluation/toy/cp2800")
        legacy.evaluate_checkpoint(ToyModel(), ToyFoundation(), ToyWindow(), ["a", "b"],
            dict(legacy_id="toy", checkpoint_update=2800), self.private / relative, self.public / relative,
            expected_B=expected, device="cpu", horizon=2, length=3)
        return self.public / relative

    def test_legacy_B_is_actually_rescored_separately_and_tampering_fails(self):
        self.registry([])
        folder = self.legacy_output()
        report = a.audit_all(self.public, self.private)
        self.assertFalse(report["errors"])
        self.assertEqual(report["prediction_rescoring_passed"], 2)
        self.assertEqual(report["legacy_B_rescoring_passed"], 1)
        self.assertFalse(report["legacy_B_counted_as_new_V_T"])
        (folder / "PUBLIC_ERROR_STATS.csv.gz").write_bytes(b"tampered")
        bad = a.audit_all(self.public, self.private)
        self.assertEqual(bad["prediction_rescoring_passed"], 2)
        self.assertEqual(bad["legacy_B_rescoring_passed"], 0)
        self.assertTrue(bad["errors"])

    def test_legacy_namespace_does_not_hide_invalid_or_curve_mismatch(self):
        self.registry([])
        folder = self.legacy_output(expected=dict(mse=0., mae=0.))
        self.assertTrue(a.audit_all(self.public, self.private)["errors"])
        path = folder / "PREDICTION_MANIFEST.json"
        original = json.loads(path.read_text())
        for changed in (dict(original, split="T_final"),
                        dict(original, historically_unseen=True),
                        dict(original, identity=dict(original["identity"], checkpoint_update=4800)),
                        dict(run_id="hidden", split="V_select")):
            with self.subTest(changed=changed.get("split")):
                path.write_text(json.dumps(changed))
                report = a.audit_all(self.public, self.private)
                self.assertEqual(report["legacy_B_rescoring_passed"], 0)
                self.assertTrue(report["errors"])


if __name__ == "__main__": unittest.main()
