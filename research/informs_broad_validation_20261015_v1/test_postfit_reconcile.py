"""CPU-only reconciliation checks; production caches and registry stay read-only."""
import copy
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import postfit
from test_postfit import Lease


PRIVATE_CACHE = Path(os.environ.get("INFORMS_CACHE_ROOT", Path(__file__).resolve().parents[2]/".cache/informs_broad_validation_20261015_v1"))


class ReconcileTests(unittest.TestCase):
    def saved(self):
        registry = {row["experiment_id"]: row for row in postfit.campaign.registry_rows()}
        records = []
        for dataset in ("beijing", "jena", "electricity", "pems08"):
            for family in ("CURRENT", "SEASONAL_NAIVE", "RAW_RIDGE"):
                run_id = postfit.parent_id(dataset, family)
                path = postfit.HERE / "predictions" / run_id / "V_select/PREDICTION_MANIFEST.json"
                if not path.is_file() or not PRIVATE_CACHE.is_dir():
                    self.skipTest("Actual private campaign caches are unavailable")
                records.append((registry[run_id], json.loads(path.read_text(encoding="utf-8"))))
        return records

    def test_actual_twelve_caches_reconcile_without_registry_writes(self):
        records = self.saved()
        with patch.object(postfit, "CACHE", PRIVATE_CACHE), \
             patch.object(postfit.campaign, "record_status") as statuses:
            result = postfit.reconcile_cpu_baselines(Lease())
        self.assertEqual(result["reconciled_run_ids"], sorted(row["experiment_id"] for row, _ in records))
        self.assertEqual(result["prediction_rescoring_passed"], 12)
        self.assertFalse(result["checkpoint_replay"])
        self.assertTrue(set(call.args[0] for call in statuses.call_args_list).issubset(result["reconciled_run_ids"]))

    def test_actual_families_have_exact_canonical_execution_kinds(self):
        for row, manifest in self.saved():
            with self.subTest(run_id=row["experiment_id"]):
                postfit._validate_cpu_baseline_identity(row, manifest)
                expected = "SEASONAL" if row["family"] == "SEASONAL_NAIVE" else row["family"]
                self.assertEqual(manifest["metadata"]["family"], expected)

    def test_bad_identity_is_rejected_for_each_actual_saved_baseline(self):
        edits = (("family", "B0"), ("run_id", "another_route"), ("split", "T_final"),
                 ("fold", "secondary"), ("cohort", "wrong_dataset:primary:V_select:stride48"),
                 ("data_sha", "wrong_data"), ("data_sha256", "wrong_data"),
                 ("dense", True), ("role", "AplusM_control"))
        for row, manifest in self.saved():
            for key, value in edits:
                with self.subTest(run_id=row["experiment_id"], field=key):
                    changed = copy.deepcopy(manifest)
                    changed["metadata"][key] = value
                    with self.assertRaisesRegex(ValueError, "identity differs"):
                        postfit._validate_cpu_baseline_identity(row, changed)
            for key in ("run_id", "split", "cohort", "data_sha"):
                changed = copy.deepcopy(manifest)
                changed[key] = "wrong_top_level_identity"
                with self.assertRaisesRegex(ValueError, "identity differs"):
                    postfit._validate_cpu_baseline_identity(row, changed)


if __name__ == "__main__":
    unittest.main()
