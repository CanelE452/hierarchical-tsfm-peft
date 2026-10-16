"""Lossless publication projection checks on synthetic public statistics only."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import publication_projection as module


class PublicationProjectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="publication_projection_")
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)
        self.public = self.repo / "research" / "campaign"
        self.public.mkdir(parents=True)
        self.cache = self.repo / "cache"; self.cache.mkdir()
        self.family, self.label, self.alias = "family_ALPHA_FIT_case", "alpha05", "family_ALPHA05_case"
        self.original_id = self.family + "__" + self.label
        self.original = self.public / "combinations" / self.family / "alpha05_V_select_error_stats.csv.gz"
        self.standard = self.public / "predictions" / self.alias / "V_select" / "PUBLIC_ERROR_STATS.csv.gz"
        self.manifest_path = self.standard.with_name("PREDICTION_MANIFEST.json")
        self.reference_path = self.standard.with_name("REFERENCE_CONTRACT.json")
        self.pre_path = self.standard.with_name("PRE_TARGET_CONTRACT.json")
        self.archive = self.cache / "combinations" / self.family / "alpha05_V_select.npz"
        self.archive.parent.mkdir(parents=True)
        self.archive.write_bytes(b"opaque synthetic archive; never interpreted as target arrays")
        arrays = {key: dict(shape=[1, 4, 1] if key.endswith("normalized") or key == "observed" else [1],
                            dtype="float32", sha256=key * 2)
                  for key in module.ARRAY_KEYS}
        identity = dict(run_id=self.alias, split="V_select", cohort="toy:V", source_sha="source",
                        config_sha="output_config", data_sha="data", scaler_sha="scaler")
        contract = dict(source_sha="source", config_sha="fit_config", data_sha="data", scaler_sha="scaler", model_sha="template")
        binding = dict(output_label=self.label, selected_artifact_cache_relative_path=None, selected_artifact_sha256=None,
                       alpha=.5, parent_model_shas={})
        common_arrays = {key: arrays[key] for key in module.REFERENCE_KEYS}
        pre = dict(identity, family_run_id=self.family, label=self.label, kind="combination", variant="alpha_family",
                   expected_receipt_contract=contract, arrays=common_arrays, parent_anchors={},
                   cache_relative_path=self.archive.relative_to(self.cache).as_posix())
        self.write_json(self.pre_path, pre)
        reference = dict(identity, model_sha=module.fingerprint(binding), arrays=common_arrays,
                         realized_model_binding=binding, pre_target_contract_sha256=module.sha256_file(self.pre_path),
                         pre_target_contract_public_relative_path=self.pre_path.relative_to(self.public).as_posix())
        self.write_json(self.reference_path, reference)
        score = dict(run_id=self.alias, split="V_select", cohort="toy:V", n_origins=1, num_channels=1, horizon=4,
                     n_observed=4, mse=.25, mae=.5)
        meta = dict(family_run_id=self.family, output_label=self.label, reference_kind="combination", actual_parent_manifests={})
        manifest = dict(identity, model_sha=reference["model_sha"], arrays=arrays, score=score, metadata=meta,
                        archive_sha256=module.sha256_file(self.archive), cache_relative_path=pre["cache_relative_path"],
                        public_stats_relative_path=self.standard.relative_to(self.public).as_posix())
        self.write_json(self.manifest_path, manifest)
        receipt = dict(run_id=self.family, execution_status="COMPLETE_VALID", variant="alpha_family", contract=contract,
                       prediction_artifacts={self.label: {"V_select": dict(file=self.archive.name,
                           sha256=manifest["archive_sha256"], arrays=arrays)}},
                       scores=[dict(score, run_id=self.original_id)])
        self.receipt_path = self.original.with_name("receipt.json")
        self.write_json(self.receipt_path, receipt)
        self.write_stats(self.original, self.original_id)
        self.write_stats(self.standard, self.alias)

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")

    def edit_json(self, path, change):
        value = json.loads(path.read_text()); change(value); self.write_json(path, value)

    def write_stats(self, path, run_id, *, sse="0.25", count="1"):
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream); writer.writerow(module.COLUMNS)
            for start, end in ((0, 1), (1, 2), (2, 4)):
                writer.writerow([run_id, "V_select", "toy:V", "20", "c", start, end, sse, ".5", count])

    def project(self, **kwargs):
        return module.create_projection(self.public, self.cache, **kwargs)

    def test_exact_stats_excluded_and_lineage_snapshot_keeps_sources_and_standard(self):
        before = {p: p.read_bytes() for p in self.public.rglob("*") if p.is_file()}
        directory = self.project()
        result = json.loads((directory / "PROJECTION.json").read_text())
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["excluded_duplicate_count"], 1)
        pair = result["duplicate_statistics"][0]
        self.assertEqual(pair["row_count"], 3)
        self.assertEqual(pair["non_run_id_columns_sha256"], pair["replacement_non_run_id_columns_sha256"])
        self.assertEqual(pair["run_id_mapping"], {self.original_id: self.alias})
        self.assertTrue(pair["opaque_private_archive_sha_verified"])
        names = (directory / "PAYLOAD_FILES.txt").read_text().splitlines()
        self.assertNotIn(self.original.relative_to(self.repo).as_posix(), names)
        for path in (self.standard, self.manifest_path, self.receipt_path, directory / "PROJECTION.json", directory / "PAYLOAD_FILES.txt"):
            self.assertIn(path.relative_to(self.repo).as_posix(), names)
        self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.assertFalse(result["NPZ_arrays_deserialized"])

    def assert_kept_failure(self):
        directory = self.project()
        result = json.loads((directory / "PROJECTION.json").read_text())
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["excluded_duplicate_count"], 0)
        self.assertTrue(result["failures"])
        self.assertIn(self.original.relative_to(self.repo).as_posix(), (directory / "PAYLOAD_FILES.txt").read_text().splitlines())

    def test_numeric_string_change_even_when_float_equal_keeps_original(self):
        self.write_stats(self.standard, self.alias, sse="0.250")
        self.assert_kept_failure()

    def test_count_mask_coverage_change_keeps_original(self):
        self.write_stats(self.standard, self.alias, count="0")
        self.assert_kept_failure()

    def test_wrong_alias_or_missing_standard_does_not_use_suffix_guess(self):
        self.edit_json(self.manifest_path, lambda m: m["metadata"].update(output_label="other"))
        self.assert_kept_failure()

    def test_array_fingerprint_tampering_is_rejected(self):
        self.edit_json(self.manifest_path, lambda m: m["arrays"]["origins"].update(sha256="changed"))
        self.assert_kept_failure()

    def test_archive_tampering_is_rejected_without_npz_deserialization(self):
        self.archive.write_bytes(b"changed")
        self.assert_kept_failure()

    def test_invalid_preseal_and_parent_evidence_are_kept(self):
        self.edit_json(self.pre_path, lambda m: m.update(config_sha="different"))
        self.assert_kept_failure()

    def test_missing_counterpart_is_kept_and_wrong_receipt_contract_is_kept(self):
        self.manifest_path.unlink()
        self.assert_kept_failure()

    def test_wrong_receipt_contract_is_kept(self):
        self.edit_json(self.receipt_path, lambda m: m["contract"].update(config_sha="different"))
        self.assert_kept_failure()

    def test_changed_model_binding_and_wrong_run_id_do_not_match(self):
        self.write_stats(self.standard, "other_alias")
        self.assert_kept_failure()

    def test_sources_or_actual_prediction_evidence_cannot_be_explicitly_excluded(self):
        with self.assertRaisesRegex(ValueError, "Extra excludes"):
            self.project(extra_excludes=["predictions"])

    def test_unreferenced_snapshots_require_explicit_exclusion_and_outputs_are_immutable(self):
        old = self.public / "figures" / "snapshot_old" / "README_KO.md"
        old.parent.mkdir(parents=True); old.write_text("old")
        first = self.project()
        names = (first / "PAYLOAD_FILES.txt").read_text()
        self.assertIn(old.relative_to(self.repo).as_posix(), names)
        before = {p: p.read_bytes() for p in first.iterdir() if p.is_file()}
        second = self.project(extra_excludes=["figures/snapshot_old"])
        self.assertNotIn(old.relative_to(self.repo).as_posix(), (second / "PAYLOAD_FILES.txt").read_text().splitlines())
        self.assertEqual(before, {p: p.read_bytes() for p in before})


if __name__ == "__main__":
    unittest.main()
