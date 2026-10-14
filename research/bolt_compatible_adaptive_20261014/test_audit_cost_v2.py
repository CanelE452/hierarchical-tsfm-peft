"""Regression for actual checked-file records that omit NPZ array metadata."""
import unittest

import audit_cost_v2 as audit


class CanonicalNPZBindingFixtures(unittest.TestCase):
    def setUp(self):
        self.canonical = {"path": "research/bolt_compatible_adaptive_20261014/fixture.npz", "sha256": "a" * 64,
                          "arrays": {"history": {"shape": [1, 512, 1], "dtype": "float32", "sha256": "b" * 64}}}
        self.checked = {"path": str(audit.resolve(self.canonical["path"])), "sha256": "a" * 64,
                        "verified_sha256": "a" * 64, "role": "actual_physical_bridge"}

    def test_file_only_actual_record_binds_to_original_array_manifest(self):
        self.assertNotIn("arrays", self.checked)
        self.assertIs(audit.bind_canonical_npz(self.checked, self.canonical, "physical"), self.canonical)

    def test_same_sha_different_path_or_same_path_different_sha_is_rejected(self):
        wrong_path = {**self.checked, "path": "research/bolt_compatible_adaptive_20261014/other.npz"}
        wrong_sha = {**self.checked, "sha256": "c" * 64}
        for record in (wrong_path, wrong_sha):
            with self.assertRaisesRegex(audit.AuditFailure, "BINDING"):
                audit.bind_canonical_npz(record, self.canonical, "physical")

    def test_file_hash_alone_cannot_replace_the_original_array_manifest(self):
        with self.assertRaisesRegex(audit.AuditFailure, "BINDING"):
            audit.bind_canonical_npz(self.checked, self.checked, "physical")


if __name__ == "__main__":
    unittest.main()
