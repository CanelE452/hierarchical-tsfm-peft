"""Regression for the actual Beijing Phase0 origin-summary schema."""
import hashlib
import unittest

import numpy as np

import audit_readout_v2 as audit


class ActualOriginSchemaFixtures(unittest.TestCase):
    def test_legacy_exact_vector_and_new_data_summary_validate_same_vector(self):
        origins = np.array([512, 592, 672], dtype=np.int64)
        summary = {"count": 3, "first": 512, "last": 672,
                   "sha256": hashlib.sha256(origins.tobytes()).hexdigest()}
        self.assertEqual(audit.verify_locked_origins(origins, origins.tolist()), "EXACT_VECTOR")
        self.assertEqual(audit.verify_locked_origins(origins, summary), "EXACT_RAW_INT64_BYTES_DIGEST")

    def test_same_endpoints_and_count_do_not_allow_changed_middle_origin(self):
        origins = np.array([512, 592, 672], dtype=np.int64)
        summary = {"count": 3, "first": 512, "last": 672,
                   "sha256": hashlib.sha256(origins.tobytes()).hexdigest()}
        changed = np.array([512, 593, 672], dtype=np.int64)
        with self.assertRaisesRegex(audit.AuditFailure, "CONTENT_DIGEST"):
            audit.verify_locked_origins(changed, summary)


if __name__ == "__main__":
    unittest.main()
