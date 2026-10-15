"""Reproduce the preserved metadata serialization failure and minimal repair."""
import json
from pathlib import Path
import unittest

import numpy as np

import audit_fit_target_comparison_v1 as original
import audit_fit_target_comparison_v2 as repaired
from test_audit_fit_target_comparison_v1 import analytic_fit

original.initialize_numpy()
repaired.initialize_numpy()


class MetadataSerializationRepair(unittest.TestCase):
    def test_preserved_original_failure_and_fresh_scalar_repair(self):
        fit, state, metadata = analytic_fit()
        before = original.check_projection_metadata(metadata, state, fit)
        with self.assertRaisesRegex(TypeError, "JSON serializable"):
            json.dumps(before, allow_nan=False)
        after = repaired.check_projection_metadata(metadata, state, fit)
        restored = json.loads(json.dumps(after, allow_nan=False))
        self.assertIs(type(restored["retained_topK_subspace_unique"]), bool)
        self.assertEqual(before, after)
        self.assertEqual(after, restored)

    def test_executed_v1_source_preserved_with_exactly_one_scalar_adapter(self):
        first = Path(original.__file__).read_text(encoding="utf-8")
        second = Path(repaired.__file__).read_text(encoding="utf-8")
        old = 'boundary_unique = k == target.shape[-1] or eigenvalues[k - 1] - eigenvalues[k] > numerical_bound'
        new = 'boundary_unique = bool(k == target.shape[-1] or eigenvalues[k - 1] - eigenvalues[k] > numerical_bound)'
        self.assertEqual(first.count(old), 1)
        self.assertEqual(first.replace(old, new, 1), second)

    def test_complete_output_subspace_metadata_serializes_without_numpy_scalars(self):
        fit, state, _ = analytic_fit()
        values = {**fit, "std": np.array([1., 10.], np.float32), "raw_forecast": fit["target"].copy()}
        history_latent = repaired.encode(values["history"], state["center"], state["encoder"])
        future = repaired.encode(values["target"], state["center"], state["encoder"])
        prediction = repaired.anchor_prediction(values["history"], values["raw_last"], future,
                                                state["center"], state["encoder"], state["decoder"])
        cached = {name: {"prediction": prediction, "latent_future": future, "latent_last": history_latent[:, -1], "origins": values["origins"]}
                  for name in ("BOLT", "NATIVE_s92601", "NATIVE_s92602", "E_Y")}
        result = repaired.output_subspace_cell(values, state, cached)
        self.assertEqual(result, json.loads(json.dumps(result, allow_nan=False)))


if __name__ == "__main__":
    unittest.main()
