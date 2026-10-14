from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest

import numpy as np


MODULE_PATH = Path(__file__).with_name("audit_matched_readout_v2.py")
SPEC = importlib.util.spec_from_file_location("audit_matched_readout_v2", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(audit)


def observed_fixture() -> np.ndarray:
    observed = np.zeros((5, 3, 2), dtype=np.bool_)
    observed[[0, 1, 2], 0, 0] = True
    observed[[0, 1], 1, 0] = True
    observed[[0, 1, 2, 3], 0, 1] = True
    observed[[0], 1, 1] = True
    observed[[2, 3], 2, 1] = True
    return observed


def coefficient_fixture(observed: np.ndarray, feature_count: int = 4) -> dict[str, np.ndarray]:
    counts = observed.sum(axis=0).astype(np.int64)
    channel_counts = observed.sum(axis=(0, 1)).astype(np.int64)
    denominators = np.tile(channel_counts, observed.shape[1]).reshape(observed.shape[1:]).astype(np.float64)
    ranks = np.where(counts > 0, np.minimum(counts - 1, feature_count), 0).astype(np.int64)
    return {
        "coefficients": np.zeros((feature_count,) + observed.shape[1:], dtype=np.float64),
        "intercept": np.zeros(observed.shape[1:], dtype=np.float64),
        "observed_counts": counts,
        "channel_counts": channel_counts,
        "normal_equation_denominators": denominators,
        "design_ranks": ranks,
        "regularized_gram_conditions": np.where(counts > 0, 1.0, np.nan).astype(np.float64),
    }


def check(arrays: dict[str, np.ndarray], observed: np.ndarray) -> list[str]:
    failures: list[str] = []
    audit.check_solution_counts("RAW33C", arrays, observed, {}, failures)
    return failures


class MatchedReadoutV2CountTests(unittest.TestCase):
    def test_accepts_h_by_c_count_and_rank_shapes(self) -> None:
        observed = observed_fixture()
        arrays = coefficient_fixture(observed)

        self.assertEqual(arrays["observed_counts"].shape, (3, 2))
        self.assertEqual(arrays["normal_equation_denominators"].dtype, np.float64)
        self.assertEqual(check(arrays, observed), [])

    def test_rejects_flattened_count_and_rank_shapes(self) -> None:
        observed = observed_fixture()
        arrays = coefficient_fixture(observed)
        arrays["observed_counts"] = arrays["observed_counts"].reshape(-1)
        arrays["normal_equation_denominators"] = arrays["normal_equation_denominators"].reshape(-1)
        arrays["design_ranks"] = arrays["design_ranks"].reshape(-1)

        failures = check(arrays, observed)

        self.assertIn("COUNT_ARRAY_MISMATCH:RAW33C:observed_counts", failures)
        self.assertIn("COUNT_ARRAY_MISMATCH:RAW33C:normal_equation_denominators", failures)
        self.assertIn("RANK_ARRAY_SCHEMA:RAW33C", failures)

    def test_denominators_use_channel_counts_and_zero_outputs_are_valid(self) -> None:
        observed = observed_fixture()
        arrays = coefficient_fixture(observed)
        counts = arrays["observed_counts"]

        self.assertEqual(int(counts[2, 0]), 0)
        self.assertNotEqual(float(arrays["normal_equation_denominators"][0, 0]), float(counts[0, 0]))
        self.assertEqual(check(arrays, observed), [])

        wrong = copy.deepcopy(arrays)
        wrong["normal_equation_denominators"] = counts.astype(np.float64)
        failures = check(wrong, observed)

        self.assertIn("COUNT_ARRAY_MISMATCH:RAW33C:normal_equation_denominators", failures)


if __name__ == "__main__":
    unittest.main()
