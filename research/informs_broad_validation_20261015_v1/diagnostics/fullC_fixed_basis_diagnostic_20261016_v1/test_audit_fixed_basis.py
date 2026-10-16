import contextlib
import csv
import gzip
import importlib.util
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("fixed_basis_diagnostic", Path(__file__).with_name("audit_fixed_basis.py"))
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def stats_fixture(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=audit.FIELDS)
    writer.writeheader()
    for row in rows:
        writer.writerow(dict(run_id="r", split="V_select", cohort="cohort", origin=row[0], channel=row[1],
                             horizon_start=row[2], horizon_end=row[3], sse=row[4], sae=row[5], count=row[6]))
    return gzip.compress(stream.getvalue().encode())


class FixedBasisTests(unittest.TestCase):
    def test_metadata_default_has_no_input_reads_or_numpy_import(self):
        output = io.StringIO()
        with mock.patch.object(Path, "read_bytes", side_effect=AssertionError("read")), mock.patch.object(audit, "load_basis", side_effect=AssertionError("numpy")), contextlib.redirect_stdout(output):
            self.assertEqual(audit.main([]), 0)
        self.assertEqual(json.loads(output.getvalue())["input_reads"], 0)

    def test_explicit_input_sha_size_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "item"
            path.write_bytes(b"immutable")
            item = dict(path=str(path), label="toy", bytes=9, sha256=audit.digest(b"immutable"))
            self.assertEqual(audit.read_bound(item, 10), b"immutable")
            with self.assertRaises(ValueError):
                audit.read_bound(dict(item, sha256="0" * 64), 10)
            with self.assertRaises(ValueError):
                audit.read_bound(item, 8)
            self.assertEqual(path.read_bytes(), b"immutable")

    def test_macro_weights_channels_equally_but_cells_by_counts(self):
        # Channel a MSE=100/100=1; b MSE=9/1=9. Macro=5, pooled=109/101.
        rows = {(1, "a", 0, 1): (100., 50., 100), (1, "b", 0, 1): (9., 3., 1)}
        result = audit.macro(rows, list(rows))
        self.assertEqual(result["mse"], 5.)
        self.assertEqual(result["mae"], 1.75)
        self.assertNotAlmostEqual(result["mse"], 109 / 101)

    def test_missing_channels_do_not_become_zero_loss(self):
        rows = {(1, "missing", 0, 1): (0., 0., 0), (1, "seen", 0, 1): (8., 4., 2)}
        self.assertEqual(audit.macro(rows, list(rows))["mse"], 4.)
        self.assertEqual(audit.macro(rows, list(rows))["active_channels"], 1)

    def test_strict_pair_wins_use_observed_cells_and_count_match(self):
        keys = [(1, "a", 0, 1), (2, "a", 0, 1), (3, "a", 0, 1)]
        left = dict(zip(keys, [(1., 1., 1), (4., 2., 1), (0., 0., 0)]))
        right = dict(zip(keys, [(2., 2., 1), (4., 1., 1), (0., 0., 0)]))
        row = audit.comparisons({"L": left, "R": right}, [1, 2, 3], ["a"], 1)[0]
        self.assertEqual((row["sse_left_wins"], row["sse_ties"], row["paired_cells"]), (1, 1, 2))
        self.assertEqual((row["sae_left_wins"], row["sae_right_wins"]), (1, 1))
        right[keys[0]] = (2., 2., 0)
        with self.assertRaises(ValueError):
            audit.comparisons({"L": left, "R": right}, [1, 2, 3], ["a"], 1)

    def test_43_contiguous_blocks_are_15_14_14_and_exhaustive(self):
        blocks = [x for x in audit.groups(list(range(43)), ["x"], 48) if x[0] == "origin_block"]
        sets = [[o for o in range(43) if predicate((o, "x", 0, 12))] for _, _, predicate in blocks]
        self.assertEqual([len(x) for x in sets], [15, 14, 14])
        self.assertEqual(sum(sets, []), list(range(43)))
        self.assertEqual(audit.bands(49), [(0, 13), (13, 25), (25, 49)])

    def test_statistics_cartesian_completeness_and_invalid_rows(self):
        rows = [(5, "x", 0, 1, 1., 1., 1), (5, "x", 1, 2, 4., 2., 1), (5, "x", 2, 4, 9., 3., 1)]
        manifest = dict(run_id="r", split="V_select", cohort="cohort", score=dict(horizon=4, num_channels=1, n_origins=1))
        parsed, origins, channels = audit.parse_stats(stats_fixture(rows), manifest)
        self.assertEqual((origins, channels, len(parsed)), ([5], ["x"], 3))
        for bad in (rows[:-1], rows + [rows[0]], [rows[0][:-3] + (-1., 1., 1)] + rows[1:]):
            with self.assertRaises(ValueError):
                audit.parse_stats(stats_fixture(bad), manifest)
        with self.assertRaises(ValueError):
            audit.parse_stats(stats_fixture(rows), dict(manifest, split="T_final"))

    def test_invertible_nonorthogonal_basis_does_not_make_transpose_inverse(self):
        import numpy as np
        arrays = dict(basis=np.eye(11, dtype=np.float32), center=np.ones(11, dtype=np.float32),
                      mean=np.zeros(11, dtype=np.float32), std=np.ones(11, dtype=np.float32))
        arrays["basis"][0, 0] = 2.
        result = audit.basis_diagnostic(arrays, np)
        self.assertEqual(result["full_basis_rank"], 11)
        self.assertEqual(result["condition_number_2"], 2.)
        self.assertEqual(result["prefix_diagnostics"][-1]["projection_identity_max_abs"], 3.)
        self.assertIsNone(result["performance_pass_threshold"])

    def test_full_orthogonal_basis_can_change_componentwise_nonlinear_forecast(self):
        import numpy as np
        # Analytic counterexample: x=(1,0), 45-degree rotation and f(z)=z^2.
        q = np.array([[1., -1.], [1., 1.]]) / math.sqrt(2.)
        x = np.array([[1., 0.]])
        self.assertTrue(np.allclose((x @ q) @ q.T, x))
        transformed_forecast = ((x @ q) ** 2) @ q.T
        self.assertFalse(np.allclose(transformed_forecast, x ** 2))

    def test_basis_archive_rejects_unrelated_arrays_before_loading(self):
        import zipfile
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("target.npy", b"not allowed")
        with self.assertRaises(ValueError):
            audit.load_basis(stream.getvalue())


if __name__ == "__main__":
    unittest.main()
