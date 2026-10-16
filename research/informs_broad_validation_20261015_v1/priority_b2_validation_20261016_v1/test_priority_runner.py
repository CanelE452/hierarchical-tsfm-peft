import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("priority_runner_under_test", Path(__file__).with_name("priority_runner.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class NoSuchProcess(Exception):
    pass


class FakeProcesses:
    NoSuchProcess = NoSuchProcess

    def __init__(self, births):
        self.births = births

    def Process(self, pid):
        if pid not in self.births:
            raise NoSuchProcess()
        return types.SimpleNamespace(create_time=lambda: self.births[pid])


def toy_saved_arrays():
    import numpy as np
    # Channel0 has three zero errors, channel1 has one error3: macro MSE4.5, pooled2.25.
    prediction = np.asarray([[[0., 3.], [0., 90.], [0., 90.]]], dtype=np.float32)
    target = np.zeros_like(prediction)
    observed = np.asarray([[[True, True], [True, False], [True, False]]])
    arrays = dict(pred_normalized=prediction, target_normalized=target, observed=observed,
                  origins=np.asarray([20]), channel_ids=np.asarray(["a", "b"]))
    statistics = []
    for channel, name in enumerate(("a", "b")):
        for start, stop in ((0, 1), (1, 2), (2, 3)):
            count = int(observed[0, start, channel])
            error = float(prediction[0, start, channel]) if count else 0.
            statistics.append(dict(run_id="toy", split="V_select", cohort="toy_V", origin="20", channel=name,
                horizon_start=str(start), horizon_end=str(stop), sse=str(error * error), sae=str(abs(error)), count=str(count)))
    manifest = dict(run_id="toy", split="V_select", cohort="toy_V", score=dict(mse=4.5, mae=1.5, n_observed=4))
    score = dict(mse="4.5", mae="1.5")
    return arrays, manifest, statistics, score, np


class PriorityTests(unittest.TestCase):
    def test_default_never_reads_plan_or_imports_scientific_context(self):
        with mock.patch.object(runner, "read", side_effect=AssertionError("read")), mock.patch.object(runner, "sha", side_effect=AssertionError("hash")), mock.patch.object(runner, "load_context", side_effect=AssertionError("science")), contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(runner.main([]), 0)
        self.assertEqual(json.loads(stream.getvalue())["input_reads"], 0)

    def test_inside_rejects_escape_and_preserves_explicit_relative_path(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.assertEqual(runner.inside(root, "a/b"), root.resolve() / "a/b")
            with self.assertRaises(ValueError):
                runner.inside(root, "../escape")

    def test_owned_old_identity_must_be_dead_even_if_pid_is_reused(self):
        identities = [dict(pid=10, create_time=3.)]
        runner.assert_old_identities_dead(identities, FakeProcesses({}))
        runner.assert_old_identities_dead(identities, FakeProcesses({10: 8.}))
        with self.assertRaises(RuntimeError):
            runner.assert_old_identities_dead(identities, FakeProcesses({10: 3.}))

    def test_process_access_failure_is_not_treated_as_dead(self):
        process = types.SimpleNamespace(NoSuchProcess=NoSuchProcess, Process=mock.Mock(side_effect=PermissionError("denied")))
        with self.assertRaises(PermissionError):
            runner.assert_old_identities_dead([dict(pid=10, create_time=3.)], process)

    def test_independent_rescore_is_channel_macro_and_masks_unobserved_errors(self):
        arrays, manifest, statistics, score, np = toy_saved_arrays()
        result = runner.independent_rescore(arrays, manifest, statistics, score, np)
        self.assertEqual(result["mse"], 4.5)
        self.assertEqual(result["mae"], 1.5)
        self.assertEqual(result["public_stats_rows"], 6)
        self.assertNotEqual(result["mse"], 2.25)

    def test_independent_rescore_rejects_missing_duplicate_and_tampered_statistics(self):
        arrays, manifest, statistics, score, np = toy_saved_arrays()
        changed = copy.deepcopy(statistics)
        changed[0]["sse"] = "1"
        for rows in (statistics[:-1], statistics + [statistics[0]], changed):
            with self.assertRaises(ValueError):
                runner.independent_rescore(arrays, manifest, rows, score, np)

    def test_rescore_rejects_foreign_identity_and_csv_disagreement(self):
        arrays, manifest, statistics, score, np = toy_saved_arrays()
        changed = copy.deepcopy(statistics)
        changed[0]["split"] = "T_final"
        with self.assertRaises(ValueError):
            runner.independent_rescore(arrays, manifest, changed, score, np)
        with self.assertRaises(ValueError):
            runner.independent_rescore(arrays, manifest, statistics, dict(score, mse="9"), np)

    def test_unobserved_channel_is_excluded_from_macro(self):
        arrays, manifest, statistics, score, np = toy_saved_arrays()
        arrays["observed"][:, :, 1] = False
        for row in statistics:
            if row["channel"] == "b":
                row.update(sse="0", sae="0", count="0")
        manifest["score"] = dict(mse=0., mae=0., n_observed=3)
        result = runner.independent_rescore(arrays, manifest, statistics, dict(mse="0", mae="0"), np)
        self.assertEqual(result["mse"], 0.)

    def test_wrapper_rejects_foundation_and_nonregistered_K(self):
        import torch
        direct = torch.nn.Linear(2, 1)
        wrapped = runner.forward_wrapper(direct, torch)
        x = torch.ones(1, 2)
        self.assertEqual(tuple(wrapped(x, None, 6).shape), (1, 1))
        with self.assertRaises(ValueError):
            wrapped(x, object(), 6)
        with self.assertRaises(ValueError):
            wrapped(x, None, 21)

    def test_replay_indices_are_six_unique_and_include_first_middle_last(self):
        import numpy as np
        indices = runner.replay_indices(43, np)
        self.assertEqual(len(set(indices.tolist())), 6)
        self.assertTrue({0, 21, 42}.issubset(set(indices.tolist())))
        self.assertEqual(indices.tolist(), runner.replay_indices(43, np).tolist())

    def test_artifact_write_is_fresh_atomic_and_refuses_overwrite(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bundle.pt"
            runner.save_bundle(path, dict(toy=torch.tensor([7.])), torch)
            original = path.read_bytes()
            self.assertEqual(list(path.parent.glob("*.tmp_*")), [])
            with self.assertRaises(FileExistsError):
                runner.save_bundle(path, dict(toy=torch.tensor([9.])), torch)
            self.assertEqual(path.read_bytes(), original)

    def test_cohort_validation_checks_mask_coordinates_and_channel_order(self):
        import numpy as np
        import hashlib
        def array_fp(value):
            value = np.ascontiguousarray(value)
            return dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(value.tobytes()).hexdigest())
        arrays = dict(target_normalized=np.zeros((1, 2, 2), np.float32), observed=np.ones((1, 2, 2), bool),
                      origins=np.asarray([4]), channel_ids=np.asarray(["a", "b"]))
        reference = dict(arrays={name: array_fp(value) for name, value in arrays.items()})
        runner.validate_cohort(arrays, reference, types.SimpleNamespace(array_fingerprint=array_fp))
        arrays["channel_ids"] = arrays["channel_ids"][::-1]
        with self.assertRaises(ValueError):
            runner.validate_cohort(arrays, reference, types.SimpleNamespace(array_fingerprint=array_fp))

    def test_safe_prefix_uses_immutable_full_contract_and_preserves_original_bounds(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            public = Path(folder)
            original = dict(shape=[20965, 21], metadata=dict(ancestry="toy"))
            (public / "DATA_CONTRACT.json").write_text(json.dumps(dict(assets=dict(jena=original))), encoding="utf-8")
            contract = dict(columns=[str(x) for x in range(21)], data_sha256="old", array_sha256={"old": "inherited"},
                primary_split_ranges=dict(A_fit=[0, 10482], A_stop=[10482, 12579], V_select=[14675, 16772]),
                A_fit_statistics_fingerprints=dict(mean="m", std="s", count="c"), search_scaler_sha256="scale", refit_scaler_sha256="scale")
            task = dict(physical=np.empty((16772, 21)), observed=np.empty((16772, 21), bool), L=512, H=48,
                        columns=contract["columns"], data_sha256="old", array_sha256=contract["array_sha256"])
            loader = types.SimpleNamespace(load_jena_prefix=mock.Mock(return_value=(task, dict(new_T_numeric_values_parsed=0))))
            mean, std, count = object(), object(), object()
            data = types.SimpleNamespace(fit_statistics=mock.Mock(return_value=(mean, std, count)),
                array_sha256=lambda value: {id(mean): "m", id(std): "s", id(count): "c"}[id(value)])
            fit = types.SimpleNamespace(scaler_fingerprint=lambda a, b: "scale")
            with mock.patch.object(runner, "module", return_value=loader):
                returned, bounds, receipt = runner.load_task_prefix("toy_assets", dict(task_contract=contract), dict(data=data, fit=fit), public)
            self.assertIs(returned, task)
            self.assertEqual(bounds["A_stop"], (10482, 12579))
            self.assertEqual(loader.load_jena_prefix.call_args.args[1], original)
            self.assertEqual(receipt["new_T_numeric_values_parsed"], 0)

    def test_deadline_fails_before_operation_when_expired(self):
        with mock.patch.object(runner.time, "time", return_value=100.):
            with self.assertRaises(TimeoutError):
                runner.check_deadline(99.)


if __name__ == "__main__":
    unittest.main()
