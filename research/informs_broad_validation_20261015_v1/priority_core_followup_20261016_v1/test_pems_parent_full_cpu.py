"""Synthetic NumPy/stdlib tests; no real checkpoint, Task or model execution."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).with_name("pems_parent_full_cpu.py")
runner = None
if SOURCE.is_file():
    spec = importlib.util.spec_from_file_location("pems_parent_full_tested",SOURCE)
    runner = importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class ParentFullTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runner,"Same-checkpoint parent CPU148 diagnostic is not implemented")

    def test_default_has_no_input_or_model_operations(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(runner.main([]),0)
        value = json.loads(output.getvalue())
        self.assertEqual(value["input_reads"],0)
        self.assertEqual(value["actual_Task_checkpoint_forward_training_cost_or_child_operations"],0)

    def test_full148_has37_four_origin_batches_and_exact_F_accounting(self):
        batches = runner.batch_slices(148)
        self.assertEqual(len(batches),37)
        self.assertEqual(batches[0],(0,4));self.assertEqual(batches[-1],(144,148))
        self.assertEqual(sum(stop-start for start,stop in batches),148)
        runner.require_full_counters(dict(streams=6364,row_calls=74,physical_streams=6364,physical_calls=74))
        for count in (147,149,True):
            with self.subTest(count=count),self.assertRaises(ValueError):runner.batch_slices(count)
        with self.assertRaises(ValueError):
            runner.require_full_counters(dict(streams=6364,row_calls=73,physical_streams=6364,physical_calls=73))

    def test_GPU_comparison_uses_fixed_tolerances_and_keeps_failure_information(self):
        import numpy as np
        expected = np.zeros((148,12,170),np.float32);actual = expected.copy();actual[0,0,0] = 1e-6
        result = runner.compare_GPU(actual,expected,np)
        self.assertEqual(result["status"],"PASS_ALL148_GPU_CPU_PARITY")
        self.assertEqual((result["atol"],result["rtol"]),(1e-5,1e-4))
        actual[0,0,0] = 2e-5
        result = runner.compare_GPU(actual,expected,np)
        self.assertEqual(result["status"],"FAIL_ALL148_GPU_CPU_PARITY")
        self.assertEqual(result["outside_tolerance_values"],1)
        self.assertGreater(result["max_abs"],1e-5)
        with self.assertRaises(ValueError):runner.compare_GPU(actual[:-1],expected[:-1],np)
        actual[0,0,0] = float("nan")
        with self.assertRaises(ValueError):runner.compare_GPU(actual,expected,np)

    def test_prefix_horizon_and_metric_contract_cannot_change(self):
        plan = dict(axes=dict(L=512,H=12,C=170,K=43,V_origins=148,observed_targets=301920),materialized_prefix_end_exclusive=14284,
            microbatch=4,row_microbatch=128,checkpoint_inputs=False,deadline_unix=1792222550)
        runner.require_full_contract(plan)
        for mutation in (dict(materialized_prefix_end_exclusive=14285),dict(microbatch=6),dict(row_microbatch=64),dict(checkpoint_inputs=True)):
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):runner.require_full_contract(dict(plan,**mutation))

    def test_changed_old_input_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"pinned.json";path.write_text("sealed",encoding="utf-8")
            digest = runner.sha(path);runner.require_pin(path,digest)
            path.write_text("changed",encoding="utf-8")
            with self.assertRaises(ValueError):runner.require_pin(path,digest)

    def test_foreign_manifest_cannot_be_rescored_as_this_parent(self):
        identity = dict(run_id="cpu_diagnostic",split="V_select",cohort="pems08:primary:V_select:stride12",data_sha="data",scaler_sha="scaler",
            source_sha="source",config_sha="plan",model_sha="same_parent_cpu_variant")
        parent = dict(run_id="original_parent",artifact_sha256="artifact")
        manifest = dict(**identity,parent_run_id="original_parent",parent_artifact_sha256="artifact",device="cpu",inference_variant="same_checkpoint_CPU_full_V148",
            full_cpu_plan_sha256="plan",full_CPU_inference_completed=True)
        runner.require_cpu_manifest(manifest,identity,parent,"plan")
        for mutation in (dict(parent_run_id="foreign"),dict(run_id="other"),dict(device="cuda"),dict(full_CPU_inference_completed=False)):
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):runner.require_cpu_manifest(dict(manifest,**mutation),identity,parent,"plan")

    def test_inference_will_not_overwrite_a_completed_prediction_or_diagnostic(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);public=root/"public";cache=root/"cache";public.mkdir();cache.mkdir()
            namespace=public/runner.NAME;namespace.mkdir();output=namespace/"diagnostics"/"fresh"
            paths = runner.output_paths(public,cache,"diagnostic",output,"inference")
            paths["public_prediction"].mkdir(parents=True)
            with self.assertRaises(FileExistsError):runner.output_paths(public,cache,"diagnostic",output,"inference")
            output.mkdir(parents=True)
            with self.assertRaises(FileExistsError):runner.output_paths(public,cache,"diagnostic",output,"rescore")
            with self.assertRaises(ValueError):runner.output_paths(public,cache,"../foreign",namespace/"another","inference")


if __name__ == "__main__":
    unittest.main()
