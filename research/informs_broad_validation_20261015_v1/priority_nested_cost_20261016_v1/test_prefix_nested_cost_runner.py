"""Synthetic prefix, quiet-window and immutable route-binding tests only."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import struct
import types
import unittest

HERE = Path(__file__).resolve().parent
SOURCE = HERE/"prefix_nested_cost_runner.py"
runner = None
if SOURCE.is_file():
    spec = importlib.util.spec_from_file_location("nested_prefix_cost_tested",SOURCE)
    runner = importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


def primitive():
    path = HERE.parent/"priority_b2_validation_20261016_v1/safe_jena_prefix.py"
    spec = importlib.util.spec_from_file_location("nested_prefix_test_primitive",path)
    value = importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


class ToyArchive:
    def __init__(self,members):
        self.members = members
        self.streams = []

    def getinfo(self,name):
        return types.SimpleNamespace(filename=name,file_size=len(self.members[name]))

    def open(self,name):
        name = name if isinstance(name,str) else name.filename
        class Recorded(io.BytesIO):
            maximum = 0
            def read(self,count=-1):
                value = super().read(count)
                self.maximum = max(self.maximum,self.tell())
                return value
        stream = Recorded(self.members[name]);self.streams.append((name,stream))
        return stream


def toy_archive(order=False):
    import numpy as np
    physical = np.asarray([[1,2],[3,4],[5,6],[91,92],[93,94],[95,96]],np.float32)
    arrays = dict(physical=physical,mask=np.ones((6,2),bool),
        times=np.asarray(["2020-01-01T00:00:00","2020-01-01T01:00:00","2020-01-01T02:00:00",
            "2020-01-01T03:00:00","2020-01-01T04:00:00","2020-01-01T05:00:00"],dtype="<U19"),columns=np.asarray(["a","b"]))
    members,headers = {},{}
    for name,value in arrays.items():
        value = np.asfortranarray(value) if order and name == "physical" else value
        stream = io.BytesIO();np.lib.format.write_array(stream,value,allow_pickle=False)
        members[name+".npy"] = stream.getvalue()
        headers[name] = dict(descr=value.dtype.str,shape=list(value.shape),fortran_order=False)
    return ToyArchive(members),headers,np


class PrefixNestedTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(runner,"New prefix-only nested cost entry has not been implemented")

    def test_only_selected_rows_are_decoded_and_timestamp_tail_is_unread(self):
        archive,headers,np = toy_archive()
        values,evidence = runner.read_prefix_members(archive,primitive(),3,headers)
        self.assertEqual(values["physical"].tolist(),[[1.,2.],[3.,4.],[5.,6.]])
        self.assertEqual(values["times"].shape,(3,))
        self.assertTrue(all(row["tail_numeric_values_parsed"] == 0 for row in evidence))
        converted = {row["member"]:row["numeric_payload_bytes_converted"] for row in evidence}
        self.assertEqual(converted,{"physical.npy":24,"mask.npy":6,"times.npy":228,"columns.npy":8})
        for name,stream in archive.streams:
            if name != "columns.npy":
                self.assertLess(stream.maximum,len(archive.members[name]))

    def test_unexpected_order_or_dtype_is_rejected_before_prefix_decoding(self):
        archive,headers,np = toy_archive(order=True)
        with self.assertRaises(ValueError):
            runner.read_prefix_members(archive,primitive(),3,headers)
        archive,headers,np = toy_archive()
        headers["physical"]["descr"] = "<f8"
        with self.assertRaises(ValueError):
            runner.read_prefix_members(archive,primitive(),3,headers)

    def test_prefix_beyond_source_or_boolean_row_count_is_rejected(self):
        for rows in (7,-1,True):
            archive,headers,np = toy_archive()
            with self.subTest(rows=rows),self.assertRaises(ValueError):
                runner.read_prefix_members(archive,primitive(),rows,headers)

    def test_costs_require_fresh_root_quiet_proof_with_both_empty_lists(self):
        valid = dict(status="QUIET_CPU_COST_WINDOW",active_model_workers=[],active_cost_workers=[],captured_unix=1000.)
        self.assertIsNone(runner.require_quiet(valid,1001.))
        mutations = [dict(status="NOT_QUIET"),dict(active_model_workers=[dict(pid=12)]),
            dict(active_cost_workers=[dict(pid=13)]),dict(active_model_workers=None),dict(captured_unix=0.),dict(captured_unix=1002.)]
        for mutation in mutations:
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                runner.require_quiet(dict(valid,**mutation),1001.)

    def test_canonical_nested_route_preserves_original_preflight_fingerprint(self):
        condition = dict(run_id="postT_cpu_beijing_E4_NESTED_PCA_k3_s92601_v1",latent_k=3,kind="nested")
        plan = dict(foundation_tensor_sha256="c2b4edf7db22d3f18e58ca79299a5c7e864e9b8f277e4e29e2f88d07dbb35874")
        binding = dict(manifest_relative="ordered_predictions/postT_cpu_beijing_E4_NESTED_PCA_k3_s92601_v1/PREDICTION_MANIFEST.json",
            manifest_sha256="6ac55cb55cc78ff9dc0f51a059ba50a8de1249d3b8ac2733a6bc80402c1b7aab")
        actual = runner.canonical_spec(condition,plan,binding)
        self.assertEqual(runner.fingerprint(actual),"47ff961ec14bce8d2d875ee46e9cb1ce514a7c530c116684cbd28cd0a0d6cb80")
        self.assertEqual(actual["canonical_fit_end"],12622)
        self.assertEqual(actual["family"],"NESTED_PCA")

    def test_invalid_k_batch_or_non_nested_condition_is_rejected(self):
        valid = dict(run_id="postT_cpu_beijing_E4_NESTED_PCA_k3_s92601_v1",latent_k=3,kind="nested")
        for bad in (dict(valid,latent_k=2),dict(valid,kind="ordered"),dict(valid,run_id="../outside")):
            with self.subTest(bad=bad),self.assertRaises(ValueError):
                runner.require_condition(bad,1)
        for batch in (0,2,True):
            with self.subTest(batch=batch),self.assertRaises(ValueError):
                runner.require_condition(valid,batch)

    def test_actual_forward_cpu_closure_cannot_be_replaced_with_file_presence(self):
        receipt = dict(status="PASS",run_id="nested",actual_F_streams=18,F_physical_streams=18,F_calls=2,
            expected_F_streams=18,atol=1e-5,rtol=1e-4,max_abs=7.2e-7,T_final_accessed=False)
        proof = dict(status="PASS",run_id="nested",mode="replay",scientific_result=dict(receipt,executed_now=True),
            actual_checkpoint_forward_in_this_command=True,checkpoint_replay_call_attempted=True,checkpoint_replay_receipt_returned=True,
            cuda_initialized=False,new_T_target_or_scoring=False,original_source_registry_selection_T_and_captured_inputs_preserved=True,
            error_class=None,postflight_error_class=None,latent_k=3,plan_sha256=runner.OLD_PLAN_SHA,verifier_sha256="verifier")
        runner.require_cpu_replay(proof,receipt,3,"verifier")
        for mutation in (dict(actual_checkpoint_forward_in_this_command=False),dict(cuda_initialized=True),dict(status="RUNNING"),dict(scientific_result=dict(receipt,executed_now=False))):
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                runner.require_cpu_replay(dict(proof,**mutation),receipt,3,"verifier")

    def test_replay_receipt_must_match_the_actual_private_archive_bytes(self):
        self.assertTrue(hasattr(runner,"require_replay_archive"),"Replay archive byte binding has not been implemented")
        run = "postT_cpu_beijing_E4_NESTED_PCA_k3_s92601_v1"
        receipt = dict(cache_relative_path=f"route_replays/{run}/V_subset.npz",archive_sha256="actual")
        runner.require_replay_archive(receipt,run,"actual")
        for mutation in (dict(archive_sha256="different"),dict(cache_relative_path="route_replays/other/V_subset.npz")):
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                runner.require_replay_archive(dict(receipt,**mutation),run,"actual")

    def test_cost_trial_accounting_rejects_wrong_stream_count_or_missing_stage(self):
        rows = []
        for phase,count in (("warmup",10),("timed",20),("stage_probe",20)):
            for trial in range(count):
                rows.append(dict(phase=phase,trial=trial,batch_size=8,F_logical_streams=88,F_physical_streams=88,F_calls=1,
                    row_microbatch=128,stage_instrumentation=phase == "stage_probe",prediction_sha256="same"))
        runner.validate_trials(rows,11,8)
        bad = copy.deepcopy(rows);bad[0]["F_physical_streams"] = 87
        with self.assertRaises(ValueError):
            runner.validate_trials(bad,11,8)
        with self.assertRaises(ValueError):
            runner.validate_trials(rows[:-1],11,8)

    def test_default_cli_reports_zero_input_and_model_operations(self):
        with contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(runner.main([]),0)
        result = json.loads(stream.getvalue())
        self.assertEqual(result["input_reads"],0)
        self.assertEqual(result["actual_Task_checkpoint_forward_training_cost_or_child_operations"],0)


if __name__ == "__main__":
    unittest.main()
