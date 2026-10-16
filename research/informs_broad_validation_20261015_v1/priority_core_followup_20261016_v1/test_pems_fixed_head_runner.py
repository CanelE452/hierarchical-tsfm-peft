"""Synthetic boundary/gradient tests; no PeMS assets, checkpoints or cost runs."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("pems_head_tested",Path(__file__).with_name("pems_fixed_head_runner.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)
PRIMITIVE_SPEC = importlib.util.spec_from_file_location("pems_test_primitive",ROOT/runner.PRIOR/"safe_jena_prefix.py")
primitive = importlib.util.module_from_spec(PRIMITIVE_SPEC)
PRIMITIVE_SPEC.loader.exec_module(primitive)


def npy_parts(shape=(6,2,3),order=False,descr="<f8"):
    header = repr(dict(descr=descr,fortran_order=order,shape=shape)).encode("latin1")+b"\n"
    start = b"\x93NUMPY\x01\x00"+struct.pack("<H",len(header))+header
    values = [float(index) for index in range(36)]
    return start,b"".join(struct.pack("<d",value) for value in values)


class PrefixTests(unittest.TestCase):
    def test_prefix_never_converts_or_reads_protected_tail(self):
        start,payload = npy_parts()
        permitted = start+payload[:3*2*3*8]
        class PrefixOnly(io.BytesIO):
            def read(self,count=-1):
                if count < 0 or self.tell()+count > len(permitted):
                    raise AssertionError("Protected tail requested")
                return super().read(count)
        actual,evidence = runner.prefix_payload(PrefixOnly(permitted),len(start)+len(payload),primitive,rows=3,expected_shape=(6,2,3))
        self.assertEqual(actual.shape,(3,2,3))
        self.assertEqual(actual[:,:,0].tolist(),[[0.,3.],[6.,9.],[12.,15.]])
        self.assertEqual(evidence["tail_numeric_values_parsed"],0)
        self.assertEqual(evidence["payload_bytes_left_unread"],144)

    def test_f_order_rejected_before_numeric_decoder(self):
        start,payload = npy_parts(order=True)
        with mock.patch.object(primitive,"decode_numeric",side_effect=AssertionError("decode")):
            with self.assertRaises(ValueError):
                runner.prefix_payload(io.BytesIO(start+payload),len(start)+len(payload),primitive,rows=3,expected_shape=(6,2,3))

    def test_size_or_prefix_tampering_rejected(self):
        start,payload = npy_parts()
        for rows,size in ((7,len(start)+len(payload)),(True,len(start)+len(payload)),(3,len(start)+len(payload)+8)):
            with self.subTest(rows=rows,size=size),self.assertRaises(ValueError):
                runner.prefix_payload(io.BytesIO(start+payload),size,primitive,rows=rows,expected_shape=(6,2,3))

    def test_truncated_prefix_rejected(self):
        start,payload = npy_parts()
        with self.assertRaises(ValueError):
            runner.prefix_payload(io.BytesIO(start+payload[:4]),len(start)+len(payload),primitive,rows=3,expected_shape=(6,2,3))


class FixedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        sys.path.insert(0,str(ROOT))
        import networks
        cls.torch,cls.np,cls.networks = torch,np,networks
        torch.set_num_threads(2)

    def model(self):
        torch,np = self.torch,self.np
        torch.manual_seed(100)
        basis = np.eye(runner.C,dtype=np.float32)[:,:runner.K]
        adapter = self.networks.ForecastAdapter("B2",runner.L,runner.H,runner.C,runner.K,np.zeros(runner.C,np.float32),basis,"pointwise",None)
        model = runner.fixed_model(adapter,torch)
        return model,dict(model={key:value.detach().clone() for key,value in adapter.state_dict().items()})

    def test_persistence_head_is_original_channel_current_anchor(self):
        model,parent = self.model()
        x = self.torch.randn(2,runner.L,runner.C)
        self.assertTrue(self.torch.allclose(model(x),x[:,-1:].expand(-1,runner.H,-1)))
        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),6156)
        runner.verify_frozen(model,parent,self.torch)

    def test_head_gradient_passes_frozen_decoder_and_full_history_matters(self):
        model,parent = self.model()
        model.train()
        self.assertFalse(model.adapter.training)
        x = self.torch.randn(2,runner.L,runner.C)
        optimizer = self.torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=0.)
        (model(x)-self.torch.zeros(2,runner.H,runner.C)).square().mean().backward()
        self.assertGreater(float(model.head.weight.grad.abs().sum()),0.)
        self.assertGreater(float(model.head.weight.grad[:,:100].abs().sum()),0.)
        optimizer.step()
        runner.verify_frozen(model,parent,self.torch)
        changed = x.clone();changed[:,:100,:] += 2.
        self.assertGreater(float((model(changed)-model(x)).detach().abs().max()),0.)

    def test_exact_parent_mutation_and_foundation_rejected(self):
        model,parent = self.model()
        with self.torch.no_grad():
            next(model.adapter.parameters()).add_(.01)
        with self.assertRaises(ValueError):
            runner.verify_frozen(model,parent,self.torch)
        with self.assertRaises(ValueError):
            model(self.torch.zeros(1,runner.L,runner.C),object(),runner.K)


class GateTests(unittest.TestCase):
    def test_parent_receipt_uses_parent_identity_and_actual_foundation_counts(self):
        spec = dict(run_id="head_id",parent=dict(run_id="parent_id"))
        for mode,result,streams in (("parent_replay",dict(actual_F_streams=258,foundation=dict(foundation_instances_created=1)),258),
                ("parent_cost",dict(actual_F_streams=17200,foundation_instances_created=1),17200)):
            with self.subTest(mode=mode):
                actual = runner.operation_receipt_fields(spec,mode,result)
                self.assertEqual(actual,dict(role="PARENT_B2",run_id="parent_id",actual_F_streams=streams,foundation_instances_created=1))
                pending = runner.operation_receipt_fields(spec,mode)
                self.assertEqual((pending["role"],pending["run_id"]),("PARENT_B2","parent_id"))
                self.assertIsNone(pending["actual_F_streams"])

    def test_head_receipt_identity_and_zero_foundation_are_unchanged(self):
        spec = dict(run_id="head_id",parent=dict(run_id="parent_id"))
        for mode in ("fit","rescore","replay","cost"):
            with self.subTest(mode=mode):
                self.assertEqual(runner.operation_receipt_fields(spec,mode),dict(role=runner.ROLE,run_id="head_id",actual_F_streams=0,foundation_instances_created=0))

    def test_parent_foundation_rejects_checkpointing_or_wrong_row_size(self):
        import torch
        model = torch.nn.Linear(1,1).requires_grad_(False)
        foundation = types.SimpleNamespace(device="cpu",horizon=12,row_microbatch=128,checkpoint_inputs=False,
            initial_hash="exact",instance_norm_source_sha256="norm",model=model,model_revision="revision",
            assert_frozen_unchanged=mock.Mock())
        route = types.SimpleNamespace(pool=types.SimpleNamespace(foundation=foundation,instances_created=1))
        spec = dict(parent_cpu=dict(foundation_tensor_sha256="exact",instance_norm_source_sha256="norm"))
        ctx = dict(fit=types.SimpleNamespace(_model_sha=lambda f:"exact"))
        receipt = runner.parent_foundation(route,spec,ctx)
        self.assertTrue(receipt["backbone_parameters_frozen"])
        for field,value in (("checkpoint_inputs",True),("row_microbatch",None),("initial_hash","different")):
            old = getattr(foundation,field);setattr(foundation,field,value)
            with self.subTest(field=field),self.assertRaises(ValueError):
                runner.parent_foundation(route,spec,ctx)
            setattr(foundation,field,old)

    def test_parent_cost_gate_rejects_failed_replay_before_model_work(self):
        with tempfile.TemporaryDirectory() as folder:
            public = Path(folder);namespace = public/runner.NAME;namespace.mkdir()
            path = namespace/"PARENT_REPLAY_RECEIPT.json"
            path.write_text(json.dumps(dict(status="FAIL_PARENT_CPU_REPLAY_PARITY")),encoding="utf-8")
            with self.assertRaises(ValueError):
                runner.parent_replay_gate(path,runner.sha(path),public,{}, {})

    def test_direct_canonical_route_matches_original_preflight_metadata(self):
        helper = runner.module("pems_contract_test_helper",ROOT/runner.PRIOR/"priority_runner.py")
        parent = dict(run_id="primary_pems08_E1_B2_k43_s92601__e1",artifact_cache_relative_path="runs/primary_pems08_E1_B2_k43_s92601__e1/selected_bundle.pt",
            artifact_sha256="0d9a5c6b40c5ca6159e5183b9089aada2d537475f67270b79a3aa7cb784e6161",model_state_sha="c3af63f084da0092d377a5ff45dd6d323bc2421fa5189c3cf50dc1ec006a2e3f",
            scaler_sha256="94e8c71c42f8d58a1d258fb200d86a24595fe1793203da1543b3e47a60569c60",fit_receipt_public_relative_path="runs/primary_pems08_E1_B2_k43_s92601__e1/BASE_FIT_RECEIPT.json",
            fit_receipt_sha256="f2cb19bd749371d15b3965048a6cc44be858a8e16f22dfd5236191c7e3a7933a",V_manifest_public_relative_path="predictions/primary_pems08_E1_B2_k43_s92601__e1/V_select/PREDICTION_MANIFEST.json",
            V_manifest_sha256="a0504b361b1f60c8521d926a9fff3bd11b91d72801093553df9b85966a7d5081")
        reference = dict(model_sha="1a93a4026548db037388dc126c434ab9ad594aa42771eaf7b5601a4068004150",archive_sha256="0c7373547e5f7377818a8463825a8fd38ae7743c8947d0e7b915b9b5b5b0e8b0")
        spec = dict(parent=parent,parent_cpu=dict(foundation_tensor_sha256="c2b4edf7db22d3f18e58ca79299a5c7e864e9b8f277e4e29e2f88d07dbb35874"))
        route = runner.canonical_parent_route(spec,reference,helper)
        self.assertEqual(helper.fingerprint(route),"5446dd3880ede53ad87aa7d983faf5262d8f0f29b54eb5cf2d4b58a52ee7ec5a")
        self.assertEqual(route["canonical_fit_end"],10713)
        self.assertEqual(route["expected_prediction_model_sha"],reference["model_sha"])

    def test_incomplete_fit_cannot_enter_accuracy_array_read(self):
        receipt = dict(status="BUDGET_LIMIT_PRESERVED")
        with mock.patch.object(runner,"read",side_effect=AssertionError("read")),self.assertRaises(ValueError):
            runner.load_prediction(None,None,None,None,None,receipt)

    def test_default_has_zero_input_operations(self):
        with mock.patch.object(runner,"read",side_effect=AssertionError("read")),mock.patch.object(runner,"sha",side_effect=AssertionError("sha")),mock.patch.object(runner,"execute",side_effect=AssertionError("execute")),contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(runner.main([]),0)
        receipt = json.loads(output.getvalue())
        self.assertEqual(receipt["input_reads"],0)
        self.assertEqual(receipt["actual_Task_checkpoint_forward_training_cost_or_child_operations"],0)

    def test_unsealed_draft_rejected_without_helper_import_or_models(self):
        with tempfile.TemporaryDirectory() as folder:
            public = Path(folder);namespace = public/runner.NAME;namespace.mkdir()
            path = namespace/"SPEC_DRAFT.json"
            path.write_text(json.dumps(dict(schema="PRIORITY_PEMS_FIXED_HEAD_SPEC_v1",status="DRAFT_NOT_EXECUTABLE")),encoding="utf-8")
            with mock.patch.object(runner,"module",side_effect=AssertionError("import")),self.assertRaises(ValueError):
                runner.metadata(public,public,path,runner.sha(path),None)


if __name__ == "__main__":
    unittest.main()
