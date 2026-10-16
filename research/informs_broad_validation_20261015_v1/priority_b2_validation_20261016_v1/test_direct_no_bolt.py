"""Synthetic tensors only: no Task, selected checkpoint, Bolt or real fit."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent


def load():
    spec=importlib.util.spec_from_file_location("tested_direct_no_bolt",HERE/"direct_no_bolt.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class DirectNoBoltTests(unittest.TestCase):
    def setUp(self):self.api=load()

    def fixture(self):
        self.api.model_class()
        import torch
        torch.set_num_threads(2)
        if torch.cuda.is_initialized():raise RuntimeError("CPU synthetic scope required")
        center=torch.zeros(21);basis=torch.eye(21)[:,:6]
        model=self.api.fresh_model(center,basis)
        return torch,model

    def test_default_reads_no_inputs_and_imports_no_new_science(self):
        before={name:name in sys.modules for name in ("torch","numpy","data_contract","fit_base")}
        with mock.patch.object(self.api,"model_class",side_effect=AssertionError("default must not construct")),mock.patch.object(Path,"read_bytes",side_effect=AssertionError("default must not read")),contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(self.api.main([]),0)
        result=json.loads(out.getvalue());self.assertEqual(result["input_reads"],0);self.assertEqual(result["writes"],0)
        self.assertEqual(before,{name:name in sys.modules for name in before})

    def test_head_persistence_has_known_mapping_and_is_shared(self):
        torch,model=self.fixture();z=torch.arange(2*512*6,dtype=torch.float32).reshape(2,512,6)
        actual=model.head(z.transpose(1,2)).transpose(1,2)
        self.assertTrue(torch.equal(actual,z[:,-1:].expand(-1,48,-1)))
        self.assertEqual(model.head.weight.shape,(48,512));self.assertEqual(sum(p.numel() for p in model.head.parameters()),24624)
        self.assertTrue(torch.equal(model.head.bias,torch.zeros(48)))

    def test_fixed_PED_keeps_head_gradient_but_has_no_D_gradient(self):
        torch,fresh=self.fixture();adapter=fresh.adapter
        state={key:value.detach().clone() for key,value in adapter.state_dict().items()}
        model=self.api.fixed_model(adapter);model.train()
        self.assertIs(model.adapter,adapter);self.assertFalse(adapter.training)
        x=torch.randn(2,512,21);y=model(x)
        self.assertEqual(tuple(y.shape),(2,48,21))
        (y-(x[:,-1:].expand_as(y)+1)).square().mean().backward()
        self.assertIsNotNone(model.head.weight.grad);self.assertTrue(torch.isfinite(model.head.weight.grad).all())
        self.assertGreater(float(model.head.weight.grad.abs().max()),0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in adapter.parameters()))
        self.assertIsNone(adapter.D.grad)
        self.assertEqual({id(p) for p in model.optimizer_parameters()},{id(p) for p in model.head.parameters()})
        self.assertTrue(all(torch.equal(value,adapter.state_dict()[key]) for key,value in state.items()))

    def test_full512_latent_input_and_original_decode_are_used(self):
        torch,model=self.fixture();model=self.api.fixed_model(model.adapter);x=torch.randn(1,512,21)
        shapes=[];handle=model.head.register_forward_pre_hook(lambda _,args:shapes.append(list(args[0].shape)))
        try:
            with mock.patch.object(model.adapter,"decode",wraps=model.adapter.decode) as decode:
                y=model(x);self.assertEqual(decode.call_count,1)
                args=decode.call_args.args;self.assertEqual(tuple(args[1].shape),(1,512,6));self.assertEqual(tuple(args[2].shape),(1,48,6))
            self.assertEqual(shapes,[[1,6,512]])
            self.assertTrue(torch.allclose(y,x[:,-1:].expand_as(y),atol=1e-6,rtol=1e-6))
        finally:handle.remove()
        with self.assertRaises(ValueError):model(x[:,-33:])

    def test_fresh_same_architecture_is_distinct_and_PED_gets_gradients(self):
        torch,one=self.fixture();two=self.api.fresh_model(torch.zeros(21),torch.eye(21)[:,:6])
        self.assertIsNot(one.adapter,two.adapter);self.assertIsNot(one.adapter.E,two.adapter.E)
        self.assertEqual(type(one.adapter),type(two.adapter))
        with torch.no_grad():two.head.weight[:,0]+=.1
        x=torch.randn(2,512,21);two(x).square().mean().backward()
        self.assertTrue(all(p.requires_grad for p in two.adapter.parameters()))
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().max()>0 for p in two.adapter.front.parameters()))
        self.assertIsNotNone(two.adapter.E.grad);self.assertIsNotNone(two.adapter.D.grad)
        self.assertGreater(float(two.adapter.E.grad.abs().max()),0)
        self.assertEqual({id(p) for p in two.optimizer_parameters()},{id(p) for p in two.parameters()})

    def test_no_foundation_or_original_fit_api_is_constructed(self):
        torch,model=self.fixture();self.assertEqual(model.mode,"R5b")
        self.assertFalse(any("foundation" in name.lower() or "bolt" in name.lower() for name,_ in model.named_modules()))
        self.assertNotIn("fit_base",set(sys.modules)-set(self.initial_modules))

    def test_wrong_adapter_contract_and_seed_rejected(self):
        torch,model=self.fixture();model.adapter.front.kind="B0"
        with self.assertRaises(ValueError):self.api.fixed_model(model.adapter)
        with self.assertRaises(ValueError):self.api.fresh_model(torch.zeros(21),torch.eye(21)[:,:6],seed=7)

    @classmethod
    def setUpClass(cls):cls.initial_modules=set(sys.modules)


if __name__=="__main__":unittest.main(verbosity=2)
