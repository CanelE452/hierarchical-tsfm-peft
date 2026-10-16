"""Tiny stdlib-only checks; no actual Task, NumPy, Torch or model is imported."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("priority_p0_tested", Path(__file__).with_name("check_priority_path.py"))
p0 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p0)


class Handle:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class Context:
    shape = (48, 512)
    requires_grad = True

    def register_hook(self, hook):
        self.hook = hook
        self.handle = Handle()
        return self.handle


class Module:
    def register_forward_pre_hook(self, hook, with_kwargs):
        assert with_kwargs is True
        self.hook = hook
        self.handle = Handle()
        return self.handle


class Trace:
    def __init__(self, text, all_observed=True):
        self.text, self.all_observed = text, all_observed

    def all(self): return self.all_observed
    def copy(self): return Trace("copy(" + self.text + ")")
    def astype(self, dtype): return Trace("float32(" + self.text + ")")
    def __sub__(self, other): return Trace("(" + self.text + "-" + other.text + ")")
    def __truediv__(self, other): return Trace("(" + self.text + "/" + other.text + ")")
    def __mul__(self, other): return Trace("(" + self.text + "*" + other.text + ")")
    def __add__(self, other): return Trace("(" + self.text + "+" + other.text + ")")
    def double(self): return Trace("FP64(" + self.text + ")")
    def cpu(self): return Trace("CPU(" + self.text + ")")
    def numpy(self): return self.text


class TinyTests(unittest.TestCase):
    def test_default_opens_nothing_and_never_calls_execute(self):
        with patch.object(p0, "sha", side_effect=AssertionError("input read")), patch.object(p0, "execute", side_effect=AssertionError("execution")):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(p0.main([]), 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["input_reads"], 0)
            self.assertFalse(result["actual_backward"])
            self.assertFalse(result["torch_imported"])

    def test_cli_requires_all_explicit_execution_bindings(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            p0.main(["--execute"])

    def test_actual_module_kwargs_context_hook_tracks_backward_and_cleanup(self):
        module, context = Module(), Context()
        probe = p0.InputGradientProbe(module, lambda value: True)
        module.hook(module, (), {"context": context})
        gradient = Context()
        self.assertIs(context.hook(gradient), gradient)
        self.assertEqual(probe.evidence()["module_context_shapes"], [[48,512]])
        probe.close()
        self.assertTrue(module.handle.removed)
        self.assertTrue(context.handle.removed)

    def test_positional_context_also_supported(self):
        module, context = Module(), Context()
        probe = p0.InputGradientProbe(module, lambda value: True)
        module.hook(module, (context,), {})
        context.hook(Context())
        self.assertEqual(len(probe.evidence()["input_backward_hooks"]), 1)
        probe.close()

    def test_detached_actual_F_input_rejected(self):
        module, context = Module(), Context()
        context.requires_grad = False
        probe = p0.InputGradientProbe(module, lambda value: True)
        with self.assertRaises(AssertionError): module.hook(module, (), {"context": context})
        probe.close()

    def test_backward_hook_absence_and_nonfinite_gradient_rejected(self):
        module, context = Module(), Context()
        probe = p0.InputGradientProbe(module, lambda value: False)
        module.hook(module, (), {"context": context})
        with self.assertRaises(AssertionError): probe.evidence()
        context.hook(Context())
        with self.assertRaises(AssertionError): probe.evidence()
        probe.close()

    def test_connected_finite_zero_gradients_are_allowed(self):
        groups = {key:dict(gradient_tensors_present=1, all_present_gradients_finite=True, any_nonzero_gradient=False) for key in ("P","E","D")}
        p0.validate_gradient_groups(groups)
        groups["E"]["gradient_tensors_present"] = 0
        with self.assertRaises(AssertionError): p0.validate_gradient_groups(groups)

    def test_nonfinite_PED_gradient_is_rejected(self):
        groups = {key:dict(gradient_tensors_present=1, all_present_gradients_finite=True) for key in ("P","E","D")}
        groups["D"]["all_present_gradients_finite"] = False
        with self.assertRaises(AssertionError): p0.validate_gradient_groups(groups)

    def test_PID_reuse_is_distinct_from_alive_old_identity(self):
        identities = [dict(pid=123,create_time=10.)]
        self.assertTrue(p0.stopped_identities(identities, lambda pid: None)[0]["old_identity_dead"])
        self.assertTrue(p0.stopped_identities(identities, lambda pid: 20.)[0]["PID_reused"])
        with self.assertRaises(RuntimeError): p0.stopped_identities(identities, lambda pid: 10.)

    def test_empty_identity_proof_is_rejected(self):
        with self.assertRaises(ValueError): p0.stopped_identities([], lambda pid: None)

    def test_row128_counter_contract_distinguishes_calls_from_streams(self):
        self.assertEqual(p0.counter_delta(dict(streams=100,physical_streams=150,physical_calls=3),
            dict(streams=148,physical_streams=198,physical_calls=4)),dict(streams=48,physical_streams=48,physical_calls=1))

    def test_transparent_wrapper_uses_same_F_and_FP64_inverse_without_adapter(self):
        calls=[]
        class NP:
            float32="float32"
            @staticmethod
            def asarray(value, dtype=None): return value
        class Torch:
            float32="float32"
            @staticmethod
            def as_tensor(value, **kwargs):
                self.assertEqual(kwargs["device"], "cpu")
                return value
        def foundation(value):
            calls.append(value.text)
            return Trace("F(" + value.text + ")")
        result = p0.transparent_raw_wrapper(Trace("physical"),Trace("mask"),Trace("mean"),Trace("std"),foundation,None,NP,Torch)
        self.assertEqual(calls,["float32(((copy(physical)-mean)/std))"])
        self.assertEqual(result,"CPU(((FP64(F(float32(((copy(physical)-mean)/std))))*FP64(std))+FP64(mean)))")

    def test_deadline_and_root_seal_fail_closed(self):
        with patch.object(p0.time,"time",return_value=100):
            with self.assertRaises(TimeoutError): p0.deadline_check(99)
            p0.deadline_check(101)
        with self.assertRaises(ValueError): p0.validate_plan(dict(status="DRAFT_ROOT_SCHEMA_NOT_EXECUTABLE",namespace=p0.NAME))

    def test_root_path_escape_rejected_without_input_read(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError): p0.inside(temp,"../escape")
            self.assertEqual(p0.inside(temp,"child"),Path(temp).resolve()/"child")


if __name__ == "__main__":
    unittest.main()
