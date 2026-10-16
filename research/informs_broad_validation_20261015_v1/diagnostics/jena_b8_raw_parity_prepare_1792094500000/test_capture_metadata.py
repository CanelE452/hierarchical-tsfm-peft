"""토이 JSON 보존과 AST만 검사. 원 GPU 함수/배열/모델을 실행하지 않는다."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("capture_helper",HERE/"capture_original_parity.py")
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


class CaptureMetadata(unittest.TestCase):
    def test_failed_original_gate_is_saved_verbatim_before_next_exception(self):
        expected=dict(status="CHECKED",reference_row_microbatch="full",variants=[dict(row_microbatch="full",status="PASS",max_abs=0.),dict(row_microbatch=4,status="FAIL",max_abs=.25)],successful_variants_parity=False,atol=1e-5,rtol=1e-4)
        calls=[]
        def original(route,physical,observed,*,lease):
            calls.append((route,physical,observed,lease));return expected
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/"RETURN.json"
            result=helper.capture(original,"route","physical","observed","lease",target)
            with self.assertRaises(RuntimeError):raise RuntimeError("후속 실패를 모의")
            self.assertEqual(result,expected)
            self.assertEqual(helper.read(target),expected)
            self.assertEqual(calls,[("route","physical","observed","lease")])
    def test_capture_never_changes_defaults_or_overwrites_saved_json(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/"RETURN.json";helper.new(target,{"original":True})
            with self.assertRaises(FileExistsError):helper.capture(lambda *a,**k:{"changed":True},None,None,None,None,target)
            self.assertEqual(helper.read(target),{"original":True})
        tree=ast.parse((HERE/"capture_original_parity.py").read_text(encoding="utf-8"))
        capture=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="capture")
        call=next(n for n in ast.walk(capture) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=="function")
        self.assertEqual(len(call.args),3);self.assertEqual([k.arg for k in call.keywords],["lease"])
        self.assertEqual(sum(isinstance(n,ast.Attribute) and n.attr=="raw_row_parity" for n in ast.walk(tree)),1)


if __name__=="__main__":
    for path in (HERE/"capture_original_parity.py",Path(__file__)):ast.parse(path.read_text(encoding="utf-8"))
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CaptureMetadata))
    clean=all(name not in sys.modules for name in ("numpy","torch","measure_costs","deployment","data_contract"))
    helper.new(HERE/"CPU_PREPARATION_TEST_REPORT.json",dict(status="PASS_SYNTHETIC_JSON_AND_AST_ONLY" if result.wasSuccessful() and clean else "FAIL",
        actual_tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),AST_files=2,
        original_GPU_function_executions=0,private_arrays_or_models_loaded=False,scientific_modules_imported=False,
        no_numpy_torch_science_modules_imported=clean,helper_sha256=helper.sha(HERE/"capture_original_parity.py"),test_sha256=helper.sha(__file__)))
    raise SystemExit(0 if result.wasSuccessful() and clean else 1)
