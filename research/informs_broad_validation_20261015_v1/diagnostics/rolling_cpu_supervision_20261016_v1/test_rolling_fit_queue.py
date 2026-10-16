"""Pure fake queue/guard fixtures: no subprocess, model or scientific imports."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent


def load():
    spec=importlib.util.spec_from_file_location("tiny_rolling_queue",HERE/"rolling_fit_queue.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class RollingQueueTests(unittest.TestCase):
    def setUp(self):self.h=load()

    def test_fast_slot_refills_before_slow_slot_finishes_without_pair_barrier(self):
        release=threading.Event();records=[];lock=threading.Lock()
        def work(unit):
            if unit["id"]=="a":self.assertTrue(release.wait(2))
            if unit["id"]=="c":
                with lock:records.append("c_started")
                release.set()
            with lock:records.append(unit["id"]+"_finished")
            return dict(id=unit["id"],status="ALL_STEPS_PASSED")
        result=self.h.rolling([dict(id=c) for c in "abcd"],work,lambda:None,lambda e:None)
        self.assertFalse(result["global_stop"])
        self.assertLess(records.index("c_started"),records.index("a_finished"))
        self.assertEqual({r["id"] for r in result["rows"]},set("abcd"))

    def test_closed_local_failure_does_not_gate_independent_jobs(self):
        def work(unit):return dict(id=unit["id"],status="LOCAL_FIT_FAILURE_PRESERVED" if unit["id"]=="a" else "ALL_STEPS_PASSED",closed=True)
        result=self.h.rolling([dict(id=c) for c in "abc"],work,lambda:None,lambda e:None)
        self.assertFalse(result["global_stop"])
        self.assertEqual(len(result["rows"]),3)

    def test_global_failure_stops_new_admission_and_waits_already_owned_fake_work(self):
        started=[];finished=[];event=threading.Event()
        def work(unit):
            started.append(unit["id"])
            if unit["id"]=="a":
                event.wait(.03)
                raise self.h.GlobalStop("synthetic unknown closure")
            event.set();time.sleep(.05);finished.append(unit["id"])
            return dict(id=unit["id"],status="ALL_STEPS_PASSED")
        result=self.h.rolling([dict(id=c) for c in "abcd"],work,lambda:None,lambda e:None)
        self.assertTrue(result["global_stop"])
        self.assertNotIn("c",started)
        self.assertNotIn("d",started)
        if "b" in started:self.assertIn("b",finished)
        self.assertEqual({row["id"] for row in result["rows"]},set("abcd"))

    def test_pre_admission_pin_failure_keeps_the_not_launched_unit_in_results(self):
        count=[0]
        def check():
            count[0]+=1
            if count[0]==3:raise self.h.GlobalStop("source pin changed")
        result=self.h.rolling([dict(id=c) for c in "abcd"],lambda u:dict(id=u["id"],status="ALL_STEPS_PASSED"),check,lambda e:None)
        self.assertTrue(result["global_stop"])
        self.assertEqual({row["id"] for row in result["rows"]},set("abcd"))
        self.assertEqual(next(r for r in result["rows"] if r["id"]=="c")["status"],"NOT_LAUNCHED_GLOBAL_STOP")

    def fake_runtime(self,root,closure,executed,scientific_bytes=None):
        h=self.h;runtime=h.Runtime.__new__(h.Runtime)
        runtime.output=root/"queue";runtime.output.mkdir()
        runtime.stop=threading.Event();runtime.check=mock.Mock();runtime.storage=mock.Mock();runtime.event=mock.Mock()
        scientific=root/"scientific";scientific.mkdir()
        guard_output=root/"guard"
        spec=dict(output_root=str(guard_output),processes=[dict(id="TINY",command=["python","--output",str(scientific)])])
        result=dict(status="COMPLETE",processes=[dict(returncode=0)])
        def supervise(value,**kwargs):
            guard_output.mkdir();h.new(guard_output/"RESULT.json",result)
            if scientific_bytes is None:h.new(scientific/"RESULT.json",dict(status="PASS",actual_checkpoint_forward_in_this_command=executed))
            else:(scientific/"RESULT.json").write_bytes(scientific_bytes)
            return result
        runtime.guard=types.SimpleNamespace(validate_spec=lambda v:v,supervise=supervise)
        runtime.core=types.SimpleNamespace(closure_state=lambda *a:(closure,{"tiny":True}),identity_state=lambda b:"DEAD")
        return runtime,spec,guard_output

    def test_unknown_owned_closure_halts_globally_and_preserves_guard_receipt(self):
        with tempfile.TemporaryDirectory() as td:
            runtime,spec,guard=self.fake_runtime(Path(td),"UNSAFE",True)
            with self.assertRaises(self.h.GlobalStop):runtime.perform(dict(id="one"),"replay_tiny",spec)
            self.assertTrue(runtime.stop.is_set())
            self.assertTrue((guard/"RESULT.json").is_file())

    def test_cached_replay_is_closed_but_is_not_a_fresh_CPU_verification(self):
        for executed in (False,True):
            with self.subTest(executed=executed),tempfile.TemporaryDirectory() as td:
                runtime,spec,_=self.fake_runtime(Path(td),"READY",executed)
                result=runtime.perform(dict(id="one"),"replay_tiny",spec)
                self.assertTrue(result["closed"])
                self.assertIs(result["success"],executed)
                self.assertTrue((runtime.output/"one/replay_tiny/TERMINAL.json").is_file())

    def test_closed_partial_or_nonobject_scientific_result_is_local_and_others_continue(self):
        for content in (b'{"status":',b'[]',b'5',b'null',b'"scalar"',b'{"status": []}',b'{"status": {}}'):
            with self.subTest(content=content),tempfile.TemporaryDirectory() as td:
                runtime,spec,_=self.fake_runtime(Path(td),"READY",True,scientific_bytes=content)
                def work(unit):
                    if unit["id"]=="one":return dict(id="one",**runtime.perform(unit,"replay_tiny",spec))
                    return dict(id=unit["id"],success=True)
                result=self.h.rolling([dict(id=x) for x in ("one","two","three")],work,lambda:None,lambda e:None)
                self.assertFalse(result["global_stop"])
                self.assertFalse(runtime.stop.is_set())
                row=next(r for r in result["rows"] if r["id"]=="one")
                self.assertTrue(row["closed"]);self.assertFalse(row["success"])
                self.assertTrue(row["scientific_result_error"])
                self.assertEqual(Path(row["scientific_result"]).read_bytes(),content)
                self.assertEqual(row["scientific_result_sha256"],self.h.sha(row["scientific_result"]))
                self.assertEqual({r["id"] for r in result["rows"]},{"one","two","three"})

    def test_exact_fifteen_fit_order_and_opaque_pin_tampering(self):
        plans=dict(multik=dict(fits=[dict(run_id="mk"+str(i)) for i in range(5)]),
            required=dict(conditions=[dict(run_id="r"+str(i)) for i in range(6)]),
            family=dict(conditions=[dict(run_id="f"+str(i)) for i in range(4)]))
        ordered=self.h.fit_order(plans)
        self.assertEqual([row["run_id"] for _,row in ordered[:2]],["mk0","mk1"])
        self.assertEqual(len(ordered),15)
        self.assertEqual({row["run_id"] for _,row in ordered},{"mk"+str(i) for i in range(5)}|{"r"+str(i) for i in range(6)}|{"f"+str(i) for i in range(4)})
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"tiny.json";path.write_text("{}",encoding="utf-8")
            pins={};self.h.pin(pins,path);path.write_text("changed",encoding="utf-8")
            with self.assertRaises(self.h.GlobalStop):self.h.pin({},path,next(iter(pins.values())))

    def test_default_cli_has_no_science_imports_child_or_experiment_reads(self):
        before={name:name in sys.modules for name in ("numpy","torch","psutil")}
        with mock.patch.object(self.h,"metadata",side_effect=AssertionError("no metadata requested")), \
             mock.patch.object(self.h,"execute",side_effect=AssertionError("no execution authorized")),contextlib.redirect_stdout(io.StringIO()) as stream:
            self.assertEqual(self.h.main([]),0)
        value=json.loads(stream.getvalue())
        self.assertEqual(value["actual_children_or_science"],0)
        self.assertEqual(value["files_written"],0)
        self.assertEqual(before,{name:name in sys.modules for name in before})


if __name__=="__main__":unittest.main(verbosity=2)
