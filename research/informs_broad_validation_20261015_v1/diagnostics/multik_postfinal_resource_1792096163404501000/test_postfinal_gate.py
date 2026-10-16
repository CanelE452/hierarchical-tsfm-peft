"""Synthetic metadata only; never import campaign runtime or access real targets."""
import copy
import importlib.util
import unittest
from pathlib import Path


class EvidenceBoundaryTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).with_name("postfinal_gate.py")
        if not path.is_file():
            self.fail("Post-final evidence gate has not been implemented")
        spec = importlib.util.spec_from_file_location("postfinal_gate_test_target", path)
        self.gate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.gate)
        self.ids = ["route_"+str(i) for i in range(46)]
        self.plan = dict(selected_run_ids=self.ids, source18={"source.py":"source"},
            legacy_checkpoints=[dict(id="legacy_"+str(i%9),update=i) for i in range(55)])
        self.lock = dict(locked=True,allowed_run_ids=self.ids,selection_sha256="lock",
                         source_bindings=self.plan["source18"])
        self.batch = dict(status="COMPLETE_SELECTED_ROUTES",phase="T_final",run_ids=self.ids,
            succeeded=46,failed=0,deferred_run_ids=[],selection_sha256="lock",
            records=[dict(run_id=i,status="PASS") for i in self.ids])
        self.legacy = dict(available_checkpoint_count=55,protected_TEST_accessed=False,C_scored=False,
            results=[dict(id=x["id"],update=x["update"],status="CURVE_SCORE_MISMATCH",score={})
                     for x in self.plan["legacy_checkpoints"]])

    def test_exact_T46_and_curve_mismatch_still_prove_attempts(self):
        self.assertEqual(self.gate.validate_T_documents(self.plan,self.lock,self.batch),46)
        result=self.gate.validate_E0_records(self.plan,self.legacy)
        self.assertEqual(result["rescored_records"],55)
        self.assertEqual(result["failed_inference_attempts"],0)

    def test_T_duplicate_cannot_hide_a_missing_route(self):
        self.batch["records"][-1]=copy.deepcopy(self.batch["records"][0])
        with self.assertRaises(ValueError):
            self.gate.validate_T_documents(self.plan,self.lock,self.batch)

    def test_T_partial_or_wrong_selection_cannot_release_reserve(self):
        for key,value in [("status","PARTIAL_ROUTE_FAILURES"),("deferred_run_ids",["route_1"]),
                          ("selection_sha256","another-lock")]:
            batch=copy.deepcopy(self.batch);batch[key]=value
            with self.assertRaises(ValueError):self.gate.validate_T_documents(self.plan,self.lock,batch)

    def test_55_budget_pending_rows_are_not_55_actual_attempts(self):
        for x in self.legacy["results"]:x.update(status="BUDGET_PENDING",actual_F_calls=0)
        with self.assertRaises(ValueError):self.gate.validate_E0_records(self.plan,self.legacy)

    def test_E0_duplicate_cannot_hide_a_missing_checkpoint(self):
        self.legacy["results"][-1]=copy.deepcopy(self.legacy["results"][0])
        with self.assertRaises(ValueError):self.gate.validate_E0_records(self.plan,self.legacy)

    def test_E0_load_failure_is_not_inference_but_positive_call_is(self):
        row=self.legacy["results"][0];row.clear();row.update(self.plan["legacy_checkpoints"][0],
            status="REPLAY_FAILED",actual_F_calls=0)
        with self.assertRaises(ValueError):self.gate.validate_E0_records(self.plan,self.legacy)
        row["actual_F_calls"]=1
        self.assertEqual(self.gate.validate_E0_records(self.plan,self.legacy)["failed_inference_attempts"],1)

    def test_evaluation_manifest_cannot_escape_public_root(self):
        with self.assertRaises(ValueError):self.gate.inside(Path("."),"../outside.json")

    def test_unrelated_closed_lease_cannot_prove_final_phase_termination(self):
        if not hasattr(self.gate,"validate_phase_owner"):self.fail("Final phase/owner cross-binding is missing")
        owner=dict(lease_id="actual",pid=123,create_time=10.,started_unix=20.,authorized_children=[])
        closure=dict(lease_id="old",status="CLOSED",started_unix=20.,finished_unix=100.,deadline_overrun_seconds=0)
        with self.assertRaises(ValueError):self.gate.validate_phase_owner(owner,closure,30.,90.)
        closure["lease_id"]="actual"
        self.assertEqual(self.gate.validate_phase_owner(owner,closure,30.,90.),[(123,10.)])

    def test_owner_closure_must_cover_actual_final_phase_time(self):
        if not hasattr(self.gate,"validate_phase_owner"):self.fail("Final phase/owner cross-binding is missing")
        owner=dict(lease_id="actual",pid=123,create_time=10.,started_unix=20.,authorized_children=[])
        closure=dict(lease_id="actual",status="CLOSED",started_unix=20.,finished_unix=80.,deadline_overrun_seconds=0)
        with self.assertRaises(ValueError):self.gate.validate_phase_owner(owner,closure,30.,90.)


if __name__=="__main__":unittest.main()
