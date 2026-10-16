"""합성 metadata 분기만 검증한다. 실제 예측/배열/모델/감사기를 읽거나 실행하지 않는다."""
import argparse
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("wrapper_v2",HERE/"discovery_wrapper_v2.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class MetadataBranches(unittest.TestCase):
    def setUp(self):
        self.route=dict(id="legacy_fixture",status="READY",spec={"type":"ADAPTER"},legacy_receipt_sha256="receipt")
        self.cp=dict(update=2800,status="READY",scope="cache",checkpoint_relative_path="legacy_e0/checkpoints/legacy_fixture/cp2800.pt",
            file_sha256="file",state_sha256="state",expected_B={"mse":1.,"mae":.5},curve_evidence_status="BOUND_RECORDED_B_SCORE")
        self.eligible={(self.route["id"],2800):dict(route=self.route,checkpoint=self.cp)}
        self.inventory=dict(source_hashes={"old.py":"old"},statistics_sha256="statistics",foundation_sha256="foundation")
        self.sources={name:"source" for name in module.FINAL_SOURCES}
        self.sources.update({"legacy_evaluation.py":"eval","legacy_continuation.py":"continue"})
        self.lock=dict(locked=True,allowed_run_ids=["selected_fixture"],route_bindings={"selected_fixture":dict(route_spec={"family":"RAW"},data_sha="data",scaler_sha="scaler")},
            allowed_route_sha={"selected_fixture":BASE.fingerprint({"family":"RAW"})},source_bindings={name:self.sources[name] for name in module.FINAL_SOURCES},registry_scientific_sha="registry")
        self.lock["selection_sha256"]=BASE.fingerprint(self.lock)
        self.tname="route_predictions/selected_fixture/T_final/PREDICTION_MANIFEST.json"
        self.t=dict(run_id="selected_fixture",split="T_final",model_sha=self.lock["allowed_route_sha"]["selected_fixture"],data_sha="data",scaler_sha="scaler",
            metadata=dict(route_sha=self.lock["allowed_route_sha"]["selected_fixture"],route_spec={"family":"RAW"},T_gate_checked=True))
        identity=dict(legacy_id="legacy_fixture",checkpoint_update=2800,spec=self.route["spec"],model_file_sha256="file",model_state_sha256="state",
            checkpoint_relative_path=self.cp["checkpoint_relative_path"],model_path_scope="cache",source_hashes=self.inventory["source_hashes"],
            evaluation_source_sha256="eval",continuation_source_sha256="continue",statistics_sha256="statistics",foundation_sha256="foundation",
            legacy_receipt_sha256="receipt",curve_evidence_status="BOUND_RECORDED_B_SCORE",recorded_curve_sha256=BASE.fingerprint(self.cp["expected_B"]),length=512,horizon=48,fold="primary")
        self.bname="legacy_e0/evaluation/legacy_fixture/cp2800/PREDICTION_MANIFEST.json"
        self.b=dict(identity=identity,split=module.LEGACY_SPLIT,private_archive_name="predictions.npz",historically_unseen=False,protected_TEST_accessed=False,C_scored=False,
            expected_B=self.cp["expected_B"],arrays={key:{} for key in ("pred_normalized","target_normalized","observed","origins","channel_ids")})

    def classify(self,name,m):return module.classify_added(name,m,self.lock,self.eligible,self.inventory,self.sources,BASE)

    def test_official_T_path_and_binding(self):self.assertEqual(self.classify(self.tname,self.t),"T_final")
    def test_legacy_schema_without_campaign_cache_relative_path(self):
        self.assertNotIn("cache_relative_path",self.b)
        self.assertEqual(self.classify(self.bname,self.b),module.LEGACY_SPLIT)
    def test_T_unselected_id_rejected(self):
        with self.assertRaises(ValueError):self.classify(self.tname.replace("selected_fixture","other"),self.t)
    def test_T_wrong_route_rejected(self):
        self.t["model_sha"]="wrong"
        with self.assertRaises(ValueError):self.classify(self.tname,self.t)
    def test_T_wrong_scaler_rejected(self):
        self.t["scaler_sha"]="wrong"
        with self.assertRaises(ValueError):self.classify(self.tname,self.t)
    def test_T_missing_gate_rejected(self):
        self.t["metadata"]["T_gate_checked"]=False
        with self.assertRaises(ValueError):self.classify(self.tname,self.t)
    def test_added_V_or_wrong_prefix_rejected(self):
        for name in (self.tname.replace("T_final","V_select"),self.bname.replace("legacy_e0/evaluation","other/evaluation")):
            with self.subTest(name=name),self.assertRaises(ValueError):self.classify(name,self.t)
    def test_legacy_unsealed_CP_rejected(self):
        with self.assertRaises(ValueError):self.classify(self.bname.replace("cp2800","cp3200"),self.b)
    def test_legacy_leading_zero_CP_rejected(self):
        with self.assertRaises(ValueError):self.classify(self.bname.replace("cp2800","cp02800"),self.b)
    def test_legacy_wrong_checkpoint_bytes_or_state_rejected(self):
        for key in ("model_file_sha256","model_state_sha256","checkpoint_relative_path","model_path_scope"):
            changed=deepcopy(self.b);changed["identity"][key]="wrong"
            with self.subTest(key=key),self.assertRaises(ValueError):self.classify(self.bname,changed)
    def test_legacy_wrong_curve_or_sources_rejected(self):
        for key in ("recorded_curve_sha256","source_hashes","statistics_sha256","foundation_sha256"):
            changed=deepcopy(self.b);changed["identity"][key]="wrong"
            with self.subTest(key=key),self.assertRaises(ValueError):self.classify(self.bname,changed)
    def test_legacy_new_evidence_or_TEST_flags_rejected(self):
        for key in ("historically_unseen","protected_TEST_accessed","C_scored"):
            changed=deepcopy(self.b);changed[key]=True
            with self.subTest(key=key),self.assertRaises(ValueError):self.classify(self.bname,changed)
    def test_legacy_campaign_run_or_array_schema_rejected(self):
        changed=deepcopy(self.b);changed["run_id"]="new_campaign"
        with self.assertRaises(ValueError):self.classify(self.bname,changed)
        changed=deepcopy(self.b);del changed["arrays"]["observed"]
        with self.assertRaises(ValueError):self.classify(self.bname,changed)
    def test_real_selection_fingerprint_valid_and_tampered_rejected(self):
        self.assertEqual(module.validate_lock(self.lock,self.sources,"registry",BASE),{"selected_fixture"})
        self.lock["allowed_run_ids"]=["other"]
        with self.assertRaises(ValueError):module.validate_lock(self.lock,self.sources,"registry",BASE)
    def test_count_relationships_keep_B_separate(self):
        t,b=2,55
        expected=module.expected_counts(t,b)
        self.assertEqual(expected,dict(manifests_found=510,prediction_rescoring_passed=455,public_stats_reaggregation_passed=407,dense_private_rescoring_passed=48,legacy_B_rescoring_passed=55))


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--public-root",required=True);args=parser.parse_args()
    BASE=module.load_base(Path(args.public_root).resolve())
    for path in (HERE/"discovery_wrapper_v2.py",Path(__file__)):ast.parse(path.read_text(encoding="utf-8"))
    runner=unittest.TextTestRunner(verbosity=2)
    outcome=runner.run(unittest.defaultTestLoader.loadTestsFromTestCase(MetadataBranches))
    untouched=all(name not in sys.modules for name in ("numpy","torch","audit_campaign","legacy_evaluation"))
    BASE.new(HERE/"SYNTHETIC_TEST_REPORT.json",dict(status="PASS_SYNTHETIC_METADATA_ONLY" if outcome.wasSuccessful() and untouched else "FAIL",
        tests_run=outcome.testsRun,failures=len(outcome.failures),errors=len(outcome.errors),AST_parsed=2,
        no_numpy_torch_auditor_legacy_import=untouched,actual_453_manifest_read=False,NPZ_weights_Gzip_or_model_reads=0,
        actual_T_E0_execution=0,actual_prediction_rescores=0,source_v2_sha256=BASE.sha(HERE/"discovery_wrapper_v2.py"),test_sha256=BASE.sha(__file__)))
    raise SystemExit(0 if outcome.wasSuccessful() and untouched else 1)
