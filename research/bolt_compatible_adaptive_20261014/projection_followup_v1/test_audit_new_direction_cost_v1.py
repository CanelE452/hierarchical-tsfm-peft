"""Independent-auditor fixtures generated from tiny CPU callbacks, never real cohorts/F."""
from pathlib import Path
import ast
import copy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import audit_new_direction_cost_v1 as audit


class NewDirectionCostAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        audit.initialize_numpy();cls.temp=tempfile.TemporaryDirectory();cls.cases={}
        import whole_new_direction_cost_v1 as producer
        from test_whole_new_direction_cost_v1 import fixture
        for data in audit.DATASETS:
            physical,mask,mean,std,origins,predictors,predictions,metadata,_=fixture(data)
            raw={"history":producer.preprocess(physical,mask,mean,std),"history_observed":mask,"mean":mean,"std":std,"origins":origins}
            raw["raw_last"]=raw["history"][:,-1].copy()
            def memory():return {"RSS_bytes":1000,"allocated_VRAM_bytes":0,"reserved_VRAM_bytes":0,"max_allocated_VRAM_bytes":0,"max_reserved_VRAM_bytes":0,"scope":audit.MEMORY_SCOPE}
            cost=producer.measure_cost(data,physical,mask,mean,std,origins,predictors,predictions,metadata,Path(cls.temp.name)/data,lambda *a:None,lambda:None,memory)
            cls.cases[data]=(cost,{"history_physical":physical,"history_observed":mask,"origins":origins},raw,predictions)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_two_full_generated_dataset_costs_all_76_cells_and_380_samples(self):
        totals=[]
        for cost,physical,raw,predictions in self.cases.values():totals.append(audit.check_dataset(cost,physical,raw,predictions,lambda:None,0.))
        self.assertEqual(sum(t["complete_cells"] for t in totals),76);self.assertEqual(sum(t["saved_samples"] for t in totals),380);self.assertTrue(all(t["F_or_timer_replayed"] is False for t in totals))

    def test_independent_dataset_routes_and_five_of_nineteen_schedule(self):
        for data in audit.DATASETS:
            routes=audit.routes(data);schedule=audit.schedule(data);self.assertEqual(len(routes),19);self.assertEqual(len(set(routes)),19);self.assertEqual(len(schedule),5)
            self.assertTrue(all(set(row)==set(routes) for row in schedule));self.assertTrue(all(len({row.index(route) for row in schedule})==5 for route in routes))
            self.assertEqual(sum("RIDGE" in r for r in routes),6);self.assertEqual(sum("_NATIVE_" in r for r in routes),8)
            for route in routes:self.assertIn(audit.matched_bolt(route),routes)

    def test_ns_and_paired_ratio_tampering_rejected_independently(self):
        base,physical,raw,predictions=self.cases["robin"]
        for field in ("ns","RAW","matched_Bolt"):
            with self.subTest(field=field):
                changed=copy.deepcopy(base);row=changed["rows"][2]
                if field=="ns":row["samples"][0]["perf_end_ns"]+=100
                else:row["paired_"+field+"_median_ratio"]+=.1
                with self.assertRaisesRegex(ValueError,"DURATIONS|TWO_RATIOS"):audit.check_dataset(changed,physical,raw,predictions,lambda:None,0.)

    def test_dropped_or_partial_route_and_native_scope_change_do_not_pass(self):
        base,physical,raw,predictions=self.cases["beijing_aotizhongxin"]
        for kind in ("drop","partial","scope"):
            with self.subTest(kind=kind):
                changed=copy.deepcopy(base)
                if kind=="drop":changed["rows"].pop()
                elif kind=="partial":changed["rows"][0]["status"]="PARTIAL_OR_FAILED_PATH_PRESERVED"
                else:next(r for r in changed["rows"] if "_NATIVE_" in r["route"])["native_head_loaded_or_timed"]=False
                with self.assertRaisesRegex(ValueError,"UNIQUE_ROWS|COMPLETE_ROW|DIRECT_C_SCOPE"):audit.check_dataset(changed,physical,raw,predictions,lambda:None,0.)

    def test_selection_seal_after_measurement_is_rejected(self):
        cost,physical,raw,predictions=self.cases["robin"];seal=cost["rows"][0]["samples"][0]["unix_start_ns"]/1e9+1.
        with self.assertRaisesRegex(ValueError,"POST_SELECTION_SEAL"):audit.check_dataset(cost,physical,raw,predictions,lambda:None,seal)

    def test_changed_physical_input_cannot_pass_existing_normalized_cache(self):
        cost,physical,raw,predictions=self.cases["robin"];changed=copy.deepcopy(physical);changed["history_physical"][0,-1,1]+=1
        with self.assertRaisesRegex(ValueError,"physical_causal_fill"):audit.check_dataset(cost,changed,raw,predictions,lambda:None,0.)

    def test_direct_ridge_first8_independent_arithmetic_and_corruption_detection(self):
        from test_run_new_direction_cost_v1 import generated_past,coefficients
        import run_new_direction_cost_v1 as producer
        from run_predictive_cross_moment_v1 import encode_torch
        for encoded in (False,True):
            with self.subTest(encoded=encoded):
                raw,physical,encoder=generated_past(encoded);route="PCA_K4_RIDGE_HISTORY512" if encoded else "RAW_CURRENT_C_RIDGE";state=coefficients(512 if encoded else 3);reference={"TUNE_prediction_FP32":producer.ridge_predictor(route,state,encoder)(raw["history"],raw["history_observed"]),"TUNE_origins":raw["origins"]};cached={"latent_history":encode_torch(encoder,raw["history"])} if encoded else None
                entry,arrays=producer.ridge_physical_precheck(physical,raw,reference,state,route,encoder,cached);frozen={key:value.detach().cpu().numpy().copy() for key,value in encoder.state_dict().items()} if encoded else None
                result=audit.ridge_precheck(entry,arrays,physical,raw,state,reference,route,frozen,cached);self.assertTrue(result["D_bypassed"] and result["independent_fused_FP32_arithmetic"]);self.assertEqual(result["encoded_E_checked"],encoded)
                bad=copy.deepcopy(arrays);bad["prediction"][0,0,0]+=.1
                with self.assertRaisesRegex(ValueError,"coefficient_arithmetic"):audit.ridge_precheck(entry,bad,physical,raw,state,reference,route,frozen,cached)

    def test_exact_timer_AST_and_actual_callbacks_no_cached_shortcut(self):
        helper=(audit.HERE/"whole_new_direction_cost_v1.py").read_text(encoding="utf-8");runner=(audit.HERE/"run_new_direction_cost_v1.py").read_text(encoding="utf-8");parent=(audit.HERE/"whole_f_input_gradient_cost_v1.py").read_text(encoding="utf-8")
        result=audit.source_contract(helper,runner,parent);self.assertFalse(result["F_or_timer_replayed"]);self.assertFalse(result["complete_nineteen_position_balance_claimed"])
        with self.assertRaisesRegex(ValueError,"ENCODED_CUDA_E"):audit.source_contract(helper,runner.replace("source.predict(feature,state,anchor)","cached_prediction"),parent)
        with self.assertRaisesRegex(ValueError,"ONE_CANONICAL_TIMED_TRY|UNCHANGED_WHOLE"):audit.source_contract(helper.replace("synchronize(); perf_end_ns, unix_end_ns", "perf_end_ns, unix_end_ns"),runner,parent)

    def test_closed_actual_gate_does_not_read_assets(self):
        with patch.object(audit,"COST_SHA",None),patch.object(audit.linear,"file_record",side_effect=AssertionError("no actual file IO")),self.assertRaisesRegex(ValueError,"NOT_YET_REGISTERED"):audit.audit_result("missing",type("Meter",(),{"check":lambda self:None})())

    def test_auditor_contains_no_model_or_producer_import_or_timing_replay(self):
        tree=ast.parse(Path(audit.__file__).read_text(encoding="utf-8"));imports=[alias.name for node in ast.walk(tree) if isinstance(node,ast.Import) for alias in node.names]
        self.assertFalse(any(name in imports for name in ("torch","run_new_direction_cost_v1","whole_new_direction_cost_v1")))
        calls=[ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node,ast.Call)];self.assertFalse(any(name.endswith(("FrozenBolt","measure_cost","fit_path","train_head","perf_counter_ns")) for name in calls))


if __name__=="__main__":unittest.main()
