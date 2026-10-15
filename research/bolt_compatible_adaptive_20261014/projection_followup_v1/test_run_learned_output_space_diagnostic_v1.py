"""Small learned-state diagnostic contracts; synthetic saved futures only."""
import importlib
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

try:
    import run_learned_output_space_diagnostic_v1 as diagnostic
except ModuleNotFoundError as error:
    if error.name != "run_learned_output_space_diagnostic_v1": raise
    diagnostic = None


def fixture(lags=(0,), seed=7):
    rng = np.random.default_rng(seed); n, c, k = 2, 3, 1
    history = rng.normal(size=(n, 512, c)).astype(np.float32)
    state = {"E": np.array([[1.], [.2], [-.1]], np.float32), "center": np.zeros(c, np.float32), "D": rng.normal(size=(len(lags)*k, c)).astype(np.float32)}
    past = (history-state["center"])@state["E"]; future = rng.normal(size=(n, 48, k)).astype(np.float32)
    import audit_causal_lag_objective_v2 as independent
    independent.initialize_numpy()
    features = independent.delta_features(past, future, lags)
    current = history[:, -1].copy(); observed = np.ones((n, 48, c), bool)
    observed[0, 1, 1] = False; target = rng.normal(size=(n, 48, c)).astype(np.float32); target[~observed] = np.nan
    raw = {"history": history, "target": target, "observed": observed, "raw_last": current, "origins": np.arange(n, dtype=np.int64)+512, "mean": np.zeros(c, np.float32), "std": np.arange(1, c+1, dtype=np.float32), "raw_forecast": np.repeat(current[:, None], 48, axis=1)}
    saved = {key: raw[key].copy() for key in ("target", "observed", "raw_last", "origins", "mean", "std")}
    saved.update(latent_history=past, latent_future=future, lag_features=features, prediction=current[:, None]+features@state["D"])
    return state, raw, saved


class LearnedOutputSpaceTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(diagnostic, "new learned-output diagnostic source is missing")
        diagnostic.initialize_numpy()

    def test_closed_gate_precedes_any_file_read(self):
        self.api()
        with patch.object(diagnostic, "STOP_AUDIT_SHA", None), patch.object(diagnostic, "record", side_effect=AssertionError("closed read")), self.assertRaisesRegex(ValueError, "ACTUAL_PARENT_PASS_NOT_REGISTERED"):
            diagnostic.load_gates("a", "b", "c", "d")

    def test_l0_real_minimum_saved_actual_EY_flat_and_observed_NaN(self):
        self.api(); state, raw, saved = fixture()
        row, arrays = diagnostic.diagnose_cell(saved, state, raw, (0,))
        self.assertEqual(row["status"], "COMPLETE"); self.assertEqual(row["complete_target_positions"], 95)
        self.assertEqual(row["joint_unknown_future_coordinates"], 48)
        self.assertEqual(set(row["methods"]), {"ACTUAL_F", "EY", "FLAT"})
        self.assertFalse(row["RAW_bound_applied"]); self.assertFalse(row["new_F_forward"])
        self.assertTrue(row["independent_reduced_LS_vs_SVD_verified"])
        np.testing.assert_array_equal(arrays["ACTUAL_F_prediction_FP32"], saved["prediction"])
        np.testing.assert_array_equal(arrays["EY_input_C_FP32"][0, 1], raw["raw_last"][0])
        self.assertTrue(row["methods"]["ACTUAL_F"]["retained_output_trace_bound_inequality_pass"])
        json.dumps(row, allow_nan=False)

    def test_l7_flat_latent_is_not_current_C_persistence(self):
        self.api(); lags=(0, 1, 2, 4, 8, 16, 32); state, raw, saved=fixture(lags)
        row, arrays=diagnostic.diagnose_cell(saved, state, raw, lags)
        self.assertGreater(np.abs(arrays["FLAT_prediction_FP32"]-raw["raw_last"][:, None]).max(), 0.)
        self.assertEqual(row["joint_unknown_future_coordinates"], 48)
        self.assertEqual(row["lag_slope_parameters_per_C"], 7)

    def test_corrupted_saved_prediction_rejected_before_floor_claim(self):
        self.api(); state, raw, saved=fixture(); saved["prediction"][0, 0, 0] += .1
        with self.assertRaisesRegex(ValueError, "independent_NumPy_fixed_D_forecast"):
            diagnostic.diagnose_cell(saved, state, raw, (0,))

    def test_new_D_space_is_computed_and_no_complete_mask_is_explicit(self):
        self.api(); state, raw, saved=fixture(); first,_=diagnostic.diagnose_cell(saved, state, raw, (0,))
        state["D"][:]=0.; saved["prediction"][:]=raw["raw_last"][:, None]
        second,_=diagnostic.diagnose_cell(saved, state, raw, (0,))
        self.assertNotEqual(first["minimum_normalized_MSE"], second["minimum_normalized_MSE"])
        raw["observed"][:]=False; raw["target"][:]=np.nan; saved["observed"][:]=False; saved["target"][:]=np.nan
        empty,_=diagnostic.diagnose_cell(saved, state, raw, (0,))
        self.assertEqual(empty["status"], "NO_COMPLETE_C_TARGET_POSITIONS")
        self.assertIsNone(empty["minimum_normalized_MSE"]); json.dumps(empty, allow_nan=False)

    def test_full_orchestration_synthetic_twenty_four_cells_sealed_parents_and_fresh_outputs(self):
        self.api(); import continuation_runtime_v1 as rt
        import fixed_decoder_span_v1 as io
        import audit_linear_e_adaptation_v1 as learned
        import run_fit_target_comparison_v1 as parent_io
        import prepare_inputs
        from copy import deepcopy
        class Meter:
            def __init__(self, *a, **kw): pass
            def check(self, *a): pass
            def protected_check(self): pass
            def close(self, status, failure): return {"status":status,"fixture_scope":"synthetic only; no campaign authorization/assets/F"}
        full, stopped, science, payloads={"datasets":{}},{"datasets":{}},{"datasets":{}},{}
        data_contract={"toyA":(3,1,"dev",2),"toyB":(3,1,"dev",2)}
        def payload(key, arrays):
            payloads[key]=arrays; return {"path":key,"sha256":"0"*64}
        for data in data_contract:
            full["datasets"][data]={"conditions":{}}; stopped["datasets"][data]={"conditions":{}}
            source_splits={}; raws={}
            for split,n in (("tune",64),("dev",2)):
                _,raw,_=fixture()
                raw={key:np.repeat(value,n//2,axis=0) if key in ("history","target","observed","raw_last","raw_forecast") else value.copy() for key,value in raw.items()}
                raw["origins"]=np.arange(n,dtype=np.int64)+512; raw["history_observed"]=np.ones(raw["history"].shape,bool)
                raws[split]=raw; source_splits[split]={"input_source":payload(data+split+"input",raw)}
            science["datasets"][data]={"spaces":{"PCA":{"conditions":{"L0_FORECAST":{"splits":source_splits}}}}}
            for lag,lags in diagnostic.LAGS.items():
                state,_,_=fixture(lags); state_source=payload(data+lag+"state",state)
                outputs={}
                for split,raw in raws.items():
                    past=(raw["history"]-state["center"])@state["E"]; future=np.repeat(past[:,-1:],48,axis=1)
                    import audit_causal_lag_objective_v2 as independent
                    features=independent.delta_features(past,future,lags)
                    saved={key:raw[key] for key in ("target","observed","raw_last","origins","mean","std")}
                    saved.update(prediction=raw["raw_last"][:,None]+features@state["D"],latent_history=past,latent_future=future,lag_features=features)
                    mask=independent.prior.oracle_mask(raw["observed"])
                    outputs[split]={"artifact":payload(data+lag+split+"saved",saved),"scores_all_observed":diagnostic.scored(saved["prediction"],raw,raw["observed"]),"scores_complete_mask":diagnostic.scored(saved["prediction"],raw,mask)}
                for seed in diagnostic.gates.SEEDS:
                    cp={"update":100,"state":state_source,"tune":{"mse":1.},**outputs["tune"]}
                    row={"seed":seed,"selected_update":100,"selected_state":state_source,"selected_state_sha256":learned.state_hash(state),"selection_lock":{"path":"syntheticlock","sha256":"0"*64},"checkpoints":[cp],"selected_tune":cp["tune"],"DEV":outputs["dev"]}
                    full["datasets"][data]["conditions"][data+"_"+lag+"_E_LEARN_s"+str(seed)]=row
                    if lag=="L0": stopped["datasets"][data]["conditions"][str(seed)]=deepcopy(row)
        with tempfile.TemporaryDirectory() as folder, ExitStack() as patches:
            output=Path(folder)/"fresh"
            for target,name,replacement in ((diagnostic,"load_gates",lambda *a:(stopped,{},full,{},science,{},{},{})),(diagnostic.gates.old,"fresh_output",lambda tag:output),(diagnostic.gates,"DATASETS",data_contract),(diagnostic,"bound",lambda item,*a:Path(item["path"])),(rt,"Meter",Meter),(io,"checked_payload",lambda item,*a:payloads[item["path"]]),(parent_io,"prior_file_records",lambda *a:[]),(prepare_inputs,"install_protected_test_guard",lambda:None)):
                patches.enter_context(patch.object(target,name,replacement))
            args=SimpleNamespace(tag="toy",full_result="full",full_audit="audit",stop_result="stop",stop_audit="audit")
            self.assertEqual(diagnostic.execute(args),0)
            result=json.loads((output/"RESULT.json").read_text(encoding="utf-8"))
            self.assertEqual((result["status"],result["completed_diagnostic_cells"],result["selected_states_diagnosed"]),("COMPLETE",24,12))
            self.assertEqual(len(list(output.glob("*_DIAGNOSTIC.npz"))),24)
            self.assertEqual(result["new_F_calls"],0)
            with self.assertRaises(FileExistsError): diagnostic.execute(args)


if __name__ == "__main__": unittest.main()
