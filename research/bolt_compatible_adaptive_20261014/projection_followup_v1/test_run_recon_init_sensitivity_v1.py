"""SOURCE-only RECON-initialization contracts; synthetic state and cached outputs."""
import json
import unittest
from unittest.mock import patch

import numpy as np

try:
    import run_recon_init_sensitivity_v1 as runner
except ModuleNotFoundError as error:
    if error.name != "run_recon_init_sensitivity_v1": raise
    runner=None


class ReconInitializationTests(unittest.TestCase):
    def api(self): self.assertIsNotNone(runner,"new two-seed RECON-initialization runner missing")

    def test_report_gate_closed_before_any_asset_read(self):
        self.api()
        with patch.object(runner,"REPORT_SHA",None), patch.object(runner.old,"record",side_effect=AssertionError("closedgate read")), self.assertRaisesRegex(ValueError,"CURRENT_OUTPUT_SPACE_REPORT_NOT_REGISTERED"):
            runner.load_gates("a","b","c","d","e")

    def test_old_trainer_prediction_functions_and_only_two_new_seeds(self):
        self.api(); import run_linear_e_adaptation_v1 as old
        self.assertIs(runner.train_condition,old.train_condition); self.assertIs(runner.predict_components,old.predict_components)
        self.assertEqual(runner.SEEDS,(92601,92602)); self.assertEqual(runner.LAGS,(0,1,2,4,8,16,32))

    def test_exact_prior_D_and_same_PCA_E_center_initializer(self):
        self.api(); e={"encoder":np.arange(8,dtype=np.float64).reshape(4,2),"center":np.arange(4,dtype=np.float64)}
        d={"decoder_FP32":np.arange(56,dtype=np.float32).reshape(14,4),"lags":np.array(runner.LAGS,np.int64)}
        old={"E":e["encoder"].astype(np.float32),"center":e["center"].astype(np.float32),"D":np.zeros((14,4),np.float32)}
        state=runner.initial_state(e,d,old)
        self.assertEqual(state["D"].tobytes(),d["decoder_FP32"].tobytes())
        self.assertEqual(state["E"].tobytes(),old["E"].tobytes()); self.assertEqual(state["center"].tobytes(),old["center"].tobytes())
        old["E"][0,0]+=.1
        with self.assertRaisesRegex(ValueError,"SAME_ORIGINAL_PCA_E_CENTER"):
            runner.initial_state(e,d,old)

    def test_two_locks_and_registry_precede_development_with_failed_seed_preserved(self):
        self.api(); events=[]; report={"datasets":{"robin":{"conditions":{}}},"failed_paths":[],"recon_initialization_training_fits":0,"training_updates_completed":0,"checkpoint_states_saved":0}
        def run(seed):
            events.append(("fit",seed))
            if seed==92601: raise ValueError("synthetic path failure")
            return {"status":"COMPLETE","training_updates_completed":600,"checkpoint_count":5}
        def seal(seed,row): events.append(("lock",seed,row["status"])); return {"path":str(seed),"sha256":"0"*64}
        def registry(rows): events.append(("registry",len(rows))); return {"path":"registry","sha256":"0"*64}
        resolved=runner.resolve_two(report,run,seal,registry)
        self.assertEqual(events[-1],("registry",2)); self.assertEqual(list(resolved),[92602]); self.assertEqual(report["resolved_selection_locks"],2)
        self.assertEqual(report["training_updates_completed"],600); self.assertEqual(len(report["failed_paths"]),1)
        self.assertTrue(report["all_2_locks_before_DEV"]); json.dumps(report,allow_nan=False)

    def test_CP0_cached_RECON_parity_has_zero_optimizer_and_saves_failed_components(self):
        self.api(); import causal_lag_decoder_v1 as pure
        rng=np.random.default_rng(5); z=rng.normal(size=(2,512,2)).astype(np.float32); f=rng.normal(size=(2,48,2)).astype(np.float32)
        current=rng.normal(size=(2,4)).astype(np.float32); d=rng.normal(size=(14,4)).astype(np.float32)
        reference={"latent_history":z,"actual_F_future":f,"ACTUAL_F_prediction_FP32":pure.predict(z,f,current,d,runner.LAGS)}
        actual={"prediction":reference["ACTUAL_F_prediction_FP32"].copy(),"latent_history":z.copy(),"latent_future":f.copy(),"lag_features":pure.lagged_delta_features(z,f,runner.LAGS)}
        row,arrays=runner.cp0_parity(actual,reference)
        self.assertEqual(row["status"],"PASS"); self.assertEqual(row["optimizer_updates"],0); self.assertEqual(len(arrays),8)
        actual["prediction"][0,0,0]+=.1; bad,saved=runner.cp0_parity(actual,reference)
        self.assertEqual(bad["status"],"FAILED_PARITY_PRESERVED"); self.assertIn("Torch_prediction",saved)
        self.assertEqual((bad["checks"]["prediction"]["atol"],bad["checks"]["prediction"]["rtol"]),(1e-5,2e-6))


if __name__=="__main__": unittest.main()
