"""Synthetic stdlib-only baseline guards; no real model, Task, NumPy or Torch."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location("legacy_baseline_tested",Path(__file__).with_name("legacy_baseline_cpu.py"))
legacy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(legacy)


def fake_metadata(baseline):
    pin=legacy.PINS[baseline]
    receipt=dict(status="COMPLETE_VALID",training_status="LIMITED_SCREEN",run_id=pin["run_id"],family=pin["family"],
        artifact_sha256=pin["artifact_sha256"],model_state_sha=pin["model_state_sha"],refit_fit_end=12579,
        search_fit_end=10482,refit_scaler_sha=legacy.SCALER,epochs=[.25],source_sha="source",
        artifact_cache_relative_path="runs/"+pin["run_id"]+"/selected_bundle.pt")
    source=dict(spec=dict(run_id=pin["run_id"],dataset="jena",family=pin["family"],latent_k=pin["latent_k"],seed=92601,fold="primary"),
        model_sha=legacy.FOUNDATION,source_sha="source",data_sha="data")
    manifest=dict(run_id=pin["run_id"],split="V_select",cohort="jena:primary:V_select:stride48",model_sha=pin["prediction_model_sha"],
        scaler_sha=legacy.SCALER,data_sha="data",archive_sha256=pin["V_archive_sha256"],metadata=dict(artifact_sha256=pin["artifact_sha256"]))
    return receipt,source,manifest


def fake_rows(baseline,batch):
    streams=legacy.PINS[baseline]["latent_k"]*batch
    return [dict(phase=phase,batch_size=batch,F_logical_streams=streams,F_physical_streams=streams,
        F_calls=(streams+127)//128,row_microbatch=128) for phase,count in (("warmup",10),("timed",20),("stage_probe",20)) for _ in range(count)]


class TinyTests(unittest.TestCase):
    def test_default_has_no_execution_or_input_read(self):
        stream=io.StringIO()
        with patch.object(legacy,"execute",side_effect=AssertionError("execution")),contextlib.redirect_stdout(stream):
            self.assertEqual(legacy.main([]),0)
        result=json.loads(stream.getvalue())
        self.assertEqual(result["input_reads"],0)
        self.assertFalse(result["torch_imported"])
        self.assertFalse(result["numpy_imported"])

    def test_draft_is_nonexecuting_and_unsealed(self):
        stream=io.StringIO()
        with patch.object(legacy,"execute",side_effect=AssertionError("execution")),contextlib.redirect_stdout(stream):
            self.assertEqual(legacy.main(["--print-spec-draft"]),0)
        spec=json.loads(stream.getvalue())
        self.assertEqual(spec["status"],"DRAFT_NOT_EXECUTABLE")
        with self.assertRaises(ValueError):legacy.validate_additive_spec(spec,"plan","entry")

    def test_additive_spec_accepts_only_exact_ROOT_seal(self):
        spec=legacy.spec_draft()
        spec.update(status="SEALED_BY_ROOT",priority_plan_sha256="plan",entrypoint_sha256="entry")
        legacy.validate_additive_spec(spec,"plan","entry")
        spec["F_row_microbatch"]=32
        with self.assertRaises(ValueError):legacy.validate_additive_spec(spec,"plan","entry")

    def test_no_unlisted_baseline_or_extra_hidden_grid(self):
        spec=legacy.spec_draft()
        spec.update(status="SEALED_BY_ROOT",priority_plan_sha256="plan",entrypoint_sha256="entry")
        spec["routes"]["OTHER"]=dict(spec["routes"]["PCA"])
        with self.assertRaises(ValueError):legacy.validate_additive_spec(spec,"plan","entry")

    def test_exact_canonical_route_matches_both_prior_preflights(self):
        for baseline in legacy.PINS:
            canonical=legacy.canonical_spec(baseline,*fake_metadata(baseline))
            self.assertEqual(legacy.fingerprint(canonical),legacy.PINS[baseline]["canonical_fingerprint"])
            self.assertEqual(canonical["canonical_fit_end"],12579)

    def test_PCA_K16_cannot_replace_K6(self):
        receipt,source,manifest=fake_metadata("PCA")
        source["spec"]["latent_k"]=16
        with self.assertRaises(ValueError):legacy.canonical_spec("PCA",receipt,source,manifest)

    def test_secondary_scaler_or_fold_cannot_replace_primary(self):
        receipt,source,manifest=fake_metadata("LORA")
        manifest["scaler_sha"]="secondary"
        with self.assertRaises(ValueError):legacy.canonical_spec("LORA",receipt,source,manifest)
        receipt,source,manifest=fake_metadata("LORA")
        source["spec"]["fold"]="secondary"
        with self.assertRaises(ValueError):legacy.canonical_spec("LORA",receipt,source,manifest)

    def test_adapter_or_V_archive_mismatch_rejected(self):
        receipt,source,manifest=fake_metadata("PCA")
        receipt["artifact_sha256"]="other"
        with self.assertRaises(ValueError):legacy.canonical_spec("PCA",receipt,source,manifest)
        receipt,source,manifest=fake_metadata("PCA")
        manifest["archive_sha256"]="other"
        with self.assertRaises(ValueError):legacy.canonical_spec("PCA",receipt,source,manifest)

    def test_limited_LoRA_is_not_relabelled_longer_training(self):
        receipt,source,manifest=fake_metadata("LORA")
        receipt["epochs"]=[.25,1.,3.]
        with self.assertRaises(ValueError):legacy.canonical_spec("LORA",receipt,source,manifest)

    def test_original_full_origin_split_never_recomputed_from_prefix_length(self):
        canonical=legacy.canonical_spec("PCA",*fake_metadata("PCA"))
        self.assertEqual(canonical["canonical_fit_end"],12579)
        self.assertNotEqual(canonical["canonical_fit_end"],int(16772*.6))

    def test_B1_B8_real_streams_and_row_call_counts(self):
        for baseline in legacy.PINS:
            for batch in (1,8):legacy.check_trial_counters(fake_rows(baseline,batch),baseline,batch)
        self.assertEqual(fake_rows("LORA",8)[0]["F_calls"],2)
        self.assertEqual(fake_rows("PCA",8)[0]["F_calls"],1)

    def test_missing_stage_trial_or_wrong_physical_count_rejected(self):
        with self.assertRaises(AssertionError):legacy.check_trial_counters(fake_rows("PCA",1)[:-1],"PCA",1)
        rows=fake_rows("LORA",8)
        rows[0]["F_physical_streams"]=21
        with self.assertRaises(AssertionError):legacy.check_trial_counters(rows,"LORA",8)

    def test_close_requires_actual_frozen_restore(self):
        class Parameter:
            requires_grad=False
            grad=None
        class Model:
            def parameters(self):return [Parameter()]
        class Foundation:
            model=Model()
        class Route:
            closed=False
            def close(self):self.closed=True
        route=Route()
        result=legacy.close_and_verify(route,Foundation(),lambda model:legacy.FOUNDATION)
        self.assertTrue(route.closed)
        self.assertEqual(result["status"],"PASS_EXACT_HASH")
        with self.assertRaises(AssertionError):legacy.close_and_verify(route,Foundation(),lambda model:"merged_wrong_hash")

    def test_missing_explicit_arguments_cannot_execute(self):
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):legacy.main(["--execute"])
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):legacy.main(["--execute","--print-spec-draft"])


if __name__=="__main__":unittest.main()
