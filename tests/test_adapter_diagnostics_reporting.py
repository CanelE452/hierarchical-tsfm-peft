"""Report extraction preserves real result schemas and missing measurements."""
import tempfile
from pathlib import Path
import unittest

from research.adapter_diagnostics_20261017_v2 import reporting
from research.adapter_diagnostics_20261017_v2.core import atomic_json, seal_plan


class ReportingTests(unittest.TestCase):
    def test_rows_wrapped_native_raw_and_cost_ms_schema_are_collected(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo, prefix="report_test_") as directory:
            report = Path(directory)
            jobs = [dict(job_id="D03_jena", question_id="D03", family="CHRONOS2_NATIVE_RAW", dataset_cohort="jena",
                         seed=None, K=21, device="cuda", scientific_identity="synthetic-native"),
                    dict(job_id="E01_jena_matched", question_id="E02", family="JENA_MATCHED_COST", dataset_cohort="jena",
                         seed=None, K=6, device="cpu", scientific_identity="synthetic-cost")]
            atomic_json(report/"EXPERIMENT_PLAN.json", seal_plan(dict(jobs=jobs,source_hashes={})))
            atomic_json(report/"receipts/D03_jena.json", dict(status="COMPLETE", result=dict(rows=[
                dict(method="Chronos2_nativeRaw", backbone="chronos2", C=21, K=21, L=512, H=48,
                     MSE=1.25, MAE=.75, split="V_select", point_statistic="quantile_0.5")])) )
            atomic_json(report/"receipts/E01_jena_matched.json", dict(status="COMPLETE", result=dict(
                cpu_heavy_workers_during_cost=1, other_campaign_workers_observed=[], rows=[
                    dict(method="B2", B=8, status="COMPLETE", median_ms_batch=16, median_ms_origin=2,
                         IQR_ms_batch=1, process_RSS_peak_sampled_bytes=123, RSS_scope="sampled shared process",
                         total_resident_parameters=100,core_dtype="float32",resident_parameter_scope="per-route weights",
                         shared_process_resident_parameters=200,shared_process_parameter_scope="one F plus all resident adapters"),
                    dict(method="RAW", B=8, status="FAILED_PARITY", median_ms_batch=None, median_ms_origin=None)])))
            value = reporting.collect(dict(repo=repo, report=report))
            native = [row for row in value["runs"] if row["method"] == "Chronos2_nativeRaw"]
            self.assertEqual(len(native), 1)
            self.assertEqual(native[0]["MSE"], 1.25)
            self.assertEqual(value["costs"][0]["median_ms_per_origin"], 2)
            self.assertTrue(value["costs"][0]["exclusive"])
            self.assertIsNone(value["costs"][1]["median_ms_per_origin"])
            self.assertEqual(value["costs"][0]["core_dtype"],"float32")
            self.assertEqual(value["costs"][0]["resident_parameter_scope"],"per-route weights")
            self.assertEqual(value["costs"][0]["shared_process_resident_parameters"],200)
            self.assertEqual(value["costs"][0]["shared_process_parameter_scope"],"one F plus all resident adapters")
            self.assertIsNone(value["costs"][1]["shared_process_resident_parameters"])

    def test_public_reaggregation_uses_channel_macro_with_unequal_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            reporting.csv_write(path/"ERROR_BY_CHANNEL.csv", [
                dict(run_id="toy", channel_id="a", SSE=100, SAE=20, count=100),
                dict(run_id="toy", channel_id="b", SSE=9, SAE=3, count=1)])
            score = reporting.public_reaggregate(path)["toy"]
            self.assertEqual(score["MSE"], 5)
            self.assertEqual(score["MAE"], 1.6)

    def test_public_horizon_sums_match_marginal_and_reject_missing_or_changed_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            reporting.csv_write(path/"RUNS_FULL_PRECISION.csv", [dict(run_id="toy", H=2)])
            reporting.csv_write(path/"ERROR_BY_CHANNEL.csv", [dict(run_id="toy", channel_id="0", SSE=5, SAE=3, count=2)])
            cells = [dict(run_id="toy", horizon=1, channel_id="0", SSE=1, SAE=1, count=1),
                     dict(run_id="toy", horizon=2, channel_id="0", SSE=4, SAE=2, count=1)]
            reporting.csv_write(path/"ERROR_BY_HORIZON.csv", cells)
            self.assertEqual(reporting.public_horizon_consistency(path)["checked_channels"], 1)
            reporting.csv_write(path/"ERROR_BY_HORIZON.csv", cells[:1])
            with self.assertRaisesRegex(ValueError, "Missing public horizon"):
                reporting.public_horizon_consistency(path)
            reporting.csv_write(path/"ERROR_BY_HORIZON.csv", [cells[0], dict(cells[1], SSE=5)])
            with self.assertRaisesRegex(ValueError, "SSE or SAE"):
                reporting.public_horizon_consistency(path)

    def test_large_fp64_reduction_uses_observed_mean_units_and_preserves_raw_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            count = 476304
            reporting.csv_write(path/"RUNS_FULL_PRECISION.csv", [dict(run_id="large", H=2)])
            reporting.csv_write(path/"ERROR_BY_CHANNEL.csv", [dict(run_id="large", channel_id="0", SSE=453192.4737698426, SAE=300000, count=count)])
            cells = [dict(run_id="large", horizon=h, channel_id="0", SSE=453192.4737697751/2, SAE=150000, count=count//2) for h in (1, 2)]
            reporting.csv_write(path/"ERROR_BY_HORIZON.csv", cells)
            check = reporting.public_horizon_consistency(path)
            self.assertGreater(check["maximum_statistic_absolute_difference"], 1e-8)
            self.assertLess(check["maximum_channel_mean_absolute_difference"], 1e-10)
            reporting.csv_write(path/"ERROR_BY_HORIZON.csv", [cells[0], dict(cells[1], SSE=cells[1]["SSE"]+count*1e-8)])
            with self.assertRaisesRegex(ValueError, "SSE or SAE"):
                reporting.public_horizon_consistency(path)

    def test_decoder_comparison_reference_is_shared_refit(self):
        common = dict(cohort="jena", seed=92601, metric_scale="canonical_A60", source_sha256="a"*64,
                      C=21, L=512, H=48, split="V_select", point_statistic="quantile_0.5",
                      device="cpu", backend="torch_eager", dtype="float64",
                      **{key: "b"*64 for key in ("scaler_sha256", "target_sha256", "mask_sha256", "origins_sha256", "channel_sha256")})
        runs = [dict(common, run_id="B03_s92601_0", method="B2_DECODER_SHARED", MSE=2, MAE=1),
                dict(common, run_id="B03_s92601_1", method="B2_DECODER_HORIZON", MSE=1, MAE=.5)]
        comparison = reporting.comparisons(dict(runs=runs))[0]
        self.assertEqual(comparison["reference"], "B2_DECODER_SHARED")
        self.assertEqual(comparison["delta_MSE_pct"], -50)

    def test_registered_primary_pairs_include_cap_raw_ridge_with_honest_reference_seed(self):
        common = dict(cohort="jena", metric_scale="canonical_A60", C=21,L=512,H=48,split="V_select",
            point_statistic="quantile_0.5",device="cpu",backend="torch_eager",dtype="float32",
            **{key:"a"*64 for key in ("scaler_sha256","target_sha256","mask_sha256","origins_sha256","channel_sha256")})
        runs = [dict(common,run_id=method+"_source",method=method,seed=92601 if method in ("Raw","Ridge") else 92602,MSE=2.,MAE=1.)
                for method in ("Raw","Ridge","B0","CAP","B2")]
        runs[1]["point_statistic"]="deterministic_ridge"
        data = dict(runs=runs)
        exact=reporting.comparisons(data)
        contextual=reporting.comparisons(data,contextual=True)
        self.assertEqual({(r["reference"],r["method"]) for r in exact+contextual}, {("Raw","B2"),("Ridge","B2"),("B0","B2"),("CAP","B2")})
        raw=next(r for r in exact if r["reference"]=="Raw")
        self.assertEqual(raw["reference_seed"],92601)
        self.assertEqual(raw["method_seed"],92602)
        self.assertIsNone(raw["reference_supervision_seed"])
        self.assertIn("fixed saved",raw["pairing_rule"])
        self.assertEqual(next(r for r in contextual if r["reference"]=="Ridge")["match_status"],"UNVERIFIED_CONTEXTUAL")

    def test_same_update_contrasts_preserve_cp0_and_require_exact_learning_progress(self):
        common=dict(cohort="jena",seed=92601,split="V_select",metric_scale="canonical_A60",C=21,L=512,H=48,
            fit="A_fit_search",device="cuda",backend="torch_eager",dtype="float32",point_statistic="quantile_0.5",
            source_file="reports/source.json",source_sha256="f"*64,
            **{key:"a"*64 for key in ("scaler_sha256","target_sha256","mask_sha256","origins_sha256","channel_sha256")})
        endpoints=[]
        for epoch,updates,exposures in ((0.,0,0),(.25,156,2481)):
            for method in ("B0","B2","B2_CAP_POINTWISE"):
                endpoints.append(dict(common,run_id=method+str(epoch),method=method,epoch=epoch,updates=updates,exposures=exposures,
                    MSE=1.04 if epoch and method=="B2" else 1.,MAE=.49 if epoch and method=="B2" else .5))
        compared=reporting.same_update_comparisons(dict(endpoints=endpoints))
        self.assertEqual(len(compared),6)
        zero=[row for row in compared if row["epoch"]==0]
        self.assertEqual(len(zero),3)
        self.assertTrue(all(row["delta_MSE"]==0 and row["delta_MAE"]==0 for row in zero))
        cap=next(row for row in compared if row["epoch"]==.25 and row["reference"]=="CAP" and row["method"]=="B2")
        self.assertAlmostEqual(cap["delta_MSE_pct"],4.)
        self.assertAlmostEqual(cap["delta_MAE_pct"],-2.)
        self.assertEqual(cap["comparison_scope"],"COMMON_SEARCH_ENDPOINT")
        self.assertEqual(cap["match_status"],"EXACT_MATCHED")
        interpretation=reporting.common_endpoint_interpretation(dict(runs=[]),compared)
        self.assertIn("일관된 우위가 아니다",interpretation)
        self.assertIn("selected fresh-A60",interpretation)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"SAME_UPDATE_RESULTS.csv"
            reporting.csv_write(path,endpoints)
            restored=reporting.same_update_comparisons(dict(endpoints=reporting.rows(path)))
            self.assertEqual(len(restored),6)
            self.assertTrue(all(row["match_status"]=="EXACT_MATCHED" for row in restored))
        endpoints[-1]["scaler_sha256"]="b"*64
        contextual=reporting.same_update_comparisons(dict(endpoints=endpoints))
        self.assertEqual(next(row for row in contextual if row["epoch"]==.25 and row["reference"]=="CAP" and row["method"]=="B2")["match_status"],"UNVERIFIED_CONTEXTUAL")
        endpoints[-1]["updates"]=157
        self.assertEqual(len(reporting.same_update_comparisons(dict(endpoints=endpoints))),4)

    def test_missing_contract_hashes_are_only_contextual(self):
        common = dict(cohort="jena", seed=92601, metric_scale="canonical_A60")
        runs = [dict(common, run_id="B03_0", method="B2_DECODER_SHARED", MSE=2, MAE=1),
                dict(common, run_id="B03_1", method="B2_DECODER_HORIZON", MSE=1, MAE=.5)]
        self.assertEqual(reporting.comparisons(dict(runs=runs)), [])
        value = reporting.comparisons(dict(runs=runs), contextual=True)[0]
        self.assertEqual(value["match_status"], "UNVERIFIED_CONTEXTUAL")
        self.assertIn("MISSING_OR_DIFFERENT_channel_sha256", value["unresolved_contract_fields"])

    def test_pems32_legacy_label_does_not_merge_with_full_cohort(self):
        import hashlib
        import numpy as np
        repo = Path(__file__).resolve().parents[1]
        selected = reporting.read(repo/"research"/reporting.BROAD/"priority_pems_subset_20261016_v1/PLAN.json")["column_selection"]["selected_ids"]
        row = dict(cohort="pems08:C32_hash:primary:V_select:stride12", C=32, L=512, H=12,
                   channel_sha256=hashlib.sha256(np.asarray(selected, dtype=str).tobytes()).hexdigest())
        value = reporting.canonical_cohort(row, repo)
        self.assertEqual(value["cohort"], "pems32")
        self.assertEqual(value["channel_order_validation"], "EXACT_REGISTERED_ORDER")
        with self.assertRaises(ValueError):
            reporting.canonical_cohort(dict(row, channel_sha256="0"*64), repo)

    def test_deterministic_native_cost_keeps_seed_na_and_joins_by_reference(self):
        common = dict(cohort="jena", C=21, L=512, H=48, MSE=1., MAE=.5,
                      method="Chronos2_nativeRaw", run_id="D03_jena_0", seed=None)
        cost = dict(cohort="jena", C=21, L=512, H=48, method="Chronos2_nativeRaw",
                    supervision_seed=None, evaluation_reference_seed=92601, accuracy_MSE=1., accuracy_MAE=.5)
        self.assertIs(reporting.cost_accuracy_match(cost, [common]), common)
        self.assertIsNone(reporting.cost_accuracy_match(dict(cost, accuracy_MSE=2.), [common]))

    def test_e03_alias_and_exclusive_schema_and_a03_source_metadata(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo, prefix="report_test_") as directory:
            report = Path(directory)
            jobs = [dict(job_id="E03_jena", question_id="E03", family="BACKBONE_COST", dataset_cohort="jena", seed=None,
                         K=6, device="cuda", scientific_identity="synthetic-cost"),
                    dict(job_id="A03_fake", question_id="A03", family="B2", dataset_cohort="jena", seed=92601,
                         K=6, device="cuda", scientific_identity="synthetic-endpoint")]
            atomic_json(report/"EXPERIMENT_PLAN.json", seal_plan(dict(jobs=jobs,source_hashes={})))
            atomic_json(report/"receipts/E03_jena.json", dict(status="COMPLETE", result=dict(measurements_exclusive=True, rows=[
                dict(method="Bolt_Raw", B=1, status="COMPLETE", median_ms_batch=2, median_ms_origin=2)])))
            atomic_json(report/"receipts/A03_fake.json", dict(status="COMPLETE", result=dict(model="B2", MSE=1, MAE=.5,
                fit="A_fit_search", selection="NOT_SELECTED_COMMON_ENDPOINT", dtype="float32", new_forward=True,
                eval="V_select", metric_scale="canonical_A60", channel_sse=[1.]*21, channel_sae=[.5]*21,
                channel_counts=[1]*21, target_sha256="a"*64)))
            value = reporting.collect(dict(repo=repo, report=report))
            self.assertEqual(value["costs"][0]["method"], "Raw")
            self.assertTrue(value["costs"][0]["exclusive"])
            self.assertIsNone(value["costs"][0]["seed"])
            endpoint = value["endpoints"][0]
            self.assertEqual(endpoint["fit"], "A_fit_search")
            self.assertEqual(endpoint["refit"], "NOT_APPLICABLE_SEARCH_ENDPOINT")
            self.assertEqual(endpoint["selection"], "NOT_SELECTED_COMMON_ENDPOINT")
            self.assertEqual(endpoint["dtype"], "float32")
            self.assertEqual(endpoint["G6_channel_coverage"], "PUBLIC_SUFFICIENT_STATS")

    def test_raw_backbones_have_no_target_selection_and_ppt_preserves_native_routing(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo, prefix="report_test_") as directory:
            report = Path(directory)
            jobs = [dict(job_id="D02_fake", question_id="D02", family="MOIRAI_TRIO", dataset_cohort="jena", seed=92601,
                         K=6, device="cuda", scientific_identity="synthetic-moirai"),
                    dict(job_id="D03_fake", question_id="D03", family="NATIVE_RAW", dataset_cohort="jena", seed=None,
                         K=21, device="cuda", scientific_identity="synthetic-native")]
            atomic_json(report/"EXPERIMENT_PLAN.json", seal_plan(dict(jobs=jobs,source_hashes={})))
            common = dict(C=21, K=21, L=512, H=48, MSE=1., MAE=.5, split="V_select",
                          fit="NO_TARGET_PARAMETER_LEARNING", trainable_params=0, refit_updates=0, refit_exposures=0)
            atomic_json(report/"receipts/D02_fake.json", dict(status="COMPLETE", result=dict(rows=[dict(common, method="Moirai_Raw")])) )
            atomic_json(report/"receipts/D03_fake.json", dict(status="COMPLETE", result=dict(rows=[dict(common,
                method="Chronos2_nativeRaw", routing="native_multivariate_per_origin")])) )
            value = reporting.collect(dict(repo=repo, report=report))
            raw = next(r for r in value["runs"] if r["run_id"] == "D02_fake_0")
            self.assertEqual(raw["fit"], "NO_TARGET_PARAMETER_LEARNING")
            self.assertEqual(raw["selection"], "NO_TARGET_SELECTION")
            self.assertEqual(raw["refit"], "NO_TARGET_PARAMETER_REFIT")
            self.assertEqual(raw["updates"], 0)
            atomic_json(report/"DATA_AUDIT.json", dict(datasets={}))
            reporting._documents(dict(report=report), value, [])
            table = reporting.rows(report/"PPT_T1_ACCURACY_COST.csv")
            native = next(r for r in table if r["method"] == "Chronos2_nativeRaw")
            self.assertIn("joint", native["F_routing"])
            self.assertNotIn("independent univariate", native["F_routing"])
            moirai = next(r for r in table if r["method"] == "Moirai_Raw")
            self.assertIn("C independent univariate", moirai["F_routing"])
            self.assertIn("No F", reporting.f_routing(dict(method="Ridge", backbone="NONE")))
            self.assertIn("K latent", reporting.f_routing(dict(method="B2", backbone="Chronos-Bolt-small")))

    def test_legacy_direct_g_has_no_backbone_and_dict_cost_parity_is_normalized(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo, prefix="report_test_") as directory:
            report = Path(directory)
            job = dict(job_id="E01_fake", question_id="E01", family="B0", dataset_cohort="pems32", seed=92601,
                       K=8, device="cpu", scientific_identity="synthetic-cost")
            atomic_json(report/"EXPERIMENT_PLAN.json", seal_plan(dict(jobs=[job],source_hashes={})))
            parity = dict(status="PASS", atol=1e-5, rtol=1e-4)
            atomic_json(report/"receipts/E01_fake.json", dict(status="COMPLETE", result=dict(
                batch_size=1, median_seconds=.1, forward_parity=parity, exclusive_queue_owned_by_caller=True,
                process_RSS_sampled_peak_bytes=2000, process_RSS_bytes=1234)))
            value = reporting.collect(dict(repo=repo, report=report))
            self.assertEqual(value["costs"][0]["parity_state"], "PASS")
            self.assertEqual(value["costs"][0]["source_forward_parity"], parity)
            self.assertEqual(value["costs"][0]["process_RSS_peak_bytes"], 2000)
            self.assertEqual(value["costs"][0]["process_RSS_snapshot_bytes"], 1234)
            direct = [r for r in value["runs"] if "_DIRECT_G_" in r["run_id"].upper()]
            self.assertTrue(direct)
            for run in direct:
                self.assertEqual(run["backbone"], "NONE")
                self.assertEqual(run["point_statistic"], "deterministic_point_regression")
                self.assertEqual(run["source_family"], "DIRECT_G")
                self.assertIn("No F", reporting.f_routing(run))
            exact = reporting.comparisons(value)
            self.assertFalse(any(c["method"].startswith("G") for c in exact))

    def test_handler_bound_c01_saved_prediction_links_cost_and_actual_gzip_geometry(self):
        import csv
        import gzip
        import numpy as np
        from research.adapter_diagnostics_20261017_v2.existing_jobs import _identity
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo, prefix="report_test_") as directory:
            root = Path(directory); report = root/"report"; cache = root/"cache"; cache.mkdir()
            first = dict(job_id="C01_fake", question_id="C01", family="B0", dataset_cohort="pems32", seed=92601,
                         K=8, device="cpu", scientific_identity="synthetic-resume")
            second = dict(first, job_id="E01_fake", question_id="E01", scientific_identity="synthetic-cost")
            atomic_json(report/"EXPERIMENT_PLAN.json", seal_plan(dict(jobs=[first, second],source_hashes={})))
            columns = reporting.read(repo/"research"/reporting.BROAD/"priority_pems_subset_20261016_v1/PLAN.json")["column_selection"]["selected_ids"]
            archive = cache/"selected_V.npz"
            np.savez_compressed(archive, pred_normalized=np.ones((1,12,32), np.float32), target_normalized=np.zeros((1,12,32), np.float32),
                                observed=np.ones((1,12,32), bool), origins=np.array([12500]), channel_ids=np.asarray(columns, dtype=str))
            digest = reporting.sha(archive)
            handler = report/"jobs/C01_fake_handler.json"
            atomic_json(handler, dict(status="COMPLETE", job_id=first["job_id"], scientific_identity_sha256=_identity(first),
                                     artifact_hashes=[dict(path="cache://selected_V.npz", sha256=digest)]))
            atomic_json(report/"receipts/C01_fake.json", dict(status="COMPLETE", artifact_hashes={"jobs/C01_fake_handler.json":reporting.sha(handler)},
                result=dict(MSE=1., MAE=1., C=32, L=512, H=12, params=1234, private_prediction_archive_sha256=digest)))
            trial = report/"costs/COST_TRIALS.csv.gz"; trial.parent.mkdir()
            with gzip.open(trial, "wt", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["F_logical_streams","F_physical_streams","F_calls"])
                writer.writeheader(); writer.writerows([dict(F_logical_streams=8,F_physical_streams=8,F_calls=1)]*50)
            atomic_json(report/"receipts/E01_fake.json", dict(status="COMPLETE", artifact_hashes={"costs/COST_TRIALS.csv.gz":reporting.sha(trial)},
                result=dict(status="COMPLETE_PROTOCOL", batch_size=1, median_seconds=.1, trials_sha256=reporting.sha(trial),
                            prediction_archive_sha256=digest, forward_parity=dict(status="PASS"), exclusive_queue_owned_by_caller=True)))
            value = reporting.collect(dict(repo=repo, report=report, cache=cache))
            cost = value["costs"][0]
            run = next(r for r in value["runs"] if r["run_id"] == "C01_fake")
            self.assertEqual(run["independent_private_score_sha256"], digest)
            self.assertEqual(cost["logical_F_series_rows"], 8)
            self.assertEqual(cost["trainable_params"], 1234)
            self.assertIs(reporting.cost_accuracy_match(cost, value["runs"]), run)
            self.assertIn(handler.relative_to(repo).as_posix(), value["bindings"])
            archive.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "binding rejected"):
                reporting.collect(dict(repo=repo, report=report, cache=cache))

    def test_g5_accepts_actual_list_profile_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory)
            job = dict(job_id="C00_fake", question_id="C00")
            atomic_json(report/"receipts/C00_fake.json", dict(status="COMPLETE", result=dict(profiles=[
                dict(family="B0", source_bundle_unchanged_after_both_profiles=True,
                     next_update_branch_audit=dict(identical_initial_model_optimizer_sampler_CPU_RNG=True))])))
            evidence = reporting.resume_evidence(dict(report=report), dict(jobs=[job]))
            self.assertTrue(evidence[0]["disposable_next_update_profiles"]["B0"]["source_bundle_unchanged_after_both_profiles"])

    def test_probe_late_binding_proves_reference_contract_and_rejects_later_mutation(self):
        import numpy as np
        from research.adapter_diagnostics_20261017_v2.existing_jobs import _identity, _array_fingerprint
        repo=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo,prefix="report_test_") as directory:
            root=Path(directory); report=root/"report"; cache=root/"cache"; cache.mkdir()
            job=dict(job_id="B01_fake",question_id="B01",family="B0",dataset_cohort="jena",seed=92601,K=6,device="cpu",scientific_identity="synthetic-probe")
            columns=reporting.read(repo/"research"/reporting.BROAD/"DATA_CONTRACT.json")["assets"]["jena"]["columns"]
            origin=np.arange(14688,14688+43*48,48,dtype=np.int64)
            target=np.zeros((43,48,21),np.float32); mask=np.ones(target.shape,bool)
            reference_values=dict(pred_normalized=np.ones_like(target),target_normalized=target,observed=mask,origins=origin,channel_ids=np.asarray(columns,dtype=str))
            reference=cache/"reference.npz"; np.savez(reference,**reference_values)
            ref_public=report/"REFERENCE.json"; atomic_json(ref_public,dict(sha256=reporting.sha(reference)))
            archive=cache/"probes/B01_fake/V_PREDICTIONS_FP64.npz"; archive.parent.mkdir(parents=True)
            np.savez(archive,prediction=np.ones(target.shape,np.float64),y=target.astype(np.float64),mask=mask,origins=origin)
            coefficient=archive.with_name("RIDGE_COEFFICIENTS_FP64.npz"); coefficient.write_bytes(b"coefficient-placeholder")
            result=dict(metadata=dict(scaler_sha256="f"*64),MSE=1.,MAE=1.,channel_SSE=[2064.]*21,channel_SAE=[2064.]*21,channel_observed=[2064]*21)
            result_path=report/"probes/B01_fake/RESULT.json"; atomic_json(result_path,result)
            handler=report/"jobs/B01_fake_handler.json"
            atomic_json(handler,dict(status="COMPLETE",job_id=job["job_id"],scientific_identity_sha256=_identity(job),artifact_hashes=[
                dict(path="report://probes/B01_fake/RESULT.json",sha256=reporting.sha(result_path)),
                dict(path="cache://probes/B01_fake/RIDGE_COEFFICIENTS_FP64.npz",sha256=reporting.sha(coefficient))]))
            receipt=report/"receipts/B01_fake.json"
            atomic_json(receipt,dict(status="COMPLETE",result=result,artifact_hashes={"probes/B01_fake/RESULT.json":reporting.sha(result_path),"jobs/B01_fake_handler.json":reporting.sha(handler)}))
            reference_run=dict(run_id="reference",cohort="jena",origin="EXISTING_RESULT_REUSED",method="B0",scaler_sha256="f"*64,
                _archive_path=reference,independent_private_score_sha256=reporting.sha(reference),source_file=ref_public.relative_to(repo).as_posix(),source_sha256=reporting.sha(ref_public),
                **reporting.contract_metadata(dict(array_fingerprints={k:_array_fingerprint(v) for k,v in reference_values.items()})))
            probe=dict(run_id=job["job_id"],cohort="jena",C=21,L=512,H=48,method="B0_FULL_HISTORY_PROBE",seed=92601,MSE=1.,MAE=1.,
                scaler_sha256="f"*64,source_file=receipt.relative_to(repo).as_posix(),source_sha256=reporting.sha(receipt))
            data=dict(runs=[reference_run,probe],bindings={},horizon=[],channel=[],new_stats_replay=[])
            ctx=dict(repo=repo,report=report,cache=cache)
            reporting.complete_probe_metadata(ctx,{job["job_id"]:(job,result,receipt)},data)
            replay=data["new_stats_replay"][0]
            self.assertTrue(replay["binding_captured_at_verification"])
            self.assertFalse(replay["historical_prediction_hash_binding"])
            self.assertEqual(probe["target_sha256"],reference_run["target_sha256"])
            self.assertNotEqual(replay["archive_arrays"]["y"]["sha256"],probe["target_sha256"])
            self.assertEqual(len(data["horizon"]),48*21)
            atomic_json(report/"NEW_STATS_REPLAY.json",dict(rows=data["new_stats_replay"]))
            self.assertTrue(reporting.replay_binding_validation(ctx)["all_bound_artifacts_verified"])
            np.savez(archive,prediction=np.ones(target.shape,np.float64),y=np.full(target.shape,2.,np.float64),mask=mask,origins=origin)
            self.assertFalse(reporting.replay_binding_validation(ctx)["all_bound_artifacts_verified"])
            with self.assertRaisesRegex(ValueError,"late-bound"):
                reporting.complete_probe_metadata(ctx,{job["job_id"]:(job,result,receipt)},data)
            (report/"NEW_STATS_REPLAY.json").unlink()
            with self.assertRaisesRegex(ValueError,"target"):
                reporting.complete_probe_metadata(ctx,{job["job_id"]:(job,result,receipt)},data)

    def test_g2_reads_bound_d01_producer_and_preserves_native_not_applicable_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            report=Path(directory)
            jobs=[]
            for backbone,routing,gradient in (("moirai2","independent_univariate","PASS"),("chronos2","native_multivariate_per_origin","NOT_APPLICABLE_NATIVE_RAW")):
                identifier="D01_"+backbone
                jobs.append(dict(job_id=identifier,question_id="D01",scientific_identity=backbone))
                result=dict(status="PASS",backbone=backbone,shape=[2,48,3],dtype="float32",device="cuda",point_statistic="quantile_0.5",
                    routing=routing,official_parity_max_abs=0.,mask_parity_max_abs=0.,tolerance=dict(atol=2e-5,rtol=2e-5),
                    routing_sentinel=dict(passed=True,origin_isolation=True,split_merge=True,channel_isolation=True if backbone=="moirai2" else None),
                    adapter_gradient=dict(status=gradient),frozen_weights_unchanged=True,save_load_prediction_parity=True,trainable_backbone_parameters=0,
                    original_scaler_forward_parity=dict(status="PASS") if backbone=="moirai2" else None,private_artifacts=[])
                artifact=report/"jobs"/(identifier+".json"); atomic_json(artifact,result)
                atomic_json(report/"receipts"/(identifier+".json"),dict(status="COMPLETE",scientific_identity=backbone,result=result,
                    artifact_hashes={"jobs/"+identifier+".json":reporting.sha(artifact)}))
            value=reporting.method_evidence(dict(report=report),dict(jobs=jobs))
            self.assertEqual(value["status"],"PASS_RECORDED_D01_APPLICABLE_SCOPE")
            self.assertEqual(len(value["backbone_smoke"]),2)
            native=next(row for row in value["backbone_smoke"] if row["backbone"]=="chronos2")
            self.assertEqual(native["adapter_gradient"]["status"],"NOT_APPLICABLE_NATIVE_RAW")
            self.assertEqual(native["optimizer_scope"]["status"],"NOT_APPLICABLE_NATIVE_RAW")
            self.assertIsNone(native["routing_sentinel"]["channel_isolation"])
            self.assertEqual(native["private_binding_verification_gate"],"G7_CACHE")
            (report/"jobs/D01_chronos2.json").write_text("{}",encoding="utf-8")
            self.assertEqual(reporting.method_evidence(dict(report=report),dict(jobs=jobs))["status"],"FAIL_D01_PUBLIC_BINDING")

    def test_changed_sealed_source_is_rejected_before_legacy_or_private_reader(self):
        from unittest.mock import patch
        repo=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo,prefix="report_test_") as directory:
            report=Path(directory)
            source=report/"sealed_source.txt"; source.write_text("before",encoding="utf-8")
            plan=seal_plan(dict(jobs=[],source_hashes={source.relative_to(repo).as_posix():reporting.sha(source)}))
            atomic_json(report/"EXPERIMENT_PLAN.json",plan)
            source.write_text("after",encoding="utf-8")
            with patch.object(reporting,"_legacy",side_effect=AssertionError("legacy reader must not run")) as legacy, \
                 patch.object(reporting,"legacy_statistics",side_effect=AssertionError("private reader must not run")) as private:
                with self.assertRaisesRegex(ValueError,"Sealed source changed"):
                    reporting.collect(dict(repo=repo,report=report))
                legacy.assert_not_called(); private.assert_not_called()

    def test_required_pre_t_audit_failure_fails_overall_verify_but_m5_block_is_allowed(self):
        repo=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo,prefix="report_test_") as directory:
            report=Path(directory); ctx=dict(repo=repo,report=report)
            audit=dict(status="PASS_WITH_M5_BLOCK",datasets={},synthetic_G1_G4=dict(status="PASS"))
            atomic_json(report/"DATA_AUDIT.json",audit)
            (report/"DATA_AUDIT_KO.md").write_text("Synthetic fixture",encoding="utf-8")
            reporting._documents(ctx,dict(jobs=[],runs=[],endpoints=[],costs=[],bindings={}),[])
            for name in ("JOB_STATUS","RUNS_FULL_PRECISION","SAME_UPDATE_RESULTS","SAME_UPDATE_COMPARISONS","MATCHED_COMPARISONS",
                         "CONTEXTUAL_COMPARISONS","COST_FULL_PRECISION","ERROR_BY_HORIZON","ERROR_BY_CHANNEL","REPRESENTATION_PROBES","INITIALIZATION_DIAGNOSTICS"):
                reporting.csv_write(report/(name+".csv"),[])
            atomic_json(report/"EXPERIMENT_PLAN.json",seal_plan(dict(jobs=[],source_hashes={})))
            atomic_json(report/"SOURCE_BINDINGS.json",dict(public_inputs={}))
            atomic_json(report/"NEW_STATS_REPLAY.json",dict(rows=[]))
            atomic_json(report/"RESOURCE_LEDGER.json",{})
            reporting.horizon_bindings(report,[])
            (report/"figures").mkdir()
            for figure in reporting.FIGURES:
                for extension in ("png","svg"):
                    (report/"figures"/(figure+"."+extension)).write_bytes(b"synthetic existence fixture")
            self.assertEqual(reporting.verify(ctx,write=True)["status"],"PASS")
            atomic_json(report/"DATA_AUDIT.json",dict(audit,status="PARTIAL_OR_FAILED"))
            with self.assertRaisesRegex(ValueError,"Required pre-T data audit failed"):
                reporting.verify(ctx,write=True)
            validation=reporting.read(report/"VALIDATION.json")
            self.assertEqual(validation["status"],"FAIL")
            self.assertEqual(validation["exit_code"],2)
            self.assertEqual(next(row["status"] for row in validation["checks"] if row["gate"]=="G1_DATA"),"FAIL_REQUIRED_PRE_T_AUDIT")

    def test_fit_metadata_reader_never_opens_storage_or_executes_unlisted_global(self):
        import os
        import pickle
        import zipfile
        from unittest.mock import patch
        class Forbidden:
            def __reduce__(self):
                return os.system, ("must not execute",)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"bundle.pt"
            with zipfile.ZipFile(path,"w") as archive:
                archive.writestr("bundle/data.pkl",pickle.dumps(dict(search_summary=dict(seconds=2.),refit_summary=dict(seconds=3.),spec={})))
                archive.writestr("bundle/data/0",b"storage must stay unopened")
            original=zipfile.ZipFile.open
            opened=[]
            def metadata_only(archive,name,*args,**kwargs):
                key=name.filename if isinstance(name,zipfile.ZipInfo) else name
                opened.append(key)
                self.assertTrue(key.endswith("/data.pkl"))
                return original(archive,name,*args,**kwargs)
            with patch.object(zipfile.ZipFile,"open",metadata_only):
                value=reporting.fit_bundle_metadata(path)
            self.assertEqual(value["search_summary"]["seconds"],2.)
            self.assertEqual(opened,["bundle/data.pkl"])
            with zipfile.ZipFile(path,"w") as archive:
                archive.writestr("bundle/data.pkl",pickle.dumps(Forbidden()))
            with patch.object(os,"system") as system:
                with self.assertRaisesRegex(ValueError,"Unsupported fit metadata global"):
                    reporting.fit_bundle_metadata(path)
                system.assert_not_called()

    def test_bound_fit_stages_join_cost_with_separate_fit_device_and_reject_changes(self):
        import pickle
        import zipfile
        repo=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo,prefix="report_test_") as directory:
            root=Path(directory);report=root/"report";cache=root/"cache"
            run_id="moirai_jena_B0_k6_s92601"
            path=cache/"runs"/run_id/"selected_bundle.pt";path.parent.mkdir(parents=True)
            search=dict(status="SEGMENT_COMPLETE",seconds=2.,selected_epoch=1.)
            refit=dict(status="SEGMENT_COMPLETE",seconds=3.,updates=4,exposures=5,contract=dict(scaler_sha="s"))
            payload=dict(search_summary=search,refit_summary=refit,spec=dict(run_id=run_id,family="B0",seed=92601,device="cuda"))
            with zipfile.ZipFile(path,"w") as archive: archive.writestr("bundle/data.pkl",pickle.dumps(payload))
            binding=dict(reference="cache://runs/"+run_id+"/selected_bundle.pt",sha256=reporting.sha(path))
            source=report/"receipts/D02_s92601.json"
            result=dict(private_artifacts=[binding]);atomic_json(source,dict(result=result))
            atomic_json(report/"runs"/run_id/"search/RECEIPT.json",search)
            refit_path=report/"runs"/run_id/"refit/RECEIPT.json";atomic_json(refit_path,refit)
            job=dict(job_id="D02_s92601",question_id="D02",seed=92601)
            run=dict(run_id="D02_s92601_0",method="Moirai_B0",cohort="jena",seed=92601,C=21,L=512,H=48,
                     device="cuda",epochs=1.,updates=4,exposures=5,scaler_sha256="s",MSE=1.,MAE=.5)
            raw=dict(run_id="D02_s92601_1",method="Moirai_Raw",cohort="jena",seed=92601,C=21,L=512,H=48,device="cuda")
            cost=dict(method="Moirai_B0",cohort="jena",supervision_seed=92601,C=21,L=512,H=48,B=1,device="cpu",fit_seconds=None)
            data=dict(runs=[run,raw],costs=[cost],bindings={});ctx=dict(repo=repo,report=report,cache=cache)
            reporting.complete_fit_metadata(ctx,{job["job_id"]:(job,result,source)},data)
            self.assertEqual(run["fit_stage_total_seconds"],5.)
            self.assertEqual(cost["fit_seconds"],5.)
            self.assertEqual(cost["fit_device"],"cuda")
            self.assertEqual(cost["device"],"cpu")
            self.assertIsNone(raw.get("fit_stage_total_seconds"))
            self.assertIn("excluded",run["fit_time_scope"])
            self.assertEqual(run["fit_source_refs"][0],binding)
            run.update(status="COMPLETE",source_sha256=reporting.sha(source))
            raw.update(status="COMPLETE",source_sha256=reporting.sha(source))
            data.update(jobs=[],endpoints=[])
            atomic_json(report/"DATA_AUDIT.json",dict(status="PASS_WITH_M5_BLOCK",datasets={}))
            reporting._documents(ctx,data,[])
            table=next(row for row in reporting.rows(report/"PPT_T1_ACCURACY_COST.csv") if row["method"]=="Moirai_B0")
            self.assertEqual(table["fit_stage_total_seconds"],"5.0")
            self.assertEqual(table["fit_device"],"cuda")
            self.assertEqual(table["device"],"cpu")
            atomic_json(report/"NEW_STATS_REPLAY.json",dict(rows=data["new_stats_replay"]))
            self.assertTrue(reporting.replay_binding_validation(ctx)["all_bound_artifacts_verified"])
            atomic_json(refit_path,dict(refit,seconds=4.))
            self.assertFalse(reporting.replay_binding_validation(ctx)["all_bound_artifacts_verified"])
            with self.assertRaisesRegex(ValueError,"Fit stage payload differs"):
                reporting.complete_fit_metadata(ctx,{job["job_id"]:(job,result,source)},data)
            atomic_json(refit_path,refit)
            path.write_bytes(b"changed bundle")
            with self.assertRaisesRegex(ValueError,"Fit bundle SHA binding rejected"):
                reporting.complete_fit_metadata(ctx,{job["job_id"]:(job,result,source)},data)

    def test_c01_fit_time_uses_bound_historical_run_lineage_and_gpu_remainder_scope(self):
        import pickle
        import zipfile
        from research.adapter_diagnostics_20261017_v2.existing_jobs import _identity
        repo=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo,prefix="report_test_") as directory:
            root=Path(directory);report=root/"report";cache=root/"cache"
            job=dict(job_id="C01_cuda",question_id="C01",family="B0",seed=92601,device="cuda")
            path=cache/"runs/pems_B0/selected_bundle.pt";path.parent.mkdir(parents=True)
            search=dict(status="SEGMENT_COMPLETE",seconds=2.,selected_epoch=1.)
            refit=dict(status="SEGMENT_COMPLETE",seconds=3.,updates=4,exposures=5,contract=dict(scaler_sha="s"))
            payload=dict(search_summary=search,refit_summary=refit,spec=dict(run_id="historical_B0",family="B0",seed=92601,device="cuda"))
            with zipfile.ZipFile(path,"w") as archive: archive.writestr("bundle/data.pkl",pickle.dumps(payload))
            handler=report/"jobs/C01_cuda_handler.json"
            atomic_json(handler,dict(job_id=job["job_id"],scientific_identity_sha256=_identity(job),artifact_hashes=[
                dict(path="cache://runs/pems_B0/selected_bundle.pt",sha256=reporting.sha(path))]))
            lineage=report/"runs/pems_B0/DEVICE_BRANCH_LINEAGE.json"
            atomic_json(lineage,dict(original_checkpoint_ref="asset://old/runs/historical_B0/search/checkpoint.pt"))
            receipt=report/"receipts/C01_cuda.json"
            atomic_json(receipt,dict(artifact_hashes={"jobs/C01_cuda_handler.json":reporting.sha(handler),
                "runs/pems_B0/DEVICE_BRANCH_LINEAGE.json":reporting.sha(lineage)}))
            atomic_json(report/"runs/pems_B0/SEARCH_SUMMARY.json",search)
            atomic_json(report/"runs/pems_B0/REFIT_SUMMARY.json",refit)
            run=dict(run_id="C01_cuda",method="B0",seed=92601,device="cuda",epochs=1.,updates=4,exposures=5,scaler_sha256="s")
            data=dict(runs=[run],costs=[],bindings={});ctx=dict(repo=repo,report=report,cache=cache)
            reporting.complete_fit_metadata(ctx,{job["job_id"]:(job,{},receipt)},data)
            self.assertEqual(run["fit_stage_total_seconds"],5.)
            self.assertIn("GPU remainder search",run["fit_time_scope"])
            self.assertIn("historical CPU time excluded",run["fit_time_scope"])
            atomic_json(lineage,dict(original_checkpoint_ref="asset://old/runs/other/search/checkpoint.pt"))
            with self.assertRaisesRegex(ValueError,"C01 fit lineage SHA binding rejected"):
                reporting.complete_fit_metadata(ctx,{job["job_id"]:(job,{},receipt)},data)

    def test_q6_cost_observations_keep_precision_limit_batch_direction_and_reject_facets(self):
        shared=dict(cohort="jena",C=21,L=512,H=48,split="V_select",metric_scale="canonical_A60",
                    **{key:"a"*64 for key in ("target_sha256","mask_sha256","origins_sha256","channel_sha256","scaler_sha256")})
        runs=[dict(shared,run_id=method,method=method,seed=92601,MSE=mse,MAE=mae)
              for method,mse,mae in (("Ridge",.5,.4),("B2",.7,.6),("Raw",1.,.8))]
        costs=[]
        for device,method,batch,latency in (("cpu","Ridge",1,.3),("cpu","B2",1,48.),
                ("cuda","Raw",1,12.),("cuda","B2",1,24.),("cuda","Raw",8,6.),("cuda","B2",8,2.)):
            costs.append(dict(cohort="jena",C=21,L=512,H=48,method=method,B=batch,device=device,
                supervision_seed=92601,dtype="float32",core_dtype="float64" if method=="Ridge" else "float32",
                median_ms_per_origin=latency,exclusive=True,parity_state="PASS",threads=2,scope="whole route",
                physical_input_sha256="b"*64,observation_mask_sha256="c"*64))
        value=reporting.practical_cost_observations(dict(runs=runs,costs=costs))
        self.assertEqual(len(value),3)
        self.assertIn("두 오차와 지연 모두 낮다",value[0])
        self.assertIn("float64",value[0]);self.assertIn("float32",value[0])
        self.assertIn("동일 정밀도 단일요인 비교가 아니다",value[0])
        self.assertIn("0.5x, B2가 느림",value[1])
        self.assertIn("3.0x, B2가 빠름",value[2])
        for row in costs:
            if row["method"]=="B2": row["physical_input_sha256"]="d"*64
        self.assertEqual(reporting.practical_cost_observations(dict(runs=runs,costs=costs)),[])
        for row in costs:
            row["physical_input_sha256"]="b"*64
            if row["method"]=="B2": row["parity_state"]="FAILED_PARITY"
        self.assertEqual(reporting.practical_cost_observations(dict(runs=runs,costs=costs)),[])

    def test_e02_b0_b2_join_requires_bound_archive_checkpoint_metrics_and_array_fingerprints(self):
        import numpy as np
        repo=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(dir=repo,prefix="report_test_") as directory:
            root=Path(directory);archive=root/"V_select.npz"
            values=dict(pred_normalized=np.ones((1,48,21),np.float32),target_normalized=np.zeros((1,48,21),np.float32),
                        observed=np.ones((1,48,21),bool),origins=np.array([14675]),channel_ids=np.asarray([str(i) for i in range(21)]))
            np.savez(archive,**values)
            run=dict(run_id="selected_B2",cohort="jena",method="B2",C=21,L=512,H=48,seed=92601,MSE=1.,MAE=1.,checkpoint_sha256="c"*64)
            score=reporting.score_prediction_archive(archive,reporting.sha(archive),None,run)
            run.update(reporting.contract_metadata(score))
            binding=dict(reference="asset://V_select.npz",sha256=reporting.sha(archive))
            result=dict(bindings=dict(B2=dict(artifact_sha256="c"*64)),private_artifacts=[binding])
            receipt=root/"receipt.json";atomic_json(receipt,dict(result=result))
            job=dict(question_id="E02");cost=dict(job_id="E02",source_method="B2",method="B2",cohort="jena",C=21,L=512,H=48,
                supervision_seed=92601,accuracy_archive_sha256=binding["sha256"])
            data=dict(runs=[run],costs=[cost],bindings={});ctx=dict(repo=repo,report=root,asset=root)
            self.assertIsNone(reporting.cost_accuracy_match(cost,[run]))
            run["checkpoint_sha256"]="d"*64
            reporting.complete_cost_metadata(ctx,{"E02":(job,result,receipt)},data)
            self.assertIsNone(reporting.cost_accuracy_match(cost,[run]))
            run["checkpoint_sha256"]="c"*64
            run["target_sha256"]="e"*64
            with self.assertRaisesRegex(ValueError,"canonical arrays disagree"):
                reporting.complete_cost_metadata(ctx,{"E02":(job,result,receipt)},data)
            run.update(reporting.contract_metadata(score))
            reporting.complete_cost_metadata(ctx,{"E02":(job,result,receipt)},data)
            self.assertIs(reporting.cost_accuracy_match(cost,[run]),run)
            self.assertEqual(data["new_stats_replay"][0]["private_artifacts"],[binding])
            b0=dict(run,run_id="selected_B0",method="B0",checkpoint_sha256="f"*64,independent_private_score_sha256=None)
            b0_cost=dict(cost,method="B0",source_method="B0")
            b0_result=dict(result,bindings=dict(B0=dict(artifact_sha256="f"*64)))
            b0_data=dict(runs=[b0],costs=[b0_cost],bindings={})
            reporting.complete_cost_metadata(ctx,{"E02":(job,b0_result,receipt)},b0_data)
            self.assertIs(reporting.cost_accuracy_match(b0_cost,[b0]),b0)
            del run["independent_private_score_sha256"]
            archive.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError,"archive SHA mismatch"):
                reporting.complete_cost_metadata(ctx,{"E02":(job,result,receipt)},data)

    def test_initialization_curve_legend_omits_missing_end_and_shortens_recorded_end(self):
        self.assertEqual(reporting.initialization_curve_label("PCA",92601,None),"PCA 92601")
        self.assertEqual(reporting.initialization_curve_label("QR",92602,10482),"QR 92602 end=10482")

    def test_error_profile_legend_has_dedicated_row_outside_data_and_inside_canvas(self):
        plt=reporting._plot_setup()
        profiles=[dict(cohort="jena",method="B2_FULL_HISTORY_PROBE_"+str(index),reference="B0_FULL_HISTORY_PROBE",
            seed=92601,match_status="EXACT_MATCHED" if index%2==0 else "UNVERIFIED_CONTEXTUAL",
            positions=[1,2],delta_MSE=[0.,1.],delta_MAE=[0.,.5]) for index in range(21)]
        fig=reporting.error_profile_figure(plt,profiles,"horizon")
        try:
            fig.canvas.draw();renderer=fig.canvas.get_renderer()
            left,right,legend_axis=fig.axes
            self.assertIsNone(left.get_legend());self.assertIsNone(right.get_legend())
            legend=legend_axis.get_legend();box=legend.get_window_extent(renderer)
            self.assertEqual(len(legend.get_texts()),21)
            self.assertLessEqual(box.y1,min(left.get_window_extent(renderer).y0,right.get_window_extent(renderer).y0))
            self.assertGreaterEqual(box.x0,fig.bbox.x0);self.assertLessEqual(box.x1,fig.bbox.x1)
            self.assertEqual(left.lines[0].get_linestyle(),"-");self.assertEqual(left.lines[1].get_linestyle(),":")
            self.assertIn("[context]",legend.get_texts()[1].get_text())
        finally:
            plt.close(fig)
        bar_profiles=[dict(profile,positions=[2,0,1],delta_MSE=[index+2,index,index+1],
                           delta_MAE=[index+2.5,index+.5,index+1.5]) for index,profile in enumerate(profiles)]
        fig=reporting.error_profile_figure(plt,bar_profiles,"channel")
        try:
            fig.canvas.draw()
            for axis,offset in zip(fig.axes[:2],(0.,.5)):
                self.assertEqual(len(axis.containers),21)
                self.assertEqual([label.get_text() for label in axis.get_xticklabels()],["0","1","2"])
                for index,container in enumerate(axis.containers):
                    self.assertEqual([bar.get_height() for bar in container.patches],[index+offset+j for j in range(3)])
                    self.assertTrue(all(abs(bar.get_width()-.8/21)<1e-12 for bar in container.patches))
                    self.assertEqual(container.patches[0].get_hatch(),None if index%2==0 else "///")
            self.assertEqual(len(fig.axes[2].get_legend().get_texts()),21)
            self.assertIn("fixed original channel order",fig._suptitle.get_text())
        finally:
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
