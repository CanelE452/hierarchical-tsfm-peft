"""Report numbers come from actual sealed arrays; all production writes are disabled."""
import copy
import csv
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import data_contract as data
import prediction_cache
import report_campaign as report
from training import fingerprint


class ReportCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="informs_report_")
        self.addCleanup(self.temp.cleanup)
        self.public, self.cache = Path(self.temp.name)/"public", Path(self.temp.name)/"private"
        self.public.mkdir(); self.cache.mkdir()
        values = np.cumsum(np.random.default_rng(31).normal(size=(240, 3)), axis=0).astype(np.float32)
        mask = np.ones_like(values, dtype=bool)
        mask[192:198, 0] = False; mask[201:215, 1] = False
        self.task = dict(name="toy", physical=values, observed=mask, times=np.arange(240, dtype=np.int64),
                         columns=["a", "b", "c"], interval_seconds=1, L=64, H=4, data_sha256="toy_data")
        self.registry = []
        for run_id, family, baseline in (("current", "CURRENT", "CURRENT"), ("seasonal", "SEASONAL_NAIVE", "SEASONAL"),
                                         ("pending", "B0", None)):
            self.registry.append(dict(experiment_id=run_id, dataset="toy", panel="", fold="primary", block="E1", family=family,
                num_channels="3", latent_k="0" if baseline else "2", seed="92601", length="64", horizon="4",
                requirement="MANDATORY_MINIMUM", execution_status="PENDING_QUEUE", training_status="NOT_STARTED",
                validity_status="NOT_CHECKED", scientific_status="UNASSESSED", publication_status="NOT_PUBLISHED", reason="", resume_command=""))
            if baseline:
                prediction_cache.predict_parent(run_id, self.task, "V_select", self.cache, self.public, None,
                                                baseline=dict(kind=baseline, period=4))
        for filename in ("EXPERIMENT_REGISTRY.csv", "EXPERIMENT_REGISTRY_INITIAL.csv"):
            self.write_registry(filename, self.registry)
        (self.public/"CONFIG_LOCK.json").write_text(json.dumps(dict(registry_rows=3, resource_limits=dict(gpu_process_seconds=43200))))

    def write_registry(self, filename, rows):
        with (self.public/filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    def generate(self):
        return Path(report.generate(self.public, self.cache))

    def test_actual_sealed_arrays_macro_score_and_separate_pending_statuses(self):
        before = (self.public/"EXPERIMENT_REGISTRY.csv").read_bytes()
        directory = self.generate()
        rows = report._rows(directory/"SCORES.csv")
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["length"],"64")
            self.assertEqual(row["panel"],"")
            self.assertEqual(row["lora_rank"],"")
            self.assertEqual(row["point_type"],"deterministic_point_forecast")
            self.assertEqual(row["foundation_point"],"not_used")
            self.assertEqual(row["execution_status"],"PENDING_QUEUE")
            self.assertEqual(row["validity_status"],"NOT_CHECKED")
            self.assertEqual(row["learning_rate"],"")
            with np.load(self.cache/f"predictions/{row['run_id']}/V_select.npz", allow_pickle=False) as saved:
                error = saved["pred_normalized"].astype(np.float64)-saved["target_normalized"].astype(np.float64)
                mask = saved["observed"]
                macro_mse = np.mean([np.mean(error[:,:,k][mask[:,:,k]]**2) for k in range(3)])
                macro_mae = np.mean([np.mean(np.abs(error[:,:,k][mask[:,:,k]])) for k in range(3)])
            self.assertAlmostEqual(float(row["mse"]), macro_mse, places=12)
            self.assertAlmostEqual(float(row["mae"]), macro_mae, places=12)
        summary = report._load(directory/"SUMMARY.json")
        self.assertEqual(summary["numeric_V_run_count"], 2)
        self.assertEqual(summary["numeric_T_run_count"], 0)
        self.assertEqual(summary["registry_unreconciled_numeric_rows"], 2)
        self.assertEqual(summary["status_counts"]["execution_status"], {"PENDING_QUEUE":3})
        self.assertEqual(summary["status_counts"]["scientific_status"], {"UNASSESSED":3})
        self.assertFalse(summary["campaign_complete"])
        self.assertEqual((self.public/"EXPERIMENT_REGISTRY.csv").read_bytes(), before)
        coverage = report._rows(directory/"COVERAGE.csv")
        self.assertEqual(sum(int(row["no_numeric_V_runs"]) for row in coverage), 1)

    def test_tampered_archive_is_excluded_not_summarized(self):
        archive = self.cache/"predictions/current/V_select.npz"
        archive.write_bytes(archive.read_bytes()+b"tampered")
        directory = self.generate()
        self.assertEqual([row["run_id"] for row in report._rows(directory/"SCORES.csv")], ["seasonal"])
        issues = report._rows(directory/"INPUT_ISSUES.csv")
        self.assertTrue(any(row["scope"]=="prediction" and "archive" in row["reason"].lower() for row in issues))

    def test_missing_manifest_keeps_pending_and_wrong_saved_score_is_excluded(self):
        current = self.public/"predictions/current/V_select/PREDICTION_MANIFEST.json"
        altered = report._load(current); altered["score"]["mse"] = -7
        current.write_text(json.dumps(altered))
        (self.public/"predictions/seasonal/V_select/PREDICTION_MANIFEST.json").unlink()
        directory = self.generate()
        self.assertEqual(report._rows(directory/"SCORES.csv"), [])
        self.assertEqual(report._load(directory/"SUMMARY.json")["numeric_V_run_count"], 0)

    def test_t_guard_rejects_before_private_target_archive_is_opened(self):
        vpath = self.public/"predictions/current/V_select/PREDICTION_MANIFEST.json"
        tpath = self.public/"route_predictions/current/T_final/PREDICTION_MANIFEST.json"
        tpath.parent.mkdir(parents=True)
        manifest = copy.deepcopy(report._load(vpath)); manifest["split"]="T_final"
        tpath.write_text(json.dumps(manifest))
        original = report.independent._load_arrays
        def guarded(manifest, *args):
            if manifest["split"]=="T_final":
                self.fail("Protected unlocked T cache must not be opened")
            return original(manifest, *args)
        with patch.object(report.independent, "_load_arrays", side_effect=guarded):
            directory = self.generate()
        self.assertEqual(report._load(directory/"SUMMARY.json")["numeric_T_run_count"], 0)
        self.assertTrue(any("unopened" in row["reason"] for row in report._rows(directory/"INPUT_ISSUES.csv")))

    def test_relative_paths_cannot_escape_and_complete_snapshot_hashes_hold(self):
        with self.assertRaises(ValueError): report._inside(self.public, "../outside")
        with self.assertRaises(ValueError): report._inside(self.public, self.cache.as_posix())
        first = self.generate()
        before = {path.name:data.sha256_file(path) for path in first.iterdir()}
        second = self.generate()
        self.assertNotEqual(first, second)
        self.assertEqual(before, {path.name:data.sha256_file(path) for path in first.iterdir()})
        for item in report._load(first/"REPORT_MANIFEST.json")["outputs"]:
            self.assertEqual(data.sha256_file(first/item["file"]), item["sha256"])
        for path in first.glob("*.md"):
            self.assertNotIn(str(self.cache), path.read_text(encoding="utf-8"))
        self.assertIn("PARTIAL", (first/"REPORT_KO.md").read_text(encoding="utf-8"))
        for filename in ("RUN_INDEX.csv","PREDICTION_MANIFEST.csv","COST.csv","CLAIM_EVIDENCE.csv","RESUME_QUEUE.json","PROFESSOR_BRIEF.md","REPRODUCE.md"):
            self.assertTrue((first/filename).is_file())
        queue = report._load(first/"RESUME_QUEUE.json")
        self.assertFalse(queue["automatically_executed"])
        pending = next(row for row in queue["runs"] if row["run_id"]=="pending")
        self.assertEqual(pending["status"], "PENDING_FIT")
        self.assertIn("--run-id pending",pending["command"])
        self.assertTrue(all(row["status"]=="UNRESOLVED" for row in report._rows(first/"CLAIM_EVIDENCE.csv")))

    def cost_fixture(self, bound=True):
        folder = self.public/"costs/toy_snapshot"; folder.mkdir(parents=True)
        path = folder/"cost.json"; trials = folder/"cost.trials.csv.gz"
        condition = "current__rows4"
        with gzip.open(trials,"wt",encoding="utf-8",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=["run_id","phase","trial","whole_seconds"])
            writer.writeheader()
            for phase,count in (("warmup",10),("timed",20),("stage_probe",20)):
                writer.writerows(dict(run_id=condition,phase=phase,trial=i,whole_seconds=.001+i*.0001) for i in range(count))
        manifest_path=self.public/"predictions/current/V_select/PREDICTION_MANIFEST.json"
        manifest=report._load(manifest_path)
        spec=dict(run_id="current",family="CURRENT",dataset="toy")
        cost_spec=dict(spec,run_id=condition,accuracy_run_id="current",row_microbatch=4)
        binding=dict(status="BOUND_EXACT_ROUTE",condition_id=condition,accuracy_run_id="current",
            cost_route_spec=cost_spec,cost_route_spec_sha256=fingerprint(cost_spec),
            accuracy_route_spec=spec,accuracy_route_spec_sha256=fingerprint(spec),artifact_bindings={},
            data_sha256=manifest["data_sha"],scaler_sha=manifest["scaler_sha"],
            accuracy_manifest_public_relative_path=manifest_path.relative_to(self.public).as_posix(),
            accuracy_manifest_sha256=data.sha256_file(manifest_path),accuracy_prediction_model_sha=manifest["model_sha"],
            accuracy_prediction_source_sha=manifest["source_sha"])
        receipt=dict(status="COMPLETE_PROTOCOL",dataset="toy",scope="fresh_process_single_route",batch_size=1,
            device="cpu",dtype="float32",trials_file=trials.name,trials_sha256=data.sha256_file(trials),
            accuracy_bindings={condition:binding} if bound else {})
        path.write_text(json.dumps(receipt),encoding="utf-8")
        return path

    def test_cost_uses_explicit_binding_and_reaggregates_actual_trials(self):
        self.cost_fixture()
        directory=self.generate()
        costs=report._rows(directory/"COST.csv")
        self.assertEqual(len(costs),1)
        self.assertEqual(costs[0]["run_id"],"current")
        self.assertEqual(costs[0]["cost_run_id"],"current__rows4")
        self.assertEqual(costs[0]["accuracy_binding_status"],"BOUND_EXACT_ROUTE")
        self.assertAlmostEqual(float(costs[0]["median_seconds"]),.00195)
        current=next(row for row in report._rows(directory/"SCORES.csv") if row["run_id"]=="current")
        self.assertEqual(current["cost_references"],"costs/toy_snapshot/cost.json")

    def test_cost_name_suffix_is_not_accuracy_identity(self):
        self.cost_fixture(bound=False)
        directory=self.generate()
        costs=report._rows(directory/"COST.csv")
        self.assertEqual(costs[0]["run_id"],"")
        self.assertEqual(costs[0]["accuracy_binding_status"],"UNBOUND_NO_DECLARED_ACCURACY")
        self.assertEqual(report._load(directory/"SUMMARY.json")["verified_cost_condition_count"],0)
        self.assertFalse(any(row["cost_references"] for row in report._rows(directory/"SCORES.csv")))

    def test_cost_exact_binding_rejects_silent_accuracy_change(self):
        path=self.cost_fixture()
        receipt=report._load(path)
        binding=receipt["accuracy_bindings"]["current__rows4"]
        binding["cost_route_spec"]["family"]="B0"
        binding["cost_route_spec_sha256"]=fingerprint(binding["cost_route_spec"])
        path.write_text(json.dumps(receipt))
        directory=self.generate()
        self.assertEqual(report._rows(directory/"COST.csv"),[])
        self.assertTrue(any(row["scope"]=="cost" for row in report._rows(directory/"INPUT_ISSUES.csv")))

    def test_old_probe_journal_rows_without_lease_id_are_all_charged(self):
        rows=[dict(kind="probe",charged_process_seconds=2),dict(kind="batching_probe",charged_process_seconds=3),
              dict(kind="fit",lease_id="real_lease",charged_process_seconds=7)]
        (self.cache/"GPU_USAGE.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        directory=self.generate()
        usage=report._load(directory/"SUMMARY.json")["resource_usage"]
        self.assertEqual(usage["closed_GPU_process_seconds"],12)
        self.assertEqual(usage["closed_lease_count"],3)
        self.assertFalse(any(row["scope"]=="resource_journal" for row in report._rows(directory/"INPUT_ISSUES.csv")))

    def test_post_selection_queue_disables_same_campaign_training(self):
        (self.public/"SELECTION_LOCK.json").write_text(json.dumps(dict(locked=True,allowed_run_ids=["current"])))
        directory=self.generate()
        queue=report._load(directory/"RESUME_QUEUE.json")
        self.assertEqual(queue["training_commands_scope"],"PRE_SELECTION_ONLY")
        self.assertTrue(queue["selection_locked"])
        pending=next(row for row in queue["runs"] if row["run_id"]=="pending")
        self.assertEqual(pending["status"],"NEW_CAMPAIGN_OR_POST_T_DEVELOPMENT_REQUIRED")
        self.assertEqual(pending["command"],"")
        self.assertIn("--run-id pending",pending["pre_selection_command"])
        self.assertFalse(pending["executable_training_under_existing_selection"])
        current=next(row for row in queue["runs"] if row["run_id"]=="current")
        self.assertEqual(current["status"],"PENDING_LOCKED_T_EXECUTION")
        seasonal=next(row for row in queue["runs"] if row["run_id"]=="seasonal")
        self.assertEqual(seasonal["status"],"NOT_SELECTED_FOR_LOCKED_T")
        self.assertEqual(seasonal["command"],"")
        script=(directory/"RESUME_COMMANDS.ps1").read_text(encoding="utf-8")
        self.assertIn("if ($selectionLocked)",script)
        self.assertIn("PRE_SELECTION_ONLY",script)

    def test_robin_stress_resume_keeps_main_minimum_and_selection_guards(self):
        datasets=("beijing","jena","electricity","pems08","robin")
        registry={ds:dict(self.registry[-1],experiment_id=ds,dataset=ds,
            requirement="STRESS_APPENDIX" if ds=="robin" else "MANDATORY_MINIMUM") for ds in datasets}
        script=report._resume_commands(registry,[])
        commands=[line for line in script.splitlines() if " -u campaign.py " in line]
        self.assertEqual(len(commands),len(datasets))
        self.assertIn("& $py -X utf8 -u campaign.py --dataset robin --epochs .25",commands)
        for ds in datasets[:-1]:
            self.assertIn(f"& $py -X utf8 -u campaign.py --dataset {ds} --minimum-only --epochs .25",commands)
        guard=script.index("if ($selectionLocked)")
        end_training=script.index("\n}\n",guard)
        self.assertTrue(all(guard<script.index(command)<end_training for command in commands))
        self.assertIn("PRE_SELECTION_ONLY",script)
        t_guard=script.index("assert (p/'SELECTION_LOCK.json').is_file()")
        self.assertGreater(script.index("print(f.execute_T("),t_guard)
        queue=report._resume_queue(registry,{},[],self.cache,dict(locked=True))
        self.assertTrue(all(row["command"]=="" for row in queue["runs"]))
        self.assertTrue(all(not row["executable_training_under_existing_selection"] for row in queue["runs"]))

    def test_replay_assertion_diagnostic_omits_private_array_values(self):
        folder=self.public/"route_replays/current";folder.mkdir(parents=True)
        private=self.cache/"route_replays/current";private.mkdir(parents=True)
        manifest_path=self.public/"predictions/current/V_select/PREDICTION_MANIFEST.json"
        source=self.cache/"predictions/current/V_select.npz"
        with np.load(source,allow_pickle=False) as stream:
            arrays={key:stream[key] for key in stream.files}
        archive=private/"V_subset.npz"
        with archive.open("xb") as stream:np.savez_compressed(stream,**arrays)
        receipt=dict(run_id="current",status="PASS",standard_manifest_public_relative_path=manifest_path.relative_to(self.public).as_posix(),
            standard_manifest_sha256=data.sha256_file(manifest_path),cache_relative_path=archive.relative_to(self.cache).as_posix(),
            archive_sha256=data.sha256_file(archive),atol=1e-5,rtol=1e-4,actual_F_streams=0)
        (folder/"REPLAY_RECEIPT.json").write_text(json.dumps(receipt))
        with patch.object(report.np.testing,"assert_array_equal",side_effect=AssertionError("PRIVATE_TARGET_ARRAY_VALUES")):
            directory=self.generate()
        self.assertNotIn("PRIVATE_TARGET_ARRAY_VALUES",(directory/"INPUT_ISSUES.csv").read_text())
        self.assertIn("private array values omitted",(directory/"INPUT_ISSUES.csv").read_text())

    def matching_row(self,family,lr,run_id):
        return dict(run_id=run_id,family=family,learning_rate=lr,split="V_select",dataset="toy",fold="primary",
            cohort="toy:primary:V_select:stride4",latent_k=2,seed=92601,max_epoch=1.,truth_cohort_sha="truth",
            data_sha="data",scaler_sha="scaler",parent_signature="",mse=1.,mae=.5)

    def test_different_LR_architectures_never_form_a_matched_pair(self):
        rows=[self.matching_row("B0",.001,"B0_high"),self.matching_row("B2",.0001,"B2_low")]
        self.assertEqual(report._matched(rows),[])

    def test_two_full_LR_pairs_are_preserved_separately(self):
        rows=[self.matching_row(family,lr,f"{family}_{lr}") for lr in (.001,.0001) for family in ("B0","B2")]
        pairs=report._matched(rows)
        self.assertEqual(len(pairs),2)
        self.assertCountEqual([pair["learning_rate"] for pair in pairs],[.001,.0001])
        for pair in pairs:
            self.assertEqual(pair["reference_learning_rate"],pair["compared_learning_rate"])
            self.assertTrue(pair["comparison_contract_verified"])

    def test_unknown_training_LR_is_explicitly_omitted_from_matching(self):
        issues=[]
        rows=[self.matching_row("B0",None,"unknown"),self.matching_row("B2",.001,"known")]
        self.assertEqual(report._matched(rows,issues),[])
        self.assertTrue(any(row["run_id"]=="unknown" and row["scope"]=="matched_comparison" for row in issues))

    def test_actual_source_spec_LR_reaches_fit_and_unique_run_index(self):
        registry={};inputs={};issues=[]
        for family in ("B0","B2"):
            for rate in (.001,.0001):
                run_id=f"{family}_{rate}";folder=self.public/"runs"/run_id;folder.mkdir(parents=True)
                (folder/"BASE_FIT_RECEIPT.json").write_text(json.dumps(dict(run_id=run_id,status="COMPLETE_VALID",family=family,epochs=[.25,1.])))
                (folder/"SOURCE_CONTRACT.json").write_text(json.dumps(dict(spec=dict(run_id=run_id,lr=rate),config_sha="cfg_"+run_id)))
                registry[run_id]=dict(self.registry[-1],experiment_id=run_id,family=family)
        fits,_=report._fits(self.public,self.cache,inputs,issues)
        index=report._run_index(registry,fits,[])
        self.assertEqual(len(index),4)
        self.assertEqual({row["run_id"]:row["learning_rate"] for row in index},
                         {f"{family}_{rate}":rate for family in ("B0","B2") for rate in (.001,.0001)})
        self.assertEqual(fits["B0_0.0001"]["learning_rate_source"],"SOURCE_CONTRACT.json:spec.lr")

    def test_closed_form_alpha_requires_exact_parents_and_known_head_LR(self):
        alpha=self.matching_row("ALPHA05",None,"alpha")
        head=self.matching_row("CONDITIONAL",.001,"head")
        alpha["parent_signature"]=head["parent_signature"]="exact_parents"
        pairs=report._matched([alpha,head])
        self.assertEqual(len(pairs),1)
        self.assertIsNone(pairs[0]["reference_learning_rate"])
        self.assertEqual(pairs[0]["compared_learning_rate"],.001)
        self.assertIn("closed_form_alpha_reference",pairs[0]["learning_rate_contract"])
        head["parent_signature"]="different_parent"
        self.assertEqual(report._matched([alpha,head]),[])
        head["parent_signature"]="exact_parents";head["learning_rate"]=None
        self.assertEqual(report._matched([alpha,head]),[])

    def test_claims_do_not_reuse_an_unverified_legacy_pair_key(self):
        a=self.matching_row("B0",.001,"B0");b=self.matching_row("B2",.001,"B2")
        a["block"]=b["block"]="E1"
        claims=report._claims([a,b],[dict(compared_run_id="B2")],[],[],{},[])
        self.assertEqual(next(row["status"] for row in claims if row["claim_id"]=="predictive_compression"),"UNRESOLVED")
        claims=report._claims([a,b],report._matched([a,b]),[],[],{},[])
        self.assertEqual(next(row["status"] for row in claims if row["claim_id"]=="predictive_compression"),"PARTIAL_EVIDENCE_UNRESOLVED")


if __name__=="__main__": unittest.main()
