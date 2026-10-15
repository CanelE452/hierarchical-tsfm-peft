"""Generated metadata fixtures only; no scientific arrays, models or fitting."""
from pathlib import Path
import csv
import importlib.util
import json
import statistics
import tempfile
import unittest
import xml.etree.ElementTree as ET
from unittest.mock import patch


SOURCE = Path(__file__).with_name("make_linear_e_adaptation_report_v1.py")


def module():
    if not SOURCE.exists(): raise AssertionError("Linear-E JSON-only report implementation is missing")
    spec = importlib.util.spec_from_file_location("linear_e_report_fixture", SOURCE)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def score(value, c):
    def one(mult):
        return {"mse": value*mult, "mae": value**.5*mult**.5,
            "per_channel_mse": [value*mult]*c, "per_channel_mae": [(value*mult)**.5]*c,
            "observed_counts": [48]*c, "active_channels": c}
    return {"normalized": one(1), "physical_per_channel_diagnostic": one(4)}


def fixture_values(folder):
    r = module(); folder.mkdir(parents=True, exist_ok=True)
    def put(name, value):
        path = folder/name; path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8"); return r.record(path)
    def opaque(name):
        return {"path": str(folder/(name+".npz")), "sha256": "a"*64, "bytes": 123,
            "arrays": {"E": {"shape": [3, 1], "dtype": "float32", "sha256": "b"*64}}}
    plan = folder/"PLAN.md"; plan.write_text("Synthetic plan\n", encoding="utf-8")
    parent = put("PARENT.json", {"status": "COMPLETE"})
    report = folder/"PRIOR.md"; report.write_text("Synthetic prior report\n", encoding="utf-8")
    source = folder/"source.py"; source.write_text("# synthetic source\n", encoding="utf-8")
    shared = {"protected_TEST_accessed": False, "F_before": {"model": "same"}, "F_after": {"model": "same"},
        "new_backbone_training": False, "independent_holdout": False, "strict_K_only": False,
        "plan": r.record(plan), "pure_source": r.record(source), "controller_source": r.record(source),
        "causal_lag_diagnostic": parent, "causal_lag_audit": parent, "parent_report": r.record(report),
        "parent_report_manifest": parent, "parent_report_audit": parent,
        "old_assets_sources_F_weights_unchanged": True, "resource": {"process_cpu_seconds": 1., "wall_seconds": 2.}}
    train = {**shared, "status": "COMPLETE", "mode": "train", "datasets": {}, "failed_paths": [],
        "adaptation_training_fits": 16, "training_updates_completed": 9600, "checkpoint_states_saved": 80,
        "resolved_selection_locks": 16, "all_16_locks_before_any_DEV": True, "whole_deployment_cost_measured": False,
        "same_initialization_seeds_change_order_only": True, "CP0_LS_and_RAW_baselines": {},
        "training_contract": {"updates_per_fit": 600, "checkpoints": [0,100,200,400,600], "batch_size": 4,
            "canonical_prediction_batch_size": 32, "full_FIT_channel_count_loss": True,
            "F_parameters_optimized": False, "F_only_gradient_probe_is_necessity_test": False}}
    probe = {**shared, "status": "COMPLETE", "mode": "probe", "cases": {}, "failed_paths": [], "probe_optimizer_updates_completed": 4}
    registry, locks, states = [], {}, {}
    for data, (c, k, dev) in r.DATASETS.items():
        train["datasets"][data] = {"conditions": {}}; probe["cases"][data] = {}; locks[data] = {}; states[data] = {}
        train["CP0_LS_and_RAW_baselines"][data] = {}
        for lag in r.LAGS:
            p = {"status": "PASS", "fixed_learned_CP0_byte_exact": True,
                "CP0_NumPy_parity": {key: {"pass": True, "max_abs": 0.} for key in ("prediction","latent_history","latent_future","lag_features")},
                "canonical32_vs_batch4": {key: {"pass": True, "max_abs": 0.} for key in ("prediction","latent_history","latent_future","lag_features")},
                "gradient_diagnostics": {"finite": True, "F_path_available": True, "E_F_path_gradient_norm": .2,
                    "E_direct_past_gradient_norm": .3, "E_gradient_sum_residual_max_abs": 1e-8,
                    "nonzero_total_E_gradient_alone_proves_F_path": False}, "E_bytes_changed": True, "D_bytes_changed": True,
                "probe_optimizer_updates_completed": 1, "loss": .5, "artifact": opaque(data+lag+"PROBE"),
                "forward_gradient_decomposition_backward_optimizer_wall_seconds": .02}
            probe["cases"][data][lag] = p
            train["CP0_LS_and_RAW_baselines"][data][lag] = {split: {"CP0_LS": {
                "scores_all_observed": score(.9,c), "scores_complete_mask": score(.91,c), "artifact": opaque(data+lag+split)},
                "RAW_scores": score(.8,c), "all_16_locks_already_sealed": True} for split in ("fit",dev)}
            for learn in (False,True):
                for seed in r.SEEDS:
                    name = f"{data}_{lag}_{'E_LEARN' if learn else 'E_FIXED'}_s{seed}"
                    cps = []
                    for update in (0,100,200,400,600):
                        mse = .9 if update == 0 else (.7 if learn else .8) + (0 if update in (200,400) else .1)
                        cps.append({"update": update, "tune": score(mse,c)["normalized"], "state": opaque(name+str(update)),
                            "scores_all_observed": score(mse,c), "scores_complete_mask": score(mse+.01,c), "artifact": opaque(name+str(update)+"P"),
                            "parameter_diagnostics": {"E_l2_norm": 1.+learn*update*.001, "D_l2_norm": 2.,
                                "E_singularvalues": [1.]*k, "D_singularvalues": [2.]*k,
                                "D_feature_to_C_operator_gain": 2., "finite": True}, "TUNE_latent_variance": [1.]*k})
                    log = folder/(name+"_UPDATES.jsonl")
                    with log.open("w",encoding="utf-8") as out:
                        for u in range(1,601):
                            out.write(json.dumps({"stage":"TRAINING_UPDATE","update":u,"loss":1/u,
                                "E_gradient_norm_before_clip": .2 if learn else 0., "D_gradient_norm_before_clip": .3,
                                "global_gradient_norm_before_clip": .4, "clipped": False})+"\n")
                    row = {"status":"COMPLETE","dataset":data,"lag":lag,"learn_E":learn,"seed":seed,
                        "training_updates_completed":600,"checkpoint_count":5,"checkpoints":cps,"selected_update":200,
                        "selected_tune":cps[2]["tune"],"selected_state":cps[2]["state"],"CP0_TUNE":cps[0],
                        "updates_log":r.record(log),"order_sha256":"c"*64,"FIT_channel_observed_counts":[12288]*c,
                        "global_clipping_fraction":0.,"RAW_TUNE_scores":score(.8,c),
                        "FIT":{"scores_all_observed":score(.6,c),"scores_complete_mask":score(.61,c),"artifact":opaque(name+"FIT")},
                        "DEV":{"scores_all_observed":score(.75 if learn else .85,c),"scores_complete_mask":score(.76,c),"artifact":opaque(name+"DEV")}}
                    lock=put(name+"_LOCK.json",{**row,"DEV_opened_before_lock":False,"sealed_unix":1.})
                    row["selection_lock"]=lock; locks[data][name]=lock; states[data][name]={"E":"fixed","D":"fixed"}
                    train["datasets"][data]["conditions"][name]=row
                    registry.append({"dataset":data,"lag":lag,"learn_E":learn,"seed":seed,"selection_lock":lock,"status":"COMPLETE"})
    train["selection_registry"]=put("REGISTRY.json",{"status":"ALL_16_RESOLVED_BEFORE_DEV","conditions":registry,"DEV_opened_before_seal":False})
    probe_ref=put("PROBE.json",probe); train["probe_result"]=probe_ref
    cost={**shared,"status":"COMPLETE","routes":list(r.ROUTES),"cost_split":"tune","cost":{},"failed_paths":[],"completed_cost_cells":36,
        "recorded_timer_samples":324,"completed_cell_timer_samples":324,"new_training_updates":0,
        "new_encoder_decoder_fit":False,"per_origin_decoder_fit":False,"cached_future_returned_instead_of_actual_F_inside_timer":False,
        "selected_states_before":states,"selected_states_after":states,"selection_registry":train["selection_registry"],"selection_locks":locks}
    for data in r.DATASETS:
        rows=[]
        for batch in (1,8):
            for i,route in enumerate(r.ROUTES):
                samples=[{"repeat":j,"position":(i-j)%9,"seconds":(.01+j*.001)*(1 if route=="RAW_BOLT" else .6),
                    "perf_start_ns":1000000000+j*100000000,"perf_end_ns":1000000000+j*100000000+round((.01+j*.001)*(1 if route=="RAW_BOLT" else .6)*1e9),
                    "unix_start_ns":2000000000+j*100000000,"unix_end_ns":2000000000+j*100000000+round((.01+j*.001)*(1 if route=="RAW_BOLT" else .6)*1e9)} for j in range(9)]
                seconds=[s["seconds"] for s in samples]
                rows.append({"route":route,"batch":batch,"median_seconds":statistics.median(seconds),"median_seconds_per_window":statistics.median(seconds)/batch,
                    "seconds":seconds,"samples":samples,"paired_RAW_ratios":[1. if route=="RAW_BOLT" else .6]*9,
                    "paired_RAW_median_ratio":1. if route=="RAW_BOLT" else .6,"source_metadata":{"shared_memory":True}})
        cost["cost"][data]={"rows":rows,"schedule_scope":"Nine circular rotations","warmup_each_route":2,"repeats":9}
    return {"probe":probe,"train":train,"audit":{"status":"PASS","protected_TEST_accessed":False,"totals":{},"parameter_changes":{}},
        "cost":cost,"cost_audit":{"status":"PASS","protected_TEST_accessed":False,"totals":{}}}


def gate_fixture(folder):
    r=module(); values=fixture_values(folder)
    def put(name,value):
        path=folder/name; path.write_text(json.dumps(value,allow_nan=False),encoding="utf-8"); return r.record(path)
    audit_source=folder/"audit_source.py"; audit_source.write_text("# generated verifier fixture\n",encoding="utf-8")
    paths={"probe":folder/"PROBE.json"}; train=values["train"]
    paths["train"]=folder/"TRAIN.json"; train_ref=put("TRAIN.json",train)
    values["audit"].update(diagnostic=train_ref,probe_result=r.record(paths["probe"]),audit_source=r.record(audit_source),totals={"adapter_fits_checked":16})
    paths["audit"]=folder/"AUDIT.json"; audit_ref=put("AUDIT.json",values["audit"])
    common={"linear_e_diagnostic":train_ref,"linear_e_audit":audit_ref,"probe_result":r.record(paths["probe"]),"selection_registry":train["selection_registry"]}
    plan=folder/"COST_PLAN.md"; plan.write_text("Synthetic cost plan\n",encoding="utf-8")
    values["cost"].update(**common,cost_plan=r.record(plan)); paths["cost"]=folder/"COST.json"; cost_ref=put("COST.json",values["cost"])
    values["cost_audit"].update(**common,cost=cost_ref,audit_source=r.record(audit_source),totals={"cost_cells_checked":36})
    paths["cost_audit"]=folder/"COST_AUDIT.json"; put("COST_AUDIT.json",values["cost_audit"])
    return paths,{key:r.record(path)["sha256"] for key,path in paths.items()},r.record(audit_source)


class ReportTests(unittest.TestCase):
    def setUp(self): self.r=module(); self.temp=tempfile.TemporaryDirectory(dir=SOURCE.parent); self.folder=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def test_pending_actual_gates_fail_before_any_input_read(self):
        with patch.object(self.r,"FIXED_SHA",{**self.r.FIXED_SHA,"cost_audit":None}),patch.object(self.r,"read",side_effect=AssertionError("must not read")):
            with self.assertRaisesRegex(ValueError,"NOT_YET_REGISTERED"): self.r.load_verified({k:"missing.json" for k in self.r.INPUTS})
    def test_exact_five_input_gate_and_mutations_without_scientific_IO(self):
        paths,hashes,audit_source=gate_fixture(self.folder)
        with patch.object(self.r,"FIXED_SHA",hashes),patch.object(self.r,"AUDITOR_SOURCES",{key:audit_source for key in ("audit","cost_audit")}),patch.object(self.r,"SCIENCE_COUNTS",{"adapter_fits_checked":16}),patch.object(self.r,"COST_COUNTS",{"cost_cells_checked":36}):
            values,bindings=self.r.load_verified(paths); self.assertEqual(values["train"]["adaptation_training_fits"],16)
            self.assertEqual(set(self.r.INPUTS)-set(bindings),set())
            original=paths["audit"].read_text(); paths["audit"].write_text(original+" ")
            with self.assertRaisesRegex(ValueError,"EXACT_ACTUAL"): self.r.load_verified(paths)
            paths["audit"].write_text(original)
            broken=json.loads(original); broken["totals"]["adapter_fits_checked"]=15; paths["audit"].write_text(json.dumps(broken)); hashes["audit"]=self.r.sha(paths["audit"])
            # The new audit file is rebound in cost to isolate missing coverage, not a stale SHA failure.
            for key in ("cost","cost_audit"):
                item=json.loads(paths[key].read_text()); item["linear_e_audit"]=self.r.record(paths["audit"])
                if key=="cost_audit": item["cost"]=self.r.record(paths["cost"])
                paths[key].write_text(json.dumps(item)); hashes[key]=self.r.sha(paths[key])
            with self.assertRaisesRegex(ValueError,"AUDIT_COVERAGE"): self.r.load_verified(paths)
    def test_same_lag_seed_pairing_preserves_masks_scales_and_baselines(self):
        values=fixture_values(self.folder); tables=self.r.collect_tables(values)
        self.assertEqual(len(tables["PAIRED_E"]),96); self.assertEqual(len(tables["CHECKPOINTS"]),80)
        self.assertEqual(len(tables["UPDATES"]),9600); self.assertEqual(len(tables["PROBES"]),4)
        row=next(x for x in tables["PAIRED_E"] if x["split"]=="tune" and x["mask"]=="all_observed" and x["scale"]=="normalized")
        self.assertAlmostEqual(row["MSE_learn_minus_fixed"],-.1)
        self.assertTrue(any(x["method"]=="CP0_LS" for x in tables["SCORES"]))
        self.assertEqual(len(tables["CHANNEL_COMPARISONS"]),336)
        channel=tables["CHANNEL_COMPARISONS"][0]; self.assertAlmostEqual(channel["learn_minus_CP0"],channel["learn_MSE"]-channel["CP0_LS_MSE"])
        self.assertIsNone(tables["CONDITIONS"][0]["independent_parameter_change"])
    def test_later_checkpoint_tie_is_rejected(self):
        values=fixture_values(self.folder); row=next(iter(values["train"]["datasets"]["robin"]["conditions"].values())); row["selected_update"]=400
        with self.assertRaisesRegex(ValueError,"EARLIEST_TUNE"): self.r.collect_tables(values)
    def test_update_gap_or_clipping_mismatch_is_rejected(self):
        values=fixture_values(self.folder); row=next(iter(values["train"]["datasets"]["robin"]["conditions"].values()))
        path=Path(row["updates_log"]["path"]); text=path.read_text(); path.write_text(text.replace('"update": 1,','"update": 2,',1))
        row["updates_log"]=self.r.record(path)
        with self.assertRaisesRegex(ValueError,"UPDATE_SEQUENCE"): self.r.collect_tables(values)
    def test_ratios_are_distinct_and_all_nine_routes_are_kept(self):
        values=fixture_values(self.folder); rows=values["cost"]["cost"]["robin"]["rows"]
        row=rows[1]; row["seconds"]=[.001]*5+[.1]*4; row["median_seconds"]=.001; row["median_seconds_per_window"]=.001
        raw=rows[0]["seconds"]; row["paired_RAW_ratios"]=[x/y for x,y in zip(row["seconds"],raw)]; row["paired_RAW_median_ratio"]=statistics.median(row["paired_RAW_ratios"])
        tables=self.r.collect_tables(values); result=tables["WHOLE_COST"][1]
        self.assertNotAlmostEqual(result["ratio_of_RAW_medians"],result["median_paired_RAW_ratio"])
        self.assertEqual(len(tables["WHOLE_COST"]),36); self.assertEqual(len(tables["WHOLE_COST_SAMPLES"]),324)
    def test_missing_same_seed_partner_is_not_silently_dropped(self):
        values=fixture_values(self.folder); values["train"]["datasets"]["robin"]["conditions"].pop("robin_L0_E_LEARN_s92601")
        with self.assertRaisesRegex(ValueError,"EXACT_16"): self.r.collect_tables(values)
    def test_csv_nested_paths_are_relative_and_protected_target_never_opened(self):
        values=fixture_values(self.folder); output=self.folder/"reports"/"toy"; output.mkdir(parents=True)
        row=values["train"]["datasets"]["robin"]["conditions"]["robin_L0_E_FIXED_s92601"]
        result=self.r.portable({"log":row["updates_log"],"state":row["selected_state"]},output)
        self.assertFalse(Path(result["log"]["path"]).is_absolute()); self.assertEqual(result["log"]["sha256"],row["updates_log"]["sha256"])
        self.assertFalse(Path(result["state"]["path"]).is_absolute()); self.assertFalse(Path(row["selected_state"]["path"]).exists())
    def test_report_text_separates_gradient_presence_from_necessity(self):
        values=fixture_values(self.folder); tables=self.r.collect_tables(values)
        text=self.r.make_text(values,tables,{},self.folder)
        for phrase in ("필요성", "기노출", "joint-Y", "RF3", "독립 일반화", "PCA", "600", "9,600", "4회"):
            self.assertIn(phrase,text)
    def test_metadata_reader_rejects_scientific_binary_and_nan(self):
        with self.assertRaisesRegex(ValueError,"JSON_METADATA"): self.r.read(self.folder/"model.npz")
        path=self.folder/"bad.json"; path.write_text('{"x":NaN}')
        with self.assertRaisesRegex(ValueError,"NONFINITE_JSON"): self.r.read(path)
    def test_generated_only_csv_png_svg_manifest_and_overwrite_guard(self):
        values=fixture_values(self.folder/"metadata"); output=self.folder/"reports"/"toy"
        with patch.object(self.r,"HERE",self.folder),patch.object(self.r,"load_verified",return_value=(values,{})):
            manifest=self.r.generate({},output)
            self.assertEqual(manifest["figure_png_count"],6); self.assertEqual(manifest["figure_svg_count"],6)
            for entry in manifest["files"]:
                path=output/entry["path"]; self.assertEqual(self.r.sha(path),entry["sha256"])
                if path.suffix==".png": self.assertEqual(path.read_bytes()[:8],b"\x89PNG\r\n\x1a\n")
                if path.suffix==".svg": ET.parse(path)
            with (output/"tables/UPDATES.csv").open(newline="",encoding="utf-8-sig") as handle: self.assertEqual(len(list(csv.DictReader(handle))),9600)
            with self.assertRaisesRegex(ValueError,"FRESH"): self.r.generate({},output)


if __name__=="__main__": unittest.main()
