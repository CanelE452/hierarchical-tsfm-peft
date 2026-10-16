"""Derived-only CPU figures: exact inputs, cohort checks and locked T boundary."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import data_contract as data
import prediction_cache
import route_catalog
import measure_costs
from types import SimpleNamespace


class ReportFiguresTests(unittest.TestCase):
    def setUp(self):
        import report_figures
        self.module = report_figures
        self.tmp = tempfile.TemporaryDirectory(prefix="informs_figures_")
        self.addCleanup(self.tmp.cleanup)
        self.public, self.private = Path(self.tmp.name) / "public", Path(self.tmp.name) / "private"
        self.public.mkdir(); self.private.mkdir()
        x = np.cumsum(np.random.default_rng(14).normal(size=(240, 3)), axis=0).astype(np.float32)
        mask = np.ones_like(x, bool); mask[173:176, 0] = False
        self.task = dict(name="toy", physical=x, observed=mask, columns=["a", "b", "c"],
                         times=np.arange(240, dtype=np.int64) * 3600 * 10**9,
                         interval_seconds=3600, L=64, H=4, data_sha256="toy_data", metadata={"source_kind": "physical"})
        self.rows = []
        for name, family, baseline in (("current", "CURRENT", "CURRENT"), ("seasonal", "SEASONAL_NAIVE", "SEASONAL")):
            prediction_cache.predict_parent(name, self.task, "V_select", self.private, self.public,
                                            None, baseline=dict(kind=baseline, period=4))
            self.rows.append(dict(experiment_id=name, dataset="toy", family=family, latent_k=0,
                                  seed=19, fold="primary", length=64, horizon=4, num_channels=3))
        self.write_registry()

    def write_registry(self):
        with (self.public / "EXPERIMENT_REGISTRY.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader(); writer.writerows(self.rows)

    def generate(self, split="V_select"):
        with patch.object(self.module.data, "load_task", return_value=self.task):
            return self.module.generate(self.public, self.private, split=split)

    def test_real_bundles_create_nonempty_figures_and_private_trace_data(self):
        result = self.generate()
        directory = Path(result)
        manifest = json.loads((directory / "FIGURE_MANIFEST.json").read_text())
        self.assertEqual(manifest["verified_prediction_count"], 2)
        self.assertEqual(manifest["split"], "V_select")
        self.assertFalse(manifest["independent_generalization_claimed"])
        groups = {row["group"]: row for row in manifest["groups"]}
        self.assertTrue(groups["accuracy_by_k"]["figures"])
        self.assertTrue(groups["representative_traces"]["figures"])
        self.assertEqual(groups["matched_controls"]["status"], "PENDING")
        self.assertEqual(groups["cost_accuracy"]["status"], "PENDING")
        for row in manifest["outputs"]:
            self.assertEqual(data.sha256_file(directory / row["file"]), row["sha256"])
        self.assertTrue(list(directory.glob("*.png")))
        self.assertTrue(list(directory.glob("*.svg")))
        self.assertTrue(list((self.private / "figures" / directory.name).glob("*trace*.csv.gz")))
        for path in directory.glob("*.csv"):
            self.assertNotIn("truth_native", path.read_text())
        self.assertNotIn(str(self.private), (directory / "FIGURE_MANIFEST.json").read_text())

    def test_snapshot_is_unique_and_preserves_previous_bytes(self):
        first = Path(self.generate())
        before = {p.name: data.sha256_file(p) for p in first.iterdir() if p.is_file()}
        second = Path(self.generate())
        self.assertNotEqual(first, second)
        self.assertEqual(before, {p.name: data.sha256_file(p) for p in first.iterdir() if p.is_file()})

    def test_t_figures_require_locked_and_allowed_artifact_without_opening_cache(self):
        with self.assertRaisesRegex(ValueError, "SELECTION_LOCK"):
            self.generate("T_final")
        (self.public / "SELECTION_LOCK.json").write_text(json.dumps({"locked": False}))
        with self.assertRaisesRegex(ValueError, "locked"):
            self.generate("T_final")

    def test_allowed_t_uses_only_fixed_cases_and_wrong_artifact_stays_closed(self):
        v = json.loads((self.public / "predictions/current/V_select/PREDICTION_MANIFEST.json").read_text())
        lock = dict(locked=True, allowed_run_ids=["current"],
                    allowed_artifact_sha={"current": v["metadata"]["artifact_sha256"]})
        (self.public / "SELECTION_LOCK.json").write_text(json.dumps(lock))
        prediction_cache.predict_parent("current", self.task, "T_final", self.private, self.public,
                                        None, baseline=dict(kind="CURRENT", period=4))
        directory = Path(self.generate("T_final"))
        manifest = json.loads((directory / "FIGURE_MANIFEST.json").read_text())
        self.assertTrue(manifest["T_selection_lock_checked"])
        self.assertFalse(manifest["T_method_selection_performed"])
        import gzip
        with gzip.open(self.private / manifest["traces"][0]["private_plotdata_relative_path"], "rt") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len({r["selection"] for r in rows}), 4)
        self.assertFalse(any("POST-HOC" in r["selection"] for r in rows))
        lock["allowed_artifact_sha"]["current"] = "different"
        (self.public / "SELECTION_LOCK.json").write_text(json.dumps(lock))
        with patch.object(self.module.independent, "_load_arrays", side_effect=AssertionError("T cache must stay closed")):
            with self.assertRaisesRegex(ValueError, "No checked locked"):
                self.generate("T_final")

    def test_tampered_private_prediction_is_excluded_with_explicit_error(self):
        path = self.private / "predictions/current/V_select.npz"
        path.write_bytes(path.read_bytes() + b"different")
        result = Path(self.generate())
        manifest = json.loads((result / "FIGURE_MANIFEST.json").read_text())
        self.assertEqual(manifest["verified_prediction_count"], 1)
        self.assertEqual(len(manifest["excluded_prediction_inputs"]), 1)
        self.assertIn("archive", manifest["excluded_prediction_inputs"][0]["reason"].lower())

    def test_budget_is_part_of_curve_identity_and_never_collapsed(self):
        a = dict(dataset="toy", fold="primary", family="B0", seed=1, k=2, budget="0.25", cohort="c")
        b = dict(a, budget="3")
        self.assertNotEqual(self.module.curve_identity(a), self.module.curve_identity(b))

    def test_source_lr_and_config_are_preserved_and_lr_separates_existing_figure(self):
        inputs, registry = {}, {}
        for family in ("B0", "B2"):
            for lr in (.001, .0001):
                run_id = f"{family}_lr{lr}"
                folder = self.public / "runs" / run_id; folder.mkdir(parents=True)
                registry[run_id] = dict(experiment_id=run_id)
                (folder / "BASE_FIT_RECEIPT.json").write_text(json.dumps(dict(
                    status="COMPLETE_VALID", run_id=run_id, epochs=[.25, 1.], family=family)))
                (folder / "SOURCE_CONTRACT.json").write_text(json.dumps(dict(
                    spec=dict(run_id=run_id, lr=lr), config_sha="config_" + run_id)))
        fits = self.module._fits(self.public, registry, inputs)
        rows = [self.figure_row(run_id.split("_")[0], 1, .2 + i / 100,
                               run_id=run_id, lr=fit["lr"], fit_config_sha=fit["config_sha"], budget="1.0")
                for i, (run_id, fit) in enumerate(fits.items())]
        self.assertEqual(len({self.module.curve_identity(row) for row in rows}), 4)
        self.assertEqual(fits["B0_lr0.0001"]["lr"], .0001)
        self.assertEqual(fits["B0_lr0.001"]["config_sha"], "config_B0_lr0.001")
        self.assertEqual(sum(key[1].endswith("SOURCE_CONTRACT.json") for key in inputs), 4)
        captured = []
        def save(fig, *args, **kwargs):
            captured.append(fig); return ["lr.png"]
        directory = self.public / "lr_accuracy"; directory.mkdir()
        with patch.object(self.module, "_save", side_effect=save):
            files = self.module._accuracy(rows, directory, "V_select")
        self.assertEqual(files, ["lr.png"])
        fig = captured[0]; self.addCleanup(self.module.plt.close, fig)
        labels = [text.get_text() for text in fig.legends[0].get_texts()]
        self.assertEqual(sum("lr=0.001" in label for label in labels), 2)
        self.assertEqual(sum("lr=0.0001" in label for label in labels), 2)
        plotted = [line for line in fig.axes[0].lines if line.get_label() in labels]
        self.assertEqual(len(plotted), 4)
        self.assertEqual(len({line.get_linestyle() for line in plotted}), 2)
        with next(directory.glob("*.csv")).open() as stream:
            saved = list(csv.DictReader(stream))
        self.assertEqual({row["run_id"] for row in saved}, set(fits))
        self.assertEqual({row["fit_config_sha"] for row in saved}, {f["config_sha"] for f in fits.values()})

    def test_matched_lr_conditions_do_not_mix_or_drop_valid_same_lr_pairs(self):
        rows = [self.figure_row(family, 1, .2, run_id=f"{family}_{lr}", lr=lr,
                               fit_config_sha=f"config_{family}_{lr}")
                for family in ("B0", "B2") for lr in (.001, .0001)]
        directory = self.public / "lr_matched"; directory.mkdir()
        self.module._matched(rows, directory, "V_select")
        with (directory / "matched_controls.csv").open() as stream:
            saved = list(csv.DictReader(stream))
        self.assertEqual(len(saved), 2)
        for row in saved:
            self.assertEqual(float(row["left_lr"]), float(row["right_lr"]))
            self.assertTrue(row["left_config_sha"].endswith(row["left_run_id"]))
            self.assertTrue(row["right_config_sha"].endswith(row["right_run_id"]))
        other = self.public / "lr_mismatch"; other.mkdir()
        mismatched = [self.figure_row("B0", 1, .2, lr=.001), self.figure_row("B2", 1, .2, lr=.0001)]
        self.assertFalse(self.module._matched(mismatched, other, "V_select"))

    def figure_row(self, family, k, value, **overrides):
        receipt = dict(search_updates=181, refit_updates=0)
        row = dict(dataset="toy", fold="primary", cohort="c", seed=19, scaler_sha="s", data_sha="d",
                   cohort_arrays_sha="a", budget="0.25", k=k, mse=value, mae=value / 2,
                   calibration_parent_signature="", parent_b_run_id="", family=family,
                   run_id=family + str(k), selected_epoch=0., last_cp_best=False,
                   training_status="LIMITED_SCREEN", fitted=dict(receipt=receipt))
        row.update(overrides)
        return row

    def test_exact_zero_matched_differences_have_visible_markers_and_numeric_labels(self):
        directory = self.public / "zero_matched"; directory.mkdir()
        captured = []
        def save(fig, *args, **kwargs):
            captured.append((fig, args[2])); return ["zero.png"]
        rows = [self.figure_row(family, k, .2) for family in ("B0", "B2") for k in (1, 2, 3)]
        with patch.object(self.module, "_save", side_effect=save):
            self.module._matched(rows, directory, "V_select")
        fig, caption = captured[0]
        self.addCleanup(self.module.plt.close, fig)
        for ax in fig.axes:
            offsets = np.concatenate([item.get_offsets() for item in ax.collections])
            self.assertEqual(len(offsets), 3)
            np.testing.assert_array_equal(offsets[:, 0], np.zeros(3))
            self.assertEqual(sum(text.get_text() == "0" for text in ax.texts), 3)
            self.assertTrue(all(bar.get_width() == 0 for bar in ax.patches))
        self.assertIn("all paired differences are exactly 0", " ".join(caption.lower().split()))

    def test_accuracy_cp0_fixed_pca_and_outlier_keep_full_context(self):
        directory = self.public / "readable_accuracy"; directory.mkdir()
        rows = [self.figure_row("B0", 1, .2), self.figure_row("B2", 1, .2),
                self.figure_row("PCA_FIXED_BOLT", 1, .2, last_cp_best=True,
                                fitted=dict(receipt=dict(search_updates=0, refit_updates=0))),
                self.figure_row("CURRENT", 0, 1.43, budget="fixed_no_optimizer", fitted=None)]
        captured = []
        def save(fig, *args, **kwargs):
            captured.append((fig, args[2])); return ["readable.png"]
        with patch.object(self.module, "_save", side_effect=save):
            self.module._accuracy(rows, directory, "V_select")
        fig, caption = captured[0]
        self.addCleanup(self.module.plt.close, fig)
        labels = [text.get_text() for legend in fig.legends for text in legend.get_texts()]
        self.assertTrue(any("B0" in label and "CP0" in label for label in labels))
        self.assertTrue(any("PCA" in label and "no optimizer" in label for label in labels))
        self.assertNotIn("convergence unresolved", caption)
        self.assertEqual(sum(len(ax.collections) for ax in fig.axes), 0)
        full = [ax for ax in fig.axes if "full range" in ax.get_title().lower()]
        zoom = [ax for ax in fig.axes if "zoom" in ax.get_title().lower()]
        self.assertEqual(len(full), 2); self.assertEqual(len(zoom), 2)
        self.assertGreater(full[0].get_ylim()[1], 1.43)
        self.assertLess(zoom[0].get_ylim()[1], 1.43)
        self.assertIn("CURRENT", caption)

    def test_matched_pairs_require_same_budget_and_same_frozen_parents(self):
        common = dict(dataset="toy", fold="primary", cohort="c", seed=1, scaler_sha="s", data_sha="d",
                      cohort_arrays_sha="a", budget=".25", k=2, mse=1., mae=.5,
                      calibration_parent_signature="parents", parent_b_run_id="raw")
        b0 = dict(common, family="B0", run_id="b0")
        b2 = dict(common, family="B2", run_id="b2", mse=.9)
        directory = self.public / "matched_toy"; directory.mkdir()
        self.assertTrue(self.module._matched([b0, b2], directory, "V_select"))
        other = self.public / "unmatched_toy"; other.mkdir()
        self.assertFalse(self.module._matched([b0, dict(b2, budget="3")], other, "V_select"))
        h = dict(common, family="CONDITIONAL", run_id="h")
        past = dict(common, family="PAST_CONTROL", run_id="past", calibration_parent_signature="other")
        self.assertFalse(self.module._matched([h, past], other, "V_select"))

    def test_cost_group_requires_actual_hashed_trials_and_explicit_timer_scope(self):
        costdir = self.public / "costs"; costdir.mkdir()
        trials = costdir / "toy.trials.csv.gz"
        import gzip
        with gzip.open(trials, "wt") as stream:
            stream.write("synthetic_unit_test_trial\n1\n")
        receipt = dict(scope="fresh_process_single_route", run_id="current", dataset="toy", family="CURRENT",
            data_sha256="toy_data",
            batch_size=1, device="cpu", dtype="float32", status="SMOKE_OR_PARTIAL_COST_PROTOCOL",
            summaries=[dict(run_id="current", phase="timed", median_seconds=.01, p90_seconds=.011, trials=2)],
            memory=dict(peak_reserved_MiB=0, resident_weights_gpu_MiB=0, resident_weights_cpu_MiB=0),
            RSS_MiB_after_close=100, training_peak_included=False, cached_forecast_reads=0,
            trials_file=trials.name, trials_sha256=data.sha256_file(trials))
        (costdir / "toy.json").write_text(json.dumps(receipt))
        directory = Path(self.generate())
        with (directory / "cost_accuracy.csv").open() as stream:
            row = list(csv.DictReader(stream))[0]
        self.assertEqual(float(row["median_whole_ms"]), 10.)
        self.assertEqual(row["timer_scope"], "fresh_process_single_route")
        self.assertEqual(row["cost_protocol_status"], "SMOKE_OR_PARTIAL_COST_PROTOCOL")
        trials.write_bytes(trials.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ValueError, "cost trials hash"):
            self.generate()

    def aliased_cost(self):
        spec,_=route_catalog.build_route('current',self.task,self.private,self.public)
        spec.update(run_id='current__rows128',accuracy_run_id='current',row_microbatch=128)
        costdir=self.public/'costs';costdir.mkdir()
        path=self.private/'cost_spec.json';path.write_text(json.dumps(spec))
        args=SimpleNamespace(cache_root=str(self.private),public_root=str(self.public),device='cpu',route_spec=str(path),
                             batch=1,model=None,warmups=1,timed_trials=2,stage_trials=1,output=str(costdir/'aliased.json'))
        task=dict(self.task,split_ranges=data.split_ranges(self.task))
        with patch.object(measure_costs,'load_task',return_value=task):receipt=measure_costs.worker(args)
        return receipt,costdir/'aliased.json'

    def test_explicit_actual_cost_alias_maps_to_original_and_creates_figure(self):
        receipt,_=self.aliased_cost()
        self.assertEqual(receipt['accuracy_bindings']['current__rows128']['accuracy_run_id'],'current')
        directory=Path(self.generate())
        manifest=json.loads((directory/'FIGURE_MANIFEST.json').read_text())
        group=next(row for row in manifest['groups'] if row['group']=='cost_accuracy')
        self.assertTrue(group['figures'])
        with (directory/'cost_accuracy.csv').open() as stream:row=list(csv.DictReader(stream))[0]
        self.assertEqual(row['run_id'],'current')
        self.assertEqual(row['condition_id'],'current__rows128')
        self.assertEqual(row['row_microbatch'],'128')

    def test_explicit_cost_mapping_cannot_substitute_another_artifact(self):
        receipt,path=self.aliased_cost()
        receipt['accuracy_bindings']['current__rows128']['artifact_bindings']={'route':dict(cache_relative_path='wrong.pt',sha256='wrong')}
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError,'artifact'):
            self.generate()

    def test_balanced_cost_exports_per_route_binding_and_memory(self):
        _,path=self.aliased_cost()
        specs=[]
        for run_id in ('current','seasonal'):
            spec,_=route_catalog.build_route(run_id,self.task,self.private,self.public)
            spec.update(run_id=run_id+'__rows128',accuracy_run_id=run_id,row_microbatch=128)
            specs.append(spec)
        source=self.private/'balanced_spec.json';source.write_text(json.dumps(specs))
        args=SimpleNamespace(cache_root=str(self.private),public_root=str(self.public),device='cpu',route_spec=str(source),
                             batch=1,model=None,warmups=1,timed_trials=2,stage_trials=1,output=str(path.parent/'balanced.json'))
        task=dict(self.task,split_ranges=data.split_ranges(self.task))
        with patch.object(measure_costs,'load_task',return_value=task):receipt=measure_costs.balanced_driver(args)
        self.assertEqual(set(receipt['accuracy_bindings']),{'current__rows128','seasonal__rows128'})
        self.assertEqual(set(receipt['memory']),set(receipt['accuracy_bindings']))
        directory=Path(self.generate())
        with (directory/'cost_accuracy.csv').open() as stream:rows=list(csv.DictReader(stream))
        self.assertEqual(sum(row['timer_scope']=='general_shared_F_balanced' for row in rows),2)

    def test_cost_mapping_cannot_replace_accuracy_with_equal_data_other_method(self):
        receipt,path=self.aliased_cost()
        receipt['accuracy_bindings']['current__rows128']['accuracy_run_id']='seasonal'
        path.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError,'accuracy'):
            self.generate()


if __name__ == "__main__":
    unittest.main()
