import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .audit import ROOT, RESULTS, sha256, utc_now, write_json
from .chronos_wrapper import load_pipeline, point_forecast
from .data import load_and_audit
from .metrics import seasonal_scales, level_weights, score
from .model import ForecastModel
from .predict import predict_origins, targets, past_context
from .reconcile import Reconciliation
from .smoke import ARMS
from .train import train_update, training_tensors

METHODS = ["unreconciled", "bottom_up", "mint_shrink", "ols"]


def select_candidate(candidates):
    if any(c["selection_role"] != "VALIDATION" for c in candidates):
        raise ValueError("Selection accepts VALIDATION metrics only")
    return min(candidates, key=lambda c: (c["validation_primary"], c["lr"], c["checkpoint"]))


def continuation(effects):
    checks = {}
    for other in ["LORA_SELF", "LORA_POOL"]:
        pair = [e for e in effects if e["comparator"] == other]
        mint = [e for e in pair if e["reconciliation"] == "mint_shrink"]
        raw = [e for e in pair if e["reconciliation"] == "unreconciled"]
        checks[f"{other}_mean_primary_favorable"] = len(mint) == 3 and np.mean([e["primary_difference"] for e in mint]) < -1e-8
        repeats = [e for e in mint if e["seed"] in [92111, 92112]]
        checks[f"{other}_both_repeats_favorable"] = len(repeats) == 2 and all(e["primary_difference"] < -1e-8 for e in repeats)
        checks[f"{other}_bottom_no_large_harm"] = len(mint) == 3 and all(e["bottom_relative_difference"] <= 0.05 for e in mint)
        checks[f"{other}_raw_mean_favorable"] = len(raw) == 3 and np.mean([e["primary_difference"] for e in raw]) < -1e-8
    return {"decision": "PROCEED_TO_TOURISM_CONFIRMATION" if all(checks.values()) else "STOP_CURRENT_HIER_ADAPTER_AFTER_LABOUR_SCREEN",
            "checks": checks, "nature": "project budget continuation rule, not statistical significance"}


class Screen:
    def __init__(self, stage):
        self.stage = stage
        self.group = "Labour" if stage == "A" else "TourismLarge"
        self.out = RESULTS if stage == "A" else RESULTS / "tourism_confirmation"
        self.out.mkdir(exist_ok=True)
        self.cache = ROOT / ".cache" / f"stage_{stage}"
        self.cache.mkdir(exist_ok=True)
        self.ds = load_and_audit(self.group, write_artifacts=False)
        self.origins = self.ds.split_manifest["origins"]
        train_end = self.ds.split_manifest["role_bounds"]["TRAIN"]["end_exclusive"]
        self.scales, self.scale_audit = seasonal_scales(self.ds.values[:, :train_end])
        self.weights = level_weights(len(self.ds.ids), self.ds.tags)
        self.cal_y = targets(self.ds.values, self.origins["CALIBRATION"])
        self.val_y = targets(self.ds.values, self.origins["VALIDATION"])
        self.snapshot = json.loads((RESULTS / "MODEL_SOURCE_MANIFEST.json").read_text())["snapshot_path"]
        self.budget = {"stage": stage, "dataset": self.group, "fits_started": 0, "fits_completed": 0, "main_updates": 0,
                       "max_fits": 16 if stage == "A" else 8, "max_main_updates": 8192 if stage == "A" else 4096,
                       "max_updates_per_fit": 512, "smoke_updates_separate": True}
        self.checkpoints, self.predictions, self.reconciliation_audits, self.resources = [], [], [], []
        self.candidates, self.selected = [], []
        self.raw_rows, self.level_rows, self.origin_rows = [], [], []
        self.started = time.monotonic()

    def flush(self):
        write_json(self.out / "TRAINING_BUDGET.json", self.budget)
        write_json(self.out / "CHECKPOINT_MANIFEST.json", self.checkpoints)
        write_json(self.out / "PREDICTIONS_MANIFEST.json", self.predictions)
        write_json(self.out / "RECONCILIATION_AUDIT.json", self.reconciliation_audits)
        write_json(self.out / "MODEL_SELECTION.json", {"candidates": self.candidates, "selected": self.selected,
                    "criterion": "VALIDATION mint_shrink level-balanced RMSSE; lower LR then earlier checkpoint"})
        pd.DataFrame(self.resources).to_csv(self.out / "RESOURCES.csv", index=False)

    def artifact(self, path, **info):
        return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size, **info}

    def evaluate_checkpoint(self, model, fit_id, arm, seed, lr, step):
        path = self.cache / f"{fit_id}_step{step}.pt"
        torch.save(model.trainable_state(), path)
        self.checkpoints.append(self.artifact(path, fit_id=fit_id, checkpoint=step, base_model_revision="772f3d25d38aec6d914c8949dab4462e2d46f5d8"))
        cal = predict_origins(model, self.ds.values, self.origins["CALIBRATION"], self.ds.parents, self.ds.children)
        val = predict_origins(model, self.ds.values, self.origins["VALIDATION"], self.ds.parents, self.ds.children)
        rec = Reconciliation(self.ds.S, self.ds.bottom_indices).fit(self.cal_y, cal, role="CALIBRATION")
        metric, _ = score(self.val_y, rec.transform(val, "mint_shrink"), self.scales, self.ds.tags, self.ds.bottom_indices, self.ds.S)
        pred_path = self.cache / f"{fit_id}_step{step}_selection.npz"
        np.savez_compressed(pred_path, calibration=cal, validation=val,
                            calibration_origins=self.origins["CALIBRATION"], validation_origins=self.origins["VALIDATION"])
        self.predictions.append(self.artifact(pred_path, fit_id=fit_id, checkpoint=step, roles=["CALIBRATION", "VALIDATION"]))
        self.reconciliation_audits.append({"fit_id": fit_id, "checkpoint": step, **rec.audit})
        candidate = {"fit_id": fit_id, "arm": arm, "seed": seed, "lr": lr, "checkpoint": step,
                     "validation_primary": metric["primary"], "selection_role": "VALIDATION",
                     "state_path": path.relative_to(ROOT).as_posix(), "predictions_path": pred_path.relative_to(ROOT).as_posix()}
        self.candidates.append(candidate)
        print(f"{fit_id} step={step} validation_mint={metric['primary']:.7f}", flush=True)
        return candidate

    def fit(self, arm, seed, lr):
        if self.budget["fits_started"] >= self.budget["max_fits"]:
            raise RuntimeError("Fit cap reached")
        fit_id = f"{self.group}_{arm}_s{seed}_lr{lr:.0e}"
        self.budget["fits_started"] += 1
        self.flush()
        t0 = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        model = ForecastModel(self.snapshot, arm, seed)
        base_before = model.frozen_hash()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0)
        candidates = [self.evaluate_checkpoint(model, fit_id, arm, seed, lr, 0)]
        origin_schedule = np.random.default_rng(seed).choice(self.origins["TRAIN"], size=512, replace=True)
        for update, origin in enumerate(origin_schedule, 1):
            if self.budget["main_updates"] >= self.budget["max_main_updates"]:
                raise RuntimeError("Main update cap reached")
            tensors = training_tensors(self.ds.values, int(origin), self.scales, self.weights)
            loss, norm = train_update(model, optimizer, *tensors, self.ds.parents, self.ds.children, chunk_size=64)
            self.budget["main_updates"] += 1
            entry = {"kind": "main", "stage": self.stage, "fit_id": fit_id, "arm": arm, "seed": seed, "lr": lr,
                     "update": update, "origin": int(origin), "loss": loss, "grad_norm": norm}
            with (self.out / "UPDATE_LEDGER.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry) + "\n")
            if update in [128, 256, 512]:
                candidates.append(self.evaluate_checkpoint(model, fit_id, arm, seed, lr, update))
                self.flush()
            if update % 32 == 0:
                write_json(self.out / "PROGRESS.json", {"status": "RUNNING", "updated_at": utc_now(), "fit_id": fit_id,
                     "fit_index": self.budget["fits_started"], "update": update, "main_updates": self.budget["main_updates"],
                     "elapsed_seconds": time.monotonic() - self.started})
        assert model.frozen_hash() == base_before
        selected = dict(select_candidate(candidates))
        selected["optimization_limit_reached"] = candidates[-1]["validation_primary"] < candidates[-2]["validation_primary"]
        self.resources.append({"dataset": self.group, "kind": "neural_fit", "fit_id": fit_id, "arm": arm, "seed": seed,
                               "lr": lr, "updates": 512, "seconds": time.monotonic() - t0,
                               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                               "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad), "precision": "float32"})
        self.budget["fits_completed"] += 1
        self.flush()
        del model, optimizer
        gc.collect()
        torch.cuda.empty_cache()
        return selected

    def record_test(self, arm, seed, lr, checkpoint, test_prediction, calibration_prediction):
        rec = Reconciliation(self.ds.S, self.ds.bottom_indices).fit(self.cal_y, calibration_prediction, role="CALIBRATION")
        y = targets(self.ds.values, self.origins["TEST"])
        key = f"{arm}_s{seed}"
        path = self.cache / f"{key}_test.npz"
        outputs = {method: rec.transform(test_prediction, method) for method in METHODS}
        np.savez_compressed(path, **outputs, test_origins=self.origins["TEST"])
        self.predictions.append(self.artifact(path, arm=arm, seed=seed, roles=["TEST"], selected_before_test=True))
        for method, prediction in outputs.items():
            metric, levels = score(y, prediction, self.scales, self.ds.tags, self.ds.bottom_indices, self.ds.S)
            keys = {"dataset": self.group, "arm": arm, "seed": seed, "lr": lr, "checkpoint": checkpoint, "reconciliation": method}
            self.raw_rows.append({**keys, **metric})
            for level, value in levels.items():
                self.level_rows.append({**keys, "level": level, "rmsse": value})
            for i, origin in enumerate(self.origins["TEST"]):
                one, _ = score(y[i:i+1], prediction[i:i+1], self.scales, self.ds.tags, self.ds.bottom_indices, self.ds.S)
                self.origin_rows.append({**keys, "origin": origin, "target_start": str(self.ds.dates[origin].date()), **one})
        self.reconciliation_audits.append({"arm": arm, "seed": seed, "evaluation_role": "TEST", **rec.audit})
        pd.DataFrame(self.raw_rows).to_csv(self.out / "RAW_SCORES.csv", index=False)
        pd.DataFrame(self.level_rows).to_csv(self.out / "LEVEL_SCORES.csv", index=False)
        pd.DataFrame(self.origin_rows).to_csv(self.out / "ORIGIN_SCORES.csv", index=False)
        self.flush()

    def evaluate_selected(self):
        for item in self.selected:
            model = ForecastModel(self.snapshot, item["arm"], item["seed"])
            model.restore_trainable(torch.load(ROOT / item["state_path"], weights_only=True))
            pred = predict_origins(model, self.ds.values, self.origins["TEST"], self.ds.parents, self.ds.children)
            with np.load(ROOT / item["predictions_path"]) as saved:
                calibration = saved["calibration"]
            self.record_test(item["arm"], item["seed"], item["lr"], item["checkpoint"], pred, calibration)
            del model
            gc.collect()
            torch.cuda.empty_cache()

    def baselines(self):
        from statsforecast.models import SeasonalNaive, AutoETS
        for arm in ["F0", "SeasonalNaive", "AutoETS"]:
            t0 = time.monotonic()
            pipeline = load_pipeline(self.snapshot) if arm == "F0" else None
            if pipeline:
                pipeline.model.eval()
            predictions = {}
            for role in ["CALIBRATION", "TEST"]:
                per_origin = []
                for origin in self.origins[role]:
                    if pipeline:
                        with torch.no_grad():
                            p = point_forecast(pipeline.model, past_context(self.ds.values, origin)).cpu().numpy()
                    else:
                        cls = SeasonalNaive if arm == "SeasonalNaive" else AutoETS
                        p = np.stack([cls(season_length=12).forecast(y=series[origin-48:origin].copy(), h=12)["mean"]
                                      for series in self.ds.values])
                    per_origin.append(p)
                predictions[role] = np.stack(per_origin)
            self.record_test(arm, -1, 0, 0, predictions["TEST"], predictions["CALIBRATION"])
            self.resources.append({"dataset": self.group, "kind": "evaluation_baseline", "fit_id": arm, "arm": arm,
                                   "seed": -1, "lr": 0, "updates": 0, "seconds": time.monotonic() - t0,
                                   "trainable_params": 0, "precision": "float32" if pipeline else "float64"})
            del pipeline
            gc.collect()
            torch.cuda.empty_cache()
            self.flush()

    def seed_effects(self):
        rows = []
        for h in self.raw_rows:
            if h["arm"] != "LORA_HIER":
                continue
            for other in ["LORA", "LORA_SELF", "LORA_POOL"]:
                o = next(r for r in self.raw_rows if r["arm"] == other and r["seed"] == h["seed"] and r["reconciliation"] == h["reconciliation"])
                rows.append({"dataset": self.group, "seed": h["seed"], "comparator": other, "reconciliation": h["reconciliation"],
                             "primary_difference": h["primary"] - o["primary"],
                             "primary_relative_difference": h["primary"] / o["primary"] - 1,
                             "bottom_relative_difference": h["bottom_rmsse"] / o["bottom_rmsse"] - 1,
                             "raw_mae_difference": h["raw_mae"] - o["raw_mae"]})
        pd.DataFrame(rows).to_csv(self.out / "SEED_EFFECTS.csv", index=False)
        return rows

    def run(self):
        if self.stage == "B":
            gate = json.loads((RESULTS / "CONTINUATION_DECISION.json").read_text())
            if gate["decision"] != "PROCEED_TO_TOURISM_CONFIRMATION":
                raise RuntimeError("TourismLarge blocked by Labour project gate")
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if self.stage == "A":
            lr_choices = {}
            for arm in ARMS:
                choices = [self.fit(arm, 92110, lr) for lr in [1e-4, 3e-4]]
                selected = select_candidate(choices)
                self.selected.append(selected)
                lr_choices[arm] = selected["lr"]
            write_json(self.out / "LR_SELECTION.json", {"selection_seed": 92110, "selection_role": "VALIDATION", "arm_lrs": lr_choices})
            for seed in [92111, 92112]:
                for arm in ARMS:
                    self.selected.append(self.fit(arm, seed, lr_choices[arm]))
        else:
            lr_choices = json.loads((RESULTS / "LR_SELECTION.json").read_text())["arm_lrs"]
            write_json(self.out / "LR_SELECTION.json", {"source": "Labour", "new_search": False, "arm_lrs": lr_choices})
            for seed in [92211, 92212]:
                for arm in ARMS:
                    self.selected.append(self.fit(arm, seed, lr_choices[arm]))
        self.flush()
        # All LR/checkpoint choices are persisted before any TEST targets are read for scoring.
        self.evaluate_selected()
        self.baselines()
        effects = self.seed_effects()
        if self.stage == "A":
            write_json(self.out / "CONTINUATION_DECISION.json", continuation(effects))
        self.flush()
        write_json(self.out / "PROGRESS.json", {"status": "COMPLETE", "updated_at": utc_now(),
                   "main_updates": self.budget["main_updates"], "elapsed_seconds": time.monotonic() - self.started})
