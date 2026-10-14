"""Post-hoc causal memory retrieval for the Bi-Hash pilot.

The paired auxiliary pilot trains hash projections, but the true future target
is unavailable at inference.  This script asks the stricter deployment question:
using only a FIT future-pattern memory bank that has ended by the query origin,
does the learned Bi-Hash retrieve useful latent future coordinates?
"""
from pathlib import Path
import argparse
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PILOT = HERE / "bihash_pilot.py"
SMALLK = ROOT / "research" / "smallk_predictive_alignment_20261013"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SMALLK))

import bihash_pilot as bp
import engine as en


def write_json(path, value):
    bp.write_json(path, value)


def decode_coordinates(history, codec, coord):
    preds = []
    with torch.no_grad():
        for i in range(0, len(history), 8):
            x = bp.tensor(history[i : i + 8])
            z = codec.encode(x)
            c = bp.tensor(coord[i : i + 8])
            future = z[:, -1:] + c
            pred = x[:, -1:] + codec.decode(future) - codec.decode(z[:, -1:])
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds)


def load_case_state(case, data):
    c = data["x"].shape[1]
    k = max(1, c // 4)
    mean, basis, _ = en.ex.initialization(data, k, "PCA")
    codec = en.Codec(mean, basis, case["seed"], "ANCHOR_NONLINEAR_D").to("cuda")
    alignment = bp.BiHashFutureAlignment(
        data["_target"]["fit"],
        data["_observed"]["fit"],
        data["_history"]["fit"],
        basis,
        data["origins"]["fit"],
        case["seed"],
    ).to("cuda")
    checkpoint = case["selected_checkpoint"]
    saved = torch.load(ROOT / checkpoint["path"], map_location="cpu", weights_only=True)
    codec.load_state_dict(saved["codec"])
    alignment.load_state_dict(saved["alignment"])
    codec.eval().requires_grad_(False)
    alignment.eval().requires_grad_(False)
    return codec, alignment


def retrieve_for_split(data, history, target, observed, origins, codec, alignment, split, limit, mode, alpha_grid):
    with torch.no_grad():
        query_delta = en.predict(history[split], codec, limit["bolt"], limit["limit"], delta=True)
        fit_delta = en.predict(history["fit"], codec, limit["bolt"], limit["limit"], delta=True)
        query_delta_t = bp.tensor(query_delta)
        fit_delta_t = bp.tensor(fit_delta)
        valid_query = torch.ones(query_delta_t.shape[:2], dtype=torch.bool, device="cuda")
        query_summary = alignment.summary(query_delta_t, valid_query)
        fit_summary_from_past = alignment.summary(fit_delta_t, alignment.valid)
        fit_summary_target = alignment.summary_target
        fit_coord_target = alignment.target.detach().cpu().numpy()

        if mode == "bihash":
            query_code = alignment.deterministic_hash(alignment.past_hash(query_summary))
            bank_code = alignment.deterministic_hash(alignment.future_hash(fit_summary_target))
            logits = query_code @ bank_code.T / math.sqrt(alignment.bits)
        elif mode == "continuous":
            logits = -torch.cdist(query_summary, fit_summary_target).square()
        elif mode == "past_continuous":
            logits = -torch.cdist(query_summary, fit_summary_from_past).square()
        else:
            raise ValueError(f"UNKNOWN_RETRIEVAL_MODE:{mode}")

        bank_origins = data["origins"]["fit"]
        query_origins = origins[split]
        causal = torch.as_tensor(bank_origins[None, :] + bp.HORIZON <= query_origins[:, None], device="cuda")
        logits = torch.where(causal, logits, torch.full_like(logits, -1e9))
        weights = torch.softmax(logits, dim=1)
        retrieved = (weights.cpu().numpy() @ fit_coord_target.reshape(len(fit_coord_target), -1)).reshape(
            len(query_origins), bp.HORIZON, -1
        )
        q_coord = alignment.q(query_delta_t).cpu().numpy()

    candidates = {}
    for alpha in alpha_grid:
        coord = alpha * q_coord + (1.0 - alpha) * retrieved
        pred = decode_coordinates(history[split], codec, coord)
        candidates[str(alpha)] = {
            "alpha_q": alpha,
            "scores": en.metrics(data, pred, target[split], observed[split]),
        }
    nearest = torch.argmax(logits, dim=1).cpu().numpy()
    return {
        "mode": mode,
        "candidate_scores": candidates,
        "retrieval_entropy": float((-(weights * weights.clamp_min(1e-12).log()).sum(1)).mean().cpu()),
        "mean_top1_weight": float(weights.max(1).values.mean().cpu()),
        "causal_bank_min": int(causal.sum(1).min().cpu()),
        "causal_bank_max": int(causal.sum(1).max().cpu()),
        "nearest_fit_origin_distance_min": int(np.min(query_origins - bank_origins[nearest])),
        "nearest_fit_target_end_lag_min": int(np.min(query_origins - (bank_origins[nearest] + bp.HORIZON))),
    }


def run(tag, source_tag):
    output = HERE / f"run_{tag}"
    output.mkdir(parents=True, exist_ok=False)
    source = HERE / f"run_{source_tag}" / "RESULT.json"
    result = {
        "status": "RUNNING",
        "source_result": str(source.relative_to(ROOT)),
        "started_unix": time.time(),
        "cases": [],
        "scope": "post-hoc retrieval on development TUNE only; no protected TEST and no new training",
    }
    limit = bp.Limit(tag, gpu=True)
    try:
        en.configure()
        protected = en.ex.access_guard()
        bolt = en.ex.previous.FrozenBolt()
        en.replace_normalizer(bolt)
        limit_payload = {"limit": limit, "bolt": bolt}
        src = json.loads(source.read_text(encoding="utf-8-sig"))
        for dataset in src["datasets"]:
            data = en.ex.load_data(dataset, sizes=(256, 64, None))
            history, target, observed, _ = en.ex.arrays(data, bp.HISTORY)
            data["_history"], data["_target"], data["_observed"] = history, target, observed
            for case in src["cases"]:
                if case["dataset"] != dataset or case["method"] not in ("BIHASH", "BIHASH_SHUFFLED"):
                    continue
                codec, alignment = load_case_state(case, data)
                record = {
                    "dataset": dataset,
                    "seed": case["seed"],
                    "method": case["method"],
                    "selected_update": case["selected_update"],
                    "selected_q_update": case.get("selected_q_update"),
                    "source_tune_mse": case["selected_checkpoint"]["forecast"]["tune"]["normalized"]["mse"],
                    "source_q_tune_mse": None if case.get("selected_q_checkpoint") is None else case["selected_q_checkpoint"]["forecast_q"]["tune"]["normalized"]["mse"],
                    "retrieval": [],
                }
                for mode in ("bihash", "continuous", "past_continuous"):
                    record["retrieval"].append(
                        retrieve_for_split(
                            data,
                            history,
                            target,
                            observed,
                            data["origins"],
                            codec,
                            alignment,
                            "tune",
                            limit_payload,
                            mode,
                            alpha_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
                        )
                    )
                result["cases"].append(record)
                bp.write_json(output / "RESULT.partial.json", result, replace=True)
        assert en.ex.previous.state_hash(bolt.model) == bolt.initial_hash
        result["status"] = "COMPLETE"
        result["protected_test_guarded"] = protected
        result["resources"] = limit.record
        write_json(output / "RESULT.json", result)
        limit.close("COMPLETE")
        print(json.dumps({"status": "COMPLETE", "path": str((output / "RESULT.json").relative_to(ROOT))}), flush=True)
    except Exception as err:
        result["status"] = "FAILED_PRESERVED"
        result["error"] = repr(err)
        result["traceback"] = traceback.format_exc()
        bp.write_json(output / "RESULT.failed.json", result)
        limit.close("FAILED_PRESERVED", repr(err))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--source-tag", default="dev_v1")
    args = parser.parse_args()
    run(args.tag, args.source_tag)


if __name__ == "__main__":
    main()
