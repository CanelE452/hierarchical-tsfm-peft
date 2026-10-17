"""Preseal and decompose four explicitly bound, saved Jena development-V forecasts."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(args):
    worktree, cache = Path(args.worktree).resolve(), Path(args.cache).resolve()
    assets = []
    common = None
    for seed in (92601, 92602):
        family_dir = "postT_V_only_cpu_lag_20261016_v1" if seed == 92601 else "priority_core_followup_20261016_v1"
        for model in ("B0", "B2"):
            run = f"postT_cpu_jena_E3_{model}_k6_s{seed}_v1" if seed == 92601 else f"priority_seed2_jena_{model}_k6_s{seed}_v1"
            relative = f"research/informs_broad_validation_20261015_v1/{family_dir}/predictions/{run}/V_select/PREDICTION_MANIFEST.json"
            path = worktree / relative
            manifest = json.loads(path.read_text(encoding="utf-8"))
            archive = cache / family_dir / "predictions" / run / "V_select.npz"
            if manifest["split"] != "V_select" or manifest["arrays"]["target_normalized"]["shape"] != [43, 48, 21]:
                raise ValueError("Only the four declared Jena development-V forecasts are in scope")
            if sha(archive) != manifest["archive_sha256"]:
                raise ValueError("Saved V archive SHA mismatch")
            contract = {key: manifest[key] for key in ("data_sha", "scaler_sha", "cohort")}
            contract["arrays"] = {key: manifest["arrays"][key] for key in ("target_normalized", "observed", "origins", "channel_ids")}
            if common is None:
                common = contract
            elif common != contract:
                raise ValueError("A different target/mask/origin/channel/scaler cohort cannot be pooled")
            assets.append(dict(seed=seed, model=model, run_id=run, archive_path=archive.as_posix(),
                               archive_sha256=sha(archive), manifest_path=path.as_posix(),
                               manifest_relative_path=relative, manifest_sha256=sha(path)))
    selection = dict(horizon_bands=[[0, 16], [16, 32], [32, 48]],
                     origin_blocks="Stable ascending origin order, numpy.array_split(order,3): sizes15/14/14",
                     channels="All21, original stored channel order; no error-dependent selection",
                     metric="FP64 observed channel-macro MSE/MAE; sum channel SSE/count then equal channel mean",
                     contribution="Subset channel SSE divided by FULL channel count and FULL active-channel count; additive partitions recover full delta",
                     subset_macro="Subset channel SSE/subset count then mean over subset active channels; not an additive contribution",
                     concentration="Descending positive channel gains (-delta contribution); top1/top3/top5 shares of gross positive gain, descriptive only",
                     significance_tests=False, independent_validation=False,
                     normalization="Original A60 channel z-score; no physical-unit or source-scaler tensor access",
                     bounds=dict(cpu_threads=2, rss_bytes=3221225472, GPU=0))
    result = dict(schema="JENA_SAVED_V_ERROR_DECOMPOSITION_PRESEAL_v1", created_unix=time.time(),
                  numeric_arrays_read=False, selection=selection, common_contract=common, assets=assets,
                  code_sha256=sha(__file__))
    put(args.inventory, result)
    print(json.dumps(dict(status="SEALED_BEFORE_ARRAY_NUMERIC_READ", inventory_sha256=sha(args.inventory), assets=4)))


def execute(args):
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "2"
    if sha(args.inventory) != args.inventory_sha:
        raise ValueError("Presealed inventory SHA differs")
    inventory = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
    if inventory["numeric_arrays_read"] or inventory["code_sha256"] != sha(__file__):
        raise ValueError("Original numeric-read boundary and analyzed source must be pinned")
    output, private = Path(args.output).resolve(), Path(args.private).resolve()
    names = ["ERROR_DECOMPOSITION_CHANNEL.csv", "ERROR_DECOMPOSITION_HORIZON.csv", "ERROR_DECOMPOSITION_BLOCK.csv",
             "ERROR_DECOMPOSITION_SUMMARY.json", "ERROR_DECOMPOSITION_KO.md", "figures/A_CHANNEL_EFFECT.svg",
             "figures/A_HORIZON_BLOCK_EFFECT.svg"]
    if any((output / name).exists() for name in names) or (private / "A_EXECUTION_RECEIPT.json").exists():
        raise FileExistsError("CreateNew outputs only; original evidence must be retained")
    import numpy as np
    arrays, bindings = {}, []
    reference = None
    score_audits = []
    for asset in inventory["assets"]:
        if sha(asset["archive_path"]) != asset["archive_sha256"] or sha(asset["manifest_path"]) != asset["manifest_sha256"]:
            raise ValueError("Bound original evidence changed")
        manifest = json.loads(Path(asset["manifest_path"]).read_text(encoding="utf-8"))
        with np.load(asset["archive_path"], allow_pickle=False) as data:
            if set(data.files) != {"pred_normalized", "target_normalized", "observed", "origins", "channel_ids"}:
                raise ValueError("Unexpected archive member")
            values = {key: data[key] for key in data.files}
        for key, value in values.items():
            fingerprint = dict(shape=list(value.shape), dtype=str(value.dtype), sha256=hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest())
            if fingerprint != manifest["arrays"][key]:
                raise ValueError("Exact saved array SHA mismatch: " + key)
        if reference is None:
            reference = values
        else:
            for key in ("target_normalized", "observed", "origins", "channel_ids"):
                if not np.array_equal(reference[key], values[key], equal_nan=values[key].dtype.kind in "fc"):
                    raise ValueError("Exact cross-run evaluation arrays differ")
        mask = values["observed"]
        if mask.dtype != np.bool_ or not np.isfinite(values["pred_normalized"][mask]).all() or not np.isfinite(values["target_normalized"][mask]).all():
            raise ValueError("Original observation mask and finite observed values required")
        difference = values["pred_normalized"].astype(np.float64) - values["target_normalized"].astype(np.float64)
        difference = np.where(mask, difference, 0.0)
        counts = mask.sum(axis=(0, 1), dtype=np.int64)
        active = counts > 0
        sse = (difference * difference).sum(axis=(0, 1), dtype=np.float64)
        sae = np.abs(difference).sum(axis=(0, 1), dtype=np.float64)
        mse = math.fsum(float(sse[c] / counts[c]) for c in np.flatnonzero(active)) / int(active.sum())
        mae = math.fsum(float(sae[c] / counts[c]) for c in np.flatnonzero(active)) / int(active.sum())
        max_delta = max(abs(mse - manifest["score"]["mse"]), abs(mae - manifest["score"]["mae"]))
        if max_delta > 1e-12 or int(counts.sum()) != manifest["score"]["n_observed"]:
            raise ValueError("FP64 full-score reconciliation failed")
        arrays[asset["seed"], asset["model"]] = dict(difference=difference, counts=counts, sse=sse, sae=sae, mse=mse, mae=mae)
        score_audits.append(dict(seed=asset["seed"], model=asset["model"], run_id=asset["run_id"], mse=mse, mae=mae,
                                 n_observed=int(counts.sum()), maximum_original_score_delta=max_delta, tolerance=1e-12))
        bindings.append({key: asset[key] for key in ("seed", "model", "run_id", "archive_sha256", "manifest_relative_path", "manifest_sha256")})
    origins, channels, mask = reference["origins"], reference["channel_ids"], reference["observed"]
    if len(set(map(int, origins))) != len(origins):
        raise ValueError("Unique original evaluation origins required")
    order = np.argsort(origins, kind="stable")
    blocks = np.array_split(order, 3)
    count_full = mask.sum(axis=(0, 1), dtype=np.int64)
    active_full = count_full > 0
    number_active = int(active_full.sum())
    channel_rows, horizon_rows, block_rows, summaries = [], [], [], []
    additive_checks = []

    def partition_row(seed, kind, label, index, b0, b2, submask):
        count = submask.sum(axis=(0, 1), dtype=np.int64)
        subactive = count > 0
        measures = {}
        for name, result in (("B0", b0), ("B2", b2)):
            diff = result["difference"]
            ss = np.where(submask, diff * diff, 0.0).sum(axis=(0, 1), dtype=np.float64)
            sa = np.where(submask, np.abs(diff), 0.0).sum(axis=(0, 1), dtype=np.float64)
            measures[name + "_subset_macro_mse"] = math.fsum(float(ss[c]/count[c]) for c in np.flatnonzero(subactive)) / int(subactive.sum()) if subactive.any() else None
            measures[name + "_subset_macro_mae"] = math.fsum(float(sa[c]/count[c]) for c in np.flatnonzero(subactive)) / int(subactive.sum()) if subactive.any() else None
            measures[name + "_full_metric_mse_contribution"] = math.fsum(float(ss[c]/count_full[c]) for c in np.flatnonzero(active_full)) / number_active
            measures[name + "_full_metric_mae_contribution"] = math.fsum(float(sa[c]/count_full[c]) for c in np.flatnonzero(active_full)) / number_active
        row = dict(dataset="jena", seed=seed, split="V_select_repeated_development", L=512, H=48, C=21, K=6,
                   partition=kind, partition_index=index, label=label, n_observed=int(count.sum()), active_channels=int(subactive.sum()), **measures)
        for metric in ("mse", "mae"):
            row["delta_subset_macro_"+metric] = measures["B2_subset_macro_"+metric] - measures["B0_subset_macro_"+metric]
            row["delta_full_metric_"+metric+"_contribution"] = measures["B2_full_metric_"+metric+"_contribution"] - measures["B0_full_metric_"+metric+"_contribution"]
        return row

    for seed in (92601, 92602):
        b0, b2 = arrays[seed, "B0"], arrays[seed, "B2"]
        seed_channel = []
        for ci, channel in enumerate(channels):
            submask = np.zeros_like(mask); submask[:, :, ci] = mask[:, :, ci]
            row = partition_row(seed, "channel", str(channel), ci, b0, b2, submask)
            row["channel_index"] = ci
            row["full_channel_observed_count"] = int(count_full[ci])
            channel_rows.append(row); seed_channel.append(row)
        seed_horizon, seed_block = [], []
        for hi in range(48):
            submask = np.zeros_like(mask); submask[:, hi, :] = mask[:, hi, :]
            row = partition_row(seed, "horizon_step", str(hi+1), hi, b0, b2, submask)
            horizon_rows.append(row); seed_horizon.append(row)
        for bi, (start, stop) in enumerate(inventory["selection"]["horizon_bands"]):
            submask = np.zeros_like(mask); submask[:, start:stop, :] = mask[:, start:stop, :]
            row = partition_row(seed, "horizon_band", f"H{start+1}-{stop}", bi, b0, b2, submask)
            horizon_rows.append(row)
        for bi, inds in enumerate(blocks):
            submask = np.zeros_like(mask); submask[inds, :, :] = mask[inds, :, :]
            row = partition_row(seed, "origin_block", ("first", "middle", "last")[bi], bi, b0, b2, submask)
            row.update(n_origins=len(inds), first_origin=int(origins[inds[0]]), last_origin=int(origins[inds[-1]]))
            block_rows.append(row); seed_block.append(row)
        for kind, rows in (("channel",seed_channel), ("horizon_step",seed_horizon), ("origin_block",seed_block)):
            for metric in ("mse", "mae"):
                reconstructed = math.fsum(row["delta_full_metric_"+metric+"_contribution"] for row in rows)
                expected = b2[metric] - b0[metric]
                if abs(reconstructed - expected) > 1e-12:
                    raise ValueError("Original-denominator additive reconstruction failed")
                additive_checks.append(dict(seed=seed, partition=kind, metric=metric, reconstructed_delta=reconstructed,
                                            full_delta=expected, residual=abs(reconstructed-expected)))
        gains = sorted((-r["delta_full_metric_mse_contribution"] for r in seed_channel if r["delta_full_metric_mse_contribution"] < 0), reverse=True)
        gross_gain = math.fsum(gains)
        gross_harm = math.fsum(r["delta_full_metric_mse_contribution"] for r in seed_channel if r["delta_full_metric_mse_contribution"] > 0)
        summaries.append(dict(seed=seed, B0_mse=b0["mse"], B2_mse=b2["mse"], delta_mse=b2["mse"]-b0["mse"],
                              B0_mae=b0["mae"], B2_mae=b2["mae"], delta_mae=b2["mae"]-b0["mae"],
                              mse_improved_channels=sum(r["delta_subset_macro_mse"]<0 for r in seed_channel),
                              mse_worse_channels=sum(r["delta_subset_macro_mse"]>0 for r in seed_channel),
                              mae_improved_channels=sum(r["delta_subset_macro_mae"]<0 for r in seed_channel),
                              mae_worse_channels=sum(r["delta_subset_macro_mae"]>0 for r in seed_channel),
                              gross_positive_mse_gain=gross_gain, gross_mse_harm=gross_harm,
                              top1_gross_gain_share=math.fsum(gains[:1])/gross_gain if gross_gain else None,
                              top3_gross_gain_share=math.fsum(gains[:3])/gross_gain if gross_gain else None,
                              top5_gross_gain_share=math.fsum(gains[:5])/gross_gain if gross_gain else None))
    direction = []
    for ci, channel in enumerate(channels):
        values = [r for r in channel_rows if r["channel_index"] == ci]
        direction.append(dict(channel_index=ci, channel_id=str(channel),
                              mse_improved_both_seeds=all(r["delta_subset_macro_mse"]<0 for r in values),
                              mse_worse_both_seeds=all(r["delta_subset_macro_mse"]>0 for r in values),
                              mae_improved_both_seeds=all(r["delta_subset_macro_mae"]<0 for r in values),
                              mae_worse_both_seeds=all(r["delta_subset_macro_mae"]>0 for r in values)))
    summary = dict(schema="SAVED_JENA_V_ERROR_DECOMPOSITION_v1", status="PASS_EXISTING_PREDICTION_DIAGNOSTIC_NOT_NEW_GENERALIZATION",
                   inventory_sha256=args.inventory_sha, source_sha256=sha(__file__), selection=inventory["selection"],
                   common_contract=inventory["common_contract"], source_bindings=bindings,
                   original_score_FP64_checks=score_audits, additive_reconstruction_checks=additive_checks,
                   summaries=summaries, channel_seed_direction=direction,
                   rows=dict(channel=len(channel_rows), horizon=len(horizon_rows), block=len(block_rows)),
                   caveats=["Repeated development V; blocks/channels/overlapping time points are not independent replicates.",
                            "Historical selected B0 refit3 vs B2 refit1 and different capacity; this is not a matched-training mechanism control.",
                            "Both models see full512 past through Bolt/lag D; C-current anchor bypass retained.",
                            "A60 z-score outputs; upstream interpolation/legacy ancestry UNKNOWN; no original-unit scaler tensor read.",
                            "Two seeds do not establish significance, independent generalization or identified common factors."],
                   new_training=0, new_inference=0, cost_runs=0, protected_T_numeric_access=0)
    write_csv(output / names[0], channel_rows); write_csv(output / names[1], horizon_rows); write_csv(output / names[2], block_rows)
    put(output / names[3], summary)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans", "svg.fonttype":"none", "font.size":10})
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    x = np.arange(21)
    for si, seed in enumerate((92601,92602)):
        rows = [r for r in channel_rows if r["seed"]==seed]
        for ax, metric in zip(axes,("mse","mae")):
            ax.bar(x+(si-.5)*.38, [r["delta_subset_macro_"+metric] for r in rows], width=.38, label=f"seed{seed}", color=("#0072B2","#D55E00")[si])
            ax.axhline(0, color="black", linewidth=.7)
            ax.set_ylabel(f"B2 - B0 channel {metric.upper()}")
            ax.grid(axis="y",alpha=.2)
    axes[0].legend(ncol=2)
    axes[0].set_title("Jena: effects across all channels | selected historical refits, repeated development V")
    axes[1].set_xticks(x, [str(v) for v in channels], rotation=65, ha="right", fontsize=8)
    axes[1].set_xlabel("Original stored channel order; negative = B2 lower error | A60 normalized units")
    fig.savefig(output/names[5], metadata={"Date":None}); plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(11,7), constrained_layout=True)
    for seed,color in zip((92601,92602),("#0072B2","#D55E00")):
        hrows = [r for r in horizon_rows if r["seed"]==seed and r["partition"]=="horizon_step"]
        brows = [r for r in block_rows if r["seed"]==seed]
        for ri,metric in enumerate(("mse","mae")):
            axes[ri,0].plot(np.arange(1,49),[r["delta_subset_macro_"+metric] for r in hrows],label=f"seed{seed}",color=color)
            axes[ri,1].plot(np.arange(3),[r["delta_subset_macro_"+metric] for r in brows],marker="o",label=f"seed{seed}",color=color)
            axes[ri,0].set_ylabel(f"B2 - B0 subset macro {metric.upper()}")
    for ax in axes.flat:
        ax.axhline(0,color="black",linewidth=.7); ax.grid(alpha=.2)
    for ax in axes[:,0]:
        ax.axvline(16.5,color="gray",ls=":"); ax.axvline(32.5,color="gray",ls=":"); ax.set_xlabel("Forecast horizon step (fixed thirds16/16/16)")
    for ax in axes[:,1]:
        ax.set_xticks([0,1,2],["First15","Middle14","Last14"]); ax.set_xlabel("Chronological origins; same development V, not new test")
    axes[0,0].legend(); axes[0,0].set_title("Prediction distance")
    axes[0,1].set_title("Fixed contiguous origin blocks")
    fig.suptitle("Jena historical B0/B2: where the observed gain persists or reverses",fontsize=13)
    fig.savefig(output/names[6],metadata={"Date":None}); plt.close(fig)
    lines=["# 기존 Jena B0/B2 오류 분해", "", "[확인] 새 학습·추론 없이 동일한 개발 V의 저장 예측4개를 FP64로 재채점했다. 채널·거리·블록은 오차를 읽기 전에 고정했다.", "", "```text", "seed   B0 MSE       B2 MSE       delta MSE     MSE improved/worse   MAE improved/worse"]
    for row in summaries:
        lines.append(f"{row['seed']}  {row['B0_mse']:.9f}  {row['B2_mse']:.9f}  {row['delta_mse']:+.9f}  {row['mse_improved_channels']}/21, {row['mse_worse_channels']}/21          {row['mae_improved_channels']}/21, {row['mae_worse_channels']}/21")
    lines += ["```", "", "[확인] 첫/중앙/마지막 origin 블록은 정렬된43개를15/14/14로 나눴고, 예측거리는1–16/17–32/33–48이다. 부분집합 macro 점수와 전체 점수 기여를 CSV에서 분리했다. 전체 관측 수를 분모로 한 기여의 합은 원 전체 B2−B0 차이를1e-12 이내로 복원한다.", ""]
    for row in summaries:
        lines.append(f"[확인] seed{row['seed']}에서 개선 채널의 총 양의 MSE 이득 중 상위1/3/5개 채널 비중은 {row['top1_gross_gain_share']:.1%}/{row['top3_gross_gain_share']:.1%}/{row['top5_gross_gain_share']:.1%}다. 이 비중의 분모는 악화가 상쇄되기 전의 총 양의 이득이며, 순 전체 이득의 비중과 다르다.")
    lines += ["", "[추정] 이 분해는 개선이 나타난 위치와 반전을 보여준다. B0 refit3epoch/B2 refit1epoch 및 용량 차이는 그대로 남아 시간 혼합의 인과 효과를 식별하지 못한다.", "", "[미완료] 두 seed와 같은 V의 세 블록은 독립 일반화나 유의성을 입증하지 않는다. 실제 공통 요인·잡음 정답과 상류 보간 계보의 확인도 남아 있다.", "", "- [원수치·검산](ERROR_DECOMPOSITION_SUMMARY.json)", "- [채널별 수치](ERROR_DECOMPOSITION_CHANNEL.csv)", "- [거리별 수치](ERROR_DECOMPOSITION_HORIZON.csv)", "- [시간 블록별 수치](ERROR_DECOMPOSITION_BLOCK.csv)", "", "![모든 채널의 오차 차이](figures/A_CHANNEL_EFFECT.svg)", "", "![예측 거리와 개발 V 블록의 차이](figures/A_HORIZON_BLOCK_EFFECT.svg)", ""]
    with (output/names[4]).open("x",encoding="utf-8",newline="\n") as stream:
        stream.write("\n".join(lines))
    import psutil
    rss = psutil.Process().memory_info().rss
    if rss > inventory["selection"]["bounds"]["rss_bytes"]:
        raise MemoryError("Analysis memory bound exceeded")
    for asset in inventory["assets"]:
        if sha(asset["archive_path"]) != asset["archive_sha256"] or sha(asset["manifest_path"]) != asset["manifest_sha256"]:
            raise ValueError("Original saved evidence changed during read")
    receipt = dict(status=summary["status"], inventory_sha256=args.inventory_sha, source_sha256=sha(__file__),
                   created_unix=time.time(), output_sha256={name:sha(output/name) for name in names},
                   current_rss_bytes=rss, numpy_matplotlib_only=True, torch_imported="torch" in __import__("sys").modules,
                   original_prepost_SHA_pass=True, new_models=0, new_inference=0, new_cost=0, protected_T_access=0)
    put(private/"A_EXECUTION_RECEIPT.json", receipt)
    print(json.dumps(dict(status=summary["status"], summaries=summaries, rows=summary["rows"], rss_bytes=rss)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    group=parser.add_mutually_exclusive_group()
    group.add_argument("--prepare",action="store_true")
    group.add_argument("--execute",action="store_true")
    for key in ("worktree","cache","inventory","inventory-sha","output","private"):
        parser.add_argument("--"+key)
    args=parser.parse_args()
    if not args.prepare and not args.execute:
        print(json.dumps(dict(status="SOURCE_ONLY_NO_IO", reads=0, writes=0, model_calls=0))); return
    required=("worktree","cache","inventory") if args.prepare else ("inventory","inventory_sha","output","private")
    if any(getattr(args,key) is None for key in required):
        parser.error("Explicit pinned paths required")
    prepare(args) if args.prepare else execute(args)


if __name__=="__main__":
    main()
