"""Read-only science closeout of the finite B2 mechanism campaign.

Default invocation reads no assets and writes nothing. Explicit execution reads
only registered public V results and exact new private V archives. It does not
load a checkpoint, model, raw time series, protected T, or the foundation model.
"""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


NAME = "informs_b2_why_where_20261017"
SEEDS = (92601, 92602)
FAMILIES = ("B0", "B2", "B2_CAP_POINTWISE")
ENDPOINTS = ((0., 0, 0), (.25, 156, 2481), (1., 622, 9923), (3., 1863, 29769))
FIELDS = ("comparison", "run_id", "dataset", "C", "K", "L", "H", "seed", "model",
          "fit", "selection", "eval", "metric_scale", "params", "epoch", "updates",
          "exposure", "MSE", "MAE", "observed", "device", "status", "same_update",
          "source_file", "source_sha256", "checkpoint_sha256", "archive_sha256", "note")


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_csv(path, rows, fields):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows({key: row.get(key, "NA") if row.get(key) is not None else "NA"
                         for key in fields} for row in rows)


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def error_summary(error):
    if isinstance(error, OSError):
        return f"{type(error).__name__}: {error.strerror or 'file access failed'}; {Path(error.filename).name if error.filename else 'registered input'}"
    return str(error)


def array_sha(value):
    import numpy as np
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def fp64_metrics(prediction, target, observed):
    """Independent per-channel FP64 totals, then equal weight active channels."""
    import numpy as np
    p, y, m = np.asarray(prediction), np.asarray(target), np.asarray(observed)
    if p.ndim != 3 or p.shape != y.shape or p.shape != m.shape or m.dtype != np.bool_:
        raise ValueError("Expected aligned [origin,horizon,channel] and boolean mask")
    if not np.isfinite(p[m]).all() or not np.isfinite(y[m]).all():
        raise ValueError("Nonfinite observed predictions/targets")
    counts, sse, sae = [], [], []
    for channel in range(p.shape[2]):
        keep = m[:, :, channel]
        error = p[:, :, channel][keep].astype(np.float64)-y[:, :, channel][keep].astype(np.float64)
        counts.append(int(keep.sum()))
        # math.fsum is independent of the producer's vectorized sum order.
        sse.append(math.fsum(float(value)*float(value) for value in error))
        sae.append(math.fsum(abs(float(value)) for value in error))
    active = [index for index, count in enumerate(counts) if count]
    if len(active) != p.shape[2]:
        raise ValueError("Registered evaluations require observed targets in every channel")
    return dict(MSE=math.fsum(sse[i]/counts[i] for i in active)/len(active),
                MAE=math.fsum(sae[i]/counts[i] for i in active)/len(active),
                observed=sum(counts), channel_sse=sse, channel_sae=sae,
                channel_counts=counts)


def audit_archive(path, result, expected_shape, cohort, contract, audits):
    """Only the caller's literal registered path is opened; SHA is pinned twice."""
    import numpy as np
    before = sha(path)
    bound = result.get("private_prediction_archive_sha256")
    if bound and before != bound:
        raise ValueError("Producer-bound private V archive SHA differs")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) == {"prediction", "y", "mask", "origins"}:
            p, y, m, origins = (archive[key] for key in ("prediction", "y", "mask", "origins"))
            channels = None
        elif set(archive.files) == {"pred_normalized", "target_normalized", "observed", "origins", "channel_ids"}:
            p, y, m, origins, channels = (archive[key] for key in
                ("pred_normalized", "target_normalized", "observed", "origins", "channel_ids"))
        else:
            raise ValueError("Unregistered private V archive schema")
    if p.shape != expected_shape or origins.ndim != 1 or len(origins) != expected_shape[0]:
        raise ValueError("Registered V dimensions differ")
    expected_origins = np.arange(14675, 16692, 48) if cohort == "jena" else np.arange(12499, 14264, 12)
    if not np.array_equal(origins, expected_origins):
        raise ValueError("Registered allowed V origins differ")
    fingerprints = dict(target=array_sha(y), mask=array_sha(m), origins=array_sha(origins))
    if channels is not None:
        fingerprints["channels"] = array_sha(channels)
    # Probe target promotes the identical FP32 original to FP64; compare values,
    # not dtype-dependent fingerprints, across producer schemas.
    canonical = dict(target=array_sha(y.astype(np.float64)), mask=array_sha(m), origins=array_sha(origins))
    original = contract.get("expected_"+cohort)
    if original:
        plain = dict(target=hashlib.sha256(y.astype(np.float32).tobytes()).hexdigest(),
                     mask=hashlib.sha256(m.tobytes()).hexdigest(), origins=hashlib.sha256(origins.tobytes()).hexdigest())
        if any(plain[key] != original[key] for key in plain):
            raise ValueError("V values differ from original published target/mask/origin contract")
        if channels is not None and hashlib.sha256(channels.tobytes()).hexdigest() != original["channels"]:
            raise ValueError("V channel order differs from original published contract")
        scaler = result.get("scaler_sha256", result.get("metadata", {}).get("scaler_sha256"))
        if scaler != original["scaler"]:
            raise ValueError("V metric normalization differs from original canonical A60 scaler")
    if cohort in contract and contract[cohort] != canonical:
        raise ValueError("Within-cohort target/mask/origin values differ")
    contract.setdefault(cohort, canonical)
    metrics = fp64_metrics(p, y, m)
    residual = max(abs(metrics[key]-float(result[key])) for key in ("MSE", "MAE"))
    if residual > 1e-12:
        raise ValueError("Independent FP64 metric residual exceeds 1e-12")
    declared_counts = result.get("channel_counts", result.get("channel_observed"))
    if declared_counts is None or list(declared_counts) != metrics["channel_counts"]:
        raise ValueError("Original per-channel observed counts differ")
    if int(result.get("observed", sum(declared_counts))) != metrics["observed"]:
        raise ValueError("Original total observed count differs")
    after = sha(path)
    if before != after:
        raise ValueError("Private V archive changed during independent rescore")
    audits.append(dict(archive_id=path.parent.name+"/"+path.name, sha256=before,
        producer_binding_present=bool(bound), shape=list(p.shape), fingerprints=fingerprints,
        independent_MSE=metrics["MSE"], independent_MAE=metrics["MAE"],
        channel_counts=metrics["channel_counts"], observed=metrics["observed"],
        max_metric_residual=residual, tolerance=1e-12, sha_unchanged=True))
    return before


def quiet_and_limits(deadline):
    import psutil
    if time.time() >= deadline:
        raise TimeoutError("Closeout's caller-supplied finite deadline reached")
    names = {"mechanism_runner.py", "pems_cost.py", "CAP_run_initialization_audit.py", "PEMS_run_restore_audit.py"}
    active = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "create_time"]):
        if process.info["pid"] == os.getpid() or "python" not in (process.info["name"] or "").lower():
            continue
        if any(Path(arg).name in names for arg in process.info["cmdline"] or []):
            active.append({key: process.info[key] for key in ("pid", "create_time")})
    if active:
        raise RuntimeError("A same-campaign model or cost worker is still active")
    if psutil.Process().memory_info().rss > 3*1024**3:
        raise MemoryError("Closeout exceeds RSS3GiB")
    return dict(checked_unix=time.time(), recognized_workers=active,
                RSS_bytes=psutil.Process().memory_info().rss, CPU_threads=2)


def base_row(model, seed, source, repo, **values):
    row = dict(model=model, seed=seed, run_id=source.parent.name, dataset="jena", C=21,
               K=6, L=512, H=48, device="cpu", source_file=source.relative_to(repo).as_posix(),
               source_sha256=sha(source) if source.is_file() else "NA", status="PENDING",
               MSE="NA", MAE="NA", metric_scale="A60_channel_zscore", same_update=False)
    row.update(values)
    return row


def collect(repo, private, output, deadline):
    rows, missing, rejected, archives, contracts = [], [], [], [], {}
    inputs = {}
    def bind(path):
        inputs[path.relative_to(repo).as_posix()] = sha(path)
    baseline_path = repo/"reports/informs_ppt_ready_20261017/PPT_MAIN_TABLES.csv"
    with baseline_path.open(encoding="utf-8-sig", newline="") as stream:
        original_rows = list(csv.DictReader(stream))
    for dataset, table in (("jena", "Temporal_Context_Effect"), ("pems08_C32", "Matched_PeMS_C32")):
        candidate = next(row for row in original_rows if row["table_id"] == table and row["target_sha256"])
        contracts["expected_"+dataset] = {key:candidate[field] for key,field in
            (("target","target_sha256"),("mask","mask_sha256"),("origins","origins_sha256"),
             ("channels","channel_sha256"),("scaler","scaler_sha256"))}
    for seed in SEEDS:
        for family in FAMILIES:
            for index, (epoch, updates, exposure) in enumerate(ENDPOINTS):
                for split in ("A_fit", "A_stop", "V_select"):
                    quiet_and_limits(deadline)
                    source = output/"same_update"/f"s{seed}"/f"{family}_e{index}_{split}.json"
                    row = base_row(family, seed, source, repo, comparison="same_update" if split == "V_select" else "optimization_diagnostic",
                        run_id=f"same_update_s{seed}_{family}_e{index}_{split}",
                        fit="A_fit_search", selection="NOT_SELECTED_COMMON_ENDPOINT", eval=split,
                        metric_scale="canonical_A60" if split == "V_select" else "search_A50",
                        epoch=epoch, updates=updates, exposure=exposure, same_update=True)
                    if not source.is_file():
                        missing.append(row["source_file"])
                    else:
                        bind(source)
                        try:
                            result = read(source)
                            for key, expected in (("seed", seed), ("model", family), ("updates", updates),
                                                  ("exposure", exposure), ("eval", split), ("epoch", epoch)):
                                if result.get(key) != expected:
                                    raise ValueError("Registered common endpoint mismatch: "+key)
                            if result.get("params") != {"B0":1008,"B2":35717,"B2_CAP_POINTWISE":35708}[family]:
                                raise ValueError("Actual active parameter count differs from registered structure")
                            if result.get("metric_scale") != row["metric_scale"]:
                                raise ValueError("Common endpoint metric scale differs")
                            if split == "V_select":
                                path = private/"same_update"/f"s{seed}"/f"{family}_e{index}_V.npz"
                                row["archive_sha256"] = audit_archive(path, result, (43, 48, 21), "jena", contracts, archives)
                            row.update({key: result[key] for key in ("params", "MSE", "MAE", "observed", "status", "checkpoint_sha256")})
                        except (ValueError, KeyError, OSError) as error:
                            row.update(status="REJECTED_AUDIT", note=error_summary(error))
                            rejected.append(dict(source=row["source_file"], reason=error_summary(error)))
                    rows.append(row)
    for dataset, family, seed, directory, shape in [("jena", "B2_CAP_POINTWISE", s, f"cap_s{s}", (43,48,21)) for s in SEEDS] + [
            ("pems08_C32", f, 92601, f"pems_{f}", (148,12,32)) for f in ("B0", "B2")]:
        source = output/"runs"/directory/"SELECTED_V_RESULT.json"
        row = base_row(family, seed, source, repo, comparison="selected_models", dataset=dataset,
            C=shape[2], K=6 if dataset == "jena" else 8, H=shape[1], eval="V_select_repeated_development")
        if not source.is_file():
            missing.append(row["source_file"])
        else:
            bind(source)
            try:
                result = read(source)
                if result.get("status") != "COMPLETE_SELECTED_V" or result.get("model") != family or result.get("seed") != seed:
                    raise ValueError("Complete original selected model result required")
                row["archive_sha256"] = audit_archive(private/"runs"/directory/"selected_V.npz", result, shape,
                    "jena" if dataset == "jena" else "pems08_C32", contracts, archives)
                row.update({key: result[key] for key in ("params", "MSE", "MAE", "observed", "fit", "selection", "status")})
                row.update(epoch=result.get("selected_epoch"), updates=result.get("updates", "NA"),
                    exposure=result.get("exposure", "NA"), checkpoint_sha256=result.get("artifact_sha256", "NA"))
                summary = output/"runs"/directory/"REFIT_SUMMARY.json"
                if summary.is_file():
                    bind(summary)
                    details = read(summary)
                    row.update(updates=details.get("updates", row["updates"]), exposure=details.get("exposures", row["exposure"]))
            except (ValueError, KeyError, OSError) as error:
                row.update(status="REJECTED_AUDIT", MSE="NA", MAE="NA", note=error_summary(error))
                rejected.append(dict(source=row["source_file"], reason=error_summary(error)))
        rows.append(row)
    for seed in SEEDS:
        for family in FAMILIES:
            directory = f"{family}_s{seed}"
            source = output/"probes"/directory/"RESULT.json"
            row = base_row(family, seed, source, repo, comparison="fixed_reader_probe", fit="A60",
                selection="fixed_lambda_1_no_V_selection", eval="V_select_repeated_development",
                note="Fixed full512x6 reader; original D/F unused; not common-factor identification")
            if not source.is_file():
                missing.append(row["source_file"])
            else:
                bind(source)
                try:
                    result = read(source)
                    if (result.get("status") != "COMPLETE" or not result.get("PE_frozen")
                            or result.get("foundation_called") is not False or result.get("original_D_called") is not False
                            or result.get("input_features") != 3072 or result.get("output_features") != 1008
                            or result.get("solver", {}).get("lambda_value") != 1.0
                            or result.get("solver", {}).get("V_used_for_fit_or_lambda") is not False):
                        raise ValueError("Registered fixed-reader protocol differs")
                    if result["metadata"]["seed"] != seed or result["metadata"]["model"] != family:
                        raise ValueError("Probe representation identity differs")
                    row["archive_sha256"] = audit_archive(private/"probes"/directory/"V_PREDICTIONS_FP64.npz",
                        result, (43,48,21), "jena", contracts, archives)
                    row.update(MSE=result["MSE"], MAE=result["MAE"], status="COMPLETE_FIXED_READER_PROBE",
                        params=result["parameter_count_with_intercept"], exposure=result["fit_exposure"],
                        checkpoint_sha256=result["metadata"]["checkpoint_sha256"], epoch=result["metadata"].get("selected_epoch"),
                        observed=sum(result["channel_observed"]))
                except (ValueError, KeyError, OSError) as error:
                    row.update(status="REJECTED_AUDIT", note=error_summary(error))
                    rejected.append(dict(source=row["source_file"], reason=error_summary(error)))
            rows.append(row)
    legacy_source = repo/"reports/informs_ppt_ready_20261017/PPT_MAIN_TABLES.csv"
    bindings = repo/"reports/informs_ppt_ready_20261017/SOURCE_BINDINGS.json"
    bind(legacy_source); bind(bindings)
    legacy_bindings = {row["row_id"]: row for row in read(bindings)["source_rows"]}
    for original in original_rows:
        table = original["table_id"]
        accepted = (table == "Temporal_Context_Effect" and original["dataset"] == "jena"
            or table == "Overall_Jena" and original["method"] not in ("B0", "B2")
            or table == "Frozen_TSFM_Contribution" and original["dataset"] == "jena" and original["method"] != "B2"
            or table == "Matched_PeMS_C32" and original["method"] in ("Raw", "G"))
        if not accepted:
            continue
        binding = legacy_bindings[original["row_id"]]
        source = repo/original["source_file"]
        if sha(source) != original["source_sha256"] or binding.get("validation") is not True:
            raise ValueError("Published legacy source or V validation binding differs")
        bind(source)
        # The original CSV is hashed opaquely, not parsed: legacy sources may
        # also transport locked history. Only the published V-only package is read.
        for key in ("run_id", "target_sha256", "mask_sha256", "origins_sha256", "scaler_sha256"):
            if binding.get(key) != original[key]:
                raise ValueError("Published V-only row binding differs: "+key)
        if "V_select" not in binding.get("cohort", ""):
            raise ValueError("Only source-bound original V rows may be reused")
        selected = table == "Temporal_Context_Effect"
        row = base_row(original["method"], int(original["seed"]), source, repo,
            comparison="selected_models" if selected else "strong_baseline_context",
            run_id=original["run_id"], dataset="pems08_C32" if table == "Matched_PeMS_C32" else "jena",
            C=int(original["C"]), K=original["K"], L=int(original["L"]), H=int(original["H"]),
            params=original["trainable_parameters"], epoch=original["selected_epoch"], updates=original["refit_updates"],
            fit="fresh_A60_selected_refit" if selected else "published_recipe_preserved",
            selection="original_A_stop" if selected else "published_recipe_preserved", eval="V_select_repeated_development",
            MSE=float(original["MSE"]), MAE=float(original["MAE"]), status="EXISTING_RESULT_REUSED",
            checkpoint_sha256=original["checkpoint_sha256"], device=original["accuracy_device"],
            note=original["comparison_note"])
        if selected:
            namespace="postT_V_only_cpu_lag_20261016_v1" if row["seed"]==92601 else "priority_core_followup_20261016_v1"
            receipt=repo/"research/informs_broad_validation_20261015_v1"/namespace/"runs"/row["run_id"]/"refit/RECEIPT.json"
            details=read(receipt);bind(receipt)
            if (details["updates"]!=int(row["updates"]) or details["selected_epoch"]!=float(row["epoch"])
                    or details["trainable_parameters"]!=int(row["params"])
                    or details["contract"]["scaler_sha"]!=contracts["expected_jena"]["scaler"]):
                raise ValueError("Original selected refit receipt differs from published metadata")
            row["exposure"]=details["exposures"]
        rows.append(row)
    return rows, missing, rejected, archives, contracts, inputs


def collect_costs(repo, output, inputs):
    costs, checks = [], []
    legacy_path=repo/"reports/informs_ppt_ready_20261017/PPT_MAIN_TABLES.csv"
    with legacy_path.open(encoding="utf-8-sig", newline="") as stream:
        legacy=[row for row in csv.DictReader(stream) if row["table_id"]=="Matched_PeMS_C32"
                and row["method"] in ("Raw","G")]
    for original in legacy:
        for batch in (1,8):
            prefix=f"B{batch}_"
            source=repo/original[prefix+"source_file"]
            if sha(source)!=original[prefix+"source_sha256"]:
                raise ValueError("Original same-C32 Raw/G cost source changed")
            inputs[source.relative_to(repo).as_posix()]=sha(source)
            costs.append(dict(dataset="pems08_C32",model=original["method"],batch=batch,
                status="EXISTING_COMPLETE_PROTOCOL_REUSED",median_seconds=float(original[prefix+"seconds"]),
                p90_seconds="NA",CPU_threads=2,source_file=source.relative_to(repo).as_posix(),
                source_sha256=sha(source)))
    for family in ("B0", "B2"):
        for batch in (1, 8):
            source = output/"costs"/f"pems_{family}_b{batch}"/"RESULT.json"
            row = dict(dataset="pems08_C32", model=family, batch=batch, status="PENDING",
                median_seconds="NA", p90_seconds="NA", CPU_threads=2,
                source_file=source.relative_to(repo).as_posix(), source_sha256="NA")
            if source.is_file():
                inputs[row["source_file"]] = row["source_sha256"] = sha(source)
                try:
                    result = read(source)
                    row["status"] = result["status"]
                    if result["status"] == "COMPLETE_PROTOCOL":
                        trials = source.parent/"COST_TRIALS.csv.gz"
                        if sha(trials) != result["trials_sha256"]:
                            raise ValueError("Cost trial hash differs")
                        with gzip.open(trials, "rt", encoding="utf-8", newline="") as stream:
                            records = list(csv.DictReader(stream))
                        timed = [item for item in records if item.get("phase") == "timed"]
                        phases=Counter(item.get("phase") for item in records)
                        if (phases!={"warmup":10,"timed":20,"stage_probe":20}
                                or result.get("CPU_threads") != 2 or result.get("family")!=family
                                or result.get("batch_size")!=batch or result.get("row_microbatch")!=128):
                            raise ValueError("Original CPU2 cost phases/family/batch/microbatch differ")
                        expected_geometry=(batch*8,batch*8,math.ceil(batch*8/128))
                        for item in records:
                            if (int(item["batch_size"])!=batch or int(item["row_microbatch"])!=128
                                    or tuple(int(item[key]) for key in ("F_logical_streams","F_physical_streams","F_calls"))!=expected_geometry
                                    or str(item["stage_instrumentation"]).lower()!=("true" if item["phase"]=="stage_probe" else "false")):
                                raise ValueError("Actual per-trial F geometry/stage instrumentation differs")
                        values = sorted(float(item["whole_seconds"]) for item in timed)
                        median = (values[9]+values[10])/2
                        p90 = .9*values[17]+.1*values[18]
                        if (not all(math.isfinite(value) and value > 0 for value in values)
                                or abs(median-result["median_seconds"]) > 1e-12
                                or abs(p90-result["p90_seconds"]) > 1e-12):
                            raise ValueError("Independent whole-pipeline median/p90 differs")
                        if result.get("forward_parity", {}).get("status") != "PASS" or result.get("GPU_initialized") is not False:
                            raise ValueError("Cost requires passed physical parity and CPU-only execution")
                        for name,digest in result["source_sha256"].items():
                            path=output/"code"/name if name in ("pems_cost.py","pems_resume.py") else repo/"research/informs_broad_validation_20261015_v1"/name
                            if sha(path)!=digest:raise ValueError("Cost producer/frozen-assert source binding changed")
                            inputs[path.relative_to(repo).as_posix()]=digest
                        selected=output/"runs"/f"pems_{family}"/"SELECTED_V_RESULT.json"
                        if sha(selected)!=result["selected_metrics_sha256"]:
                            raise ValueError("Cost does not bind current selected accuracy result")
                        for key in ("physical_input_sha256","observation_mask_sha256"):
                            if result[key]!=legacy[0][f"B{batch}_"+key]:
                                raise ValueError("New/Raw/G cost input identity differs")
                        row.update(median_seconds=result["median_seconds"], p90_seconds=result["p90_seconds"])
                        checks.append(dict(model=family, batch=batch, trials=50, timed_trials=20,
                            phases=dict(phases),per_trial_F_geometry=list(expected_geometry),
                            foundation_immutability_assurance="Producer frozen-weight assertion source-bound; not rerun by read-only closeout",
                            independent_median=median, independent_p90=p90,
                            source_sha256=row["source_sha256"], trials_sha256=sha(trials)))
                        inputs[trials.relative_to(repo).as_posix()] = sha(trials)
                except (ValueError, KeyError, OSError) as error:
                    row.update(status="REJECTED_COST_AUDIT",median_seconds="NA",p90_seconds="NA")
                    checks.append(dict(model=family,batch=batch,status="REJECTED_COST_AUDIT",reason=error_summary(error)))
            costs.append(row)
    return costs, checks


def claims(rows, decomposition=None):
    result = []
    def add(layer, question, claim, evidence, limitation, next_decision):
        result.append(dict(layer=layer, question=question, claim=claim, evidence=evidence,
            limitation=limitation, next_decision=next_decision))
    for summary in (decomposition or {}).get("summaries",[]):
        add("확인", f"기존 예측 오차 분해/seed{summary['seed']}",
            f"MSE 개선 채널 {summary['mse_improved_channels']}/21, 악화 {summary['mse_worse_channels']}/21; "
            f"상위5 채널은 양의 총 개선 {100*summary['top5_gross_gain_share']:.3f}%를 차지. "
            f"MAE 개선/악화 채널 {summary['mae_improved_channels']}/{summary['mae_worse_channels']}",
            "ERROR_DECOMPOSITION_SUMMARY.json; ERROR_DECOMPOSITION_CHANNEL.csv",
            "양의 총 개선 share는 순 개선 점유율과 다름. 같은 개발 V의 기술통계이며 seed·채널은 독립 표본 아님.",
            "평균 이득과 채널별 손해를 동시에 발표")
    for seed in SEEDS:
        for comparison in ("selected_models", "fixed_reader_probe"):
            pair = {row["model"]: row for row in rows if row["comparison"] == comparison
                    and row["dataset"] == "jena" and row["seed"] == seed and number(row.get("MSE")) is not None}
            if "B0" in pair and "B2" in pair:
                left, right = pair["B0"], pair["B2"]
                delta = float(right["MSE"])-float(left["MSE"])
                add("확인", f"{comparison}/seed{seed}",
                    f"B0 {float(left['MSE']):.12g}, B2 {float(right['MSE']):.12g}, B2−B0={delta:.12g}",
                    left["source_file"]+"; "+right["source_file"],
                    "반복 개발 V; 선택 모델과 공통 update를 구분. Probe는 고정 읽기 모델의 결과이며 공통 요인 식별 아님.",
                    "두 seed와 근접 용량 대조의 방향을 함께 판단")
            if "B2" in pair and "B2_CAP_POINTWISE" in pair:
                left, right = pair["B2_CAP_POINTWISE"], pair["B2"]
                add("확인", f"{comparison}/용량 대조/seed{seed}",
                    f"CAP {float(left['MSE']):.12g}, B2 {float(right['MSE']):.12g}, B2−CAP={float(right['MSE'])-float(left['MSE']):.12g}",
                    left["source_file"]+"; "+right["source_file"],
                    "근접 모수 수; 함수공간·연산량·최적화 난이도 차이와 전체 창 LayerNorm 통계 공유는 남음.",
                    "선택 성능을 시간 혼합의 독립적 인과효과로 확대하지 않음")
        common = {row["model"]: row for row in rows if row["comparison"] == "same_update"
                  and row["seed"] == seed and row["epoch"] == 3. and number(row.get("MSE")) is not None}
        if len(common) == 3:
            add("확인", f"공통1863update/29769exposure/seed{seed}",
                "; ".join(f"{name}={float(common[name]['MSE']):.12g}" for name in FAMILIES),
                "; ".join(common[name]["source_file"] for name in FAMILIES),
                "A50 검색 CP의 A60 척도 V 재평가. Fresh A60 선택 refit 표와 합치지 않음; 수렴 증명 아님.",
                "추가 학습량 설명과 용량 설명을 부분적으로만 통제")
    pems = {row["model"]: row for row in rows if row["comparison"] == "selected_models"
            and row["dataset"] == "pems08_C32" and number(row.get("MSE")) is not None}
    if len(pems) == 2:
        add("확인", "PeMS C32/K8 적용 범위", f"B0={float(pems['B0']['MSE']):.12g}; B2={float(pems['B2']['MSE']):.12g}; "
            f"B2−B0={float(pems['B2']['MSE'])-float(pems['B0']['MSE']):.12g}",
            pems['B0']['source_file']+"; "+pems['B2']['source_file'],
            "같은 C32/K8의 별도 학습 반복 V; C170/K43 대체 없음; 새 시간 일반화 아님.", "자료별 효과 방향을 그대로 보존")
    else:
        add("미완료", "PeMS C32/K8 적용 범위", "등록된 쌍의 최종 검산 가능 결과가 모두 있지 않음", "COMPARISONS.csv",
            "다른 C/K 수치로 빈칸을 채우지 않음", "정상 저장점과 중단 이유에 따라 재개 범위를 판단")
    add("추정", "B2 개선 원인", "동일 update·근접 용량·고정 probe는 경쟁 설명을 나눠 검사하지만 하나의 원인을 확정하지 않음", "COMPARISONS.csv",
        "시점별 대조도 전체 창 정규화 통계 공유. 학습량·용량·F 적합성의 상호작용 가능.", "결과의 방향과 제한을 함께 발표")
    add("미완료", "독립 시간 일반화", "허용된 미사용 미래 구간의 성능 증거 없음", "PLAN_KO.md",
        "기존 V를 시간 블록으로 나눠도 독립 검증이 되지 않음. 보호 T는 잠금 유지.",
        "새 자료 또는 미사용 미래 구간의 구체적인 접근 승인이 필요")
    add("미완료", "공통 요인 식별과 충분 학습", "공통 요인 식별·완전 수렴·독립 일반화는 입증되지 않음", "PLAN_KO.md; COMPARISONS.csv",
        "1차 공통 최대3epoch; 추가6epoch 여부는 별도 원 결정 기록만 따름", "학습 완료를 가설 지지로 바꾸지 않음")
    return result


def make_figures(rows, blocks, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "axes.spines.top":False,
        "axes.spines.right":False, "svg.fonttype":"none", "svg.hashsalt":NAME})
    colors = {"B0":"#0072B2", "B2":"#D55E00", "B2_CAP_POINTWISE":"#009E73"}
    figure_paths = []
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 6.4), constrained_layout=True)
    for r, seed in enumerate(SEEDS):
        for c, split in enumerate(("A_fit", "A_stop", "V_select")):
            ax = axes[r,c]
            for family in FAMILIES:
                selected = sorted((row for row in rows if row["seed"] == seed and row["model"] == family
                    and row.get("eval") == split and row.get("same_update") is True
                    and number(row.get("MSE")) is not None), key=lambda row:row["updates"])
                if selected:
                    ax.plot([row["updates"] for row in selected], [float(row["MSE"]) for row in selected],
                        marker="o", markersize=3, color=colors[family], label=family.replace("B2_CAP_POINTWISE","CAP pointwise"))
            ax.set_title(f"Seed {seed}: {split}")
            ax.set_xlabel("Completed optimizer updates")
            ax.set_ylabel("Channel-macro MSE"+(" (A60 scale)" if split == "V_select" else " (A50 scale)"))
            ax.grid(alpha=.18)
            if r == c == 0: ax.legend(fontsize=8)
    fig.suptitle("Matched training endpoints: full A-fit diagnostics, A-stop, and reused V\nSearch A50 checkpoints; curves are recomputed full-set metrics, not online losses",fontsize=12)
    path=output/"figures/MATCHED_TRAINING.svg"; fig.savefig(path,metadata={"Date":None}); plt.close(fig);figure_paths.append(path)
    fig, axes=plt.subplots(1,2,figsize=(11.4,4.4),constrained_layout=True)
    for ax,seed in zip(axes,SEEDS):
        for index,family in enumerate(FAMILIES):
            for offset,kind,hatch in ((-.17,"selected_models",""),(.17,"fixed_reader_probe","//")):
                selected=[row for row in rows if row["dataset"]=="jena" and row["seed"]==seed
                    and row["model"]==family and row["comparison"]==kind and number(row.get("MSE")) is not None]
                if selected:
                    value=float(selected[0]["MSE"])
                    ax.bar(index+offset,value,width=.32,color=colors[family],hatch=hatch,alpha=.87,
                        label=("Bolt + original D" if kind=="selected_models" else "Fixed full-history ridge") if index==0 else None)
                    ax.text(index+offset,value,f"{value:.3f}",ha="center",va="bottom",fontsize=8)
                else: ax.text(index+offset,.02,"NA",ha="center",rotation=90,fontsize=8)
        ax.set_xticks(range(3),["B0","B2","CAP pointwise"]);ax.set_ylabel("Channel-macro V MSE (A60 scale)")
        ax.set_title(f"Jena seed {seed}");ax.legend(fontsize=8);ax.grid(axis="y",alpha=.18)
    fig.suptitle("Selected representations: original Bolt route vs identical diagnostic reader\nProbe keeps P/E fixed; no original D or F; fixed lambda=1; repeated development V",fontsize=12)
    path=output/"figures/FIXED_READER_PROBE.svg";fig.savefig(path,metadata={"Date":None});plt.close(fig);figure_paths.append(path)
    fig,axes=plt.subplots(1,2,figsize=(11.8,4.4),constrained_layout=True)
    names=[("jena",92601,"Jena seed1"),("jena",92602,"Jena seed2"),("pems08_C32",92601,"PeMS C32")]
    for index,(dataset,seed,label) in enumerate(names):
        pair={row["model"]:row for row in rows if row["comparison"]=="selected_models"
            and row["dataset"]==dataset and row["seed"]==seed and row["model"] in ("B0","B2")
            and number(row.get("MSE")) is not None}
        for offset,family in ((-.18,"B0"),(.18,"B2")):
            if family in pair:
                value=float(pair[family]["MSE"])
                axes[0].bar(index+offset,value,.34,color=colors[family],label=family if index==0 else None)
                axes[0].text(index+offset,value,f"{value:.3f}",ha="center",va="bottom",fontsize=8)
            else:axes[0].text(index+offset,.03,"NA",ha="center",rotation=90)
    axes[0].set_xticks(range(3),[item[2] for item in names]);axes[0].set_ylabel("Channel-macro MSE (per-dataset A60 scale)")
    axes[0].set_title("Selected routes: separate dataset recipes");axes[0].legend();axes[0].grid(axis="y",alpha=.18)
    for seed,style in zip(SEEDS,("-","--")):
        selected=sorted((row for row in blocks if int(row["seed"])==seed),key=lambda row:int(row["partition_index"]))
        axes[1].plot(range(len(selected)),[float(row["delta_subset_macro_mse"]) for row in selected],style,
            marker="o",label=f"Jena seed{seed}")
    axes[1].axhline(0,color="black",linewidth=.8);axes[1].set_xticks(range(3),["First15","Middle14","Last14"])
    axes[1].set_ylabel("B2 - B0 block channel-macro MSE");axes[1].set_title("Partitions of the SAME V43; not new time tests")
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.18)
    fig.suptitle("Applicability is bounded by dataset, recipe, and repeated V\nNegative block delta favors B2; cross-dataset MSE magnitudes are not a common-unit ranking",fontsize=12)
    path=output/"figures/DATASET_TIME_SCOPE.svg";fig.savefig(path,metadata={"Date":None});plt.close(fig);figure_paths.append(path)
    return figure_paths


def report(rows, costs, claim_rows, audit):
    lines=["# B2 개선 원인과 적용 범위: 실제 결과 보고", "",
        "이 보고서는 기존 완료 결과와 신규 유한 검증의 실제 생성 파일을 구분한다. 새로운 원자료·보호 T·모델 checkpoint를 열지 않고, 등록된 신규 V 예측만 독립 FP64로 재집계했다.", "",
        f"[확인] 결과 수집 상태는 **{audit['status']}**다. 신규 V archive 검산 {len(audit['archive_checks'])}/34개, 누락 결과 {len(audit['missing_results'])}개, 거부 결과 {len(audit['rejected_results'])}개다. 검산 통과는 성능 개선이나 독립 일반화를 뜻하지 않는다.", "",
        "## 실제 차이와 발표 가능 문장", ""]
    for claim in claim_rows:
        lines.extend([f"- [{claim['layer']}] {claim['question']}: {claim['claim']} 근거: `{claim['evidence']}`. {claim['limitation']}"])
    lines.extend(["", "## 같은 학습량·근접 용량", "",
        "공통 update 표는 A50 검색 checkpoint에 대한 동일 update·표본 노출량 비교다. A_fit/A_stop은 검색 A50 척도이며 V만 공통 A60 척도로 환산했다. A_fit은 각 저장점의 전체 학습 구간 재평가이며 online minibatch loss가 아니다. 전체 창 LayerNorm 통계를 공유하므로 CAP을 완전히 시간 독립이라고 부르지 않는다. B2/CAP의 활성 모수 수는 실제 결과 열에 기록한다. 공통3epoch는 수렴 증거가 아니다.", "", "```text",
        "seed   model              epoch  updates exposure       V_MSE        V_MAE status"])
    for row in rows:
        if row["comparison"]=="same_update":
            lines.append(f"{row['seed']}  {row['model']:<18} {float(row['epoch']):4.2f} {row['updates']:7} {row['exposure']:8} {str(row.get('MSE','NA')):>13} {str(row.get('MAE','NA')):>13} {row['status']}")
    lines.extend(["```", "", "![같은 학습량 비교](figures/MATCHED_TRAINING.svg)", "", "## 동일 선택 규칙과 표현 probe", "",
        "아래 선택 모델은 원 A_stop 선택과 fresh A60 refit 결과다. 위 검색 checkpoint 표와 합치지 않는다. Probe는 선택 표현 P/E를 고정하고 전체 512×6 과거로 48×21 미래 변화를 읽는 같은 선형 모델이다. 원 D를 물려주지 않으며 λ=1·FIT만의 정규화·동일 표본을 유지한다. 결과는 이 읽기 모델에서의 접근성에 한정된다.", "", "```text",
        "comparison         dataset       seed  model                params epoch updates exposure MSE MAE status"])
    for row in rows:
        if row["comparison"] in ("selected_models","fixed_reader_probe"):
            lines.append(" ".join(str(row.get(key,"NA")) for key in ("comparison","dataset","seed","model","params","epoch","updates","exposure","MSE","MAE","status")))
    lines.extend(["```", "", "![동일 probe 비교](figures/FIXED_READER_PROBE.svg)", "", "## 자료·시간 범위와 강한 기준선", "",
        "C32/K8의 B0/B2는 같은 C32 계약의 결과만 사용한다. C170/K43 결과를 넣지 않는다. 다른 자료에서 각각 학습한 비교는 적용 범위의 단서이며 zero-shot 검증이 아니다. Jena의 첫·중앙·마지막 블록은 이미 본 V의 부분집합이다. 부분집합 지표를 다시 단순 평균해 전체 성능으로 쓰지 않으며, 기존 분해 표의 전체 분모 가산 기여를 보존한다.", "",
        "![자료와 시간의 범위](figures/DATASET_TIME_SCOPE.svg)", "",
        "기존 채널·거리별 효과: [분해 보고](ERROR_DECOMPOSITION_KO.md), [채널](ERROR_DECOMPOSITION_CHANNEL.csv), [거리](ERROR_DECOMPOSITION_HORIZON.csv), [시간 블록](ERROR_DECOMPOSITION_BLOCK.csv). 원 그림 두 개는 수정하지 않았다.", "", "```text",
        "dataset       model          MSE MAE   recipe/source"])
    for row in rows:
        if row["comparison"]=="strong_baseline_context":
            lines.append(f"{row['dataset']} {row['model']} {row['MSE']} {row['MAE']} {row['source_file']}")
    lines.extend(["```", "",
        "[확인] 강한 기준선·no-Bolt는 기존 완료 결과를 재사용했으며 재학습하지 않았다. 같은 최종 P/E를 읽는 비교와 별도 end-to-end 구조 학습은 원 recipe가 다르다. Ridge·G·LoRA보다 낮은 오차라는 보편적 주장은 하지 않는다. B0도 Bolt와 D를 통해 전체 과거를 사용한다.", "", "## 실제 전체 비용", "",
        "새 비용은 학습 종료 후 단독 CPU2, 고정 물리 과거 입력/관측 mask, 전처리→모델→역정규화→CPU 반환을 측정한 결과만 넣는다. Warm trial에는 모델 로딩이 포함되지 않는다. 물리 경로와 저장 V 예측의 parity 실패·누락 시 NA를 유지한다. 이전 같은 C32 Raw/G는 재측정하지 않았다.", "", "```text",
        "model batch median_s p90_s status"])
    for row in costs:
        lines.append(" ".join(str(row[key]) for key in ("model","batch","median_seconds","p90_seconds","status")))
    lines.extend(["```", "", "## 남은 증거와 재개", "",
        "[추정] 공통 update와 근접 모수 수는 선택 학습량·단순 용량 설명을 부분적으로 줄인다. 명시적 시간 혼합의 이득은 probe 및 두 seed 방향과 함께 판단해야 한다. 서로 다른 함수공간과 최적화 난이도, 표현과 동결 Bolt의 적합성은 남는 설명이다. P/E/D는 이미 최종 미래 MSE로 학습하므로 실패를 재구성 목적 탓으로 설명하지 않는다. 현재 C값 anchor bypass를 사용한다.", "",
        "[미완료] 새 미사용 미래 구간·독립 자료·공통 요인 식별·완전 수렴에 관한 증거는 없다. 사용자는 새 자료가 없으며 현재 진단을 계속하라고 답했다. 따라서 독립 시간 일반화는 INDEPENDENT_GENERALIZATION_PENDING이며 허가 대기로 다른 진단을 멈춘 상태가 아니다. 기존 T 잠금은 유지되며 상류 보간 계보가 미확정인 범위도 보존한다. 향후 독립 일반화를 위해서는 새 미사용 미래 구간(동일 채널과 관측 mask) 또는 별도 자료가 필요하다. 누락 경로는 아래 실제 경로를 기준으로 root가 정상 저장점·원인·남은 자원을 확인해야 하며 완료 실험을 자동 재실행하지 않는다.", ""])
    for source in audit["missing_results"]:lines.append(f"- 미생성 결과: `{source}`")
    for item in audit["rejected_results"]:lines.append(f"- 검산 거부: `{item['source']}` — {item['reason']}")
    lines.extend(["", "[전체 비교 원수치](COMPARISONS.csv) · [비용](COSTS.csv) · [발표 문장과 근거](CLAIM_EVIDENCE.csv) · [실제 검산](FINAL_AUDIT.json)", ""])
    return "\n".join(lines)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--repo",type=Path)
    parser.add_argument("--asset",type=Path)
    parser.add_argument("--deadline",type=float)
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="SOURCE_ONLY",input_reads=0,writes=0,torch_imported="torch" in sys.modules)))
        return 0
    if args.repo is None or args.asset is None or args.deadline is None:
        parser.error("Explicit roots and finite closeout deadline required")
    for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        if os.environ.get(key)!="2":raise ValueError("Set CPU2 environment before interpreter startup")
    repo,asset=args.repo.resolve(),args.asset.resolve()
    output,private=repo/"reports"/NAME,asset/".cache"/NAME
    targets=[output/name for name in ("COMPARISONS.csv","COSTS.csv","CLAIM_EVIDENCE.csv","RESULTS_KO.md","FINAL_AUDIT.json")]
    targets += [output/"figures"/(name+".svg") for name in ("MATCHED_TRAINING","FIXED_READER_PROBE","DATASET_TIME_SCOPE")]
    if any(path.exists() for path in targets):raise FileExistsError("Final closeout is CreateNew; preserve previous outputs")
    started=time.time();quiet_before=quiet_and_limits(args.deadline)
    scope=read(private/"CAMPAIGN_v1.json")
    if scope.get("max_Jena_conditions")!=6 or scope.get("max_fitted_probes")!=6 or scope.get("new_GPU_seconds")!=0:
        raise ValueError("Registered finite new-campaign scope differs")
    preexecution=[]
    for filename in ("CAP_INITIALIZATION_AUDIT_RESULT.json","PEMS_RESTORE_AUDIT_RESULT.json"):
        path=private/filename
        if path.is_file():
            receipt=read(path)
            preexecution.append(dict(evidence_id=filename,sha256=sha(path),status=receipt.get("status"),
                reused_existing_diagnostic=True,new_audit_optimizer_updates=0,
                checks=[dict(identity=item.get("seed",item.get("family")),status=item.get("status"),
                             checks=item.get("checks",{})) for item in receipt.get("rows",[])]))
    preserved={path.relative_to(output).as_posix():sha(path) for path in output.glob("ERROR_DECOMPOSITION*") if path.is_file()}
    preserved.update({path.relative_to(output).as_posix():sha(path) for path in (output/"figures").glob("*.svg")})
    rows,missing,rejected,archives,contracts,inputs=collect(repo,private,output,args.deadline)
    costs,cost_checks=collect_costs(repo,output,inputs)
    for row in costs:
        if row["status"]=="PENDING":missing.append(row["source_file"])
        elif row["status"] not in ("COMPLETE_PROTOCOL","EXISTING_COMPLETE_PROTOCOL_REUSED"):
            rejected.append(dict(source=row["source_file"],reason=row["status"]))
    for row in rows:
        if row["comparison"]=="selected_models" and row["dataset"]=="pems08_C32":
            source=output/"costs"/f"pems_{row['model']}_b1"/"RESULT.json"
            if source.is_file() and any(item["model"]==row["model"] and item["batch"]==1 and item["status"]=="COMPLETE_PROTOCOL" for item in costs):
                receipt=read(source)
                if receipt.get("status")=="COMPLETE_PROTOCOL" and receipt.get("selected_metrics_sha256")==row["source_sha256"]:
                    row["checkpoint_sha256"]=receipt["artifact_sha256"]
    expected_archive_count=34
    status="COMPLETE_REGISTERED_RESULTS_NOT_HYPOTHESIS_CONFIRMATION" if not missing and not rejected and len(archives)==expected_archive_count else "PARTIAL_RESULTS_PRESERVED_NOT_METHOD_FAILURE"
    audit=dict(schema="B2_MECHANISM_FINAL_READONLY_AUDIT_v1",status=status,
        source_sha256=sha(__file__),campaign_scope_sha256=sha(private/"CAMPAIGN_v1.json"),
        archive_checks=archives,expected_new_V_archives=expected_archive_count,
        cost_checks=cost_checks,missing_results=missing,rejected_results=rejected,
        original_input_sha256=inputs,within_cohort_contracts=contracts,
        preserved_original_A_sha256=preserved,quiet_before=quiet_before,
        preexecution_diagnostic_receipts=preexecution,
        operation_counts_scope="READ_ONLY_CLOSEOUT_ONLY",
        campaign_evidence_counts=dict(
            reused_selected_Jena_models=sum(row["comparison"]=="selected_models" and row["dataset"]=="jena" and row["status"]=="EXISTING_RESULT_REUSED" for row in rows),
            CAP_selected_result_statuses={str(row["seed"]):row["status"] for row in rows if row["comparison"]=="selected_models" and row["model"]=="B2_CAP_POINTWISE"},
            PeMS_resumed_selected_result_statuses={row["model"]:row["status"] for row in rows if row["comparison"]=="selected_models" and row["dataset"]=="pems08_C32"},
            fixed_probe_result_statuses=dict(Counter(row["status"] for row in rows if row["comparison"]=="fixed_reader_probe")),
            new_cost_result_statuses=dict(Counter(row["status"] for row in costs if row["model"] in ("B0","B2"))),
            counts_are="Available condition result receipts; not new operations performed by closeout"),
        protected_T_numeric_access=0,raw_series_reads=0,checkpoint_loads=0,model_calls=0,
        new_training_runs=0,new_inference_runs=0,new_cost_measurements=0,
        statistics="FP64 observed SSE/SAE per channel, then equal active-channel mean; independent math.fsum",
        independence="Repeated development V; no independent future evaluation; no significance test",
        independent_future_status="INDEPENDENT_GENERALIZATION_PENDING",no_new_data_user_confirmed=True,
        optional_six_epoch_execution="Not inferred by closeout; root decision/actual output required")
    user_status=private/"USER_DATA_STATUS_v1.json"
    audit["user_data_status_evidence"]={"evidence_id":user_status.name,"sha256":sha(user_status)} if user_status.is_file() else None
    with (output/"ERROR_DECOMPOSITION_BLOCK.csv").open(encoding="utf-8-sig",newline="") as stream:blocks=list(csv.DictReader(stream))
    decomposition_path=output/"ERROR_DECOMPOSITION_SUMMARY.json"
    inputs[decomposition_path.relative_to(repo).as_posix()]=sha(decomposition_path)
    claim_rows=claims(rows,read(decomposition_path))
    figures=make_figures(rows,blocks,output)
    quiet_after=quiet_and_limits(args.deadline)
    if "torch" in sys.modules:raise RuntimeError("Read-only closeout unexpectedly imported torch")
    if any(sha(output/path)!=digest for path,digest in preserved.items()):raise ValueError("Original A evidence changed")
    if any(sha(repo/path)!=digest for path,digest in inputs.items()):raise ValueError("Science input changed during closeout")
    write_csv(targets[0],rows,FIELDS)
    write_csv(targets[1],costs,("dataset","model","batch","status","median_seconds","p90_seconds","CPU_threads","source_file","source_sha256"))
    write_csv(targets[2],claim_rows,("layer","question","claim","evidence","limitation","next_decision"))
    with targets[3].open("x",encoding="utf-8",newline="\n") as stream:stream.write(report(rows,costs,claim_rows,audit))
    audit.update(quiet_after=quiet_after,elapsed_seconds=time.time()-started,torch_imported=False,
        output_sha256={path.relative_to(output).as_posix():sha(path) for path in targets[:4]+figures})
    new(targets[4],audit)
    print(json.dumps(dict(status=status,new_V_archive_checks=len(archives),missing=len(missing),rejected=len(rejected),
        new_figures=len(figures),comparison_rows=len(rows),model_calls=0,protected_T_numeric_access=0)))
    return 0 if not rejected else 2


if __name__=="__main__":raise SystemExit(main())
