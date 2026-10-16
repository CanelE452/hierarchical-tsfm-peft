"""Metadata-only scientific synthesis; never opens arrays, weights, gzip, GPU, or T."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

DATASETS = ("beijing", "jena", "electricity", "pems08")
MUTABLE = {"execution_status", "training_status", "validity_status", "scientific_status", "publication_status", "reason", "resume_command"}
UNION = "diagnostics/raw_deep_g_saved_finalize_1792091230000/UNION_BUNDLES.csv"
UNION_SHA = "12b46d151739edb970862763a222387d7056e758787dabefc264b4364d3770f0"
MATCHED = "diagnostics/raw_deep_g_saved_finalize_1792091230000/MATCHED_G_COMPARISON.csv"
MATCHED_SHA = "43c2970c899301ee20043eae0c598e58a78c55bd0e5d347671e724ed7e46fde8"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            value.update(block)
    return value.hexdigest()


def new(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def csv_new(path, rows):
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", required=True)
    args = parser.parse_args()
    public, output = Path(args.public_root).resolve(), Path(__file__).resolve().parent
    started = time.time()
    forbidden = []
    def guard(event, values):
        if event == "import" and str(values[0]).split(".")[0] in {"numpy", "torch", "chronos"}:
            forbidden.append("ARRAY_OR_MODEL_IMPORT");raise RuntimeError("Metadata-only imports")
        if event != "open" or not isinstance(values[0], (str, bytes)):
            return
        path = Path(values[0]).resolve()
        if path.suffix.casefold() in {".npz", ".npy", ".pt", ".safetensors", ".gz"} or "/t_final/" in path.as_posix().casefold():
            forbidden.append("ARRAY_WEIGHT_GZIP_T");raise RuntimeError("Metadata-only file scope")
        flags = values[2] if len(values)>2 and isinstance(values[2], int) else 0
        if flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND) and not path.is_relative_to(output):
            forbidden.append("OUTSIDE_WRITE");raise RuntimeError("Writes confined to owned synthesis diagnostic")
    sys.addaudithook(guard)
    inputs = {}
    def record(path):
        path = Path(path)
        key = path.relative_to(public).as_posix()
        value = dict(sha256=sha(path), bytes=path.stat().st_size)
        if key in inputs and inputs[key] != value:raise ValueError("Metadata changed during extraction")
        inputs[key] = value
        return path
    def read(path):return json.loads(record(path).read_text(encoding="utf-8-sig"))
    def rows(path):return list(csv.DictReader(record(path).open(encoding="utf-8-sig", newline="")))
    prior = read(public / "diagnostics/raw_deep_g_saved_finalize_1792091230000/RESULT.json")
    source_before = {name: sha(public / name) for name in prior["source39_after"]}
    if source_before != prior["source39_after"]:raise ValueError("Frozen source39 changed")
    if sha(public / UNION) != UNION_SHA or sha(public / MATCHED) != MATCHED_SHA:raise ValueError("Exact evidence input changed")
    union, matched = rows(public / UNION), rows(public / MATCHED)
    if len(union) != 457:raise ValueError("Bundle inventory differs")
    if sum(row["split"] == "V_select" for row in union) != 405 or sum(row["split"] == "M_cal" for row in union) != 52:raise ValueError("V/M count differs")
    registry_bytes = (public / "EXPERIMENT_REGISTRY.csv").read_bytes()
    registry = list(csv.DictReader(io.StringIO(registry_bytes.decode("utf-8-sig"))))
    by_id = {row["experiment_id"]: row for row in registry}
    assets = read(public / "DATA_CONTRACT.json")["assets"]
    inventory, missing, definitions = [], [], {}
    V_ids = set()
    def trained_assets(identity):
        folder = public / "runs" / identity
        receipt_path = next((folder / name for name in ("BASE_FIT_RECEIPT.json", "CONTROL_FIT_RECEIPT.json", "RIDGE_FIT_RECEIPT.json") if (folder / name).is_file()), None)
        source_path = next((folder / name for name in ("SOURCE_CONTRACT.json", "CONTROL_SOURCE_CONTRACT.json") if (folder / name).is_file()), None)
        receipt = read(receipt_path) if receipt_path else {}
        source = read(source_path) if source_path else {}
        return receipt, source, receipt_path, source_path
    for bundle in union:
        identity = bundle["run_id"]
        if bundle["split"] != "V_select" or identity not in by_id:continue
        row = by_id[identity]
        if row["fold"] != "primary" or row["dataset"] not in DATASETS:continue
        path = public / bundle["manifest_public_relative_path"]
        manifest = read(path)
        if sha(path) != bundle["manifest_sha256"] or manifest["run_id"] != identity or manifest["split"] != "V_select":raise ValueError("Scientific score source identity differs")
        score, meta = manifest["score"], manifest.get("metadata", {})
        if not score:raise ValueError("Primary V score absent")
        receipt, source, rp, sp = trained_assets(identity)
        parent = meta.get("parent_run_id")
        if not receipt and parent and parent != "nested_pca":receipt, source, rp, sp = trained_assets(parent)
        if meta.get("family_run_id"):
            category = "raw_calibration" if meta.get("reference_kind") == "raw_calibration" else "combinations"
            rp = public / category / meta["family_run_id"] / "receipt.json"
            receipt = read(rp)
            sp = None;source = {}
            spec_path = public / "combinations" / meta["family_run_id"] / "spec.json"
            if spec_path.is_file():source["spec"] = read(spec_path)
        spec = source.get("spec", {})
        family = row["family"]
        info = "A50 search / A_stop checkpoint selection / fresh A60 refit" if receipt.get("refit_fit_end") is not None else ""
        if receipt.get("role") == "AplusM_control":info = "A+M direct supervised fit; A60 normalization; V checkpoint selection"
        elif receipt.get("variant"):info = "Frozen A-only parents; M_cal head/alpha fit; V neural checkpoint selection" if receipt.get("variant") != "alpha_family" else "Frozen A-only parents; analytic alpha on M_cal; optimizer updates0"
        elif family == "RAW_RIDGE":info = "All legal A50 origins closed form; A_stop alpha choice; fresh all-origin A60 refit"
        elif family.startswith("RAW_BOLT_"):info = "Official frozen foundation; no optimizer; A60 normalization only"
        elif family in {"CURRENT", "SEASONAL_NAIVE"}:info = "No optimizer; fixed baseline; A60 normalization"
        elif family == "NESTED_PCA":info = "A60 covariance projection; no optimizer; frozen Bolt; nested prefix"
        elif family == "RAW_CALIBRATED":info = "Frozen Raw; M_cal constant-bias closed form; no neural optimizer"
        if meta.get("variant") == "posthoc":info += "; prefix truncated from Kmax model; this prefix not trained separately"
        elif meta.get("variant") == "shared":info += "; shared multi-K checkpoint prefix; not independent fit per K"
        epochs = receipt.get("epochs", receipt.get("registered_epoch_endpoints", []))
        fit_end = receipt.get("refit_fit_end", receipt.get("control_fit_end", receipt.get("fit_end", meta.get("parent_fit_end"))))
        checkpoint_split = receipt.get("checkpoint_selection_split", receipt.get("checkpoint_split", "A_stop" if receipt.get("selected_astop_mse") is not None else "A_stop" if family == "RAW_RIDGE" else ""))
        entry = dict(run_id=identity, dataset=row["dataset"], fold=row["fold"], block=row["block"], family=family, latent_k=int(row["latent_k"]), seed=int(row["seed"]),
            split="V_select", cohort=manifest["cohort"], mse=score["mse"], mae=score["mae"], n_observed=score["n_observed"], n_origins=score["n_origins"], H=score["horizon"], C=score["num_channels"],
            metric="A60_normalized_observed_original_channel_macro_MSE", point_output="Bolt_median_quantile0.5_or_native_deterministic", registry_execution_status=row["execution_status"], registry_training_status=row["training_status"],
            actual_receipt_status=receipt.get("status", receipt.get("execution_status", "FROZEN_OR_ALIAS_NO_OWN_FIT")), actual_training_status=receipt.get("training_status", row["training_status"]),
            attempted_epoch_endpoints="|".join(map(str,epochs)), selected_epoch=receipt.get("selected_epoch"), actual_head_epochs=receipt.get("actual_epochs"),
            selected_updates=receipt.get("selected_updates"), optimizer_updates=receipt.get("updates", receipt.get("search_updates", receipt.get("actual_optimizer_updates"))), fresh_refit_updates=receipt.get("refit_updates"),
            fit_information_scope=info, actual_fit_end=fit_end, normalization_fit_end=receipt.get("normalization_fit_end", receipt.get("refit_fit_end", meta.get("canonical_scaler_fit_end"))),
            A60_end=assets[row["dataset"]]["primary_split_ranges"]["A_stop"][1], M_cal_end=assets[row["dataset"]]["primary_split_ranges"]["M_cal"][1], checkpoint_selection_split=checkpoint_split,
            fit_device=spec.get("device", "cpu" if family == "RAW_RIDGE" else "no optimizer" if family.startswith("RAW_BOLT") else "not recorded in public metadata"), fit_learning_rate=spec.get("lr", spec.get("learning_rate")),
            last_cp_best=receipt.get("last_cp_best"), last_scores_still_falling=receipt.get("last_scores_still_falling"), parent_or_alias_run_id=parent or meta.get("family_run_id", ""),
            ridge_selected_alpha=receipt.get("selected_alpha"), ridge_features=receipt.get("config", {}).get("features", ""), closed_form_effective_epochs=receipt.get("effective_epochs", ""),
            training_source_files_match_current=all(sources_digest == source_before.get(name) for name,sources_digest in source["files"].items()) if source.get("files") else None,
            source_manifest_public_relative_path=path.relative_to(public).as_posix(), source_manifest_sha256=sha(path), source_score_location="/score/mse;/score/mae", source_score_mse_literal=score["mse"],
            source_receipt_public_relative_path=rp.relative_to(public).as_posix() if rp else "", source_receipt_sha256=sha(rp) if rp else "", source_contract_public_relative_path=sp.relative_to(public).as_posix() if sp else "",
            source_contract_sha256=sha(sp) if sp else "", prediction_source_sha=manifest["source_sha"], training_source_sha=receipt.get("source_sha", receipt.get("contract", {}).get("source_sha", "")),
            model_sha=manifest["model_sha"], data_sha=manifest["data_sha"], scaler_sha=manifest["scaler_sha"], selected_artifact_declared_sha=receipt.get("artifact_sha256", receipt.get("selected_checkpoint_sha256", meta.get("parent_artifact_sha256", ""))),
            new_FP64_scoring_performed=False, T_selection_performed=False, deployment_cost_status="PENDING_ROOT_FROZEN_PROTOCOL")
        inventory.append(entry);V_ids.add(identity);definitions[identity] = entry
    for row in registry:
        if row["fold"] != "primary" or row["dataset"] not in DATASETS or row["experiment_id"] in V_ids:continue
        if row["block"] in {"E3", "E4"} or row["family"] in {"DIRECT_G", "G_REFIT_APLUSM", "LORA_R8", "LORA_REFIT_APLUSM", "ADAPTS_BOLT_PORT"}:
            missing.append(dict(run_id=row["experiment_id"], dataset=row["dataset"], block=row["block"], family=row["family"], latent_k=row["latent_k"], execution_status=row["execution_status"], training_status=row["training_status"], reason=row["reason"], score_status="NO_V_SCORE_IN_PINNED_AUDITED_INVENTORY", fake_zero_score=False))
    csv_new(output / "PRIMARY_V_METADATA_INVENTORY.csv", inventory)
    csv_new(output / "UNAVAILABLE_RELEVANT_CONDITIONS.csv", missing)
    families = {"RAW_BOLT_SMALL","RAW_BOLT_MINI","RAW_BOLT_TINY","RAW_RIDGE","DIRECT_G","G_REFIT_APLUSM","LORA_R8","LORA_REFIT_APLUSM","ADAPTS_BOLT_PORT"}
    strong = [row for row in inventory if row["family"] in families]
    csv_new(output / "STRONG_BASELINES_V.csv", strong)
    csv_new(output / "E3_E4_V_ALL_AVAILABLE.csv", [row for row in inventory if row["block"] in {"E3","E4"}])
    csv_new(output / "SAME_G_E2_REPORTED.csv", matched)
    paired = []
    for dataset in DATASETS:
        conditions = {row["method"]: row for row in matched if row["dataset"] == dataset}
        for method in ("H_B0","Hpast","RawH","ALPHA_FIT","ALPHA05","G"):
            b2, other = conditions["H_B2"], conditions[method]
            paired.append(dict(dataset=dataset, comparison="H_B2_minus_"+method, run_id_a=b2["run_id"], run_id_b=other["run_id"], mse_a=float(b2["mse"]), mse_b=float(other["mse"]),
                mse_delta=float(b2["mse"])-float(other["mse"]), mae_delta=float(b2["mae"])-float(other["mae"]), scalar_arithmetic_on_existing_scores_only=True,
                same_G_model_sha=b2["G_model_sha"], same_G_artifact_sha=b2["G_artifact_sha256"], source_csv_relative_path=MATCHED, source_csv_sha256=MATCHED_SHA,
                no_significance_or_T_selection_claim=True))
    csv_new(output / "SAME_G_PAIR_DELTAS.csv", paired)
    labels = [("RawSmall","E1_RAW_BOLT_SMALL_k0_s92601"),("RawMini","E1_RAW_BOLT_MINI_k0_s92601"),("RawTiny","E1_RAW_BOLT_TINY_k0_s92601"),("Ridge_A60","E1_RAW_RIDGE_k0_s92601"),
        ("G_initial025","E1_DIRECT_G_k0_s92601"),("G_GPU_to1","E1_DIRECT_G_k0_s92601__e1"),("G_GPU_to3","E1_DIRECT_G_k0_s92601__e1__e3"),("G_CPU_fresh3","E1_DIRECT_G_k0_s92601__cpu_fresh_e3"),
        ("G_AplusM025","E2_G_REFIT_APLUSM_k0_s92601_G"),("G_AplusM3","E2_G_REFIT_APLUSM_k0_s92601_G__e3"),("LoRA025","E1_LORA_R8_k0_s92601"),("LoRA_GPU3","E1_LORA_R8_k0_s92601__e3"),
        ("LoRA_AplusM025","E2_LORA_REFIT_APLUSM_k0_s92601_LORA")]
    matrix = []
    for label, tail in labels:
        for dataset in DATASETS:
            identity = "primary_"+dataset+"_"+tail
            if identity in definitions:matrix.append(dict(table_label=label, **definitions[identity]))
            else:matrix.append(dict(table_label=label, dataset=dataset, run_id=identity, score_status="NO_PINNED_V_MANIFEST", mse=None, mae=None, no_fake_zero=True))
    for dataset, k in (("beijing",3),("jena",6),("electricity",81),("pems08",43)):
        for suffix,label in (("","AdaPTS_port025"),("__e3","AdaPTS_port3")):
            identity=f"primary_{dataset}_E1_ADAPTS_BOLT_PORT_k{k}_s92601{suffix}"
            matrix.append(dict(table_label=label, **definitions[identity]) if identity in definitions else dict(table_label=label,dataset=dataset,run_id=identity,mse=None,mae=None,score_status="NO_PINNED_V_MANIFEST",no_fake_zero=True))
    csv_new(output / "PROFESSOR_BASELINE_MATRIX.csv", matrix)
    sources_after = {name: sha(public / name) for name in source_before}
    inputs_after = {name: dict(sha256=sha(public/name),bytes=(public/name).stat().st_size) for name in inputs}
    registry_after = list(csv.DictReader((public / "EXPERIMENT_REGISTRY.csv").open(encoding="utf-8-sig", newline="")))
    after_map = {row["experiment_id"]:row for row in registry_after}
    scientific_changes = [identity for identity,row in by_id.items() if identity not in after_map or any(after_map[identity][key]!=value for key,value in row.items() if key not in MUTABLE)]
    if sources_after != source_before or inputs_after != inputs or scientific_changes or forbidden:raise ValueError("Metadata/source/registry science changed")
    new(output / "SOURCE_PINS.json", dict(source39_before=source_before,source39_after=sources_after,inputs=inputs,input_hashes_unchanged=True,registry_before_sha256=hashlib.sha256(registry_bytes).hexdigest(),registry_after_sha256=sha(public/"EXPERIMENT_REGISTRY.csv"),registry_scientific_changes=scientific_changes))
    new(output / "RESULT.json", dict(status="METADATA_ONLY_EXTRACTION_COMPLETE",primary_V_rows=len(inventory),strong_baseline_rows=len(strong),E3_E4_rows=sum(row["block"] in {"E3","E4"} for row in inventory),missing_rows=len(missing),
        source39_unchanged=True,input_metadata_files=len(inputs),all_score_rows_link_to_exact_manifest=True,canonical_V_bundles=405,canonical_dense_M_manifests=48,audit_only_posthoc_M_contracts=4,
        union457_is_saved_identity_inventory_not457_original_manifests=True,model_array_gzip_or_T_opened=False,new_statistics_recomputed=False,source_or_registry_written=False,Git_calls=0,forbidden_attempts=forbidden,wall_seconds=time.time()-started,created_unix=time.time(),driver_sha256=sha(__file__)))
    print(json.dumps(dict(status="METADATA_ONLY_EXTRACTION_COMPLETE",primary_V=len(inventory),strong=len(strong),E3E4=sum(row["block"] in {"E3","E4"} for row in inventory),seconds=time.time()-started)),flush=True)


if __name__ == "__main__":main()
