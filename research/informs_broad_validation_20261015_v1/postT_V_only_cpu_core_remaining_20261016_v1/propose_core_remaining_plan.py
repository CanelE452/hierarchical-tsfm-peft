"""Metadata-only proposal for 29 explicit post-T fresh CPU core conditions."""
import argparse
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

NAME = "postT_V_only_cpu_core_remaining_20261016_v1"
SCOPE = "diagnostics/postT_core56_scope_20261016_v1/CORE56_CORRESPONDENCE.csv"
SCOPE_SHA = "fe39487020d08dc7c14ee7e4f287227136d965591dbb69484aaa848950515bc5"
PCA_PLAN = "postT_V_only_cpu_fixed_pca_20261016_v1/PLAN.json"
PCA_SHA = "b4207edec63331cf95b24909c24c692091251dd8e29316f8867b9f66c65a0930"
DEPENDENCIES = (
    "postT_V_only_cpu_required_minima_20261016_v1/required_minima_runner.py",
    "postT_V_only_cpu_fixed_pca_20261016_v1/fixed_pca_runner.py",
    "postT_V_only_cpu_lag_20261016_v1/cpu_lag_runner.py",
    "postT_V_only_cpu_lag_20261016_v1/cpu_group_supervisor.py",
)
DEADLINE = 1792222550
SOURCE6 = {"fit_base.py", "training.py", "networks.py", "data_contract.py", "gpu_runtime.py", "evaluation.py"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def new(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def plain(path):
    path = Path(path).absolute()
    if any(p.is_symlink() or getattr(p, "is_junction", lambda: False)() for p in (path, *path.parents)):
        raise ValueError("Symlink/junction paths are forbidden")
    return path.resolve()


def relative(root, name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or ":" in str(name):
        raise ValueError("Safe root-relative path required")
    root = plain(root)
    result = plain(root / path)
    if not result.is_relative_to(root):
        raise ValueError("Path leaves its declared root")
    return result


def check_pins(root, pins):
    for name, expected in pins.items():
        if sha(relative(root, name)) != expected:
            raise ValueError("Frozen input/source changed: " + name)


def selected_scope(rows):
    if len(rows) != 56 or len({r["initial_registered_ID"] for r in rows}) != 56:
        raise ValueError("Unique original core56 required")
    selected = [r for r in rows if r["remaining_after_all_planned8_success_SCENARIO_ONLY"] == "True"]
    if len(selected) != 29 or {r["dataset"] for r in selected} != {"beijing", "jena", "electricity", "pems08"}:
        raise ValueError("The exact conditional remaining29 scope is required")
    return selected


def condition(row, original, task, source39):
    dataset, label, registered_k = row["dataset"], row["family"], int(row["registered_K"])
    family = {"LORA_R8": "LORA", "ADAPTS_BOLT_PORT": "ADAPTS"}.get(label, label)
    if family not in {"B0", "B2", "B4", "LORA", "ADAPTS"}:
        raise ValueError("Out-of-scope core family")
    channels = len(task["columns"])
    if (int(row["C"]) != channels or int(row["L"]) != task["L"] or int(row["H"]) != task["H"]
            or int(row["seed"]) != 92601 or row["fold"] != "primary"
            or original.get("data_sha") != task["data_sha256"]):
        raise ValueError("Original core/task/data identity differs")
    old = original["spec"]
    expected = dict(dataset=dataset, family=family, latent_k=channels if family == "LORA" else registered_k,
                    seed=92601, fold="primary", microbatch=4, effective_batch=16,
                    F_row_microbatch=128, checkpoint_inputs=False, lr=.0001 if family == "LORA" else .001)
    if any(old.get(k) != v for k, v in expected.items()) or old.get("run_id") != row["initial_registered_ID"]:
        raise ValueError("Original registered science recipe differs")
    run_id = f"postT_cpu_{dataset}_E1_{label}_k{registered_k}_s92601_remaining_v1"
    spec = copy.deepcopy(old)
    spec.update(run_id=run_id, device="cpu", epochs=[.25, 1., 3.],
                microbatch=1 if dataset in {"electricity", "pems08"} else 4,
                stage_label="postT_cpu_fresh_V_only_remaining_core29",
                provenance=dict(upstream_registered_id=row["initial_registered_ID"],
                    old_weights_optimizer_RNG_or_state_reused=False,
                    training_microbatch_changed=dataset in {"electricity", "pems08"},
                    bitwise_training_equivalence_claimed=False))
    files_equal = (set(original.get("files", {})) == SOURCE6
                   and all(source39.get(name) == digest for name, digest in original["files"].items()))
    return dict(run_id=run_id, upstream_registered_id=row["initial_registered_ID"], dataset=dataset,
                registered_family=label, registered_latent_k=registered_k, family=family,
                latent_k=spec["latent_k"], effective_F_width=channels if family == "LORA" else registered_k,
                spec=spec, preflight_status="SOURCE_METADATA_COMPATIBLE" if files_equal else "BLOCKED_SOURCE_METADATA",
                preflight_reason=None if files_equal else "Original source6 differs; ROOT reconciliation required; route retained")


def registry_bytes(conditions, tasks):
    fields = ("experiment_id", "dataset", "family", "latent_k", "num_channels", "length", "horizon", "seed", "fold",
              "upstream_registered_id", "device", "microbatch", "effective_batch", "epochs", "stage")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for c in conditions:
        task = tasks[c["dataset"]]
        writer.writerow(dict(experiment_id=c["run_id"], dataset=c["dataset"], family=c["registered_family"],
            latent_k=c["registered_latent_k"], num_channels=len(task["columns"]), length=task["L"], horizon=task["H"],
            seed=92601, fold="primary", upstream_registered_id=c["upstream_registered_id"], device="cpu",
            microbatch=c["spec"]["microbatch"], effective_batch=16, epochs="0.25|1|3", stage="POST_T_V_ONLY_DEVELOPMENT"))
    return stream.getvalue().encode("utf-8")


def build_proposal(public, cache, asset):
    public, cache, asset = map(plain, (public, cache, asset))
    if sha(relative(public, SCOPE)) != SCOPE_SHA or sha(relative(public, PCA_PLAN)) != PCA_SHA:
        raise ValueError("Original scope/PCA metadata pin differs")
    base = read(relative(public, PCA_PLAN))
    if len(base["source39"]) != 39 or len(base["immutable_public"]) != 9:
        raise ValueError("Original source39 and immutable9 are required")
    check_pins(public, base["source39"]); check_pins(public, base["immutable_public"])
    if read(public / "SELECTION_LOCK.json").get("locked") is not True:
        raise ValueError("Observed T selection must remain locked")
    with relative(public, SCOPE).open(encoding="utf-8", newline="") as stream:
        scope = selected_scope(list(csv.DictReader(stream)))
    upstream_pins, conditions = {}, []
    for row in scope:
        path = "runs/" + row["initial_registered_ID"] + "/SOURCE_CONTRACT.json"
        value = read(relative(public, path))
        if value.get("model_sha") != base["foundation_tensor_sha256"]:
            raise ValueError("Original recipe foundation identity differs")
        c = condition(row, value, base["task_contracts"][row["dataset"]], base["source39"])
        c.update(upstream_contract_relative_path=path, upstream_contract_sha256=sha(public / path))
        conditions.append(c); upstream_pins[path] = c["upstream_contract_sha256"]
    plan = {key: copy.deepcopy(base[key]) for key in ("source39", "immutable_public", "task_contracts",
        "model_asset_relative_path", "model_revision", "foundation_tensor_sha256",
        "instance_norm_source_relative_path", "instance_norm_source_sha256")}
    plan.update(schema="POST_T_CPU_CORE_REMAINING29_v1", status="DRAFT_NOT_SEALED", stage="POST_T_V_ONLY_DEVELOPMENT",
        output_public_relative_root=NAME, output_cache_relative_root=NAME,
        scope_input_relative_path=SCOPE, scope_input_sha256=SCOPE_SHA, upstream_recipe_metadata=upstream_pins,
        runner_sha256=sha(Path(__file__).with_name("core_remaining_runner.py")), proposer_sha256=sha(__file__),
        dependencies={name: sha(relative(public, name)) for name in DEPENDENCIES}, conditions=conditions,
        postT_registry_relative_path="REGISTRY.csv", postT_registry_sha256=hashlib.sha256(registry_bytes(conditions, plan["task_contracts"])).hexdigest(),
        CPU_threads=2, max_RSS_bytes=3*1024**3, maximum_concurrent_workers=2, total_owned_RSS_limit_bytes=6*1024**3,
        storage_limit_bytes=30*1024**3, admission_reserve_bytes=256*1024**2, deadline_unix=DEADLINE,
        requires_ROOT_seal=True, requires_all_prior_fit_and_cost_and_audit_owners_closed=True,
        no_original_registry_or_selection_updates=True, entire_MASTER_completion_claimed=False,
        scope_note="Conditional core29 subset only; current15 success not presumed; E2/E3/E4/seeds/subsets/M5 remain separate",
        microbatch_note="E/P CPU micro1 vs original4; effective16/permutation/epochs/exposures unchanged; bitwise equivalence not claimed",
        memory_note="CPU RSS3GiB remains unmeasured until the actual probe; no estimated-RSS blocking. checkpoint_inputs=False remains fixed. An explicit future True alternate would recompute F during backward and require separate actual physical counters and provenance.")
    for root in (asset, public.parent):
        norm = relative(asset, plan["instance_norm_source_relative_path"]) if root == asset else root / "temporal_frontend_campaign_20261015_v1/runtime.py"
        if sha(norm) != plan["instance_norm_source_sha256"]:
            raise ValueError("Actual foundation instance-normalization source differs")
    model = relative(asset, plan["model_asset_relative_path"])
    if not model.is_dir() or model.name != plan["model_revision"]:
        raise ValueError("Pinned local foundation revision missing")
    return plan


def metadata(public, cache, asset, plan_sha, run_id=None):
    if not plan_sha:
        raise ValueError("Explicit sealed PLAN SHA required")
    path = relative(Path(public) / NAME, "PLAN.json")
    if sha(path) != plan_sha:
        raise ValueError("PLAN SHA changed")
    plan = read(path)
    if plan.get("status") != "SEALED_BY_ROOT":
        raise ValueError("DRAFT proposal cannot execute; ROOT seal is required")
    expected = build_proposal(public, cache, asset)
    expected["status"] = "SEALED_BY_ROOT"
    comparison = {key: value for key, value in plan.items() if key not in {"ROOT_seal", "ROOT_seal_unix"}}
    if comparison != expected:
        raise ValueError("Sealed scientific/resource proposal differs from exact original metadata")
    registry = relative(Path(public) / NAME, plan["postT_registry_relative_path"])
    if sha(registry) != plan["postT_registry_sha256"] or registry.read_bytes() != registry_bytes(plan["conditions"], plan["task_contracts"]):
        raise ValueError("Separate immutable preregistration differs")
    if run_id is not None:
        selected = [c for c in plan["conditions"] if c["run_id"] == run_id]
        if len(selected) != 1 or selected[0]["preflight_status"] != "SOURCE_METADATA_COMPATIBLE":
            raise ValueError("Explicit supported registered condition required")
    return plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("public-root", "cache-root", "asset-root", "output"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--write-proposal", action="store_true")
    args = parser.parse_args(argv)
    if not args.write_proposal:
        print(json.dumps(dict(status="DRAFT_TOOL_ONLY", experiment_inputs_read=0, writes=0, scientific_imports=False)))
        return 0
    if any(getattr(args, key) is None for key in ("public_root", "cache_root", "asset_root", "output")):
        parser.error("Proposal writing requires explicit roots and a new private output")
    output = plain(args.output)
    if output.exists() or output.is_relative_to(plain(args.public_root)):
        raise ValueError("New private proposal output required; public PLAN is ROOT-owned")
    plan = build_proposal(args.public_root, args.cache_root, args.asset_root)
    output.mkdir(parents=True, exist_ok=False)
    new(output / "DRAFT_PLAN.json", plan)
    with (output / "REGISTRY_PROPOSAL.csv").open("xb") as stream:
        stream.write(registry_bytes(plan["conditions"], plan["task_contracts"]))
    print(json.dumps(dict(status="DRAFT_NOT_SEALED", conditions=len(plan["conditions"]), scientific_execution=0, ROOT_seal_required=True)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
