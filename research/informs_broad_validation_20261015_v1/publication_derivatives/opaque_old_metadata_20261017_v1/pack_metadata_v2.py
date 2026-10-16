"""PREPARED ONLY: exact bytes of three already-public metadata files.

The default CLI performs no reads or writes. ROOT must separately verify current
process closure, review this source and seal a new plan before actual execution.
No original JSON/UTF-8/numeric payload is interpreted, including during restore.
"""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import re
import time

CAMPAIGN = "research/informs_broad_validation_20261015_v1"
NAMESPACE = "publication_derivatives/opaque_old_metadata_20261017_v1"
DEADLINE = 1792222550
INVENTORY_SHA = "e91e00c1622a05f8648ba6b998b6919532aa66c42ccb49454f3cfc5a002a068d"
RESTORER_SHA = "e88b5cf8d37a667ba60b1666982924be7e1aa796d86101ab64a2a7834b59ab5b"
NUMERIC_SHA = "660a5c9bf56526a3b3bd073aa8b312e2719aeb705b8704868ba8a4595d26785f"
GUARD_HELPER_SHA = "82a837ea0cfee8b52765f4ac1c7325d564e886c9e94857bd0f957e4a40c75f8b"
FIT_SPEC_SHA = "3faf463d4c89269a47d20e83929d76b5ae7a23ae6c7766377aa2e92e9d765390"
MAX_FILE = 20 * 1024**2
MAX_BUNDLE = 250 * 1024**2
STORAGE_LIMIT = 30 * 1024**3
STORAGE_RESERVE = 256 * 1024**2
ALLOCATION_BOUND = 32 * 1024**2
RUN_IDS = (
    "postT_cpu_beijing_E4_INDEPENDENT_B0_k6_s92601_v1",
    "postT_cpu_beijing_E1_B2_k11_s92601_required_v1",
)
ORIGINALS = (
    (CAMPAIGN + "/RESUME_QUEUE.json", 2084960, "ad323d434e76346cd6599187b146e4faae771e41fea775c64d0124210142db8e"),
    (CAMPAIGN + "/diagnostics/publication_multik_lossless_1792094047117000000/bundle_1792096101396562900/files/BATCH_LAUNCHER_RECEIPT.json", 267714, "000c71479d54e0108f3e5974a3c246a7c18e0ca20a53bd8908202f2c2d9f76c7"),
    (CAMPAIGN + "/diagnostics/publication_multik_fresh_lossless_20261016_v1/bundle_20261016_actual_01/files/BATCH_LAUNCHER_RECEIPT.json", 552925, "915a9682adc17184759c43eab0d9fd130f0edc360ac7f4cf3e12f352a7c1bedd"),
)
GZIP_NAMES = tuple(f"files/item_{index:04d}.opaque.json.gz" for index in range(3))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def safe(name):
    if (not isinstance(name, str) or not name or "\\" in name or PurePosixPath(name).is_absolute()
            or any(part in {"", ".", ".."} or ":" in part for part in name.split("/"))):
        raise ValueError("Canonical relative path required")
    return name


def plain(path):
    path = Path(path).absolute()
    if any(p.is_symlink() or getattr(p, "is_junction", lambda: False)() for p in (path, *path.parents)):
        raise ValueError("Symlink or junction rejected")
    return path.resolve()


def inside(root, name):
    root = plain(root)
    path = plain(root / safe(name))
    if not path.is_relative_to(root):
        raise ValueError("Escaped input path")
    return path


def bound(path, expected, size_limit=MAX_FILE):
    path = plain(path)
    if not re.fullmatch(r"[0-9a-f]{64}", expected or "") or not path.is_file() or path.stat().st_size > size_limit:
        raise ValueError("Bounded file and exact SHA required")
    raw = path.read_bytes()
    if digest(raw) != expected:
        raise ValueError("Input byte SHA differs")
    return raw


def read_descriptor(row):
    return json.loads(bound(row["path"], row["sha256"]))


def json_new(path, value):
    with plain(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def inventory_rows(raw):
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"), newline="")))
    names = [safe(row["file"]) for row in rows]
    if not rows or set(rows[0]) != {"file", "bytes", "sha256"} or len(names) != len(set(names)):
        raise ValueError("Exact selected inventory schema required")
    indexed = {row["file"]: row for row in rows}
    for name, length, sha in ORIGINALS:
        if name not in indexed or (int(indexed[name]["bytes"]), indexed[name]["sha256"]) != (length, sha):
            raise ValueError("One of exact three original bindings absent from old selected inventory")
    return rows


def validate_manifest(value):
    if set(value) != {"format", "items"} or value["format"] != "exact_auxiliary_gzip_v1" or len(value["items"]) != 3:
        raise ValueError("Exact existing manifest format and three members required")
    for item, original, gzip_name in zip(value["items"], ORIGINALS, GZIP_NAMES):
        if set(item) != {"original_file", "original_bytes", "original_sha256", "gzip_file", "gzip_bytes", "gzip_sha256"}:
            raise ValueError("Unexpected manifest fields")
        if (item["original_file"], item["original_bytes"], item["original_sha256"]) != original or item["gzip_file"] != gzip_name:
            raise ValueError("Manifest outside exact three bindings or namespace")
        if type(item["gzip_bytes"]) is not int or not 0 < item["gzip_bytes"] <= MAX_FILE or not re.fullmatch(r"[0-9a-f]{64}", item["gzip_sha256"]):
            raise ValueError("Invalid gzip byte binding")
    return value


def deterministic_gzip(raw):
    stream = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=stream, mtime=0) as target:
        target.write(raw)
    return stream.getvalue()


def load_restorer(path):
    bound(path, RESTORER_SHA)
    spec = importlib.util.spec_from_file_location("opaque_metadata_frozen_restorer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def closed_guard(binding, now, expected_workers, exact_spec_sha=None):
    spec = read_descriptor(binding["spec"])
    guard = read_descriptor(binding["guard"])
    closure = read_descriptor(binding["closure"])
    if exact_spec_sha is not None and binding["spec"]["sha256"] != exact_spec_sha:
        raise ValueError("Exact actual fit dispatch SPEC required")
    if (guard.get("schema") != "CPU_GROUP_SUPERVISOR_RESULT_v1"
            or guard.get("status") not in {"COMPLETE", "STOPPED_RESOURCE_OR_WORKER_FAILURE"}
            or guard.get("helper_sha256") != GUARD_HELPER_SHA or guard.get("helper_unchanged") is not True
            or guard.get("spec_file_sha256") != binding["spec"]["sha256"]
            or guard.get("supervisor_failure_class") is not None or guard.get("remaining_owned_processes") != []
            or len(guard.get("processes", [])) != expected_workers
            or spec.get("schema") != "CPU_GROUP_SUPERVISOR_v1" or spec.get("deadline_unix") != DEADLINE
            or len(spec.get("processes", [])) != expected_workers
            or {p.get("id") for p in spec["processes"]} != {p.get("id") for p in guard["processes"]}):
        raise ValueError("Actual exact helper/SPEC/terminal guard binding required")
    identities = []
    for worker in guard["processes"]:
        if (worker.get("returncode") is None or worker.get("end_unix") is None
                or worker.get("observed_tree_join_confirmed") is not True
                or worker.get("process_tree_errors") != [] or worker.get("remaining_owned_processes") != []
                or not worker.get("owned_processes") or not isinstance(worker.get("root"), dict)):
            raise ValueError("Every actually executed worker must have strong observed-tree closure")
        root = worker["root"]
        if sum(row.get("pid") == root.get("pid") and row.get("create_time") == root.get("create_time") for row in worker["owned_processes"]) != 1:
            raise ValueError("Root PID/creation-time must occur once in actual owned tree")
        for identity in worker["owned_processes"]:
            if type(identity.get("pid")) is not int or not isinstance(identity.get("create_time"), (int, float)):
                raise ValueError("Actual PID and creation-time identity required")
            identities.append(dict(pid=identity["pid"], create_time=identity["create_time"]))
    expected = sorted(identities, key=lambda row: (row["pid"], row["create_time"]))
    if (closure.get("schema") != "ROOT_OPAQUE_METADATA_OWNER_ABSENCE_v2"
            or closure.get("status") != "VERIFIED_STRONG_OBSERVED_TREE_CLOSURE_AND_CURRENT_ABSENCE"
            or closure.get("guard_sha256") != binding["guard"]["sha256"]
            or closure.get("spec_sha256") != binding["spec"]["sha256"]
            or closure.get("actual_PID_creation_time_check") is not True
            or closure.get("remaining_exact_owners") != [] or closure.get("errors") != []
            or not 0 <= now - closure.get("checked_unix", -1) <= 120
            or closure.get("absent_identities") != expected):
        raise ValueError("Fresh ROOT actual PID/creation-time absence proof required")
    return guard


def validate_quiet(value, now):
    if (value.get("schema") != "ROOT_OPAQUE_METADATA_QUIET_ADMISSION_v2" or value.get("status") != "ADMITTED"
            or value.get("actual_current_process_inspection") is not True
            or value.get("actual_current_campaign_CIM_empty") is not True
            or value.get("remaining_campaign_model_cost_pack_processes") != []
            or value.get("errors") != [] or value.get("worker_deadline_unix") != DEADLINE
            or value.get("cpu_threads") != 2 or value.get("RSS_limit_bytes") != 1024**3
            or value.get("storage_limit_bytes") != STORAGE_LIMIT or value.get("storage_reserve_bytes") != STORAGE_RESERVE
            or not 0 <= now - value.get("checked_unix", -1) <= 120):
        raise ValueError("Fresh actual quiet/resource admission required; this helper never queries processes")
    used = value.get("storage_used_bytes")
    if type(used) is not int or used < 0 or used + STORAGE_RESERVE + ALLOCATION_BOUND > STORAGE_LIMIT:
        raise ValueError("Original storage cap, reserve and conservative allocation bound required")
    fit = read_descriptor(value["fit_dispatch_result"])
    if (fit.get("status") not in {"COMPLETE_TWO_FITS_AND_FULL_V", "PARTIAL_TWO_FITS_PRESERVED"}
            or fit.get("error") is not None or fit.get("pins_before_after_unchanged") is not True
            or fit.get("strong_observed_tree_closure_verified") is not True
            or fit.get("changed_or_unreadable_pinned_paths") != [] or fit.get("source39_immutable9_count") != 48
            or fit.get("original_worker_deadline") != DEADLINE or len(fit.get("conditions", [])) != 2
            or {row.get("run_id") for row in fit["conditions"]} != set(RUN_IDS)):
        raise ValueError("Actual terminal two-fit dispatch with source preservation required")
    closed_guard(value["fit_guard_binding"], now, 2, FIT_SPEC_SHA)
    post = read_descriptor(value["post_result"])
    if (post.get("status") not in {"COMPLETE_TWO_VALIDATIONS_AND_FOUR_COSTS", "PARTIAL_VALIDATIONS_COSTS_PRESERVED"}
            or post.get("error") is not None or post.get("pins_before_after_unchanged") is not True
            or post.get("changed_or_unreadable_pinned_paths") != [] or post.get("original_worker_deadline") != DEADLINE
            or post.get("fit_dispatch_result_sha256") != value["fit_dispatch_result"]["sha256"]):
        raise ValueError("Actual terminal post result bound to current exact fit dispatch required")
    steps = post.get("steps", [])
    bindings = value.get("executed_post_guard_bindings", [])
    if (len({row.get("id") for row in steps}) != len(steps)
            or len({row.get("id") for row in bindings}) != len(bindings)
            or {row.get("id") for row in steps} != {row.get("id") for row in bindings}):
        raise ValueError("Exactly all actually executed post guards required; never invent skipped guards")
    by_id = {row["id"]: row for row in bindings}
    for row in steps:
        binding = by_id[row["id"]]
        if (row.get("run_id") not in RUN_IDS or row.get("strong_observed_tree_closure_verified") is not True
                or row.get("guard_sha256") != binding["guard"]["sha256"] or row.get("spec_sha256") != binding["spec"]["sha256"]):
            raise ValueError("Executed post step and actual guard/SPEC differ")
        closed_guard(binding, now, 1)
    if any(row.get("run_id") not in RUN_IDS for row in post.get("skipped", [])):
        raise ValueError("Only exact current routes may be preserved as skipped")
    science = dict(fit_dispatch_status=fit["status"], fit_routes=[
        {key: row.get(key) for key in ("run_id", "outer_status", "worker_science_complete", "budget_partial_preserved")}
        for row in fit["conditions"]], post_status=post["status"], post_steps=[
        {key: row.get(key) for key in ("id", "run_id", "mode", "batch", "success", "outer_status")}
        for row in steps], skipped=[{key: row.get(key) for key in ("run_id", "reason", "status", "budget_partial")}
        for row in post.get("skipped", [])])
    return dict(value, scientific_terminal_statuses=science)


def guarded_sources(repo, plan):
    groups = (plan["source39"], plan["immutable9"])
    if tuple(map(len, groups)) != (39, 9) or len(set(groups[0]) | set(groups[1])) != 48:
        raise ValueError("Exact disjoint source39 and immutable9 required")
    result = {}
    for group in groups:
        for name, sha in group.items():
            if not name.startswith(CAMPAIGN + "/"):
                raise ValueError("Source39/immutable9 keys require canonical repository CAMPAIGN prefix")
            raw = bound(inside(repo, name), sha)
            result[name] = dict(bytes=len(raw), sha256=digest(raw))
    return result


def restore_two(helper, manifest_path, items, cache_root):
    cache_root = plain(cache_root)
    cache_root.mkdir(parents=True, exist_ok=False)
    restored = []
    for label in ("restore_a", "restore_b"):
        destination = cache_root / label
        destination.mkdir()
        first = helper.restore(manifest_path, destination)
        repeat = helper.restore(manifest_path, destination)
        if first != dict(status="PASS", restored=3, already_matching=0) or repeat != dict(status="PASS", restored=0, already_matching=3):
            raise ValueError("Exact fresh and repeated restore counts differ")
        files = []
        for item in items:
            raw = inside(destination, item["original_file"]).read_bytes()
            if (len(raw), digest(raw)) != (item["original_bytes"], item["original_sha256"]):
                raise ValueError("Restored original byte SHA differs")
            files.append(dict(file=item["original_file"], bytes=len(raw), sha256=digest(raw)))
        restored.append(dict(destination_label=label, first=first, repeated=repeat, files=files))
    refuse = cache_root / "refuse_probe"
    refuse.mkdir()
    mismatch = inside(refuse, items[0]["original_file"])
    mismatch.parent.mkdir(parents=True)
    with mismatch.open("xb") as stream:
        stream.write(b"existing-different-byte-sentinel")
    sentinel = mismatch.read_bytes()
    try:
        helper.restore(manifest_path, refuse)
    except ValueError as error:
        if "Existing destination differs" not in str(error):
            raise
    else:
        raise ValueError("Existing-byte mismatch was not refused")
    if mismatch.read_bytes() != sentinel or any(inside(refuse, item["original_file"]).exists() for item in items[1:]):
        raise ValueError("Mismatch refusal changed a destination")
    return restored


def execute(repo, cache_root, plan_path, plan_sha):
    now = time.time()
    if now >= DEADLINE:
        raise ValueError("Original worker deadline reached")
    plan_raw = bound(plan_path, plan_sha)
    plan = json.loads(plan_raw)
    if (plan.get("schema") != "OPAQUE_OLD_METADATA_PLAN_v2" or plan.get("status") != "ROOT_SEALED"
            or plan.get("ROOT_admitted") is not True or plan.get("worker_deadline_unix") != DEADLINE
            or plan.get("runner_sha256") != digest(Path(__file__).read_bytes())
            or plan.get("inspector_sha256") != digest(Path(__file__).with_name("inspect_opaque_metadata_v2.py").read_bytes())):
        raise ValueError("Future ROOT seal and exact reviewed sources required")
    repo = plain(repo)
    output = inside(repo, CAMPAIGN + "/" + NAMESPACE)
    cache_boundary = plain(plan["cache_boundary"])
    if (plain(__file__).parent != output or not plain(cache_root).is_relative_to(cache_boundary)
            or plain(cache_root).is_relative_to(repo) or cache_boundary.is_relative_to(repo)
            or cache_boundary.name != "informs_broad_validation_20261015_v1" or cache_boundary.parent.name != ".cache"):
        raise ValueError("Exact owned output and ROOT private cache boundary required")
    inventory = bound(plan["old_inventory_path"], INVENTORY_SHA)
    inventory_rows(inventory)
    admission = validate_quiet(read_descriptor(plan["quiet_admission"]), now)
    before = guarded_sources(repo, plan)
    bound(output / "restore_originals.py", RESTORER_SHA)
    bound(inside(repo, CAMPAIGN + "/publication_check_numeric.py"), NUMERIC_SHA)
    outputs = [output / "files", output / "GZIP_MANIFEST.json", output / "PROOF.json"]
    if any(path.exists() for path in outputs) or plain(cache_root).exists():
        raise FileExistsError("Fresh pack outputs and two fresh private restore destinations required")
    original_before = {}
    for name, length, sha in ORIGINALS:
        raw = bound(inside(repo, name), sha)
        if len(raw) != length:
            raise ValueError("Original byte size differs")
        original_before[name] = dict(bytes=len(raw), sha256=digest(raw))
    (output / "files").mkdir()
    items = []
    for original, gzip_name in zip(ORIGINALS, GZIP_NAMES):
        if time.time() >= DEADLINE:
            raise ValueError("Original worker deadline reached; preserve partial outputs")
        name, length, sha = original
        raw = bound(inside(repo, name), sha)
        blob = deterministic_gzip(raw)
        if blob != deterministic_gzip(raw) or len(blob) > MAX_FILE:
            raise ValueError("Deterministic gzip or file bound failed")
        with inside(output, gzip_name).open("xb") as stream:
            stream.write(blob)
        items.append(dict(original_file=name, original_bytes=length, original_sha256=sha,
                          gzip_file=gzip_name, gzip_bytes=len(blob), gzip_sha256=digest(blob)))
    manifest = validate_manifest(dict(format="exact_auxiliary_gzip_v1", items=items))
    json_new(output / "GZIP_MANIFEST.json", manifest)
    restored = restore_two(load_restorer(output / "restore_originals.py"), output / "GZIP_MANIFEST.json", items, cache_root)
    original_after = {name: dict(bytes=len(raw), sha256=digest(raw)) for name, _, sha in ORIGINALS for raw in [bound(inside(repo, name), sha)]}
    if original_after != original_before or guarded_sources(repo, plan) != before or bound(plan_path, plan_sha) != plan_raw:
        raise ValueError("Original, source48 or plan changed; preserve partial outputs")
    bound(plan["old_inventory_path"], INVENTORY_SHA)
    validate_quiet(read_descriptor(plan["quiet_admission"]), time.time())
    bound(output / "restore_originals.py", RESTORER_SHA)
    bound(inside(repo, CAMPAIGN + "/publication_check_numeric.py"), NUMERIC_SHA)
    if time.time() >= DEADLINE:
        raise ValueError("Deadline reached before proof; partial output is not success")
    proof = dict(schema="OPAQUE_OLD_METADATA_BYTE_PROOF_v2", status="PASS_EXACT_THREE_ALREADY_PUBLIC_METADATA_BYTES",
        plan_sha256=plan_sha, old_inventory_sha256=INVENTORY_SHA, manifest_sha256=digest((output / "GZIP_MANIFEST.json").read_bytes()),
        existing_restorer_sha256=RESTORER_SHA, inspector_sha256=plan["inspector_sha256"], runner_sha256=plan["runner_sha256"],
        quiet_admission_sha256=plan["quiet_admission"]["sha256"], original_before=original_before, original_after=original_after,
        source48_before=before, source48_after=before, fresh_restores=restored, mismatch_refused_without_write=True,
        original_files_modified=False, original_payload_JSON_UTF8_or_numeric_parse=False, new_T_numeric_parse=0,
        model_Task_NPZ_GPU_Git_install_or_process_query_operations=0, worker_deadline_unix=DEADLINE,
        CPU2_RSS1GiB_external_supervisor_admission=admission["status"], whole_selected_cap_index_tree_blob_verification="REQUIRED_SEPARATELY_BY_ROOT",
        resource_terminal_admission_independent_of_scientific_success=True,
        scientific_terminal_statuses=admission["scientific_terminal_statuses"],
        original_bytes=sum(x[1] for x in ORIGINALS), gzip_bytes=sum(x["gzip_bytes"] for x in items), completed_unix=time.time())
    json_new(output / "PROOF.json", proof)
    return proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--root-triggered", action="store_true")
    parser.add_argument("--repo-root")
    parser.add_argument("--private-restore-root")
    parser.add_argument("--plan")
    parser.add_argument("--plan-sha")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="PREP_ONLY_DEFAULT_ZERO_IO", input_reads=0, files_written=0, actual_pack_restore_or_archive_operations=0)))
        return 0
    if not args.root_triggered or not all((args.repo_root, args.private_restore_root, args.plan, args.plan_sha)):
        parser.error("ROOT trigger and future sealed plan/SHA/explicit roots required")
    proof = execute(args.repo_root, args.private_restore_root, args.plan, args.plan_sha)
    print(json.dumps(dict(status=proof["status"], original_files=3, original_bytes=proof["original_bytes"], gzip_bytes=proof["gzip_bytes"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
