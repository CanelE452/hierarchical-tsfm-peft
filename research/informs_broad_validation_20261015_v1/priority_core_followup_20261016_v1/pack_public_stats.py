"""Exact three-file P2 aggregate-statistics writer using the frozen numeric codec.

The default command reads and writes nothing. ROOT must seal a new plan before
--execute. Existing statistics, codec, format and restore reader stay unchanged.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import threading
import time


NAME = "priority_core_followup_20261016_v1"
PUBLIC = "research/informs_broad_validation_20261015_v1"
PREFIX = PUBLIC + "/" + NAME
CODEC_SHA = "c9c42c780306b6e75eb68c939db1c3a9658b3e2680a3d57c15c9971cbefd8743"
CORE_SHA = "0d8c4e4cc9881b70907acf7a154133a64cb582227c1df357521a7668e7fde3fd"
PRIORITY_PLAN = "priority_b2_validation_20261016_v1/PREREG_PLAN.json"
PRIORITY_SHA = "475f983058871ff36b4112b6aa56dc252b17aea47eb6ae1744ff119c2ad5138f"
BASE = "85d2f60d040221f60ffe8df8a96195a62e87365f"
OUTPUT = PUBLIC + "/stats_packs/pack_numeric_priority_followup_20261016_v1"
DEADLINE = 1792222550
RUNS = {"priority_seed2_jena_B0_k6_s92602_v1": ("jena", 43, 48, 21),
        "priority_seed2_jena_B2_k6_s92602_v1": ("jena", 43, 48, 21),
        "priority_pems08_fixedB2_linear_k43_s92601_v1": ("pems08", 148, 12, 170)}
LIMITS = dict(max_wall_seconds=1100, max_RSS_bytes=1073741824,
              storage_limit_bytes=32212254720, admission_reserve_bytes=268435456)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def plain(path):
    path = Path(path).absolute()
    if any(p.is_symlink() or getattr(p, "is_junction", lambda: False)() for p in (path, *path.parents)):
        raise ValueError("Plain nonsymlink and nonjunction paths required")
    return path.resolve()


def inside(root, name):
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or any(p in {"", ".", ".."} for p in name.split("/"))):
        raise ValueError("Canonical repository-relative path required")
    path = plain(root / name)
    if not path.is_relative_to(root):
        raise ValueError("Path escaped the explicit root")
    return path


def check_pins(root, pins):
    for name, digest in pins.items():
        if not re.fullmatch("[0-9a-f]{64}", digest or "") or sha(inside(root, name)) != digest:
            raise ValueError("Frozen source or metadata SHA changed")


def bound(repo, record):
    if set(record) != {"repository_relative_path", "bytes", "sha256"}:
        raise ValueError("Exact path/bytes/SHA binding required")
    size = record["bytes"]
    if type(size) is not int or not 0 < size <= 20 * 1024**2:
        raise ValueError("Positive bounded public file size required")
    path = inside(repo, record["repository_relative_path"])
    if not path.is_file() or path.stat().st_size != size or sha(path) != record["sha256"]:
        raise ValueError("Bound public file changed")
    return path


def require_sealed(plan):
    if plan.get("status") != "SEALED_BY_ROOT" or plan.get("runner_sha256") != sha(__file__):
        raise ValueError("A new ROOT-sealed plan bound to this exact writer is required")


def validate_records(records, prefix):
    fields = {"run_id", "repository_relative_path", "gzip_bytes", "gzip_sha256", "prediction_manifest", "score_csv"}
    if prefix != PREFIX or not isinstance(records, list) or len(records) != 3:
        raise ValueError("Only the exact P2 three-statistic scope is supported")
    seen = set()
    for row in records:
        if not isinstance(row, dict) or set(row) != fields or row["run_id"] not in RUNS or row["run_id"] in seen:
            raise ValueError("Exact three distinct registered runs required")
        seen.add(row["run_id"])
        base = prefix + "/predictions/" + row["run_id"] + "/V_select/"
        if row["repository_relative_path"] != base + "PUBLIC_ERROR_STATS.csv.gz":
            raise ValueError("Foreign namespace, T and parent_full_cpu are excluded")
        if type(row["gzip_bytes"]) is not int or not 0 < row["gzip_bytes"] < 20 * 1024**2 or not re.fullmatch("[0-9a-f]{64}", row["gzip_sha256"] or ""):
            raise ValueError("Exact bounded gzip bytes/SHA required")
        for key, filename in (("prediction_manifest", "PREDICTION_MANIFEST.json"), ("score_csv", "SCORES.csv")):
            record = row[key]
            if (not isinstance(record, dict) or set(record) != {"repository_relative_path", "bytes", "sha256"}
                    or record["repository_relative_path"] != base + filename or type(record["bytes"]) is not int
                    or not 0 < record["bytes"] <= 20 * 1024**2 or not re.fullmatch("[0-9a-f]{64}", record["sha256"] or "")):
                raise ValueError("Exact same-run manifest and score sidecars required")
    return records


def load_codec(public):
    check_pins(public, {"stats_pack_numeric.py": CODEC_SHA, "stats_pack.py": CORE_SHA})
    spec = importlib.util.spec_from_file_location("priority_frozen_numeric_codec", public / "stats_pack_numeric.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_v_metadata(manifest, scores, row):
    dataset, origins, horizon, channels = RUNS[row["run_id"]]
    expected_relative = row["repository_relative_path"][len(PREFIX) + 1:]
    if (manifest.get("run_id") != row["run_id"] or manifest.get("split") != "V_select"
            or manifest.get("cohort") != dataset + ":primary:V_select:stride" + str(horizon)
            or manifest.get("device") != "cpu" or len(scores) != 1
            or manifest["arrays"]["target_normalized"]["shape"] != [origins, horizon, channels]
            or ("public_stats_relative_path" in manifest and manifest["public_stats_relative_path"] != expected_relative)):
        raise ValueError("Exact completed CPU V manifest/statistic identity differs")
    # These P2 manifests omit a statistics-path field; the sealed plan binds it.
    for key, value in manifest["score"].items():
        if (isinstance(value, (int, float)) and float(scores[0][key]) != value) or (not isinstance(value, (int, float)) and scores[0][key] != str(value)):
            raise ValueError("Completed manifest and score CSV differ")


def metadata(repo, public, cache, plan_path, plan_sha):
    repo, public, cache, plan_path = map(plain, (repo, public, cache, plan_path))
    if public != repo / PUBLIC or not cache.is_dir() or not plan_path.is_relative_to(public / NAME):
        raise ValueError("Explicit repository/campaign/cache roots and P2 plan required")
    if not re.fullmatch("[0-9a-f]{64}", plan_sha or "") or sha(plan_path) != plan_sha:
        raise ValueError("Exact sealed plan SHA required")
    plan = read(plan_path)
    require_sealed(plan)
    if (plan.get("schema") != "P2_PUBLIC_STATS_PACK_PLAN_v1" or plan.get("campaign_repository_relative") != PREFIX
            or plan.get("output_repository_relative") != OUTPUT or plan.get("deadline_unix") != DEADLINE
            or plan.get("limits") != LIMITS or plan.get("base_publication_commit") != BASE
            or plan.get("numeric_source_sha256") != CODEC_SHA or plan.get("original_stats_pack_sha256") != CORE_SHA
            or plan.get("original_priority_plan") != dict(relative_path=PRIORITY_PLAN, sha256=PRIORITY_SHA)
            or plan.get("post_T_V_only") is not True or plan.get("new_T_numeric_values_parsed") != 0):
        raise ValueError("Sealed scope, sources, resource or V-only contract changed")
    check_pins(public, {PRIORITY_PLAN: PRIORITY_SHA})
    original = read(public / PRIORITY_PLAN)
    if (plan.get("source39") != original["source39"] or plan.get("immutable_public") != original["immutable_public"]
            or len(plan["source39"]) != 39 or len(plan["immutable_public"]) != 9):
        raise ValueError("All original39+9 pins must be retained")
    pins = {**plan["source39"], **plan["immutable_public"], PRIORITY_PLAN: PRIORITY_SHA,
            "stats_pack_numeric.py": CODEC_SHA, "stats_pack.py": CORE_SHA,
            NAME + "/pack_public_stats.py": plan["runner_sha256"]}
    check_pins(public, pins)
    selected = validate_records(plan["files"], PREFIX)
    for row in selected:
        bound(repo, dict(repository_relative_path=row["repository_relative_path"], bytes=row["gzip_bytes"], sha256=row["gzip_sha256"]))
        manifest = read(bound(repo, row["prediction_manifest"]))
        with bound(repo, row["score_csv"]).open(encoding="utf-8-sig", newline="") as stream:
            scores = list(csv.DictReader(stream))
        validate_v_metadata(manifest, scores, row)
    return dict(repo=repo, public=public, cache=cache, plan_path=plan_path, plan_sha=plan_sha,
                plan=plan, records=selected, pins=pins)


def deadline_check(deadline):
    if time.time() >= deadline:
        raise TimeoutError("Original wall/deadline reached; preserve partial output")


def storage(state, pending):
    used = sum(p.stat().st_size for root in (state["public"], state["cache"]) for p in root.rglob("*") if p.is_file())
    reserve = LIMITS["admission_reserve_bytes"]
    if used + pending + reserve > LIMITS["storage_limit_bytes"] or shutil.disk_usage(state["cache"]).free < pending + reserve:
        raise RuntimeError("Original combined30GiB/256MiB reserve exceeded")
    return used


def write_payloads(codec, repo, records, prefix, output, deadline, check, manifest_name="MANIFEST.json"):
    validate_records(records, prefix)
    output = plain(output)
    output.mkdir(parents=True, exist_ok=False)
    core = codec.load_core()
    archives, encoded, group, assignments, seen = [], [], [], {}, set()
    for row in sorted(records, key=lambda x: x["repository_relative_path"]):
        check(); deadline_check(deadline)
        path = bound(repo, dict(repository_relative_path=row["repository_relative_path"], bytes=row["gzip_bytes"], sha256=row["gzip_sha256"]))
        record, payload = codec.encode_blob(path.read_bytes(), deadline=deadline)
        member = "members/" + core.sha(payload) + (".numeric" if record["storage_mode"] == "numeric_common9" else ".csv.gz")
        if member not in seen:
            if group and sum(len(data) for _, data in group) + len(payload) > codec.MAX_MEMBER:
                assignments.update(codec._emit(group, output, archives, deadline)); group = []
            group.append((member, payload)); seen.add(member)
        encoded.append(dict(record, repository_relative_path=row["repository_relative_path"], member=member))
    if group:
        assignments.update(codec._emit(group, output, archives, deadline))
    for row in encoded:
        row["archive"] = assignments[row["member"]]
    manifest = dict(format=codec.SCHEMA, status="PENDING_RESTORE_VERIFICATION", original_stats_pack_sha256=CORE_SHA,
        numeric_source_sha256=CODEC_SHA, namespace_guard_sha256=sha(__file__), campaign_repository_relative=prefix,
        namespace=NAME, records=encoded, archives=archives, original_total_bytes=sum(r["gzip_bytes"] for r in records),
        packed_total_bytes=sum(a["bytes"] for a in archives), original_files_modified=False,
        python_version=".".join(map(str, sys.version_info[:3])), zlib_compile_version=codec.zlib.ZLIB_VERSION,
        zlib_runtime_version=codec.zlib.ZLIB_RUNTIME_VERSION, original_gzip_SHA_requires_reproducible_zlib=True,
        numeric_values_changed=False, full_rows_or_conditions_dropped=False, protected_T_numeric_values_parsed=0)
    codec._new(output / manifest_name, manifest)
    codec.load_manifest(output / manifest_name)
    return manifest


def verify_restoration(codec, manifest_path, destination, manifest, deadline, check):
    check(); first = codec.restore(manifest_path, destination, deadline=deadline)
    if first["restored"] != 3:
        raise ValueError("All three originals must be independently restored")
    core = codec.load_core()
    for row in manifest["records"]:
        check(); deadline_check(deadline)
        blob = inside(destination, row["repository_relative_path"]).read_bytes()
        raw = core._gunzip(blob)
        if (len(blob) != row["gzip_bytes"] or core.sha(blob) != row["gzip_sha256"]
                or len(raw) != row["csv_bytes"] or core.sha(raw) != row["csv_sha256"]
                or core._csv_schema(raw)[1] != row["row_count"]):
            raise ValueError("Original gzip/CSV bytes, row order or token fidelity differs")
    repeat = codec.restore(manifest_path, destination, deadline=deadline)
    if repeat["restored"] != 0 or repeat["already_matching"] != 3:
        raise ValueError("Repeated restore must only skip three byte-exact originals")
    return dict(first=first, repeat=repeat)


class Guard:
    def __init__(self, deadline, output):
        import psutil
        self.process, self.deadline, self.output = psutil.Process(), deadline, output
        self.stop = threading.Event()

    def check(self):
        deadline_check(self.deadline)
        memory = self.process.memory_info()
        if max(memory.rss, getattr(memory, "peak_wset", 0)) > LIMITS["max_RSS_bytes"]:
            raise MemoryError("Own packing process exceeds1GiB RSS/observed peak")

    def __enter__(self):
        self.check()
        def watch():
            while not self.stop.wait(.1):
                try:
                    self.check()
                except Exception as error:
                    if self.output.is_dir():
                        try:
                            with (self.output / "RESOURCE_STOP.json").open("x", encoding="utf-8") as stream:
                                json.dump(dict(status="RESOURCE_STOP", error_class=type(error).__name__), stream)
                        except OSError:
                            pass
                    os._exit(98)
        self.thread = threading.Thread(target=watch, daemon=True); self.thread.start()
        return self

    def __exit__(self, *error):
        self.stop.set(); self.thread.join(timeout=1)


def execute(repo, public, cache, plan_path, plan_sha, output, wall_seconds=1100):
    if type(wall_seconds) not in (int, float) or not math.isfinite(wall_seconds) or not 0 < wall_seconds <= 1100:
        raise ValueError("Positive wall bound not exceeding1100s required")
    state = metadata(repo, public, cache, plan_path, plan_sha)
    output = plain(output)
    if output != state["repo"] / OUTPUT or output.exists():
        raise FileExistsError("Exact new planned pack output required; no overwrite/resume")
    codec = load_codec(state["public"])
    started = time.time(); deadline = min(started + wall_seconds, DEADLINE)
    source_bytes = sum(row["gzip_bytes"] for row in state["records"])
    storage(state, 64 * 1024**2 + 3 * source_bytes)
    error = post_error = result = None
    try:
        with Guard(deadline, output) as guard:
            manifest = write_payloads(codec, state["repo"], state["records"], PREFIX, output, deadline, guard.check, ".MANIFEST_VERIFY.json")
            with tempfile.TemporaryDirectory(prefix="priority_stats_verify_", dir=state["cache"]) as folder:
                temporary = plain(folder)
                if temporary == state["cache"] or not temporary.is_relative_to(state["cache"]):
                    raise ValueError("Private fresh verification temp must remain below cache")
                restored = verify_restoration(codec, output / ".MANIFEST_VERIFY.json", temporary, manifest, deadline, guard.check)
                plain(temporary)
            if temporary.exists():
                raise ValueError("Private verification temp was not removed")
            guard.check()
            after = metadata(repo, public, cache, plan_path, plan_sha)
            if after["pins"] != state["pins"]:
                raise ValueError("Original sources, immutable metadata or statistics changed")
            storage(state, 0)
            manifest.update(status="PASS", plan_sha256=plan_sha, separate_full_restore_verified=True, repeated_restore_verified=True)
            codec._new(output / "MANIFEST.json", manifest)
            for archive in manifest["archives"]:
                codec.inspect_archive(output / archive["file"]); guard.check()
            result = dict(status="PASS_EXACT_THREE_P2_PUBLIC_STATISTICS", plan_sha256=plan_sha,
                source39_immutable9_preserved=True, separate_restore=restored, original_gzip_files=3,
                original_gzip_bytes=source_bytes, numeric_archive_bytes=manifest["packed_total_bytes"],
                CSV_rows=sum(row["row_count"] for row in manifest["records"]), manifest_sha256=sha(output / "MANIFEST.json"),
                writer_sha256=sha(__file__), numeric_source_sha256=CODEC_SHA, original_stats_pack_sha256=CORE_SHA,
                original_source_and_statistics_unchanged=True, temporary_removed=True,
                row_token_newline_float64_and_original_gzip_bytes_exact=True,
                actual_raw_series_model_Task_array_child_operations=0, new_T_numeric_values_parsed=0,
                whole_publication_budget_or_index_replacement_approved=False, wall_seconds=time.time() - started)
            readme = ("# P2 공개 충분통계3개 무손실 표현\n\nCSV 모든 행/순서/원문 token/개행과 원 gzip byte SHA를 보존한다. 원자료·모델·새 평가가 아니다. 원 로컬 파일은 유지한다.\n\n"
                "[Manifest](MANIFEST.json) · [실제 복원 검산](PROOF.json)\n\n저장소 root에서 복원:\n\n```text\npython " + PUBLIC + "/stats_pack_numeric.py restore --manifest " + OUTPUT + "/MANIFEST.json --destination .\n```\n\n"
                "같은 파일은 SHA검증 후 skip, 다른 기존 파일은 거부한다. frozen reader/FORMAT은 변경하지 않았다. 전체250MiB/게시 검사와 index 표현 대체는 ROOT의 별도 단계다.\n")
            with (output / "README_KO.md").open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(readme)
            names = ["MANIFEST.json", "PROOF.json", "README_KO.md", "ADD_LIST.txt"] + [a["file"] for a in manifest["archives"]]
            with (output / "ADD_LIST.txt").open("x", encoding="utf-8", newline="\n") as stream:
                stream.write("\n".join(sorted(OUTPUT + "/" + name for name in names)) + "\n")
    except Exception as failure:
        error = failure
    finally:
        try:
            metadata(repo, public, cache, plan_path, plan_sha)
        except Exception as failure:
            post_error = failure
        if (error or post_error) and output.is_dir():
            codec._new(output / "FAILED_PRESERVED.json", dict(status="FAILED_PRESERVED",
                error_class=None if error is None else type(error).__name__,
                postflight_error_class=None if post_error is None else type(post_error).__name__, partial_files_preserved=True))
    if error is not None:
        raise error
    if post_error is not None:
        raise post_error
    guard.check()
    result["wall_seconds"] = time.time() - started
    codec._new(output / "PROOF.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true"); action.add_argument("--metadata", action="store_true")
    parser.add_argument("--root-triggered", action="store_true")
    for name in ("repo-root", "public-root", "cache-root", "plan", "plan-sha", "output"):
        parser.add_argument("--" + name)
    parser.add_argument("--wall-seconds", type=float, default=1100)
    args = parser.parse_args(argv)
    if not args.execute and not args.metadata:
        print(json.dumps(dict(status="METADATA_ONLY_NOT_EXECUTED", input_reads=0, files_written=0,
            numpy_imported="numpy" in sys.modules, torch_imported="torch" in sys.modules)))
        return 0
    if not all((args.repo_root, args.public_root, args.cache_root, args.plan, args.plan_sha)):
        parser.error("Explicit roots and ROOT-sealed PLAN/SHA required")
    if args.metadata:
        state = metadata(args.repo_root, args.public_root, args.cache_root, args.plan, args.plan_sha)
        print(json.dumps(dict(status="SEALED_METADATA_VERIFIED_NO_STATISTICS_DECODE", files=3,
            original_gzip_bytes=sum(r["gzip_bytes"] for r in state["records"]), files_written=0)))
        return 0
    if not args.root_triggered or not args.output:
        parser.error("Explicit ROOT trigger and new planned output required")
    result = execute(args.repo_root, args.public_root, args.cache_root, args.plan, args.plan_sha, args.output, args.wall_seconds)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
