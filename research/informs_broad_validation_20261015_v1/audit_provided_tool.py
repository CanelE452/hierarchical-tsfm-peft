"""Apply the supplied aggregate auditor to each real public prediction table.

The temporary schema projection is private. Original statistics are copied
without numeric conversion; raw predictions and model checkpoints are not read.
"""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import time

import data_contract as data


def audit(public_root, cache_root):
    public, cache = Path(public_root).resolve(), Path(cache_root).resolve()
    source = public / "plan/audit_public_bundle.py"
    module_spec = importlib.util.spec_from_file_location("provided_public_arithmetic_auditor", source)
    provided = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(provided)
    self_test = provided.self_test()
    stamp = str(time.time_ns())
    output = public / "audits" / ("provided_tool_" + stamp)
    private = cache / "provided_tool_projection" / stamp
    output.mkdir(parents=True, exist_ok=False)
    private.mkdir(parents=True, exist_ok=False)
    records, excluded = [], []
    for path in sorted(public.rglob("PREDICTION_MANIFEST.json")):
        relative = path.relative_to(public).as_posix()
        if "legacy_e0" in path.relative_to(public).parts:
            excluded.append(dict(manifest=relative, reason="Separate legacy B schema and independent legacy evaluator"))
            continue
        manifest = json.loads(path.read_text(encoding="utf-8-sig"))
        if manifest.get("score") is None:
            excluded.append(dict(manifest=relative, reason="Dense private calibration cache has no public score table"))
            continue
        row = dict(manifest["score"])
        if manifest.get("split") == "T_final":
            lock_path = public / "SELECTION_LOCK.json"
            lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.is_file() else {}
            if not lock.get("locked") or manifest["run_id"] not in lock.get("allowed_run_ids", []):
                records.append(dict(manifest=relative, run_id=manifest["run_id"], status="FAIL", error="T score lacks selection authorization"))
                continue
        row.update({key: manifest[key] for key in ("run_id", "split", "cohort")})
        stats = (public / manifest["public_stats_relative_path"]).resolve()
        try:
            if not stats.is_relative_to(public) or not stats.is_file():
                raise ValueError("Statistics escape public root or are missing")
            with TemporaryDirectory(prefix="run_", dir=private) as temporary:
                projection = Path(temporary)
                (projection / "PUBLIC_ERROR_STATS").mkdir()
                with (projection / "SCORES.csv").open("x", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=sorted(provided.SCORE_FIELDS))
                    writer.writeheader()
                    writer.writerow({key: row[key] for key in writer.fieldnames})
                copied = projection / "PUBLIC_ERROR_STATS" / ("stats.csv.gz" if stats.suffix == ".gz" else "stats.csv")
                shutil.copyfile(stats, copied)
                if data.sha256_file(copied) != data.sha256_file(stats):
                    raise ValueError("Projection changed the original public statistics bytes")
                result = provided.audit(projection)
            records.append(dict(manifest=relative, manifest_sha256=data.sha256_file(path),
                                statistics=stats.relative_to(public).as_posix(), statistics_sha256=data.sha256_file(stats),
                                run_id=manifest["run_id"], split=manifest["split"], cohort=manifest["cohort"],
                                status=result["status"], scores=result["scores"],
                                checked_score_rows=result["checked_score_rows"]))
        except (ValueError, KeyError, OSError, csv.Error, AssertionError) as error:
            records.append(dict(manifest=relative, run_id=manifest.get("run_id"), status="FAIL", error_type=type(error).__name__,
                                error=str(error).replace(str(public), "<public-root>").replace(str(cache), "<private-cache>")))
    failed = sum(row["status"] != "PASS" for row in records)
    receipt = dict(status="FAIL" if failed else ("PASS_AVAILABLE_PUBLIC_STATISTICS" if records else "NO_PUBLIC_STATISTICS"),
                   source_sha256=data.sha256_file(source), wrapper_source_sha256=data.sha256_file(__file__),
                   supplied_tool_self_test=self_test, checked_manifests=len(records), failed_manifests=failed,
                   records=records, excluded=excluded, projection="Per-manifest private copy, byte-identical SSE/SAE/count; temporary copies removed",
                   prediction_array_rescoring=False, checkpoint_replay=False, training_rerun=False, timer_reexecution=False,
                   registry_execution_audit=False, scientific_success=False, created_unix=time.time())
    with (output / "PROVIDED_TOOL_AUDIT.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False, allow_nan=False)
    return output, receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-root", required=True)
    parser.add_argument("--cache-root", required=True)
    args = parser.parse_args()
    folder, result = audit(args.public_root, args.cache_root)
    print(json.dumps(dict(output=folder.relative_to(Path(args.public_root).resolve()).as_posix(),
                          status=result["status"], checked_manifests=result["checked_manifests"],
                          failed_manifests=result["failed_manifests"])))
