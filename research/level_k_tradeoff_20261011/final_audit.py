"""Audit completed receipts and preserve final artifacts without science execution."""

from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import math
import os
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / "final_verification.json"


def read(name):
    return json.loads((HERE / name).read_text(encoding="utf-8-sig"))


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def function_asts(path):
    result = {}
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            result[node.name] = ast.dump(node, include_attributes=False)
        elif isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    result[f"{node.name}.{member.name}"] = ast.dump(member, include_attributes=False)
    return result


def main():
    if OUTPUT.exists():
        raise FileExistsError("Preserve an existing audit before any rerun")
    checks = []

    def check(name, passed, **detail):
        checks.append({"name": name, "passed": bool(passed), **detail})

    result, method, benchmark = (read(n) for n in ("result.json", "method_result.json", "benchmark.json"))
    summary, verify, protocol = (read(n) for n in ("summary.json", "verification.json", "protocol.json"))
    check("science stages complete", result["status"] == method["status"] == "complete")
    check("no inherited baseline failure", "error" not in result and "failed_utc" not in result)
    check("independent verification", verify["status"] == "PASS" and len(verify["checks"]) == 2518
          and verify["failure_count"] == 0 and len(verify["warnings"]) == 2,
          warnings=verify["warnings"])
    check("successful verification copy", sha(HERE / "verification.json") == sha(HERE / "verification.attempt2.json"))
    check("failed verification preserved", read("verification.attempt1.json")["failure_count"] == 3
          and (HERE / "verify.before_static_guard_fix.py").is_file())
    for name, expected in summary["inputs"].items():
        check(f"summary input:{name}", sha(HERE / name) == expected)

    counts = result["execution_counts"]
    check("baseline actual counts", (counts["fit_starts"], counts["completed_fits"], counts["ridge_fits"])
          == (10, 8, 200))
    check("continuation actual counts", counts["continuation_counts"]
          == {"fit_starts": 3, "completed_fits": 3, "ridge_fits": 100})
    native = method["execution_counts"]
    check("native actual counts", (native["fit_starts"], native["completed_fits"], native["ridge_fits"])
          == (4, 4, 40))
    check("fresh successful updates", counts["selected_fit_updates"] + 4 * 2880 == 34560)
    for name in ("k", "continuation", "method"):
        sealed = read(f"source_seal.{name}.json")
        bad = [row["path"] for row in sealed["artifacts"] if sha(resolve(row["path"])) != row["sha256"]]
        check(f"source seal:{name}", not bad, artifact_count=len(sealed["artifacts"]), changed=bad)
    continuation = result["continuation"]
    check("continuation start/end source receipt",
          continuation["source_seal_continuation"] == continuation["source_seal_continuation_after"]
          and sha(resolve(continuation["source_seal_continuation"]["path"]))
          == continuation["source_seal_continuation"]["sha256"])
    for attempt in (1, 2):
        manifest = read(f"failed_attempt{attempt}_preservation.json")
        bad = [row["path"] for row in manifest["archived_artifacts"]
               if sha(resolve(row["path"])) != row["sha256"]]
        check(f"failed archive:{attempt}", not bad,
              artifact_count=len(manifest["archived_artifacts"]), changed=bad)
    for row in read("recovery_source_diff.json"):
        archived = HERE / "failed_attempts" / row["attempt"] / "sources" / row["file"]
        # Archives retain source files at their top level.
        if not archived.is_file():
            archived = HERE / "failed_attempts" / row["attempt"] / row["file"]
        old, new = function_asts(archived), function_asts(HERE / row["file"])
        changed = sorted(k for k in old.keys() & new.keys() if old[k] != new[k])
        added, removed = sorted(new.keys() - old.keys()), sorted(old.keys() - new.keys())
        check(f"I/O-only AST receipt:{row['attempt']}:{row['file']}",
              changed == sorted(row["changed_common_functions"])
              and added == sorted(row["added_functions"]) and removed == sorted(row["removed_functions"]))

    rows = benchmark["rows"]
    check("benchmark completed case count", benchmark["status"] == "complete"
          and benchmark["case_count"] == len(rows) == 52)
    check("unique timing cases", len({(r["dataset"], r["adapter"], r["batch"]) for r in rows}) == 52)
    check("measured timing iterations", all(r["warmups"] == 5 and r["repetitions"] == 20
          and len(r["timings_seconds"]) == 20 for r in rows))
    check("benchmark prediction/input parity", all(r["parity"]["passed"]
          and r["timed_input_parity"]["passed"] for r in rows))
    check("benchmark frozen state", all(r["backbone_state_sha256_before"]
          == r["backbone_state_sha256_after"] for r in rows))
    for dataset, data in summary["datasets"].items():
        raw = next(c for c in data["conditions"] if c["family"] == "RAW_F0")
        for condition in data["conditions"]:
            selected = [r for r in rows if r["dataset"] == dataset
                        and r["adapter"] in condition["adapters"] and r["batch"] == 1]
            mse = sum(r["prediction_entry"]["future_tsfm"]["all"]["mse"] for r in selected) / len(selected)
            mae = sum(r["prediction_entry"]["future_tsfm"]["all"]["mae"] for r in selected) / len(selected)
            actual = condition["metrics"]["all"]
            check(f"summary metric mean:{dataset}:{condition['id']}", math.isclose(mse, actual["mse"], abs_tol=1e-12)
                  and math.isclose(mae, actual["mae"], abs_tol=1e-12))
            for batch in (1, 32):
                selected = [r for r in rows if r["dataset"] == dataset
                            and r["adapter"] in condition["adapters"] and r["batch"] == batch]
                average = lambda key: sum(key(r) for r in selected) / len(selected)
                cost = condition["cost"][str(batch)]
                check(f"summary cost mean:{dataset}:{condition['id']}:B{batch}",
                      math.isclose(cost["median_ms"], average(lambda r: r["median_seconds"] * 1000), abs_tol=1e-10)
                      and math.isclose(cost["allocated_mib"], average(lambda r: r["cuda_peak_allocated_bytes"] / 2**20), abs_tol=1e-10)
                      and math.isclose(cost["sampled_rss_mib"], average(lambda r: r["rss"]["peak_rss_bytes"] / 2**20), abs_tol=1e-10))
            check(f"summary MSE/MAE deltas:{dataset}:{condition['id']}",
                  math.isclose(condition["mse_change_vs_raw_pct"], 100 * (mse / raw["metrics"]["all"]["mse"] - 1), abs_tol=1e-9)
                  and math.isclose(condition["mae_change_vs_raw_pct"], 100 * (mae / raw["metrics"]["all"]["mae"] - 1), abs_tol=1e-9))
    contracts = [value["contracts"]["FORECAST_RIDGE_D8"] for data in result["datasets"].values()
                 for value in data["decoder_contracts"].values()]
    check("20 baseline lambda selections unchanged and interior", len(contracts) == 20
          and all(c["selected_lambda"] == c["initial_grid_selection"]["selected_lambda"]
                  and protocol["ridge_lambdas"][0] < c["selected_lambda"] < protocol["ridge_lambdas"][-1]
                  for c in contracts))
    review = read("report_review.json")
    check("independent report review", review["status"] == "PASS" and not review["actionable_errors"]
          and sha(resolve(review["report"])) == review["report_sha256"])
    for stem in ("accuracy_cost", "batch_cost", "ridge_sensitivity", "readout_generalization"):
        for extension in ("png", "pdf"):
            path = HERE / "figures" / f"{stem}.{extension}"
            check(f"figure exists:{path.name}", path.is_file() and path.stat().st_size > 1000)
    check("execution note exists", (HERE / "EXECUTION_NOTE_KO.md").stat().st_size > 1000)

    parent = read("parent_preservation.before.json")
    bad = [r["path"] for r in parent["artifacts"] if sha(resolve(r["path"])) != r["sha256"]]
    check("preceding artifacts unchanged", len(parent["artifacts"]) == 4662 and not bad, changed=bad)
    history = parent["history_append_only"]
    history_bytes = resolve(history["path"]).read_bytes()
    check("history prefix unchanged and append present", len(history_bytes) > history["bytes"]
          and hashlib.sha256(history_bytes[:history["bytes"]]).hexdigest() == history["prefix_sha256"])
    check("HEAD unchanged", subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() == parent["head"])
    check("tracked unstaged code unchanged", subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode == 0)
    check("tracked staged code unchanged", subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0)
    artifacts = []
    paths = {resolve(history["path"])}
    for base in (HERE, ROOT / ".cache" / HERE.name):
        for folder, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            paths.update(Path(folder) / name for name in files if not name.endswith(".tmp"))
    for path in sorted(paths):
        if path == OUTPUT:
            continue
        artifacts.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path), "bytes": path.stat().st_size})
    failed = [r for r in checks if not r["passed"]]
    receipt = {"status": "FAIL" if failed else "PASS", "checked_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "CPU/file/receipt audit; no training, frozen-F call, GPU benchmark, or TEST evaluation",
               "checks": checks, "check_count": len(checks), "failure_count": len(failed), "failures": failed,
               "independent_checks": 2518, "independent_warnings": verify["warnings"],
               "artifact_count": len(artifacts), "artifacts": artifacts}
    OUTPUT.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Final audit {receipt['status']}: {len(checks)} checks, {len(artifacts)} artifacts, {len(failed)} failures")
    for failure in failed:
        print(failure)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
