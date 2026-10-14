"""Guarded launcher for matched-readout synthetic fixture batches.

SOURCE-only until ROOT creates the matched-readout authorization and run manifest.
When executed later, this launcher constructs matched_readout_runtime_v1.ContinuationLimit
before reading fixture files or importing pytest, so process birth, fixture collection, and
pytest runtime are charged to the matched-readout CPU-only budget.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time
from typing import Iterable


# Process-local CPU / no-CUDA defaults. These do not authorize execution by themselves.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for entry in (str(ROOT), str(HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

DEFAULT_TESTS = (
    HERE / "test_matched_readout_math_v1.py",
    HERE / "test_matched_readout_runtime_v1.py",
    HERE / "test_run_matched_readout_v1.py",
)
DEFAULT_OUT = HERE / "validation_v1" / "MATCHED_READOUT_FIXTURES_v1.json"
SAFE_TAG_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789_-")


class LauncherContractError(RuntimeError):
    pass


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(path: Path) -> dict:
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _snapshot(paths: Iterable[Path]) -> list[dict]:
    return [_source_record(path) for path in paths]


def _snapshot_delta(before: list[dict], after: list[dict]) -> list[dict]:
    before_by_path = {record["path"]: record for record in before}
    changes = []
    for record in after:
        prior = before_by_path.get(record["path"])
        if prior is None or prior.get("sha256") != record.get("sha256") or prior.get("bytes") != record.get("bytes"):
            changes.append({"before": prior, "after": record})
    return changes


def _validate_tag(tag: str) -> str:
    if not tag or any(char not in SAFE_TAG_CHARS for char in tag):
        raise LauncherContractError("SAFE_MATCHED_READOUT_FIXTURE_TAG_REQUIRED")
    return tag


def _resolve_output(value: str | None) -> Path:
    path = DEFAULT_OUT if value is None else Path(value)
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if not _is_relative_to(resolved, ROOT):
        raise LauncherContractError("OUTPUT_MUST_BE_UNDER_REPOSITORY_ROOT")
    if ".git" in resolved.parts:
        raise LauncherContractError("OUTPUT_MUST_NOT_BE_UNDER_GIT_INTERNALS")
    return resolved


def _candidate_test_paths(value: str) -> list[Path]:
    raw = Path(value)
    if raw.is_absolute():
        return [raw.resolve()]
    return [(HERE / raw).resolve(), (ROOT / raw).resolve()]


def _resolve_test(value: str) -> Path:
    candidates = _candidate_test_paths(value)
    existing = [path for path in candidates if path.is_file()]
    resolved = existing[0] if existing else candidates[0]
    if not _is_relative_to(resolved, ROOT):
        raise LauncherContractError("TEST_FILE_MUST_BE_UNDER_REPOSITORY_ROOT")
    if ".git" in resolved.parts:
        raise LauncherContractError("TEST_FILE_MUST_NOT_BE_UNDER_GIT_INTERNALS")
    if resolved.suffix != ".py":
        raise LauncherContractError("TEST_FILE_MUST_BE_PYTHON_SOURCE")
    stem = resolved.stem.lower()
    if not (resolved.name.startswith("test_") or "fixture" in stem or "contract" in stem):
        raise LauncherContractError("TEST_FILE_NAME_MUST_BE_A_TEST_OR_FIXTURE_SOURCE")
    if not resolved.is_file():
        raise FileNotFoundError(str(resolved))
    return resolved


def _resolve_tests(values: list[str] | None) -> list[Path]:
    if values:
        tests = [_resolve_test(value) for value in values]
    else:
        tests = [_resolve_test(str(path)) for path in DEFAULT_TESTS]
    seen = set()
    unique = []
    for path in tests:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)
    if not unique:
        raise LauncherContractError("AT_LEAST_ONE_TEST_FILE_REQUIRED")
    return unique


def _error_record(error: BaseException) -> dict:
    return {"type": type(error).__name__, "message": str(error)}


def _describe() -> dict:
    return {
        "status": "SOURCE_ONLY_NOT_EXECUTED",
        "scope": "matched-readout synthetic fixture launcher; runtime authorization required before fixture reads/imports",
        "launcher": _source_record(Path(__file__).resolve()),
        "default_tests": [str(path.relative_to(ROOT)).replace("\\", "/") for path in DEFAULT_TESTS],
        "default_out": str(DEFAULT_OUT.relative_to(ROOT)).replace("\\", "/"),
        "cpu_no_cuda_process_defaults": {
            key: os.environ.get(key)
            for key in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        },
    }


def execute(args: argparse.Namespace) -> int:
    import matched_readout_runtime_v1 as runtime

    tag = _validate_tag(args.tag)
    output = _resolve_output(args.out)
    if output.exists():
        raise FileExistsError(str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    output_ready = True
    limit = None
    result = {
        "schema": "smallk_matched_readout_fixture_launcher_v1",
        "status": "RUNNING",
        "tag": tag,
        "started_unix": time.time(),
        "scope": "synthetic fixture tests only; no pretrained data, scientific assets, model F access, GPU, CUDA, or matched-readout execution",
    }
    try:
        limit = runtime.ContinuationLimit(tag, gpu=False)
        result.update(
            runtime_authorization=getattr(limit, "authorization_record", None),
            runtime_manifest=getattr(limit, "run_manifest_record", None),
            guard_adapter=getattr(limit, "guard_adapter_record", None),
        )
        limit.checkfull()

        test_paths = _resolve_tests(args.tests)
        runtime_source = Path(runtime.__file__).resolve()
        source_paths = [Path(__file__).resolve(), runtime_source, *test_paths]
        source_before = _snapshot(source_paths)
        result["source_before"] = source_before
        result["tests"] = [str(path.relative_to(ROOT)).replace("\\", "/") for path in test_paths]
        result["output"] = str(output.relative_to(ROOT)).replace("\\", "/")
        limit.checkfull()

        import pytest

        limit.checkfull()
        pytest_args = ["-q", *[str(path) for path in test_paths]]
        result["pytest_args"] = ["-q", *[str(path.relative_to(ROOT)).replace("\\", "/") for path in test_paths]]
        code = pytest.main(pytest_args)
        result["pytest_exit_code"] = int(code)
        limit.checkfull()

        source_after = _snapshot(source_paths)
        result["source_after"] = source_after
        source_changes = _snapshot_delta(source_before, source_after)
        result["source_changes"] = source_changes
        if source_changes:
            result["status"] = "FAILED_PRESERVED"
            result["error"] = {"type": "SourceChangedDuringFixtureRun", "message": "fixture or guard source changed during execution"}
        else:
            result["status"] = "COMPLETE" if int(code) == 0 else "FAILED_PRESERVED"
            if int(code) != 0:
                result["error"] = {"type": "PytestFailed", "message": f"pytest exit code {int(code)}"}
        limit.checkfull()
    except Exception as error:
        result["status"] = "BUDGET_HELD_PRESERVED" if isinstance(error, runtime.BudgetHeld) else "FAILED_PRESERVED"
        result["error"] = _error_record(error)
    finally:
        if limit is not None:
            try:
                receipt = limit.close(result.get("status", "FAILED_PRESERVED"), result.get("error"))
            except Exception as close_error:
                receipt = {
                    "terminal_status": "FAILED_PRESERVED",
                    "terminal_error": _error_record(close_error),
                    "scope": "resource close failed after guarded fixture launcher attempted terminal preservation",
                }
                result["status"] = "FAILED_PRESERVED"
                result["error"] = _error_record(close_error)
        else:
            receipt = getattr(runtime, "LAST_CONSTRUCTOR_TERMINAL", None) or {
                "terminal_status": "FAILED_PRESERVED",
                "terminal_error": result.get("error"),
                "scope": "runtime constructor failed before active resource receipt; fixture files were not read",
            }
        result["resource_receipt"] = receipt
        if receipt.get("terminal_status") != "COMPLETE":
            result["status"] = receipt.get("terminal_status", "FAILED_PRESERVED")
            result["resource_terminal_error"] = receipt.get("terminal_error")
        result["finished_unix"] = time.time()
        if output_ready:
            runtime.write_new(output, result)
    print(json.dumps({"status": result["status"], "out": str(output.relative_to(ROOT)).replace("\\", "/")}, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "COMPLETE" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="construct matched-readout runtime and execute the requested synthetic fixture batch")
    parser.add_argument("--describe", action="store_true", help="print source-only launcher metadata without constructing runtime or reading fixtures")
    parser.add_argument("--tests", nargs="+", help="explicit repository-local pytest/fixture source files; default is the matched-readout fixture set")
    parser.add_argument("--tag", default="matched_readout_fixtures_v1", help="safe runtime resource tag")
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)).replace("\\", "/"), help="fresh repository-local RESULT JSON path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.describe or not args.execute:
        print(json.dumps(_describe(), ensure_ascii=False, indent=2), flush=True)
        return 0
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
