"""Activate matched-readout fixture resources and launch the initial synthetic fixtures.

SOURCE-only until ROOT creates matched_readout_authorization_v1.json and explicitly runs
this script. The activator does not create authorization. At execution time it verifies the
fresh human approval, writes one fixed matched_readout_run_manifest_v1.json from the
current process birth time, then calls newrun_matched_readout_fixtures_v1.execute() in the
same process so bootstrap, manifest work, fixture collection, and pytest are charged by
matched_readout_runtime_v1.ContinuationLimit.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import os
import sys


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

HUMAN_RESPONSE_IDENTIFIER = "call_eef6ce160feb4a6db164c356abaf4f2c:0"
HUMAN_RESPONSE_TEXT = "추가 범위·자원 승인"
PHASE = "matched_readout_information_v1"
DEFAULT_TAG = "matched_readout_initial_fixtures_v1"
DEFAULT_OUT = HERE / "validation_v1" / "MATCHED_READOUT_FIXTURES_v1.json"
STAGE_RUN_MANIFEST = HERE / "stage_separation_run_manifest_v1.json"
CONTINUATION_RUN_MANIFEST = HERE / "continuation_run_manifest_v1.json"
RESOURCE_DIRS_TO_PROTECT = (HERE / "continuation_resources_v1", HERE / "stage_separation_resources_v1")


class ActivationContractError(RuntimeError):
    pass


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def _error(message: str) -> ActivationContractError:
    return ActivationContractError(message)


def _validate_authorization(runtime) -> dict:
    auth = runtime.read(runtime.AUTHORIZATION)
    if (
        auth.get("schema") != "smallk_matched_readout_authorization_v1"
        or auth.get("status") != "APPROVED"
        or auth.get("origin") != "human_user_response"
        or auth.get("response_identifier") != HUMAN_RESPONSE_IDENTIFIER
        or auth.get("user_response") != HUMAN_RESPONSE_TEXT
        or auth.get("phase") != PHASE
        or auth.get("approved_plan_sha256") != runtime.sha(runtime.APPROVED_PLAN)
        or auth.get("method_lock_sha256") != runtime.sha(runtime.METHOD_LOCK)
        or auth.get("runtime_source_sha256") != runtime.sha(runtime.CORE_RUNTIME)
        or auth.get("limits") != runtime.MAX_LIMITS
        or auth.get("gpu_allowed") is not False
        or auth.get("cuda_allowed") is not False
        or auth.get("F_access_allowed") is not False
    ):
        raise PermissionError("ACTUAL_MATCHED_READOUT_HUMAN_APPROVAL_REQUIRED")
    return auth


def _resource_receipt_record(runtime, path: Path) -> dict:
    receipt = runtime.read(path)
    if receipt.get("status") not in runtime.TERMINAL:
        raise RuntimeError("PRIOR_RECEIPT_NOT_TERMINAL:" + str(path))
    resolved = path.resolve()
    return {"path": str(resolved), "sha256": runtime.sha(resolved)}


def _resolve_record_path(path_value: str) -> Path:
    raw = Path(path_value)
    return raw.resolve() if raw.is_absolute() else (ROOT / raw).resolve()


def _add_protected_record(runtime, protected: dict[str, dict], path_value: str, sha256: str) -> None:
    resolved = _resolve_record_path(path_value)
    record = {"path": str(resolved), "sha256": sha256}
    if runtime.sha(resolved) != sha256:
        raise RuntimeError("PRIOR_RECEIPT_CHANGED_BEFORE_MATCHED_ACTIVATION:" + str(resolved))
    protected[str(resolved)] = record


def _collect_protected_prior_receipts(runtime, stage_manifest: dict) -> list[dict]:
    protected: dict[str, dict] = {}
    for record in stage_manifest.get("protected_prior_receipts", []):
        if not isinstance(record, dict) or not record.get("path") or not record.get("sha256"):
            raise RuntimeError("STAGE_RUN_MANIFEST_PROTECTED_RECEIPT_RECORD_INVALID")
        _add_protected_record(runtime, protected, record["path"], record["sha256"])

    for folder in RESOURCE_DIRS_TO_PROTECT:
        if not folder.is_dir():
            raise FileNotFoundError("PRIOR_RESOURCE_FOLDER_MISSING:" + str(folder))
        for path in sorted(folder.glob("*.json")):
            if path.name == "ACTIVE.json":
                raise RuntimeError("PRIOR_PHASE_ACTIVE_LEASE_PRESENT:" + str(path))
            record = _resource_receipt_record(runtime, path)
            protected[record["path"]] = record

    if not protected:
        raise RuntimeError("NO_PRIOR_RECEIPTS_TO_PROTECT")
    for record in protected.values():
        if runtime.sha(record["path"]) != record["sha256"]:
            raise RuntimeError("PRIOR_RECEIPT_CHANGED_AFTER_PROTECTION_COLLECTION")
    return [protected[key] for key in sorted(protected)]


def _fail_if_existing_outputs(runtime, output: Path) -> None:
    if runtime.RUN_MANIFEST.exists():
        raise FileExistsError("DO_NOT_REPLACE_MATCHED_READOUT_RUN_MANIFEST:" + str(runtime.RUN_MANIFEST))
    if output.exists():
        raise FileExistsError("DO_NOT_REPLACE_MATCHED_READOUT_FIXTURE_RESULT:" + str(output))
    if runtime.ACTIVE.exists():
        raise RuntimeError("MATCHED_READOUT_ACTIVE_LEASE_PRESENT:" + str(runtime.ACTIVE))


def _resolve_output(value: str | None) -> Path:
    path = DEFAULT_OUT if value is None else Path(value)
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise _error("OUTPUT_MUST_BE_UNDER_REPOSITORY_ROOT") from exc
    return resolved


def _write_manifest(runtime, auth: dict, stage_manifest: dict, protected: list[dict], birth: float) -> dict:
    storage_roots = stage_manifest.get("storage_roots")
    if not isinstance(storage_roots, list) or not storage_roots:
        raise RuntimeError("STAGE_RUN_MANIFEST_STORAGE_ROOTS_REQUIRED")
    baseline_storage = runtime.storage_bytes(storage_roots)
    manifest = {
        "schema": "smallk_matched_readout_run_manifest_v1",
        "phase": PHASE,
        "authorization_sha256": runtime.sha(runtime.AUTHORIZATION),
        "runtime_source_sha256": runtime.sha(runtime.CORE_RUNTIME),
        "started_unix": birth,
        "expires_unix": birth + auth["limits"]["campaign_wall_seconds"],
        "baseline_storage_bytes": baseline_storage,
        "storage_roots": storage_roots,
        "protected_prior_receipts": protected,
        "gpu_allowed": False,
        "cuda_allowed": False,
        "F_access_allowed": False,
        "scope": "fresh matched-readout phase approval; this activation only starts the initial fixture batch while preserving previous continuation/stage receipts and clocks",
        "bootstrap_process_birth_wall_in_first_CPU_receipt": True,
        "activation_source": {"path": _relative(Path(__file__)), "sha256": runtime.sha(Path(__file__).resolve())},
        "fixture_launcher_source": {"path": _relative(HERE / "newrun_matched_readout_fixtures_v1.py"), "sha256": runtime.sha(HERE / "newrun_matched_readout_fixtures_v1.py")},
        "source_manifests_preserved": {
            "stage_run_manifest": {"path": _relative(STAGE_RUN_MANIFEST), "sha256": runtime.sha(STAGE_RUN_MANIFEST)},
            "continuation_run_manifest": {"path": _relative(CONTINUATION_RUN_MANIFEST), "sha256": runtime.sha(CONTINUATION_RUN_MANIFEST)} if CONTINUATION_RUN_MANIFEST.exists() else None,
        },
        "human_response_identifier": HUMAN_RESPONSE_IDENTIFIER,
    }
    runtime.write_new(runtime.RUN_MANIFEST, manifest)
    return manifest


def execute(args: argparse.Namespace) -> int:
    import psutil
    import matched_readout_runtime_v1 as runtime

    output = _resolve_output(args.out)
    _fail_if_existing_outputs(runtime, output)
    auth = _validate_authorization(runtime)
    birth = psutil.Process().create_time()
    if birth < auth["approved_unix"]:
        raise PermissionError("PROCESS_STARTED_BEFORE_MATCHED_READOUT_APPROVAL_RECORD")
    if not STAGE_RUN_MANIFEST.is_file():
        raise FileNotFoundError("STAGE_RUN_MANIFEST_REQUIRED:" + str(STAGE_RUN_MANIFEST))
    stage_manifest = runtime.read(STAGE_RUN_MANIFEST)
    protected = _collect_protected_prior_receipts(runtime, stage_manifest)

    import newrun_matched_readout_fixtures_v1 as fixture_launcher

    manifest = _write_manifest(runtime, auth, stage_manifest, protected, birth)
    launcher_args = argparse.Namespace(execute=True, describe=False, tests=args.tests, tag=args.tag, out=str(output))
    code = fixture_launcher.execute(launcher_args)
    print(
        json.dumps(
            {
                "status": "COMPLETE" if code == 0 else "FAILED_PRESERVED",
                "fixed_expiry_unix": manifest["expires_unix"],
                "matched_run_manifest": _relative(runtime.RUN_MANIFEST),
                "fixture_result": _relative(output),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return int(code)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="verify AUTH, write fresh matched run manifest, and execute guarded fixture launcher")
    parser.add_argument("--tag", default=DEFAULT_TAG, help="runtime tag passed to the guarded fixture launcher")
    parser.add_argument("--out", default=str(DEFAULT_OUT.relative_to(ROOT)).replace("\\", "/"), help="fresh fixture RESULT JSON path")
    parser.add_argument("--tests", nargs="+", help="optional explicit repository-local synthetic fixture test files passed through to the launcher")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "SOURCE_ONLY_NOT_EXECUTED",
                    "activation_source": {"path": _relative(Path(__file__)), "sha256": hashlib_sha(Path(__file__))},
                    "default_tag": DEFAULT_TAG,
                    "default_out": str(DEFAULT_OUT.relative_to(ROOT)).replace("\\", "/"),
                    "required_authorization_response_identifier": HUMAN_RESPONSE_IDENTIFIER,
                    "phase": PHASE,
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return 0
    return execute(args)


def hashlib_sha(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
