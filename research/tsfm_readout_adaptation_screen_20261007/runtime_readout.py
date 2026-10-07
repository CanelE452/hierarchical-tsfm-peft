"""Runtime helpers for the bounded TSFM readout S0/S1 contract.

This module is intentionally small and side-effect free at import time.  It
provides canonical data access, origin sampling, feature-cache I/O and a guarded
S0 calibration ledger.  It does not create caches, run models, touch TEST data,
or start timers until callers enter :class:`Budget`.
"""
from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT / ".cache" / HERE.name
V18 = ROOT / "research" / "level_matched_learned_ed_v18_20261006"

DATASETS = ("robin", "peacock_education", "jena")
SEEDS = (92601, 92602)
PHASE_PERIODS = {"robin": 24, "peacock_education": 24, "jena": 144}
SAMPLE_TRAIN_ORIGINS = 24
BATCH_ORIGINS = 4
HISTORY = 512
HORIZON = 48
FEATURE_HIDDEN = 512

_COUNTER_ALIASES = {
    "prefix": "prefix_calls",
    "prefix_call": "prefix_calls",
    "prefix_calls": "prefix_calls",
    "head": "head_calls",
    "head_call": "head_calls",
    "head_calls": "head_calls",
    "output": "head_calls",
    "output_call": "head_calls",
    "output_calls": "head_calls",
    "update": "technical_updates",
    "updates": "technical_updates",
    "technical_update": "technical_updates",
    "technical_updates": "technical_updates",
    "fit": "scientific_fits",
    "fits": "scientific_fits",
    "scientific_fit": "scientific_fits",
    "scientific_fits": "scientific_fits",
    "scientific_update": "scientific_updates",
    "scientific_updates": "scientific_updates",
    "test": "new_test_predictions",
    "test_prediction": "new_test_predictions",
    "test_predictions": "new_test_predictions",
    "new_test_prediction": "new_test_predictions",
    "new_test_predictions": "new_test_predictions",
}


def resolve(path: str | os.PathLike[str]) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _public_path(value: Any) -> Any:
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        root_s = str(ROOT)
        root_p = ROOT.as_posix()
        value = value.replace(root_s + os.sep, "").replace(root_s + "/", "")
        value = value.replace(root_p + "/", "")
        return value
    if isinstance(value, dict):
        return {k: _public_path(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_public_path(v) for v in value]
    return value


def _require_read_path(path: Path) -> Path:
    path = path.resolve()
    if not _is_within(path, ROOT):
        raise PermissionError(f"read path outside repository root: {path}")
    return path


def _require_write_path(path: Path) -> Path:
    path = path.resolve()
    if not (_is_within(path, HERE) or _is_within(path, CACHE)):
        raise PermissionError(f"writes limited to campaign/cache directories: {path}")
    return path


def read_json(path: str | os.PathLike[str]) -> Any:
    path = _require_read_path(resolve(path))
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: str | os.PathLike[str], value: Any) -> None:
    path = _require_write_path(resolve(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    payload = json.dumps(_public_path(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def digest(path: str | os.PathLike[str]) -> str:
    path = _require_read_path(resolve(path))
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(array: Any) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _import_file(path: Path, name: str):
    path = _require_read_path(path)
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _v18_runtime():
    return _import_file(V18 / "runtime_v18.py", "_tsfm_readout_exact_v18_runtime")


def configure() -> None:
    """Configure deterministic low-thread CPU defaults and disable TF32."""
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


configure_torch = configure


def _dataset_from_data(data: Mapping[str, Any] | str) -> str:
    if isinstance(data, str):
        dataset = data
    else:
        dataset = str(data.get("_dataset", ""))
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset!r}")
    return dataset


def _test_choice_is_sealed() -> bool:
    for name in ("selection_seal.json", "global_choice_seal.json"):
        path = HERE / name
        if not path.exists():
            continue
        seal = read_json(path)
        accepted = (
            seal.get("all_trainval_selection_complete_before_test")
            or seal.get("global_choice_sealed_before_test")
            or seal.get("test_access_authorized")
        )
        if accepted:
            return True
    return False


def load_data(dataset: str, include_test: bool = False) -> dict[str, Any]:
    """Load canonical v18-bound data, blocking TEST until global choice seal."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset!r}")
    if include_test and not _test_choice_is_sealed():
        raise RuntimeError("TEST data access rejected until S1 global choice/selection seal exists")
    data = _v18_runtime().load_data(dataset, include_test=include_test)
    data["_dataset"] = dataset
    return data


def inputs(data: Mapping[str, Any], origins: Any):
    """Return canonical CPU float32 context tensor [B,512,C]."""
    import numpy as np
    import torch

    origins = np.asarray(origins, dtype=np.int64)
    x = np.stack([data["x"][int(o) - HISTORY:int(o)] for o in origins]).astype(np.float32, copy=False)
    return torch.from_numpy(np.ascontiguousarray(x)).to(device="cpu", dtype=torch.float32)


def targets(data: Mapping[str, Any], origins: Any):
    """Return CPU target float32 and observed mask bool tensors, both [B,48,C]."""
    import numpy as np
    import torch

    origins = np.asarray(origins, dtype=np.int64)
    y = np.stack([data["x"][int(o):int(o) + HORIZON] for o in origins]).astype(np.float32, copy=False)
    m = np.stack([data["finite"][int(o):int(o) + HORIZON] for o in origins]).astype(bool, copy=False)
    return (
        torch.from_numpy(np.ascontiguousarray(y)).to(device="cpu", dtype=torch.float32),
        torch.from_numpy(np.ascontiguousarray(m)).to(device="cpu", dtype=torch.bool),
    )


def sample_epoch(data: Mapping[str, Any] | str, seed: int, epoch: int):
    """Literal parent phase-balanced TRAIN sampler for one 512-origin epoch."""
    import numpy as np

    dataset = _dataset_from_data(data)
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}: {seed}")
    if not 1 <= int(epoch) <= 120:
        raise ValueError("sample_epoch is defined for training epochs 1..120")
    if isinstance(data, str):
        data = load_data(dataset, include_test=False)
    train = np.asarray(data["train_origins"], dtype=np.int64)
    period = PHASE_PERIODS[dataset]
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch)]))
    selected = []
    for phase in range(period):
        group = train[train % period == phase]
        count = HISTORY // period + int(phase < HISTORY % period)
        if len(group) < count:
            raise ValueError("TRAIN phase insufficient for approved sampling without replacement")
        selected.append(rng.choice(group, size=count, replace=False))
    result = np.concatenate(selected).astype(np.int64, copy=False)
    rng.shuffle(result)
    return np.ascontiguousarray(result)


def sample_origins(data: Mapping[str, Any] | str):
    """Return (origins, seal) for 24 chronological TRAIN origins plus full VAL."""
    import numpy as np

    dataset = _dataset_from_data(data)
    if isinstance(data, str):
        data = load_data(dataset, include_test=False)
    train = np.asarray(data["train_origins"], dtype=np.int64)
    val = np.asarray(data["val_origins"], dtype=np.int64)
    if len(train) < SAMPLE_TRAIN_ORIGINS:
        raise ValueError("canonical TRAIN has fewer than 24 origins")
    first_train = np.ascontiguousarray(train[:SAMPLE_TRAIN_ORIGINS])
    origins = np.ascontiguousarray(np.concatenate([first_train, val]).astype(np.int64, copy=False))
    seal = {
        "dataset": dataset,
        "kind": "s0_sample_train_first24_plus_full_val",
        "train_count": int(len(first_train)),
        "val_count": int(len(val)),
        "origin_count": int(len(origins)),
        "train_origins_sha256": array_hash(first_train),
        "val_origins_sha256": array_hash(val),
        "origins_sha256": array_hash(origins),
        "order": "canonical chronological TRAIN prefix followed by canonical VAL",
    }
    return origins, seal


def _as_numpy_cpu(value: Any, *, dtype: Any | None = None):
    import numpy as np

    if hasattr(value, "detach"):
        value = value.detach().to("cpu").contiguous().numpy()
    arr = np.ascontiguousarray(value)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def features_to_cpu(features: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize model-worker feature tensors to CPU float32 numpy arrays."""
    import numpy as np

    out: dict[str, Any] = {}
    if "loc_scale" in features and ("loc" not in features or "scale" not in features):
        loc_scale = _as_numpy_cpu(features["loc_scale"], dtype=np.float32)
        if loc_scale.ndim != 3 or loc_scale.shape[-1] != 2:
            raise ValueError("loc_scale must have shape [B,C,2]")
        out["loc"] = np.ascontiguousarray(loc_scale[..., 0:1])
        out["scale"] = np.ascontiguousarray(loc_scale[..., 1:2])
    for key in ("h", "loc", "scale", "point"):
        if key in features:
            out[key] = _as_numpy_cpu(features[key], dtype=np.float32)
    missing = {"h", "loc", "scale", "point"} - set(out)
    if missing:
        raise KeyError(f"missing feature arrays: {sorted(missing)}")
    h, loc, scale, point = out["h"], out["loc"], out["scale"], out["point"]
    if h.ndim != 3 or h.shape[2] != FEATURE_HIDDEN:
        raise ValueError(f"h must have shape [B,C,{FEATURE_HIDDEN}]")
    b, c, _ = h.shape
    if loc.shape != (b, c, 1) or scale.shape != (b, c, 1):
        raise ValueError("loc and scale must have shape [B,C,1] matching h")
    if point.shape != (b, HORIZON, c):
        raise ValueError(f"point must have shape [B,{HORIZON},C] matching h")
    return out


def _features_as_torch(features: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    return {key: torch.from_numpy(value).to(device="cpu", dtype=torch.float32) for key, value in features.items()}


def _feature_path(dataset: str, origins: Any) -> Path:
    import numpy as np

    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset: {dataset!r}")
    origins = np.asarray(origins, dtype=np.int64)
    return CACHE / "features" / dataset / f"features_{array_hash(origins)[:24]}.npz"


def _feature_receipt(path: Path, dataset: str, origins: Any, features: Mapping[str, Any], *, created: bool) -> dict[str, Any]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "dataset": dataset,
        "created": bool(created),
        "bytes": path.stat().st_size,
        "sha256": digest(path),
        "origins_sha256": array_hash(origins),
        "feature_hashes": {key: array_hash(value) for key, value in features.items()},
        "compression": "none_npz_zip_store",
    }


def save_features(dataset: str, origins: Any, features: Mapping[str, Any]) -> dict[str, Any]:
    """Save exact uncompressed feature cache and verify by read-only reload."""
    import numpy as np

    origins = np.ascontiguousarray(np.asarray(origins, dtype=np.int64))
    cpu = features_to_cpu(features)
    path = _require_write_path(_feature_path(dataset, origins))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        loaded_origins, loaded = load_features(path)
        loaded_np = {key: _as_numpy_cpu(value, dtype=np.float32) for key, value in loaded.items()}
        if not np.array_equal(loaded_origins, origins):
            raise RuntimeError("existing feature cache origin mismatch")
        for key, value in cpu.items():
            if key not in loaded_np or array_hash(loaded_np[key]) != array_hash(value):
                raise RuntimeError(f"existing feature cache differs for {key}")
        return _feature_receipt(path, dataset, origins, cpu, created=False)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("xb") as handle:
        np.savez(
            handle,
            origins=origins,
            origin_sha256=np.array(array_hash(origins)),
            dataset=np.array(dataset),
            h=cpu["h"],
            loc=cpu["loc"],
            scale=cpu["scale"],
            point=cpu["point"],
        )
    os.replace(temporary, path)
    loaded_origins, loaded = load_features(path)
    loaded_np = {key: _as_numpy_cpu(value, dtype=np.float32) for key, value in loaded.items()}
    if not np.array_equal(loaded_origins, origins) or array_hash(loaded_origins) != array_hash(origins):
        raise RuntimeError("saved feature cache origin hash mismatch")
    for key, value in cpu.items():
        if array_hash(loaded_np[key]) != array_hash(value):
            raise RuntimeError(f"saved feature cache reload mismatch for {key}")
    return _feature_receipt(path, dataset, origins, cpu, created=True)


def load_features(path: str | os.PathLike[str]):
    """Load an existing feature .npz as (origins int64, CPU torch feature dict)."""
    import numpy as np

    path = _require_read_path(resolve(path))
    if not (_is_within(path, HERE) or _is_within(path, CACHE)):
        raise PermissionError("feature cache loads are limited to campaign/cache directories")
    with np.load(path, allow_pickle=False) as archive:
        origins = np.ascontiguousarray(archive["origins"].astype(np.int64, copy=False))
        expected = str(archive["origin_sha256"].tolist()) if "origin_sha256" in archive.files else None
        if expected is not None and expected != array_hash(origins):
            raise RuntimeError("feature cache origin_sha256 does not match origins")
        features = {key: np.ascontiguousarray(archive[key].astype(np.float32, copy=False)) for key in ("h", "loc", "scale", "point")}
    return origins, _features_as_torch(features)


def batches(indices: Any, batch4: int = BATCH_ORIGINS):
    import numpy as np

    if int(batch4) <= 0 or int(batch4) > BATCH_ORIGINS:
        raise ValueError("batch4 must be in 1..4")
    values = np.asarray(indices)
    for start in range(0, len(values), int(batch4)):
        yield values[start:start + int(batch4)]


def _approved_limits(stage: str = "s0") -> dict[str, Any]:
    authorization = read_json(HERE / "authorization.json")
    if stage == "s0":
        if not bool(authorization.get("s0_calibration_authorized", False)):
            raise RuntimeError("S0 calibration is not authorized in authorization.json")
        limits = authorization.get("approved_limits")
        label = "S0 approved_limits"
        required_caps = (
            "maximum_gpu_job_wall_seconds",
            "maximum_cpu_only_job_wall_seconds",
            "maximum_cpu_process_time_seconds",
            "maximum_elapsed_operation_wall_seconds",
            "maximum_new_disk_bytes",
            "maximum_process_ram_bytes",
            "maximum_gpu_allocated_or_reserved_bytes",
            "maximum_technical_optimizer_updates",
            "maximum_full_prefix_batch_calls",
            "maximum_head_or_output_only_batch_calls",
        )
    elif stage == "s1":
        if not bool(authorization.get("s1_execution_authorized", False)):
            raise RuntimeError("S1 execution is not authorized in authorization.json")
        limits = authorization.get("s1_approved_limits") or authorization.get("approved_s1_limits")
        label = "S1 approved limits"
        required_caps = (
            "maximum_gpu_job_wall_seconds",
            "maximum_cpu_only_job_wall_seconds",
            "maximum_cpu_process_time_seconds",
            "maximum_elapsed_operation_wall_seconds",
            "maximum_new_disk_bytes",
            "maximum_process_ram_bytes",
            "maximum_gpu_allocated_or_reserved_bytes",
            "maximum_technical_optimizer_updates",
            "maximum_full_prefix_batch_calls",
            "maximum_head_or_output_only_batch_calls",
        )
    else:
        raise ValueError("Budget stage must be 's0' or 's1'")
    if not isinstance(limits, dict):
        raise RuntimeError(f"{label} missing in authorization.json")
    missing = [key for key in required_caps if limits.get(key) is None]
    if stage == "s1":
        if limits.get("maximum_scientific_fits", limits.get("maximum_fits")) is None:
            missing.append("maximum_scientific_fits|maximum_fits")
        if limits.get("maximum_scientific_optimizer_updates", limits.get("maximum_scientific_updates")) is None:
            missing.append("maximum_scientific_optimizer_updates|maximum_scientific_updates")
    if missing:
        raise RuntimeError(f"{label} missing required caps: {missing}")
    if stage == "s0":
        if limits.get("scientific_fits") not in (None, 0) or limits.get("selected_test_models") not in (None, 0):
            raise RuntimeError("S0 approval must keep scientific_fits and selected_test_models at 0")
        if bool(limits.get("test_access", False)):
            raise RuntimeError("S0 approval must not allow TEST access")
    return limits


def _limit_seconds(limits: Mapping[str, Any], *names: str, default: float | None = None) -> float | None:
    for name in names:
        if name in limits and limits[name] is not None:
            return float(limits[name])
    return default


def _limit_int(limits: Mapping[str, Any], *names: str, default: int | None = None) -> int | None:
    for name in names:
        if name in limits and limits[name] is not None:
            return int(limits[name])
    return default


def _dir_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                total += item.stat().st_size
        except OSError:
            pass
    return total


def _process_rss_bytes() -> int | None:
    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass
    if os.name == "nt":
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        return int(counters.WorkingSetSize) if ok else None
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(rss * (1024 if sys.platform != "darwin" else 1))
    except Exception:
        return None




def _process_memory_bytes() -> dict[str, int | None]:
    if os.name == "nt":
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        if ok:
            return {"rss": int(counters.WorkingSetSize), "peak_rss": int(counters.PeakWorkingSetSize)}
    try:
        import psutil  # type: ignore

        return {"rss": int(psutil.Process(os.getpid()).memory_info().rss), "peak_rss": None}
    except Exception:
        pass
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = int(peak * (1024 if sys.platform != "darwin" else 1))
        return {"rss": None, "peak_rss": peak_bytes}
    except Exception:
        return {"rss": None, "peak_rss": None}


def _gpu_memory_bytes() -> dict[str, int | None]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0}
        return {
            "allocated": int(torch.cuda.memory_allocated()),
            "reserved": int(torch.cuda.memory_reserved()),
            "max_allocated": int(torch.cuda.max_memory_allocated()),
            "max_reserved": int(torch.cuda.max_memory_reserved()),
        }
    except Exception:
        return {"allocated": None, "reserved": None, "max_allocated": None, "max_reserved": None}


class Budget:
    """S0/S1 central budget guard.

    The timer starts in ``__enter__``.  The guard records a ledger under the
    campaign directory and never terminates other processes.  Callers should use
    ``tick('prefix'|'head'|'update'|'scientific_fit'|'scientific_update'|'test')``
    around billable operations and ``check`` before/after larger steps.
    """

    def __init__(self, stage: str = "s0", category: str = "gpu", label: str | None = None,
                 ledger_path: str | os.PathLike[str] | None = None):
        if stage not in ("s0", "s1"):
            raise ValueError("Budget stage must be 's0' or 's1'")
        if category not in ("gpu", "cpu", "cpu_synthetic"):
            raise ValueError("category must be 'gpu', 'cpu', or 'cpu_synthetic'")
        self.stage = stage
        self.category = category
        self.label = label or f"{stage}_{category}_budget"
        default_ledger = HERE / ("s0_budget_ledger.json" if stage == "s0" else "s1_budget_ledger.json")
        self.ledger_path = _require_write_path(resolve(ledger_path or default_ledger))
        self.limits: dict[str, Any] | None = None
        self.job_id: int | None = None
        self._start_perf: float | None = None
        self._start_process: float | None = None

    def _empty_ledger(self) -> dict[str, Any]:
        return {
            "schema": "tsfm_readout_budget_ledger_v2",
            "stage": self.stage,
            "operation_started_utc": None,
            "disk_baseline_bytes": None,
            "jobs": [],
            "counters": {
                "prefix_calls": 0,
                "head_calls": 0,
                "technical_updates": 0,
                "scientific_fits": 0,
                "scientific_updates": 0,
                "new_test_predictions": 0,
            },
            "peaks": {"rss_bytes": 0, "gpu_allocated_bytes": 0, "gpu_reserved_bytes": 0},
        }

    def _load_ledger(self) -> dict[str, Any]:
        if self.ledger_path.exists():
            ledger = read_json(self.ledger_path)
            ledger.setdefault("jobs", [])
            counters = ledger.setdefault("counters", {})
            for key in ("prefix_calls", "head_calls", "technical_updates", "scientific_fits", "scientific_updates", "new_test_predictions"):
                counters.setdefault(key, 0)
            ledger.setdefault("peaks", {"rss_bytes": 0, "gpu_allocated_bytes": 0, "gpu_reserved_bytes": 0})
            return ledger
        return self._seeded_empty_ledger()

    def _seeded_empty_ledger(self) -> dict[str, Any]:
        ledger = self._empty_ledger()
        if self.stage != "s0":
            return ledger
        receipt_path = HERE / "cpu_checks.json"
        if not receipt_path.exists():
            return ledger
        try:
            receipt = read_json(receipt_path)
            wall = float(receipt.get("wall_seconds", 0.0) or 0.0)
            cpu = float(receipt.get("cpu_process_seconds", 0.0) or 0.0)
            peak_ram = int(receipt.get("peak_ram_bytes", 0) or 0)
            ended = receipt_path.stat().st_mtime
            started = max(0.0, ended - wall)
            ledger["operation_started_utc"] = started
            ledger["jobs"].append({
                "id": 0,
                "stage": "s0",
                "category": "cpu_synthetic",
                "label": "cpu_checks_receipt",
                "pid": None,
                "status": "complete",
                "started_utc": started,
                "ended_utc": ended,
                "elapsed_wall_seconds": wall,
                "process_time_seconds": cpu,
                "receipt": "cpu_checks.json",
                "receipt_status": receipt.get("status"),
            })
            if peak_ram > 0:
                ledger["peaks"]["rss_bytes"] = peak_ram
        except Exception as exc:
            ledger["cpu_checks_seed_error"] = repr(exc)
        return ledger

    def _save_ledger(self, ledger: Mapping[str, Any]) -> None:
        save_json(self.ledger_path, dict(ledger))

    def _running_rows(self, ledger: Mapping[str, Any]) -> list[dict[str, Any]]:
        return [dict(row) for row in ledger.get("jobs", []) if row.get("status") == "running"]

    def __enter__(self):
        self.limits = _approved_limits(self.stage)
        ledger = self._load_ledger()
        running = self._running_rows(ledger)
        if running:
            other = [row for row in running if row.get("pid") != os.getpid()]
            if other:
                raise RuntimeError(f"STOP_RESOURCE: pre-existing running budget row from another PID: {other}")
            raise RuntimeError(f"STOP_RESOURCE: pre-existing running budget row: {running}")
        now = time.time()
        if ledger.get("operation_started_utc") is None:
            ledger["operation_started_utc"] = now
        if ledger.get("disk_baseline_bytes") is None:
            ledger["disk_baseline_bytes"] = _dir_bytes(HERE) + _dir_bytes(CACHE)
        self.job_id = len(ledger["jobs"])
        self._start_perf = time.perf_counter()
        self._start_process = time.process_time()
        ledger["jobs"].append({
            "id": self.job_id,
            "stage": self.stage,
            "category": self.category,
            "label": self.label,
            "pid": os.getpid(),
            "status": "running",
            "started_utc": now,
            "elapsed_wall_seconds": 0.0,
            "process_time_seconds": 0.0,
        })
        self._save_ledger(ledger)
        try:
            self.check()
        except Exception as exc:
            self._finalize_current(status="failed", error="".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
            raise
        return self

    def _current_elapsed(self) -> tuple[float, float]:
        wall = float(time.perf_counter() - self._start_perf) if self._start_perf is not None else 0.0
        proc = float(time.process_time() - self._start_process) if self._start_process is not None else 0.0
        return wall, proc

    def _raw_snapshot(self) -> dict[str, Any]:
        ledger = self._load_ledger()
        now = time.time()
        operation_started = ledger.get("operation_started_utc")
        if operation_started is None:
            operation_started = now
        disk_now = _dir_bytes(HERE) + _dir_bytes(CACHE)
        disk_base = ledger.get("disk_baseline_bytes")
        if disk_base is None:
            disk_base = disk_now
        memory = _process_memory_bytes()
        gpu = _gpu_memory_bytes()
        current_wall, current_process = self._current_elapsed()
        gpu_wall = 0.0
        cpu_wall = 0.0
        cpu_process = 0.0
        for row in ledger.get("jobs", []):
            if row.get("status") == "running":
                if row.get("id") != self.job_id:
                    raise RuntimeError(f"STOP_RESOURCE: unexpected concurrent running budget row: {row}")
                wall = current_wall
                proc = current_process
            else:
                wall = float(row.get("elapsed_wall_seconds", 0.0) or 0.0)
                proc = float(row.get("process_time_seconds", 0.0) or 0.0)
            if row.get("category") == "gpu":
                gpu_wall += wall
            elif row.get("category") in ("cpu", "cpu_synthetic"):
                cpu_wall += wall
            cpu_process += proc
        return {
            "operation_elapsed_seconds": float(now - float(operation_started)),
            "gpu_wall_seconds": gpu_wall,
            "cpu_wall_seconds": cpu_wall,
            "cpu_process_seconds": cpu_process,
            "new_disk_bytes": max(0, disk_now - int(disk_base)),
            "rss_bytes": memory.get("rss"),
            "peak_rss_bytes": memory.get("peak_rss"),
            "gpu_memory": gpu,
            "counters": dict(ledger.get("counters", {})),
            "limits": self.limits or {},
        }

    def _apply_peak_snapshot(self, ledger: dict[str, Any], snap: Mapping[str, Any]) -> None:
        peaks = ledger.setdefault("peaks", {})
        rss_candidates = [snap.get("rss_bytes"), snap.get("peak_rss_bytes")]
        for value in rss_candidates:
            if isinstance(value, int):
                peaks["rss_bytes"] = max(int(peaks.get("rss_bytes") or 0), value)
        gpu = snap.get("gpu_memory") or {}
        if isinstance(gpu, dict):
            allocated_candidates = [gpu.get("allocated"), gpu.get("max_allocated")]
            reserved_candidates = [gpu.get("reserved"), gpu.get("max_reserved")]
            for value in allocated_candidates:
                if isinstance(value, int):
                    peaks["gpu_allocated_bytes"] = max(int(peaks.get("gpu_allocated_bytes") or 0), value)
            for value in reserved_candidates:
                if isinstance(value, int):
                    peaks["gpu_reserved_bytes"] = max(int(peaks.get("gpu_reserved_bytes") or 0), value)

    def snapshot(self) -> dict[str, Any]:
        snap = self._raw_snapshot()
        ledger = self._load_ledger()
        self._apply_peak_snapshot(ledger, snap)
        if self.job_id is not None and 0 <= self.job_id < len(ledger["jobs"]):
            wall, proc = self._current_elapsed()
            ledger["jobs"][self.job_id].update({
                "elapsed_wall_seconds": wall,
                "process_time_seconds": proc,
                "last_snapshot_utc": time.time(),
            })
        self._save_ledger(ledger)
        return snap

    def check(self) -> dict[str, Any]:
        if self.limits is None:
            self.limits = _approved_limits(self.stage)
        snap = self.snapshot()
        limits = self.limits
        checks = [
            (snap["operation_elapsed_seconds"], _limit_seconds(limits, "maximum_elapsed_operation_wall_seconds"), "operation elapsed wall"),
            (snap["gpu_wall_seconds"], _limit_seconds(limits, "maximum_gpu_job_wall_seconds"), "GPU job wall"),
            (snap["cpu_wall_seconds"], _limit_seconds(limits, "maximum_cpu_only_job_wall_seconds"), "CPU-only job wall"),
            (snap["cpu_process_seconds"], _limit_seconds(limits, "maximum_cpu_process_time_seconds"), "CPU process time"),
            (snap["new_disk_bytes"], _limit_int(limits, "maximum_new_disk_bytes"), "new disk bytes"),
            (snap["counters"].get("technical_updates", 0), _limit_int(limits, "maximum_technical_optimizer_updates"), "technical optimizer updates"),
            (snap["counters"].get("prefix_calls", 0), _limit_int(limits, "maximum_full_prefix_batch_calls"), "prefix batch calls"),
            (snap["counters"].get("head_calls", 0), _limit_int(limits, "maximum_head_or_output_only_batch_calls"), "head/output batch calls"),
        ]
        if self.stage == "s1":
            checks.extend([
                (snap["counters"].get("scientific_fits", 0), _limit_int(limits, "maximum_scientific_fits", "maximum_fits"), "scientific fits"),
                (snap["counters"].get("scientific_updates", 0), _limit_int(limits, "maximum_scientific_optimizer_updates", "maximum_scientific_updates"), "scientific optimizer updates"),
                (snap["counters"].get("new_test_predictions", 0), _limit_int(limits, "maximum_new_test_predictions", "maximum_selected_test_models", "selected_test_models"), "TEST predictions"),
            ])
        for actual, limit, label in checks:
            if limit is not None and actual is not None and float(actual) > float(limit):
                raise RuntimeError(f"STOP_RESOURCE: {label} cap exceeded ({actual} > {limit})")
        ram_limit = _limit_int(limits, "maximum_process_ram_bytes")
        peak_ram = max([value for value in (snap.get("rss_bytes"), snap.get("peak_rss_bytes")) if isinstance(value, int)] or [0])
        if ram_limit is not None and peak_ram > ram_limit:
            raise RuntimeError(f"STOP_RESOURCE: process RAM cap exceeded ({peak_ram} > {ram_limit})")
        gpu_limit = _limit_int(limits, "maximum_gpu_allocated_or_reserved_bytes")
        gpu = snap.get("gpu_memory") or {}
        if gpu_limit is not None and isinstance(gpu, dict):
            for key in ("allocated", "reserved", "max_allocated", "max_reserved"):
                value = gpu.get(key)
                if isinstance(value, int) and value > gpu_limit:
                    raise RuntimeError(f"STOP_RESOURCE: GPU {key} cap exceeded ({value} > {gpu_limit})")
        if self.stage == "s0":
            if snap["counters"].get("scientific_fits", 0) != 0 or snap["counters"].get("scientific_updates", 0) != 0 or snap["counters"].get("new_test_predictions", 0) != 0:
                raise RuntimeError("STOP_DEBUG: S0 guard recorded scientific fit/update or TEST prediction")
        return snap

    def tick(self, kind: str, n: int = 1) -> dict[str, Any]:
        if int(n) < 0:
            raise ValueError("tick n must be non-negative")
        canonical = _COUNTER_ALIASES.get(kind)
        if canonical is None:
            raise ValueError(f"unknown budget counter kind: {kind!r}")
        if canonical == "new_test_predictions" and not _test_choice_is_sealed():
            raise RuntimeError("TEST prediction counter rejected until selection_seal/global_choice_seal exists")
        ledger = self._load_ledger()
        ledger.setdefault("counters", {}).setdefault(canonical, 0)
        ledger["counters"][canonical] = int(ledger["counters"][canonical]) + int(n)
        if self.job_id is not None and 0 <= self.job_id < len(ledger["jobs"]):
            ledger["jobs"][self.job_id]["last_tick"] = {"kind": canonical, "n": int(n), "utc": time.time()}
        self._save_ledger(ledger)
        return self.check()

    def heartbeat(self, progress: Any | None = None, **extra: Any) -> dict[str, Any]:
        snap = self.snapshot()
        ledger = self._load_ledger()
        if self.job_id is not None and 0 <= self.job_id < len(ledger["jobs"]):
            row = ledger["jobs"][self.job_id]
            row["heartbeat_utc"] = time.time()
            if progress is not None:
                row["progress"] = progress
            row.update(extra)
        self._save_ledger(ledger)
        return snap

    def _finalize_current(self, status: str, error: str | None = None) -> None:
        ledger = self._load_ledger()
        if self.job_id is not None and 0 <= self.job_id < len(ledger["jobs"]):
            row = ledger["jobs"][self.job_id]
            wall, proc = self._current_elapsed()
            row.update(status=status, ended_utc=time.time(), elapsed_wall_seconds=wall, process_time_seconds=proc)
            if error:
                row["error"] = error
        try:
            snap = self._raw_snapshot()
            self._apply_peak_snapshot(ledger, snap)
        finally:
            self._save_ledger(ledger)

    def __exit__(self, exc_type, exc, tb) -> bool:
        error = "".join(traceback.format_exception(exc_type, exc, tb)) if exc_type else None
        self._finalize_current(status="failed" if exc_type else "complete", error=error)
        return False


ReferenceBudget = Budget
from budget_guard import FastBudget as Budget
BudgetGuard = Budget
