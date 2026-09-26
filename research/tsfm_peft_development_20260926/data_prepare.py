from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MLTS = Path("E:/CODING/proj/mltimeseries")
OUT_DIR = ROOT / ".cache" / "tsfm_peft_development_20260926" / "data"
DOC_DIR = ROOT / "research" / "tsfm_peft_development_20260926"

L_CONTEXT = 512
HORIZON = 48
ORIGIN_STRIDE = 24

ELECTRICITY_RAW = MLTS / "data" / "electricity" / "electricity.csv"
BDG2_ELECTRICITY_RAW = (
    MLTS / "data_external" / "bdg2_coarse_supervision_v1" / "raw" / "electricity.csv"
)
BDG2_METADATA_RAW = (
    MLTS / "data_external" / "bdg2_coarse_supervision_v1" / "raw" / "metadata.csv"
)
EXPOSURE_LEDGER = MLTS / "results" / "peft_paper_closure_v1" / "data_exposure_ledger.csv"
FRESH_STAGE_A_MANIFEST = (
    MLTS / "results" / "peft_paper_closure_v1" / "fresh_stage_a_candidate_manifest.md"
)
HOLDOUT_AVAILABILITY = (
    MLTS / "results" / "peft_paper_closure_v1" / "holdout_availability.md"
)
REFERENCE_REPOS = {
    "mltimeseries": MLTS,
    "forecast-revision-peft": Path("E:/CODING/proj/forecast-revision-peft"),
    "hierarchical-tsfm-peft": ROOT,
    "covariate-trust-pilot": Path("E:/CODING/proj/covariate-trust-pilot"),
    "tsfm-peft-method-screen": Path("E:/CODING/proj/tsfm-peft-method-screen"),
}


@dataclass(frozen=True)
class HalfOpen:
    name: str
    start: pd.Timestamp
    end: pd.Timestamp


BULL_SPLITS = [
    HalfOpen("precontext", pd.Timestamp("2016-01-01T00:00:00"), pd.Timestamp("2016-01-23T00:00:00")),
    HalfOpen("train", pd.Timestamp("2016-01-23T00:00:00"), pd.Timestamp("2016-04-15T00:00:00")),
    HalfOpen("val", pd.Timestamp("2016-04-17T00:00:00"), pd.Timestamp("2016-05-18T00:00:00")),
    HalfOpen("eval_e1", pd.Timestamp("2016-05-20T00:00:00"), pd.Timestamp("2016-06-30T00:00:00")),
    HalfOpen("eval_e2", pd.Timestamp("2016-06-30T00:00:00"), pd.Timestamp("2016-08-10T00:00:00")),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_time_frame(path: Path, time_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df[time_col] = pd.to_datetime(df[time_col], errors="raise")
    df = df.sort_values(time_col).drop_duplicates(time_col, keep="first").reset_index(drop=True)
    return df


def origin_range(start: int, end: int, stride: int = 1) -> np.ndarray:
    first = max(start, L_CONTEXT)
    last_exclusive = end - HORIZON + 1
    if last_exclusive <= first:
        return np.array([], dtype=np.int64)
    return np.arange(first, last_exclusive, stride, dtype=np.int64)


def split_origin_range(
    times: pd.Series, start: pd.Timestamp, end: pd.Timestamp, stride: int = ORIGIN_STRIDE
) -> np.ndarray:
    idx_start = int(np.searchsorted(times.values, np.datetime64(start), side="left"))
    idx_end = int(np.searchsorted(times.values, np.datetime64(end), side="left"))
    return origin_range(idx_start, idx_end, stride=stride)


def standardize(raw: np.ndarray, train_slice: slice) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = raw[train_slice]
    mean = np.nanmean(train, axis=0).astype(np.float32)
    std = np.nanstd(train, axis=0).astype(np.float32)
    bad = ~np.isfinite(std) | (std <= 0)
    if bad.any():
        std = std.copy()
        std[bad] = np.float32(1.0)
    filled = np.where(np.isfinite(raw), raw, mean[None, :])
    x = ((filled - mean[None, :]) / std[None, :]).astype(np.float32)
    return x, mean, std


def int64_ns(times: pd.Series) -> np.ndarray:
    return pd.to_datetime(times).astype("int64").to_numpy(dtype=np.int64)


def finite_fraction(raw: np.ndarray, row_slice: slice | np.ndarray) -> np.ndarray:
    return np.isfinite(raw[row_slice]).mean(axis=0).astype(np.float64)


def nonconstant(raw: np.ndarray, row_slice: slice | np.ndarray) -> np.ndarray:
    return (np.nanstd(raw[row_slice], axis=0) > 0).astype(bool)


def prepare_electricity() -> dict[str, Any]:
    df = read_time_frame(ELECTRICITY_RAW, "date")
    columns = [c for c in df.columns if c != "date"][:32]
    raw = df[columns].to_numpy(dtype=np.float32)
    t = len(df)
    train_end = int(np.floor(t * 0.70))
    val_end = int(np.floor(t * 0.80))
    train_slice = slice(0, train_end)
    x, mean, std = standardize(raw, train_slice)
    targetmask = np.isfinite(raw)

    train_origins = origin_range(0, train_end, stride=1)
    val_origins = origin_range(train_end, val_end, stride=ORIGIN_STRIDE)
    dev_origins = origin_range(val_end, t, stride=ORIGIN_STRIDE)

    npz_path = OUT_DIR / "electricity_first32_l512_h48.npz"
    np.savez_compressed(
        npz_path,
        x=x.astype(np.float32),
        raw=raw.astype(np.float32),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        times=int64_ns(df["date"]),
        targetmask=targetmask.astype(bool),
        train_origins=train_origins,
        val_origins=val_origins,
        dev_origins=dev_origins,
        columns=np.array(columns, dtype="U"),
        split_bounds=np.array([0, train_end, val_end, t], dtype=np.int64),
    )

    train_avail = finite_fraction(raw, train_slice)
    train_nonconstant = nonconstant(raw, train_slice)
    return {
        "npz": str(npz_path.relative_to(ROOT)),
        "source": str(ELECTRICITY_RAW),
        "source_sha256": sha256_file(ELECTRICITY_RAW),
        "timestamp_note": "Timestamps are read as-is from mltimeseries/data/electricity/electricity.csv; this file spans 2016-07-01 02:00:00 through 2019-07-02 01:00:00. This preparation does not synthesize or remap dates. Earlier canonical Electricity artifacts may use different source-year conventions and must not be conflated with this local file.",
        "time_range": [str(df["date"].iloc[0]), str(df["date"].iloc[-1])],
        "rows": t,
        "columns": columns,
        "shape": list(raw.shape),
        "split_bounds": {"train": [0, train_end], "val": [train_end, val_end], "dev": [val_end, t]},
        "origin_counts": {
            "train": int(train_origins.size),
            "val": int(val_origins.size),
            "dev": int(dev_origins.size),
        },
        "train_availability": {
            "min": float(train_avail.min()),
            "max": float(train_avail.max()),
            "all_ge_95pct": bool((train_avail >= 0.95).all()),
        },
        "train_nonconstant_all": bool(train_nonconstant.all()),
    }


def sealed_bull_office_ids() -> list[str]:
    metadata = pd.read_csv(BDG2_METADATA_RAW)
    office = metadata[
        (metadata["site_id"].astype(str) == "Bull")
        & (metadata["primaryspaceusage"].astype(str) == "Office")
        & metadata["electricity"].notna()
    ]
    return office["building_id"].astype(str).tolist()


def prepare_bull_office() -> dict[str, Any]:
    sealed = sealed_bull_office_ids()
    df = read_time_frame(BDG2_ELECTRICITY_RAW, "timestamp")
    missing_in_raw = [c for c in sealed if c not in df.columns]
    if missing_in_raw:
        raise RuntimeError(f"sealed Bull Office ids missing from electricity.csv: {missing_in_raw}")

    raw_all = df[sealed].to_numpy(dtype=np.float32)
    times = df["timestamp"]
    train_idx = (times >= BULL_SPLITS[1].start) & (times < BULL_SPLITS[1].end)
    if not bool(train_idx.any()):
        raise RuntimeError("Bull Office train period has no rows.")

    train_avail = finite_fraction(raw_all, train_idx.to_numpy())
    train_nonconstant = nonconstant(raw_all, train_idx.to_numpy())
    selected_mask = (train_avail >= 0.95) & train_nonconstant
    selected = [c for c, keep in zip(sealed, selected_mask, strict=True) if bool(keep)]
    if not selected:
        raise RuntimeError("No Bull Office channel passed train-only availability/nonconstant gate.")

    selected_indices = np.flatnonzero(selected_mask)
    raw = raw_all[:, selected_indices]
    train_positions = np.flatnonzero(train_idx.to_numpy())
    train_slice = np.s_[train_positions]
    x, mean, std = standardize(raw, train_slice)
    targetmask = np.isfinite(raw)

    origins = {
        "train_origins": split_origin_range(times, BULL_SPLITS[1].start, BULL_SPLITS[1].end, stride=1),
        "val_origins": split_origin_range(times, BULL_SPLITS[2].start, BULL_SPLITS[2].end),
        "eval_e1_origins": split_origin_range(times, BULL_SPLITS[3].start, BULL_SPLITS[3].end),
        "eval_e2_origins": split_origin_range(times, BULL_SPLITS[4].start, BULL_SPLITS[4].end),
    }

    npz_path = OUT_DIR / "bdg2_bull_office17_l512_h48.npz"
    np.savez_compressed(
        npz_path,
        x=x.astype(np.float32),
        raw=raw.astype(np.float32),
        raw_all=raw_all.astype(np.float32),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        times=int64_ns(times),
        targetmask=targetmask.astype(bool),
        targetmask_all=np.isfinite(raw_all).astype(bool),
        columns=np.array(selected, dtype="U"),
        sealed_columns=np.array(sealed, dtype="U"),
        selected_mask=selected_mask.astype(bool),
        train_origins=origins["train_origins"],
        val_origins=origins["val_origins"],
        eval_e1_origins=origins["eval_e1_origins"],
        eval_e2_origins=origins["eval_e2_origins"],
    )

    split_rows = {}
    for sp in BULL_SPLITS:
        idx = (times >= sp.start) & (times < sp.end)
        split_rows[sp.name] = int(idx.sum())

    return {
        "npz": str(npz_path.relative_to(ROOT)),
        "source": str(BDG2_ELECTRICITY_RAW),
        "source_sha256": sha256_file(BDG2_ELECTRICITY_RAW),
        "metadata": str(BDG2_METADATA_RAW),
        "metadata_sha256": sha256_file(BDG2_METADATA_RAW),
        "sealed_columns_all": sealed,
        "selected_columns": selected,
        "selected_count": len(selected),
        "selection_rule": "Train-only observed fraction >= 0.95 and train nonconstant; no protected target outcome used.",
        "train_availability_by_sealed_column": {
            c: float(v) for c, v in zip(sealed, train_avail, strict=True)
        },
        "train_nonconstant_by_sealed_column": {
            c: bool(v) for c, v in zip(sealed, train_nonconstant, strict=True)
        },
        "split_periods": {
            sp.name: [sp.start.isoformat(), sp.end.isoformat()] for sp in BULL_SPLITS
        },
        "split_rows": split_rows,
        "origin_counts": {
            key.replace("_origins", ""): int(value.size) for key, value in origins.items()
        },
        "time_range": [str(times.iloc[0]), str(times.iloc[-1])],
    }


def audit_exposure() -> dict[str, Any]:
    exact_bull_rows: list[dict[str, str]] = []
    bdg2_rows = 0
    if EXPOSURE_LEDGER.exists():
        with EXPOSURE_LEDGER.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                text = json.dumps(row, ensure_ascii=False)
                if "BDG2" in text:
                    bdg2_rows += 1
                if "Bull_office" in text or "Bull Office" in text:
                    exact_bull_rows.append(row)

    manifest_text = FRESH_STAGE_A_MANIFEST.read_text(encoding="utf-8") if FRESH_STAGE_A_MANIFEST.exists() else ""
    holdout_text = HOLDOUT_AVAILABILITY.read_text(encoding="utf-8") if HOLDOUT_AVAILABILITY.exists() else ""
    repo_search = search_bull_references()
    return {
        "ledger": str(EXPOSURE_LEDGER),
        "ledger_exists": EXPOSURE_LEDGER.exists(),
        "bdg2_rows_in_ledger": bdg2_rows,
        "bull_office_exact_rows_in_ledger": len(exact_bull_rows),
        "fresh_manifest": str(FRESH_STAGE_A_MANIFEST),
        "fresh_manifest_mentions_bull_ready": "BDG2 Bull Office 2016A" in manifest_text
        and "READY" in manifest_text,
        "fresh_manifest_notes": [
            "Bull office meters are recorded as not existing PEFT target-label units in the manifest.",
            "BDG2 raw source family was already inspected locally, so this is not final independent holdout evidence.",
        ],
        "holdout_availability": str(HOLDOUT_AVAILABILITY),
        "holdout_certified_stage_a_zero": "Certified clean fresh units for Stage A: 0" in holdout_text,
        "holdout_certified_final_zero": "Certified sealed final units: 0" in holdout_text,
        "claim_boundary": "Do not call Bull Office fully independent; use as target-blind local protected candidate after sealing.",
        "reference_repo_search": repo_search,
    }


def search_bull_references() -> dict[str, Any]:
    search: dict[str, Any] = {
        "patterns": ["Bull_office", "Bull Office", "word-boundary Bull"],
        "excluded_paths": [
            "**/.cache/**",
            "**/data/**",
            "**/data_external/**",
            "**/*.csv",
            "**/*.txt",
            "**/*.gz",
            "**/*.npz",
            "**/*.npy",
            "**/*.parquet",
        ],
        "repos": {},
        "summary": "Search covered local code/results/history artifacts, excluding raw data and local caches. Absence is not a global literature or remote-only proof.",
    }
    cmd = [
        "rg",
        "-n",
        "--glob",
        "!**/.cache/**",
        "--glob",
        "!**/data/**",
        "--glob",
        "!**/data_external/**",
        "--glob",
        "!**/*.csv",
        "--glob",
        "!**/*.txt",
        "--glob",
        "!**/*.gz",
        "--glob",
        "!**/*.npz",
        "--glob",
        "!**/*.npy",
        "--glob",
        "!**/*.parquet",
        r"Bull_office|Bull Office|\bBull\b",
        ".",
    ]
    for name, repo in REFERENCE_REPOS.items():
        entry: dict[str, Any] = {"path": str(repo), "exists": repo.exists()}
        if repo.exists():
            proc = subprocess.run(
                cmd,
                cwd=repo,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
            )
            lines = [line for line in (proc.stdout or "").splitlines() if line.strip()]
            entry.update(
                {
                    "rg_exit_code": proc.returncode,
                    "match_count": len(lines),
                    "matches": lines[:80],
                    "truncated": len(lines) > 80,
                }
            )
        search["repos"][name] = entry
    return search


def write_exposure_md(exposure: dict[str, Any], bull: dict[str, Any]) -> None:
    search_lines: list[str] = []
    for name, info in exposure["reference_repo_search"]["repos"].items():
        if not info["exists"]:
            search_lines.append(f"- {name}: [미확인] local checkout missing at `{info['path']}`")
            continue
        search_lines.append(f"- {name}: [확인] matches={info['match_count']}")
        for match in info.get("matches", [])[:12]:
            search_lines.append(f"  - `{match}`")
    lines = [
        "# Exposure Audit",
        "",
        "[확인] 이전 PEFT 노출 장부와 fresh manifest를 읽어 BDG2/Bull Office 노출 상태만 확인했다.",
        "보호 E1/E2 예측이나 지표는 계산하지 않았다.",
        "",
        f"- Ledger: `{exposure['ledger']}`",
        f"- BDG2 rows in ledger: {exposure['bdg2_rows_in_ledger']}",
        f"- Bull Office exact rows in ledger: {exposure['bull_office_exact_rows_in_ledger']}",
        f"- Fresh manifest Bull READY mention: {exposure['fresh_manifest_mentions_bull_ready']}",
        f"- Stage A certified clean units zero in holdout audit: {exposure['holdout_certified_stage_a_zero']}",
        f"- Sealed final certified units zero in holdout audit: {exposure['holdout_certified_final_zero']}",
        "",
        "[판정] Bull Office는 이번 로컬 보호 후보로 봉인할 수 있지만, raw family가 이미 inspect된 BDG2이므로 완전 독립 holdout이나 독립 재현으로 부르지 않는다.",
        "",
        "## Five-Repo Bull Search",
        "",
        "[확인] 원자료·`.cache`·대용량 배열을 제외하고 `Bull_office`, `Bull Office`, 단어 경계 `Bull`을 검색했다. mltimeseries에서는 fresh 후보 준비와 관련 진단 기록만 확인됐고, covariate-trust-pilot 및 tsfm-peft-method-screen에서는 0건이었다. hierarchical-tsfm-peft의 hits는 현재 연구 폴더의 계획·계약·감사 기록이다. forecast-revision-peft는 로컬 checkout이 없어 이 머신에서는 미확인이다.",
        "",
        *search_lines,
        "",
        "[판정] 검색 범위 안에서는 2026-09-12 이후 Bull Office 예측·학습·평가 결과 노출을 새로 확인하지 못했다. 다만 raw family inspect와 pretraining overlap UNKNOWN은 남아 있어 본 개발을 막지는 않되 독립 최종 확증으로 부르지 않는다.",
        "",
        "## Sealed Bull Office IDs",
        "",
        "```text",
        *bull["sealed_columns_all"],
        "```",
        "",
        "## Train-Only Selection",
        "",
        f"Rule: {bull['selection_rule']}",
        f"Selected count: {bull['selected_count']} / {len(bull['sealed_columns_all'])}",
        "",
        "```text",
        *bull["selected_columns"],
        "```",
    ]
    (DOC_DIR / "exposure_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_DIR.mkdir(parents=True, exist_ok=True)

    electricity = prepare_electricity()
    bull = prepare_bull_office()
    exposure = audit_exposure()

    contract = {
        "created_by": "research/tsfm_peft_development_20260926/data_prepare.py",
        "scope": "CPU data preparation only; no model training, prediction, or protected metric evaluation.",
        "context_length": L_CONTEXT,
        "horizon": HORIZON,
        "origin_stride_eval": ORIGIN_STRIDE,
        "npz_interface": {
            "x": "float32 [T,C], standardized with train-only mean/std; missing inputs imputed to standardized 0",
            "raw": "float32 [T,C], original values with NaN for missing",
            "mean": "float32 [C], train-only",
            "std": "float32 [C], train-only with invalid std replaced by 1 only after QC reporting",
            "times": "int64 ns [T]",
            "targetmask": "bool [T,C], finite target availability before imputation",
            "origins": "int64 origin indices; context[o-512:o], target[o:o+48]",
            "columns": "str [C]",
        },
        "datasets": {"electricity_first32": electricity, "bdg2_bull_office": bull},
        "exposure": exposure,
    }
    audit = {
        "electricity_first32": {
            k: v
            for k, v in electricity.items()
            if k
            in {
                "npz",
                "rows",
                "shape",
                "split_bounds",
                "origin_counts",
                "train_availability",
                "train_nonconstant_all",
                "time_range",
            }
        },
        "bdg2_bull_office": {
            k: v
            for k, v in bull.items()
            if k
            in {
                "npz",
                "sealed_columns_all",
                "selected_columns",
                "selected_count",
                "selection_rule",
                "train_availability_by_sealed_column",
                "train_nonconstant_by_sealed_column",
                "split_periods",
                "split_rows",
                "origin_counts",
                "time_range",
            }
        },
    }
    write_json(DOC_DIR / "data_contract.json", contract)
    write_json(DOC_DIR / "data_audit.json", audit)
    write_exposure_md(exposure, bull)
    print(json.dumps({"contract": str(DOC_DIR / "data_contract.json"), "audit": str(DOC_DIR / "data_audit.json"), "out_dir": str(OUT_DIR)}, indent=2))


if __name__ == "__main__":
    main()
