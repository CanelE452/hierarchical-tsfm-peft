"""Diagnose Beijing UCI archive schema without reading forecast targets.

The script inspects archive members, headers, station fields, and timestamp
components only.  It does not create training payloads and does not open any
EVAL target store.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import io
import json
import zipfile

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_RAW = ROOT / ".cache" / HERE.name / "raw_dataset" / "beijing_aotizhongxin" / "download" / "uci_501.zip"
DEFAULT_OUT = HERE / "beijing_data_diagnosis" / "current_raw_metadata_v2.json"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_members(zip_path: Path) -> tuple[list[str], list[tuple[str, bytes]]]:
    top_level = []
    csv_payloads = []
    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            top_level.append(name)
            if name.lower().endswith(".csv"):
                csv_payloads.append((name, archive.read(name)))
            elif name.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(archive.read(name))) as nested:
                    for inner_name in nested.namelist():
                        if inner_name.lower().endswith(".csv"):
                            csv_payloads.append((f"{name}!{inner_name}", nested.read(inner_name)))
    return top_level, csv_payloads


def member_report(member: str, payload: bytes) -> dict:
    header = payload.splitlines()[0].decode("utf-8-sig", errors="replace")
    columns = pd.read_csv(io.BytesIO(payload), nrows=0).columns.tolist()
    report = {"member": member, "header": header, "columns": columns}

    metadata_columns = [c for c in ("year", "month", "day", "hour", "station", "Date") if c in columns]
    if not metadata_columns:
        report["metadata_error"] = "No recognized timestamp/station metadata columns"
        return report
    frame = pd.read_csv(io.BytesIO(payload), usecols=metadata_columns)
    report["rows"] = int(len(frame))
    if "station" in frame.columns:
        report["station_values"] = sorted(frame["station"].astype(str).dropna().unique().tolist())
    if {"year", "month", "day", "hour"}.issubset(frame.columns):
        timestamps = pd.to_datetime(
            {"year": frame["year"], "month": frame["month"], "day": frame["day"], "hour": frame["hour"]},
            errors="raise",
        )
        report["timestamp_source"] = "year/month/day/hour"
    elif "Date" in frame.columns:
        timestamps = pd.to_datetime(frame["Date"], errors="raise")
        report["timestamp_source"] = "Date"
    else:
        return report
    report["first_timestamp"] = timestamps.min().isoformat()
    report["last_timestamp"] = timestamps.max().isoformat()
    report["duplicate_timestamps_within_member"] = int(timestamps.duplicated().sum())
    report["valid_timestamp_count"] = int(timestamps.notna().sum())
    report["nat_timestamp_count"] = int(timestamps.isna().sum())
    if len(timestamps) > 1:
        diffs = timestamps.sort_values().diff().dropna().dt.total_seconds()
        report["median_interval_seconds"] = float(diffs.median())
    return report


def frozen_parser_simulation(csv_payloads: list[tuple[str, bytes]]) -> dict:
    source_members = [name for name, _payload in csv_payloads]
    source_text = ",".join(source_members).lower()
    selected_metadata_frames = []
    per_member = []
    for name, payload in csv_payloads:
        columns_list = pd.read_csv(io.BytesIO(payload), nrows=0).columns.tolist()
        columns = set(columns_list)
        metadata_columns = [c for c in ("year", "month", "day", "hour", "station", "Date") if c in columns]
        frame = pd.read_csv(io.BytesIO(payload), usecols=metadata_columns)
        record = {
            "member": name,
            "rows": int(len(frame)),
            "has_station": "station" in columns,
            "has_year_month_day_hour": {"year", "month", "day", "hour"}.issubset(columns),
            "has_date": "Date" in columns,
            "would_append_by_frozen_logic": False,
        }
        if "station" in frame.columns:
            record["station_counts"] = frame["station"].astype(str).value_counts().to_dict()
            selected = frame[frame["station"].astype(str) == "Aotizhongxin"]
            record["selected_rows"] = int(len(selected))
            record["would_append_by_frozen_logic"] = True
            selected_metadata_frames.append(selected)
        elif "aotizhongxin" in source_text:
            record["selected_rows"] = int(len(frame))
            record["would_append_by_frozen_logic"] = True
            selected_metadata_frames.append(frame)
        else:
            record["selected_rows"] = 0

        if record["has_year_month_day_hour"]:
            timestamps = pd.to_datetime(
                {"year": frame["year"], "month": frame["month"], "day": frame["day"], "hour": frame["hour"]},
                errors="coerce",
            )
            record["valid_ymdh_timestamps"] = int(timestamps.notna().sum())
            record["nat_ymdh_timestamps"] = int(timestamps.isna().sum())
        else:
            record["valid_ymdh_timestamps"] = 0
            record["nat_ymdh_timestamps"] = int(len(frame))
        if record["has_date"]:
            date_timestamps = pd.to_datetime(frame["Date"], errors="coerce")
            record["valid_date_timestamps"] = int(date_timestamps.notna().sum())
            record["duplicate_date_timestamps_within_member"] = int(date_timestamps.duplicated().sum())
        per_member.append(record)

    concat = pd.concat(selected_metadata_frames, ignore_index=True) if selected_metadata_frames else pd.DataFrame()
    if len(concat):
        timestamps = pd.to_datetime(
            {
                "year": concat.get("year"),
                "month": concat.get("month"),
                "day": concat.get("day"),
                "hour": concat.get("hour"),
            },
            errors="coerce",
        )
        duplicated = int(timestamps.duplicated().sum())
        valid = int(timestamps.notna().sum())
        nat = int(timestamps.isna().sum())
    else:
        duplicated = valid = nat = 0
    return {
        "source_members_contain_aotizhongxin": "aotizhongxin" in source_text,
        "per_member": per_member,
        "frozen_concat_rows": int(len(concat)),
        "frozen_concat_valid_ymdh": valid,
        "frozen_concat_nat": nat,
        "frozen_concat_duplicated_timestamps": duplicated,
        "root_cause": (
            "Frozen parser appends stationless top-level CSVs because a global source member string contains "
            "Aotizhongxin; their missing year/month/day/hour become NaT, and pandas counts 1005 duplicate NaT."
        ),
    }


def diagnose(zip_path: Path) -> dict:
    top_level, payloads = archive_members(zip_path)
    reports = [member_report(member, payload) for member, payload in payloads]
    recognized_prsa_members = [
        item
        for item in reports
        if {"year", "month", "day", "hour", "station"}.issubset(set(item.get("columns", [])))
    ]
    aotizhongxin_members = [
        item
        for item in recognized_prsa_members
        if "Aotizhongxin" in item.get("station_values", []) or "aotizhongxin" in item["member"].lower()
    ]
    duplicate_across_date_members = None
    date_frames = []
    for member, payload in payloads:
        columns = pd.read_csv(io.BytesIO(payload), nrows=0).columns.tolist()
        if "Date" not in columns:
            continue
        frame = pd.read_csv(io.BytesIO(payload), usecols=["Date"])
        frame["member"] = member
        date_frames.append(frame)
    if date_frames:
        merged = pd.concat(date_frames, ignore_index=True)
        timestamps = pd.to_datetime(merged["Date"], errors="raise")
        duplicate_across_date_members = int(timestamps.duplicated().sum())

    simulation = frozen_parser_simulation(payloads)
    status = "PASS_PRSA_PRESENT_WITH_FROZEN_PARSER_STOCK_CSV_BUG" if aotizhongxin_members else "FAIL_NOT_PRSA_AOTIZHONGXIN_ARCHIVE"
    conclusion = (
        "Current archive contains nested PRSA Aotizhongxin data; failure is parser selection, not a missing raw dataset."
        if status.startswith("PASS")
        else "Current archive does not contain PRSA year/month/day/hour/station metadata for Aotizhongxin."
    )
    return {
        "zip_path": str(zip_path.resolve()),
        "zip_sha256": sha256_path(zip_path),
        "zip_size_bytes": zip_path.stat().st_size,
        "top_level_members": top_level,
        "csv_member_count_recursive": len(payloads),
        "members": reports,
        "prsa_member_count": len(recognized_prsa_members),
        "aotizhongxin_member_count": len(aotizhongxin_members),
        "duplicate_timestamps_across_date_members": duplicate_across_date_members,
        "frozen_parser_simulation": simulation,
        "status": status,
        "conclusion": conclusion,
        "no_eval_target_store_opened": True,
    }


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = diagnose(args.zip)
    write_json(args.out, result)
    print(json.dumps({"status": result["status"], "out": str(args.out), "zip_sha256": result["zip_sha256"]}, indent=2))


if __name__ == "__main__":
    main()

