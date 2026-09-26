from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = ROOT / "experiments" / "text_event_favorita_v2_20260925"
RESULTS_DIR = ROOT / "results" / "text_event_favorita_v2_20260925"
ZIP_PATH = EXPERIMENT_DIR / ".cache" / "store-sales-time-series-forecasting.zip"

EVAL_START = pd.Timestamp("2016-01-08")
PREFIXES = ("Traslado ", "Puente ", "Recupero ")
CONCEPT_SUFFIX_RE = re.compile(r"[+\-\u2212]\d+$")
EARTHQUAKE_RE = re.compile(r"terremoto", re.IGNORECASE)
OTHER_CONCEPT_WINDOW_BEFORE = 4
OTHER_CONCEPT_WINDOW_AFTER = 3


@dataclass(frozen=True)
class Paths:
    concepts: Path
    event_counts: Path
    event_details: Path
    event_store_details: Path
    exclusions: Path
    ambiguous: Path
    verification: Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )


def normalize_concept(description: str) -> dict[str, Any]:
    raw = str(description)
    text = raw.strip()
    flags: list[str] = []
    removed_prefixes: list[str] = []

    while True:
        matched = next((prefix for prefix in PREFIXES if text.startswith(prefix)), None)
        if matched is None:
            break
        removed_prefixes.append(matched.strip())
        text = text[len(matched) :].strip()

    lower_text = text.lower()
    for prefix in PREFIXES:
        lower_prefix = prefix.lower()
        if lower_text.startswith(lower_prefix) and not text.startswith(prefix):
            flags.append(f"case_mismatched_prefix:{text[:len(prefix)].strip()}")
            break

    if CONCEPT_SUFFIX_RE.search(text):
        flags.append("numeric_suffix_removed")
        text = CONCEPT_SUFFIX_RE.sub("", text).strip()

    if ":" in text:
        flags.append("colon_detail_removed")
        text = text.split(":", 1)[0].strip()

    alt = raw.strip()
    alt_removed: list[str] = []
    while True:
        lower_alt = alt.lower()
        hit = None
        for prefix in PREFIXES:
            if lower_alt.startswith(prefix.lower()):
                hit = prefix
                break
        if hit is None:
            break
        alt_removed.append(alt[: len(hit)].strip())
        alt = alt[len(hit) :].strip()
    alt = CONCEPT_SUFFIX_RE.sub("", alt).strip()
    alt = alt.split(":", 1)[0].strip()
    if alt != text:
        flags.append("case_insensitive_normalization_differs")

    return {
        "concept": text,
        "removed_prefixes": "|".join(removed_prefixes),
        "normalization_flags": "|".join(flags),
        "case_insensitive_concept": alt,
        "case_insensitive_removed_prefixes": "|".join(alt_removed),
    }


def concept_family(concept: str) -> str:
    if concept.startswith("Fundacion de ") or concept.startswith("Fundacion "):
        return "Fundacion"
    if concept.startswith("Cantonizacion de ") or concept.startswith("Cantonizacion "):
        return "Cantonizacion"
    if concept.startswith("Provincializacion de ") or concept.startswith("Provincializacion "):
        return "Provincializacion"
    return ""


def open_member(zf: zipfile.ZipFile, name: str) -> bytes:
    with zf.open(name) as f:
        return f.read()


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Favorita ZIP not found: {ZIP_PATH}")

    with zipfile.ZipFile(ZIP_PATH) as zf:
        names = set(zf.namelist())
        required = {"train.csv", "holidays_events.csv", "stores.csv"}
        missing = sorted(required - names)
        if missing:
            raise RuntimeError(f"Missing required ZIP members: {missing}")

        holiday_bytes = open_member(zf, "holidays_events.csv")
        store_bytes = open_member(zf, "stores.csv")
        train_header = pd.read_csv(zf.open("train.csv"), nrows=0)
        holidays = pd.read_csv(
            zf.open("holidays_events.csv"),
            parse_dates=["date"],
            dtype={
                "type": "string",
                "locale": "string",
                "locale_name": "string",
                "description": "string",
                "transferred": "bool",
            },
        )
        stores = pd.read_csv(
            zf.open("stores.csv"),
            dtype={
                "store_nbr": "int16",
                "city": "string",
                "state": "string",
                "type": "string",
                "cluster": "int16",
            },
        )
        train = pd.read_csv(
            zf.open("train.csv"),
            usecols=["date", "store_nbr", "sales"],
            parse_dates=["date"],
            dtype={"store_nbr": "int16", "sales": "float64"},
        )

    expected_holidays = ["date", "type", "locale", "locale_name", "description", "transferred"]
    expected_stores = ["store_nbr", "city", "state", "type", "cluster"]
    expected_train = ["id", "date", "store_nbr", "family", "sales", "onpromotion"]
    if holidays.columns.tolist() != expected_holidays:
        raise RuntimeError(f"Unexpected holidays_events.csv columns: {holidays.columns.tolist()}")
    if stores.columns.tolist() != expected_stores:
        raise RuntimeError(f"Unexpected stores.csv columns: {stores.columns.tolist()}")
    if train_header.columns.tolist() != expected_train:
        raise RuntimeError(f"Unexpected train.csv columns: {train_header.columns.tolist()}")

    manifest = {
        "zip_path": str(ZIP_PATH),
        "zip_sha256": sha256_file(ZIP_PATH),
        "members": {
            "holidays_events.csv": {
                "rows": len(holidays),
                "sha256": sha256_bytes(holiday_bytes),
                "columns": holidays.columns.tolist(),
            },
            "stores.csv": {
                "rows": len(stores),
                "sha256": sha256_bytes(store_bytes),
                "columns": stores.columns.tolist(),
            },
            "train.csv": {
                "columns": train_header.columns.tolist(),
                "loaded_columns_for_c0": ["date", "store_nbr", "sales"],
            },
        },
    }
    return holidays, stores, train, manifest


def build_concepts(holidays: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in holidays.reset_index(names="holiday_row_id").iterrows():
        norm = normalize_concept(row["description"])
        rows.append(
            {
                "holiday_row_id": int(row["holiday_row_id"]),
                "date": row["date"].strftime("%Y-%m-%d"),
                "type": str(row["type"]),
                "locale": str(row["locale"]),
                "locale_name": str(row["locale_name"]),
                "description": str(row["description"]),
                "transferred": bool(row["transferred"]),
                **norm,
                "concept_family": concept_family(norm["concept"]),
                "excluded_before_application": bool(row["transferred"]) or str(row["type"]) == "Work Day",
                "pre_application_exclusion_reason": "transferred"
                if bool(row["transferred"])
                else ("work_day" if str(row["type"]) == "Work Day" else ""),
            }
        )
    return pd.DataFrame(rows)


def expand_to_store_days(concepts: pd.DataFrame, stores: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    active = concepts[
        (~concepts["transferred"]) & (concepts["type"] != "Work Day")
    ].copy()
    expanded: list[dict[str, Any]] = []
    unmatched: list[dict[str, Any]] = []

    for row in active.itertuples(index=False):
        if row.locale == "National":
            matched = stores
        elif row.locale == "Regional":
            matched = stores[stores["state"] == row.locale_name]
        elif row.locale == "Local":
            matched = stores[stores["city"] == row.locale_name]
        else:
            matched = stores.iloc[0:0]

        if matched.empty:
            unmatched.append(
                {
                    "holiday_row_id": row.holiday_row_id,
                    "date": row.date,
                    "locale": row.locale,
                    "locale_name": row.locale_name,
                    "description": row.description,
                    "concept": row.concept,
                }
            )
            continue

        for store in matched.itertuples(index=False):
            expanded.append(
                {
                    "holiday_row_id": int(row.holiday_row_id),
                    "date": pd.Timestamp(row.date),
                    "store_nbr": int(store.store_nbr),
                    "city": str(store.city),
                    "state": str(store.state),
                    "type": row.type,
                    "locale": row.locale,
                    "locale_name": row.locale_name,
                    "description": row.description,
                    "concept": row.concept,
                    "concept_family": row.concept_family,
                }
            )

    expanded_df = pd.DataFrame(expanded)
    if expanded_df.empty:
        raise RuntimeError("No active event-store rows after transferred/work_day filtering")

    duplicate_keys = ["store_nbr", "concept", "date"]
    expanded_df["raw_duplicate_same_store_concept_date"] = expanded_df.duplicated(duplicate_keys, keep=False)
    duplicate_flagged_rows = int(expanded_df["raw_duplicate_same_store_concept_date"].sum())
    before_dedup = len(expanded_df)
    expanded_df = expanded_df.drop_duplicates(duplicate_keys).copy()
    expanded_df.attrs["raw_duplicate_same_store_concept_date_flagged_rows"] = duplicate_flagged_rows
    expanded_df.attrs["raw_duplicate_same_store_concept_date_rows_dropped"] = int(before_dedup - len(expanded_df))
    return expanded_df, unmatched


def group_consecutive(expanded: pd.DataFrame) -> pd.DataFrame:
    grouped_rows: list[dict[str, Any]] = []
    expanded = expanded.sort_values(["store_nbr", "concept", "date", "holiday_row_id"]).copy()

    for (store_nbr, concept), group in expanded.groupby(["store_nbr", "concept"], sort=False):
        group = group.sort_values("date")
        current: list[pd.Series] = []
        last_date: pd.Timestamp | None = None
        for _, row in group.iterrows():
            if last_date is None or row["date"] == last_date + pd.Timedelta(days=1):
                current.append(row)
            else:
                grouped_rows.append(make_store_occurrence(store_nbr, concept, current))
                current = [row]
            last_date = row["date"]
        if current:
            grouped_rows.append(make_store_occurrence(store_nbr, concept, current))

    return pd.DataFrame(grouped_rows)


def make_store_occurrence(store_nbr: int, concept: str, rows: list[pd.Series]) -> dict[str, Any]:
    dates = [pd.Timestamp(r["date"]) for r in rows]
    first = min(dates)
    last = max(dates)
    return {
        "store_nbr": int(store_nbr),
        "concept": concept,
        "eval_date": first,
        "group_end_date": last,
        "group_length_days": int((last - first).days + 1),
        "all_dates": "|".join(d.strftime("%Y-%m-%d") for d in dates),
        "holiday_row_ids": "|".join(str(int(r["holiday_row_id"])) for r in rows),
        "types": "|".join(sorted({str(r["type"]) for r in rows})),
        "locales": "|".join(sorted({str(r["locale"]) for r in rows})),
        "locale_names": "|".join(sorted({str(r["locale_name"]) for r in rows})),
        "descriptions": "|".join(dict.fromkeys(str(r["description"]) for r in rows)),
        "city": str(rows[0]["city"]),
        "state": str(rows[0]["state"]),
        "concept_family": str(rows[0]["concept_family"]),
    }


def build_sales_daily(train: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[int, pd.Timestamp], float], dict[str, Any]]:
    daily = train.groupby(["store_nbr", "date"], as_index=False, sort=False)["sales"].sum()
    total_raw = float(train["sales"].sum())
    total_daily = float(daily["sales"].sum())
    last_sales_date = pd.Timestamp(daily["date"].max())
    first_sales_date = pd.Timestamp(daily["date"].min())
    sales_lookup = {
        (int(row.store_nbr), pd.Timestamp(row.date)): float(row.sales)
        for row in daily.itertuples(index=False)
    }
    audit = {
        "train_rows_loaded_for_c0": int(len(train)),
        "daily_store_rows": int(len(daily)),
        "first_sales_date": first_sales_date.strftime("%Y-%m-%d"),
        "last_sales_date": last_sales_date.strftime("%Y-%m-%d"),
        "raw_sales_sum": total_raw,
        "daily_grouped_sales_sum": total_daily,
        "sales_sum_abs_diff": abs(total_raw - total_daily),
    }
    return daily, sales_lookup, audit


def add_exclusions(
    store_occurrences: pd.DataFrame,
    expanded_days: pd.DataFrame,
    daily_sales: pd.DataFrame,
    sales_lookup: dict[tuple[int, pd.Timestamp], float],
    last_sales_date: pd.Timestamp,
) -> pd.DataFrame:
    occurrences = store_occurrences.copy()
    occurrences["in_eval_window"] = (occurrences["eval_date"] >= EVAL_START) & (
        occurrences["eval_date"] <= last_sales_date
    )
    occurrences["earthquake_excluded"] = occurrences["concept"].str.contains(EARTHQUAKE_RE) | occurrences[
        "descriptions"
    ].str.contains(EARTHQUAKE_RE)

    daily_by_store = {
        int(store): group.set_index("date")["sales"].sort_index()
        for store, group in daily_sales.groupby("store_nbr", sort=False)
    }

    closed_flags: list[bool] = []
    eval_sales_values: list[float] = []
    previous_means: list[float] = []
    closure_ratios: list[float | None] = []
    closure_missing_days: list[int] = []
    for row in occurrences.itertuples(index=False):
        series = daily_by_store.get(int(row.store_nbr))
        eval_sales = sales_lookup.get((int(row.store_nbr), pd.Timestamp(row.eval_date)), 0.0)
        start = pd.Timestamp(row.eval_date) - pd.Timedelta(days=28)
        end = pd.Timestamp(row.eval_date) - pd.Timedelta(days=1)
        if series is None:
            window = pd.Series(dtype="float64")
        else:
            window = series.loc[(series.index >= start) & (series.index <= end)]
        previous_mean = float(window.mean()) if len(window) else 0.0
        missing_days = int(28 - len(window))
        ratio = eval_sales / previous_mean if previous_mean > 0 else None
        is_closed = previous_mean > 0 and eval_sales < 0.10 * previous_mean
        closed_flags.append(bool(is_closed))
        eval_sales_values.append(float(eval_sales))
        previous_means.append(float(previous_mean))
        closure_ratios.append(ratio)
        closure_missing_days.append(missing_days)

    occurrences["eval_day_store_sales"] = eval_sales_values
    occurrences["previous_28d_mean_store_sales"] = previous_means
    occurrences["closure_ratio"] = closure_ratios
    occurrences["closure_missing_preceding_days"] = closure_missing_days
    occurrences["closure_excluded"] = closed_flags

    other_days: dict[int, list[tuple[pd.Timestamp, str]]] = defaultdict(list)
    for row in expanded_days.itertuples(index=False):
        other_days[int(row.store_nbr)].append((pd.Timestamp(row.date), str(row.concept)))

    overlap_flags: list[bool] = []
    overlap_details: list[str] = []
    for row in occurrences.itertuples(index=False):
        lower = pd.Timestamp(row.eval_date) - pd.Timedelta(days=OTHER_CONCEPT_WINDOW_BEFORE)
        upper = pd.Timestamp(row.eval_date) + pd.Timedelta(days=OTHER_CONCEPT_WINDOW_AFTER)
        hits = sorted(
            {
                f"{date.strftime('%Y-%m-%d')}::{concept}"
                for date, concept in other_days.get(int(row.store_nbr), [])
                if concept != row.concept and lower <= date <= upper
            }
        )
        overlap_flags.append(bool(hits))
        overlap_details.append("|".join(hits[:20]))

    occurrences["other_concept_window_start"] = occurrences["eval_date"] - pd.Timedelta(
        days=OTHER_CONCEPT_WINDOW_BEFORE
    )
    occurrences["other_concept_window_end"] = occurrences["eval_date"] + pd.Timedelta(
        days=OTHER_CONCEPT_WINDOW_AFTER
    )
    occurrences["other_concept_overlap_excluded"] = overlap_flags
    occurrences["other_concept_overlap_details"] = overlap_details

    reason_columns = [
        "earthquake_excluded",
        "closure_excluded",
        "other_concept_overlap_excluded",
    ]
    occurrences["passed_c0"] = occurrences["in_eval_window"] & ~occurrences[reason_columns].any(axis=1)
    reason_values: list[str] = []
    for row in occurrences.itertuples(index=False):
        reasons = []
        if not bool(row.in_eval_window):
            reasons.append("outside_eval_window")
        if bool(row.earthquake_excluded):
            reasons.append("earthquake")
        if bool(row.closure_excluded):
            reasons.append("closure_lt_10pct_prev28_mean")
        if bool(row.other_concept_overlap_excluded):
            reasons.append("other_concept_within_eval_window_pm3")
        reason_values.append("|".join(reasons))
    occurrences["exclusion_reasons"] = reason_values
    return occurrences


def build_event_details(occurrences: pd.DataFrame) -> pd.DataFrame:
    passed = occurrences[occurrences["passed_c0"]].copy()
    if passed.empty:
        return pd.DataFrame(
            columns=[
                "concept",
                "eval_date",
                "affected_store_count",
                "event_store_units",
                "types",
                "locales",
                "locale_names",
                "states",
                "cities",
                "concept_family",
                "year",
            ]
        )

    rows: list[dict[str, Any]] = []
    for (concept, eval_date), group in passed.groupby(["concept", "eval_date"], sort=True):
        rows.append(
            {
                "concept": concept,
                "eval_date": pd.Timestamp(eval_date).strftime("%Y-%m-%d"),
                "affected_store_count": int(group["store_nbr"].nunique()),
                "event_store_units": int(len(group)),
                "types": "|".join(sorted(set("|".join(group["types"]).split("|")))),
                "locales": "|".join(sorted(set("|".join(group["locales"]).split("|")))),
                "locale_names": "|".join(sorted(set("|".join(group["locale_names"]).split("|")))),
                "states": "|".join(sorted(group["state"].astype(str).unique())),
                "cities": "|".join(sorted(group["city"].astype(str).unique())),
                "concept_family": str(group["concept_family"].iloc[0]),
                "year": int(pd.Timestamp(eval_date).year),
            }
        )
    return pd.DataFrame(rows).sort_values(["eval_date", "concept"]).reset_index(drop=True)


def count_by_reason(occurrences: pd.DataFrame) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for reasons in occurrences["exclusion_reasons"].fillna(""):
        for reason in str(reasons).split("|"):
            if reason:
                counter[reason] += 1
    return dict(sorted(counter.items()))


def count_eval_exclusion_flags(occurrences: pd.DataFrame) -> dict[str, int]:
    in_eval = occurrences[occurrences["in_eval_window"]].copy()
    return {
        "earthquake": int(in_eval["earthquake_excluded"].sum()),
        "closure_lt_10pct_prev28_mean": int(in_eval["closure_excluded"].sum()),
        "other_concept_within_eval_window_pm3": int(in_eval["other_concept_overlap_excluded"].sum()),
        "any_store_level_exclusion_nonexclusive": int(
            (
                in_eval["earthquake_excluded"]
                | in_eval["closure_excluded"]
                | in_eval["other_concept_overlap_excluded"]
            ).sum()
        ),
    }


def event_level_stage_counts(occurrences: pd.DataFrame) -> dict[str, int]:
    in_eval = occurrences[occurrences["in_eval_window"]].copy()
    after_earthquake = in_eval[~in_eval["earthquake_excluded"]]
    after_closure = after_earthquake[~after_earthquake["closure_excluded"]]
    after_overlap = after_closure[~after_closure["other_concept_overlap_excluded"]]

    def unique_events(df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        return int(df[["concept", "eval_date"]].drop_duplicates().shape[0])

    return {
        "active_grouped_event_concept_date_in_eval_before_exclusions": unique_events(in_eval),
        "after_earthquake_exclusion_event_concept_date": unique_events(after_earthquake),
        "after_closure_exclusion_event_concept_date": unique_events(after_closure),
        "after_overlap_exclusion_event_concept_date": unique_events(after_overlap),
    }


def build_counts(
    concepts: pd.DataFrame,
    expanded: pd.DataFrame,
    occurrences: pd.DataFrame,
    event_details: pd.DataFrame,
    unmatched: list[dict[str, Any]],
    manifest: dict[str, Any],
    sales_audit: dict[str, Any],
) -> dict[str, Any]:
    final_event_count = int(len(event_details))
    final_concepts = int(event_details["concept"].nunique()) if not event_details.empty else 0
    final_local_regional_concepts = (
        int(
            event_details[
                event_details["locales"].str.contains("Local|Regional", regex=True, na=False)
            ]["concept"].nunique()
        )
        if not event_details.empty
        else 0
    )
    final_event_store_units = int(occurrences["passed_c0"].sum())
    family_members = (
        event_details[event_details["concept_family"] != ""]
        .groupby("concept_family")["concept"]
        .nunique()
        .to_dict()
        if not event_details.empty
        else {}
    )
    family_groups_ge2 = int(sum(1 for n in family_members.values() if n >= 2))

    thresholds = {
        "excluded_eval_event_concept_date_min": 40,
        "eval_distinct_concepts_min": 25,
        "local_or_regional_distinct_concepts_min": 10,
        "event_store_units_min": 300,
        "sibling_template_groups_ge2_members_min": 3,
    }
    observed = {
        "excluded_eval_event_concept_date": final_event_count,
        "eval_distinct_concepts": final_concepts,
        "local_or_regional_distinct_concepts": final_local_regional_concepts,
        "event_store_units": final_event_store_units,
        "sibling_template_groups_ge2_members": family_groups_ge2,
    }
    threshold_pass = {
        key: observed[key.replace("_min", "")] >= value
        for key, value in thresholds.items()
    }

    type_locale_distribution = (
        event_details.assign(type_locale=event_details["types"] + " x " + event_details["locales"])[
            "type_locale"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
        if not event_details.empty
        else {}
    )

    concept_counts = (
        event_details.groupby("concept")
        .agg(
            event_count=("eval_date", "count"),
            total_event_store_units=("event_store_units", "sum"),
            max_affected_store_count=("affected_store_count", "max"),
            years=("year", lambda x: "|".join(str(v) for v in sorted(set(x)))),
            locales=("locales", lambda x: "|".join(sorted(set("|".join(x).split("|"))))),
            concept_family=("concept_family", "first"),
        )
        .reset_index()
        .sort_values(["event_count", "concept"], ascending=[False, True])
        .to_dict(orient="records")
        if not event_details.empty
        else []
    )

    return {
        "status": "PASS_C0" if all(threshold_pass.values()) else "INCONCLUSIVE_FEW_EVENTS",
        "pass_all": bool(all(threshold_pass.values())),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "contract_scope": "C0 counting only; no model, GPU, install, fitting, or inference",
        "source": manifest,
        "sales_audit": sales_audit,
        "normalization": {
            "literal_prefixes_removed_in_sequence": list(PREFIXES),
            "numeric_suffix_regex": CONCEPT_SUFFIX_RE.pattern,
            "colon_detail_removed_after_prefix_and_suffix": True,
            "case_mismatched_prefix_rows": int(
                concepts["normalization_flags"].fillna("").str.contains("case_mismatched_prefix").sum()
            ),
            "ambiguous_rows_written": "C0_AMBIGUITIES.csv",
        },
        "event_definition": {
            "counting_unit": "event(concept,date), not store replications",
            "event_store_unit": "same store and same concept consecutive dates collapsed; eval_date is first date",
            "eval_date_min": EVAL_START.strftime("%Y-%m-%d"),
            "eval_date_max": sales_audit["last_sales_date"],
            "transferred_true": "excluded before application",
            "work_day": "excluded before application",
            "earthquake": "excluded from main analysis",
            "closure_rule": "eval day store total sales < 10% of preceding 28 calendar-day mean",
            "other_concept_rule": "other concept for the same store in [d-4,d+3], where W={d-1,d}",
            "other_concept_window_start_offset": -OTHER_CONCEPT_WINDOW_BEFORE,
            "other_concept_window_end_offset": OTHER_CONCEPT_WINDOW_AFTER,
        },
        "pre_application_counts": {
            "holiday_rows": int(len(concepts)),
            "transferred_rows": int(concepts["transferred"].sum()),
            "work_day_rows": int((concepts["type"] == "Work Day").sum()),
            "active_holiday_rows_after_transfer_workday": int(
                ((~concepts["transferred"]) & (concepts["type"] != "Work Day")).sum()
            ),
            "expanded_event_store_day_rows": int(len(expanded)),
            "unmatched_locale_rows": len(unmatched),
            "raw_duplicate_same_store_concept_date_flagged_rows_before_dedup": int(
                expanded.attrs.get("raw_duplicate_same_store_concept_date_flagged_rows", 0)
            ),
            "raw_duplicate_same_store_concept_date_rows_dropped": int(
                expanded.attrs.get("raw_duplicate_same_store_concept_date_rows_dropped", 0)
            ),
            "grouped_store_event_occurrences_all_dates": int(len(occurrences)),
            "grouped_store_event_occurrences_in_eval": int(occurrences["in_eval_window"].sum()),
        },
        "stage_counts_event_concept_date": event_level_stage_counts(occurrences),
        "store_occurrence_exclusion_reason_counts_nonexclusive": count_by_reason(occurrences),
        "eval_store_occurrence_exclusion_flag_counts_nonexclusive": count_eval_exclusion_flags(
            occurrences
        ),
        "observed": observed,
        "thresholds": thresholds,
        "threshold_pass": threshold_pass,
        "family_groups": {
            "distinct_concepts_by_family": family_members,
            "groups_with_at_least_2_concepts": family_groups_ge2,
            "note": "Provincializacion without 'de' is assigned to the Provincializacion family and documented in CONCEPTS.csv",
        },
        "type_locale_distribution_final_events": type_locale_distribution,
        "concept_counts_final": concept_counts,
        "data_concerns": [
            "C0 uses literal prefix normalization and flags case-mismatched prefixes; affected rows are Work Day rows excluded before application.",
            "Provincializacion Santa Elena lacks 'de'; it is retained in the Provincializacion sibling family for template-group counting.",
            "Closure and overlap are store-level exclusions, so one event(concept,date) may survive with fewer stores.",
        ],
    }


def write_csvs(
    paths: Paths,
    concepts: pd.DataFrame,
    occurrences: pd.DataFrame,
    event_details: pd.DataFrame,
) -> None:
    concepts.to_csv(paths.concepts, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)

    store_out = occurrences.copy()
    for col in [
        "eval_date",
        "group_end_date",
        "other_concept_window_start",
        "other_concept_window_end",
    ]:
        store_out[col] = pd.to_datetime(store_out[col]).dt.strftime("%Y-%m-%d")
    store_out = store_out.sort_values(["eval_date", "concept", "store_nbr"]).reset_index(drop=True)
    store_out.to_csv(paths.event_store_details, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)

    event_details.to_csv(paths.event_details, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)

    exclusions = store_out[store_out["exclusion_reasons"] != ""].copy()
    exclusions.to_csv(paths.exclusions, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)

    ambiguous = concepts[
        (concepts["normalization_flags"].fillna("") != "")
        | (concepts["concept"] == "Provincializacion Santa Elena")
    ].copy()
    ambiguous.to_csv(paths.ambiguous, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL)


def verify_outputs(paths: Paths, counts: dict[str, Any]) -> dict[str, Any]:
    event_details = pd.read_csv(paths.event_details)
    store_details = pd.read_csv(paths.event_store_details)
    concepts = pd.read_csv(paths.concepts)

    passed_stores = store_details[store_details["passed_c0"] == True]  # noqa: E712
    recomputed_event_count = int(event_details[["concept", "eval_date"]].drop_duplicates().shape[0])
    recomputed_store_units = int(len(passed_stores))
    recomputed_concepts = int(event_details["concept"].nunique()) if len(event_details) else 0
    recomputed_local_regional = (
        int(
            event_details[
                event_details["locales"].astype(str).str.contains("Local|Regional", regex=True, na=False)
            ]["concept"].nunique()
        )
        if len(event_details)
        else 0
    )
    recomputed_family_groups = (
        int(
            sum(
                n >= 2
                for n in event_details[event_details["concept_family"].fillna("") != ""]
                .groupby("concept_family")["concept"]
                .nunique()
            )
        )
        if len(event_details)
        else 0
    )

    observed = counts["observed"]
    checks = {
        "event_details_unique_concept_date": int(event_details.duplicated(["concept", "eval_date"]).sum()) == 0,
        "event_count_matches": recomputed_event_count == observed["excluded_eval_event_concept_date"],
        "store_units_matches": recomputed_store_units == observed["event_store_units"],
        "distinct_concepts_matches": recomputed_concepts == observed["eval_distinct_concepts"],
        "local_regional_concepts_matches": recomputed_local_regional
        == observed["local_or_regional_distinct_concepts"],
        "family_group_count_matches": recomputed_family_groups
        == observed["sibling_template_groups_ge2_members"],
        "all_final_events_within_eval_bounds": bool(
            (pd.to_datetime(event_details["eval_date"]) >= EVAL_START).all()
            and (
                pd.to_datetime(event_details["eval_date"])
                <= pd.Timestamp(counts["sales_audit"]["last_sales_date"])
            ).all()
        )
        if len(event_details)
        else True,
        "concept_rows_match_holiday_rows": int(len(concepts)) == counts["pre_application_counts"]["holiday_rows"],
        "no_transferred_or_workday_in_passed_rows": bool(
            ~passed_stores["types"].astype(str).str.contains("Work Day").any()
        ),
        "no_earthquake_in_passed_rows": bool(
            ~passed_stores["concept"].astype(str).str.contains(EARTHQUAKE_RE).any()
        ),
        "other_concept_bounds_documented_as_d_minus_4_to_d_plus_3": counts["event_definition"][
            "other_concept_window_start_offset"
        ]
        == -4
        and counts["event_definition"]["other_concept_window_end_offset"] == 3,
        "threshold_status_matches": (counts["status"] == "PASS_C0") == all(counts["threshold_pass"].values()),
    }

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "verification_scope": "C0 artifact self-check from written CSV/JSON outputs",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "recomputed_observed": {
            "excluded_eval_event_concept_date": recomputed_event_count,
            "eval_distinct_concepts": recomputed_concepts,
            "local_or_regional_distinct_concepts": recomputed_local_regional,
            "event_store_units": recomputed_store_units,
            "sibling_template_groups_ge2_members": recomputed_family_groups,
        },
        "artifact_sha256": {
            paths.concepts.name: sha256_file(paths.concepts),
            paths.event_details.name: sha256_file(paths.event_details),
            paths.event_store_details.name: sha256_file(paths.event_store_details),
            paths.exclusions.name: sha256_file(paths.exclusions),
            paths.ambiguous.name: sha256_file(paths.ambiguous),
            paths.event_counts.name: sha256_file(paths.event_counts),
        },
    }


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = Paths(
        concepts=RESULTS_DIR / "CONCEPTS.csv",
        event_counts=RESULTS_DIR / "EVENT_COUNTS.json",
        event_details=RESULTS_DIR / "C0_EVENT_DETAILS.csv",
        event_store_details=RESULTS_DIR / "C0_EVENT_STORE_DETAILS.csv",
        exclusions=RESULTS_DIR / "C0_EXCLUSIONS.csv",
        ambiguous=RESULTS_DIR / "C0_AMBIGUITIES.csv",
        verification=RESULTS_DIR / "C0_VERIFICATION.json",
    )

    holidays, stores, train, manifest = load_inputs()
    concepts = build_concepts(holidays)
    expanded, unmatched = expand_to_store_days(concepts, stores)
    store_occurrences = group_consecutive(expanded)
    daily_sales, sales_lookup, sales_audit = build_sales_daily(train)
    occurrences = add_exclusions(
        store_occurrences=store_occurrences,
        expanded_days=expanded,
        daily_sales=daily_sales,
        sales_lookup=sales_lookup,
        last_sales_date=pd.Timestamp(sales_audit["last_sales_date"]),
    )
    event_details = build_event_details(occurrences)
    counts = build_counts(
        concepts=concepts,
        expanded=expanded,
        occurrences=occurrences,
        event_details=event_details,
        unmatched=unmatched,
        manifest=manifest,
        sales_audit=sales_audit,
    )
    write_csvs(paths, concepts, occurrences, event_details)
    write_json(paths.event_counts, counts)
    verification = verify_outputs(paths, counts)
    write_json(paths.verification, verification)

    summary = {
        "status": counts["status"],
        "observed": counts["observed"],
        "threshold_pass": counts["threshold_pass"],
        "verification_all_checks_passed": verification["all_checks_passed"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
