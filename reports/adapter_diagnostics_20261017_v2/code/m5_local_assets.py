"""Verify approved M5 file hashes and header schemas without reading numeric rows."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

FILENAMES = ("sales_train_validation.csv", "sales_train_evaluation.csv", "calendar.csv", "sell_prices.csv")
IDS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
CALENDAR = ["date", "wm_yr_wk", "weekday", "wday", "month", "year", "d", "event_name_1", "event_type_1", "event_name_2", "event_type_2", "snap_CA", "snap_TX", "snap_WI"]
PRICES = ["store_id", "item_id", "wm_yr_wk", "sell_price"]
INVENTORY_REFERENCE = "cache://m5_local_assets/INVENTORY.json"
SOURCE_REFERENCE = "repo://reports/adapter_diagnostics_20261017_v2/code/m5_local_assets.py"
ACCESS_SCOPE = "HASH_BYTES_AND_CSV_FIRST_HEADER_LINE_ONLY"


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def _observe_csv(path, filename):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError("Missing approved M5 source: "+filename)
    with path.open("rb") as stream:
        first = stream.readline(1024*1024)
    if not first or not first.endswith(b"\n"):
        raise ValueError("Unsupported M5 first header line: "+filename)
    fields = next(csv.reader([first.decode("utf-8-sig").rstrip("\r\n")], strict=True))
    if filename.startswith("sales_train_"):
        days = 1913 if filename == FILENAMES[0] else 1941
        expected = IDS+[f"d_{number}" for number in range(1, days+1)]
    elif filename == "calendar.csv":
        expected = CALENDAR
    elif filename == "sell_prices.csv":
        expected = PRICES
    else:
        raise ValueError("Unapproved M5 source filename")
    if fields != expected:
        raise ValueError("M5 header schema differs: "+filename)
    result = dict(filename=filename, sha256=sha(path), bytes=path.stat().st_size,
                  header_fields=fields, header_field_count=len(fields))
    if filename.startswith("sales_train_"):
        result["day_column_range"] = dict(first="d_1", last=f"d_{days}", count=days)
    return result


def _typed_entry(value, filename):
    if not isinstance(value, dict) or not {"absolute_path", "sha256", "bytes"}.issubset(value):
        raise ValueError("Invalid typed M5 inventory entry: "+filename)
    path = Path(value["absolute_path"])
    digest, size = value["sha256"], value["bytes"]
    if (not path.is_absolute() or path.name != filename or not isinstance(digest, str)
            or len(digest) != 64 or any(letter not in "0123456789abcdef" for letter in digest)
            or not isinstance(size, int) or isinstance(size, bool) or size < 0):
        raise ValueError("Invalid typed M5 inventory path/hash/bytes: "+filename)
    return path


def verify_local_m5_assets(ctx, discovery=None):
    if discovery is None:
        discovery = json.loads((Path(ctx["report"])/"M5_LOCAL_ASSET_DISCOVERY.json").read_text(encoding="utf-8"))
    if (discovery.get("schema") != "M5_LOCAL_ASSET_DISCOVERY_v1"
            or discovery.get("status") != "LOCAL_ASSETS_SCHEMA_VERIFIED"
            or discovery.get("new_numeric_values_parsed") != 0
            or discovery.get("no_M5_evaluation") is not True or discovery.get("no_fitting") is not True):
        raise ValueError("Invalid M5 public discovery scope/schema")
    binding = discovery.get("inventory_binding", {})
    if binding.get("reference") != INVENTORY_REFERENCE:
        raise ValueError("Unapproved M5 inventory reference")
    inventory_path = Path(ctx["cache"])/"m5_local_assets/INVENTORY.json"
    if not inventory_path.is_file() or inventory_path.is_symlink() or sha(inventory_path) != binding.get("sha256"):
        raise ValueError("Inventory SHA binding rejected")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if (inventory.get("schema") != "M5_LOCAL_ASSET_INVENTORY_v1"
            or inventory.get("approved_access_scope") != ACCESS_SCOPE
            or not isinstance(inventory.get("files"), dict) or set(inventory["files"]) != set(FILENAMES)):
        raise ValueError("Invalid typed M5 inventory schema/approved file set")
    public = discovery.get("source_files", [])
    if (not isinstance(public, list) or len(public) != 4
            or [item.get("filename") for item in public] != list(FILENAMES)):
        raise ValueError("M5 public discovery file set/order differs")
    source = discovery.get("verifier_source")
    if source is not None and source != dict(reference=SOURCE_REFERENCE, sha256=sha(__file__)):
        raise ValueError("M5 verifier source SHA binding rejected")
    records = []
    for item in public:
        filename = item["filename"]
        entry = inventory["files"][filename]
        actual = _observe_csv(_typed_entry(entry, filename), filename)
        if actual["bytes"] != entry["bytes"]:
            raise ValueError("Source bytes binding rejected: "+filename)
        if actual["sha256"] != entry["sha256"]:
            raise ValueError("Source SHA binding rejected: "+filename)
        if actual != item:
            raise ValueError("M5 public source/header binding rejected: "+filename)
        records.append(actual)
    archive = inventory.get("archive", {})
    archive_name = archive.get("filename")
    if archive_name != "m5-forecasting-accuracy.zip":
        raise ValueError("Unapproved M5 archive filename")
    archive_path = _typed_entry(archive, archive_name)
    if not archive_path.is_file() or archive_path.is_symlink():
        raise FileNotFoundError("Missing approved M5 archive")
    actual_archive = dict(filename=archive_name, sha256=sha(archive_path), bytes=archive_path.stat().st_size)
    if (actual_archive["sha256"] != archive["sha256"] or actual_archive["bytes"] != archive["bytes"]
            or actual_archive != discovery.get("archive")):
        raise ValueError("M5 archive SHA/bytes binding rejected")
    return dict(status="LOCAL_ASSETS_SCHEMA_VERIFIED", reverified_utc=datetime.now(timezone.utc).isoformat(),
        inventory_binding=dict(binding), verifier_source=dict(reference=SOURCE_REFERENCE, sha256=sha(__file__)),
        checked_csv_files=4, checked_archives=1, source_files=records, archive=actual_archive,
        csv_header_lines_parsed=4, new_numeric_values_parsed=0, archive_contents_inspected=False,
        no_M5_evaluation=True, no_fitting=True, model_calls=0, foundation_calls=0, new_downloads=0,
        source_file_modifications=0, research_splits="NOT_REGISTERED", license="UNKNOWN",
        source_provenance="UNKNOWN_NOT_INDEPENDENTLY_VERIFIED", read_scope=ACCESS_SCOPE)
