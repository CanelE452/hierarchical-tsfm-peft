"""Append predeclared derived routes without changing the initial registry."""
import csv
import hashlib
import json
from pathlib import Path
import time
import uuid


def require_pre_selection(public_root, operation):
    path = Path(public_root) / "SELECTION_LOCK.json"
    if not path.is_file():
        return
    lock = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(lock, dict) or not isinstance(lock.get("locked"), bool):
        raise ValueError("Malformed final selection lock; training/registration must remain closed")
    if lock["locked"]:
        raise ValueError(f"Final selection is locked: {operation} requires a separate new campaign or explicitly separated post-T development; preserve the existing selection and T results")


def register(public_root, proposed_rows, reason):
    root = Path(public_root)
    path = root / "EXPERIMENT_REGISTRY.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    existing = {row["experiment_id"]: row for row in rows}
    additions = []
    for proposed in proposed_rows:
        row = {key: str(proposed.get(key, "")) for key in fields}
        run_id = row["experiment_id"]
        if run_id in existing:
            keys = ("dataset", "panel", "fold", "block", "family", "num_channels", "latent_k", "seed", "length", "horizon")
            if any(row[key] != existing[run_id][key] for key in keys):
                raise ValueError("An existing experiment identity cannot be changed")
            continue
        if not run_id or Path(run_id).name != run_id or row["execution_status"] != "PENDING_QUEUE":
            raise ValueError("Only a safe, not-yet-executed new route can be registered")
        if run_id in {v["experiment_id"] for v in additions}:
            raise ValueError("Duplicate proposed extension")
        additions.append(row)
    if not additions:
        return []
    require_pre_selection(root, "new registry additions")
    stamp = str(time.time_ns()) + "_" + uuid.uuid4().hex
    extension = root / ("EXPERIMENT_REGISTRY_EXTENSION_" + stamp + ".csv")
    with extension.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(additions)
    with extension.with_suffix(".json").open("x", encoding="utf-8") as handle:
        json.dump(dict(registered_unix=time.time(), reason=reason, run_ids=[r["experiment_id"] for r in additions],
                       csv_sha256=hashlib.sha256(extension.read_bytes()).hexdigest(), before_execution=True, before_T_selection=True,
                       existing_rows_preserved=True, status="REGISTERED_BEFORE_NEW_EXECUTION"), handle, indent=2)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows + additions)
    return additions
