"""M5 discovery verifies only approved file bytes and one CSV header line."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1]/"reports/adapter_diagnostics_20261017_v2/code/m5_local_assets.py"
SPEC = importlib.util.spec_from_file_location("m5_local_assets", SOURCE)
m5 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m5)

IDS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
CALENDAR = ["date", "wm_yr_wk", "weekday", "wday", "month", "year", "d", "event_name_1", "event_type_1", "event_name_2", "event_type_2", "snap_CA", "snap_TX", "snap_WI"]
PRICES = ["store_id", "item_id", "wm_yr_wk", "sell_price"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class M5AssetTests(unittest.TestCase):
    def fixture(self, root):
        cache, report, assets = root/"private", root/"report", root/"approved"
        cache.mkdir(); report.mkdir(); assets.mkdir()
        headers = {"sales_train_validation.csv": IDS+[f"d_{i}" for i in range(1, 1914)],
                   "sales_train_evaluation.csv": IDS+[f"d_{i}" for i in range(1, 1942)],
                   "calendar.csv": CALENDAR, "sell_prices.csv": PRICES}
        entries, public = {}, []
        for name, fields in headers.items():
            path = assets/name
            path.write_bytes((",".join(fields)+'\n"deliberately unclosed body; must never parse\n').encode())
            entries[name] = dict(absolute_path=str(path.resolve()), sha256=digest(path), bytes=path.stat().st_size)
            item = dict(filename=name, sha256=digest(path), bytes=path.stat().st_size,
                        header_fields=fields, header_field_count=len(fields))
            if name.startswith("sales_"):
                item["day_column_range"] = dict(first="d_1", last=fields[-1], count=len(fields)-6)
            public.append(item)
        archive = assets/"m5-forecasting-accuracy.zip"
        archive.write_bytes(b"opaque archive bytes: do not inspect or decompress")
        archive_entry = dict(filename=archive.name, absolute_path=str(archive.resolve()), sha256=digest(archive), bytes=archive.stat().st_size)
        inventory = dict(schema="M5_LOCAL_ASSET_INVENTORY_v1", approved_access_scope="HASH_BYTES_AND_CSV_FIRST_HEADER_LINE_ONLY",
                         files=entries, archive=archive_entry)
        inventory_path = cache/"m5_local_assets/INVENTORY.json"
        inventory_path.parent.mkdir(); inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
        discovery = dict(schema="M5_LOCAL_ASSET_DISCOVERY_v1", status="LOCAL_ASSETS_SCHEMA_VERIFIED",
                         inventory_binding=dict(reference="cache://m5_local_assets/INVENTORY.json", sha256=digest(inventory_path)),
                         source_files=public, archive={key:archive_entry[key] for key in ("filename", "sha256", "bytes")},
                         new_numeric_values_parsed=0, no_M5_evaluation=True, no_fitting=True)
        (report/"M5_LOCAL_ASSET_DISCOVERY.json").write_text(json.dumps(discovery), encoding="utf-8")
        return dict(cache=cache, report=report), discovery, assets, inventory_path

    def test_valid_headers_and_opaque_body_are_verified_without_numeric_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            ctx, discovery, _, _ = self.fixture(Path(directory))
            value = m5.verify_local_m5_assets(ctx)
            self.assertEqual(value["status"], "LOCAL_ASSETS_SCHEMA_VERIFIED")
            self.assertEqual(value["checked_csv_files"], 4)
            self.assertEqual(value["checked_archives"], 1)
            self.assertEqual(value["source_files"], discovery["source_files"])
            self.assertEqual(value["new_numeric_values_parsed"], 0)
            self.assertTrue(value["no_M5_evaluation"] and value["no_fitting"])

    def test_source_or_inventory_tamper_is_rejected_before_rebinding(self):
        with tempfile.TemporaryDirectory() as directory:
            ctx, discovery, assets, inventory = self.fixture(Path(directory))
            (assets/"calendar.csv").write_bytes((assets/"calendar.csv").read_bytes()+b"tamper")
            with self.assertRaisesRegex(ValueError, "Source.*SHA|Source.*bytes"):
                m5.verify_local_m5_assets(ctx, discovery)
            inventory.write_bytes(inventory.read_bytes()+b" ")
            with patch.object(m5, "_observe_csv") as observe:
                with self.assertRaisesRegex(ValueError, "Inventory SHA"):
                    m5.verify_local_m5_assets(ctx, discovery)
                observe.assert_not_called()

    def test_missing_approved_file_or_wrong_header_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            ctx, discovery, assets, inventory_path = self.fixture(Path(directory))
            (assets/"calendar.csv").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "calendar.csv"):
                m5.verify_local_m5_assets(ctx, discovery)
            path = assets/"calendar.csv"
            path.write_bytes(b"date,wrong_column\n")
            inventory = json.loads(inventory_path.read_text())
            inventory["files"]["calendar.csv"].update(sha256=digest(path), bytes=path.stat().st_size)
            inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
            discovery["inventory_binding"]["sha256"] = digest(inventory_path)
            with self.assertRaisesRegex(ValueError, "header schema"):
                m5.verify_local_m5_assets(ctx, discovery)


if __name__ == "__main__":
    unittest.main()
