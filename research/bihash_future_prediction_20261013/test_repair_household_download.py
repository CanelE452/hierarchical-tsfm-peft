from __future__ import annotations

from pathlib import Path
import json
import sys

import pytest

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import download_new
import repair_household_download as repair


def test_plus_and_percent_encoded_paths_are_same_static_resource() -> None:
    proof = repair.prove_same_static_resource(
        "https://archive.ics.uci.edu/static/public/235/individual%2Bhousehold%2Belectric%2Bpower%2Bconsumption.zip",
        "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip",
    )
    assert proof["status"] == "SAME_STATIC_RESOURCE"
    assert proof["expected"]["decoded_path"] == proof["locked"]["decoded_path"]
    assert " " not in proof["locked"]["decoded_path"]
    assert proof["normalization"].endswith("unquote_plus is intentionally not used")


@pytest.mark.parametrize(
    "bad_url",
    [
        "https://example.com/static/public/235/individual+household+electric+power+consumption.zip",
        "https://archive.ics.uci.edu/static/public/235/individual-household-electric-power-consumption.zip",
        "https://archive.ics.uci.edu/static/public/274/sml2010.zip",
        "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip?x=1",
    ],
)
def test_changed_host_path_or_query_is_rejected(bad_url: str) -> None:
    with pytest.raises(ValueError):
        repair.prove_same_static_resource(
            "https://archive.ics.uci.edu/static/public/235/individual%2Bhousehold%2Belectric%2Bpower%2Bconsumption.zip",
            bad_url,
        )


def test_process_local_override_restores_frozen_dataset_constant() -> None:
    original = download_new.DATASETS[repair.DATASET]["download_url"]
    locked = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
    with repair._process_local_download_url(repair.DATASET, locked):
        assert download_new.DATASETS[repair.DATASET]["download_url"] == locked
    assert download_new.DATASETS[repair.DATASET]["download_url"] == original


def test_repair_uses_locked_path_and_process_local_url(monkeypatch, tmp_path: Path) -> None:
    locked_path = tmp_path / "raw" / "uci235_household_hourly" / "source_v1.zip"
    lock = {
        "locked": True,
        "newdata_sources": {
            repair.DATASET: {
                "path": str(locked_path),
                "url": "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip",
            }
        },
    }
    method_lock = tmp_path / "METHOD_LOCK.json"
    method_lock.write_text(json.dumps(lock), encoding="utf-8")

    monkeypatch.setattr(download_new, "raw_dir", lambda dataset, cache_root=None: locked_path.parent)

    captured: dict[str, object] = {}

    def fake_download_one(dataset, cache_root=None, timeout=120.0, method_lock_path=None):
        captured["dataset"] = dataset
        captured["method_lock_path"] = Path(method_lock_path)
        captured["url_seen"] = download_new.DATASETS[repair.DATASET]["download_url"]
        return {
            "status": "DOWNLOADED",
            "dataset": repair.DATASET,
            "artifact": {"path": str(locked_path), "sha256": "x", "bytes": 1},
        }

    monkeypatch.setattr(download_new, "download_one", fake_download_one)
    before = repair._source_hashes()
    result = repair.repair_household_download(method_lock)

    assert result["status"] == "COMPLETE"
    assert captured["dataset"] == repair.DATASET
    assert captured["method_lock_path"] == method_lock.resolve()
    assert captured["url_seen"] == lock["newdata_sources"][repair.DATASET]["url"]
    assert repair._source_hashes() == before
