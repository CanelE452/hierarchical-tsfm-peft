import csv
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


MODULE = "research.b2_dataset_search_20261019_v1.data_safe_etth2"
DATASET = "ETTh2_safe17400"


def api():
    return importlib.import_module(MODULE)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def source(tmp_path):
    original = importlib.import_module("research.b2_dataset_search_20261019_v1.data")
    asset = tmp_path / "assets/hierarchical-tsfm-peft"
    asset.mkdir(parents=True)
    spec = original.DATASETS["ETTh2"]
    locations = original.source_locations(asset, "ETTh2")
    path = locations["source"]
    path.parent.mkdir(parents=True)
    first = np.datetime64(spec["first_timestamp"], "s")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(spec["header"])
        for row in range(13920):
            stamp = str(first + np.timedelta64(row, "h")).replace("T", " ")
            values = [str((row % 73) + col / 10) for col in range(7)]
            if row == 20:
                values[1] = ""
            writer.writerow([stamp, *values])
        handle.write("FORBIDDEN_PROTECTED_TAIL\n")
    record = dict(raw_bytes=path.stat().st_size, raw_sha256="historical_metadata_only",
        schema=dict(n_rows=17420, header=list(spec["header"]), target_columns=list(spec["columns"])),
        time_axis=dict(first_timestamp=spec["first_timestamp"], last_timestamp=spec["last_timestamp"],
                       all_hourly_diffs=True, duplicate_timestamps=0))
    write_json(locations["metadata"][0], dict(source=record))
    write_json(locations["metadata"][1], dict(datasets=dict(ett_small=dict(files=[dict(
        name="ETTh2.csv", bytes=path.stat().st_size, sha256="historical_metadata_only")]))))
    repo = tmp_path / "repo"
    amendment = repo / "reports/b2_dataset_search_20261019_v1/PREPARATION_AMENDMENT_001.json"
    replacement = dict(dataset=DATASET, source_original_rows=17420, cohort_rows=17400, prefix_rows=13920,
        A_fit=[0, 8700], A_stop=[8700, 10440], A60=[0, 10440], M_cal=[10440, 12180],
        V_select=[12180, 13920], T_final_within_cohort=[13920, 17400], source_tail_outside_cohort=[17400, 17420],
        training_recipe_change=False, forecast_result_used_to_choose_cutoff=False)
    write_json(amendment, dict(status="REGISTERED_BEFORE_ANY_NEW_GPU_TRAINING_OR_FORECAST_SCORES",
        original_candidate="ETTh2", source_original_rows=17420, original_prepared_prefix_rows=13936,
        per_file_historical_ETTh2_TEST_start=13936,
        same_station_ETTm2_protected_TEST_start_15minute_rows=55680,
        same_station_earliest_hourly_cutoff=13920,
        earliest_protected_cutoff_timestamp="2018-02-01 00:00:00", replacement=replacement))
    return dict(asset=asset, path=path, locations=locations, repo=repo, amendment=amendment,
                output=tmp_path / "prepared")


def prepare(source, monkeypatch):
    mod = api()
    monkeypatch.setattr(mod, "REPO", source["repo"])
    return mod.prepare_dataset(source["asset"], DATASET, source["output"], amendment_path=source["amendment"])


def test_exact_safe_prefix_and_clock_and_consumer_contract(source, monkeypatch):
    original_open = Path.open
    consumed = []

    class Guarded:
        def __init__(self, handle): self.handle = handle
        def __enter__(self): self.handle.__enter__(); return self
        def __exit__(self, *args): return self.handle.__exit__(*args)
        def readline(self, *args):
            line = self.handle.readline(*args)
            assert b"FORBIDDEN" not in line, "A protected record was requested"
            consumed.append(line)
            return line

    def guarded(path, *args, **kwargs):
        assert not (path.name == "PREFIX_PHYSICAL.npz" and path.parent.name == "ETTh2"), "Old prefix access"
        handle = original_open(path, *args, **kwargs)
        if path == source["path"]:
            assert args == ("rb",)
            return Guarded(handle)
        return handle

    monkeypatch.setattr(Path, "open", guarded)
    result = prepare(source, monkeypatch)
    assert result["status"] == "READY"
    assert result["original_rows"] == result["source_rows"] == 17420
    assert result["cohort_rows"] == 17400 and result["prefix_rows"] == 13920
    assert result["split_ranges"] == dict(A_fit=[0,8700], A_stop=[8700,10440], A60=[0,10440],
        M_cal=[10440,12180], V_select=[12180,13920], T_final=[13920,17400])
    assert len(consumed) == 13921
    assert result["source_prefix_consumed_records"] == 13920
    assert result["source_prefix_consumed_sha256"] == hashlib.sha256(b"".join(consumed)).hexdigest()
    assert result["source_full_file_hashed"] is False
    assert result["source_dataset"] == "ETTh2"
    assert result["protection_boundaries"]["per_file_historical_TEST_start"] == 13936
    assert result["protection_boundaries"]["same_station_protected_hourly_start"] == 13920
    assert result["protection_boundaries"]["value_equivalence_claimed"] is False
    assert result["amendment_binding"]["sha256"] == hashlib.sha256(source["amendment"].read_bytes()).hexdigest()
    assert result["amendment_binding"]["reference"].startswith("repo://reports/")
    assert "research/b2_dataset_search_20261019_v1/data.py" in result["dependency_source_hashes"]
    prefix = source["output"] / DATASET / "PREFIX_PHYSICAL.npz"
    config_path = prefix.with_name("DATA_DIAGNOSTIC.json")
    with np.load(prefix, allow_pickle=False) as payload:
        assert set(payload.files) == {"physical", "observed", "times", "columns"}
        assert payload["physical"].dtype == np.float32 and payload["physical"].shape == (13920, 7)
        assert payload["observed"].dtype == bool
        assert payload["times"].dtype == np.int64
        assert payload["columns"].dtype.kind == "U"
        assert np.isnan(payload["physical"][20, 1]) and not payload["observed"][20, 1]
        assert payload["times"][-1] == np.datetime64("2018-01-31T23:00", "ns").astype(np.int64)
    train = importlib.import_module("research.b2_dataset_search_20261019_v1.train")
    config, task = train.load_task(config_path, prefix)
    assert config["cohort_rows"] == 17400
    assert task["dataset"] == DATASET and task["split_ranges"]["T_final"] == (13920, 17400)
    assert task["metadata"]["T_access"] is False


def test_original_admission_and_cohort_parse_use_distinct_row_counts(source, monkeypatch):
    mod = api()
    original_admit, original_parse = mod.base._admit_csv, mod.base._csv_prefix
    calls = []

    def admit(asset_root, dataset, locations, spec):
        calls.append(("admit", dataset, spec["rows"]))
        return original_admit(asset_root, dataset, locations, spec)

    def parse(path, dataset, spec):
        calls.append(("parse", dataset, spec["rows"]))
        return original_parse(path, dataset, spec)

    monkeypatch.setattr(mod.base, "_admit_csv", admit)
    monkeypatch.setattr(mod.base, "_csv_prefix", parse)
    prepare(source, monkeypatch)
    assert calls == [("admit", "ETTh2", 17420), ("parse", "ETTh2", 17400)]
    assert mod.base.DATASETS["ETTh2"]["rows"] == 17420


@pytest.mark.parametrize("change", ["source_rows", "cohort_rows", "prefix_rows", "cutoff", "scope"])
def test_bad_amendment_rejected_before_source_read(source, monkeypatch, change):
    value = json.loads(source["amendment"].read_text())
    if change == "source_rows": value["replacement"]["source_original_rows"] = 17400
    elif change == "cohort_rows": value["replacement"]["cohort_rows"] = 17420
    elif change == "prefix_rows": value["replacement"]["prefix_rows"] = 13936
    elif change == "cutoff": value["earliest_protected_cutoff_timestamp"] = "2018-02-01 16:00:00"
    else: value["replacement"]["forecast_result_used_to_choose_cutoff"] = True
    write_json(source["amendment"], value)
    original_open = Path.open
    def guarded(path, *args, **kwargs):
        assert path != source["path"], "CSV read before amendment admission"
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises(ValueError, match="amendment"):
        prepare(source, monkeypatch)
    assert not source["output"].exists()


def test_original_metadata_mismatch_rejected_before_numeric_read(source, monkeypatch):
    proof = source["locations"]["metadata"][0]
    value = json.loads(proof.read_text())
    value["source"]["schema"]["n_rows"] = 17400
    write_json(proof, value)
    original_open = Path.open
    def guarded(path, *args, **kwargs):
        assert path != source["path"], "CSV read before original N admission"
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises(ValueError, match="metadata"):
        prepare(source, monkeypatch)


def test_safe_clock_end_is_enforced(source, monkeypatch):
    mod = api()
    original_parse = mod.base._csv_prefix
    def changed(path, dataset, spec):
        physical, observed, times, columns, provenance = original_parse(path, dataset, spec)
        times += np.timedelta64(1, "h").astype("timedelta64[ns]").astype(np.int64)
        return physical, observed, times, columns, provenance
    monkeypatch.setattr(mod.base, "_csv_prefix", changed)
    with pytest.raises(ValueError, match="clock"):
        prepare(source, monkeypatch)
    assert not source["output"].exists()


def test_existing_safe_prefix_is_preserved(source, monkeypatch):
    prepare(source, monkeypatch)
    with pytest.raises(FileExistsError):
        prepare(source, monkeypatch)


def test_unsupported_id_rejected(source):
    with pytest.raises(ValueError, match="Unsupported dataset"):
        api().prepare_dataset(source["asset"], "ETTh2", source["output"], amendment_path=source["amendment"])
