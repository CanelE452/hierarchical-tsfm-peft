import csv
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


MODULE = "research.b2_dataset_search_20261019_v1.data"


def api():
    return importlib.import_module(MODULE)


def test_search_data_preparer_exists():
    try:
        spec = importlib.util.find_spec(MODULE)
    except ModuleNotFoundError:
        spec = None
    assert spec is not None, "The new bounded data preparer is missing"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def csv_source(tmp_path, request):
    mod = api()
    dataset = getattr(request, "param", "ETTh2")
    spec = mod.DATASETS[dataset]
    asset = tmp_path / "hierarchical-tsfm-peft"
    asset.mkdir()
    locations = mod.source_locations(asset, dataset)
    path = locations["source"]
    path.parent.mkdir(parents=True, exist_ok=True)
    end = spec["rows"] * 4 // 5
    first = np.datetime64(spec["first_timestamp"], "s")
    interval = np.timedelta64(spec["interval_seconds"], "s")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter=spec["delimiter"], lineterminator="\n")
        writer.writerow(spec["header"])
        for index in range(end):
            stamp = str(first + index * interval)
            values = [str((index % 97) + channel / 10) for channel in range(len(spec["columns"]))]
            if dataset == "uci360_air_quality":
                date, clock = stamp.split("T")
                year, month, day = date.split("-")
                values = [v.replace(".", ",") for v in values]
                if index >= spec["rows"] // 2:
                    values[2] = "-200"
                writer.writerow([day + "/" + month + "/" + year, clock.replace(":", "."), *values, "", ""])
            else:
                writer.writerow([stamp.replace("T", " "), *values])
        handle.write("FORBIDDEN_PROTECTED_TAIL_MUST_NOT_BE_PARSED\n")
    if dataset.startswith("ETTh"):
        source = {"raw_bytes": path.stat().st_size, "raw_sha256": "historical_only",
                  "schema": {"n_rows": spec["rows"], "header": list(spec["header"]), "target_columns": list(spec["columns"])},
                  "time_axis": {"first_timestamp": spec["first_timestamp"], "last_timestamp": spec["last_timestamp"],
                                "all_hourly_diffs": True, "duplicate_timestamps": 0}}
        proof = {"source": source} if dataset == "ETTh2" else {"sources": {"ETTh1": source}}
        manifest = {"datasets": {"ett_small": {"files": [{"name": dataset + ".csv", "bytes": path.stat().st_size,
                    "sha256": "historical_only"}]}}}
        write_json(locations["metadata"][0], proof)
        write_json(locations["metadata"][1], manifest)
    else:
        proof = {"datasets": {"uci_air_quality": {"rows": spec["rows"], "columns": list(spec["header"][:-2]),
                  "first_timestamp": spec["first_timestamp"], "last_timestamp": spec["last_timestamp"],
                  "timestamp_parse_errors": 0, "duplicate_timestamps": 0,
                  "hour_gap_seconds_distribution": {"3600": spec["rows"] - 1},
                  "full_hour_grid_missing_between_first_last": 0}}}
        acquisition = {"dataset_id": 360, "current_raw_files": [{"path": "data_external/uci_air_quality/raw/AirQualityUCI.csv",
                       "bytes": path.stat().st_size, "sha256": "historical_only"}]}
        write_json(locations["metadata"][0], proof)
        write_json(locations["metadata"][1], acquisition)
    return dict(asset=asset, dataset=dataset, spec=spec, path=path, locations=locations,
                output=tmp_path / "prepared", rows=end)


def test_csv_reads_exact_prefix_without_requesting_one_tail_record(csv_source, monkeypatch):
    mod = api()
    original = Path.open
    reads = []

    class Guarded:
        def __init__(self, wrapped): self.wrapped = wrapped
        def __enter__(self): self.wrapped.__enter__(); return self
        def __exit__(self, *args): return self.wrapped.__exit__(*args)
        def readline(self, *args):
            value = self.wrapped.readline(*args)
            assert b"FORBIDDEN" not in value, "A protected tail record was read"
            reads.append(value)
            return value

    def guarded(path, *args, **kwargs):
        handle = original(path, *args, **kwargs)
        if path == csv_source["path"]:
            assert args and args[0] == "rb"
            return Guarded(handle)
        return handle

    monkeypatch.setattr(Path, "open", guarded)
    result = mod.prepare_dataset(csv_source["asset"], csv_source["dataset"], csv_source["output"])
    assert result["status"] == "READY"
    assert len(reads) == csv_source["rows"] + 1
    assert result["source_prefix_consumed_sha256"] == hashlib.sha256(b"".join(reads)).hexdigest()
    assert result["source_prefix_consumed_records"] == csv_source["rows"]
    assert result["source_full_file_hashed"] is False
    with np.load(csv_source["output"] / csv_source["dataset"] / "PREFIX_PHYSICAL.npz", allow_pickle=False) as z:
        assert set(z.files) == {"physical", "observed", "times", "columns"}
        assert z["physical"].shape == (csv_source["rows"], 7)
        assert z["physical"].dtype == np.float32 and z["observed"].dtype == bool
        assert z["times"].dtype == np.int64


@pytest.mark.parametrize("csv_source", ["ETTh1"], indirect=True)
def test_etth1_support_uses_its_own_metadata(csv_source):
    result = api().prepare_dataset(csv_source["asset"], csv_source["dataset"], csv_source["output"])
    assert result["status"] == "READY" and result["dataset"] == "ETTh1"
    assert result["original_rows"] == 17420 and result["C"] == 7


@pytest.mark.parametrize("csv_source", ["uci360_air_quality"], indirect=True)
def test_air_decimal_sentinel_and_astop_unobserved_preserve_full_columns(csv_source):
    result = api().prepare_dataset(csv_source["asset"], csv_source["dataset"], csv_source["output"])
    assert result["status"] == "NOT_READY_UNOBSERVED_CHANNEL"
    assert result["C"] == 13
    assert result["role_observability"]["A_stop"]["unobserved_columns"] == ["NMHC(GT)"]
    assert result["qualification_scope"] == "A_fit_and_A_stop_observability_only"
    assert result["readiness_issues"] == [{"role": "A_stop", "unobserved_columns": ["NMHC(GT)"]}]
    with np.load(csv_source["output"] / csv_source["dataset"] / "PREFIX_PHYSICAL.npz", allow_pickle=False) as z:
        assert np.isnan(z["physical"][5000, 2]) and not z["observed"][5000, 2]
        assert z["physical"][0, 1] == np.float32(.1)
        assert z["physical"].shape[1] == 13
    assert result["source_full_file_hashed"] is False


def test_metadata_wrong_n_rejected_before_csv_read(csv_source, monkeypatch):
    path = csv_source["locations"]["metadata"][0]
    value = json.loads(path.read_text())
    value["source"]["schema"]["n_rows"] += 1
    write_json(path, value)
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert path != csv_source["path"], "Source opened before metadata admission"
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    with pytest.raises(ValueError, match="metadata"):
        api().prepare_dataset(csv_source["asset"], csv_source["dataset"], csv_source["output"])


def test_changed_csv_header_is_rejected(csv_source):
    with csv_source["path"].open("r+b") as f:
        f.write(b"Xate")
    with pytest.raises(ValueError, match="header"):
        api().prepare_dataset(csv_source["asset"], csv_source["dataset"], csv_source["output"])


def test_existing_prepared_data_is_not_overwritten(csv_source):
    mod = api()
    mod.prepare_dataset(csv_source["asset"], csv_source["dataset"], csv_source["output"])
    with pytest.raises(FileExistsError):
        mod.prepare_dataset(csv_source["asset"], csv_source["dataset"], csv_source["output"])


def test_v_only_missingness_does_not_exclude_dataset(csv_source):
    mod = api()
    spec = csv_source["spec"]
    ranges = mod.split_ranges(spec["rows"])
    x = np.ones((ranges["V_select"][1], 7), np.float32)
    mask = np.ones_like(x, bool)
    x[ranges["V_select"][0]:, 0] = np.nan
    mask[ranges["V_select"][0]:, 0] = False
    roles = mod.role_observability(x, mask, list(spec["columns"]), ranges)
    assert roles["V_select"]["unobserved_columns"] == ["HUFL"]
    assert mod.readiness_issues(roles) == []


def test_unrecognized_dataset_refused(tmp_path):
    with pytest.raises(ValueError, match="Unsupported dataset"):
        api().prepare_dataset(tmp_path, "Jena", tmp_path / "output")


@pytest.fixture
def appliances_source(tmp_path):
    mod = api()
    dataset = "appliances"
    spec = mod.DATASETS[dataset]
    asset = tmp_path / "hierarchical-tsfm-peft"
    asset.mkdir()
    locations = mod.source_locations(asset, dataset)
    locations["source"].parent.mkdir(parents=True)
    n, c = spec["rows"], len(spec["columns"])
    end = n * 4 // 5
    values = (np.arange(end, dtype=np.float32)[:, None] % 41 + np.arange(c, dtype=np.float32)[None] / 7)
    labels = (np.datetime64("2016-01-11T17:00", "s") + np.arange(end) * np.timedelta64(600, "s")).astype(str)
    np.savez_compressed(locations["source"], physical=values, mask=np.ones_like(values, bool),
        x=np.full_like(values, np.inf), filled=np.full_like(values, np.inf), mean=np.full(c, np.nan, np.float32),
        std=np.full(c, np.nan, np.float32), origins_fit=np.array([512]), origins_tune=np.array([n * 3 // 5]),
        columns=np.array(spec["columns"]), times=labels)
    contract = dict(dataset=dataset, source=dict(parse=dict(rows=n, columns=list(spec["columns"]))),
        spec=dict(channels=list(spec["columns"]), excluded_columns=["date", "rv1", "rv2"], interval_seconds=600),
        bounds=dict(fit=[0, n * 3 // 5], tune=[n * 3 // 5, end], eval=[end, n]),
        quality=dict(domain_leakage_checks=dict(training_payload_contains_eval_rows=False)),
        payloads=dict(fit_tune=dict(path=str(locations["source"]), sha256=hashlib.sha256(locations["source"].read_bytes()).hexdigest()),
                      eval_future_target=dict(path="FORBIDDEN_EVAL.npz", sha256="forbidden")))
    write_json(locations["metadata"][0], contract)
    return dict(asset=asset, spec=spec, values=values, locations=locations, contract=contract, output=tmp_path / "prepared")


def test_appliances_uses_original_physical_without_normalized_or_eval_arrays(appliances_source, monkeypatch):
    source = appliances_source
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert "FORBIDDEN" not in str(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", guarded)
    result = api().prepare_dataset(source["asset"], "appliances", source["output"])
    assert result["status"] == "READY" and result["prefix_rows"] == 15788
    assert result["physical_inverse_transform_required"] is False
    assert result["original_normalized_arrays_consumed"] is False
    with np.load(source["output"] / "appliances/PREFIX_PHYSICAL.npz", allow_pickle=False) as z:
        np.testing.assert_array_equal(z["physical"], source["values"])


def test_appliances_bad_header_refused_before_numeric_load_or_payload_hash(appliances_source, monkeypatch):
    source = appliances_source
    mod = api()
    values = np.concatenate([source["values"], source["values"][:1]])
    with np.load(source["locations"]["source"], allow_pickle=False) as z:
        arrays = {key: z[key].copy() for key in z.files}
    arrays["physical"] = values
    np.savez_compressed(source["locations"]["source"], **arrays)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("numeric load of a tail-bearing prefix"))
    monkeypatch.setattr(mod, "sha256_file", lambda *a, **k: pytest.fail("payload hash before header admission"))
    with pytest.raises(ValueError, match="prefix header"):
        mod.prepare_dataset(source["asset"], "appliances", source["output"])


def test_appliances_contract_cannot_redirect_to_eval(appliances_source):
    source = appliances_source
    source["contract"]["payloads"]["fit_tune"]["path"] = str(source["asset"] / "FORBIDDEN_EVAL.npz")
    write_json(source["locations"]["metadata"][0], source["contract"])
    with pytest.raises(ValueError, match="FIT_TUNE path"):
        api().prepare_dataset(source["asset"], "appliances", source["output"])
