import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


MODULE = "research.b2_dataset_extension_20261018_v1.data"


def api():
    return importlib.import_module(MODULE)


def test_data_module_is_available():
    try:
        spec = importlib.util.find_spec(MODULE)
    except ModuleNotFoundError:
        spec = None
    assert spec is not None, "The protected-prefix data preparer is missing"


@pytest.fixture
def source(tmp_path, request):
    mod = api()
    dataset = getattr(request, "param", "uci274_sml2010")
    spec = mod.DATASETS[dataset]
    n = spec["rows"]
    end = n * 4 // 5
    columns = list(spec["columns"])
    rng = np.random.default_rng(123)
    physical = rng.normal(size=(end, len(columns))).astype(np.float32)
    physical[:, -1] = 7
    physical[23:39, 2] = np.nan
    observed = np.isfinite(physical)
    times = (np.datetime64("2012-01-01T00:00", "ns") + np.arange(end) * np.timedelta64(spec["interval_seconds"], "s"))
    labels = times.astype("datetime64[s]").astype(str)
    if dataset == "uci274_sml2010":
        labels = np.char.add(labels, "Z")
    root = tmp_path / "assets"
    folder = root / ".cache" / mod.SOURCE_CAMPAIGN / dataset / mod.SOURCE_LOCK
    folder.mkdir(parents=True)
    payload = folder / "FIT_TUNE_payload.npz"

    def write_payload(**changes):
        values = dict(physical=physical, mask=observed, times=labels,
                      columns=np.array(columns, dtype="U64"),
                      mean=np.full(len(columns), np.nan, dtype=np.float32),
                      std=np.full(len(columns), np.nan, dtype=np.float32))
        values.update(changes)
        np.savez_compressed(payload, **values)
        return hashlib.sha256(payload.read_bytes()).hexdigest()

    contract = {
        "dataset": dataset, "L": 512, "H": 48, "C": len(columns), "columns": columns,
        "parser_metadata": {spec["rows_field"]: n},
        "bounds": {"fit": [0, n * 3 // 5], "tune": [n * 3 // 5, end], "eval": [end, n]},
        "row_split": "floor 60/20/20 by post-parser rows",
        "history_contract": "causal forward fill is reset to FIT mean at each L512 window start",
        "target_contract": "future target is not forward filled; missing cells are NaN in physical payload and 0 with bool mask in normalized payload",
        "eval_separation": "FIT_TUNE, EVAL past input, and EVAL future target are separate payloads",
        "payloads": {
            "fit_tune": {"path": str(payload.relative_to(root)), "sha256": write_payload(), "bytes": payload.stat().st_size},
            "eval_future_target": {"path": "NEVER_OPEN_EVAL.npz", "sha256": "forbidden"},
        },
        "source_records": [{"role": "raw_source", "path": "NEVER_OPEN_RAW.zip", "sha256": "forbidden"}],
    }
    path = folder / "contract.json"

    def seal(**changes):
        if changes:
            contract["payloads"]["fit_tune"]["sha256"] = write_payload(**changes)
            contract["payloads"]["fit_tune"]["bytes"] = payload.stat().st_size
        path.write_text(json.dumps(contract), encoding="utf-8")

    seal()
    return dict(root=root, dataset=dataset, physical=physical, observed=observed,
                labels=labels, times=times.astype(np.int64), columns=columns,
                payload=payload, contract=contract, contract_path=path, seal=seal,
                output=tmp_path / "new_data")


def test_preparer_preserves_physical_missingness_and_only_opens_prefix(source, monkeypatch):
    mod = api()
    opened = []
    original = Path.open

    def guarded(path, *args, **kwargs):
        opened.append(str(path))
        assert "NEVER_OPEN" not in str(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    result = mod.prepare_dataset(source["root"], source["dataset"], source["output"])
    loaded = mod.load_prepared(source["output"] / source["dataset"])
    np.testing.assert_array_equal(loaded["physical"], source["physical"])
    np.testing.assert_array_equal(loaded["observed"], source["observed"])
    np.testing.assert_array_equal(loaded["times"], source["times"])
    assert loaded["columns"].tolist() == source["columns"]
    assert result["prefix_only_approved"] is True
    assert result["prefix_rows"] == source["physical"].shape[0]
    assert result["train_structure"]["constant_columns"] == [source["columns"][-1]]
    assert result["train_structure"]["scope"] == "A_fit_only"
    assert not any("NEVER_OPEN" in p for p in opened)
    assert not any("mean" in k or "std" in k for k in loaded if k != "metadata")


def test_structure_diagnostics_ignore_every_post_afit_numeric_value(source):
    mod = api()
    first = mod.prepare_dataset(source["root"], source["dataset"], source["output"])
    poisoned = source["physical"].copy()
    poisoned[mod.DATASETS[source["dataset"]]["rows"] // 2:] *= 1000
    source["seal"](physical=poisoned)
    second = mod.prepare_dataset(source["root"], source["dataset"], source["output"].with_name("other_data"))
    assert first["train_structure"] == second["train_structure"]


def test_exact_split_and_origin_containment():
    mod = api()
    ranges = mod.split_ranges(4784)
    assert ranges == {"A_fit": [0, 2392], "A_stop": [2392, 2870], "A60": [0, 2870],
                      "M_cal": [2870, 3348], "V_select": [3348, 3827], "T_final": [3827, 4784]}
    for name in ("A_fit", "A_stop", "A60", "V_select"):
        origins = mod.legal_origins(ranges[name])
        assert np.all(origins >= max(512, ranges[name][0]))
        assert np.all(origins + 48 <= ranges[name][1])
        assert np.all(np.diff(origins) == 48)


def test_contract_cannot_redirect_to_eval_payload(source, monkeypatch):
    mod = api()
    source["contract"]["payloads"]["fit_tune"]["path"] = "NEVER_OPEN_EVAL.npz"
    source["seal"]()
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("numeric load before prefix admission"))
    with pytest.raises(ValueError, match="FIT_TUNE path"):
        mod.prepare_dataset(source["root"], source["dataset"], source["output"])


def test_tail_rejected_by_headers_before_numeric_load_or_hash(source, monkeypatch):
    mod = api()
    physical = np.concatenate([source["physical"], source["physical"][:1]])
    source["seal"](physical=physical)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("numeric load with forbidden tail"))
    real_hash = mod.sha256_file
    def guarded_hash(path):
        assert Path(path) != source["payload"], "full payload hash before header admission"
        return real_hash(path)
    monkeypatch.setattr(mod, "sha256_file", guarded_hash)
    with pytest.raises(ValueError, match="prefix header"):
        mod.prepare_dataset(source["root"], source["dataset"], source["output"])


def test_wrong_numeric_header_rejected(source, monkeypatch):
    mod = api()
    source["seal"](mask=source["observed"].astype(np.int8))
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("numeric load with invalid schema"))
    with pytest.raises(ValueError, match="prefix header"):
        mod.prepare_dataset(source["root"], source["dataset"], source["output"])


@pytest.mark.parametrize("corruption", ["clock", "missingness", "column"])
def test_source_semantics_rejected(source, corruption):
    mod = api()
    if corruption == "clock":
        labels = source["labels"].copy()
        labels[10] = labels[9]
        source["seal"](times=labels)
    elif corruption == "missingness":
        mask = source["observed"].copy()
        mask[0, 0] = False
        source["seal"](mask=mask)
    else:
        cols = np.array(source["columns"], dtype="U64")
        cols[0] = "Day_of_week"
        source["seal"](columns=cols)
    with pytest.raises(ValueError):
        mod.prepare_dataset(source["root"], source["dataset"], source["output"])


def test_payload_fingerprint_mismatch_rejected(source):
    mod = api()
    source["contract"]["payloads"]["fit_tune"]["sha256"] = "0" * 64
    source["seal"]()
    with pytest.raises(ValueError, match="fingerprint"):
        mod.prepare_dataset(source["root"], source["dataset"], source["output"])


def test_existing_outputs_are_never_replaced(source):
    mod = api()
    mod.prepare_dataset(source["root"], source["dataset"], source["output"])
    with pytest.raises(FileExistsError):
        mod.prepare_dataset(source["root"], source["dataset"], source["output"])


def test_prepared_payload_must_match_bound_fingerprint(source):
    mod = api()
    mod.prepare_dataset(source["root"], source["dataset"], source["output"])
    folder = source["output"] / source["dataset"]
    with (folder / "PREFIX_PHYSICAL.npz").open("ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="fingerprint"):
        mod.load_prepared(folder)


def test_unknown_dataset_refused_without_filesystem_access(tmp_path):
    with pytest.raises(ValueError, match="Unsupported dataset"):
        api().prepare_dataset(tmp_path, "jena", tmp_path / "output")


@pytest.mark.parametrize("source", ["uci235_household_hourly"], indirect=True)
def test_household_hourly_clock_and_boundary(source):
    mod = api()
    result = mod.prepare_dataset(source["root"], source["dataset"], source["output"])
    loaded = mod.load_prepared(source["output"] / source["dataset"])
    assert result["prefix_rows"] == 27669
    assert result["interval_seconds"] == 3600
    assert result["split_ranges"]["A_fit"] == [0, 17293]
    assert "timezone unspecified" in result["clock"]
    np.testing.assert_array_equal(loaded["times"], source["times"])


def test_unobserved_train_channel_is_marked_not_ready(source):
    mod = api()
    physical = source["physical"].copy()
    physical[:mod.DATASETS[source["dataset"]]["rows"] // 2, 0] = np.nan
    source["seal"](physical=physical, mask=np.isfinite(physical))
    result = mod.prepare_dataset(source["root"], source["dataset"], source["output"])
    assert result["status"] == "NOT_READY_TRAIN_STRUCTURE"
    assert result["train_structure"]["unobserved_columns"] == [source["columns"][0]]
