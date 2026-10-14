from pathlib import Path
import json
import zipfile

import numpy as np
import pytest

import newdata


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return newdata.sha256_path(path)


def method_lock(tmp_path: Path) -> Path:
    path = tmp_path / "METHOD_LOCK.json"
    frozen_source = tmp_path / "frozen_method_source.py"
    frozen_source.write_text("METHOD = 'synthetic'\n", encoding="utf-8")
    write_json(
        path,
        {
            "status": "METHODS_LOCKED_BEFORE_NEW_RAW_DOWNLOAD",
            "created_unix": 1.0,
            "source_hashes": {str(frozen_source): sha(frozen_source)},
        },
    )
    return path


def appliances_zip(tmp_path: Path, rows: int = 1200) -> Path:
    path = tmp_path / "appliances.zip"
    columns = ["date", *newdata.APPLIANCE_CHANNELS, "rv1", "rv2"]
    lines = [",".join(columns)]
    start = np.datetime64("2020-01-01T00:00")
    for i in range(rows):
        timestamp = str(start + np.timedelta64(10 * i, "m")).replace("T", " ")
        values = [timestamp]
        for j, _ in enumerate(newdata.APPLIANCE_CHANNELS):
            if i in (0, 513) and j == 1:
                values.append("")
            else:
                values.append(f"{i + j * 0.1:.3f}")
        values.extend([f"{i * 0.01:.3f}", f"{i * 0.02:.3f}"])
        lines.append(",".join(values))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("energydata_complete.csv", "\n".join(lines))
    return path


def beijing_zip(tmp_path: Path, rows: int = 1200) -> Path:
    path = tmp_path / "beijing.zip"
    columns = ["No", "year", "month", "day", "hour", *newdata.BEIJING_CHANNELS, "wd", "station"]
    lines = [",".join(columns)]
    start = np.datetime64("2013-03-01T00:00")
    for i in range(rows):
        stamp = start + np.timedelta64(i, "h")
        text = str(stamp)
        date, hour_text = text.split("T")
        year, month, day = date.split("-")
        hour = hour_text.split(":")[0]
        values = [str(i + 1), year, month, day, str(int(hour))]
        for j, _ in enumerate(newdata.BEIJING_CHANNELS):
            values.append("" if (i == 20 and j == 0) else f"{100 + i * 0.01 + j:.3f}")
        values.extend(["NE", "Aotizhongxin"])
        lines.append(",".join(values))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("PRSA_Data_Aotizhongxin_20130301-20170228.csv", "\n".join(lines))
    return path


def model_lock(tmp_path: Path, fields: dict, selected=True, bad_model_sha=False) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "MODEL_SELECTION_LOCK.json"
    payload = {"status": "MODEL_SELECTION_LOCKED_BEFORE_EVAL_TARGET_OPEN", "newdata_contract": dict(fields)}
    if selected:
        model_path = tmp_path / "selected_model.pt"
        model_path.write_bytes(b"synthetic model artifact\n")
        payload["newdata_contract"]["selected_models"] = [
            {
                "name": "synthetic",
                "path": str(model_path),
                "sha256": "0" * 64 if bad_model_sha else sha(model_path),
            }
        ]
    write_json(path, payload)
    return path


def test_appliances_training_payload_excludes_eval_and_random_variables(tmp_path):
    lock = method_lock(tmp_path)
    source = appliances_zip(tmp_path)
    contract = newdata.prepare("appliances", lock, source_path=source, cache_root=tmp_path / "cache")
    data = newdata.load_training("appliances", lock, cache_root=tmp_path / "cache")

    assert data["dataset"] == "appliances"
    assert len(data["columns"]) == 26
    assert "rv1" not in data["columns"]
    assert "rv2" not in data["columns"]
    assert data["x"].shape[0] == data["tune_bounds"][1]
    assert set(data["origins"]) == {"fit", "tune"}
    assert data["eval_target_loaded"] is False
    assert Path(contract["payloads"]["eval_future_target"]["path"]).exists()
    assert data["source_hash"] == contract["source"]["sha256"]


def test_open_evaluation_requires_lock_fields_and_selected_models(tmp_path):
    lock = method_lock(tmp_path)
    source = appliances_zip(tmp_path)
    newdata.prepare("appliances", lock, source_path=source, cache_root=tmp_path / "cache")
    fields = newdata.required_model_selection_fields("appliances", lock, cache_root=tmp_path / "cache")

    no_model_lock = model_lock(tmp_path, fields, selected=False)
    with pytest.raises(ValueError, match="selected model"):
        newdata.open_evaluation("appliances", no_model_lock, cache_root=tmp_path / "cache")

    good_lock = model_lock(tmp_path, fields, selected=True)
    evaluation = newdata.open_evaluation("appliances", good_lock, cache_root=tmp_path / "cache")
    assert evaluation["history"].shape[1:] == (newdata.L, 26)
    assert evaluation["physical_history"].shape == evaluation["history"].shape
    for physical, mask, normalized in zip(evaluation["physical_history"], evaluation["history_observed"], evaluation["history"]):
        rebuilt = newdata.window_forward_fill_normalize(physical, mask, evaluation["mean"], evaluation["std"])
        np.testing.assert_array_equal(rebuilt, normalized)
    assert evaluation["target"].shape[1:] == (newdata.H, 26)
    assert np.all(np.diff(evaluation["origins"]) >= newdata.H)

    bad = dict(fields)
    bad["contract_sha256"] = "0" * 64
    bad_lock = model_lock(tmp_path / "bad", bad, selected=True)
    with pytest.raises(ValueError, match="contract_sha256"):
        newdata.open_evaluation("appliances", bad_lock, cache_root=tmp_path / "cache")


def test_training_and_required_fields_do_not_hash_eval_target_until_lock(tmp_path):
    lock = method_lock(tmp_path)
    source = appliances_zip(tmp_path)
    contract = newdata.prepare("appliances", lock, source_path=source, cache_root=tmp_path / "cache")
    target_path = Path(contract["payloads"]["eval_future_target"]["path"])
    target_path.write_bytes(b"tampered eval target bytes")

    data = newdata.load_training("appliances", lock, cache_root=tmp_path / "cache")
    fields = newdata.required_model_selection_fields("appliances", lock, cache_root=tmp_path / "cache")

    assert data["eval_target_loaded"] is False
    assert fields["eval_input_payload_sha256"] == contract["payloads"]["eval_past_input"]["sha256"]
    with pytest.raises(ValueError, match="Payload hash mismatch"):
        newdata.open_evaluation("appliances", model_lock(tmp_path, fields), cache_root=tmp_path / "cache")


def test_open_evaluation_rejects_missing_or_tampered_artifacts_and_sources(tmp_path):
    lock = method_lock(tmp_path)
    source = appliances_zip(tmp_path)
    newdata.prepare("appliances", lock, source_path=source, cache_root=tmp_path / "cache")
    fields = newdata.required_model_selection_fields("appliances", lock, cache_root=tmp_path / "cache")

    bad_model_lock = model_lock(tmp_path / "bad_model", fields, bad_model_sha=True)
    with pytest.raises(ValueError, match="Selected model artifact hash mismatch"):
        newdata.open_evaluation("appliances", bad_model_lock, cache_root=tmp_path / "cache")

    selected_lock = model_lock(tmp_path / "missing_model", fields)
    lock_payload = json.loads(selected_lock.read_text(encoding="utf-8"))
    missing_model_path = Path(lock_payload["newdata_contract"]["selected_models"][0]["path"])
    missing_model_path.unlink()
    selected_lock.write_text(json.dumps(lock_payload, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Selected model artifact missing"):
        newdata.open_evaluation("appliances", selected_lock, cache_root=tmp_path / "cache")

    source_path = Path(next(iter(fields["source_hashes"])))
    source_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Declared source hash mismatch"):
        newdata.open_evaluation("appliances", model_lock(tmp_path / "bad_source", fields), cache_root=tmp_path / "cache")


def test_registry_contract_hash_is_checked(tmp_path):
    lock = method_lock(tmp_path)
    source = appliances_zip(tmp_path)
    contract = newdata.prepare("appliances", lock, source_path=source, cache_root=tmp_path / "cache")
    Path(contract["payloads"]["fit_tune"]["path"]).with_name("contract.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="Registered contract JSON hash mismatch"):
        newdata.load_training("appliances", lock, cache_root=tmp_path / "cache")


def test_window_forward_fill_resets_to_fit_mean_not_prior_global_state():
    physical = np.array([[np.nan], [14.0], [np.nan]], dtype=np.float32)
    mask = np.array([[False], [True], [False]])
    mean = np.array([10.0], dtype=np.float32)
    std = np.array([2.0], dtype=np.float32)
    normalized = newdata.window_forward_fill_normalize(physical, mask, mean, std)
    np.testing.assert_allclose(normalized[:, 0], np.array([0.0, 2.0, 2.0], dtype=np.float32))


def test_beijing_nested_archive_preserves_fixed_station_contract(tmp_path):
    inner = beijing_zip(tmp_path)
    outer = tmp_path / "uci_download.zip"
    with zipfile.ZipFile(outer, "w") as archive:
        archive.writestr("PRSA_Data.zip", inner.read_bytes())
    physical, mask, columns, times, metadata = newdata._parse_source("beijing_aotizhongxin", outer)
    assert physical.shape == mask.shape == (1200, 11)
    assert columns == list(newdata.BEIJING_CHANNELS)
    assert len(times) == 1200
    assert "PRSA_Data.zip!" in metadata["source_member"]


def test_beijing_station_schema_and_eval_target_mask(tmp_path):
    lock = method_lock(tmp_path)
    source = beijing_zip(tmp_path)
    contract = newdata.prepare("beijing_aotizhongxin", lock, source_path=source, cache_root=tmp_path / "cache")
    data = newdata.load_training("beijing_aotizhongxin", lock, cache_root=tmp_path / "cache")

    assert data["columns"] == list(newdata.BEIJING_CHANNELS)
    assert data["interval_seconds"] == 3600
    assert contract["spec"]["station"] == "Aotizhongxin"
    assert contract["source"]["parse"]["missing_values_total"] >= 1

    fields = newdata.required_model_selection_fields("beijing_aotizhongxin", lock, cache_root=tmp_path / "cache")
    evaluation = newdata.open_evaluation(
        "beijing_aotizhongxin", model_lock(tmp_path, fields), cache_root=tmp_path / "cache"
    )
    assert evaluation["observed"].dtype == np.bool_
    assert evaluation["target_physical"].shape == evaluation["target"].shape

