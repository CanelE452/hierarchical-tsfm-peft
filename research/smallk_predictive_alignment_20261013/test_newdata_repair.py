from pathlib import Path
import io
import json
import zipfile

import numpy as np
import pytest

import newdata
import newdata_repair


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return newdata.sha256_path(path)


def method_lock(tmp_path: Path) -> Path:
    source = tmp_path / "frozen_source.py"
    source.write_text("SOURCE='repair-test'\n", encoding="utf-8")
    path = tmp_path / "METHOD_LOCK.json"
    write_json(
        path,
        {
            "status": "METHODS_LOCKED_BEFORE_NEW_RAW_DOWNLOAD",
            "created_unix": 1.0,
            "source_hashes": {str(source): sha(source)},
        },
    )
    return path


def wrong_stock_style_zip(tmp_path: Path) -> Path:
    path = tmp_path / "wrong_beijing.zip"
    lines = ["Date,Open,High,Low,Close,Adj Close,Volume"]
    for i in range(4):
        lines.append(f"2018-07-0{i+1},{i+1},{i+2},{i},{i+1.5},{i+1.5},{1000+i}")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data.csv", "\n".join(lines))
        archive.writestr("test.csv", "\n".join(lines))
    return path


def prsa_zip(tmp_path: Path, rows: int = 1200) -> Path:
    path = tmp_path / "uci_packaged_501.zip"
    header = ["No", "year", "month", "day", "hour", *newdata.BEIJING_CHANNELS, "wd", "station"]
    start = np.datetime64("2013-03-01T00:00")

    def member(station: str) -> str:
        lines = [",".join(header)]
        for i in range(rows):
            stamp = start + np.timedelta64(i, "h")
            date, hour_text = str(stamp).split("T")
            year, month, day = date.split("-")
            hour = str(int(hour_text.split(":")[0]))
            values = [str(i + 1), year, month, day, hour]
            for j, _ in enumerate(newdata.BEIJING_CHANNELS):
                values.append("" if (station == "Aotizhongxin" and i == 15 and j == 0) else f"{10 + i * 0.1 + j:.3f}")
            values.extend(["NE", station])
            lines.append(",".join(values))
        return "\n".join(lines)

    nested_bytes = io.BytesIO()
    with zipfile.ZipFile(nested_bytes, "w") as nested:
        nested.writestr("PRSA_Data_20130301-20170228/PRSA_Data_Aotizhongxin_20130301-20170228.csv", member("Aotizhongxin"))
        nested.writestr("PRSA_Data_20130301-20170228/PRSA_Data_Changping_20130301-20170228.csv", member("Changping"))
    stock_lines = ["Date,Open,High,Low,Close,Adj Close,Volume"]
    for i in range(503):
        stock_lines.append(f"2018-07-{(i % 28) + 1:02d},{i+1},{i+2},{i},{i+1.5},{i+1.5},{1000+i}")
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("PRSA2017_Data_20130301-20170228.zip", nested_bytes.getvalue())
        archive.writestr("data.csv", "\n".join(stock_lines))
        archive.writestr("test.csv", "\n".join(stock_lines))
    return path


def test_current_wrong_artifact_fixture_is_rejected(tmp_path):
    lock = method_lock(tmp_path)
    with pytest.raises(ValueError, match="No PRSA member"):
        newdata_repair.load_training(
            "beijing_aotizhongxin",
            lock,
            source_path=wrong_stock_style_zip(tmp_path),
            cache_root=tmp_path / "repair_cache",
        )


def test_repair_wrapper_accepts_prsa_aotizhongxin_and_isolates_cache(tmp_path):
    lock = method_lock(tmp_path)
    source = prsa_zip(tmp_path)
    data = newdata_repair.load_training(
        "beijing_aotizhongxin",
        lock,
        source_path=source,
        cache_root=tmp_path / "repair_cache",
    )
    assert data["columns"] == list(newdata.BEIJING_CHANNELS)
    assert data["interval_seconds"] == 3600
    assert data["source"].endswith("uci_packaged_501.zip")
    contract = newdata_repair.get_contract("beijing_aotizhongxin", lock, cache_root=tmp_path / "repair_cache")
    assert contract["source"]["parse"]["selected_member_count"] == 1
    assert contract["source"]["parse"]["skipped_top_level_stock_csv"] == ["data.csv", "test.csv"]
    fields = newdata_repair.required_model_selection_fields(
        "beijing_aotizhongxin", lock, cache_root=tmp_path / "repair_cache"
    )
    assert fields["contract_sha256"] == contract["contract_sha256"]
    assert data["x"].shape[0] == data["tune_bounds"][1]
    assert set(data["origins"]) == {"fit", "tune"}


def test_repair_load_evaluation_inputs_returns_original_physical_history(tmp_path):
    lock = method_lock(tmp_path)
    source = prsa_zip(tmp_path)
    newdata_repair.load_training(
        "beijing_aotizhongxin",
        lock,
        source_path=source,
        cache_root=tmp_path / "repair_cache",
    )
    eval_inputs = newdata_repair.load_evaluation_inputs(
        "beijing_aotizhongxin", lock, cache_root=tmp_path / "repair_cache"
    )
    assert eval_inputs["history"].shape[1:] == (newdata.L, len(newdata.BEIJING_CHANNELS))
    assert eval_inputs["history_physical"].shape == eval_inputs["history"].shape

    import pandas as pd

    with zipfile.ZipFile(source) as archive:
        with zipfile.ZipFile(io.BytesIO(archive.read("PRSA2017_Data_20130301-20170228.zip"))) as nested:
            frame = pd.read_csv(nested.open("PRSA_Data_20130301-20170228/PRSA_Data_Aotizhongxin_20130301-20170228.csv"))
    physical = frame[list(newdata.BEIJING_CHANNELS)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    origin = int(eval_inputs["origins"][0])
    np.testing.assert_allclose(eval_inputs["history_physical"][0], physical[origin - newdata.L : origin], equal_nan=True)


def test_repair_metadata_names_parser_fix_and_cache():
    metadata = newdata_repair.repair_metadata()
    assert metadata["repair_kind"] == "select_per_member_aotizhongxin_prsa_schema"
    assert "load_evaluation_inputs" in metadata["public_api"]

