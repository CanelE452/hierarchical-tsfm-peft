import hashlib

import pytest

from test_verify_past_constant_gate_report_v1 import timers, sample_row, write_rows
import verify_past_constant_gate_report_v2 as verifier


def reference(tmp_path):
    path = tmp_path / "audited.json"; path.write_text('{"status":"PASS"}', encoding="utf-8")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def test_same_semantic_reference_with_extra_correct_bytes(tmp_path):
    item = reference(tmp_path)
    assert verifier.same_bound_record({**item, "bytes": 17}, item)


def test_wrong_extra_bytes_cannot_pass(tmp_path):
    item = reference(tmp_path)
    assert not verifier.same_bound_record({**item, "bytes": 18}, item)


def test_changed_sha_or_path_is_rejected(tmp_path):
    item = reference(tmp_path)
    assert not verifier.same_bound_record({**item, "sha256": "invalid"}, item)
    assert not verifier.same_bound_record({**item, "path": str(tmp_path / "other.json")}, item)


def test_changed_bound_file_is_rejected(tmp_path):
    item = reference(tmp_path); (tmp_path / "audited.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="BOUND_FILE_SHA_CHANGED"): verifier.same_bound_record(item, item)


def test_v2_still_recomputes_timer_and_checks_every_csv_leaf(tmp_path, monkeypatch):
    row = timers(); assert verifier.timer_statistics(row, row) == (1.0, 1.0)
    expected = sample_row(); path = tmp_path / "scores.csv"; write_rows(path, [expected])
    monkeypatch.setitem(verifier.COUNTS, "SCORES", 1)
    assert verifier.check_csv(path, "SCORES", [expected]) == 10
    altered = {**expected, "mse": .26}; write_rows(path, [altered])
    with pytest.raises(ValueError, match="CSV_SOURCE_VALUE_MISMATCH"): verifier.check_csv(path, "SCORES", [expected])
