"""Preservation and scope tests with isolated synthetic files."""
import json
from pathlib import Path
import subprocess
import sys

from . import common

def test_write_guard_blocks_existing_mutations_but_allows_new_temp(tmp_path):
    report = tmp_path / "report"
    report.mkdir()
    frozen = tmp_path / "frozen.txt"
    frozen.write_text("unchanged", encoding="utf-8")
    baseline = {"files": [{"path": str(frozen), "bytes": frozen.stat().st_size,
                           "sha256": common.sha(frozen)}]}
    (report / "PRESERVATION_BASELINE.json").write_text(json.dumps(baseline), encoding="utf-8")
    program = """
import sys
from pathlib import Path
from research.comparative_ablation_followup_20261027_v1 import common
common.REPORT=Path(sys.argv[1])
frozen=Path(sys.argv[2])
common.install_write_guard()
for action in (lambda:frozen.write_text('bad'),lambda:frozen.unlink(),
               lambda:frozen.rename(frozen.with_name('moved.txt'))):
    try: action()
    except PermissionError: pass
    else: raise AssertionError('frozen mutation was allowed')
fresh=frozen.with_name('new-temp.txt')
fresh.write_text('allowed')
fresh.unlink()
assert frozen.read_text()=='unchanged'
"""
    completed = subprocess.run([sys.executable, "-B", "-c", program, str(report), str(frozen)],
                               cwd=common.REPO, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert frozen.read_text() == "unchanged"

def test_new_writer_excludes_original_namespace_and_preserves_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "REPORT", tmp_path / "new-report")
    monkeypatch.setattr(common, "CACHE", tmp_path / "new-cache")
    target = common.REPORT / "receipt.json"
    common.write_new(target, {"first": True})
    import pytest
    with pytest.raises(FileExistsError):
        common.write_new(target, {"first": False})
    with pytest.raises(ValueError):
        common.write_new(tmp_path / "old-report" / "receipt.json", {})
    assert common.read(target) == {"first": True}

def test_all_required_seeds_and_priorities_are_in_registered_selection():
    jobs = common.selected_jobs()
    assert len(jobs) == 50
    assert [row["method_id"] for row in jobs] == ["A4"] * 40 + ["A1"] * 10
    for index, seed in enumerate(common.SEEDS):
        group = jobs[index * 4:index * 4 + 4]
        assert [x["dataset"] for x in group] == list(common.DATASETS)
        assert all(x["seed"] == seed and not x["extended"] for x in group)
    assert all(x["dataset"] == "jena" and x["extended"] for x in jobs[40:])

