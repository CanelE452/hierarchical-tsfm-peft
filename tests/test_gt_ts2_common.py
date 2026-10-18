from pathlib import Path
import importlib
import json

import numpy as np
import pytest


def common():
    try:
        return importlib.import_module('research.gt_ts2_fastcheck_20261018_v1.common')
    except ModuleNotFoundError:
        pytest.fail('new campaign immutable artifacts and budget functions are missing')


def test_sealed_receipt_cannot_overwrite_failed_attempt(tmp_path):
    c = common()
    path = tmp_path / 'failure.json'
    c.seal_json(path, {'status': 'FAILED'}, allowed_root=tmp_path)
    with pytest.raises(FileExistsError):
        c.seal_json(path, {'status': 'COMPLETE'}, allowed_root=tmp_path)
    assert json.loads(path.read_text())['status'] == 'FAILED'


def test_artifact_writer_rejects_escape(tmp_path):
    c = common()
    with pytest.raises(ValueError):
        c.seal_json(tmp_path.parent / 'old.json', {}, allowed_root=tmp_path)


def test_public_json_rejects_nonfinite(tmp_path):
    c = common()
    with pytest.raises(ValueError):
        c.seal_json(tmp_path / 'nan.json', {'x': float('nan')}, allowed_root=tmp_path)
    assert not (tmp_path / 'nan.json').exists()


def test_npz_is_exclusive_and_array_hash_binds_order(tmp_path):
    c = common()
    x = np.arange(24).reshape(2, 3, 4)
    path = tmp_path / 'private.npz'
    c.seal_npz(path, allowed_root=tmp_path, x=x)
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['x'], x)
    with pytest.raises(FileExistsError):
        c.seal_npz(path, allowed_root=tmp_path, x=x[::-1])
    assert c.array_sha(x) != c.array_sha(x[::-1])


def test_uniform_update_decision_uses_time_and_reserves_only():
    c = common()
    assert c.choose_updates(1., 0., 6000., 0.)['updates'] == 300
    assert c.choose_updates(2., 500., 6000., 100.)['updates'] == 100
    assert c.choose_updates(8., 1000., 6000., 100.)['updates'] is None
    # GPU reserve and eight matched arms both matter.
    assert c.choose_updates(1., 4000., 6000., 0.)['updates'] is None
    assert c.choose_updates(1., 0., 9000., 0.)['updates'] is None


def test_budget_counts_current_process_time_and_storage():
    c = common()
    assert c.budget_status(100., 50., 20., 30., 1)['GPU_used_seconds'] == 50.
    with pytest.raises(c.BudgetExceeded):
        c.budget_status(0., 10801., 0., 0., 1)
    with pytest.raises(c.BudgetExceeded):
        c.budget_status(0., 0., 5300., 101., 1)
    with pytest.raises(c.BudgetExceeded):
        c.budget_status(0., 0., 0., 0., 6 * 1024**3 + 1)
