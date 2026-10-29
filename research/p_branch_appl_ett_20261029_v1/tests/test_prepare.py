"""Preparation binding checks, using only synthetic input files."""
import json
from pathlib import Path

import pytest

from research.p_branch_appl_ett_20261029_v1 import prepare, runtime


def make_inputs(tmp_path, monkeypatch):
    source, asset = tmp_path/'historical', tmp_path/'asset'
    monkeypatch.setattr(runtime, 'DEFAULT_SOURCE', source)
    monkeypatch.setattr(runtime, 'DEFAULT_ASSET', asset)
    monkeypatch.setattr(runtime, 'MODEL_REL', 'model')
    paths = [source/'reports/k_tau_rule_20261023_v1/CURRENT_DATA_FINGERPRINTS.json',
             asset/'model/config.json', asset/'model/weights.bin', tmp_path/'prefix.npz']
    for dataset, axes in prepare.AXES.items():
        paths.extend([source/'reports/b2_dataset_search_20261019_v1/datasets'/dataset/'TRAIN_IDENTITY.json',
                      source/'.cache/k_tau_rule_20261023_v1/results'/f'KTAU_main_{dataset}_K{axes["K"]}_s92603.json'])
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'synthetic opaque bytes')
    return paths


def test_external_bindings_cover_prefix_provenance_and_foundation(tmp_path, monkeypatch):
    paths = make_inputs(tmp_path, monkeypatch)
    audit = dict(bindings={str(tmp_path/'prefix.npz'): runtime.sha(tmp_path/'prefix.npz')})
    actual = prepare.external_input_bindings(audit)
    assert {Path(row['path']) for row in actual['files']} == {p.resolve() for p in paths}
    assert all(row['sha256'] == runtime.sha(row['path']) for row in actual['files'])
    assert actual['protected_T_numeric_values_read'] == 0


def test_missing_external_provenance_rejected(tmp_path, monkeypatch):
    paths = make_inputs(tmp_path, monkeypatch)
    paths[-1].unlink()
    with pytest.raises(FileNotFoundError):
        prepare.external_input_bindings(dict(bindings={}))


def test_foundation_symlink_keeps_snapshot_and_blob_bindings(tmp_path, monkeypatch):
    paths = make_inputs(tmp_path, monkeypatch)
    snapshot = paths[2]
    blob = tmp_path/'asset/blobs/model.bin'
    blob.parent.mkdir(parents=True)
    snapshot.replace(blob)
    try:
        snapshot.symlink_to(blob)
    except OSError:
        pytest.skip('Symlink creation unavailable')
    rows = prepare.external_input_bindings(dict(bindings={}))['files']
    names = {row['path'] for row in rows}
    assert str(snapshot.absolute()) in names and str(blob.resolve()) in names


def test_preservation_excludes_new_outputs_and_includes_bound_external(tmp_path, monkeypatch):
    repo = tmp_path/'repo'
    code, report = repo/'research/new', repo/'reports/new'
    monkeypatch.setattr(prepare, 'REPO', repo)
    monkeypatch.setattr(prepare, 'CODE', code)
    monkeypatch.setattr(prepare, 'REPORT', report)
    paths = [repo/'research/old/model.py', repo/'reports/old/result.json', code/'run.py',
             report/'RESULT.json', repo/'reports/old/__pycache__/item.pyc', tmp_path/'external.bin']
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'test')
    actual = prepare.preserved_paths([], dict(files=[dict(path=str(paths[-1]))]))
    assert set(actual) == {paths[0].resolve(), paths[1].resolve(), paths[-1].resolve()}


def test_registration_write_does_not_overwrite_changed_evidence(tmp_path):
    path = tmp_path/'registration.json'
    prepare.write(path, dict(value=1))
    prepare.write(path, dict(value=1))
    with pytest.raises(FileExistsError):
        prepare.write(path, dict(value=2))
    assert json.loads(path.read_text(encoding='utf8')) == dict(value=1)
