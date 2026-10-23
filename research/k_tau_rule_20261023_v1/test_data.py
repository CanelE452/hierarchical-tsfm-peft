"""Data identity and pre-T boundary fixtures; no real data or model reads."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from research.k_tau_rule_20261023_v1 import data


def prefix_config():
    return dict(dataset='ETTh1', prefix_only_approved=True, L=512, H=48,
        prefix_rows=8000, prefix_sha256='a' * 64,
        split_ranges=dict(A_fit=[0, 5000], A_stop=[5000, 6000],
            M_cal=[6000, 7000], V_select=[7000, 8000], T_final=[8000, 10000]))


def test_missing_historical_data_hash_is_rejected():
    with pytest.raises(ValueError, match='historical.*data_sha256'):
        data._validate_prefix_config(prefix_config(), 'ETTh1', {})


@pytest.mark.parametrize('change', ['tail', 'overlap', 'approval', 'dataset'])
def test_config_rejects_boundary_or_identity_drift(change):
    config = prefix_config()
    if change == 'tail':
        config['prefix_rows'] = 8001
    elif change == 'overlap':
        config['split_ranges']['T_final'][0] = 7999
    elif change == 'approval':
        config['prefix_only_approved'] = False
    else:
        config['dataset'] = 'ETTh2'
    identity = dict(data_sha256='b' * 64, prefix_sha256='a' * 64)
    with pytest.raises(ValueError):
        data._validate_prefix_config(config, 'ETTh1', identity)


def test_valid_config_is_not_mutated():
    config = prefix_config()
    original = copy.deepcopy(config)
    data._validate_prefix_config(config, 'ETTh1',
        dict(data_sha256='b' * 64, prefix_sha256='a' * 64))
    assert config == original


def test_loaded_task_tail_is_rejected():
    task = dict(dataset='ETTh1', L=512, H=48, data_sha256='b' * 64,
        physical=SimpleNamespace(shape=(8001, 7)),
        observed=SimpleNamespace(shape=(8001, 7)), columns=list('1234567'),
        split_ranges=prefix_config()['split_ranges'])
    with pytest.raises(ValueError, match='pre-T'):
        data._validate_task(task, 'ETTh1', dict(data_sha256='b' * 64))


def test_unknown_dataset_rejected_before_modules_or_loader(monkeypatch):
    monkeypatch.setattr(data.oldexisting, '_modules', lambda ctx: pytest.fail('modules called'))
    with pytest.raises(ValueError, match='dataset'):
        data.build_context('../TEST', 1.)


def test_missing_identity_hash_rejected_before_prefix_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(data, 'REPO', tmp_path)
    report = tmp_path / 'reports' / data.SEARCH_NAME / 'datasets' / 'ETTh1'
    report.mkdir(parents=True)
    (report / 'TRAIN_IDENTITY.json').write_text('{"dataset":"ETTh1"}')
    monkeypatch.setattr(data.core, 'load_task', lambda *args: pytest.fail('prefix loaded'))
    with pytest.raises(ValueError, match='historical.*data_sha256'):
        data.build_context('ETTh1', 1.)


def test_synthetic_prefix_context_has_no_training_permission_without_preregistration(tmp_path, monkeypatch):
    monkeypatch.setattr(data, 'REPO', tmp_path)
    root = tmp_path / '.cache' / data.SEARCH_NAME / 'data' / 'ETTh1'
    root.mkdir(parents=True)
    config = prefix_config()
    config['interval_seconds'] = 3600
    prefix = root / 'PREFIX_PHYSICAL.npz'
    np.savez(prefix, physical=np.zeros((8000, 7), dtype=np.float32),
        observed=np.ones((8000, 7), dtype=bool),
        times=np.arange(8000, dtype=np.int64) * 3600 * 1_000_000_000,
        columns=np.array(list('1234567')))
    config['prefix_sha256'] = data.sha(prefix)
    config_path = root / 'DATA_DIAGNOSTIC.json'
    import json
    config_path.write_text(json.dumps(config))
    _, expected_task = data.core.load_task(config_path, prefix)
    identity = dict(dataset='ETTh1', data_sha256=expected_task['data_sha256'],
        prefix_sha256=config['prefix_sha256'], input_config_sha256=data.sha(config_path))
    monkeypatch.setattr(data, '_historical_records',
        lambda dataset: (tmp_path / 'identity.json', identity, tmp_path / 'source.json', {}))
    monkeypatch.setattr(data, '_source_hashes', lambda source: ({}, {}))
    model_dir = tmp_path / 'models' / 'fixture'
    model_dir.mkdir(parents=True)
    monkeypatch.setattr(data.oldexisting, '_modules',
        lambda ctx: dict(legacy=SimpleNamespace(MODEL_REL='models/fixture')))
    context = data.build_context('ETTh1', 123., asset_root=tmp_path, device='cpu')
    assert context['task']['data_sha256'] == identity['data_sha256']
    assert context['cache'] == tmp_path / '.cache' / data.NAME / 'datasets' / 'ETTh1'
    assert context['report'] == tmp_path / 'reports' / data.NAME / 'datasets' / 'ETTh1'
    assert context['protocol_sha256'] is None
    assert context['training_allowed'] is False
    assert 'foundation' not in context
    assert context['task_access']['T_numeric_values_parsed'] == 0


def test_historical_core_file_set_is_read_from_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(data, 'REPO', tmp_path)
    with pytest.raises(ValueError, match='exactly six'):
        data._source_hashes(dict(files={'fit_base.py': 'a' * 64}))
