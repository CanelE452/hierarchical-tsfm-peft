from types import SimpleNamespace

import pytest

from research.b2_dataset_extension_20261018_v1 import baselines


def test_split_and_producer_hooks_restored_after_failure():
    original_split = lambda *args, **kwargs: 'original'
    original_source = lambda: ('old-sha', {'existing.py': '123'})
    modules = {'data': SimpleNamespace(split_ranges=original_split),
               'fit': SimpleNamespace(_source_contract=original_source),
               'training': SimpleNamespace(fingerprint=lambda files: tuple(sorted(files)))}
    task = {'split_ranges': {'A_fit': (0, 600)}}
    with pytest.raises(RuntimeError, match='injected'):
        with baselines.direct_scope(modules, task):
            assert modules['data'].split_ranges(task) == task['split_ranges']
            assert 'research/' + baselines.train.NAME + '/baselines.py' in modules['fit']._source_contract()[1]
            with pytest.raises(ValueError, match='exact registered task'):
                modules['data'].split_ranges(dict(task))
            raise RuntimeError('injected')
    assert modules['data'].split_ranges is original_split
    assert modules['fit']._source_contract is original_source


def test_direct_identity_rejects_unregistered_seed():
    with pytest.raises(ValueError, match='two fixed seeds'):
        baselines.direct_id({'dataset': 'household'}, 92603)
