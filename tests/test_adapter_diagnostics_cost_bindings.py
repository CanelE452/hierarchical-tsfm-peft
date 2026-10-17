import json

import numpy as np
import pytest

from research.adapter_diagnostics_20261017_v2 import costs
from research.adapter_diagnostics_20261017_v2 import existing_jobs as existing


def test_private_cost_binding_rejects_changed_asset_and_keeps_relative_reference(tmp_path):
    path = tmp_path/'weights.bin'
    path.write_bytes(b'original')
    ctx = dict(asset=tmp_path, repo=tmp_path/'repo', cache=tmp_path/'cache', report=tmp_path/'report')
    expected = existing.sha(path)
    binding = costs.bound_file(path, ctx, expected)
    assert binding == dict(reference='asset://weights.bin', sha256=expected)
    path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='SHA'):
        costs.bound_file(path, ctx, expected)


def test_cost_archive_checks_stored_archive_sha_and_each_array_fingerprint(tmp_path):
    path = tmp_path/'V.npz'
    arrays = dict(pred_normalized=np.ones((1, 2, 1), np.float32),
                  target_normalized=np.zeros((1, 2, 1), np.float32),
                  observed=np.ones((1, 2, 1), bool), origins=np.array([10], np.int64),
                  channel_ids=np.array(['c']))
    np.savez(path, **arrays)
    manifest = dict(archive_sha256=existing.sha(path),
                    arrays={key: existing._array_fingerprint(value) for key,value in arrays.items()})
    costs.verified_archive(path, manifest)
    manifest['arrays']['origins']['sha256'] = '0'*64
    with pytest.raises(ValueError, match='fingerprint'):
        costs.verified_archive(path, manifest)
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='SHA'):
        costs.verified_archive(path, manifest)
