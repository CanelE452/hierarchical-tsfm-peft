import importlib

import numpy as np
import pytest


def execution():
    try:
        return importlib.import_module('research.gt_ts2_fastcheck_20261018_v1.execution')
    except ModuleNotFoundError:
        pytest.fail('actual parent parity and fixed experiment registry are missing')


def test_parent_parity_rejects_shape_nonfinite_and_excess_error():
    e = execution()
    expected = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    assert e.parity(expected.copy(), expected)['status'] == 'PASS'
    for bad in (expected[:, :2], np.full_like(expected, np.nan), expected + .01):
        with pytest.raises(ValueError):
            e.parity(bad, expected)


def test_predeclared_candidates_preserve_frozen_and_eight_matched_arms():
    e = execution()
    specs = e.candidate_specs()
    frozen = [s for s in specs if s['phase'] == 'A']
    adapted = [s for s in specs if s['phase'] == 'B']
    assert len(frozen) == 13
    assert len(adapted) == 8
    assert len({s['run_id'] for s in specs}) == len(specs)
    assert {(s['arm'], s['seed']) for s in adapted} == {
        (arm, seed) for arm in ('GT_FT_L512', 'GT_FT_L1024', 'GT_TS_PRE_L1024', 'GT_TS_RAND_L1024')
        for seed in (92601, 92602)}
    assert all(s['L'] <= 1024 for s in adapted)
    assert {s['L'] for s in frozen if s['mode'] == 'raw'} == {512, 1024, 2048}


def test_feature_consumer_rejects_post_export_tampering(tmp_path):
    e = execution()
    path = tmp_path / 'features.npz'
    features = np.zeros((2,256,320), np.float32)
    origins = np.array([1024,1152])
    np.savez(path, features=features, origins=origins)
    proof = dict(file_sha256=e.c.file_sha(path),features_sha256=e.c.array_sha(features),
                 origin_sha256=e.c.array_sha(origins),shape=list(features.shape))
    record = dict(features_paths={'train':str(path)},_receipt={'files':{'train':proof}})
    np.testing.assert_array_equal(e._features(record,'train'),features)
    np.savez(path, features=features+1, origins=origins)
    with pytest.raises(ValueError):
        e._features(record,'train')
