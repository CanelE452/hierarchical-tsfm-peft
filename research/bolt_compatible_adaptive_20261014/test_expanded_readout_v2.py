"""Canonical stride and serialization regression fixtures for the FIT512 repair."""
from io import BytesIO
import unittest

import numpy as np
import readout_math_v1 as math
from run_expanded_readout_v2 import canonicalize_payload, legal_fit_origins
from run_expanded_readout_v1 import legal_fit_origins as original_legal_origins


class CanonicalLayoutRepairTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(77231)
        self.strided = rng.normal(size=(21, 3, 512)).astype(np.float32).transpose(0, 2, 1)
        self.target = rng.normal(size=(21, 3, 48)).astype(np.float32).transpose(0, 2, 1)
        self.mask = np.ones((21, 3, 48), bool).transpose(0, 2, 1)
        self.mask[::3, 8:13, 1] = False
        self.payload = {'history': self.strided, 'target': self.target, 'observed': self.mask,
                        'raw_last': self.strided[:, -1].copy(), 'origins': np.arange(21, dtype=np.int64)}

    def test_canonical_payload_changes_only_memory_order(self):
        self.assertFalse(self.strided.flags.c_contiguous)
        canonical = canonicalize_payload(self.payload)
        for key, value in self.payload.items():
            np.testing.assert_array_equal(value, canonical[key])
            self.assertEqual(value.dtype, canonical[key].dtype)
            self.assertEqual(value.shape, canonical[key].shape)
            self.assertTrue(canonical[key].flags.c_contiguous)

    def test_canonical_features_statistics_and_digest_survive_npz(self):
        canonical = canonicalize_payload(self.payload)
        file = BytesIO()
        np.savez(file, **canonical)
        file.seek(0)
        with np.load(file, allow_pickle=False) as saved:
            loaded = {k: saved[k].copy() for k in saved.files}
        for route in ('SELF33', 'SELF512'):
            a = math.fit_self_features(route, canonical['history'])
            b = math.fit_self_features(route, loaded['history'])
            np.testing.assert_array_equal(a.center, b.center)
            np.testing.assert_array_equal(a.scale, b.scale)
            self.assertEqual(a.fit_raw_features_sha256, b.fit_raw_features_sha256)
            self.assertEqual(a.sha256, b.sha256)
        a = math.joint.fit_feature_state('RAW33C', canonical['history'])
        b = math.joint.fit_feature_state('RAW33C', loaded['history'])
        self.assertEqual(a.sha256, b.sha256)

    def test_strided_and_contiguous_input_produce_identical_canonical_fit(self):
        canonical = canonicalize_payload(self.payload)
        contiguous = canonicalize_payload({key: np.array(value, copy=True, order='C') for key, value in self.payload.items()})
        a = math.prepare_self('SELF33', canonical['history'], canonical['target'], canonical['raw_last'], canonical['observed'])
        b = math.prepare_self('SELF33', contiguous['history'], contiguous['target'], contiguous['raw_last'], contiguous['observed'])
        self.assertEqual(a.feature_state.sha256, b.feature_state.sha256)
        for alpha in math.ALPHAS:
            am, bm = a.fit(alpha), b.fit(alpha)
            self.assertEqual(am.sha256, bm.sha256)
            np.testing.assert_array_equal(am.forecast(canonical['history'], canonical['raw_last']),
                                          bm.forecast(contiguous['history'], contiguous['raw_last']))
        np.testing.assert_array_equal(math.fit_shared_mean(canonical['target'], canonical['raw_last'], canonical['observed'])['innovation'],
                                      math.fit_shared_mean(contiguous['target'], contiguous['raw_last'], contiguous['observed'])['innovation'])

    def test_repair_keeps_original_legal_origin_selector(self):
        for bounds in ((528, 2121), (0, 21038), (0, 570)):
            np.testing.assert_array_equal(legal_fit_origins(bounds), original_legal_origins(bounds))

    def test_canonical_deployment_has_no_future_payload_dependency(self):
        canonical = canonicalize_payload(self.payload)
        prepared = math.prepare_self('SELF33', canonical['history'], canonical['target'],
                                     canonical['raw_last'], canonical['observed'])
        model = prepared.fit(.01)
        prediction = model.forecast(canonical['history'], canonical['raw_last'])
        feature_digest = model.feature_state.sha256
        canonical['target'][:] = np.float32(1.e6)
        canonical['observed'][:] = False
        np.testing.assert_array_equal(prediction, model.forecast(canonical['history'], canonical['raw_last']))
        self.assertEqual(feature_digest, model.feature_state.sha256)


if __name__ == '__main__':
    unittest.main()
