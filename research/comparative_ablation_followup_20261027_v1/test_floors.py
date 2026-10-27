"""Synthetic checks of affine geometry, complete cases, and saved-array identities."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research.informs_broad_validation_20261015_v1 import data_contract
from . import floors


def arrays(target, prediction=None, observed=None):
    target = np.asarray(target, np.float32)
    if target.ndim == 2:
        target = target[None]
    prediction = target.copy() if prediction is None else np.asarray(prediction, np.float32).reshape(target.shape)
    return dict(target_normalized=target, pred_normalized=prediction,
                observed=np.ones(target.shape, bool) if observed is None else np.asarray(observed, bool).reshape(target.shape),
                origins=np.arange(target.shape[0], dtype=np.int64),
                channel_ids=np.asarray([f'c{i}' for i in range(target.shape[-1])]))


class AffineGeometryTests(unittest.TestCase):
    def test_nonzero_offset_and_nonorthonormal_weight(self):
        weight = np.asarray([[2., 0., 0.]])
        bias = np.asarray([7., 11., 13.])
        target = np.asarray([[9., 14., 17.], [3., 8., 9.]])
        prediction = np.asarray([[9., 11., 13.], [3., 11., 13.]])
        result = floors.affine_diagnostic(arrays(target, prediction), weight, bias)
        self.assertAlmostEqual(result['floor_MSE'], 25 / 3)
        self.assertAlmostEqual(result['actual_same_cohort_MSE'], 25 / 3)
        self.assertEqual(result['rank'], 1)
        self.assertEqual(result['floor_fraction'], 1)

    def test_rank_deficiency_retains_true_affine_space(self):
        weight = np.asarray([[1., 0., 0.], [2., 0., 0.]])
        target = np.asarray([[4., 3., -2.], [8., -3., 2.]])
        prediction = np.asarray([[4., 0., 0.], [8., 0., 0.]])
        result = floors.affine_diagnostic(arrays(target, prediction), weight, np.zeros(3))
        self.assertEqual(result['rank'], 1)
        self.assertAlmostEqual(result['floor_MSE'], 13 / 3)

    def test_small_real_singular_value_is_not_dropped_as_fp32_noise(self):
        space = floors.row_space(np.asarray([[1., 0., 0.], [0., 1e-8, 0.]], np.float32))
        self.assertEqual(space['rank'], 2)
        np.testing.assert_allclose(space['basis'].T @ space['basis'], np.eye(2), atol=1e-15)

    def test_zero_decoder_is_constant_affine_space(self):
        bias = np.asarray([2., -4.])
        values = arrays([[5., 0.], [2., -4.]], [[2., -4.], [2., -4.]])
        result = floors.affine_diagnostic(values, np.zeros((1, 2)), bias)
        self.assertEqual(result['rank'], 0)
        self.assertAlmostEqual(result['floor_MSE'], 25 / 4)

    def test_partial_mask_excludes_whole_origin_horizon_point(self):
        target = [[2., 3.], [800., 1000.], [4., 5.], [6., 7.]]
        prediction = [[2., 0.], [-900., -1000.], [4., 0.], [6., 0.]]
        observed = [[True, True], [True, False], [True, True], [True, True]]
        result = floors.affine_diagnostic(arrays(target, prediction, observed), [[1., 0.]], [0., 0.])
        self.assertEqual(result['complete_points'], 3)
        self.assertEqual(result['excluded_points'], 1)
        self.assertEqual(result['excluded_fraction'], .25)
        self.assertAlmostEqual(result['floor_MSE'], (9 + 25 + 49) / 6)

    def test_empty_complete_cohort_stops_even_when_each_channel_observed(self):
        values = arrays([[1., 3.], [4., 2.]], observed=[[True, False], [False, True]])
        with self.assertRaisesRegex(ValueError, 'No fully observed'):
            floors.complete_cohort(values)

    def test_scale_changes_squared_errors_without_changing_fraction(self):
        target = np.asarray([[9., 14., 17.], [3., 8., 9.]])
        prediction = np.asarray([[10., 11., 13.], [1., 11., 13.]])
        weight, bias = np.asarray([[2., 0., 0.]]), np.asarray([7., 11., 13.])
        baseline = floors.affine_diagnostic(arrays(target, prediction), weight, bias)
        for scale in (1 / 8, 32):
            scaled = floors.affine_diagnostic(arrays(target * scale, prediction * scale), weight * scale, bias * scale)
            self.assertAlmostEqual(scaled['floor_MSE'], baseline['floor_MSE'] * scale ** 2)
            self.assertAlmostEqual(scaled['actual_same_cohort_MSE'], baseline['actual_same_cohort_MSE'] * scale ** 2)
            self.assertAlmostEqual(scaled['floor_fraction'], baseline['floor_fraction'])

    def test_forecast_below_mathematical_floor_stops(self):
        values = arrays([[3., 4.], [5., 6.]])
        with self.assertRaisesRegex(ValueError, 'Prediction outside saved decoder space|lower-bound violation'):
            floors.affine_diagnostic(values, [[1., 0.]], [0., 0.])

    def test_incorrect_decoder_space_stops_even_when_actual_above_floor(self):
        values = arrays([[3., 4.], [5., 6.]], [[30., 1.], [50., 1.]])
        with self.assertRaisesRegex(ValueError, 'Prediction outside saved decoder space'):
            floors.affine_diagnostic(values, [[1., 0.]], [0., 0.])

    def test_fp32_affine_rounding_passes_nonaxis_aligned_space(self):
        rng = np.random.default_rng(191)
        weight = rng.normal(size=(3, 8)).astype(np.float32)
        bias = rng.normal(size=8).astype(np.float32)
        latent = rng.normal(size=(128, 3)).astype(np.float32)
        prediction = latent @ weight + bias
        target = rng.normal(size=(128, 8)).astype(np.float32)
        result = floors.affine_diagnostic(arrays(target, prediction), weight, bias)
        self.assertGreater(result['prediction_space_residual_MSE'], 0)
        self.assertLess(result['prediction_space_residual_MSE'], result['prediction_space_residual_limit'])
        self.assertGreaterEqual(result['actual_same_cohort_MSE'] + result['lower_bound_tolerance'], result['floor_MSE'])


class IdentityAndPcaTests(unittest.TestCase):
    def test_registered_pca_keeps_causal_fill_center(self):
        import torch
        physical = np.asarray([[0., 1., 3.], [0., 3., 1.], [8., 5., 5.], [8., 7., 7.]], np.float32)
        observed = np.ones(physical.shape, bool)
        observed[2:, 0] = False
        task = dict(physical=physical, observed=observed, columns=['a', 'b', 'c'])
        # Missing first-channel values remain at 0 under causal fill; vary another
        # observed prefix to make the filled normalized center nonzero.
        task['physical'][1, 0] = 4
        scales = floors.analysis.training_scales(task, 4)
        pca = data_contract.initialize_pca(task, 4, 1, scales['mean'], scales['std'])
        self.assertGreater(abs(float(pca['pca_center'][0])), .1)
        center, weight = pca['pca_center'], pca['E'].T
        prediction = np.asarray([[1.], [-2.]], np.float32) @ weight + center
        values = arrays(prediction + np.asarray([0., 1., -1.]), prediction)
        bundle = dict(model={'E': torch.from_numpy(pca['E']), 'center': torch.from_numpy(center)})
        summaries = {('synthetic', method): dict(MSE_median=1., n=10, extended=False) for method in floors.REFERENCES}
        with patch.dict(floors.runtime.KS, synthetic=1), patch.object(floors, '_bundle', return_value=bundle):
            row, shifts = floors._d0('synthetic', task, scales, {'data': data_contract}, {}, values, summaries)
        filled = data_contract.causal_forward_fill(task['physical'], task['observed'], scales['mean'])
        normalized = ((filled - scales['mean']) / scales['std']).astype(np.float64)
        basis = floors.row_space(weight)['basis']
        expected = np.square(floors.outside(normalized, center, basis)).mean()
        self.assertAlmostEqual(row['A60_floor_MSE'], expected)
        self.assertEqual(shifts[0]['PCA_center'], float(center[0]))
        self.assertGreater(abs(shifts[0]['A60_causalfilled_normalized_mean']), .1)
        self.assertEqual(shifts[0]['V_minus_A60_observed_scaler_mean'], shifts[0]['V_complete_case_mean'])
        self.assertLessEqual(row['V_optimal_rankK_floor_MSE'], row['V_floor_MSE'] + 1e-12)

    def test_bound_arrays_verify_normalization_origins_channels_and_hash(self):
        physical = np.asarray([[9., 19.], [11., 21.], [9., 21.], [11., 19.],
                               [12., 23.], [10., 24.], [14., 20.], [15., 25.]], np.float32)
        task = dict(physical=physical, observed=np.ones(physical.shape, bool), columns=['a', 'b'],
                    split_ranges={'V_select': (4, 8)}, L=2, H=2)
        scales = floors.analysis.training_scales(task, 4)
        target = ((physical[4:] - scales['mean']) / scales['std']).reshape(2, 2, 2)
        values = arrays(target, target + 2)
        values.update(origins=np.asarray([4, 6]), channel_ids=np.asarray(['a', 'b']), mean=scales['mean'], std=scales['std'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'arrays.npz'
            np.savez(path, **values)
            result = {'row': dict(prediction_path=str(path), prediction_sha256=floors.common.sha(path),
                                  normalization_fit_end=4, MSE=4., MAE=2.)}
            bound = floors._bound_arrays(result, task, scales, {'data': data_contract})
            np.testing.assert_array_equal(bound['target_physical'], physical[4:].reshape(2, 2, 2))
            for field, replacement in [('mean', scales['mean'] + 1), ('origins', np.asarray([4, 5])),
                                       ('channel_ids', np.asarray(['b', 'a']))]:
                altered = dict(values, **{field: replacement})
                np.savez(path, **altered)
                result['row']['prediction_sha256'] = floors.common.sha(path)
                with self.assertRaises(ValueError):
                    floors._bound_arrays(result, task, scales, {'data': data_contract})
            result['row']['prediction_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'SHA changed'):
                floors._bound_arrays(result, task, scales, {'data': data_contract})

    def test_summary_uses_median_of_ratios_and_primary_cohort(self):
        rows = [dict(dataset='ETTh1', method_id='A4', is_primary=True, status='PASS',
                     floor_MSE=floor, actual_same_cohort_MSE=actual, floor_fraction=floor / actual,
                     excluded_fraction=0, prediction_space_residual_MSE=0)
                for floor, actual in [(1, 2), (2, 100), (3, 3)]]
        rows.append(dict(rows[0], is_primary=False, floor_fraction=1000))
        summary = next(row for row in floors.summarize_d1(rows, ('A4',)) if row['dataset'] == 'ETTh1')
        self.assertEqual(summary['n'], 3)
        self.assertEqual(summary['floor_fraction_median'], .5)
        self.assertFalse(summary['complete'])
        self.assertEqual(summary['status'], 'INCOMPLETE')
        self.assertNotEqual(summary['floor_fraction_median'], summary['floor_MSE_median'] / summary['actual_same_cohort_MSE_median'])

    def test_exclusive_csv_writer_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'output.csv'
            floors._csv_new(path, [{'value': 1}])
            with self.assertRaises(FileExistsError):
                floors._csv_new(path, [{'value': 2}])
            self.assertIn('1', path.read_text())


class SealedReuseTests(unittest.TestCase):
    def seal(self, output):
        names = list(floors.FLOOR_FILES)
        manifest = dict(algorithm_version=floors.ALGORITHM_VERSION, methods=['C2'], include_d0=True,
                        source='synthetic sealed inputs')
        rows = [dict(dataset='synthetic', rank=2, floor_MSE=.125, floor_fraction=None,
                     is_primary=True, extended=False, singular_values=[2., .5]),
                dict(dataset='synthetic', status='SKIPPED_MISSING_BUNDLE')]
        tables = {name: rows for name in names if name.endswith('.csv')}
        for name, values in tables.items():
            floors._csv_new(output / name, values)
        receipt = dict(status='PASS', input_manifest=manifest, input_digest=floors._digest(manifest),
                       artifact_sha256={name: floors.common.sha(output / name) for name in tables},
                       csv_types={name: floors._csv_types(values) for name, values in tables.items()})
        (output / 'D0_D1_VALIDATION.json').write_text(json.dumps(receipt), encoding='utf8')
        expected = dict(d0=rows, d1=rows, d1_summary=rows, validation=receipt,
                        artifacts={name: str(output / name) for name in names})
        return names, manifest, expected

    def test_run_reuses_exact_typed_values_without_loading_data_or_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory).resolve()
            names, manifest, expected = self.seal(output)
            before = {name: ((output / name).read_bytes(), (output / name).stat().st_mtime_ns) for name in names}
            with patch.object(floors.common, 'REPORT', output), \
                    patch.object(floors.common, 'guard_preserved', return_value={'status': 'PASS'}), \
                    patch.object(floors, '_input_manifest', return_value=manifest), \
                    patch.object(floors.runtime, 'load_modules', side_effect=AssertionError('No numerical reload')), \
                    patch.object(floors.common, 'install_write_guard', side_effect=AssertionError('No new write phase')):
                actual = floors.run(output, methods=('C2',), include_d0=True, results=[])
            self.assertEqual(actual, expected)
            self.assertIs(type(actual['d1'][0]['rank']), int)
            self.assertIs(type(actual['d1'][0]['is_primary']), bool)
            self.assertIsNone(actual['d1'][0]['floor_fraction'])
            self.assertNotIn('rank', actual['d1'][1])
            self.assertEqual(before, {name: ((output / name).read_bytes(), (output / name).stat().st_mtime_ns) for name in names})

    def test_tampered_artifact_stops_without_rewriting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            names, manifest, _ = self.seal(output)
            path = output / 'D1_DECODER_FLOOR.csv'
            path.write_bytes(path.read_bytes() + b'\n')
            damaged = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'SHA changed.*D1_DECODER_FLOOR.csv'):
                floors._reuse(output, names, manifest)
            self.assertEqual(path.read_bytes(), damaged)

    def test_partial_artifacts_and_changed_input_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            names, manifest, _ = self.seal(output)
            with self.assertRaisesRegex(ValueError, 'input/algorithm mismatch.*D0_D1_VALIDATION.json'):
                floors._reuse(output, names, dict(manifest, source='changed input'))
            (output / 'D0_MEAN_SHIFT_CHANNELS.csv').unlink()
            with self.assertRaisesRegex(FileExistsError, 'Partial.*D0_MEAN_SHIFT_CHANNELS.csv'):
                floors._reuse(output, names, manifest)

    def test_input_manifest_binds_saved_result_prediction_bundle_and_seal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old, new = root / 'old', root / 'new'
            job_dir = old / 'jobs' / 'synthetic_C2'
            job_dir.mkdir(parents=True)
            (old / 'RUNS.csv').write_text('synthetic frozen reference', encoding='utf8')
            paths = {name: root / filename for name, filename in
                     [('prediction', 'prediction.npz'), ('bundle', 'bundle.pt'), ('selection', 'SELECTION.json')]}
            for kind, path in paths.items():
                path.write_bytes(kind.encode())
            result = dict(status='COMPLETE', job={'job_id': 'synthetic_C2', 'method_id': 'C2'},
                          row=dict(prediction_path=str(paths['prediction']), prediction_sha256=floors.common.sha(paths['prediction'])),
                          bundle_path=str(paths['bundle']), bundle_sha256=floors.common.sha(paths['bundle']),
                          selection_path=str(paths['selection']), selection_sha256=floors.common.sha(paths['selection']))
            result_path = job_dir / 'RESULT.json'
            result_path.write_text(json.dumps(result), encoding='utf8')
            with patch.object(floors.common, 'OLD_REPORT', old), patch.object(floors.common, 'REPORT', new):
                original = floors._input_manifest([result], ('C2',), False)
                self.assertEqual({item['kind'] for item in original['results'][0]['files']}, {'prediction', 'bundle', 'selection'})
                self.assertEqual(original['results'][0]['result_sha256'], floors.common.sha(result_path))
                paths['prediction'].write_bytes(b'changed prediction')
                with self.assertRaisesRegex(ValueError, 'prediction.*SHA changed.*prediction.npz'):
                    floors._input_manifest([result], ('C2',), False)
                paths['prediction'].write_bytes(b'prediction')
                paths['bundle'].unlink()
                missing = floors._input_manifest([result], ('C2',), False)
                self.assertNotEqual(floors._digest(original), floors._digest(missing))
                altered = dict(result, status='ALTERED')
                with self.assertRaisesRegex(ValueError, 'RESULT object differs.*RESULT.json'):
                    floors._input_manifest([altered], ('C2',), False)

    def test_dataset_specific_no_t_receipts_fail_closed(self):
        floors._check_no_t('jena', dict(new_T_numeric_values_parsed=0,
                                       members=[dict(member='x.npy', tail_numeric_values_parsed=0)]))
        floors._check_no_t('ETTh1', dict(T_numeric_values_parsed=0))
        for dataset, receipt in [('jena', dict(T_numeric_values_parsed=0)),
                                 ('jena', dict(new_T_numeric_values_parsed=0, members=[])),
                                 ('jena', dict(new_T_numeric_values_parsed=0, members=[dict(member='x.npy')])),
                                 ('jena', dict(new_T_numeric_values_parsed=0, members=[dict(member='x.npy', tail_numeric_values_parsed=1)])),
                                 ('ETTh1', {})]:
            with self.assertRaises(ValueError):
                floors._check_no_t(dataset, receipt)


if __name__ == '__main__':
    unittest.main()
