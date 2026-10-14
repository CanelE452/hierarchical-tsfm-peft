"""Prepared CPU contract fixtures; no real assets/model/runtime execution."""
import ast
import inspect
from pathlib import Path
import sys
import types
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_stage_native_ridge_v1 as runner
import stage_separation_math_v1 as math


class SourceAndIdentityTests(unittest.TestCase):
    def test_runtime_constructed_before_assets_torch_or_solver(self):
        tree = ast.parse(inspect.getsource(runner.execute))
        statements = tree.body[0].body
        self.assertIsInstance(statements[0], ast.Import)
        self.assertEqual(statements[0].names[0].name, 'stage_separation_runtime_v1')
        self.assertIsInstance(statements[1], ast.Assign)
        self.assertEqual(statements[1].value.func.attr, 'ContinuationLimit')
        self.assertEqual(statements[1].value.keywords[0].arg, 'gpu')
        self.assertFalse(statements[1].value.keywords[0].value.value)
        source = inspect.getsource(runner)
        self.assertNotIn('import_old_engine(', source)
        self.assertNotIn('.cuda(', source)
        self.assertIn("map_location='cpu', weights_only=True", source)

    def test_tag_rejects_path_traversal(self):
        for tag in ('../old', 'a/b', 'UPPER', ''):
            with self.assertRaises(ValueError):
                runner.safe_tag(tag)
        self.assertEqual(runner.safe_tag('new_s92601_v1'), 'new_s92601_v1')

    def test_d2_identity_rejects_wrong_seed_and_missing_parity(self):
        p = types.SimpleNamespace(OLD_F_HASH='a', BI_F_HASH='b')
        d = {'status': 'COMPLETE', 'stage': 'D2', 'dataset': 'robin', 'seed': 92601,
             'F_frozen_pass': True, 'MAIN_codec_frozen_pass': True,
             'F_before': {'old_algorithm': 'a', 'bi_algorithm': 'b'},
             'F_after': {'old_algorithm': 'a', 'bi_algorithm': 'b'},
             'source_GPU_D_vs_stored_MAIN': {s: {'pass': True} for s in ('fit', 'tune', 'val')}}
        runner.validate_d2_identity(d, 'robin', 92601, p)
        with self.assertRaises(ValueError):
            runner.validate_d2_identity(d, 'robin', 92602, p)
        d['source_GPU_D_vs_stored_MAIN'] = {}
        with self.assertRaises(ValueError):
            runner.validate_d2_identity(d, 'robin', 92601, p)

    def test_oracle_is_same_cp_and_d2_diagnostic_only(self):
        value = {'status': 'COMPLETE', 'dataset': 'robin', 'seed': 92601,
                 'diagnostic_only': True, 'future_observations_used': True,
                 'original_codec_frozen_pass': True, 'codec_before': {'a': 1}, 'codec_after': {'a': 1},
                 'MAIN_checkpoint': {'checkpoint': {'sha256': 'cp'}}, 'D2_source': {'sha256': 'd2'},
                 'artifacts': {s: {} for s in ('fit', 'tune', 'val')},
                 'checks': {s: {'prefix': {'pass': True}} for s in ('fit', 'tune', 'val')}}
        runner.validate_oracle_identity(value, 'robin', 92601, 'cp', 'd2', value['artifacts'])
        for field, replacement in (('seed', 92602), ('diagnostic_only', False)):
            changed = {**value, field: replacement}
            with self.assertRaises(ValueError):
                runner.validate_oracle_identity(changed, 'robin', 92601, 'cp', 'd2', value['artifacts'])
        with self.assertRaises(ValueError):
            runner.validate_oracle_identity(value, 'robin', 92601, 'other', 'd2', value['artifacts'])

    def test_exact_array_contract_preserves_dtype_and_nan_mask_cells(self):
        a = np.array([1., np.nan], np.float32)
        runner.exact_array(a, a.copy(), 'same')
        with self.assertRaises(ValueError):
            runner.exact_array(a, a.astype(np.float64), 'cast')
        with self.assertRaises(ValueError):
            runner.exact_array(a, np.array([1., 0.], np.float32), 'filled')

    def test_independent_n2_precedes_oracle_load(self):
        source = inspect.getsource(runner.execute)
        self.assertLess(source.index("n2 = family('N2'"), source.index('if not args.oracle_result:'))
        self.assertIn("for name in ('O1', 'N1', 'N3'):", source)
        self.assertIn("if isinstance(error, runtime.BudgetHeld):\n                    raise", source)

    def test_rank_nonfinite_diagnostics_are_not_model_json(self):
        p = np.zeros((2, 1))
        y = np.ones((2, 48, 1))
        ridge = math.prepare_masked_ridge(p, y, np.ones_like(y, bool), kind='N2_DIRECT_C', feature_state_sha256='s')
        solution = ridge.fit(0.)
        self.assertTrue(np.isinf(solution.regularized_gram_conditions).all())
        metadata = runner.solution_metadata(solution)
        self.assertNotIn('regularized_gram_conditions', metadata)
        self.assertIn('regularized_gram_conditions', runner.solution_arrays(solution))


class SavedModelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(8)
        self.past = rng.normal(size=(4, 512, 1)).astype(np.float32)
        self.last = np.zeros((4, 2), np.float32)
        self.y = rng.normal(size=(4, 48, 2)).astype(np.float32)
        self.mask = np.ones(self.y.shape, bool)
        self.state = math.fit_feature_state(self.past)
        oracle = np.repeat(self.past[:, -1:], 48, axis=1) + np.float32(.2)
        self.n1 = math.prepare_n1(self.state, self.past, oracle, self.mask).fit(.1)
        self.n2 = math.prepare_n2(self.state, self.past, self.y, self.last, self.mask).fit(.1)
        self.hidden_weight = np.zeros((32, 1), np.float32)
        self.hidden_bias = np.zeros(32, np.float32)
        self.n3 = math.prepare_n3(self.n1, self.past, self.y, self.last, self.mask,
            self.hidden_weight, self.hidden_bias, np.zeros((33, 2), np.float32)).fit(.1)

    def test_native_coefficient_roundtrip_no_refit_and_same_predictions(self):
        for model in (self.n1, self.n2):
            arrays = runner.model_arrays(model)
            solution = runner.restore_solution(runner.solution_metadata(model.solution), arrays)
            restored = math.NativeModel(self.state, solution)
            self.assertEqual(model.sha256, restored.sha256)
            if model is self.n1:
                np.testing.assert_array_equal(model.latent_forecast(self.past), restored.latent_forecast(self.past))
            else:
                np.testing.assert_array_equal(model.direct_forecast(self.past, self.last), restored.direct_forecast(self.past, self.last))

    def test_changed_coefficient_rejected_by_solution_hash(self):
        arrays = {k: v.copy() for k, v in runner.model_arrays(self.n2).items()}
        arrays['coefficients'].flat[0] += 1
        with self.assertRaises(ValueError):
            runner.restore_solution(runner.solution_metadata(self.n2.solution), arrays)

    def test_n3_public_prediction_calls_bound_selected_n1_no_future_parameter(self):
        parameters = inspect.signature(runner.predict_selected_cpu).parameters
        self.assertNotIn('z_future', parameters)
        source = inspect.getsource(runner.predict_selected_cpu)
        self.assertIn('forecast_from_selected_n1(n1_model, latent_past, raw_last)', source)
        pred = runner.predict_selected_cpu('N3', self.n3, self.past, self.last, None, self.n1)
        np.testing.assert_array_equal(pred, self.n3.forecast_from_selected_n1(self.n1, self.past, self.last))
        with self.assertRaises(ValueError):
            runner.predict_selected_cpu('N3', self.n3, self.past, self.last, None, self.n2)

    def test_diagnostic_oracle_not_deployment_family(self):
        with self.assertRaises(ValueError):
            runner.predict_selected_cpu('O1', self.n3, self.past, self.last, None, self.n1)

    def test_selected_coefficients_restore_not_training_api(self):
        source = inspect.getsource(runner.load_selected_models)
        self.assertNotIn('prepare_n1(', source)
        self.assertNotIn('.fit(', source)
        self.assertIn('RESTORED_N3_MUST_BIND_SELECTED_N1', source)

    def test_terminal_receipt_never_overrides_failure_as_complete(self):
        source = inspect.getsource(runner.execute)
        self.assertIn("if receipt.get('terminal_status', result['status']) != 'COMPLETE':", source)
        self.assertIn("result['status'] = receipt['terminal_status']", source)
        self.assertIn("runtime.write_new(output / 'RESULT.json', result)", source)


if __name__ == '__main__':
    unittest.main()
