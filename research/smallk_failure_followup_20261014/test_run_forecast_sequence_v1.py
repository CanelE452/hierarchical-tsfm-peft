"""SOURCE ONLY prepared CPU fixtures; not run before continuation approval."""
import ast
import inspect
from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_forecast_sequence_v1 as runner


class RecipeAndGateTests(unittest.TestCase):
    def test_describe_exact_recipe_and_no_asset_read(self):
        with mock.patch.object(Path, 'open', side_effect=AssertionError('ASSET_READ')):
            record = runner.describe()
        self.assertEqual(len(record['conditions']), 8)
        self.assertFalse(record['actual_data_opened'])
        self.assertEqual(record['new_updates'], 600)
        self.assertEqual(record['batch_size'], 4)
        self.assertEqual(record['checkpoints'], [0, 100, 200, 400, 600])
        self.assertEqual(record['local_fit_cap_seconds'], 150)
        self.assertEqual(record['fit_GPU_process_cap_sum_seconds'], 1200)

    def test_same_seed_order_across_heads_and_original_recipe(self):
        for seed in runner.SEEDS:
            first, second = (runner.training_order(seed, 256) for _ in runner.HEADS)
            np.testing.assert_array_equal(first, second)
            np.testing.assert_array_equal(first, np.random.default_rng(seed).integers(0, 256, (600, 4)))

    def test_prediction_public_API_excludes_actual_targets_and_masks(self):
        self.assertEqual(list(inspect.signature(runner.forward_trace).parameters), ['codec', 'bolt', 'head', 'history'])
        self.assertEqual(list(inspect.signature(runner.predict_trace).parameters),
                         ['engine', 'codec', 'bolt', 'head', 'history', 'check'])

    def test_continuation_runtime_is_lazy_and_precedes_real_data_model_reads(self):
        source = Path(runner.__file__).read_text(encoding='utf-8')
        tree = ast.parse(source)
        self.assertFalse(any(isinstance(node, (ast.Import, ast.ImportFrom))
                             and 'continuation_runtime' in ast.unparse(node) for node in tree.body))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'execute')
        body = ast.unparse(function)
        self.assertLess(body.index('runtime.ContinuationLimit('), body.index('probes.load_bridge('))
        self.assertLess(body.index('runtime.ContinuationLimit('), body.index('probes.import_old_engine('))
        self.assertIn('probes.apply_resource_terminal(result, receipt)', body)

    def test_actual_Y_TUNE_selection_includes_zero_and_earliest_tie(self):
        rows = [{'new_update': 0, 'tune_actual_Y_mse': 1.},
                {'new_update': 100, 'tune_actual_Y_mse': 2.},
                {'new_update': 200, 'tune_actual_Y_mse': 1.}]
        self.assertEqual(runner.choose_checkpoint(rows), 0)
        rows[2]['tune_actual_Y_mse'] = .9
        self.assertEqual(runner.choose_checkpoint(rows), 2)

    def test_invalid_scope_and_path_tag_rejected(self):
        with self.assertRaises(SystemExit):
            runner.parse_args(['--execute'])
        with self.assertRaises(SystemExit):
            runner.parse_args(['--describe', '--tag', '../old'])
        args = runner.parse_args(['--execute', '--dataset', 'robin', '--seed', '92601', '--head', 'TEMPORAL_RF3',
                                  '--input-bridge', 'not_opened.json', '--tag', 'new_rf3_fixture'])
        self.assertEqual(args.head, 'TEMPORAL_RF3')

    def test_failed_terminal_status_cannot_return_success_exit(self):
        args = ['--execute', '--dataset', 'robin', '--seed', '92601', '--head', 'STATIC_RF1',
                '--input-bridge', 'not_opened.json', '--tag', 'fresh_exit_fixture']
        with mock.patch.object(runner, 'execute', return_value={'status': 'BUDGET_HELD_PRESERVED'}):
            self.assertEqual(runner.main(args), 1)

    def test_selected_checkpoint_wrong_head_rejected_before_module_restore(self):
        fake_torch = mock.Mock()
        fake_torch.load.return_value = {'dataset': 'robin', 'seed': 92601, 'head_type': 'STATIC_RF1',
            'objective': 'Y', 'architecture': 'ANCHOR_NONLINEAR_D',
            'new_forecast_sequence_comparison': True, 'new_update': 100}
        probes = mock.Mock()
        probes.verified_file.return_value = {'path': 'synthetic_no_read.pt'}
        codec, head = mock.Mock(), mock.Mock()
        with self.assertRaisesRegex(ValueError, 'CP_IDENTITY'):
            runner.restore_selected_state(fake_torch, codec, head, {'path': 'unused'},
                                          'robin', 92601, 'TEMPORAL_RF3', probes)
        codec.load_state_dict.assert_not_called()
        head.load_state_dict.assert_not_called()

    def test_finalization_cannot_claim_COMPLETE_at_or_above_local_cap(self):
        result = {'status': 'COMPLETE', 'learning_executed': {'optimizer_updates': 600}}
        runner.apply_local_finalization_cap(result, 149.999, 'pre_resource_close')
        self.assertEqual(result['status'], 'COMPLETE')
        result['shared_resource_receipt'] = {'terminal_status': 'COMPLETE', 'immutable_actual_receipt': True}
        runner.apply_local_finalization_cap(result, 150., 'post_resource_close')
        self.assertEqual(result['status'], 'BUDGET_HELD_PRESERVED')
        self.assertEqual(result['failure']['phase'], 'post_resource_close')
        self.assertEqual(result['learning_executed']['optimizer_updates'], 600)
        self.assertEqual(result['shared_resource_receipt']['terminal_status'], 'COMPLETE')

    def test_finalization_preserves_prior_scientific_failure(self):
        failure = {'type': 'ValueError', 'message': 'F_WEIGHT_OR_GRADIENT_CHANGED'}
        result = {'status': 'FAILED_PRESERVED', 'failure': failure, 'F_frozen_pass': False}
        runner.apply_local_finalization_cap(result, 151., 'pre_resource_close')
        runner.apply_local_finalization_cap(result, 152., 'post_resource_close')
        self.assertEqual(result['failure_before_local_finalization'], failure)
        self.assertFalse(result['F_frozen_pass'])
        self.assertEqual(result['status'], 'BUDGET_HELD_PRESERVED')


class MaskAndGradientTests(unittest.TestCase):
    def test_masked_nan_and_empty_channel_finite_backward(self):
        import torch
        prediction = torch.tensor([[[1., 2., 3.], [3., 4., 5.]]], requires_grad=True)
        target = torch.tensor([[[0., float('nan'), float('nan')], [1., 1., float('nan')]]])
        observed = torch.tensor([[[True, False, False], [True, True, False]]])
        loss = runner.observed_macro_loss(prediction, target, observed)
        self.assertAlmostEqual(float(loss), ((1. + 4.) / 2. + 9.) / 2.)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))
        self.assertTrue(torch.equal(prediction.grad[:, :, 2], torch.zeros_like(prediction.grad[:, :, 2])))

    def test_real_CPU_autograd_F_branch_and_E_D_head_updates_after_zero_init(self):
        import torch
        import forecast_sequence_decoder as sequence
        from test_run_joint_objectives import FrozenGradientTests
        for factory in (lambda: sequence.StaticResidualHead(2, 3, 5), lambda: sequence.TemporalResidualHead(2, 3)):
            codec, frozen = FrozenGradientTests().models()
            head = factory()
            initial_codec = {name: value.detach().clone() for name, value in codec.state_dict().items()}
            initial_head = {name: value.detach().clone() for name, value in head.state_dict().items()}
            initial_f = {name: value.detach().clone() for name, value in frozen.state_dict().items()}
            parameters = list(codec.parameters()) + list(head.parameters())
            optimizer = torch.optim.AdamW(parameters, lr=.001, weight_decay=0.)
            history, target = torch.randn(4, 6, 3), torch.randn(4, 2, 3)
            observed = torch.ones_like(target, dtype=torch.bool)
            for update in range(2):
                optimizer.zero_grad(set_to_none=True)
                prediction, past, future, _ = runner.forward_trace(codec, frozen, head, history)
                loss = runner.observed_macro_loss(prediction, target, observed)
                proof = runner.parameter_gradient_proof(codec, head, loss, past, future)
                self.assertGreater(proof['frozen_F_input_gradient_norm'], 0.)
                self.assertTrue(proof['frozen_F_input_gradient_finite'])
                self.assertTrue(proof['Head_gradients']['finite'])
                self.assertGreater(proof['Head_gradients']['norm'], 0.)
                loss.backward()
                if update == 0:
                    self.assertTrue(torch.equal(head.hidden.weight.grad, torch.zeros_like(head.hidden.weight.grad)))
                else:
                    self.assertGreater(float(head.hidden.weight.grad.norm()), 0.)
                torch.nn.utils.clip_grad_norm_(parameters, 1.)
                optimizer.step()
            import run_joint_objectives_v2 as joint
            changes = joint.parameter_change_proof(initial_codec, codec.state_dict())
            self.assertGreater(changes['E']['changed_parameter_tensors'], 0)
            self.assertGreater(changes['D']['changed_parameter_tensors'], 0)
            self.assertGreater(runner.head_change_proof(initial_head, head.state_dict())['changed_parameter_tensors'], 0)
            for name, value in frozen.state_dict().items():
                self.assertTrue(torch.equal(value, initial_f[name]))
            self.assertTrue(all(parameter.grad is None for parameter in frozen.parameters()))
            self.assertFalse(torch.cuda.is_initialized())


if __name__ == '__main__':
    unittest.main()
