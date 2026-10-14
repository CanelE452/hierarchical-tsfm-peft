"""CPU-only loss/guard/source-contract fixtures; no real Appliances assets."""
from pathlib import Path
import ast
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_appliances_losses_v1 as app


class ApplianceLossTests(unittest.TestCase):
    def test_describe_is_preparation_only(self):
        self.assertFalse(app.describe()['actual_data_opened'])
        self.assertFalse(app.describe()['GPU_executed'])
        self.assertEqual(app.OBJECTIVES, ('MSE', 'MAE', 'HUBER_DELTA1'))

    def test_MSE_finite_matches_original_observed_macro(self):
        import torch
        torch.manual_seed(241)
        prediction = torch.randn(4, 48, 26, requires_grad=True)
        target = torch.randn_like(prediction)
        mask = torch.rand_like(prediction) > .4
        mask[:, :, 4] = False
        count = mask.sum((0, 1))
        old = (torch.where(mask, (prediction-target).square(), 0.).sum((0, 1)) / count.clamp_min(1))[count > 0].mean()
        actual = app.observed_objective_loss(prediction, target, mask, 'MSE')
        self.assertTrue(torch.equal(old, actual))
        self.assertTrue(torch.equal(torch.autograd.grad(old, prediction, retain_graph=True)[0],
                                    torch.autograd.grad(actual, prediction)[0]))

    def test_all_losses_missing_nan_gradients_finite_zero_inactive_channels(self):
        import torch
        for objective in app.OBJECTIVES:
            with self.subTest(objective=objective):
                prediction = torch.tensor([[[1., 2., 3.], [2., 4., 5.]]], requires_grad=True)
                target = torch.tensor([[[0., float('nan'), float('nan')], [0., 1., float('nan')]]])
                mask = torch.tensor([[[True, False, False], [True, True, False]]])
                loss = app.observed_objective_loss(prediction, target, mask, objective)
                loss.backward()
                self.assertTrue(bool(torch.isfinite(loss)))
                self.assertTrue(bool(torch.isfinite(prediction.grad).all()))
                self.assertEqual(float(prediction.grad[0, 0, 1]), 0.)
                self.assertTrue(torch.equal(prediction.grad[:, :, 2], torch.zeros_like(prediction.grad[:, :, 2])))

    def test_Huber_delta1_preserves_quadratic_body_and_caps_tail_gradient(self):
        import torch
        gradients = {}
        for objective in app.OBJECTIVES:
            prediction = torch.tensor([[[.2], [5.]]], requires_grad=True)
            loss = app.observed_objective_loss(prediction, torch.zeros_like(prediction), torch.ones_like(prediction, dtype=torch.bool), objective)
            gradients[objective] = torch.autograd.grad(loss, prediction)[0].reshape(-1)
        self.assertAlmostEqual(float(gradients['HUBER_DELTA1'][0]), .1, places=6)
        self.assertAlmostEqual(float(gradients['HUBER_DELTA1'][1]), .5, places=6)
        self.assertGreater(float(gradients['MSE'][1]), float(gradients['HUBER_DELTA1'][1]))
        self.assertLess(float(gradients['HUBER_DELTA1'][0]), float(gradients['MAE'][0]))

    def test_RAW_reference_is_diagnostic_and_does_not_change_any_training_loss(self):
        import torch
        for objective in app.OBJECTIVES:
            prediction = torch.tensor([[[1., 2.]]], requires_grad=True)
            y = torch.zeros_like(prediction)
            mask = torch.ones_like(prediction, dtype=torch.bool)
            left = app.objective_losses(prediction, y, torch.zeros_like(prediction), mask, objective)[0]
            right = app.objective_losses(prediction, y, torch.ones_like(prediction)*100., mask, objective)[0]
            self.assertTrue(torch.equal(left, right))

    def test_actual_Y_TUNE_selection_keeps_initial_earliest_tie_for_MAE_Huber(self):
        rows = [{'tune_actual_Y_mse': .3}, {'tune_actual_Y_mse': .4}, {'tune_actual_Y_mse': .3}]
        self.assertEqual(app.choose_checkpoint(rows), 0)
        np.testing.assert_array_equal(app.training_order(92601, 256), app.training_order(92601, 256))

    def test_all_empty_observed_loss_rejected(self):
        import torch
        for objective in app.OBJECTIVES:
            with self.assertRaisesRegex(ValueError, 'NO_OBSERVED_TARGET'):
                app.observed_objective_loss(torch.zeros(2, 3, 2), torch.zeros(2, 3, 2), torch.zeros(2, 3, 2, dtype=torch.bool), objective)

    def test_v2_guard_cadence_and_update_recipe_unchanged(self):
        left = ast.parse((HERE/'run_joint_objectives_v2.py').read_text(encoding='utf-8'))
        right = ast.parse((HERE/'run_appliances_losses_v1.py').read_text(encoding='utf-8'))
        for name in ('GuardCadence', 'parameter_gradient_proof', 'parameter_change_proof', 'forward_trace', 'training_order', 'choose_checkpoint', 'predict'):
            with self.subTest(name=name):
                old = next(node for node in left.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name)
                current = next(node for node in right.body if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name)
                self.assertEqual(ast.dump(old, include_attributes=False), ast.dump(current, include_attributes=False))
        self.assertEqual((app.UPDATES, app.BATCH_SIZE, app.LEARNING_RATE, app.CLIP_NORM, app.LOCAL_SECONDS), (600,4,.001,1.,150))

    def test_restore_rejects_original_q_aux_method_change_before_model_construction(self):
        import prepare_appliances_inputs as bridge
        bundle = {'checkpoint_cases': {'92601': {'source_checkpoint': {'path':'synthetic','sha256':'1'*64},
                                                 'selected_update':600}}}
        payload = {'seed':92601,'update':600,'architecture':'ANCHOR_NONLINEAR_D','method':'MAIN','weight':.1}
        with patch.object(bridge,'load_codec_checkpoint_state',return_value=payload):
            with self.assertRaisesRegex(ValueError, 'APP_SELECTED_ORIGINAL_CP_METADATA'):
                app.restore_appliances_codec(SimpleNamespace(torch=None),bundle,92601,
                    SimpleNamespace(verified_file=lambda record,**kwargs:record))


if __name__ == '__main__':
    unittest.main()
