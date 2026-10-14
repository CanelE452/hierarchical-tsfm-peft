"""Cadence repair plus unchanged scientific CPU fixtures; no actual assets."""
from pathlib import Path
from types import SimpleNamespace
import ast
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_joint_objectives_v2 as joint_v2
import test_run_joint_objectives as fixtures


class ObjectiveContractV2(fixtures.ObjectiveContractTests):
    def setUp(self):
        self.fixture_module = patch.object(fixtures, 'joint', joint_v2)
        self.fixture_module.start()
    def tearDown(self):
        self.fixture_module.stop()


class FrozenGradientV2(fixtures.FrozenGradientTests):
    def setUp(self):
        self.fixture_module = patch.object(fixtures, 'joint', joint_v2)
        self.fixture_module.start()
    def tearDown(self):
        self.fixture_module.stop()


class CadenceTests(unittest.TestCase):
    def test_numerical_AST_and_recipe_constants_equal_preserved_v1(self):
        original = ast.parse((HERE / 'run_joint_objectives.py').read_text(encoding='utf-8'))
        current = ast.parse((HERE / 'run_joint_objectives_v2.py').read_text(encoding='utf-8'))
        names = ('training_order', 'observed_macro_loss', 'objective_losses', 'choose_checkpoint',
                 'forward_trace', 'parameter_group', 'parameter_gradient_proof', 'parameter_change_proof',
                 'array_score', 'prediction_scores', 'save_checkpoint', 'predict', 'parse_args', 'main')
        for name in names:
            with self.subTest(function=name):
                left = next(node for node in original.body if isinstance(node, ast.FunctionDef) and node.name == name)
                right = next(node for node in current.body if isinstance(node, ast.FunctionDef) and node.name == name)
                self.assertEqual(ast.dump(left, include_attributes=False), ast.dump(right, include_attributes=False))
        constants = ('OBJECTIVES', 'UPDATES', 'CHECKPOINTS', 'BATCH_SIZE', 'LEARNING_RATE',
                     'WEIGHT_DECAY', 'CLIP_NORM', 'LOCAL_SECONDS', 'DIAGNOSTIC_UPDATES')
        for name in constants:
            with self.subTest(constant=name):
                def value(tree):
                    return next(node.value for node in tree.body if isinstance(node, ast.Assign)
                                and any(isinstance(target, ast.Name) and target.id == name for target in node.targets))
                self.assertEqual(ast.dump(value(original)), ast.dump(value(current)))

    def test_fast_keeps_base_guard_full_keeps_supplementary_guard(self):
        calls = []
        limit = SimpleNamespace(check=lambda: calls.append('full'),
                                shared=SimpleNamespace(check=lambda: calls.append('base')))
        guard = joint_v2.GuardCadence(limit, lambda: calls.append('local150'))
        guard.fast()
        guard.full()
        self.assertEqual(calls, ['local150', 'base', 'local150', 'full'])
        self.assertEqual(guard.summary()['explicit_calls'], {'full': 1, 'fast_base': 1})
        self.assertEqual(guard.summary()['local_process_cap_seconds'], 150)

    def test_local_cap_blocks_both_guard_modes_without_extending_cap(self):
        def rejected():
            raise RuntimeError('local150')
        limit = SimpleNamespace(check=lambda: self.fail('full called after cap'),
                                shared=SimpleNamespace(check=lambda: self.fail('base called after cap')))
        guard = joint_v2.GuardCadence(limit, rejected)
        for callback in (guard.fast, guard.full):
            with self.assertRaisesRegex(RuntimeError, 'local150'):
                callback()

    def test_every_CP_NPZ_write_has_full_guard_before_and_after(self):
        tree = ast.parse((HERE / 'run_joint_objectives_v2.py').read_text(encoding='utf-8'))
        execute = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'execute')
        writes = []
        def check_call(node):
            return isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'check'
        for parent in ast.walk(execute):
            for attribute in ('body', 'orelse', 'finalbody'):
                sequence = getattr(parent, attribute, [])
                if not isinstance(sequence, list):
                    continue
                for index, node in enumerate(sequence):
                    called = getattr(node, 'value', None)
                    if not isinstance(called, ast.Call):
                        continue
                    function = called.func
                    name = function.id if isinstance(function, ast.Name) else function.attr if isinstance(function, ast.Attribute) else ''
                    if name in ('save_checkpoint', 'save_npz'):
                        writes.append(name)
                        self.assertGreater(index, 0)
                        self.assertLess(index + 1, len(sequence))
                        self.assertTrue(check_call(sequence[index - 1]))
                        self.assertTrue(check_call(sequence[index + 1]))
        self.assertEqual(len(writes), 5)


if __name__ == '__main__':
    unittest.main()
