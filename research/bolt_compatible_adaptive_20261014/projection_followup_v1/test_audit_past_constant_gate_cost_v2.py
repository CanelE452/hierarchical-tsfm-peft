"""Reproduce date-prefix replacement error; keep the registered seed and rules."""
import ast
from pathlib import Path
import unittest

import audit_past_constant_gate_cost_v1 as original
import audit_past_constant_gate_cost_v2 as repaired

repaired.initialize_numpy()


def declared_seed_guard(source):
    values = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Subscript):
            left = node.left
            if isinstance(left.value, ast.Name) and left.value.id == "cost" and isinstance(left.slice, ast.Constant) and left.slice.value == "order_seed":
                values.append(ast.literal_eval(node.comparators[0]))
    if len(values) != 1:
        raise AssertionError("Exactly one recorded cost seed guard required")
    return values[0]


class RegisteredSeedRepair(unittest.TestCase):
    def test_original_seed_guard_rejects_the_registered_schedule_seed(self):
        old = Path(original.__file__).read_text(encoding="utf-8")
        self.assertEqual(declared_seed_guard(old), 24261015)
        self.assertNotEqual(declared_seed_guard(old), 20261015)

    def test_new_seed_guard_and_independent_Latin_schedule_keep_registered_seed(self):
        new = Path(repaired.__file__).read_text(encoding="utf-8")
        self.assertEqual(declared_seed_guard(new), 20261015)
        order = repaired.np.random.default_rng(20261015).permutation(6)
        expected = [[repaired.ROUTES[index] for index in repaired.np.roll(order, -repeat)] for repeat in range(6)]
        self.assertEqual(repaired.independent_schedule(), expected)

    def test_executed_source_diff_contains_only_one_seed_literal_correction(self):
        old = Path(original.__file__).read_text(encoding="utf-8")
        new = Path(repaired.__file__).read_text(encoding="utf-8")
        before, after = 'cost["order_seed"] == 24261015', 'cost["order_seed"] == 20261015'
        self.assertEqual(old.count(before), 1)
        self.assertEqual(old.replace(before, after, 1), new)

    def test_full_metadata_guard_uses_real_helper_seed_and_rejects_corrupted_seed(self):
        helper = ast.parse((repaired.HERE / "whole_past_constant_gate_cost_v1.py").read_text(encoding="utf-8-sig"))
        seed = next(ast.literal_eval(node.value) for node in helper.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "ORDER_SEED" for target in node.targets))
        new_source = Path(repaired.__file__).read_text(encoding="utf-8")
        order = repaired.np.random.default_rng(seed).permutation(6)
        schedule = [[repaired.ROUTES[index] for index in repaired.np.roll(order, -repeat)] for repeat in range(6)]
        def checked(metadata):
            repaired.require(metadata["order_seed"] == declared_seed_guard(new_source)
                             and metadata["schedule"] == repaired.independent_schedule(), "REGISTERED_METADATA_SEED_AND_LATIN6")
        checked({"order_seed": seed, "schedule": schedule})
        self.assertEqual(seed, 20261015)
        with self.assertRaisesRegex(repaired.prior.AuditFailure, "METADATA_SEED_AND_LATIN6"):
            checked({"order_seed": 24261015, "schedule": schedule})
        for route in repaired.ROUTES:
            self.assertEqual(sorted(row.index(route) for row in schedule), list(range(6)))


if __name__ == "__main__":
    unittest.main()
