"""Keep actual cost receipts while excluding unrelated diagnostic metadata."""
import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest


ORIGINAL_BLOCK = '''    for path in sorted(public.rglob("*.json")):
        relative = path.relative_to(public)
        if "figures" in relative.parts or not (any("cost" in part.lower() for part in relative.parts)):
            continue
'''
REVISED_BLOCK = '''    for path in sorted((public / "costs").rglob("*.json")):
        relative = path.relative_to(public)
'''


class FigureCostNamespaceTests(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module(os.environ.get(
            "FIGURE_NAMESPACE_TEST_TARGET", "report_figures_cost_namespace_v2"))
        self.temporary = tempfile.TemporaryDirectory(prefix="figure_namespace_")
        self.addCleanup(self.temporary.cleanup)
        self.public = Path(self.temporary.name) / "public"
        self.directory = self.public / "new_figure_output"
        self.directory.mkdir(parents=True)

    def cost_receipt(self):
        folder = self.public / "costs" / "actual_protocol"
        folder.mkdir(parents=True)
        trials = folder / "trials.csv.gz"
        trials.write_bytes(b"synthetic sealed trial bytes")
        receipt = folder / "receipt.json"
        receipt.write_text(json.dumps(dict(
            scope="fresh_process_single_route", status="COMPLETE_PROTOCOL",
            trials_file=trials.name,
            trials_sha256=hashlib.sha256(trials.read_bytes()).hexdigest(),
            summaries=[])), encoding="utf-8")
        return receipt, trials

    def diagnostic_list(self):
        folder = self.public / "diagnostics" / "final_cost_queue_ready"
        folder.mkdir(parents=True)
        path = folder / "EXACT_PEMS9_REUSE.json"
        path.write_text(json.dumps([dict(run_id="synthetic_reference")]), encoding="utf-8")
        return path

    def test_diagnostics_list_not_read_and_actual_cost_receipt_retained(self):
        receipt, trials = self.cost_receipt()
        diagnostic = self.diagnostic_list()
        inputs = {}
        result = self.module._costs([], self.directory, "V_select", self.public, inputs)
        self.assertEqual(result, [])
        included = {key[1] for key in inputs}
        self.assertEqual(included, {
            receipt.relative_to(self.public).as_posix(),
            trials.relative_to(self.public).as_posix()})
        self.assertNotIn(diagnostic.relative_to(self.public).as_posix(), included)

    def test_malformed_cost_namespace_list_still_fails_closed(self):
        self.cost_receipt()
        (self.public / "costs" / "zz_malformed.json").write_text("[]", encoding="utf-8")
        with self.assertRaises(AttributeError):
            self.module._costs([], self.directory, "V_select", self.public, {})

    def test_changed_trial_bytes_still_fail_closed(self):
        _, trials = self.cost_receipt()
        trials.write_bytes(b"changed trial bytes")
        with self.assertRaisesRegex(ValueError, "trials hash changed"):
            self.module._costs([], self.directory, "V_select", self.public, {})

    def test_unsupported_cost_scope_remains_ignored(self):
        folder = self.public / "costs"
        folder.mkdir()
        (folder / "metadata.json").write_text(json.dumps(dict(scope="planning_only")), encoding="utf-8")
        self.assertEqual(self.module._costs([], self.directory, "V_select", self.public, {}), [])

    def test_original_failure_is_reproduced_by_same_fixture(self):
        self.cost_receipt()
        self.diagnostic_list()
        original = importlib.import_module("report_figures")
        with self.assertRaisesRegex(AttributeError, "get"):
            original._costs([], self.directory, "V_select", self.public, {})

    def test_variant_is_exact_namespace_only_patch(self):
        root = Path(__file__).resolve().parent
        original = (root / "report_figures.py").read_bytes()
        variant = (root / "report_figures_cost_namespace_v2.py").read_bytes()
        old = ORIGINAL_BLOCK.replace("\n", "\r\n").encode() if b"\r\n" in original else ORIGINAL_BLOCK.encode()
        new = REVISED_BLOCK.replace("\n", "\r\n").encode() if b"\r\n" in original else REVISED_BLOCK.encode()
        self.assertEqual(original.count(old), 1)
        self.assertEqual(variant, original.replace(old, new, 1))
        before = ast.parse(original)
        after = ast.parse(variant)
        functions_before = {node.name: ast.dump(node) for node in before.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        functions_after = {node.name: ast.dump(node) for node in after.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        self.assertEqual(set(functions_before), set(functions_after))
        changed = [name for name in functions_before if functions_before[name] != functions_after[name]]
        self.assertEqual(changed, ["_costs"])


if __name__ == "__main__":
    unittest.main()
