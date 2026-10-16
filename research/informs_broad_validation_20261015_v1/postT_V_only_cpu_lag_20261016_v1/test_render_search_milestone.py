"""Synthetic metadata/aggregate tests; never load research checkpoints or models."""
import contextlib
import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("milestone_under_test", HERE / "render_search_milestone.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def curves(values=(.6, .5, .4, .45)):
    return [dict(epoch=epoch, exposures=exposures, updates=updates, astop_mse=value,
                 segment_seconds=0. if index == 0 else 1., clipped_steps=0)
            for index, (epoch, exposures, updates, value) in enumerate(zip(module.EPOCHS, module.EXPOSURES, module.UPDATES, values))]


def write_csv(path, records):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=module.FIELDS)
        writer.writeheader()
        writer.writerows(records)


def records():
    result = []
    for family in module.COLORS:
        rows = curves()
        selected = module.selected_curve(rows)
        result.extend(dict(run_id=module.run_id(family), family=family, **row, selected_by_Astop=int(row is selected)) for row in rows)
    return result


class MilestoneTests(unittest.TestCase):
    def test_initial_zero_and_earliest_ties(self):
        for values, expected in (((.3, .3, .4, .5), 0.), ((.6, .4, .4, .5), .25), ((.6, .5, .4, .3), 3.)):
            self.assertEqual(module.validate_curves(curves(values))["epoch"], expected)

    def test_wrong_endpoints_counts_and_nonfinite(self):
        changes = [(0, "epoch", .1), (1, "exposures", 2497), (3, "updates", 1862), (2, "astop_mse", float("nan")),
                   (2, "segment_seconds", float("inf")), (0, "clipped_steps", 1), (1, "clipped_steps", 157)]
        for index, key, value in changes:
            rows = curves()
            rows[index][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                module.validate_curves(rows)
        with self.assertRaises(ValueError):
            module.validate_curves(curves()[:3])

    def test_saved_and_events_contract(self):
        plan = dict(task_contract=dict(L=512, H=48, C=21, columns=["x"], array_sha256={"x": "a"},
                    data_sha256="data", search_scaler_sha256="scaler"), foundation_tensor_sha256="model",
                    source39={name: name for name in module.SOURCE_NAMES})
        condition = dict(run_id=module.run_id("B0"), family="B0", spec=dict(fold="primary", family="B0"))
        contract = dict(module.expected_contract(plan, condition), initialization_sha="a"*64)
        saved = dict(contract=contract, updates=1863, sampler=dict(exposures=29769),
                     extra=dict(curves=curves(), completed_epochs=list(module.EPOCHS), best_mse=.4, best_epoch=1.))
        events = [dict(status="SEGMENT_COMPLETE", epoch=epoch, exposures=exposure, updates=update, contract=contract)
                  for epoch, exposure, update in zip(module.EPOCHS, module.EXPOSURES, module.UPDATES)]
        self.assertEqual(len(module.validate_saved(saved, events, plan, condition)), 4)
        for location in ("event", "best", "sampler", "source"):
            altered, edited = copy.deepcopy(saved), copy.deepcopy(events)
            if location == "event": edited[3]["status"] = "BUDGET_LIMIT"
            if location == "best": altered["extra"]["best_epoch"] = 3.
            if location == "sampler": altered["sampler"]["exposures"] -= 1
            if location == "source": altered["contract"]["source_sha"] = "other"
            with self.subTest(location=location), self.assertRaises(ValueError):
                module.validate_saved(altered, edited, plan, condition)

    def test_csv_exact_rows_duplicate_and_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = records()
            for name, rows in (("valid", original), ("short", original[:-1]), ("duplicate", original[:-1]+[original[0]])):
                path = root / (name + ".csv")
                write_csv(path, rows)
                if name == "valid": self.assertEqual(len(module.read_records(path)), 16)
                else:
                    with self.assertRaises(ValueError): module.read_records(path)
            wrong = copy.deepcopy(original)
            wrong[0]["selected_by_Astop"] = 1
            path = root / "wrong.csv"
            write_csv(path, wrong)
            with self.assertRaises(ValueError): module.read_records(path)

    def test_output_boundary_and_existing_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / module.NAME / "diagnostics" / "fresh"
            module.require_new_output(valid, root)
            for path in (root / "escape", root / module.NAME / "diagnostics"):
                with self.assertRaises(ValueError): module.require_new_output(path, root)
            existing = root / "existing"
            existing.mkdir()
            with self.assertRaises(FileExistsError): module.require_new_output(existing)
            with self.assertRaises(ValueError): module.require_new_output(valid, root, root / "preview.svg", root)

    def test_default_has_no_torch_or_writes(self):
        result = subprocess.run([sys.executable, "-B", str(HERE / "render_search_milestone.py")], capture_output=True, text=True, check=True)
        value = json.loads(result.stdout)
        self.assertFalse(value["torch_imported"])
        self.assertEqual(value["writes"], 0)

    def test_csv_render_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "synthetic.csv"
            write_csv(path, records())
            output = root / "render"
            with contextlib.redirect_stdout(io.StringIO()):
                module.main(["--input-csv", str(path), "--output", str(output), "--preview", str(root / "preview.png")])
            self.assertTrue((output / module.SVG_NAME).is_file())
            binding = module.read(output / "FIGURE_BINDING.json")
            self.assertEqual(binding["public_csv_sha256"], module.sha(path))
            self.assertEqual(binding["svg_sha256"], module.sha(output / module.SVG_NAME))
            self.assertFalse(binding["final_V_result"])
            svg_before = module.sha(output / module.SVG_NAME)
            with self.assertRaises(FileExistsError):
                module.main(["--input-csv", str(path), "--output", str(output)])
            self.assertEqual(module.sha(output / module.SVG_NAME), svg_before)


if __name__ == "__main__":
    unittest.main()
