"""Check hard training allocation before any real task or GPU is loaded."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import deepen
from resources import BudgetExceeded


class DeepenBudgetTests(unittest.TestCase):
    def test_long_extension_keeps_final_audit_allocation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            parent = root / "runs" / "parent"
            parent.mkdir(parents=True)
            (parent / "BASE_FIT_RECEIPT.json").write_text(json.dumps(
                dict(status="COMPLETE_VALID", epochs=[.25])), encoding="utf-8")
            row = dict(experiment_id="parent", family="DIRECT_G", dataset="toy",
                       block="E1", fold="primary", requirement="MANDATORY_MINIMUM")
            with patch.multiple(deepen, HERE=root, CACHE=root / "cache"), \
                 patch.object(deepen.campaign, "registry_rows", return_value=[row]), \
                 patch.object(deepen.registry_extension, "require_pre_selection"), \
                 patch.object(deepen.registry_extension, "register"), \
                 patch.object(deepen.data, "load_task") as load, \
                 patch.object(deepen, "GPUReservation", side_effect=BudgetExceeded("reserved")) as reservation, \
                 patch.object(sys, "argv", ["deepen.py", "--parent-id", "parent", "--max-epoch", "1"]):
                with self.assertRaises(BudgetExceeded):
                    deepen.main()
                load.assert_not_called()
            self.assertEqual(reservation.call_args.kwargs["total_gpu_seconds"], 43200)
            self.assertEqual(reservation.call_args.kwargs["reserve_seconds"], 9000)
            self.assertEqual(reservation.call_args.kwargs["purpose"], "training")


if __name__ == "__main__":
    unittest.main()
