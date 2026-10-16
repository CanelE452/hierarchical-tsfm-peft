"""A locked final selection blocks new fitting before registration or GPU use."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import campaign
import deepen
import postfit
import registry_extension


class SelectionTrainingGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="informs_selection_guard_")
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.row=dict(experiment_id="pending",dataset="toy",panel="",fold="primary",block="E1",family="B0",
            num_channels="3",latent_k="1",seed="92601",length="64",horizon="4",
            execution_status="PENDING_QUEUE",requirement="MANDATORY_MINIMUM")
        with (self.root/"EXPERIMENT_REGISTRY.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(self.row));writer.writeheader();writer.writerow(self.row)

    def lock(self,value=True):
        (self.root/"SELECTION_LOCK.json").write_text(json.dumps(dict(locked=value)))

    def test_locked_selection_rejects_new_registry_ids_before_any_write(self):
        self.lock()
        before={p.name:p.read_bytes() for p in self.root.iterdir()}
        with self.assertRaisesRegex(ValueError,"selection"):
            registry_extension.register(self.root,[dict(self.row,experiment_id="new")],"test")
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.root.iterdir()})

    def test_existing_registry_id_no_addition_remains_read_only(self):
        self.lock()
        before=(self.root/"EXPERIMENT_REGISTRY.csv").read_bytes()
        self.assertEqual(registry_extension.register(self.root,[self.row],"read-only"),[])
        self.assertEqual(before,(self.root/"EXPERIMENT_REGISTRY.csv").read_bytes())

    def test_no_lock_and_explicit_unlocked_draft_allow_normal_registration(self):
        self.assertEqual(len(registry_extension.register(self.root,[dict(self.row,experiment_id="new")],"before selection")),1)
        self.lock(False)
        self.assertEqual(len(registry_extension.register(self.root,[dict(self.row,experiment_id="new2")],"draft only")),1)
        receipts=list(self.root.glob("EXPERIMENT_REGISTRY_EXTENSION_*.json"))
        self.assertEqual(len(receipts),2)
        self.assertTrue(all(json.loads(path.read_text())["before_T_selection"] for path in receipts))

    def test_equal_clock_ticks_keep_two_distinct_immutable_registration_receipts(self):
        (self.root/"EXPERIMENT_REGISTRY_INITIAL.csv").write_bytes((self.root/"EXPERIMENT_REGISTRY.csv").read_bytes())
        with patch.object(registry_extension.time,"time_ns",return_value=123456789):
            registry_extension.register(self.root,[dict(self.row,experiment_id="same_tick_a")],"first")
            first=next(self.root.glob("EXPERIMENT_REGISTRY_EXTENSION_*.csv"))
            before=first.read_bytes()
            registry_extension.register(self.root,[dict(self.row,experiment_id="same_tick_b")],"second")
        files=list(self.root.glob("EXPERIMENT_REGISTRY_EXTENSION_*.csv"))
        self.assertEqual(len(files),2)
        self.assertEqual(first.read_bytes(),before)
        ids=[]
        for path in files:
            proof=json.loads(path.with_suffix(".json").read_text())
            self.assertEqual(proof["csv_sha256"],hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertTrue(proof["before_execution"])
            self.assertTrue(proof["before_T_selection"])
            ids.extend(proof["run_ids"])
        self.assertCountEqual(ids,["same_tick_a","same_tick_b"])
        with (self.root/"EXPERIMENT_REGISTRY.csv").open(newline="") as stream:
            self.assertCountEqual([row["experiment_id"] for row in csv.DictReader(stream)],["pending","same_tick_a","same_tick_b"])
        import audit_campaign
        errors=[]
        _,declared,_,extensions=audit_campaign._registry_contract(self.root,errors)
        self.assertEqual(errors,[])
        self.assertEqual(len(declared),3)
        self.assertEqual(len(extensions),2)

    def test_campaign_main_and_direct_fit_reject_before_GPU(self):
        self.lock()
        with patch.object(campaign,"HERE",self.root),patch.object(campaign,"GPUReservation") as gpu, \
             patch.object(campaign,"FrozenBolt") as foundation:
            with patch("sys.argv",["campaign.py","--run-id","pending"]):
                with self.assertRaisesRegex(ValueError,"selection"):campaign.main()
            with self.assertRaisesRegex(ValueError,"selection"):
                campaign.execute_fits(["pending"],[.25],10**12)
            gpu.assert_not_called();foundation.assert_not_called()

    def test_postfit_training_phases_reject_before_GPU_or_registry_reconciliation(self):
        self.lock()
        for phase in ("combine","controls","ordered"):
            with self.subTest(phase=phase),patch.object(campaign,"HERE",self.root), \
                 patch.object(postfit,"HERE",self.root),patch.object(postfit,"GPUReservation") as gpu, \
                 patch.object(postfit,"reconcile_cpu_baselines") as reconcile, \
                 patch("sys.argv",["postfit.py","--phase",phase,"--dataset","toy"]):
                with self.assertRaisesRegex(ValueError,"selection"):postfit.main()
                gpu.assert_not_called();reconcile.assert_not_called()

    def test_deepen_rejects_even_existing_registered_extension_before_GPU(self):
        self.lock()
        with patch.object(campaign,"HERE",self.root),patch.object(deepen,"HERE",self.root), \
             patch.object(deepen,"GPUReservation") as gpu,patch.object(deepen.registry_extension,"register") as register, \
             patch("sys.argv",["deepen.py","--parent-id","pending","--max-epoch","3"]):
            with self.assertRaisesRegex(ValueError,"selection"):deepen.main()
            gpu.assert_not_called();register.assert_not_called()

    def test_preselection_campaign_flow_reaches_existing_fit_driver(self):
        lease=Mock(deadline=10**12)
        owner=Mock();owner.__enter__=Mock(return_value=lease);owner.__exit__=Mock(return_value=False)
        with patch.object(campaign,"HERE",self.root),patch.object(campaign,"GPUReservation",return_value=owner), \
             patch.object(campaign,"execute_fits") as fits,patch("sys.argv",["campaign.py","--run-id","pending"]):
            campaign.main()
            fits.assert_called_once_with(["pending"],[.25],10**12-5,lease)


if __name__=="__main__":unittest.main()
