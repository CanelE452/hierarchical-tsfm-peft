"""CPU-only cost parent control tests; no GPU or actual subprocess launched."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cost_campaign as c
from resources import BudgetExceeded


class Lease:
    remaining=100
    def check(self):pass
    def child_environment(self):return {}


class CostCampaignTests(unittest.TestCase):
    def folders(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        private,public=Path(temp.name)/'private',Path(temp.name)/'public'
        private.mkdir();public.mkdir()
        return private,public

    def test_native_naive_and_ridge_cost_device_matches_final_route_policy(self):
        routes=[(dict(run_id=family,family=family,dataset='toy'),None) for family in ('CURRENT','SEASONAL','RAW_RIDGE','DIRECT_G')]
        plan=c.condition_plan(routes)
        fresh=[row for row in plan if row['mode']=='worker' and row['batch']==1]
        self.assertEqual([row['device'] for row in fresh],['cpu','cpu','cpu','cuda'])

    def test_prelaunch_budget_has_explicit_condition_receipt(self):
        private,public=self.folders();lease=Lease();lease.remaining=20
        with self.assertRaises((BudgetExceeded,TimeoutError)):
            c.execute({},None,'cuda',1,'worker',private,public,'condition',lease)
        self.assertTrue((public/'condition.execution.json').is_file())
        self.assertEqual(json.loads((public/'condition.execution.json').read_text())['failure_class'],'DEFERRED_RESOURCE')

    def test_child_budget_is_not_reported_as_implementation_failure(self):
        private,public=self.folders()
        def process(*args,**kwargs):
            kwargs['stdout'].write(b'resources.BudgetExceeded: Reserve 15 seconds for teardown\n')
            return type('Process',(),{'wait':lambda self:1})()
        with patch.object(c.subprocess,'Popen',side_effect=process):
            with self.assertRaises(BudgetExceeded):
                c.execute({},None,'cuda',1,'worker',private,public,'condition',Lease())
        self.assertEqual(json.loads((public/'condition.execution.json').read_text())['failure_class'],'DEFERRED_RESOURCE')

    def test_child_identity_failure_is_preserved_and_propagated(self):
        private,public=self.folders()
        def process(*args,**kwargs):
            kwargs['stdout'].write(b'ValueError: Raw route foundation identity differs from its accuracy artifact\n')
            return type('Process',(),{'wait':lambda self:1})()
        with patch.object(c.subprocess,'Popen',side_effect=process):
            with self.assertRaises(ValueError):
                c.execute({},None,'cuda',1,'worker',private,public,'condition',Lease())
        self.assertEqual(json.loads((public/'condition.execution.json').read_text())['failure_class'],'FAILED_IDENTITY_CONTRACT')

    def test_initial_reservation_exhaustion_preserves_full_pending_matrix(self):
        private,public=self.folders()
        with patch.multiple(c,HERE=public,CACHE=private), \
             patch.object(c.data,'load_task',return_value={'name':'toy'}), \
             patch.object(c.route_catalog,'build_route',return_value=(dict(run_id='g',family='DIRECT_G',dataset='toy'),None)), \
             patch.object(c,'GPUReservation',side_effect=BudgetExceeded('already exhausted')), \
             patch.object(c.sys,'argv',['cost_campaign.py','--dataset','toy','--run-id','g']):
            c.main()
        receipt=next((public/'costs').glob('*/COST_CAMPAIGN_RECEIPT.json'))
        result=json.loads(receipt.read_text())
        self.assertEqual(result['status'],'DEFERRED_RESOURCE')
        self.assertEqual(len(result['remaining_conditions']),4)

    def test_budget_preserves_all_remaining_fresh_and_balanced_condition_names(self):
        private,public=self.folders()
        routes=[(dict(run_id='raw',family='RAW',dataset='toy'),'small'),
                (dict(run_id='g',family='DIRECT_G',dataset='toy'),None)]
        with patch.object(c,'execute',side_effect=BudgetExceeded('budget reached')):
            result=c.run_conditions(routes,private,public,Lease())
        self.assertEqual(result['status'],'DEFERRED_RESOURCE')
        remaining={item['label'] for item in result['remaining_conditions']}
        self.assertIn('fresh_b1_m000_rowsfull',remaining)
        self.assertIn('fresh_b8_m001_rows128',remaining)
        self.assertIn('balanced_b1_g00',remaining)
        self.assertIn('balanced_b8_g00',remaining)

    def test_OOM_condition_does_not_prevent_independent_fresh_conditions(self):
        private,public=self.folders();calls=[]
        routes=[(dict(run_id='raw',family='RAW',dataset='toy'),'small'),
                (dict(run_id='g',family='DIRECT_G',dataset='toy'),None)]
        def execute(*args):
            label=args[-2];calls.append(label)
            if 'rowsfull' in label:return None
            return dict(summaries=[dict(phase='timed',median_seconds=.01)])
        with patch.object(c,'execute',side_effect=execute):
            result=c.run_conditions(routes,private,public,Lease())
        self.assertIn('fresh_b1_m001_rows128',calls)
        self.assertIn('balanced_b8_g00',calls)
        self.assertFalse(result['remaining_conditions'])


if __name__=='__main__':unittest.main()
