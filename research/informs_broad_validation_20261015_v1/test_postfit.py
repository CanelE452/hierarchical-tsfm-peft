"""CPU-only integration guard tests; no campaign registry or GPU writes."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import postfit as p
import prediction_cache
import data_contract as data
import test_combination as fixtures


class Lease:
    deadline = 10**12
    _entered = True
    def __init__(self):
        self.lock = type('Lock', (), {'file': object()})()
    def check(self):
        pass


class PostfitTests(unittest.TestCase):
    def rows(self):
        return [dict(experiment_id=f'primary_synthetic_E2_{family}_k{k}_s92601_{suffix}',
                     dataset='synthetic', fold='primary', block='E2', family=family,
                     latent_k=str(k), seed='92601', execution_status='PENDING_QUEUE')
                for family, k, suffix in [('CONDITIONAL', 1, 'B2'), ('PAST_CONTROL', 0, 'G'),
                                          ('RAW_CALIBRATED', 0, 'RAW')]]

    def exercise_missing_parent(self, missing):
        task, *_ = fixtures.CombinationTests().toy()
        calls, statuses = [], {}
        class Pool:
            def __init__(self, *args, **kwargs): pass
            def close(self): pass
            def predict(self, run_id, *args):
                if f'_E1_{missing}_' in run_id:
                    raise ValueError('controlled parent absent')
                return {'metadata': {'run_id': run_id}}
        def fitted(spec, *args):
            calls.append(spec['variant'])
            return dict(execution_status='COMPLETE_VALID', training_status='toy', updates=0)
        def raw_fit(*args):
            calls.append('raw_calibration')
            return dict(status='COMPLETE_VALID', training_status='toy')
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); (root/'cache').mkdir()
            with patch.multiple(p, HERE=root/'public', CACHE=root/'cache', ParentPool=Pool), \
                 patch.object(p.data, 'load_task', return_value=task), \
                 patch.object(p.campaign, 'registry_rows', return_value=self.rows()), \
                 patch.object(p.campaign, 'record_status', side_effect=lambda rid, **kw: statuses.update({rid: kw})), \
                 patch.object(p, 'check_storage_budget'), \
                 patch.object(p.audit_campaign, 'prepare_combination_reference', return_value={}), \
                 patch.object(p.audit_campaign, 'wrap_combination_output'), \
                 patch.object(p.combination, 'fit_combination', side_effect=fitted), \
                 patch.object(p.raw_calibration, 'fit_raw_calibration', side_effect=raw_fit):
                p.combine('synthetic', Lease(), [.25])
            failures = list((root/'cache/failures').glob('*.json'))
            self.assertTrue(failures)
            for path in failures:
                self.assertIn('traceback', json.loads(path.read_text()))
        return calls, statuses

    def test_one_bad_b_parent_keeps_past_and_raw_running(self):
        calls, statuses = self.exercise_missing_parent('B2')
        self.assertEqual(calls, ['past_control', 'raw_calibration'])
        self.assertEqual(sum(v['execution_status']=='EXECUTION_FAILED' for v in statuses.values()), 1)

    def test_missing_g_blocks_only_g_dependent_rows(self):
        calls, statuses = self.exercise_missing_parent('DIRECT_G')
        self.assertEqual(calls, ['raw_calibration'])
        self.assertEqual(sum(v['execution_status']=='EXECUTION_FAILED' for v in statuses.values()), 2)

    def test_reconcile_real_cpu_cache_rejects_tampering_before_status_write(self):
        task, *_ = fixtures.CombinationTests().toy()
        task = dict(task, name='beijing', period=4)
        run_id = 'primary_beijing_E1_CURRENT_k0_s92601'
        rows = [dict(experiment_id=run_id, dataset='beijing', family='CURRENT', fold='primary', block='E1',
                     latent_k='0', length='40', horizon='4', num_channels='3',
                     execution_status='PENDING_QUEUE', reason='')]
        updates=[]
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); cache=root/'cache'; public=root/'public'
            result=prediction_cache.predict_parent(run_id,task,'V_select',cache,public,None,baseline='CURRENT')
            manifest=public/f'predictions/{run_id}/V_select/PREDICTION_MANIFEST.json'
            batch=public/'execution_batches';batch.mkdir()
            receipt=dict(status='ACTUAL_V_INFERENCE_COMPLETE',GPU_calls=0,T_final_accessed=False,
                         source_sha256=data.sha256_file(Path(p.__file__).with_name('cpu_baselines.py')),
                         outputs=[dict(run_id=run_id,score=result['score'],actual_F_streams=0,
                          prediction_manifest_relative_path=manifest.relative_to(public).as_posix())])
            (batch/'cpu_baselines_toy.json').write_text(json.dumps(receipt))
            with patch.multiple(p,HERE=public,CACHE=cache), \
                 patch.object(p.campaign,'registry_rows',return_value=rows), \
                 patch.object(p.campaign,'record_status',side_effect=lambda *a,**kw:updates.append((a,kw))):
                with self.assertRaises(Exception): p.reconcile_cpu_baselines(LeaseInactive())
                self.assertEqual(updates,[])
                outcome=p.reconcile_cpu_baselines(Lease())
                self.assertEqual(outcome['reconciled_run_ids'],[run_id])
                self.assertEqual(len(updates),1)
                altered=copy.deepcopy(result['manifest']);altered['cost_record']['actual_F_streams']=1
                manifest.write_text(json.dumps(altered));updates.clear()
                with self.assertRaises(ValueError):p.reconcile_cpu_baselines(Lease())
                self.assertEqual(updates,[])


class LeaseInactive(Lease):
    _entered=False


if __name__=='__main__':unittest.main()
