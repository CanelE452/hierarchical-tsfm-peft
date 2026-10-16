"""CPU toy checks for explicit final selection and guarded evaluation."""
import copy
import csv
import importlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

import psutil

import audit_campaign
import data_contract as data
import prediction_cache
import route_predictions
import test_route_predictions as fixtures


class AuditLease:
    purpose = 'audit'
    _entered = True
    def __init__(self):
        self.lock = type('Lock', (), {'file': object()})()
    def check(self):
        pass


class FinalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(Path(__file__).with_name('final_evaluation.py').is_file(), 'Final evaluation mechanism is missing')
        self.mod = importlib.import_module('final_evaluation')
        self.fx = fixtures.RoutePredictionTests()
        self.fx.setUp()
        self.addCleanup(self.fx.doCleanups)
        self.cache, self.public, self.task = self.fx.cache, self.fx.public, self.fx.task
        self.cache.mkdir(exist_ok=True)
        self.ids=[]

    def baseline(self, family='CURRENT'):
        run=f'primary_toy_E1_{family}_k0_s92601'
        self.fx.register(run, family)
        baseline='CURRENT' if family=='CURRENT' else dict(kind='SEASONAL', period=self.task['period'])
        prediction_cache.predict_parent(run,self.task,'V_select',self.cache,self.public,None,baseline=baseline)
        self.ids.append(run)
        return run

    def audit(self, partial=False):
        rows=self.fx.rows
        for row in rows:
            row.update(requirement='MANDATORY_MINIMUM', reason='0.25 screen only' if partial else '')
            if partial:row['execution_status']='PARTIAL_SCREEN025'
        path=self.public/'EXPERIMENT_REGISTRY.csv'
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        shutil.copyfile(path,self.public/'EXPERIMENT_REGISTRY_INITIAL.csv')
        result=audit_campaign.audit_all(self.public,self.cache)
        self.assertFalse(result['errors'])
        return result

    def lock(self):
        with patch.object(data,'load_task',return_value=self.task):
            return self.mod.lock_selection(self.ids,dict(stop_training=True,reason='Explicit toy stop decision'),self.public,self.cache)

    def test_requires_audit_stop_decision_and_quiescent_training(self):
        run=self.baseline()
        with self.assertRaises(ValueError):self.lock()
        self.audit()
        with patch.object(data,'load_task',return_value=self.task):
            with self.assertRaises(ValueError):
                self.mod.lock_selection(self.ids,dict(stop_training=False,reason='keep training'),self.public,self.cache)
        self.fx.rows[0]['execution_status']='RUNNING'
        path=self.public/'EXPERIMENT_REGISTRY.csv'
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(self.fx.rows[0]));writer.writeheader();writer.writerows(self.fx.rows)
        with self.assertRaises(ValueError):self.lock()
        self.fx.rows[0]['execution_status']='COMPLETE_VALID'
        with path.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(self.fx.rows[0]));writer.writeheader();writer.writerows(self.fx.rows)
        (self.cache/'ACTIVE_GPU_LEASE.json').write_text(json.dumps(dict(purpose='training',pid=os.getpid(),create_time=psutil.Process().create_time())))
        with self.assertRaises(ValueError):self.lock()

    def test_partial_lock_keeps_registry_and_never_reads_targets(self):
        run=self.baseline();self.audit(partial=True)
        before=(self.public/'EXPERIMENT_REGISTRY.csv').read_bytes()
        with patch.object(data,'target_window',side_effect=AssertionError('selection read a future target')):
            lock=self.lock()
        self.assertFalse(lock['mandatory_coverage_complete'])
        self.assertEqual(lock['mandatory_incomplete'][0]['execution_status'],'PARTIAL_SCREEN025')
        self.assertEqual((self.public/'EXPERIMENT_REGISTRY.csv').read_bytes(),before)
        self.assertIn(run,lock['allowed_route_sha'])
        self.assertIn('final_evaluation.py',lock['source_bindings'])
        self.assertEqual(lock['decision']['output_definition']['foundation_quantile'],.5)
        self.assertEqual(lock,self.lock())

    def test_reference_change_after_audit_is_rejected(self):
        run=self.baseline();self.audit()
        path=self.public/'predictions'/run/'V_select/REFERENCE_CONTRACT.json'
        value=json.loads(path.read_text())
        value['arrays']['target_normalized']['sha256']='changed after audit'
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):self.lock()

    def test_source_gate_precedes_task_or_T_target_loading(self):
        self.baseline();self.audit();lock=self.lock()
        lock['source_bindings']['deployment.py']='changed'
        (self.public/'SELECTION_LOCK.json').write_text(json.dumps(lock))
        with patch.object(data,'load_task',side_effect=AssertionError('task loaded before source gate')), \
             patch.object(data,'target_window',side_effect=AssertionError('T read before gate')):
            with self.assertRaises(ValueError):self.mod.execute_T(self.cache,self.public,AuditLease(),time.time()+30)

    def test_artifact_tamper_precedes_T_target_loading(self):
        run=self.fx.base_G();self.ids.append(run)
        prediction_cache.predict_parent(run,self.task,'V_select',self.cache,self.public,None)
        self.audit();lock=self.lock()
        relative=next(iter(lock['route_bindings'][run]['artifact_bindings'].values()))['cache_relative_path']
        path=self.cache/relative;path.write_bytes(path.read_bytes()+b'tamper')
        with patch.object(data,'target_window',side_effect=AssertionError('T read before artifact gate')):
            with self.assertRaises(ValueError):self.mod.execute_T(self.cache,self.public,AuditLease(),time.time()+30)

    def test_guarded_T_really_runs_cpu_toy_and_returns_paths_only(self):
        run=self.baseline();self.audit();self.lock()
        with patch.object(data,'load_task',return_value=self.task):
            result=self.mod.execute_T(self.cache,self.public,AuditLease(),time.time()+30)
        self.assertEqual(result['succeeded'],1)
        self.assertEqual(result['failed'],0)
        self.assertNotIn('pred_normalized',result)
        manifest=self.public/result['records'][0]['manifest_public_relative_path']
        self.assertEqual(json.loads(manifest.read_text())['split'],'T_final')

    def test_failed_replay_does_not_stop_independent_route(self):
        bad=self.baseline();good=self.baseline('SEASONAL_NAIVE')
        original=route_predictions.replay_route
        def replay(run_id,*args,**kwargs):
            if run_id==bad:raise ValueError('controlled one-route failure')
            return original(run_id,*args,**kwargs)
        with patch.object(data,'load_task',return_value=self.task),patch.object(route_predictions,'replay_route',side_effect=replay):
            result=self.mod.execute_replays([bad,good],self.cache,self.public,AuditLease(),time.time()+30)
        self.assertEqual(result['succeeded'],1);self.assertEqual(result['failed'],1)
        self.assertEqual(result['records'][1]['run_id'],good)

    def test_exhausted_budget_reports_all_remaining_routes_without_forward(self):
        run=self.baseline()
        with patch.object(route_predictions,'replay_route',side_effect=AssertionError('expired work ran')):
            result=self.mod.execute_replays([run],self.cache,self.public,AuditLease(),time.time()-1)
        self.assertEqual(result['status'],'DEFERRED_RESOURCE')
        self.assertEqual(result['deferred_run_ids'],[run])
        self.assertEqual(result['succeeded'],0)


if __name__=='__main__':unittest.main()
