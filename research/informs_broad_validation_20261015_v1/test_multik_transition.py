import collections
import tempfile
from pathlib import Path
import unittest
import os
import json
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import measure_multik_transition as m
import deployment
import fit_base
from data_contract import array_sha256, split_ranges, sha256_file
from training import fingerprint
from test_costs import ToyFoundation


class PredictiveToyFoundation(ToyFoundation):
    def forward(self,x):
        current=super().forward(x)
        drift=torch.arange(1,self.horizon+1,dtype=x.dtype,device=x.device)[None,:,None]*.1
        return current*1.2 + x.mean(1,keepdim=True)*.05 + drift


class MutableDeadline:
    def __init__(self):
        self.expired=False
        self.deadline=time.time()+1000
    def expire(self):
        self.expired=True
        self.deadline=time.time()-1
    def check(self):
        if self.expired:raise m.BudgetExceeded('toy operation returned after owner deadline')
        return self


class MultiKTests(unittest.TestCase):
    def test_each_directed_edge_has_ten_warm_and_twenty_timed_exposures(self):
        schedule = m.balanced_edges((1, 2, 4), 20)
        counts = collections.Counter((a, b) for _, _, a, b in schedule)
        self.assertEqual(counts, {(1, 2): 20, (1, 4): 20, (2, 1): 20, (2, 4): 20, (4, 1): 20, (4, 2): 20})
        first_six_rounds = [x for x in schedule if x[0] < 6]
        for edge in counts:
            self.assertEqual(sorted(pos for _, pos, a, b in first_six_rounds if (a, b) == edge), list(range(6)))
        self.assertEqual(len(m.balanced_edges((1, 2, 4), 10)), 60)
        with self.assertRaises(ValueError):
            m.balanced_edges((1, 1, 4), 20)

    def fixture(self, directory, variant='shared'):
        t=np.arange(800,dtype=np.float32)
        physical=np.stack([np.sin(t/9)+.01*t,np.cos(t/13)+.002*t,.005*(t%17)**2,np.sin(t/5+.2)*.5+.001*t],1).astype(np.float32)
        task = {'name': 'toy', 'L': 40, 'H': 4, 'columns': ['a','b','c','d'],
                'physical': physical,
                'observed': np.ones((800,4), bool), 'split_ranges': split_ranges(800), 'data_sha256': 'toy'}
        pool = deployment.FoundationPool(None, 'cpu', factory=lambda p,h,r,d: PredictiveToyFoundation(h))
        foundation = pool.get(task['H'])
        specs = {}
        for k in ((4,) if variant != 'independent_hot' else (1,2,4)):
            spec = {'run_id': 'model'+str(k), 'family': 'B0', 'seed': 7, 'device':'cpu', 'latent_k':k,
                    'k_grid': [1,2,4] if variant=='shared' else [k]}
            model, norm = fit_base.create_model(spec,task,task['split_ranges']['A_stop'][1],foundation)
            with torch.no_grad():
                model.E.copy_(torch.eye(4)[:,:k]);model.center.zero_();model.D.zero_()
                model.D[0].copy_(torch.eye(k,4))
            state={key:v.detach().cpu().clone() for key,v in model.state_dict().items()}
            bundle={'spec':spec,'normalization':norm,'model':state,
                    'contract':{'data_sha':'toy','model_sha':fit_base._model_sha(foundation),'scaler_sha':fit_base.scaler_fingerprint(norm['mean'],norm['std'])},
                    'model_state_sha':fingerprint({key:array_sha256(v.numpy()) for key,v in state.items()})}
            path=Path(directory)/('model'+str(k)+'.pt');torch.save(bundle,path)
            specs[k]={'run_id':'alias'+str(k),'family':'B0','latent_k':k,'artifact_cache_relative_path':path.name,'artifact_sha256':sha256_file(path)}
        if variant != 'independent_hot':
            specs={k:dict(specs[4],run_id='alias'+str(k),latent_k=k,**({'prefix_variant':'posthoc'} if variant=='posthoc' else {})) for k in (1,2,4)}
        created=[]
        def factory(spec):
            route=deployment.Route(spec,task,directory,device='cpu',foundation_pool=pool)
            created.append(route)
            return route
        x=task['physical'][500:540][None].copy();mask=np.ones_like(x,bool)
        return specs,factory,created,pool,x,mask

    def test_single_route_keeps_one_model_and_original_parameters_across_k(self):
        with tempfile.TemporaryDirectory() as directory:
            specs,factory,created,pool,x,mask=self.fixture(directory)
            runtime=m.MultiKRuntime('shared',specs,factory)
            runtime.initialize()
            model=runtime.active.model
            before={k:v.detach().clone() for k,v in model.state_dict().items()}
            parity=runtime.verify_canonical(x,mask)
            self.assertTrue(parity['all_passed'])
            result=m.measure(runtime,x,mask,warmups=1,timed_trials=2)
            self.assertEqual(result['status'],'COMPLETE_PROTOCOL')
            self.assertEqual(len(result['rows']),18)
            self.assertEqual(runtime.resident_route_count,1)
            self.assertIs(runtime.active.model,model)
            self.assertTrue(all(torch.equal(v,model.state_dict()[k]) for k,v in before.items()))
            self.assertTrue(all(row['F_streams']==row['to_K'] for row in result['rows']))
            runtime.close();pool.close()

    def test_posthoc_expands_only_runtime_prefix_registry_after_maxonly_check(self):
        with tempfile.TemporaryDirectory() as directory:
            specs,factory,created,pool,x,mask=self.fixture(directory,'posthoc')
            runtime=m.MultiKRuntime('posthoc',specs,factory);runtime.initialize()
            self.assertEqual(runtime.active.bundle['spec']['k_grid'],[4])
            self.assertEqual(tuple(runtime.active.model.k_grid),(1,2,4))
            self.assertTrue(runtime.verify_canonical(x,mask)['all_passed'])
            runtime.close();pool.close()

    def test_single_strategy_rejects_multiple_artifacts_and_unsupported_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            specs,factory,created,pool,x,mask=self.fixture(directory,'independent_hot')
            runtime=m.MultiKRuntime('shared',specs,factory)
            with self.assertRaises(ValueError):runtime.initialize()
            self.assertFalse(created)
            pool.close()

    def test_independent_hot_keeps_three_routes_while_cold_reloads_one(self):
        with tempfile.TemporaryDirectory() as directory:
            specs,factory,created,pool,x,mask=self.fixture(directory,'independent_hot')
            hot=m.MultiKRuntime('independent_hot',specs,factory);hot.initialize()
            self.assertEqual(hot.resident_route_count,3)
            self.assertTrue(hot.verify_canonical(x,mask)['all_passed'])
            hot.close()
            cold=m.MultiKRuntime('independent_cold',specs,factory);cold.initialize()
            previous=cold.active
            cold.set_k(2)
            self.assertTrue(previous._closed)
            self.assertIsNot(cold.active,previous)
            self.assertEqual(cold.resident_route_count,1)
            self.assertGreaterEqual(cold.set_k(4)['artifact_load_seconds'],0)
            cold.close();pool.close()

    def test_no_measurement_before_parity_and_partial_counts_are_actual(self):
        with tempfile.TemporaryDirectory() as directory:
            specs,factory,created,pool,x,mask=self.fixture(directory)
            runtime=m.MultiKRuntime('shared',specs,factory);runtime.initialize()
            with self.assertRaises(ValueError):m.measure(runtime,x,mask,warmups=1,timed_trials=2)
            runtime.verify_canonical(x,mask)
            lease=MutableDeadline();calls=[];predict=runtime.predict
            def stop_after_third_actual_prediction(*args):
                value=predict(*args);calls.append(1)
                if len(calls)==3:lease.expire()
                return value
            runtime.predict=stop_after_third_actual_prediction
            result=m.measure(runtime,x,mask,warmups=1,timed_trials=2,lease=lease)
            self.assertEqual(result['status'],'PARTIAL_RESOURCE_LIMIT')
            self.assertEqual(len(result['rows']),3)
            self.assertEqual(sum(x['remaining'] for x in result['transition_ledger']),15)
            runtime.close();pool.close()

    def test_parity_rejects_fake_small_k_streams(self):
        with tempfile.TemporaryDirectory() as directory:
            specs,factory,created,pool,x,mask=self.fixture(directory)
            runtime=m.MultiKRuntime('shared',specs,factory);runtime.initialize()
            original=runtime.active.predict_physical
            def wrong_streams(*args,**kwargs):
                result=original(*args,**kwargs);runtime.active.last_receipt['F_logical_streams']=999;return result
            runtime.active.predict_physical=wrong_streams
            with self.assertRaises(AssertionError):runtime.verify_canonical(x,mask)
            self.assertFalse(runtime.parity_passed)
            runtime.close();pool.close()

    def test_canonical_constructor_overrun_starts_no_prediction(self):
        with tempfile.TemporaryDirectory() as directory:
            specs,factory,created,pool,x,mask=self.fixture(directory)
            runtime=m.MultiKRuntime('shared',specs,factory);runtime.initialize()
            lease=MutableDeadline();predictions=[];canonical_routes=[]
            def constructor(spec):
                route=factory(spec);canonical_routes.append(route)
                original=route.predict_physical
                def predict(*args):
                    predictions.append(1)
                    return original(*args)
                route.predict_physical=predict
                lease.expire()
                return route
            runtime.factory=constructor
            try:
                with self.assertRaises(m.BudgetExceeded):runtime.verify_canonical(x,mask,lease)
                self.assertEqual(predictions,[])
                self.assertEqual(len(canonical_routes),1)
                self.assertTrue(canonical_routes[0]._closed)
                self.assertFalse(runtime.parity_passed)
                self.assertEqual(runtime.canonical_predictions,{})
            finally:
                runtime.close();pool.close()

    def test_unique_disk_paths_and_unique_content_are_distinct_storage_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            a=Path(directory)/'a.bin';b=Path(directory)/'b.bin';a.write_bytes(b'abc');b.write_bytes(b'abc')
            sha=sha256_file(a)
            specs={1:{'artifact_cache_relative_path':'a.bin','artifact_sha256':sha},2:{'artifact_cache_relative_path':'a.bin','artifact_sha256':sha},4:{'artifact_cache_relative_path':'b.bin','artifact_sha256':sha}}
            receipt=m.storage_inventory(specs,directory)
            self.assertEqual(receipt['unique_artifact_paths'],2)
            self.assertEqual(receipt['actual_artifact_bytes'],6)
            self.assertEqual(receipt['unique_content_sha256_count'],1)
            b.write_bytes(b'changed')
            with self.assertRaises(ValueError):m.storage_inventory(specs,directory)

    def test_pre_measurement_budget_boundary_keeps_all_180_transitions_pending(self):
        ledger=m.transition_ledger((1,2,4),10,20)
        self.assertEqual(len(ledger),12)
        self.assertEqual(sum(x['remaining'] for x in ledger),180)
        self.assertTrue(all(x['actual_complete']==0 for x in ledger))

    def test_late_prediction_failure_does_not_count_a_failed_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            specs,factory,created,pool,x,mask=self.fixture(directory)
            runtime=m.MultiKRuntime('shared',specs,factory);runtime.initialize();runtime.verify_canonical(x,mask)
            original=runtime.active.predict_physical
            def wrong(*args,**kwargs):return original(*args,**kwargs)+1
            runtime.active.predict_physical=wrong
            result=m.measure(runtime,x,mask,warmups=1,timed_trials=2)
            self.assertEqual(result['status'],'FAILED_STRATEGY_AFTER_PARTIAL_MEASUREMENT')
            self.assertEqual(result['rows'],[])
            self.assertEqual(sum(x['remaining'] for x in result['transition_ledger']),18)
            runtime.close();pool.close()

    def test_nested_single_basis_uses_actual_k_streams_without_model_copies(self):
        with tempfile.TemporaryDirectory() as directory:
            specs,factory,created,pool,x,mask=self.fixture(directory)
            probe=factory(specs[4]);task=probe.task;norm=probe.bundle['normalization'];probe.close()
            path=Path(directory)/'nested.npz'
            np.savez_compressed(path,basis=norm['basis'],center=norm['pca_center'],mean=norm['mean'],std=norm['std'],
                                data_sha=np.asarray('toy'),fit_end=np.asarray(task['split_ranges']['A_stop'][1]))
            specs={k:{'run_id':'nested'+str(k),'family':'NESTED_PCA','latent_k':k,
                      'artifact_cache_relative_path':path.name,'artifact_sha256':sha256_file(path),
                      'foundation_sha':fit_base._model_sha(pool.foundation)} for k in (1,2,4)}
            runtime=m.MultiKRuntime('nested',specs,factory);runtime.initialize()
            model=runtime.active.model
            self.assertTrue(runtime.verify_canonical(x,mask)['all_passed'])
            result=m.measure(runtime,x,mask,warmups=1,timed_trials=1)
            self.assertEqual(result['status'],'COMPLETE_PROTOCOL')
            self.assertIs(runtime.active.model,model)
            self.assertTrue(all(row['F_streams']==row['to_K'] for row in result['rows']))
            runtime.close();pool.close()

    def test_tensor_mutation_is_detected_separately_from_allowed_k_attribute_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            specs,factory,created,pool,x,mask=self.fixture(directory)
            runtime=m.MultiKRuntime('shared',specs,factory);runtime.initialize();runtime.set_k(1)
            self.assertTrue(runtime.assert_model_states_unchanged()['all_equal'])
            with torch.no_grad():runtime.active.model.E[0,0]+=1
            with self.assertRaises(AssertionError):runtime.assert_model_states_unchanged()
            runtime.close();pool.close()

    def test_unowned_gpu_worker_fails_before_any_task_or_foundation_load(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{},clear=True),patch('data_contract.load_task',side_effect=AssertionError('asset load before owner')):
            with self.assertRaises(KeyError):
                m.worker(SimpleNamespace(cache_root=directory,device='cuda'))
            self.assertFalse(torch.cuda.is_initialized())

    def test_deadline_after_preparation_or_switch_starts_no_prediction(self):
        for expire_after in (1,2):
            with self.subTest(expire_after=expire_after),tempfile.TemporaryDirectory() as directory:
                specs,factory,created,pool,x,mask=self.fixture(directory)
                runtime=m.MultiKRuntime('shared',specs,factory);runtime.initialize();runtime.verify_canonical(x,mask)
                lease=MutableDeadline();calls=[];predictions=[];original=runtime.set_k;original_predict=runtime.predict
                def switching(k):
                    value=original(k);calls.append(k)
                    if len(calls)==expire_after:lease.expire()
                    return value
                def predict(*args):predictions.append(1);return original_predict(*args)
                runtime.set_k=switching;runtime.predict=predict
                result=m.measure(runtime,x,mask,1,1,lease)
                self.assertEqual(predictions,[])
                self.assertEqual(result['rows'],[])
                self.assertEqual(result['status'],'PARTIAL_RESOURCE_LIMIT')
                runtime.close();pool.close()

    def test_last_prediction_overrun_is_numeric_complete_but_not_protocol_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            specs,factory,created,pool,x,mask=self.fixture(directory)
            runtime=m.MultiKRuntime('shared',specs,factory);runtime.initialize();runtime.verify_canonical(x,mask)
            lease=MutableDeadline();calls=[];original=runtime.predict
            def predict(*args):
                value=original(*args);calls.append(1)
                if len(calls)==180:lease.expire()
                return value
            runtime.predict=predict
            result=m.measure(runtime,x,mask,10,20,lease)
            self.assertEqual(len(result['rows']),180)
            self.assertEqual(result['numeric_execution_status'],'COMPLETE_REQUESTED_TRIALS')
            self.assertEqual(result['resource_status'],'DEADLINE_OVERRUN')
            self.assertFalse(result['exact_minimum_protocol_complete'])
            self.assertEqual(sum(x['remaining'] for x in result['transition_ledger']),0)
            self.assertEqual(sum(x['remaining_within_resource'] for x in result['transition_ledger']),1)
            runtime.close();pool.close()

    def worker_fixture(self,directory,strategy='shared'):
        specs,factory,created,pool,x,mask=self.fixture(directory)
        probe=factory(specs[4]);task=dict(probe.task,name='beijing');probe.close()
        plan=m.default_plan('beijing',4)
        for name in plan['strategies']:plan['strategies'][name]={str(k):v['run_id'] for k,v in specs.items()}
        path=Path(directory)/'plan.json';path.write_text(json.dumps(plan))
        args=SimpleNamespace(cache_root=directory,public_root=directory,device='cpu',model='toy',plan_json=path,
                             batch=1,strategy=strategy,warmups=1,timed_trials=1,output=str(Path(directory)/'output'))
        build=lambda run_id,*a: (next(dict(v) for v in specs.values() if v['run_id']==run_id),'small')
        return args,pool,task,build

    def test_post_measure_validation_errors_preserve_durable_rows_and_ledgers(self):
        for name in ('memory','state','summary'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as directory:
                args,pool,task,build=self.worker_fixture(directory)
                target={'memory':'measure_multik_transition._memory','state':'measure_multik_transition.MultiKRuntime.assert_model_states_unchanged','summary':'measure_multik_transition.statistics.median'}[name]
                with patch('data_contract.load_task',return_value=task),patch('deployment.FoundationPool',return_value=pool),patch('route_catalog.build_route',side_effect=build),patch('measure_costs.accuracy_binding',return_value={'status':'CPU_TOY'}),patch(target,side_effect=RuntimeError('injected post measure '+name)):
                    m.worker(args)
                output=Path(args.output)/'shared';value=json.loads((output/'RESULT.json').read_text())
                self.assertIn('rows',value)
                self.assertEqual(len(value['rows']),12)
                self.assertEqual(len(list((output/'actual_trials').glob('trial_*.json'))),12)
                with (output/'TRIALS.csv').open(encoding='utf-8',newline='') as handle:
                    self.assertEqual(len(list(__import__('csv').DictReader(handle))),12)
                self.assertEqual(sum(x['remaining'] for x in value['transition_ledger']),0)
                self.assertEqual(value['post_measure_validation_status'],'FAILED')
                self.assertFalse(value['exact_minimum_protocol_complete'])

    def test_close_deadline_overrun_blocks_next_scope_without_losing_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            args,pool,task,build=self.worker_fixture(directory,'all');lease=MutableDeadline();original=pool.close
            def close():original();lease.expire()
            pool.close=close
            with patch('data_contract.load_task',return_value=task),patch('deployment.FoundationPool',return_value=pool),patch('route_catalog.build_route',side_effect=build),patch('measure_costs.accuracy_binding',return_value={'status':'CPU_TOY'}),patch('measure_costs.worker_lease_check',return_value=lease):
                values=m.worker(args)
            value=json.loads((Path(args.output)/'shared/RESULT.json').read_text())
            self.assertEqual(len(value['rows']),12)
            self.assertEqual(value['resource_status'],'DEADLINE_OVERRUN')
            self.assertTrue(all(x['status']=='NOT_STARTED_RESOURCE_LIMIT' for x in values[1:]))
            self.assertFalse((Path(args.output)/'posthoc').exists())

    def test_nondegenerate_k_predictions_detect_wrong_lag_major_decoder_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            specs,factory,created,pool,x,mask=self.fixture(directory)
            runtime=m.MultiKRuntime('shared',specs,factory);runtime.initialize();runtime.verify_canonical(x,mask)
            self.assertFalse(np.allclose(runtime.canonical_predictions[1],runtime.canonical_predictions[2]))
            self.assertFalse(np.allclose(runtime.canonical_predictions[2],runtime.canonical_predictions[4]))
            model=runtime.active.model
            def wrong_decode(x,z,future,latent_k):
                k=latent_k;joined=torch.cat((z,future),1)
                features=torch.cat([joined[:,model.length-lag:model.length-lag+model.horizon]-z[:,-1-lag,None] for lag in (0,1,2,4,8,16,32)],-1)
                return x[:,-1:]+features @ model.D.reshape(7*model.kmax,model.channels)[:7*k]
            model.decode=wrong_decode
            with self.assertRaises(AssertionError):runtime.verify_canonical(x,mask)
            self.assertFalse(runtime.parity_passed)
            runtime.close();pool.close()

    def test_wrong_encoder_columns_fail_canonical_parity_even_with_correct_k_stream_count(self):
        with tempfile.TemporaryDirectory() as directory:
            specs,factory,created,pool,x,mask=self.fixture(directory)
            runtime=m.MultiKRuntime('shared',specs,factory);runtime.initialize()
            model=runtime.active.model
            def wrong_encode(x,latent_k):return (model.front(x)-model.center) @ model.E[:,-latent_k:]
            model.encode=wrong_encode
            with self.assertRaises(AssertionError):runtime.verify_canonical(x,mask)
            self.assertFalse(runtime.parity_passed)
            self.assertEqual(runtime.active.last_receipt['F_logical_streams'],1)
            runtime.close();pool.close()

    def test_durable_integrity_rejects_csv_value_tampering_not_just_row_count(self):
        with tempfile.TemporaryDirectory() as directory:
            args,pool,task,build=self.worker_fixture(directory)
            with patch('data_contract.load_task',return_value=task),patch('deployment.FoundationPool',return_value=pool),patch('route_catalog.build_route',side_effect=build),patch('measure_costs.accuracy_binding',return_value={'status':'CPU_TOY'}):
                m.worker(args)
            folder=Path(args.output)/'shared';value=json.loads((folder/'RESULT.json').read_text())
            import csv
            with (folder/'TRIALS.csv').open(encoding='utf-8',newline='') as handle:
                rows=list(csv.DictReader(handle))
            rows[0]['F_streams']='999'
            with (folder/'TRIALS.csv').open('w',encoding='utf-8',newline='') as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
            with self.assertRaises(AssertionError):m._durable_trial_check(folder,value)

    def test_disjoint_switch_prediction_guard_and_other_segments_reconstruct_whole(self):
        with tempfile.TemporaryDirectory() as directory:
            specs,factory,created,pool,x,mask=self.fixture(directory)
            runtime=m.MultiKRuntime('shared',specs,factory);runtime.initialize();runtime.verify_canonical(x,mask)
            value=m.measure(runtime,x,mask,1,1,MutableDeadline())
            for row in value['rows']:
                fields=('switch_execution_segment_seconds','physical_prediction_segment_seconds','controller_guard_seconds_inside_whole','other_clock_overhead_seconds')
                self.assertTrue(all(row[key]>=0 for key in fields))
                self.assertAlmostEqual(sum(row[key] for key in fields),row['transition_plus_physical_seconds'],places=12)
            runtime.close();pool.close()


if __name__=='__main__':unittest.main()
