"""Independent cost audit on tiny synthetic saved artifacts only; no actual assets/GPU."""
from pathlib import Path
from unittest.mock import patch
import copy
import json
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
import audit_horizon_decoder_cost_v1 as audit
import run_horizon_decoder_cost_v1 as producer
import whole_horizon_decoder_cost_v1 as whole
import run_horizon_decoder_diagnostic_v1 as scientific
import causal_lag_decoder_v1 as lag
from test_horizon_decoder_cost_v1 import fixture

audit.initialize_numpy()


def source_texts():return [(audit.HERE/name).read_text(encoding='utf-8') for name in ('whole_horizon_decoder_cost_v1.py','run_horizon_decoder_cost_v1.py','whole_f_input_gradient_cost_v1.py')]


def physical_fixture():
    history,model,state,coefficient,tensor,bolt,calls=fixture();n=len(history);future=(history[:,-1:]+.01).repeat(48,axis=1);origins=np.arange(n,dtype=np.int64)+512;observed=np.ones_like(history,dtype=bool)
    raw={'history':history,'raw_last':history[:,-1].copy(),'raw_forecast':future,'history_observed':observed,'origins':origins,'mean':np.zeros(3,np.float32),'std':np.ones(3,np.float32)};physical={**raw,'history_physical':history.copy()};z=history@state['E'];f=future@state['E'];cache={'latent_history':z,'latent_future':f,'lag_features':lag.lagged_delta_features(z,f,audit.LAGS),'prediction':lag.predict(z,f,raw['raw_last'],state['D'],audit.LAGS),'FLAT_prediction':lag.predict(z,np.broadcast_to(z[:,-1:],f.shape).copy(),raw['raw_last'],state['D'],audit.LAGS)}
    return raw,physical,model,state,coefficient,tensor,bolt,cache


class HorizonCostAuditTests(unittest.TestCase):
    def test_source_control_flow_timer_boundary_and_missing_ZERO_or_F_rules_rejected(self):
        helper,runner,parent=source_texts();result=audit.source_contract(helper,runner,parent);self.assertTrue(result['five_of_eighteen_positions_only']);self.assertFalse(result['runtime_F_event_or_timer_replayed'])
        for before,after in (("selected_kind!=\"ZERO_DELTA\"","selected_kind==\"ZERO_DELTA\""),("baseline=future.copy()","baseline=future+history[:,-1,None]")):
            self.assertIn(before,runner)
            with self.subTest(change=before),self.assertRaises(ValueError):audit.source_contract(helper,runner.replace(before,after),parent)
        changed=helper.replace('synchronize(); perf_end_ns, unix_end_ns = time.perf_counter_ns(), time.time_ns()','perf_end_ns, unix_end_ns = time.perf_counter_ns(), time.time_ns()')
        with self.assertRaises(ValueError):audit.source_contract(changed,runner,parent)

    def test_routes_schedule_independently_match_registered_18_five_positions(self):
        for data in audit.DATASETS:
            self.assertEqual(audit.routes(data),whole.routes_for(data));self.assertEqual(audit.schedule(data),whole.balanced_schedule(data))
            for name in audit.routes(data):self.assertEqual(audit.matched_baseline(name),whole.matched_baseline(name));self.assertEqual(len({r.index(name) for r in audit.schedule(data)}),5)

    def test_independent_first8_PCA_F_FLAT_and_two_corrections_saved_arithmetic(self):
        raw,physical,model,state,coefficient,tensor,bolt,cache=physical_fixture()
        for mode in ('F','FLAT'):
            stream=scientific.stream_payload(raw['history'],raw['raw_last'],raw['raw_forecast'],'PCA_K2',mode,cache,state)
            for arm,kind in (('STATIC_DELTA','ALPHA'),('RAMP_DELTA','ALPHA'),('RAMP_DELTA','ZERO_DELTA'),('BASELINE',None)):
                with self.subTest(mode=mode,arm=arm,kind=kind):
                    reference=producer.forecast_components(raw['history'],'PCA_K2',mode,arm,model,bolt,tensor,coefficient,kind);entry,arrays=producer.physical_precheck(physical,raw,reference,'PCA_K2',mode,arm,model,bolt,tensor,coefficient,kind,stream)
                    result=audit.precheck(entry,arrays,physical,raw,reference,stream,state,'PCA_K2',mode,arm,coefficient,kind);self.assertTrue(result['first8_B8_component_arithmetic_checked']);self.assertFalse(result['all_timed_B1_B8_latent_components_replayed'])
        entry,arrays=producer.physical_precheck(physical,raw,reference,'PCA_K2','FLAT','BASELINE',model,bolt,tensor,coefficient,None,stream);arrays['latent_future'][0,0,0]+=.1
        with self.assertRaisesRegex(ValueError,'FLAT_is_known|independent_original_PCA_D'):audit.precheck(entry,arrays,physical,raw,reference,stream,state,'PCA_K2','FLAT','BASELINE',coefficient,None)

    def test_wrong_selected_coefficient_or_missing_lag_origin_cannot_pass_first8(self):
        raw,physical,model,state,coefficient,tensor,bolt,cache=physical_fixture();stream=scientific.stream_payload(raw['history'],raw['raw_last'],raw['raw_forecast'],'PCA_K2','F',cache,state);reference=producer.forecast_components(raw['history'],'PCA_K2','F','STATIC_DELTA',model,bolt,tensor,coefficient,'ALPHA');entry,arrays=producer.physical_precheck(physical,raw,reference,'PCA_K2','F','STATIC_DELTA',model,bolt,tensor,coefficient,'ALPHA',stream)
        altered=copy.deepcopy(coefficient);altered['coefficient_FP32'][0,0]+=.1
        with self.assertRaisesRegex(ValueError,'selected_CPU_FP32'):audit.precheck(entry,arrays,physical,raw,reference,stream,state,'PCA_K2','F','STATIC_DELTA',altered,'ALPHA')
        altered_arrays=copy.deepcopy(arrays);altered_arrays['features'][0,0,2]+=.01
        with self.assertRaisesRegex(ValueError,'lag_specific'):audit.precheck(entry,altered_arrays,physical,raw,reference,stream,state,'PCA_K2','F','STATIC_DELTA',coefficient,'ALPHA')

    def test_RAW_F_identity_cancellation_is_not_an_invalid_baseline_gate(self):
        history=np.full((8,512,2),np.float32(1e8));future=np.ones((8,48,2),np.float32);observed=np.ones_like(history,dtype=bool);raw={'history':history,'raw_last':history[:,-1].copy(),'raw_forecast':future,'history_observed':observed,'origins':np.arange(8,dtype=np.int64)+512,'mean':np.zeros(2,np.float32),'std':np.ones(2,np.float32)};physical={**raw,'history_physical':history.copy()};stream=scientific.stream_payload(history,raw['raw_last'],future,'RAW_C','F')
        def bolt(x):return torch.ones((len(x),48,2),dtype=torch.float32)
        entry,arrays=producer.physical_precheck(physical,raw,future,'RAW_C','F','BASELINE',None,bolt,torch.as_tensor,None,None,stream);result=audit.precheck(entry,arrays,physical,raw,future,stream,None,'RAW_C','F','BASELINE',None,None)
        self.assertTrue(result['ZERO_or_baseline_exact_copy']);self.assertEqual(arrays['baseline'].tobytes(),future.tobytes());self.assertFalse(stream['checks']['same_old_decoder']['pass'])

    def test_full_saved_synthetic_dataset_ns_outputs_ratios_memory_and_tampering(self):
        physical=np.arange(8*512*2,dtype=np.float32).reshape(8,512,2)*1e-5;observed=np.ones_like(physical,dtype=bool);mean=np.zeros(2,np.float32);std=np.ones(2,np.float32);origins=np.arange(8,dtype=np.int64)+512;raw={'history':physical.copy(),'history_observed':observed,'raw_last':physical[:,-1].copy(),'origins':origins,'mean':mean,'std':std}
        routes=whole.routes_for('robin');callbacks={name:(lambda x,m:np.broadcast_to(x[:,-1:],(len(x),48,2)).copy()) for name in routes};predictions={name:callbacks[name](physical,observed) for name in routes};metadata={name:{'synthetic_only':True} for name in routes};memory={'RSS_bytes':128,'allocated_VRAM_bytes':16,'reserved_VRAM_bytes':32,'max_allocated_VRAM_bytes':16,'max_reserved_VRAM_bytes':32,'scope':audit.MEMORY_SCOPE}
        with tempfile.TemporaryDirectory() as directory:
            cost=whole.measure_cost('robin',physical,observed,mean,std,origins,callbacks,predictions,metadata,Path(directory)/'cost',lambda:None,lambda:None,lambda:memory.copy());result=audit.check_dataset(cost,{'history_physical':physical,'history_observed':observed},raw,predictions,lambda:None,0.)
            self.assertEqual((result['complete_cells'],result['saved_samples']),(36,180));self.assertFalse(result['F_or_timer_replayed']);self.assertEqual(len(result['rows']),36)
            for mutate,label in ((lambda d:d['rows'][0]['seconds'].__setitem__(0,1.),'NS_DURATIONS'),(lambda d:d['rows'][0].update(paired_RAW_median_ratio=22.),'TWO_RATIOS'),(lambda d:d['rows'][0]['samples'][0]['memory'].update(scope='exclusive'),'SAMPLE_JSON'),(lambda d:d['rows'].pop(),'36_UNIQUE')):
                broken=copy.deepcopy(cost);mutate(broken)
                with self.subTest(label=label),self.assertRaisesRegex(ValueError,label):audit.check_dataset(broken,{'history_physical':physical,'history_observed':observed},raw,predictions,lambda:None,0.)
            broken_raw=copy.deepcopy(raw);broken_raw['history'][0,0,0]+=.1
            with self.assertRaisesRegex(ValueError,'causal_fill'):audit.check_dataset(cost,{'history_physical':physical,'history_observed':observed},broken_raw,predictions,lambda:None,0.)

    def test_metadata_allows_only_exact_two_cohorts_and_rejects_third_before_json(self):
        self.assertEqual(len(audit.ALLOWED_COST_SHA),2)
        with patch.object(audit,'ALLOWED_COST_SHA',None),patch.object(audit.linear,'file_record',side_effect=AssertionError('forbidden IO')):
            with self.assertRaisesRegex(ValueError,'NOT_REGISTERED'):audit.metadata('missing',lambda:None)
        with patch.object(audit.linear,'file_record',return_value={'path':'missing','sha256':'0'*64}),patch.object(audit.prior,'verified_json',side_effect=AssertionError('third cohort must stop before JSON')):
            with self.assertRaisesRegex(ValueError,'PRIMARY_OR_REPEAT'):audit.metadata('missing',lambda:None)
        code='import sys;sys.path.insert(0,'+repr(str(audit.HERE))+');import audit_horizon_decoder_cost_v1;assert "numpy" not in sys.modules and "torch" not in sys.modules;print("STDLIB_AUDIT_IMPORT_PASS")'
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True);self.assertIn('STDLIB_AUDIT_IMPORT_PASS',result.stdout)


if __name__=='__main__':unittest.main()
