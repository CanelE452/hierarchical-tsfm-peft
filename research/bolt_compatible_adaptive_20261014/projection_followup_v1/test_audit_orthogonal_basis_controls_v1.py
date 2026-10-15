"""Independent audit checks against small generated CPU controls and metadata."""
from pathlib import Path
import ast
import copy
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

import audit_orthogonal_basis_controls_v1 as audit
import run_orthogonal_basis_controls_v1 as producer
from orthogonal_latent_basis_v1 import fixed_orthogonal_controls
from test_run_orthogonal_basis_controls_v1 import fixture,toy_F

audit.initialize_numpy()


def cell(control="IDENTITY"):
    model,raw,cached,frozen,Q=fixture(control)
    outputs,calls=producer.infer_control(model,toy_F,raw["history"],cached,lambda x:None)
    row,independent=producer.evaluate_arrays(outputs,raw,cached,frozen,Q,control)
    row.update(canonical_batch_size=32,actual_F_calls=calls)
    return row,{**outputs,"origins":raw["origins"].copy()},independent,raw,cached,frozen,Q,control


def Q_fixture(control):
    model,_,_,_,_=fixture(control);item=fixed_orthogonal_controls()[control]
    arrays={"Q_FP64":item["Q"],"Q_FP32":model.basis().detach().numpy().copy()}
    row={"Q_metadata":item["metadata"],"Q_before":model.basis_diagnostics(),"Q_after":model.basis_diagnostics(),
        "Q_learned_or_selected":False,"frozen_status":model.frozen_status()}
    return arrays,row,control


def registry_fixture():
    result={"all_15_locks_before_DEV":True,"resolved_control_locks":15,
        "states":{state:{"controls":{}} for state in audit.STATES}}
    registry={"status":"ALL_FIFTEEN_RESOLVED_BEFORE_DEV","DEV_opened_before_seal":False,
        "Q_selection_performed":False,"sealed_unix":20.,"conditions":[]};locks={}
    for state in audit.STATES:
        for control in audit.CONTROLS:
            name=state+'_'+control;binding={"path":name,"sha256":"a"*64};Q={"path":name+'_Q',"sha256":"b"*64}
            result['states'][state]['controls'][control]={"control_lock":binding,"Q_artifact":Q}
            registry['conditions'].append({"state":state,"control":control,"status":"COMPLETE","control_lock":binding})
            locks[name]={"state":state,"control":control,"status":"COMPLETE","DEV_opened_before_lock":False,
                "Q_selection_performed":False,"FIT_TUNE_statuses":{"fit":"COMPLETE","tune":"COMPLETE"},"sealed_unix":10.,"Q_artifact":Q}
    return result,registry,locks


class FixedQIndependentAuditTests(unittest.TestCase):
    def test_all_five_control_arithmetic_masks_scores_commutators_and_null_scope(self):
        for control in audit.CONTROLS:
            with self.subTest(control=control):
                result=audit.check_cell(*cell(control));self.assertEqual(result['score_fields_checked'],96)
                self.assertEqual(result['null_control'],control in ('IDENTITY','PERMUTATION'))
                self.assertFalse(result['F_inference_independently_replayed']);self.assertEqual(len(result['C_commutator']['by_horizon_RMS']),48)
                json.dumps(result,allow_nan=False)

    def test_all_canonical_controls_and_Q_before_after_have_independent_certificates(self):
        for control in audit.CONTROLS:
            with self.subTest(control=control):
                result=audit.check_Q(*Q_fixture(control));self.assertLess(result['orthogonality_FP32_as_FP64_max_abs'],1e-5)
                if control.startswith('DENSE'):self.assertGreater(np.linalg.det(result['Q_FP64']),0)
                json.dumps(result,allow_nan=False)

    def test_mutated_Q_dtype_value_SO_provenance_or_trainability_rejected(self):
        for kind in ('Q32','Q64','metadata','trainable','after'):
            values=copy.deepcopy(Q_fixture('DENSE_SO_s92711'))
            if kind=='Q32':values[0]['Q_FP32'][0,0]+=.01
            elif kind=='Q64':values[0]['Q_FP64']=values[0]['Q_FP64'].astype(np.float32)
            elif kind=='metadata':values[1]['Q_metadata']['seed']=1
            elif kind=='trainable':values[1]['frozen_status']['theta_trainable']=True
            else:values[1]['Q_after']['determinant']+=.1
            with self.subTest(kind=kind),self.assertRaises(ValueError):audit.check_Q(*values)

    def test_changed_actual_inverse_D_output_or_independent_artifact_rejected(self):
        for kind in ('rotated','inverse','C','transport','independent'):
            values=copy.deepcopy(cell('DENSE_SO_s92712'))
            if kind=='rotated':values[1]['rotated_past'][0,0,0]+=.1
            elif kind=='inverse':values[1]['unrotated_future'][0,0,0]+=.1
            elif kind=='C':values[1]['prediction'][0,0,0]+=.1
            elif kind=='transport':values[1]['transport_prediction'][0,0,0]+=.1
            else:values[2]['independent_NumPy_future'][0,0,0]+=.1
            with self.subTest(kind=kind),self.assertRaises(ValueError):audit.check_cell(*values)

    def test_score_mask_count_origins_and_registered_call_counts_cannot_be_relabelled(self):
        for kind in ('score','mask','count','origins','calls','tolerance','summary'):
            values=copy.deepcopy(cell())
            if kind=='score':values[0]['scores']['all_observed']['normalized']['mse']+=.1
            elif kind=='mask':values[3]['observed'][0,0,0]=True
            elif kind=='count':values[0]['complete_target_positions']+=1
            elif kind=='origins':values[1]['origins'][0]+=1
            elif kind=='calls':values[0]['actual_F_calls']+=1
            elif kind=='tolerance':values[0]['checks']['old_past_vs_cached']['atol']=.1
            else:values[0]['C_commutator']['by_horizon_RMS'][0]+=.1
            with self.subTest(kind=kind),self.assertRaises(ValueError):audit.check_cell(*values)

    def test_unobserved_target_nonfinite_is_ignored_but_observed_nonfinite_rejected(self):
        values=cell('DENSE_SO_s92711');expected=audit.check_cell(*values)
        values[3]['target'][~values[3]['observed']]=float('inf')
        self.assertEqual(expected,audit.check_cell(*values))
        values[3]['target'][values[3]['observed']]=float('nan')
        with self.assertRaises(ValueError):audit.check_cell(*values)

    def test_fifteen_complete_unique_locks_resolve_before_registry_without_Q_selection(self):
        result,registry,locks=registry_fixture()
        verified=audit.check_registry(result,registry,lambda item:locks[item['path']])
        self.assertEqual(len(verified),15);json.dumps(list(verified.values()),allow_nan=False)
        for kind in ('duplicate','early_DEV','selection','after_registry','splitfailed'):
            r,g,l=copy.deepcopy((result,registry,locks))
            if kind=='duplicate':g['conditions'][-1]=copy.deepcopy(g['conditions'][0])
            elif kind=='early_DEV':r['all_15_locks_before_DEV']=False
            elif kind=='selection':g['Q_selection_performed']=True
            elif kind=='after_registry':next(iter(l.values()))['sealed_unix']=21.
            else:next(iter(l.values()))['FIT_TUNE_statuses']['fit']='FAILED'
            with self.subTest(kind=kind),self.assertRaises(ValueError):audit.check_registry(r,g,lambda item:l[item['path']])

    def test_no_observed_complete_horizon_is_not_reported_as_zero_error(self):
        values=cell();raw=values[3];raw['observed'][:,:,0]=False
        raw['target'][:,:,0]=np.nan
        outputs={k:v for k,v in values[1].items() if k!='origins'}
        row,independent=producer.evaluate_arrays(outputs,raw,values[4],values[5],values[6],values[7])
        row.update(canonical_batch_size=32,actual_F_calls=1)
        values=(row,values[1],independent,*values[3:])
        result=audit.check_cell(*values)
        self.assertEqual(result['complete_target_positions'],0);self.assertEqual(result['scores']['complete_C'],{'status':'NO_COMPLETE_C'})
        self.assertEqual(result['score_fields_checked'],48)

    def test_closed_actual_binding_precedes_input_IO_and_auditor_has_no_producer_import(self):
        with patch.object(audit,'RESULT_SHA',None),patch.object(audit.base,'file_record',side_effect=AssertionError('IO forbidden')):
            with self.assertRaisesRegex(ValueError,'NOT_REGISTERED'):audit.audit_result('missing',None)
        source=Path(audit.__file__).read_text(encoding='utf-8');tree=ast.parse(source)
        imports=[node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
        self.assertFalse(any(name in ('run_orthogonal_basis_controls_v1','orthogonal_latent_basis_v1') for name in imports))
        command='import sys;sys.path.insert(0,'+repr(str(audit.HERE))+');import audit_orthogonal_basis_controls_v1;assert "torch" not in sys.modules and "numpy" not in sys.modules;print("PURE_IMPORT_PASS")'
        completed=subprocess.run([sys.executable,'-c',command],capture_output=True,text=True,check=True)
        self.assertIn('PURE_IMPORT_PASS',completed.stdout)


if __name__=='__main__':unittest.main()
