"""Synthetic CPU controls and metadata gates; no actual arrays, model, F or GPU."""
from pathlib import Path
from types import SimpleNamespace
import copy
import inspect
import json
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_orthogonal_basis_controls_v1 as run
from orthogonal_latent_basis_v1 import OrthogonalLatentBasisAdapter,fixed_orthogonal_controls


def toy_F(z):return z[:,-1:].square().expand(-1,48,-1)+z[:,-2:-1].sin()


def fixture(control="IDENTITY"):
    rng=np.random.default_rng(775);frozen={"center":rng.normal(scale=.1,size=6).astype(np.float32),
        "E":rng.normal(scale=.2,size=(6,4)).astype(np.float32),"D":rng.normal(scale=.1,size=(28,6)).astype(np.float32)}
    history=rng.normal(size=(3,512,6)).astype(np.float32);target=rng.normal(size=(3,48,6)).astype(np.float32)
    observed=np.ones_like(target,dtype=bool);observed[0,0,0]=False;target[~observed]=np.nan
    raw={"history":history,"target":target,"observed":observed,"raw_last":history[:,-1].copy(),
        "mean":np.arange(6,dtype=np.float32),"std":np.arange(1,7,dtype=np.float32),"origins":np.arange(3,dtype=np.int64)+512}
    controls=fixed_orthogonal_controls();Q=controls[control]["Q"]
    identity=OrthogonalLatentBasisAdapter(frozen["center"],frozen["E"],frozen["D"],run.LAGS,identity=True)
    with torch.no_grad():p,z,u=identity(torch.from_numpy(history),toy_F,return_components=True)
    cached={"prediction":p.numpy().copy(),"latent_history":z.numpy().copy(),"latent_future":u.numpy().copy()}
    raw["raw_forecast"]=p.numpy().copy()+np.float32(.2)
    model=OrthogonalLatentBasisAdapter(frozen["center"],frozen["E"],frozen["D"],run.LAGS,Q,identity=control=="IDENTITY")
    return model,raw,cached,frozen,Q.astype(np.float32)


def result_fixture():
    return {"states":{state:{"controls":{}} for state in run.STATES},"failed_paths":[],
        "resolved_control_locks":0,"all_15_locks_before_DEV":False,"DEV_evaluations_completed":0}


class OrthogonalControlRunnerTests(unittest.TestCase):
    def test_five_actual_toy_controls_save_original_and_rotated_paths_and_diagnostics(self):
        for control in run.CONTROLS:
            with self.subTest(control=control):
                model,raw,cached,frozen,Q=fixture(control);checks=[]
                outputs,calls=run.infer_control(model,toy_F,raw["history"],cached,checks.append,batch_size=2)
                self.assertEqual(calls,2);self.assertEqual(len(checks),2)
                row,independent=run.evaluate_arrays(outputs,raw,cached,frozen,Q,control)
                self.assertEqual(row["status"],"COMPLETE");self.assertEqual(row["null_control_required"],control in ("IDENTITY","PERMUTATION"))
                self.assertEqual(outputs["old_past"].shape,(3,512,4));self.assertEqual(outputs["F_rotated_future"].shape,(3,48,4))
                self.assertEqual(set(independent),{"independent_NumPy_future","independent_NumPy_prediction"})
                self.assertEqual(len(row["latent_commutator"]["by_horizon_RMS"]),48)
                self.assertLess(row["transport_C_gap"]["max_abs"],1e-5)
                if control in ("SIGN_FLIP","DENSE_SO_s92711","DENSE_SO_s92712"):
                    self.assertGreater(row["latent_commutator"]["max_abs"],1e-3)
                json.dumps(row,allow_nan=False)

    def test_null_failure_and_wrong_inverse_are_preserved_as_failed_checks(self):
        model,raw,cached,frozen,Q=fixture();outputs,_=run.infer_control(model,toy_F,raw["history"],cached,lambda x:None)
        outputs["unrotated_future"][0,0,0]+=.1
        row,_=run.evaluate_arrays(outputs,raw,cached,frozen,Q,"IDENTITY")
        self.assertEqual(row["status"],"FAILED_PARITY_PRESERVED");self.assertFalse(row["checks"]["null_F_future_vs_original"]["pass"])
        self.assertFalse(row["checks"]["NumPy_inverse_rotated_F_future"]["pass"])
        self.assertIn("scores",row)

    def test_labels_and_unobserved_values_never_enter_control_forward(self):
        model,raw,cached,frozen,Q=fixture("DENSE_SO_s92712")
        self.assertEqual(tuple(inspect.signature(run.infer_control).parameters),('model','bolt','history','cached','check','batch_size'))
        outputs,_=run.infer_control(model,toy_F,raw["history"],cached,lambda x:None)
        row,_=run.evaluate_arrays(outputs,raw,cached,frozen,Q,"DENSE_SO_s92712")
        changed=copy.deepcopy(raw);changed["target"][~changed["observed"]]=float('inf')
        other,_=run.evaluate_arrays(outputs,changed,cached,frozen,Q,"DENSE_SO_s92712")
        self.assertEqual(row,other)
        changed["target"][changed["observed"]]+=1
        other,_=run.evaluate_arrays(outputs,changed,cached,frozen,Q,"DENSE_SO_s92712")
        self.assertEqual(row['C_commutator'],other['C_commutator']);self.assertNotEqual(row['scores'],other['scores'])
        again,_=run.infer_control(model,toy_F,raw["history"],cached,lambda x:None)
        for key in outputs:self.assertEqual(outputs[key].tobytes(),again[key].tobytes())

    def test_actual_F_callback_is_invoked_for_identity_and_rotated_control(self):
        for control in ("IDENTITY","DENSE_SO_s92711"):
            model,raw,cached,_,_=fixture(control);seen=[]
            def F(z):seen.append(z.clone());return toy_F(z)
            outputs,calls=run.infer_control(model,F,raw["history"],cached,lambda x:None,batch_size=1)
            self.assertEqual(calls,len(raw['history']));self.assertEqual(len(seen),3)
            np.testing.assert_array_equal(torch.cat(seen).numpy(),outputs['rotated_past'])
            if control=="IDENTITY":np.testing.assert_array_equal(outputs['rotated_past'],outputs['old_past'])

    def test_global_and_by_horizon_commutator_use_original_value_scales(self):
        a=np.arange(24,dtype=np.float32).reshape(2,3,4);b=np.zeros_like(a);d=run.difference(a,b)
        self.assertEqual(d['max_abs'],23.);self.assertEqual(d['RMS'],float(np.sqrt(np.mean(a.astype(np.float64)**2))))
        np.testing.assert_array_equal(d['by_horizon_RMS'],np.sqrt(np.mean(a.astype(np.float64)**2,axis=(0,2))))

    def test_fifteen_controls_are_all_sealed_before_any_DEV_and_failure_is_local(self):
        result=result_fixture();events=[]
        def execute(state,control):
            events.append(('run',state,control))
            if (state,control)==run.CONDITIONS[0]:raise ValueError('synthetic path failure')
            return {'status':'COMPLETE','splits':{'fit':{'status':'COMPLETE'},'tune':{'status':'COMPLETE'}}}
        def seal(state,control,row):events.append(('lock',state,control));return {'path':state+'_'+control,'sha256':'a'*64}
        def registry(rows):events.append(('registry',len(rows)));return {'path':'registry','sha256':'b'*64}
        resolved=run.resolve_all(result,execute,seal,registry)
        self.assertEqual(len(resolved),14);self.assertEqual(len(result['failed_paths']),1)
        self.assertEqual(sum(e[0]=='lock' for e in events),15)
        def development(state,control,row):
            self.assertEqual(sum(e[0]=='lock' for e in events),15);self.assertIn(('registry',15),events)
            events.append(('DEV',state,control));return {'status':'COMPLETE'}
        run.infer_development(result,resolved,development)
        self.assertEqual(result['DEV_evaluations_completed'],14)
        self.assertGreater(next(i for i,e in enumerate(events) if e[0]=='DEV'),next(i for i,e in enumerate(events) if e[0]=='registry'))

    def test_DEV_fails_closed_before_fifteen_locks_and_keeps_failed_check(self):
        result=result_fixture()
        with self.assertRaisesRegex(ValueError,'FIFTEEN'):run.infer_development(result,{},lambda *args:self.fail('DEV opened'))
        result.update(all_15_locks_before_DEV=True,resolved_control_locks=15)
        row={'splits':{}};resolved={run.CONDITIONS[0]:row}
        run.infer_development(result,resolved,lambda *args:{'status':'FAILED_PARITY_PRESERVED','artifact':{'path':'synthetic'}})
        self.assertEqual(result['DEV_evaluations_completed'],0);self.assertEqual(row['splits']['val']['status'],'FAILED_PARITY_PRESERVED')
        self.assertEqual(len(result['failed_paths']),1)

    def test_complete_metadata_orchestration_roundtrips_synthetic_artifacts_and_locks(self):
        with tempfile.TemporaryDirectory(dir=run.HERE) as temp:
            folder=Path(temp);result=result_fixture()
            def execute(state,control):
                model,raw,cached,frozen,Q=fixture(control)
                outputs,_=run.infer_control(model,toy_F,raw['history'],cached,lambda x:None)
                row,_=run.evaluate_arrays(outputs,raw,cached,frozen,Q,control)
                path=folder/(state+'_'+control+'.npz')
                with path.open('xb') as f:np.savez_compressed(f,**outputs,origins=raw['origins'])
                entry={'status':row['status'],'artifact':run.old.record(path),'scores':row['scores']}
                return {'status':row['status'],'splits':{'fit':entry,'tune':copy.deepcopy(entry)}}
            def write(path,value):path.write_text(json.dumps(value,allow_nan=False),encoding='utf-8');return run.old.record(path)
            def seal(state,control,row):return write(folder/(state+'_'+control+'_LOCK.json'),{'status':row['status'],'DEV_opened_before_lock':False,'Q_selection_performed':False})
            def registry(rows):return write(folder/'REGISTRY.json',{'status':'ALL_FIFTEEN_RESOLVED_BEFORE_DEV','conditions':rows,'DEV_opened_before_seal':False})
            resolved=run.resolve_all(result,execute,seal,registry)
            run.infer_development(result,resolved,lambda state,control,row:copy.deepcopy(row['splits']['fit']))
            self.assertEqual((len(resolved),result['DEV_evaluations_completed']),(15,15));self.assertFalse(result['failed_paths'])
            literal=json.loads(json.dumps(result,allow_nan=False))
            self.assertEqual(len(json.loads(run.old.bound(literal['control_registry']).read_text(encoding='utf-8'))['conditions']),15)
            self.assertEqual(len(list(folder.glob('*.npz'))),15)

    def test_closed_report_gate_precedes_all_IO_and_imports_no_scientific_runtime(self):
        with patch.object(run,'REQUIRE_REPORT_AUDIT_SHA',None),patch.object(run.old,'record',side_effect=AssertionError('IO forbidden')):
            with self.assertRaisesRegex(ValueError,'NOT_YET_REGISTERED'):run.load_gates(SimpleNamespace())
        code='import sys;sys.path.insert(0,'+repr(str(run.HERE))+');import run_orthogonal_basis_controls_v1 as r;assert "numpy" not in sys.modules and "torch" not in sys.modules;print("METADATA_ONLY_IMPORT_PASS")'
        result=subprocess.run([sys.executable,'-c',code],capture_output=True,text=True,check=True)
        self.assertIn('METADATA_ONLY_IMPORT_PASS',result.stdout)


if __name__=='__main__':unittest.main()
