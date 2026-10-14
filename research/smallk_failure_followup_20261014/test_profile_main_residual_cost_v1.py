"""Prepared CPU fixtures only. No scientific asset/model/runtime invocation."""
from pathlib import Path
from types import SimpleNamespace
import importlib.util
import json
import sys
import tempfile
import unittest

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import profile_main_residual_cost_v1 as profile


class Tensor:
    transfers=0
    def __init__(self,value): self.value=np.asarray(value,np.float32)
    def __getitem__(self,key): return Tensor(self.value[key])
    def __add__(self,other): return Tensor(self.value+other.value)
    def __sub__(self,other): return Tensor(self.value-other.value)
    def detach(self): return self
    def cpu(self): Tensor.transfers+=1; return self
    def numpy(self): return self.value


class Codec:
    def __init__(self,channels): self.channels=channels; self.encodes=0; self.decodes=0
    def encode(self,x): self.encodes+=1; return Tensor(x.value[:,:,:1])
    def decode(self,z): self.decodes+=1; return Tensor(np.repeat(z.value,self.channels,axis=2))


class ProtocolTests(unittest.TestCase):
    def test_six_route_Latin_coverage_and_FIXED_seed_batch_protocol(self):
        schedule=profile.balanced_schedule()
        self.assertEqual(schedule,profile.balanced_schedule())
        self.assertEqual(len(schedule),6)
        for row in schedule:self.assertEqual(set(row),set(profile.ROUTES))
        for route in profile.ROUTES:self.assertEqual({row.index(route) for row in schedule},set(range(6)))
        self.assertEqual(profile.SEED,92601);self.assertEqual(profile.WARMUP,2)
        self.assertEqual(profile.REPEATS,6);self.assertEqual(profile.LOCAL_SECONDS,240)

    def test_zero_skip_transfers_preserves_MAIN_bits_and_encodes_once(self):
        history=Tensor(np.linspace(-1,1,2*512*3,dtype=np.float32).reshape(2,512,3))
        codec=Codec(3);forecasts=[]
        def forecaster(z):forecasts.append(True);return Tensor(np.repeat(z.value[:,-1:],48,axis=1)+np.float32(.03125))
        model=SimpleNamespace(solution=None,mean_control=None,forecast=lambda *_:self.fail('ZERO called feature forecast'))
        def to_device(_):self.fail('ZERO called normalized H2D')
        route=profile.HybridCorrectionRoute(codec,model,'E512K',to_device)
        Tensor.transfers=0;actual=route.forward_path(history,forecaster,return_delta=False)
        expected=history.value[:,-1:]+np.repeat(history.value[:,-1:,:1]+np.float32(.03125),48,axis=1).repeat(3,axis=2)-np.repeat(history.value[:,-1:,:1],3,axis=2)
        np.testing.assert_array_equal(actual.value.view(np.uint32),expected.view(np.uint32))
        self.assertEqual(codec.encodes,1);self.assertEqual(codec.decodes,2);self.assertEqual(len(forecasts),1)
        self.assertEqual(Tensor.transfers,0)
        with self.assertRaises(ValueError):route.forward_path(history,forecaster,return_delta=True)

    def test_mean_uses_only_MAIN_CPU_transfer_and_shared_control(self):
        history=Tensor(np.zeros((1,512,3),np.float32));codec=Codec(3);calls=[]
        correction=np.full((48,3),np.float32(.125))
        control=SimpleNamespace(forecast=lambda main:(calls.append(main.copy()) or main+correction))
        for route_name in ('MEAN_CONTROL','E33K'):
            calls.clear();Tensor.transfers=0;uploads=[]
            model=None if route_name=='MEAN_CONTROL' else SimpleNamespace(solution=None,mean_control=control)
            route=profile.HybridCorrectionRoute(codec,model,route_name,lambda a:(uploads.append(a.copy()) or Tensor(a)),control)
            value=route.forward_path(history,lambda z:Tensor(np.zeros((1,48,1),np.float32)))
            np.testing.assert_array_equal(value.value,np.full((1,48,3),.125,np.float32))
            self.assertEqual(Tensor.transfers,1);self.assertEqual(len(calls),1);self.assertEqual(len(uploads),1)

    def test_ridge_RAW_and_E_features_use_declared_source_with_FP32_main(self):
        history=Tensor(np.arange(512*3,dtype=np.float32).reshape(1,512,3)/np.float32(1000))
        for name,channels in (('RAW33C',3),('E33K',1),('E512K',1)):
            captured=[];Tensor.transfers=0;codec=Codec(3)
            def forecast(past,main):captured.append((past.copy(),main.copy()));return (main+np.float32(.25)).astype(np.float32)
            model=SimpleNamespace(solution=object(),mean_control=None,forecast=forecast)
            route=profile.HybridCorrectionRoute(codec,model,name,Tensor)
            value=route.forward_path(history,lambda z:Tensor(np.repeat(z.value[:,-1:],48,axis=1)))
            self.assertEqual(captured[0][0].shape,(1,512,channels))
            self.assertEqual(captured[0][1].dtype,np.float32);self.assertEqual(value.value.dtype,np.float32)
            self.assertEqual(Tensor.transfers,2);self.assertEqual(codec.encodes,1)

    def test_PASS_audit_requires_four_conditions_and_cost_producer_SHA(self):
        producers=[{'dataset':d,'seed':92601} for d in profile.DATASETS];records=[{'sha256':'a'*64},{'sha256':'b'*64}]
        rows=[{'dataset':d,'seed':s,'status':'PASS','producer_sha256':records[i]['sha256'] if s==92601 else 'c'*64} for i,d in enumerate(profile.DATASETS) for s in (92601,92602)]
        audit={'status':'PASS','failed_count':0,'conditions':rows};profile.audit_binding(audit,producers,records)
        with self.assertRaises(ValueError):profile.audit_binding({**audit,'conditions':rows[:2]},producers,records)
        with self.assertRaises(ValueError):profile.audit_binding({**audit,'status':'PASS_WITH_UNKNOWN_SCOPE'},producers,records)
        rows[0]['producer_sha256']='d'*64
        with self.assertRaises(ValueError):profile.audit_binding(audit,producers,records)

    def test_partial_measurement_preserves_samples_instead_of_complete(self):
        backend=SimpleNamespace(synchronize=lambda:None,reset_memory=lambda:None,to_numpy=lambda x:x,
                                memory_snapshot=lambda:{'scope':'fixture'})
        prediction=np.zeros((1,48,2),np.float32)
        pipelines={r:SimpleNamespace(backend=backend,whole=lambda *_:(prediction.copy(),prediction.copy())) for r in profile.ROUTES}
        references={r:{'normalized':prediction,'physical':prediction} for r in profile.ROUTES}
        guards=[];saved=[]
        def fast():
            guards.append(True)
            if len(guards)>26:raise RuntimeError('fixture interrupted after first timed output')
        def save(path,arrays):saved.append(arrays);return {'path':str(path),'arrays':{k:list(v.shape) for k,v in arrays.items()}}
        base=SimpleNamespace(parity=lambda *_:{'pass':True})
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):profile.measure_group(pipelines,prediction,np.ones_like(prediction,bool),references,references,Path(directory),fast,lambda:None,save,base)
            rows=json.loads((Path(directory)/'ROWS.json').read_text())['rows']
            self.assertEqual(len(rows),6);self.assertEqual(len(saved),1)
            self.assertFalse(any(r['status']=='COMPLETE' for r in rows));self.assertEqual(sum(r['repeats'] for r in rows),1)

    def test_source_authorization_before_assets_and_uses_driver_v2(self):
        source=(HERE/'profile_main_residual_cost_v1.py').read_text(encoding='utf-8')
        start=source.index('def execute(args):');body=source[start:]
        gate=body.index("runtime.ContinuationLimit('main_residual_cost_'+args.tag,gpu=True)")
        for fragment in ('output.mkdir','import numpy as np','import torch','probes.read(path)','probes.import_old_engine()'):
            self.assertLess(gate,body.index(fragment))
        self.assertLess(body.index("declare_gpu_forward_scope('frozen_F_cost')"),body.index('probes.import_old_engine()'))
        self.assertIn('import run_main_residual_v2 as worker',body)
        self.assertIn('MAIN_RESIDUAL_AUDIT_v1_ACTUAL.json',body)


if __name__=='__main__':unittest.main()
