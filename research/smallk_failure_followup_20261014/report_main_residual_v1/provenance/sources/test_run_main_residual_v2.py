"""SOURCE fixtures prepared only; no real assets or executed solver/model.

Future execution requires the fresh main-residual runtime. Synthetic NPZ files
use original raw-byte manifests; manufactured coefficients are not fit results.
"""
import ast
import copy
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import run_main_residual_v2 as runner
import run_condition_probes as probes
import main_forecast_residual_readout_v1 as math
import stage_separation_math_v1 as stage


def test_runtime_precedes_output_assets_numpy_and_math_import():
    body=ast.parse(inspect.getsource(runner.execute)).body[0].body
    assert isinstance(body[0],ast.Import) and body[0].names[0].name=='main_residual_runtime_v1'
    assert body[1].value.func.attr=='ContinuationLimit'
    assert body[1].value.keywords[0].arg=='gpu' and body[1].value.keywords[0].value.value is False
    source=inspect.getsource(runner.execute)
    gate=source.index('runtime.ContinuationLimit(')
    for token in ('output.mkdir(','import numpy','import main_forecast_residual','probes.read(','probes.load_checked_npz('):
        assert gate<source.index(token)
    assert 'torch.load' not in source and 'import_old_engine(' not in source and '.cuda(' not in source


def test_missing_runtime_authorization_opens_no_assets_or_output(tmp_path,monkeypatch):
    class GateDenied(RuntimeError): pass
    def deny(tag,gpu):
        assert gpu is False
        raise GateDenied('synthetic approval gate')
    monkeypatch.setitem(sys.modules,'main_residual_runtime_v1',SimpleNamespace(ContinuationLimit=deny))
    monkeypatch.setattr(runner,'CACHE',tmp_path/'never_created')
    with pytest.raises(GateDenied): runner.execute(SimpleNamespace(tag='source_fixture',dataset='robin',seed=92601))
    assert not (tmp_path/'never_created').exists()


def test_matched_source_condition_and_scope_rejected_when_changed():
    valid={'status':'COMPLETE','dataset':'robin','seed':92601,'GPU_execution':False,
           'E_D_F_loaded':False,'E_D_F_forward_replayed':False,'E_D_F_updated':False,
           'future_oracle_read':False,'independent_holdout':False}
    runner.validate_matched_flags(valid,'robin',92601)
    for key,value in (('dataset','beijing_aotizhongxin'),('seed',92602),('future_oracle_read',True),('E_D_F_loaded',True)):
        with pytest.raises(ValueError): runner.validate_matched_flags({**valid,key:value},'robin',92601)


def test_canonical_report_and_audit_bind_exact_matched_producer(tmp_path,monkeypatch):
    record={'path':'/synthetic/matched.json','sha256':'a'*64}
    conditions=[{'dataset':d,'seed':s,'status':'PASS','producer_sha256':'a'*64 if (d,s)==('robin',92601) else str(i+1)*64}
                for i,(d,s) in enumerate((('robin',92601),('robin',92602),('beijing_aotizhongxin',92601),('beijing_aotizhongxin',92602)))]
    audit={'status':'PASS','failed_count':0,'conditions':conditions}
    audit_path=tmp_path/'AUDIT.json'; probes.write(audit_path,audit)
    manifest={'status':'COMPLETE','audit_gate':{'passed':True},'input_records':[
        {'role':'producer',**record},{'role':'independent_audit','path':str(audit_path),'sha256':probes.sha(audit_path)},
        {'path':'synthetic_authorization.json','sha256':'b'*64}]}
    manifest_path=tmp_path/'MANIFEST.json'; probes.write(manifest_path,manifest)
    monkeypatch.setattr(runner,'CANONICAL_MATCHED_REPORT_SHA',probes.sha(manifest_path))
    assert len(runner.verify_canonical_audit(probes,manifest_path,record,'robin',92601))==2
    with pytest.raises(ValueError,match='NOT_BOUND'):
        runner.verify_canonical_audit(probes,manifest_path,{**record,'sha256':'f'*64},'robin',92601)
    with pytest.raises(ValueError,match='EXACT_PRODUCER_SHA'):
        runner.verify_canonical_audit(probes,manifest_path,record,'robin',92602)
    bad=copy.deepcopy(audit); bad['conditions'][0]['producer_sha256']='f'*64
    bad_path=tmp_path/'BAD_AUDIT.json'; probes.write(bad_path,bad)
    altered=copy.deepcopy(manifest)
    altered['input_records'][1]={'role':'independent_audit','path':str(bad_path),'sha256':probes.sha(bad_path)}
    alt_path=tmp_path/'BAD_MANIFEST.json'; probes.write(alt_path,altered)
    monkeypatch.setattr(runner,'CANONICAL_MATCHED_REPORT_SHA',probes.sha(alt_path))
    with pytest.raises(ValueError,match='EXACT_PRODUCER_SHA'):
        runner.verify_canonical_audit(probes,alt_path,record,'robin',92601)


def synthetic_selected_family(tmp_path,kind):
    past=np.zeros((3,512,1),np.float32)
    state=math.fit_feature_state('E33K',past)
    main=np.full((3,48,2),.25,np.float32)
    target=main+np.float32(.125); observed=np.ones(main.shape,bool)
    control=math.fit_intercept_only(main,target,observed)
    arrays=None
    if kind==math.ZERO:
        model=math.zero_correction_model(state,2)
    elif kind==math.MEAN:
        model=math.mean_correction_model(state,2,control)
        arrays={'correction':control.correction,'observed_counts':control.observed_counts}
    else:
        coefficients=np.zeros((33,48,2),np.float64)
        intercept=np.full((48,2),.125,np.float64)
        counts=np.full((48,2),3,np.int64)
        solution=stage.RidgeSolution('MAIN_RESIDUAL_E33K',.1,stage._readonly(coefficients),stage._readonly(intercept),
            stage._readonly(counts,np.int64),stage._readonly([144,144],np.int64),stage._readonly(np.full((48,2),144.)),
            stage._readonly(np.zeros((48,2)),np.int64),stage._readonly(np.ones((48,2))),state.sha256,True)
        model=math.CorrectionModel(state,2,solution); arrays=runner.solution_arrays(solution)
    feature_record=probes.save_npz(tmp_path/(kind+'_FEATURE.npz'),{'center':state.center,'scale':state.scale})
    coefficient_record=None if arrays is None else probes.save_npz(tmp_path/(kind+'_MODEL.npz'),arrays)
    row={'status':'COMPLETE','feature_state':{**state.metadata(),'artifact':feature_record},
        'selected_model_metadata':model.metadata(),'selected_coefficient_artifact':coefficient_record,
        'selected_model_sha256':model.sha256}
    return {'families':{'E33K':row}},model,past,main


@pytest.mark.parametrize('candidate',(math.ZERO,math.MEAN,'RIDGE'))
def test_selected_zero_mean_ridge_restored_with_raw_manifest_no_refit(tmp_path,candidate):
    producer,model,past,main=synthetic_selected_family(tmp_path,candidate)
    restored=runner.load_selected_models(producer,probes)
    assert restored['RAW33C'] is None and restored['E512K'] is None
    assert restored['E33K'].sha256==model.sha256
    np.testing.assert_array_equal(restored['E33K'].forecast(past,main),model.forecast(past,main))
    manifest=producer['families']['E33K']['feature_state']['artifact']['arrays']['center']
    assert manifest['sha256']==probes.array_sha(model.feature_state.center)
    assert manifest['sha256']!=stage.array_sha(model.feature_state.center)
    source=inspect.getsource(runner.load_selected_models)
    assert '.fit(' not in source and 'prepare_' not in source


def test_changed_selected_model_or_coefficients_rejected(tmp_path):
    producer,model,_,_=synthetic_selected_family(tmp_path,'RIDGE')
    bad=copy.deepcopy(producer); bad['families']['E33K']['selected_model_sha256']='0'*64
    with pytest.raises(ValueError): runner.load_selected_models(bad,probes)
    arrays={key:value.copy() for key,value in runner.solution_arrays(model.solution).items()}
    arrays['coefficients'][0,0,0]=1.
    producer['families']['E33K']['selected_coefficient_artifact']=probes.save_npz(tmp_path/'CHANGED_COEFF.npz',arrays)
    with pytest.raises(ValueError,match='SOLUTION_SHA'): runner.load_selected_models(producer,probes)


def test_controls_shared_once_before_any_SVD_and_all_routes_keep_fixed_pool():
    source=inspect.getsource(runner.execute)
    assert source.count('mean_control=math.fit_intercept_only(')==1
    assert source.index('mean_control=math.fit_intercept_only(')<source.index('prepared=math.prepare_with_state(')
    assert source.index('zero=math.zero_correction_model(')<source.index('prepared=math.prepare_with_state(')
    assert source.index('mean_model=math.mean_correction_model(')<source.index('prepared=math.prepare_with_state(')
    assert "fit_route('RAW33C'" in source and "for route in ('E33K','E512K')" in source
    assert 'index=math.select_candidate(row[\'candidate_grid\'])' in source


def test_candidate_failure_is_preserved_but_budget_propagates_and_inputs_not_copied():
    source=inspect.getsource(runner.execute)
    assert "status='FAILED_PRESERVED',tune_actual_Y_mse=None" in source
    assert 'if isinstance(error,runtime.BudgetHeld): raise' in source
    assert 'selected_original_MAIN_unchanged=model.candidate_id==math.ZERO' in source
    assert "'big_input_arrays_copied':False" in source
    assert "cached_records[split]=declared" in source
    assert "original_prediction'],values['main_forecast']" in source
    assert "'actual-Y" not in inspect.getsource(runner.load_selected_models)


def test_candidate_model_prediction_requires_only_past_and_frozen_MAIN():
    assert tuple(inspect.signature(math.CorrectionModel.forecast).parameters)==('self','past','main_forecast')
    source=inspect.getsource(runner.execute)
    assert "raw_splits[split]['main_forecast']" in source
    assert "fit['target'],fit['observed']" in source
    assert "model.forecast(past_by_split[split],values['main_forecast'])" in source


def test_resource_terminal_cannot_override_budget_failure_as_complete():
    source=inspect.getsource(runner.execute)
    assert "if receipt.get('terminal_status')!='COMPLETE':" in source
    assert "result.update(status=receipt['terminal_status']" in source
    assert "if initialized: runtime.write_new(output/'RESULT.json',result)" in source
