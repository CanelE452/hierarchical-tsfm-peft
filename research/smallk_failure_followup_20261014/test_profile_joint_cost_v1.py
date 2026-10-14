"""Synthetic source/metadata/timing checks; no assets or model/GPU execution."""
from copy import deepcopy
from pathlib import Path
import inspect

import numpy as np
import pytest

from research.smallk_failure_followup_20261014 import profile_joint_cost_v1 as jc
from research.smallk_failure_followup_20261014 import profile_latency as base


def producers():
    values=[]
    for dataset in jc.DATASETS:
        for seed in jc.SEEDS:
            for objective in jc.OBJECTIVES:
                cps=[{'new_update':u,'tune_actual_Y_mse':1.-u/1000,
                      'checkpoint':{'path':str(u),'state_sha_old_algorithm':'selected','state_sha_Bi_algorithm':'bi'}} for u in (0,100,200,400,600)]
                values.append({'dataset':dataset,'seed':seed,'objective':objective,'status':'COMPLETE',
                    'learning_executed':{'optimizer_updates':600,'full_learning_proof_pass':True,'E_updated':True,'D_updated':True,'frozen_F_input_gradient_nonzero_diagnostic_count':5},
                    'F_frozen_pass':True,'F_has_no_parameter_gradients':True,'initial_fixed_basis_unchanged':True,
                    'selected_checkpoint':cps[4]['checkpoint'],'selected_index':4,'checkpoints':cps,
                    'selected_new_update':600,'selected_codec_hash':'selected','selected_is_initial_original_MAIN':False,
                    'training_order_sha256':str(seed),'same_initialization_across_objectives_key':{
                        'dataset':dataset,'seed':seed,'original_MAIN_CP_sha256':dataset+str(seed),'initial_codec_hash':'init'}})
    return values


def test_all12_conditions_and_learning_proof_are_required():
    values=producers();assert len(jc.validate_joint_producers(values))==12
    with pytest.raises(ValueError,match='EXACT12'):
        jc.validate_joint_producers(values[:-1])
    with pytest.raises(ValueError,match='EXACT12'):
        jc.validate_joint_producers(values+[values[0]])
    values[0]['learning_executed']['D_updated']=False
    with pytest.raises(ValueError,match='600_LEARNING'):
        jc.validate_joint_producers(values)


def test_shared_initialization_order_and_actualY_TUNE_selection_are_checked():
    values=producers();values[1]['training_order_sha256']='different'
    with pytest.raises(ValueError,match='SAME_INITIALIZATION'):
        jc.validate_joint_producers(values)
    values=producers();values[0]['checkpoints'][0]['tune_actual_Y_mse']=.1
    with pytest.raises(ValueError,match='TUNE_SELECTED'):
        jc.validate_joint_producers(values)


def test_initial0_selection_is_preserved_without_new_training_advantage():
    values=producers();p=values[0]
    p['checkpoints'][0]['tune_actual_Y_mse']=.1
    p.update(selected_index=0,selected_checkpoint=p['checkpoints'][0]['checkpoint'],
             selected_new_update=0,selected_is_initial_original_MAIN=True)
    assert jc.validate_joint_producers(values)[('robin',92601,'Y')]['selected_new_update']==0


def test_passed_audit_is_bound_to_exact_all12_producer_hashes():
    values=producers();records=[{'sha256':str(i)} for i in range(12)]
    rows=[{'dataset':p['dataset'],'seed':p['seed'],'objective':p['objective'],'producer_sha256':r['sha256']}
          for p,r in zip(values,records)]
    jc.verify_audit_binding({'conditions':rows},values,records)
    rows[0]['producer_sha256']='other-result'
    with pytest.raises(ValueError,match='AUDIT_PRODUCER_SHA'):
        jc.verify_audit_binding({'conditions':rows},values,records)
    with pytest.raises(ValueError,match='AUDIT_PRODUCER_SHA'):
        jc.verify_audit_binding({'conditions':rows[:-1]},values,records)


def test_selected_checkpoint_metadata_and_original_target_contract_are_required():
    p=producers()[0]
    saved={'architecture':'ANCHOR_NONLINEAR_D','new_joint_comparison':True,'seed':p['seed'],'objective':p['objective'],'new_update':600}
    jc.verify_checkpoint_metadata(saved,p)
    saved['objective']='RAW'
    with pytest.raises(ValueError,match='CHECKPOINT_METADATA'):
        jc.verify_checkpoint_metadata(saved,p)
    values={'target':np.array([[[1.,np.nan]]],np.float32),'observed':np.array([[[True,False]]]),'origins':np.array([512],np.int64)}
    mean,std=np.zeros(2,np.float32),np.ones(2,np.float32)
    arrays={**values,'mean':mean,'std':std,'prediction':np.zeros((1,1,2),np.float32)}
    jc.verify_bulk_conditions(arrays,values,mean,std)
    arrays['origins']=np.array([513],np.int64)
    with pytest.raises(ValueError,match='CONDITION_MISMATCH'):
        jc.verify_bulk_conditions(arrays,values,mean,std)


class Backend:
    def synchronize(self):pass
    def reset_memory(self):pass
    def to_numpy(self,value):return value.copy()
    def memory_snapshot(self):return {'synthetic':True}


class Pipeline:
    def __init__(self):self.backend=Backend();self.calls=[]
    def whole(self,physical,observed,kind):
        self.calls.append((physical.shape,kind))
        future=np.repeat(physical[:,-1:,:],48,axis=1)
        return future,future*2+1


def save(path,arrays):
    with Path(path).open('xb') as f:np.savez_compressed(f,**arrays)
    return {'path':str(path)}


def test_all7_outputs_saved_and_original_bulk_failure_separate(tmp_path):
    pipeline=Pipeline();physical=np.arange(1024,dtype=np.float32).reshape(1,512,2);mask=np.ones_like(physical,bool)
    normalized,actual=pipeline.whole(physical,mask,'vectorized')
    reference={'normalized':normalized,'physical':actual};bulk={'normalized':normalized+1,'physical':actual+2}
    checks=[]
    row=jc.measure_whole(pipeline,physical,mask,reference,bulk,lambda:checks.append(1),tmp_path,save,base)
    assert row['status']=='COMPLETE' and len(row['seconds'])==7
    assert len(pipeline.calls)==10 and all(shape==(1,512,2) for shape,_ in pipeline.calls)
    assert all(c['matched_batch_physical']['pass'] for c in row['parity'])
    assert not any(c['original_bulk_physical']['pass'] for c in row['parity'])
    with np.load(tmp_path/'TIMED_ALL7.npz',allow_pickle=False) as a:
        assert a['normalized'].shape==(7,1,48,2) and a['physical'].shape==(7,1,48,2)
    assert row['normalized_D2H_copy_outside_timer'] is True and row['stage_decomposition'] is False


def test_matched_batch_failure_retains_outputs(tmp_path):
    pipeline=Pipeline();physical=np.zeros((1,512,2),np.float32);mask=np.ones_like(physical,bool)
    normalized,actual=pipeline.whole(physical,mask,'vectorized')
    wrong={'normalized':normalized+1,'physical':actual+1}
    row=jc.measure_whole(pipeline,physical,mask,wrong,wrong,lambda:None,tmp_path,save,base)
    assert row['status']=='FAILED_PRESERVED' and (tmp_path/'TIMED_ALL7.npz').exists()


def test_existing_D3_whole_pipeline_reused_and_no_automatic_execution():
    source=inspect.getsource(jc)
    assert 'base.DeploymentPipeline(' in source and 'base.validate_raw_inputs(' in source
    assert 'def vectorized_preprocess' not in source and 'decomposed(' not in source
    assert 'no_contention_attested' in source and 'PASSED_JOINT_INDEPENDENT_AUDIT_REQUIRED' in source
    assert 'limit.shared.check()' in source and 'limit.check()' in source
