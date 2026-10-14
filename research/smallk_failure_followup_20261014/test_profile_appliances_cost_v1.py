"""Synthetic source/metadata tests; no actual APP asset or GPU execution."""
from copy import deepcopy
import inspect
import pytest
from research.smallk_failure_followup_20261014 import profile_appliances_cost_v1 as app
from research.smallk_failure_followup_20261014.test_profile_joint_cost_v1 import producers


def fixture():
    values=[deepcopy(p) for p in producers() if p['dataset']=='robin']
    for p,o in zip(values,app.OBJECTIVES*2):
        p.update(dataset='appliances',objective=o,initial_codec_hash='init',original_MAIN_checkpoint={
            'q_aux_used_in_new_training':False,'q_aux_used_in_new_deployment':False,'checkpoint':{'sha256':str(p['seed'])}})
    return values


def test_all6_complete600_and_actual_selection_are_required():
    assert len(app.validate_producers(fixture()))==6
    with pytest.raises(ValueError,match='EXACT6'):app.validate_producers(fixture()[:-1])
    values=fixture();values[0]['learning_executed']['optimizer_updates']=599
    with pytest.raises(ValueError,match='COMPLETE600'):app.validate_producers(values)
    values=fixture();values[0]['checkpoints'][0]['tune_actual_Y_mse']=0.
    with pytest.raises(ValueError,match='TUNE_SELECTED'):app.validate_producers(values)


def test_original_q_is_preserved_unused_and_same_batch_order_checked():
    values=fixture();values[0]['original_MAIN_checkpoint']['q_aux_used_in_new_deployment']=True
    with pytest.raises(ValueError,match='Q_PRESERVED'):app.validate_producers(values)
    values=fixture();values[1]['training_order_sha256']='different'
    with pytest.raises(ValueError,match='SAME_INITIAL'):app.validate_producers(values)


def test_APP_pass_audit_must_match_actual_producer_SHA():
    values=fixture();records=[{'sha256':str(i)} for i in range(6)]
    rows=[{'dataset':p['dataset'],'seed':p['seed'],'objective':p['objective'],'producer_sha256':r['sha256']} for p,r in zip(values,records)]
    app.audit_binding({'status':'PASS','failed_count':0,'conditions':rows},values,records)
    rows[0]['producer_sha256']='other'
    with pytest.raises(ValueError,match='SHA_BINDING'):app.audit_binding({'status':'PASS','failed_count':0,'conditions':rows},values,records)


def test_frozen_whole_helpers_reused_without_new_math_or_stage_profiles():
    source=inspect.getsource(app)
    assert 'joint.measure_whole(' in source and 'base.DeploymentPipeline(' in source and 'base.validate_raw_inputs(' in source
    assert 'def fill_' not in source and 'decomposed(' not in source
    assert 'limit.shared.check()' in source and 'limit.check()' in source
    assert "'RAW_PLUS_R_cost_measured':False" in source and "'q_aux_used_in_new_deployment':False" in source
