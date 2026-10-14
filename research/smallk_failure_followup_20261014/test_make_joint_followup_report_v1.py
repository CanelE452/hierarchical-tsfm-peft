"""Report adapter tests with synthetic terminal JSON only; no assets/GPU/plots."""
from copy import deepcopy
import inspect
from pathlib import Path

import pytest

from research.smallk_failure_followup_20261014 import make_joint_followup_report_v1 as r
from research.smallk_failure_followup_20261014.test_profile_joint_cost_v1 import producers


def joint_fixture():
    values=producers()
    score={'normalized_actual_Y':{'mse':.5,'mae':.4},'physical_actual_Y':{'mse':5.,'mae':4.},'normalized_RAW_teacher':{'mse':.1,'mae':.2}}
    for p in values:
        splits=('fit','tune','val' if p['dataset']=='robin' else 'eval')
        p.update(selected_inference={s:{'scores':deepcopy(score)} for s in splits},
            original_MAIN_scores={s:deepcopy(score) for s in splits},original_RAW_scores={s:deepcopy(score) for s in splits},
            curve=[{'new_update':u,'objective_loss':.2,'actual_Y_loss':.5,'cached_RAW_teacher_loss':.1} for u in range(1,601)],
            FIT_target_finite_contract={'actual_Y_unobserved_nonfinite_count':0},new_update_parameter_proof={'E':{'changed_parameter_tensors':2},'D':{'changed_parameter_tensors':3}},
            training_wall_seconds=10.,runner_wall_seconds=20.,guard_cadence={'explicit_wall_seconds':{'full':2.,'fast_base':.1}},
            cached_teacher_cost={'historical_teacher_generation_compute_wall_seconds':None},
            gradient_diagnostics=[{'new_update':u,'frozen_F_input_gradient_norm':.02,'frozen_F_input_gradient_finite':True} for u in (0,99,199,399,599)])
        for row in p['checkpoints']:row['checkpoint']['sha256']='cp'
    return values


def cost_fixture():
    rows=[]
    for d in r.DATASETS:
        for s,o in [(None,None)]+[(s,o) for s in r.SEEDS for o in r.OBJECTIVES]:
            for b in (1,8):
                rows.append({'dataset':d,'seed':s,'objective':o,'batch':b,'status':'COMPLETE','warmup':2,'repeats':7,
                    'method':'RAW_BOLT' if s is None else 'JOINT_SELECTED','seconds':[.01]*7,
                    'parity':[{'matched_batch_normalized':{'pass':True},'matched_batch_physical':{'pass':True},
                              'original_bulk_normalized':{'pass':False},'original_bulk_physical':{'pass':True}} for _ in range(7)],
                    'selected_new_update':None if s is None else 0,'selected_is_original_MAIN':None if s is None else True,
                    'median_seconds':.01,'median_seconds_per_window':.01/b,'timed_outputs':{'path':'literal/cache.npz','sha256':'outputs'},
                    'memory':{'scope':'co-resident'},'preprocessing':'same D3 vectorized causal CPU fill'})
    return {'status':'COMPLETE','actual_complete_rows':28,'rows':rows,'F_before':{'old':'same'},'F_after':{'old':'same'}}


def test_joint_adapter_preserves_all12_600_curves_and_selected_metadata():
    tables=r.joint_tables(joint_fixture())
    assert len(tables['joint_learning'])==12 and len(tables['joint_curve'])==7200
    assert len(tables['joint_checkpoints'])==60 and len(tables['joint_gradients'])==60
    assert all(row['historical_teacher_generation_seconds'] is None for row in tables['joint_learning'])
    assert all(row['guard_full_check_wall_seconds']==2. for row in tables['joint_learning'])
    assert len(r.metric_groups(tables['joint_scores']))==20


def test_duplicate_shared_baseline_is_not_counted_six_times_or_averaged_silently():
    rows=[{'dataset':'robin','split':'val','seed':92601,'method':'RAW_BOLT','mse':.5,'mae':.4}]*3
    assert len(r.metric_groups(rows))==1
    values=[dict(row) for row in rows];values[1]['mse']=.6
    with pytest.raises(ValueError,match='BASELINE_DUPLICATION'):r.metric_groups(values)


def test_cost28_requires_complete_matched_outputs_but_keeps_bulk_failures():
    value=cost_fixture();rows=r.cost_tables(value)
    assert len(rows)==28 and all(row['original_bulk_normalized_failed_repeats']==7 for row in rows)
    assert any(row['selected0'] is True for row in rows)
    value['rows'][0]['parity'][0]['matched_batch_physical']['pass']=False
    with pytest.raises(ValueError,match='MATCHED_BATCH_GATE'):r.cost_tables(value)


def test_cost_incomplete_duplicate_and_f_changed_do_not_create_actual_chart():
    value=cost_fixture();value['actual_complete_rows']=27
    with pytest.raises(ValueError,match='TERMINAL_COMPLETE'):r.cost_tables(value)
    value=cost_fixture();value['rows'][-1]=value['rows'][0]
    with pytest.raises(ValueError,match='COVERAGE'):r.cost_tables(value)
    value=cost_fixture();value['F_after']={'old':'changed'}
    with pytest.raises(ValueError,match='PRESERVATION'):r.cost_tables(value)


def test_APP_missing_requires_explicit_reason_and_remains_NOT_EXECUTED():
    with pytest.raises(ValueError,match='REASONS_REQUIRED'):r.app_tables([],None)
    rows,statuses=r.app_tables([],'asset preparation is still pending; no fits started')
    assert rows==[] and len(statuses)==6 and {x['status'] for x in statuses}=={'NOT_EXECUTED'}


def test_APP_complete_vs_failed_are_separate_and_learning_proof_required():
    p=joint_fixture()[0];p.update(dataset='appliances',objective='MSE')
    rows,statuses=r.app_tables([p],'remaining5 not executed under current cap')
    assert rows and statuses[0]['status']=='COMPLETE'
    p['learning_executed']['optimizer_updates']=599
    with pytest.raises(ValueError,match='600_PROOF'):r.app_tables([p],'remaining5 missing')
    p.update(status='BUDGET_HELD_PRESERVED',failure={'message':'wall cap'})
    rows,statuses=r.app_tables([p],'remaining5 missing')
    assert rows==[] and statuses[0]['status']=='BUDGET_HELD_PRESERVED'


def test_audit_failure_and_nonterminal_never_count_as_pass():
    r.passed_audit({'status':'PASS','failed_count':0})
    for value in ({'status':'RUNNING','failed_count':0},{'status':'PASS','failed_count':1}):
        with pytest.raises(ValueError,match='INDEPENDENT_AUDIT'):r.passed_audit(value)


def test_GitHub_links_require_real_files_and_reject_cache_or_invented_images(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'ROOT',tmp_path)
    out=tmp_path/'report';out.mkdir();(out/'figure.png').write_bytes(b'fixture')
    context=tmp_path/'prior';context.mkdir();(context/'REPORT.md').write_text('actual')
    md='![actual](figure.png) [context](../prior/REPORT.md)'
    assert len(r.checked_links(md,out)[1])==2
    with pytest.raises(ValueError,match='UNSAFE'):r.checked_links('![invented](missing.png)',out)
    cache=tmp_path/'.cache';cache.mkdir();(cache/'RESULT.json').write_text('{}')
    with pytest.raises(ValueError,match='UNSAFE'):r.checked_links('[cache](../.cache/RESULT.json)',out)


def test_import_and_describe_have_no_scientific_reads_or_cuda():
    source=inspect.getsource(r)
    assert 'np.load(' not in source and 'torch.load(' not in source and "import torch" not in source
    assert "--no-contention-attested" in source and "whole_goal_complete':False" in source
    assert len(r.FIGURES)==6
