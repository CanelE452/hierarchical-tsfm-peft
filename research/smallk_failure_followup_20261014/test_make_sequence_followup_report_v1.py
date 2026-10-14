"""Synthetic report contracts only; actual tests require root CPU GO."""
from copy import deepcopy
import inspect
import pytest
from research.smallk_failure_followup_20261014 import make_sequence_followup_report_v1 as report


def queue_fixture():
    return {'status':'COMPLETE','conditions':[{'dataset':d,'seed':s,'head':h,'status':'COMPLETE',
        'producer_result':{'path':f'actual/{d}_{s}_{h}/RESULT.json','sha256':f'{d}_{s}_{h}'}}
        for d,s,h in sorted(report.EXPECTED)]}


def test_cost_arguments_use_all8_declared_actual_paths_and_reject_partial():
    queue=queue_fixture();args=report.cost_cli_arguments(queue,'bridge.json','audit.json','fresh')
    assert all(row['producer_result']['path'] in args for row in queue['conditions'])
    assert '--no-contention-attested' in args
    queue['conditions'][0]['status']='BUDGET_HELD_PRESERVED'
    with pytest.raises(ValueError,match='ALL8_COMPLETE'):report.cost_cli_arguments(queue,'b','a','t')


def test_empty_or_unbound_audit_cannot_become_pass():
    assert report.audit_gate({'status':'PASS','failed_count':0,'conditions':[]},[],[])['passed'] is False
    records=report.producer_refs(queue_fixture());producers=[{k:r[k] for k in ('dataset','seed','head','status')} for r in records]
    audit={'status':'PASS_WITHOUT_F_REPLAY','failed_count':0,'checks':17,'conditions':[
        {**p,'producer_sha256':r['sha256']} for p,r in zip(producers,records)]}
    assert report.audit_gate(audit,producers,records)['passed'] is True
    audit['conditions'][0]['producer_sha256']='wrong'
    assert report.audit_gate(audit,producers,records)['passed'] is False


def test_balanced_cost_binds_all8_training_producers():
    records=report.producer_refs(queue_fixture());cost={'sequence_producers':deepcopy(records)}
    report.cost_binding(cost,records)
    cost['sequence_producers'][0]['sha256']='wrong'
    with pytest.raises(ValueError,match='ALL8_PRODUCER_SHA'):report.cost_binding(cost,records)


def test_optional_flat_summary_is_complete24_and_descriptive_only():
    records=report.producer_refs(queue_fixture())
    value={'status':'COMPLETE','source_records':records,'E_or_F_replayed':False,'causal_failure_identified':False,
        'rows':[{'dataset':r['dataset'],'seed':r['seed'],'head':r['head'],'split':split}
        for r in records for split in ('fit','tune',report.FINAL[r['dataset']])]}
    assert len(report.flat_binding(value,records))==24
    value['causal_failure_identified']=True
    with pytest.raises(ValueError,match='DESCRIPTIVE_FLAT'):report.flat_binding(value,records)


def test_paired_differences_use_same_seed_and_keep_mse_mae_disagreement():
    rows=[{'dataset':'robin','seed':92601,'split':'val','method':m,'mse':mse,'mae':mae}
        for m,mse,mae in [('RAW_BOLT',.3,.2),('STATIC_RF1',.6,.4),('TEMPORAL_RF3',.5,.5)]]
    diffs=report.paired_differences(rows)
    assert len(diffs)==2
    static=next(r for r in diffs if r['reference']=='STATIC_RF1')
    assert static['mse_difference']==pytest.approx(-.1) and static['mae_difference']==pytest.approx(.1)


def test_duplicate_original_baseline_must_have_equal_metrics():
    row={'dataset':'robin','seed':92601,'split':'val','method':'RAW_BOLT','mse':.3,'mae':.2}
    assert len(report.final_scores([row,deepcopy(row)]))==1
    other={**row,'mse':.31}
    with pytest.raises(ValueError,match='DUPLICATION_MISMATCH'):report.final_scores([row,other])


def test_report_never_loads_scientific_arrays_or_models_and_cost_is_optional_honest():
    source=inspect.getsource(report)
    assert 'np.load(' not in source and 'torch.load(' not in source
    assert '--allow-incomplete' in source and 'NOT_MEASURED' in source
    assert len(report.FIGURES)==6
