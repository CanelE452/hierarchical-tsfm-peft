"""NEW v2 source adapter checks; no assets/GPU/actual figures."""
from copy import deepcopy
import inspect
import pytest
from research.smallk_failure_followup_20261014 import make_joint_followup_report_v2 as r
from research.smallk_failure_followup_20261014.test_make_joint_followup_report_v1 import joint_fixture,cost_fixture


def app_fixture():
    rows=[p for p in joint_fixture() if p['dataset']=='robin']
    for p,o in zip(rows,('MSE','MAE','HUBER_DELTA1')*2):
        p.update(dataset='appliances',objective=o,original_RAW_PLUS_R_scores=deepcopy(p['original_RAW_scores']))
        p['selected_inference']['eval']=p['selected_inference'].pop('val')
        p['original_MAIN_scores']['eval']=p['original_MAIN_scores'].pop('val')
        p['original_RAW_scores']['eval']=p['original_RAW_scores'].pop('val')
        p['original_RAW_PLUS_R_scores']['eval']=p['original_RAW_PLUS_R_scores'].pop('val')
    return rows


def app_cost_fixture():
    value=cost_fixture();rows=[row for row in value['rows'] if row['dataset']=='robin']
    labels={'Y':'MSE','RAW':'MAE','MIXED':'HUBER_DELTA1',None:None}
    for row in rows:row.update(dataset='appliances',objective=labels[row['objective']],method='RAW_BOLT' if row['seed'] is None else 'APP_SELECTED')
    value.update(actual_complete_rows=14,rows=rows,RAW_PLUS_R_cost_measured=False)
    return value


def test_stable_APP_three_losses_and_stronger_RAW_R_baseline_all_preserved():
    rows,statuses=r.app_tables(app_fixture(),None)
    assert len(statuses)==6 and {s['status'] for s in statuses}=={'COMPLETE'}
    final=r.metric_groups(rows)
    assert len(final)==12 and {v['method'] for v in final}=={'MSE','MAE','HUBER_DELTA1','RAW_BOLT','RAW_PLUS_R','OLD_LEARNED'}


def test_actual_APP14_whole_cost_keeps_bulk_flags_and_RAW_R_unmeasured():
    rows=r.app_cost_tables(app_cost_fixture())
    assert len(rows)==14 and all(v['RAW_PLUS_R_cost']=='NOT_MEASURED' for v in rows)
    assert all(v['original_bulk_normalized_failed_repeats']==7 for v in rows)
    value=app_cost_fixture();value['rows'][0]['parity'][0]['matched_batch_normalized']['pass']=False
    with pytest.raises(ValueError,match='MATCHED_BATCH'):r.app_cost_tables(value)


def test_APP14_partial_or_fake_measured_RAW_R_never_count_as_actual_cost():
    value=app_cost_fixture();value['actual_complete_rows']=13
    with pytest.raises(ValueError,match='COMPLETE'):r.app_cost_tables(value)
    value=app_cost_fixture();value['RAW_PLUS_R_cost_measured']=True
    with pytest.raises(ValueError,match='COMPLETE'):r.app_cost_tables(value)


def test_v1_source_is_preserved_and_actual_generation_still_opt_in():
    source=inspect.getsource(r)
    assert 'np.load(' not in source and 'torch.load(' not in source
    assert '--no-contention-attested' in source and 'APP 신규14행 비용: **NOT_MEASURED**' in source
    assert len(r.FIGURES)==6
