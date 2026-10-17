import importlib
import json

import numpy as np
import pytest


def impl():
    try:
        return importlib.import_module('research.adapter_improvements_20261018_v1.multik_audit')
    except ModuleNotFoundError:
        pytest.fail('Read-only multi-K audit implementation is missing')


def test_lag_major_prefix_keeps_each_lag_and_disagrees_with_flat_prefix():
    values = np.arange(7*5*3).reshape(7, 5, 3)
    actual = impl().lag_major_prefix(values.reshape(35,3), 5, 2)
    assert np.array_equal(actual, values[:, :2].reshape(14,3))
    assert not np.array_equal(actual, values.reshape(35,3)[:14])
    with pytest.raises(ValueError):
        impl().lag_major_prefix(values.reshape(35,3), 5, 6)


def test_actual_exposure_fields_are_never_invented_from_update_count():
    source = dict(spec=dict(run_id='shared', k_grid=[2,4,8], latent_k=8, dataset='x'))
    receipt = dict(status='SEGMENT_COMPLETE', selected_epoch=3, updates=99, exposures=1500,
                   k_exposures={'2':400,'4':500,'8':600}, F_training_logical_streams=7600)
    rows = impl().stage_rows(source, receipt, stage='refit', parent='shared')
    assert [r['K_origin_exposures'] for r in rows] == [400,500,600]
    assert [r['direct_K_exposure_fraction'] for r in rows] == [400/1500,500/1500,600/1500]
    assert rows[0]['actual_F_rows'] == 7600
    assert rows[0]['per_K_F_rows'] == 800
    assert rows[0]['per_K_F_rows_evidence'] == 'DERIVED_K_TIMES_LOGGED_ORIGIN_EXPOSURES'
    old = impl().stage_rows(source, dict(status='SEGMENT_COMPLETE',updates=99,exposures=1500), stage='search',parent='shared')
    assert all(r['K_origin_exposures']=='UNKNOWN' and r['actual_F_rows']=='UNKNOWN' for r in old)


def test_read_only_audit_binds_sources_and_preserves_incomplete_stages(tmp_path):
    root = tmp_path/'repo'
    broad = root/'research/informs_broad_validation_20261015_v1'
    run = broad/'campaign/runs/shared'
    (run/'search').mkdir(parents=True)
    (run/'refit').mkdir()
    (run/'SOURCE_CONTRACT.json').write_text(json.dumps({'spec':{'run_id':'shared','k_grid':[2,4],'latent_k':4}}))
    (run/'search/RECEIPT.json').write_text(json.dumps({'status':'SEGMENT_COMPLETE','updates':10,'exposures':100,'k_exposures':{'2':48,'4':52}}))
    (run/'refit/event_000001.json').write_text(json.dumps({'status':'SEGMENT_COMPLETE','updates':2,'exposures':16}))
    before = {p:p.read_bytes() for p in broad.rglob('*') if p.is_file()}
    result = impl().audit_sources(root)
    assert len(result['rows']) == 4
    refit = [r for r in result['rows'] if r['stage']=='refit']
    assert all(r['completion']=='PARTIAL_EVENT_ONLY' for r in refit)
    assert all(r['checkpoint_bytes']=='UNKNOWN' for r in result['rows'])
    assert all(not p.is_absolute() for p in map(__import__('pathlib').Path, result['source_hashes']))
    assert all(p.read_bytes()==value for p,value in before.items())


def test_existing_transition_trials_report_observed_switch_time_and_f_rows(tmp_path):
    import gzip
    root = tmp_path/'repo'
    folder = root/'research/informs_broad_validation_20261015_v1/diagnostics/publication_multik_lossless_x/bundle_x/files/beijing_B1_shared/shared'
    folder.mkdir(parents=True)
    rows = [dict(phase='timed',from_K=2,to_K=4,switch_total_seconds=t,
                 transition_plus_physical_seconds=t+.2,physical_pipeline_seconds=.2,F_streams=4,
                 numeric_prediction_valid=True) for t in (.01,.02,.03)]
    value = dict(rows=rows,strategy='shared',dataset='beijing',batch=1,status='COMPLETE_PROTOCOL',
                 storage={'actual_artifact_bytes':123}, exact_minimum_protocol_complete=False)
    with gzip.open(folder/'RESULT.json.gz','wt') as stream:
        json.dump(value,stream)
    result = impl().audit_sources(root)
    switch = [r for r in result['rows'] if r['stage']=='K_SWITCH']
    assert len(switch) == 1
    assert switch[0]['switch_time'] == .02
    assert switch[0]['actual_F_rows'] == 4
    assert switch[0]['checkpoint_bytes'] == 123
    assert switch[0]['switch_time_evidence'] == 'MEDIAN_OF_HISTORICAL_TIMED_TRIALS'
    assert not switch[0]['exact_minimum_protocol_complete']
