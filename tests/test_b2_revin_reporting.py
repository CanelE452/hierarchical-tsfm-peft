import csv
import hashlib
import importlib
import json

import numpy as np
import pytest


def reporter():
    return importlib.import_module('research.b2_revin_20261018_v1.reporting')


def write_csv(path, rows):
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fixture_report(tmp_path):
    report = tmp_path / 'inputs'
    report.mkdir()
    runs, history, comparisons, stats, pairs = [], [], [], [], []
    for dataset in ('jena', 'pems32'):
        for cap in (3, 6):
            for seed in (92601, 92602):
                for enabled in (False, True):
                    identifier = f'{dataset}_{seed}_{enabled}_cap{cap}'
                    delta = -.1 if seed == 92601 else .2
                    mse = 1. + (delta if enabled else 0.)
                    mae = .5 + (-delta / 4 if enabled else 0.)
                    runs.append(dict(run_id=identifier, dataset=dataset, family='B2',
                        method='B2 + RevIN' if enabled else 'B2 OFF', seed=seed,
                        revin=str(enabled), cap=cap, final=str(cap == 6),
                        MSE=mse, MAE=mae, observed=8, C=2, K=1, L=4, H=2,
                        selected_epoch=cap, search_updates=cap * 2,
                        refit_updates=cap * 3, search_seconds=1., refit_seconds=2.,
                        training_status='TRAINING_CAP_STILL_IMPROVING',
                        source_kind='NEW' if enabled else 'REUSED_VERIFIED',
                        prediction_binding=json.dumps({'reference':'cache://'+identifier+'.npz','sha256':'a'*64})))
                    for epoch in (0., .25, 1., 3., 4., 5., 6.):
                        if epoch > cap:
                            continue
                        history.append(dict(run_id=identifier, dataset=dataset, family='B2',
                            seed=seed, revin=enabled, cap=cap, epoch=epoch,
                            astop_mse=1. - epoch * .02 + (delta if enabled else 0.),
                            stage='search'))
                    for channel in (0, 1):
                        for horizon in (1, 2):
                            stats.append(dict(run_id=identifier, channel_id=channel,
                                horizon=horizon, SSE=mse * 2, SAE=mae * 2, count=2))
                comparison_id = f'{dataset}_{seed}_{cap}'
                comparisons.append(dict(comparison_id=comparison_id, kind='REVIN',
                    dataset=dataset, seed=seed, cap=cap, match_status='MATCHED',
                    candidate_run_id=f'{dataset}_{seed}_True_cap{cap}',
                    control_run_id=f'{dataset}_{seed}_False_cap{cap}', n_origins=2))
                pairs.append(dict(comparison_id=comparison_id, origin=0, delta_MSE=delta))
    runs.append(dict(run_id='jena_raw', dataset='jena', family='RAW', method='Raw',
        final='True', seed='', revin='', cap='', MSE=2., MAE=.7,
        C=2, K=2, L=4, H=2, training_status='NO_TRAINING', source_kind='HISTORICAL'))
    costs = [dict(accuracy_run_id='jena_raw', dataset='jena', device='cuda',
        batch_size=batch, median_ms_per_origin=10. / batch, cost_valid='True',
        status='COMPLETE', cost_id=f'raw_gpu_{batch}') for batch in (1, 8)]
    costs.append(dict(accuracy_run_id='jena_raw', dataset='jena', device='cpu',
        batch_size=1, median_ms_per_origin=100., cost_valid='True',
        status='COMPLETE', cost_id='raw_cpu_1'))
    costs.append(dict(accuracy_run_id='jena_92601_True_cap6', dataset='jena',
        device='cuda', batch_size=8, median_ms_per_origin=3., cost_valid='False',
        status='FAILED_PARITY', cost_id='failed'))
    for name, rows in [('RUNS', runs), ('COST', costs), ('TRAINING_HISTORY', history),
                       ('COMPARISONS', comparisons), ('ERROR_SUFFICIENT_STATS', stats),
                       ('PAIR_ORIGIN_DIFFS', pairs)]:
        write_csv(report / (name + '.csv'), rows)
    for name, value in [('CAMPAIGN_CONTRACT', {'V_exposure':'previously_reused_development',
            'old_T_exposure':'DISCLOSED', 'pretraining_overlap':'UNKNOWN'}),
                       ('DECISIONS', {'datasets': {'jena': {'final_cap':6, 'revin':False},
                                                   'pems32': {'final_cap':6, 'revin':False}}})]:
        (report / (name+'.json')).write_text(json.dumps(value), encoding='utf-8')
    return report, runs, costs, comparisons, stats, history


def test_csv_booleans_are_explicit_and_missing_is_not_false():
    m = reporter()
    assert m.parse_bool('False') is False
    assert m.parse_bool('True') is True
    assert m.parse_bool('') is None
    with pytest.raises(ValueError):
        m.parse_bool('MAYBE')


def test_accuracy_cost_uses_exact_run_and_never_substitutes_raw_metric(tmp_path):
    m = reporter()
    _, runs, costs, *_ = fixture_report(tmp_path)
    table = m.accuracy_cost_rows(runs, costs)
    raw = [row for row in table if row['run_id']=='jena_raw' and row['device']=='cuda']
    assert len(raw)==2
    assert {row['MSE'] for row in raw}=={2.}
    other = [row for row in table if row['run_id']=='jena_92602_True_cap6']
    assert {row['MSE'] for row in other}=={1.2}
    assert all(row['median_ms_per_origin'] is None for row in other)
    failed = next(row for row in table if row.get('cost_id')=='failed')
    assert failed['cost_valid'] is False
    assert failed['median_ms_per_origin'] is None


def test_revin_retains_both_seed_directions_and_cap3_cap6(tmp_path):
    m = reporter()
    _, runs, _, comparisons, *_ = fixture_report(tmp_path)
    result = m.comparison_rows(runs, comparisons)
    assert len(result)==8
    assert {row['cap'] for row in result}=={3,6}
    first = next(row for row in result if row['dataset']=='jena' and row['seed']==92601)
    second = next(row for row in result if row['dataset']=='jena' and row['seed']==92602)
    assert first['delta_MSE']==pytest.approx(-.1)
    assert second['delta_MSE']==pytest.approx(.2)
    assert first['delta_MAE']==pytest.approx(.025)
    assert second['delta_MAE']==pytest.approx(-.05)


def test_count_weighted_statistics_keep_registered_macro_channels():
    m = reporter()
    rows = [dict(channel_id=0, horizon=1, SSE=2., SAE=1., count=1),
            dict(channel_id=0, horizon=2, SSE=18., SAE=9., count=9),
            dict(channel_id=1, horizon=1, SSE=9., SAE=3., count=1)]
    assert m.aggregate_statistics(rows)['MSE']==pytest.approx((2.+9.)/2)
    assert m.aggregate_statistics(rows)['MAE']==pytest.approx((1.+3.)/2)
    with pytest.raises(ValueError):
        m.aggregate_statistics(rows+[dict(channel_id=2,horizon=1,SSE=0,SAE=0,count=0)])


def test_heatmap_cell_delta_comes_from_sse_counts(tmp_path):
    m = reporter()
    _, runs, _, comparisons, stats, _ = fixture_report(tmp_path)
    paired = m.comparison_rows(runs, comparisons)
    result = m.error_change_rows(stats, paired, runs)
    assert len(result)==16
    first = next(row for row in result if row['dataset']=='jena' and row['seed']==92601)
    second = next(row for row in result if row['dataset']=='jena' and row['seed']==92602)
    assert first['delta_MSE']==pytest.approx(-.1)
    assert second['delta_MSE']==pytest.approx(.2)


def test_latency_figure_separates_cpu_gpu_and_batch(tmp_path):
    m = reporter()
    _, runs, costs, *_ = fixture_report(tmp_path)
    fig = m.accuracy_cost_figure(m.accuracy_cost_rows(runs, costs))
    titles = [axis.get_title() for axis in fig.axes]
    assert any('cuda' in title and 'B1' in title for title in titles)
    assert any('cuda' in title and 'B8' in title for title in titles)
    assert any('cpu' in title and 'B1' in title for title in titles)
    for axis in fig.axes:
        points = [p for collection in axis.collections for p in collection.get_offsets()]
        if 'cpu' in axis.get_title():
            assert all(point[0]==100 for point in points)
    m.close_figure(fig)


def test_cap_status_does_not_claim_convergence():
    assert reporter().training_note('TRAINING_CAP_STILL_IMPROVING')=='상한에서도 개선 중; 수렴 여부 미확정'
    assert '수렴 여부 미확정' in reporter().training_note('COMPLETE')


def test_public_build_is_deterministic_and_never_opens_private_arrays(tmp_path, monkeypatch):
    m = reporter()
    report, *_ = fixture_report(tmp_path)
    initial = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in report.iterdir()}
    monkeypatch.setattr(np, 'load', lambda *a, **k:pytest.fail('Public reporting opened private NPZ'))
    first, second = tmp_path/'first', tmp_path/'second'
    state1 = m.build(report, first)
    state2 = m.build(report, second)
    assert state1==state2
    assert state1['evaluation_scope']=='REPEATED_DEVELOPMENT_V'
    assert len(state1['figures'])==4
    assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in report.iterdir()}==initial
    hashes1 = {p.relative_to(first).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in first.rglob('*') if p.is_file()}
    hashes2 = {p.relative_to(second).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in second.rglob('*') if p.is_file()}
    assert hashes1==hashes2
    text = (first/'RESULT_ANALYSIS_KO.md').read_text('utf-8')
    assert 'seed92601' in text and 'seed92602' in text
    assert '독립 테스트' in text and '반복 개발' in text
    assert '수렴 여부 미확정' in text


def test_inconsistent_cost_cohort_is_rejected(tmp_path):
    m = reporter()
    _, runs, costs, *_ = fixture_report(tmp_path)
    costs[0]['dataset']='pems32'
    with pytest.raises(ValueError, match='dataset'):
        m.accuracy_cost_rows(runs,costs)


def test_composition_requires_same_revin_and_preserves_b0_result(tmp_path):
    m = reporter()
    _, runs, _, comparisons, *_ = fixture_report(tmp_path)
    control = dict(next(row for row in runs if row['run_id']=='jena_92601_False_cap6'),
                   run_id='jena_b0',family='B0',method='B0',MSE=.8,MAE=.4)
    runs.append(control)
    pair = dict(comparison_id='composition',kind='COMPOSITION',dataset='jena',seed=92601,
                cap=6,match_status='MATCHED',candidate_run_id='jena_92601_False_cap6',control_run_id='jena_b0')
    row = m.comparison_rows(runs,[pair])[0]
    assert row['delta_MSE']==pytest.approx(.2)
    assert row['control_MSE']==.8
    control['revin']='True'
    with pytest.raises(ValueError,match='normalization'):
        m.comparison_rows(runs,[pair])


def test_wrong_published_delta_and_missing_final_error_cells_fail(tmp_path):
    m = reporter()
    _, runs, _, comparisons, stats, _ = fixture_report(tmp_path)
    comparisons[0]['delta_MSE']=.99
    with pytest.raises(ValueError,match='referenced metric'):
        m.comparison_rows(runs,comparisons)
    comparisons[0].pop('delta_MSE')
    paired = m.comparison_rows(runs,comparisons)
    stats = [row for row in stats if not (row['run_id']=='jena_92601_True_cap6' and row['horizon']==1)]
    with pytest.raises(ValueError,match='coordinates'):
        m.error_change_rows(stats,paired,runs)


def test_build_rejects_incomplete_revin_seed_coverage_before_outputs(tmp_path):
    m = reporter()
    report, _, _, comparisons, *_ = fixture_report(tmp_path)
    write_csv(report/'COMPARISONS.csv',[row for row in comparisons if row['seed']!=92602])
    output = tmp_path/'incomplete'
    with pytest.raises(ValueError,match='both seeds'):
        m.build(report,output)
    assert not output.exists()


def test_shared_missing_channel_is_not_silently_dropped(tmp_path):
    m = reporter()
    _, runs, _, comparisons, stats, _ = fixture_report(tmp_path)
    stats = [row for row in stats if row['channel_id']!=1]
    with pytest.raises(ValueError,match='registered channel'):
        m.error_change_rows(stats,m.comparison_rows(runs,comparisons),runs)


def test_root_schema_without_comparison_id_counts_origins_once(tmp_path):
    m = reporter()
    report, runs, _, comparisons, *_ = fixture_report(tmp_path)
    for row in runs:
        if 'observed' in row:
            row['n_targets']=row.pop('observed')
    pairs = []
    for row in comparisons:
        row.pop('comparison_id')
        for origin in (11,23):
            for channel in (0,1):
                pairs.append(dict(kind=row['kind'],dataset=row['dataset'],seed=row['seed'],
                                  cap=row['cap'],origin=origin,channel_id=channel,SSEdiff=1.,SAEdiff=1.,count=2))
    write_csv(report/'RUNS.csv',runs)
    write_csv(report/'COMPARISONS.csv',comparisons)
    write_csv(report/'PAIR_ORIGIN_DIFFS.csv',pairs)
    output = tmp_path/'root_schema'
    m.build(report,output)
    with (output/'T2_REVIN_MATCHED.csv').open(encoding='utf-8',newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows)==8
    assert {row['paired_origin_rows'] for row in rows}=={'2'}
    assert {row['paired_stat_rows'] for row in rows}=={'4'}


def test_documents_preserve_reused_artifact_path_and_sha(tmp_path):
    m = reporter()
    _, runs, costs, comparisons, *_ = fixture_report(tmp_path)
    contract = dict(reused_public_artifacts=[dict(reference='repo://reports/old/figures/baseline.svg',
                    sha256='1'*64,scope='Historical measured baseline context')])
    documents = m._documents(runs,m.accuracy_cost_rows(runs,costs),
        m.comparison_rows(runs,comparisons),contract,{}, {})
    assert 'repo://reports/old/figures/baseline.svg' in documents['REPORT_METHOD_NOTES_KO.md']
    assert '1'*64 in documents['REPORT_METHOD_NOTES_KO.md']


def test_optional_historical_metadata_is_bound_and_root_methods_remain_untouched(tmp_path):
    m = reporter()
    report, *_ = fixture_report(tmp_path)
    historical = dict(artifacts=[dict(reference='repo://reports/old/figures/baseline.svg',
                                     sha256='2'*64,scope='Historical baseline cost context')])
    metadata = report/'HISTORICAL_ARTIFACTS.json'
    metadata.write_text(json.dumps(historical),encoding='utf-8')
    official_notes = report/'METHOD_REFERENCES.md'
    official_notes.write_text('Root-owned pinned official reference',encoding='utf-8')
    state = m.build(report)
    assert official_notes.read_text('utf-8')=='Root-owned pinned official reference'
    assert state['input_hashes']['HISTORICAL_ARTIFACTS.json']==hashlib.sha256(metadata.read_bytes()).hexdigest()
    notes = (report/'REPORT_METHOD_NOTES_KO.md').read_text('utf-8')
    assert '2'*64 in notes and 'reports/old/figures/baseline.svg' in notes
    assert '../../reports/old/figures/baseline.svg' in (report/'README_KO.md').read_text('utf-8')


def test_error_figure_keeps_revin_and_composition_same_seed_cap_separate():
    m = reporter()
    rows = []
    for kind, values, normalization in [('REVIN',(-.1,.3),(True,False)),
                                        ('COMPOSITION',(.2,-.4),(False,False))]:
        for horizon, delta in enumerate(values,1):
            rows.append(dict(kind=kind,comparison_id=kind+':jena:92601:3',dataset='jena',
                seed=92601,cap=3,channel_id='pressure',horizon=horizon,count=4,
                candidate_revin=normalization[0],control_revin=normalization[1],delta_MSE=delta))
    fig = m.error_change_figure(rows)
    try:
        axes = [axis for axis in fig.axes if axis.images]
        assert len(axes)==2
        revin = next(axis for axis in axes if 'REVIN ON − OFF' in axis.get_title())
        composition = next(axis for axis in axes if 'COMPOSITION B2 − B0' in axis.get_title())
        np.testing.assert_array_equal(revin.images[0].get_array(),[[-.1,.3]])
        np.testing.assert_array_equal(composition.images[0].get_array(),[[.2,-.4]])
        assert 'RevIN OFF' in composition.get_title()
        assert 'cap 3' in revin.get_title() and 'cap 3' in composition.get_title()
    finally:
        m.close_figure(fig)


def test_error_figure_rejects_duplicate_comparison_cell():
    row = dict(kind='REVIN',comparison_id='r',dataset='jena',seed=92601,cap=3,
               channel_id='pressure',horizon=1,count=4,candidate_revin=True,control_revin=False,delta_MSE=.1)
    with pytest.raises(ValueError,match='Duplicate comparison/channel/horizon'):
        reporter().error_change_figure([row,dict(row,delta_MSE=.2)])


def test_error_source_rows_preserve_composition_identity_and_normalization(tmp_path):
    m = reporter()
    _, runs, _, comparisons, stats, _ = fixture_report(tmp_path)
    b0 = dict(next(row for row in runs if row['run_id']=='jena_92601_False_cap6'),
              run_id='jena_b0',family='B0',method='B0',MSE=.8,MAE=.4)
    runs.append(b0)
    stats.extend(dict(row,run_id='jena_b0',SSE=1.6,SAE=.8)
                 for row in list(stats) if row['run_id']=='jena_92601_False_cap6')
    comparisons.append(dict(comparison_id='composition',kind='COMPOSITION',dataset='jena',seed=92601,
        cap=6,match_status='MATCHED',candidate_run_id='jena_92601_False_cap6',control_run_id='jena_b0'))
    rows = m.error_change_rows(stats,m.comparison_rows(runs,comparisons),runs)
    composed = [row for row in rows if row.get('kind')=='COMPOSITION']
    assert len(composed)==4
    assert {row['comparison_id'] for row in composed}=={'composition'}
    assert all(row['candidate_revin'] is False and row['control_revin'] is False for row in composed)
    assert all(row['delta_MSE']==pytest.approx(.2) for row in composed)


def test_native_raw_ridge_keeps_two_cpu_rows_without_false_cuda_gaps():
    m = reporter()
    run = dict(run_id='BASELINE_JENA_RIDGE',dataset='jena',family='RAW_RIDGE',method='Ridge',
               seed=92601,revin='',cap=0,final=True,MSE=.556,MAE=.405,C=21,K=21,L=512,H=48,
               source_kind='HISTORICAL',training_status='CLOSED_FORM',core_dtype='float64',prediction_dtype='float32')
    costs = [dict(accuracy_run_id=run['run_id'],dataset='jena',device='cpu',batch_size=batch,
                  cost_valid=True,status='COMPLETE',median_ms_per_origin=latency,cost_id=f'cpu{batch}',core_dtype='float64')
             for batch,latency in ((1,.3),(8,.07))]
    rows = m.accuracy_cost_rows([run],costs)
    assert len(rows)==2
    assert {row['device'] for row in rows}=={'cpu'}
    assert sum(row['cost_valid'] is not True for row in rows)==0
    assert {row['MSE'] for row in rows}=={.556}
    assert all(row['core_dtype']=='float64' and row['prediction_dtype']=='float32' for row in rows)


def test_matched_pairs_reject_declared_cap_different_from_actual_run_budget(tmp_path):
    m = reporter()
    _, runs, _, comparisons, *_ = fixture_report(tmp_path)
    b0 = dict(next(row for row in runs if row['run_id']=='jena_92601_False_cap3'),
              run_id='jena_b0_cap3',family='B0',method='B0')
    runs.append(b0)
    revin = next(row for row in comparisons if row['dataset']=='jena' and row['seed']==92601 and row['cap']==3)
    composition = dict(comparison_id='composition',kind='COMPOSITION',dataset='jena',seed=92601,
                       cap=3,match_status='MATCHED',candidate_run_id='jena_92601_False_cap3',control_run_id='jena_b0_cap3')
    for pair in (revin,composition):
        with pytest.raises(ValueError,match='cap'):
            m.comparison_rows(runs,[dict(pair,cap=6)])
        with pytest.raises(ValueError,match='cap'):
            m.comparison_rows(runs,[dict(pair,cap='')])


def test_cap6_status_is_explicit_in_generated_analysis_and_legacy_label(tmp_path):
    m = reporter()
    _, runs, costs, comparisons, *_ = fixture_report(tmp_path)
    run = next(row for row in runs if row['run_id']=='jena_92601_True_cap6')
    for status in ('TRAINING_CAP_STILL_IMPROVING','CAP_OR_UNDERTRAINING'):
        note = m.training_note(status)
        assert '상한' in note and '개선' in note and '수렴 여부 미확정' in note
        assert '방법 실패' not in note and '수렴 완료' not in note
        run['training_status']=status
        documents = m._documents(runs,m.accuracy_cost_rows(runs,costs),
            m.comparison_rows(runs,comparisons),{}, {}, {})
        text = documents['RESULT_ANALYSIS_KO.md']
        assert m._label(m._run(run))+': '+note in text
