import hashlib
import importlib
import json
from pathlib import Path
import time

import numpy as np
import pytest

from research.adapter_improvements_20261018_v1 import contracts, run


def impl():
    try:
        return importlib.import_module('research.adapter_improvements_20261018_v1.verification')
    except ModuleNotFoundError:
        pytest.fail('Independent campaign verifier is not implemented')


def write(path, value):
    contracts.atomic_json(path, value)


def expected_statistics(pred, target, mask):
    n, h, c = pred.shape
    sse = np.zeros((h, c))
    sae = np.zeros((h, c))
    count = np.zeros((h, c), np.int64)
    for i in range(n):
        for j in range(h):
            for k in range(c):
                if mask[i, j, k]:
                    error = float(pred[i, j, k])-float(target[i, j, k])
                    sse[j, k] += error*error
                    sae[j, k] += abs(error)
                    count[j, k] += 1
    return dict(MSE=float(np.mean(sse.sum(0)/count.sum(0))),
                MAE=float(np.mean(sae.sum(0)/count.sum(0))),
                channel_sse=sse.sum(0).tolist(), channel_sae=sae.sum(0).tolist(),
                channel_counts=count.sum(0).tolist(), horizon_sse=sse.tolist(),
                horizon_sae=sae.tolist(), horizon_counts=count.tolist())


@pytest.fixture
def campaign(tmp_path):
    ctx = dict(repo=tmp_path, report=tmp_path/'report', cache=tmp_path/'cache',
               previous_cache=tmp_path/'old', asset=tmp_path/'assets')
    for key in ('report', 'cache', 'previous_cache', 'asset'):
        ctx[key].mkdir()
    science = tmp_path/'science.py'
    science.write_text('pinned scientific producer\n', encoding='utf-8')
    protected = tmp_path/'historical.txt'
    protected.write_text('sealed parent\n', encoding='utf-8')
    inventory = {'historical.txt': contracts.sha(protected)}
    write(ctx['cache']/'PROTECTED_PARENT_FILES.json', inventory)
    start = time.time()-20
    ledger = dict(campaign=contracts.NAME, started_unix=start, limits=contracts.DEFAULT_LIMITS,
                  lineage=[], gpu_seconds=2., cpu_worker_seconds=1., process_cpu_seconds=1.,
                  download_reserved_bytes=0)
    write(ctx['cache']/'CAMPAIGN.json', ledger)
    (ctx['cache']/'GPU_USAGE.jsonl').write_text(json.dumps(dict(lease_id='one', charged_process_seconds=2.))+'\n')
    write(ctx['report']/'START_STATE.json', dict(new_campaign_start_unix=start,
          protected_parent_inventory_sha256=contracts.sha(ctx['cache']/'PROTECTED_PARENT_FILES.json'),
          protected_historical_file_count=1, limits=contracts.DEFAULT_LIMITS))
    write(ctx['report']/'REUSE_AUDIT.json', {'status': 'PASS'})
    sources = {'science.py': contracts.sha(science)}
    job = contracts.seal_job(dict(job_id='P_SPLIT_s92601', kind='training', family='P_SPLIT',
                            seed=92601, cohort='synthetic', K=1, device='cpu', dependency_ids=[]), sources)
    pending = contracts.seal_job(dict(job_id='DCA_optional', kind='initialization',
                                dependency_ids=[], conditional=True), sources)
    plan = contracts.seal_plan(dict(campaign=contracts.NAME, campaign_started_unix=start,
              source_hashes=sources, execution_control_sha256=contracts.sha(run.__file__),
              reuse_audit_sha256=contracts.sha(ctx['report']/'REUSE_AUDIT.json'),
              limits=contracts.DEFAULT_LIMITS, jobs=[job, pending],
              numerical_tolerances={'FP32_route': {'atol': 1e-5, 'rtol': 1e-4}}))
    write(ctx['report']/'EXPERIMENT_PLAN.json', plan)
    pred = np.arange(24, dtype=np.float32).reshape(4, 3, 2)/8
    target = np.ones_like(pred)
    mask = np.ones_like(pred, bool)
    mask[:, :1, 0] = False
    arrays = dict(pred_normalized=pred, target_normalized=target, observed=mask,
                  origins=np.array([20, 23, 26, 29], np.int64), channel_ids=np.array(['a', 'b']),
                  mean=np.zeros(2, np.float32), std=np.ones(2, np.float32))
    archive = ctx['cache']/'predictions.npz'
    np.savez(archive, **arrays)
    row = dict(expected_statistics(pred, target, mask), method='P_SPLIT_PCA', run_id=job['job_id'],
               seed=92601, dataset='synthetic', stage='selected_refit', evaluation='V_select', fit='fresh_A60',
               C=2, H=3, L=10, K=1, n_origins=4, selected_epoch=1.,
               selection_split='A_stop', V_exposure='previously_reused_development',
               prediction_binding=contracts.binding(archive, ctx), scaler_sha256=None,
               array_fingerprints={key: dict(shape=list(value.shape), dtype=str(value.dtype),
                   sha256=hashlib.sha256(value.tobytes()).hexdigest()) for key, value in arrays.items()},
               metric_accumulation_dtype='float64', prediction_dtype='float32', logical_F_rows=4,
               source_sha='source', model_sha='F', units='normalized_A60')
    result = dict(status='COMPLETE', rows=[row], selection_sealed_before_V=True,
                  recipe={'selection': 'A_fit/A_stop then fresh A60'}, model_sha='F',
                  fit_receipt={'search_fit_end': 10, 'refit_fit_end': 15},
                  private_artifacts=[contracts.binding(archive, ctx)])
    public = ctx['report']/'jobs'/(job['job_id']+'.json')
    write(public, result)
    ctx['ledger'] = ledger
    ctx['_test_job'] = job
    ctx['_test_arrays'] = arrays
    ctx['_test_result'] = result
    ctx['_test_path'] = public
    ctx['_test_plan'] = plan
    refresh(ctx)
    return ctx


def refresh(ctx):
    write(ctx['_test_path'], ctx['_test_result'])
    contracts.finish_job(ctx['report']/'receipts'/(ctx['_test_job']['job_id']+'.json'),
                         ctx['_test_job'], dict(status='COMPLETE', result=ctx['_test_result'],
                         artifact_paths=[str(ctx['_test_path'])]), dict(ctx, plan=ctx['_test_plan']))


def test_saved_arrays_independent_macro_stats_bands_and_canonical_fingerprints(campaign):
    m = impl()
    value = m.verify(campaign, write=False)
    assert value['status'] == 'PASS', value['failures']
    assert value['scientific_status'] == 'PARTIAL'
    assert value['coverage']['pending_job_ids'] == ['DCA_optional']
    proof, = value['verified_prediction_archives']
    assert proof['source_job_id'] == 'P_SPLIT_s92601'
    assert proof['validation_status'] == 'PASS'
    assert proof['canonical_cohort_signature']['mask']['dtype'] == 'bool'
    assert proof['statistics']['bands'][0]['status'] == 'UNDEFINED'
    assert proof['statistics']['bands'][0]['coverage_channels'] == 1
    assert proof['statistics']['MSE'] == campaign['_test_result']['rows'][0]['MSE']
    assert not (campaign['report']/'VALIDATION_REPORT.json').exists()
    m.verify(campaign)
    saved = contracts.read(campaign['report']/'VALIDATION_REPORT.json')
    assert saved['publication_status'] == 'UNVERIFIED'
    assert str(campaign['cache']) not in json.dumps(saved)


@pytest.mark.parametrize('kind', ['metric', 'fingerprint', 'origin_order', 'mask', 'sha', 'empty_channel'])
def test_prediction_mutations_never_fall_back_to_reported_green(campaign, kind):
    row = campaign['_test_result']['rows'][0]
    arrays = campaign['_test_arrays']
    if kind == 'metric':
        row['MSE'] += .1
    elif kind == 'fingerprint':
        row['array_fingerprints']['observed']['sha256'] = '0'*64
    elif kind == 'sha':
        Path(campaign['cache']/'predictions.npz').write_bytes(b'changed')
    else:
        arrays = dict(arrays)
        if kind == 'origin_order':
            arrays['origins'] = arrays['origins'][::-1]
        else:
            arrays['observed'] = arrays['observed'].copy()
            arrays['observed'][0, 1, 0] = False
            if kind == 'empty_channel':
                arrays['observed'][..., 1] = False
        np.savez(campaign['cache']/'predictions.npz', **arrays)
        row['prediction_binding'] = contracts.binding(campaign['cache']/'predictions.npz', campaign)
        campaign['_test_result']['private_artifacts'] = [row['prediction_binding']]
    refresh(campaign)
    value = impl().verify(campaign, write=False)
    assert value['status'] == 'FAIL'
    assert value['failures']
    assert not value['verified_prediction_archives']


def test_complete_without_predictions_is_not_a_verified_scored_result(campaign):
    campaign['_test_result']['rows'][0].pop('prediction_binding')
    refresh(campaign)
    assert impl().verify(campaign, write=False)['status'] == 'FAIL'


def test_parent_source_and_resume_ledger_guards_are_real(campaign):
    (campaign['repo']/'historical.txt').write_text('tampered')
    assert impl().verify(campaign, write=False)['status'] == 'FAIL'
    (campaign['repo']/'science.py').write_text('changed')
    value = impl().verify(campaign, write=False)
    assert sum(check['status'] == 'FAIL' for check in value['checks']) >= 2


def test_false_receipt_status_and_identity_are_rejected(campaign):
    receipt_path = campaign['report']/'receipts'/'P_SPLIT_s92601.json'
    receipt = contracts.read(receipt_path)
    receipt['scientific_identity'] = 'other'
    write(receipt_path, receipt)
    assert impl().verify(campaign, write=False)['status'] == 'FAIL'
    receipt['scientific_identity'] = campaign['_test_job']['scientific_identity']
    receipt['status'] = 'REUSED_VERIFIED'
    write(receipt_path, receipt)
    assert impl().verify(campaign, write=False)['status'] == 'FAIL'


def test_missing_or_nonconverged_job_keeps_partial_separate_from_validity(campaign):
    pending = campaign['_test_plan']['jobs'][1]
    contracts.finish_job(campaign['report']/'receipts'/(pending['job_id']+'.json'), pending,
                         dict(status='NOT_CONVERGED', reason='registered solver limit', artifact_paths=[]),
                         dict(campaign, plan=campaign['_test_plan']))
    value = impl().verify(campaign, write=False)
    assert value['status'] == 'PASS'
    assert value['scientific_status'] == 'PARTIAL'
    assert value['coverage']['statuses']['NOT_CONVERGED'] == 1


def test_invalid_cost_cannot_be_ranked_or_join_another_accuracy_archive(campaign):
    row = dict(method='P_SPLIT_PCA', cost_valid=True, status='COMPLETE',
               accuracy_job_id='another_job', accuracy_run_id='another_job',
               accuracy_prediction_binding=campaign['_test_result']['rows'][0]['prediction_binding'],
               device='cpu', batch_size=1, warmups=10, timed_trials=20,
               saved_V_parity={'status': 'PASS', 'n_origins': 4, 'atol': 1e-5, 'rtol': 1e-4},
               serialized_reload_saved_V_parity={'status': 'PASS', 'n_origins': 4},
               serialized_reload_parity={'status': 'PASS', 'n_origins': 4},
               F_rows_per_timed_batch=[1], backbone_dtype='float32')
    cost_job = contracts.seal_job(dict(job_id='COST', kind='costs', dependency_ids=[]),
                                 campaign['_test_plan']['source_hashes'])
    campaign['_test_plan']['jobs'].append(cost_job)
    campaign['_test_plan'] = contracts.seal_plan(campaign['_test_plan'])
    write(campaign['report']/'EXPERIMENT_PLAN.json', campaign['_test_plan'])
    result = {'status': 'COMPLETE', 'family': 'WHOLE_PATH_COST', 'rows': [row]}
    path = campaign['report']/'jobs'/'COST.json'
    write(path, result)
    contracts.finish_job(campaign['report']/'receipts'/'COST.json', cost_job,
                         dict(status='COMPLETE', result=result, artifact_paths=[str(path)]),
                         dict(campaign, plan=campaign['_test_plan']))
    refresh(campaign)
    value = impl().verify(campaign, write=False)
    assert value['status'] == 'FAIL'
    assert any(check['id'].startswith('cost:') and check['status'] == 'FAIL' for check in value['checks'])


def test_forged_cohort_with_self_consistent_metrics_and_fingerprints_is_rejected(campaign):
    import copy
    first = campaign['_test_job']
    second = contracts.seal_job(dict(job_id='P_CONTROL_s92601', kind='training', family='P_LOCAL_CONTROL',
                        seed=92601, cohort='synthetic', K=1, device='cpu', dependency_ids=[]),
                        campaign['_test_plan']['source_hashes'])
    campaign['_test_plan']['jobs'].append(second)
    campaign['_test_plan'] = contracts.seal_plan(campaign['_test_plan'])
    write(campaign['report']/'EXPERIMENT_PLAN.json', campaign['_test_plan'])
    result = copy.deepcopy(campaign['_test_result'])
    arrays = dict(campaign['_test_arrays'])
    arrays['target_normalized'] = arrays['target_normalized']+.5
    archive = campaign['cache']/'second.npz'
    np.savez(archive, **arrays)
    row = result['rows'][0]
    row.update(expected_statistics(arrays['pred_normalized'], arrays['target_normalized'], arrays['observed']))
    row['prediction_binding'] = contracts.binding(archive, campaign)
    row['array_fingerprints'] = {name: dict(shape=list(a.shape), dtype=str(a.dtype),
                             sha256=hashlib.sha256(a.tobytes()).hexdigest()) for name, a in arrays.items()}
    result['private_artifacts'] = [row['prediction_binding']]
    path = campaign['report']/'jobs'/(second['job_id']+'.json')
    write(path, result)
    contracts.finish_job(campaign['report']/'receipts'/(second['job_id']+'.json'), second,
              dict(status='COMPLETE', result=result, artifact_paths=[str(path)]), dict(campaign, plan=campaign['_test_plan']))
    refresh(campaign)
    value = impl().verify(campaign, write=False)
    assert value['status'] == 'FAIL'
    assert any(c['id'].startswith('cohort:') and c['status'] == 'FAIL' for c in value['checks'])


def test_public_run_and_channel_horizon_mutations_are_independently_checked(campaign):
    import csv
    proof, = impl().verify(campaign, write=False)['verified_prediction_archives']
    full_stats = impl().independent_statistics(campaign['_test_arrays']['pred_normalized'],
                           campaign['_test_arrays']['target_normalized'], campaign['_test_arrays']['observed'])
    row = dict(row_id='one', job_id=proof['source_job_id'], MSE=proof['statistics']['MSE'],
               MAE=proof['statistics']['MAE'], C=2, H=3,
               prediction_binding=json.dumps(campaign['_test_result']['rows'][0]['prediction_binding']))
    def save_csv(path, values):
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
    save_csv(campaign['report']/'RUNS.csv', [row])
    stats = [dict(row_id='one', horizon=h+1, channel_id=c,
              SSE=full_stats['horizon_sse'][h][c], SAE=full_stats['horizon_sae'][h][c],
              count=full_stats['horizon_counts'][h][c]) for h in range(3) for c in range(2)]
    save_csv(campaign['report']/'CHANNEL_HORIZON_STATS.csv', stats)
    assert impl().verify(campaign, write=False)['status'] == 'PASS'
    stats[-1]['count'] = 0
    save_csv(campaign['report']/'CHANNEL_HORIZON_STATS.csv', stats)
    assert impl().verify(campaign, write=False)['status'] == 'FAIL'
    row['MSE'] += .1
    save_csv(campaign['report']/'RUNS.csv', [row])
    assert impl().verify(campaign, write=False)['status'] == 'FAIL'


def test_compact_contextual_statistics_use_joined_runs_origin_without_weakening_new_proofs(campaign):
    import csv
    report = campaign['report']
    run_row = dict(row_id='legacy', job_id='old', record_origin='CONTEXTUAL', MSE=1., MAE=1.)
    statistic = dict(row_id='legacy', horizon=1, channel_id=0, SSE=3., SAE=3., count=3)
    def save(path, row):
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader(); writer.writerow(row)
    save(report/'RUNS.csv', run_row)
    save(report/'CHANNEL_HORIZON_STATS.csv', statistic)
    assert impl()._public_tables(campaign, [], [])['rows'] == 1
    run_row['record_origin'] = 'NEW'
    save(report/'RUNS.csv', run_row)
    with pytest.raises(ValueError, match='statistics_without_saved_prediction_proof'):
        impl()._public_tables(campaign, [], [])
    run_row['record_origin'] = 'CONTEXTUAL'
    statistic['record_origin'] = 'NEW'
    save(report/'RUNS.csv', run_row)
    save(report/'CHANNEL_HORIZON_STATS.csv', statistic)
    with pytest.raises(ValueError, match='statistics_record_origin_mismatch'):
        impl()._public_tables(campaign, [], [])
    statistic.pop('record_origin')
    run_row['prediction_binding'] = json.dumps(campaign['_test_result']['rows'][0]['prediction_binding'])
    save(report/'RUNS.csv', run_row)
    save(report/'CHANNEL_HORIZON_STATS.csv', statistic)
    with pytest.raises(ValueError, match='RUNS_scored_archive_not_independently_verified'):
        impl()._public_tables(campaign, [], [])


def test_resource_double_charge_and_ledger_restart_are_not_green(campaign):
    journal = campaign['cache']/'GPU_USAGE.jsonl'
    journal.write_text(journal.read_text()*2)
    assert impl().verify(campaign, write=False)['status'] == 'FAIL'
    ledger = contracts.read(campaign['cache']/'CAMPAIGN.json')
    ledger['started_unix'] += 1
    write(campaign['cache']/'CAMPAIGN.json', ledger)
    result = impl().verify(campaign, write=False)
    assert any(c['id'] == 'plan:scientific_controller_and_resume_guards' and c['status'] == 'FAIL' for c in result['checks'])


def test_figure_available_schema_requires_actual_files_source_hash_and_count(campaign):
    report = campaign['report']
    (report/'F1.png').write_bytes(b'\x89PNG\r\n\x1a\nsynthetic_container')
    (report/'F1.svg').write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>')
    (report/'F1.csv').write_text('cohort,method,value\nsynthetic,P_SPLIT,1.0\n')
    index = dict(figures=[dict(figure_id='F1', png='F1.png', svg='F1.svg', source_csv='F1.csv',
                         source_sha256=contracts.sha(report/'F1.csv'), n_rows=1, status='AVAILABLE')])
    write(report/'FIGURE_INDEX.json', index)
    value = impl().verify(campaign, write=False)
    assert value['status'] == 'PASS'
    assert any(g['id'] == 'public_visual_inspection' for g in value['unverified_gates'])
    (report/'F1.csv').write_text('cohort,method,value\nsynthetic,P_SPLIT,2.0\n')
    assert impl().verify(campaign, write=False)['status'] == 'FAIL'


def test_supplemental_plans_preserve_each_receipts_own_plan_identity(campaign):
    job = contracts.seal_job(dict(job_id='COST_SUPPLEMENT', kind='profile', dependency_ids=[]),
                             campaign['_test_plan']['source_hashes'])
    supplemental = contracts.seal_plan(dict(campaign['_test_plan'], jobs=[job]))
    write(campaign['report']/'SUPPLEMENTAL_PLANS'/'costs.json', supplemental)
    path = campaign['report']/'profiles'/'supplement.json'
    write(path, {'profile': 'CPU synthetic'})
    receipt_path = campaign['report']/'receipts'/(job['job_id']+'.json')
    contracts.finish_job(receipt_path, job,
                         dict(status='COMPLETE', result={'status': 'COMPLETE'}, artifact_paths=[str(path)]),
                         dict(campaign, plan=supplemental))
    value = impl().verify(campaign, write=False)
    assert value['status'] == 'PASS', value['failures']
    assert value['coverage']['registered_jobs'] == 3
    assert value['supplemental_plan_sha256'] == [supplemental['plan_sha256']]
    receipt = contracts.read(receipt_path)
    receipt['plan_sha256'] = campaign['_test_plan']['plan_sha256']
    write(receipt_path, receipt)
    value = impl().verify(campaign, write=False)
    assert value['status'] == 'FAIL'
    assert any('receipt_plan_identity_mismatch' in c.get('reason', '') for c in value['failures'])


def test_manifest_checks_ordinary_public_bytes_and_rejects_self_cycle(campaign):
    file = campaign['report']/'public.csv'
    file.write_text('a,b\n1,2\n')
    manifest = dict(schema='public-v1', files=[dict(path=file.relative_to(campaign['repo']).as_posix(),
                       sha256=contracts.sha(file), bytes=file.stat().st_size)],
                    exclusions=['ARTIFACT_MANIFEST.json', 'VALIDATION_REPORT.json'])
    path = campaign['report']/'ARTIFACT_MANIFEST.json'
    write(path, manifest)
    assert impl().verify(campaign, write=False)['status'] == 'PASS'
    file.write_text('a,b\n9,2\n')
    assert impl().verify(campaign, write=False)['status'] == 'FAIL'
    manifest['files'] = [dict(path=path.relative_to(campaign['repo']).as_posix(),
                              sha256=contracts.sha(path), bytes=path.stat().st_size)]
    write(path, manifest)
    assert impl().verify(campaign, write=False)['status'] == 'FAIL'


def test_cost_full_V_exact_run_dtype_and_calibrator_join(campaign):
    m = impl()
    proof, = m.verify(campaign, write=False)['verified_prediction_archives']
    row = dict(method=proof['method'], cost_valid=True, status='COMPLETE',
               accuracy_job_id=proof['source_job_id'], accuracy_run_id=proof['source_run_id'],
               accuracy_prediction_binding=campaign['_test_result']['rows'][0]['prediction_binding'],
               device='cpu', batch_size=1, warmups=10, timed_trials=20, C=2, K=1,
               saved_V_parity={'status': 'PASS', 'n_origins': 4, 'atol': 1e-5, 'rtol': 1e-4},
               serialized_reload_saved_V_parity={'status': 'PASS', 'n_origins': 4},
               serialized_reload_parity={'status': 'PASS', 'n_origins': 4},
               F_rows_per_timed_batch=[1], backbone_dtype='float32', head_dtype='float32', prediction_dtype='float32',
               cold_observations=[{'scope': 'load1'}, {'scope': 'load2'}], whole_path='physical_to_CPU')
    assert m._cost(row, [proof], campaign)['status'] == 'PASS'
    trials = [dict(accuracy_run_id=proof['source_run_id'], batch_size='1', phase='warmup',
                   whole_seconds='.01', F_logical_streams='1') for _ in range(10)]
    trials += [dict(accuracy_run_id=proof['source_run_id'], batch_size='1', phase='timed',
                    whole_seconds=str(.01+i*.001), F_logical_streams='1') for i in range(20)]
    q25, median, q75, p95 = np.quantile(np.arange(10., 30.), [.25, .5, .75, .95])
    measured = dict(row, _source_trials=trials, median_batch_ms=median, IQR_batch_ms=q75-q25,
                    p95_batch_ms=p95, median_ms_per_origin=median, IQR_ms_per_origin=q75-q25,
                    p95_ms_per_origin=p95)
    assert m._cost(measured, [proof], campaign)['status'] == 'PASS'
    with pytest.raises(ValueError, match='raw_cost_median_batch_ms_mismatch'):
        m._cost(dict(measured, median_batch_ms=median+1), [proof], campaign)
    wrong = dict(row, prediction_dtype='float64')
    with pytest.raises(ValueError, match='dtype'):
        m._cost(wrong, [proof], campaign)
    wrong = dict(row, serialized_reload_saved_V_parity={'status': 'PASS', 'n_origins': 1})
    with pytest.raises(ValueError, match='full_saved_V'):
        m._cost(wrong, [proof], campaign)
    proof = dict(proof, calibrator_binding={'reference': 'cache://one.pt', 'sha256': '1'*64})
    wrong = dict(row, accuracy_refiner_binding={'reference': 'cache://two.pt', 'sha256': '2'*64})
    with pytest.raises(ValueError, match='refiner_binding_join'):
        m._cost(wrong, [proof], campaign)


def test_public_raw_display_alias_preserves_exact_producer_cost_identity(campaign):
    import csv
    report = campaign['report']
    (report/'RUNS.csv').write_text('row_id,job_id,MSE,MAE\n')
    producer = dict(_source_job_id='COST_JENA_CPU', accuracy_job_id='HISTORICAL_RAW_V', method='RAW', device='cpu', batch_size=1,
                    cost_valid=True, _verification_status='PASS', median_ms_per_origin=10.,
                    p95_ms_per_origin=11., median_batch_ms=10.)
    public = dict(producer, method='Raw', job_id='COST_JENA_CPU')
    public.pop('_source_job_id')
    public.pop('_verification_status')
    path = report/'COST.csv'
    def save():
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(public))
            writer.writeheader(); writer.writerow(public)
    save()
    assert impl()._public_tables(campaign, [], [producer])['cost_table_checked']
    public['method'] = 'B2'
    save()
    with pytest.raises(ValueError, match='ranked_COST_row'):
        impl()._public_tables(campaign, [], [producer])


def test_repeated_raw_accuracy_costs_join_their_own_measurement_jobs(campaign):
    import csv
    report = campaign['report']
    (report/'RUNS.csv').write_text('row_id,job_id,MSE,MAE\n')
    original = dict(_source_job_id='COST_JENA_CPU', accuracy_job_id='HISTORICAL_RAW_V',
                    method='RAW', device='cpu', batch_size=1, cost_valid=True, _verification_status='PASS',
                    median_ms_per_origin=10., p95_ms_per_origin=11., median_batch_ms=10.)
    supplemental = dict(original, _source_job_id='COST_JENA_CPU_s92602',
                        median_ms_per_origin=20., p95_ms_per_origin=21., median_batch_ms=20.)
    public = []
    for producer in (original, supplemental):
        row = dict(producer, job_id=producer['_source_job_id'], method='Raw')
        row.pop('_source_job_id'); row.pop('_verification_status')
        public.append(row)
    path = report/'COST.csv'
    def save():
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(public[0]))
            writer.writeheader(); writer.writerows(public)
    save()
    assert impl()._public_tables(campaign, [], [original, supplemental])['cost_table_checked']
    public[0]['job_id'] = 'COST_OTHER_MEASUREMENT'
    save()
    with pytest.raises(ValueError, match='ranked_COST_row'):
        impl()._public_tables(campaign, [], [original, supplemental])


def paired_case(campaign, missing_channel=False):
    import csv
    m = impl()
    report = campaign['report']
    arrays = {key: value.copy() for key, value in campaign['_test_arrays'].items()}
    if missing_channel:
        arrays['observed'][0, :, 0] = False
    raw_rows, run_rows, proofs = [], [], []
    for index, name in enumerate(('left', 'right')):
        values = dict(arrays)
        values['pred_normalized'] = arrays['pred_normalized']+np.float32(.25 if index == 0 else 0.)
        path = campaign['cache']/(name+'.npz')
        np.savez(path, **values)
        raw = dict(expected_statistics(values['pred_normalized'], values['target_normalized'], values['observed']),
                   method=name, run_id=name+'-job', seed=92601, dataset='synthetic', stage='selected_refit',
                   evaluation='V_select', fit='fresh_A60', C=2, H=3, K=1, L=10, n_origins=4,
                   selected_epoch=1., selection_split='A_stop', prediction_binding=contracts.binding(path, campaign),
                   model_sha='F', core_dtype='float32', units='normalized_A60')
        proof = m._proof(raw, {}, dict(job_id=name+'-job', kind='training', scientific_identity=name,
                         cohort='synthetic', K=1, device='cpu'), campaign)
        proofs.append(proof); raw_rows.append(raw)
        run_rows.append(dict(raw, row_id=name, job_id=name+'-job', cohort='synthetic', record_origin='NEW',
                    device='cpu', point_statistic='quantile_0.5', recipe_signature='registered-recipe',
                    backend_signature='same-backend', source_signature='same-common-source',
                    scaler_sha256=proof['scaler_sha256_ifproven'], evaluation_stride=3,
                    source_file='report/pair_source.json', source_collection='rows', source_row_index=index))
    write(report/'pair_source.json', {'rows': raw_rows})
    for row in run_rows:
        row['source_sha256'] = contracts.sha(report/'pair_source.json')
    comparison = dict(comparison_id='pair-1', candidate_row_id='left', control_row_id='right',
                      candidate='left', control='right', cohort='synthetic', seed=92601,
                      match_status='MATCHED', stage='selected_refit', evaluation='V_select',
                      delta_MSE=raw_rows[0]['MSE']-raw_rows[1]['MSE'],
                      delta_MAE=raw_rows[0]['MAE']-raw_rows[1]['MAE'], horizon_overlap=False)
    pairs = []
    for i in range(4):
        counts = arrays['observed'][i].sum(0)
        defined = bool(np.all(counts > 0))
        row = dict(comparison_id='pair-1', candidate_row_id='left', control_row_id='right',
                   candidate='left', control='right', cohort='synthetic', seed=92601, origin_index=i,
                   C=2, n_origins=4, evaluation_stride=3, horizon_overlap=False,
                   status='DEFINED' if defined else 'UNDEFINED', candidate_status='DEFINED' if defined else 'UNDEFINED',
                   control_status='DEFINED' if defined else 'UNDEFINED', candidate_coverage_channels=int((counts > 0).sum()),
                   control_coverage_channels=int((counts > 0).sum()))
        for label, run_row, proof in zip(('candidate', 'control'), run_rows, proofs):
            if defined:
                delta = arrays['pred_normalized'][i]+np.float32(.25 if label == 'candidate' else 0.)-arrays['target_normalized'][i]
                row[label+'_MSE'] = float(np.mean(np.where(arrays['observed'][i], delta.astype(np.float64)**2, 0).sum(0)/counts))
                row[label+'_MAE'] = float(np.mean(np.where(arrays['observed'][i], np.abs(delta.astype(np.float64)), 0).sum(0)/counts))
            else:
                row[label+'_MSE'] = row[label+'_MAE'] = None
            for key in ('source_file', 'source_sha256', 'source_collection', 'source_row_index'):
                row[label+'_'+key] = run_row[key]
            row[label+'_prediction_reference'] = proof['prediction_reference']
            row[label+'_prediction_sha256'] = proof['prediction_sha256']
        for metric in ('MSE', 'MAE'):
            row['delta_'+metric] = row['candidate_'+metric]-row['control_'+metric] if defined else None
        pairs.append(row)
    def save_pairs():
        save_csv(report/'PAIRED_ORIGIN_EFFECTS.csv', pairs)
    def save_csv(path, rows):
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({key: json.dumps(value) if isinstance(value, dict) else value for key, value in row.items()})
    save_csv(report/'RUNS.csv', run_rows)
    save_csv(report/'COMPARISONS.csv', [comparison])
    save_pairs()
    return proofs, pairs, save_pairs


@pytest.mark.parametrize('mutation', ['delta', 'index', 'metadata', 'source', 'comparison', 'coverage', 'missing_origin'])
def test_paired_origin_effects_are_verified_from_actual_saved_prediction_proofs(campaign, mutation):
    proofs, rows, save = paired_case(campaign)
    checked = impl()._public_tables(campaign, proofs, [])
    assert checked['rows'] == 2
    assert checked['paired_origin_statistics']['origins'] == 4
    if mutation == 'delta':
        rows[0]['delta_MSE'] *= -1
    elif mutation == 'index':
        rows[0]['origin_index'] = 4
    elif mutation == 'metadata':
        rows[0]['evaluation_stride'] = 1
    elif mutation == 'source':
        rows[0]['candidate_prediction_sha256'] = '0'*64
    elif mutation == 'comparison':
        rows[0]['comparison_id'] = 'unregistered-comparison'
    elif mutation == 'coverage':
        rows[0]['candidate_coverage_channels'] = 1
    else:
        rows.pop()
    save()
    with pytest.raises(ValueError, match='paired_'):
        impl()._public_tables(campaign, proofs, [])


def test_paired_origin_missing_channel_is_undefined_and_never_zero_filled(campaign):
    proofs, rows, save = paired_case(campaign, missing_channel=True)
    assert rows[0]['status'] == 'UNDEFINED'
    assert impl()._public_tables(campaign, proofs, [])['rows'] == 2
    rows[0]['candidate_MSE'] = rows[0]['control_MSE'] = rows[0]['delta_MSE'] = 0.
    save()
    with pytest.raises(ValueError, match='paired_'):
        impl()._public_tables(campaign, proofs, [])


def test_public_proof_summary_is_bounded_and_hashes_every_full_statistic():
    m = impl()
    prediction = (np.arange(3000*4*3).reshape(3000, 4, 3)%17)/8
    target = np.ones_like(prediction)
    mask = np.ones_like(prediction, bool)
    mask[0, :, 0] = False
    full = m.independent_statistics(prediction, target, mask)
    original_digest = contracts.digest(full)
    proof = dict(source_job_id='large-origin-fixture', prediction_reference='cache://fixture.npz',
                 prediction_sha256='1'*64, statistics=full, validation_status='PASS')
    compact = m._compact_proof(proof)
    assert compact['statistics']['MSE'] == full['MSE']
    assert compact['statistics']['n_origins'] == 3000
    assert compact['full_statistics_sha256'] == original_digest
    assert compact['statistics_blocks']['origin_sse']['sha256'] == contracts.digest(full['origin_sse'])
    assert compact['statistics_blocks']['origin_sse']['shape'] == [3000, 3]
    assert 'origin_sse' not in compact['statistics'] and 'origin_statistics' not in compact['statistics']
    assert compact['prediction_reference'] == proof['prediction_reference']
    assert len(json.dumps(proof, indent=2).encode()) > 1_000_000
    assert len(json.dumps(compact, indent=2).encode()) < 32*1024
    assert contracts.digest(proof['statistics']) == original_digest
    changed = dict(full, origin_sse=[list(row) for row in full['origin_sse']])
    changed['origin_sse'][0][0] += .5
    assert m._compact_proof(dict(proof, statistics=changed))['full_statistics_sha256'] != original_digest


def test_compaction_happens_after_full_statistics_tamper_gate(campaign):
    m = impl()
    value = m.verify(campaign, write=False)
    proof, = value['verified_prediction_archives']
    assert 'horizon_sse' not in proof['statistics']
    assert proof['statistics_blocks']['horizon_sse']['shape'] == [3, 2]
    campaign['_test_result']['rows'][0]['horizon_sse'][0][0] += .5
    refresh(campaign)
    failed = m.verify(campaign, write=False)
    assert failed['status'] == 'FAIL'
    assert not failed['verified_prediction_archives']
    assert any('horizon_sse_mismatch' in row.get('reason', '') for row in failed['failures'])


@pytest.mark.parametrize('mutation', [None, 'tampered', 'missing', 'escape', 'outside_scope',
                                    'executable', 'scientific_field', 'reference'])
def test_multik_metadata_sources_are_hashed_without_relaxing_the_scientific_seal(campaign, mutation):
    repo = campaign['repo']
    prefix = 'research/informs_broad_validation_20261015_v1'
    sources = {}
    for relative in ('runs/parent/SOURCE_CONTRACT.json', 'protocol.md', 'summary.csv', 'RESULT.json.gz'):
        name = prefix+'/'+relative
        path = repo/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'historical evidence\n')
        sources[name] = contracts.sha(path)
    first = next(iter(sources))
    result = dict(status='COMPLETE', rows=[{'source_reference': first}], source_hashes=sources,
                  scientific_source_hashes=campaign['_test_plan']['source_hashes'])
    bound_path = campaign['cache']/'historical_binding.json'
    bound_path.write_text('existing bound historical artifact')
    result['private_artifacts'] = [contracts.binding(bound_path, campaign)]
    if mutation in ('escape', 'outside_scope', 'executable'):
        name = {'escape': prefix+'/../../escaping.json',
                'outside_scope': 'research/unregistered_history/evidence.json',
                'executable': prefix+'/unsealed_producer.py'}[mutation]
        target = repo/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'unregistered evidence\n')
        result['source_hashes'][name] = contracts.sha(target)
    elif mutation == 'scientific_field':
        result['scientific_source_hashes'] = {first: sources[first]}
    job = contracts.seal_job(dict(job_id='MULTIK_AUDIT', kind='multik_audit', dependency_ids=[]),
                             campaign['_test_plan']['source_hashes'])
    campaign['_test_plan']['jobs'].append(job)
    campaign['_test_plan'] = contracts.seal_plan(campaign['_test_plan'])
    write(campaign['report']/'EXPERIMENT_PLAN.json', campaign['_test_plan'])
    public = campaign['report']/'jobs'/'MULTIK_AUDIT.json'
    write(public, result)
    contracts.finish_job(campaign['report']/'receipts'/'MULTIK_AUDIT.json', job,
                         dict(status='COMPLETE', result=result, artifact_paths=[str(public)]),
                         dict(campaign, plan=campaign['_test_plan']))
    refresh(campaign)
    if mutation == 'tampered':
        (repo/first).write_bytes(b'changed historical source')
    elif mutation == 'missing':
        (repo/first).unlink()
    elif mutation == 'reference':
        bound_path.write_text('changed existing binding')
    value = impl().verify(campaign, write=False)
    assert value['status'] == ('PASS' if mutation is None else 'FAIL'), value['failures']
    if mutation is None:
        assert any(check['id'] == 'producer_sources:MULTIK_AUDIT' and check['status'] == 'PASS'
                   for check in value['checks'])
    else:
        expected_gate = 'receipt:identity_and_artifacts:MULTIK_AUDIT' if mutation == 'reference' else 'producer_sources:MULTIK_AUDIT'
        assert any(check['id'] == expected_gate and check['status'] == 'FAIL' for check in value['checks'])


def test_multik_metadata_exception_does_not_apply_to_other_producers_or_code_fields(campaign):
    name = 'research/informs_broad_validation_20261015_v1/SOURCE_CONTRACT.json'
    path = campaign['repo']/name
    path.parent.mkdir(parents=True)
    path.write_text('historical metadata')
    sources = {name: contracts.sha(path)}
    for kind, field in (('training', 'source_hashes'), ('multik_audit', 'source_files'),
                        ('multik_audit', 'scientific_source_hashes')):
        with pytest.raises(ValueError, match='producer_source_seal_mismatch'):
            impl()._producer_sources({field: sources}, campaign['_test_plan'], campaign, {'kind': kind})
    impl()._producer_sources({'source_hashes': campaign['_test_plan']['source_hashes']},
                            campaign['_test_plan'], campaign, {'kind': 'multik_audit'})


@pytest.mark.parametrize('historical_H', ['48.0', '48.5', 'NaN', '48.00000000000000001'])
def test_public_historical_integer_representation_is_exact_not_truncated_or_rounded(campaign, historical_H):
    import csv
    m = impl()
    arrays = dict(campaign['_test_arrays'])
    for key in ('pred_normalized', 'target_normalized', 'observed'):
        arrays[key] = np.tile(arrays[key], (1, 16, 1))
    arrays['origins'] = np.arange(4, dtype=np.int64)*48+20
    archive = campaign['cache']/'predictions.npz'
    np.savez(archive, **arrays)
    raw = campaign['_test_result']['rows'][0]
    raw.update(expected_statistics(arrays['pred_normalized'], arrays['target_normalized'], arrays['observed']))
    raw.update(H=48, prediction_binding=contracts.binding(archive, campaign),
               array_fingerprints={key: dict(shape=list(value.shape), dtype=str(value.dtype),
                   sha256=hashlib.sha256(value.tobytes()).hexdigest()) for key, value in arrays.items()})
    refresh(campaign)
    proof = m._proof(raw, campaign['_test_result'], campaign['_test_job'], campaign)
    row = dict(row_id='historical-float-format', job_id=proof['source_job_id'], record_origin='CONTEXTUAL',
               MSE=raw['MSE'], MAE=raw['MAE'], H=historical_H, C='2.0', prediction_binding=raw['prediction_binding'])
    statistics = [dict(row_id=row['row_id'], horizon=h+1, channel_id=c,
                  SSE=proof['statistics']['horizon_sse'][h][c], SAE=proof['statistics']['horizon_sae'][h][c],
                  count=proof['statistics']['horizon_counts'][h][c]) for h in range(48) for c in range(2)]
    for name, values in (('RUNS.csv', [row]), ('CHANNEL_HORIZON_STATS.csv', statistics)):
        with (campaign['report']/name).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows({key: json.dumps(value) if isinstance(value, dict) else value for key, value in r.items()}
                            for r in values)
    if historical_H == '48.0':
        assert m._public_tables(campaign, [proof], [])['rows'] == 1
    else:
        with pytest.raises(ValueError, match='integer'):
            m._public_tables(campaign, [proof], [])


def test_public_csv_integer_parser_keeps_full_precision_and_rejects_nonfinite_values():
    m = impl()
    for value, expected in [('48.0', 48), ('0.0', 0), ('9007199254740993.0', 9007199254740993)]:
        assert m._csv_integer(value) == expected
    for value in ('48.5', '48.00000000000000001', 'NaN', 'Infinity', '-Infinity', '', None, True):
        with pytest.raises(ValueError, match='integer'):
            m._csv_integer(value)
