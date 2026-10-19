import json

import numpy as np
import pytest


def test_channel_macro_uses_each_observed_channel_equally_and_fp64():
    from research.b2_dataset_search_20261019_v1 import independent_wave003 as audit
    prediction = np.array([[[1., 2.], [9., 2.]], [[9., 2.], [9., 2.]]], np.float32)
    target = np.zeros_like(prediction)
    mask = np.ones_like(prediction, bool); mask[:, :, 0] = False; mask[0, 0, 0] = True
    result = audit.metrics(prediction, target, mask)
    assert result['MSE'] == 2.5 and result['MAE'] == 1.5
    assert result['channel_counts'] == [1, 4] and result['observed'] == 5


def test_scaler_uses_only_observed_prefix_and_constant_scale_one():
    from research.b2_dataset_search_20261019_v1 import independent_wave003 as audit
    physical = np.array([[0., 7.], [2., 7.], [1000., 7.], [9000., 8.]], np.float32)
    mask = np.ones_like(physical, bool); mask[2, 0] = False
    mean, std = audit.scaler(physical, mask, 3)
    assert mean.tolist() == [1., 7.] and std.tolist() == [1., 1.]


def cohort_fixture():
    from research.b2_dataset_search_20261019_v1 import independent_wave003 as audit
    physical = np.arange(40, dtype=np.float32).reshape(20, 2)
    observed = np.ones_like(physical, bool)
    config = dict(L=4, H=2, columns=['a', 'b'], split_ranges=dict(A_fit=[0, 8], A_stop=[8, 10], V_select=[12, 18]))
    mean, std = audit.scaler(physical, observed, 10)
    origins = np.array([12, 14, 16], np.int64)
    rows = origins[:, None] + np.arange(2)[None]
    target = ((physical[rows]-mean)/std).astype(np.float32)
    arrays = dict(origins=origins, channel_ids=np.array(['a', 'b']), mean=mean, std=std,
        observed=observed[rows], target_normalized=target,
        reference_pred_normalized=target.copy(), pred_normalized=target.astype(np.float64))
    return config, physical, observed, arrays


@pytest.mark.parametrize('change', ('origin', 'mask', 'target', 'scaler', 'channel', 'parity'))
def test_canonical_arrays_reject_changed_cohort_or_predictions(change):
    from research.b2_dataset_search_20261019_v1 import independent_wave003 as audit
    config, physical, observed, arrays = cohort_fixture()
    if change == 'origin': arrays['origins'][0] += 1
    elif change == 'mask': arrays['observed'][0, 0, 0] = False
    elif change == 'target': arrays['target_normalized'][0, 0, 0] += .1
    elif change == 'scaler': arrays['mean'][0] += .1
    elif change == 'channel': arrays['channel_ids'][0] = 'z'
    else: arrays['pred_normalized'][0, 0, 0] += .1
    with pytest.raises(ValueError):
        audit.verify_arrays(arrays, config, physical, observed)


def test_valid_canonical_predictions_are_independently_rescored():
    from research.b2_dataset_search_20261019_v1 import independent_wave003 as audit
    config, physical, observed, arrays = cohort_fixture()
    result = audit.verify_arrays(arrays, config, physical, observed)
    assert result['MSE'] == 0. and result['MAE'] == 0.
    assert result['n_origins'] == 3 and result['observed'] == 12


def test_selection_requires_full_schedule_and_earliest_exact_tie():
    from research.b2_dataset_search_20261019_v1 import independent_wave003 as audit
    curve = [dict(epoch=e, astop_mse=m) for e, m in [(0, 3), (.25, 1), (1, 1), (3, 2)]]
    assert audit.select_epoch(curve) == (.25, 1.)
    with pytest.raises(ValueError, match='schedule'):
        audit.select_epoch(curve[:-1])
    assert audit.exposure_arithmetic(17, 3) == (51, 5)
    assert audit.exposure_arithmetic(17, .25, refit=True) == (5, 1)


def cost_fixture():
    trials = [dict(phase=phase, index=i, trial=i, whole_seconds=(i+1)/1000,
        peak_allocated_MiB=90., peak_reserved_MiB=100., pid=123,
        F_calls=1, F_logical_streams=16, F_physical_streams=16)
        for phase, count in [('warmup', 10), ('timed', 20)] for i in range(count)]
    summary = dict(warmups=10, timed_trials=20, status='COMPLETE', parity_status='PASS',
        median_batch_ms=10.5, IQR_batch_ms=9.5, p95_batch_ms=19.05,
        median_ms_per_origin=1.3125, origins_per_second=8000/10.5,
        peak_allocated_MiB=90., peak_reserved_MiB=100., process_pid=123, pid=123,
        process_create_time=1., F_calls=1, F_logical_streams=16,
        process_scope='fresh OS process, one facet, one serialized model load')
    return dict(trials=trials, summary=summary, family='B2', K=2, C=7, row_policy='DEFAULT128',
        requested_row_cap=128, effective_row_cap=16, actual_F_width=2, model_K=2,
        expected_work=dict(q=2, width=2, streams=16, requested_row_cap=128, effective_row_cap=16,
            calls=1, row_sizes=[16]))


@pytest.mark.parametrize('change', ('quantile', 'count', 'pid', 'memory', 'work'))
def test_saved_raw_trials_are_recomputed_and_tampering_is_rejected(change):
    from research.b2_dataset_search_20261019_v1 import independent_wave003 as audit
    receipt = cost_fixture()
    result = audit.cost_statistics(receipt)
    assert result['p50_ms'] == 10.5 and result['peak_allocated_MiB'] == 90.
    if change == 'quantile': receipt['summary']['median_batch_ms'] = 11.
    elif change == 'count': receipt['trials'].pop()
    elif change == 'pid': receipt['trials'][-1]['pid'] = 99
    elif change == 'memory': receipt['trials'][-1]['peak_allocated_MiB'] = 200.
    else: receipt['trials'][-1]['F_calls'] = 2
    with pytest.raises(ValueError): audit.cost_statistics(receipt)


def test_positive_rule_preserves_negative_widths_both_seeds_and_mae():
    from research.b2_dataset_search_20261019_v1 import independent_wave003 as audit
    rows = [dict(family='RAW', K=7, seed=92601, MSE=1., MAE=.5, p50_ms=10., peak_allocated_MiB=100.)]
    for k, errors in [(1, [1.2, 1.2]), (2, [.989, .991]), (4, [.9, 1.1])]:
        rows += [dict(family='B2', K=k, seed=seed, MSE=mse, MAE=.6, p50_ms=8., peak_allocated_MiB=90.)
                 for seed, mse in zip((92601, 92602), errors)]
    rows += [dict(family='DIRECT_G', K=0, seed=seed, MSE=.8, MAE=.4, p50_ms=1., peak_allocated_MiB=5.)
             for seed in (92601, 92602)]
    result = audit.judge(rows)
    assert result['positive'] and list(result['widths']) == ['1', '2', '4']
    assert not result['widths']['1']['positive'] and not result['widths']['4']['positive']
    assert result['widths']['2']['positive']
    assert result['widths']['2']['MAE_change_pct_vs_Raw'] == pytest.approx(20.)
    assert result['widths']['2']['Direct_dominates_two_seed_mean']
    rows[4]['p50_ms'] = 10.
    assert not audit.judge(rows)['positive']


def test_terminal_failure_prevents_even_prefix_or_prediction_loading(tmp_path, monkeypatch):
    from research.b2_dataset_search_20261019_v1 import independent_wave003 as audit
    def forbidden(*args, **kwargs): raise AssertionError('Numeric arrays cannot be read before terminal success')
    monkeypatch.setattr(audit.np, 'load', forbidden)
    with pytest.raises(ValueError, match='terminal'):
        audit.audit(tmp_path)
    cache = tmp_path / '.cache' / audit.NAME; cache.mkdir(parents=True)
    (cache / 'TERMINAL_003.json').write_text(json.dumps(dict(status='FAILED', exit_code=1, wave=3)))
    with pytest.raises(ValueError, match='terminal'):
        audit.audit(tmp_path)


@pytest.mark.parametrize('change', ('unfinished_costs', 'changed_wave', 'duplicate_stage_time'))
def test_terminal_exit0_also_requires_the_three_correct_completed_producer_stages(tmp_path, monkeypatch, change):
    from research.b2_dataset_search_20261019_v1 import independent_wave003 as audit
    cache = tmp_path / '.cache' / audit.NAME; cache.mkdir(parents=True)
    report = tmp_path / 'reports' / audit.NAME / 'waves'; report.mkdir(parents=True)
    wave_path = report / 'WAVE_003.json'; wave_path.write_text('{}')
    source = tmp_path / 'research' / audit.NAME / 'run.py'; source.parent.mkdir(parents=True); source.write_text('# synthetic')
    (cache / 'TERMINAL_003.json').write_text(json.dumps(dict(status='COMPLETE', exit_code=0, wave=3,
        started_unix=1., finished_unix=10.)))
    events = []
    for stage, when in [('train', 2.), ('baselines', 4.), ('costs', 6.)]:
        for status, offset in [('STARTED', 0), ('COMPLETE', 1)]:
            events.append(dict(stage=stage, status=status, wave=3, wave_sha256=audit.sha(wave_path),
                producer_sha256=audit.sha(source), observed_unix=when+offset))
    journal = cache / 'PROCESS_ATTEMPTS.jsonl'
    journal.write_text(''.join(json.dumps(row)+'\n' for row in events))
    assert set(audit.require_terminal(tmp_path)['stage_completions']) == {'train', 'baselines', 'costs'}
    if change == 'unfinished_costs': events[-1]['status'] = 'FAILED'
    elif change == 'changed_wave': events[-1]['wave_sha256'] = 'changed'
    else: events[-1]['observed_unix'] = 3.
    journal.write_text(''.join(json.dumps(row)+'\n' for row in events))
    def forbidden(*args, **kwargs): raise AssertionError('A terminal driver alone does not authorize array reads')
    monkeypatch.setattr(audit.np, 'load', forbidden)
    with pytest.raises(ValueError, match='stage|journal|Producer'):
        audit.audit(tmp_path)


def test_refit_counter_audit_checks_selected_epoch_ceiling_and_segmented_search_updates():
    from research.b2_dataset_search_20261019_v1 import independent_wave003 as audit
    config, _, _, _ = cohort_fixture()
    curve = [dict(epoch=e, astop_mse=m) for e, m in [(0, 3), (.25, 1), (1, 1), (3, 2)]]
    fit = dict(status='COMPLETE_VALID', optimizer='AdamW', weight_decay=0, selected_epoch=.25,
        selected_astop_mse=1., search_fit_end=8, refit_fit_end=10, normalization_fit_end=10,
        search_train_origins=3, search_stop_origins=1, search_exposures=9, search_updates=3,
        refit_exposures=2, refit_updates=1)
    search = dict(status='SEGMENT_COMPLETE', train_origins=3, exposures=9, updates=3, actual_epochs=3.)
    refit = dict(status='SEGMENT_COMPLETE', train_origins=5, exposures=2, updates=1, actual_epochs=.4)
    assert audit._fit_arithmetic(fit, curve, search, refit, config)['refit_exposures'] == 2
    fit['search_updates'] = 1
    with pytest.raises(ValueError, match='arithmetic'):
        audit._fit_arithmetic(fit, curve, search, refit, config)


def test_report_writer_is_idempotent_and_preserves_existing_bytes(tmp_path):
    from research.b2_dataset_search_20261019_v1 import independent_wave003 as audit
    path = tmp_path / 'INDEPENDENT_AUDIT_WAVE_003.json'
    audit.write_report(path, dict(status='PASS', positive=False))
    before = path.read_bytes()
    audit.write_report(path, dict(status='PASS', positive=False))
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError): audit.write_report(path, dict(status='PASS', positive=True))


def recovery_fixture(repo):
    from research.b2_dataset_search_20261019_v1 import independent_wave003 as audit
    cache = repo / '.cache' / audit.NAME; cache.mkdir(parents=True)
    report = repo / 'reports' / audit.NAME
    (report / 'waves').mkdir(parents=True)
    wave = report / 'waves' / 'WAVE_003.json'; wave.write_text('{"wave":3}')
    source = repo / 'research' / audit.NAME / 'run.py'
    source.parent.mkdir(parents=True); source.write_text('# Independent synthetic coordinator')
    previous_path = cache / 'TERMINAL_003.json'
    previous_path.write_text(json.dumps(dict(status='FAILED', exit_code=1073807364,
        wave=3, started_unix=1., finished_unix=10., wall_seconds=9.)))
    retry_path = cache / 'TERMINAL_003_ATTEMPT_002.json'
    retry_path.write_text(json.dumps(dict(status='COMPLETE', exit_code=0,
        wave=3, started_unix=12., finished_unix=25., wall_seconds=13.)))
    amendment_path = report / 'EXECUTION_RECOVERY_WAVE_003_ATTEMPT_002.json'
    amendment_path.write_text(json.dumps(dict(status='REGISTERED_BEFORE_RESUMED_PRODUCER', wave=3, attempt=2,
        previous_terminal=dict(reference='cache://TERMINAL_003.json', sha256=audit.sha(previous_path)),
        next_terminal_reference='cache://TERMINAL_003_ATTEMPT_002.json', unchanged_producer_sources=True,
        wave_sha256=audit.sha(wave), producer_sha256=audit.sha(source), registered_unix=11.)))
    events = [dict(stage=stage, status=status, wave=3, wave_sha256=audit.sha(wave), producer_sha256=audit.sha(source),
        observed_unix=when+offset) for stage, when in [('train', 13.), ('baselines', 16.), ('costs', 19.)]
        for status, offset in [('STARTED', 0), ('COMPLETE', 1)]]
    (cache / 'PROCESS_ATTEMPTS.jsonl').write_text(''.join(json.dumps(event)+'\n' for event in events))
    return report, cache, amendment_path


def test_independent_recovery_returns_registration_and_preserved_failure_evidence(tmp_path, monkeypatch):
    from research.b2_dataset_search_20261019_v1 import independent_wave003 as audit
    _, cache, amendment_path = recovery_fixture(tmp_path)
    original = (cache / 'TERMINAL_003.json').read_bytes()
    def forbidden(*args, **kwargs): raise AssertionError('No arrays belong in terminal lineage verification')
    monkeypatch.setattr(audit.np, 'load', forbidden)
    result = audit.require_terminal(tmp_path)
    assert result['driver']['exit_code'] == 0
    assert result['terminal_sha256'] == audit.sha(cache / 'TERMINAL_003_ATTEMPT_002.json')
    recovery = result['execution_recovery']
    assert recovery['binding']['sha256'] == audit.sha(amendment_path)
    assert recovery['previous_terminal']['status'] == 'FAILED'
    assert recovery['previous_terminal']['exit_code'] == 1073807364
    assert recovery['previous_terminal']['sha256'] == audit.sha(cache / 'TERMINAL_003.json')
    assert (cache / 'TERMINAL_003.json').read_bytes() == original


@pytest.mark.parametrize('change', ('unregistered_retry', 'missing_retry', 'failed_retry', 'retry_wrong_wave',
    'wrong_status', 'wrong_wave', 'wrong_attempt', 'failed_hash', 'failed_path', 'next_path', 'changed_source',
    'changed_wave', 'source_flag', 'registration_before_failure', 'registration_after_retry', 'registration_nan',
    'registration_bool', 'previous_success', 'previous_bool_exit', 'retry_nonfinite_time', 'stage_before_retry',
    'unfinished_costs'))
def test_independent_recovery_fails_before_prefix_or_prediction_loading(tmp_path, monkeypatch, change):
    from research.b2_dataset_search_20261019_v1 import independent_wave003 as audit
    report, cache, amendment_path = recovery_fixture(tmp_path)
    registration = json.loads(amendment_path.read_text())
    retry_path = cache / 'TERMINAL_003_ATTEMPT_002.json'
    if change == 'unregistered_retry': amendment_path.unlink()
    elif change == 'missing_retry': retry_path.unlink()
    elif change in ('failed_retry', 'retry_wrong_wave', 'retry_nonfinite_time'):
        retry = json.loads(retry_path.read_text())
        if change == 'failed_retry': retry.update(status='FAILED', exit_code=1)
        elif change == 'retry_wrong_wave': retry['wave'] = 4
        else: retry['finished_unix'] = float('inf')
        retry_path.write_text(json.dumps(retry))
    elif change == 'changed_source':
        (tmp_path / 'research' / audit.NAME / 'run.py').write_text('# Different producer')
    elif change == 'changed_wave': (report / 'waves' / 'WAVE_003.json').write_text('{}')
    elif change in ('previous_success', 'previous_bool_exit'):
        previous_path = cache / 'TERMINAL_003.json'
        previous = json.loads(previous_path.read_text())
        previous.update(status='COMPLETE', exit_code=0) if change == 'previous_success' else previous.update(exit_code=True)
        previous_path.write_text(json.dumps(previous))
        registration['previous_terminal']['sha256'] = audit.sha(previous_path)
    elif change in ('stage_before_retry', 'unfinished_costs'):
        journal = cache / 'PROCESS_ATTEMPTS.jsonl'
        events = [json.loads(line) for line in journal.read_text().splitlines()]
        events[-1].update(observed_unix=9.) if change == 'stage_before_retry' else events[-1].update(status='FAILED')
        journal.write_text(''.join(json.dumps(event)+'\n' for event in events))
    else:
        key, value = {'wrong_status': ('status', 'COMPLETE'), 'wrong_wave': ('wave', True),
            'wrong_attempt': ('attempt', 1), 'next_path': ('next_terminal_reference', 'cache://TERMINAL_003.json'),
            'source_flag': ('unchanged_producer_sources', False),
            'registration_before_failure': ('registered_unix', 10.), 'registration_after_retry': ('registered_unix', 13.),
            'registration_nan': ('registered_unix', float('nan')), 'registration_bool': ('registered_unix', False),
            'failed_hash': ('previous_terminal', dict(reference='cache://TERMINAL_003.json', sha256='wrong')),
            'failed_path': ('previous_terminal', dict(reference='cache://../TERMINAL_003.json',
                sha256=registration['previous_terminal']['sha256']))}[change]
        registration[key] = value
    if change != 'unregistered_retry': amendment_path.write_text(json.dumps(registration))
    def forbidden(*args, **kwargs): raise AssertionError('Arrays cannot precede registered terminal and stage completion')
    monkeypatch.setattr(audit.np, 'load', forbidden)
    with pytest.raises((ValueError, FileNotFoundError)):
        audit.audit(tmp_path)


def recovery_chain_fixture(repo):
    from research.b2_dataset_search_20261019_v1 import independent_wave003 as audit
    report, cache, preceding_path = recovery_fixture(repo)
    previous_path = cache / 'TERMINAL_003_ATTEMPT_002.json'
    previous = json.loads(previous_path.read_text()); previous.update(status='FAILED', exit_code=1)
    previous_path.write_text(json.dumps(previous))
    retry_path = cache / 'TERMINAL_003_ATTEMPT_003.json'
    retry_path.write_text(json.dumps(dict(status='COMPLETE', exit_code=0, wave=3,
        started_unix=27., finished_unix=40., wall_seconds=13.)))
    adapter = repo / 'research' / audit.NAME / 'resume_wave003.py'; adapter.write_text('# Synthetic adapter source')
    registration = json.loads(preceding_path.read_text())
    registration.update(attempt=3, previous_terminal=dict(reference='cache://' + previous_path.name, sha256=audit.sha(previous_path)),
        next_terminal_reference='cache://' + retry_path.name, registered_unix=26.,
        preceding_recovery=dict(reference='report://' + preceding_path.name, sha256=audit.sha(preceding_path)),
        execution_adapter=dict(reference='repo://research/' + audit.NAME + '/resume_wave003.py', sha256=audit.sha(adapter)),
        identity_serialization_repair='JSON_ROUNDTRIP_ONLY_EXACT_TRAIN_IDENTITY')
    amendment_path = report / 'EXECUTION_RECOVERY_WAVE_003_ATTEMPT_003.json'
    amendment_path.write_text(json.dumps(registration))
    journal = cache / 'PROCESS_ATTEMPTS.jsonl'
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    events += [dict(event, observed_unix=event['observed_unix']+15.) for event in events]
    journal.write_text(''.join(json.dumps(event)+'\n' for event in events))
    return report, cache, amendment_path, preceding_path


def test_independent_third_attempt_preserves_both_prior_terminal_evidence(tmp_path, monkeypatch):
    from research.b2_dataset_search_20261019_v1 import independent_wave003 as audit
    _, cache, amendment_path, preceding_path = recovery_chain_fixture(tmp_path)
    originals = {name: (cache / name).read_bytes() for name in ('TERMINAL_003.json', 'TERMINAL_003_ATTEMPT_002.json')}
    def forbidden(*args, **kwargs): raise AssertionError('Lineage verification has no numeric array reads')
    monkeypatch.setattr(audit.np, 'load', forbidden)
    result = audit.require_terminal(tmp_path)
    assert result['terminal_sha256'] == audit.sha(cache / 'TERMINAL_003_ATTEMPT_003.json')
    recovery = result['execution_recovery']
    assert recovery['binding']['sha256'] == audit.sha(amendment_path)
    assert recovery['previous_terminal']['exit_code'] == 1
    assert len(recovery['preceding_recoveries']) == 1
    assert recovery['preceding_recoveries'][0]['binding']['sha256'] == audit.sha(preceding_path)
    assert recovery['preceding_recoveries'][0]['previous_terminal']['exit_code'] == 1073807364
    assert all((cache / name).read_bytes() == value for name, value in originals.items())


@pytest.mark.parametrize('change', ('unregistered_third', 'missing_third_terminal', 'failed_third', 'missing_preceding',
    'preceding_hash', 'preceding_path', 'original_hash', 'preceding_late_registration', 'wrong_previous_path',
    'third_registration_early', 'third_registration_late', 'stage_outside_third', 'adapter_path', 'adapter_hash',
    'adapter_source_changed', 'adapter_missing', 'adapter_scope'))
def test_independent_third_attempt_requires_complete_sealed_recovery_chain(tmp_path, monkeypatch, change):
    from research.b2_dataset_search_20261019_v1 import independent_wave003 as audit
    _, cache, amendment_path, preceding_path = recovery_chain_fixture(tmp_path)
    registration = json.loads(amendment_path.read_text())
    if change == 'unregistered_third': amendment_path.unlink()
    elif change == 'missing_third_terminal': (cache / 'TERMINAL_003_ATTEMPT_003.json').unlink()
    elif change == 'failed_third':
        path = cache / 'TERMINAL_003_ATTEMPT_003.json'
        value = json.loads(path.read_text()); value.update(status='FAILED', exit_code=1); path.write_text(json.dumps(value))
    elif change == 'missing_preceding': preceding_path.unlink()
    elif change == 'original_hash':
        path = cache / 'TERMINAL_003.json'; path.write_bytes(path.read_bytes()+b'\n')
    elif change == 'preceding_late_registration':
        preceding = json.loads(preceding_path.read_text()); preceding['registered_unix'] = 13.
        preceding_path.write_text(json.dumps(preceding)); registration['preceding_recovery']['sha256'] = audit.sha(preceding_path)
    elif change == 'stage_outside_third':
        journal = cache / 'PROCESS_ATTEMPTS.jsonl'
        events = [json.loads(line) for line in journal.read_text().splitlines()]; events[-1]['observed_unix'] = 20.
        journal.write_text(''.join(json.dumps(event)+'\n' for event in events))
    elif change == 'preceding_hash': registration['preceding_recovery']['sha256'] = 'wrong'
    elif change == 'preceding_path': registration['preceding_recovery']['reference'] = 'report://elsewhere.json'
    elif change == 'wrong_previous_path': registration['previous_terminal']['reference'] = 'cache://TERMINAL_003.json'
    elif change == 'adapter_path': registration['execution_adapter']['reference'] = 'repo://research/elsewhere.py'
    elif change == 'adapter_hash': registration['execution_adapter']['sha256'] = 'wrong'
    elif change == 'adapter_scope': registration['identity_serialization_repair'] = 'ALTER_RECIPE'
    elif change == 'adapter_source_changed': (tmp_path / 'research' / audit.NAME / 'resume_wave003.py').write_text('# Different')
    elif change == 'adapter_missing': (tmp_path / 'research' / audit.NAME / 'resume_wave003.py').unlink()
    else: registration['registered_unix'] = 25. if change == 'third_registration_early' else 28.
    if change != 'unregistered_third': amendment_path.write_text(json.dumps(registration))
    def forbidden(*args, **kwargs): raise AssertionError('Invalid recovery must precede and prevent all numeric array reads')
    monkeypatch.setattr(audit.np, 'load', forbidden)
    with pytest.raises((ValueError, FileNotFoundError)):
        audit.audit(tmp_path)
