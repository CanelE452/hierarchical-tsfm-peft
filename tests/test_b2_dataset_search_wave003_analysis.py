import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


def synthetic_inputs(positive=True):
    from research.b2_dataset_search_20261019_v1 import analysis as base
    rows = []
    for policy in base.POLICIES:
        common = dict(dataset='ETTm1', wave=3, row_policy=policy, C=7, L=512, H=48, B=8,
            interval_seconds=900, n_origins=145, scored_origins=145, observed_targets=48720,
            row_cap=56 if policy == 'DEFAULT128' else 16, F_calls=1, F_streams=56)
        rows.append(dict(common, family='RAW', K=7, seed=92601, MSE=1., MAE=.5,
            p50_ms=10., peak_allocated_MiB=250., peak_reserved_MiB=260.))
        for k in (1, 2, 4):
            for seed in base.SEEDS:
                mse = (.94 if seed == 92601 else .96) if positive and k == 4 else 1.1
                rows.append(dict(common, family='B2', K=k, seed=seed, MSE=mse, MAE=.6,
                    p50_ms=8., peak_allocated_MiB=240., peak_reserved_MiB=260.))
        for seed in base.SEEDS:
            rows.append(dict(common, family='DIRECT_G', K=0, seed=seed, MSE=.8, MAE=.4,
                p50_ms=1., peak_allocated_MiB=10., peak_reserved_MiB=20.))
    proof = dict(status='PASS', wave=3, judgments=base.classify(rows), route_facets=18,
        warmup_trials=180, timed_trials=360, B2_searches=6, B2_fresh_refits=6,
        Direct_MLP_search_refits=2, training={}, historical=[], candidates=[
            dict(wave=2, dataset='ETTh1', status='PRESERVED_PRIOR_WAVE', detail='Prior wave'),
            dict(wave=3, dataset='ETTm1', status='POSITIVE_EXPLORATORY' if positive else 'NEGATIVE_EXPLORATORY',
                K_grid='1,2,4', detail='All K and both seeds disclosed')],
        datasets=dict(ETTm1=dict(C=7, columns=['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'],
            interval_seconds=900, original_rows=69680, prefix_rows=55744, train_structure={})),
        terminal_evidence=dict(status='COMPLETE', exit_code=0, wave=3),
        activation_amendment=dict(activated_wave=3, prior_positive_datasets=['Appliances', 'ETTh1']),
        pending_candidates=[dict(wave='NEXT_UNREGISTERED', originally_planned_wave=3,
            dataset='solar_full137', C=137, K_grid='18,35,70',
            status='READY_NOT_EXECUTED' if positive else 'READY_NEXT_WAVE_AFTER_NO_POSITIVE',
            measured_in_this_report=False, forecast_results_counted=0, detail='Prepared metadata only')])
    return rows, proof


def terminal_fixture(repo, *, terminal=True, stages=('train', 'baselines', 'costs')):
    from research.b2_dataset_search_20261019_v1 import analysis as base
    report = repo / 'reports' / base.NAME
    cache = repo / '.cache' / base.NAME
    (report / 'waves').mkdir(parents=True)
    cache.mkdir(parents=True)
    source = repo / 'research' / base.NAME / 'run.py'
    source.parent.mkdir(parents=True)
    source.write_text('# SYNTHETIC coordinator\n')
    plan = report / 'CANDIDATE_PLAN_003.json'
    plan.write_text(json.dumps(dict(planned_execution_wave=3, parent_wave=2,
        numeric_forecast_scores_used_to_choose_candidates=False, unchanged_recipe=True,
        unchanged_K_rule=True, ordered_candidates=[
            dict(dataset='ETTm1', C=7, columns=['a'] * 7, prefix_rows=55744, interval_seconds=900, K_grid=[1, 2, 4]),
            dict(dataset='solar_full137', C=137, columns=[str(i) for i in range(137)],
                prefix_rows=42048, interval_seconds=600, K_grid=[18, 35, 70])])), encoding='utf8')
    amendment = dict(status='REGISTERED_BEFORE_NEW_FORECASTS', registered_at_unix=1.,
        activated_wave=3, activated_datasets=['ETTm1'], new_authorization_overrides_prior_conditional_stop=True,
        preserve_prior_activation_rule=True, unchanged_recipe=True, unchanged_K_grid=True,
        V_scores_used_to_choose_wave=False, protected_T_access=False,
        prior_plan_binding=dict(reference='repo://reports/' + base.NAME + '/CANDIDATE_PLAN_003.json',
            sha256=base.sha(plan)), prior_completed_wave=2, prior_positive_datasets=['Appliances', 'ETTh1'])
    amendment_path = report / 'ACTIVATION_AMENDMENT_WAVE_003.json'
    amendment_path.write_text(json.dumps(amendment), encoding='utf8')
    wave = dict(status='REGISTERED_BEFORE_TRAINING', wave=3, registered_at_unix=2.,
        datasets=[dict(dataset='ETTm1', K_grid=[1, 2, 4])], candidate_plan_sha256=base.sha(plan),
        activation_amendment_sha256=base.sha(amendment_path), V_forecast_scores_used=False,
        T_consumed_by_this_wave_training=False)
    wave_path = report / 'waves' / 'WAVE_003.json'
    wave_path.write_text(json.dumps(wave), encoding='utf8')
    events = []
    for index, stage in enumerate(stages):
        events.extend([dict(stage=stage, status=status, wave=3, pid=123, process_started_unix=3.,
            observed_unix=3. + index * 2 + offset, wave_sha256=base.sha(wave_path),
            producer_sha256=base.sha(source), **({'wall_seconds': 1.} if status == 'COMPLETE' else {}))
            for offset, status in enumerate(('STARTED', 'COMPLETE'))])
    (cache / 'PROCESS_ATTEMPTS.jsonl').write_text(''.join(json.dumps(event) + '\n' for event in events), encoding='utf8')
    if terminal:
        (cache / 'TERMINAL_003.json').write_text(json.dumps(dict(status='COMPLETE', exit_code=0,
            wave=3, started_unix=3., finished_unix=10., wall_seconds=7., driver_pid=100)), encoding='utf8')
    return report, cache


@pytest.mark.parametrize('change', ('missing_terminal', 'failed_terminal', 'unfinished_costs', 'wrong_wave_hash'))
def test_actual_collection_refuses_incomplete_execution_before_reading_scores(tmp_path, monkeypatch, change):
    from research.b2_dataset_search_20261019_v1 import analysis as base, analysis_wave003 as wave3
    _, cache = terminal_fixture(tmp_path, terminal=change != 'missing_terminal',
        stages=('train', 'baselines') if change == 'unfinished_costs' else ('train', 'baselines', 'costs'))
    if change == 'failed_terminal':
        path = cache / 'TERMINAL_003.json'
        value = json.loads(path.read_text()); value.update(status='FAILED', exit_code=1)
        path.write_text(json.dumps(value))
    if change == 'wrong_wave_hash':
        path = cache / 'PROCESS_ATTEMPTS.jsonl'
        path.write_text(path.read_text().replace('"wave_sha256": "', '"wave_sha256": "wrong'))
    def forbidden(*args, **kwargs):
        raise AssertionError('No actual score reader may run before terminal success')
    monkeypatch.setattr(base, 'collect', forbidden)
    with pytest.raises((ValueError, FileNotFoundError), match='[Tt]erminal|[Ss]tage|[Jj]ournal'):
        wave3.collect(tmp_path)


@pytest.mark.parametrize('change', ('old_authorization', 'changed_plan', 'unregistered_width'))
def test_current_activation_binding_is_required_before_actual_score_reading(tmp_path, monkeypatch, change):
    from research.b2_dataset_search_20261019_v1 import analysis as base, analysis_wave003 as wave3
    report, cache = terminal_fixture(tmp_path)
    if change == 'old_authorization':
        path = report / 'ACTIVATION_AMENDMENT_WAVE_003.json'
        value = json.loads(path.read_text()); value['new_authorization_overrides_prior_conditional_stop'] = False
        path.write_text(json.dumps(value))
        wave_path = report / 'waves' / 'WAVE_003.json'
        value = json.loads(wave_path.read_text()); value['activation_amendment_sha256'] = base.sha(path)
        wave_path.write_text(json.dumps(value))
    elif change == 'changed_plan':
        path = report / 'CANDIDATE_PLAN_003.json'
        path.write_text(path.read_text() + '\n')
    else:
        wave_path = report / 'waves' / 'WAVE_003.json'
        value = json.loads(wave_path.read_text()); value['datasets'][0]['K_grid'] = [1, 2]
        wave_path.write_text(json.dumps(value))
    journal = cache / 'PROCESS_ATTEMPTS.jsonl'
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    for event in events:
        event['wave_sha256'] = base.sha(report / 'waves' / 'WAVE_003.json')
    journal.write_text(''.join(json.dumps(event) + '\n' for event in events))
    def forbidden(*args, **kwargs):
        raise AssertionError('Scores cannot be read under changed activation or recipe bindings')
    monkeypatch.setattr(base, 'collect', forbidden)
    with pytest.raises(ValueError, match='activation|Activation'):
        wave3.collect(tmp_path)


@pytest.mark.parametrize('positive', (True, False))
def test_collect_activates_ettm1_and_retains_unexecuted_solar_metadata(tmp_path, monkeypatch, positive):
    from research.b2_dataset_search_20261019_v1 import analysis as base, analysis_wave003 as wave3
    report, cache = terminal_fixture(tmp_path)
    rows, proof = synthetic_inputs(positive)
    plan = json.loads((report / 'CANDIDATE_PLAN_003.json').read_text())
    solar = plan['ordered_candidates'][1]
    folder = cache / 'data' / 'solar_full137'; folder.mkdir(parents=True)
    (folder / 'DATA_DIAGNOSTIC.json').write_text(json.dumps(dict(solar, status='READY', prefix_sha256='metadata_only')))
    (folder / 'PREFIX_PHYSICAL.npz').write_bytes(b'NEVER_OPEN_UNEXECUTED_NUMERIC_PREFIX')
    monkeypatch.setattr(base, 'collect', lambda repo, wave: (rows, copy.deepcopy(proof)))
    def forbidden(*args, **kwargs):
        raise AssertionError('Unexecuted Solar prefix must not be opened')
    monkeypatch.setattr(base.np, 'load', forbidden)
    _, actual = wave3.collect(tmp_path)
    solar_row = next(row for row in actual['candidates'] if row['dataset'] == 'solar_full137')
    assert solar_row['measured_in_this_report'] is False
    assert solar_row['forecast_results_counted'] == 0
    assert solar_row['status'] == ('READY_NOT_EXECUTED' if positive else 'READY_NEXT_WAVE_AFTER_NO_POSITIVE')
    assert actual['terminal_evidence']['exit_code'] == 0
    assert actual['activation_amendment_sha256'] == base.sha(report / 'ACTIVATION_AMENDMENT_WAVE_003.json')
    ettm1_row = next(row for row in actual['candidates'] if row['dataset'] == 'ETTm1')
    assert ettm1_row['status'] == ('POSITIVE_EXPLORATORY' if positive else 'NEGATIVE_EXPLORATORY')


@pytest.mark.parametrize('positive', (True, False))
def test_rendered_wave3_results_disclose_executed_grid_ancestry_mae_and_practical_mlp(positive):
    from research.b2_dataset_search_20261019_v1 import analysis_wave003 as wave3
    rows, proof = synthetic_inputs(positive)
    outputs, validation = wave3.render_outputs(rows, proof, synthetic=True)
    ko = outputs['RESULTS_KO.md'].decode()
    en = outputs['SLIDE_CONTENT_EN.md'].decode()
    assert 'ETTm1' in ko and 'H48=12시간' in ko and 'L512=128시간' in ko
    assert 'ETTh1' in ko and 'ETTm2' in ko and '역사적' in ko
    assert 'ETTm1 was executed' in en and 'H48=12h' in en and 'L512=128h' in en
    assert 'same transformer station' in en
    assert 'ETTm1 and full137-channel Solar are registry metadata only' not in en
    assert 'unrelated to ETTm1' in en
    assert 'Direct MLP' in ko and '+20.00%' in ko
    assert 'K=1' in en and 'K=2' in en and 'K=4' in en
    assert ('POSITIVE_EXPLORATORY' if positive else 'NEGATIVE_EXPLORATORY') in en
    assert b'SYNTHETIC' in outputs['ETTm1_COMPARISON.svg']
    assert validation['outputs']['RUNS.csv'] == hashlib.sha256(outputs['RUNS.csv']).hexdigest()
    assert set(outputs) == {'RUNS.csv', 'SUMMARY.csv', 'CANDIDATES.csv', 'JENA_HISTORICAL_CONTEXT.csv',
        'RESULTS_KO.md', 'SLIDE_CONTENT_EN.md', 'ETTm1_DEFAULT128.csv', 'ETTm1_COMMON_CAP.csv',
        'ETTm1_COMPARISON.png', 'ETTm1_COMPARISON.svg', 'VALIDATION.json'}


def test_output_publication_is_idempotent_and_never_overwrites_different_bytes(tmp_path):
    from research.b2_dataset_search_20261019_v1 import analysis_wave003 as wave3
    destination = tmp_path / 'reports' / wave3.NAME / 'waves' / 'WAVE_003_results'
    wave3.write_outputs({'x.csv': b'one\n'}, destination, tmp_path)
    wave3.write_outputs({'x.csv': b'one\n'}, destination, tmp_path)
    with pytest.raises(FileExistsError, match='Preserve existing'):
        wave3.write_outputs({'x.csv': b'two\n', 'new.csv': b'new\n'}, destination, tmp_path)
    assert (destination / 'x.csv').read_bytes() == b'one\n'
    assert not (destination / 'new.csv').exists()
    with pytest.raises(ValueError, match='namespace'):
        wave3.write_outputs({'x.csv': b'one\n'}, tmp_path / 'elsewhere', tmp_path)


def test_all_synthetic_output_hashes_replay_in_a_separate_cpu_process():
    from research.b2_dataset_search_20261019_v1 import analysis_wave003 as wave3
    rows, proof = synthetic_inputs()
    first, _ = wave3.render_outputs(rows, proof, synthetic=True)
    expected = {name: hashlib.sha256(value).hexdigest() for name, value in first.items()}
    code = ('import hashlib,json,sys;from research.b2_dataset_search_20261019_v1 import analysis_wave003 as a;'
        'v=json.load(sys.stdin);out,_=a.render_outputs(v["rows"],v["proof"],synthetic=True);'
        'print(json.dumps({k:hashlib.sha256(b).hexdigest() for k,b in out.items()},sort_keys=True))')
    result = subprocess.run([sys.executable, '-c', code], cwd=Path(wave3.__file__).resolve().parents[2],
        input=json.dumps(dict(rows=rows, proof=proof)), capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == expected


def test_cli_check_only_stops_before_incomplete_score_reading_and_creates_no_outputs(tmp_path):
    from research.b2_dataset_search_20261019_v1 import analysis_wave003 as wave3
    destination = tmp_path / 'reports' / wave3.NAME / 'waves' / 'WAVE_003_results'
    result = subprocess.run([sys.executable, '-m', 'research.b2_dataset_search_20261019_v1.analysis_wave003',
        '--repo', str(tmp_path), '--wave', '3', '--check-only', '--output-dir', str(destination)],
        cwd=Path(wave3.__file__).resolve().parents[2], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Terminal success is required' in result.stderr
    assert not destination.exists()


def recovery_fixture(repo):
    from research.b2_dataset_search_20261019_v1 import analysis as base
    report, cache = terminal_fixture(repo)
    previous_path = cache / 'TERMINAL_003.json'
    previous = json.loads(previous_path.read_text())
    previous.update(status='FAILED', exit_code=1073807364)
    previous_path.write_text(json.dumps(previous))
    retry = dict(status='COMPLETE', exit_code=0, wave=3, started_unix=12., finished_unix=25., wall_seconds=13.)
    (cache / 'TERMINAL_003_ATTEMPT_002.json').write_text(json.dumps(retry))
    registration = dict(status='REGISTERED_BEFORE_RESUMED_PRODUCER', wave=3, attempt=2,
        previous_terminal=dict(reference='cache://TERMINAL_003.json', sha256=base.sha(previous_path)),
        next_terminal_reference='cache://TERMINAL_003_ATTEMPT_002.json', unchanged_producer_sources=True,
        wave_sha256=base.sha(report / 'waves' / 'WAVE_003.json'),
        producer_sha256=base.sha(repo / 'research' / base.NAME / 'run.py'), registered_unix=11.)
    amendment_path = report / 'EXECUTION_RECOVERY_WAVE_003_ATTEMPT_002.json'
    amendment_path.write_text(json.dumps(registration))
    journal = cache / 'PROCESS_ATTEMPTS.jsonl'
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    resumed = [dict(event, observed_unix=event['observed_unix']+10.) for event in events]
    journal.write_text(''.join(json.dumps(event)+'\n' for event in events+resumed))
    return report, cache, amendment_path


def test_registered_recovery_selects_retry_and_preserves_failed_terminal_bytes(tmp_path, monkeypatch):
    from research.b2_dataset_search_20261019_v1 import analysis as base, analysis_wave003 as wave3
    _, cache, amendment_path = recovery_fixture(tmp_path)
    original = (cache / 'TERMINAL_003.json').read_bytes()
    def forbidden(*args, **kwargs): raise AssertionError('Terminal verification must not open numeric arrays')
    monkeypatch.setattr(base.np, 'load', forbidden)
    evidence = wave3.terminal_evidence(tmp_path)
    assert evidence['exit_code'] == 0
    assert evidence['source_reference'].endswith('/TERMINAL_003_ATTEMPT_002.json')
    assert set(evidence['stage_completions']) == {'train', 'baselines', 'costs'}
    recovery = evidence['execution_recovery']
    assert recovery['binding']['sha256'] == base.sha(amendment_path)
    assert recovery['previous_terminal']['exit_code'] == 1073807364
    assert recovery['previous_terminal']['sha256'] == hashlib.sha256(original).hexdigest()
    assert (cache / 'TERMINAL_003.json').read_bytes() == original


@pytest.mark.parametrize('change', ('unregistered_retry', 'missing_retry', 'failed_retry', 'retry_wrong_wave',
    'wrong_status', 'wrong_wave', 'wrong_attempt', 'failed_hash', 'failed_path', 'next_path', 'changed_source',
    'changed_wave', 'source_flag', 'registration_before_failure', 'registration_after_retry', 'registration_nan',
    'registration_bool', 'previous_success', 'previous_bool_exit', 'retry_nonfinite_time', 'stage_before_retry',
    'unfinished_costs'))
def test_recovery_rejects_invalid_lineage_before_numeric_reading(tmp_path, monkeypatch, change):
    from research.b2_dataset_search_20261019_v1 import analysis as base, analysis_wave003 as wave3
    report, cache, amendment_path = recovery_fixture(tmp_path)
    registration = json.loads(amendment_path.read_text())
    retry_path = cache / 'TERMINAL_003_ATTEMPT_002.json'
    if change == 'unregistered_retry': amendment_path.unlink()
    elif change == 'missing_retry': retry_path.unlink()
    elif change in ('failed_retry', 'retry_wrong_wave', 'retry_nonfinite_time'):
        retry = json.loads(retry_path.read_text())
        if change == 'failed_retry': retry.update(status='FAILED', exit_code=1)
        elif change == 'retry_wrong_wave': retry['wave'] = 2
        else: retry['started_unix'] = float('nan')
        retry_path.write_text(json.dumps(retry))
    elif change == 'changed_source':
        (tmp_path / 'research' / base.NAME / 'run.py').write_text('# Changed producer')
    elif change == 'changed_wave': (report / 'waves' / 'WAVE_003.json').write_text('{}')
    elif change in ('previous_success', 'previous_bool_exit'):
        previous_path = cache / 'TERMINAL_003.json'
        previous = json.loads(previous_path.read_text())
        previous.update(status='COMPLETE', exit_code=0) if change == 'previous_success' else previous.update(exit_code=True)
        previous_path.write_text(json.dumps(previous))
        registration['previous_terminal']['sha256'] = base.sha(previous_path)
    elif change in ('stage_before_retry', 'unfinished_costs'):
        journal = cache / 'PROCESS_ATTEMPTS.jsonl'
        events = [json.loads(line) for line in journal.read_text().splitlines()]
        events[-1].update(observed_unix=8.) if change == 'stage_before_retry' else events[-1].update(status='FAILED')
        journal.write_text(''.join(json.dumps(event)+'\n' for event in events))
    else:
        key, value = {'wrong_status': ('status', 'COMPLETE'), 'wrong_wave': ('wave', 2),
            'wrong_attempt': ('attempt', True), 'next_path': ('next_terminal_reference', 'cache://elsewhere.json'),
            'source_flag': ('unchanged_producer_sources', False),
            'registration_before_failure': ('registered_unix', 10.), 'registration_after_retry': ('registered_unix', 13.),
            'registration_nan': ('registered_unix', float('nan')), 'registration_bool': ('registered_unix', True),
            'failed_hash': ('previous_terminal', dict(reference='cache://TERMINAL_003.json', sha256='changed')),
            'failed_path': ('previous_terminal', dict(reference='cache://../TERMINAL_003.json',
                sha256=registration['previous_terminal']['sha256']))}[change]
        registration[key] = value
    if change != 'unregistered_retry': amendment_path.write_text(json.dumps(registration))
    def forbidden(*args, **kwargs): raise AssertionError('No numeric reader may precede valid recovery completion')
    monkeypatch.setattr(base, 'collect', forbidden)
    monkeypatch.setattr(base.np, 'load', forbidden)
    with pytest.raises((ValueError, FileNotFoundError)):
        wave3.collect(tmp_path)


def recovery_chain_fixture(repo):
    from research.b2_dataset_search_20261019_v1 import analysis as base
    report, cache, preceding_path = recovery_fixture(repo)
    previous_path = cache / 'TERMINAL_003_ATTEMPT_002.json'
    previous = json.loads(previous_path.read_text()); previous.update(status='FAILED', exit_code=1)
    previous_path.write_text(json.dumps(previous))
    retry_path = cache / 'TERMINAL_003_ATTEMPT_003.json'
    retry_path.write_text(json.dumps(dict(status='COMPLETE', exit_code=0, wave=3,
        started_unix=27., finished_unix=40., wall_seconds=13.)))
    adapter = repo / 'research' / base.NAME / 'resume_wave003.py'; adapter.write_text('# Synthetic identity adapter')
    registration = json.loads(preceding_path.read_text())
    registration.update(attempt=3, previous_terminal=dict(reference='cache://' + previous_path.name, sha256=base.sha(previous_path)),
        next_terminal_reference='cache://' + retry_path.name, registered_unix=26.,
        preceding_recovery=dict(reference='report://' + preceding_path.name, sha256=base.sha(preceding_path)),
        execution_adapter=dict(reference='repo://research/' + base.NAME + '/resume_wave003.py', sha256=base.sha(adapter)),
        identity_serialization_repair='JSON_ROUNDTRIP_ONLY_EXACT_TRAIN_IDENTITY')
    amendment_path = report / 'EXECUTION_RECOVERY_WAVE_003_ATTEMPT_003.json'
    amendment_path.write_text(json.dumps(registration))
    journal = cache / 'PROCESS_ATTEMPTS.jsonl'
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    events += [dict(event, observed_unix=event['observed_unix']+15.) for event in events[-6:]]
    journal.write_text(''.join(json.dumps(event)+'\n' for event in events))
    return report, cache, amendment_path, preceding_path


def test_third_attempt_requires_and_exposes_both_preserved_failed_lineages(tmp_path):
    from research.b2_dataset_search_20261019_v1 import analysis as base, analysis_wave003 as wave3
    _, cache, amendment_path, preceding_path = recovery_chain_fixture(tmp_path)
    originals = {name: (cache / name).read_bytes() for name in ('TERMINAL_003.json', 'TERMINAL_003_ATTEMPT_002.json')}
    evidence = wave3.terminal_evidence(tmp_path)
    assert evidence['source_reference'].endswith('/TERMINAL_003_ATTEMPT_003.json')
    recovery = evidence['execution_recovery']
    assert recovery['binding']['sha256'] == base.sha(amendment_path)
    assert recovery['previous_terminal']['exit_code'] == 1
    assert len(recovery['preceding_recoveries']) == 1
    assert recovery['preceding_recoveries'][0]['binding']['sha256'] == base.sha(preceding_path)
    assert recovery['preceding_recoveries'][0]['previous_terminal']['exit_code'] == 1073807364
    assert all((cache / name).read_bytes() == value for name, value in originals.items())


@pytest.mark.parametrize('change', ('unregistered_third', 'missing_third_terminal', 'failed_third', 'missing_preceding',
    'preceding_hash', 'preceding_path', 'original_hash', 'preceding_late_registration', 'wrong_previous_path',
    'third_registration_early', 'third_registration_late', 'stage_outside_third', 'adapter_path', 'adapter_hash',
    'adapter_source_changed', 'adapter_missing', 'adapter_scope'))
def test_third_attempt_cannot_skip_prior_registration_failure_or_current_stage_gates(tmp_path, monkeypatch, change):
    from research.b2_dataset_search_20261019_v1 import analysis as base, analysis_wave003 as wave3
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
        preceding_path.write_text(json.dumps(preceding)); registration['preceding_recovery']['sha256'] = base.sha(preceding_path)
    elif change == 'stage_outside_third':
        journal = cache / 'PROCESS_ATTEMPTS.jsonl'
        events = [json.loads(line) for line in journal.read_text().splitlines()]; events[-1]['observed_unix'] = 20.
        journal.write_text(''.join(json.dumps(event)+'\n' for event in events))
    elif change == 'preceding_hash': registration['preceding_recovery']['sha256'] = 'changed'
    elif change == 'preceding_path': registration['preceding_recovery']['reference'] = 'report://../elsewhere.json'
    elif change == 'wrong_previous_path': registration['previous_terminal']['reference'] = 'cache://TERMINAL_003.json'
    elif change == 'adapter_path': registration['execution_adapter']['reference'] = 'repo://../resume_wave003.py'
    elif change == 'adapter_hash': registration['execution_adapter']['sha256'] = 'changed'
    elif change == 'adapter_scope': registration['identity_serialization_repair'] = 'ALTER_TRAINING'
    elif change == 'adapter_source_changed': (tmp_path / 'research' / base.NAME / 'resume_wave003.py').write_text('# Changed')
    elif change == 'adapter_missing': (tmp_path / 'research' / base.NAME / 'resume_wave003.py').unlink()
    else: registration['registered_unix'] = 25. if change == 'third_registration_early' else 28.
    if change != 'unregistered_third': amendment_path.write_text(json.dumps(registration))
    def forbidden(*args, **kwargs): raise AssertionError('Unproven recovery cannot read numeric scores')
    monkeypatch.setattr(base, 'collect', forbidden); monkeypatch.setattr(base.np, 'load', forbidden)
    with pytest.raises((ValueError, FileNotFoundError)):
        wave3.collect(tmp_path)
