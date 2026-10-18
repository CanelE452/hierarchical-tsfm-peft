import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.memory_budget_20261018_v1 import common
from research.memory_budget_20261018_v1 import training


@pytest.fixture
def context(tmp_path, monkeypatch):
    report, cache = tmp_path/'reports', tmp_path/'cache'
    report.mkdir()
    cache.mkdir()
    source = tmp_path/'producer.py'
    source.write_text('frozen producer\n', encoding='utf-8')
    (report/'PROTOCOL.md').write_text('registered protocol\n', encoding='utf-8')
    (report/'A_CONFIRMED.json').write_text('{"status":"COMPLETE"}\n', encoding='utf-8')
    deadline = time.time()+10000
    monkeypatch.setattr(common, 'SCIENTIFIC_DEADLINE', deadline)
    ctx = dict(repo=tmp_path, asset=tmp_path/'asset', report=report, cache=cache,
        scientific_source_hashes={'producer.py': common.sha(source)}, deadline=deadline,
        lease=SimpleNamespace(check=lambda: None))
    gate = dict(status='PROCEED', eligible=True, phase_a_all_confirmed=True,
        protocol_binding=common.binding(report/'PROTOCOL.md', ctx),
        phase_a_evidence=[common.binding(report/'A_CONFIRMED.json', ctx)],
        budget=dict(scientific_deadline_unix=deadline, estimated_training_seconds=1000.,
                    estimated_measurement_seconds=1000.))
    write_gate(ctx, gate)
    return ctx


def write_gate(ctx, value):
    value = {k: v for k, v in value.items() if k != 'decision_identity'}
    value['decision_identity'] = common.fingerprint(value)
    path = ctx['report']/'PHASE_B_DECISION.json'
    common.atomic_json(path, value)
    return path


def fake_runtime(ctx, calls, *, fail_search=None, mutate_source=False):
    foundation = SimpleNamespace(assert_frozen_unchanged=lambda: calls.append('F_CHECK'))
    resource = dict(ctx, foundation=foundation)

    def search(job, run_ctx, epochs):
        calls.append(('search', job['job_id'], epochs))
        if mutate_source:
            (ctx['repo']/'producer.py').write_text('changed producer\n', encoding='utf-8')
        cp = ctx['cache']/(job['job_id']+'.pt')
        cp.write_bytes(b'checkpoint')
        return dict(status='PARTIAL_CHECKPOINTED' if fail_search == job['job_id'] else 'COMPLETE',
            cap=3, spec=dict(latent_k=job['K'], family='B2', revin=False, seed=job['seed'],
                            dataset='jena', device='cuda', epochs=[.25, 1., 3.]),
            V_used=False, selected_epoch=1.,
            summary=dict(selected_astop_mse=.2), checkpoint_binding=common.binding(cp, ctx),
            curves=[dict(epoch=0., astop_mse=.4), dict(epoch=1., astop_mse=.2),
                    dict(epoch=3., astop_mse=.3)])

    def refit(job, run_ctx, selected, cap):
        assert len([x for x in calls if isinstance(x, tuple) and x[0] == 'search']) == 4
        assert common.sha(run_ctx['selection_decision']['path']) == run_ctx['selection_decision']['sha256']
        calls.append(('refit', job['job_id']))
        prediction = ctx['cache']/(job['job_id']+'.npz')
        prediction.write_bytes(b'private predictions')
        bundle = ctx['cache']/(job['job_id']+'_bundle.pt')
        bundle.write_bytes(b'private trained bundle')
        receipt = ctx['report']/(job['job_id']+'_RECEIPT.json')
        receipt.write_text('{}', encoding='utf-8')
        row = dict(family='B2', revin=False, seed=job['seed'], dataset='jena', K=job['K'], cap=3,
            C=21, L=512, H=48, device='cuda', evaluation='V_select', fit='fresh_A60',
            independent_FP64='PASS', core_dtype='float32', prediction_dtype='float32',
            prediction_binding=common.binding(prediction, ctx), MSE=.3, MAE=.2,
            selected_epoch=1., target_trained_params=34709+168*job['K'],
            P_params=34709, E_params=21*job['K'], D_params=147*job['K'],
            search_seconds=200., refit_seconds=80.)
        return dict(status='COMPLETE', rows=[row], bundle_binding=common.binding(bundle, ctx),
            selection_receipt_binding=common.binding(receipt, ctx),
            selection_decision_binding=common.binding(run_ctx['selection_decision']['path'], ctx),
            fit_receipt=dict(normalization_fit_end=12579, model_state_sha='a'*64),
            model_sha='b'*64, refit_summary=dict(contract=dict(initialization_sha='c'*64)),
            recipe=dict(optimizer='AdamW', weight_decay=0., gradient_clip=1., dtype='float32'))

    return SimpleNamespace(search=search, _refit_predict=refit), resource


def test_registered_jobs_are_exactly_four_independent_off_models():
    jobs = training.registered_jobs()
    assert {(job['K'], job['seed']) for job in jobs} == {
        (3, 92601), (3, 92602), (12, 92601), (12, 92602)}
    assert all(job['family'] == 'B2' and job['revin'] is False for job in jobs)
    assert len({job['job_id'] for job in jobs}) == 4


@pytest.mark.parametrize('field,value', [('eligible', False), ('eligible', 1),
    ('phase_a_all_confirmed', False), ('status', 'SKIPPED_GATE')])
def test_gate_rejects_ineligible_or_unconfirmed_before_runtime(context, field, value):
    path = context['report']/'PHASE_B_DECISION.json'
    gate = common.read(path)
    gate[field] = value
    write_gate(context, gate)
    calls = []
    with pytest.raises(ValueError, match='eligible'):
        training.execute(context, path, lambda ctx: fake_runtime(ctx, calls))
    assert calls == []


def test_gate_requires_exact_identity_protocol_and_evidence_bytes(context):
    path = context['report']/'PHASE_B_DECISION.json'
    gate = common.read(path)
    gate['eligible'] = False
    common.atomic_json(path, gate)
    with pytest.raises(ValueError, match='identity'):
        training.validate_gate(path, context)
    gate['eligible'] = True
    write_gate(context, gate)
    (context['report']/'PROTOCOL.md').write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError):
        training.validate_gate(path, context)


def test_gate_rejects_changed_evidence_and_empty_evidence(context):
    path = context['report']/'PHASE_B_DECISION.json'
    gate = common.read(path)
    (context['report']/'A_CONFIRMED.json').write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError):
        training.validate_gate(path, context)
    gate['phase_a_evidence'] = []
    write_gate(context, gate)
    with pytest.raises(ValueError, match='bound Phase A'):
        training.validate_gate(path, context)


def test_gate_rejects_fake_path_and_insufficient_full_phase_budget(context):
    path = context['report']/'PHASE_B_DECISION.json'
    other = context['report']/'OTHER.json'
    other.write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match='registered report decision'):
        training.validate_gate(other, context)
    with pytest.raises(TimeoutError, match='full Phase B'):
        training.validate_gate(path, context, now=common.SCIENTIFIC_DEADLINE-1999)


@pytest.mark.parametrize('estimate', [-1., 0., float('inf'), True, None])
def test_gate_rejects_invalid_budget_estimates(context, estimate):
    path = context['report']/'PHASE_B_DECISION.json'
    gate = common.read(path)
    if estimate == float('inf'):
        gate['budget']['estimated_training_seconds'] = 'Infinity'
    else:
        gate['budget']['estimated_training_seconds'] = estimate
    write_gate(context, gate)
    with pytest.raises(ValueError, match='finite positive'):
        training.validate_gate(path, context)


def test_all_four_searches_are_sealed_before_any_refit_and_rows_have_new_context(context):
    calls = []
    result = training.execute(context, context['report']/'PHASE_B_DECISION.json',
                             lambda ctx: fake_runtime(ctx, calls))
    assert result['status'] == 'COMPLETE' and result['trained_models'] == 4
    operations = [x[0] for x in calls if isinstance(x, tuple)]
    assert operations == ['search']*4+['refit']*4
    rows = common.read(context['cache']/'B_PHASE_ROWS.json')
    assert len(rows) == 4 and len({row['run_id'] for row in rows}) == 4
    assert all(row['source_context'] == 'MEMORY_BUDGET_PHASE_B' and row['revin'] is False for row in rows)
    assert all(row['bundle_binding']['reference'].startswith('cache://') for row in rows)
    assert result['new_training_search_seconds'] == 800.
    public = (context['report']/'B_TRAINING_SUMMARY.json').read_text(encoding='utf-8')
    assert str(context['repo']) not in public


def test_partial_search_preserves_attempt_and_never_refits(context):
    calls = []
    failed = training.registered_jobs()[2]['job_id']
    with pytest.raises(RuntimeError, match='search incomplete'):
        training.execute(context, context['report']/'PHASE_B_DECISION.json',
            lambda ctx: fake_runtime(ctx, calls, fail_search=failed))
    assert not any(isinstance(x, tuple) and x[0] == 'refit' for x in calls)
    assert not (context['report']/'B_SELECTIONS_SEALED.json').exists()
    assert not (context['cache']/'B_PHASE_ROWS.json').exists()
    attempts = [json.loads(line) for line in (context['cache']/'B_TRAINING_ATTEMPTS.jsonl').read_text().splitlines()]
    assert [row['event'] for row in attempts] == ['STARTED', 'FAILED']
    assert attempts[-1]['phase'] == 'SEARCH_'+failed


def test_producer_change_stops_before_second_search(context):
    calls = []
    with pytest.raises(ValueError, match='producer source bytes changed'):
        training.execute(context, context['report']/'PHASE_B_DECISION.json',
            lambda ctx: fake_runtime(ctx, calls, mutate_source=True))
    assert len([x for x in calls if isinstance(x, tuple)]) == 1


def test_same_frozen_sources_can_resume_without_replacing_public_identity(context):
    path = context['report']/'PHASE_B_DECISION.json'
    for _ in range(2):
        calls = []
        training.execute(context, path, lambda ctx: fake_runtime(ctx, calls))
    attempts = [json.loads(line) for line in (context['cache']/'B_TRAINING_ATTEMPTS.jsonl').read_text().splitlines()]
    assert [row['event'] for row in attempts] == ['STARTED', 'COMPLETE', 'STARTED', 'COMPLETE']


def test_parent_lease_check_precedes_runtime(context):
    def reject():
        raise RuntimeError('parent authorization missing')
    context['lease'] = SimpleNamespace(check=reject)
    with pytest.raises(RuntimeError, match='parent authorization'):
        training.execute(context, context['report']/'PHASE_B_DECISION.json',
                         lambda ctx: pytest.fail('runtime must not load'))


@pytest.mark.parametrize('key,value', [('K', 6), ('revin', True), ('prediction_dtype', 'float16'),
    ('independent_FP64', 'FAIL'), ('selected_epoch', 3.), ('C', 32)])
def test_refit_rejects_changed_axes_precision_score_or_selection(context, key, value):
    calls = []
    def runtime(ctx):
        helper, resource = fake_runtime(ctx, calls)
        original = helper._refit_predict
        def changed(*args):
            result = original(*args)
            result['rows'][0][key] = value
            return result
        helper._refit_predict = changed
        return helper, resource
    with pytest.raises(ValueError, match='unregistered model axes'):
        training.execute(context, context['report']/'PHASE_B_DECISION.json', runtime)
    assert not (context['cache']/'B_PHASE_ROWS.json').exists()


def test_cli_refuses_unregistered_cache_before_source_context(tmp_path, monkeypatch):
    monkeypatch.setattr(common, 'source_context', lambda asset: pytest.fail('must not load context'))
    with pytest.raises(SystemExit):
        training.main(['--asset-root', str(tmp_path), '--gate', str(tmp_path/'gate.json'),
                       '--resource-cache', str(tmp_path/'wrong')])
