import copy
import hashlib
import json

import pytest

import evaluate_protected as ep
from protected_protocol import validate_protected_evaluation


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def prepared(tmp_path):
    root = tmp_path
    here = root / 'research' / 'experiment'
    here.mkdir(parents=True)
    data_path = root / '.cache' / 'synthetic_data.npz'
    data_path.parent.mkdir()
    data_path.write_bytes(b'synthetic bytes; deliberately not a valid NPZ')
    write_json(here / 'data_contract.json', {'datasets': {'bdg2_bull_office': {
        'npz': '.cache/synthetic_data.npz', 'selected_count': 16,
    }}})
    sources = ('model.py', 'run.py', 'protocol.json', 'evaluate_protected.py',
               'evaluate_development.py', 'masked_linear.py', 'protected_protocol.py', 'data_prepare.py')
    for name in sources:
        (here / name).write_text('synthetic ' + name, encoding='utf-8')
    hashes = {name: sha(here / name) for name in ('data_contract.json', 'protocol.json', 'model.py', 'run.py')}
    hashes['protected_npz'] = sha(data_path)
    fits = [dict(id=f'bull_{arm}_1', dataset='bull', arm=arm, seed=1,
                 lr=0.001, latent=8, epochs=120, residual_rank=32, loss='mse')
            for arm in ('residual', 'raw_bypass')]
    source_names = ('evaluate_protected.py', 'evaluate_development.py', 'masked_linear.py',
                    'protected_protocol.py', 'data_prepare.py')
    seal = dict(schema_version=1, status='sealed', method_fixed=True, hashes=hashes,
                allowed_fits=fits,
                evaluation=dict(periods=['e1', 'e2'], block_origins=7, linear_penalty=0.001,
                                seasonal_period=24, primary_metric='train_std_equal_channel_macro_mse', latent=8),
                evaluation_source_hashes={name: sha(here / name) for name in source_names})
    seal_path = here / 'final_seal.json'
    write_json(seal_path, seal)
    receipt = validate_protected_evaluation(seal_path, [spec['id'] for spec in fits], hashes)
    write_json(here / 'protected_adaptation_receipt.json', receipt)
    for spec in fits:
        checkpoint = root / '.cache' / spec['id'] / 'best.pt'
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(b'synthetic checkpoint; no model deserialization')
        result = dict(spec, status='complete', checkpoint=str(checkpoint),
                      checkpoint_sha256=sha(checkpoint), data_sha256=hashes['protected_npz'],
                      protected_receipt=copy.deepcopy(receipt))
        write_json(here / 'runs' / spec['id'] / 'result.json', result)
    return here, root, seal


def first_result(prepared):
    here, _, seal = prepared
    path = here / 'runs' / seal['allowed_fits'][0]['id'] / 'result.json'
    return path, json.loads(path.read_text(encoding='utf-8'))


def refresh_receipts(prepared):
    here, _, seal = prepared
    receipt = validate_protected_evaluation(here / 'final_seal.json',
                                           [spec['id'] for spec in seal['allowed_fits']], seal['hashes'])
    write_json(here / 'protected_adaptation_receipt.json', receipt)
    for spec in seal['allowed_fits']:
        path = here / 'runs' / spec['id'] / 'result.json'
        row = json.loads(path.read_text(encoding='utf-8'))
        row['protected_receipt'] = receipt
        write_json(path, row)


def test_preflight_uses_all_sealed_fits_without_loading_fake_archives(prepared):
    here, root, seal = prepared
    plan = ep.preflight(here, root)
    assert [row['id'] for row in plan['results']] == ['bull_residual_1', 'bull_raw_bypass_1']
    assert plan['seal']['allowed_fits'] == seal['allowed_fits']
    assert plan['receipt']['seal_sha256'] == sha(here / 'final_seal.json')
    assert not (here / 'protected_evaluation_attempt.json').exists()


@pytest.mark.parametrize('field,value', [
    ('status', 'failed'), ('lr', 0.01), ('seed', 2), ('epochs', 121),
    ('microbatch_origins', 4), ('data_sha256', 'f' * 64), ('checkpoint_sha256', 'f' * 64),
])
def test_bad_result_blocks_execution_before_data_or_model_use(prepared, monkeypatch, field, value):
    here, root, _ = prepared
    path, result = first_result(prepared)
    result[field] = value
    write_json(path, result)
    monkeypatch.setattr(ep, '_evaluate_plan', lambda *args: pytest.fail('execution reached'))
    with pytest.raises(ValueError):
        ep.run_evaluation(here, root)
    assert not (here / 'protected_evaluation_attempt.json').exists()


@pytest.mark.parametrize('location', ['result', 'adaptation'])
def test_receipt_must_match_the_original_adaptation_seal(prepared, location):
    here, root, _ = prepared
    if location == 'result':
        path, row = first_result(prepared)
        row['protected_receipt']['seal_sha256'] = 'f' * 64
    else:
        path = here / 'protected_adaptation_receipt.json'
        row = json.loads(path.read_text(encoding='utf-8'))
        row['seal_sha256'] = 'f' * 64
    write_json(path, row)
    with pytest.raises(ValueError, match='receipt'):
        ep.preflight(here, root)


@pytest.mark.parametrize('name', [
    'evaluate_protected.py', 'evaluate_development.py', 'masked_linear.py',
    'protected_protocol.py', 'data_prepare.py', 'model.py', 'run.py', 'protocol.json',
])
def test_changed_execution_source_blocks_preflight(prepared, name):
    here, root, _ = prepared
    (here / name).write_text('changed execution source', encoding='utf-8')
    with pytest.raises(ValueError, match='hash'):
        ep.preflight(here, root)


def test_changed_data_bytes_block_preflight_without_decoding(prepared):
    here, root, _ = prepared
    (root / '.cache' / 'synthetic_data.npz').write_bytes(b'changed archive')
    with pytest.raises(ValueError, match='hash'):
        ep.preflight(here, root)


def test_changed_checkpoint_bytes_block_preflight_without_deserializing(prepared):
    here, root, _ = prepared
    _, row = first_result(prepared)
    from pathlib import Path
    Path(row['checkpoint']).write_bytes(b'changed checkpoint')
    with pytest.raises(ValueError, match='checkpoint'):
        ep.preflight(here, root)


@pytest.mark.parametrize('field,value', [
    ('periods', ['e2', 'e1']), ('block_origins', 8), ('linear_penalty', 0.1),
    ('seasonal_period', 48), ('primary_metric', 'pooled_mse'), ('latent', 0),
])
def test_evaluation_contract_cannot_silently_change(prepared, field, value):
    here, root, seal = prepared
    seal['evaluation'][field] = value
    write_json(here / 'final_seal.json', seal)
    refresh_receipts(prepared)
    with pytest.raises(ValueError):
        ep.preflight(here, root)


def test_missing_fit_is_not_dropped_from_the_evaluation(prepared):
    here, root, seal = prepared
    (here / 'runs' / seal['allowed_fits'][1]['id'] / 'result.json').unlink()
    with pytest.raises(ValueError, match='result'):
        ep.preflight(here, root)


def test_custom_preflight_paths_cannot_load_the_real_experiment(prepared):
    here, root, _ = prepared
    plan = ep.preflight(here, root)
    with pytest.raises(ValueError, match='checkout'):
        ep._evaluate_plan(plan, root / 'unused', {})


def test_fit_id_cannot_collide_with_a_baseline_cache_key(prepared):
    here, root, seal = prepared
    _, row = first_result(prepared)
    seal['allowed_fits'][0]['id'] = 'f0'
    row['id'] = 'f0'
    write_json(here / 'runs' / 'f0' / 'result.json', row)
    write_json(here / 'final_seal.json', seal)
    refresh_receipts(prepared)
    with pytest.raises(ValueError, match='reserved'):
        ep.preflight(here, root)


@pytest.mark.parametrize('existing', ['protected_evaluation_attempt.json', 'protected_evaluation.json'])
def test_existing_attempt_or_report_cannot_be_overwritten(prepared, monkeypatch, existing):
    here, root, _ = prepared
    path = here / existing
    path.write_text('preserve this original', encoding='utf-8')
    monkeypatch.setattr(ep, '_evaluate_plan', lambda *args: pytest.fail('execution reached'))
    with pytest.raises(FileExistsError):
        ep.run_evaluation(here, root)
    assert path.read_text(encoding='utf-8') == 'preserve this original'


def test_exclusive_claim_and_failure_preserve_partial_artifacts(prepared, monkeypatch):
    here, root, _ = prepared
    calls = []
    def fail_after_partial(plan, local, record):
        calls.append(1)
        (local / 'partial.npz').write_bytes(b'preserve partial prediction')
        record['partial_rows'] = [{'period': 'e1', 'fit': 'bull_residual_1'}]
        raise RuntimeError('synthetic evaluation failure')
    monkeypatch.setattr(ep, '_evaluate_plan', fail_after_partial)
    with pytest.raises(RuntimeError, match='synthetic evaluation failure'):
        ep.run_evaluation(here, root)
    attempt_path = here / 'protected_evaluation_attempt.json'
    record = json.loads(attempt_path.read_text(encoding='utf-8'))
    assert record['status'] == 'failed'
    assert 'synthetic evaluation failure' in record['error']
    assert record['partial_rows'] == [{'period': 'e1', 'fit': 'bull_residual_1'}]
    assert (root / '.cache' / 'tsfm_peft_development_20260926' / 'protected_evaluation' / 'partial.npz').read_bytes() == b'preserve partial prediction'
    with pytest.raises(FileExistsError):
        ep.run_evaluation(here, root)
    assert calls == [1]


def test_synthetic_full_pipeline_preserves_all_fits_periods_and_effects(prepared, monkeypatch):
    import numpy as np
    import torch
    import evaluate_development
    import model as model_module
    import run

    here, root, seal = prepared
    specs = seal['allowed_fits']
    values = np.random.default_rng(72).normal(size=(760, 16)).astype(np.float32)
    data = dict(x=values, finite=np.ones_like(values, dtype=bool),
                train_start=512, train_end=568, train_origins=np.arange(512, 520),
                eval_e1_origins=np.arange(600, 608), eval_e2_origins=np.arange(680, 688),
                columns=np.array([f'channel_{i}' for i in range(16)]),
                _path=root / '.cache' / 'synthetic_data.npz')
    events = []
    monkeypatch.setattr(ep, 'HERE', here)
    monkeypatch.setattr(ep, 'ROOT', root)

    def load_data(dataset):
        assert dataset == 'bull'
        events.append('load_synthetic_data')
        return data

    def fit_baselines(data_arg, basis, local, penalty):
        assert data_arg is data and penalty == 0.001
        assert basis.shape == (16, 8)
        events.append('cpu_baseline_fit')
        shared = dict(weight=np.zeros((512, 48)), bias=np.full(48, 0.2))
        factor = dict(basis=basis, common=shared,
                      residual=dict(weight=np.zeros((512, 48)), bias=np.full(48, 0.3)))
        return ({'shared_linear': shared, 'factor_linear': factor},
                {arm: dict(status='complete', fit_seconds=0.0)
                 for arm in ('shared_linear', 'factor_linear')})

    class FakeJob:
        def __init__(self, label):
            assert label == 'protected_evaluation'
            self.record = {'elapsed_s': 0.125}

        def __enter__(self):
            assert 'cpu_baseline_fit' in events
            events.append('gpu_job_enter')
            return self

        def __exit__(self, *args):
            events.append('gpu_job_exit')

    class FakeModel:
        def __init__(self, arm):
            self.arm = arm

        def restore_adapter(self, state):
            assert state == {'synthetic': self.arm}
            events.append('restore_' + self.arm)

    def make_model(arm, basis, residual_rank=8):
        assert 'gpu_job_enter' in events and basis.shape == (16, 8)
        events.append('make_' + arm)
        return FakeModel(arm)

    def load_checkpoint(path, **kwargs):
        from pathlib import Path
        fit_id = Path(path).parent.name
        spec = next(spec for spec in specs if spec['id'] == fit_id)
        events.append('checkpoint_' + fit_id)
        return dict(spec=copy.deepcopy(spec), basis=np.eye(16, 8, dtype=np.float32),
                    state={'synthetic': spec['arm']})

    def measured(model, data_arg, origins, job, label):
        assert data_arg is data and isinstance(job, FakeJob)
        assert 'gpu_job_enter' in events and 'gpu_job_exit' not in events
        target = np.stack([values[o:o + 48] for o in origins])
        delta = {'f0': 0.75, 'residual': 0.5, 'raw_bypass': 1.0}[model.arm]
        events.append('measure_' + model.arm + '_' + str(origins[0]))
        return {}, target.astype(np.float64) + delta, {'wall_seconds': 0.0}

    original_effects = ep._effects
    def effects(*args):
        assert events[-1] == 'gpu_job_exit' or events[-1] == 'paired_effects'
        events.append('paired_effects')
        return original_effects(*args)

    monkeypatch.setattr(run, 'load_data', load_data)
    monkeypatch.setattr(run, 'GPUJob', FakeJob)
    monkeypatch.setattr(model_module, 'make_model', make_model)
    monkeypatch.setattr(evaluate_development, 'measured_evaluate', measured)
    monkeypatch.setattr(torch, 'load', load_checkpoint)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: events.append('cuda_cleanup_stub'))
    monkeypatch.setattr(ep, '_fit_baselines', fit_baselines)
    monkeypatch.setattr(ep, '_effects', effects)
    report = ep.run_evaluation(here, root)

    assert report['status'] == 'complete'
    assert len(report['rows']) == 12
    assert events[:2] == ['load_synthetic_data', 'cpu_baseline_fit']
    assert [item for item in events if item.startswith('checkpoint_')] == [
        'checkpoint_bull_residual_1', 'checkpoint_bull_raw_bypass_1']
    assert [item for item in events if item.startswith('restore_')] == ['restore_residual', 'restore_raw_bypass']
    assert events.count('cuda_cleanup_stub') == 3
    assert events.count('paired_effects') == 2
    assert report['gpu_accounted_seconds'] == 0.125
    local = root / '.cache' / 'tsfm_peft_development_20260926' / 'protected_evaluation'
    assert len(list(local.glob('*.npz'))) == 12
    for period in ('e1', 'e2'):
        rows = [row for row in report['rows'] if row['period'] == period]
        assert {row['arm'] for row in rows} == {'f0', 'residual', 'raw_bypass', 'shared_linear', 'factor_linear', 'seasonal24'}
        residual_row = next(row for row in rows if row['arm'] == 'residual')
        assert residual_row['seed'] == 1
        assert residual_row['scores']['channel_mse'] == pytest.approx([0.25] * 16)
        assert residual_row['scores']['channel_mae'] == pytest.approx([0.5] * 16)
        with np.load(local / residual_row['prediction_cache'], allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['origins'], data[f'eval_{period}_origins'])
        assert set(report['periods'][period]['residual_vs']) == {'f0', 'raw_bypass', 'shared_linear', 'factor_linear', 'seasonal24'}
        assert all(effect['status'] == 'available' and effect['draws'] == 2000
                   for effect in report['periods'][period]['residual_vs'].values())
    assert json.loads((here / 'protected_evaluation_attempt.json').read_text(encoding='utf-8'))['status'] == 'complete'
