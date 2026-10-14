"""Synthetic CPU-only checks; no historical assets, models or GPU execution."""
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from research.smallk_failure_followup_20261014 import profile_latency_burst_v2 as lp


def fixture():
    rng = np.random.default_rng(92601)
    physical = rng.normal(size=(8, 512, 5)).astype(np.float32)
    observed = rng.random(physical.shape) > .25
    observed[0, :, 0] = False
    physical[~observed] = np.nan
    mean = np.array([1., -2., 0., 3., .2], np.float32)
    std = np.array([.1, 1., 3., .3, 9.], np.float32)
    return physical, observed, mean, std


def old_preprocess(physical, observed, mean, std):
    filled = np.empty_like(physical, dtype=np.float32)
    last = np.broadcast_to(mean, (len(physical), len(mean))).copy()
    for t in range(physical.shape[1]):
        last = np.where(observed[:, t], physical[:, t], last)
        filled[:, t] = last
    return ((filled - mean) / std).astype(np.float32)


class Backend:
    def __init__(self):
        self.syncs = 0

    def tensor(self, value):
        return value.copy()

    def to_numpy(self, value):
        return value.copy()

    def synchronize(self):
        self.syncs += 1


class Limit:
    def __init__(self, fail_at=None):
        self.checks, self.fail_at = 0, fail_at

    def check(self):
        self.checks += 1
        if self.checks == self.fail_at:
            raise RuntimeError('synthetic_budget')


def forecaster(x):
    return np.repeat(.8 * x[:, -1:] + .2 * x[:, -2:-1], 48, axis=1).astype(np.float32)


class Codec:
    def encode(self, x):
        return x * np.float32(.3) + np.float32(1.7)

    def decode(self, z):
        return np.tanh(z).astype(np.float32) * np.float32(1.4)

    def forward_path(self, x, future):
        z = self.encode(x)
        return x[:, -1:] + self.decode(future(z)) - self.decode(z[:, -1:])


@pytest.mark.parametrize('batch', [1, 8])
def test_vectorized_fill_is_exact_original_loop_with_missing_first_and_all_missing(batch):
    physical, observed, mean, std = fixture()
    actual = lp.vectorized_preprocess(physical[:batch], observed[:batch], mean, std)
    expected = old_preprocess(physical[:batch], observed[:batch], mean, std)
    assert np.array_equal(actual, expected)
    assert np.array_equal(actual[0, :, 0], np.zeros(512, np.float32))
    assert np.isnan(physical[~observed]).all()


def test_future_changes_do_not_change_past_fills():
    physical, observed, mean, _ = fixture()
    before = lp.fill_vectorized(physical, observed, mean)
    changed = physical.copy()
    changed[:, 250:] = 999
    after = lp.fill_vectorized(changed, observed, mean)
    assert np.array_equal(before[:, :250], after[:, :250])


def test_raw_validator_rejects_denormalized_filled_history_as_original_past():
    physical, observed, mean, std = fixture()
    history = old_preprocess(physical, observed, mean, std)
    origins = np.arange(8, dtype=np.int64) + 700
    lp.validate_raw_inputs(physical, observed, history, mean, std, origins)
    wrong = physical.copy()
    wrong[observed] += .3
    with pytest.raises(ValueError, match='DOES_NOT_REPRODUCE'):
        lp.validate_raw_inputs(wrong, observed, history, mean, std, origins)
    wrong[observed] = np.nan
    with pytest.raises(ValueError, match='NONFINITE'):
        lp.validate_raw_inputs(wrong, observed, history, mean, std, origins)


@pytest.mark.parametrize('codec', [None, Codec()])
@pytest.mark.parametrize('preprocessing', ['old_loop', 'vectorized'])
def test_decomposed_pipeline_preserves_whole_anchor_order(codec, preprocessing):
    physical, observed, mean, std = fixture()
    backend = Backend()
    pipeline = lp.DeploymentPipeline(backend, codec, forecaster, mean, std, old_preprocess)
    normalized, actual = pipeline.whole(physical[:1], observed[:1], preprocessing)
    segmented, output, stages = pipeline.decomposed(physical[:1], observed[:1], preprocessing, Limit())
    assert np.array_equal(normalized, segmented)
    assert np.array_equal(actual, output)
    assert tuple(stages) == lp.STAGES
    assert backend.syncs == 2 * len(lp.STAGES)
    assert all(value >= 0 for value in stages.values())


def save_arrays(path, arrays):
    with Path(path).open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    return {'path': str(path), 'arrays': list(arrays)}


def test_all_repeats_reuse_original_past_save_outputs_and_report_bulk_failure(tmp_path):
    physical, observed, mean, std = fixture()
    physical, observed = physical[:1], observed[:1]
    pipeline = lp.DeploymentPipeline(Backend(), Codec(), forecaster, mean, std, old_preprocess)
    normalized, physical_forecast = pipeline.whole(physical, observed, 'old_loop')
    reference = {'normalized': normalized, 'physical': physical_forecast}
    wrong_bulk = {'normalized': normalized + np.float32(.2), 'physical': physical_forecast + np.float32(1)}
    row = lp.measure_route(pipeline, physical, observed, 'vectorized', reference, wrong_bulk, tmp_path, Limit(), save_arrays)
    assert row['status'] == 'COMPLETE'
    assert len(row['whole_seconds']) == len(row['decomposed_seconds']) == 7
    assert all(check['matched_batch_physical']['pass'] for check in row['whole_parity'])
    assert not any(check['original_bulk_physical']['pass'] for check in row['whole_parity'])
    with np.load(tmp_path / 'whole_outputs.npz', allow_pickle=False) as outputs:
        assert outputs['physical'].shape == (7, 1, 48, 5)
        assert np.array_equal(outputs['normalized'], np.repeat(normalized[None], 7, axis=0))
    with np.load(tmp_path / 'decomposed_outputs.npz', allow_pickle=False) as outputs:
        assert outputs['physical'].shape == (7, 1, 48, 5)
    assert physical.shape == (1, 512, 5)


def test_whole_failure_still_saves_actual_outputs_and_skips_decomposition(tmp_path):
    physical, observed, mean, std = fixture()
    pipeline = lp.DeploymentPipeline(Backend(), None, forecaster, mean, std, old_preprocess)
    normalized, actual = pipeline.whole(physical[:1], observed[:1], 'old_loop')
    wrong = {'normalized': normalized + 1, 'physical': actual + 1}
    row = lp.measure_route(pipeline, physical[:1], observed[:1], 'old_loop', wrong, wrong, tmp_path, Limit(), save_arrays)
    assert row['status'] == 'FAILED_PRESERVED'
    assert (tmp_path / 'whole_outputs.npz').exists()
    assert 'decomposed_outputs' not in row


def test_budget_check_is_cooperative_during_each_stage():
    physical, observed, mean, std = fixture()
    pipeline = lp.DeploymentPipeline(Backend(), None, forecaster, mean, std, old_preprocess)
    with pytest.raises(RuntimeError, match='synthetic_budget'):
        pipeline.decomposed(physical[:1], observed[:1], 'old_loop', Limit(fail_at=3))


def test_failed_rows_are_not_used_to_claim_speedup():
    rows = [{'method': method, 'batch': batch, 'preprocessing': kind,
             'status': 'COMPLETE', 'whole_median_seconds': value}
            for method in ('RAW_BOLT', 'MAIN') for batch in (1, 8)
            for kind, value in (('old_loop', .03), ('vectorized', .02))]
    changes = lp.paired_changes(rows)
    assert len(changes) == 4
    assert all(np.isclose(row['relative_change_percent'], -100 / 3) for row in changes)
    rows[0]['status'] = 'FAILED_PRESERVED'
    assert len(lp.paired_changes(rows)) == 3


def test_parity_rejects_shape_or_nan():
    assert not lp.parity(np.ones(2), np.ones(3), 1e-5, 2e-6)['pass']
    assert not lp.parity(np.array([np.nan]), np.array([np.nan]), 1e-5, 2e-6)['pass']


def test_preparation_import_and_cli_have_no_asset_or_gpu_side_effect():
    description = lp.describe()
    assert description['scientific_arrays_opened'] is False
    assert description['model_or_GPU_executed'] is False
    source = inspect.getsource(lp)
    assert 'torch.load(' not in source
    assert 'en.Limit(' not in source
    assert 'compile(' not in source
    assert 'cuda_graph' not in source
    assert json.loads(json.dumps(description))['warmup'] == 2


def test_execution_opt_in_precedes_directory_creation():
    args = type('Arguments', (), {'execute_after_bi': False})()
    with pytest.raises(ValueError, match='EXPLICIT_AFTER_BI'):
        lp.execute(args)


def test_budget_constructor_rejection_preserves_terminal_result(tmp_path, monkeypatch):
    class Runner:
        @staticmethod
        def write(path, value):
            with Path(path).open('x', encoding='utf-8') as handle:
                json.dump(value, handle)

    class Runtime:
        @staticmethod
        def FollowupLimit(tag, gpu):
            raise RuntimeError('synthetic_budget_constructor')

    modules = {'run_condition_probes': Runner, 'prepare_inputs': object(), 'followup_runtime': Runtime}
    monkeypatch.setattr(lp, 'CACHE', tmp_path)
    monkeypatch.setattr(lp, 'import_sibling', lambda name: modules[name])
    args = type('Arguments', (), {'execute_after_bi': True, 'tag': 'fresh_budget_failed',
        'dataset': 'robin', 'seed': 92601})()
    with pytest.raises(RuntimeError, match='synthetic_budget_constructor'):
        lp.execute(args)
    saved = json.loads((tmp_path / 'fresh_budget_failed' / 'RESULT.json').read_text(encoding='utf-8'))
    assert saved['status'] == 'FAILED_PRESERVED'
    assert saved['partial_coverage']['complete_routes'] == 0
    assert saved['error']['message'] == 'synthetic_budget_constructor'


def test_protected_guard_precedes_bridge_scientific_read():
    source = inspect.getsource(lp.execute)
    assert source.index('bridge_module.install_protected_test_guard()') < source.index('runner.load_bridge(')


@pytest.mark.parametrize('receipt,expected', [
    ({'terminal_status': 'BUDGET_HELD_PRESERVED', 'terminal_error': {'message': 'synthetic_close_budget'}}, 'BUDGET_HELD_PRESERVED'),
    ({'legacy_receipt': True}, 'FAILED_PRESERVED')])
def test_close_status_is_propagated_and_legacy_receipt_preserves_result(tmp_path, monkeypatch, receipt, expected):
    class Runner:
        @staticmethod
        def sha(path):
            return 'synthetic_hash'

        @staticmethod
        def verified_file(record):
            return dict(record)

        @staticmethod
        def write(path, value):
            with Path(path).open('x', encoding='utf-8') as handle:
                json.dump(value, handle)

    class Bridge:
        @staticmethod
        def install_protected_test_guard():
            raise ValueError('synthetic_before_read_failure')

    class Runtime:
        class FollowupLimit:
            def __init__(self, tag, gpu):
                pass

            def check(self):
                pass

            def close(self, status, error):
                return dict(receipt)

    modules = {'run_condition_probes': Runner, 'prepare_inputs': Bridge, 'followup_runtime': Runtime}
    monkeypatch.setattr(lp, 'CACHE', tmp_path)
    monkeypatch.setattr(lp, 'import_sibling', lambda name: modules[name])
    args = type('Arguments', (), {'execute_after_bi': True, 'tag': 'fresh_close_failed',
        'dataset': 'robin', 'seed': 92601})()
    with pytest.raises(ValueError, match='synthetic_before_read_failure'):
        lp.execute(args)
    saved = json.loads((tmp_path / 'fresh_close_failed' / 'RESULT.json').read_text(encoding='utf-8'))
    assert saved['status'] == expected
    assert saved['shared_resource_receipt'] == receipt
    if expected == 'BUDGET_HELD_PRESERVED':
        assert saved['error'] == receipt['terminal_error']
        assert saved['before_resource_close']['error']['message'] == 'synthetic_before_read_failure'
    else:
        assert saved['error']['message'] == 'synthetic_before_read_failure'


def test_cli_noncomplete_result_has_nonzero_exit(monkeypatch, capsys):
    monkeypatch.setattr(lp.sys, 'argv', ['profile_latency.py', '--execute-after-bi',
        '--bridge-result', 'synthetic.json', '--dataset', 'robin', '--seed', '92601', '--tag', 'synthetic'])
    monkeypatch.setattr(lp, 'execute', lambda args: {'status': 'BUDGET_HELD_PRESERVED', 'rows': []})
    with pytest.raises(SystemExit) as error:
        lp.main()
    assert error.value.code == 1
    assert 'BUDGET_HELD_PRESERVED' in capsys.readouterr().out


def test_numerical_functions_identical_to_preserved_v1():
    import ast
    from research.smallk_failure_followup_20261014 import profile_latency as original
    for name in ('fill_loop','fill_vectorized','normalize','vectorized_preprocess', 'validate_raw_inputs','parity','measure_route','paired_changes'):
        assert ast.dump(ast.parse(inspect.getsource(getattr(lp,name)))) == ast.dump(ast.parse(inspect.getsource(getattr(original,name))))
    assert ast.dump(ast.parse(inspect.getsource(lp.DeploymentPipeline))) == ast.dump(ast.parse(inspect.getsource(original.DeploymentPipeline)))
