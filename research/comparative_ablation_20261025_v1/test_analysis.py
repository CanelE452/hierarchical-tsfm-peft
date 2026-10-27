"""Hand-computed metric, selection, and synthetic report verification."""
import copy
import json
from pathlib import Path
import tempfile

import numpy as np
import pytest

from research.comparative_ablation_20261025_v1 import analysis as a


def arrays(pred, target, mask=None):
    p, y = np.asarray(pred, np.float32), np.asarray(target, np.float32)
    if p.ndim == 1:
        p, y = p[None, :, None], y[None, :, None]
    elif p.ndim == 2:
        p, y = p[None], y[None]
    return dict(pred_normalized=p, target_normalized=y,
                observed=np.ones_like(y, bool) if mask is None else np.asarray(mask, bool),
                origins=np.arange(len(p), dtype=np.int64), channel_ids=np.arange(p.shape[-1]))


def scales(c=1):
    return dict(mean=np.zeros(c, np.float32), std=np.ones(c, np.float32),
                physical_std=np.ones(c), mase_scale=np.ones(c), mase_pairs=np.full(c, 2), fit_end=4)


def test_hand_calculated_channel_macro_metrics():
    values = arrays([[2, 4], [1, 2]], [[1, 2], [3, 2]])
    s = scales(2)
    s['mase_scale'] = np.array([1, 2])
    result = a.metrics(values, s, expected_mse=2.25, expected_mae=1.25)
    assert result['MSE'] == 2.25
    assert result['MAE'] == 1.25
    assert result['R2'] == -1.5 and result['R2_excluded_channels'] == 1
    assert result['MASE'] == 1
    assert result['MAPE_nonzero_pct'] == pytest.approx(200/3)
    assert result['MAPE_far_from_zero_pct'] == result['MAPE_nonzero_pct']
    assert result['nearzero_fraction'] == result['zero_fraction'] == 0


def test_zero_and_nearzero_mape_and_undefined_mase_are_explicit():
    values = arrays([1, .15, 1], [0, .05, 2])
    s = scales()
    s['mase_scale'][:] = 0
    result = a.metrics(values, s)
    assert result['MAPE_nonzero_pct'] == pytest.approx(125, abs=2e-5)
    assert result['MAPE_far_from_zero_pct'] == 50
    assert result['nearzero_fraction'] == pytest.approx(2/3)
    assert result['zero_fraction'] == pytest.approx(1/3)
    assert result['MASE'] is None and result['MASE_valid_channel_mean'] is None
    assert result['MASE_undefined_channels'] == 1


def test_physical_zero_is_excluded_despite_nonzero_normalization_roundtrip():
    s = scales()
    s.update(mean=np.array([1.3], np.float32), std=np.array([2.7], np.float32))
    task = dict(physical=np.array([[1], [1], [0], [2]], np.float32),
                observed=np.ones((4, 1), bool), columns=[0], H=2,
                split_ranges={'V_select': (2, 4)})
    physical_truth = task['physical'][2:]
    normalized_truth = (physical_truth - s['mean']) / s['std']
    normalized_prediction = (np.array([[1], [3]], np.float32) - s['mean']) / s['std']
    values = arrays(normalized_prediction, normalized_truth)
    values['origins'][:] = 2
    reconstructed_zero = normalized_truth[0].astype(np.float64) * s['std'] + s['mean']
    assert reconstructed_zero[0] != 0
    with pytest.raises(ValueError, match='Exact physical V targets'):
        a.metrics(values, s)
    bound = a.bind_physical_targets(values, task, s)
    result = a.metrics(bound, s)
    physical_prediction = normalized_prediction.astype(np.float64) * s['std'] + s['mean']
    assert result['zero_fraction'] == .5
    assert result['nearzero_fraction'] == .5
    assert result['MAPE_nonzero_pct'] == pytest.approx(abs(physical_prediction[1, 0] - 2) / 2 * 100)
    assert result['MAPE_nonzero_pct'] == pytest.approx(50, abs=1e-5)
    normalized_error = normalized_prediction.astype(np.float64) - normalized_truth.astype(np.float64)
    assert result['MSE'] == float(np.square(normalized_error).mean())
    assert result['MAE'] == float(np.abs(normalized_error).mean())


def test_physical_target_binding_requires_exact_normalized_values_and_mask():
    task = dict(physical=np.array([[0], [2]], np.float32), observed=np.ones((2, 1), bool),
                columns=[0], H=2, split_ranges={'V_select': (0, 2)})
    values = arrays([1, 3], [0, 2])
    changed = copy.deepcopy(values)
    changed['target_normalized'][0, 1, 0] += .0001
    with pytest.raises(ValueError, match='reproduce saved FP32'):
        a.bind_physical_targets(changed, task, scales())
    task['observed'][0, 0] = False
    with pytest.raises(ValueError, match='observed mask'):
        a.bind_physical_targets(values, task, scales())


def test_mase_uses_only_adjacent_observed_pairs_inside_a60():
    task = dict(physical=np.array([[0, 8], [1, 8], [99, 8], [4, 8], [9999, 99]], np.float32),
                observed=np.array([[1, 1], [1, 1], [0, 1], [1, 1], [1, 1]], bool))
    s = a.training_scales(task, 4)
    assert s['mean'][0] == np.float32(5/3)
    np.testing.assert_array_equal(s['mase_pairs'], [1, 3])
    assert s['mase_scale'][0] == pytest.approx(1 / s['std'][0])
    assert s['mase_scale'][1] == 0
    assert s['std'][1] == 1 and s['physical_std'][1] == 0
    task['physical'][-1] *= 100
    np.testing.assert_array_equal(a.training_scales(task, 4)['mase_scale'], s['mase_scale'])


def test_observation_mask_controls_channel_mean_and_receipt_tolerance():
    values = arrays([[2, 100], [4, 3]], [[1, 0], [2, 1]], mask=[[[1, 0], [1, 1]]])
    result = a.metrics(values, scales(2))
    assert result['MSE'] == (2.5 + 4) / 2
    assert result['MAE'] == (1.5 + 2) / 2
    a.metrics(values, scales(2), expected_mse=result['MSE'] + .5e-9)
    with pytest.raises(ValueError, match='MSE'):
        a.metrics(values, scales(2), expected_mse=result['MSE'] + 2e-9)


def test_error_quantiles_and_raw_ranked_top_five_percent():
    raw = arrays(np.arange(1, 21), np.zeros(20))
    ours = arrays(np.arange(1, 21) / 2, np.zeros(20))
    result = a.error_distribution(ours, raw)
    assert result['abs_error_q50'] == 5.25
    assert result['raw_top5_count'] == 1
    assert result['raw_top5_SSE'] == 400
    assert result['raw_total_SSE'] == 2870
    assert result['raw_top5_SSE_fraction'] == pytest.approx(400/2870)
    assert result['family_SSE_on_raw_top5'] == 100
    assert result['fraction_abs_error_better_than_raw'] == 1
    tied = arrays(np.ones(20), np.zeros(20))
    first_worse = arrays([100] + [0]*19, np.zeros(20))
    assert a.error_distribution(first_worse, tied)['family_SSE_on_raw_top5'] == 10000


@pytest.mark.parametrize('field', ['target_normalized', 'observed', 'origins', 'channel_ids'])
def test_cohort_mismatch_is_rejected(field):
    reference = arrays([1, 2], [0, 0])
    other = copy.deepcopy(reference)
    other[field].flat[0] = not other[field].flat[0] if field == 'observed' else other[field].flat[0] + 1
    with pytest.raises(ValueError, match='cohort mismatch'):
        a.validate_arrays(other, reference)


def rows_for_decisions():
    rows = []
    for d in a.DATASETS:
        for method in a.METHODS:
            seeds = [92601] if method in ('C1', 'RAW') else sorted(a.SEEDS)
            value = dict(RAW=2, OURS=1, C1=3, C2=4, C3=5).get(method, 2)
            for seed in seeds:
                rows.append(dict(dataset=d, method_id=method, seed=seed, job_id=f'{method}_{d}_{seed}',
                    MSE=float(value), MAE=.5, MASE=1., R2=.5, extended=False, cap_selected=False))
    return rows


def test_aggregate_excludes_etth2_and_incomplete_cannot_win():
    rows = rows_for_decisions()
    for row in rows:
        if row['dataset'] == 'ETTh2_safe17400' and row['method_id'] == 'OURS':
            row['MSE'] = 10
    summary = a.choose_primary(rows)
    decision = a.decisions(rows, summary)
    assert decision['comparative_overall'] == 'SUPPORTED'
    assert next(r for r in decision['comparative'] if r['dataset'] == 'ETTh2_safe17400')['decision'] == 'LOSS'
    rows = [r for r in rows if not (r['dataset'] == 'jena' and r['method_id'] == 'C3' and r['seed'] == 92610)]
    assert a.decisions(rows, a.choose_primary(rows))['comparative_overall'] == 'INCOMPLETE'


def test_threshold_equality_is_tie_and_unclear():
    rows = rows_for_decisions()
    for row in rows:
        if row['method_id'] == 'OURS':
            row['MSE'] = 10 + row['seed'] - 92601
        elif row['method_id'] in ('C1', 'C2', 'C3', 'A1'):
            row['MSE'] = 19
    decision = a.decisions(rows, a.choose_primary(rows))
    assert all(r['decision'] == 'TIE' and r['gap'] == r['spread'] == 4.5 for r in decision['comparative'])
    assert all(r['decision'] == 'UNCLEAR' for r in decision['ablation'] if r['method_id'] == 'A1')


def test_extensions_replace_only_complete_groups_and_provisional_is_visible():
    rows = rows_for_decisions()
    base = [r for r in rows if r['dataset'] == 'jena' and r['method_id'] == 'C2']
    for row in base[:5]:
        row['cap_selected'] = True
    extensions = [dict(r, job_id=r['job_id']+'_ext10', extended=True, MSE=.25, cap_selected=True) for r in base]
    partial = rows + extensions[:9]
    summary = a.choose_primary(partial)
    item = next(r for r in summary if (r['dataset'], r['method_id']) == ('jena', 'C2'))
    assert item['n'] == 10 and item['MSE_median'] == 4 and item['provisional'] and not item['extended']
    assert a.decisions(partial, summary)['comparative_overall'] == 'SUPPORTED_PROVISIONAL'
    complete = rows + extensions
    summary = a.choose_primary(complete)
    item = next(r for r in summary if (r['dataset'], r['method_id']) == ('jena', 'C2'))
    assert item['MSE_median'] == .25 and item['extended'] and not item['provisional']
    assert item['undertrained_at_10']
    assert sum(r['is_primary'] for r in complete if (r['dataset'], r['method_id']) == ('jena', 'C2')) == 10


def test_duplicate_seed_is_rejected():
    rows = rows_for_decisions()
    rows.append(dict(rows[0], job_id='different_id_same_seed'))
    with pytest.raises(ValueError, match='Duplicate seed'):
        a.choose_primary(rows)


def test_error_summary_uses_seed_medians_and_only_primary_rows():
    records = [dict(dataset='jena', method_id='A1', seed=92601+i, is_primary=True,
                    **{f'abs_error_q{q}': value for q in a.QUANTILES})
               for i, value in enumerate((1., 3., 50.))]
    records.append(dict(records[0], is_primary=False, abs_error_q50=10000.))
    summary = a.summarize_errors(records)
    assert summary[0]['n'] == 3
    assert summary[0]['abs_error_q50_median'] == 3


def test_documents_qualify_provisional_ablation_and_delimit_large_metrics(monkeypatch):
    rows = rows_for_decisions()
    for row in rows:
        if row['method_id'] == 'A1' and row['dataset'] in ('jena', 'ETTh2_safe17400'):
            row['cap_selected'] = True
        if row['dataset'] == 'appliances' and row['method_id'] == 'RAW':
            row.update(MAPE_nonzero_pct=123456789012345.5, MAPE_far_from_zero_pct=42.125)
    summary = a.choose_primary(rows)
    decision = a.decisions(rows, summary)
    decision.update(Q3=[], error_summary=[])
    original_decision = copy.deepcopy(decision)
    documents = {}
    monkeypatch.setattr(a, '_write', lambda path, text: documents.update({path.name: text}))
    a._write_documents(Path('synthetic_unused'), summary, decision, {})
    text = documents['RESULTS_KO.md']
    aggregate = text.split('절제 집계', 1)[1].split('Raw 대 OURS 보조 지표', 1)[0]
    assert 'A1/jena=CONTRIBUTES_PROVISIONAL' in aggregate
    assert 'A1/ETTh2_safe17400' not in aggregate
    assert '확정된 구성요소 기여로 해석할 수 없다' in aggregate
    metrics = text.split('Raw 대 OURS 보조 지표', 1)[1].split('```', 2)[1]
    appliance_raw = next(line for line in metrics.splitlines() if line.startswith('appliances') and 'RAW' in line)
    cells = [cell.strip() for cell in appliance_raw.split(' | ')]
    assert len(cells) == 8
    assert cells[5:7] == ['123456789012345.500000', '42.125000']
    assert decision == original_decision


def test_synthetic_report_roundtrip_preserves_existing_files_and_writes_figures(monkeypatch):
    rows = [r for r in rows_for_decisions() if r['method_id'] in ('RAW', 'OURS')]
    errors = []
    for row in rows:
        row.update(display_name=a.LABELS[row['method_id']])
        errors.append(dict(dataset=row['dataset'], method_id=row['method_id'], seed=row['seed'],
                           job_id=row['job_id'], extended=False,
                           **{f'abs_error_q{q}': float(q) / (50 if row['method_id'] == 'RAW' else 75) for q in a.QUANTILES}))
    monkeypatch.setattr(a, '_prepare_existing', lambda *args: (copy.deepcopy(rows), copy.deepcopy(errors), {}, {}, {}))
    root = Path(__file__).resolve().parents[2] / '.cache/comparative_ablation_20261025_v1/analysis_test_tmp'
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as directory:
        output = Path(directory)
        a.report_existing(output, existing_audit_path='synthetic_unused.json', task_loader=lambda _: None)
        before = (output/'METRICS_EXISTING.csv').read_bytes()
        decision = a.report(output, [], dict(status='PARTIAL', stop_reason='synthetic'),
                            existing_audit_path='synthetic_unused.json', task_loader=lambda _: None)
        assert decision['comparative_overall'] == 'INCOMPLETE'
        assert (output/'METRICS_EXISTING.csv').read_bytes() == before
        assert json.loads((output/'DECISION.json').read_text())['execution_status']['stop_reason'] == 'synthetic'
        for name in ('FIG_COMPARATIVE', 'FIG_ABLATION', 'FIG_ERROR_QUANTILES'):
            for suffix in ('png', 'svg', 'csv'):
                assert (output/f'{name}.{suffix}').stat().st_size > 0
        before = {name: ((output/name).read_bytes(), (output/name).stat().st_mtime_ns) for name in a.GENERATED_FILES}
        # A prior crash may leave a strict subset of the complete artifacts.
        (output/'FIG_ABLATION.png').unlink()
        a.report(output, [], dict(status='PARTIAL', stop_reason='synthetic'),
                 existing_audit_path='synthetic_unused.json', task_loader=lambda _: None)
        for name, (payload, modified) in before.items():
            assert (output/name).read_bytes() == payload
            if name != 'FIG_ABLATION.png':
                assert (output/name).stat().st_mtime_ns == modified
        with pytest.raises(FileExistsError):
            a.report(output, [], {}, existing_audit_path='synthetic_unused.json', task_loader=lambda _: None)
