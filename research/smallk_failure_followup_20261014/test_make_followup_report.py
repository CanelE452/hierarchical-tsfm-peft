"""Synthetic metadata/table checks; no old assets, models or GPU execution."""
from copy import deepcopy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from research.smallk_failure_followup_20261014 import make_followup_report as mr


def complete_items():
    items = []
    for block in ('D0', 'D1', 'D2', 'D3'):
        for dataset in mr.DATASETS:
            for seed in (mr.SEEDS if block in ('D2', 'D3') else (None,)):
                p = {'dataset': dataset, 'seed': seed, 'status': 'COMPLETE'}
                item = {'block': block, 'producer': p}
                if block == 'D0':
                    p['native_same_MAIN_lineage'] = {str(s): {'same_E_D_state_torch_equal': True} for s in mr.SEEDS}
                elif block == 'D1':
                    p['F_frozen_pass'] = True
                    item['scientific'] = {'candidates': list(range(7))}
                elif block == 'D2':
                    p.update(F_frozen_pass=True, MAIN_codec_frozen_pass=True)
                    item['scientific'] = {'refits': {objective: {'selected_alpha': .1, 'selected_index': 1,
                        'alpha_grid': [{'alpha': 0., 'tune_mse': .3}, {'alpha': .1, 'tune_mse': .2}]}
                        for objective in ('Y', 'RAW', 'MIXED')}}
                else:
                    p.update(F_before={'old': 'same'}, F_after={'old': 'same'}, codec_before='same', codec_after='same')
                    checks = [{'repeat': i, 'matched_batch_normalized': {'pass': True},
                        'matched_batch_physical': {'pass': True}, 'original_bulk_physical': {'pass': False}}
                        for i in range(7)]
                    stages = [{'repeat': i, 'matched_batch_normalized': {'pass': True},
                        'matched_batch_physical': {'pass': True}} for i in range(7)]
                    p['rows'] = [{'status': 'COMPLETE', 'method': method, 'batch': batch,
                        'preprocessing': kind, 'warmup': 2, 'repeats': 7, 'whole_seconds': [.01] * 7,
                        'whole_parity': deepcopy(checks), 'decomposed_parity': deepcopy(stages),
                        'decomposed_seconds': [{}] * 7}
                        for method in ('RAW_BOLT', 'MAIN') for batch in (1, 8) for kind in ('old_loop', 'vectorized')]
                items.append(item)
    return items


def test_full_coverage_remains_distinct_from_improvement_and_holdout():
    coverage = mr.assess_coverage(complete_items())
    assert coverage['expected_conditions'] == coverage['supplied_conditions'] == coverage['complete_conditions'] == 12
    assert coverage['all_declared_blocks_complete'] is True
    assert coverage['independent_holdout'] is False
    assert coverage['improvement_established_by_coverage'] is False


def test_duplicate_or_missing_condition_is_not_full_completion():
    items = complete_items()
    with pytest.raises(ValueError, match='DUPLICATE_REPORT_CONDITION'):
        mr.assess_coverage(items + [items[0]])
    coverage = mr.assess_coverage(items[:-1])
    assert coverage['all_declared_blocks_complete'] is False
    assert len(coverage['missing_conditions']) == 1
    items[-1]['producer']['status'] = 'BUDGET_HELD_PRESERVED'
    coverage = mr.assess_coverage(items)
    assert coverage['complete_conditions'] == 11
    assert len(coverage['noncomplete_conditions']) == 1


def test_same_MAIN_native_proof_is_required():
    items = complete_items()
    items[0]['producer']['native_same_MAIN_lineage']['92601']['same_E_D_state_torch_equal'] = False
    with pytest.raises(ValueError, match='SAME_MAIN_NATIVE'):
        mr.assess_coverage(items)


def test_all_D1_candidates_and_frozen_F_are_required():
    items = complete_items()
    item = next(i for i in items if i['block'] == 'D1')
    item['scientific']['candidates'] = list(range(6))
    with pytest.raises(ValueError, match='D1_SEVEN_CANDIDATES'):
        mr.assess_coverage(items)


def test_D2_tune_choice_and_earliest_tie_are_checked():
    items = complete_items()
    item = next(i for i in items if i['block'] == 'D2')
    item['scientific']['refits']['Y']['selected_index'] = 0
    with pytest.raises(ValueError, match='D2_TUNE_SELECTION'):
        mr.assess_coverage(items)
    row = item['scientific']['refits']['Y']
    row['alpha_grid'][0]['tune_mse'] = .2
    row['selected_alpha'] = 0.
    assert mr.assess_coverage(items)['all_declared_blocks_complete'] is True


def test_D3_bulk_failure_is_retained_without_rejecting_matching_batch_success():
    items = complete_items()
    assert mr.assess_coverage(items)['all_declared_blocks_complete'] is True
    item = next(i for i in items if i['block'] == 'D3')
    item['producer']['rows'][0]['whole_parity'][1]['matched_batch_physical']['pass'] = False
    with pytest.raises(ValueError, match='D3_MATCHED_BATCH'):
        mr.assess_coverage(items)


def test_D3_all_repeats_and_stage_parity_are_required():
    items = complete_items()
    item = next(i for i in items if i['block'] == 'D3')
    item['producer']['rows'][0]['whole_parity'] = []
    with pytest.raises(ValueError, match='D3_WARM2_REPEAT7'):
        mr.assess_coverage(items)
    items = complete_items()
    item = next(i for i in items if i['block'] == 'D3')
    item['producer']['rows'][0]['decomposed_parity'][0]['matched_batch_normalized']['pass'] = False
    with pytest.raises(ValueError, match='D3_DECOMPOSED'):
        mr.assess_coverage(items)


def test_seed_average_is_metric_average_not_ensemble():
    rows = [{'dataset': 'robin', 'method': 'MAIN', 'mse': 4, 'mae': 2},
            {'dataset': 'robin', 'method': 'MAIN', 'mse': 16, 'mae': 4}]
    row = mr.aggregate_metrics(rows, ('dataset', 'method'), ('mse', 'mae'))[0]
    assert row['mse'] == 10 and row['mae'] == 3
    assert row['metric_rows_averaged'] == 2
    assert row['seed_aggregation'].endswith('not_prediction_ensemble')


def test_checked_child_hash_rejects_changed_scientific_metadata(tmp_path):
    child = tmp_path / 'child.json'
    child.write_text(json.dumps({'status': 'COMPLETE', 'scientific_execution': True}), encoding='utf-8')
    producer = tmp_path / 'producer.json'
    producer.write_text(json.dumps({'status': 'COMPLETE', 'dataset': 'robin',
        'probe_result': {'path': str(child), 'sha256': mr.sha(child)}}), encoding='utf-8')
    assert mr.load_producer(producer, 'D1')['scientific']['scientific_execution'] is True
    child.write_text(json.dumps({'status': 'COMPLETE', 'scientific_execution': False}), encoding='utf-8')
    with pytest.raises(ValueError, match='HASH_MISMATCH'):
        mr.load_producer(producer, 'D1')


def test_running_producer_and_resource_status_mismatch_are_rejected(tmp_path):
    path = tmp_path / 'producer.json'
    path.write_text(json.dumps({'status': 'RUNNING', 'dataset': 'robin'}), encoding='utf-8')
    with pytest.raises(ValueError, match='TERMINAL_PRODUCER'):
        mr.load_producer(path, 'D0')
    path.write_text(json.dumps({'status': 'FAILED_PRESERVED', 'dataset': 'robin',
        'shared_resource_receipt': {'terminal_status': 'COMPLETE'}}), encoding='utf-8')
    with pytest.raises(ValueError, match='RESOURCE_TERMINAL'):
        mr.load_producer(path, 'D0')


def test_links_are_relative_existing_and_no_ignored_cache_hyperlinks(tmp_path):
    (tmp_path / 'figures').mkdir()
    (tmp_path / 'figures/a.png').write_bytes(b'synthetic_image_stub')
    (tmp_path / 'tables').mkdir()
    (tmp_path / 'tables/a.csv').write_text('mse\n1\n', encoding='utf-8')
    markdown = '![圖](figures/a.png) [수치](tables/a.csv) [없음](tables/missing.csv)'
    actual, paths = mr.local_links(markdown, tmp_path)
    assert paths == ['figures/a.png', 'tables/a.csv']
    assert '[없음]' not in actual and '미생성' in actual
    with pytest.raises(ValueError, match='OUTSIDE_ARTIFACT'):
        mr.local_links('[cache](../.cache/a.npz)', tmp_path)
    with pytest.raises(ValueError, match='IMAGE_MISSING'):
        mr.local_links('![missing](figures/missing.png)', tmp_path)


def test_preparation_mode_has_no_scientific_reads_or_gpu_inference():
    description = mr.describe()
    assert description['actual_report_generated'] is False
    assert description['GPU_execution'] is False
    assert description['scientific_reads_on_import'] is False
    source = inspect.getsource(mr)
    assert 'torch.load(' not in source
    assert 'np.load(' not in source
    assert "gpu=False" in source


def test_generation_opt_in_precedes_output_creation():
    args = type('Arguments', (), {'generate': False})()
    with pytest.raises(ValueError, match='OPT_IN_REQUIRED'):
        mr.generate(args)


def test_protocol_repair_requires_preserved_original_and_guard_receipt():
    args = type('Arguments', (), {'generate': True, 'tag': 'synthetic',
                                 'preserved_d3': None, 'measurement_diagnostic': None})()
    with pytest.raises(ValueError, match='ORIGINAL_D3_AND_GUARD_DIAGNOSTIC_REQUIRED'):
        mr.generate(args)


def test_report_keeps_cadence_coldwarm_and_recipe_uncertainty():
    tables = {key: [] for key in ('d0_summary_mean', 'd0_rank', 'd1', 'd2_selected_mean', 'd3')}
    coverage = {'complete_conditions': 0, 'expected_conditions': 12,
                'missing_conditions': [['D3', 'robin', 92601]], 'noncomplete_conditions': []}
    markdown = mr.report_markdown(tables, coverage, [], [], [])
    assert 'storage/path scan' in markdown
    assert 'cold start' in markdown
    assert '무조건적인 speedup으로 주장하지 않는다' in markdown
    assert 'old 2-seed' in markdown and '3-seed' in markdown
    assert 'prediction ensemble이 아니다' in markdown


def test_audit_list_is_counted_and_D0_nested_status_is_preserved():
    row = mr.summarize_audit({'status': 'PASS_WITH_SCOPE', 'checks': [{'passed': True}] * 3,
                              'failed_count': 0}, 'synthetic.json')
    assert row['checks'] == 3 and row['all_declared_scopes_passed'] is True
    with pytest.raises(ValueError, match='AUDIT_FAILURE_COUNT_MISMATCH'):
        mr.summarize_audit({'status': 'PASS_WITH_SCOPE', 'checks': [{'passed': False}], 'failed_count': 0}, 'bad.json')
    d0 = {'status': 'COMPLETE', 'schema': 'smallk_d0_independent_array_audit_v1',
          'datasets': {'robin': {'status': 'PASS'}, 'beijing_aotizhongxin': {'status': 'PASS'}}}
    row = mr.summarize_audit(d0, 'D0.json')
    assert row['status'] == 'COMPLETE'
    assert row['failed_count'] is None
    assert row['all_declared_scopes_passed'] is True
    d0['datasets']['robin']['status'] = 'FAIL'
    assert mr.summarize_audit(d0, 'D0.json')['all_declared_scopes_passed'] is False


def test_rank_prose_uses_literal_mean_centered_mode_and_separate_target_error():
    tables = {key: [] for key in ('d0_summary_mean', 'd0_rank', 'd1', 'd2_selected_mean', 'd3')}
    tables['d0_rank'] = [{'dataset': 'robin', 'projection_mode': 'mean_centered', 'split': 'val',
        'k': '4', 'retained_fit_energy_fraction': '.96', 'normalized_mse': '.59',
        'normalized_mae': '.45', 'raw_actual_target_normalized_mse': '.34'}]
    coverage = {'complete_conditions': 0, 'expected_conditions': 12,
                'missing_conditions': [], 'noncomplete_conditions': []}
    markdown = mr.report_markdown(tables, coverage, [], [], [])
    assert 'K4 FIT retained=0.960000000' in markdown
    assert 'projected RAW target MSE=0.590000000' in markdown
    assert 'RAW MSE=0.340000000' in markdown
    assert 'VAL 설명률이 아니다' in markdown
