# Reporting repair 01: numeric rendering verification

Scope: verify_v6.py only. This verifier had not been executed when the defect was found; no experiment result, model, checkpoint, prediction or score is changed.

Cause confirmed by source inspection: a source value of 2.1 and a manuscript rendered string of 3.2 could pass because the former was compared with the source JSON while the latter was only searched in the artifact.

Change: bind rendered text to the actual source value. With format_spec, require Python format(value * multiplier, format_spec), with multiplier default 1. Without format_spec, require exact json.dumps(value, ensure_ascii=False, allow_nan=False). Then retain the artifact-text presence check.

Compatibility inspection: plot_v6.py numeric_claims emits json.dumps(value) for finite numeric or null values; these have the same representation under the new default. No plot code was changed.

The first edit helper stopped at an assertion because its LF-only search did not match the source CRLF bytes. It wrote no files. The second helper preserves the source line endings and applies the intended single block change. This was a file-edit failure, not a verifier/model/experiment execution.

Validation status: source inspection only. No AST check, verifier execution, model import, forward, fit or benchmark was performed by this repair. Root reserves and executes checks.

Original SHA256: `ff427e730945ae9652a7541853112b6d9a522f06a69e62ab2caf36e846d183ae`

Repaired SHA256: `c89c10fdf46754561ec1b03773278fe803b08c8d04cca14a63d62d8a4eba21df`

Original and repaired Python source use UTF-8 with CRLF line endings. The exact textual diff below is displayed with LF; reverse it and restore CRLF to recover the original bytes/hash.

```diff
--- verify_v6.py.before_reporting01
+++ verify_v6.py.after_reporting01
@@ -501,6 +501,11 @@
         for component in claim['json_path']:
             value = value[component]
         close(claim['value'], value, claim['label'])
+        if 'format_spec' in claim:
+            expected_rendered = format(value * claim.get('multiplier', 1), claim['format_spec'])
+        else:
+            expected_rendered = json.dumps(value, ensure_ascii=False, allow_nan=False)
+        assert claim['rendered'] == expected_rendered, claim['label'] + ': displayed number differs from source'
         artifact = resolve(claim['artifact'])
         assert str(artifact.resolve()) in artifacts
         assert claim['rendered'] and claim['rendered'] in artifact.read_text(encoding='utf-8'), claim['label']
```

## Derived reporting extension before first verifier execution

Root requested direct verification of summarize_v6 report_values and assembled JSON/CSV outputs, when present. The verifier now checks reported family centers, block seed means, memory extrema/MiB, relative costs, channel directions and selected-training values against original completed JSON; assembly JSON and all CSV cells against evaluation/cost sources. It also disallows data_prepare jobs ending after first TEST exposure. Missing optional reporting outputs are explicitly not_present/checked:false. No model/data/result is changed.

This extension was edited without AST/test/verifier execution. Only verify_v6.py and this repair record changed. Core experiment files remain frozen.

Before SHA256: `c89c10fdf46754561ec1b03773278fe803b08c8d04cca14a63d62d8a4eba21df`

After SHA256: `08169a3ecb28582537223987694a25b4404ff732fdc2d80f7dec2d7501d4750c`

Source remains UTF-8 CRLF; the diff is displayed with LF. Reverse the diff and restore CRLF to recover prior bytes.

```diff
--- verify_v6.py.before_derived_reporting
+++ verify_v6.py.after_derived_reporting
@@ -124,7 +124,7 @@
     for key in ('gpu_seconds', 'cpu_check_seconds', 'storage_bytes'):
         assert snapshot[key] <= LIMITS[key], key
     assert all(r.get('ended_utc', float('inf')) <= read(HERE / 'test_exposure.json')['first_exposed_utc']
-               for r in jobs if r['category'] == 'real_fit'), 'Fitting continued after TEST exposure'
+               for r in jobs if r['category'] in ('real_fit', 'data_prepare')), 'Fitting or data preparation continued after TEST exposure'
     return snapshot
 
 
@@ -613,6 +613,139 @@
             'drift_does_not_fail_audit': True}
 
 
+def check_csv_rows(path, expected_rows, numeric_fields):
+    import csv
+    with Path(path).open(newline='', encoding='utf-8') as handle:
+        reader = csv.DictReader(handle)
+        rows = list(reader)
+        assert set(reader.fieldnames) == set(expected_rows[0]), 'CSV columns changed: ' + str(path)
+    assert len(rows) == len(expected_rows), 'CSV row count changed: ' + str(path)
+    for position, (actual, expected) in enumerate(zip(rows, expected_rows)):
+        for key, value in expected.items():
+            label = str(path) + '/' + str(position) + '/' + key
+            if key in numeric_fields:
+                parsed = None if actual[key] == '' else float(actual[key])
+                close(parsed, value, label)
+            else:
+                assert actual[key] == ('' if value is None else str(value)), label
+    return {'source': receipt(path), 'rows': len(rows), 'all_cells_checked': True}
+
+
+def check_derived_reports(evaluation, costs, selection, evaluation_path):
+    paths = {read(path)['stage']: resolve(path) for path in costs}
+    original = {stage: read(path) for stage, path in paths.items()}
+    report = {}
+    values_path = HERE / 'report_values.json'
+    if values_path.exists():
+        values = read(values_path)
+        assert values['status'] == 'complete'
+        available_sources = {str(check_receipt(item).resolve()) for item in values['sources']}
+        assert {str(resolve(evaluation_path).resolve()), *(str(path.resolve()) for path in paths.values())} <= available_sources
+        check_receipt(values['source_code'])
+        assert values['accuracy'] == evaluation['role_scores'] and values['comparisons'] == evaluation['comparisons']
+        for stage in ('gpu', 'cpu'):
+            measured = original[stage]
+            wanted_families = {group['family'] for group in measured['summary']['groups']}
+            assert set(values[stage]) == wanted_families
+            for group in measured['summary']['groups']:
+                family, batch = group['family'], str(group['batch_origins'])
+                item = values[stage][family][batch]
+                rows = [row for row in measured['rows'] if row['include_in_primary'] and
+                        row['family'] == family and str(row['batch_origins']) == batch]
+                assert rows
+                close(item['throughput'], group['mean_fit_median_origins_per_second'], stage + family + ' throughput')
+                close(item['ms_per_origin'], group['mean_fit_median_ms_per_origin'], stage + family + ' latency')
+                block_means = [float(np.mean([row['origins_per_second']['median'] for row in rows if row['block'] == block]))
+                               for block in range(3)]
+                close(item['block_seed_mean_throughput'], block_means, 'Per-block seed-mean throughput')
+                for key, source_key in (('deployed_parameters', 'total_deployed_parameters'),
+                                         ('deployed_tensor_bytes', 'total_deployed_tensor_bytes'),
+                                         ('trainable_before_merge', 'registered_trainable_parameters_before_merge')):
+                    assert all(row[source_key] == rows[0][source_key] for row in rows)
+                    assert item[key] == rows[0][source_key]
+                assert item['fitted_coefficients'] == (24624 if family == 'shared' else rows[0]['registered_trainable_parameters_before_merge'])
+                for field in ('peak_allocated_bytes', 'peak_reserved_bytes'):
+                    if field in rows[0]:
+                        peaks = [row[field] for row in rows]
+                        assert item[field] == max(peaks) and item[field + '_min'] == min(peaks)
+                        close(item[field.replace('_bytes', '_mib')], max(peaks) / 2**20, 'Max-peak bytes to MiB')
+                    else:
+                        assert field not in item and field.replace('_bytes', '_mib') not in item, 'CPU memory must not be invented as zero'
+        level = values['gpu']['level_res']['4']
+        expected_references = {'old_res', 'level_raw', 'f0', 'lora', 'level_linear_res'}
+        assert set(values['cost_relative_to']) == expected_references
+        for family, relative in values['cost_relative_to'].items():
+            reference = values['gpu'][family]['4']
+            ratio = level['throughput'] / reference['throughput']
+            close(relative['batch4_throughput_ratio'], ratio, family + ' throughput ratio')
+            close(relative['batch4_throughput_change_pct'], 100 * (ratio - 1), family + ' throughput relative percent')
+            close(relative['peak_allocated_change_pct'], 100 * (level['peak_allocated_bytes'] / reference['peak_allocated_bytes'] - 1), family + ' allocated relative percent')
+            close(relative['batch1_latency_change_pct'], 100 * (values['gpu']['level_res']['1']['ms_per_origin'] /
+                  values['gpu'][family]['1']['ms_per_origin'] - 1), family + ' batch1 latency relative percent')
+        assert set(values['channel_directions']) == set(evaluation['roles'])
+        channel_means = {family: np.mean([evaluation['models'][identifier]['scores']['combined']['channel_mse']
+                                        for identifier in identifiers], axis=0)
+                         for family, identifiers in evaluation['roles'].items()}
+        for family, item in values['channel_directions'].items():
+            differences = channel_means['level_res'] - channel_means[family]
+            assert item['channel_order'] == evaluation['channels']
+            close(item['delta_mse_by_channel'], differences, family + ' channel deltas')
+            assert item['lower'] == int(np.sum(differences < 0))
+            assert item['higher'] == int(np.sum(differences > 0))
+            assert item['ties'] == int(np.sum(differences == 0))
+        training = {row['id']: row for row in values['training']}
+        assert len(training) == len(values['training']) == 10
+        assert set(training) == {row['id'] for row in selection['all_neural_runs']}
+        for row in selection['all_neural_runs']:
+            selected_result = read(check_receipt({'path': row['result_path'], 'sha256': row['result_sha256']}))
+            assert str(resolve(row['result_path']).resolve()) in available_sources
+            for key, value in training[row['id']].items():
+                assert value == selected_result[key], 'Reported training value differs from selected run: ' + row['id'] + '/' + key
+        report['report_values'] = {'source': receipt(values_path), 'cost_centers_extrema_units_ratios': True,
+                                   'accuracy_comparisons_channel_directions_training': True}
+    else:
+        report['report_values'] = {'status': 'not_present', 'checked': False}
+    assembly_path = HERE / 'final_comparison.json'
+    csv_path = HERE / 'final_comparison.csv'
+    channel_path = HERE / 'final_channel_metrics.csv'
+    if any(path.exists() for path in (assembly_path, csv_path, channel_path)):
+        assert all(path.exists() for path in (assembly_path, csv_path, channel_path)), 'Incomplete final assembly artifacts'
+        assembled = read(assembly_path)
+        assert all(assembled[key] == value for key, value in evaluation.items()), 'Assembled evaluation fields changed'
+        expected_sources = {'evaluation': resolve(evaluation_path), 'gpu_cost': paths['gpu'], 'cpu_cost': paths['cpu']}
+        for key, path in expected_sources.items():
+            assert check_receipt(assembled['evidence_sources'][key]).resolve() == path.resolve()
+        assert assembled['gpu_cost_summary'] == original['gpu']['summary']
+        assert assembled['cpu_cost_summary'] == original['cpu']['summary']
+        assert assembled['cost_drift'] == original['gpu']['drift_checks']
+        assembly_receipt = read(HERE / 'assembly_receipt.json')
+        assert assembly_receipt['status'] == 'complete' and assembly_receipt['new_predictions'] == assembly_receipt['new_fits'] == 0
+        check_receipt(assembly_receipt['source'])
+        assert {str(check_receipt(item).resolve()) for item in assembly_receipt['outputs']} == {
+            str(path.resolve()) for path in (assembly_path, csv_path, channel_path)}
+        expected_rows, channel_rows = [], []
+        for family, periods in evaluation['role_scores'].items():
+            for period, score in periods.items():
+                expected_rows.append(dict(family=family, model_id='', seed='', period=period,
+                                          mse=score['mse'], mae=score['mae'], aggregation='mean individual model loss'))
+        for identifier, model in evaluation['models'].items():
+            for period, score in model['scores'].items():
+                expected_rows.append(dict(family=model['spec']['family'], model_id=identifier, seed=model['spec'].get('seed'),
+                                          period=period, mse=score['mse'], mae=score['mae'], aggregation='individual model'))
+                for index, channel in enumerate(evaluation['channels']):
+                    channel_rows.append(dict(family=model['spec']['family'], model_id=identifier, seed=model['spec'].get('seed'),
+                        period=period, channel=channel, mse=score['channel_mse'][index], mae=score['channel_mae'][index],
+                        target_count=score['channel_target_count'][index], squared_error_sum=score['channel_squared_error_sum'][index],
+                        absolute_error_sum=score['channel_absolute_error_sum'][index]))
+        report['assembly'] = {'json': receipt(assembly_path),
+            'metrics_csv': check_csv_rows(csv_path, expected_rows, {'mse', 'mae'}),
+            'channel_csv': check_csv_rows(channel_path, channel_rows, {'mse', 'mae', 'target_count', 'squared_error_sum', 'absolute_error_sum'}),
+            'evaluation_and_cost_fields_match_original_sources': True}
+    else:
+        report['assembly'] = {'status': 'not_present', 'checked': False}
+    return report
+
+
 def audit(args, job):
     report = {'preservation': check_preservation(job)}
     report['budget_during_audit'] = check_budget(job.id)
@@ -647,6 +780,7 @@
     report['evaluation'] = check_test(evaluation, seal, job)
     report['costs'] = [check_cost(path, seal) for path in args.cost]
     assert {read(path)['stage'] for path in args.cost} == {'gpu', 'cpu'}
+    report['derived_reporting'] = check_derived_reports(evaluation, args.cost, selection, args.evaluation)
     forbidden = [str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file() and
                  (p.suffix.lower() in ('.npz', '.npy', '.pt', '.pth', '.safetensors', '.parquet', '.arrow', '.feather', '.bin', '.pkl', '.joblib')
                   or p.name.lower() in ('electricity.csv', 'metadata.csv'))]
```
