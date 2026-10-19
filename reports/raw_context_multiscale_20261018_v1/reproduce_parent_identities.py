"""Publish parent identities from completed, SHA-bound metadata; never load arrays or weights."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[2]
CAMPAIGN = 'raw_context_multiscale_20261018_v1'
CSV_NAME = 'PARENT_IDENTITIES.csv'
VALIDATION_NAME = 'PARENT_IDENTITIES_VALIDATION.json'
SEEDS = (92601, 92602)
ENDS = dict(jena=(10482, 12579), pems32=(8928, 10713))


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def context(asset_root, repo=REPO):
    repo = Path(repo).resolve()
    return dict(repo=repo, report=repo/'reports'/CAMPAIGN, cache=repo/'.cache'/CAMPAIGN,
                asset=Path(asset_root).resolve())


def path_for(reference, ctx):
    name, relative = reference.split('://', 1)
    if name not in ('report', 'cache', 'repo', 'asset'):
        raise ValueError('Unknown source reference')
    root = Path(ctx[name]).resolve()
    path = (root/relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Source reference escapes its root')
    return path


def binding(path, ctx):
    path = Path(path).resolve()
    for name in ('report', 'cache', 'repo', 'asset'):
        root = Path(ctx[name]).resolve()
        if path.is_relative_to(root):
            return dict(reference=name+'://'+path.relative_to(root).as_posix(), sha256=sha(path))
    raise ValueError('Unregistered source root')


def resolve(value, ctx):
    path = path_for(value['reference'], ctx)
    if sha(path) != value['sha256']:
        raise ValueError('Source SHA changed: '+value['reference'])
    return path


def public_refit_path(parent, ctx):
    reference = parent['search_result']['reference']
    if not reference.startswith('repo://.cache/') or not reference.endswith('/SEARCH_RESULT.json'):
        raise ValueError('Exact historical S search pointer required')
    return path_for(reference.replace('repo://.cache/', 'repo://reports/', 1).replace('/SEARCH_RESULT.json', '/REFIT_RESULT.json'), ctx)


def verify_refit(path, row, ctx, *, historical):
    value = read(path)
    fit = value['fit_receipt']
    afit, a60 = ENDS[row['dataset']]
    if (value.get('status') != 'COMPLETE' or fit.get('fresh_A60') is not True
            or fit['search_fit_end'] != afit or fit['refit_fit_end'] != a60
            or fit['normalization_fit_end'] != a60
            or fit['artifact_sha256'] != row['source_artifact_sha256']
            or value['bundle_binding']['sha256'] != row['source_artifact_sha256']
            or value['cap'] != row['search_cap']
            or fit['selected_epoch'] != row['selected_epoch']
            or value['search_summary']['selected_epoch'] != row['selected_epoch']):
        raise ValueError('Public S refit is not the exact deployed artifact/scope/epoch/cap')
    if not historical and (value.get('V_used') is not False or fit.get('V_used') is not False
            or value['spec']['scientific_identity']['raw_context_mode'] != 'S'
            or value['spec']['dataset'] != row['dataset'] or value['spec']['seed'] != row['seed']
            or value['spec']['run_id'] != row['run_id']):
        raise ValueError('Fresh final S public refit contract differs')
    result = binding(path, ctx)
    row.update(public_refit_reference=result['reference'], public_refit_sha256=result['sha256'])
    return dict(binding=result, artifact_sha256=fit['artifact_sha256'],
                artifact_bundle_binding_in_receipt=value['bundle_binding'],
                cap=value['cap'], selected_epoch=fit['selected_epoch'],
                A_fit_end=afit, A60_fit_end=a60, historical_schema=historical)


def parent_rows(inventory, seal, finals, ctx):
    rows, evidence, fixed = [], [], {}
    inventory_binding = binding(ctx['report']/'DATA_AND_PARENT_BINDINGS.json', ctx)
    seal_binding = binding(ctx['report']/'MODELS_SEALED_BEFORE_V.json', ctx)
    if inventory.get('status') != 'PASS' or seal.get('status') != 'SEALED' or seal.get('V_used') is not False:
        raise ValueError('Complete original parent inventory and before-V seal required')
    for index, parent in enumerate(inventory['parents']):
        role, dataset = parent['role'], parent['dataset']
        if role not in ('S', 'G', 'Raw'):
            continue
        route = parent['route_spec']
        a60 = ENDS[dataset][1]
        if parent['parent_fit_end'] != a60 or route['canonical_fit_end'] != a60:
            raise ValueError('Original parent fit boundary differs')
        raw = role == 'Raw'
        row = dict(identity_scope='fixed_calibration', role=role, dataset=dataset, seed=parent.get('seed'),
            run_id=parent.get('run_id', route['run_id']),
            source_artifact_sha256=None if raw else route['artifact_sha256'],
            F_tensor_sha256=inventory['F_tensor_sha256'] if raw else None,
            A60_fit_end=a60, normalization_fit_end=a60,
            selected_epoch=None if raw else parent['selected_epoch'], search_cap=parent.get('training_cap'),
            source_metadata_reference=inventory_binding['reference'], source_metadata_sha256=inventory_binding['sha256'],
            source_item_pointer='/parents/'+str(index), public_refit_reference=None, public_refit_sha256=None,
            original_calibration_S_run_id=None, original_calibration_S_artifact_sha256=None,
            same_artifact_as_calibration_S=None, G_unique_fitted_model_count_per_dataset=1 if role == 'G' else None,
            referenced_S_or_head_seeds=list(SEEDS) if role == 'G' else None)
        if not raw and parent['normalization_fit_end'] != a60:
            raise ValueError('Original trained parent scaler boundary differs')
        proof = dict(source_binding=inventory_binding, source_item_pointer=row['source_item_pointer'])
        if role == 'G' and (parent.get('actual_G_seed_count') != 1 or parent.get('reused_across_S_seeds') is not True
                            or parent['seed'] != 92601 or row['search_cap'] is not None):
            raise ValueError('One actually fitted G seed must be shared without invented cap')
        if role == 'S':
            fixed[(dataset, parent['seed'])] = (parent, row)
            proof['public_refit'] = verify_refit(public_refit_path(parent, ctx), row, ctx, historical=True)
        rows.append(row)
        evidence.append(proof)
    expected_fixed = {(dataset, role, seed) for dataset in ENDS for role, seeds in
                      (('S', SEEDS), ('G', (92601,)), ('Raw', (None,))) for seed in seeds}
    if len(rows) != 8 or {(r['dataset'], r['role'], r['seed']) for r in rows} != expected_fixed:
        raise ValueError('Exactly eight fixed calibration role identities required')
    final_rows = [item for item in enumerate(finals) if item[1]['mode'] == 'S']
    if len(final_rows) != 4 or {(r['dataset'], r['seed']) for _, r in final_rows} != set(fixed):
        raise ValueError('Exactly four final S identities required')
    for private_index, final in final_rows:
        matches = [(i, item) for i, item in enumerate(seal['finals']) if item['run_id'] == final['run_id']]
        if len(matches) != 1 or matches[0][1]['route_spec'] != final['route_spec'] or matches[0][1]['reused'] != final['reused']:
            raise ValueError('Final private body identity differs from before-V public seal')
        seal_index, _ = matches[0]
        parent, original = fixed[(final['dataset'], final['seed'])]
        route = final['route_spec']
        if route['canonical_fit_end'] != ENDS[final['dataset']][1]:
            raise ValueError('Final S boundary differs')
        row = dict(original, identity_scope='final_S', run_id=final['run_id'],
            source_artifact_sha256=route['artifact_sha256'], selected_epoch=final['search']['selected_epoch'],
            search_cap=final['search']['cap'], source_metadata_reference=seal_binding['reference'],
            source_metadata_sha256=seal_binding['sha256'], source_item_pointer='/finals/'+str(seal_index),
            original_calibration_S_run_id=original['run_id'],
            original_calibration_S_artifact_sha256=original['source_artifact_sha256'],
            same_artifact_as_calibration_S=route['artifact_sha256'] == original['source_artifact_sha256'])
        if final['reused']:
            receipt_path = public_refit_path(final['parent'], ctx)
        else:
            candidates = [p for p in (ctx['report']/'runs'/final['run_id']).glob('cap*/REFIT_RESULT.json')
                          if read(p)['bundle_binding']['sha256'] == route['artifact_sha256']]
            if len(candidates) != 1:
                raise ValueError('Exactly one public refit must bind each fresh final S artifact')
            receipt_path = candidates[0]
        proof = verify_refit(receipt_path, row, ctx, historical=final['reused'])
        evidence.append(dict(source_binding=seal_binding, source_item_pointer=row['source_item_pointer'],
            private_final_body_item_pointer='/'+str(private_index), public_refit=proof))
        rows.append(row)
    return sorted(rows, key=lambda r:(r['identity_scope'], r['dataset'], r['role'], r['seed'] or 0)), evidence, fixed


def endpoint_proofs(ctx, fixed):
    index_path = ctx['report']/'COST_INDEX.json'
    index = read(index_path)
    if index.get('status') != 'COMPLETE':
        raise ValueError('Completed immutable whole-path cost index required')
    proofs, coverage = [], set()
    for facet in index['facets']:
        if not facet['run_id'].endswith('_alpha1'):
            continue
        receipt_path = resolve(facet['output'], ctx)
        value = read(receipt_path)
        route, row = value['route_spec'], value['row']
        parts = facet['run_id'].split('_')
        dataset, role, seed = parts[1], parts[2], int(parts[4][1:])
        key = (dataset, role, seed, facet['batch'])
        original = fixed[(dataset, seed)][1]
        if (key in coverage or value.get('status') != 'COMPLETE' or role not in ('Raw', 'G')
                or route.get('family') != 'COMBINED' or route.get('operation') != 'alpha' or route.get('alpha') != 1
                or route['b_route']['family'] != 'B2'
                or route['b_route']['artifact_sha256'] != original['source_artifact_sha256']
                or row['dataset'] != dataset or row['accuracy_run_id'] != facet['run_id']
                or row['cost_id'] != facet['cost_id'] or row['device'] != 'cuda'
                or row['batch_size'] != facet['batch'] or row['schedule'] != 'native' or facet['schedule'] != 'native'
                or route['g_route']['family'] != ('RAW' if role == 'Raw' else 'DIRECT_G')):
            raise ValueError('Alpha1 endpoint must bind original calibration S, never a different final S')
        request_path = resolve(value['request_binding'], ctx)
        request = read(request_path)
        if request['route_spec'] != route or request['accuracy_run_id'] != facet['run_id']:
            raise ValueError('Endpoint request route identity differs')
        coverage.add(key)
        proofs.append(dict(dataset=dataset, base_role=role, seed=seed, batch_size=facet['batch'],
            cost_id=facet['cost_id'], cost_receipt_binding=binding(receipt_path, ctx), request_binding=value['request_binding'],
            original_calibration_S_run_id=original['run_id'], original_calibration_S_artifact_sha256=original['source_artifact_sha256'],
            actual_b_route_artifact_sha256=route['b_route']['artifact_sha256'], canonical_Raw_endpoint=role == 'Raw'))
    expected = {(d, role, seed, batch) for d in ENDS for role in ('Raw', 'G') for seed in SEEDS for batch in (1, 8)}
    if coverage != expected or len(proofs) != 16:
        raise ValueError('All16 Raw/G alpha1 facets including canonical8 B1/B8 endpoints required')
    return sorted(proofs, key=lambda r:(r['dataset'], r['base_role'], r['seed'], r['batch_size']))


def generate(ctx, output_dir):
    from research.raw_context_multiscale_completion_recovery_20261019_v1.revision2 import validate_completion
    completion = validate_completion(ctx)
    note_path = ctx['cache']/'PARENT_IDENTITIES_MAPPING_NOTE_001.json'
    note = read(note_path)
    inventory_path = resolve(note['metadata_binding'], ctx)
    if inventory_path != (ctx['report']/'DATA_AND_PARENT_BINDINGS.json').resolve():
        raise ValueError('Parent row pointers must bind the exact canonical public inventory')
    scope_path = ctx['report']/'HEAD_COMPONENT_COST_SCOPE_001.json'
    sources = read(scope_path)['source_files']
    if len(sources) != 20 or any(sha(ctx['repo']/name) != digest for name, digest in sources.items()):
        raise ValueError('Existing frozen20 source closure changed')
    seal_path, final_path = ctx['report']/'MODELS_SEALED_BEFORE_V.json', ctx['cache']/'FINAL_BODY_MODELS.json'
    rows, evidence, fixed = parent_rows(read(inventory_path), read(seal_path), read(final_path), ctx)
    endpoints = endpoint_proofs(ctx, fixed)
    inputs = dict(mapping_note=binding(note_path, ctx), original_source_scope=binding(scope_path, ctx),
        parent_inventory=binding(inventory_path, ctx), model_seal=binding(seal_path, ctx),
        private_final_bodies=binding(final_path, ctx), cost_index=binding(ctx['report']/'COST_INDEX.json', ctx),
        completion_terminal=binding(completion['terminal_path'], ctx), completion_validation=binding(completion['validation_path'], ctx))
    for item in evidence:
        if 'public_refit' in item:
            resolve(item['public_refit']['binding'], ctx)
    fields = note['proposed_output']['header']
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text, fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows({key:json.dumps(value, separators=(',', ':')) if isinstance(value, list) else value
                     for key, value in row.items()} for row in rows)
    data = text.getvalue().encode('utf8')
    validation = dict(status='PASS', role_rows=12, unique_role_rows=True,
        scope_counts=dict(fixed_calibration=8, final_S=4), G_unique_fitted_models_per_dataset=1,
        artifact_matches_calibration_by_dataset={dataset:sum(r['same_artifact_as_calibration_S'] is True
            for r in rows if r['identity_scope']=='final_S' and r['dataset']==dataset) for dataset in ENDS},
        public_S_refit_relationships=sum('public_refit' in item for item in evidence),
        unique_public_S_refits=len({item['public_refit']['binding']['reference'] for item in evidence if 'public_refit' in item}),
        source_bindings=inputs, helper_source_binding=binding(Path(__file__), ctx),
        frozen_source_files=sources, row_evidence=evidence, alpha1_endpoint_proofs=endpoints,
        canonical_Raw_alpha1_facets=8, additional_G_alpha1_facets=8,
        endpoint_scope='Exact original-S route SHA identity only; whole-path numerical/parity validity is inherited from recovery PASS. No latency/score fields or T3 CSV were inspected.',
        interpretation='12 role rows are not12 newly fitted models. PeMS finalS cap12 is distinct from original calibrationS cap6. Raw adapter/epoch/cap are NA; one G seed92601 per dataset is shared by both S/head seeds.',
        V_numeric_metrics_inspected_or_used=False, SCORES_file_opened=False,
        cost_receipt_JSON_contains_uninspected_V_metric_fields=True,
        inspected_cost_receipt_fields=['status', 'route_spec', 'request_binding',
            'row.dataset', 'row.accuracy_run_id', 'row.cost_id', 'row.device', 'row.batch_size', 'row.schedule'],
        prediction_arrays_or_weights_read=False, GPU_execution=False,
        completion_lineage='revision2.validate_completion',
        table_binding=dict(reference=CSV_NAME, reference_base='validation_file_parent',
                           sha256=hashlib.sha256(data).hexdigest()))
    for value in inputs.values():
        resolve(value, ctx)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output/name).exists() for name in (CSV_NAME, VALIDATION_NAME)):
        raise FileExistsError('Preserve existing parent identity outputs')
    with (output/CSV_NAME).open('xb') as stream:
        stream.write(data)
    with (output/VALIDATION_NAME).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(validation, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root', required=True, type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(REPO))
    ctx = context(args.asset_root)
    result = generate(ctx, args.output_dir or ctx['report'])
    print(json.dumps(dict(status=result['status'], role_rows=result['role_rows'])))


if __name__ == '__main__':
    main()
