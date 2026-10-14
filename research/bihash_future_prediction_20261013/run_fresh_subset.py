"""Use the sealed fresh algorithm for a declared dataset-only recovery attempt."""
from pathlib import Path
import argparse
import time
import traceback
import shutil
import runtime as rt
import run as ru
import merge_fresh_results as mf

EXPECTED_METHOD_LOCK_SHA = '3338c264ce51d82f9104a0af90fc29c97d6511aa8ec0a6385e929d99317fdb29'


def validate_request(datasets, tag, method_lock, root=rt.ROOT, expected_lock_sha=EXPECTED_METHOD_LOCK_SHA):
    mf.valid_tag(tag)
    if not datasets or len(set(datasets)) != len(datasets) or not set(datasets) <= set(mf.DATASETS):
        raise ValueError('INVALID_OR_DUPLICATE_FRESH_SUBSET_DATASET')
    root = Path(root); here = root / 'research/bihash_future_prediction_20261013'; cache = root / '.cache/bihash_future_prediction_20261013'
    for path in (here / ('run_' + tag), cache / ('run_' + tag), cache / 'source_snapshots' / tag):
        if path.exists(): raise FileExistsError('RECOVERY_TAG_ALREADY_EXISTS: ' + str(path))
    path = Path(method_lock).resolve()
    if not path.is_file() or rt.sha(path) != expected_lock_sha: raise ValueError('METHOD_LOCK_NOT_IDENTICAL_TO_APPROVED_LOCK')
    lock = rt.read(path)
    if not lock.get('locked') or len(lock['source_hashes']) != 16: raise ValueError('METHOD_LOCK_NOT_SEALED_SOURCE16')
    for relative, digest in lock['source_hashes'].items():
        if rt.sha(root / relative) != digest: raise ValueError('SCIENTIFIC_SOURCE_CHANGED_BEFORE_GPU')
    return lock


def contracts_for_reuse(datasets, method_lock, root=rt.ROOT):
    cache = Path(root) / '.cache/bihash_future_prediction_20261013'
    lock_sha = rt.sha(method_lock)
    contracts, metadata = {}, {}
    for dataset in datasets:
        registry_path = cache / 'contract' / (dataset + '_' + lock_sha + '.json')
        base = cache / dataset / lock_sha
        if registry_path.is_file():
            registry = rt.read(registry_path)
            contract_path = mf.resolve(root, registry['contract_path'])
            contract = rt.read(contract_path)
            checked = mf.reference(contract, root, lock_sha)
            contracts[dataset] = contract
            metadata[dataset] = {'mode': 'REUSE_REGISTERED_IMMUTABLE_CONTRACT',
                                 'contract': checked['contract_artifact'], 'fit_tune_payload': checked['fit_tune_payload'],
                                 'EVAL_payloads_not_opened_or_hashed': checked['unopened_EVAL_payloads']}
        else:
            if base.exists() and any(path.is_file() for path in base.rglob('*')):
                raise ValueError('PARTIAL_CANONICAL_DATA_PRESERVED_REFUSE_OVERWRITE: ' + str(base))
            metadata[dataset] = {'mode': 'FIRST_MATERIALIZATION_BY_FROZEN_PREPARE_NEW', 'canonical_path': str(base.relative_to(root))}
    return contracts, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', action='append', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--method-lock', type=Path, default=rt.HERE / 'METHOD_LOCK.json')
    parser.add_argument('--data-repair-result', type=Path)
    args = parser.parse_args()
    method_lock = args.method_lock.resolve()
    lock = validate_request(args.dataset, args.tag, method_lock)
    reuse_contracts, materialization = contracts_for_reuse(args.dataset, method_lock)
    if not (rt.HERE / 'PREFLIGHT.json').is_file(): raise RuntimeError('PREFLIGHT_NOT_SEALED_NO_GPU_START')
    if args.data_repair_result is not None and not args.data_repair_result.is_file(): raise FileNotFoundError('DATA_REPAIR_RESULT_MISSING')
    folder = rt.HERE / ('run_' + args.tag); folder.mkdir(parents=True, exist_ok=False)
    sources = rt.source_snapshot(args.tag)
    if sources != lock['source_hashes']: raise ValueError('RECOVERY_SNAPSHOT_NOT_SEALED_SOURCE16')
    snapshot_folder = rt.CACHE / 'source_snapshots' / args.tag
    snapshot_script = snapshot_folder / 'RECOVERY_PRODUCER.py'
    with Path(__file__).open('rb') as source, snapshot_script.open('xb') as target: shutil.copyfileobj(source, target)
    snapshot_helper = snapshot_folder / 'RECOVERY_MERGE_HELPER.py'
    with Path(mf.__file__).open('rb') as source, snapshot_helper.open('xb') as target: shutil.copyfileobj(source, target)
    output = {'tag': args.tag, 'stage': 'fresh', 'started_unix': time.time(), 'source_hashes': sources,
              'execution_subset': list(args.dataset), 'selected_datasets': list(args.dataset), 'failed_paths': [], 'protected_TEST_opened': False,
              'EVAL_future_target_opened': False, 'previous_cases_retrained': False,
              'producer': {'script': mf.ref(__file__, rt.ROOT), 'snapshot_script': mf.ref(snapshot_script, rt.ROOT),
                           'helper_script': mf.ref(mf.__file__, rt.ROOT), 'snapshot_helper_script': mf.ref(snapshot_helper, rt.ROOT),
                           'algorithm': 'Frozen run.fresh learning called unchanged; FRESH iteration tuple restricted. Existing registered data returned without regeneration, new data uses original prepare_new.',
                           'scientific_fresh_function_source': {'path': 'research/bihash_future_prediction_20261013/run.py', 'sha256': sources['research/bihash_future_prediction_20261013/run.py']},
                           'planned_core_updates': 600, 'planned_future_AE_updates': 400},
              'source_snapshot': mf.ref(snapshot_folder / 'MANIFEST.json', rt.ROOT), 'materialization_policy': materialization}
    if args.data_repair_result is not None: output['data_repair_result'] = mf.ref(args.data_repair_result, rt.ROOT)
    limit = rt.Limit(args.tag, gpu=True); status, error = 'COMPLETE', None
    try:
        ru.bm.en.configure()
        forecaster = ru.bm.en.ex.previous.FrozenBolt(); ru.bm.en.replace_normalizer(forecaster)
        before = rt.state_hash(forecaster.model); output['backbone_before_sha256'] = before
        original_datasets = ru.FRESH
        original_prepare = ru.dt.prepare_new
        def guarded_prepare(dataset, supplied_lock):
            if dataset not in args.dataset or Path(supplied_lock).resolve() != method_lock:
                raise ValueError('FRESH_PREPARE_OUTSIDE_DECLARED_SUBSET_OR_LOCK')
            if dataset in reuse_contracts:
                existing = ru.dt._load_registered_contract(dataset, method_lock)
                if existing != reuse_contracts[dataset]: raise ValueError('REUSED_CONTRACT_CHANGED_DURING_RECOVERY')
                return existing
            return original_prepare(dataset, supplied_lock)
        try:
            ru.FRESH = tuple(args.dataset)
            ru.dt.prepare_new = guarded_prepare
            output = ru.fresh(output, forecaster, limit, sources, folder, method_lock)
        finally:
            ru.FRESH = original_datasets
            ru.dt.prepare_new = original_prepare
        ru.source_check(sources)
        if rt.sha(__file__) != output['producer']['script']['sha256']: raise ValueError('RECOVERY_PRODUCER_CHANGED_WHILE_ACTIVE')
        if rt.sha(mf.__file__) != output['producer']['helper_script']['sha256']: raise ValueError('RECOVERY_HELPER_CHANGED_WHILE_ACTIVE')
        for dataset in reuse_contracts:
            for key in ('contract', 'fit_tune_payload'): mf.verify(rt.ROOT, materialization[dataset][key])
        output['backbone_after_sha256'] = rt.state_hash(forecaster.model)
        if output['backbone_after_sha256'] != before: raise ValueError('BACKBONE_CHANGED')
        expected = {(ds, seed, method, lock['weights'][method]) for ds in args.dataset for seed in mf.SEEDS for method in mf.METHODS}
        actual = [(c['dataset'], c['seed'], c['method'], c['weight']) for c in output['cases']]
        ae = [(a['dataset'], a['seed']) for a in output['future_pretrains']]
        if len(actual) != len(expected) or set(actual) != expected or len(ae) != 3*len(args.dataset) or set(ae) != {(ds, seed) for ds in args.dataset for seed in mf.SEEDS}:
            raise ValueError('SUBSET_ACTUAL_COVERAGE_INCOMPLETE')
        if any(c['training']['updates'] != 600 or len(c['training']['curve']) != 600 for c in output['cases']): raise ValueError('SUBSET_CORE_NOT_600')
        if any(a['updates'] != 400 or len(a['curve']) != 400 for a in output['future_pretrains']): raise ValueError('SUBSET_FUTURE_AE_NOT_400')
        if output['failed_paths']: status = 'FAILED_PRESERVED'
    except Exception as exc:
        status = 'BUDGET_HELD_PRESERVED' if isinstance(exc, rt.BudgetHeld) else 'FAILED_PRESERVED'
        error = str(exc); output.update(error=error, traceback=traceback.format_exc())
    output.update(status=status, finished_unix=time.time(), resource_receipt=limit.close(status, error))
    rt.write(folder / 'RESULT.json', output)
    print(rt.json_line({'fresh_subset_terminal': str(folder / 'RESULT.json'), 'status': status, 'case_count': len(output.get('cases', []))}), flush=True)
    if status != 'COMPLETE': raise SystemExit(2)


if __name__ == '__main__': main()
