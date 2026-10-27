"""Load approved pre-T prefixes and frozen F once, without PCA, training or prediction."""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime, timezone


NAME = 'comparative_ablation_20261025_v1'
DATASETS = ('jena', 'ETTh1', 'ETTh2_safe17400', 'appliances')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--asset', type=Path, required=True)
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    source, asset = args.source.resolve(), args.asset.resolve()
    directory = repo / 'reports' / NAME
    suffix = '' if args.attempt == 1 else f'_ATTEMPT{args.attempt:03d}'
    results_path = directory / f'G1_RUNTIME_AUDIT{suffix}.json'
    opens_path = directory / f'OPENED_FILES_RUNTIME{suffix}.json'
    if results_path.exists() or opens_path.exists():
        raise FileExistsError('Refusing to overwrite runtime audit reports')
    opened, denied = {}, []
    temporary_root = repo / '.cache' / NAME / 'runtime_audit_tmp'
    temporary_root.mkdir(parents=True, exist_ok=True)
    for name in ('TEMP', 'TMP', 'TMPDIR'):
        os.environ[name] = str(temporary_root)
    permitted_outputs = {os.path.normcase(str(x.resolve())) for x in (results_path, opens_path)}

    def audit(event, args):
        if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path, mode, flags = args
        absolute = os.path.abspath(os.fsdecode(path))
        write = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(
            flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        opened[(absolute, str(mode), int(flags))] = True
        if os.fsdecode(path).lower() in ('nul', r'\\.\nul'):
            return
        if (write and os.path.normcase(absolute) not in permitted_outputs
                and not Path(absolute).is_relative_to(temporary_root)):
            denied.append(dict(path=absolute, mode=mode, flags=flags))
            raise PermissionError('Runtime audit forbids writes outside its reports: ' + absolute)

    sys.addaudithook(audit)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '2'
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(repo))
    broad = repo / 'research/informs_broad_validation_20261015_v1'
    sys.path.insert(0, str(broad))
    output = dict(created_utc=datetime.now(timezone.utc).isoformat(), status='INCOMPLETE',
        source_worktree=str(source), current_worktree=str(repo), datasets=[],
        training_executed=False, PCA_executed=False, V_predictions_executed=False,
        T_numeric_values_parsed=0, failures=[])
    started = time.perf_counter()
    foundation = None

    def read(path):
        return json.loads(path.read_text(encoding='utf-8-sig'))

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    try:
        import torch
        import data_contract
        import fit_base
        import gpu_runtime
        from research.b2_dataset_extension_20261018_v1 import train
        torch.set_num_threads(2)
        modules = (data_contract, fit_base, gpu_runtime)
        if any(Path(module.__file__).resolve().parent != broad for module in modules):
            raise ValueError('Scientific module loaded from wrong checkout')
        old_data = {x['dataset']: x for x in read(source / 'reports/k_tau_rule_20261023_v1/CURRENT_DATA_FINGERPRINTS.json')}
        environment = read(source / 'reports/k_tau_rule_20261023_v1/CURRENT_ENVIRONMENT.json')
        for dataset in DATASETS:
            if dataset == 'jena':
                primitive_path = broad / 'priority_b2_validation_20261016_v1/safe_jena_prefix.py'
                spec = importlib.util.spec_from_file_location('comparative_safe_jena', primitive_path)
                primitive = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(primitive)
                contract = read(broad / 'DATA_CONTRACT.json')['assets']['jena']
                task, access = primitive.load_jena_prefix(asset, contract)
                current_prefix = task['materialized_prefix_sha256']
                expected_prefix = old_data[dataset]['materialized_prefix_sha256']
                prefix_path = asset / primitive.SOURCE_REL
                current_config = None
            else:
                data_root = source / '.cache/b2_dataset_search_20261019_v1/data' / dataset
                prefix_path, config_path = data_root / 'PREFIX_PHYSICAL.npz', data_root / 'DATA_DIAGNOSTIC.json'
                identity = read(source / 'reports/b2_dataset_search_20261019_v1/datasets' / dataset / 'TRAIN_IDENTITY.json')
                config, task = train.load_task(config_path, prefix_path)
                current_prefix, expected_prefix = sha(prefix_path), identity['prefix_sha256']
                current_config = sha(config_path)
                if current_config != identity['input_config_sha256']:
                    raise ValueError('Dataset configuration SHA differs: ' + dataset)
                access = dict(numeric_scope='Existing approved prefix archive only',
                              T_numeric_values_parsed=0, prefix_rows=len(task['physical']))
            historical = old_data[dataset]
            boundaries = {k: list(v) for k, v in task['split_ranges'].items()}
            checks = dict(data_sha=task['data_sha256'] == historical['data_sha256'],
                          array_sha=task['array_sha256'] == historical['array_sha256'],
                          split_ranges=boundaries == historical['split_ranges'],
                          prefix_sha=current_prefix == expected_prefix,
                          materialized_rows=len(task['physical']) == historical['materialized_rows'],
                          ends_before_T=len(task['physical']) == boundaries['T_final'][0],
                          L_H=(task['L'], task['H']) == (512, 48))
            selected_k = {'jena': 6, 'ETTh1': 4, 'ETTh2_safe17400': 4, 'appliances': 10}[dataset]
            historical_result = read(source / '.cache/k_tau_rule_20261023_v1/results' /
                                     f'KTAU_main_{dataset}_K{selected_k}_s92603.json')['result']
            normalization = []
            for label, bound, historical_key in (('A_fit', boundaries['A_fit'][1], 'search_summary'),
                                                  ('A60', boundaries['A_stop'][1], 'refit_summary')):
                mean, std, count = data_contract.fit_statistics(task, bound)
                value = fit_base.scaler_fingerprint(mean, std)
                expected = historical_result[historical_key]['contract']['scaler_sha']
                normalization.append(dict(stage=label, fit_end=bound, scaler_sha256=value,
                                          expected_sha256=expected, matches=value == expected,
                                          observed_counts=count.tolist()))
            checks['A_fit_A60_normalization'] = all(x['matches'] for x in normalization)
            item = dict(dataset=dataset, status='PASS' if all(checks.values()) else 'FAIL', checks=checks,
                        data_sha256=task['data_sha256'], split_ranges=boundaries,
                        prefix_path=str(prefix_path), prefix_sha256=current_prefix,
                        expected_prefix_sha256=expected_prefix, config_sha256=current_config,
                        materialized_rows=len(task['physical']), normalization=normalization,
                        numeric_access=access,
                        data_identity_provenance=task.get('data_identity_provenance', 'RECOMPUTED_FROM_APPROVED_PREFIX'))
            output['datasets'].append(item)
            if not all(checks.values()):
                raise ValueError('Current dataset or normalization contract mismatch: ' + dataset)
        model_path = Path(environment['model_path']).resolve()
        if not model_path.is_relative_to(asset):
            raise ValueError('Foundation model asset escapes approved asset root')
        foundation = gpu_runtime.FrozenBolt(model_path, 48, row_microbatch=128,
                                            checkpoint_inputs=False, device='cuda')
        foundation.assert_frozen_unchanged()
        output['foundation'] = dict(F_SHA=foundation.initial_hash, expected_F_SHA=environment['F_SHA'],
            matches=foundation.initial_hash == environment['F_SHA'], model_path=str(model_path),
            revision=foundation.model_revision, dtype=str(next(foundation.model.parameters()).dtype),
            row_microbatch=foundation.row_microbatch, checkpoint_inputs=foundation.checkpoint_inputs,
            logical_F_rows=foundation.streams, logical_F_calls=foundation.row_calls,
            physical_F_rows=foundation.physical_streams, physical_F_calls=foundation.physical_calls,
            load_seconds=foundation.load_seconds, all_parameters_frozen=all(not p.requires_grad for p in foundation.model.parameters()),
            GPU=torch.cuda.get_device_name(0), torch=torch.__version__, cuda_build=torch.version.cuda)
        if not output['foundation']['matches']:
            raise ValueError('Current frozen foundation F_SHA differs from K_tau')
        output['status'] = 'PASS_CURRENT_F_DATA_SPLITS_NORMALIZATION'
    except Exception as exc:
        output['status'] = 'FAIL'
        output['failures'].append(f'{type(exc).__name__}: {exc}')
    finally:
        foundation = None
        gc.collect()
        if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
            sys.modules['torch'].cuda.empty_cache()
        output['elapsed_seconds'] = time.perf_counter() - started
        output['denied_writes'] = denied
        with results_path.open('x', encoding='utf8') as stream:
            json.dump(output, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
        file_log = dict(scope='Python audit open events after scientific-audit hook registration; native loader reads may not emit Python open events',
            source_script=str(Path(__file__).resolve()), T_numeric_values_parsed=0,
            native_model_assets=[str(Path(environment['model_path']) / name) for name in ('config.json','model.safetensors')] if 'environment' in locals() else [],
            denied_writes=denied,
            opens=[dict(path=k[0], mode=k[1], flags=k[2]) for k in sorted(opened)])
        with opens_path.open('x', encoding='utf8') as stream:
            json.dump(file_log, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
    print(json.dumps(dict(status=output['status'], dataset_count=len(output['datasets']),
                         foundation=output.get('foundation'), failures=output['failures'])))
    return 0 if output['status'].startswith('PASS') else 1


if __name__ == '__main__':
    raise SystemExit(main())
