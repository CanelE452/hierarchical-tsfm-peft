"""Root-invoked Hog deployment costs for one retained v5 modification only."""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import re
import sys
import time
import traceback

from runtime_v5 import (HERE, ROOT, V4, Job, digest, environment_receipt,
                        load_data, save_json, source_receipt)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def receipt(path):
    path = resolve(path)
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': digest(path)}


def check_receipt(record):
    path = resolve(record['path'])
    if digest(path) != record['sha256']:
        raise ValueError(f'Cost input artifact hash changed: {path}')
    return path


def load_module(name, path):
    path = Path(path).resolve()
    if name in sys.modules:
        module = sys.modules[name]
        if Path(module.__file__).resolve() != path:
            raise RuntimeError(f'Import collision for {name}')
        return module
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def measurement_helpers():
    # Only the pure timing/schedule/summary functions are called. Their v4
    # launchers, model loader and output writers are never invoked.
    load_module('runtime_v4', V4 / 'runtime_v4.py')
    load_module('evaluate_v4', V4 / 'evaluate_v4.py')
    helpers = load_module('_v5_readonly_cost_helpers_v4', V4 / 'cost_v4.py')
    helpers.np, helpers.torch, helpers.psutil = np, torch, psutil
    return helpers


def model_entries(args):
    manifest_path = HERE / 'reuse_manifest.json'
    manifest = read_json(manifest_path)
    hog = manifest['datasets']['hog']
    for record in manifest['backbone']['files'].values():
        check_receipt(record)
    check_receipt(hog['data']['trainval'])
    entries = {}
    for role, family in [('f0', 'f0'), ('original_res', 'old_res'), ('lora', 'lora')]:
        for identifier in hog['roles'][role]:
            original = hog['models'][identifier]
            check_receipt(original['result'])
            check_receipt(original['checkpoint'])
            check_receipt(original['validation_prediction'])
            entries[identifier] = {
                'id': identifier, 'restore_version': 4, 'family': family,
                'seed': original['seed'], 'ed_mode': original['ed_mode'],
                'latent': original['latent'], 'method': 'original',
                'checkpoint': original['checkpoint'], 'result': original['result'],
                'best_val_prediction': original['validation_prediction'],
                'selected_epoch': original['selected_epoch'],
            }
    seen = set()
    for identifier in args.runs:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', identifier):
            raise ValueError('A run argument must be a local v5 run identifier')
        path = HERE / 'runs' / identifier / 'result.json'
        result = read_json(path)
        spec = result['spec']
        if result['status'] != 'complete' or spec['dataset'] != 'hog' or spec['method'] != args.method:
            raise ValueError('All new cost entries must be complete Hog fits of the retained method')
        key = (spec['variant'], spec['seed'])
        if key in seen or spec['ed_mode'] != 'fixed_ed':
            raise ValueError('Duplicate seed/variant or changed Hog E/D mode')
        seen.add(key)
        checkpoint = {'path': result['checkpoint'], 'sha256': result['checkpoint_sha256']}
        check_receipt(checkpoint)
        check_receipt(result['best_val_prediction'])
        entries[identifier] = {
            'id': identifier, 'restore_version': 5,
            'family': 'modified_res' if spec['variant'] == 'res' else 'matched_raw',
            'seed': spec['seed'], 'ed_mode': spec['ed_mode'], 'latent': spec['latent'],
            'method': args.method, 'variant': spec['variant'],
            'checkpoint': receipt(checkpoint['path']), 'result': receipt(path),
            'best_val_prediction': receipt(result['best_val_prediction']['path']),
            'selected_epoch': result['selected_epoch'],
        }
    if seen != {(variant, seed) for variant in ('res', 'raw') for seed in (92601, 92602)}:
        raise ValueError('Cost requires exactly RES/RAW x seeds 92601/92602')
    if len(entries) != 9:
        raise ValueError('Expected F0 + oldRES2 + LoRA2 + modifiedRES2 + matchedRAW2')
    order = {'f0': 0, 'old_res': 1, 'modified_res': 2, 'matched_raw': 3, 'lora': 4}
    identifiers = sorted(entries, key=lambda identifier: (order[entries[identifier]['family']],
                                                         entries[identifier]['seed'] or 0))
    return entries, identifiers, manifest


def prepare_inputs(manifest):
    data = load_data('hog', evaluation=False)
    metadata = manifest['datasets']['hog']['data']
    if receipt(data['_path']) != {key: metadata['trainval'][key] for key in ('path', 'sha256')}:
        raise ValueError('Cost data differs from the reused Hog TRAIN/VAL contract')
    origins = np.asarray(data['val_origins'], dtype=np.int64)
    if len(origins) != 30 or data['columns'].astype(str).tolist() != metadata['columns']:
        raise ValueError('Hog VAL origins or channel identities differ from the existing contract')
    contract = read_json(V4 / 'data_contract.json')
    if origins.tolist() != contract['origins']['lists']['val']:
        raise ValueError('Exact Hog VAL origin sequence differs')
    indices = np.linspace(0, len(origins) - 1, 24, dtype=int)
    chosen = origins[indices]
    inputs = torch.from_numpy(np.stack([data['x'][origin-512:origin] for origin in chosen]))
    if inputs.shape != (24, 512, 32) or inputs.dtype != torch.float32 or not torch.isfinite(inputs).all():
        raise ValueError('Expected standardized finite CPU FP32 [24,512,32]')
    return {'inputs': inputs, 'origins': chosen, 'all_val_origins': origins, 'indices': indices,
            'data': receipt(data['_path'])}


def load_candidate(entry, prepared, job, helpers):
    path = check_receipt(entry['checkpoint'])
    model_path = V4 / 'model_v4.py' if entry['restore_version'] == 4 else HERE / 'model_v5.py'
    module = load_module(f'_v5_cost_model_version_{entry["restore_version"]}', model_path)
    model = module.restore_model(path, device='cuda').eval()
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise ValueError('All model parameters must use FP32')
    info = {
        'registered_trainable_parameters_before_merge': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'total_parameters_before_merge': sum(p.numel() for p in model.parameters()),
        'task_channels': 32,
        'backbone_series_per_origin': 32 if entry['family'] in ('f0', 'lora') else entry['latent'],
        'checkpoint': entry['checkpoint'], 'result': entry['result'],
        'selected_epoch': entry['selected_epoch'],
    }
    before = torch.cat([helpers.forward_cpu(model, chunk, 'cuda') for chunk in prepared['inputs'].split(4)])
    if before.shape != (24, 48, 32) or not torch.isfinite(before).all():
        raise ValueError('Every method must return the complete original-channel forecast')
    prediction_path = check_receipt(entry['best_val_prediction'])
    with np.load(prediction_path, allow_pickle=False) as archive:
        if not np.array_equal(archive['origins'], prepared['all_val_origins']):
            raise ValueError('Stored VAL predictions and benchmark origins are not aligned')
        saved = torch.from_numpy(archive['prediction'][prepared['indices']].copy()).float()
    atol, rtol = 1e-5, 1e-4
    error = (before - saved).abs()
    torch.testing.assert_close(before, saved, atol=atol, rtol=rtol)
    info['validation_replay'] = {
        'pass': True, 'atol': atol, 'rtol': rtol, 'max_abs_error': float(error.max()),
        'elements': before.numel(), 'prediction': entry['best_val_prediction'],
        'historical_stricter_tolerance_violations': int(torch.count_nonzero(error > 1e-6 + 1e-5 * saved.abs())),
        'historical_stricter_atol': 1e-6, 'historical_stricter_rtol': 1e-5,
        'scope': 'Output restoration check against existing full-VAL predictions; no target loss is computed.',
    }
    del saved, error
    if entry['family'] == 'lora':
        job.heartbeat(f'{entry["id"]}: verify returned merged LoRA on the same VAL inputs')
        model.backbone = model.backbone.merge_and_unload(safe_merge=True)
        model.eval()
        after = torch.cat([helpers.forward_cpu(model, chunk, 'cuda') for chunk in prepared['inputs'].split(4)])
        torch.testing.assert_close(after, before, atol=atol, rtol=rtol)
        info['merge_parity'] = {'pass': True, 'atol': atol, 'rtol': rtol,
                                'max_abs_error': float((after-before).abs().max()), 'origins': 24,
                                'returned_merged_backbone_used': True}
        del after
    info['total_deployed_parameters'] = sum(p.numel() for p in model.parameters())
    info['deployed_parameter_bytes'] = sum(p.numel()*p.element_size() for p in model.parameters())
    info['deployed_buffer_bytes'] = sum(b.numel()*b.element_size() for b in model.buffers())
    info['total_deployed_tensor_bytes'] = info['deployed_parameter_bytes'] + info['deployed_buffer_bytes']
    del before
    return model, info


def measure_all(args, entries, identifiers, prepared, job, helpers):
    f0 = next(identifier for identifier in identifiers if entries[identifier]['family'] == 'f0')
    schedule = helpers.measurement_schedule(identifiers, f0, 'cuda')
    rows = []
    code_hash = digest(Path(__file__))
    contract = {'method': args.method, 'model_ids': identifiers,
                'entries': entries, 'val_origins': prepared['origins'].tolist(),
                'val_data': prepared['data'], 'order': [list(row) for row in schedule]}
    save_json(HERE / f'{args.name}_contract.json', contract)
    for block, ordinal, identifier, sentinel in schedule:
        job.check_limits()
        baseline = helpers.cuda_cleanup()
        if baseline['allocated']:
            raise RuntimeError('A previous model/output remains allocated on CUDA')
        entry = entries[identifier]
        model, info = load_candidate(entry, prepared, job, helpers)
        for batch in (1, 4):
            job.heartbeat(f'{args.name}: block{block} {identifier} {sentinel} batch{batch}')
            started = time.time()
            before = helpers.telemetry()
            measured = helpers.measure(model, prepared['inputs'], batch, 'cuda', job)
            rows.append({
                'id': identifier, 'family': entry['family'], 'method': entry['method'],
                'seed': entry['seed'], 'dataset': 'hog', 'ed_mode': entry['ed_mode'],
                'latent': entry['latent'], 'block': block, 'order_position': ordinal,
                'batch_origins': batch, 'sentinel_role': sentinel,
                'include_in_primary': sentinel != 'post', 'scope': 'deployment_cpu_to_cpu',
                'measurement_run': args.name, 'measurement_cost_source_sha256': code_hash,
                'measurement_started_utc': started, 'measurement_completed_utc': time.time(),
                'baseline_memory': baseline, 'telemetry_before': before,
                'telemetry_after': helpers.telemetry(), **info, **measured,
            })
            save_json(HERE / f'{args.name}_partial.json', {
                'status': 'running', 'method': args.method, 'contract': receipt(HERE / f'{args.name}_contract.json'),
                'rows': rows, 'expected_rows': 60, 'resume': None})
        del model
        gc.collect()
        if helpers.cuda_cleanup()['allocated']:
            raise RuntimeError('Model cleanup left CUDA allocations')
    if len(rows) != 60:
        raise ValueError('The fixed 9-model / 3-block / batch1+4 design is incomplete')
    return {'rows': rows, 'expected_rows': 60, 'model_ids': identifiers,
            'summary': helpers.cost_summary(rows), 'drift_checks': helpers.drift_checks(rows),
            'contract': receipt(HERE / f'{args.name}_contract.json'),
            'device': torch.cuda.get_device_name(), 'resume': None,
            'scope': 'Standardized FP32 CPU input -> H2D -> complete original-channel CPU [B,48,32] output; synchronized wall time. Load/disk/scoring/ledger excluded.',
            'memory_scope': 'Peak allocated/reserved after batch-shape-specific warmup; process RSS includes runtime/data. Deployed tensor bytes include parameters and buffers.',
            'claim_boundary': 'Same-session descriptive comparison only. Preserve all blocks and drift; no stable speed claim from center alone. No ratios against historical v4 timing.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('level', 'full', 'k'), required=True)
    parser.add_argument('--runs', nargs=4, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--reserve-seconds', type=float, default=1200)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', args.name):
        raise ValueError('Use a simple unique cost run name')
    if args.reserve_seconds <= 0:
        raise ValueError('Positive GPU reservation required')
    paths = {key: HERE / f'{args.name}{suffix}.json' for key, suffix in
             [('result', ''), ('partial', '_partial'), ('attempt', '_attempt'), ('contract', '_contract')]}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('Preserve original cost attempts; inspect failures and use a new name for an authorized retry')
    attempt = {'status': 'running', 'name': args.name, 'method': args.method, 'command': sys.argv,
               'started_utc': time.time(), 'source': source_receipt(), 'environment': environment_receipt(),
               'execution_scope': 'Root launches only after deciding the single modification merits retention; no automatic retry or diagnostic.'}
    save_json(paths['attempt'], attempt)
    try:
        with Job(args.name, category='gpu', reserve_seconds=args.reserve_seconds,
                 metadata={'stage': 'retained_modification_cost', 'method': args.method,
                           'real_fit_attempts': 0, 'models': 9}) as job:
            global np, torch, psutil
            import numpy as np
            import torch
            import psutil
            from threadpoolctl import threadpool_info, threadpool_limits
            torch.set_num_threads(4)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            helpers = measurement_helpers()
            entries, identifiers, manifest = model_entries(args)
            with threadpool_limits(limits=4), torch.inference_mode():
                prepared = prepare_inputs(manifest)
                result = measure_all(args, entries, identifiers, prepared, job, helpers)
                result.update(
                    schema_version=1, status='complete', name=args.name, method=args.method,
                    stage='gpu', source=source_receipt(), environment=environment_receipt(),
                    reused_helper_sources=[receipt(V4 / filename) for filename in
                                           ['cost_v4.py', 'model_v4.py', 'runtime_v4.py', 'evaluate_v4.py']],
                    reuse_manifest=receipt(HERE / 'reuse_manifest.json'),
                    val_origins=prepared['origins'].tolist(), precision='float32',
                    allow_tf32_matmul=False, allow_tf32_cudnn=False, torch_threads=torch.get_num_threads(),
                    threadpools=threadpool_info(), inference_mode=True,
                    warmup_calls=10, passes_per_order_block=20, order_blocks=3,
                    no_target_scores=True, new_fits=0, single_uninterrupted_session=True,
                    uncertainty='Timing passes/blocks are not independent scientific replications; F0 post sentinel excluded from centers but retained.',
                    completed_utc=time.time())
                save_json(paths['result'], result)
            attempt['status'] = 'complete'
            print(json.dumps({'output': str(paths['result']), 'rows': len(result['rows'])}), flush=True)
    except BaseException:
        attempt.update(status='failed', traceback=traceback.format_exc())
        raise
    finally:
        attempt['ended_utc'] = time.time()
        save_json(paths['attempt'], attempt)
        if attempt['status'] == 'failed' and paths['partial'].exists():
            partial = read_json(paths['partial'])
            partial.update(status='failed', failure_attempt=receipt(paths['attempt']))
            save_json(paths['partial'], partial)


if __name__ == '__main__':
    main()
