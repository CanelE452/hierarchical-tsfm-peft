"""No-training shape smoke and centered A60 PCA equivalence gate."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time


NAME = 'comparative_ablation_20261025_v1'
KS = {'jena': 6, 'ETTh1': 4, 'ETTh2_safe17400': 4, 'appliances': 10}
FAMILIES = {'C1': 'PCA_FIXED', 'C2': 'ADAPTS', 'A1': 'B0'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt', type=int, default=1)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    output = repo / 'reports' / NAME
    suffix = '' if args.attempt == 1 else f'_ATTEMPT{args.attempt:03d}'
    smoke_path = output / f'STEP0_SMOKE{suffix}.json'
    pca_path = output / f'PCA_SUBSPACE_VALIDATION{suffix}.json'
    if smoke_path.exists() or pca_path.exists():
        raise FileExistsError('Refusing to overwrite smoke or PCA gate output')
    sys.path.insert(0, str(repo))
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    temp = repo / '.cache' / NAME / 'runtime_audit_tmp'
    if not temp.is_dir():
        raise FileNotFoundError('The existing namespace-local runtime temporary directory is required')
    for name in ('TMP', 'TEMP', 'TMPDIR'):
        os.environ[name] = str(temp)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '2'
    opened = set()

    def audit(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            path = Path(os.path.abspath(os.fsdecode(args[0])))
            if 'hierarchical-tsfm-peft' in path.parts:
                opened.add(str(path))

    sys.addaudithook(audit)
    smoke = dict(status='INCOMPLETE', rows=[], training_executed=False,
                 V_selection_or_metrics_executed=False, T_numeric_values_parsed=0, failures=[])
    pca = dict(status='INCOMPLETE', rows=[], fit_scope='Only [0,A60_end) causal-filled normalized rows',
               sklearn_solver='full', tolerance=1e-4, training_executed=False, failures=[])
    started = time.perf_counter()
    foundation = None
    try:
        import numpy as np
        import torch
        from sklearn.decomposition import PCA
        from research.comparative_ablation_20261025_v1 import runtime
        modules = runtime.load_modules()
        fit, data = modules['fit'], modules['data']
        foundation = runtime.load_foundation()
        foundation.assert_frozen_unchanged()
        initial_hash = foundation.initial_hash
        for dataset, k in KS.items():
            task = runtime.load_task(dataset)
            fit_end = task['split_ranges']['A_stop'][1]
            origin = task['split_ranges']['V_select'][0]
            if len(task['physical']) != task['split_ranges']['T_final'][0]:
                raise ValueError('Task is not limited to pre-T rows')
            for method, family in FAMILIES.items():
                spec = dict(family=family, latent_k=k, seed=92601, device='cuda')
                model, normalization = fit.create_model(spec, task, fit_end, foundation)
                model.eval()
                history, _ = data.history_window(task, origin, normalization['mean'], normalization['std'])
                inputs = torch.as_tensor(history[None], device='cuda', dtype=torch.float32)
                with torch.no_grad():
                    value = fit.predict_model(model, inputs, foundation, family)
                expected_shape = (1, 48, len(task['columns']))
                passed = tuple(value.shape) == expected_shape and bool(torch.isfinite(value).all().item())
                row = dict(dataset=dataset, method_id=method, family=family, model_class=type(model).__name__,
                           K=k, seed=92601, input_shape=list(inputs.shape), output_shape=list(value.shape),
                           expected_shape=list(expected_shape), finite=bool(torch.isfinite(value).all().item()),
                           trainable_parameters=sum(x.numel() for x in model.parameters() if x.requires_grad),
                           evaluation_origin=int(origin), initialization_fit_end=int(fit_end),
                           status='PASS' if passed else 'FAIL')
                smoke['rows'].append(row)
                if not passed:
                    raise ValueError(f'Forward smoke failed: {dataset}/{method}')
                if method == 'C1':
                    filled = data.causal_forward_fill(task['physical'][:fit_end], task['observed'][:fit_end], normalization['mean'])
                    normalized = ((filled - normalization['mean']) / normalization['std']).astype(np.float64)
                    reference = PCA(n_components=k, svd_solver='full').fit(normalized)
                    basis = model.E.detach().cpu().numpy().astype(np.float64)
                    center = model.center.detach().cpu().numpy().astype(np.float64)
                    projector = basis @ basis.T
                    expected_projector = reference.components_.T @ reference.components_
                    reconstruction = (normalized - center) @ projector + center
                    expected_reconstruction = reference.inverse_transform(reference.transform(normalized))
                    projection_error = float(np.max(np.abs(projector - expected_projector)))
                    reconstruction_error = float(np.max(np.abs(reconstruction - expected_reconstruction)))
                    passed = max(projection_error, reconstruction_error) <= pca['tolerance']
                    pca['rows'].append(dict(dataset=dataset, K=k, fit_end=int(fit_end),
                        fitted_rows=len(normalized), channels=len(task['columns']),
                        projector_max_abs_error=projection_error, reconstruction_max_abs_error=reconstruction_error,
                        center_max_abs_error=float(np.max(np.abs(center - reference.mean_))),
                        comparison='Sign-invariant E E.T projection and centered reconstruction on identical rows',
                        status='PASS' if passed else 'FAIL'))
                    if not passed:
                        raise ValueError('G2 PCA projection/reconstruction tolerance failed: ' + dataset)
                del model, inputs, value
            foundation.assert_frozen_unchanged()
        foundation.assert_frozen_unchanged()
        smoke.update(status='PASS', F_SHA_before=initial_hash, F_SHA_after=foundation.initial_hash,
            F_frozen_assert_pass=True, logical_F_rows=foundation.streams, logical_F_calls=foundation.row_calls,
            physical_F_rows=foundation.physical_streams, physical_F_calls=foundation.physical_calls,
            dtype='float32', row_microbatch=foundation.row_microbatch, checkpoint_inputs=foundation.checkpoint_inputs)
        pca['status'] = 'PASS'
        smoke['module_sources'] = {name:dict(path=str(Path(module.__file__).resolve()),
            sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
            for name, module in modules.items() if hasattr(module, '__file__')}
    except Exception as exc:
        message = f'{type(exc).__name__}: {exc}'
        smoke['status'] = 'FAIL'
        smoke['failures'].append(message)
        if pca['status'] == 'INCOMPLETE':
            pca['status'] = 'FAIL_OR_NOT_COMPLETED'
            pca['failures'].append(message)
    finally:
        foundation = None
        gc.collect()
        if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
            sys.modules['torch'].cuda.empty_cache()
        smoke['elapsed_seconds'] = time.perf_counter() - started
        smoke['created_utc'] = datetime.now(timezone.utc).isoformat()
        smoke['opened_project_files'] = sorted(opened)
        smoke['file_audit_scope'] = 'Python open events for project paths; native model-loader IO is not fully covered'
        for path, value in ((smoke_path, smoke), (pca_path, pca)):
            with path.open('x', encoding='utf8') as stream:
                json.dump(value, stream, indent=2, ensure_ascii=False)
                stream.write('\n')
    print(json.dumps(dict(smoke_status=smoke['status'], forwards=len(smoke['rows']),
                         pca_status=pca['status'], pca_datasets=len(pca['rows']), failures=smoke['failures'])))
    return 0 if smoke['status'] == pca['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
