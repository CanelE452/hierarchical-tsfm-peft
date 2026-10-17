"""Split-bound DCA preparation, using the pinned official objective and estimator.

Official code remains in a verified private source cache with its LICENSE.txt.
Imports and deployment loaders neither download sources nor optimize projections.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import multiprocessing
from pathlib import Path
import sys
import time

import numpy as np
import scipy
from scipy.optimize import minimize
import sklearn
import torch

from research.informs_broad_validation_20261015_v1 import data_contract, evaluation


DCA_COMMIT = 'a140e6fa9524a3bd6d043ae9e96f6e149b9c3745'
SETTINGS = dict(T=5, n_init=1, stride=1, chunk_cov_estimate=None, tol=1e-6,
                ortho_lambda=10., verbose=False, block_toeplitz=False,
                method='toeplitzify', regularization=None, device='cpu', dtype='float64',
                rng_or_seed=0, maxiter=200, maxfun=1000, max_seconds=120.,
                rotation='thin-SVD-orth_then_QtPCA-SVD_UVt_v1')
MANIFEST = Path(__file__).resolve().parents[2] / 'reports/adapter_improvements_20261018_v1/OFFICIAL_SOURCE_MANIFEST.json'


def _sha(path):
    return data_contract.sha256_file(path)


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _array_sha(value):
    return data_contract.array_sha256(np.asarray(value))


def _get(ctx, key, default=None):
    return ctx.get(key, default) if isinstance(ctx, dict) else getattr(ctx, key, default)


def official_api(source_root):
    """Load only verified official modules, avoiding unrelated package imports."""
    source_root = Path(source_root)
    root = source_root / 'DCA' if (source_root / 'DCA').is_dir() else source_root
    source = next(item for item in json.loads(MANIFEST.read_text(encoding='utf-8'))['sources'] if item['name'] == 'DCA')
    if source['commit'] != DCA_COMMIT:
        raise ValueError('Official DCA commit mismatch')
    required = {'src/dca/base.py', 'src/dca/cov_util.py', 'src/dca/dca.py', 'LICENSE.txt'}
    for item in source['files']:
        if item['path'] in required:
            path = root / item['path']
            if not path.is_file() or _sha(path) != item['sha256']:
                raise ValueError('Official source hash mismatch: ' + item['path'])
            required.remove(item['path'])
    if required:
        raise ValueError('Official source manifest incomplete')
    name = '_adapter_dca_' + DCA_COMMIT + '_' + hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:12]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, root / 'src/dca/__init__.py',
                                                     submodule_search_locations=[str(root / 'src/dca')])
        sys.modules[name] = importlib.util.module_from_spec(spec)
    return {key: importlib.import_module(name + '.' + module)
            for key, module in [('base', 'base'), ('cov', 'cov_util'), ('dca', 'dca')]}


def _spd(matrix):
    matrix = np.asarray(matrix, np.float64)
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T, rtol=1e-11, atol=1e-11):
        raise ValueError('Covariance must be finite and symmetric positive definite')
    sign, logdet = np.linalg.slogdet(matrix)
    if sign != 1 or not np.isfinite(logdet):
        raise ValueError('Covariance is not positive definite (slogdet sign)')
    try:
        np.linalg.cholesky(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError('Covariance is not positive definite (Cholesky)') from exc
    return dict(min_eigenvalue=float(np.linalg.eigvalsh(matrix)[0]), slogdet_sign=float(sign),
                logdet=float(logdet), symmetry_error=float(np.max(np.abs(matrix-matrix.T))))


def _block_cov(cross):
    n = len(cross)
    return np.block([[cross[abs(i-j)] if i > j else cross[abs(i-j)].T
                      for j in range(n)] for i in range(n)])


def gaussian_pi(cross_covs, projection):
    """Independent sign/Cholesky-checked verification of Gaussian PI, in nats."""
    cross = np.asarray(cross_covs, np.float64)
    projection = np.asarray(projection, np.float64)
    if len(cross) % 2 or len(cross) < 2:
        raise ValueError('Gaussian PI requires 2*T covariance lags')
    projected = projection.T @ cross @ projection
    cov = _block_cov(projected)
    half = cov.shape[0] // 2
    joint, marginal = _spd(cov), _spd(cov[:half, :half])
    return marginal['logdet'] - .5*joint['logdet']


class FreshObjective:
    """Cache detached values/gradients only; every new graph owns a fresh leaf."""
    def __init__(self, loss, deadline=None, maxfun=1000):
        self.loss = loss
        self.deadline = deadline
        self.maxfun = maxfun
        self.params = self.value = self.gradient = None
        self.n_eval = 0

    def _evaluate(self, parameters):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise TimeoutError('DCA fit deadline reached')
        parameters = np.asarray(parameters, np.float64)
        if self.params is None or not np.array_equal(parameters, self.params):
            if self.n_eval >= self.maxfun:
                raise _EvaluationLimit('DCA function evaluation limit reached')
            leaf = torch.tensor(parameters, dtype=torch.float64, requires_grad=True)
            value = self.loss(leaf)
            gradient, = torch.autograd.grad(value, leaf)
            self.value = float(value.detach())
            self.gradient = gradient.detach().cpu().numpy().copy()
            if not np.isfinite(self.value) or not np.isfinite(self.gradient).all():
                raise ValueError('Nonfinite official DCA objective or gradient')
            self.params = parameters.copy()
            self.n_eval += 1
        return self.value, self.gradient

    def func(self, parameters):
        return self._evaluate(parameters)[0]

    def grad(self, parameters):
        return self._evaluate(parameters)[1].copy()


class _EvaluationLimit(RuntimeError):
    pass


def official_statistics(stream, source_root, T=5, deadline=None, init=None):
    """Instrument rectification while calling the unchanged official estimator."""
    api = official_api(source_root)
    cov = api['cov']
    diagnostics = {}
    original = cov.rectify_spectrum

    def rectify(matrix, epsilon=1e-6, logger=None):
        minimum = float(scipy.linalg.eigvalsh(matrix)[0])
        diagnostics.update(min_eigenvalue_before=minimum,
                           rectification_shift=(-minimum+epsilon) if minimum <= 0 else 0.)
        original(matrix, epsilon=epsilon, logger=logger)
        diagnostics['covariance'] = _spd(matrix)
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('DCA deadline reached during covariance estimation')

    cov.rectify_spectrum = rectify
    try:
        initial = np.ones((stream.shape[1], 1)) if init is None else np.asarray(init, np.float64)
        model = api['dca'].DynamicalComponentsAnalysis(
            d=initial.shape[1], T=T, init=initial, n_init=1, stride=1,
            chunk_cov_estimate=None, tol=1e-6, ortho_lambda=10., verbose=False,
            block_toeplitz=False, method='toeplitzify', device='cpu',
            dtype=torch.float64, rng_or_seed=0)
        model.estimate_data_statistics(np.asarray(stream, np.float64), regularization=None)
    finally:
        cov.rectify_spectrum = original
    cross = model.cross_covs.detach().cpu().numpy().copy()
    diagnostics.update(lag_rows=int(len(stream)-2*T+1), first_lag_row=0,
                       last_lag_row=int(len(stream)-2*T), joint_window=2*T,
                       internal_mean=model.mean_.ravel().tolist(),
                       reconstructed_covariance=_spd(_block_cov(cross)))
    return cross, diagnostics


def align_to_pca(projection, pca):
    projection, pca = np.asarray(projection, np.float64), np.asarray(pca, np.float64)
    if projection.shape != pca.shape or not np.isfinite(projection).all():
        raise ValueError('DCA/PCA projection shape or finite-value mismatch')
    k = pca.shape[1]
    rank = int(np.linalg.matrix_rank(projection))
    if rank != k:
        raise ValueError('DCA projection lost rank K')
    q = scipy.linalg.orth(projection)
    u, singular, vt = np.linalg.svd(q.T @ pca, full_matrices=False)
    rotation = u @ vt
    aligned = q @ rotation
    return aligned, dict(rank=rank, projection_singular_values=np.linalg.svd(projection, compute_uv=False).tolist(),
                         alignment_singular_values=singular.tolist(),
                         alignment_rank=int(np.linalg.matrix_rank(q.T@pca)),
                         alignment_nonunique=bool(np.linalg.matrix_rank(q.T@pca) < k),
                         alignment_error=float(np.linalg.norm(aligned-pca)),
                         orthogonality_error=float(np.linalg.norm(aligned.T@aligned-np.eye(k))),
                         subspace_error=float(np.linalg.norm(aligned@aligned.T-q@q.T)),
                         rotation_sha256=_array_sha(rotation), E64_sha256=_array_sha(aligned))


def _fit_direct(stream, pca, source_root, limits, deadline):
    started = time.monotonic()
    torch.set_num_threads(2)
    cross, covariance = official_statistics(stream, source_root, SETTINGS['T'], deadline, pca)
    api = official_api(source_root)
    c = torch.tensor(cross, dtype=torch.float64)
    k = pca.shape[1]
    initial = api['base'].init_coef(pca.shape[0], k, np.random.RandomState(0), np.asarray(pca, np.float64))
    objective = FreshObjective(api['dca'].build_loss(c, k, 10., False), deadline, limits['maxfun'])
    first_loss = objective.func(initial.ravel())
    first_pi = gaussian_pi(cross, initial)
    try:
        result = minimize(objective.func, initial.ravel(), method='L-BFGS-B', jac=objective.grad,
                          options={'ftol': 1e-6, 'maxiter': limits['maxiter'], 'maxfun': limits['maxfun']})
    except _EvaluationLimit as exc:
        return dict(status='NOT_CONVERGED', converged=False, optimizer_message=str(exc),
                    nfev=objective.n_eval, initial_objective=first_loss, initial_pi=first_pi,
                    covariance=covariance, wall_seconds=time.monotonic()-started)
    diagnostics = dict(status='COMPLETE' if result.success else 'NOT_CONVERGED',
                       converged=bool(result.success), optimizer_status=int(result.status),
                       optimizer_message=str(result.message), nit=int(result.nit), nfev=int(result.nfev),
                       njev=int(result.njev), actual_graph_evaluations=objective.n_eval,
                       initial_objective=first_loss, initial_pi=first_pi,
                       final_objective=float(result.fun), covariance=covariance,
                       wall_seconds=time.monotonic()-started)
    if not result.success:
        return diagnostics
    raw = result.x.reshape(pca.shape)
    aligned, rotation = align_to_pca(raw, pca)
    pi_raw, pi_aligned = gaussian_pi(cross, raw), gaussian_pi(cross, aligned)
    if abs(pi_raw-pi_aligned) > 1e-8 * max(1., abs(pi_raw)):
        raise ValueError('DCA orthogonalization/rotation failed PI invariance')
    diagnostics.update(rotation=rotation, final_pi=pi_raw, aligned_pi=pi_aligned,
                       pi_invariance_error=abs(pi_raw-pi_aligned),
                       E_movement_fro=float(np.linalg.norm(aligned-pca)),
                       ortho_penalty=float(api['base'].ortho_reg_fn(10., raw)),
                       E64=aligned, cross_covariances=cross)
    return diagnostics


def _fit_child(connection, stream, pca, source_root, limits, deadline):
    cpu_started = time.process_time()
    try:
        result = _fit_direct(stream, pca, source_root, limits, deadline)
    except TimeoutError as exc:
        result = dict(status='TIMEOUT', converged=False, reason=str(exc))
    except Exception as exc:
        result = dict(status='FAILED_NUMERICAL', converged=False, reason=type(exc).__name__+': '+str(exc))
    result['process_cpu_seconds'] = time.process_time()-cpu_started
    try:
        connection.send(result)
    finally:
        connection.close()


def fit_official_dca(stream, pca, source_root, *, maxiter=200, maxfun=1000, max_seconds=120.):
    """One isolated CPU fit; the parent enforces wall time across covariance+solve."""
    if maxiter > 200 or maxiter < 0 or not 1 <= maxfun <= 1000 or not 0 < max_seconds <= 120:
        raise ValueError('DCA solver exceeds registered operational limits')
    started = time.monotonic()
    deadline = started + max_seconds
    limits = dict(maxiter=maxiter, maxfun=maxfun, max_seconds=max_seconds)
    spawn = multiprocessing.get_context('spawn')
    parent, child = spawn.Pipe(duplex=False)
    process = spawn.Process(target=_fit_child, args=(child, stream, pca, str(source_root), limits, deadline))
    process.start()
    child.close()
    try:
        if parent.poll(max(0., deadline-time.monotonic())):
            try:
                result = parent.recv()
            except EOFError:
                result = dict(status='FAILED_NUMERICAL', converged=False, reason='DCA child ended without result')
        else:
            result = dict(status='TIMEOUT', converged=False, reason='120s covariance+solver deadline (or stricter test bound)')
    finally:
        if process.is_alive():
            process.terminate()
        process.join()
        parent.close()
    result['wall_seconds'] = time.monotonic()-started
    result['solver_limits'] = limits
    return result


def _norm_arrays(pca_norm):
    basis = pca_norm.get('E', pca_norm.get('basis'))
    if basis is None or pca_norm.get('pca_center') is None:
        raise ValueError('PCA normalization must include basis and pca_center')
    return tuple(np.asarray(v, np.float32) for v in
                 (pca_norm['mean'], pca_norm['std'], pca_norm['pca_center'], basis))


def initialization_binding(task, fit_end, k, pca_norm):
    fit_end, k = int(fit_end), int(k)
    splits = task['split_ranges']
    scopes = {int(splits['A_fit'][1]): 'A_fit', int(splits['A_stop'][1]): 'A60'}
    if fit_end not in scopes or not 10 < fit_end <= len(task['physical']):
        raise ValueError('DCA fitting scope must be registered A_fit or A60')
    if int(pca_norm.get('fit_end', fit_end)) != fit_end:
        raise ValueError('PCA/DCA fitting scope mismatch')
    mean, std, center, pca = _norm_arrays(pca_norm)
    channels = task['physical'].shape[1]
    if pca.shape != (channels, k) or mean.shape != (channels,) or std.shape != mean.shape or center.shape != mean.shape:
        raise ValueError('PCA/DCA axes mismatch')
    if not all(np.isfinite(a).all() for a in (mean, std, center, pca)) or np.any(std <= 0):
        raise ValueError('Invalid PCA scaler or basis')
    producer = {name: _sha(path) for name, path in
                [('initialization.py', Path(__file__)), ('data_contract.py', Path(data_contract.__file__))]}
    versions = dict(numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__, sklearn=sklearn.__version__)
    binding = dict(schema='dca-init-v1', dataset=str(task.get('name', task.get('dataset', 'UNKNOWN'))),
                   fit_scope=dict(name=scopes[fit_end], start=0, fit_end=fit_end, covariance_rows=fit_end,
                                  joint_window=10, lag_pair_max_row=fit_end-1), k=k, channels=channels,
                   physical_prefix_sha256=_array_sha(task['physical'][:fit_end]),
                   mask_prefix_sha256=_array_sha(np.asarray(task['observed'][:fit_end], bool)),
                   time_prefix_sha256=_array_sha(task['times'][:fit_end]) if 'times' in task else None,
                   channel_order_sha256=_fingerprint(task.get('columns', list(range(channels)))),
                   scaler_sha256=_fingerprint(dict(mean=evaluation.array_fingerprint(mean),
                                                   std=evaluation.array_fingerprint(std))),
                   center_sha256=_array_sha(center), pca_sha256=_array_sha(pca),
                   observed_counts_sha256=_array_sha(np.asarray(task['observed'][:fit_end], bool).sum(0)),
                   official_commit=DCA_COMMIT,
                   official_source_manifest_sha256=_fingerprint(next(s for s in json.loads(MANIFEST.read_text(encoding='utf-8'))['sources'] if s['name'] == 'DCA')),
                   producer_sources=producer, versions=versions, settings=SETTINGS)
    return dict(key=_fingerprint(binding), binding=binding)


def prepare_stream(task, fit_end, k, pca_norm):
    binding = initialization_binding(task, fit_end, k, pca_norm)['binding']
    mean, std, center, pca = _norm_arrays(pca_norm)
    expected = data_contract.initialize_pca(task, fit_end, k)
    for supplied, exact in [(mean, expected['mean']), (std, expected['std']),
                            (center, expected['pca_center']), (pca, expected['E'])]:
        if not np.array_equal(supplied, exact):
            raise ValueError('DCA must use exact fitting-scope PCA normalization')
    filled = data_contract.causal_forward_fill(task['physical'][:fit_end], task['observed'][:fit_end], mean)
    normalized = ((filled-mean)/std).astype(np.float64)
    # Official estimation centers this stream internally; model.center stays PCA's FP32 center.
    missing = int(np.count_nonzero(~np.asarray(task['observed'][:fit_end], bool)))
    return normalized, dict(fit_scope=binding['fit_scope'], imputed_count=missing,
                            imputed_fraction=missing/filled.size,
                            normalized_stream_sha256=_array_sha(normalized),
                            center_residual_max=float(np.max(np.abs(normalized.mean(0)-center.astype(np.float64)))))


def install_initialization(model, e32):
    e32 = np.asarray(e32)
    if e32.dtype != np.float32 or model.E.dtype != torch.float32 or model.D.dtype != torch.float32:
        raise ValueError('Install actual FP32 E before computing its FP32 pinv')
    if tuple(model.E.shape) != e32.shape or np.linalg.matrix_rank(e32.astype(np.float64)) != e32.shape[1]:
        raise ValueError('DCA initialization shape/rank mismatch')
    with torch.no_grad():
        model.E.copy_(torch.as_tensor(e32, device=model.E.device))
        model.D.zero_()
        model.D[0].copy_(torch.linalg.pinv(model.E.detach()))


def _cache_paths(ctx, identity):
    root = Path(_get(ctx, 'cache_root', _get(ctx, 'cache')))
    directory = root / 'initialization' / identity['key']
    return root, directory / 'initialization.npz', directory / 'receipt.json'


def _read_receipt(path, identity):
    receipt = json.loads(path.read_text(encoding='utf-8'))
    if receipt.get('identity') != identity:
        raise ValueError('DCA initialization scope/source hash mismatch')
    return receipt


def fit_dca_initialization(ctx, task, fit_end, k, pca_norm):
    started = time.monotonic()
    identity = initialization_binding(task, fit_end, k, pca_norm)
    root, arrays_path, receipt_path = _cache_paths(ctx, identity)
    if receipt_path.is_file():
        receipt = _read_receipt(receipt_path, identity)
        if receipt['public']['status'] == 'COMPLETE':
            _, public = load_initialization(ctx, task, fit_end, k, pca_norm)
            return public
        return receipt['public']
    stream, stream_meta = prepare_stream(task, fit_end, k, pca_norm)
    source_root = _get(ctx, 'official_source_root', root / 'official')
    _, _, _, pca = _norm_arrays(pca_norm)
    result = fit_official_dca(stream, pca.astype(np.float64), source_root)
    result.update(stream=stream_meta, preparation_wall_seconds=time.monotonic()-started,
                  seed_sharing='one deterministic fit per dataset/K/scope; separate A_fit/A60',
                  F_calls=0)
    public = dict(status=result['status'], converged=result['converged'], fit_scope=identity['binding']['fit_scope'],
                  cache_sha256=None, cache_binding=None, identity_sha256=identity['key'],
                  scaler_sha256=identity['binding']['scaler_sha256'], center_sha256=identity['binding']['center_sha256'],
                  pca_sha256=identity['binding']['pca_sha256'], diagnostics=result)
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    if result['status'] == 'COMPLETE':
        e64 = result.pop('E64')
        cross = result.pop('cross_covariances')
        e32 = e64.astype(np.float32)
        d32 = torch.linalg.pinv(torch.from_numpy(e32)).numpy()
        result.update(E32_sha256=_array_sha(e32), D0_32_sha256=_array_sha(d32),
                      E32_rank=int(np.linalg.matrix_rank(e32.astype(np.float64))))
        if result['E32_rank'] != k:
            raise ValueError('FP32 DCA projection lost rank K')
        with arrays_path.open('xb') as handle:
            np.savez_compressed(handle, E64=e64, E32=e32, D0_32=d32, cross_covariances=cross,
                                pca=pca, center=_norm_arrays(pca_norm)[2])
        public.update(cache_sha256=_sha(arrays_path), cache_binding=arrays_path.relative_to(root).as_posix())
    with receipt_path.open('x', encoding='utf-8') as handle:
        json.dump(dict(identity=identity, public=public), handle, indent=2, allow_nan=False)
    return public


def load_initialization(ctx, task, fit_end, k, pca_norm):
    """Read-only deployment: validate scope/source/bytes, never fit or write."""
    identity = initialization_binding(task, fit_end, k, pca_norm)
    _, arrays_path, receipt_path = _cache_paths(ctx, identity)
    receipt = _read_receipt(receipt_path, identity)
    public = receipt['public']
    if public['status'] != 'COMPLETE' or public['converged'] is not True:
        raise ValueError('DCA initialization is not converged: ' + public['status'])
    if not arrays_path.is_file() or _sha(arrays_path) != public['cache_sha256']:
        raise ValueError('DCA initialization cache hash mismatch')
    with np.load(arrays_path, allow_pickle=False) as arrays:
        e32 = arrays['E32'].copy()
        if (e32.dtype != np.float32 or e32.shape != (task['physical'].shape[1], k)
                or not np.isfinite(e32).all() or _array_sha(e32) != public['diagnostics']['E32_sha256']):
            raise ValueError('DCA saved FP32 initialization hash/axes mismatch')
    return e32, public


def execute_job(job, ctx):
    """Prefit registered scopes without loading the frozen foundation model."""
    config = job.get('config', {})
    cohort = job.get('cohort', config.get('cohort', job.get('dataset', config.get('dataset'))))
    loader = _get(ctx, 'load_task')
    if loader is None:
        raise ValueError('DCA prefit requires caller-injected ctx.load_task')
    task = loader(cohort)
    scope = job.get('scope', config.get('scope', job.get('stage', config.get('stage'))))
    if scope not in {'A_fit', 'A60'}:
        raise ValueError('DCA job requires registered A_fit/A60 scope')
    fit_end = int(task['split_ranges']['A_fit' if scope == 'A_fit' else 'A_stop'][1])
    k = int(job.get('k', job.get('K', config.get('k', config.get('K', job.get('latent_k', config.get('latent_k', 6 if cohort == 'jena' else 8)))))))
    pca = data_contract.initialize_pca(task, fit_end, k)
    return fit_dca_initialization(ctx, task, fit_end, k, pca)
