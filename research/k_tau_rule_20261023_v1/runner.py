"""Thin, import-inert adapter for the fixed K-tau B2 campaign.

The caller owns runtime construction, the run plan, budget and a single retry.
Only seed registration and per-job selection validation are scoped here; the
existing search, fresh refit and saved-array FP64 scorer execute unchanged.
Run in an isolated worker process because the core module adapters are scoped
module attributes, not thread-local state.
"""
from contextlib import contextmanager
import math
from pathlib import Path
import re
import time

from research.b2_revin_20261018_v1 import contracts


NAME = 'k_tau_rule_20261023_v1'
EPOCHS = (.25, 1., 3.)
SEEDS = tuple(range(92601, 92611))
COHORTS = ('jena', 'ETTh1', 'ETTh2_safe17400', 'appliances')


class IncompleteRun(RuntimeError):
    """Core checkpoint is preserved; only the controller may schedule a retry."""

    def __init__(self, phase, result):
        self.phase, self.result = phase, result
        super().__init__(f'{phase}: {result.get("status", "INCOMPLETE")}; '
                         'checkpoints preserved; resume the same job_id')


def _validate(ctx, job):
    repo = Path(ctx['repo']).resolve()
    for key, parent in (('cache', '.cache'), ('report', 'reports')):
        if not Path(ctx[key]).resolve().is_relative_to(repo / parent / NAME):
            raise ValueError('Outputs must remain in the K-tau campaign namespace')
    if (job.get('family') != 'B2' or job.get('revin') is not False
            or job.get('init_kind') != 'PCA' or job.get('device') != 'cuda'):
        raise ValueError('Fixed B2/PCA/RevIN-off/cuda scientific recipe required')
    if type(job.get('seed')) is not int or job['seed'] not in SEEDS:
        raise ValueError('Approved seeds are exactly 92601 through 92610')
    task = ctx['task']
    if job.get('cohort') not in COHORTS or job['cohort'] != task['dataset']:
        raise ValueError('Job cohort differs from the approved task')
    if (type(job.get('K')) is not int or not 1 <= job['K'] <= len(task['columns'])
            or (task['L'], task['H']) != (512, 48)):
        raise ValueError('Invalid registered K or L512/H48 task')
    if not re.fullmatch(r'[A-Za-z0-9_-]+', str(job.get('job_id', ''))):
        raise ValueError('Safe immutable job_id required')
    if not ctx.get('protocol_sha256') or not ctx['foundation'].initial_hash:
        raise ValueError('Preregistration and frozen foundation SHA bindings required')


@contextmanager
def _registration_scope(helper):
    original = helper._spec

    def registered(job, ctx):
        _validate(ctx, job)
        # The original registration creates no model. Preserve every scientific
        # field it emits and replace only the approved sampler/model seed.
        spec = dict(original(dict(job, seed=92601), ctx))
        spec['seed'] = job['seed']
        return spec

    helper._spec = registered
    try:
        yield
    finally:
        helper._spec = original


def _verify_search(job, result):
    if result.get('status') != 'COMPLETE':
        raise IncompleteRun('search', result)
    expected = dict(dataset=job['cohort'], family='B2', revin=False,
                    seed=job['seed'], latent_k=job['K'], epochs=list(EPOCHS))
    if (result.get('cap') != 3 or result.get('V_used') is not False
            or result.get('selection_split') != 'A_stop'
            or any(result.get('spec', {}).get(k) != v for k, v in expected.items())):
        raise ValueError('Search scientific recipe or information boundary changed')
    curve = [(float(row['epoch']), float(row['astop_mse'])) for row in result['curves']
             if row.get('astop_mse') is not None]
    if ([epoch for epoch, _ in curve] != [0., *EPOCHS]
            or not all(math.isfinite(mse) for _, mse in curve)):
        raise ValueError('All four fixed A_stop checkpoints must be finite')
    epoch, value = min(curve, key=lambda pair: (pair[1], pair[0]))
    if (float(result['selected_epoch']) != epoch
            or float(result['summary']['selected_epoch']) != epoch
            or float(result['summary']['selected_astop_mse']) != value):
        raise ValueError('Selection must be minimum A_stop with the earlier exact tie')
    return epoch, value


def _search_path(ctx, job):
    return Path(ctx['cache']) / 'runs' / job['job_id'] / 'cap3' / 'SEARCH_RESULT.json'


def _selection_payload(ctx, job, search, identity_path):
    epoch, value = _verify_search(job, search)
    stored = _search_path(ctx, job)
    if contracts.read(stored) != search:
        raise ValueError('Exact completed search must be stored before sealing')
    contracts.resolve_binding(search['checkpoint_binding'], ctx)
    if (search['contract']['data_sha'] != ctx['task']['data_sha256']
            or search['contract']['model_sha'] != ctx['foundation'].initial_hash):
        raise ValueError('Search data/F binding differs from the task/runtime')
    return dict(status='SEALED', cap=3, V_used=False, selection_split='A_stop',
        job_id=job['job_id'], cohort=job['cohort'], selected_epoch=epoch,
        selected_astop_mse=value, spec=search['spec'], contract=search['contract'],
        foundation_sha256=ctx['foundation'].initial_hash,
        data_sha256=ctx['task']['data_sha256'], protocol_sha256=ctx['protocol_sha256'],
        identity_binding=contracts.binding(identity_path, ctx),
        search_binding=contracts.binding(stored, ctx),
        checkpoint_binding=search['checkpoint_binding'])


@contextmanager
def _selection_scope(helper, ctx, job, search, seal_path, identity_path):
    original = helper._selection_decision
    expected = _selection_payload(ctx, job, search, identity_path)
    sealed_sha = contracts.sha(seal_path)

    def validate(active_job, active_ctx, cap):
        descriptor = active_ctx.get('selection_decision', {})
        if (active_job != job or active_ctx is not ctx or cap != 3
                or Path(descriptor.get('path', '')).resolve() != seal_path.resolve()
                or descriptor.get('sha256') != sealed_sha
                or contracts.sha(seal_path) != sealed_sha
                or contracts.read(seal_path) != expected):
            raise ValueError('Exact per-job selection seal binding required before V')
        for name in ('identity_binding', 'search_binding', 'checkpoint_binding'):
            contracts.resolve_binding(expected[name], ctx)
        if _selection_payload(ctx, job, search, identity_path) != expected:
            raise ValueError('Search/spec/data/F binding changed after selection seal')
        return contracts.binding(seal_path, ctx)

    helper._selection_decision = validate
    try:
        validate(job, ctx, 3)
        yield
    finally:
        helper._selection_decision = original


def _verify_result(result, ctx):
    if result.get('status') != 'COMPLETE':
        raise IncompleteRun('fresh_A60_refit', result)
    if len(result.get('rows', [])) != 1 or result.get('selection_sealed_before_V') is not True:
        raise ValueError('Exactly one sealed fresh-A60 V result required')
    row = result['rows'][0]
    if (row.get('independent_FP64') != 'PASS'
            or row.get('metric_accumulation_dtype') != 'float64'
            or not all(math.isfinite(float(row[key])) for key in ('MSE', 'MAE'))):
        raise ValueError('Existing saved-array independent FP64 verification required')
    contracts.resolve_binding(result['bundle_binding'], ctx)
    contracts.resolve_binding(row['prediction_binding'], ctx)
    contracts.resolve_binding(result['selection_decision_binding'], ctx)


def run_one(ctx, job, *, role='main'):
    """Execute one registered search -> seal -> fresh refit/V, without retries.

    ``ctx`` is the context returned by the existing extension runtime with the
    campaign roots and ``protocol_sha256`` (PREREGISTRATION.json) supplied by the
    controller. Optional ``helper`` permits CPU fixture testing. Completed jobs
    return the immutable cached public result. Failed attempts remain separate.
    """
    _validate(ctx, job)
    if 'helper' in ctx:
        helper = ctx['helper']
    else:
        from research.b2_revin_20261018_v1 import training_jobs as helper
    active = dict(ctx)
    directory = Path(active['report']) / 'jobs' / job['job_id']
    directory.mkdir(parents=True, exist_ok=True)
    identity_path = directory / 'IDENTITY.json'
    with _registration_scope(helper):
        spec = helper._spec(job, active)
        identity = dict(campaign=NAME, job=job, role=str(role), spec=spec,
            protocol_sha256=active['protocol_sha256'], source_hashes=active['scientific_source_hashes'],
            wrapper_sha256=contracts.sha(__file__), foundation_sha256=active['foundation'].initial_hash,
            data_sha256=active['task']['data_sha256'],
            split_ranges={key: list(value) for key, value in active['task']['split_ranges'].items()},
            roots={key: str(Path(active[key]).resolve()) for key in ('repo', 'cache', 'report', 'asset')},
            epochs=list(EPOCHS), cap=3, T_access=False)
        contracts.sealed_json(identity_path, identity)
        output = directory / 'RESULT.json'
        if output.exists():
            result = contracts.read(output)
            _verify_result(result, active)
            return result
        attempts = sorted(directory.glob('attempt_*'))
        if len(attempts) >= 2:
            raise RuntimeError('Two attempts already exist; controller must stop this job_id')
        attempt = directory / f'attempt_{len(attempts) + 1:03d}'
        attempt.mkdir()
        started, started_utc = time.time(), contracts.utc()
        contracts.sealed_json(attempt / 'START.json', dict(job_id=job['job_id'],
            role=str(role), started_utc=started_utc, identity_binding=contracts.binding(identity_path, active)))
        phase, status, error, search, result = 'search', 'FAILED', None, None, None
        selection, frozen_checked = None, False
        try:
            if time.time() >= active['deadline']:
                raise IncompleteRun('search', dict(status='BUDGET_EXHAUSTED'))
            search = helper.search(job, active, epochs=EPOCHS)
            contracts.sealed_json(attempt / 'SEARCH_RETURN.json', search)
            _verify_search(job, search)
            if search['spec'] != dict(spec, epochs=list(EPOCHS)):
                raise ValueError('Search spec differs from the immutable registered spec')
            seal_path = directory / 'SELECTION.json'
            contracts.sealed_json(seal_path, _selection_payload(active, job, search, identity_path))
            selection = contracts.binding(seal_path, active)
            active['selection_decision'] = dict(path=str(seal_path.resolve()), sha256=selection['sha256'])
            phase = 'fresh_A60_refit_and_V'
            with _selection_scope(helper, active, job, search, seal_path, identity_path):
                if time.time() >= active['deadline']:
                    raise IncompleteRun(phase, dict(status='BUDGET_EXHAUSTED'))
                result = helper._refit_predict(job, active, search, 3)
            contracts.sealed_json(attempt / 'REFIT_RETURN.json', result)
            _verify_result(result, active)
            if result['selection_decision_binding'] != selection:
                raise ValueError('Returned selection seal differs from the pre-refit binding')
            result = dict(result, rows=[dict(result['rows'][0], bundle_binding=result['bundle_binding'])])
            active['foundation'].assert_frozen_unchanged()
            frozen_checked = True
            contracts.sealed_json(output, result)
            status = 'COMPLETE'
            return result
        except IncompleteRun as exc:
            status, error = exc.result.get('status', 'INCOMPLETE'), str(exc)
            raise
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            receipt = dict(job_id=job['job_id'], role=str(role), status=status, phase=phase,
                started_utc=started_utc, ended_utc=contracts.utc(), elapsed_seconds=time.time() - started,
                error=error, identity_binding=contracts.binding(identity_path, active),
                selection_binding=selection, selection_sealed_before_refit_and_V=selection is not None,
                source_hashes=active['scientific_source_hashes'], protocol_sha256=active['protocol_sha256'],
                search_seconds=None if search is None else search.get('summary', {}).get('seconds'),
                refit_seconds=None if result is None else result.get('refit_summary', {}).get('seconds'),
                checkpoint_binding=None if search is None else search.get('checkpoint_binding'),
                automatic_retries=0, T_access=False)
            contracts.sealed_json(attempt / 'RECEIPT.json', receipt)
            if not frozen_checked:
                active['foundation'].assert_frozen_unchanged()
