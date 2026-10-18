"""Finite B2/RevIN identities and hash-bound local artifacts; import is inert."""
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

NAME = 'b2_revin_20261018_v1'
SEEDS = (92601, 92602)
COHORTS = ('jena', 'pems32')
INITIAL_EPOCHS = (.25, 1., 3.)
EXTENDED_EPOCHS = (.25, 1., 3., 4., 5., 6.)
RECIPE = dict(lr=.001, effective_batch=16, microbatch=4,
              F_row_microbatch=128, checkpoint_inputs=False)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def sealed_json(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError('Sealed artifact changed: ' + path.name)
    else:
        atomic_json(path, value)
    return value


def binding(path, ctx):
    path = Path(path).resolve()
    for key in ('report', 'cache', 'repo', 'asset'):
        root = Path(ctx[key]).resolve()
        if path.is_relative_to(root):
            return dict(reference=key + '://' + path.relative_to(root).as_posix(), sha256=sha(path))
    raise ValueError('Artifact outside registered roots')


def resolve_binding(value, ctx):
    key, relative = value['reference'].split('://', 1)
    if key not in ('report', 'cache', 'repo', 'asset'):
        raise ValueError('Unregistered artifact root')
    root = Path(ctx[key]).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or sha(path) != value['sha256']:
        raise ValueError('Artifact path or SHA mismatch')
    return path


def job_id(cohort, enabled, seed, family='B2'):
    if cohort not in COHORTS or seed not in SEEDS or type(enabled) is not bool:
        raise ValueError('Unregistered cohort/normalization/seed')
    if family not in ('B2', 'B0') or (family == 'B0' and cohort != 'jena'):
        raise ValueError('Unregistered family')
    return f'{family}_{cohort}_REVIN_{"ON" if enabled else "OFF"}_s{seed}'


def _check_group(searches):
    if len(searches) != 4:
        raise ValueError('Whole OFF/ON x two-seed group required')
    specs = [result['spec'] for result in searches]
    if len({spec['dataset'] for spec in specs}) != 1:
        raise ValueError('Extension must be decided per dataset')
    if any(type(spec['revin']) is not bool for spec in specs):
        raise ValueError('Boolean normalization identity required')
    if {(spec['seed'], spec['revin']) for spec in specs} != {(seed, enabled) for seed in SEEDS for enabled in (False, True)}:
        raise ValueError('Both seeds of each normalization required')


def needs_extension(searches, cap):
    """A_stop-only decision; strict last best and falling from preceding point."""
    _check_group(searches)
    if cap not in (3, 6):
        raise ValueError('Registered cap required')
    continuing = []
    for result in searches:
        curves = result['curves']
        values = [(float(row['epoch']), float(row['astop_mse'])) for row in curves
                  if row.get('astop_mse') is not None and row['epoch'] <= cap]
        if not values or values[-1][0] != cap:
            raise ValueError('Group has not completed its registered cap')
        continuing.append(len(values) > 1 and values[-1][1] < min(value for _, value in values[:-1]))
    return any(continuing)


def common_normalization(searches):
    _check_group(searches)
    groups = {False: [], True: []}
    for result in searches:
        groups[bool(result['spec']['revin'])].append(min(float(row['astop_mse'])
            for row in result['curves'] if row.get('astop_mse') is not None))
    if any(len(values) != 2 for values in groups.values()):
        raise ValueError('Both seeds of each normalization required')
    means = {key: sum(values) / 2 for key, values in groups.items()}
    return means[True] < means[False], means


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
