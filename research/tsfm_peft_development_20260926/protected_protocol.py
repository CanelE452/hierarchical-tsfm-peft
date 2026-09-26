"""Validate the final Bull adaptation contract without opening protected data.

Only create the final seal after fixing the method on Electricity. Schema v1 is::

    {
        "schema_version": 1,
        "status": "sealed",
        "method_fixed": true,
        "hashes": {
            "data_contract.json": "<sha256>",
            "protected_npz": "<sha256>",
            "protocol.json": "<sha256>",
            "model.py": "<sha256>",
            "run.py": "<sha256>"
        },
        "allowed_fits": [
            {"id": "bull_residual_1", "dataset": "bull", "arm": "residual",
             "seed": 1, "lr": 0.001, "latent": 8, "epochs": 120,
             "residual_rank": 32, "loss": "mse"}
        ]
    }

The five hashes must be recomputed by the caller from the current artifacts,
not copied from the seal. Each fit must match its complete sealed JSON spec;
omitted optional parameters retain defaults pinned by the source/protocol hashes.
Persist the returned receipt before starting work and require the same seal SHA
for subsequent fits and evaluation. This module does not make a file immutable
or authenticate its author; method_fixed records an assertion, not proof of when
selection ended. It neither creates a seal nor reads datasets/checkpoints.

Call the fit guard before load_data/GPUJob. For evaluation, pass exactly the
allowed IDs whose result/checkpoint completion the caller has verified. The
evaluation guard does not verify results or enforce a one-time evaluation itself.
"""

import hashlib
import json
import math
from pathlib import Path
import re


HASH_KEYS = ('data_contract.json', 'protected_npz', 'protocol.json', 'model.py', 'run.py')
FIT_KEYS = {'id', 'dataset', 'arm', 'seed', 'lr', 'latent', 'epochs'}
ARMS = {'lora', 'compress', 'residual', 'raw_bypass'}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key in seal: {key}')
        result[key] = value
    return result


def _canonical_spec(spec):
    try:
        return json.dumps(spec, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError('fit spec must contain finite JSON values') from exc


def _validate_spec(spec):
    if not isinstance(spec, dict):
        raise ValueError('each allowed fit must be a JSON object')
    missing = FIT_KEYS - spec.keys()
    if missing:
        raise ValueError(f'fit spec missing explicit fields: {sorted(missing)}')
    if not isinstance(spec['id'], str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', spec['id']):
        raise ValueError('fit id must be a nonempty filename-safe identifier')
    if spec['dataset'] != 'bull':
        raise ValueError('protected seal only permits bull fits')
    if not isinstance(spec['arm'], str) or spec['arm'] not in ARMS:
        raise ValueError('unknown protected fit arm')
    for key in ('seed', 'latent', 'epochs'):
        minimum = 0 if key == 'seed' else 1
        if type(spec[key]) is not int or spec[key] < minimum:
            raise ValueError(f'fit {key} must be an integer >= {minimum}')
    lr = spec['lr']
    if type(lr) not in (int, float) or not math.isfinite(lr) or lr <= 0:
        raise ValueError('fit lr must be finite and positive')
    return _canonical_spec(spec)


def _validate_hashes(hashes, label):
    if not isinstance(hashes, dict) or set(hashes) != set(HASH_KEYS):
        raise ValueError(f'{label} hashes must contain exactly {HASH_KEYS}')
    for key, value in hashes.items():
        if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
            raise ValueError(f'{label} hash is not lowercase SHA256: {key}')


def _read_seal(seal_path, expected_hashes):
    try:
        raw = Path(seal_path).read_bytes()
    except OSError as exc:
        raise ValueError(f'cannot read final seal: {seal_path}') from exc
    seal = json.loads(raw.decode('utf-8'), object_pairs_hook=_unique_object)
    if not isinstance(seal, dict):
        raise ValueError('final seal must be a JSON object')
    if type(seal.get('schema_version')) is not int or seal['schema_version'] != 1:
        raise ValueError('unsupported final seal schema_version')
    if seal.get('status') != 'sealed' or seal.get('method_fixed') is not True:
        raise ValueError('final seal requires status sealed and method_fixed true')
    _validate_hashes(expected_hashes, 'current')
    _validate_hashes(seal.get('hashes'), 'sealed')
    for key in HASH_KEYS:
        if seal['hashes'][key] != expected_hashes[key]:
            raise ValueError(f'protected artifact hash mismatch: {key}')
    fits = seal.get('allowed_fits')
    if not isinstance(fits, list) or not fits:
        raise ValueError('seal must contain a nonempty allowed_fits list')
    indexed = {}
    for spec in fits:
        canonical = _validate_spec(spec)
        if spec['id'] in indexed:
            raise ValueError(f'duplicate allowed fit id: {spec["id"]}')
        indexed[spec['id']] = canonical
    receipt = {
        'schema_version': 1,
        'seal_sha256': hashlib.sha256(raw).hexdigest(),
        'allowed_fit_ids': list(indexed),
        'hashes': dict(seal['hashes']),
    }
    return indexed, receipt


def validate_protected_fit(spec, seal_path, expected_hashes):
    """Return the common seal receipt, or raise ValueError before a Bull fit."""
    allowed, receipt = _read_seal(seal_path, expected_hashes)
    canonical = _validate_spec(spec)
    if spec['id'] not in allowed:
        raise ValueError(f'fit id is not allowed by final seal: {spec["id"]}')
    if canonical != allowed[spec['id']]:
        raise ValueError(f'fit spec differs from final seal: {spec["id"]}')
    return receipt


def validate_protected_evaluation(seal_path, completed_fit_ids, expected_hashes):
    """Require the whole sealed cohort; completion evidence remains caller-owned."""
    allowed, receipt = _read_seal(seal_path, expected_hashes)
    if isinstance(completed_fit_ids, str):
        raise ValueError('completed fit IDs must be a collection of identifiers')
    try:
        completed = list(completed_fit_ids)
    except TypeError as exc:
        raise ValueError('completed fit IDs must be a collection of identifiers') from exc
    if not all(isinstance(fit_id, str) for fit_id in completed):
        raise ValueError('completed fit IDs must be strings')
    if len(set(completed)) != len(completed):
        raise ValueError('duplicate completed fit id')
    missing = set(allowed) - set(completed)
    unexpected = set(completed) - set(allowed)
    if missing or unexpected:
        raise ValueError(f'protected cohort incomplete: missing={sorted(missing)}, unexpected={sorted(unexpected)}')
    return receipt
