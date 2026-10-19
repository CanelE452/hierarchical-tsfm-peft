"""Explicit, registered JSON-identity adapter for the interrupted ETTm1 wave."""
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path

from research.b2_revin_20261018_v1.contracts import read, sha


NAME = 'b2_dataset_search_20261019_v1'
REPO = Path(__file__).resolve().parents[2]


@contextmanager
def identity_scope(module, identity_path):
    """Preserve saved bytes while comparing the same serialized JSON contract."""
    identity_path = Path(identity_path).resolve()
    if not identity_path.is_file():
        raise ValueError('An existing sealed TRAIN_IDENTITY is required')
    original = module.sealed_json

    def serialize_identity(path, value):
        if Path(path).resolve() == identity_path:
            value = json.loads(json.dumps(value, allow_nan=False))
        return original(path, value)

    module.sealed_json = serialize_identity
    try:
        yield
    finally:
        module.sealed_json = original


def authorization(repo=REPO):
    repo = Path(repo).resolve()
    report, cache = repo/'reports'/NAME, repo/'.cache'/NAME
    amendment = read(report/'EXECUTION_RECOVERY_WAVE_003_ATTEMPT_003.json')
    preceding = report/'EXECUTION_RECOVERY_WAVE_003_ATTEMPT_002.json'
    prior = cache/'TERMINAL_003_ATTEMPT_002.json'
    failed = read(prior)
    expected_adapter = dict(reference='repo://research/'+NAME+'/resume_wave003.py',
                            sha256=sha(repo/'research'/NAME/'resume_wave003.py'))
    if (amendment.get('status') != 'REGISTERED_BEFORE_RESUMED_PRODUCER'
            or amendment.get('wave') != 3 or amendment.get('attempt') != 3
            or amendment.get('previous_terminal') != dict(reference='cache://'+prior.name, sha256=sha(prior))
            or amendment.get('preceding_recovery') != dict(reference='report://'+preceding.name, sha256=sha(preceding))
            or failed.get('status') != 'FAILED' or failed.get('exit_code') != 1
            or amendment.get('execution_adapter') != expected_adapter
            or amendment.get('identity_serialization_repair') != 'JSON_ROUNDTRIP_ONLY_EXACT_TRAIN_IDENTITY'
            or amendment.get('unchanged_producer_sources') is not True
            or amendment.get('next_terminal_reference') != 'cache://TERMINAL_003_ATTEMPT_003.json'
            or not isinstance(amendment.get('registered_unix'), (int, float))
            or not math.isfinite(amendment['registered_unix'])
            or amendment['registered_unix'] <= failed['finished_unix']):
        raise ValueError('Exact registered serialization-recovery authorization is required')
    manifest = read(report/'SOURCE_MANIFEST_TRAIN_003.json')
    for name, digest in manifest['sources'].items():
        if sha(repo/name) != digest:
            raise ValueError('Frozen producer source changed: '+name)
    if (sha(repo/'research'/NAME/'run.py') != manifest['coordinator_sha256']
            or amendment.get('producer_sha256') != manifest['coordinator_sha256']
            or amendment.get('wave_sha256') != sha(report/'waves/WAVE_003.json')):
        raise ValueError('Frozen coordinator or wave changed')
    return amendment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    authorization()
    from . import run, train
    identity = REPO/'reports'/NAME/'datasets/ETTm1/TRAIN_IDENTITY.json'
    with identity_scope(train, identity):
        run.main(['all', '--wave', '3', '--model-dir', str(args.model_dir)])


if __name__ == '__main__':
    main()
