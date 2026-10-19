import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.b2_revin_20261018_v1.contracts import sealed_json


def test_resume_canonicalizes_only_exact_identity_without_changing_saved_bytes(tmp_path):
    from research.b2_dataset_search_20261019_v1.resume_wave003 import identity_scope
    path = tmp_path/'TRAIN_IDENTITY.json'
    proposed = dict(split_ranges=dict(A_fit=(0, 12), A_stop=(12, 16)), epochs=[.25, 1, 3])
    sealed_json(path, proposed)
    before = path.read_bytes()
    with pytest.raises(ValueError, match='Sealed artifact changed'):
        sealed_json(path, proposed)
    module = SimpleNamespace(sealed_json=sealed_json)
    with identity_scope(module, path):
        assert module.sealed_json(path, proposed) == json.loads(before)
        with pytest.raises(ValueError, match='Sealed artifact changed'):
            module.sealed_json(path, dict(proposed, epochs=[1]))
    assert module.sealed_json is sealed_json and path.read_bytes() == before


def test_resume_never_canonicalizes_other_paths_and_restores_after_error(tmp_path):
    from research.b2_dataset_search_20261019_v1.resume_wave003 import identity_scope
    path, other = tmp_path/'TRAIN_IDENTITY.json', tmp_path/'other/TRAIN_IDENTITY.json'
    value = dict(bounds=(0, 10))
    for target in (path, other):
        sealed_json(target, value)
    module = SimpleNamespace(sealed_json=sealed_json)
    with pytest.raises(RuntimeError):
        with identity_scope(module, path):
            with pytest.raises(ValueError, match='Sealed artifact changed'):
                module.sealed_json(other, value)
            raise RuntimeError('interrupted')
    assert module.sealed_json is sealed_json


@pytest.mark.parametrize('field,value', [('split_ranges', dict(A_fit=(0, 13))),
                                       ('sources', dict(model='changed')), ('epochs', [0, 1])])
def test_resume_keeps_scientific_contract_mismatch_fatal(tmp_path, field, value):
    from research.b2_dataset_search_20261019_v1.resume_wave003 import identity_scope
    path = tmp_path/'TRAIN_IDENTITY.json'
    proposed = dict(split_ranges=dict(A_fit=(0, 12)), sources=dict(model='fixed'), epochs=[.25, 1, 3])
    sealed_json(path, proposed)
    before = path.read_bytes()
    module = SimpleNamespace(sealed_json=sealed_json)
    with identity_scope(module, path), pytest.raises(ValueError, match='Sealed artifact changed'):
        module.sealed_json(path, dict(proposed, **{field:value}))
    assert path.read_bytes() == before


def test_scope_requires_existing_sealed_identity(tmp_path):
    from research.b2_dataset_search_20261019_v1.resume_wave003 import identity_scope
    module = SimpleNamespace(sealed_json=sealed_json)
    with pytest.raises(ValueError, match='existing'):
        with identity_scope(module, tmp_path/'TRAIN_IDENTITY.json'):
            pytest.fail('Missing identity must fail before execution')


@pytest.fixture
def authorization_repo(tmp_path):
    from research.b2_dataset_search_20261019_v1 import resume_wave003 as resume
    from research.b2_revin_20261018_v1.contracts import atomic_json, sha
    report, cache = tmp_path/'reports'/resume.NAME, tmp_path/'.cache'/resume.NAME
    module = tmp_path/'research'/resume.NAME
    module.mkdir(parents=True); cache.mkdir(parents=True); (report/'waves').mkdir(parents=True)
    (module/'resume_wave003.py').write_text('adapter fixture')
    (module/'run.py').write_text('coordinator fixture')
    (module/'train.py').write_text('frozen training fixture')
    atomic_json(report/'waves/WAVE_003.json', dict(wave=3))
    atomic_json(cache/'TERMINAL_003_ATTEMPT_002.json', dict(status='FAILED', exit_code=1, finished_unix=10.))
    atomic_json(report/'EXECUTION_RECOVERY_WAVE_003_ATTEMPT_002.json', dict(attempt=2))
    atomic_json(report/'SOURCE_MANIFEST_TRAIN_003.json', dict(
        sources={'research/'+resume.NAME+'/train.py':sha(module/'train.py')},
        coordinator_sha256=sha(module/'run.py')))
    path = report/'EXECUTION_RECOVERY_WAVE_003_ATTEMPT_003.json'
    value = dict(status='REGISTERED_BEFORE_RESUMED_PRODUCER', wave=3, attempt=3, registered_unix=11.,
        previous_terminal=dict(reference='cache://TERMINAL_003_ATTEMPT_002.json', sha256=sha(cache/'TERMINAL_003_ATTEMPT_002.json')),
        preceding_recovery=dict(reference='report://EXECUTION_RECOVERY_WAVE_003_ATTEMPT_002.json', sha256=sha(report/'EXECUTION_RECOVERY_WAVE_003_ATTEMPT_002.json')),
        execution_adapter=dict(reference='repo://research/'+resume.NAME+'/resume_wave003.py', sha256=sha(module/'resume_wave003.py')),
        identity_serialization_repair='JSON_ROUNDTRIP_ONLY_EXACT_TRAIN_IDENTITY', unchanged_producer_sources=True,
        next_terminal_reference='cache://TERMINAL_003_ATTEMPT_003.json', producer_sha256=sha(module/'run.py'),
        wave_sha256=sha(report/'waves/WAVE_003.json'))
    atomic_json(path, value)
    return tmp_path, path, value


def test_resume_authorization_requires_matching_adapter_and_frozen_sources(authorization_repo):
    from research.b2_dataset_search_20261019_v1 import resume_wave003 as resume
    repo, _, value = authorization_repo
    assert resume.authorization(repo) == value
    (repo/'research'/resume.NAME/'train.py').write_text('changed algorithm')
    with pytest.raises(ValueError, match='Frozen producer'):
        resume.authorization(repo)


@pytest.mark.parametrize('field,value', [('wave', 2), ('attempt', 2), ('unchanged_producer_sources', False),
    ('registered_unix', 9.), ('execution_adapter', {}), ('previous_terminal', {}), ('preceding_recovery', {})])
def test_unregistered_or_tampered_resume_cannot_run(authorization_repo, field, value):
    from research.b2_dataset_search_20261019_v1 import resume_wave003 as resume
    from research.b2_revin_20261018_v1.contracts import atomic_json
    repo, path, amendment = authorization_repo
    atomic_json(path, dict(amendment, **{field:value}))
    with pytest.raises(ValueError, match='authorization'):
        resume.authorization(repo)
