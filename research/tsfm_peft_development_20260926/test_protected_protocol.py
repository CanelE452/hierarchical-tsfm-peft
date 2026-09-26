import copy
import hashlib
import json

import pytest

import protected_protocol


@pytest.fixture
def contract(tmp_path):
    hashes = {
        'data_contract.json': 'a' * 64,
        'protected_npz': 'b' * 64,
        'protocol.json': 'c' * 64,
        'model.py': 'd' * 64,
        'run.py': 'e' * 64,
    }
    fits = [
        dict(id='bull_residual_1', dataset='bull', arm='residual', seed=1,
             lr=0.001, latent=8, epochs=120, residual_rank=32, loss='mse'),
        dict(id='bull_raw_1', dataset='bull', arm='raw_bypass', seed=1,
             lr=0.001, latent=8, epochs=120, residual_rank=32, loss='mse'),
    ]
    seal = dict(schema_version=1, status='sealed', method_fixed=True,
                hashes=hashes, allowed_fits=fits)
    path = tmp_path / 'final_seal.json'
    path.write_text(json.dumps(seal), encoding='utf-8')
    return path, seal, copy.deepcopy(hashes)


def write_seal(path, seal):
    path.write_text(json.dumps(seal), encoding='utf-8')


def test_exact_fit_and_completed_evaluation_return_the_same_seal_receipt(contract):
    path, seal, hashes = contract
    original = path.read_bytes()
    fit_receipt = protected_protocol.validate_protected_fit(
        copy.deepcopy(seal['allowed_fits'][0]), path, hashes)
    evaluation_receipt = protected_protocol.validate_protected_evaluation(
        path, ['bull_raw_1', 'bull_residual_1'], hashes)
    assert fit_receipt == evaluation_receipt
    assert fit_receipt['seal_sha256'] == hashlib.sha256(original).hexdigest()
    assert fit_receipt['allowed_fit_ids'] == ['bull_residual_1', 'bull_raw_1']
    assert fit_receipt['hashes'] == hashes
    assert path.read_bytes() == original
    assert sorted(p.name for p in path.parent.iterdir()) == ['final_seal.json']


@pytest.mark.parametrize('field,value', [
    ('id', 'unlisted_fit'), ('dataset', 'electricity'), ('arm', 'compress'),
    ('seed', 2), ('lr', 0.0001), ('latent', 16), ('epochs', 121),
    ('residual_rank', 8), ('loss', 'native'), ('weight_decay', 0.01),
])
def test_fit_cannot_change_or_add_a_sealed_parameter(contract, field, value):
    path, seal, hashes = contract
    spec = dict(seal['allowed_fits'][0], **{field: value})
    with pytest.raises(ValueError):
        protected_protocol.validate_protected_fit(spec, path, hashes)


def test_fit_cannot_omit_a_sealed_optional_parameter(contract):
    path, seal, hashes = contract
    spec = dict(seal['allowed_fits'][0])
    del spec['residual_rank']
    with pytest.raises(ValueError):
        protected_protocol.validate_protected_fit(spec, path, hashes)


@pytest.mark.parametrize('name', [
    'data_contract.json', 'protected_npz', 'protocol.json', 'model.py', 'run.py',
])
def test_changed_current_hash_blocks_fit_and_evaluation(contract, name):
    path, seal, hashes = contract
    hashes[name] = 'f' * 64
    with pytest.raises(ValueError, match='hash'):
        protected_protocol.validate_protected_fit(seal['allowed_fits'][0], path, hashes)
    with pytest.raises(ValueError, match='hash'):
        protected_protocol.validate_protected_evaluation(
            path, ['bull_residual_1', 'bull_raw_1'], hashes)


@pytest.mark.parametrize('hashes_location', ['current', 'sealed'])
@pytest.mark.parametrize('invalid_hash', [None, 'bad-sha256', 'g' * 64])
def test_malformed_hash_is_rejected_even_if_equal(contract, hashes_location, invalid_hash):
    path, seal, hashes = contract
    hashes['run.py'] = invalid_hash
    if hashes_location == 'sealed':
        seal['hashes']['run.py'] = invalid_hash
        write_seal(path, seal)
    with pytest.raises(ValueError, match='hash'):
        protected_protocol.validate_protected_fit(seal['allowed_fits'][0], path, hashes)


@pytest.mark.parametrize('location', ['current', 'sealed'])
def test_missing_required_hash_is_rejected(contract, location):
    path, seal, hashes = contract
    if location == 'current':
        del hashes['model.py']
    else:
        del seal['hashes']['model.py']
        write_seal(path, seal)
    with pytest.raises(ValueError, match='hash'):
        protected_protocol.validate_protected_fit(seal['allowed_fits'][0], path, hashes)


def test_evaluation_rejects_missing_fit(contract):
    path, _, hashes = contract
    with pytest.raises(ValueError, match='missing'):
        protected_protocol.validate_protected_evaluation(path, ['bull_residual_1'], hashes)


@pytest.mark.parametrize('completed', [
    ['bull_residual_1', 'bull_raw_1', 'unlisted_fit'],
    ['bull_residual_1', 'bull_raw_1', 'bull_raw_1'],
])
def test_evaluation_rejects_unknown_or_duplicate_completion_ids(contract, completed):
    path, _, hashes = contract
    with pytest.raises(ValueError):
        protected_protocol.validate_protected_evaluation(path, completed, hashes)


def test_duplicate_fit_id_in_seal_is_rejected(contract):
    path, seal, hashes = contract
    seal['allowed_fits'][1]['id'] = 'bull_residual_1'
    write_seal(path, seal)
    with pytest.raises(ValueError, match='duplicate'):
        protected_protocol.validate_protected_fit(seal['allowed_fits'][0], path, hashes)


def test_non_bull_fit_anywhere_in_seal_is_rejected(contract):
    path, seal, hashes = contract
    seal['allowed_fits'][1]['dataset'] = 'electricity'
    write_seal(path, seal)
    with pytest.raises(ValueError, match='bull'):
        protected_protocol.validate_protected_fit(seal['allowed_fits'][0], path, hashes)


@pytest.mark.parametrize('field,value', [
    ('status', 'draft'), ('status', None), ('method_fixed', False),
    ('method_fixed', 1), ('schema_version', 2), ('schema_version', True),
    ('allowed_fits', []),
])
def test_unsealed_or_invalid_contract_is_rejected(contract, field, value):
    path, seal, hashes = contract
    spec = copy.deepcopy(seal['allowed_fits'][0])
    seal[field] = value
    write_seal(path, seal)
    with pytest.raises(ValueError):
        protected_protocol.validate_protected_fit(spec, path, hashes)


@pytest.mark.parametrize('field,value', [
    ('seed', True), ('lr', float('nan')), ('latent', 0), ('epochs', -1),
    ('arm', 'unknown'),
])
def test_invalid_sealed_fit_cannot_authorize_itself(contract, field, value):
    path, seal, hashes = contract
    seal['allowed_fits'][0][field] = value
    write_seal(path, seal)
    with pytest.raises(ValueError):
        protected_protocol.validate_protected_fit(seal['allowed_fits'][0], path, hashes)


def test_seal_requires_explicit_epoch_cap(contract):
    path, seal, hashes = contract
    del seal['allowed_fits'][0]['epochs']
    write_seal(path, seal)
    with pytest.raises(ValueError, match='epochs'):
        protected_protocol.validate_protected_fit(seal['allowed_fits'][0], path, hashes)


def test_missing_seal_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='seal'):
        protected_protocol.validate_protected_fit({}, tmp_path / 'absent.json', {})


def test_duplicate_json_key_is_rejected(contract):
    path, seal, hashes = contract
    text = json.dumps(seal).replace('"status": "sealed"', '"status": "draft", "status": "sealed"')
    path.write_text(text, encoding='utf-8')
    with pytest.raises(ValueError, match='duplicate'):
        protected_protocol.validate_protected_fit(seal['allowed_fits'][0], path, hashes)
