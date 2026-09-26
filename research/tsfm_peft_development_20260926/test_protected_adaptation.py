import json

import pytest

from protected_protocol import current_protected_hashes, prepare_protected_adaptation


def fixture_contract(tmp_path):
    here = tmp_path / 'research'
    here.mkdir()
    (tmp_path / 'synthetic.npz').write_bytes(b'synthetic bytes; never parsed as an array')
    contract = {'datasets': {'bdg2_bull_office': {'npz': 'synthetic.npz'}}}
    (here / 'data_contract.json').write_text(json.dumps(contract), encoding='utf-8')
    for name in ('protocol.json', 'model.py', 'run.py'):
        (here / name).write_text('synthetic content', encoding='utf-8')
    spec = dict(id='bull_synthetic', dataset='bull', arm='residual', seed=1,
                lr=0.001, latent=4, epochs=120, residual_rank=32, loss='mse')
    seal = dict(schema_version=1, status='sealed', method_fixed=True,
                hashes=current_protected_hashes(here, tmp_path), allowed_fits=[spec])
    (here / 'final_seal.json').write_text(json.dumps(seal), encoding='utf-8')
    return here, spec, seal


def test_electricity_preflight_needs_no_protected_artifacts(tmp_path):
    assert prepare_protected_adaptation([{'dataset': 'electricity'}], tmp_path, tmp_path) == {}
    assert not list(tmp_path.iterdir())


def test_receipt_pinned_and_reused_without_parsing_npz(tmp_path):
    here, spec, _ = fixture_contract(tmp_path)
    first = prepare_protected_adaptation([spec], here, tmp_path)
    pin = here / 'protected_adaptation_receipt.json'
    original = pin.read_bytes()
    assert json.loads(original)["seal_sha256"] == first[spec['id']]['seal_sha256']
    assert prepare_protected_adaptation([spec], here, tmp_path) == first
    assert pin.read_bytes() == original


def test_all_requested_specs_validated_before_pin(tmp_path):
    here, spec, _ = fixture_contract(tmp_path)
    with pytest.raises(ValueError, match='not allowed'):
        prepare_protected_adaptation([spec, dict(spec, id='unsealed')], here, tmp_path)
    assert not (here / 'protected_adaptation_receipt.json').exists()


def test_artifact_mutation_rejected_before_pin(tmp_path):
    here, spec, _ = fixture_contract(tmp_path)
    (tmp_path / 'synthetic.npz').write_bytes(b'changed synthetic bytes')
    with pytest.raises(ValueError, match='artifact hash mismatch'):
        prepare_protected_adaptation([spec], here, tmp_path)
    assert not (here / 'protected_adaptation_receipt.json').exists()


def test_later_seal_replacement_rejected_and_original_pin_preserved(tmp_path):
    here, spec, seal = fixture_contract(tmp_path)
    prepare_protected_adaptation([spec], here, tmp_path)
    pin = here / 'protected_adaptation_receipt.json'
    original = pin.read_bytes()
    seal['new_note'] = 'a later replacement, even with identical fits'
    (here / 'final_seal.json').write_text(json.dumps(seal), encoding='utf-8')
    with pytest.raises(ValueError, match='originally pinned'):
        prepare_protected_adaptation([spec], here, tmp_path)
    assert pin.read_bytes() == original


def test_training_entry_rejects_missing_seal_before_data_and_gpu(tmp_path, monkeypatch):
    import run
    here, spec, _ = fixture_contract(tmp_path)
    (here / 'final_seal.json').unlink()
    specs_path = here / 'specs.json'
    specs_path.write_text(json.dumps([spec]), encoding='utf-8')
    monkeypatch.setattr(run, 'HERE', here)
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run.sys, 'argv', ['run.py', 'fits', '--specs', str(specs_path)])
    monkeypatch.setattr(run, 'load_data', lambda *args: pytest.fail('unsealed data loading reached'))
    monkeypatch.setattr(run, 'GPUJob', lambda *args: pytest.fail('unsealed GPU job reached'))
    with pytest.raises(ValueError, match='cannot read final seal'):
        run.main()


def test_training_entry_passes_original_receipt_to_fit(tmp_path, monkeypatch):
    import run
    here, spec, _ = fixture_contract(tmp_path)
    specs_path = here / 'specs.json'
    specs_path.write_text(json.dumps([spec]), encoding='utf-8')
    monkeypatch.setattr(run, 'HERE', here)
    monkeypatch.setattr(run, 'ROOT', tmp_path)
    monkeypatch.setattr(run.sys, 'argv', ['run.py', 'fits', '--specs', str(specs_path)])
    events = []
    def load(dataset):
        assert dataset == 'bull'
        assert (here / 'protected_adaptation_receipt.json').exists()
        events.append('data')
        return {'synthetic': True}
    class FakeJob:
        def __init__(self, label):
            assert label == 'fits'
        def __enter__(self):
            events.append('gpu')
            return self
        def __exit__(self, *args):
            return False
    def fit(data, requested, job, protected_receipt=None):
        assert data == {'synthetic': True} and requested == spec
        assert protected_receipt == json.loads((here / 'protected_adaptation_receipt.json').read_text(encoding='utf-8'))
        events.append('fit')
    monkeypatch.setattr(run, 'load_data', load)
    monkeypatch.setattr(run, 'GPUJob', FakeJob)
    monkeypatch.setattr(run, 'fit', fit)
    run.main()
    assert events == ['data', 'gpu', 'fit']
