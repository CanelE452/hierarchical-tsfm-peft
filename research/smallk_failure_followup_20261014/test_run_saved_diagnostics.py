"""Same-MAIN native lineage is checked before interpreting saved scores."""
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_condition_probes as probes
import run_saved_diagnostics as wrapper


def fixture(tmp_path, monkeypatch, change=None):
    case = tmp_path / 'S1_ANCHOR_NONLINEAR_D_MAIN_w0_s92601'
    case.mkdir()
    checkpoint = case / 'u100.pt'
    models = tmp_path / 'models.pt'
    native = tmp_path / 'NATIVE.npz'
    native.write_bytes(b'synthetic-prediction-placeholder-not-opened')
    state = {'weight': torch.tensor([[1., 2.]], dtype=torch.float32)}
    torch.save({'codec': state}, checkpoint)
    torch.save({'codec': {'weight': state['weight'] + (1 if change == 'state' else 0)}}, models)
    record = lambda path: {'path': str(path), 'sha256': probes.sha(path)}
    control = {'dataset': 'robin', 'seed': 92601, 'architecture': 'ANCHOR_NONLINEAR_D',
               'source_case': case.name, 'source_checkpoint': record(checkpoint),
               'selected_models': record(models),
               'methods': [{'method': 'NATIVE', 'prediction': record(native)}]}
    if change == 'checkpoint':
        control['source_checkpoint']['sha256'] = '0' * 64
    if change == 'prediction':
        control['methods'][0]['prediction']['sha256'] = '0' * 64
    source = tmp_path / 'CONTROL.json'
    source.write_text(json.dumps(control), encoding='utf-8')
    monkeypatch.setitem(wrapper.NATIVE_CONTROL_JSONS, ('robin', 92601), (str(source), probes.sha(source)))
    bridge = {'declared_asset_hashes': {'checkpoints_file_hash_only': {'robin_92601': record(checkpoint)},
              'predictions': {'robin': {'native_92601': record(native)}}}}
    return bridge


def test_exact_cpu_state_and_json_chain(tmp_path, monkeypatch):
    bridge = fixture(tmp_path, monkeypatch)
    result = wrapper.verify_native_lineage(probes, bridge, 'robin', 92601)
    assert result['same_E_D_state_torch_equal'] is True
    assert result['control_json']['sha256'] == probes.sha(result['control_json']['path'])


@pytest.mark.parametrize('change,message', [
    ('checkpoint', 'SOURCE_CHECKPOINT_LINEAGE'),
    ('prediction', 'PREDICTION_LINEAGE'),
    ('state', 'CODEC_STATE_NOT_EXACT_MAIN'),
])
def test_mismatched_native_reference_is_rejected(tmp_path, monkeypatch, change, message):
    bridge = fixture(tmp_path, monkeypatch, change)
    with pytest.raises(ValueError, match=message):
        wrapper.verify_native_lineage(probes, bridge, 'robin', 92601)
