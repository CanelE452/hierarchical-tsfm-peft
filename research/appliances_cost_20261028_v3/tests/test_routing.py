"""Synthetic v3 routing checks only; no real data, GPU or environment changes."""
import ast
import hashlib
from pathlib import Path
import sys
import pytest
from research.appliances_cost_20261028_v3 import cost_profile as routed
from research.appliances_cost_20261028_v3 import verify_cost_results as verified
from research.appliances_cost_20261028_v2 import cost_profile as frozen_cost
from research.appliances_cost_20261028_v2 import verify_cost_results as frozen_verify
from research.ppt_justification_20261028_v1 import runtime as original_runtime

PINS = {
    'runtime.py': '2264dc68c61f26ae4d45832fef676ce801e78a591925c5c337519c68c22e138b',
    'cost_profile.py': '54edcebed072ca02918eb19f15ae12f36c17b38e0476c214d3b13f53e4bf7e97',
    'verify_cost_results.py': '70b2fb4cd6d80f8f96761bc27386d4bcb4d987661c4a49d2b4a60a4cd0a0fd5c',
    'tests/test_isolation.py': '979dc2efbdfe1d8cb5b911bdebbc560009099b9aa5b0eca14e09e64c27f37e3c',
}

@pytest.mark.parametrize('relative,digest', PINS.items())
def test_frozen_v2_bytes(relative, digest):
    path = routed.core.runtime.REPO / 'research/appliances_cost_20261028_v2' / relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_measurement_and_validator_function_identity():
    assert routed.core is frozen_cost and verified.core is frozen_verify
    assert routed.core.measure is frozen_cost.measure
    assert routed.core.make_inputs is frozen_cost.make_inputs
    assert routed.core.StateCheckedRoute is frozen_cost.StateCheckedRoute
    assert verified.core.validate_measurement is frozen_verify.validate_measurement
    assert verified.core.validate_sampling is frozen_verify.validate_sampling


def test_routes_and_shared_locks_are_separated():
    rt = routed.core.runtime
    assert rt.NAME == verified.core.NAME == 'appliances_cost_20261028_v3'
    assert routed.core.COST_REPORT == verified.REPORT == verified.core.REPORT == routed.REPORT/'cost_profile'
    assert routed.core.COST_CACHE == routed.CACHE/'cost_profile'
    assert rt.REPORT == original_runtime.REPORT and rt.CACHE == original_runtime.CACHE
    assert rt.load_task is original_runtime.load_task and rt.CampaignLock is original_runtime.CampaignLock
    assert rt.output_path(routed.REPORT/'test.json') == (routed.REPORT/'test.json').resolve()
    assert rt.output_path(routed.CACHE/'test.json') == (routed.CACHE/'test.json').resolve()
    for name in ('ppt_justification_20261028_v1', 'appliances_cost_20261028_v2'):
        for directory in ('reports', '.cache'):
            with pytest.raises(ValueError):
                rt.output_path(rt.REPO/directory/name/'NEVER_WRITTEN.json')


@pytest.mark.parametrize('name,expected', [('appliances_cost_20261028_v3', True), ('appliances_cost_20261028_v2', False),
                                         ('ppt_justification_20261028_v1', False), ('arbitrary', False)])
def test_exact_current_command(name, expected):
    command = ['python', '-B', '-m', f'research.{name}.cost_profile', '--run']
    assert routed.core.cost_command({'command': command}, '--run') is expected


def test_only_expected_host_module_added():
    path = original_runtime.REPO/'research/ppt_justification_20261028_v1/host_gpu_gate.py'
    tree = ast.parse(path.read_text(encoding='utf8'))
    literal = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == 'MODULES' for t in n.targets))
    expected = ast.literal_eval(literal) | {'research.appliances_cost_20261028_v2.cost_profile',
                                         'research.appliances_cost_20261028_v3.cost_profile'}
    assert routed.core.host_gpu_gate.MODULES == expected


@pytest.mark.parametrize('module', ['research.ppt_justification_20261028_v1.run_one',
                                  'research.crossdata_component_screen_20261028_v1.run_one', 'arbitrary.compute'])
def test_arbitrary_compute_still_rejected(module):
    policy = {'device': {'gpu_uuid': 'GPU-synthetic'}, 'desktop_processes': []}
    records = {20: {'pid': 20, 'created': 100., 'command': ['python','-m',module]}}
    with pytest.raises(RuntimeError):
        routed.core.host_gpu_gate.validate_identity(policy, records, [], {20: 100.})


def test_new_cost_parent_can_receive_exact_lease():
    policy = {'device': {'gpu_uuid': 'GPU-synthetic'}, 'desktop_processes': []}
    records = {20: {'pid': 20, 'created': 100., 'command': ['python','-m','research.appliances_cost_20261028_v3.cost_profile','--run']}}
    value = routed.core.host_gpu_gate.validate_identity(policy, records, [], {20: 100.})
    assert value['allowed_processes'] == {'20': 100.}


def test_validator_passes_explicit_new_path(monkeypatch):
    called = []
    def fake(path):
        called.append(path)
        return {'status': 'PASS', 'facets': [{}, {}, {}]}
    monkeypatch.setattr(verified.core, 'validate', fake)
    monkeypatch.setattr(sys, 'argv', ['validator', '--check-only'])
    verified.main()
    assert called == [verified.REPORT]


def test_validation_output_never_overwrites(monkeypatch, tmp_path):
    original = b'preserved receipt'
    destination = tmp_path/'COST_VALIDATION.json'
    destination.write_bytes(original)
    monkeypatch.setattr(verified, 'REPORT', tmp_path)
    monkeypatch.setattr(verified.core, 'validate', lambda path: {'status': 'PASS', 'facets': [{}, {}, {}]})
    monkeypatch.setattr(sys, 'argv', ['validator'])
    with pytest.raises(FileExistsError):
        verified.main()
    assert destination.read_bytes() == original
