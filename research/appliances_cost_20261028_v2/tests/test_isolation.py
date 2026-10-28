"""CPU-only isolation and unchanged-core checks; no real arrays or GPU."""
import ast
from pathlib import Path
import pytest
from research.appliances_cost_20261028_v2 import runtime, cost_profile as cost, verify_cost_results as verify
from research.ppt_justification_20261028_v1 import runtime as old_runtime

OLD = runtime.REPO / 'research/ppt_justification_20261028_v1'
NEW = runtime.REPO / 'research/appliances_cost_20261028_v2'

def nodes(path):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text(encoding='utf8')).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}

@pytest.mark.parametrize('name', ['StateCheckedRoute', 'measure', 'make_inputs', 'verify_reference'])
def test_measurement_core_unchanged(name):
    assert nodes(OLD/'cost_profile.py')[name] == nodes(NEW/'cost_profile.py')[name]

@pytest.mark.parametrize('name', ['validate_measurement', 'validate_sampling', 'table_values', 'validate_prefix', 'adapter_hash', 'validate_host_evidence', 'macro_metrics'])
def test_independent_validation_core_unchanged(name):
    assert nodes(OLD/'verify_cost_results.py')[name] == nodes(NEW/'verify_cost_results.py')[name]

def test_runtime_isolation_and_delegation():
    assert runtime.CACHE == old_runtime.CACHE and runtime.REPORT == old_runtime.REPORT
    assert runtime.load_modules is old_runtime.load_modules
    assert runtime.CampaignLock is old_runtime.CampaignLock
    assert cost.COST_REPORT == verify.REPORT == runtime.OUTPUT_REPORT/'cost_profile'
    assert cost.COST_CACHE == runtime.OUTPUT_CACHE/'cost_profile'
    for root in (old_runtime.REPORT, old_runtime.CACHE):
        with pytest.raises(ValueError):
            runtime.write_new(root/'NEVER_WRITTEN.json', {})
        with pytest.raises(ValueError):
            runtime.atomic(root/'NEVER_WRITTEN.json', {})

def test_exact_new_command_guard():
    good = {'command': ['python', '-B', '-m', 'research.appliances_cost_20261028_v2.cost_profile', '--run']}
    assert cost.cost_command(good, '--run')
    assert not cost.cost_command({'command': ['python','-m','research.ppt_justification_20261028_v1.cost_profile','--run']}, '--run')
    assert not cost.cost_command({'command': ['python','other.py','--run','research.appliances_cost_20261028_v2.cost_profile']}, '--run')

def test_all_source_files_exist_and_unique_trace_labels():
    assert all((runtime.REPO / name).is_file() for name in cost.SOURCE_FILES)
    source = (NEW/'cost_profile.py').read_text(encoding='utf8')
    assert "runtime.NAME+'_'+run_id+'_preflight'" in source
    assert "runtime.NAME+'_'+run_id+'_measurement'" in source


def test_host_extension_exactly_one_module():
    tree = ast.parse((OLD/'host_gpu_gate.py').read_text(encoding='utf8'))
    literal = next(node.value for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'MODULES' for target in node.targets))
    previous = ast.literal_eval(literal)
    assert cost.host_gpu_gate.MODULES == previous | {'research.appliances_cost_20261028_v2.cost_profile'}
    assert verify.host_gpu_gate.MODULES == cost.host_gpu_gate.MODULES


def test_new_parent_worker_identity_and_gpu_lease_admitted():
    policy = {'device': {'gpu_uuid': 'GPU-synthetic'}, 'desktop_processes': []}
    records = {20: {'pid': 20, 'created': 100., 'command': ['python', '-m', 'research.appliances_cost_20261028_v2.cost_profile', '--run']},
               21: {'pid': 21, 'created': 101., 'command': ['python', '-m', 'research.appliances_cost_20261028_v2.cost_profile', '--worker']}}
    snapshot = cost.host_gpu_gate.validate_identity(policy, records, [{'pid': 21, 'gpu_uuid': 'GPU-synthetic'}],
                                                     {20: 100., 21: 101.}, {21: 101.})
    assert snapshot['observed'] == [{'pid': 21, 'role': 'registered_worker', 'created': 101.}]


@pytest.mark.parametrize('module', ['research.ppt_justification_20261028_v1.controller',
                                  'research.ppt_justification_20261028_v1.run_one',
                                  'research.crossdata_component_screen_20261028_v1.run_one',
                                  'arbitrary.compute'])
def test_unrelated_compute_cannot_receive_host_exception(module):
    policy = {'device': {'gpu_uuid': 'GPU-synthetic'}, 'desktop_processes': []}
    records = {20: {'pid': 20, 'created': 100., 'command': ['python', '-m', module]}}
    with pytest.raises(RuntimeError):
        cost.host_gpu_gate.validate_identity(policy, records, [], {20: 100.})


def test_unknown_gpu_context_remains_rejected():
    policy = {'device': {'gpu_uuid': 'GPU-synthetic'}, 'desktop_processes': []}
    with pytest.raises(RuntimeError, match='Unknown GPU context'):
        cost.host_gpu_gate.validate_identity(policy, {}, [{'pid': 99, 'gpu_uuid': 'GPU-synthetic'}])


def test_new_cost_module_without_lease_remains_rejected():
    policy = {'device': {'gpu_uuid': 'GPU-synthetic'}, 'desktop_processes': []}
    records = {20: {'pid': 20, 'created': 100., 'command': ['python', '-m', 'research.appliances_cost_20261028_v2.cost_profile', '--worker']}}
    with pytest.raises(RuntimeError, match='Live experiment process'):
        cost.host_gpu_gate.validate_identity(policy, records, [])
