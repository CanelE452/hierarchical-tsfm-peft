"""Source AST checks prepared only; require root GO before executing tests."""
import ast
from pathlib import Path

HERE=Path(__file__).resolve().parent


def tree(name):return ast.parse((HERE/name).read_text(encoding='utf-8'))


def functions(parsed):return {node.name:node for node in parsed.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}


def exact(node):return ast.dump(node,include_attributes=False)


def test_APP_runtime_derivative_preserves_validation_and_entire_timed_loop():
    previous=functions(tree('profile_appliances_cost_v1.py'));current=functions(tree('profile_appliances_cost_v2.py'))
    for name in ('validate_producers','audit_binding'):assert exact(previous[name])==exact(current[name])
    def timed(parsed):return [n for n in ast.walk(parsed['execute']) if isinstance(n,ast.With)]
    assert [exact(n) for n in timed(previous)]==[exact(n) for n in timed(current)]
    source=(HERE/'profile_appliances_cost_v2.py').read_text(encoding='utf-8')
    assert 'runtime.ContinuationLimit(' in source and 'runtime.FollowupLimit(' not in source
    assert 'LOCAL_SECONDS=300' in source


def test_report_runtime_derivative_preserves_all_scientific_tables_plots_text():
    previous=functions(tree('make_joint_followup_report_v2.py'));current=functions(tree('make_joint_followup_report_v3.py'))
    for name,node in previous.items():
        if name!='generate':assert exact(node)==exact(current[name])
    source=(HERE/'make_joint_followup_report_v3.py').read_text(encoding='utf-8')
    assert 'runtime.ContinuationLimit(' in source and 'runtime.FollowupLimit(' not in source
    assert 'thermal·clock·route-order' in source
