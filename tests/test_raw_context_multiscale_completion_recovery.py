"""The verifier recovery admits only the registered foundation-free parents."""
import ast
from pathlib import Path

import pytest

from research.raw_context_multiscale_completion_recovery_20261019_v1 import run


def contract(**changes):
    return dict(source_sha='a'*64, config_sha='b'*64, data_sha='c'*64,
                model_sha='NO_FOUNDATION', **changes)


def test_exact_registered_direct_parent_only():
    route = dict(family='DIRECT_G', run_id='g', artifact_sha256='d'*64)
    allowed = [route]
    assert run.identity_ok(route, contract(), allowed)
    for key, value in [('family','B2'), ('run_id','other'), ('artifact_sha256','e'*64)]:
        assert not run.identity_ok(dict(route, **{key:value}), contract(), allowed)
    for sentinel in ['NO_F', 'NO_FOUNDATION ', '', None]:
        bad = contract(); bad['model_sha'] = sentinel
        assert not run.identity_ok(route, bad, allowed)
    for key in ('source_sha','data_sha','config_sha'):
        bad = contract(); bad[key] = 'bad'
        assert not run.identity_ok(route, bad, allowed)


def test_existing_foundation_identity_behavior_is_preserved():
    saved = contract(); saved['model_sha'] = 'f'*64
    assert run.identity_ok(dict(family='B2'), saved, [])


def test_ast_repair_retains_every_other_check():
    from research.raw_context_multiscale_20261018_v1 import report
    import inspect
    original = ast.parse(inspect.getsource(report._verify_route))
    fixed = run.route_tree(report._verify_route)
    original_tests = [n for n in ast.walk(original) if isinstance(n, ast.If)]
    fixed_tests = [n for n in ast.walk(fixed) if isinstance(n, ast.If)]
    changed = [(a,b) for a,b in zip(original_tests,fixed_tests)
               if ast.dump(a.test) != ast.dump(b.test)]
    assert len(changed) == 1
    changed[0][0].test = changed[0][1].test
    assert ast.dump(original) == ast.dump(fixed)


def test_source_pattern_change_is_rejected():
    def unrelated():
        return True
    with pytest.raises(ValueError, match='exact identity guard'):
        run.route_tree(unrelated)


def test_validation_path_repair_is_only_literal_change():
    from research.raw_context_multiscale_20261018_v1 import report
    import inspect
    tree = run.validation_tree(report.verify)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == 'RECOVERY_VALIDATION_REPORT_001.json':
            node.value = 'VALIDATION_REPORT.json'
    assert ast.dump(tree) == ast.dump(ast.parse(inspect.getsource(report.verify)))


def test_registration_is_required_before_reading_success(tmp_path):
    from research.raw_context_multiscale_completion_recovery_20261019_v1.lineage import validate_completion
    with pytest.raises((FileNotFoundError, ValueError)):
        validate_completion(dict(repo=tmp_path, report=tmp_path/'reports', cache=tmp_path/'cache'))
