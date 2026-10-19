"""Explicit consumer gates for the separately verified completion recovery.

The frozen scientific modules are not edited. Only the five named functions
below are adapted, and only their enumerated receipt filename literals change.
Every adapted invocation validates the recovery lineage before the original
function can read numerical inputs. No file reader or binding resolver changes.
"""
import ast
import copy
from contextlib import contextmanager
import functools
import importlib
import inspect
from pathlib import Path
import types

from .lineage import validate_completion


TERMINAL_NAME = "TERMINAL_COMPLETION_RECOVERY_001.json"
VALIDATION_NAME = "RECOVERY_VALIDATION_REPORT_001.json"
SOURCE_BINDINGS = {
    "research/raw_context_multiscale_supplement_20261019_v1/cp0.py":
        "fd84887346e188462c8bdd54087fd42204a93e3b1547da7a75c51f4ad9e4c7bc",
    "research/raw_context_multiscale_supplement_20261019_v1/report.py":
        "2491863203ad6aaa0ab1606ad4fc05f5f91919a4e7df15180d328265072046f6",
    "research/raw_context_multiscale_independent_20261019_v1/run.py":
        "e979bf8abff3ff4f48c66c50d099996df29a8a4540a2cd3f2b962fbb1b78ba9a",
    "research/raw_context_multiscale_mcal_diagnostics_20261019_v1/run.py":
        "d52f0c1515bff3531d83a7d30b1a217b40dafa9660a7e1e6c09492b61f32f462",
}
# Exact occurrence counts in each immutable function, including the final
# report's public primary_input_bindings key as well as its validation read.
FUNCTIONS = (
    ("research.raw_context_multiscale_supplement_20261019_v1.cp0", "validate_plan", {}),
    ("research.raw_context_multiscale_supplement_20261019_v1.report", "generate",
     {"VALIDATION_REPORT.json": (VALIDATION_NAME, 2)}),
    ("research.raw_context_multiscale_independent_20261019_v1.run", "numerical_gate",
     {"TERMINAL_CORE_002.json": (TERMINAL_NAME, 1), "VALIDATION_REPORT.json": (VALIDATION_NAME, 1)}),
    ("research.raw_context_multiscale_independent_20261019_v1.run", "audit_cp0",
     {"TERMINAL_CORE_002.json": (TERMINAL_NAME, 1)}),
    ("research.raw_context_multiscale_mcal_diagnostics_20261019_v1.run", "gates",
     {"TERMINAL_CORE_002.json": (TERMINAL_NAME, 1), "VALIDATION_REPORT.json": (VALIDATION_NAME, 1)}),
)


def _rewrite_tree(original, replacements):
    """Return the same function AST with only the declared constants replaced."""
    if not isinstance(original, ast.FunctionDef) or original.decorator_list:
        raise ValueError("An undecorated named frozen function is required")
    result = copy.deepcopy(original)
    counts = {old: 0 for old in replacements}
    for node in ast.walk(result):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in replacements:
            old = node.value
            node.value = replacements[old][0]
            counts[old] += 1
    if counts != {old: count for old, (_, count) in replacements.items()}:
        raise ValueError("Frozen function receipt literal occurrence count changed")
    restored = copy.deepcopy(result)
    inverse = {new: old for old, (new, _) in replacements.items()}
    if len(inverse) != len(replacements) or any(
            isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in inverse
            for node in ast.walk(original)):
        raise ValueError("Recovery filename is ambiguous in the original function")
    for node in ast.walk(restored):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in inverse:
            node.value = inverse[node.value]
    if ast.dump(restored, include_attributes=True) != ast.dump(original, include_attributes=True):
        raise ValueError("Consumer adaptation changed more than enumerated receipt filenames")
    return result


def _same_context(actual, registered):
    for key in ("repo", "report", "cache"):
        if Path(actual[key]).resolve() != Path(registered[key]).resolve():
            raise ValueError("Recovery adapter called for a different campaign context")


def _evidence_paths(ctx):
    evidence = validate_completion(ctx)
    expected = {
        "terminal_path": Path(ctx["cache"]) / TERMINAL_NAME,
        "validation_path": Path(ctx["report"]) / VALIDATION_NAME,
    }
    for key, path in expected.items():
        if Path(evidence[key]).resolve() != path.resolve():
            raise ValueError("Recovery lineage returned an unexpected receipt path: " + key)
    return evidence


def _adapt(module, name, replacements, ctx):
    original = getattr(module, name)
    if getattr(original, "_completion_recovery_adapter", False):
        raise ValueError("Recovery adapters are already installed")
    source_path = Path(inspect.getsourcefile(original)).resolve()
    relative = source_path.relative_to(Path(ctx["repo"]).resolve()).as_posix()
    from research.raw_context_multiscale_20261018_v1 import common
    if relative not in SOURCE_BINDINGS or common.sha(source_path) != SOURCE_BINDINGS[relative]:
        raise ValueError("Frozen consumer source fingerprint changed: " + relative)
    functions = [node for node in ast.parse(source_path.read_text(encoding="utf8")).body
                 if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(functions) != 1:
        raise ValueError("Exactly one registered consumer function is required")
    tree = ast.fix_missing_locations(ast.Module(body=[_rewrite_tree(functions[0], replacements)], type_ignores=[]))
    compiled = compile(tree, str(source_path), "exec")
    codes = [code for code in compiled.co_consts if isinstance(code, types.CodeType) and code.co_name == name]
    if len(codes) != 1 or original.__closure__:
        raise ValueError("A single module-level consumer function is required")
    adapted = types.FunctionType(codes[0], module.__dict__, name, original.__defaults__)
    adapted.__kwdefaults__ = copy.copy(original.__kwdefaults__)
    adapted.__annotations__ = copy.copy(original.__annotations__)
    signature = inspect.signature(original)

    @functools.wraps(original)
    def guarded(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        if "ctx" in bound.arguments:
            _same_context(bound.arguments["ctx"], ctx)
        elif Path(bound.arguments["repo"]).resolve() != Path(ctx["repo"]).resolve():
            raise ValueError("Recovery adapter called for a different repository")
        evidence = _evidence_paths(ctx)
        if "terminal_path" in bound.arguments and Path(bound.arguments["terminal_path"]).resolve() != Path(evidence["terminal_path"]).resolve():
            raise ValueError("Final report must explicitly receive the actual recovery terminal")
        return adapted(*args, **kwargs)

    guarded._completion_recovery_adapter = True
    guarded._completion_recovery_replacements = copy.deepcopy(replacements)
    return original, guarded


@contextmanager
def install(ctx):
    """Temporarily install only named, lineage-checked consumer adaptations."""
    _evidence_paths(ctx)  # Fail before importing or replacing any consumer gate.
    changes = []
    for module_name, name, replacements in FUNCTIONS:
        module = importlib.import_module(module_name)
        original, adapted = _adapt(module, name, replacements, ctx)
        changes.append((module, name, original, adapted))
    cp0 = changes[0][0]
    if cp0.TERMINAL_NAME != "TERMINAL_CORE_002.json":
        raise ValueError("Frozen CP0 terminal name changed before recovery installation")
    original_terminal_name = cp0.TERMINAL_NAME
    try:
        for module, name, _, adapted in changes:
            setattr(module, name, adapted)
        cp0.TERMINAL_NAME = TERMINAL_NAME
        yield {
            "source_bindings": dict(SOURCE_BINDINGS),
            "functions": [{"module": module_name, "function": name,
                           "literal_changes": {old: dict(new=new, count=count)
                                               for old, (new, count) in replacements.items()}}
                          for module_name, name, replacements in FUNCTIONS],
            "terminal_name": TERMINAL_NAME,
            "validation_name": VALIDATION_NAME,
            "global_reader_or_binding_changes": False,
            "frozen_source_files_modified": False,
        }
    finally:
        cp0.TERMINAL_NAME = original_terminal_name
        for module, name, original, _ in reversed(changes):
            setattr(module, name, original)
