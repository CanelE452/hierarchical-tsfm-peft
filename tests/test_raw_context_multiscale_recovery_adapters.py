"""Synthetic and source-only checks for explicit completion consumer gates."""
import ast
import importlib
from pathlib import Path

import pytest

from research.raw_context_multiscale_completion_recovery_20261019_v1 import adapters


@pytest.fixture
def ctx(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    report, cache = tmp_path / "reports", tmp_path / ".cache"
    report.mkdir()
    cache.mkdir()
    return dict(repo=repo, report=report, cache=cache)


def evidence(ctx):
    return dict(terminal_path=ctx["cache"] / adapters.TERMINAL_NAME,
                validation_path=ctx["report"] / adapters.VALIDATION_NAME)


def test_only_receipt_literals_change_and_checks_are_preserved():
    tree = ast.parse('def gate(x):\n    assert x == 56\n    return ("old.json", x * 120)\n').body[0]
    changed = adapters._rewrite_tree(tree, {"old.json": ("new.json", 1)})
    assert ast.unparse(changed) == "def gate(x):\n    assert x == 56\n    return ('new.json', x * 120)"
    assert ast.unparse(tree).endswith("return ('old.json', x * 120)")


@pytest.mark.parametrize("source,replacements", [
    ('def gate():\n    return "old.json"', {"old.json": ("new.json", 2)}),
    ('def gate():\n    return ("old.json", "new.json")', {"old.json": ("new.json", 1)}),
    ('@decorator\ndef gate():\n    return "old.json"', {"old.json": ("new.json", 1)}),
])
def test_ambiguous_or_changed_function_fails_closed(source, replacements):
    with pytest.raises(ValueError):
        adapters._rewrite_tree(ast.parse(source).body[0], replacements)


def test_failed_lineage_installs_nothing(ctx, monkeypatch):
    imported = []
    def fail(_):
        raise ValueError("actual recovery has not completed")
    monkeypatch.setattr(adapters, "validate_completion", fail)
    monkeypatch.setattr(adapters.importlib, "import_module", lambda name: imported.append(name))
    with pytest.raises(ValueError, match="has not completed"):
        with adapters.install(ctx):
            pytest.fail("Failed lineage may not enter installation")
    assert imported == []


def test_unexpected_receipt_path_fails_closed(ctx, monkeypatch):
    bad = evidence(ctx)
    bad["terminal_path"] = ctx["cache"] / "TERMINAL_CORE_002.json"
    monkeypatch.setattr(adapters, "validate_completion", lambda _: bad)
    with pytest.raises(ValueError, match="unexpected receipt"):
        with adapters.install(ctx):
            pytest.fail("The failed original receipt cannot substitute for recovery")


def test_install_restores_symbols_and_preserves_failed_bytes(ctx, monkeypatch):
    old_terminal = ctx["cache"] / "TERMINAL_CORE_002.json"
    old_validation = ctx["report"] / "VALIDATION_REPORT.json"
    old_terminal.write_bytes(b'{"status":"FAILED","exit_code":1}\n')
    old_validation.write_bytes(b'{"status":"FAIL"}\n')
    before = old_terminal.read_bytes(), old_validation.read_bytes()
    originals = [(importlib.import_module(module), name) for module, name, _ in adapters.FUNCTIONS]
    functions = [getattr(module, name) for module, name in originals]
    cp0 = originals[0][0]
    monkeypatch.setattr(adapters, "validate_completion", lambda _: evidence(ctx))
    with pytest.raises(RuntimeError, match="synthetic body failure"):
        with adapters.install(ctx) as manifest:
            assert cp0.TERMINAL_NAME == adapters.TERMINAL_NAME
            assert manifest["global_reader_or_binding_changes"] is False
            assert manifest["frozen_source_files_modified"] is False
            assert len(manifest["functions"]) == 5
            for (module, name), original in zip(originals, functions):
                assert getattr(module, name) is not original
            raise RuntimeError("synthetic body failure")
    assert cp0.TERMINAL_NAME == "TERMINAL_CORE_002.json"
    assert [getattr(module, name) for module, name in originals] == functions
    assert (old_terminal.read_bytes(), old_validation.read_bytes()) == before


def test_every_consumer_revalidates_lineage_before_numerical_reads(ctx, monkeypatch):
    monkeypatch.setattr(adapters, "validate_completion", lambda _: evidence(ctx))
    with adapters.install(ctx):
        def fail(_):
            raise ValueError("synthetic broken recovery lineage")
        monkeypatch.setattr(adapters, "validate_completion", fail)
        cp0 = importlib.import_module(adapters.FUNCTIONS[0][0])
        supplement = importlib.import_module(adapters.FUNCTIONS[1][0])
        independent = importlib.import_module(adapters.FUNCTIONS[2][0])
        mcal = importlib.import_module(adapters.FUNCTIONS[4][0])
        calls = (
            lambda: cp0.validate_plan(ctx, ctx["report"] / "not_read.json", "cuda"),
            lambda: supplement.generate(ctx, plan_path=ctx["report"] / "not_read.json",
                                        terminal_path=evidence(ctx)["terminal_path"]),
            lambda: independent.numerical_gate(ctx["repo"]),
            lambda: independent.audit_cp0(ctx["repo"], {}, {}, {}, {}),
            lambda: mcal.gates(ctx, ctx["report"] / "not_read.json", ctx["report"] / "not_read.json"),
        )
        for call in calls:
            with pytest.raises(ValueError, match="broken recovery lineage"):
                call()


def test_explicit_terminal_and_campaign_context_are_required(ctx, monkeypatch):
    monkeypatch.setattr(adapters, "validate_completion", lambda _: evidence(ctx))
    with adapters.install(ctx):
        supplement = importlib.import_module(adapters.FUNCTIONS[1][0])
        with pytest.raises(ValueError, match="explicitly receive"):
            supplement.generate(ctx, plan_path=ctx["report"] / "not_read.json",
                                terminal_path=ctx["cache"] / "TERMINAL_CORE_002.json")
        independent = importlib.import_module(adapters.FUNCTIONS[2][0])
        with pytest.raises(ValueError, match="different repository"):
            independent.numerical_gate(ctx["repo"] / "other")
        wrong = dict(ctx, cache=ctx["cache"] / "other")
        cp0 = importlib.import_module(adapters.FUNCTIONS[0][0])
        with pytest.raises(ValueError, match="different campaign context"):
            cp0.validate_plan(wrong, ctx["report"] / "not_read.json", "cuda")


def test_frozen_source_change_is_rejected_before_install(ctx, monkeypatch):
    monkeypatch.setattr(adapters, "validate_completion", lambda _: evidence(ctx))
    monkeypatch.setattr(adapters, "SOURCE_BINDINGS", {name: "0" * 64 for name in adapters.SOURCE_BINDINGS})
    with pytest.raises(ValueError, match="fingerprint changed"):
        with adapters.install(ctx):
            pytest.fail("Changed scientific source cannot be installed")


def test_nested_install_rejected_and_outer_restored(ctx, monkeypatch):
    monkeypatch.setattr(adapters, "validate_completion", lambda _: evidence(ctx))
    cp0 = importlib.import_module(adapters.FUNCTIONS[0][0])
    original = cp0.validate_plan
    with adapters.install(ctx):
        installed = cp0.validate_plan
        with pytest.raises(ValueError, match="already installed"):
            with adapters.install(ctx):
                pytest.fail("Nested mutation of consumer gates is forbidden")
        assert cp0.validate_plan is installed
    assert cp0.validate_plan is original
