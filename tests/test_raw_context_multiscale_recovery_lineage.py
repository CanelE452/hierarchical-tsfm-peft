"""Synthetic on-disk lineage tests; no scientific arrays or model loads."""
import ast
import copy
import inspect
from pathlib import Path

import pytest

from research.raw_context_multiscale_20261018_v1 import common
from research.raw_context_multiscale_completion_recovery_20261019_v1 import lineage, run


@pytest.fixture
def registered(tmp_path):
    actual_repo = Path(__file__).resolve().parents[1]
    ctx = common.paths(tmp_path)
    ctx["report"].mkdir(parents=True)
    ctx["cache"].mkdir(parents=True)

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        common.atomic_json(path, value)
        return path

    frozen = {}
    for namespace in ("raw_context_multiscale_20261018_v1", "raw_context_multiscale_supplement_20261019_v1",
                      "raw_context_multiscale_independent_20261019_v1", "raw_context_multiscale_mcal_diagnostics_20261019_v1"):
        for source in (actual_repo / "research" / namespace).glob("*.py"):
            relative = source.relative_to(actual_repo)
            target = tmp_path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            frozen[relative.as_posix()] = common.sha(target)
    assert len(frozen) == 20
    recovery = {}
    for source in (actual_repo / "research" / lineage.NAME).glob("*.py"):
        relative = source.relative_to(actual_repo)
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        recovery[relative.as_posix()] = common.sha(target)
    tests = {}
    for name in ("test_raw_context_multiscale_completion_recovery.py",
                 "test_raw_context_multiscale_recovery_adapters.py",
                 "test_raw_context_multiscale_recovery_lineage.py"):
        source, target = actual_repo / "tests" / name, tmp_path / "tests" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        tests["tests/" + name] = common.sha(target)

    inputs = {
        "failed_terminal": write(ctx["cache"] / "TERMINAL_CORE_002.json",
                                 dict(status="FAILED", exit_code=1, finished_unix=1001.0)),
        "failed_chain_terminal": write(ctx["cache"] / "TERMINAL_SUPPLEMENT_CHAIN_001.json",
                                       dict(status="FAILED_PRESERVED_FOR_ROOT_DIAGNOSIS", stages=[])),
        "failed_validation": write(ctx["report"] / "VALIDATION_REPORT.json",
                                   dict(status="FAIL", reason="Complete source/model/config/data identities required")),
        "failed_stderr": ctx["cache"] / "EXECUTION_CORE_ATTEMPT_002.stderr.log",
        "execution_start": write(ctx["cache"] / "CORE_EXECUTION_START.json",
                                 dict(status="STARTED", started_unix=1000.0,
                                      limits=dict(wall_seconds=21600, GPU_seconds=14400))),
    }
    inputs["failed_stderr"].write_text("Synthetic verifier identity failure\n", encoding="utf8")
    for key, name, location in (("primary_plan", "EXPERIMENT_PLAN_REPAIR_002.json", "report"),
                                ("scores", "SCORES.json", "report"),
                                ("cost_index", "COST_INDEX.json", "report"),
                                ("model_seal", "MODELS_SEALED_BEFORE_V.json", "report"),
                                ("selection", "SELECTIONS_SEALED.json", "report"),
                                ("final_body_models", "FINAL_BODY_MODELS.json", "cache")):
        inputs[key] = write(ctx[location] / name, {"synthetic_metadata_only": True})
    routes = []
    for dataset in ("jena", "pems32"):
        path = ctx["cache"] / "parents" / ("G_" + dataset + ".pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"Synthetic artifact identity only; never deserialized")
        routes.append(dict(family="DIRECT_G", run_id="G_" + dataset,
                           artifact_cache_relative_path=path.relative_to(ctx["cache"]).as_posix(),
                           artifact_sha256=common.sha(path)))
    inputs["parents"] = write(ctx["report"] / "DATA_AND_PARENT_BINDINGS.json",
                              dict(parents=[dict(family="DIRECT_G", route_spec=route) for route in routes]))
    driver = ctx["cache"] / "driver_completion_recovery_001.py"
    driver.write_text("# Synthetic driver identity; never executed\n", encoding="utf8")
    plan = dict(status="REGISTERED_VERIFIER_ONLY_RECOVERY", registered_unix=1002.0,
                scientific_contract_changed=False, new_training_updates=0, new_V_forecasts=0,
                new_GPU_measurements=0, T_access=False,
                identity_exception="EXACT_REGISTERED_DIRECT_G_NO_FOUNDATION_ONLY",
                frozen_source_files=frozen, recovery_source_files=recovery, test_source_files=tests,
                input_bindings={key: common.binding(path, ctx) for key, path in inputs.items()},
                original_deadline_unix=22600.0, limits=dict(wall_seconds=21600, GPU_seconds=14400),
                allowed_direct_routes=routes, completion_driver_binding=common.binding(driver, ctx))
    write(ctx["report"] / lineage.REGISTRATION, plan)
    return ctx, plan, inputs, write


def completed(registered):
    ctx, plan, inputs, write = registered
    validation_path = write(ctx["report"] / lineage.VALIDATION,
                            dict(status="PASS", models=56, cost_facets=120, T_access=False,
                                 V_independent_test=False, manifest_bindings={
                                     "registered_plan": common.binding(inputs["primary_plan"], ctx),
                                     **{inputs[key].name: common.binding(inputs[key], ctx)
                                        for key in ("scores", "cost_index", "model_seal")}}))
    generated = {name: write(ctx["report"] / name, {"synthetic_metadata_only": True})
                 for name in ("REPORT_INPUTS.json", "RESOURCE_LEDGER.json", "README_KO.md")}
    registration_binding = common.binding(ctx["report"] / lineage.REGISTRATION, ctx)
    validation_binding = common.binding(validation_path, ctx)
    work = dict(status="PASS_VERIFIED_AND_RENDERED", started_unix=1003.0, finished_unix=1004.0,
                wall_seconds=1.0,
                registration_binding=registration_binding, validation_binding=validation_binding,
                generated_bindings={key: common.binding(path, ctx) for key, path in generated.items()},
                GPU_seconds=0, new_training_updates=0, new_V_forecasts=0, new_GPU_measurements=0, T_access=False)
    work_path = write(ctx["cache"] / lineage.WORK_RESULT, work)
    terminal = dict(status="COMPLETE", exit_code=0, started_unix=1003.0, finished_unix=1005.0,
                    wall_seconds=2.0, GPU_seconds=0, driver_binding=plan["completion_driver_binding"],
                    original_deadline_unix=plan["original_deadline_unix"],
                    registration_binding=registration_binding, validation_binding=validation_binding,
                    work_result_binding=common.binding(work_path, ctx))
    write(ctx["cache"] / lineage.TERMINAL, terminal)
    return ctx, plan, inputs, write, work, terminal


def test_complete_lineage_passes_and_original_fail_bytes_are_preserved(registered):
    ctx, _, inputs, _, _, _ = completed(registered)
    before = {key: path.read_bytes() for key, path in inputs.items() if key.startswith("failed_")}
    result = lineage.validate_completion(ctx)
    assert result == dict(terminal_path=ctx["cache"] / lineage.TERMINAL,
                          validation_path=ctx["report"] / lineage.VALIDATION)
    assert {key: inputs[key].read_bytes() for key in before} == before


@pytest.mark.parametrize("mutation", ("source", "artifact", "failed_terminal", "failed_validation", "deadline"))
def test_registration_rejects_changed_immutable_inputs(registered, mutation):
    ctx, plan, inputs, write = registered
    if mutation == "source":
        source = ctx["repo"] / next(iter(plan["frozen_source_files"]))
        source.write_bytes(source.read_bytes() + b"\n# Synthetic mutation\n")
    elif mutation == "artifact":
        source = ctx["cache"] / plan["allowed_direct_routes"][0]["artifact_cache_relative_path"]
        source.write_bytes(b"Changed synthetic artifact")
    elif mutation == "failed_terminal":
        write(inputs["failed_terminal"], dict(status="COMPLETE", exit_code=0, finished_unix=1001.0))
    elif mutation == "failed_validation":
        write(inputs["failed_validation"], dict(status="PASS"))
    else:
        plan["original_deadline_unix"] += 1
        write(ctx["report"] / lineage.REGISTRATION, plan)
    with pytest.raises(ValueError):
        lineage.validate_registration(ctx)


@pytest.mark.parametrize("mutation", ("terminal_status", "work_status", "model_count", "cost_count", "chronology", "generated_file",
                                      "empty_generated", "new_training", "new_V", "work_GPU", "terminal_GPU", "T_access", "wrong_driver"))
def test_completion_rejects_bad_success_receipts(registered, mutation):
    ctx, _, _, write, work, terminal = completed(registered)
    if mutation == "terminal_status":
        terminal["status"] = "FAILED"
        write(ctx["cache"] / lineage.TERMINAL, terminal)
    elif mutation == "work_status":
        work["status"] = "FAILED"
        write(ctx["cache"] / lineage.WORK_RESULT, work)
    elif mutation in ("model_count", "cost_count"):
        validation = common.read(ctx["report"] / lineage.VALIDATION)
        validation["models" if mutation == "model_count" else "cost_facets"] -= 1
        write(ctx["report"] / lineage.VALIDATION, validation)
    elif mutation == "chronology":
        terminal["started_unix"] = 999.0
    elif mutation == "generated_file":
        (ctx["report"] / "README_KO.md").write_bytes(b"Changed generated file")
    elif mutation == "empty_generated":
        work["generated_bindings"] = {}
    elif mutation == "new_training":
        work["new_training_updates"] = 1
    elif mutation == "new_V":
        work["new_V_forecasts"] = 1
    elif mutation == "work_GPU":
        work["GPU_seconds"] = 1
    elif mutation == "terminal_GPU":
        terminal["GPU_seconds"] = 1
    elif mutation == "T_access":
        work["T_access"] = True
    elif mutation == "wrong_driver":
        terminal["driver_binding"] = terminal["registration_binding"]
    # Refresh every downstream binding so tests exercise the semantic checks,
    # rather than merely failing on a stale JSON checksum.
    work["validation_binding"] = common.binding(ctx["report"] / lineage.VALIDATION, ctx)
    write(ctx["cache"] / lineage.WORK_RESULT, work)
    terminal["validation_binding"] = work["validation_binding"]
    terminal["work_result_binding"] = common.binding(ctx["cache"] / lineage.WORK_RESULT, ctx)
    write(ctx["cache"] / lineage.TERMINAL, terminal)
    with pytest.raises(ValueError):
        lineage.validate_completion(ctx)


def test_route_repair_preserves_full_state_scaler_source_config_guards():
    from research.raw_context_multiscale_20261018_v1 import report
    original = ast.parse(inspect.getsource(report._verify_route))
    changed = run.route_tree(report._verify_route)
    identity_guard = next(node for node in ast.walk(changed)
                          if isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp)
                          and isinstance(node.test.operand, ast.Call)
                          and isinstance(node.test.operand.func, ast.Name)
                          and node.test.operand.func.id == "_recovery_identity_ok")
    original_identity = next(node for node in ast.walk(original)
                             if isinstance(node, ast.If) and len(node.body) == 1
                             and isinstance(node.body[0], ast.Raise)
                             and isinstance(node.body[0].exc, ast.Call)
                             and any(isinstance(value, ast.Constant) and value.value ==
                                     "Complete source/model/config/data identities required"
                                     for value in node.body[0].exc.args))
    identity_guard.test = copy.deepcopy(original_identity.test)
    assert ast.dump(changed) == ast.dump(original)
    source = ast.unparse(changed)
    for check in ("model_state_sha", "scaler_fingerprint", "config_sha", "new_scientific_source_hashes"):
        assert check in source
