"""CPU-only regression of archived two-field versus deployed three-field parent identities."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
RUNNER = REPO/"reports/adapter_diagnostics_20261017_v2/code/lora_cost_runner_v2.py"


def load():
    assert RUNNER.is_file(), "Versioned Jena parent correction runner is missing"
    spec = importlib.util.spec_from_file_location("lora_parent_v2_fixture", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metadata(module):
    receipt = json.loads((REPO/module.PUBLIC_RUN_REL/"BASE_FIT_RECEIPT.json").read_text())
    manifest = json.loads((REPO/module.MANIFEST_REL).read_text())
    source = json.loads((REPO/module.PUBLIC_RUN_REL/"SOURCE_CONTRACT.json").read_text())
    return receipt, manifest, source


def test_parent_identity_reproduces_actual_deployment_formula_and_preserves_legacy_label():
    module = load()
    receipt, manifest, source = metadata(module)
    spec = module.build_route_spec(receipt, manifest, source)
    identity = module.parent_identity(receipt, manifest)
    assert manifest["model_sha"] == identity["legacy_manifest_parent_sha256"] == "a1896d9d70055f44dc471599a623ddfa746357b1fe93ec8c43fd0b0f03f6600f"
    expected = hashlib.sha256(json.dumps(dict(artifact_sha=receipt["artifact_sha256"],
        model_state_sha=receipt["model_state_sha"], normalization=receipt["refit_scaler_sha"]),
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert spec["expected_prediction_model_sha"] == expected == "8621b2eef34e081941fb1ce4e637c330f30da64fff578fef6969660504726b1b"
    assert expected != manifest["model_sha"]
    assert identity["deployment_parent_includes_scaler"] is True


@pytest.mark.parametrize("scope", ["legacy_label", "receipt_state", "receipt_scaler"])
def test_wrong_saved_label_state_or_scaler_remains_rejected(scope):
    module = load()
    receipt, manifest, source = metadata(module)
    receipt, manifest = copy.deepcopy(receipt), copy.deepcopy(manifest)
    if scope == "legacy_label":
        manifest["model_sha"] = "0"*64
    elif scope == "receipt_state":
        receipt["model_state_sha"] = "0"*64
    else:
        receipt["refit_scaler_sha"] = "0"*64
    with pytest.raises(ValueError, match="LoRA|parent"):
        module.build_route_spec(receipt, manifest, source)


def test_v2_cannot_dispatch_old_job_or_pems_before_any_runtime_import():
    module = load()
    for job in (dict(job_id="E04_jena_lora", question_id="E04", device="cuda", cohort="jena", routes=["LoRA"]),
                dict(job_id=module.JOB_ID, question_id="E04", device="cuda", cohort="pems08", routes=["LoRA"])):
        with pytest.raises(ValueError, match="v2|registered"):
            module.execute_job(job, {})
    with pytest.raises(RuntimeError, match="lease"):
        module.execute_job(dict(job_id=module.JOB_ID, question_id="E04", device="cuda", cohort="jena", routes=["LoRA"]), {})


def test_legacy_runner_is_unchanged_and_v2_owns_distinct_outputs():
    module = load()
    legacy = RUNNER.with_name("lora_cost_runner.py")
    assert module.LEGACY_RUNNER_SHA == hashlib.sha256(legacy.read_bytes()).hexdigest() == "04701c6809c5d12b04663fff6a0f43da25253cbabcf54b7c9f5e9bdcc526e6ac"
    assert module.OUTPUT_NAME == "E04_JENA_LORA_PARENT_V2"
    source = RUNNER.read_text()
    assert '"costs/E04_JENA_LORA.json"' not in source
    assert '"lora_cost/E04_JENA_LORA_CUDA_V43.npz"' not in source
