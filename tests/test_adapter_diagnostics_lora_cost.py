"""CPU-only identity/parity guards for the separately scheduled LoRA cost runner."""
import importlib.util
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


RUNNER = (Path(__file__).resolve().parents[1] / "reports" /
          "adapter_diagnostics_20261017_v2/code/lora_cost_runner.py")


def load_runner():
    assert RUNNER.is_file(), "The registered LoRA cost runner is missing"
    spec = importlib.util.spec_from_file_location("lora_cost_test_runner", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metadata(module):
    receipt = dict(status="COMPLETE_VALID", run_id=module.RUN_ID, family="LORA", seed=92601,
                   artifact_sha256="a" * 64, artifact_cache_relative_path=module.ARTIFACT_REL,
                   refit_fit_end=12579, search_fit_end=10482, refit_scaler_sha=module.SCALER_SHA,
                   selected_epoch=1., model_state_sha="b" * 64)
    manifest = dict(run_id=module.RUN_ID, family="LORA", seed=92601, split="V_select",
                    data_sha=module.DATA_SHA, scaler_sha=module.SCALER_SHA,
                    artifact_sha256="a" * 64, archive_sha256="c" * 64,
                    model_sha="d" * 64, score=dict(mse=.56, mae=.39, n_origins=43,
                    num_channels=21, horizon=48), arrays={key: dict(shape=shape) for key, shape in
                    (("pred_normalized", [43, 48, 21]), ("target_normalized", [43, 48, 21]),
                     ("observed", [43, 48, 21]), ("origins", [43]), ("channel_ids", [21]))})
    source = dict(data_sha=module.DATA_SHA, model_sha=module.F_SHA,
                  spec=dict(run_id=module.RUN_ID, family="LORA", dataset="jena", seed=92601,
                            lora_rank=8, lora_alpha=16, F_row_microbatch=128))
    return receipt, manifest, source


def test_cost_route_binds_the_fresh_selected_lora_and_saved_v():
    module = load_runner()
    receipt, manifest, source = metadata(module)
    spec = module.build_route_spec(receipt, manifest, source)
    assert spec["artifact_sha256"] == receipt["artifact_sha256"]
    assert spec["artifact_cache_relative_path"] == module.ARTIFACT_REL
    assert spec["expected_prediction_model_sha"] == manifest["model_sha"]
    assert spec["row_microbatch"] == 128
    assert spec["canonical_fit_end"] == 12579


@pytest.mark.parametrize("field,value", [("artifact_sha256", "e" * 64),
                                        ("scaler_sha", "e" * 64),
                                        ("data_sha", "e" * 64),
                                        ("run_id", "another_run")])
def test_cost_route_rejects_a_different_accuracy_contract(field, value):
    module = load_runner()
    receipt, manifest, source = metadata(module)
    manifest[field] = value
    with pytest.raises(ValueError, match="LoRA"):
        module.build_route_spec(receipt, manifest, source)


def test_saved_parity_checks_targets_masks_origins_channels_and_predictions():
    module = load_runner()
    origins = np.array([14675, 14723], np.int64)
    physical = np.arange(2 * 48 * 2, dtype=np.float32).reshape(2, 48, 2)
    mean, std = np.array([2., 3.], np.float32), np.array([4., 5.], np.float32)
    arrays = dict(pred_normalized=physical.copy(), target_normalized=physical.copy(),
                  observed=np.ones_like(physical, bool), origins=origins.copy(),
                  channel_ids=np.array(["first", "second"]))
    actual = physical.astype(np.float64) * std + mean
    result = module.saved_prediction_parity(actual, arrays, origins, arrays["channel_ids"], mean, std)
    assert result["status"] == "PASS"
    assert result["maximum_absolute_difference"] == 0
    arrays["origins"][1] += 1
    with pytest.raises(ValueError, match="origin"):
        module.saved_prediction_parity(actual, arrays, origins, arrays["channel_ids"], mean, std)
    arrays["origins"] = origins.copy()
    arrays["channel_ids"] = arrays["channel_ids"][::-1]
    with pytest.raises(ValueError, match="channel"):
        module.saved_prediction_parity(actual, arrays, origins, np.array(["first", "second"]), mean, std)
    arrays["channel_ids"] = np.array(["first", "second"])
    actual[0, 0, 0] += 100.
    assert module.saved_prediction_parity(actual, arrays, origins, arrays["channel_ids"], mean, std)["status"] == "FAILED"


def test_invalid_job_or_missing_lease_fails_before_any_runtime_import():
    module = load_runner()
    with pytest.raises(ValueError, match="E04"):
        module.execute_job(dict(question_id="E03", device="cuda"), {})
    with pytest.raises(RuntimeError, match="lease"):
        module.execute_job(dict(question_id="E04", device="cuda", cohort="jena"), {})


def test_default_cli_is_source_only(capsys):
    module = load_runner()
    assert module.main([]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "SOURCE_ONLY_NOOP"


def test_exact_registered_peMs_routes_reject_an_unregistered_or_incomplete_comparison():
    module = load_runner()
    methods = ["Raw", "B2", "Ridge", "LoRA", "Chronos2_nativeRaw"]
    assert module.registered_routes(dict(cohort="pems08", routes=methods)) == methods
    with pytest.raises(ValueError, match="registered"):
        module.registered_routes(dict(cohort="pems08", routes=["Raw", "LoRA"]))
    assert module.registered_routes(dict(cohort="jena", routes=["LoRA"])) == ["LoRA"]


def test_public_metadata_gzip_requires_both_archive_and_original_hash(tmp_path):
    module = load_runner()
    relative = "research/informs_broad_validation_20261015_v1/runs/a/RECEIPT.json"
    directory = tmp_path / "research/informs_broad_validation_20261015_v1/publication_derivatives/gzip_extra_aux_20261016_v1"
    directory.mkdir(parents=True)
    raw = b'{"status":"COMPLETE_VALID"}'
    packed = gzip.compress(raw)
    (directory / "receipt.json.gz").write_bytes(packed)
    entry = dict(original_file=relative, gzip_file="receipt.json.gz", gzip_sha256=hashlib.sha256(packed).hexdigest(),
                 original_sha256=hashlib.sha256(raw).hexdigest(), original_bytes=len(raw))
    manifest = directory / "GZIP_MANIFEST.json"
    manifest.write_text(json.dumps(dict(items=[entry])))
    actual, bindings = module.read_public_json(tmp_path, relative)
    assert actual == dict(status="COMPLETE_VALID")
    assert len(bindings) == 2
    entry["original_sha256"] = "0" * 64
    manifest.write_text(json.dumps(dict(items=[entry])))
    with pytest.raises(ValueError, match="original"):
        module.read_public_json(tmp_path, relative)


@pytest.mark.parametrize("batch", [1, 8])
def test_measured_batch_parity_uses_matching_saved_replica_origins(batch):
    module = load_runner()
    origins = np.arange(20, 32, dtype=np.int64)
    predictions = np.arange(12 * 3 * 2, dtype=np.float32).reshape(12, 3, 2)
    channels = np.array(["first", "second"])
    replica = dict(pred_normalized=predictions, target_normalized=np.zeros_like(predictions),
                   observed=np.ones_like(predictions, bool), origins=origins, channel_ids=channels)
    selected = origins[2:2+batch]
    mean, std = np.array([2., 3.], np.float32), np.array([4., 5.], np.float32)
    output = predictions[2:2+batch].astype(np.float64) * std + mean
    result = module.measured_batch_parity(output, replica, selected, channels, mean, std)
    assert result["status"] == "PASS"
    assert result["origins"] == selected.tolist()
    assert result["B"] == batch
    assert result["maximum_absolute_difference"] == 0
    output[0, 0, 0] += 100.
    assert module.measured_batch_parity(output, replica, selected, channels, mean, std)["status"] == "FAILED"


def test_measured_batch_rejects_missing_replica_origin():
    module = load_runner()
    replica = dict(origins=np.array([10]), channel_ids=np.array(["channel"]),
                   pred_normalized=np.zeros((1, 2, 1)), target_normalized=np.zeros((1, 2, 1)),
                   observed=np.ones((1, 2, 1), bool))
    with pytest.raises(ValueError, match="origin"):
        module.measured_batch_parity(np.zeros((1, 2, 1)), replica, np.array([11]),
                                     np.array(["channel"]), np.zeros(1), np.ones(1))
