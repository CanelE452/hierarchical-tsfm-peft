import copy
import hashlib
import importlib
import json

import numpy as np
import pytest


def verifier():
    try:
        return importlib.import_module("research.gt_ts2_fastcheck_20261018_v1.verify")
    except ModuleNotFoundError:
        pytest.fail("independent final campaign verifier is missing")


def array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def prediction_fixture():
    prediction = np.ones((2, 48, 2), np.float32)
    prediction[..., 1] = 2.
    bundle = {"prediction": prediction, "target": np.zeros_like(prediction),
              "mask": np.ones_like(prediction, bool), "origins": np.array([1024, 1048]),
              "mean": np.array([3., 4.], np.float32), "std": np.array([2., 3.], np.float32)}
    history = np.zeros((2, 512, 2), np.float32)
    receipt = {"run_id": "fixture", "L": 512, "H": 48, "C": 2, "origin_count": 2,
               "MSE": 2.5, "MAE": 1.5, "input_sha256": array_hash(history)}
    receipt.update({key + "_sha256": array_hash(value) for key, value in bundle.items()})
    return receipt, bundle, history


def test_prediction_recompute_rejects_scalar_tampering_and_wrong_window():
    module = verifier()
    receipt, bundle, history = prediction_fixture()
    result = module.verify_prediction(receipt, bundle, history, bundle)
    assert result["mse"] == 2.5 and result["mae"] == 1.5
    changed = {**receipt, "MSE": 2.6}
    with pytest.raises(ValueError, match="metric"):
        module.verify_prediction(changed, bundle, history, bundle)
    with pytest.raises(ValueError, match="input window"):
        module.verify_prediction({**receipt, "L": 1024}, bundle, history, bundle)


def test_prediction_recompute_rejects_changed_target_even_with_updated_receipt_hash():
    receipt, bundle, history = prediction_fixture()
    original = copy.deepcopy(bundle)
    bundle["target"][0, 0, 0] = 10.
    receipt["target_sha256"] = array_hash(bundle["target"])
    with pytest.raises(ValueError, match="cohort"):
        verifier().verify_prediction(receipt, bundle, history, original)


def test_missing_eighth_training_arm_is_rejected():
    module = verifier()
    required = [row["run_id"] for row in module.expected_runs() if row["phase"] == "B"]
    assert len(required) == 8
    with pytest.raises(ValueError, match="training.*missing"):
        module.require_exact_ids(required[:-1], required, "training")


def test_cost_verification_rejects_cached_ts2_in_timed_path():
    spec = {"run_id": "A_TS2_L1024_s92601", "mode": "ts2", "L": 1024, "phase": "A", "updates": 0}
    with pytest.raises(ValueError, match="online"):
        verifier().verify_cost_receipt({"TS2cache_used": True}, spec, np.arange(371), {}, "fixture")


def test_frozen_foundation_state_change_is_rejected():
    with pytest.raises(ValueError, match="Frozen"):
        verifier().verify_frozen_states({"F": "canonical", "TS2_SWA": "same"},
                                         {"F": "changed", "TS2_SWA": "same"}, "canonical")


def test_foundation_work_counts_contexts_and_physical_calls():
    module = verifier()
    expected = (("raw", 2048, 168, 2, [2048]), ("gt", 2048, 336, 3, [512]),
                ("ts2", 1024, 2560, 20, [1024]), ("ensemble", 1024, 2896, 23, [1024, 256]))
    for mode, length, streams, calls, contexts in expected:
        result = module.expected_F_work({"mode": mode, "L": length}, 8, 2048)
        assert result["streams"] == streams and result["calls"] == calls
        assert result["requested"] == contexts and result["actual"] == contexts


def test_missing_final_tests_and_replay_cannot_be_certified(tmp_path, monkeypatch):
    module = verifier()
    monkeypatch.setattr(module.c, "REPORT", tmp_path / "report")
    monkeypatch.setattr(module.c, "CACHE", tmp_path / "cache")
    readiness = module.readiness()
    assert readiness["status"] == "INCOMPLETE"
    assert "TEST_VALIDATION.json" in readiness["missing_evidence"]
    assert "PUBLIC_REPLAY_VALIDATION.json" in readiness["missing_evidence"]
    with pytest.raises(FileNotFoundError, match="Cannot certify"):
        module.run_verification()
    assert not (tmp_path / "report").exists()


def test_verify_cli_accepts_the_run_wrapper_stage_argument(monkeypatch, capsys):
    module = verifier()
    monkeypatch.setattr(module, "readiness", lambda: {"status": "INCOMPLETE", "writes_performed": False})
    assert module.main(["verify", "--check-only"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "INCOMPLETE"


def test_public_csv_accepts_fp64_reduction_rounding_but_rejects_metric_tampering(tmp_path):
    path = tmp_path / "stats.csv"
    path.write_text('run_id,sse,channels\nx,3.0,"[0.10000000000000002, 2.0]"\n')
    expected = [{"run_id": "x", "sse": 3., "channels": [.1, 2.]}]
    assert verifier().verify_csv(path, expected, ("run_id",)) == 1
    path.write_text('run_id,sse,channels\nx,3.1,"[0.10000000000000002, 2.0]"\n')
    with pytest.raises(ValueError, match="differs"):
        verifier().verify_csv(path, expected, ("run_id",))


def test_independent_schedule_keeps_each_five_window_tail_and_equal_exposure():
    module = verifier()
    batches, matrix, digest = module.independent_schedule(92601, 197, 300)
    exposure = np.bincount(np.concatenate(batches), minlength=197)
    np.testing.assert_array_equal(exposure, 12)
    assert sum(len(batch) == 5 for batch in batches) == 12
    assert matrix.shape == (300, 8)
    assert np.count_nonzero(matrix == -1) == 36
    assert digest != module.independent_schedule(92602, 197, 300)[2]


def test_adam_checkpoint_rejects_an_incorrect_actual_step():
    module = verifier()
    torch = module.torch
    saved = {"update": 100, "optimizer": "Adam", "learning_rate": 1e-4, "weight_decay": 0.,
             "gradient_norm_clip": 1., "adapter_state": {"weight": torch.ones(2, 2)},
             "optimizer_state": {"param_groups": [{"params": [0], "lr": 1e-4, "weight_decay": 0.}],
                                 "state": {0: {"step": torch.tensor(100.), "exp_avg": torch.zeros(2, 2),
                                                "exp_avg_sq": torch.ones(2, 2)}}}}
    module._optimizer(saved, ["weight"])
    saved["optimizer_state"]["state"][0]["step"] = torch.tensor(99.)
    with pytest.raises(ValueError, match="Adam step"):
        module._optimizer(saved, ["weight"])


def test_used_source_checks_ignore_incidental_snapshot_modules():
    module = verifier()
    module._sources({"project/model.py": "pinned", "project/reporting.py": "old-unused"},
                    ("model.py",), {"model.py": "pinned", "reporting.py": "new-unused"})
    with pytest.raises(ValueError, match="producer source"):
        module._sources({"project/model.py": "changed"}, ("model.py",), {"model.py": "pinned"})


def test_historical_a0_exception_requires_exact_preserved_source(tmp_path, monkeypatch):
    module = verifier()
    monkeypatch.setattr(module.c, "REPORT", tmp_path)
    preserved = tmp_path / "source_history/old_execution.py"
    preserved.parent.mkdir()
    preserved.write_bytes(b"historical exact source")
    digest = hashlib.sha256(preserved.read_bytes()).hexdigest()
    proof = {"status": "EXACT_HISTORICAL_SOURCE_PRESERVED", "amendments": [
        {"module": "execution.py", "before_sha256": digest, "after_sha256": "current",
         "historical_file": "source_history/old_execution.py", "parent_reproduction_evidence_preserved": True}]}
    (tmp_path / "SOURCE_AMENDMENTS.json").write_text(json.dumps(proof))
    assert module._source_history({"execution.py": "current", "model.py": "same"}) == {"execution.py": digest, "model.py": "same"}
    preserved.write_bytes(b"different source")
    with pytest.raises(ValueError, match="source bytes"):
        module._source_history({"execution.py": "current"})


def cost_fixture():
    module = verifier()
    spec = {"run_id": "A_RAW_L512", "phase": "A", "mode": "raw", "seed": 0, "L": 512, "updates": 0, "arm": "RAW_F"}
    origins = np.arange(371, dtype=np.int64) + 1024
    contract = {"cost": {"fixed_group_indices": {"1": [0, 185, 370]}}}
    zero = {"parameters": 0, "parameter_bytes": 0, "buffer_bytes": 0}
    gpu = {"parameters": 2, "parameter_bytes": 8, "buffer_bytes": 0}
    inventory = {"by_device": {"cpu": zero, "cuda": gpu}, "by_module": {"F": {"cpu": zero, "cuda": gpu}},
                 "total_unique_parameters": 2, "total_unique_parameter_bytes": 8, "total_unique_buffer_bytes": 0}
    summary = {"p50_ms": 19.5, "q25_ms": 14.75, "q75_ms": 24.25, "iqr_ms": 9.5, "p95_ms": 28.05,
               "origins_per_second_at_median": 1000 / 19.5, "peak_allocated_MiB": 72., "peak_reserved_MiB": 100.,
               "RSS_MiB": 120., "repetitions": 20, "updates": 0, "resident_inventory": inventory,
               "resident_parameters": 2, "CUDA_resident_parameters": 2, "CPU_resident_parameters": 0,
               "CUDA_resident_parameter_bytes": 8, "CPU_resident_parameter_bytes": 0,
               "CUDA_resident_buffer_bytes": 0, "CPU_resident_buffer_bytes": 0, "system_resident_parameters": 2}
    trials = []
    for stage, count in (("warmup", 10), ("timed", 20)):
        for i in range(count):
            start = [0, 185, 370][i % 3]
            ms = 100. + i if stage == "warmup" else 10. + i
            trials.append({**spec, "stage": stage, "iteration": i, "group_index": i % 3, "groupstart": start,
                           "origins_sha256": array_hash(origins[start:start + 1]), "batch": 1,
                           "TS2cache_used": False, "prediction_parity": {"status": "PASS", "atol": 1e-5, "rtol": 1e-4},
                           "F_calls": 1, "F_streams": 21, "requested_context_length": [512], "actual_context_length": [512],
                           "F_context_records": [{"streams": 21, "row_sizes": [21], "requested_context_length": 512, "actual_context_length": 512}],
                           "whole_batch_ms": ms, "origins_per_second": 1000 / ms, "peak_allocated_MiB": 72.,
                           "peak_reserved_MiB": 100., "RSS_MiB": 120., "pid": 1, "process_create_time_unix": 100.,
                           "checkpoint_sha256": {}, "loaded_source_sha256": {},
                           "memory_peak_scope": "cumulative warmup phase" if stage == "warmup" else "one timed repetition including all resident models"})
    value = {"status": "COMPLETE", "TS2cache_used": False, "spec": spec, "batch": 1, "F_row_microbatch": 128,
             "TS2_present": False, "TS2_pretrained_weights_used": False,
             "states_before_sha256": {"F": "canonical"}, "states_after_sha256": {"F": "canonical"},
             "validation": {"status": "PASS", "parameter_gradients_absent": {"F": True}, "F_unchanged": True, "TS2_unchanged": True},
             "trials": trials, "summary": summary, "pid": 1, "process_create_time_unix": 100.,
             "checkpoint_sha256": {}, "loaded_source_sha256": {}, "loaded_baseline": {"allocated_MiB": 64.},
             "warmup_peaks": {"peak_allocated_MiB": 72., "peak_reserved_MiB": 100., "RSS_MiB": 120.},
             "resident_inventory": inventory}
    return value, spec, origins, contract


@pytest.mark.parametrize("mutation", [None, "calls", "context", "median", "resident_memory"])
def test_full_cost_receipt_rechecks_actual_calls_contexts_quantiles_and_resident_memory(mutation):
    value, spec, origins, contract = cost_fixture()
    if mutation == "calls":
        value["trials"][10]["F_calls"] = 2
    elif mutation == "context":
        value["trials"][10]["actual_context_length"] = [128]
    elif mutation == "median":
        value["summary"]["p50_ms"] = 20.
    elif mutation == "resident_memory":
        value["trials"][10]["peak_allocated_MiB"] = 8.
    if mutation is None:
        assert verifier().verify_cost_receipt(value, spec, origins, contract, "canonical")["status"] == "PASS"
    else:
        with pytest.raises(ValueError):
            verifier().verify_cost_receipt(value, spec, origins, contract, "canonical")


@pytest.mark.parametrize("mutation", [None, "tested_source", "local_log", "local_artifact"])
def test_final_proofs_bind_tested_sources_and_actual_full_local_replay(tmp_path, monkeypatch, mutation):
    module = verifier()
    monkeypatch.setattr(module.c, "ROOT", tmp_path)
    monkeypatch.setattr(module.c, "CODE", tmp_path / "research/fixture")
    monkeypatch.setattr(module.c, "REPORT", tmp_path / "reports/fixture")
    monkeypatch.setattr(module.c, "CACHE", tmp_path / "cache/fixture")
    module.c.CODE.mkdir(parents=True)
    module.c.REPORT.mkdir(parents=True)
    module.c.CACHE.mkdir(parents=True)
    sources = {}
    for name in (*module.CORE_NAMES, "verify.py"):
        path = module.c.CODE / name
        path.write_text(name)
        sources[path.relative_to(tmp_path).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = tmp_path / "tests/test_gt_ts2_verify.py"
    path.parent.mkdir()
    path.write_text("fixture test source")
    sources[path.relative_to(tmp_path).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    logs = {}
    for name, contents in (("tests.log", "3 passed in 0.1s"), ("public.log", "public replay passed"), ("local.log", "local replay passed")):
        path = module.c.CACHE / name
        path.write_text(contents)
        logs[name] = {"exit_code": 0, "log_reference": "cache://" + name,
                      "log_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    names = ["RUNS.csv", "COMPARISONS.csv", "CHANNEL_HORIZON_STATS.csv", "ORIGIN_EFFECTS.csv", "COST.csv", "COST_TRIALS.csv",
             "TRAINING_CURVES.csv", "JOB_STATUS.csv", "RESOURCE_LEDGER.json", "RESUME_QUEUE.md", "README_KO.md", "METHOD_DELTA_KO.md"]
    names += [f"FIG{index}_fixture.{suffix}" for index in (1, 2, 3) for suffix in ("png", "svg", "csv")]
    files = {}
    for name in names:
        path = module.c.REPORT / name
        path.write_text(name)
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    tests = {"status": "PASS", **logs["tests.log"], "passed": 3, "failed": 0, "errors": 0,
             "source_sha256": sources, "source_sha256_after": sources, "source_unchanged": True,
             "excluded_files": ["tests/test_model_parity.py", "tests/test_baselines.py"],
             "exclusion_reason": "old Labour training/data integrations excluded from this campaign"}
    replay = {"status": "PASS", **logs["public.log"], "files": files,
              "public_only_scope": "public statistics only",
              "full_local_replay": {"status": "PASS", **logs["local.log"], "files": files.copy()}}
    if mutation == "tested_source":
        (module.c.CODE / "model.py").write_text("changed after tests")
    elif mutation == "local_log":
        (module.c.CACHE / "local.log").write_text("changed actual log")
    elif mutation == "local_artifact":
        replay["full_local_replay"]["files"]["RUNS.csv"] = "0" * 64
    (module.c.REPORT / "TEST_VALIDATION.json").write_text(json.dumps(tests))
    (module.c.REPORT / "PUBLIC_REPLAY_VALIDATION.json").write_text(json.dumps(replay))
    if mutation is None:
        assert module._final_proofs()["replayed_public_files"] == 21
    else:
        with pytest.raises(ValueError):
            module._final_proofs()
