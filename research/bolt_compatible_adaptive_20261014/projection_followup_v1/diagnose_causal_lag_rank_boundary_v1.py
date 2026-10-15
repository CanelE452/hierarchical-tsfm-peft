"""Compare rank algorithms on saved joint matrices; no targets or forecasts loaded."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import continuation_runtime_v1 as rt


def binding(path):
    return {"path": str(path), "sha256": rt.sha(path)}


def execute():
    import numpy as np
    from threadpoolctl import threadpool_limits
    result_path = HERE / "runs/actual_causal_lag_objective_v1/RESULT.json"
    audit_path = HERE / "audits/CAUSAL_LAG_OBJECTIVE_AUDIT_v1.json"
    output = HERE / "audits/CAUSAL_LAG_RANK_BOUNDARY_DIAGNOSTIC_ROOT_v1.json"
    assert not output.exists()
    assert rt.sha(result_path) == "4168cd05962e479130a722919310f4c192c1bab03d9b78cb7983754f2e502c17"
    assert rt.sha(audit_path) == "7d4999aed252e343dba18c2f009bb3dc20498af5ab2b90c6ac01c0d8c6c8df7a"
    result = rt.read(result_path)
    meter = rt.Meter("causal_lag_saved_rank_diagnostic_root_v1", gpu=False)
    report = {"status": "RUNNING", "producer": binding(result_path), "failed_audit": binding(audit_path),
              "source": binding(Path(__file__).resolve()), "cases": [], "target_or_forecast_arrays_deserialized": False,
              "new_D_fit_or_model_forward": False, "registered_cutoff_changed": False,
              "scope": "Only saved joint_A and complete_horizons. LS RHS is zero with the same number of columns as mask-group origins. Dimension-aware cutoff is a separate sensitivity diagnostic, not a replacement of original results."}
    failure = None
    try:
        meter.protected_check()
        with threadpool_limits(limits=2):
            spaces = result["datasets"]["beijing_aotizhongxin"]["spaces"]
            for space, parent in spaces.items():
                for condition in ("L7_RECON", "L7_FORECAST"):
                    for split, entry in parent["conditions"][condition]["splits"].items():
                        artifact = entry["artifact"]; path = Path(artifact["path"])
                        assert rt.sha(path) == artifact["sha256"]
                        with np.load(path, allow_pickle=False) as stored:
                            arrays = {name: stored[name] for name in ("joint_A_FP64", "complete_horizons")}
                        for name, value in arrays.items():
                            expected = artifact["arrays"][name]
                            assert str(value.dtype) == expected["dtype"] and list(value.shape) == expected["shape"]
                            assert hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() == expected["sha256"]
                        a, complete = arrays["joint_A_FP64"], arrays["complete_horizons"]
                        assert a.dtype == np.float64 and complete.dtype == np.bool_ and a.shape == (48 * 11, 48 * 2)
                        groups = {}
                        for i, mask in enumerate(complete): groups.setdefault(mask.tobytes(), []).append(i)
                        case = {"dataset": "beijing_aotizhongxin", "space": space, "condition": condition,
                                "split": split, "artifact": artifact, "groups": []}
                        for indices in groups.values():
                            rows = np.repeat(complete[indices[0]], 11)
                            if not rows.any(): continue
                            matrix = a[rows]
                            _, s, _ = np.linalg.svd(matrix, full_matrices=False)
                            _, _, rank, ls_s = np.linalg.lstsq(matrix, np.zeros((len(matrix), len(indices))), rcond=1e-15)
                            rank_at = lambda spectrum, cut: int(np.count_nonzero(spectrum > spectrum[0] * cut)) if spectrum.size and spectrum[0] > 0 else 0
                            dimension_cut = max(matrix.shape) * np.finfo(np.float64).eps
                            case["groups"].append({"windows_zero_based": indices, "matrix_shape": list(matrix.shape),
                                "exact_zero_columns": int(np.all(matrix == 0, axis=0).sum()), "registered_cutoff": 1e-15,
                                "SVD_rank": rank_at(s, 1e-15), "lstsq_returned_rank": int(rank),
                                "lstsq_spectrum_rank": rank_at(ls_s, 1e-15), "SVD_singularvalues": s.tolist(),
                                "lstsq_singularvalues": ls_s.tolist(), "dimension_aware_sensitivity_cutoff": dimension_cut,
                                "SVD_dimension_aware_rank": rank_at(s, dimension_cut),
                                "lstsq_dimension_aware_rank": rank_at(ls_s, dimension_cut)})
                        report["cases"].append(case)
                        assert rt.sha(path) == artifact["sha256"]
                        meter.check({"space": space, "condition": condition, "split": split})
        groups = [g for case in report["cases"] for g in case["groups"]]
        report.update(status="COMPLETE_SAVED_MATRIX_RANK_SENSITIVITY_DIAGNOSTIC", cases_checked=len(report["cases"]),
                      nonempty_mask_groups_checked=len(groups),
                      registered_rank_disagreements=sum(g["SVD_rank"] != g["lstsq_returned_rank"] for g in groups),
                      dimension_aware_rank_disagreements=sum(g["SVD_dimension_aware_rank"] != g["lstsq_dimension_aware_rank"] for g in groups))
        meter.protected_check()
    except Exception as error:
        failure = {"type": type(error).__name__, "message": str(error)}
        report.update(status="FAILED_PRESERVED", failure=failure)
    finally:
        report["resource_terminal"] = meter.close(report["status"], failure)
        rt.write_new(output, report)
    print(json.dumps({key: report.get(key) for key in ("status", "cases_checked", "nonempty_mask_groups_checked", "registered_rank_disagreements", "dimension_aware_rank_disagreements", "failure")}))
    return 0 if failure is None else 2


if __name__ == "__main__":
    raise SystemExit(execute())
