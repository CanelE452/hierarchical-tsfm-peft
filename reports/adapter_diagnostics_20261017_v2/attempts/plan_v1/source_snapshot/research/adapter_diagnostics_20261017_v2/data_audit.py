"""Bounded data and masked-ridge audit. Default CLI performs no asset IO."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import zipfile

CAMPAIGN = "informs_broad_validation_20261015_v1"
NAMESPACE = "adapter_diagnostics_20261017_v2"


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _prepared_prefix(name, cache, contract, primitive):
    import numpy as np
    receipt_path = cache / "data" / (name + "_receipt.json")
    receipt = read(receipt_path)
    relative = Path(receipt["cache_reference"])
    path = (cache / relative).resolve()
    if not path.is_relative_to(cache.resolve()):
        raise ValueError("Prepared asset escapes cache root")
    if (sha(path) != receipt["cache_sha256"] or path.stat().st_size != receipt["cache_bytes"]
            or receipt["shape"] != contract["shape"]):
        raise ValueError("Prepared asset no longer matches receipt")
    rows = contract["primary_split_ranges"]["V_select"][1]
    values, evidence = {}, []
    with zipfile.ZipFile(path) as archive:
        for key in ("physical", "observed", "times", "columns"):
            value, access = primitive.read_member(archive, key + ".npy", None if key == "columns" else rows)
            values[key] = value
            evidence.append(access)
    if (list(values["physical"].shape) != [rows, contract["shape"][1]]
            or values["physical"].shape != values["observed"].shape
            or values["observed"].dtype != np.dtype(bool)
            or values["columns"].tolist() != contract["columns"]):
        raise ValueError("Prepared prefix axes, mask or channel order changed")
    values["physical"][~values["observed"]] = np.nan
    task = dict(values, name=name, columns=values["columns"].tolist(), L=contract["L"], H=contract["H"],
                data_sha256=contract["data_sha256"], split_ranges=contract["primary_split_ranges"])
    return task, dict(source_reference=relative.as_posix(), source_sha256=sha(path),
                      receipt_sha256=sha(receipt_path), materialized_rows=rows, members=evidence,
                      new_T_numeric_values_parsed=0,
                      source_blob_IO="Opaque compressed file hashing; only pre-T members decoded numerically")


def _dataset_checks(task, contract, data, ridge):
    import numpy as np
    x, mask = task["physical"], task["observed"]
    if not np.isfinite(x[mask]).all():
        raise ValueError("Observed nonfinite data")
    if not np.all(np.diff(task["times"]) == contract["interval_seconds"] * 10**9):
        raise ValueError("Timestamp interval mismatch")
    a50, a60 = contract["primary_split_ranges"]["A_fit"][1], contract["primary_split_ranges"]["A_stop"][1]
    mean, std, count = data.fit_statistics(task, a50)
    stats = dict(mean=data.array_sha256(mean), std=data.array_sha256(std), count=data.array_sha256(count))
    if stats != contract["A_fit_statistics_fingerprints"]:
        raise ValueError("A_fit scaler fingerprint changed")
    origins, origin_hashes = {}, {}
    for split in ("A_fit", "A_stop", "M_cal", "V_select"):
        bounds = contract["primary_split_ranges"][split]
        points = data.origins(bounds, task["L"], task["H"], 1 if split == "A_fit" else task["H"])
        if not (np.all(points >= max(bounds[0], task["L"])) and np.all(points + task["H"] <= bounds[1])):
            raise ValueError("Target crosses split boundary")
        origins[split], origin_hashes[split] = len(points), data.array_sha256(points)
        if len(points) != contract["legal_origin_counts_primary"][split]:
            raise ValueError("Legal origin count differs")
    mean60, std60, _ = data.fit_statistics(task, a60)
    origin = int(data.origins(contract["primary_split_ranges"]["V_select"], task["L"], task["H"], task["H"])[0])
    before = data.history_window(task, origin, mean60, std60)[0]
    changed = dict(task, physical=x.copy())
    changed["physical"][origin:] = np.float32(1e6)
    np.testing.assert_array_equal(before, data.history_window(changed, origin, mean60, std60)[0])
    changed["physical"][a50:] = np.float32(1e6)
    for expected, actual in zip((mean, std, count), data.fit_statistics(changed, a50)):
        np.testing.assert_array_equal(expected, actual)
    hashes = {key: data.array_sha256(task[key]) for key in ("physical", "observed", "times")}
    hashes["columns"] = hashlib.sha256(json.dumps(task["columns"], ensure_ascii=False).encode()).hexdigest()
    return dict(status="PASS", prefix_rows=len(x), channels=x.shape[1],
                actual_observed_zero_count_preT=int(np.count_nonzero(mask & (x == 0))),
                observed_count_preT=int(mask.sum()), missing_count_preT=int((~mask).sum()),
                A_fit_min_channel_observation_count=int(count.min()),
                constant_channels_A_fit=int(np.count_nonzero(np.nanmax(np.where(mask[:a50], x[:a50], np.nan), axis=0)
                                                            == np.nanmin(np.where(mask[:a50], x[:a50], np.nan), axis=0))),
                A_fit_statistics_fingerprints=stats, A60_scaler_sha256=ridge.scaler_fingerprint(mean60, std60),
                origins=origins, origin_hashes=origin_hashes, prefix_array_sha256=hashes,
                first_materialized_timestamp_ns=int(task["times"][0]), last_materialized_timestamp_ns=int(task["times"][-1]),
                future_mutation_history_unchanged=True, A_outside_mutation_statistics_unchanged=True,
                inherited_full_task_data_sha256=contract["data_sha256"], inherited_identity_recomputed=False)


def _ridge_reuse(name, task, contract, cache, public, data, ridge):
    import numpy as np
    run_id = "primary_" + name + "_E1_RAW_RIDGE_k0_s92601"
    artifact_path = cache / "ridge" / run_id / "artifact.npz"
    replay_path = public / "runs" / run_id / "RIDGE_REPLAY_RECEIPT.json"
    if not artifact_path.is_file() or not replay_path.is_file():
        return dict(status="BLOCKED_MISSING_ASSET", reason="Pinned ridge artifact or public replay receipt absent")
    replay = read(replay_path)
    if replay["status"] != "PASS" or sha(artifact_path) != replay["artifact_sha256"]:
        raise ValueError("Historical ridge replay artifact identity changed")
    artifact = ridge.load_artifact(artifact_path)
    source_sha = ridge.fingerprint({"ridge_baseline.py": sha(public / "ridge_baseline.py"),
                                    "data_contract.py": sha(public / "data_contract.py")})
    if str(artifact["source_sha"]) != source_sha or str(artifact["data_sha"]) != contract["data_sha256"]:
        raise ValueError("Historical ridge code or data identity mismatch")
    end = contract["primary_split_ranges"]["A_stop"][1]
    mean, std, _ = data.fit_statistics(task, end)
    np.testing.assert_array_equal(mean, artifact["mean"])
    np.testing.assert_array_equal(std, artifact["std"])
    alpha = float(artifact["alpha"])
    if alpha not in ridge.ALPHAS or int(artifact["actual_fit_end"]) != end:
        raise ValueError("Historical ridge alpha or refit boundary mismatch")
    model_sha = ridge.fingerprint(dict(coefficients=data.array_sha256(artifact["coefficients"]),
        scaler_sha=ridge.scaler_fingerprint(mean, std), feature_sha=str(artifact["feature_sha"]), family="RAW_RIDGE"))
    if model_sha != str(artifact["model_sha"]):
        raise ValueError("Ridge model fingerprint changed")
    return dict(status="REUSED_VERIFIED", run_id=run_id, artifact_sha256=sha(artifact_path),
                replay_receipt_reference=replay_path.relative_to(public).as_posix(), replay_receipt_sha256=sha(replay_path),
                source_sha256=source_sha, data_sha256=contract["data_sha256"], model_sha256=model_sha,
                scaler_sha256=ridge.scaler_fingerprint(mean, std), selected_alpha=alpha,
                parameters=int(artifact["coefficients"].size),
                historical_KKT_max_abs_residual=replay["representative_channel_KKT_max_abs_residual"],
                KKT_evidence="Historical verified receipt reused; no new real-data refit or KKT claimed",
                selection_grid_boundary="Selected .01 is lower grid boundary; no global optimum claim")


def synthetic_checks(data, ridge):
    import numpy as np
    rng = np.random.default_rng(92601)
    physical = rng.normal(size=(220, 3)).astype(np.float32).cumsum(0)
    mask = np.ones_like(physical, bool)
    mask[::4, 0] = False
    mask[35:105, 1] = False
    physical[:, 2] = 0
    physical[~mask] = np.nan
    task = dict(physical=physical, observed=mask, L=64, H=4)
    mean, std, count = data.fit_statistics(task, 110)
    assert std[2] == 1 and count[2] == 110
    empty = dict(task, observed=mask.copy())
    empty["observed"][:110, 0] = False
    try:
        data.fit_statistics(empty, 110)
    except ValueError:
        pass
    else:
        raise AssertionError("Zero-observation fit channel must be rejected")
    train = data.origins((0, 110), 64, 4, 1)
    prepared = ridge.PrefixFeatures(task, mean, std, 132)
    coefficients, _, fallback = ridge.fit_coefficients(prepared, train, [.1])
    maximum_coefficient_difference, maximum_residual = 0., 0.
    penalty = np.eye(33)
    penalty[-1, -1] = 0
    for channel in range(3):
        design = np.concatenate([ridge.temporal_features(data.history_window(task, int(o), mean, std)[0][None])[:, channel]
                                 for o in train])
        targets = np.stack([data.target_window(task, int(o), mean, std)[0][:, channel] for o in train]).astype(np.float64)
        target_mask = np.stack([mask[o:o+4, channel] for o in train])
        anchors = np.array([data.history_window(task, int(o), mean, std)[0][-1, channel] for o in train])
        targets -= anchors[:, None]
        total = int(target_mask.sum())
        for h in range(4):
            selected = target_mask[:, h]
            a = np.concatenate([design[selected] / np.sqrt(total), np.sqrt(.1) * penalty])
            b = np.concatenate([targets[selected, h] / np.sqrt(total), np.zeros(33)])
            independent = np.linalg.lstsq(a, b, rcond=None)[0]
            actual = coefficients[0, channel, :, h]
            maximum_coefficient_difference = max(maximum_coefficient_difference, float(np.max(np.abs(independent - actual))))
            residual = design[selected].T @ (design[selected] @ actual - targets[selected, h]) / total + .1 * penalty @ actual
            maximum_residual = max(maximum_residual, float(np.max(np.abs(residual))))
    assert maximum_coefficient_difference < 1e-9 and maximum_residual < 1e-10
    sample = np.stack([data.history_window(task, int(o), mean, std)[0] for o in [64, 90, 110]])
    with tempfile.TemporaryDirectory(prefix="adapter_diagnostic_ridge_") as temporary:
        path = Path(temporary) / "synthetic.npz"
        np.savez(path, coefficients=coefficients[0])
        loaded = ridge.load_artifact(path)
        np.testing.assert_array_equal(ridge.predict_normalized(loaded, sample),
                                      ridge.predict_normalized(dict(coefficients=coefficients[0]), sample))
    reset = dict(physical=np.array([[99, 99], [np.nan, 0], [np.nan, 4], [3, np.nan], [5, 6]], np.float32), L=3, H=1)
    reset["observed"] = np.isfinite(reset["physical"])
    history, _ = data.history_window(reset, 4, np.array([1, 2]), np.ones(2))
    np.testing.assert_array_equal(history, [[0, -2], [0, 2], [2, 2]])
    source = "@data\nT1:2012-01-01 00-00-01:0,1,PROTECTED_POISON,PROTECTED_POISON\n"
    parsed, _, metadata = data.parse_electricity_prefix(io.StringIO(source), cutoff="2012-01-01T02:00:00", expected_channels=1)
    np.testing.assert_array_equal(parsed[:, 0], [0, 1])
    phi, weight = rng.normal(size=(5, 4, 7)), rng.normal(size=(7, 3))
    targets, observed = rng.normal(size=(5, 4, 3)), rng.random((5, 4, 3)) > .2
    shared = np.einsum("nhp,pc->nhc", phi, weight)
    replicated = np.einsum("nhp,hpc->nhc", phi, np.repeat(weight[None], 4, axis=0))
    np.testing.assert_array_equal(shared, replicated)
    def objective(pred, weights):
        return float(np.mean(np.sum(np.where(observed, (pred-targets)**2, 0), axis=(0, 1)) / observed.sum(axis=(0, 1)))
                     + .1 * np.mean(np.sum(weights**2, axis=(1, 2))))
    shared_loss = objective(shared, weight[None])
    replicated_loss = objective(replicated, np.repeat(weight[None], 4, axis=0))
    assert abs(shared_loss - replicated_loss) < 1e-12
    return dict(status="PASS", masked_augmented_lstsq_max_abs_coefficient_difference=maximum_coefficient_difference,
                masked_normal_equation_max_abs_residual=maximum_residual, solver_fallbacks=fallback,
                save_load_predictions_exact=True, window_reset_and_observed_zero=True,
                zero_observation_fit_channel_rejected=True,
                electricity_poison_tail_numeric_values_parsed=0,
                shared_and_replicated_Dh_prediction_max_abs_difference=float(np.max(np.abs(shared-replicated))),
                shared_and_replicated_Dh_objective_abs_difference=abs(shared_loss-replicated_loss),
                shared_Dh_penalty_convention="alpha times mean_h ||D_h||^2; shared alpha ||D||^2",
                ridge_objective_convention="SSE_c / N_c across all origin,h; intercept unpenalized; not sklearn unscaled alpha",
                sklearn_equivalent_alpha="alpha_sklearn = alpha_repository * N_c for each channel")


def run_audit(asset_root, cache_root=None, report_root=None):
    started = time.perf_counter()
    repo = Path(__file__).resolve().parents[2]
    public = repo / "research" / CAMPAIGN
    asset = Path(asset_root).resolve()
    cache = Path(cache_root).resolve() if cache_root else asset / ".cache" / CAMPAIGN
    sys.path.insert(0, str(public))
    data = module(public / "data_contract.py", "adapter_audit_data_contract")
    previous = sys.modules.get("data_contract")
    sys.modules["data_contract"] = data
    try:
        ridge = module(public / "ridge_baseline.py", "adapter_audit_ridge")
    finally:
        if previous is None:
            sys.modules.pop("data_contract", None)
        else:
            sys.modules["data_contract"] = previous
    primitive_path = public / "priority_b2_validation_20261016_v1" / "safe_jena_prefix.py"
    primitive = module(primitive_path, "adapter_audit_primitive")
    contracts = read(public / "DATA_CONTRACT.json")["assets"]
    bindings = {str(path.relative_to(repo).as_posix()): sha(path) for path in
                [Path(__file__), public / "DATA_CONTRACT.json", public / "ASSET_SOURCES.json",
                 public / "data_contract.py", public / "ridge_baseline.py", public / "RIDGE_PROTOCOL_KO.md", primitive_path]}
    output = dict(schema="ADAPTER_DATA_AUDIT_v1", checked_utc=datetime.now(timezone.utc).isoformat(),
                  source_bindings=bindings, datasets={}, synthetic_G1_G4=synthetic_checks(data, ridge),
                  protected_T_numeric_values_parsed=0, new_raw_sources_parsed=0, F_calls=0, model_loads=0,
                  scientific_scope="Repeated development V; protected T remains locked; no independent generalization claim")
    model_sources = repo / "reports" / NAMESPACE / "BACKBONE_SOURCES.json"
    if model_sources.is_file():
        sources = read(model_sources)
        output["model_card_overlap_review"] = dict(source_reference=model_sources.relative_to(repo).as_posix(),
            source_sha256=sha(model_sources),
            models={key: dict(revision=value["revision"], pretraining=value.get("pretraining", {}))
                    for key, value in sources["models"].items()},
            Jena_PeMS_Electricity_exact_series_and_time_overlap="UNKNOWN",
            independent_pretraining_holdout_claim_allowed=False)
    for name in ("jena", "pems08", "electricity"):
        contract = contracts[name]
        row = dict(status="PLANNED", inherited_source_metadata=contract["metadata"],
                   full_permitted_shape=contract["shape"], L=contract["L"], H=contract["H"],
                   primary_split_ranges=contract["primary_split_ranges"],
                   pretraining_overlap="UNKNOWN", pretraining_note="Dataset lineage alone cannot prove model training overlap")
        try:
            task, access = (primitive.load_jena_prefix(asset, contract) if name == "jena"
                            else _prepared_prefix(name, cache, contract, primitive))
            row.update(status="COMPLETE", access=access,
                       checks=_dataset_checks(task, contract, data, ridge),
                       ridge=_ridge_reuse(name, task, contract, cache, public, data, ridge))
        except FileNotFoundError:
            row.update(status="BLOCKED_MISSING_ASSET", reason="Manifest-listed source, contract or cache receipt absent")
        except (ValueError, AssertionError) as exc:
            row.update(status="FAILED", reason=str(exc))
        output["datasets"][name] = row
    m5_path = cache / "m5_access_reaudit_20261016_v1" / "RESULT.json"
    m5 = dict(status="BLOCKED_ACCESS_OR_LICENSE", approved_sales_calendar_prices_assets_verified=False,
              new_authentication_requests=0, new_downloads=0, new_broad_local_searches=0,
              reason="No approved M5 sales/calendar/prices in the registered inventory; no mirror or authentication retry")
    if m5_path.is_file():
        prior = read(m5_path)
        m5.update(prior_receipt_sha256=sha(m5_path), prior_http_status=prior["http_status"],
                  prior_local_raw_candidates=prior["local_m5_raw_candidates_found"],
                  prior_status=prior["status"], prior_record_created_utc=prior["record_created_utc"])
    output["datasets"]["m5"] = m5
    output["wall_seconds"] = time.perf_counter() - started
    output["status"] = "PASS_WITH_M5_BLOCK" if all(output["datasets"][name]["status"] == "COMPLETE" for name in ("jena", "pems08", "electricity")) else "PARTIAL_OR_FAILED"
    if report_root is not None:
        destination = Path(report_root)
        atomic(destination / "DATA_AUDIT.json", json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
        atomic(destination / "DATA_AUDIT_KO.md", render_report(output))
    return output


def render_report(value):
    lines = ["# 데이터·Ridge 감사", "", "[확인] 상태: `" + value["status"] + "`. 실제 실행 시각: " + value["checked_utc"] + ".",
             "기존 승인 자산에서 V 끝까지만 수치 변환했다. 보호 T·Electricity 원자료 tail 수치 접근, 새 다운로드·모델 로드·F 호출은 0회다.",
             "전체 blob SHA는 opaque bytes 검증이며 수치 변환 경계와 구분한다. 전체 자료 hash는 기존 계약에서 상속했으며 T를 다시 읽어 재계산하지 않았다.", "",
             "모든 성능 해석은 반복 사용한 개발 V에 한정된다. 기존 T의 46조건 평가 이력을 지우지 않으며 새 독립 test라고 부르지 않는다.", ""]
    for name in ("jena", "pems08", "electricity"):
        row = value["datasets"][name]
        lines += ["## " + name, "", "[확인] `" + row["status"] + "`, 계약 shape " + str(row["full_permitted_shape"]) + ", L/H=" + str(row["L"]) + "/" + str(row["H"]) + "."]
        if "checks" in row:
            check = row["checks"]
            lines += ["실제 materialized pre-T 행 " + str(check["prefix_rows"]) + ", 관측 0 " + str(check["actual_observed_zero_count_preT"]) + ", 결측 " + str(check["missing_count_preT"]) + ".",
                      "A_fit mean/std/count와 채널 순서·간격·평가 origin count가 기존 계약과 일치했다. 미래값 변경에 대한 history 불변·A 밖 변경에 대한 scaler 불변을 확인했다.",
                      "A60 scaler SHA256: `" + check["A60_scaler_sha256"] + "`."]
        else:
            lines += [row.get("reason", "추가 확인 필요")]
        lines += ["Ridge: `" + row.get("ridge", {}).get("status", "NOT_RUN_DEPENDENCY") + "`.", ""]
    lines += ["Jena는 legacy normalized FP32를 기존 mean/std로 역변환한 계보다. bit-exact 원자료도, 역변환 잔차가 성능에 영향을 준다는 결론도 아니다. 컬럼별 단위는 이름을 유지하고 10분 실제 timestamp를 사용한다.",
              "PeMS08은 저자 배포 [17856,170,3]의 feature0 flow·원 센서 축 순서다. 5분 경과 인덱스이며 1970년 실제 관측이 아니다. 외부 센서 ID·정확한 물리 counting 단위·상류 보간의 미래 접근은 UNKNOWN이다.",
              "Electricity는 기존 승인된 321 hourly Lai panel이다. 2012-01-01 00:00:01을 임의 보정하지 않았고 보호 cutoff 2014-10-07 이전 prefix 계약을 유지했다. 이번 검사는 그 안에서도 V까지만 읽었다. 집계·단위변환·상류 보간은 재구축하지 않았다.", "",
              "## Ridge와 solver", "", "[확인] own-channel 512과거 → 32개 비중첩 block 평균−마지막 값 + intercept, 33 features, horizon별 계수다. alpha=(.01,.1,1,10,100), A_stop stride H, A60 scaler fresh refit 계약이다. 저장 alpha .01은 grid 하한이며 전역 최적값이 아니다.",
              "목적식은 채널별 전체 관측 target 수 N_c로 SSE를 나눈 뒤 alpha L2를 더한다. intercept는 비규제다. sklearn의 비정규화 SSE 관례에 대응하는 alpha는 채널별 alpha×N_c다.",
              "[확인] synthetic masked regression에서 독립 augmented least-squares와 기존 solver, 정상방정식 잔차, 저장·로드 예측을 확인했다. coefficient 최대 차이 `" + str(value["synthetic_G1_G4"]["masked_augmented_lstsq_max_abs_coefficient_difference"]) + "`, KKT 최대 잔차 `" + str(value["synthetic_G1_G4"]["masked_normal_equation_max_abs_residual"]) + "`.",
              "공유 D를 D_h에 복제하면 같은 예측·목적값임을 확인했다. penalty는 shared alpha||D||²와 D_h alpha mean_h||D_h||²로 맞췄다. 이 합성 검사는 아직 실제 B03 decoder 구현의 검증을 대신하지 않는다.",
              "기존 Ridge artifact/source/data/scaler/model hash와 PASS replay 영수증이 일치하는 경우 REUSED_VERIFIED다. 실제 데이터의 KKT는 기존 receipt 수치를 재사용했고 새 대규모 fit/KKT를 주장하지 않는다.", "",
              "## M5·상류 한계", "", "[확인] M5는 `BLOCKED_ACCESS_OR_LICENSE`다. 기존 공식 재감사는 인증401과 로컬 원자산0을 기록했다. 이번에는 인증 재시도·mirror·새 다운로드·범위 확대 검색을 하지 않았다. 승인된 sales/calendar/prices가 실제 제공되어야 AVAILABLE로 바꿀 수 있다.",
              "[미검증] 정확한 시계열·시간구간의 pretraining overlap은 UNKNOWN이다. [공식 model card 대조](BACKBONE_SOURCES.json)에서 Moirai2는 GiftEvalPretrain·GiftEval train context·Chronos corpus 일부·합성/내부 자료를, Chronos2는 chronos_datasets·GiftEvalPretrain을 기술하지만 현재 Jena/PeMS/Electricity 구간의 명시적 포함·제외까지 입증하지 못했다. NO_OVERLAP_EVIDENCE도 무누출 보증이 아니다.", "",
              "공식 근거: [Ridge 정의](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), [Electricity 배포판](https://zenodo.org/records/3889830), [저자 PeMS feature0 전처리](https://github.com/guoshnBJTU/ASTGCN-r-pytorch/blob/2e7a4faa2a6f89da8d1cb37acb7e267c9bc87296/prepareData.py).", "",
              "재현: `python -m research.adapter_diagnostics_20261017_v2.data_audit --execute --asset-root ASSET_ROOT --report-root REPORT_ROOT`. 인수 없는 기본 실행은 source-only no-op이다. 전체 hashes·접근 영수증·검증 수치는 [DATA_AUDIT.json](DATA_AUDIT.json)에 있다.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--report-root", type=Path)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="SOURCE_ONLY_NOOP", asset_reads=0, model_loads=0)))
        return 0
    if args.asset_root is None or args.report_root is None:
        parser.error("Explicit --asset-root and --report-root required")
    value = run_audit(args.asset_root, args.cache_root, args.report_root)
    print(json.dumps(dict(status=value["status"], synthetic=value["synthetic_G1_G4"]["status"],
                          datasets={key: row["status"] for key, row in value["datasets"].items()})))
    return 0 if value["status"] == "PASS_WITH_M5_BLOCK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
