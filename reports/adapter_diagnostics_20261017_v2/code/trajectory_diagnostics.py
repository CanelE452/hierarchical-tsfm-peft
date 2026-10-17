"""Read bound trajectory metadata and fixed decoder features without loading a model."""
from collections import OrderedDict
from contextlib import ExitStack
import hashlib
import io
import json
from pathlib import Path
import pickle
import pickletools
import zipfile

import numpy as np


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            value.update(chunk)
    return value.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class _Ignored:
    def __new__(cls, *args):
        return super().__new__(cls)

    def __init__(self, *args):
        pass

    def __setstate__(self, state):
        pass


class _Tensor:
    def __init__(self, storage, offset, shape, stride, *unused):
        self.storage, self.offset, self.shape, self.stride = storage, offset, tuple(shape), tuple(stride)


class _Reader(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _Tensor
        if module == "torch" and name in ("FloatStorage", "ByteStorage"):
            return {"FloatStorage": "<f4", "ByteStorage": "u1"}[name]
        if (module, name) in {("_codecs", "encode"), ("numpy", "dtype"), ("numpy", "ndarray"),
                              ("numpy._core.multiarray", "_reconstruct")}:
            return _Ignored
        raise ValueError("Unsupported trajectory metadata global: "+module+"."+name)

    def persistent_load(self, value):
        if not isinstance(value, tuple) or len(value) != 5 or value[0] != "storage" or value[1] not in ("<f4", "u1"):
            raise ValueError("Unsupported trajectory storage metadata")
        return dict(key=str(value[2]), dtype=value[1], count=int(value[4]))


def checkpoint_metadata(path):
    with zipfile.ZipFile(path) as archive:
        entries = [entry for entry in archive.infolist() if entry.filename.endswith("/data.pkl")]
        if len(entries) != 1 or entries[0].file_size > 2*1024*1024:
            raise ValueError("Unsupported trajectory pickle layout/size")
        raw = archive.read(entries[0])
        if any(op.name in ("EXT1", "EXT2", "EXT4") for op, _, _ in pickletools.genops(raw)):
            raise ValueError("Unsupported trajectory pickle extension")
        value = _Reader(io.BytesIO(raw)).load()
    if not isinstance(value, dict):
        raise ValueError("Unsupported trajectory checkpoint schema")
    return value


def tensor_array(path, descriptor):
    if not isinstance(descriptor, _Tensor):
        raise ValueError("Expected a recorded tensor descriptor")
    storage = descriptor.storage
    if storage["count"] > 100000:
        raise ValueError("Only small decoder/RNG storage reads are allowed")
    with zipfile.ZipFile(path) as archive:
        prefix = next(name.rsplit("/", 1)[0] for name in archive.namelist() if name.endswith("/data.pkl"))
        byteorder = archive.read(prefix+"/byteorder").decode() if prefix+"/byteorder" in archive.namelist() else "little"
        if byteorder != "little":
            raise ValueError("Unsupported checkpoint byte order")
        raw = archive.read(prefix+"/data/"+storage["key"])
    dtype = np.dtype(storage["dtype"])
    if len(raw) != storage["count"]*dtype.itemsize:
        raise ValueError("Tensor storage byte count mismatch")
    return np.ndarray(descriptor.shape, dtype=dtype, buffer=raw, offset=descriptor.offset*dtype.itemsize,
                      strides=tuple(value*dtype.itemsize for value in descriptor.stride)).copy()


def bound_path(ctx, binding):
    scheme, relative = binding["reference"].split("://", 1)
    if scheme not in ctx:
        raise ValueError("Missing bound trajectory root: "+scheme)
    root = Path(ctx[scheme]).resolve(); path = (root/relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or sha(path) != binding["sha256"]:
        raise ValueError("Trajectory private SHA binding rejected")
    return path


def sampler_record(checkpoint):
    state = checkpoint["sampler"]
    required = ("order", "position", "exposures", "epochs_completed", "rng", "origins_sha")
    if any(key not in state for key in required):
        raise ValueError("Incomplete actual sampler checkpoint state")
    order = np.asarray(state["order"], dtype=np.int64)
    if sorted(order.tolist()) != list(range(len(order))) or not 0 <= state["position"] < len(order):
        raise ValueError("Recorded sampler permutation/cursor invalid")
    if state["exposures"] != state["epochs_completed"]*len(order)+state["position"]:
        raise ValueError("Recorded sampler epoch/exposure arithmetic differs")
    return dict(sampler_sha256=fingerprint({key: state[key] for key in required}),
                order_sha256=hashlib.sha256(order.tobytes()).hexdigest(), rng_sha256=fingerprint(state["rng"]),
                origins_sha256=state["origins_sha"], position=state["position"], exposures=state["exposures"],
                epochs_completed=state["epochs_completed"], order_length=len(order))


def _array_stream(stream):
    version = np.lib.format.read_magic(stream)
    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, fortran, dtype = reader(stream)
    if fortran or dtype.hasobject:
        raise ValueError("Only C-order numeric feature cache members are allowed")
    return shape, dtype


def fixed_decoder_scores(path, weights, chunk_origins=128):
    from threadpoolctl import threadpool_limits
    weights = np.asarray(weights)
    totals = {mode: [np.zeros(weights.shape[-1]), np.zeros(weights.shape[-1])] for mode in ("float64", "native_float32")}
    counts = np.zeros(weights.shape[-1], np.int64)
    with threadpool_limits(limits=2, user_api="blas"), zipfile.ZipFile(path) as archive, ExitStack() as stack:
        streams = {key: stack.enter_context(archive.open(key+".npy")) for key in ("phi", "target", "mask", "anchor")}
        headers = {key: _array_stream(stream) for key, stream in streams.items()}
        shape = headers["phi"][0]
        if len(shape) != 3 or weights.shape != (shape[2], headers["target"][0][-1]):
            raise ValueError("Fixed decoder feature/output axes differ")
        if headers["mask"][0] != headers["target"][0] or headers["mask"][1] != np.dtype(bool):
            raise ValueError("Fixed decoder target/mask contract differs")
        for start in range(0, shape[0], chunk_origins):
            count = min(chunk_origins, shape[0]-start)
            values = {}
            for key, stream in streams.items():
                member_shape, dtype = headers[key]
                items = count*int(np.prod(member_shape[1:])); raw = stream.read(items*dtype.itemsize)
                if len(raw) != items*dtype.itemsize:
                    raise ValueError("Truncated fixed feature member")
                values[key] = np.frombuffer(raw, dtype=dtype).reshape((count, *member_shape[1:]))
            counts += values["mask"].sum((0, 1))
            for mode, dtype in (("float64", np.float64), ("native_float32", np.float32)):
                prediction = values["anchor"].astype(dtype)+values["phi"].astype(dtype)@weights.astype(dtype)
                error = np.where(values["mask"], prediction.astype(np.float64)-values["target"].astype(np.float64), 0.)
                if not np.isfinite(error).all():
                    raise ValueError("Nonfinite fixed decoder observed error")
                totals[mode][0] += np.sum(error*error, (0, 1), dtype=np.float64)
                totals[mode][1] += np.sum(np.abs(error), (0, 1), dtype=np.float64)
    active = counts > 0
    if not active.any():
        raise ValueError("No observed fixed decoder targets")
    return {mode: dict(MSE=float(np.mean(sse[active]/counts[active])), MAE=float(np.mean(sae[active]/counts[active])),
                       observed_count=int(counts.sum())) for mode, (sse, sae) in totals.items()}


def build_trajectory_diagnostics(ctx, data):
    from research.adapter_diagnostics_20261017_v2.core import validate_plan
    from research.adapter_diagnostics_20261017_v2.existing_jobs import _old_run
    report, repo = Path(ctx["report"]), Path(ctx["repo"])
    read = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    plan = data.get("plan") or read(report/"EXPERIMENT_PLAN.json")
    validate_plan(plan, repo)
    public_sources, private_bindings, records, fit_rows = {}, {}, [], []
    checked_b03 = 0
    def public(path):
        path = Path(path)
        public_sources[path.resolve().relative_to(repo.resolve()).as_posix()] = sha(path)
        return dict(reference="report://"+path.relative_to(report).as_posix(), sha256=sha(path))
    def private(binding):
        path = bound_path(ctx, binding)
        private_bindings[binding["reference"]] = binding
        return path
    for job in plan["jobs"]:
        if job["question_id"] not in ("A03", "B03") or (job["question_id"] == "A03" and job["split"] != "V_select"):
            continue
        receipt_path = report/"receipts"/(job["job_id"]+".json")
        if not receipt_path.is_file():
            continue
        receipt = read(receipt_path)
        if receipt.get("status") not in ("COMPLETE", "REUSED_VERIFIED"):
            continue
        if receipt.get("scientific_identity") != job["scientific_identity"]:
            raise ValueError("Trajectory receipt scientific identity differs")
        sources = [public(receipt_path)]
        for relative, expected in receipt.get("artifact_hashes", {}).items():
            path = (report/relative).resolve()
            if not path.is_relative_to(report.resolve()) or sha(path) != expected:
                raise ValueError("Trajectory public producer SHA rejected")
            sources.append(public(path))
        result = receipt["result"]
        if job["question_id"] == "A03":
            handler = read(report/"jobs"/(job["job_id"]+"_handler.json"))
            candidates = [item for item in handler["artifact_hashes"] if item["path"].endswith(".pt")
                          and item["sha256"] == result["checkpoint_sha256"]]
            if len(candidates) != 1:
                raise ValueError("A03 checkpoint binding is ambiguous")
            binding = dict(reference=candidates[0]["path"], sha256=candidates[0]["sha256"])
            path = private(binding); checkpoint = checkpoint_metadata(path)
            sampler = sampler_record(checkpoint)
            if sampler["exposures"] != result["exposure"] or checkpoint["updates"] != result["updates"]:
                raise ValueError("A03 saved sampler progress differs from endpoint")
            rng = tensor_array(path, checkpoint["torch_rng"])
            records.append(dict(job_id=job["job_id"], family=job["family"], seed=job["seed"], epoch=result["epoch"],
                checkpoint=binding, public_references=sources, torch_rng_sha256=hashlib.sha256(rng.tobytes()).hexdigest(), **sampler))
        else:
            seed = job["seed"]
            caches = result["feature_caches"]
            cache_by_label = {}
            for cache in caches:
                binding = cache["private_artifacts"][0]
                path = private(binding)
                if cache["sha256"] != binding["sha256"]:
                    raise ValueError("B03 cache content hash differs")
                label = Path(binding["reference"].split("://", 1)[1]).name
                for suffix in ("A60_fit", "A60_V_select"):
                    if label.startswith(str(seed)+"_"+suffix+"_"):
                        cache_by_label[suffix] = (path, cache)
            if set(cache_by_label) != {"A60_fit", "A60_V_select"}:
                raise ValueError("B03 A60 fit/V feature pair missing")
            fit_path, fit_cache = cache_by_label["A60_fit"]; val_path, val_cache = cache_by_label["A60_V_select"]
            for key in ("PE_checkpoint_sha256", "scaler_sha", "weights_sha", "point", "feature", "routing"):
                if fit_cache["key"][key] != val_cache["key"][key]:
                    raise ValueError("B03 fit/V cache identity differs: "+key)
            relative = Path(".cache/informs_broad_validation_20261015_v1")/_old_run(seed, "B2")/"selected_bundle.pt"
            binding = dict(reference="asset://"+relative.as_posix(), sha256=fit_cache["key"]["PE_checkpoint_sha256"])
            bundle_path = private(binding); bundle = checkpoint_metadata(bundle_path)
            if bundle["refit_summary"]["contract"]["scaler_sha"] != fit_cache["key"]["scaler_sha"]:
                raise ValueError("B03 selected bundle scaler differs from features")
            weights = tensor_array(bundle_path, bundle["model"]["D"]).reshape(42, 21)
            fit_score, val_score = fixed_decoder_scores(fit_path, weights), fixed_decoder_scores(val_path, weights)
            original = next(row for row in result["rows"] if row["method"] == "B2_ORIGINAL_D")
            native_difference = max(abs(val_score["native_float32"][key]-original[key]) for key in ("MSE", "MAE"))
            fp64_difference = max(abs(val_score["float64"][key]-original[key]) for key in ("MSE", "MAE"))
            if native_difference > 1e-12 or fp64_difference > 1e-6:
                raise ValueError("B03 original decoder V cross-parity failed")
            for index, row in enumerate(result["rows"]):
                entry = dict(run_id=job["job_id"]+"_"+str(index), method=row["method"], seed=seed,
                             stage=row["stage"], evaluation=row["evaluation"], lambda_value=row.get("lambda_value"),
                             fit_MSE=row.get("fit_MSE"), fit_MAE=row.get("fit_MAE"), MSE=row["MSE"], MAE=row["MAE"],
                             public_references=sources)
                if row["method"] == "B2_ORIGINAL_D":
                    entry.update(fit_MSE=fit_score["float64"]["MSE"], fit_MAE=fit_score["float64"]["MAE"],
                        independent_FP64_V=val_score["float64"], native_float32_V=val_score["native_float32"],
                        native_V_max_abs_difference=native_difference, FP64_V_max_abs_difference=fp64_difference,
                        native_tolerance=1e-12, FP32_core_vs_FP64_core_tolerance=1e-6,
                        score_scope="Original fixed D/phi/anchor; FP64 core and accumulation, observed channel macro; no fit",
                        private_artifacts=[binding, fit_cache["private_artifacts"][0], val_cache["private_artifacts"][0]])
                fit_rows.append(entry)
            checked_b03 += 1
    groups = {}
    for record in records:
        groups.setdefault((record["seed"], record["epoch"]), []).append(record)
    pairs = []
    for (seed, epoch), group in sorted(groups.items()):
        matched = len({row["sampler_sha256"] for row in group}) == 1 and {row["family"] for row in group} == {"B0", "B2", "B2_CAP_POINTWISE"}
        pairs.append(dict(seed=seed, epoch=epoch, status="PASS_ACTUAL_SAVED_SAMPLER_STATE" if matched else "UNVERIFIED_OR_DIFFERENT",
            records=group, historical_consumed_order_trace_verified=False,
            scope="Actual saved current permutation, cursor, epoch, sampler RNG and origins hash match; historical intervening permutations are not logged; torch RNG is recorded separately and may differ across architectures"))
    expected_a03 = sum(job["question_id"] == "A03" and job["split"] == "V_select" for job in plan["jobs"])
    expected_b03 = sum(job["question_id"] == "B03" for job in plan["jobs"])
    complete = pairs and len(records) == expected_a03 and checked_b03 == expected_b03 and all(p["status"].startswith("PASS_") for p in pairs)
    return dict(schema="TRAJECTORY_DIAGNOSTICS_v1", status="PASS_RECORDED_SCOPE" if complete else "PARTIAL",
        expected_a03_snapshots=expected_a03, checked_a03_snapshots=len(records), expected_b03_jobs=expected_b03, checked_b03_jobs=checked_b03,
        a03_sampler_pairs=pairs, b03_fit_rows=fit_rows, public_sources=public_sources,
        private_artifacts=list(private_bindings.values()), model_calls=0, new_training=False,
        limitations=["Snapshot state equality does not establish an independently logged complete historical consumed-order trace."])
