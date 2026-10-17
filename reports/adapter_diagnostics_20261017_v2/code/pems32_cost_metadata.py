"""Derive PeMS32 CPU route metadata from sealed sources and existing bound artifacts."""
import importlib.util
import math
from pathlib import Path

from research.adapter_diagnostics_20261017_v2.core import read, sha, validate_plan


BROAD = "research/informs_broad_validation_20261015_v1"
REPORT = "reports/adapter_diagnostics_20261017_v2"
SUBSET_PLAN_SHA256 = "004ed9908b5e013e837bfc03dbfb3facad9cdb901c4a3e9672ef46e8d7a13bfa"
SOURCE_PINS = {
    BROAD+"/deployment.py": "9ccf6a6e142ce49ab3cb1c1a0dc8d1594890f05c3a7dfea80fa217111c57deba",
    BROAD+"/gpu_runtime.py": "c633f1ebf4eb24ea3fa2b025ffa6cea4ec3697c8e9285d971174179560596158",
    BROAD+"/fit_base.py": "c59cc77ba48e3f061ffa8d2385442286ea989b3ab86c17ea352caeb6d2c1bf20",
    BROAD+"/networks.py": "787f9caf91f32a001f929f324c7bcb570b9270331f6f60f0f56872529f86dc54",
    "research/adapter_diagnostics_20261017_v2/existing_jobs.py": "6a8e7d4d940414ad71c2583581b681dd0a05f54e79293db1350d76a7b90d61d6",
    "research/adapter_diagnostics_20261017_v2/backbone_costs.py": "9395acf47e1ad5f1cb2e1993c306350f6c1450feaabef639eca2f943957a49e7",
    "reports/informs_b2_why_where_20261017/code/pems_cost.py": "8a7b35c6ea1ef25bc6875c283365383e52d4c50c3f40496be242e10fc18b39a6",
}


def _checkpoint_metadata(ctx, path):
    source = Path(ctx["repo"])/REPORT/"code/trajectory_diagnostics.py"
    spec = importlib.util.spec_from_file_location("pems32_metadata_only_reader", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.checkpoint_metadata(path)


def complete_cost_metadata(ctx, results, data):
    """Validate first, then update present E01 C32 cost rows; no model/storage loads or writes."""
    targets = [cost for cost in data["costs"] if cost.get("cohort") == "pems32"
               and cost.get("job_id") in results and results[cost["job_id"]][0]["question_id"] == "E01"]
    if not targets:
        return dict(status="NOT_APPLICABLE", checked_rows=0, private_artifacts=[], public_verification_sources=[])
    repo, report = Path(ctx["repo"]).resolve(), Path(ctx["report"]).resolve()
    plan_path = report/"EXPERIMENT_PLAN.json"
    plan = read(plan_path)
    validate_plan(plan, repo)
    plan_jobs = {job["job_id"]: job for job in plan["jobs"]}
    public_sources, private_sources = {}, {}

    def public(path, expected=None):
        path = Path(path).resolve()
        if not path.is_relative_to(repo):
            raise ValueError("Cost metadata public source escapes repository")
        digest = sha(path)
        if expected is not None and digest != expected:
            raise ValueError("Cost metadata public source SHA mismatch")
        public_sources[path.relative_to(repo).as_posix()] = digest
        return read(path) if path.suffix == ".json" else None

    def private(reference, expected):
        scheme, relative = reference.split("://", 1)
        root = Path(ctx[scheme]).resolve()
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or sha(path) != expected:
            raise ValueError("Cost metadata private artifact/weights SHA mismatch")
        canonical = scheme+"://"+path.relative_to(root).as_posix()
        private_sources[canonical] = dict(reference=canonical, sha256=expected)
        return path

    def producer(job, result, receipt_path, result_path):
        if (job["job_id"] not in plan_jobs or plan_jobs[job["job_id"]]["scientific_identity"] != job["scientific_identity"]):
            raise ValueError("Cost metadata job differs from sealed plan identity")
        receipt = public(receipt_path)
        if (receipt.get("status") not in ("COMPLETE", "REUSED_VERIFIED")
                or receipt.get("scientific_identity") != job["scientific_identity"] or receipt.get("result") != result):
            raise ValueError("Cost metadata receipt identity/result mismatch")
        relative = result_path.relative_to(report).as_posix()
        expected = receipt.get("artifact_hashes", {}).get(relative)
        if expected is None:
            raise ValueError("Cost metadata result lacks receipt hash binding")
        if public(result_path, expected) != result:
            raise ValueError("Cost metadata bound producer result differs")

    e03_candidates = [entry for entry in results.values() if entry[0]["question_id"] == "E03"
                      and entry[0].get("dataset_cohort") == "jena"]
    if not e03_candidates:
        return dict(status="UNKNOWN_MISSING_E03_EVIDENCE", checked_rows=0,
                    reason="No successful source-bound identical-snapshot E03 F parameter count; existing unknown fields retained",
                    private_artifacts=[], public_verification_sources=[])
    if len(e03_candidates) != 1:
        raise ValueError("Cost metadata E03 F count source is ambiguous")
    public(plan_path)
    for relative, expected in SOURCE_PINS.items():
        if plan["source_hashes"].get(relative) != expected:
            raise ValueError("Cost metadata sealed measurement source identity mismatch: "+relative)
        public(repo/relative, expected)
    reader_path = repo/REPORT/"code/trajectory_diagnostics.py"
    public(reader_path)
    subset_path = repo/BROAD/"priority_pems_subset_20261016_v1/PLAN.json"
    subset = public(subset_path, SUBSET_PLAN_SHA256)
    if (len(subset["column_selection"]["selected_ids"]) != 32
            or len(set(subset["column_selection"]["selected_ids"])) != 32):
        raise ValueError("Cost metadata subset does not identify 32 distinct channels")
    e03_job, e03, e03_receipt_path = e03_candidates[0]
    producer(e03_job, e03, e03_receipt_path, report/"costs/E03_JENA_BACKBONES.json")
    raw = [row for row in e03["rows"] if row["method"] == "Bolt_Raw"]
    if ({row.get("B") for row in raw} != {1, 8} or len(raw) != 2
            or any(row.get("status") != "COMPLETE" or row.get("parity") != "PASS"
                   or row.get("core_dtype") != "float32" or row.get("trainable_params") != 0 for row in raw)):
        raise ValueError("Cost metadata E03 Raw parameter/dtype evidence is incomplete")
    f_counts = {row["total_resident_parameters"] for row in raw}
    if len(f_counts) != 1:
        raise ValueError("Cost metadata E03 F parameter counts disagree")
    foundation_count = f_counts.pop()
    if not isinstance(foundation_count, int) or foundation_count <= 0:
        raise ValueError("Cost metadata F parameter count is invalid")
    e03_weights = {item["reference"]: item["sha256"] for item in e03["private_artifacts"]}
    for item in subset["model_snapshot_bindings"]:
        reference = "asset://"+item["resolved_blob_relative_path"]
        if e03_weights.get(reference) != item["sha256"]:
            raise ValueError("Cost metadata E03 and PeMS foundation weights/config identity differs")
        private(reference, item["sha256"])
    pending, evidence, seen = [], [], set()
    for cost in targets:
        job, result, receipt_path = results[cost["job_id"]]
        family, batch = cost["method"], int(float(cost["B"]))
        if (family not in ("B0", "B2") or batch not in (1, 8) or (family, batch) in seen
                or result.get("family") != family or result.get("batch_size") != batch
                or result.get("status") != "COMPLETE_PROTOCOL" or result.get("device") != "cpu"
                or result.get("CPU_threads") != 2 or result.get("C") != 32 or result.get("K") != 8
                or result.get("physical_output_and_inverse_dtype") != "float64"
                or result.get("forward_parity", {}).get("status") != "PASS"):
            raise ValueError("Cost metadata registered measurement spec/dtype/device mismatch")
        seen.add((family, batch))
        result_path = report/"costs"/f"pems_{family}_b{batch}"/"RESULT.json"
        producer(job, result, receipt_path, result_path)
        for filename, expected in result["source_protocol"]["original_scientific_sources"].items():
            relative = BROAD+"/"+filename
            if plan["source_hashes"].get(relative) != expected:
                raise ValueError("Cost metadata E01 measurement source differs from sealed source")
            public(repo/relative, expected)
        reference = "cache://runs/pems_"+family+"/selected_bundle.pt"
        bundle_path = private(reference, result["artifact_sha256"])
        bundle = _checkpoint_metadata(ctx, bundle_path)
        spec, contract, refit = bundle["spec"], bundle["contract"], bundle["refit_summary"]
        data_hashes = {item["data_sha256"] for item in result["source_protocol"]["original_receipts"]}
        if (spec.get("family") != family or spec.get("dataset") != "pems08" or spec.get("latent_k") != 8
                or spec.get("seed") != job["seed"] or contract.get("model_sha") != subset["foundation_tensor_sha256"]
                or refit.get("contract") != contract or refit.get("status") != "SEGMENT_COMPLETE"
                or data_hashes != {contract.get("data_sha")}):
            raise ValueError("Cost metadata selected bundle spec/F/data identity mismatch")
        descriptors = bundle["model"]
        if (not descriptors or any(value.storage.get("dtype") != "<f4" for value in descriptors.values())
                or tuple(descriptors["E"].shape) != (32, 8) or tuple(descriptors["D"].shape) != (7, 8, 32)):
            raise ValueError("Cost metadata adapter state dtype/shape mismatch")
        adapter_count = refit.get("trainable_parameters")
        state_count = sum(math.prod(value.shape) for value in descriptors.values())
        if not isinstance(adapter_count, int) or not 0 < adapter_count <= state_count:
            raise ValueError("Cost metadata adapter parameter count is invalid")
        total = foundation_count+adapter_count
        measured_bytes = result["memory"]["resident_weights_cpu_MiB"]*1024**2
        if not math.isfinite(measured_bytes) or measured_bytes < total*4 or not measured_bytes.is_integer():
            raise ValueError("Cost metadata measured weights/buffers contradict parameter metadata")
        values = dict(device="cpu", threads=result["CPU_threads"], core_dtype="float32",
            dtype="FP32 core; FP64 inverse; CPU return", trainable_params=adapter_count,
            backbone_resident_parameters=foundation_count, total_resident_model_params=total,
            resident_weights_and_buffers_cpu_bytes=int(measured_bytes), source_derived_parameter_bytes=total*4,
            resident_buffer_bytes_above_parameters=int(measured_bytes)-total*4,
            resident_parameter_scope="SOURCE_DERIVED: identical pinned E03 F parameter count plus selected adapter metadata; buffers excluded; no new parameter measurement in this E01 CPU run",
            resident_weights_and_buffers_scope="Recorded deployment.memory_state CPU parameters+buffers, excluding normalization arrays and process RSS; not activation memory",
            metadata_derivation="Bound immutable E01 result + sealed FP32 route/F sources + identical-snapshot E03 count + selected-bundle data.pkl descriptors; no tensor storage loaded")
        pending.append((cost, values))
        evidence.append(dict(job_id=job["job_id"], family=family, B=batch, artifact_sha256=result["artifact_sha256"],
                             F_state_sha256=contract["model_sha"], adapter_training_device=spec.get("device"), **values))
    for cost, values in pending:
        cost.update(values)
    data.setdefault("bindings", {}).update(public_sources)
    return dict(status="PASS_SOURCE_DERIVED_PRESENT_ROWS", checked_rows=len(evidence), expected_registered_rows=4,
                helper_sha256=sha(__file__), rows=evidence,
                public_verification_sources=[dict(reference="repo://"+name, sha256=digest) for name, digest in sorted(public_sources.items())],
                private_artifacts=list(private_sources.values()), model_calls=0, tensor_storage_loaded=False,
                scope="Source-derived dtype/parameter metadata plus existing measured weights+buffers bytes; no new cost, model, or memory experiment")
