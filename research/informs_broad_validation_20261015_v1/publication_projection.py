"""Preserve scientific evidence while omitting proven duplicate public statistics.

Only new immutable projection metadata are written. Original files, scores,
receipts, archives, Git state, and scientific validity decisions are untouched.
Opaque V archives are hashed, never opened as NPZ arrays or forecast inputs.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import time

import publication_check as publication


COLUMNS = ("run_id", "split", "cohort", "origin", "channel", "horizon_start", "horizon_end", "sse", "sae", "count")
REFERENCE_KEYS = ("target_normalized", "observed", "origins", "channel_ids")
ARRAY_KEYS = ("pred_normalized", *REFERENCE_KEYS)
IDENTITY_KEYS = ("run_id", "split", "cohort", "source_sha", "config_sha", "data_sha", "scaler_sha", "model_sha")
KNOWN_LOCAL_ONLY = ("diagnostics/initialization_anchor_1792059651372799700",
                    "execution_batches/cpu_V_checkpoint_replay_1792060139101125900.json")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _inside(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    path = (root / relative).resolve()
    if relative.is_absolute() or path == root or not path.is_relative_to(root):
        raise ValueError("Evidence path escapes its declared root")
    if path.is_symlink():
        raise ValueError("Symlink evidence is outside the projection contract")
    return path


def _same(left, right, message):
    if left != right:
        raise ValueError(message)


def _file_evidence(path, root):
    return dict(file=path.relative_to(root).as_posix(), sha256=sha256_file(path), bytes=path.stat().st_size)


def _compare_rows(original, replacement, original_id, replacement_id, split, cohort):
    if any(char in value for value in (original_id, replacement_id) for char in ',"\r\n'):
        raise ValueError("CSV run IDs require a direct unquoted first-column mapping")
    raw_left, raw_right, common_left, common_right = (hashlib.sha256() for _ in range(4))
    rows = 0
    with gzip.open(original, "rb") as left, gzip.open(replacement, "rb") as right:
        for number, (a, b) in enumerate(itertools.zip_longest(left, right)):
            if a is None or b is None:
                raise ValueError("Statistic row counts differ")
            raw_left.update(a); raw_right.update(b)
            if number == 0:
                _same(a, b, "Statistic CSV headers differ")
                _same(tuple(a.decode("utf-8").rstrip("\r\n").split(",")), COLUMNS, "Unexpected statistic CSV schema")
                header = a.decode("utf-8")
                continue
            if not a.endswith(b"\n") or not b.endswith(b"\n"):
                raise ValueError("Incomplete statistic CSV row")
            first_a, rest_a = a.rstrip(b"\r\n").split(b",", 1)
            first_b, rest_b = b.rstrip(b"\r\n").split(b",", 1)
            _same(first_a.decode("utf-8"), original_id, "Internal statistic run ID differs from receipt mapping")
            _same(first_b.decode("utf-8"), replacement_id, "Standard statistic run ID differs from manifest")
            _same(rest_a, rest_b, "Non-run-ID statistic columns differ exactly")
            # The first two common columns cannot contain delimiters in these contracts.
            fields = rest_a.split(b",", 2)
            _same(fields[0].decode(), split, "Statistic split differs from manifest")
            _same(fields[1].decode(), cohort, "Statistic cohort differs from manifest")
            common_left.update(rest_a + b"\n"); common_right.update(rest_b + b"\n")
            rows += 1
    if rows == 0:
        raise ValueError("Empty sufficient statistics cannot replace an evidence file")
    return dict(row_count=rows, original_uncompressed_sha256=raw_left.hexdigest(),
                replacement_uncompressed_sha256=raw_right.hexdigest(),
                non_run_id_columns_sha256=common_left.hexdigest(),
                replacement_non_run_id_columns_sha256=common_right.hexdigest(),
                original_header=header, canonical_common_row_line_ending="LF",
                exact_non_run_id_column_comparison=True, float_tolerance_used=False)


def verify_duplicate(original, manifest_path, campaign, cache_root):
    campaign, cache_root = Path(campaign).resolve(), Path(cache_root).resolve()
    evidence = [_file_evidence(original, campaign), _file_evidence(manifest_path, campaign),
                _file_evidence(manifest_path.with_name("REFERENCE_CONTRACT.json"), campaign)]
    manifest, reference = _json(manifest_path), _json(manifest_path.with_name("REFERENCE_CONTRACT.json"))
    pre_path = _inside(campaign, reference["pre_target_contract_public_relative_path"])
    _same(sha256_file(pre_path), reference["pre_target_contract_sha256"], "Pre-target contract SHA differs")
    pre, receipt_path = _json(pre_path), original.parent / "receipt.json"
    evidence.extend((_file_evidence(pre_path, campaign), _file_evidence(receipt_path, campaign)))
    receipt = _json(receipt_path)
    family, label, split = pre["family_run_id"], pre["label"], pre["split"]
    # Current internal combination writers publish V only. Do not access a protected target archive.
    if split != "V_select":
        raise ValueError("Only existing V_select combination statistics are authorized for opaque archive hashing")
    _same(manifest_path, campaign / "predictions" / manifest["run_id"] / split / "PREDICTION_MANIFEST.json", "Standard manifest directory differs from sealed run ID")
    _same(receipt["execution_status"], "COMPLETE_VALID", "Incomplete combination receipt is not a replacement proof")
    _same(receipt["run_id"], family, "Combination receipt family differs")
    _same(original.parent.name, family, "Internal statistic directory differs from family")
    _same(original.name, f"{label}_{split}_error_stats.csv.gz", "Internal filename differs from sealed output label")
    _same(pre["kind"], "combination", "This projection only substitutes combination-internal statistics")
    _same(pre["variant"], receipt["variant"], "Combination variant differs")
    for key, value in pre["expected_receipt_contract"].items():
        _same(receipt["contract"].get(key), value, "Receipt fit contract differs: " + key)
    for key in IDENTITY_KEYS:
        _same(manifest.get(key), reference.get(key), "Manifest/reference identity differs: " + key)
        if not manifest.get(key):
            raise ValueError("Missing manifest identity: " + key)
        if key != "model_sha":
            _same(pre.get(key), reference.get(key), "Pre-target/reference identity differs: " + key)
    _same(fingerprint(reference["realized_model_binding"]), reference["model_sha"], "Realized selected model binding SHA differs")
    metadata = manifest["metadata"]
    for key, expected in (("family_run_id", family), ("output_label", label), ("reference_kind", "combination")):
        _same(metadata.get(key), expected, "Standard manifest metadata differs: " + key)
    _same(metadata.get("actual_parent_manifests"), pre["parent_anchors"], "Parent anchor mappings differ")
    _same(set(manifest["arrays"]), set(ARRAY_KEYS), "Standard manifest array schema differs")
    for key in REFERENCE_KEYS:
        _same(pre["arrays"][key], reference["arrays"][key], "Pre-target/reference array fingerprint differs: " + key)
        _same(manifest["arrays"][key], reference["arrays"][key], "Manifest/reference array fingerprint differs: " + key)
    declared = receipt["prediction_artifacts"][label][split]
    _same(declared["arrays"], manifest["arrays"], "Receipt/standard array fingerprints differ")
    _same(declared["sha256"], manifest["archive_sha256"], "Receipt/standard archive SHA differs")
    _same(pre["cache_relative_path"], manifest["cache_relative_path"], "Private archive paths differ")
    archive = _inside(cache_root, manifest["cache_relative_path"])
    _same(archive.name, declared["file"], "Receipt private archive filename differs")
    _same(sha256_file(archive), manifest["archive_sha256"], "Opaque private archive SHA differs")
    parent_files = []
    for anchor in pre["parent_anchors"].values():
        path = _inside(campaign, anchor["manifest_public_relative_path"])
        _same(sha256_file(path), anchor["manifest_sha256"], "Actual parent manifest SHA differs")
        parent = _json(path)
        for key in ("run_id", "model_sha", "archive_sha256"):
            _same(parent[key], anchor[key], "Actual parent identity differs: " + key)
        parent_files.append(_file_evidence(path, campaign))
    binding = reference["realized_model_binding"]
    artifact_relative = binding.get("selected_artifact_cache_relative_path")
    if artifact_relative is not None:
        _same(sha256_file(_inside(cache_root, artifact_relative)), binding["selected_artifact_sha256"], "Selected combination artifact SHA differs")
    internal_id = family + "__" + label if pre["variant"] == "alpha_family" else family
    scores = [score for score in receipt["scores"] if score["run_id"] == internal_id and score["split"] == split]
    if len(scores) != 1:
        raise ValueError("Receipt has no unique exact internal-output score mapping")
    for key in ("cohort", "mse", "mae", "n_origins", "num_channels", "horizon", "n_observed"):
        _same(scores[0][key], manifest["score"][key], "Receipt/standard score differs: " + key)
    replacement = _inside(campaign, manifest["public_stats_relative_path"])
    if replacement != manifest_path.with_name("PUBLIC_ERROR_STATS.csv.gz"):
        raise ValueError("Replacement is not the canonical standard statistic path")
    evidence.append(_file_evidence(replacement, campaign))
    _same(binding["output_label"], label, "Realized model output label differs from sealed label")
    if pre["parent_anchors"] and not binding.get("parent_model_shas"):
        raise ValueError("Realized model binding has no active parent identity")
    for key, model_sha in binding.get("parent_model_shas", {}).items():
        _same(model_sha, pre["parent_anchors"][key]["model_sha"], "Realized model parent differs from sealed anchor")
    if pre["variant"] != "alpha_family":
        _same(binding["selected_artifact_sha256"], receipt["selected_checkpoint_sha256"], "Selected checkpoint differs from combination receipt")
    for key in ("run_id", "split", "cohort"):
        _same(manifest["score"][key], manifest[key], "Standard score identity differs: " + key)
    compared = _compare_rows(original, replacement, internal_id, manifest["run_id"], split, manifest["cohort"])
    horizon = int(manifest["score"]["horizon"])
    edges = {0, math.ceil(horizon / 4), math.ceil(horizon / 2), horizon}
    expected_rows = int(manifest["score"]["n_origins"]) * int(manifest["score"]["num_channels"]) * (len(edges) - 1)
    _same(compared["row_count"], expected_rows, "Statistic coverage row count differs from frozen manifest axes")
    for item in [*evidence, *parent_files]:
        _same(sha256_file(_inside(campaign, item["file"])), item["sha256"], "Public evidence changed during projection verification")
    return dict(status="PROVEN_DUPLICATE_EXCLUDED_FROM_PUBLIC_PAYLOAD", original=_file_evidence(original, campaign),
        replacement=_file_evidence(replacement, campaign), **compared, run_id_mapping={internal_id: manifest["run_id"]},
        family_run_id=family, output_label=label, split=split, cohort=manifest["cohort"],
        receipt=_file_evidence(receipt_path, campaign), standard_manifest=_file_evidence(manifest_path, campaign),
        reference_contract=_file_evidence(manifest_path.with_name("REFERENCE_CONTRACT.json"), campaign),
        pre_target_contract=_file_evidence(pre_path, campaign), parent_manifests=parent_files,
        private_archive_cache_relative_path=manifest["cache_relative_path"], private_archive_sha256=manifest["archive_sha256"],
        opaque_private_archive_sha_verified=True, array_fingerprints=manifest["arrays"],
        declaration_array_fingerprints_exact=True, actual_array_fingerprints_recomputed=False,
        NPZ_arrays_deserialized=False, selected_model_binding=binding, fit_contract=receipt["contract"],
        original_preserved_locally=True, original_gzip_byte_reconstruction_claimed=False)


def create_projection(campaign, cache_root, *, extra_excludes=()):
    campaign, cache_root = Path(campaign).resolve(), Path(cache_root).resolve()
    repo = publication._root(campaign)
    extra_excludes = tuple(str(value).replace("\\", "/").rstrip("/") for value in extra_excludes)
    for value in extra_excludes:
        if not value.startswith(("figures/snapshot_", "reports/snapshot_")) or ".." in Path(value).parts:
            raise ValueError("Extra excludes are limited to explicitly named intermediate figure/report snapshots")
    names, excluded = publication.candidate_files(campaign, excluded_prefixes=(*KNOWN_LOCAL_ONLY, *extra_excludes))
    selected = set(names)
    manifest_map = {}
    manifest_errors = []
    for path in sorted((campaign / "predictions").glob("*/V_select/PREDICTION_MANIFEST.json")):
        try:
            value = _json(path); meta = value.get("metadata", {})
            key = (meta.get("family_run_id"), meta.get("output_label"), value.get("split"))
            if all(key) and meta.get("reference_kind") == "combination":
                manifest_map.setdefault(key, []).append(path)
        except (OSError, ValueError, TypeError) as exc:
            manifest_errors.append(dict(file=path.relative_to(campaign).as_posix(), rule="UNREADABLE_STANDARD_MANIFEST", error_type=type(exc).__name__))
    proofs, failures = [], list(manifest_errors)
    for original in sorted((campaign / "combinations").glob("*/*_error_stats.csv.gz")):
        if original.relative_to(repo).as_posix() not in selected:
            continue
        try:
            matches = []
            for (family, label, split), paths in manifest_map.items():
                if original.parent.name == family and original.name == f"{label}_{split}_error_stats.csv.gz":
                    matches.extend(paths)
            if len(matches) != 1:
                raise ValueError("No unique standardized manifest mapping; original kept")
            proof = verify_duplicate(original, matches[0], campaign, cache_root)
            proofs.append(proof)
            selected.remove(original.relative_to(repo).as_posix())
        except (OSError, ValueError, KeyError, TypeError, EOFError) as exc:
            reason = str(exc)
            for path, label in ((campaign, "<campaign>"), (cache_root, "<private cache>")):
                reason = reason.replace(str(path), label).replace(path.as_posix(), label)
            failures.append(dict(file=original.relative_to(campaign).as_posix(), rule="UNPROVEN_DUPLICATE_KEPT",
                                 error_type=type(exc).__name__, reason=reason, original_kept_in_payload=True))
    snapshot = "snapshot_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + str(time.time_ns())
    directory = campaign / "publication_projections" / snapshot
    directory.mkdir(parents=True, exist_ok=False)
    output_names = ("PROJECTION.json", "PAYLOAD_FILES.txt", "README_KO.md")
    selected.update((directory / name).relative_to(repo).as_posix() for name in output_names)
    payload = ("\n".join(sorted(selected)) + "\n").encode("utf-8")
    removed_bytes = sum(proof["original"]["bytes"] for proof in proofs)
    result = dict(schema=1, status="FAIL" if failures else "PASS", created_unix=time.time(),
        scope="Lossless sufficient-statistic publication projection; no scientific revalidation or array deserialization",
        source_sha256=sha256_file(Path(__file__)), publication_helper_sha256=sha256_file(Path(publication.__file__)),
        candidate_file_count=len(names), candidate_bytes=sum((repo / name).stat().st_size for name in names),
        excluded_duplicate_count=len(proofs), excluded_duplicate_bytes=removed_bytes,
        payload_file_count=len(selected), payload_file_list_sha256=hashlib.sha256(payload).hexdigest(),
        known_local_only_prefixes=KNOWN_LOCAL_ONLY, explicit_intermediate_snapshot_excludes=extra_excludes,
        default_and_explicit_exclusions=excluded, duplicate_statistics=proofs, failures=failures,
        originals_modified_or_deleted=False, standard_statistics_excluded=False, registry_changes=0,
        GPU_calls=0, scientific_fits=0, NPZ_arrays_deserialized=False, TEST_arrays_read=False,
        actual_array_fingerprints_recomputed=False, independent_scientific_audit_performed=False,
        requires_separate_final_publication_check=True, original_gzip_byte_reconstruction_claimed=False,
        failed_or_unproven_internal_statistics_remain_public=True)
    for name, content in (("PAYLOAD_FILES.txt", payload), ("PROJECTION.json", publication._json_bytes(result))):
        with (directory / name).open("xb") as stream:
            stream.write(content)
    readme = ["# 무손실 공개 projection", "", f"상태 `{result['status']}`. 검증된 중복 통계 {len(proofs)}개, {removed_bytes:,} byte를 공개 후보에서 제외한다.",
        "원본은 로컬에 그대로 보존한다. 표준 통계·예측 manifest·receipt·계약·실패 기록은 유지한다.",
        "run ID 매핑을 제외한 9열의 행별 문자열이 정확히 같음을 검사했다. 수치 허용오차나 재압축을 사용하지 않았다.",
        "NPZ는 해석하지 않고 V archive의 byte SHA만 검증했다. 배열 지문은 기존 선언 간 동일성 검사이며 새 예측 검산·학습·일반화 검증이 아니다.",
        "원 gzip byte의 재구축을 주장하지 않는다. 원 압축/해제 SHA와 표준 통계 및 run ID 매핑으로 충분통계의 계보를 보존한다.", "",
        "- [계보 증명·실패 기록](PROJECTION.json)", "- [공개 payload 파일 목록](PAYLOAD_FILES.txt)", "", "표준 충분통계:", ""]
    for proof in proofs:
        target = "../../" + proof["replacement"]["file"]
        readme.append(f"- [{proof['family_run_id']} / {proof['output_label']}]({target})")
    with (directory / "README_KO.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(readme) + "\n")
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--exclude-prefix", action="append", default=[])
    args = parser.parse_args()
    directory = create_projection(args.campaign, args.cache_root, extra_excludes=args.exclude_prefix)
    result = _json(directory / "PROJECTION.json")
    print(json.dumps(dict(status=result["status"], projection=directory.relative_to(args.campaign.resolve()).as_posix(),
        excluded_duplicate_count=result["excluded_duplicate_count"], excluded_duplicate_bytes=result["excluded_duplicate_bytes"],
        failure_count=len(result["failures"])), ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
