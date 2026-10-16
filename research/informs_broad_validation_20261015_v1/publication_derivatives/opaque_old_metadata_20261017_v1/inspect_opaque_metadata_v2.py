"""Narrow byte-only exception for exactly three already-public metadata blobs.

Frozen numeric.inspect still inspects every other explicitly selected file. This
does not replace ROOT's complete selected-inventory/cap/index/tree/blob checks.
The default CLI performs no reads, writes, imports of frozen sources or scans.
"""
import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_pack():
    spec = importlib.util.spec_from_file_location("opaque_publication_pack_contract_v2", HERE / "pack_metadata_v2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_proof(proof, manifest, pack):
    expected = {name: dict(bytes=size, sha256=sha) for name, size, sha in pack.ORIGINALS}
    if (proof.get("schema") != "OPAQUE_OLD_METADATA_BYTE_PROOF_v2"
            or proof.get("status") != "PASS_EXACT_THREE_ALREADY_PUBLIC_METADATA_BYTES"
            or proof.get("old_inventory_sha256") != pack.INVENTORY_SHA
            or proof.get("existing_restorer_sha256") != pack.RESTORER_SHA
            or proof.get("original_before") != expected or proof.get("original_after") != expected
            or proof.get("source48_before") != proof.get("source48_after") or len(proof.get("source48_before", {})) != 48
            or proof.get("mismatch_refused_without_write") is not True or proof.get("original_files_modified") is not False
            or proof.get("original_payload_JSON_UTF8_or_numeric_parse") is not False or proof.get("new_T_numeric_parse") != 0
            or proof.get("worker_deadline_unix") != pack.DEADLINE
            or proof.get("resource_terminal_admission_independent_of_scientific_success") is not True
            or not isinstance(proof.get("scientific_terminal_statuses"), dict)
            or proof.get("CPU2_RSS1GiB_external_supervisor_admission") != "ADMITTED"):
        raise ValueError("Pinned actual exact-restoration proof required")
    restores = proof.get("fresh_restores", [])
    if len(restores) != 2 or [row.get("destination_label") for row in restores] != ["restore_a", "restore_b"]:
        raise ValueError("Actual two distinct fresh restore receipts required")
    for row in restores:
        if (row.get("first") != dict(status="PASS", restored=3, already_matching=0)
                or row.get("repeated") != dict(status="PASS", restored=0, already_matching=3)
                or row.get("files") != [dict(file=name, **expected[name]) for name, _, _ in pack.ORIGINALS]):
            raise ValueError("Exact original bytes and fresh/repeat restore counts differ")
    pack.validate_manifest(manifest)
    return expected


class ExactVirtualFacade:
    def __init__(self, frozen, exact_names):
        self.frozen = frozen
        self.exact_names = tuple(exact_names)

    def __getattr__(self, name):
        return getattr(self.frozen, name)

    def inspect(self, campaign, files, tracked=None, virtual_files=()):
        if set(files) & set(self.exact_names):
            raise ValueError("Opaque binary must never enter the frozen text/gzip scanner")
        return self.frozen.inspect(campaign, files, tracked=tracked,
                                   virtual_files=tuple(virtual_files) + self.exact_names)


def delegate_numeric(numeric, campaign, normal, tracked, opaque):
    original_loader = numeric.load_dependencies

    def loader():
        frozen, codec = original_loader()
        return ExactVirtualFacade(frozen, opaque), codec

    numeric.load_dependencies = loader
    try:
        return numeric.inspect(campaign, normal, tracked=set(tracked) | set(opaque))
    finally:
        numeric.load_dependencies = original_loader


def inspect(repo, files, tracked, inventory_path, proof_sha):
    pack = load_pack()
    repo = pack.plain(repo)
    campaign = pack.inside(repo, pack.CAMPAIGN)
    namespace = pack.inside(campaign, pack.NAMESPACE)
    if namespace != pack.plain(HERE):
        raise ValueError("Exact reviewed namespace required")
    pack.inventory_rows(pack.bound(inventory_path, pack.INVENTORY_SHA))
    names = [pack.safe(name) for name in files]
    if len(names) != len(set(name.casefold() for name in names)) or not names:
        raise ValueError("Nonempty canonical explicit list without duplicates required")
    tracked = [pack.safe(name) for name in tracked]
    prefix = pack.CAMPAIGN + "/" + pack.NAMESPACE + "/"
    opaque = [prefix + name for name in pack.GZIP_NAMES]
    required = set(opaque + [prefix + name for name in ("GZIP_MANIFEST.json", "PROOF.json", "restore_originals.py")])
    if not required.issubset(names):
        raise ValueError("All exact three binaries, manifest, proof and frozen reader must be explicitly selected")
    for name in names:
        if name.endswith(".opaque.json.gz") and name not in opaque:
            raise ValueError("Fourth or out-of-namespace opaque binary rejected")
        if name in {row[0] for row in pack.ORIGINALS}:
            raise ValueError("Old original payload must not enter new text inspection")
    proof_path = namespace / "PROOF.json"
    proof_raw = pack.bound(proof_path, proof_sha)
    proof = json.loads(proof_raw)
    manifest_path = namespace / "GZIP_MANIFEST.json"
    manifest_raw = pack.bound(manifest_path, proof.get("manifest_sha256"))
    manifest = json.loads(manifest_raw)
    check_proof(proof, manifest, pack)
    pack.bound(namespace / "restore_originals.py", pack.RESTORER_SHA)
    pack.bound(HERE / "pack_metadata_v2.py", proof.get("runner_sha256"))
    pack.bound(HERE / "inspect_opaque_metadata_v2.py", proof.get("inspector_sha256"))
    before = {}
    for item, name in zip(manifest["items"], opaque):
        raw = pack.bound(pack.inside(repo, name), item["gzip_sha256"])
        if len(raw) != item["gzip_bytes"]:
            raise ValueError("Opaque byte size differs")
        before[name] = dict(file=name, bytes=len(raw), sha256=pack.digest(raw))
    numeric_path = campaign / "publication_check_numeric.py"
    pack.bound(numeric_path, pack.NUMERIC_SHA)
    spec = importlib.util.spec_from_file_location("opaque_private_frozen_numeric", numeric_path)
    numeric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(numeric)
    normal = [name for name in names if name not in opaque]
    result = delegate_numeric(numeric, campaign, normal, tracked, opaque)
    if {row["file"] for row in result["files"]} != set(normal):
        raise ValueError("Frozen numeric inspector did not cover every other selected file")
    rows = result["files"] + list(before.values())
    total = sum(row["bytes"] for row in rows)
    failures = list(result["failures"])
    if total > pack.MAX_BUNDLE:
        failures.append(dict(file="<bundle>", rule="BUNDLE_SIZE_LIMIT", bytes=total, limit=pack.MAX_BUNDLE))
    for name, row in before.items():
        if len(pack.bound(pack.inside(repo, name), row["sha256"])) != row["bytes"]:
            raise ValueError("Opaque input changed during inspection")
    if pack.bound(proof_path, proof_sha) != proof_raw or pack.bound(manifest_path, proof["manifest_sha256"]) != manifest_raw:
        raise ValueError("Proof or manifest changed during inspection")
    pack.bound(numeric_path, pack.NUMERIC_SHA)
    result.update(status="FAIL" if failures else "PASS", schema="EXACT_THREE_OPAQUE_METADATA_PUBLICATION_INSPECTION_v2",
        files=sorted(rows, key=lambda row: row["file"]), file_count=len(rows), selected_bytes=total,
        public_increment_upper_bound_bytes=total, failures=failures, opaque_files=list(before.values()),
        actual_exact_restore_proof_sha256=proof_sha, old_selected_inventory_sha256=pack.INVENTORY_SHA,
        opaque_payload_JSON_UTF8_numeric_or_gzip_decode_operations=0, frozen_numeric_source_changed=False,
        exact_three_passed_as_frozen_pc_virtual_files=True, frozen_numeric_inspection_used_for_all_other_files=True,
        whole_ROOT_selected_inventory_cap_index_tree_blob_verification_still_required=True,
        source_or_original_files_written=0, Git_model_Task_or_process_query_operations=0)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("repo-root", "file-list", "file-list-sha", "tracked-list", "tracked-list-sha", "old-inventory", "proof-sha"):
        parser.add_argument("--" + key)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--root-triggered", action="store_true")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="PREP_ONLY_DEFAULT_ZERO_IO", input_reads=0, files_written=0, frozen_inspector_calls=0)))
        return 0
    if not args.root_triggered or not all((args.repo_root, args.file_list, args.file_list_sha, args.tracked_list,
                                         args.tracked_list_sha, args.old_inventory, args.proof_sha)):
        parser.error("Explicit ROOT trigger, exact lists/SHA, old inventory and actual proof SHA required")
    pack = load_pack()
    listing = pack.bound(args.file_list, args.file_list_sha)
    tracking = pack.bound(args.tracked_list, args.tracked_list_sha)
    result = inspect(args.repo_root, listing.decode("utf-8").splitlines(), tracking.decode("utf-8").splitlines(), args.old_inventory, args.proof_sha)
    pack.bound(args.file_list, args.file_list_sha)
    pack.bound(args.tracked_list, args.tracked_list_sha)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
