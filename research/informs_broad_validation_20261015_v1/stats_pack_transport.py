"""Separate exact-byte transport for one explicitly selected legacy stats archive.

Only execute-pack/restore/inspect decompress public aggregate members. Original
archives and manifests are never edited. This third format is not understood by
the frozen publication inspector and does not promote any original audit gate.
"""
import argparse
import hashlib
import importlib.util
import json
import lzma
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import time


HERE = Path(__file__).resolve().parent
CORE_SHA = "0d8c4e4cc9881b70907acf7a154133a64cb582227c1df357521a7668e7fde3fd"
NUMERIC_SHA = "c9c42c780306b6e75eb68c939db1c3a9658b3e2680a3d57c15c9971cbefd8743"
LEGACY_PRODUCER_SHA = "3fbe41fae32c940ddec35a2db027599b77f5fea0703305d64f02a5fd82d351fe"
SCHEMA = "public_error_stats_archive_transport_v1"
SELECTION_SCHEMA = "public_error_stats_transport_selection_v1"
MAX_ARCHIVE = 20 * 1024**2 - 1
MAX_MEMBER = 64 * 1024**2
MAX_TAR = 128 * 1024**2
MAX_MANIFEST = 8 * 1024**2
MAX_MEMBERS = 10000
DEADLINE = 1792222550
ORIGINAL_MANIFEST_MEMBER = "metadata/original_manifest.json"
XZ_RECIPE = dict(format="FORMAT_XZ", preset=6, check="CHECK_CRC64")


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _source_pins():
    pins = dict(core=_sha((HERE / "stats_pack.py").read_bytes()),
                numeric=_sha((HERE / "stats_pack_numeric.py").read_bytes()),
                transport=_sha(Path(__file__).read_bytes()))
    if pins["core"] != CORE_SHA or pins["numeric"] != NUMERIC_SHA:
        raise ValueError("Frozen core/numeric source changed")
    return pins


def load_dependencies():
    _source_pins()
    modules = []
    for name in ("stats_pack", "stats_pack_numeric"):
        spec = importlib.util.spec_from_file_location("transport_pinned_" + name, HERE / (name + ".py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return tuple(modules)


def _int(value, maximum, *, positive=True):
    if isinstance(value, bool) or not isinstance(value, int) or value < (1 if positive else 0) or value > maximum:
        raise ValueError("Invalid bounded transport integer")
    return value


def _digest(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("Exact SHA256 binding required")
    return value


def _name(value):
    if not isinstance(value, str) or not value or "\\" in value or PurePosixPath(value).is_absolute():
        raise ValueError("Repository-relative path required")
    for part in value.split("/"):
        if (part in {"", ".", ".."} or any(ord(c) < 32 or c in ':*?"<>|' for c in part)
                or part.endswith((".", " ")) or re.fullmatch(r"(?i)(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", part)):
            raise ValueError("Unsafe or aliased transport path")
    return value


def _plain(path):
    path = Path(path).absolute()
    if any(p.is_symlink() or getattr(p, "is_junction", lambda: False)() for p in (path, *path.parents)):
        raise ValueError("Symlink/junction transport paths are forbidden")
    return path.resolve()


def _inside(root, name):
    root = _plain(root)
    path = _plain(root / _name(name))
    if not path.is_relative_to(root):
        raise ValueError("Transport path escapes root")
    return path


def _check(deadline):
    if deadline is not None and time.time() >= deadline:
        raise TimeoutError("Transport wall limit reached; no original file is overwritten")


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON key")
        value[key] = item
    return value


def _json(raw):
    if not 0 < len(raw) <= MAX_MANIFEST:
        raise ValueError("Transport JSON size limit exceeded")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs)
    if not isinstance(value, dict):
        raise ValueError("Transport JSON object required")
    return value


def _read_json(path):
    path = _plain(path)
    _int(path.stat().st_size, MAX_MANIFEST)
    return _json(path.read_bytes())


def _list(value, maximum=MAX_MEMBERS):
    if not isinstance(value, list) or not 1 <= len(value) <= maximum or any(not isinstance(row, dict) for row in value):
        raise ValueError("Bounded nonempty object list required")
    return value


def _tar_size(sizes):
    total = 1024
    for size in sizes:
        total += 512 + ((_int(size, MAX_MEMBER) + 511) // 512) * 512
        if total > MAX_TAR:
            raise ValueError("Reconstructed tar exceeds expansion budget")
    total = ((total + 10239) // 10240) * 10240
    if total > MAX_TAR:
        raise ValueError("Canonical tar padding exceeds expansion budget")
    return total


def _binding(value):
    if not isinstance(value, dict):
        raise ValueError("Explicit archive binding required")
    archive = _name(value["original_archive_repository_relative_path"])
    manifest = _name(value["original_manifest_repository_relative_path"])
    p = PurePosixPath(archive)
    if (re.fullmatch(r"part_\d{4}\.tar\.xz", p.name) is None or p.parent.parent.name != "stats_packs"
            or re.fullmatch(r"pack_[A-Za-z0-9_-]+", p.parent.name) is None or manifest != str(p.with_name("MANIFEST.json"))):
        raise ValueError("Only an explicit manifest-bound legacy stats part is supported")
    _int(value["original_archive_bytes"], MAX_ARCHIVE)
    _int(value["original_manifest_bytes"], MAX_MANIFEST)
    _digest(value["original_archive_sha256"])
    _digest(value["original_manifest_sha256"])
    return value


def _original_manifest(raw, binding):
    if len(raw) != binding["original_manifest_bytes"] or _sha(raw) != binding["original_manifest_sha256"]:
        raise ValueError("Embedded original manifest byte SHA differs")
    value = _json(raw)
    if (value.get("format") != "public_error_stats_pack_v1" or value.get("status") != "PASS"
            or value.get("source_sha256") not in {CORE_SHA, LEGACY_PRODUCER_SHA}
            or value.get("separate_temporary_restore_all_sha_verified") is not True):
        raise ValueError("Only already-verified frozen legacy packs may be transported")
    prefix = _name(value["campaign_repository_relative"])
    archive_path = PurePosixPath(binding["original_archive_repository_relative_path"])
    if archive_path.parent.parent.parent.as_posix() != prefix:
        raise ValueError("Legacy archive is outside its campaign")
    archives = _list(value.get("archives"))
    names = [row.get("file") for row in archives]
    if any(not isinstance(name, str) or re.fullmatch(r"part_\d{4}\.tar\.xz", name) is None for name in names) or len(set(names)) != len(names):
        raise ValueError("Legacy archive names differ")
    selected = [row for row in archives if row["file"] == archive_path.name]
    if len(selected) != 1:
        raise ValueError("Unique original archive binding required")
    info = selected[0]
    if info["bytes"] != binding["original_archive_bytes"] or info["sha256"] != binding["original_archive_sha256"]:
        raise ValueError("Selected archive and old manifest disagree")
    _int(info["bytes"], MAX_ARCHIVE); _digest(info["sha256"]); _int(info["tar_bytes"], MAX_TAR)
    members = _list(info.get("members")); member_map = {}
    for row in members:
        name = _name(row["name"])
        if re.fullmatch(r"members/[0-9a-f]{64}\.csv(?:\.gz)?", name) is None or name in member_map:
            raise ValueError("Unsafe or duplicate original member")
        _int(row["bytes"], MAX_MEMBER); _digest(row["sha256"])
        if name.split("/")[1].split(".")[0] != row["sha256"]:
            raise ValueError("Original content-addressed member differs")
        member_map[name] = row
    if _tar_size(row["bytes"] for row in members) != info["tar_bytes"]:
        raise ValueError("Original declared tar size/padding differs")
    records = [row for row in _list(value.get("records")) if row.get("archive") == archive_path.name]
    if not records:
        raise ValueError("Selected original archive has no public records")
    seen = set(); modes = {}
    for row in records:
        name = _name(row["repository_relative_path"])
        try:
            rel = PurePosixPath(name).relative_to(prefix)
        except ValueError:
            raise ValueError("Original statistic is outside campaign") from None
        if (not rel.parts or rel.parts[0] not in {"predictions", "ordered_predictions", "route_predictions", "route_replays", "raw_calibration"}
                or rel.name != "PUBLIC_ERROR_STATS.csv.gz" or name.casefold() in seen):
            raise ValueError("Only existing public aggregate statistic records are supported")
        seen.add(name.casefold())
        for key, limit in (("gzip_bytes", MAX_ARCHIVE), ("csv_bytes", MAX_MEMBER), ("row_count", MAX_MEMBER)):
            _int(row[key], limit)
        _digest(row["gzip_sha256"]); _digest(row["csv_sha256"])
        mode = row.get("storage_mode"); member = row.get("member")
        if member not in member_map or mode not in {"csv_common9", "opaque_gzip"}:
            raise ValueError("Unbound original public member")
        if not member.endswith(".csv" if mode == "csv_common9" else ".csv.gz") or modes.setdefault(member, mode) != mode:
            raise ValueError("Original storage mode/suffix differs")
    if set(modes) != set(member_map):
        raise ValueError("Unbound original archive content")
    return value, info, records, modes


def metadata(repo_root, selection, selection_sha):
    core, numeric = load_dependencies()
    repo = _plain(repo_root); selection = _plain(selection)
    _digest(selection_sha)
    if core.sha_file(selection) != selection_sha:
        raise ValueError("Explicit selection SHA differs")
    selected = _read_json(selection)
    if selected.get("format") != SELECTION_SCHEMA:
        raise ValueError("Explicit transport selection format required")
    binding = {key: selected[key] for key in ("original_archive_repository_relative_path", "original_archive_bytes", "original_archive_sha256",
               "original_manifest_repository_relative_path", "original_manifest_bytes", "original_manifest_sha256")}
    _binding(binding)
    archive = _inside(repo, binding["original_archive_repository_relative_path"])
    manifest = _inside(repo, binding["original_manifest_repository_relative_path"])
    if (archive.stat().st_size != binding["original_archive_bytes"] or core.sha_file(archive) != binding["original_archive_sha256"]
            or manifest.stat().st_size != binding["original_manifest_bytes"] or core.sha_file(manifest) != binding["original_manifest_sha256"]):
        raise ValueError("Selected legacy file byte SHA/size differs")
    raw = manifest.read_bytes()
    original, info, records, modes = _original_manifest(raw, binding)
    return dict(repo=repo, selection=selection, selection_sha256=selection_sha, archive_path=archive, manifest_path=manifest,
                manifest_raw=raw, original=original, info=info, records=records, modes=modes, binding=binding,
                core=core, numeric=numeric, source_pins=_source_pins())


def _validate_payload(core, name, payload, mode, records):
    related = [row for row in records if row["member"] == name]
    raw = payload if mode == "csv_common9" else core._gunzip(payload)
    text, rows = core._csv_schema(raw, common=mode == "csv_common9")
    for row in related:
        if rows != row["row_count"]:
            raise ValueError("Original aggregate row count differs")
        if mode == "opaque_gzip":
            if len(payload) != row["gzip_bytes"] or _sha(payload) != row["gzip_sha256"] or len(raw) != row["csv_bytes"] or _sha(raw) != row["csv_sha256"]:
                raise ValueError("Original opaque gzip/CSV byte SHA differs")
        else:
            prefix = row.get("run_id_prefix")
            if not isinstance(prefix, str) or not 1 < len(prefix) <= row["csv_bytes"]:
                raise ValueError("Original run ID prefix type/size differs")
            try:
                prefix = prefix.encode("utf-8")
            except UnicodeError:
                raise ValueError("Invalid original UTF8 run ID prefix") from None
            if (not prefix.endswith(b",") or prefix.count(b",") != 1 or any(c in prefix for c in (b'"', b"\n", b"\r"))
                    or len(raw) + len(b"run_id,") + rows * len(prefix) != row["csv_bytes"]):
                raise ValueError("Original common9/full CSV size identity differs")
    return dict(member_name=name, storage_mode=mode, row_count=rows, csv_text=text, original_member_sha256=_sha(payload))


def _load_manifest(path):
    value = _read_json(path)
    if (value.get("format") != SCHEMA or value.get("status") not in {"TRANSPORT_CANDIDATE", "PASS_ARCHIVE_BYTE_TRANSPORT"}
            or value.get("source_pins") != _source_pins() or value.get("xz_recipe") != XZ_RECIPE
            or value.get("original_publication_gate_pass_claimed") is not False):
        raise ValueError("Unsupported transport format/source pin/recipe")
    binding = _binding(value.get("binding")); _digest(value.get("selection_sha256")); _digest(value.get("original_tar_sha256"))
    if value.get("original_writer_source_sha256") not in {CORE_SHA, LEGACY_PRODUCER_SHA}:
        raise ValueError("Unsupported original writer provenance")
    _int(value.get("original_tar_bytes"), MAX_TAR)
    members = _list(value.get("members"))
    originals = set(); transported = set()
    for row in members:
        original_name = _name(row["original_name"]); transport_name = _name(row["transport_name"])
        _int(row["original_bytes"], MAX_MEMBER); _int(row["transport_bytes"], MAX_MEMBER)
        _digest(row["original_sha256"]); _digest(row["transport_sha256"])
        suffix = {"numeric_common9": ".numeric", "raw_common9": ".common9", "opaque_gzip": ".csv.gz"}.get(row.get("transport_mode"))
        if (suffix is None or re.fullmatch(r"members/[0-9a-f]{64}\.csv(?:\.gz)?", original_name) is None
                or transport_name != "members/" + row["transport_sha256"] + suffix
                or original_name in originals or transport_name in transported):
            raise ValueError("Unsafe, duplicate or aliased transport member")
        originals.add(original_name); transported.add(transport_name)
    if _tar_size(row["original_bytes"] for row in members) != value["original_tar_bytes"]:
        raise ValueError("Reconstructed original tar declared size differs")
    archive = value.get("archive")
    if not isinstance(archive, dict) or archive.get("file") != PurePosixPath(binding["original_archive_repository_relative_path"]).name.replace(".tar.xz", ".transport.xz"):
        raise ValueError("Transport archive filename differs")
    _int(archive["bytes"], MAX_ARCHIVE); _int(archive["tar_bytes"], MAX_TAR); _digest(archive["sha256"]); _digest(archive["tar_sha256"])
    expected = [dict(name=ORIGINAL_MANIFEST_MEMBER, bytes=binding["original_manifest_bytes"], sha256=binding["original_manifest_sha256"])]
    expected += [dict(name=row["transport_name"], bytes=row["transport_bytes"], sha256=row["transport_sha256"]) for row in members]
    if archive.get("members") != expected or _tar_size(row["bytes"] for row in expected) != archive["tar_bytes"]:
        raise ValueError("Transport member binding/order/expanded size differs")
    return value


def _decode_numeric(numeric, payload, original_bytes, original_sha, rows):
    if not 12 < len(payload) <= MAX_MEMBER or payload[:8] != numeric.MAGIC:
        raise ValueError("Bounded numeric transport member required")
    length = int.from_bytes(payload[8:12], "little")
    if not 0 < length <= numeric.MAX_HEADER or 12 + length > len(payload):
        raise ValueError("Numeric transport header size differs")
    header = _json(payload[12:12 + length])
    _int(header.get("common_csv_bytes"), MAX_MEMBER)
    if (header.get("common_csv_bytes") != original_bytes or header.get("common_csv_sha256") != original_sha
            or header.get("rows") != rows or not isinstance(header.get("header"), str)
            or not 0 < len(header["header"]) <= original_bytes or len(header["header"].encode("utf-8")) > original_bytes):
        raise ValueError("Numeric inner expansion/row binding differs before decode")
    _int(header["rows"], numeric.MAX_ROWS)
    return numeric.decode_common9(payload)


def _rebuild(manifest_path, *, deadline=None, manifest=None):
    _check(deadline)
    core, numeric = load_dependencies(); manifest_path = _plain(manifest_path)
    value = _load_manifest(manifest_path) if manifest is None else manifest
    archive = value["archive"]; path = _inside(manifest_path.parent, archive["file"])
    synthetic = dict(archives=[archive], records=[dict(archive=path.name, member=row["name"]) for row in archive["members"]])
    transported = core._archive_members(synthetic, path)
    if _sha(core._tar_bytes(list(transported.items()))) != archive["tar_sha256"]:
        raise ValueError("Transport tar SHA differs")
    original, info, records, modes = _original_manifest(transported[ORIGINAL_MANIFEST_MEMBER], value["binding"])
    if value["original_writer_source_sha256"] != original["source_sha256"]:
        raise ValueError("Original writer provenance differs from exact embedded manifest")
    if value["original_tar_bytes"] != info["tar_bytes"] or [row["original_name"] for row in value["members"]] != [row["name"] for row in info["members"]]:
        raise ValueError("Original canonical member order differs")
    group = []; inspected = []
    for row, expected in zip(value["members"], info["members"]):
        _check(deadline)
        if (row["original_bytes"], row["original_sha256"]) != (expected["bytes"], expected["sha256"]):
            raise ValueError("Original member binding differs")
        mode = modes[row["original_name"]]
        if (mode == "opaque_gzip") != (row["transport_mode"] == "opaque_gzip"):
            raise ValueError("Common9 and opaque gzip modes may not be exchanged")
        payload = transported[row["transport_name"]]
        row_counts = {record["row_count"] for record in records if record["member"] == row["original_name"]}
        if len(row_counts) != 1:
            raise ValueError("Shared original member row counts disagree")
        raw = _decode_numeric(numeric, payload, row["original_bytes"], row["original_sha256"], row_counts.pop()) if row["transport_mode"] == "numeric_common9" else payload
        if len(raw) != row["original_bytes"] or _sha(raw) != row["original_sha256"]:
            raise ValueError("Restored original member byte SHA differs")
        inspected.append(_validate_payload(core, row["original_name"], raw, mode, records))
        group.append((row["original_name"], raw))
    raw = core._tar_bytes(group)
    if len(raw) != info["tar_bytes"] or _sha(raw) != value["original_tar_sha256"]:
        raise ValueError("Reconstructed original tar byte SHA differs")
    _check(deadline)
    blob = lzma.compress(raw, format=lzma.FORMAT_XZ, preset=6, check=lzma.CHECK_CRC64)
    if len(blob) != value["binding"]["original_archive_bytes"] or _sha(blob) != value["binding"]["original_archive_sha256"]:
        raise ValueError("Original XZ byte SHA differs; reproducible liblzma environment required")
    _check(deadline)
    if _source_pins() != value["source_pins"]:
        raise ValueError("Codec source changed during restore")
    return value, blob, inspected


def restore(manifest_path, destination, *, max_output_bytes=MAX_ARCHIVE, deadline=None):
    manifest_path = _plain(manifest_path); value = _load_manifest(manifest_path)
    _int(max_output_bytes, MAX_ARCHIVE)
    if value["binding"]["original_archive_bytes"] > max_output_bytes:
        raise ValueError("Original archive restore output budget exceeded")
    target = _inside(destination, value["binding"]["original_archive_repository_relative_path"])
    core, _ = load_dependencies()
    if target.exists() and (not target.is_file() or target.stat().st_size != value["binding"]["original_archive_bytes"] or core.sha_file(target) != value["binding"]["original_archive_sha256"]):
        raise ValueError("Existing original archive differs; never overwrite")
    value, blob, _ = _rebuild(manifest_path, deadline=deadline, manifest=value)
    target = _inside(destination, value["binding"]["original_archive_repository_relative_path"])
    matching = target.exists()
    if matching:
        if not target.is_file() or target.stat().st_size != len(blob) or core.sha_file(target) != _sha(blob):
            raise ValueError("Existing original archive changed during restore")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target = _inside(destination, value["binding"]["original_archive_repository_relative_path"])
        with target.open("xb") as stream:
            stream.write(blob)
        if core.sha_file(target) != _sha(blob):
            raise ValueError("Written original archive SHA differs")
    return dict(status="PASS_ARCHIVE_BYTE_TRANSPORT", restored=int(not matching), already_matching=int(matching),
                original_members_tar_and_XZ_byte_SHA_verified=True, original_publication_gate_pass_claimed=False,
                original_archive_repository_relative_path=value["binding"]["original_archive_repository_relative_path"])


def inspect_archive(path):
    path = _plain(path)
    value, _, inspected = _rebuild(path.with_name("MANIFEST.json"))
    if path.name != value["archive"]["file"]:
        raise ValueError("Unbound transport inspection path")
    return inspected


def _unchanged(state):
    core = state["core"]
    if (core.sha_file(state["selection"]) != state["selection_sha256"] or _source_pins() != state["source_pins"]
            or state["archive_path"].stat().st_size != state["binding"]["original_archive_bytes"]
            or core.sha_file(state["archive_path"]) != state["binding"]["original_archive_sha256"]
            or state["manifest_path"].stat().st_size != state["binding"]["original_manifest_bytes"]
            or core.sha_file(state["manifest_path"]) != state["binding"]["original_manifest_sha256"]):
        raise ValueError("Selected source/selection/codec changed during transport")


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def pack(repo_root, selection, selection_sha, output, *, wall_seconds=1100):
    if isinstance(wall_seconds, bool) or not isinstance(wall_seconds, (int, float)) or not 0 < wall_seconds <= 1100:
        raise ValueError("Transport wall budget must be within 1100 seconds")
    deadline = min(time.time() + wall_seconds, DEADLINE); _check(deadline)
    state = metadata(repo_root, selection, selection_sha); core = state["core"]; numeric = state["numeric"]
    output = _plain(output)
    if output.exists() or output.is_relative_to(state["repo"]):
        raise ValueError("Fresh explicit private output outside original repository required")
    members = core._archive_members(state["original"], state["archive_path"])
    if list(members) != [row["name"] for row in state["info"]["members"]]:
        raise ValueError("Original tar and manifest member order differ")
    old_tar = core._tar_bytes(list(members.items()))
    regenerated = lzma.compress(old_tar, format=lzma.FORMAT_XZ, preset=6, check=lzma.CHECK_CRC64)
    if len(regenerated) != state["binding"]["original_archive_bytes"] or _sha(regenerated) != state["binding"]["original_archive_sha256"]:
        raise ValueError("Selected original archive has no exact supported XZ6 recipe")
    group = [(ORIGINAL_MANIFEST_MEMBER, state["manifest_raw"])]; descriptors = []
    for name, payload in members.items():
        _check(deadline)
        mode = state["modes"][name]; _validate_payload(core, name, payload, mode, state["records"])
        encoded = numeric.encode_common9(payload, deadline=deadline) if mode == "csv_common9" else None
        transport_mode = "numeric_common9" if encoded is not None else "raw_common9" if mode == "csv_common9" else "opaque_gzip"
        encoded = payload if encoded is None else encoded
        suffix = {"numeric_common9": ".numeric", "raw_common9": ".common9", "opaque_gzip": ".csv.gz"}[transport_mode]
        transport_name = "members/" + _sha(encoded) + suffix
        descriptors.append(dict(original_name=name, original_bytes=len(payload), original_sha256=_sha(payload),
                                transport_name=transport_name, transport_bytes=len(encoded), transport_sha256=_sha(encoded), transport_mode=transport_mode))
        group.append((transport_name, encoded))
    _tar_size(len(raw) for _, raw in group)
    raw = core._tar_bytes(group); _check(deadline)
    packed = lzma.compress(raw, format=lzma.FORMAT_XZ, preset=6, check=lzma.CHECK_CRC64)
    if not 0 < len(packed) <= MAX_ARCHIVE:
        raise ValueError("Single transport exceeds compressed archive budget")
    archive_name = state["archive_path"].name.replace(".tar.xz", ".transport.xz")
    value = dict(format=SCHEMA, status="TRANSPORT_CANDIDATE", source_pins=state["source_pins"], selection_sha256=selection_sha,
                 binding=state["binding"], original_writer_source_sha256=state["original"]["source_sha256"],
                 original_tar_bytes=len(old_tar), original_tar_sha256=_sha(old_tar), xz_recipe=XZ_RECIPE,
                 members=descriptors, archive=dict(file=archive_name, bytes=len(packed), sha256=_sha(packed), tar_bytes=len(raw), tar_sha256=_sha(raw),
                 members=[dict(name=name, bytes=len(payload), sha256=_sha(payload)) for name, payload in group]),
                 original_publication_gate_pass_claimed=False, individual_original_gzip_regeneration_executed=False,
                 existing_original_files_modified=False, statistics_values_rounded_or_rows_dropped=False,
                 legacy_publication_inspector_compatible=False, new_Task_model_array_or_T_target_operations=0)
    _check(deadline); _unchanged(state)
    output.mkdir(parents=True, exist_ok=False)
    with (output / archive_name).open("xb") as stream:
        stream.write(packed)
    draft = output / "CANDIDATE_MANIFEST.json"; _write_json(draft, value)
    with tempfile.TemporaryDirectory(prefix="stats_transport_exact_") as temp:
        first = restore(draft, Path(temp), deadline=deadline)
        repeated = restore(draft, Path(temp), deadline=deadline)
        if first["restored"] != 1 or repeated["already_matching"] != 1:
            raise ValueError("Separate exact archive restore/repeat proof incomplete")
    _check(deadline); _unchanged(state)
    value["status"] = "PASS_ARCHIVE_BYTE_TRANSPORT"
    proof = dict(status="PASS_ARCHIVE_BYTE_TRANSPORT", separate_restore=first, repeated_restore=repeated,
                 original_archive_bytes=state["binding"]["original_archive_bytes"], transport_archive_bytes=len(packed),
                 archive_only_bytes_saved=state["binding"]["original_archive_bytes"] - len(packed),
                 sidecar_bytes_not_in_savings=True, original_source_and_manifest_unchanged=True,
                 original_publication_gate_pass_claimed=False, actual_experiment_data_model_Task_or_T_target_operations=0)
    _write_json(output / "PROOF.json", proof)
    readme = ("# Legacy public-statistic archive byte transport\n\n"
              "This is a third transport format. The frozen publication inspector does not support it; independent publication admission is required.\n"
              "The original archive and original MANIFEST remain unchanged locally. No removal, staging, or publication is performed.\n\n"
              "Restore the exact original repository-relative part into a clone or separate private directory:\n\n"
              "```text\npython research/" + HERE.name + "/stats_pack_transport.py restore --manifest TRANSPORT_DIR/MANIFEST.json --destination CLONE_ROOT\n```\n\n"
              "Restored part: `" + state["binding"]["original_archive_repository_relative_path"] + "`.\n"
              "The embedded original MANIFEST, each original member, canonical tar header/order/padding, and original XZ bytes are SHA-verified.\n"
              "The original preset6/CRC64 XZ bytes require a reproducible liblzma environment; a mismatch fails without writing.\n"
              "Exact existing destinations are read and skipped; mismatches are never overwritten. Original sufficient-statistic gzips are restored separately by the original stats_pack.py and original MANIFEST.\n"
              "Only already-public aggregate members are transported, with exact numeric/literal or raw common9 fallback and opaque gzip bytes. No rounding, row removal, new targets, model access, or scientific re-evaluation occurs.\n"
              "Archive-only savings exclude JSON/README sidecars. This byte-transport PASS does not replace the old publication or scientific gates.\n")
    with (output / "README.md").open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(readme)
    _check(deadline); _unchanged(state)
    _write_json(output / "MANIFEST.json", value)
    return proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    command = sub.add_parser("pack")
    command.add_argument("--repo-root", required=True); command.add_argument("--selection", required=True)
    command.add_argument("--selection-sha", required=True); command.add_argument("--output"); command.add_argument("--execute", action="store_true")
    command.add_argument("--wall-seconds", type=float, default=1100)
    command = sub.add_parser("restore")
    command.add_argument("--manifest", required=True); command.add_argument("--destination", required=True)
    command.add_argument("--max-output-bytes", type=int, default=MAX_ARCHIVE); command.add_argument("--deadline", type=float)
    command = sub.add_parser("inspect"); command.add_argument("--archive", required=True)
    args = parser.parse_args(argv)
    if args.command == "pack":
        if args.execute:
            if not args.output:
                parser.error("Actual transport pack requires a fresh explicit private output")
            result = pack(args.repo_root, args.selection, args.selection_sha, args.output, wall_seconds=args.wall_seconds)
        else:
            state = metadata(args.repo_root, args.selection, args.selection_sha)
            result = dict(status="METADATA_ONLY_TRANSPORT_SELECTION", selected_archives=1, selected_members=len(state["info"]["members"]), statistics_content_read=False, compressions=0, files_written=0)
    elif args.command == "restore":
        result = restore(args.manifest, args.destination, max_output_bytes=args.max_output_bytes, deadline=args.deadline)
    elif args.command == "inspect":
        result = inspect_archive(args.archive)
    else:
        result = dict(status="METADATA_ONLY_ARCHIVE_TRANSPORT", format=SCHEMA, selection_format=SELECTION_SCHEMA,
                      statistics_content_read=False, compressions=0, files_written=0, numpy_imported="numpy" in sys.modules,
                      torch_imported="torch" in sys.modules, legacy_publication_inspector_compatible=False,
                      original_core_sha256=CORE_SHA, numeric_codec_sha256=NUMERIC_SHA)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
