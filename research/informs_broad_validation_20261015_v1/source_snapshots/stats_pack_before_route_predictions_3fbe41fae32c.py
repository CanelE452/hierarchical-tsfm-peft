"""Bounded, byte-exact public error-statistic packs; no model/data-array access."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import lzma
import math
from pathlib import Path, PurePosixPath
import re
import struct
import tarfile
import tempfile
import time
import zlib


COLUMNS = ("run_id", "split", "cohort", "origin", "channel", "horizon_start", "horizon_end", "sse", "sae", "count")
CSV_HEADER = (",".join(COLUMNS) + "\r\n").encode()
MAX_ARCHIVE = 20 * 1024**2 - 1
MAX_MEMBER = 64 * 1024**2
MAX_TAR = 128 * 1024**2
MAX_OUTPUT = 3 * 1024**3
MAX_MANIFEST = 8 * 1024**2
SCHEMA = "public_error_stats_pack_v1"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_name(name):
    p = PurePosixPath(name)
    if not name or "\\" in name or p.is_absolute() or any(part in {"", ".", ".."} or ":" in part for part in name.split("/")):
        raise ValueError("Unsafe relative path")
    return p.as_posix()


def _inside(root, name):
    name = _safe_name(name); root = Path(root).resolve(); path = root / name
    for part in (path, *path.parents):
        if part == root: break
        if part.is_symlink(): raise ValueError("Symlink in restore/evidence path")
    if not path.resolve().is_relative_to(root): raise ValueError("Path escapes destination")
    return path


def _gunzip(blob):
    with gzip.GzipFile(fileobj=io.BytesIO(blob)) as stream:
        raw = stream.read(MAX_MEMBER + 1)
    if len(raw) > MAX_MEMBER: raise ValueError("CSV decompression limit exceeded")
    return raw


def _csv_schema(raw, common=False):
    text = raw.decode("utf-8")
    rows = csv.reader(io.StringIO(text, newline=""))
    if tuple(next(rows, ())) != COLUMNS[1 if common else 0:]: raise ValueError("Error statistic CSV schema differs")
    n = 0
    for row in rows:
        fields = row if common else row[1:]
        if len(fields) != 9 or (not common and not row[0]): raise ValueError("Error statistic column count differs")
        if not fields[0] or not fields[1] or not fields[3]: raise ValueError("Missing statistic identity")
        origin, start, end, count = (int(fields[i]) for i in (2, 4, 5, 8))
        sse, sae = float(fields[6]), float(fields[7])
        if origin < 0 or start < 0 or end <= start or not 0 <= count <= end - start or not math.isfinite(sse + sae) or sse < 0 or sae < 0:
            raise ValueError("Invalid error statistic values")
        n += 1
    if n == 0: raise ValueError("Empty error statistics")
    return text, n


def regenerate_gzip(raw, recipe):
    compressor = zlib.compressobj(recipe["level"], zlib.DEFLATED, -15, recipe["memLevel"], zlib.Z_DEFAULT_STRATEGY)
    body = compressor.compress(raw)
    if recipe["sync_flush_before_finish"]: body += compressor.flush(zlib.Z_SYNC_FLUSH)
    body += compressor.flush(zlib.Z_FINISH)
    return bytes.fromhex(recipe["header_hex"]) + body + bytes.fromhex(recipe["trailer_hex"])


def find_gzip_recipe(blob, raw):
    if blob[:3] != b"\x1f\x8b\x08" or len(blob) < 18 or blob[3] & 0xE0: return None
    offset, flags = 10, blob[3]
    try:
        if flags & 4: offset += 2 + struct.unpack("<H", blob[offset:offset + 2])[0]
        if flags & 8: offset = blob.index(b"\0", offset) + 1
        if flags & 16: offset = blob.index(b"\0", offset) + 1
        if flags & 2: offset += 2
        header = blob[:offset].hex(); trailer = blob[-8:].hex()
        for level in (9, 6, 1):
            for sync in (True, False):
                recipe = dict(level=level, wbits=-15, memLevel=8, strategy="Z_DEFAULT_STRATEGY",
                              sync_flush_before_finish=sync, header_hex=header, trailer_hex=trailer,
                              zlib_compile_version=zlib.ZLIB_VERSION, zlib_runtime_version=zlib.ZLIB_RUNTIME_VERSION)
                if regenerate_gzip(raw, recipe) == blob: return recipe
    except (ValueError, IndexError, struct.error, zlib.error):
        return None
    return None


def bounded_xz_decompress(blob, limit=MAX_TAR):
    if len(blob) > MAX_ARCHIVE: raise ValueError("Compressed archive limit exceeded")
    decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=256 * 1024**2)
    raw = decoder.decompress(blob, max_length=limit + 1)
    if len(raw) > limit: raise ValueError("XZ decompression limit exceeded")
    if not decoder.eof or decoder.unused_data: raise ValueError("Truncated or concatenated XZ archive")
    return raw


def _load_manifest(path):
    path = Path(path)
    if path.stat().st_size > MAX_MANIFEST: raise ValueError("Manifest limit exceeded")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("format") != SCHEMA or len(value.get("records", [])) > 10000 or len(value.get("archives", [])) > 10000:
        raise ValueError("Unsupported pack manifest")
    prefix = _safe_name(value["campaign_repository_relative"])
    archive_names = [a["file"] for a in value["archives"]]
    if len(set(archive_names)) != len(archive_names) or any(re.fullmatch(r"part_\d{4}\.tar\.xz", name) is None for name in archive_names):
        raise ValueError("Invalid or duplicate archive filename")
    seen = set()
    for record in value["records"]:
        name = _safe_name(record["repository_relative_path"])
        rel = PurePosixPath(name).relative_to(prefix)
        if rel.parts[0] not in {"predictions", "ordered_predictions", "route_replays", "raw_calibration"} or rel.name != "PUBLIC_ERROR_STATS.csv.gz":
            raise ValueError("Pack source is outside public sufficient-statistic scope")
        if name.casefold() in seen: raise ValueError("Duplicate original statistic path")
        seen.add(name.casefold())
        if not 0 < record["gzip_bytes"] <= MAX_ARCHIVE or not 0 < record["csv_bytes"] <= MAX_MEMBER:
            raise ValueError("Declared statistic size limit exceeded")
        if record["storage_mode"] not in {"csv_common9", "opaque_gzip"}: raise ValueError("Unsupported packed storage mode")
        if record["archive"] not in archive_names or re.fullmatch(r"members/[0-9a-f]{64}\.csv(?:\.gz)?", record["member"]) is None:
            raise ValueError("Unbound or unsafe member path")
    return value


def _archive_members(manifest, path):
    infos = [a for a in manifest["archives"] if a["file"] == path.name]
    if len(infos) != 1: raise ValueError("Archive has no unique manifest binding")
    info = infos[0]
    if path.stat().st_size != info["bytes"] or sha_file(path) != info["sha256"]: raise ValueError("Archive SHA/size differs")
    if not 0 < info["tar_bytes"] <= MAX_TAR: raise ValueError("Declared tar limit exceeded")
    raw = bounded_xz_decompress(path.read_bytes(), info["tar_bytes"])
    if len(raw) != info["tar_bytes"]: raise ValueError("Expanded tar size differs")
    expected = {v["name"]: v for v in info["members"]}
    if len(expected) != len(info["members"]) or len(expected) > 10000: raise ValueError("Invalid member count")
    result = {}
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        for member in archive:
            name = _safe_name(member.name)
            if not member.isfile() or member.pax_headers or name not in expected or name in result:
                raise ValueError("Unexpected or nonregular tar member")
            declaration = expected[name]
            if not 0 < member.size <= MAX_MEMBER or member.size != declaration["bytes"]: raise ValueError("Member size limit differs")
            payload = archive.extractfile(member).read(member.size + 1)
            if len(payload) != member.size or sha(payload) != declaration["sha256"]: raise ValueError("Member SHA differs")
            result[name] = payload
    if set(result) != set(expected): raise ValueError("Missing tar members")
    if set(result) != {r["member"] for r in manifest["records"] if r["archive"] == path.name}: raise ValueError("Unbound archive content")
    if raw != _tar_bytes(list(result.items())): raise ValueError("Noncanonical tar metadata, padding, or trailing content")
    return result


def inspect_archive(path):
    """No-extract publication API; returns bounded validated statistic CSV text."""
    path = Path(path)
    if not path.name.endswith(".tar.xz") or path.parent.parent.name != "stats_packs" or not path.parent.name.startswith("pack_"):
        raise ValueError("Only manifest-bound stats_packs/pack_* archives are supported")
    manifest = _load_manifest(path.with_name("MANIFEST.json"))
    members = _archive_members(manifest, path)
    result = []
    for name, payload in members.items():
        records = [r for r in manifest["records"] if r["archive"] == path.name and r["member"] == name]
        modes = {r["storage_mode"] for r in records}
        if len(modes) != 1: raise ValueError("Shared member storage modes differ")
        mode = modes.pop()
        text, _ = _csv_schema(payload if mode == "csv_common9" else _gunzip(payload), common=mode == "csv_common9")
        for record in records: _original_csv(record, payload)
        result.append(dict(member_name=name, payload_bytes=len(payload), sha256=sha(payload), csv_text=text, storage_mode=mode))
    return result


def _original_csv(record, payload):
    if record["storage_mode"] == "opaque_gzip":
        raw = _gunzip(payload)
    else:
        prefix = record["run_id_prefix"].encode("utf-8")
        if not prefix.endswith(b",") or b"\n" in prefix or b"\r" in prefix or b'"' in prefix or prefix.count(b",") != 1:
            raise ValueError("Invalid run ID prefix")
        lines = payload.splitlines(keepends=True)
        raw = b"run_id," + lines[0] + b"".join(prefix + line for line in lines[1:])
    if len(raw) != record["csv_bytes"] or sha(raw) != record["csv_sha256"]: raise ValueError("Restored CSV SHA differs")
    _csv_schema(raw)
    return raw


def _restore_blob(record, payload):
    raw = _original_csv(record, payload)
    blob = payload if record["storage_mode"] == "opaque_gzip" else regenerate_gzip(raw, record["gzip_recipe"])
    if len(blob) != record["gzip_bytes"] or sha(blob) != record["gzip_sha256"]:
        raise ValueError("Original gzip byte SHA differs; use the recorded zlib environment")
    return blob


def restore(manifest_path, destination, *, max_output_bytes=MAX_OUTPUT, deadline=None):
    manifest_path = Path(manifest_path); manifest = _load_manifest(manifest_path)
    if sum(r["gzip_bytes"] for r in manifest["records"]) > min(max_output_bytes, MAX_OUTPUT): raise ValueError("Restore output budget limit exceeded")
    destination = Path(destination).resolve()
    for record in manifest["records"]:
        target = _inside(destination, record["repository_relative_path"])
        if target.exists() and (not target.is_file() or sha_file(target) != record["gzip_sha256"]):
            raise ValueError("Existing destination differs; never overwrite")
    restored, matching = 0, 0
    for info in manifest["archives"]:
        if deadline is not None and time.time() >= deadline: raise TimeoutError("Restore wall limit reached; existing exact files remain reusable")
        archive_path = _inside(manifest_path.parent, info["file"])
        members = _archive_members(manifest, archive_path)
        for record in [r for r in manifest["records"] if r["archive"] == info["file"]]:
            blob = _restore_blob(record, members[record["member"]])
            target = _inside(destination, record["repository_relative_path"])
            if target.exists():
                if sha_file(target) != record["gzip_sha256"]: raise ValueError("Existing destination changed during restore")
                matching += 1; continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream: stream.write(blob)
            restored += 1
    return dict(status="PASS", restored=restored, already_matching=matching, CSV_and_original_gzip_SHA_verified=True)


def _tar_bytes(group):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name, payload in group:
            info = tarfile.TarInfo(name); info.size = len(payload); info.mtime = 0
            tar.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _emit(group, directory, archives):
    raw = _tar_bytes(group)
    if len(raw) > MAX_TAR: raise ValueError("Expanded tar group limit exceeded")
    packed = lzma.compress(raw, format=lzma.FORMAT_XZ, preset=6)
    if len(packed) > MAX_ARCHIVE:
        if len(group) == 1: raise ValueError("Single sufficient-statistic member exceeds archive limit")
        middle = len(group) // 2
        return {**_emit(group[:middle], directory, archives), **_emit(group[middle:], directory, archives)}
    filename = f"part_{len(archives):04d}.tar.xz"
    with (directory / filename).open("xb") as stream: stream.write(packed)
    archives.append(dict(file=filename, bytes=len(packed), sha256=sha(packed), tar_bytes=len(raw),
                         members=[dict(name=name, bytes=len(payload), sha256=sha(payload)) for name, payload in group]))
    return {name: filename for name, payload in group}


def pack(campaign, *, only=None, file_list=None, wall_seconds=1100):
    campaign = Path(campaign).resolve(); repo = campaign.parents[1]
    started = time.time(); deadline = started + min(float(wall_seconds), 1100)
    import publication_check as publication
    if file_list is not None:
        base_names = Path(file_list).read_text(encoding="utf-8").splitlines()
    else:
        base_names, _ = publication.candidate_files(campaign)
    candidates = []
    relative_names = list(only) if only is not None else [str(PurePosixPath(n).relative_to(campaign.relative_to(repo).as_posix())) for n in base_names
                                                       if n.endswith("/PUBLIC_ERROR_STATS.csv.gz")]
    for name in relative_names:
        path = _inside(campaign, name)
        if path.name != "PUBLIC_ERROR_STATS.csv.gz" or Path(name).parts[0] not in {"predictions", "ordered_predictions", "route_replays", "raw_calibration"}:
            raise ValueError("Only standard public sufficient-statistic gzip files may be packed")
        candidates.append(path)
    if len(set(candidates)) != len(candidates): raise ValueError("Duplicate source path")
    if not candidates: raise ValueError("No standard sufficient statistics selected")
    directory = campaign / "stats_packs" / ("pack_" + str(time.time_ns()))
    directory.mkdir(parents=True, exist_ok=False)
    records, archives, group, seen, assignment = [], [], [], set(), {}
    for path in sorted(candidates):
        if time.time() >= deadline: raise TimeoutError("Pack wall limit reached; incomplete pack is not a public payload")
        blob = path.read_bytes()
        if len(blob) > MAX_ARCHIVE: raise ValueError("Original gzip file limit exceeded")
        raw = _gunzip(blob); _, row_count = _csv_schema(raw)
        recipe = find_gzip_recipe(blob, raw)
        lines = raw.splitlines(keepends=True); prefix = lines[1].split(b",", 1)[0] + b","
        constant = all(line.startswith(prefix) for line in lines[1:]) and b'"' not in prefix
        if recipe is not None and constant:
            payload = lines[0].split(b",", 1)[1] + b"".join(line[len(prefix):] for line in lines[1:])
            mode, suffix = "csv_common9", ".csv"
        else:
            payload, mode, suffix, recipe = blob, "opaque_gzip", ".csv.gz", None
        member = "members/" + sha(payload) + suffix
        if member not in seen:
            if group and sum(len(v) for n, v in group) + len(payload) > 64 * 1024**2:
                assignment.update(_emit(group, directory, archives)); group = []
            group.append((member, payload)); seen.add(member)
        record = dict(repository_relative_path=path.relative_to(repo).as_posix(), member=member, storage_mode=mode,
                      gzip_bytes=len(blob), gzip_sha256=sha(blob), csv_bytes=len(raw), csv_sha256=sha(raw), row_count=row_count,
                      gzip_recipe=recipe, run_id_prefix=prefix.decode("utf-8") if mode == "csv_common9" else None,
                      same_environment_original_gzip_restore_verified=True)
        if _restore_blob(record, payload) != blob: raise ValueError("Pack source regeneration differs")
        records.append(record)
    if group: assignment.update(_emit(group, directory, archives))
    for record in records: record["archive"] = assignment[record["member"]]
    manifest = dict(format=SCHEMA, status="PENDING_RESTORE_VERIFICATION", source_sha256=sha_file(Path(__file__)), created_unix=time.time(),
        campaign_repository_relative=campaign.relative_to(repo).as_posix(), records=records, archives=archives,
        zlib_compile_version=zlib.ZLIB_VERSION, zlib_runtime_version=zlib.ZLIB_RUNTIME_VERSION,
        compression="stdlib XZ preset6; solid USTAR; exact common9-column members deduplicated by SHA",
        original_total_bytes=sum(r["gzip_bytes"] for r in records), packed_total_bytes=sum(a["bytes"] for a in archives),
        max_archive_bytes=MAX_ARCHIVE, max_expanded_tar_bytes=MAX_TAR, max_member_bytes=MAX_MEMBER,
        original_files_modified=False, NPZ_or_weights_included=False, array_values_read=False,
        originals_missing_in_clone_until_restore=True, cross_environment_CSV_SHA_portable=True,
        original_gzip_SHA_requires_reproducible_zlib=True, separate_temporary_restore_all_sha_verified=False)
    manifest_path = directory / "MANIFEST.json"
    draft_path = directory / ".MANIFEST_VERIFY.json"
    with draft_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    # Verification restores a separate disposable tree, never the source campaign.
    with tempfile.TemporaryDirectory(prefix="stats_pack_verify_") as temp:
        result = restore(draft_path, Path(temp), deadline=deadline)
        if result["restored"] != len(records): raise ValueError("Separate restore did not cover every statistic")
    if time.time() >= deadline: raise TimeoutError("Pack verification wall limit reached; no complete manifest published")
    manifest.update(status="PASS", separate_temporary_restore_all_sha_verified=True)
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    for archive in archives: inspect_archive(directory / archive["file"])
    selected = set(base_names) - {p.relative_to(repo).as_posix() for p in candidates}
    output_files = ("MANIFEST.json", "PAYLOAD_FILES.txt", "README_KO.md", "MARKDOWN_LINK_REPLACEMENTS.json")
    selected.update((directory / name).relative_to(repo).as_posix() for name in output_files)
    selected.update((directory / a["file"]).relative_to(repo).as_posix() for a in archives)
    (directory / "PAYLOAD_FILES.txt").write_text("\n".join(sorted(selected)) + "\n", encoding="utf-8")
    removed = {path.resolve(): path.relative_to(repo).as_posix() for path in candidates}; links = []
    from urllib.parse import unquote, urlsplit
    for path in campaign.rglob("*.md"):
        for match in publication.MD_LINK.finditer(path.read_text(encoding="utf-8-sig")):
            target = match.group(1) or match.group(2); parsed = urlsplit(target)
            if parsed.scheme: continue
            resolved = (path.parent / unquote(parsed.path)).resolve()
            if resolved in removed: links.append(dict(document=path.relative_to(repo).as_posix(), original_target=target,
                                                     restored_file=removed[resolved], action="Point public documentation to pack index and restore instructions"))
    (directory / "MARKDOWN_LINK_REPLACEMENTS.json").write_text(json.dumps(links, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "README_KO.md").write_text("# 공개 충분통계 pack\n\nClone 직후 원 PUBLIC_ERROR_STATS.csv.gz 경로는 비어 있다. 아래 restore 명령을 실행해야 원 상대 경로와 원 gzip byte가 복원된다.\n\n숫자·행·run ID·CSV byte를 보존한다. 같은 zlib 환경에서 원 gzip byte의 SHA를 별도 임시 디렉터리 복원으로 전수 확인했다. zlib가 다르면 원 gzip 재생성이 달라질 수 있으며 덮어쓰지 않고 실패한다.\n\n- [원본 SHA·형식·archive manifest](MANIFEST.json)\n- [공개 payload 목록](PAYLOAD_FILES.txt)\n- [수정이 필요한 기존 Markdown 링크 목록](MARKDOWN_LINK_REPLACEMENTS.json)\n\n저장소 루트에서 다음 명령을 사용한다.\n\n```text\npython research/" + campaign.name + "/stats_pack.py restore --manifest " + manifest_path.relative_to(repo).as_posix() + " --destination .\n```\n\n이미 일치하는 파일은 읽기만 하며, 불일치 파일을 덮어쓰지 않는다. 원본은 로컬에서 삭제하거나 변경하지 않았다.\n", encoding="utf-8")
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="action", required=True)
    build = sub.add_parser("pack"); build.add_argument("--campaign", type=Path, default=Path(__file__).resolve().parent)
    build.add_argument("--file-list", type=Path); build.add_argument("--only", action="append"); build.add_argument("--wall-seconds", type=float, default=1100)
    unpack = sub.add_parser("restore"); unpack.add_argument("--manifest", type=Path, required=True); unpack.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "pack":
        directory = pack(args.campaign, only=args.only, file_list=args.file_list, wall_seconds=args.wall_seconds)
        print(json.dumps(dict(status="PASS", pack=directory.relative_to(args.campaign.resolve()).as_posix())))
    else: print(json.dumps(restore(args.manifest, args.destination)))


if __name__ == "__main__":
    main()
