"""Read Markdown local links against exact POSIX path spellings; stdlib only.

This is metadata inspection, never a final payload/privacy/science approval.
The optional repository-path-list must be supplied by ROOT from exact Git path
names when validating a clone or index; this tool never invokes Git.
"""
import argparse
import ast
import collections
import datetime
import hashlib
import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import sys
from urllib.parse import unquote, urlsplit

INLINE = re.compile(r'(?P<image>!?)\[[^\]\n]*\]\(\s*(?:<(?P<angle>[^>]+)>|(?P<plain>[^\s)]+))(?:\s+[^)]*)?\)')
REFERENCE = re.compile(r'^\s{0,3}\[[^\]]+\]:\s*(?:<([^>]+)>|(\S+))', re.MULTILINE)
FENCES = re.compile(r'(?ms)^\s*(```|~~~).*?^\s*\1\s*$')
PRIVATE_NAMES = {".git", ".cache", "__pycache__", ".pytest_cache", ".venv", "private", "private_cache", "weights", "checkpoints", "raw"}
EXTRA_EXCLUDES = (
    "diagnostics/initialization_anchor_1792059651372799700",
    "execution_batches/cpu_V_checkpoint_replay_1792060139101125900.json",
    "SOURCE_PATCH_CONTROL_LR_METADATA.json",
    "source_snapshots/CONTROL_LR_METADATA_ROOT_PYTEST_INCIDENT.json",
    "source_snapshots/AUDIT_PROTOCOL_KO_before_legacy_namespace_675bcf808bd4.md",
    "source_snapshots/COST_PROTOCOL_KO_before_cost_final_reserve_3100c3f435e3.md",
    "publication_projections/snapshot_20261015T104409Z_1792061049890264400/README_KO.md",
)
SOURCE_LINK_GAP = {
    "source": "publication_check.py",
    "source_sha256": "65cb9a05da3bc4a981db83733633704345a71dadeafd2f2d7bb10318b0a7cbb3",
    "lines": [149, 178],
    "controlled_read_only_probe": {
        "document": "PUBLIC_EXCLUSIONS_KO.md",
        "synthetic_target_not_written_to_any_document": "PUBLICATION_CHECK.PY",
        "exact_existing_filename": "publication_check.py",
        "Windows_Path_exists": True,
        "Windows_Path_resolve_name": "publication_check.py",
        "actual_extracted_link_function_failures": [],
        "accepted_basis": "selected",
        "framework_imports": [],
    },
    "scope": "Markdown inline/reference, quoted HTML href/src and SVG href all call _link; Windows resolve canonicalizes wrong case before exact set comparison.",
}

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def source_constants(path):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    wanted = {"EXCLUDED_NAMES", "EXCLUDED_PREFIXES", "ALLOWED_SUFFIXES"}
    return {target.id: ast.literal_eval(node.value)
            for node in tree.body if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Name) and target.id in wanted}

def paths_file(path):
    result = set()
    if path is None:
        return result
    for value in Path(path).read_text(encoding="utf-8-sig").splitlines():
        if not value:
            continue
        p = PurePosixPath(value)
        if "\\" in value or p.is_absolute() or ".." in p.parts or re.match(r"^[A-Za-z]:", value):
            raise ValueError("Path list must contain safe exact repository-relative POSIX names")
        result.add(p.as_posix())
    return result

def prefixes(files):
    result = {""}
    for name in files:
        parts = PurePosixPath(name).parts
        result.update("/".join(parts[:i]) for i in range(1, len(parts)))
    return result

def physical_paths(root):
    files, directories, symlinks = set(), {""}, set()
    for directory, subdirs, names in os.walk(root, followlinks=False):
        current = Path(directory)
        relative = current.relative_to(root)
        for name in subdirs:
            item = current / name
            rel = (relative / name).as_posix()
            directories.add(rel)
            if item.is_symlink():
                symlinks.add(rel)
        subdirs[:] = [name for name in subdirs if name.lower() not in PRIVATE_NAMES
                      and not (current / name).is_symlink()]
        for name in names:
            item = current / name
            rel = (relative / name).as_posix()
            files.add(rel)
            if item.is_symlink():
                symlinks.add(rel)
    return files, directories, symlinks

def blank_preserving_lines(match):
    return re.sub(r"[^\n]", " ", match.group())

class HtmlLinks(HTMLParser):
    def __init__(self, text):
        super().__init__(convert_charrefs=True)
        self.items = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key.lower() in {"href", "src"} and value is not None:
                self.items.append({"line": self.getpos()[0], "kind": "html_image" if tag.lower() == "img" and key.lower() == "src" else "html_" + key.lower(), "target": value})

def markdown_links(text):
    markup = FENCES.sub(blank_preserving_lines, text)
    markup = re.sub(r"(`+)[^\n]*?\1", blank_preserving_lines, markup)
    found = []
    for match in INLINE.finditer(markup):
        found.append({"line": markup.count("\n", 0, match.start()) + 1,
                      "kind": "markdown_image" if match.group("image") else "markdown_link",
                      "target": match.group("angle") or match.group("plain")})
    for match in REFERENCE.finditer(markup):
        found.append({"line": markup.count("\n", 0, match.start()) + 1,
                      "kind": "markdown_reference_definition",
                      "target": match.group(1) or match.group(2)})
    found.extend(HtmlLinks(markup).items)
    return found

def normalized_target(document, raw):
    value = html.unescape(raw.strip())
    if not value or value.startswith("#"):
        return None, "ANCHOR_ONLY", None
    parsed = urlsplit(value)
    if parsed.scheme in {"http", "https", "mailto", "tel", "data"} or parsed.netloc:
        return None, "EXTERNAL", None
    if parsed.scheme or value.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", value):
        return None, "ABSOLUTE_OR_UNSUPPORTED_SCHEME", None
    path = unquote(parsed.path).replace("\\", "/")
    # Split fragment/query before decoding; decode URL path once, not twice.
    if not path:
        return None, "SAME_DOCUMENT_FRAGMENT_OR_QUERY", None
    target = posixpath.normpath(posixpath.join(posixpath.dirname(document), path))
    if target == ".." or target.startswith("../") or target.startswith("/"):
        return None, "ESCAPES_REPOSITORY", None
    return target, None, {"fragment": parsed.fragment, "query_present": bool(parsed.query),
                          "percent_decoded": path != parsed.path, "double_decode_would_differ": unquote(path) != path}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--file-list", type=Path, help="Final payload paths; scans listed Markdown only")
    parser.add_argument("--tracked-file-list", type=Path, help="Exact existing HEAD paths ROOT supplies; no Git is run")
    parser.add_argument("--repository-path-list", type=Path, help="Exact union of HEAD/index/payload paths for POSIX case basis")
    parser.add_argument("--virtual-file-list", type=Path, help="Exact restore-only statistic paths; never considered clone-ready links")
    parser.add_argument("--output-report", required=True, type=Path)
    args = parser.parse_args()
    root, campaign = args.repository_root.resolve(), args.campaign.resolve()
    cp = campaign.relative_to(root).as_posix()
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    constants = source_constants(campaign / "publication_check.py")
    excluded = set(constants["EXCLUDED_NAMES"]) | set(constants["EXCLUDED_PREFIXES"]) | set(EXTRA_EXCLUDES)
    projection = campaign / "publication_projections/snapshot_20261015T104409Z_1792061049890264400/PROJECTION.json"
    duplicate_paths = set()
    if projection.is_file():
        value = json.loads(projection.read_text(encoding="utf-8"))
        excluded.update(value.get("explicit_intermediate_snapshot_excludes", []))
        duplicate_paths.update(item["original"]["file"] for item in value.get("duplicate_statistics", []))

    def excluded_document(name):
        if not name.startswith(cp + "/"):
            return False
        rel = name[len(cp) + 1:]
        return (any(part.lower() in PRIVATE_NAMES for part in PurePosixPath(rel).parts)
                or any(rel == p or rel.startswith(p.rstrip("/") + "/") for p in excluded))

    disk_files, disk_dirs, symlinks = physical_paths(root)
    supplied_inventory = paths_file(args.repository_path_list)
    inventory = supplied_inventory if args.repository_path_list else disk_files
    inventory_dirs = prefixes(inventory) if args.repository_path_list else disk_dirs
    payload = paths_file(args.file_list)
    tracked = paths_file(args.tracked_file_list)
    explicit_virtual = paths_file(args.virtual_file_list)
    selected_pack_metadata, restored_by_pack = [], set()
    for path in sorted((campaign / "stats_packs").glob("pack_*/MANIFEST.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        name = path.relative_to(root).as_posix()
        archive_names = [(path.parent / item["file"]).relative_to(root).as_posix() for item in value.get("archives", [])]
        bound_for_explicit_payload = bool(payload and name in payload and all(n in payload for n in archive_names))
        stats = {item["repository_relative_path"] for item in value.get("records", [])}
        if bound_for_explicit_payload:
            restored_by_pack.update(stats)
        selected_pack_metadata.append({"manifest": name, "manifest_status": value.get("status"), "record_count": len(stats),
                                      "archive_count": len(archive_names), "selected_with_all_archive_names": bound_for_explicit_payload,
                                      "archive_or_content_verification_performed": False})
    virtual = explicit_virtual | restored_by_pack
    all_names = inventory | inventory_dirs | virtual | prefixes(virtual)
    folded = collections.defaultdict(list)
    for name in all_names:
        folded[name.casefold()].append(name)
    scope = payload if args.file_list else {name for name in disk_files if name.startswith(cp + "/")}
    skipped_documents = sorted(name for name in scope if name.lower().endswith(".md") and excluded_document(name))
    documents = sorted(name for name in scope if name.lower().endswith(".md") and not excluded_document(name))
    findings, links, skipped, md_inputs = [], [], collections.Counter(), []
    published = payload | tracked

    for document in documents:
        path = root / document
        if not path.is_file():
            findings.append({"document": document, "rule": "DOCUMENT_MISSING_ON_DISK"})
            continue
        raw = path.read_bytes()
        md_inputs.append({"file": document, "bytes": len(raw), "sha256": digest(raw)})
        for item in markdown_links(raw.decode("utf-8-sig")):
            target, skip, decoded = normalized_target(document, item["target"])
            if skip:
                skipped[skip] += 1
                if skip in {"ABSOLUTE_OR_UNSUPPORTED_SCHEME", "ESCAPES_REPOSITORY"}:
                    findings.append({"document": document, "line": item["line"], "kind": item["kind"], "rule": skip,
                                     "target": "<absolute-local-target>" if skip.startswith("ABSOLUTE") else item["target"]})
                continue
            record = {"document": document, **item, "normalized_POSIX_target": target, **decoded}
            exact = target in all_names
            options = sorted(folded.get(target.casefold(), []))
            if not exact and options:
                record.update(rule="CASE_MISMATCH", exact_target_case_candidates=options)
                findings.append(record)
            elif not exact and target not in virtual:
                record.update(rule="MISSING_TARGET")
                findings.append(record)
            else:
                record["case_status"] = "EXACT" if exact else "EXACT_VIRTUAL_ONLY"
            if target in symlinks:
                findings.append({**record, "rule": "SYMLINK_TARGET"})
            if args.file_list:
                ready = target in published or target in prefixes(published)
                if not ready and target in virtual:
                    record["publication_status"] = "RESTORE_ONLY_NOT_CLONE_READY"
                    findings.append({**record, "rule": "RESTORE_ONLY_TARGET_REQUIRES_PACK_INDEX_LINK"})
                elif not ready:
                    record["publication_status"] = "NOT_SELECTED_OR_TRACKED"
                    findings.append({**record, "rule": "TARGET_NOT_PUBLISHED"})
                else:
                    record["publication_status"] = "SELECTED_OR_TRACKED"
            elif target.startswith(cp + "/") and target[len(cp) + 1:] in duplicate_paths:
                record["publication_status"] = "KNOWN_PROVEN_DUPLICATE_EXCLUDED"
            elif target.endswith("/PUBLIC_ERROR_STATS.csv.gz"):
                record["publication_status"] = "CURRENT_PHYSICAL_GZIP_FINAL_PACK_REPLACEMENT_REQUIRED"
            elif excluded_document(target):
                record["publication_status"] = "KNOWN_FINAL_EXCLUDED_TARGET"
            else:
                record["publication_status"] = "CURRENT_PHYSICAL_ONLY_FINAL_PAYLOAD_UNSET"
            links.append(record)

    report = {
        "status": "CASE_OR_LINK_FINDINGS" if findings else "PASS_CURRENT_METADATA_LINK_CASE",
        "scope": "Current Markdown/HTML href/src metadata snapshot; no final payload, Git index, archive, license, privacy, scientific or rendering approval.",
        "observed_UTC_start": started, "observed_UTC_end": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "campaign_repository_relative": cp,
        "exact_case_basis": "ROOT-supplied exact POSIX repository paths" if args.repository_path_list else "os.walk entry spellings; no Path.resolve used on link target",
        "explicit_final_payload_supplied": bool(args.file_list), "Git_executed": False,
        "models_arrays_raw_data_archives_read": False, "framework_imports": [n for n in ("torch", "numpy", "campaign", "stats_pack") if n in sys.modules],
        "known_final_exclusion_prefixes_or_files": sorted(excluded),
        "excluded_documents_not_opened": skipped_documents,
        "explicitly_listed_excluded_documents": skipped_documents if args.file_list else [],
        "historical_source_link_checker_gap": SOURCE_LINK_GAP,
        "source_current_sha256": digest((campaign / "publication_check.py").read_bytes()),
        "source_gap_probe_version_matches_current": digest((campaign / "publication_check.py").read_bytes()) == SOURCE_LINK_GAP["source_sha256"],
        "pack_metadata_only": selected_pack_metadata, "restore_only_virtual_names": sorted(virtual),
        "documents_scanned": len(md_inputs), "local_links_or_images_scanned": len(links),
        "link_kind_counts": dict(collections.Counter(v["kind"] for v in links)), "skipped_targets": dict(skipped),
        "finding_counts": dict(collections.Counter(v["rule"] for v in findings)),
        "findings": findings, "links": links, "Markdown_input_manifest": md_inputs,
        "path_list_casefold_collisions": {key: sorted(value) for key, value in folded.items() if len(value) > 1},
        "limits": [
            "An absent final file list leaves publication membership unverified.",
            "Exact Git spelling must later come from ROOT's HEAD/index/payload union; physical Windows spelling is only the current snapshot basis.",
            "Archive manifests declare restore-only paths; archive bytes and restored statistics were not opened or verified.",
            "Fragments are separated and retained but heading-anchor existence/case is not validated.",
            "Markdown inline parsing follows the existing simple link grammar; nested literal parentheses or escaped delimiters require a renderer-level check.",
            "Growing output directories mean this read interval is not an atomic final snapshot.",
        ],
    }
    if args.file_list and skipped_documents:
        report["status"] = "EXCLUDED_DOCUMENT_LISTED_IN_EXPLICIT_PAYLOAD"
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    with args.output_report.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in ("status", "documents_scanned", "local_links_or_images_scanned", "link_kind_counts", "finding_counts", "explicit_final_payload_supplied")}, ensure_ascii=False))

if __name__ == "__main__":
    main()

