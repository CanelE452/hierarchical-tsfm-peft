"""Read-only publication inspection and exclusive, explicit bundle preparation."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import time
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET


MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_BUNDLE_BYTES = 250 * 1024 * 1024
TOP_FILES = ("PUBLIC_ALLOWLIST.txt", "PUBLICATION_CHECK.json", "PUBLIC_MANIFEST.sha256")
ALLOWED_SUFFIXES = {".py", ".ps1", ".sh", ".md", ".json", ".jsonl", ".csv", ".gz", ".png", ".svg", ".yaml", ".yml", ".toml", ".txt", ".sha256"}
PRIVATE_PARTS = {".git", ".cache", "__pycache__", "private", "private_cache", "weights", "checkpoints", "raw"}
EXCLUDED_NAMES = {"plan/MASTER_PLAN.md", "plan/PACKAGE_MANIFEST.sha256", "plan/PACKAGE_README.md"}
EXCLUDED_PREFIXES = ("reports/snapshot_20261015T095115Z_1792057875518554100", "checks")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MD_LINK = re.compile(r"!?\[[^\]\n]*\]\(\s*(?:<([^>]+)>|([^\s)]+))(?:\s+[^)]*)?\)")
MD_REFERENCE = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*(?:<([^>]+)>|(\S+))", re.MULTILINE)
HTML_LINK = re.compile(r"\b(?:href|src)\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
TOKEN_PATTERNS = (
    re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b"),
    re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
PHONE = re.compile(r"(?<![A-Za-z0-9_.])(?:01[016789]|\+82[- .]?1[016789])(?:[- .]\d{3,4}[- .]\d{4}|\d{7,8})(?![A-Za-z0-9_.])")
WINDOWS_PATH = re.compile(r"(?<![A-Za-z0-9_])(?:[A-Za-z]:[\\/])[^\s\"'`<>]*")
UNIX_PRIVATE_PATH = re.compile(r"(?<![:/A-Za-z0-9_])/(?:Users|home|mnt/[a-zA-Z])/(?!<)[^\s\"'`<>]+")
SECRET_ASSIGNMENT = re.compile(r"(?im)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*['\"]([^'\"\r\n]{12,})['\"]")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _git(root, args):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError("Read-only Git command failed")
    return result.stdout


def tracked_files(root):
    """Only files present in HEAD qualify as existing repository artifacts."""
    return {p.decode("utf-8") for p in _git(root, ["ls-tree", "-r", "--name-only", "-z", "HEAD"]).split(b"\0") if p}


def index_paths(root):
    return {p.decode("utf-8") for p in _git(root, ["diff", "--cached", "--name-only", "-z"]).split(b"\0") if p}


def index_blob(root, name):
    return _git(root, ["show", ":" + name])


def _root(campaign):
    return Path(campaign).resolve().parents[1]


def _failure(failures, name, rule, **details):
    failures.append({"file": name, "rule": rule, **details})


def _canonical_name(root, name):
    name = str(name).replace("\\", "/")
    pure = PurePosixPath(name)
    if pure.is_absolute() or re.match(r"^[A-Za-z]:", name) or ".." in pure.parts or not name or name.startswith("./"):
        raise ValueError("Unsafe repository-relative filename")
    path = root.joinpath(*pure.parts)
    path.resolve().relative_to(root)
    for ancestor in (path, *path.parents):
        if ancestor == root:
            break
        if ancestor.is_symlink():
            raise ValueError("Symlink is outside the publication contract")
    return pure.as_posix(), path


def _excluded(campaign, path, extra_prefixes=()):
    relative = path.relative_to(campaign).as_posix()
    if any(p.lower() in PRIVATE_PARTS for p in path.relative_to(campaign).parts):
        return "PRIVATE_DIRECTORY"
    if relative in EXCLUDED_NAMES:
        return "PRIVATE_PLAN_OR_PACKAGE"
    if any(relative == p or relative.startswith(p.rstrip("/") + "/") for p in (*EXCLUDED_PREFIXES, *extra_prefixes)):
        return "EXCLUDED_SNAPSHOT"
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        return "DISALLOWED_SUFFIX"
    if path.name.startswith(".") or path.name.endswith("~"):
        return "TEMPORARY_FILE"
    return None


def candidate_files(campaign, extras=(), excluded_prefixes=()):
    campaign = Path(campaign).resolve(); root = _root(campaign)
    selected = []; excluded = []
    for path in sorted(campaign.rglob("*")):
        if not path.is_file():
            continue
        reason = _excluded(campaign, path, excluded_prefixes)
        if path.parent == campaign and path.name in TOP_FILES:
            reason = "GENERATED_FINAL_METADATA"
        name = path.relative_to(root).as_posix()
        if reason:
            excluded.append({"file": name, "reason": reason})
        else:
            selected.append(name)
    selected.extend(str(p).replace("\\", "/") for p in extras)
    return sorted(set(selected)), excluded


def _scan_text(name, text, failures):
    for number, line in enumerate(text.splitlines(), 1):
        if any(pattern.search(line) for pattern in TOKEN_PATTERNS):
            _failure(failures, name, "SECRET_TOKEN", line=number)
        for match in SECRET_ASSIGNMENT.finditer(line):
            value = match.group(1)
            if not re.search(r"example|placeholder|redacted|dummy|<|\$\{|os\.environ|your[_ -]", value, re.IGNORECASE):
                _failure(failures, name, "SECRET_ASSIGNMENT", line=number)
        if PHONE.search(line):
            _failure(failures, name, "PHONE_NUMBER", line=number)
        for match in WINDOWS_PATH.finditer(line):
            suffix = line[match.end():]
            if match.group().endswith((":/", ":\\")) and suffix.startswith(("<", "$", "%", "[")):
                continue
            _failure(failures, name, "PRIVATE_ABSOLUTE_PATH", line=number)
        if UNIX_PRIVATE_PATH.search(line):
            _failure(failures, name, "PRIVATE_ABSOLUTE_PATH", line=number)


def _link(root, name, target, published, tracked, virtual, failures, links):
    target = unquote(target.strip())
    if not target or target.startswith("#"):
        return
    parsed = urlsplit(target)
    if parsed.scheme in {"http", "https", "mailto", "tel", "data"}:
        return
    if parsed.scheme or target.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", target):
        _failure(failures, name, "LINK_ABSOLUTE_OR_SCHEME")
        return
    target_path = unquote(parsed.path).replace("\\", "/")
    try:
        path = ((root / name).parent / target_path).resolve()
        relative = path.relative_to(root).as_posix()
    except ValueError:
        _failure(failures, name, "LINK_ESCAPES_REPOSITORY")
        return
    if path.is_symlink():
        _failure(failures, name, "LINK_SYMLINK")
        return
    if not path.exists() and relative not in virtual:
        _failure(failures, name, "LINK_MISSING", target=relative)
        return
    allowed = relative in published or relative in tracked
    if path.is_dir():
        allowed = any(p.startswith(relative.rstrip("/") + "/") for p in published | tracked)
    if not allowed:
        _failure(failures, name, "LINK_NOT_PUBLISHED", target=relative)
        return
    links.append({"file": name, "target": relative, "basis": "selected" if relative in published else "HEAD"})


def inspect(campaign, files, tracked=None, virtual_files=()):
    """Inspect explicit repository-relative files; never changes files or Git."""
    campaign = Path(campaign).resolve(); root = _root(campaign)
    failures = []; inventory = []; links = []; total = 0
    tracked = tracked_files(root) if tracked is None else set(tracked)
    names = []; paths = {}; seen = set()
    for raw in files:
        try:
            name, path = _canonical_name(root, raw)
        except ValueError:
            _failure(failures, "<invalid-relative-name>", "FILE_PATH_UNSAFE")
            continue
        if name.casefold() in seen:
            _failure(failures, name, "DUPLICATE_FILE")
            continue
        seen.add(name.casefold()); names.append(name); paths[name] = path
    published = set(names) | set(virtual_files)
    for name in names:
        path = paths[name]
        if not path.is_file():
            _failure(failures, name, "FILE_MISSING")
            continue
        if any(part.lower() in PRIVATE_PARTS for part in PurePosixPath(name).parts):
            _failure(failures, name, "DISALLOWED_FILE", reason="PRIVATE_DIRECTORY")
        elif path.is_relative_to(campaign):
            reason = _excluded(campaign, path)
            if reason:
                _failure(failures, name, "DISALLOWED_FILE", reason=reason)
        elif path.suffix.lower() not in ALLOWED_SUFFIXES:
            _failure(failures, name, "DISALLOWED_FILE", reason="DISALLOWED_SUFFIX")
        data = path.read_bytes(); total += len(data)
        inventory.append({"file": name, "bytes": len(data), "sha256": _sha(data)})
        if len(data) > MAX_FILE_BYTES:
            _failure(failures, name, "FILE_SIZE_LIMIT", bytes=len(data), limit=MAX_FILE_BYTES)
        suffix = path.suffix.lower()
        if suffix == ".png":
            if not data.startswith(PNG_SIGNATURE):
                _failure(failures, name, "PNG_SIGNATURE")
            continue
        try:
            if suffix == ".gz":
                data = gzip.decompress(data)
            text = data.decode("utf-8-sig")
        except (UnicodeDecodeError, OSError, EOFError):
            _failure(failures, name, "UTF8_OR_GZIP_INVALID")
            continue
        _scan_text(name, text, failures)
        if suffix == ".md":
            markup = re.sub(r"(?ms)^\s*(```|~~~).*?^\s*\1\s*$", "", text)
            for match in (*MD_LINK.finditer(markup), *MD_REFERENCE.finditer(markup)):
                _link(root, name, match.group(1) or match.group(2), published, tracked, set(virtual_files), failures, links)
            for match in HTML_LINK.finditer(markup):
                _link(root, name, match.group(1), published, tracked, set(virtual_files), failures, links)
        if suffix == ".svg":
            try:
                tree = ET.fromstring(text)
                if tree.tag.rsplit("}", 1)[-1] != "svg":
                    raise ValueError("Wrong SVG root")
                for element in tree.iter():
                    if element.tag.rsplit("}", 1)[-1].lower() in {"script", "foreignobject"}:
                        _failure(failures, name, "SVG_ACTIVE_CONTENT")
                    for key, value in element.attrib.items():
                        localkey = key.rsplit("}", 1)[-1].lower()
                        if localkey.startswith("on"):
                            _failure(failures, name, "SVG_ACTIVE_CONTENT")
                        if localkey == "href":
                            _link(root, name, value, published, tracked, set(virtual_files), failures, links)
            except (ET.ParseError, ValueError):
                _failure(failures, name, "SVG_SIGNATURE")
    if total > MAX_BUNDLE_BYTES:
        _failure(failures, "<bundle>", "BUNDLE_SIZE_LIMIT", bytes=total, limit=MAX_BUNDLE_BYTES)
    return {"status": "FAIL" if failures else "PASS", "schema": 1,
            "scope": "publication bytes, paths, local links and Git scope; no scientific rerun",
            "files": inventory, "file_count": len(inventory), "selected_bytes": total,
            "public_increment_upper_bound_bytes": total, "links": links, "failures": failures}


def _top_names(campaign):
    prefix = Path(campaign).resolve().relative_to(_root(campaign)).as_posix()
    return [prefix + "/" + name for name in TOP_FILES]


def prepare(campaign, files):
    """Exclusively create final metadata only after a clean explicit preflight."""
    campaign = Path(campaign).resolve(); root = _root(campaign)
    for name in TOP_FILES:
        if (campaign / name).exists():
            raise FileExistsError("Final publication metadata already exists: " + name)
    files = list(files)
    if set(files) & set(_top_names(campaign)):
        raise ValueError("Pass payload files only; generated metadata is added explicitly")
    result = inspect(campaign, files, virtual_files=_top_names(campaign))
    if result["status"] != "PASS":
        return result
    allowlist = sorted([item["file"] for item in result["files"]] + _top_names(campaign))
    allowlist_bytes = ("\n".join(allowlist) + "\n").encode("utf-8")
    receipt = {**result, "prepared_unix_ns": time.time_ns(), "allowlist_sha256": _sha(allowlist_bytes),
               "manifest_excludes_only_itself": True,
               "self_hash_policy": "Manifest hashes payload, allowlist and receipt; excludes only manifest. Receipt hashes payload and allowlist, never itself or manifest.",
               "preparation_does_not_stage_or_commit": True}
    receipt_bytes = _json_bytes(receipt)
    # Recheck immediately before exclusive writes; ongoing payload changes invalidate preparation.
    for item in result["files"]:
        if _sha((root / item["file"]).read_bytes()) != item["sha256"]:
            raise RuntimeError("Payload changed after preflight")
    digest_by_name = {item["file"]: item["sha256"] for item in result["files"]}
    digest_by_name[_top_names(campaign)[0]] = _sha(allowlist_bytes)
    digest_by_name[_top_names(campaign)[1]] = _sha(receipt_bytes)
    manifest_name = _top_names(campaign)[2]
    manifest_bytes = "".join(digest_by_name[name] + "  " + name + "\n" for name in allowlist if name != manifest_name).encode("utf-8")
    generated = (allowlist_bytes, receipt_bytes, manifest_bytes)
    projected = result["selected_bytes"] + sum(map(len, generated))
    if max(map(len, generated)) > MAX_FILE_BYTES or projected > MAX_BUNDLE_BYTES:
        return {"status": "FAIL", "failures": [{"file": "<generated-metadata>", "rule": "GENERATED_SIZE_LIMIT"}]}
    for name, data in zip(TOP_FILES, generated):
        with (campaign / name).open("xb") as stream:
            stream.write(data)
    return verify(campaign)


def verify(campaign):
    campaign = Path(campaign).resolve(); root = _root(campaign)
    if any(not (campaign / name).is_file() for name in TOP_FILES):
        return {"status": "FAIL", "failures": [{"file": "<final-metadata>", "rule": "FINAL_METADATA_MISSING"}]}
    allowlist_data = (campaign / TOP_FILES[0]).read_bytes()
    try:
        files = allowlist_data.decode("utf-8").splitlines()
        receipt = json.loads((campaign / TOP_FILES[1]).read_text(encoding="utf-8"))
        lines = (campaign / TOP_FILES[2]).read_text(encoding="utf-8").splitlines()
        manifest = {}
        for line in lines:
            digest, name = line.split("  ", 1)
            if not re.fullmatch(r"[0-9a-f]{64}", digest) or name in manifest:
                raise ValueError("Invalid or duplicated manifest line")
            manifest[name] = digest
    except (ValueError, UnicodeDecodeError):
        return {"status": "FAIL", "failures": [{"file": "<final-metadata>", "rule": "FINAL_METADATA_INVALID"}]}
    result = inspect(campaign, files)
    failures = result["failures"]
    if files != sorted(set(files)) or not set(_top_names(campaign)).issubset(files):
        _failure(failures, "<allowlist>", "ALLOWLIST_INVALID")
    if receipt.get("allowlist_sha256") != _sha(allowlist_data):
        _failure(failures, "<allowlist>", "ALLOWLIST_HASH_MISMATCH")
    expected_manifest = set(files) - {_top_names(campaign)[2]}
    if set(manifest) != expected_manifest:
        _failure(failures, "<manifest>", "MANIFEST_SCOPE")
    inventory = {item["file"]: item for item in result["files"]}
    for name, digest in manifest.items():
        if name not in inventory or inventory[name]["sha256"] != digest:
            _failure(failures, name, "MANIFEST_HASH_MISMATCH")
    for item in receipt.get("files", []):
        current = inventory.get(item["file"])
        if current != item:
            _failure(failures, item["file"], "PREFLIGHT_PAYLOAD_MISMATCH")
    expected_payload = set(files) - set(_top_names(campaign))
    if {item["file"] for item in receipt.get("files", [])} != expected_payload:
        _failure(failures, "<receipt>", "RECEIPT_PAYLOAD_SCOPE")
    result["status"] = "FAIL" if failures else "PASS"
    return result


def check_index(root, selected):
    root = Path(root).resolve(); selected = set(selected); failures = []
    staged = index_paths(root)
    if staged != selected:
        _failure(failures, "<index>", "STAGED_SCOPE", missing=sorted(selected - staged), extra=sorted(staged - selected))
    for name in sorted(selected & staged):
        try:
            _, path = _canonical_name(root, name)
            same = path.is_file() and index_blob(root, name) == path.read_bytes()
        except (ValueError, RuntimeError):
            same = False
        if not same:
            _failure(failures, name, "INDEX_DISK_MISMATCH")
    return {"status": "FAIL" if failures else "PASS", "failures": failures, "selected_count": len(selected), "staged_count": len(staged), "git_mutations": 0}


def preflight(campaign, files, excluded=()):
    campaign = Path(campaign).resolve()
    result = inspect(campaign, files, virtual_files=_top_names(campaign))
    result["excluded"] = list(excluded)
    output = campaign / "checks" / ("preflight_" + str(time.time_ns()))
    output.mkdir(parents=True, exist_ok=False)
    (output / "PREFLIGHT.json").write_bytes(_json_bytes(result))
    return result, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, default=Path(__file__).resolve().parent)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--staged", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    parser.add_argument("--file-list", type=Path, help="Explicit payload names, one repository-relative file per line")
    parser.add_argument("--extra", action="append", default=[])
    parser.add_argument("--exclude-prefix", action="append", default=[])
    args = parser.parse_args()
    if args.verify or args.staged:
        result = verify(args.campaign)
        if args.staged and result["status"] == "PASS":
            names = (args.campaign / TOP_FILES[0]).read_text(encoding="utf-8").splitlines()
            result = check_index(_root(args.campaign), names)
    else:
        if args.file_list:
            files = [name for name in args.file_list.read_text(encoding="utf-8").splitlines() if name]
            excluded = []
        else:
            files, excluded = candidate_files(args.campaign, args.extra, args.exclude_prefix)
        if args.prepare:
            result = prepare(args.campaign, files)
        else:
            result, output = preflight(args.campaign, files, excluded)
            result = {"status": result["status"], "file_count": result["file_count"], "selected_bytes": result["selected_bytes"], "failures": result["failures"], "preflight_directory": output.relative_to(_root(args.campaign)).as_posix()}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
