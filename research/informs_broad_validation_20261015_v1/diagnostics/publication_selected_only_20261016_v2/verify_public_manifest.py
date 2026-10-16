"""Check every explicitly published file against the final SHA256 allowlist."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


MAX_FILE_BYTES = 20 * 1024**2
MAX_BUNDLE_BYTES = 250 * 1024**2


def sha_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inside(root, name):
    if (not name or '\\' in name or ':' in name or PurePosixPath(name).is_absolute()
            or any(part in {'', '.', '..', '.git'} for part in name.split('/'))):
        raise ValueError('Unsafe repository-relative manifest path')
    path = root / name
    for part in [path, *path.parents]:
        if part == root:
            break
        if part.is_symlink():
            raise ValueError('Symlink manifest target is not supported')
    if not path.resolve().is_relative_to(root):
        raise ValueError('Manifest target escaped repository root')
    return path


def verify(root, manifest):
    root = Path(root).resolve()
    manifest = Path(manifest).absolute()
    manifest_name = manifest.relative_to(root).as_posix()
    manifest = inside(root, manifest_name)
    entries = {}
    for line in manifest.read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        if re.fullmatch(r'[0-9a-f]{64}', digest) is None or name in entries:
            raise ValueError('Invalid or duplicate manifest entry')
        inside(root, name)
        entries[name] = digest
    if not entries or manifest_name in entries:
        raise ValueError('Manifest must hash its nonempty allowlist and exclude itself')
    if list(entries) != sorted(entries):
        raise ValueError('Manifest entries must be sorted')
    total = manifest.stat().st_size
    if total > MAX_FILE_BYTES:
        raise ValueError('Manifest exceeds the public per-file limit')
    for name, digest in entries.items():
        path = inside(root, name)
        size = path.stat().st_size
        if size > MAX_FILE_BYTES or sha_file(path) != digest:
            raise ValueError('Public file size or SHA mismatch: ' + name)
        total += size
    if total > MAX_BUNDLE_BYTES:
        raise ValueError('Published allowlist exceeds the total public byte limit')
    return dict(status='PASS_EXPLICIT_PUBLIC_ALLOWLIST_SHA256',
                hashed_files=len(entries), selected_files_including_manifest=len(entries) + 1,
                selected_bytes_including_manifest=total, manifest_sha256=sha_file(manifest),
                scope='Published bytes and stated file limits only; no numerical rescoring, learning, links, or remote-Git verification.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.repo_root, args.manifest), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
