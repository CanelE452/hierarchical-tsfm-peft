import argparse
import configparser
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import time


SCHEMA = 'GIT_HISTORY_PACK_BYTE_TRANSPORT_v1'
COMMIT = 'a32f88c184a5241585ca1fb50119f9462d91c402'
DEADLINE = 1792222550
TARGET = dict(file='research/informs_broad_validation_20261015_v1/stats_packs/pack_1792097901794080200/part_0000.tar.xz', bytes=11900560, sha256='b3f22f6675d5ca605440be7929cd2eeb10f3be38977798444c835352bc281a6a')


def blob_oid(payload):
    return hashlib.sha1(b'blob ' + str(len(payload)).encode() + b'\0' + payload).hexdigest()


def no_links(path):
    for part in (path, *path.parents):
        if part.is_symlink() or (part.exists() and getattr(part.lstat(), 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT):
            raise ValueError('symlink or reparse point refused')


def git(repo, *args, worker_deadline=None):
    if worker_deadline is not None and time.time() >= worker_deadline:
        raise TimeoutError('original worker deadline reached')
    env = {k: v for k, v in os.environ.items() if not k.upper().startswith('GIT_')}
    env.update(GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_COUNT='0', GIT_NO_LAZY_FETCH='1', GIT_NO_REPLACE_OBJECTS='1', GIT_OPTIONAL_LOCKS='0', GIT_TERMINAL_PROMPT='0')
    result = subprocess.run(['git', '-c', 'protocol.allow=never', '-C', str(repo), *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=30 if worker_deadline is None else min(30, worker_deadline - time.time()), check=False)
    if result.returncode:
        raise RuntimeError('local Git object unavailable or metadata check failed; no fetch attempted')
    return result.stdout


def full_local_clone(repo, worker_deadline=None):
    lines = git(repo, 'rev-parse', '--is-inside-work-tree', '--is-shallow-repository', '--path-format=absolute', '--git-common-dir', '--git-path', 'config.worktree', worker_deadline=worker_deadline).decode('utf-8').splitlines()
    if len(lines) != 4 or lines[:2] != ['true', 'false']:
        raise ValueError('full local clone required (worktrees supported)')
    common = Path(lines[2])
    if not common.is_absolute() or not common.is_dir():
        raise ValueError('full local clone common directory unavailable')
    no_links(common)
    bindings, forbidden = [], False
    for config_path in dict.fromkeys((common / 'config', Path(lines[3]))):
        if not config_path.is_absolute():
            raise ValueError('absolute Git configuration metadata required')
        no_links(config_path)
        config_bytes = config_path.read_bytes() if config_path.exists() else None
        bindings.append((config_path, config_bytes))
        config = configparser.ConfigParser(interpolation=None, strict=False)
        config.read_string((config_bytes or b'').decode('utf-8'))
        forbidden = forbidden or any(s.lower().startswith(('include', 'includeif')) or any(k.lower() in ('partialclone', 'partialclonefilter') or (k.lower() == 'promisor' and v.strip().lower() not in ('false', 'no', 'off', '0')) for k, v in config.items(s)) for s in config.sections())
    for name in ('objects/info/alternates', 'objects/info/http-alternates'):
        target = common / name
        no_links(target)
        forbidden = forbidden or (target.exists() and bool(target.read_bytes().strip()))
    no_links(common / 'objects/pack')
    if forbidden or any((common / 'objects/pack').glob('*.promisor')):
        raise ValueError('full local clone without includes, partial objects or alternates required')
    return bindings


def restore(repo, destination, manifest, worker_deadline=None):
    if worker_deadline is not None and time.time() >= worker_deadline:
        raise TimeoutError('original worker deadline reached')
    if manifest.get('schema') != SCHEMA or manifest.get('status') != 'SEALED_BY_ROOT' or manifest.get('commit') != COMMIT:
        raise ValueError('exact ROOT-sealed history manifest required')
    items = manifest.get('items', [])
    if not isinstance(items, list) or len(items) != 1:
        raise ValueError('exactly one sealed opaque historical pack item required')
    item = items[0]
    if set(item) != {'file', 'bytes', 'sha256', 'git_blob_oid'} or {key:item[key] for key in TARGET} != TARGET or type(item['bytes']) is not int:
        raise ValueError('only the exact original opaque historical pack path/size/SHA256 is permitted')
    name = item['file']
    p = PurePosixPath(name)
    if not isinstance(name, str) or '\\' in name or ':' in name or p.is_absolute() or any(x in ('', '.', '..') for x in name.split('/')) or not name.endswith('.tar.xz'):
        raise ValueError('unsafe repository-relative opaque pack path')
    if not isinstance(item['git_blob_oid'], str) or not re.fullmatch('[0-9a-f]{40}', item['git_blob_oid']):
        raise ValueError('actual Git blob OID required; a future null OID is not executable')
    repo, destination = Path(repo).absolute(), Path(destination).absolute()
    no_links(repo)
    no_links(destination)
    repo, destination = repo.resolve(), destination.resolve()
    if not repo.is_dir() or not destination.is_relative_to(repo):
        raise ValueError('fresh destination must be inside the declared repository')
    config_bindings = full_local_clone(repo, worker_deadline)
    result = []
    for item in items:
        oid = git(repo, 'rev-parse', '--verify', COMMIT + ':' + item['file'], worker_deadline=worker_deadline).decode('ascii').strip()
        if oid != item['git_blob_oid']:
            raise ValueError('commit:path blob binding mismatch')
        payload = git(repo, 'cat-file', 'blob', oid, worker_deadline=worker_deadline)
        if len(payload) != item['bytes'] or hashlib.sha256(payload).hexdigest() != item['sha256'] or blob_oid(payload) != oid:
            raise ValueError('blob binding size/SHA256/OID mismatch')
        target = destination / item['file']
        no_links(target)
        if worker_deadline is not None and time.time() >= worker_deadline:
            raise TimeoutError('original worker deadline reached before write')
        if target.exists():
            if not target.is_file() or target.read_bytes() != payload:
                raise ValueError('different existing destination refused; no overwrite')
            action = 'IDENTICAL_SKIPPED'
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            no_links(target)
            with target.open('xb') as handle:
                handle.write(payload)
            action = 'CREATED'
        result.append({**item, 'action': action})
    for config_path, original in config_bindings:
        if (config_path.read_bytes() if config_path.exists() else None) != original:
            raise ValueError('Git repository configuration changed during restoration')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--root-triggered', action='store_true')
    parser.add_argument('--worker-deadline', type=float)
    for key in ('repo-root', 'destination-root', 'manifest', 'manifest-sha', 'source-sha'):
        parser.add_argument('--' + key)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({'status': 'PREP_ONLY', 'input_reads': 0, 'Git_calls': 0, 'writes': 0}))
        return 0
    if not args.root_triggered or any(getattr(args, key.replace('-', '_')) is None for key in ('repo-root', 'destination-root', 'manifest', 'manifest-sha', 'source-sha')):
        raise ValueError('explicit ROOT trigger, roots and exact manifest/source SHA required')
    if args.worker_deadline is not None and (not math.isfinite(args.worker_deadline) or args.worker_deadline <= 0):
        raise ValueError('finite positive worker deadline required when specified')
    source = Path(__file__).read_bytes()
    manifest_bytes = Path(args.manifest).read_bytes()
    if hashlib.sha256(source).hexdigest() != args.source_sha or hashlib.sha256(manifest_bytes).hexdigest() != args.manifest_sha:
        raise ValueError('manifest/source SHA binding mismatch')
    rows = restore(Path(args.repo_root), Path(args.destination_root), json.loads(manifest_bytes), args.worker_deadline)
    print(json.dumps({'schema': 'GIT_HISTORY_PACK_BYTE_RESTORE_RESULT_v1', 'status': 'RESTORED_EXACT_BYTES', 'commit': COMMIT, 'manifest_sha256': args.manifest_sha, 'restorer_sha256': args.source_sha, 'worker_deadline_unix': args.worker_deadline, 'restored': sum(r['action'] == 'CREATED' for r in rows), 'already_matching': sum(r['action'] == 'IDENTICAL_SKIPPED' for r in rows), 'items': rows}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
