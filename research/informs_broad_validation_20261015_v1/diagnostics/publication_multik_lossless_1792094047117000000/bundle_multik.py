"""Byte-exact publication projection for the already measured K-transition rows."""
import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tempfile
import time


WRITER_SHA256 = 'db06213bbf1ac315022e76d7946d2b070f49dfd163c73ff20a7b6a2be36c85cb'
SOURCE_RELATIVE = 'costs/multik_transition_1792090892017452100'
SCHEMA = 'multik_cost_byte_exact_projection_v1'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def safe_relative(name):
    if (not isinstance(name, str) or not name or '\\' in name or
            PurePosixPath(name).is_absolute() or
            any(part in {'', '.', '..'} or ':' in part for part in name.split('/'))):
        raise ValueError('Unsafe relative path')
    return name


def inside(root, relative):
    relative = safe_relative(relative)
    root = Path(root).resolve(); path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Unsafe relative path escapes root')
    return path


def trial_bytes(row):
    text = json.dumps(row, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    return text.replace('\n', '\r\n').encode('utf-8')


def json_new(path, value):
    with Path(path).open('xb') as stream:
        stream.write((json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8'))


def validate_scope(result_relative, source_bytes):
    value = json.loads(source_bytes[result_relative])
    folder = PurePosixPath(result_relative).parent.as_posix()
    rows = value['rows']
    trial_paths = sorted(name for name in source_bytes if name.startswith(folder + '/actual_trials/') and name.endswith('.json'))
    expected_paths = [folder + f'/actual_trials/trial_{index:06}.json' for index in range(len(rows))]
    if trial_paths != expected_paths:
        raise ValueError('trial index/count mismatch')
    records = []
    for index, (relative, row) in enumerate(zip(trial_paths, rows)):
        raw = source_bytes[relative]
        if json.loads(raw) != row:
            raise ValueError('trial parsed row mismatch')
        if trial_bytes(row) != raw:
            raise ValueError('trial byte reconstruction mismatch')
        records.append(dict(kind='trial', relative=relative, bytes=len(raw), sha256=sha(raw),
                            result_relative=result_relative, row_index=index))
    actual_csv = list(csv.DictReader(io.StringIO(source_bytes[folder + '/TRIALS.csv'].decode('utf-8'), newline='')))
    expected_csv = [{key: '' if item is None else str(item) for key, item in row.items()} for row in rows]
    if actual_csv != expected_csv:
        raise ValueError('csv row/value mismatch')
    for entry in value['transition_ledger']:
        selected = [row for row in rows if row['phase'] == entry['phase'] and
                    row['from_K'] == entry['from_K'] and row['to_K'] == entry['to_K']]
        within = sum(row.get('resource_within_limit', True) for row in selected)
        if (entry['actual_complete'] != len(selected) or entry['remaining'] != entry['requested'] - len(selected) or
                entry['completed_within_resource'] != within or
                entry['remaining_within_resource'] != entry['requested'] - within):
            raise ValueError('ledger count mismatch')
    numeric = json.loads(source_bytes[folder + '/NUMERIC_EXECUTION.json'])
    if numeric['rows'] != rows or numeric['transition_ledger'] != value['transition_ledger']:
        raise ValueError('numeric execution rows/ledger mismatch')
    summary = dict(result_relative=result_relative, trial_count=len(rows),
                   ledger_entries=len(value['transition_ledger']),
                   exact_minimum_protocol_complete=value.get('exact_minimum_protocol_complete'),
                   numeric_execution_status=value.get('numeric_execution_status'),
                   resource_status=value.get('resource_status'),
                   post_measure_validation_status=value.get('post_measure_validation_status'))
    return records, summary


def build(public_root, source, output, expected_scopes, expected_trials, writer_pin):
    public_root = Path(public_root).resolve(); source = Path(source).resolve(); output = Path(output).resolve()
    if not source.is_relative_to(public_root) or output.is_relative_to(source):
        raise ValueError('Source/output boundary mismatch')
    source_relative = safe_relative(source.relative_to(public_root).as_posix())
    paths = sorted(path for path in source.rglob('*') if path.is_file())
    if not paths or any(path.suffix.lower() not in {'.csv', '.json'} for path in paths):
        raise ValueError('Unexpected or empty source file set')
    originals = {safe_relative(path.relative_to(source).as_posix()): path.read_bytes() for path in paths}
    result_paths = sorted(name for name in originals if name.endswith('/RESULT.json'))
    if len(result_paths) != expected_scopes:
        raise ValueError('scope count mismatch')
    trial_records, scopes = [], []
    for relative in result_paths:
        records, summary = validate_scope(relative, originals)
        trial_records.extend(records); scopes.append(summary)
    if len(trial_records) != expected_trials:
        raise ValueError('trial count mismatch')
    trial_names = {record['relative'] for record in trial_records}
    actual_trial_names = {name for name in originals if '/actual_trials/' in name}
    if trial_names != actual_trial_names:
        raise ValueError('Unregistered trial source file')
    if output.exists():
        raise FileExistsError('Output must be a new immutable directory')
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for relative, raw in originals.items():
        if relative in trial_names:
            continue
        compressed = relative.endswith('/RESULT.json') or relative.endswith('/NUMERIC_EXECUTION.json')
        payload = 'files/' + relative + ('.gz' if compressed else '')
        encoded = gzip.compress(raw, compresslevel=9, mtime=0) if compressed else raw
        destination = inside(output, payload); destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(encoded)
        records.append(dict(kind='gzip' if compressed else 'copy', relative=relative,
                            bytes=len(raw), sha256=sha(raw), payload=payload,
                            payload_bytes=len(encoded), payload_sha256=sha(encoded)))
    records.extend(trial_records)
    manifest = dict(schema=SCHEMA, source_relative=source_relative, created_unix=time.time(),
                    original_files=len(originals), original_bytes=sum(map(len, originals.values())),
                    scope_count=len(scopes), trial_count=len(trial_records),
                    writer_pin=writer_pin,
                    trial_serialization=dict(ensure_ascii=False, indent=2, allow_nan=False,
                                             sort_keys=False, trailing_newline=True,
                                             newline_bytes='CRLF', encoding='utf-8'),
                    original_files_preserved=True, arrays_or_models_read=False,
                    numerical_precision_reduced=False, queue_or_ledger_reduced=False,
                    records=records, scopes=scopes)
    json_new(output / 'MANIFEST.json', manifest)
    with tempfile.TemporaryDirectory(prefix='multik_exact_restore_') as temporary:
        restored = restore(output / 'MANIFEST.json', Path(temporary))
        repeated = restore(output / 'MANIFEST.json', Path(temporary))
    after_paths = sorted(path for path in source.rglob('*') if path.is_file())
    unchanged = (paths == after_paths and all(path.read_bytes() == originals[path.relative_to(source).as_posix()] for path in paths))
    if not unchanged:
        raise ValueError('Original source changed during projection')
    result = dict(status='PASS_BYTE_EXACT_ALL_ORIGINAL_FILES', source_files=len(originals),
                  trial_count=len(trial_records), separate_temporary_restore=restored,
                  repeated_restore=repeated, original_source_unchanged=True,
                  original_bytes=sum(map(len, originals.values())),
                  payload_bytes=sum(record.get('payload_bytes', 0) for record in records),
                  manifest_sha256=sha((output / 'MANIFEST.json').read_bytes()),
                  scope_count=len(scopes), full_campaign_complete_claimed=False,
                  measurement_rerun=False, GPU_or_model_execution=False)
    json_new(output / 'BUILD_RESULT.json', result)
    return manifest


def materialize(manifest_path):
    manifest_path = Path(manifest_path).resolve(); base = manifest_path.parent
    manifest = json.loads(manifest_path.read_bytes())
    if manifest['schema'] != SCHEMA:
        raise ValueError('Unknown manifest schema')
    safe_relative(manifest['source_relative'])
    records = manifest['records']; names = [safe_relative(record['relative']) for record in records]
    if len(names) != len(set(names)) or len(records) != manifest['original_files']:
        raise ValueError('Manifest duplicate/count mismatch')
    content = {}
    for record in records:
        if record['kind'] == 'trial':
            continue
        if record['kind'] not in {'copy', 'gzip'}:
            raise ValueError('Unknown record kind')
        encoded = inside(base, record['payload']).read_bytes()
        if len(encoded) != record['payload_bytes'] or sha(encoded) != record['payload_sha256']:
            raise ValueError('Damaged payload bytes/SHA')
        raw = gzip.decompress(encoded) if record['kind'] == 'gzip' else encoded
        if len(raw) != record['bytes'] or sha(raw) != record['sha256']:
            raise ValueError('Original payload reconstruction SHA mismatch')
        content[record['relative']] = raw
    trial_count = 0
    for record in records:
        if record['kind'] != 'trial':
            continue
        result_relative = safe_relative(record['result_relative'])
        raw = trial_bytes(json.loads(content[result_relative])['rows'][record['row_index']])
        if len(raw) != record['bytes'] or sha(raw) != record['sha256']:
            raise ValueError('Reconstructed trial bytes/SHA mismatch')
        content[record['relative']] = raw; trial_count += 1
    if trial_count != manifest['trial_count'] or sum(map(len, content.values())) != manifest['original_bytes']:
        raise ValueError('Restoration total count/bytes mismatch')
    return manifest, content


def restore(manifest_path, public_root):
    manifest, content = materialize(manifest_path)
    root = inside(public_root, manifest['source_relative'])
    targets = [(inside(root, relative), raw) for relative, raw in content.items()]
    for path, raw in targets:
        if path.exists() and (not path.is_file() or path.read_bytes() != raw):
            raise ValueError('Refusing to overwrite a different existing file')
    restored, matching = 0, 0
    for path, raw in targets:
        if path.exists():
            matching += 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
        if sha(path.read_bytes()) != sha(raw):
            raise ValueError('Written restoration SHA mismatch')
        restored += 1
    return dict(status='PASS', restored=restored, already_matching=matching,
                all_file_sha_matches=len(content), trial_sha_matches=manifest['trial_count'],
                original_bytes=manifest['original_bytes'])


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('prepare')
    create.add_argument('--public-root', type=Path, required=True)
    create.add_argument('--output', type=Path, required=True)
    recreate = sub.add_parser('restore')
    recreate.add_argument('--manifest', type=Path, required=True)
    recreate.add_argument('--public-root', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        writer = args.public_root / 'measure_multik_transition.py'
        actual_sha = sha(writer.read_bytes())
        if actual_sha != WRITER_SHA256:
            raise ValueError('Pinned original measurement writer changed')
        manifest = build(args.public_root, args.public_root / SOURCE_RELATIVE, args.output,
                         8, 1440, dict(relative='measure_multik_transition.py', sha256=actual_sha))
        print(json.dumps(dict(status='PASS', original_files=manifest['original_files'],
                              trial_count=manifest['trial_count']), allow_nan=False))
    else:
        print(json.dumps(restore(args.manifest, args.public_root), allow_nan=False))


if __name__ == '__main__':
    main()
