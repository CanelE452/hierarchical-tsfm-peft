"""Byte-only auxiliary publication pack; no semantic data transformations."""
import argparse
import csv
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import tempfile
import time

CAMPAIGN = 'research/informs_broad_validation_20261015_v1'
RESTORER_SHA = 'e88b5cf8d37a667ba60b1666982924be7e1aa796d86101ab64a2a7834b59ab5b'
CANONICAL5 = {CAMPAIGN + '/' + n for n in ['REPORT_KO.md', 'PROFESSOR_BRIEF.md', 'README_KO.md', 'REPRODUCE.md', 'RESUME.md']}
APPROVED_PROJECTION_DIAGNOSTICS = [
    {'file': CAMPAIGN + '/publication_projections/lossless_1792092403733272600/STANDARD_ALIASES12.json',
     'bytes': 37096, 'sha256': '69509f2c0ee41252c63804b11e818d8c91277f703a8c7889c1e170e21b92b8c5'},
    {'file': CAMPAIGN + '/publication_projections/lossless_1792092403733272600/STATS_RESULT.json',
     'bytes': 19549, 'sha256': '6e623d4b8f236b3147186678eb5fb8b1712c806f0d58d097b2dda7d5294b2dce'},
]

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def safe(name):
    if (not isinstance(name, str) or not name or '\\' in name or PurePosixPath(name).is_absolute()
            or any(x in {'', '.', '..'} or ':' in x for x in name.split('/'))):
        raise ValueError('Unsafe repository-relative path')
    return name

def inside(root, name, must_exist=True):
    root = Path(root).resolve(); unresolved = root / safe(name)
    for p in [unresolved, *unresolved.parents]:
        if p == root:
            break
        if p.is_symlink():
            raise ValueError('Symlink path rejected')
    p = unresolved.resolve()
    if not p.is_relative_to(root) or (must_exist and not p.is_file()):
        raise ValueError('Missing or escaped input path')
    return p

def pinned(descriptor):
    p = Path(descriptor['path'])
    if p.is_symlink():
        raise ValueError('Pinned symlink rejected')
    raw = p.read_bytes()
    if sha(raw) != descriptor['sha256']:
        raise ValueError('Pinned input SHA differs')
    return raw

def allowed_original(name, row, exceptions=()):
    safe(name)
    prefix = CAMPAIGN + '/'
    if not name.startswith(prefix):
        raise ValueError('Original outside campaign')
    tail = PurePosixPath(name[len(prefix):])
    if len(tail.parts) < 2 or name in CANONICAL5 or tail.suffix not in {'.json', '.csv'}:
        raise ValueError('Root canonical/source/non-auxiliary input rejected')
    if any(x in {'stats_packs', 'publication_derivatives'} or x.startswith('publication_multik') for x in tail.parts):
        raise ValueError('Restore bootstrap namespace rejected')
    if tail.parts[0] == 'publication_projections':
        binding = {'file': name, 'bytes': int(row['original_bytes']), 'sha256': row['original_sha256']}
        if binding not in exceptions or binding not in APPROVED_PROJECTION_DIAGNOSTICS:
            raise ValueError('Projection requires explicit exact non-bootstrap exception')

def selection_rows(selection):
    raw = pinned(selection)
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'), newline='')))
    fields = {'original_file', 'original_bytes', 'original_sha256', 'estimated_gzip_bytes', 'estimated_saving_bytes'}
    if not rows or set(rows[0]) != fields:
        raise ValueError('Frozen selection CSV schema differs')
    names = [safe(r['original_file']) for r in rows]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate original path')
    if len(rows) != selection['expected_count'] or sum(int(r['original_bytes']) for r in rows) != selection['expected_original_bytes']:
        raise ValueError('Explicit selection count/bytes differ')
    for r in rows:
        if int(r['original_bytes']) < 1024 or len(r['original_sha256']) != 64:
            raise ValueError('Invalid original byte binding')
        allowed_original(r['original_file'], r, selection.get('allowed_projection_originals', []))
    return raw, rows

def guards(repo, descriptor):
    meta = json.loads(pinned(descriptor))
    source39 = meta['source39']
    if len(source39) != 39:
        raise ValueError('Exact source39 required')
    result = {}
    for name, expected in source39.items():
        rel = CAMPAIGN + '/' + safe(name)
        actual = sha(inside(repo, rel).read_bytes())
        if actual != expected:
            raise ValueError('Frozen source39 differs')
        result[rel] = actual
    for name in ['EXPERIMENT_REGISTRY.csv', 'SELECTION_LOCK.json']:
        rel = CAMPAIGN + '/' + name
        result[rel] = sha(inside(repo, rel).read_bytes())
    return result

def json_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')

def load_restorer(path):
    raw = Path(path).read_bytes()
    if sha(raw) != RESTORER_SHA:
        raise ValueError('Existing restorer pin differs')
    spec = importlib.util.spec_from_file_location('extra_aux_exact_existing_restorer', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def independent_restore(manifest_path, destination):
    spec = json.loads(Path(manifest_path).read_bytes())
    names = [r['original_file'] for r in spec['items']]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate restoration path')
    count = 0
    for row in spec['items']:
        blob = inside(Path(manifest_path).parent, row['gzip_file']).read_bytes()
        if (len(blob), sha(blob)) != (row['gzip_bytes'], row['gzip_sha256']):
            raise ValueError('Independent gzip binding differs')
        raw = gzip.decompress(blob)
        if (len(raw), sha(raw)) != (row['original_bytes'], row['original_sha256']):
            raise ValueError('Independent original binding differs')
        target = inside(destination, row['original_file'], must_exist=False)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as f:
            f.write(raw)
        if (target.stat().st_size, sha(target.read_bytes())) != (row['original_bytes'], row['original_sha256']):
            raise ValueError('Independent restored byte binding differs')
        count += 1
    return {'status': 'PASS', 'restored': count, 'all_original_byte_SHA_exact': True}

def build(repo, selection, restorer_descriptor, source_guard, public_output, execute=False, root_triggered=False):
    repo = Path(repo).resolve(); public_output = Path(public_output).resolve()
    raw_list, rows = selection_rows(selection)
    restorer_raw = pinned(restorer_descriptor)
    if sha(restorer_raw) != RESTORER_SHA:
        raise ValueError('Existing restorer identity differs')
    pinned(source_guard)
    plan = dict(status='METADATA_ONLY_NO_PACK_NO_SOURCE_BODY_READ', original_files=len(rows),
                original_bytes=sum(int(x['original_bytes']) for x in rows),
                estimated_gzip_bytes=sum(int(x['estimated_gzip_bytes']) for x in rows),
                expected_list_sha256=selection['sha256'], restorer_sha256=RESTORER_SHA,
                output_name=public_output.name, source_or_original_files_modified=False,
                semantic_parse_or_numeric_transformation=False, GPU_model_NPZ_T_or_Git_operations=False)
    if not execute:
        return plan
    if not root_triggered:
        raise ValueError('Explicit ROOT execution trigger required')
    campaign = inside(repo, CAMPAIGN, must_exist=False)
    if not campaign.is_dir():
        raise ValueError('Campaign directory missing')
    expected_parent = campaign / 'publication_derivatives'
    if public_output.parent != expected_parent or not public_output.name.startswith('gzip_extra_aux_') or public_output.exists():
        raise ValueError('New owned public pack boundary required')
    before_guards = guards(repo, source_guard)
    before_originals = {}
    for row in rows:
        p = inside(repo, row['original_file'])
        raw = p.read_bytes()
        actual = (len(raw), sha(raw))
        if actual != (int(row['original_bytes']), row['original_sha256']):
            raise ValueError('Actual original differs from explicit frozen selection')
        before_originals[row['original_file']] = actual
    started = time.perf_counter()
    public_output.mkdir(); (public_output / 'files').mkdir()
    items = []
    for index, row in enumerate(rows):
        raw = inside(repo, row['original_file']).read_bytes()
        if (len(raw), sha(raw)) != before_originals[row['original_file']]:
            raise ValueError('Original changed during compression')
        blob = gzip.compress(raw, compresslevel=9, mtime=0)
        gzip_name = f'files/item_{index:04d}{PurePosixPath(row["original_file"]).suffix}.gz'
        with (public_output / gzip_name).open('xb') as f:
            f.write(blob)
        items.append(dict(original_file=row['original_file'], original_bytes=len(raw), original_sha256=sha(raw),
                          gzip_file=gzip_name, gzip_bytes=len(blob), gzip_sha256=sha(blob)))
    manifest = {'format': 'exact_auxiliary_gzip_v1', 'items': items}
    json_new(public_output / 'GZIP_MANIFEST.json', manifest)
    (public_output / 'restore_originals.py').write_bytes(restorer_raw)
    (public_output / 'ORIGINAL_FILE_LIST.csv').write_bytes(raw_list)
    (public_output / 'BUILD_RECIPE.py').write_bytes(Path(__file__).read_bytes())
    helper = load_restorer(public_output / 'restore_originals.py')
    with tempfile.TemporaryDirectory(prefix='extra_aux_existing_restore_') as temp1:
        restored = helper.restore(public_output / 'GZIP_MANIFEST.json', temp1)
        if restored['restored'] != len(rows):
            raise ValueError('First temporary full restore count differs')
        for row in items:
            raw = inside(temp1, row['original_file']).read_bytes()
            if (len(raw), sha(raw)) != (row['original_bytes'], row['original_sha256']):
                raise ValueError('First full temporary restore differs')
    with tempfile.TemporaryDirectory(prefix='extra_aux_independent_restore_') as temp2:
        external = independent_restore(public_output / 'GZIP_MANIFEST.json', temp2)
        repeated = helper.restore(public_output / 'GZIP_MANIFEST.json', temp2)
        if external['restored'] != len(rows) or repeated['already_matching'] != len(rows) or repeated['restored'] != 0:
            raise ValueError('External/repeat full restore counts differ')
    after_originals = {r['original_file']: (inside(repo, r['original_file']).stat().st_size,
                       sha(inside(repo, r['original_file']).read_bytes())) for r in rows}
    after_guards = guards(repo, source_guard)
    if before_originals != after_originals or before_guards != after_guards:
        raise ValueError('Original/source/registry/lock changed; partial pack preserved')
    manifest_sha = sha((public_output / 'GZIP_MANIFEST.json').read_bytes())
    result = dict(status='PASS_EXACT_EXTRA_AUX_FULL', original_files=len(rows), original_bytes=plan['original_bytes'],
                  gzip_bytes=sum(r['gzip_bytes'] for r in items),
                  manifest_sha256=manifest_sha, selection_list_sha256=selection['sha256'],
                  existing_restorer_sha256=RESTORER_SHA, build_recipe_sha256=sha(Path(__file__).read_bytes()),
                  source_pre_post_identical=True, original_files_unchanged=True,
                  source_registry_lock_before=before_guards, source_registry_lock_after=after_guards,
                  first_temporary_restore=restored, independent_temporary_restore=external, repeat_restore=repeated,
                  both_temporary_directories_removed=True, original_file_semantic_parse=False,
                  numbers_rows_conditions_or_precision_removed=False, overwrite_existing_original=False,
                  GPU_model_NPZ_T_or_Git_operations=False, wall_seconds=time.perf_counter()-started)
    json_new(public_output / 'BUILD_RESULT.json', result)
    readme = '''# 부속 JSON/CSV 원바이트 무손실 포장

원본 내용·모든 숫자·행·조건을 바꾸지 않고 gzip(level9,mtime0)으로 보존했다. 원파일은 로컬에서 변경하지 않았다. [모든 원경로·원 SHA·gzip SHA](GZIP_MANIFEST.json), [동결 원 목록](ORIGINAL_FILE_LIST.csv), [전수 임시·독립 복원 및 반복 검증](BUILD_RESULT.json)을 함께 확인한다. 원본의 JSON 의미를 다시 직렬화하거나 CSV 숫자를 반올림하지 않았다.

저장소 루트에서 `python <이 폴더>/restore_originals.py --destination .`를 실행하면 정확한 원경로로 복원한다. 기존 byte가 같으면 skip하고 다르면 실패하며 덮어쓰지 않는다. 이 helper는 기존 e88b5cf8…ab5b의 정확한 복사본이다. 재포장 소스는 [BUILD_RECIPE.py](BUILD_RECIPE.py)이며 명시된 입력 SHA·source39·registry·lock을 확인하고 --execute --root-triggered로만 새 출력에 쓴다.

루트 canonical MD/CSV, stats pack manifest/archive, 원 복원 bootstrap은 이 부속 포장의 대상이 아니다. 공개 Markdown은 원경로 직접 링크 대신 이 manifest/index 또는 실제 gzip 경로를 사용한다. 복원 PASS는 새 학습·예측·독립 일반화·원격 게시 완료를 뜻하지 않는다.
'''
    (public_output / 'README_KO.md').write_text(readme, encoding='utf-8', newline='\n')
    additions = {p.relative_to(repo).as_posix() for p in public_output.rglob('*') if p.is_file()}
    additions.add((public_output / 'ADD_LIST.txt').relative_to(repo).as_posix())
    (public_output / 'ADD_LIST.txt').write_text(''.join(n+'\n' for n in sorted(additions)), encoding='utf-8', newline='\n')
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--selection-config', type=Path, required=True)
    parser.add_argument('--restorer', type=Path, required=True)
    parser.add_argument('--source-guard', type=Path, required=True)
    parser.add_argument('--source-guard-sha256', required=True)
    parser.add_argument('--public-output', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--root-triggered', action='store_true')
    args = parser.parse_args()
    selection = json.loads(args.selection_config.read_bytes())
    result = build(args.repo_root, selection, {'path': str(args.restorer), 'sha256': RESTORER_SHA},
                   {'path': str(args.source_guard), 'sha256': args.source_guard_sha256}, args.public_output,
                   args.execute, args.root_triggered)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))

if __name__ == '__main__':
    main()
