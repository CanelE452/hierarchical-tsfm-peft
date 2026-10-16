"""Reuse the immutable K8 byte-exact packer after the fresh cost owner closes."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys


HELPER_REL = 'diagnostics/publication_multik_lossless_1792094047117000000/bundle_multik.py'
HELPER_SHA = '5632fa60313c41d62b1573ca67225cfb9ad9dadd7e494ba14c506388548a24dc'
PLAN_REL = 'diagnostics/multik_postfinal_resource_1792096163404501000/PLAN.json'
PLAN_SHA = '45908dfa6bf5354c655bc879143d81a440484927885acda8ea4989cbf9808601'
OLD_BUNDLE_REL = 'diagnostics/publication_multik_lossless_1792094047117000000/bundle_1792096101396562900'
OLD_MANIFEST_SHA = '7d5e1e25fd436e5d18129a0341295f58bdc187792a5c7b20a23b7f7e00dff2c7'
OLD_BUILD_SHA = '50313829567552c3f7be9ca57e4cde0ef0455fa488d1d298c559da6b9d1a0f40'
SOURCE_REL = 'costs/multik_transition_1792098251740023100'
PINS_SHA = 'b520924528606a1f2eb83f6302a247cfd9b1f4e5a179adb2f1ff83b5fc3c2ad0'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pinned(path, expected):
    if digest(path) != expected:
        raise ValueError('Pinned input SHA mismatch')
    return json.loads(Path(path).read_bytes())


def metadata(public):
    path = public / HELPER_REL
    if digest(path) != HELPER_SHA:
        raise ValueError('Original lossless helper changed')
    spec = importlib.util.spec_from_file_location('unchanged_multik_lossless', path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    plan = pinned(public / PLAN_REL, PLAN_SHA)
    if digest(public / 'measure_multik_transition.py') != helper.WRITER_SHA256:
        raise ValueError('Pinned measurement writer changed')
    return helper, plan


def execute(args, public, helper, plan):
    if not args.root_triggered:
        raise ValueError('Explicit ROOT terminal review and trigger required')
    cache = Path(os.environ['INFORMS_CACHE_ROOT']).resolve()
    if (cache / 'ACTIVE_GPU_LEASE.json').exists():
        raise ValueError('Cost owner is still active')
    terminal_path = helper.inside(public, args.terminal_result)
    terminal = pinned(terminal_path, args.terminal_sha256)
    closure = terminal.get('central_closure') or {}
    if (closure.get('status') != 'CLOSED' or not terminal.get('source39_preserved')
            or not terminal.get('immutable_selection_and_registry_preserved')):
        raise ValueError('Terminal source/selection/central closure evidence failed')
    closure_path = helper.inside(cache, closure['cache_relative_path'])
    actual_closure = pinned(closure_path, closure['sha256'])
    if any(actual_closure.get(key) != closure.get(key) for key in
           ('lease_id', 'status', 'charged_process_seconds', 'deadline_overrun_seconds')):
        raise ValueError('Actual central closure binding mismatch')
    receipt_path = public / SOURCE_REL / 'BATCH_LAUNCHER_RECEIPT.json'
    receipt = pinned(receipt_path, args.batch_sha256)
    if terminal.get('multik_batch_result') != receipt:
        raise ValueError('Terminal result does not bind this exact final batch')
    if (receipt.get('public_batch_relative_path') != SOURCE_REL
            or receipt.get('pins_sha256') != PINS_SHA or not receipt.get('finished_unix')):
        raise ValueError('Wrong or unterminated batch')
    rows = receipt['scope_ledger']
    queue = plan['full_scope_queue']
    if len(rows) != 20 or len({row['scope_id'] for row in rows}) != 20:
        raise ValueError('Original20 scope identity/count mismatch')
    if any(any(row.get(key) != value for key, value in registered.items() if key != 'status')
           for row, registered in zip(rows, queue)):
        raise ValueError('Original full20 queue/science changed')
    before_sources = {name: digest(helper.inside(public, name)) for name in plan['source39']}
    if before_sources != plan['source39']:
        raise ValueError('Source39 mismatch')
    old_folder = public / OLD_BUNDLE_REL
    old_manifest = pinned(old_folder / 'MANIFEST.json', OLD_MANIFEST_SHA)
    old_build = pinned(old_folder / 'BUILD_RESULT.json', OLD_BUILD_SHA)
    if old_build['status'] != 'PASS_BYTE_EXACT_ALL_ORIGINAL_FILES':
        raise ValueError('Original K8 package is not a verified exact package')
    old_records = {record['relative']: record['sha256'] for record in old_manifest['records']}
    reused, fresh, unstarted = [], [], []
    for row in rows:
        output = row.get('output_public_relative_path')
        if row.get('status') == 'REUSED_VERIFIED_COMPLETE':
            prefix = old_manifest['source_relative'] + '/'
            if not output or not output.startswith(prefix):
                raise ValueError('Reused scope does not bind original K8 package')
            suffix = output[len(prefix):] + '/'
            expected = {name[len(suffix):]: value for name, value in old_records.items()
                        if name.startswith(suffix)}
            if not expected or expected != row.get('result_files_sha256'):
                raise ValueError('Reused original scope file SHA map differs')
            reused.append(row['scope_id'])
        elif row.get('fresh_child_completed_in_this_call'):
            if output != SOURCE_REL + '/' + row['scope_id']:
                raise ValueError('Fresh output escaped this batch')
            scope = helper.inside(public, output)
            actual = {path.relative_to(scope).as_posix(): digest(path)
                      for path in scope.rglob('*') if path.is_file()}
            if actual != row.get('result_files_sha256'):
                raise ValueError('Fresh receipt/file SHA map differs; no files dropped')
            fresh.append(row)
        else:
            if output or row.get('actual_numeric_trials', 0):
                raise ValueError('Unclassified output or numeric rows; preserve and inspect')
            unstarted.append(row['scope_id'])
    if len(reused) != 8 or len(fresh) > 12:
        raise ValueError('Expected eight exact old scopes and at most12 fresh scopes')
    scopes = sum(sum(name.endswith('/RESULT.json') for name in row['result_files_sha256'])
                 for row in fresh)
    if scopes != len(fresh):
        raise ValueError('Fresh partial scope lacks exactly one durable RESULT; originals preserved')
    trials = sum(row['actual_numeric_trials'] for row in fresh)
    if trials > 2160:
        raise ValueError('Fresh trial count exceeds original requested scope')
    output = helper.inside(public, args.output)
    manifest = helper.build(public, public / SOURCE_REL, output, scopes, trials,
                            dict(relative='measure_multik_transition.py', sha256=helper.WRITER_SHA256))
    if before_sources != {name: digest(helper.inside(public, name)) for name in before_sources}:
        raise ValueError('Source39 changed while packaging; keep output for inspection')
    if digest(receipt_path) != args.batch_sha256 or digest(terminal_path) != args.terminal_sha256:
        raise ValueError('Terminal metadata changed while packaging')
    helper.json_new(output / 'EXTERNAL_REUSE_BINDINGS.json', dict(
        original_K8_bundle_relative_path=OLD_BUNDLE_REL,
        original_manifest_sha256=OLD_MANIFEST_SHA, original_build_sha256=OLD_BUILD_SHA,
        reused_scope_ids=reused, reused_numeric_trials=1440,
        source39_sha256=before_sources, terminal_result_relative_path=args.terminal_result,
        terminal_result_sha256=args.terminal_sha256, final_batch_sha256=args.batch_sha256,
        original_central_closure=closure,
        fresh_scope_ids=[row['scope_id'] for row in fresh], unstarted_scope_ids=unstarted,
        requested_fresh_scope_maximum=12, requested_fresh_trial_maximum=2160,
        actual_fresh_scopes=scopes, actual_fresh_trials=trials,
        original_full20_status=receipt['status'], full20_complete_claimed=False))
    text = ('# Fresh K 비용 무손실 공개 포장\n\n'
            '새 측정이 아니라 원 JSON/CSV/영수증의 바이트 동일 공개 파생본입니다. '
            '완료·부분·미실행 상태는 원 BATCH와 full20 큐 그대로 보존합니다.\n\n'
            f'- 현재 포장: {scopes} fresh scope, {trials} trial. 계획상 최대12/2160과 구별합니다.\n'
            '- [복원 원본 목록과 모든 SHA](MANIFEST.json)\n'
            '- [별도 임시 복원·반복 검산](BUILD_RESULT.json)\n'
            '- [기존8 재사용 및 원 terminal 바인딩](EXTERNAL_REUSE_BINDINGS.json)\n'
            f'- [기존 K8 공개 포장](../' + Path(OLD_BUNDLE_REL).relative_to(Path('diagnostics')).as_posix() + '/MANIFEST.json)\n\n'
            '복원은 기존 bundle_multik.py의 restore --manifest <이 폴더/MANIFEST.json> '
            '--public-root <복원 R>를 사용합니다. 기존과 다른 파일은 덮어쓰지 않습니다. '
            '원 trial 경로는 MANIFEST에서 복원하며 Markdown의 원 trial 직접 링크를 만들지 않습니다.\n')
    # Compute the link from the actual immutable package directory, rather than guessing depth.
    old_link = Path(os.path.relpath(old_folder / 'MANIFEST.json', output)).as_posix()
    marker = f'../' + Path(OLD_BUNDLE_REL).relative_to(Path('diagnostics')).as_posix() + '/MANIFEST.json'
    text = text.replace(marker, old_link)
    with (output / 'README_KO.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)
    return dict(status='PASS_EXACT_FRESH_PACKAGE', scope_count=manifest['scope_count'],
                trial_count=manifest['trial_count'], original_files=manifest['original_files'],
                full20_status_preserved=True, measurement_or_GPU_execution=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-package', action='store_true')
    parser.add_argument('--root-triggered', action='store_true')
    parser.add_argument('--terminal-result')
    parser.add_argument('--terminal-sha256')
    parser.add_argument('--batch-sha256')
    parser.add_argument('--output')
    args = parser.parse_args()
    public = Path(os.environ['INFORMS_PUBLIC_ROOT']).resolve()
    helper, plan = metadata(public)
    if args.execute_package:
        if not all((args.terminal_result, args.terminal_sha256, args.batch_sha256, args.output)):
            parser.error('Explicit ROOT terminal/BATCH SHA and new relative output required')
        result = execute(args, public, helper, plan)
    else:
        result = dict(status='PREPARATION_ONLY_CURRENT_BATCH_NOT_READ', helper_sha256=HELPER_SHA,
                      reused_build_restore_code_changed=False, original_queue_scopes=len(plan['full_scope_queue']),
                      requested_fresh_scope_maximum=12, requested_fresh_trial_maximum=2160,
                      compression_or_current_folder_glob=False, models_arrays_GPU_read=False,
                      framework_imports=[name for name in ('numpy', 'torch', 'pandas') if name in sys.modules])
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
