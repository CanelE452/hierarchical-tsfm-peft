"""Preserve the separate historical resource-limited run without counting it twice."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time


HELPER = 'diagnostics/publication_multik_lossless_1792094047117000000/bundle_multik.py'
HELPER_SHA = '5632fa60313c41d62b1573ca67225cfb9ad9dadd7e494ba14c506388548a24dc'
SOURCE = 'costs/multik_transition_1792096238448295300'
BATCH_SHA = '743d5207b604c3d5476521da001e8eab24928b80f6479fad592fa7f88cdaba04'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-root', type=Path, required=True)
    parser.add_argument('--private-root', type=Path, required=True)
    parser.add_argument('--execute-pack', action='store_true')
    args = parser.parse_args()
    public = args.public_root.resolve()
    if sha(public / HELPER) != HELPER_SHA or sha(public / SOURCE / 'BATCH_LAUNCHER_RECEIPT.json') != BATCH_SHA:
        raise ValueError('Pinned original helper or historical batch changed')
    if not args.execute_pack:
        print(json.dumps(dict(status='METADATA_ONLY_NO_COMPRESSION', helper_sha256=HELPER_SHA, historical_batch_sha256=BATCH_SHA)))
        return
    if (args.private_root / 'ACTIVE_GPU_LEASE.json').exists():
        raise ValueError('Wait for cost owner terminal')
    frozen = json.loads((public / 'diagnostics/whole_available_audit_1792092300000/SOURCE_FREEZE.json').read_bytes())['source39']
    if len(frozen) != 39 or any(sha(public / name) != value for name, value in frozen.items()):
        raise ValueError('Original source39 changed')
    metadata = {name: sha(public / name) for name in ('EXPERIMENT_REGISTRY.csv', 'SELECTION_LOCK.json')}
    batch = json.loads((public / SOURCE / 'BATCH_LAUNCHER_RECEIPT.json').read_bytes())
    if batch['complete_scopes'] != 8 or batch['actual_numeric_trials'] != 1606 or batch['actual_resource_valid_trials'] != 1605:
        raise ValueError('Historical reused1440 plus partial166/valid165 changed')
    paths = list((public / SOURCE).glob('*/*/RESULT.json'))
    if len(paths) != 1:
        raise ValueError('Expected one historical partial scope')
    partial = json.loads(paths[0].read_bytes())
    if len(partial['rows']) != 166 or sum(row['resource_within_limit'] for row in partial['rows']) != 165 or partial['exact_minimum_protocol_complete']:
        raise ValueError('Historical partial status or rows changed')
    spec = importlib.util.spec_from_file_location('unchanged_multik_lossless', public / HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    if sha(public / 'measure_multik_transition.py') != helper.WRITER_SHA256:
        raise ValueError('Original measurement writer changed')
    out = Path(__file__).resolve().parent / ('bundle_' + str(time.time_ns()))
    manifest = helper.build(public, public / SOURCE, out, 1, 166,
                            dict(relative='measure_multik_transition.py', sha256=helper.WRITER_SHA256))
    if any(sha(public / name) != value for name, value in {**frozen, **metadata}.items()):
        raise ValueError('Source39, registry, or selection changed during packing')
    helper.json_new(out / 'HISTORICAL_SCOPE.json', dict(status='HISTORICAL_PARTIAL_PRESERVED_NOT_ADDED_TO_FINAL3600',
        original_batch_sha256=BATCH_SHA, actual_fresh_trials=166, resource_valid_fresh_trials=165,
        reused_old_trials=1440, full_protocol_complete=False, original_source_preserved=True,
        source39_registry_lock_preserved=True, model_GPU_or_new_measurement=False))
    (out / 'README_KO.md').write_text('# 기존 자원 제한 부분 실행의 무손실 보존\n\n'
        '이 묶음은 이전 실행의 새 trial166개(자원 유효165개)와 전체 queue·부분 상태를 보존한다. '
        '원8개 재사용1440은 외부 원 참조로 남아 있으며 여기서 다시 포장하거나 최종 K20의3600에 더하지 않는다. '
        '부분 실행의 실패·자원 상태를 완료로 바꾸지 않았다. 새 학습·추론·비용 측정이 아니다.\n\n'
        '- [원 경로·바이트·SHA](MANIFEST.json)\n- [별도 임시 전수 복원·반복 복원](BUILD_RESULT.json)\n'
        '- [중복 계산하지 않는 실행 범위](HISTORICAL_SCOPE.json)\n\n'
        '원 bundle_multik.py restore --manifest <이 폴더/MANIFEST.json> --public-root <campaign>로 복원한다.\n', encoding='utf-8')
    print(json.dumps(dict(status='PASS_HISTORICAL_PARTIAL_BYTE_PRESERVATION', output=out.relative_to(public).as_posix(),
                          original_files=manifest['original_files'], trial_count=manifest['trial_count'])))


if __name__ == '__main__':
    main()
