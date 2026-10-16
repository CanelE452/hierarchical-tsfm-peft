"""Explicit public-only caller of the unchanged audit supplied with the plan."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import time


ORIGINAL_RELATIVE = 'plan/audit_public_bundle.py'
ORIGINAL_SHA = '7c5f705770361e81f64c0963245013b2b54d50bcebab6b7c7278df25389fbc2c'
HORIZON_RELATIVE = 'diagnostics/public_horizon_summary_20261016_v1/horizon_summary.py'
HORIZON_SHA = 'ca7d4711fca27c1dcc0903945b3d915f6a016279824736a19c29d58beaa9d9d2'
SOURCE_REFERENCE = 'diagnostics/whole_available_audit_1792092300000/SOURCE_FREEZE.json'
SOURCE_REFERENCE_SHA = 'd5499db440871a928ef4d0275c26a222e657ac9b044d5eaa1295e0bdb2bed9fa'


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def source_root():
    return Path(__file__).resolve().parents[2]


def _load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def dependencies():
    modules = []
    for relative, expected, name in ((HORIZON_RELATIVE, HORIZON_SHA, 'provided_entry_horizon_metadata'),
                                      (ORIGINAL_RELATIVE, ORIGINAL_SHA, 'unchanged_provided_public_audit')):
        path = source_root() / relative
        if sha(path) != expected:
            raise ValueError('Pinned helper SHA differs: ' + relative)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        modules.append(module)
    return tuple(modules)


def source_snapshot(public_root):
    public = Path(public_root).resolve(); source = source_root()
    if sha(source / SOURCE_REFERENCE) != SOURCE_REFERENCE_SHA:
        raise ValueError('Frozen source39 reference SHA differs')
    expected = _load(source / SOURCE_REFERENCE)['source39']
    actual = {name: sha(source / name) for name in expected}
    if len(actual) != 39 or actual != expected:
        raise ValueError('Frozen source39 differs')
    if sha(source / ORIGINAL_RELATIVE) != ORIGINAL_SHA or sha(source / HORIZON_RELATIVE) != HORIZON_SHA:
        raise ValueError('Pinned supplied/metadata helper changed')
    return dict(source39=actual, source39_reference=SOURCE_REFERENCE, source39_reference_sha256=SOURCE_REFERENCE_SHA,
                provided_original_sha256=ORIGINAL_SHA, horizon_metadata_sha256=HORIZON_SHA,
                registry_sha256=sha(public / 'EXPERIMENT_REGISTRY.csv'),
                selection_lock_sha256=sha(public / 'SELECTION_LOCK.json') if (public / 'SELECTION_LOCK.json').is_file() else None)


def metadata_only(scores, public_root, input_manifest):
    horizon, _ = dependencies()
    plan = horizon.metadata_plan(scores, public_root, input_manifest)
    keys = [(r['score']['run_id'], r['score']['split'], r['score']['cohort']) for r in plan['routes']]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate provided-audit score key; do not rename, split, or drop rows')
    plan.update(requested_rows=len(keys), original_provided_source=ORIGINAL_RELATIVE,
                original_provided_sha256=ORIGINAL_SHA, source_snapshot=source_snapshot(public_root),
                caller_sha256=sha(__file__), provided_audit_executed=False)
    return plan


def execute_audit(scores, public_root, input_manifest):
    public = Path(public_root).resolve(); scores = Path(scores).resolve()
    plan = metadata_only(scores, public, input_manifest)
    horizon, original = dependencies()
    output = public / 'diagnostics/provided_public_audit_entry_20261016_v1' / ('snapshot_' + str(time.time_ns()))
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    mapping = []
    for index, route in enumerate(plan['routes']):
        mapping.append(dict(run_id=route['score']['run_id'], split=route['score']['split'], cohort=route['score']['cohort'],
            manifest_relative_path=route['manifest_relative_path'], manifest_sha256=route['manifest_sha256'],
            original_public_relative_path=route['public_stats_relative_path'],
            original_sha256=route['public_stats_sha256'], original_bytes=route['public_stats_bytes'],
            temporary_relative_path=f'PUBLIC_ERROR_STATS/{index:06d}_{route["public_stats_sha256"]}.csv.gz', copy_verified=False))
    audit_result, error, audit_called, flat_path = None, None, False, None
    flat_manifest_text = ''; score_copy_verified = False
    try:
        with tempfile.TemporaryDirectory(prefix='flat_public_', dir=output) as temporary:
            flat_path = Path(temporary)
            (flat_path / 'PUBLIC_ERROR_STATS').mkdir()
            shutil.copyfile(scores, flat_path / 'SCORES.csv')
            if sha(flat_path / 'SCORES.csv') != plan['scores_sha256']:
                raise ValueError('Byte-exact flat score copy differs')
            score_copy_verified = True
            hashes = [(plan['scores_sha256'], 'SCORES.csv')]
            for record in mapping:
                source = horizon.inside(public, record['original_public_relative_path'])
                if sha(source) != record['original_sha256'] or source.stat().st_size != record['original_bytes']:
                    raise ValueError('Original public gzip SHA/size differs before copy')
                copied = flat_path / record['temporary_relative_path']
                shutil.copyfile(source, copied)
                if sha(copied) != record['original_sha256'] or copied.stat().st_size != record['original_bytes']:
                    raise ValueError('Copied public gzip SHA/size differs')
                record['copy_verified'] = True
                hashes.append((record['original_sha256'], record['temporary_relative_path']))
            flat_manifest_text = ''.join(f'{digest}  {relative}\n' for digest, relative in hashes)
            (flat_path / 'PUBLIC_MANIFEST.sha256').write_text(flat_manifest_text, encoding='utf-8')
            audit_called = True
            audit_result = original.audit(flat_path)
            expected = plan['requested_rows']
            if (audit_result['status'] != 'PASS' or audit_result['checked_score_rows'] != expected
                    or audit_result['statistics_files'] != expected or audit_result['manifest_status'] != 'PASS'
                    or audit_result['verified_file_hashes'] != expected + 1):
                raise ValueError('Provided original audit did not cover the entire exact flat bundle')
    except (ValueError, KeyError, OSError, AssertionError, csv.Error, MemoryError) as failure:
        error = dict(type=type(failure).__name__, reason=str(failure) if isinstance(failure, ValueError) else 'Declared public audit/copy failed')
    postflight = []
    try:
        if source_snapshot(public) != plan['source_snapshot']:
            postflight.append('source39, registry, lock, or pinned helper changed')
        if sha(__file__) != plan['caller_sha256']:
            postflight.append('Caller source changed during the audit')
        if sha(scores) != plan['scores_sha256'] or sha(input_manifest) != plan['input_manifest_sha256']:
            postflight.append('Final scores or public input-binding table changed')
        for route in plan['routes']:
            if sha(horizon.inside(public, route['manifest_relative_path'])) != route['manifest_sha256']:
                postflight.append('Official prediction manifest changed: ' + route['manifest_relative_path'])
            if sha(horizon.inside(public, route['public_stats_relative_path'])) != route['public_stats_sha256']:
                postflight.append('Original public statistic changed: ' + route['public_stats_relative_path'])
    except (ValueError, KeyError, OSError):
        postflight.append('Cannot verify declared original inputs after the audit')
    temporary_removed = flat_path is None or not flat_path.exists()
    if not temporary_removed:
        postflight.append('Temporary flat bundle cleanup incomplete')
    status = 'FAIL_PROVIDED_AUDIT_OR_COPY' if error else ('INCOMPLETE_POSTFLIGHT' if postflight else 'PASS_PROVIDED_ORIGINAL_AUDIT')
    bindings = {key: value for key, value in plan.items() if key != 'routes'}
    bindings.update(flat_mapping=mapping, original_score_copy=dict(original_public_relative_path=plan['scores_relative_path'],
                    temporary_relative_path='SCORES.csv', sha256=plan['scores_sha256'], copy_verified=score_copy_verified),
                    flat_manifest_relative_name='PUBLIC_MANIFEST.sha256', flat_manifest_sha256=hashlib.sha256(flat_manifest_text.encode()).hexdigest())
    _write(output / 'INPUT_BINDINGS.json', bindings)
    # This is an evidence copy of an ephemeral flat manifest; original public files stay at their original paths.
    (output / 'FLAT_PUBLIC_MANIFEST.sha256.txt').write_text(flat_manifest_text, encoding='utf-8')
    result = dict(status=status, requested_score_rows=plan['requested_rows'],
        copied_statistics_files=sum(row['copy_verified'] for row in mapping),
        copied_statistics_bytes=sum(row['original_bytes'] for row in mapping if row['copy_verified']),
        provided_audit_called=audit_called, provided_audit=audit_result, failure=error,
        source_and_input_postflight_errors=postflight, postflight_unchanged=not postflight,
        temporary_directory_removed=temporary_removed, elapsed_seconds=time.time() - started,
        provided_original_source=ORIGINAL_RELATIVE, provided_original_sha256=ORIGINAL_SHA,
        provided_original_modified=False, source39_registry_lock_or_scores_modified=False,
        original_gzip_modified=False, source_rows_renamed_split_or_dropped=False,
        public_statistics_reaggregation_only=True, prediction_array_rescoring=False,
        model_NPZ_raw_data_GPU_access=False, checkpoint_replay=False, training_rerun=False,
        T_selection_or_new_inference=False, supplied_selftests_are_separate_synthetic_only=True)
    _write(output / 'RESULT.json', result)
    (output / 'README_KO.md').write_text('# 제공 원본 공개 검산기의 실행 영수증\n\n'
        '최종 점수표의 모든 행과 결속된 원 gzip을 바이트 그대로 임시 flat bundle에 복사한 뒤 제공 audit()를 수정 없이 호출했다. '
        '임시 SCORES와 모든 통계 SHA를 flat manifest에 넣었다. 원본 파일은 보존했고 임시 폴더는 자동 정리한다. '
        '원본 통계의 수치/coverage/SAE-SSE/manifest 검산과 source·registry·lock 사후검사를 구분한다. '
        '이 결과는 원예측 검산·학습 재실행·추론·T 선택·비용 측정이 아니다. 제공3개 selftest는 별도 합성 준비 증거다.\n\n'
        '- [실행·사후 상태](RESULT.json)\n- [원→임시 상대 경로와 입력 SHA](INPUT_BINDINGS.json)\n'
        '- [임시 flat manifest 원문](FLAT_PUBLIC_MANIFEST.sha256.txt)\n', encoding='utf-8')
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-root', type=Path, required=True)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--input-manifest', type=Path, required=True)
    parser.add_argument('--execute-audit', action='store_true')
    args = parser.parse_args(argv)
    if args.execute_audit:
        output = execute_audit(args.scores, args.public_root, args.input_manifest)
        result = _load(output / 'RESULT.json')
        print(json.dumps(dict(status=result['status'], requested_rows=result['requested_score_rows'],
                             output=output.relative_to(args.public_root.resolve()).as_posix())))
        return 0 if result['status'] == 'PASS_PROVIDED_ORIGINAL_AUDIT' else 1
    plan = metadata_only(args.scores, args.public_root, args.input_manifest)
    print(json.dumps(dict(status=plan['status'], requested_rows=plan['requested_rows'], gzip_bytes_read=0,
                         original_provided_sha256=ORIGINAL_SHA, caller_sha256=plan['caller_sha256'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
