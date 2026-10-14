"""Activate the already human-approved new window and meter its CPU fixtures.

No scientific data/CP/model assets are read. Bootstrap wall is included in the
first CPU process receipt by the process-birth accounting in the stage runtime.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))


def main():
    import psutil
    import stage_separation_runtime_v1 as runtime
    auth = runtime.read(runtime.AUTHORIZATION)
    if (auth.get('status') != 'APPROVED' or auth.get('origin') != 'human_user_response'
            or auth.get('response_identifier') != 'call_1f2cabd6f99548fabaa58afa3a764c21:0'
            or auth.get('user_response') != '추가 범위·자원 승인'
            or auth.get('approved_plan_sha256') != runtime.sha(runtime.APPROVED_PLAN)
            or auth.get('method_lock_sha256') != runtime.sha(runtime.METHOD_LOCK)
            or auth.get('phase') != 'stage_separation_diagnostics_v1'
            or auth.get('oracle_scope') != 'diagnostic_only_not_deployment'):
        raise PermissionError('ACTUAL_NEW_HUMAN_APPROVAL_REQUIRED')
    if auth.get('limits') != runtime.MAX_LIMITS:
        raise PermissionError('EXACT_NEW_APPROVED_PROPOSAL_LIMITS_REQUIRED')
    fixture_paths = [HERE/'test_stage_separation_math_v1.py', HERE/'test_stage_separation_oracle_v1.py',
                     HERE/'test_stage_separation_runtime_v1.py', HERE/'test_run_stage_oracle_extract_v1.py']
    for path in fixture_paths:
        if not path.is_file():
            raise FileNotFoundError('SOURCE_FIXTURE_NOT_READY:' + str(path))
    output = HERE/'validation_v1/STAGE_INITIAL_FIXTURES_v1.json'
    if output.exists() or runtime.RUN_MANIFEST.exists():
        raise FileExistsError('DO_NOT_REPLACE_ACTIVATION_OR_RECEIPT')
    old_manifest = runtime.read(HERE/'continuation_run_manifest_v1.json')
    protected = {r['path']: r for r in old_manifest['protected_prior_receipts']}
    for path in (HERE/'continuation_resources_v1').glob('*.json'):
        if path.name == 'ACTIVE.json':
            raise RuntimeError('PRIOR_PHASE_ACTIVE_LEASE_PRESENT')
        receipt = runtime.read(path)
        if receipt.get('status') not in runtime.TERMINAL:
            raise RuntimeError('PRIOR_RECEIPT_NOT_TERMINAL:' + str(path))
        protected[str(path.resolve())] = {'path': str(path.resolve()), 'sha256': runtime.sha(path)}
    for record in protected.values():
        if runtime.sha(record['path']) != record['sha256']:
            raise RuntimeError('PRIOR_RECEIPT_CHANGED_BEFORE_ACTIVATION')
    birth = psutil.Process().create_time()
    if birth < auth['approved_unix']:
        raise PermissionError('PROCESS_STARTED_BEFORE_APPROVAL_RECORD')
    storage = runtime.storage_bytes(old_manifest['storage_roots'])
    manifest = {'schema': 'smallk_stage_separation_run_manifest_v1',
                'authorization_sha256': runtime.sha(runtime.AUTHORIZATION),
                'runtime_source_sha256': runtime.sha(runtime.CORE_RUNTIME),
                'started_unix': birth, 'expires_unix': birth+auth['limits']['campaign_wall_seconds'],
                'baseline_storage_bytes': storage, 'storage_roots': old_manifest['storage_roots'],
                'protected_prior_receipts': list(protected.values()),
                'scope': 'new stage-separation approval only; old exhausted ledgers preserved',
                'bootstrap_process_birth_wall_in_first_CPU_receipt': True,
                'activation_source_sha256': runtime.sha(__file__)}
    runtime.write_new(runtime.RUN_MANIFEST, manifest)
    limit = runtime.ContinuationLimit('stage_initial_fixtures_v1', gpu=False)
    result = {'status': 'RUNNING', 'scope': 'synthetic CPU contract fixtures; no scientific assets or pretrained models',
              'run_manifest': limit.run_manifest_record, 'GPU_execution': False,
              'sources': [{'path': str(p), 'sha256': runtime.sha(p)} for p in fixture_paths]}
    try:
        import pytest
        code = pytest.main(['-q', *[str(path) for path in fixture_paths]])
        result.update(status='COMPLETE' if code == 0 else 'FAILED_PRESERVED', pytest_exit_code=int(code))
        limit.check()
    except Exception as error:
        result.update(status='FAILED_PRESERVED', error={'type': type(error).__name__, 'message': str(error)})
    finally:
        result['resource_receipt'] = limit.close(result['status'], result.get('error'))
        if result['resource_receipt']['terminal_status'] != 'COMPLETE':
            result['status'] = result['resource_receipt']['terminal_status']
        result['activation_source_sha256'] = runtime.sha(__file__)
        runtime.write_new(output, result)
    print(json.dumps({'status': result['status'], 'fixed_expiry_unix': manifest['expires_unix']}))
    raise SystemExit(0 if result['status'] == 'COMPLETE' else 1)


if __name__ == '__main__':
    main()
