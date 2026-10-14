"""Metered fresh runtime-fixture corrections and first CPU-runner fixtures."""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))


def main():
    import stage_separation_runtime_v1 as runtime
    output = HERE/'validation_v1/STAGE_REPAIRED_RUNTIME_AND_NATIVE_FIXTURES_v1.json'
    if output.exists():
        raise FileExistsError(output)
    limit = runtime.ContinuationLimit('stage_repaired_runtime_and_native_fixtures_v1', gpu=False)
    paths = [HERE/'test_stage_separation_runtime_v2.py', HERE/'test_run_stage_native_ridge_v1.py']
    result = {'status': 'RUNNING', 'scope': 'synthetic CPU fixture data only; no scientific assets/pretrained models',
              'sources': [{'path': str(p), 'sha256': runtime.sha(p)} for p in paths]}
    try:
        import pytest
        code = pytest.main(['-q', *[str(path) for path in paths]])
        result.update(status='COMPLETE' if code == 0 else 'FAILED_PRESERVED', pytest_exit_code=int(code))
        limit.check()
    except Exception as error:
        result.update(status='FAILED_PRESERVED', error={'type': type(error).__name__, 'message': str(error)})
    finally:
        result['resource_receipt'] = limit.close(result['status'], result.get('error'))
        if result['resource_receipt']['terminal_status'] != 'COMPLETE':
            result['status'] = result['resource_receipt']['terminal_status']
        runtime.write_new(output, result)
    print(json.dumps({'status': result['status']}))
    raise SystemExit(0 if result['status'] == 'COMPLETE' else 1)


if __name__ == '__main__':
    main()
