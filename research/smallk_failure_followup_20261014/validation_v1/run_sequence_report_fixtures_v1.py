"""Metered new CPU fixtures after all actual fitting has ended."""
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY.parents[1]))
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(HERE))


def main():
    from continuation_runtime_v2 import ContinuationLimit
    output = HERE/'SEQUENCE_REPORT_FIXTURES_v1.json'
    if output.exists():
        raise FileExistsError(output)
    limit = ContinuationLimit('sequence_report_fixtures_v1', gpu=False)
    paths = [STUDY/'test_make_sequence_followup_report_v1.py']
    result = {'status': 'RUNNING', 'scope': 'synthetic CPU fixture tests only; no scientific assets or models',
              'GPU_execution': False, 'whole_goal_complete': False,
              'sources': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]}
    try:
        import pytest
        code = pytest.main(['-q', *[str(p) for p in paths]])
        result.update(pytest_exit_code=int(code), status='COMPLETE' if code == 0 else 'FAILED_PRESERVED')
        limit.check()
    except Exception as error:
        result.update(status='FAILED_PRESERVED', error={'type': type(error).__name__, 'message': str(error)})
    finally:
        result['shared_resource_receipt'] = limit.close(result['status'], result.get('error'))
        if result['shared_resource_receipt']['terminal_status'] != 'COMPLETE':
            result['status'] = result['shared_resource_receipt']['terminal_status']
        with output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return 0 if result['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
