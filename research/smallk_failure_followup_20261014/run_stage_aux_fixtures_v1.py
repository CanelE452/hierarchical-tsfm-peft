"""Explicitly metered CPU fixtures, fresh outputs only; synthetic assets only."""
from pathlib import Path
import argparse
import json
import os
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tests', nargs='+', required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    import stage_separation_runtime_v1 as runtime
    output = ROOT / args.out
    if output.exists(): raise FileExistsError(output)
    limit = runtime.ContinuationLimit(args.tag, gpu=False)
    paths = [ROOT / p for p in args.tests]
    result = {'status': 'RUNNING', 'scope': 'synthetic CPU fixtures only; no real scientific assets/GPU',
              'sources': [{'path': str(p), 'sha256': runtime.sha(p)} for p in paths]}
    try:
        import pytest
        code = pytest.main(['-q', *map(str, paths)])
        result.update(status='COMPLETE' if code == 0 else 'FAILED_PRESERVED', pytest_exit_code=int(code))
        limit.checkfull()
    except Exception as error:
        result.update(status='FAILED_PRESERVED', error={'type': type(error).__name__, 'message': str(error)})
    finally:
        result['resource_receipt'] = limit.close(result['status'], result.get('error'))
        if result['resource_receipt']['terminal_status'] != 'COMPLETE': result['status'] = result['resource_receipt']['terminal_status']
        runtime.write_new(output, result)
    print(json.dumps({'status': result['status']}))
    return 0 if result['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
