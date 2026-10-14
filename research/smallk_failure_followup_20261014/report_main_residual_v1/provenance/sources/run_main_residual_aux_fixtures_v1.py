"""Run synthetic fixtures under the immutable MAIN-residual CPU ledger."""
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
    import main_residual_runtime_v1 as runtime
    limit = runtime.ContinuationLimit(args.tag, gpu=False)
    output = ROOT / args.out
    result = {'status': 'RUNNING', 'scope': 'synthetic CPU fixtures only; no real data/models/GPU'}
    try:
        if output.exists():
            raise FileExistsError(output)
        paths = [ROOT / p for p in args.tests]
        result['sources_before'] = [{'path': str(p), 'sha256': runtime.sha(p)} for p in paths]
        limit.checkfull()
        import pytest
        code = pytest.main(['-q', *map(str, paths)])
        result.update(status='COMPLETE' if code == 0 else 'FAILED_PRESERVED', pytest_exit_code=int(code))
        result['sources_after'] = [{'path': str(p), 'sha256': runtime.sha(p)} for p in paths]
        if result['sources_before'] != result['sources_after']:
            raise RuntimeError('FIXTURE_SOURCE_CHANGED')
        limit.checkfull()
    except Exception as error:
        result.update(status='BUDGET_HELD_PRESERVED' if isinstance(error, runtime.BudgetHeld) else 'FAILED_PRESERVED',
                      error={'type': type(error).__name__, 'message': str(error)})
    finally:
        result['resource_receipt'] = limit.close(result['status'], result.get('error'))
        if result['resource_receipt']['terminal_status'] != 'COMPLETE':
            result['status'] = result['resource_receipt']['terminal_status']
        if not output.exists():
            runtime.write_new(output, result)
    print(json.dumps({'status': result['status']}))
    return 0 if result['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
