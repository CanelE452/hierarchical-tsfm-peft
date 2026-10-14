from pathlib import Path
import argparse
import os
import sys
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(name,'2')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE),str(ROOT)]
import adaptive_runtime_v1 as runtime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag',required=True)
    parser.add_argument('tests',nargs='+')
    args = parser.parse_args()
    paths = [(ROOT/Path(p)).resolve() for p in args.tests]
    if any(not p.is_relative_to(HERE) or not p.name.startswith('test_') for p in paths):
        raise ValueError('ONLY_NEW_CAMPAIGN_TEST_SOURCES_ALLOWED')
    limit = runtime.ContinuationLimit(args.tag,gpu=False)
    try:
        import pytest
        code = pytest.main(['-q','-p','no:cacheprovider',*[str(p) for p in paths]])
        terminal = limit.close('COMPLETE' if code == 0 else 'FAILED_PRESERVED',None if code == 0 else {'pytest_exit':code})
        return code if terminal['terminal_status'] == 'COMPLETE' else max(1,int(code))
    except BaseException as error:
        if not limit.closed:
            limit.close('FAILED_PRESERVED',{'type':type(error).__name__,'message':str(error)})
        raise

if __name__ == '__main__':
    raise SystemExit(main())
