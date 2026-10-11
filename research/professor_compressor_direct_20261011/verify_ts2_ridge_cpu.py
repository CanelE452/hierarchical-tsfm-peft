"""Independent sklearn cross-check of the observed-mask native ridge math."""
import time

from common_direct import HERE, Resources, digest, write_json
from runner_ts2 import verify_ridge_math


if __name__ == '__main__':
    resources = Resources()
    with resources.stage('TS2_native_ridge_independent_CPU', gpu=False):
        result = verify_ridge_math()
    resources.flush()
    path = HERE / f'TS2_RIDGE_CPU_VERIFICATION_{time.time_ns()}.json'
    write_json(path, {**result, 'resource_receipt': str(resources.path),
                     'runner_sha256': digest(HERE/'runner_ts2.py'),
                     'actual_data_fit': 0, 'GPU_jobs': 0, 'protected_TEST_reads': 0})
    print(path, flush=True)
