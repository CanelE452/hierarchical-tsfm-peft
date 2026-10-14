"""CPU guard timing after the actual bounded joint-fit stop; no model forward."""
from pathlib import Path
import json
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import followup_runtime as runtime


def main():
    output = HERE / 'JOINT_GUARD_TIMING_DIAGNOSTIC_v1.json'
    if output.exists():
        raise FileExistsError(output)
    limit = runtime.FollowupLimit('joint_guard_timing_v1', gpu=False)
    result = {'status': 'RUNNING', 'GPU_inference': False, 'protected_TEST_read': False,
              'scope': 'CPU timing of resource guard components only; not proof of GPU/training failure cause', 'measurements': {}}
    try:
        functions = {'base_process_campaign_guard': limit.shared.check,
                     'CPU_receipt_scan': lambda: runtime._previous_cpu_seconds(limit.rt),
                     'followup_storage_scan': lambda: runtime._storage_bytes(limit.followup_roots),
                     'full_followup_guard': limit.check}
        for name, function in functions.items():
            samples = []
            for _ in range(5):
                started = time.perf_counter()
                function()
                samples.append(time.perf_counter() - started)
            result['measurements'][name] = {'seconds': samples, 'median_seconds': statistics.median(samples)}
        result['status'] = 'COMPLETE'
    except Exception as error:
        result.update(status='FAILED_PRESERVED', error={'type': type(error).__name__, 'message': str(error)})
        raise
    finally:
        result['shared_resource_receipt'] = limit.close(result['status'], result.get('error'))
        if result['shared_resource_receipt'].get('terminal_status') != 'COMPLETE':
            result['status'] = result['shared_resource_receipt']['terminal_status']
        with output.open('x', encoding='utf-8') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
    print(json.dumps({'status': result['status'], 'medians_seconds': {name: row['median_seconds'] for name, row in result['measurements'].items()}, 'path': str(output)}, ensure_ascii=True))
    return 0 if result['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
