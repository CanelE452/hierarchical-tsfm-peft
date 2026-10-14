"""Actual CPU guard timing only; no dataset, checkpoint, torch or F reads."""
import argparse
from pathlib import Path
import time
import json

import continuation_runtime_v1 as runtime


def execute():
    output = runtime.HERE / 'validation_v1' / 'CONTINUATION_GUARD_DIAGNOSTIC_v1.json'
    if output.exists():
        raise FileExistsError('FRESH_GUARD_DIAGNOSTIC_REQUIRED')
    limit = runtime.ContinuationLimit('cpu_continuation_guard_diagnostic_v1', gpu=False)
    result = {'status': 'RUNNING', 'GPU_execution': False, 'scientific_assets_opened': False,
              'samples_seconds': {}, 'source_sha256': runtime.sha(__file__),
              'authorization': limit.authorization_record, 'run_manifest': limit.run_manifest_record,
              'scope': 'CPU/file-system guard costs; does not identify GPU clock/thermal causes'}
    try:
        def receipt_hashes():
            for record in limit.run_manifest['protected_prior_receipts']:
                if runtime.sha(record['path']) != record['sha256']:
                    raise ValueError('PROTECTED_PRIOR_RECEIPT_CHANGED')
        callbacks = {'protected_prior_receipt_hashes': receipt_hashes,
                     'storage_scan': lambda: runtime.storage_bytes(limit.run_manifest['storage_roots']),
                     'full_guard': limit.check, 'base_guard': limit.shared.check}
        for name, callback in callbacks.items():
            values = []
            for _ in range(100 if name == 'base_guard' else 3):
                limit.shared.check()
                start = time.perf_counter()
                callback()
                values.append(time.perf_counter() - start)
            result['samples_seconds'][name] = values
        result['protected_prior_receipt_count'] = len(limit.run_manifest['protected_prior_receipts'])
        result['status'] = 'COMPLETE'
    except Exception as error:
        result.update(status='BUDGET_HELD_PRESERVED' if isinstance(error, runtime.BudgetHeld) else 'FAILED_PRESERVED',
                      error={'type': type(error).__name__, 'message': str(error)})
    finally:
        receipt = limit.close(result['status'], result.get('error'))
        result['shared_resource_receipt'] = receipt
        if receipt['terminal_status'] != 'COMPLETE':
            result.update(status=receipt['terminal_status'], terminal_error=receipt['terminal_error'])
        runtime.write_new(output, result)
    print(json.dumps({'status': result['status'], 'output': str(output),
                      'median_seconds': {k: sorted(v)[len(v) // 2] for k, v in result['samples_seconds'].items()}}))
    return 0 if result['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', required=True, action='store_true')
    parser.parse_args()
    raise SystemExit(execute())
