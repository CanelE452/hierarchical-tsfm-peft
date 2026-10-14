"""Final receipt accounting; no model, data, or scientific-array reads."""
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY))


def main():
    import continuation_runtime_v2 as runtime
    output = HERE/'CONTINUATION_FINAL_ACCOUNTING_v1.json'
    if output.exists():
        raise FileExistsError(output)
    limit = runtime.ContinuationLimit('continuation_final_accounting_v1', gpu=False)
    last = limit.check()
    receipt = limit.close('COMPLETE')
    rows = []
    for path in sorted(runtime.RESOURCES.glob('*.json')):
        if path.name == 'ACTIVE.json':
            raise RuntimeError('NEXT_ACTUAL_JOB_LIVE')
        record = json.loads(path.read_text(encoding='utf-8'))
        if record['status'] not in runtime.TERMINAL:
            raise ValueError('NONTERMINAL_RECEIPT')
        rows.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'gpu': record['gpu'], 'wall_seconds': record['wall_seconds'], 'status': record['status']})
    gpu = sum(r['wall_seconds'] for r in rows if r['gpu'])
    cpu = sum(r['wall_seconds'] for r in rows if not r['gpu'])
    result = {'status': 'COMPLETE', 'captured_unix': time.time(), 'receipt_rows': rows,
              'gpu_process_wall_seconds': gpu, 'cpu_process_wall_seconds': cpu,
              'remaining_gpu_seconds': 1800-gpu, 'remaining_cpu_seconds': 1800-cpu,
              'remaining_campaign_wall_seconds': max(0, limit.run_manifest['expires_unix']-time.time()),
              'fixed_expiry_unix': limit.run_manifest['expires_unix'],
              'storage_at_last_guard': last, 'active_actual_execution_lease_absent': True,
              'all_receipts_terminal': True, 'shared_resource_receipt': receipt,
              'whole_research_goal_complete': False,
              'note': 'Process wall includes guards/load/verification; not GPU kernel time. Remaining GPU/CPU do not extend fixed wall deadline.'}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('status','gpu_process_wall_seconds','cpu_process_wall_seconds','remaining_gpu_seconds','remaining_cpu_seconds','remaining_campaign_wall_seconds')}))


if __name__ == '__main__':
    main()
