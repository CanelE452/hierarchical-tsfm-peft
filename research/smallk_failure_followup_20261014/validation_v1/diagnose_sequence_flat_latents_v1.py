"""Saved latent arithmetic only; conditional toy frequency, not a causal test."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
ROOT = STUDY.parents[1]
sys.path.insert(0, str(STUDY))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(record):
    path = Path(record['path'])
    path = path if path.is_absolute() else ROOT/path
    if any(part.lower() == 'test' for part in path.parts):
        raise PermissionError('PROTECTED_TEST_PATH')
    if sha(path) != record['sha256']:
        raise ValueError('SOURCE_SHA_MISMATCH:' + str(path))
    return path


def main():
    from continuation_runtime_v2 import ContinuationLimit
    queue_path = ROOT/'.cache'/STUDY.name/'forecast_sequence_v1/FORECAST_SEQUENCE_QUEUE_v1.json'
    output = HERE/'SEQUENCE_FLAT_LATENT_DIAGNOSTIC_v1.json'
    csv_path = HERE/'SEQUENCE_FLAT_LATENT_DIAGNOSTIC_v1.csv'
    if output.exists() or csv_path.exists():
        raise FileExistsError('FRESH_OUTPUT_REQUIRED')
    limit = ContinuationLimit('sequence_flat_latent_diagnostic_v1', gpu=False)
    result = {'status': 'RUNNING', 'rows': [], 'source_records': [],
              'scope': 'saved forecast-latent minus saved last-past latent arithmetic only',
              'GPU_execution': False, 'E_or_F_replayed': False,
              'causal_failure_identified': False, 'whole_goal_complete': False}
    try:
        queue_before = sha(queue_path)
        queue = json.loads(queue_path.read_text(encoding='utf-8-sig'))
        if queue['status'] != 'COMPLETE' or len(queue['conditions']) != 8:
            raise ValueError('ALL8_TERMINAL_REQUIRED')
        for item in queue['conditions']:
            limit.shared.check()
            producer_path = checked(item['producer_result'])
            producer = json.loads(producer_path.read_text(encoding='utf-8'))
            if producer['status'] != 'COMPLETE':
                raise ValueError('PRODUCER_COMPLETE_REQUIRED')
            result['source_records'].append(item['producer_result'])
            for split, record in producer['selected_inference'].items():
                path = checked(record['artifact'])
                with np.load(path, allow_pickle=False) as arrays:
                    past = arrays['past_z_last_two'].astype(np.float64)
                    future = arrays['absolute_future_z'].astype(np.float64)
                innovation = future-past[:, -1:]
                if not np.isfinite(innovation).all():
                    raise ValueError('NONFINITE_SAVED_LATENTS')
                maximum = np.max(np.abs(innovation), axis=(1, 2))
                result['rows'].append({'dataset': producer['dataset'], 'seed': producer['seed'],
                    'head': producer['head'], 'split': split, 'windows': len(maximum),
                    'selected_new_update': producer['selected_new_update'],
                    'exact_flat_windows': int((maximum == 0).sum()),
                    'exact_flat_fraction': float((maximum == 0).mean()),
                    'abs_le_1e_minus5_windows': int((maximum <= 1e-5).sum()),
                    'abs_le_1e_minus5_fraction': float((maximum <= 1e-5).mean()),
                    'max_abs_innovation_min': float(maximum.min()),
                    'max_abs_innovation_median': float(np.median(maximum)),
                    'max_abs_innovation_p95': float(np.quantile(maximum, .95)),
                    'max_abs_innovation_max': float(maximum.max()),
                    'threshold_scope': 'absolute latent units 1e-5; descriptive, not normalized across encoders'})
                result['source_records'].append(record['artifact'])
                checked(record['artifact'])
        if len(result['rows']) != 24 or sha(queue_path) != queue_before:
            raise ValueError('COVERAGE_OR_QUEUE_PRESERVATION_FAILED')
        result['queue_source'] = {'path': str(queue_path), 'sha256': queue_before}
        result['status'] = 'COMPLETE'
        limit.check()
    except Exception as error:
        result.update(status='FAILED_PRESERVED', error={'type': type(error).__name__, 'message': str(error)})
    finally:
        result['shared_resource_receipt'] = limit.close(result['status'], result.get('error'))
        if result['shared_resource_receipt']['terminal_status'] != 'COMPLETE':
            result['status'] = result['shared_resource_receipt']['terminal_status']
        result['diagnostic_source_sha256'] = sha(__file__)
        if result['rows']:
            with csv_path.open('x', encoding='utf-8-sig', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(result['rows'][0]))
                writer.writeheader()
                writer.writerows(result['rows'])
            result['csv'] = {'path': str(csv_path), 'sha256': sha(csv_path)}
        with output.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'status': result['status'], 'rows': len(result['rows'])}))
    return 0 if result['status'] == 'COMPLETE' else 1


if __name__ == '__main__':
    raise SystemExit(main())
