"""Actual prediction persistence and small full-path timing for direct prototypes."""
import time

import numpy as np
import torch

from common_direct import HERE, ROOT, CACHE, write_json, digest, observed_metrics
from data_io import make_windows


def benchmark_path(infer, data, resources, label, reference_prediction):
    output = []
    past = make_windows(data, data['val_origins'])
    for batch in (1, 32):
        positions = np.arange(batch) % len(past)
        physical = np.asarray(past[positions] * data['std'] + data['mean'], dtype=np.float32)

        def deployed():
            normalized = (physical - data['mean']) / data['std']
            x = torch.from_numpy(normalized).cuda()
            with torch.no_grad():
                prediction = infer(x)
            host = prediction.detach().cpu().numpy()
            torch.cuda.synchronize()
            return host

        with resources.stage('actual_full_path_timing', gpu=True, method=label, batch=batch):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            reference = deployed()
            first_seconds = time.perf_counter() - start
            error = float(np.max(np.abs(reference - reference_prediction[positions])))
            if not np.allclose(reference, reference_prediction[positions], atol=5e-4, rtol=5e-4):
                raise RuntimeError(f'Benchmark prediction differs from full VAL: {label} B{batch}')
            for _ in range(3):
                deployed()
            samples = []
            for _ in range(10):
                started = time.perf_counter()
                deployed()
                samples.append(time.perf_counter() - started)
                resources.check(stage='timing', method=label, batch=batch)
            output.append({'method': label, 'batch': batch, 'distinct_origins': len(np.unique(positions)),
                           'fixture_origin_positions': positions,
                           'warmup': 3, 'timed': 10, 'seconds': samples,
                           'p50_ms': float(np.median(samples) * 1000),
                           'p95_ms': float(np.percentile(samples, 95) * 1000),
                           'first_forward_seconds': first_seconds,
                           'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                           'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
                           'sampled_process_rss_bytes': resources.proc.memory_info().rss,
                           'prediction_parity_max_abs': error,
                           'memory_scope': 'reset before parity/3warm/10timed; all currently resident prototype modules',
                           'boundary': 'preloaded physical FP32 past -> CPU normalization -> H2D -> entire method -> D2H -> CUDA sync',
                           'excluded': 'startup, model load, file I/O, window extraction, scoring',
                           'device': torch.cuda.get_device_name(),
                           'precision': 'FP32, no AMP, TF32 disabled'})
    return output


def save_case(job, data, reconstruction, forecasts, predictions, diagnostic,
              training, calibration, shape, cost, resources, source_files):
    folder = HERE / 'runs' / job['id']
    arrays = CACHE / 'runs' / job['id']
    folder.mkdir(parents=True, exist_ok=True)
    arrays.mkdir(parents=True, exist_ok=True)
    prediction_path = arrays / 'prediction.npz'
    diagnostic_path = arrays / 'diagnostic.npz'
    np.savez_compressed(prediction_path, targets=make_windows(data, data['val_origins'], future=True),
                        mask=make_windows(data, data['val_origins'], future=True, masked=True)[1],
                        origins=data['val_origins'], mean=data['mean'], std=data['std'], **predictions)
    np.savez_compressed(diagnostic_path, **diagnostic)
    pins = {str(path.relative_to(ROOT)): digest(path) for path in source_files}
    result = {'status': 'complete', 'ended_unix': time.time(), 'job': job,
              'data_contract_hash': data['contract_hash'], 'reconstruction': reconstruction,
              'forecast': forecasts, 'prediction_path': prediction_path,
              'prediction_sha256': digest(prediction_path), 'diagnostic_path': diagnostic_path,
              'diagnostic_sha256': digest(diagnostic_path), 'shape': shape,
              'training': training, 'calibration': calibration, 'cost': cost,
              'resources': dict(resources.value), 'source_pins': pins,
              'scope': 'existing development VAL; no protected TEST access; no LEVEL/D8/residual G'}
    write_json(folder / 'result.json', result)
    interim = HERE / 'INTERIM_RESULTS_KO.md'
    if not interim.exists():
        lines = ['# 교수님 제안 직접 실험 — 첫 실제 결과', '',
                 '실제 학습·미래 추론을 완료한 첫 조건이다. 전체 16개 표현학습 조건의 완료를 뜻하지 않는다.', '',
                 f"조건: `{job['id']}`. 기존 개발 VAL이며 독립 TEST 검증이 아니다.", '',
                 f"입력 재구성 MSE={reconstruction['mse']:.6f}, MAE={reconstruction['mae']:.6f}.", '',
                 '```text', '미래 예측 경로              MSE          MAE']
        for name, metric in forecasts.items():
            lines.append(f"{name:26} {metric['mse']:12.6f} {metric['mae']:12.6f}")
        lines += ['```', '', f"상세 실제 산출물: [result](runs/{job['id']}/result.json).", '',
                  '재구성 디코더와 같은 디코더의 supervised forecast calibration을 구분했다. '
                  'GT는 native graph-time decoder를 유지하며 TS2Vec는 별도 비선형 복원기를 사용한다. '
                  '원문 변경과 수치 검산·예측 곡선은 후속 보고에 연결한다.']
        interim.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result
