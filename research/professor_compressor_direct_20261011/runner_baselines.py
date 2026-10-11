"""New matched Raw F and fixed mean-time-compression controls."""
import argparse
from pathlib import Path
import time

import numpy as np
import torch

from common_direct import HERE, CACHE, Resources, write_json, digest, observed_metrics
from data_io import load_data, make_windows
from foundation import Foundation
from run_support import benchmark_path


def run(dataset):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    resources = Resources()
    data = load_data(dataset)
    with resources.stage('matched_control_frozen_F', gpu=True, dataset=dataset):
        foundation = Foundation(resources)
        parity = foundation.verify_pipeline_parity()
        past = make_windows(data, data['val_origins'])
        target, mask = make_windows(data, data['val_origins'], future=True, masked=True)
        predictions = {}
        for name in ('RAW_F', 'MEAN_BLOCK_F'):
            chunks = []
            for start in range(0, len(past), 4):
                x = torch.as_tensor(past[start:start + 4], device='cuda')
                if name == 'RAW_F':
                    y = foundation.forecast(x, 48)
                else:
                    z = x.reshape(len(x), 128, 4, x.shape[-1]).mean(2)
                    y = foundation.forecast(z, 12).repeat_interleave(4, dim=1)
                chunks.append(y.cpu().numpy())
                resources.check(dataset=dataset, phase=name, origins_done=min(start + 4, len(past)), origins_total=len(past))
            predictions[name] = np.concatenate(chunks)
    costs = []
    for name in predictions:
        def infer(x, name=name):
            if name == 'RAW_F':
                return foundation.forecast(x, 48)
            z = x.reshape(len(x), 128, 4, x.shape[-1]).mean(2)
            return foundation.forecast(z, 12).repeat_interleave(4, dim=1)
        costs.extend(benchmark_path(infer, data, resources, name, predictions[name]))
    arrays = CACHE / 'baselines' / dataset
    arrays.mkdir(parents=True, exist_ok=True)
    path = arrays / 'prediction.npz'
    np.savez_compressed(path, targets=target, mask=mask, origins=data['val_origins'],
                        mean=data['mean'], std=data['std'], **predictions)
    resources.flush()
    result = {'status': 'complete', 'ended_unix': time.time(),
              'job': {'id': dataset + '_BASELINES', 'dataset': dataset, 'family': 'BASELINES', 'seed': None},
              'data_contract_hash': data['contract_hash'], 'forecast': {
                  name: observed_metrics(value, target, mask, data['std']) for name, value in predictions.items()},
              'prediction_path': path, 'prediction_sha256': digest(path), 'cost': costs,
              'F_pipeline_parity': parity, 'F_guard': foundation.verify_unchanged(),
              'source_pins': {str(file): digest(file) for file in (Path(__file__), HERE / 'data_io.py',
                             HERE / 'foundation.py', HERE / 'run_support.py', HERE / 'common_direct.py')},
              'comparison_note': 'mean block compression R4 and repetition are fixed; GT has twice as many F rows and trained filters, so capacity is not matched',
              'resources': resources.value}
    write_json(HERE / 'baselines' / dataset / 'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('robin', 'jena'), required=True)
    run(parser.parse_args().dataset)
