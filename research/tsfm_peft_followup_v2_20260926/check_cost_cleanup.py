"""Reproduce benchmark cleanup failure and test the cuBLAS-workspace cause."""
import gc
import json
import weakref

import numpy as np
import torch

from measure_cost import cuda_cleanup, infer, load_candidate
from run import GPUJob, HERE, load_data, save_json, source_receipt


def ordinary_cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return {'allocated': torch.cuda.memory_allocated(), 'reserved': torch.cuda.memory_reserved()}


def main():
    output = HERE / 'cost_cleanup_check.json'
    assert not output.exists()
    data = load_data('electricity')
    origins = data['val_origins'][np.linspace(0, len(data['val_origins']) - 1, 24, dtype=int)]
    inputs = torch.from_numpy(np.stack([data['x'][o - 512:o] for o in origins]))
    result = {'source': source_receipt(), 'fit_attempts': 0,
              'hypothesis': 'cuBLAS workspaces survive model deletion and ordinary empty_cache',
              'reference': 'https://docs.pytorch.org/docs/2.10/notes/cuda.html#cublas-workspaces'}
    with GPUJob('cost_cleanup_diagnostic') as job:
        model, _ = load_candidate({'id': 'cleanup_f0', 'dataset': 'electricity'}, data, inputs, job)
        reference = weakref.ref(model)
        infer(model, inputs[:1])
        infer(model, inputs[:4])
        del model
        result['ordinary_cleanup'] = ordinary_cleanup()
        result['model_released'] = reference() is None
        result['after_workspace_clear'] = cuda_cleanup()
        assert result['model_released']
        assert result['ordinary_cleanup']['allocated'] > 0
        assert result['after_workspace_clear']['allocated'] == 0
        result['status'] = 'pass'
        job.heartbeat('workspace cause reproduced; model released and allocator returned to zero')
    save_json(output, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'source'}))


if __name__ == '__main__':
    main()
