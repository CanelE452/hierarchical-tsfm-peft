"""Fresh full-size TS2 run with process-local allocator/cache operating settings."""
import atexit
import json
import os
from pathlib import Path
import runpy
import sys
import time


# Apply before PyTorch import; neither network geometry nor contrastive objective changes.
if 'torch' in sys.modules:
    raise RuntimeError('Allocator retry settings must precede torch import')
settings = {'PYTORCH_ALLOC_CONF': 'backend:native,garbage_collection_threshold:0.6',
            'TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT': '128'}
os.environ.update(settings)
os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)
started = time.time()
source = Path(__file__).resolve()
arguments = sys.argv[1:]


def argument(name):
    return arguments[arguments.index(name) + 1]


identifier = f"{argument('--dataset')}_TS2_d{argument('--dim')}_s{argument('--seed')}_retry_{argument('--retry-tag')}"
from common_direct import HERE, ROOT, digest, write_json
import torch

receipt = {'job_id': identifier, 'pid': os.getpid(), 'started_unix': started,
           'settings_before_torch_import': True,
           'operational_environment': settings,
           'encoder_batch': int(argument('--encoder-batch')) if '--encoder-batch' in arguments else 8,
           'official_geometry_or_loss_changed': False,
           'cuda_allocator_backend': torch.cuda.get_allocator_backend(),
           'wrapper_sha256': digest(source),
           'runner_sha256': digest(HERE / 'runner_ts2.py'),
           'command': [sys.executable, str(source), *arguments],
           'why': 'Prior Jena attempts hit host RSS cap while CUDA cached reservation was 19-21GiB; cache/allocator cause remains a hypothesis.',
           'primary_references': ['https://docs.pytorch.org/docs/2.10/notes/cuda.html',
                                  'https://docs.pytorch.org/docs/stable/cuda_environment_variables.html'],
           'status': 'RUNNING'}
path = HERE / 'execution_environment' / f'{identifier}.json'
if path.exists():
    raise FileExistsError('Preserve an existing memory retry environment receipt')
write_json(path, receipt)


def finish():
    receipt['ended_unix'] = time.time()
    receipt['result_exists'] = (HERE / 'runs' / identifier / 'result.json').exists()
    receipt['status'] = 'RESULT_EXISTS_PENDING_AUDIT' if receipt['result_exists'] else 'INCOMPLETE'
    write_json(path, receipt)


atexit.register(finish)
sys.argv = [str(HERE / 'runner_ts2.py'), *arguments]
runpy.run_path(str(HERE / 'runner_ts2.py'), run_name='__main__')
