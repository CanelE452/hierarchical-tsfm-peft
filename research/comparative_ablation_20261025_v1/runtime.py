"""Pinned scientific modules and bounded historical data for this campaign."""
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys


NAME = 'comparative_ablation_20261025_v1'
REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / 'reports' / NAME
CACHE = REPO / '.cache' / NAME
DEFAULT_SOURCE = Path('C:/Users/User/.codex/worktrees/adapter-diagnostics-v2/hierarchical-tsfm-peft')
DEFAULT_ASSET = Path('E:/CODING/proj/hierarchical-tsfm-peft')
KS = dict(jena=6, ETTh1=4, ETTh2_safe17400=4, appliances=10)
FIT_ENDS = dict(jena=(10482, 12579), ETTh1=(8710, 10452),
               ETTh2_safe17400=(8700, 10440), appliances=(9867, 11841))
F_SHA = 'c2b4edf7db22d3f18e58ca79299a5c7e864e9b8f277e4e29e2f88d07dbb35874'
MODEL_REL = '.cache/huggingface/models--amazon--chronos-bolt-small/snapshots/772f3d25d38aec6d914c8949dab4462e2d46f5d8'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path, value):
    path = Path(path)
    if not any(path.resolve().is_relative_to(root) for root in (REPORT, CACHE)):
        raise ValueError('Output is outside the comparative campaign')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def load_modules():
    temporary = CACHE / 'runtime_tmp'
    temporary.mkdir(parents=True, exist_ok=True)
    os.environ['TEMP'] = os.environ['TMP'] = str(temporary)
    import tempfile
    tempfile.tempdir = str(temporary)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '2'
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.dont_write_bytecode = True
    broad = REPO / 'research/informs_broad_validation_20261015_v1'
    if str(broad) not in sys.path:
        sys.path.insert(0, str(broad))
    names = dict(fit='fit_base', data='data_contract', training='training',
                 evaluation='evaluation', networks='networks', gpu='gpu_runtime')
    modules = {key: importlib.import_module(name) for key, name in names.items()}
    if any(Path(module.__file__).resolve().parent != broad for module in modules.values()):
        raise ValueError('A scientific module came from another checkout')
    import torch
    torch.set_num_threads(2)
    return modules


def load_task(dataset, *, source=DEFAULT_SOURCE, asset=DEFAULT_ASSET):
    if dataset not in KS:
        raise ValueError('Dataset is outside the registered campaign')
    source, asset = Path(source).resolve(), Path(asset).resolve()
    modules = load_modules()
    historical = next(row for row in read(source / 'reports/k_tau_rule_20261023_v1/CURRENT_DATA_FINGERPRINTS.json')
                      if row['dataset'] == dataset)
    if dataset == 'jena':
        broad = REPO / 'research/informs_broad_validation_20261015_v1'
        location = broad / 'priority_b2_validation_20261016_v1/safe_jena_prefix.py'
        spec = importlib.util.spec_from_file_location('comparative_safe_jena_runtime', location)
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        task, access = loader.load_jena_prefix(asset, read(broad / 'DATA_CONTRACT.json')['assets']['jena'])
        prefix_matches = task['materialized_prefix_sha256'] == historical['materialized_prefix_sha256']
    else:
        from research.b2_dataset_extension_20261018_v1.train import load_task as load_prefix
        directory = source / '.cache/b2_dataset_search_20261019_v1/data' / dataset
        config_path, prefix_path = directory / 'DATA_DIAGNOSTIC.json', directory / 'PREFIX_PHYSICAL.npz'
        identity = read(source / 'reports/b2_dataset_search_20261019_v1/datasets' / dataset / 'TRAIN_IDENTITY.json')
        if sha(config_path) != identity['input_config_sha256']:
            raise ValueError('Historical prefix configuration changed')
        _, task = load_prefix(config_path, prefix_path)
        prefix_matches = sha(prefix_path) == identity['prefix_sha256']
        access = dict(prefix_path=str(prefix_path), config_path=str(config_path),
                      prefix_sha256=sha(prefix_path), T_numeric_values_parsed=0)
    ranges = {key: list(value) for key, value in task['split_ranges'].items()}
    fit, refit = FIT_ENDS[dataset]
    if (not prefix_matches or task['data_sha256'] != historical['data_sha256']
            or task['array_sha256'] != historical['array_sha256']
            or ranges != historical['split_ranges']
            or (task['L'], task['H']) != (512, 48)
            or (ranges['A_fit'][1], ranges['A_stop'][1]) != (fit, refit)
            or len(task['physical']) != ranges['V_select'][1]
            or len(task['physical']) != ranges['T_final'][0]):
        raise ValueError('Historical pre-T data, registered bounds or L/H changed')
    result = read(source / '.cache/k_tau_rule_20261023_v1/results' /
                  f'KTAU_main_{dataset}_K{KS[dataset]}_s92603.json')['result']
    for endpoint, stage in ((fit, 'search_summary'), (refit, 'refit_summary')):
        mean, std, _ = modules['data'].fit_statistics(task, endpoint)
        if modules['fit'].scaler_fingerprint(mean, std) != result[stage]['contract']['scaler_sha']:
            raise ValueError('Historical A_fit/A60 normalization changed')
    task['_campaign_access'] = access
    return task


def load_foundation(*, asset=DEFAULT_ASSET, device='cuda'):
    if device != 'cuda':
        raise ValueError('Registered real campaign device is CUDA')
    model_path = Path(asset).resolve() / MODEL_REL
    foundation = load_modules()['gpu'].FrozenBolt(model_path, 48, row_microbatch=128,
                                                   checkpoint_inputs=False, device=device)
    if foundation.initial_hash != F_SHA:
        raise ValueError('Foundation state changed from the fixed K-tau reference')
    foundation.assert_frozen_unchanged()
    return foundation


def guard_preserved():
    entries = read(REPORT / 'PRE_HASHES.json')['files']
    changed = [entry['path'] for entry in entries if sha(entry['path']) != entry['sha256']]
    if changed:
        raise ValueError('Existing protected file changed: ' + ', '.join(changed))
    return dict(scope='PRE_HASHES explicit allowlist', checked=len(entries), changed=[])


def record_opens():
    """Collect Python open events; bounded loaders enforce the numeric boundary."""
    records = set()

    def audit(event, arguments):
        if event == 'open' and isinstance(arguments[0], (str, bytes, os.PathLike)):
            path, mode, flags = arguments
            records.add((os.path.abspath(os.fsdecode(path)), str(mode), int(flags)))

    sys.addaudithook(audit)
    return records
