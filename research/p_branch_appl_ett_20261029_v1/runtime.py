"""Pinned scientific modules and bounded historical data for this campaign."""
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


NAME = 'p_branch_appl_ett_20261029_v1'
REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / 'reports' / NAME
CACHE = REPO / '.cache' / NAME
SHARED_CACHE = REPO / '.cache/ppt_justification_20261028_v1'
DEFAULT_SOURCE = Path('C:/Users/User/.codex/worktrees/adapter-diagnostics-v2/hierarchical-tsfm-peft')
DEFAULT_ASSET = Path('E:/CODING/proj/hierarchical-tsfm-peft')
DATASETS = ('appliances', 'ETTh1', 'ETTh2_safe17400')
KS = dict(appliances=10, ETTh1=4, ETTh2_safe17400=4)
V_SHAPES = dict(appliances=(41, 48, 26), ETTh1=(36, 48, 7), ETTh2_safe17400=(36, 48, 7))
T_STARTS = dict(appliances=15788, ETTh1=13936, ETTh2_safe17400=13920)
V_RANGES = dict(appliances=(13814, 15788), ETTh1=(12194, 13936), ETTh2_safe17400=(12180, 13920))
EXPECTED_PARAMETERS = dict(appliances=dict(OURS=37114, P_TIME_ONLY=35392, P_CHANNEL_ONLY=3802),
    ETTh1=dict(OURS=34023, P_TIME_ONLY=33536, P_CHANNEL_ONLY=711),
    ETTh2_safe17400=dict(OURS=34023, P_TIME_ONLY=33536, P_CHANNEL_ONLY=711))
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
        header_proof = bounded_prefix_headers(prefix_path, T_STARTS[dataset], V_SHAPES[dataset][2])
        _, task = load_prefix(config_path, prefix_path)
        prefix_matches = sha(prefix_path) == identity['prefix_sha256']
        access = dict(dataset=dataset, prefix_path=str(prefix_path), config_path=str(config_path),
                      prefix_sha256=sha(prefix_path), config_sha256=sha(config_path),
                      prefix_rows=len(task['physical']), T_start=T_STARTS[dataset],
                      prefix_header_proof=header_proof, T_numeric_values_parsed=0)
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


_PRESERVATION_HASH_CACHE = {}


def critical_preservation_paths():
    paths = set()
    protocol_path = REPORT / 'PREREGISTRATION.json'
    if protocol_path.is_file():
        protocol = read(protocol_path)
        paths.update((REPO / path).resolve() for path in protocol.get('registered_sources', {}))
    references_path = REPORT / 'REFERENCE_RUNS.json'
    if references_path.is_file():
        paths.update(Path(row['prediction_path']).resolve() for row in read(references_path).get('rows', [])
                     if row.get('prediction_path'))
    return paths


def guard_preserved(*, full=False):
    import stat
    entries = read(REPORT / 'PRESERVATION_BASELINE.json')['files']
    critical = critical_preservation_paths()
    broad = (REPO / 'research/informs_broad_validation_20261015_v1').resolve()
    dependency = (REPO / 'research/adapter_diagnostics_20261017_v2/existing_jobs.py').resolve()
    changed, fresh_count, cached_count = [], 0, 0
    for entry in entries:
        path = Path(entry['path']).resolve()
        try:
            details = path.stat()
        except FileNotFoundError:
            changed.append(entry['path'])
            continue
        if not stat.S_ISREG(details.st_mode):
            changed.append(entry['path'])
            continue
        signature = (details.st_dev, details.st_ino, details.st_size, details.st_mtime_ns, details.st_ctime_ns)
        previous = _PRESERVATION_HASH_CACHE.get(path)
        mandatory = full or path in critical or path.is_relative_to(broad) or path == dependency
        if not mandatory and previous is not None and previous[:2] == (signature, entry['sha256']):
            actual = previous[2]
            cached_count += 1
        else:
            actual = sha(path)
            _PRESERVATION_HASH_CACHE[path] = (signature, entry['sha256'], actual)
            fresh_count += 1
        if actual != entry['sha256']:
            changed.append(entry['path'])
    if changed:
        raise ValueError('Existing protected file changed: ' + ', '.join(changed))
    return dict(scope='PRESERVATION_BASELINE explicit allowlist', checked=len(entries), changed=[],
                scan_mode='FULL_SHA256' if cached_count == 0 else 'MIXED_FRESH_HASH_AND_METADATA_CACHE',
                requested_full=bool(full), full_hash_checked=fresh_count, cached_checked=cached_count,
                cache_scope='Per-process memory only; metadata size/mtime/ctime/file identity checked every call; critical inputs always freshly hashed')


def record_opens():
    """Collect Python open events; bounded loaders enforce the numeric boundary."""
    records = set()

    def audit(event, arguments):
        if event == 'open' and isinstance(arguments[0], (str, bytes, os.PathLike)):
            path, mode, flags = arguments
            records.add((os.path.abspath(os.fsdecode(path)), str(mode), int(flags)))

    sys.addaudithook(audit)
    return records


def atomic(path, value):
    path = Path(path).resolve()
    if not any(path.is_relative_to(root.resolve()) for root in (REPORT, CACHE)):
        raise ValueError('Atomic output outside new campaign')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf8')
    for attempt in range(6):
        try:
            temporary.replace(path)
            return
        except PermissionError as exc:
            if getattr(exc, 'winerror', None) not in (5, 32, 33) or attempt == 5:
                raise
            time.sleep(.1 * 2 ** attempt)


class CampaignLock:
    def __init__(self, path):
        self.path, self.stream = Path(path), None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('a+b')
        if self.path.stat().st_size == 0:
            self.stream.write(b'0')
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            self.stream = None
            raise RuntimeError('Another live process holds ' + self.path.name) from exc
        return self

    def __exit__(self, *args):
        if self.stream is not None:
            self.stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()


def pid_alive(pid):
    if os.name == 'nt':
        import ctypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        kernel.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
        kernel.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_ulong)
        kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
        handle = kernel.OpenProcess(0x100000, False, int(pid))
        if not handle:
            return ctypes.get_last_error() == 5
        try:
            return kernel.WaitForSingleObject(handle, 0) != 0
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def bounded_prefix_headers(path, rows, channels):
    import zipfile
    import numpy as np
    expected = dict(physical=((rows, channels), np.dtype('float32')),
                    observed=((rows, channels), np.dtype('bool')))
    proof = {}
    with zipfile.ZipFile(path) as archive:
        for key, (shape, dtype) in expected.items():
            with archive.open(key + '.npy') as stream:
                version = np.lib.format.read_magic(stream)
                if version not in ((1, 0), (2, 0)):
                    raise ValueError('Unsupported bounded prefix numeric header')
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                actual_shape, fortran, actual_dtype = reader(stream)
            if actual_shape != shape or fortran or actual_dtype != dtype:
                raise ValueError('Bounded prefix numeric header exceeds registered pre-T range')
            proof[key] = dict(shape=list(actual_shape), dtype=str(actual_dtype), fortran_order=False)
    return proof


def assert_protected_T(result):
    access = result['task_access']
    dataset = result.get('job', {}).get('dataset', access.get('dataset'))
    if dataset == 'jena' or (dataset is None and 'members' in access):
        members = access.get('members', [])
        if (access.get('new_T_numeric_values_parsed') != 0
                or {row['member'] for row in members} != {'x.npy', 'finite.npy', 'times.npy', 'mean.npy', 'std.npy', 'columns.npy'}
                or any(row.get('tail_numeric_values_parsed') != 0 for row in members)):
            raise ValueError('Jena bounded loader did not prove zero protected T parsing')
        return
    if dataset not in DATASETS:
        raise ValueError('Protected T proof has no registered dataset')
    rows, channels = T_STARTS[dataset], V_SHAPES[dataset][2]
    expected = dict(physical=dict(shape=[rows, channels], dtype='float32', fortran_order=False),
                    observed=dict(shape=[rows, channels], dtype='bool', fortran_order=False))
    if (access.get('T_numeric_values_parsed') != 0 or access.get('prefix_rows') != rows
            or access.get('T_start') != rows or access.get('prefix_header_proof') != expected
            or sha(access['prefix_path']) != access.get('prefix_sha256')
            or sha(access['config_path']) != access.get('config_sha256')):
        raise ValueError('Bounded prefix loader did not prove zero protected T parsing')


def install_write_guard():
    protected = {os.path.normcase(os.path.abspath(row['path']))
                 for row in read(REPORT / 'PRESERVATION_BASELINE.json')['files']}

    def reject(paths):
        for path in paths:
            if isinstance(path, (str, bytes, os.PathLike)):
                normalized = os.path.normcase(os.path.abspath(os.fsdecode(path)))
                if normalized in protected:
                    raise PermissionError('Cannot change protected existing file: ' + normalized)

    def audit(event, args):
        if event == 'open':
            path, mode, flags = args
            if ((isinstance(mode, str) and any(c in mode for c in 'wax+'))
                    or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
                reject((path,))
        elif event in ('os.remove', 'os.unlink', 'os.truncate', 'os.chmod', 'os.utime'):
            reject(args[:1])
        elif event in ('os.rename', 'os.replace'):
            reject(args[:2])

    sys.addaudithook(audit)
