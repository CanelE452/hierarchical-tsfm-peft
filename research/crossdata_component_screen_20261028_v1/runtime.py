"""Pinned scientific modules and bounded historical data for this campaign."""
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import subprocess


NAME = 'crossdata_component_screen_20261028_v1'
REPO = Path(__file__).resolve().parents[2]
REPORT = REPO / 'reports' / NAME
CACHE = REPO / '.cache' / NAME
DEFAULT_SOURCE = Path('C:/Users/User/.codex/worktrees/adapter-diagnostics-v2/hierarchical-tsfm-peft')
DEFAULT_ASSET = Path('E:/CODING/proj/hierarchical-tsfm-peft')
DATASETS = dict(pems32=dict(K=8,L=512,H=12,C=32,cap=6),
                beijing=dict(K=6,L=512,H=48,C=11,cap=3))
KS = {name: item['K'] for name,item in DATASETS.items()}
FIT_ENDS = dict(pems32=(8928,10713),beijing=(10519,12622))
F_SHA = 'c2b4edf7db22d3f18e58ca79299a5c7e864e9b8f277e4e29e2f88d07dbb35874'
MODEL_REL = '.cache/huggingface/models--amazon--chronos-bolt-small/snapshots/772f3d25d38aec6d914c8949dab4462e2d46f5d8'
UPSTREAM_NAME = 'ppt_justification_20261028_v1'
UPSTREAM_REPORT = REPO / 'reports' / UPSTREAM_NAME
SHARED_CACHE = REPO / '.cache' / UPSTREAM_NAME
HOST_GPU_HELPER_REL = f'research/{NAME}/host_gpu_gate.py'
HOST_GPU_CONTRACT_REL = f'reports/{NAME}/HOST_GPU_CONTRACT.json'


def process_snapshot():
    if os.name != 'nt':
        raise RuntimeError('The registered execution host is Windows')
    script = "Get-CimInstance Win32_Process | Where-Object {$_.Name -match 'python'} | ForEach-Object { [pscustomobject]@{pid=$_.ProcessId;ppid=$_.ParentProcessId;created_utc=$_.CreationDate.ToUniversalTime().ToString('o');command_line=$_.CommandLine} } | ConvertTo-Json -Compress"
    output = subprocess.check_output(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', script],
                                     text=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    records = json.loads(output) if output.strip() else []
    return {int(row['pid']): row for row in (records if isinstance(records, list) else [records])}


def require_upstream_finished(snapshot=None):
    completion = read(UPSTREAM_REPORT / 'EXECUTION_COMPLETION.json')
    if completion.get('status') not in ('TRAINING_COMPLETE', 'TRAINING_INCOMPLETE'):
        raise ValueError('The upstream training campaign is not terminal without a scientific stop')
    processes = process_snapshot() if snapshot is None else snapshot
    e_report = REPO/'reports/ppt_e_fixed_20261028_v1'
    e_completion = read(e_report/'EXECUTION_COMPLETION.json')
    if e_completion.get('status') not in ('TRAINING_COMPLETE','TRAINING_INCOMPLETE','NOT_RUN_BUDGET'):
        raise ValueError('The preceding E campaign is not terminal')
    modules = [f'research.{UPSTREAM_NAME}.{name}' for name in ('controller', 'run_one', 'cost_profile')]
    modules += [f'research.ppt_e_fixed_20261028_v1.{name}' for name in ('controller','run_one','postprocess')]
    live = [row['pid'] for row in processes.values()
            if any(module in str(row.get('command_line', '')) for module in modules)]
    if live:
        raise RuntimeError('Upstream training or cost processes still live: ' + str(live))
    return dict(status='PASS', upstream_status=completion['status'],
                completion_sha256=sha(UPSTREAM_REPORT / 'EXECUTION_COMPLETION.json'),
                E_completion_sha256=sha(e_report/'EXECUTION_COMPLETION.json'),live_upstream_pids=[])


def require_root_gate():
    gate = read(REPORT / 'ROOT_LAUNCH_GATE.json')
    if (gate.get('status') != 'PASS' or gate.get('allow_gpu') is not True
            or gate.get('registration_seal_sha256') != sha(REPORT / 'REGISTRATION_SEAL.json')):
        raise ValueError('A root launch gate bound to the current sealed study is required')
    return gate


def require_active_lease(token):
    require_root_gate()
    lease = read(CACHE / 'GPU_LEASE.json')
    if (lease.get('status') != 'ACTIVE' or lease.get('token') != token
            or lease.get('registration_seal_sha256') != sha(REPORT / 'REGISTRATION_SEAL.json')):
        raise ValueError('Worker lease differs from the active registered controller')
    processes = process_snapshot()
    require_upstream_finished(processes)
    owner = processes.get(lease['pid'])
    if (owner is None or owner['created_utc'] != lease['created_utc']
            or f'research.{NAME}.controller' not in str(owner.get('command_line', ''))):
        raise RuntimeError('The lease owner is not the original live cross-dataset controller')
    ancestor = os.getpid()
    verified_ancestry = []
    for _ in range(5):
        if ancestor == lease['pid']:
            verified_ancestry.append(dict(owner))
            marker = f'-m research.{NAME}.controller'
            owner_line = str(owner.get('command_line',''))
            owner_args = owner_line.split(marker,1)[1].strip() if marker in owner_line else None
            launcher = owner
            for _ in range(3):
                parent = processes.get(int(launcher.get('ppid',0)),{})
                line = str(parent.get('command_line',''))
                if owner_args is None or marker not in line or line.split(marker,1)[1].strip()!=owner_args:
                    break
                verified_ancestry.append(dict(parent))
                launcher = parent
            return dict(lease, verified_ancestry=verified_ancestry)
        intermediate = processes.get(ancestor, {})
        line = str(intermediate.get('command_line', ''))
        if f'research.{NAME}.run_one' not in line or '--lease-token' not in line or token not in line:
            raise ValueError('The worker ancestry contains an unrelated Python process')
        verified_ancestry.append(dict(intermediate))
        ancestor = int(intermediate.get('ppid', 0))
    raise ValueError('Direct worker invocation is forbidden; the lease owner must be its ancestor')


def host_gpu_registration():
    return dict(mode='identity_only', helper_relative_path=HOST_GPU_HELPER_REL,
                helper_sha256=sha(REPO/HOST_GPU_HELPER_REL), contract_relative_path=HOST_GPU_CONTRACT_REL,
                contract_sha256=sha(REPO/HOST_GPU_CONTRACT_REL), desktop_concurrent_load=True,
                load_policy='Only prospectively sealed exact desktop identities may coexist. Unknown/new/Python/WSL GPU contexts are rejected. No utilization ceiling: this component screen is not a performance benchmark.')


def verify_host_gpu_binding(protocol=None):
    protocol = read(REPORT/'PREREGISTRATION.json') if protocol is None else protocol
    binding = protocol.get('host_gpu_gate')
    if binding != host_gpu_registration():
        raise ValueError('Shared host GPU helper/contract differs from this study registration')
    return binding


def require_idle_gpu(token):
    lease = require_active_lease(token)
    binding = verify_host_gpu_binding()
    helper = importlib.import_module(f'research.{NAME}.host_gpu_gate')
    if Path(helper.__file__).resolve() != (REPO/HOST_GPU_HELPER_REL).resolve():
        raise ValueError('Host GPU gate loaded from another checkout')
    allowed = helper.creation_times([row['pid'] for row in lease['verified_ancestry']])
    if require_active_lease(token)['verified_ancestry'] != lease['verified_ancestry']:
        raise RuntimeError('Verified worker/controller identity changed before host admission')
    receipt = helper.admit_identity(REPO/HOST_GPU_CONTRACT_REL, allowed_processes=allowed)
    verify_host_gpu_binding()
    if receipt.get('status')!='PASS' or receipt.get('mode')!='identity_only' or receipt.get('unexpected_gpu_pids')!=[]:
        raise ValueError('Shared host GPU identity gate did not pass')
    return dict(receipt, host_gpu_binding=binding,
                registration_seal_sha256=sha(REPORT/'REGISTRATION_SEAL.json'))


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


def import_source(path, label):
    spec = importlib.util.spec_from_file_location(NAME+'_'+label,path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def describe_task(task):
    name=task['dataset'];fit,refit=FIT_ENDS[name]
    modules=load_modules();scalers={}
    for phase,endpoint in (('A_fit',fit),('A60',refit)):
        mean,std,_=modules['data'].fit_statistics(task,endpoint)
        scalers[phase]=modules['fit'].scaler_fingerprint(mean,std)
    return dict(K=KS[name],L=task['L'],H=task['H'],C=task['physical'].shape[1],
                fit_end=fit,refit_end=refit,T_start=task['split_ranges']['T_final'][0],
                split_ranges={k:list(v) for k,v in task['split_ranges'].items()},
                columns=list(task['columns']),data_sha256=task['data_sha256'],
                array_sha256=task['array_sha256'],
                materialized_prefix_sha256=task['materialized_prefix_sha256'],
                normalization_sha256=scalers)


def validate_task(task, access, expected=None):
    dataset=task['dataset'];axes=DATASETS[dataset];fit,refit=FIT_ENDS[dataset]
    bounds=task['split_ranges']
    if ((task['L'],task['H'],task['physical'].shape[1])!=(axes['L'],axes['H'],axes['C'])
            or bounds['A_fit']!=[0,fit] or bounds['A_stop']!=[fit,refit]
            or len(task['physical'])!=bounds['V_select'][1]
            or len(task['physical'])!=bounds['T_final'][0]
            or task['observed'].shape!=task['physical'].shape
            or access.get('new_T_numeric_values_parsed')!=0
            or not access.get('members')
            or any(row.get('tail_numeric_values_parsed')!=0 for row in access['members'])):
        raise ValueError('Dataset axes, prefix, A_stop/refit or protected T proof changed')
    description=describe_task(task)
    if expected is not None and description!=expected:
        raise ValueError('Dataset/scaler cohort identity differs from sealed DATA_REGISTRY')
    return description


def load_task(dataset, *, source=DEFAULT_SOURCE, asset=DEFAULT_ASSET, require_registry=True):
    if dataset not in DATASETS:
        raise ValueError('Dataset is outside the registered screen')
    modules=load_modules();broad=REPO/'research/informs_broad_validation_20261015_v1'
    primitive=import_source(broad/'priority_b2_validation_20261016_v1/safe_jena_prefix.py','primitive')
    if dataset=='pems32':
        subset=import_source(broad/'priority_pems_subset_20261016_v1/subset_minimum_runner.py','subset')
        plan=read(broad/'priority_pems_subset_20261016_v1/PLAN.json')
        subset.require_sealed(plan)
        for name,digest in plan['dependencies'].items():
            if sha(broad/name)!=digest:raise ValueError('Pinned PeMS prefix dependency changed: '+name)
        priority=import_source(broad/subset.DEPENDENCY_PATHS['priority'],'priority')
        pems=import_source(broad/subset.DEPENDENCY_PATHS['pems'],'pems')
        parent,access=pems.load_prefix(Path(asset)/'.cache/informs_broad_validation_20261015_v1',
                                      plan['parent_task_contract'],priority,primitive)
        import numpy as np
        task=subset.subset_task(parent,plan,dict(np=np,data=modules['data']))
        task['dataset']=task['name']='pems32'
        access.update(actual_C=32,K=8,subset_plan_sha256=sha(broad/'priority_pems_subset_20261016_v1/PLAN.json'),
                      derived_subset_data_sha256=task['data_sha256'])
    else:
        import numpy as np
        loader=import_source(broad/'priority_nested_cost_20261016_v1/prefix_nested_cost_runner.py','beijing')
        old=read(broad/'DATA_CONTRACT.json')['assets']['beijing']
        task,access=loader.safe_task(Path(asset),dict(task_contract=old),primitive,np)
    expected=read(REPORT/'DATA_REGISTRY.json')['datasets'][dataset] if require_registry else None
    validate_task(task,access,expected)
    task['_campaign_access']=access
    return task


def load_foundation(dataset, *, asset=DEFAULT_ASSET, device='cuda'):
    if device != 'cuda':
        raise ValueError('Registered real campaign device is CUDA')
    model_path = Path(asset).resolve() / MODEL_REL
    foundation = load_modules()['gpu'].FrozenBolt(model_path, DATASETS[dataset]['H'], row_microbatch=128,
                                                   checkpoint_inputs=False, device=device)
    if foundation.initial_hash != F_SHA:
        raise ValueError('Foundation state changed from the fixed K-tau reference')
    foundation.assert_frozen_unchanged()
    return foundation


class NoFoundation:
    initial_hash = 'NO_FOUNDATION'
    streams = row_calls = physical_streams = physical_calls = 0

    def __init__(self, device='cuda'):
        self.device = device

    def __call__(self, x):
        raise AssertionError('NO_F_LINEAR must never invoke a foundation')

    def assert_frozen_unchanged(self):
        if any(getattr(self, name) != 0 for name in ('streams', 'row_calls', 'physical_streams', 'physical_calls')):
            raise AssertionError('NO_FOUNDATION counters must remain zero')


def guard_preserved():
    entries = read(REPORT / 'PRESERVATION_BASELINE.json')['files']
    changed = [entry['path'] for entry in entries if not Path(entry['path']).is_file() or sha(entry['path']) != entry['sha256']]
    if changed:
        raise ValueError('Existing protected file changed: ' + ', '.join(changed))
    return dict(scope='PRESERVATION_BASELINE explicit allowlist', checked=len(entries), changed=[])


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


def assert_protected_T(result):
    access=result['task_access']
    if (access.get('new_T_numeric_values_parsed')!=0 or not access.get('members')
            or any(row.get('tail_numeric_values_parsed')!=0 for row in access['members'])):
        raise ValueError('Bounded prefix did not prove zero protected T numeric parsing')


def allowed_write_path(path):
    path=Path(path).resolve()
    return (any(path.is_relative_to(root.resolve()) for root in (REPORT,CACHE))
            or path in {(SHARED_CACHE/name).resolve() for name in ('controller.lock','worker.lock')})


def is_native_null_device(path):
    if not isinstance(path,(str,bytes,os.PathLike)):
        return False
    raw=os.fsdecode(path)
    if os.name=='nt':
        return raw.replace('/','\\').casefold() in {os.devnull.casefold(),r'\\.\nul'}
    return raw==os.devnull


def install_write_guard():
    protected = {os.path.normcase(os.path.abspath(row['path']))
                 for row in read(REPORT / 'PRESERVATION_BASELINE.json')['files']}

    def reject(paths):
        for path in paths:
            if isinstance(path, (str, bytes, os.PathLike)):
                normalized = os.path.normcase(os.path.abspath(os.fsdecode(path)))
                if normalized in protected:
                    raise PermissionError('Cannot change protected existing file: ' + normalized)
                if not allowed_write_path(normalized):
                    raise PermissionError('Write outside this campaign and exact shared lock files: '+normalized)

    def audit(event, args):
        if event == 'open':
            path, mode, flags = args
            if ((isinstance(mode, str) and any(c in mode for c in 'wax+'))
                    or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)):
                if not is_native_null_device(path):
                    reject((path,))
        elif event in ('os.remove', 'os.unlink', 'os.truncate', 'os.chmod', 'os.utime'):
            reject(args[:1])
        elif event in ('os.rename', 'os.replace'):
            reject(args[:2])

    sys.addaudithook(audit)
