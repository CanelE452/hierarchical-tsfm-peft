"""New human-approved campaign; immutable clocks, exclusive metered jobs."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AUTHORIZATION = HERE / 'AUTHORIZATION_v1.json'
RUN_MANIFEST = HERE / 'CAMPAIGN_MANIFEST_v1.json'
APPROVED_PLAN = HERE / 'PHASE1_PLAN_KO.md'
CORE_RUNTIME = Path(__file__).resolve()
RESOURCES = HERE / 'resources_v1'
ACTIVE = RESOURCES / 'ACTIVE.json'
EXPECTED_REPLY_ID = 'call_72d48d8ce73647df9d34c9a54d7f817e:0'
MAX_LIMITS = dict(gpu_process_wall_seconds=5400, cpu_process_wall_seconds=7200,
                  campaign_wall_seconds=14400, additional_storage_bytes=6*1024**3,
                  process_rss_bytes=8*1024**3, reserved_vram_bytes=9*1024**3)
TERMINAL = ('COMPLETE', 'FAILED_PRESERVED', 'BUDGET_HELD_PRESERVED')
LAST_CONSTRUCTOR_TERMINAL = None

class BudgetHeld(RuntimeError):
    pass

class AuthorizationRequired(RuntimeError):
    pass

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')

def storage_bytes(roots):
    total = 0
    for root in roots:
        if Path(root).exists():
            for directory, _, names in os.walk(root, followlinks=False):
                for name in names:
                    path = Path(directory)/name
                    if path.is_file() and not path.is_symlink():
                        total += path.stat().st_size
    return total

def authorized_records():
    auth, run = read(AUTHORIZATION), read(RUN_MANIFEST)
    limits = auth.get('limits', {})
    if (auth.get('status') != 'APPROVED' or auth.get('origin') != 'human_user_response'
        or auth.get('response_identifier') != EXPECTED_REPLY_ID
        or auth.get('user_response') != '계획과 자원 상한 승인'
        or auth.get('approved_plan_sha256') != sha(APPROVED_PLAN)
        or auth.get('runtime_source_sha256') != sha(CORE_RUNTIME)
        or set(limits) != set(MAX_LIMITS)
        or any(not isinstance(limits[k], (int,float)) or not 0 < limits[k] <= cap
               for k,cap in MAX_LIMITS.items())
        or run.get('authorization_sha256') != sha(AUTHORIZATION)
        or run.get('runtime_source_sha256') != sha(CORE_RUNTIME)
        or run.get('started_unix',0) < auth.get('approved_unix',float('inf'))
        or run.get('expires_unix') != run.get('started_unix',0)+limits['campaign_wall_seconds']
        or not run.get('storage_roots') or not run.get('protected_records')):
        raise AuthorizationRequired('NEW_APPROVAL_FIXED_MANIFEST_OR_LIMIT_INVALID')
    return auth, run

class ContinuationLimit:
    def __init__(self, tag, gpu=False):
        global LAST_CONSTRUCTOR_TERMINAL
        LAST_CONSTRUCTOR_TERMINAL = None
        self.authorization, self.run_manifest = authorized_records()
        if not tag or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in tag):
            raise ValueError('SAFE_RESOURCE_TAG_REQUIRED')
        import psutil
        self.process = psutil.Process()
        self.started_unix = self.process.create_time()
        self.start = time.perf_counter()
        self.initial_elapsed = max(0., time.time()-self.started_unix)
        self.tag, self.gpu = tag, bool(gpu)
        self.limits = self.authorization['limits']
        self.authorization_record = {'path':str(AUTHORIZATION),'sha256':sha(AUTHORIZATION)}
        self.run_manifest_record = {'path':str(RUN_MANIFEST),'sha256':sha(RUN_MANIFEST)}
        self.rss = self.vram = self.reserved = 0
        self.storage = self.run_manifest['baseline_storage_bytes']
        self.closed = False
        self.last_check = None
        self.rt = SimpleNamespace(BudgetHeld=BudgetHeld,ROOT=ROOT,HERE=HERE)
        self.shared = SimpleNamespace(check=self.base_check,checkfast=self.base_check)
        RESOURCES.mkdir(exist_ok=True)
        self.previous = []
        for path in RESOURCES.glob('*.json'):
            if path == ACTIVE:
                raise RuntimeError('ACTIVE_JOB_LEASE_PRESERVE_AND_REVIEW')
            record = read(path)
            if record.get('status') not in TERMINAL or record.get('authorization',{}).get('sha256') != self.authorization_record['sha256']:
                raise RuntimeError('NONTERMINAL_OR_OTHER_AUTH_RECEIPT')
            self.previous.append(record)
        self.path = RESOURCES/f'{os.getpid()}_{time.time_ns()}_{tag}.json'
        write_new(ACTIVE,dict(pid=os.getpid(),process_created_unix=self.started_unix,tag=tag,gpu=self.gpu))
        self.lease_hash = sha(ACTIVE)
        try:
            self.check()
        except Exception as error:
            LAST_CONSTRUCTOR_TERMINAL = self.close('BUDGET_HELD_PRESERVED' if isinstance(error,BudgetHeld) else 'FAILED_PRESERVED',dict(type=type(error).__name__,message=str(error)))
            raise

    def _elapsed(self):
        return self.initial_elapsed + time.perf_counter()-self.start

    def base_check(self):
        self.rss = max(self.rss,self.process.memory_info().rss)
        torch = sys.modules.get('torch')
        initialized = torch is not None and torch.cuda.is_initialized()
        if initialized and not self.gpu:
            raise RuntimeError('CPU_JOB_CUDA_FORBIDDEN')
        if initialized:
            self.vram = max(self.vram,torch.cuda.max_memory_allocated())
            self.reserved = max(self.reserved,torch.cuda.max_memory_reserved())
        elapsed = self._elapsed()
        charged = sum(r['wall_seconds'] for r in self.previous if bool(r['gpu']) == self.gpu)+elapsed
        if charged > self.limits['gpu_process_wall_seconds' if self.gpu else 'cpu_process_wall_seconds']:
            raise BudgetHeld('PROCESS_WALL_CAP')
        if time.time() > self.run_manifest['expires_unix']:
            raise BudgetHeld('FIXED_CAMPAIGN_WALL_CAP')
        if self.rss > self.limits['process_rss_bytes'] or self.reserved > self.limits['reserved_vram_bytes']:
            raise BudgetHeld('RSS_OR_RESERVED_VRAM_CAP')
        return dict(tag=self.tag,gpu=self.gpu,elapsed_seconds=elapsed,budget_counted_seconds=charged,
                    sampled_peak_rss=self.rss,peak_allocated_vram=self.vram,peak_reserved_vram=self.reserved)

    def check(self):
        value = self.base_check()
        if sha(AUTHORIZATION) != self.authorization_record['sha256'] or sha(RUN_MANIFEST) != self.run_manifest_record['sha256'] or sha(CORE_RUNTIME) != self.authorization['runtime_source_sha256']:
            raise RuntimeError('AUTH_MANIFEST_RUNTIME_CHANGED')
        for record in self.run_manifest['protected_records']:
            if sha(record['path']) != record['sha256']:
                raise RuntimeError('PROTECTED_ORIGINAL_CHANGED:'+record['path'])
        self.storage = storage_bytes(self.run_manifest['storage_roots'])
        growth = max(0,self.storage-self.run_manifest['baseline_storage_bytes'])
        if growth > self.limits['additional_storage_bytes'] or shutil.disk_usage(ROOT).free < 2*1024**3:
            raise BudgetHeld('STORAGE_CAP_OR_DISK_FULL')
        value.update(storage_bytes=self.storage,additional_storage_bytes=growth,
                     authorization=self.authorization_record,run_manifest=self.run_manifest_record)
        self.last_check = value
        self.base_check()
        return value

    def checkfull(self):
        return self.check()

    def declare_gpu_forward_scope(self, scope):
        if not self.gpu:
            raise RuntimeError('GPU_SCOPE_ON_CPU_JOB')
        return {'gpu_forward_scope':scope,'backbone_training_allowed':False}

    def close(self,status='COMPLETE',error=None):
        if self.closed:
            return self.terminal
        requested = status
        if status == 'COMPLETE' and error:
            status = 'FAILED_PRESERVED'
        check_failure = None
        try:
            torch = sys.modules.get('torch')
            if self.gpu and torch is not None and torch.cuda.is_initialized():
                torch.cuda.synchronize()
            self.check()
        except Exception as failure:
            status = 'BUDGET_HELD_PRESERVED' if isinstance(failure,BudgetHeld) else 'FAILED_PRESERVED'
            check_failure = dict(type=type(failure).__name__,message=str(failure))
        if status not in TERMINAL:
            raise ValueError('TERMINAL_STATUS_REQUIRED')
        receipt = dict(tag=self.tag,gpu=self.gpu,pid=os.getpid(),status=status,started_unix=self.started_unix,
                       finished_unix=time.time(),wall_seconds=self._elapsed(),sampled_peak_rss=self.rss,
                       peak_allocated_vram=self.vram,peak_reserved_vram=self.reserved,storage_bytes=self.storage,
                       authorization=self.authorization_record,run_manifest=self.run_manifest_record,
                       requested_status=requested,caller_error=error,check_failure=check_failure,
                       prior_ledgers_preserved=True,wall_scope='Whole process birth through terminal, not kernel time')
        write_new(self.path,receipt)
        if sha(ACTIVE) != self.lease_hash:
            raise RuntimeError('OWN_ACTIVE_LEASE_CHANGED')
        ACTIVE.unlink()
        self.terminal = dict(path=str(self.path.relative_to(ROOT)).replace('\\','/'),sha256=sha(self.path),
                             terminal_status=status,terminal_error=check_failure or error,continuation_limit=self.last_check)
        self.closed = True
        print(json.dumps({'resource_terminal':self.terminal,'wall_seconds':receipt['wall_seconds']}),flush=True)
        return self.terminal
