"""Same approved continuation; faster equivalent storage scan. No cap/deadline reset."""
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
AUTHORIZATION = HERE / 'continuation_authorization_v1.json'
RUN_MANIFEST = HERE / 'continuation_run_manifest_v1.json'
CORE_RUNTIME = HERE / 'continuation_runtime_v1.py'
RESOURCES = HERE / 'continuation_resources_v1'
ACTIVE = RESOURCES / 'ACTIVE.json'
MAX_LIMITS = {'gpu_process_wall_seconds': 1800, 'cpu_process_wall_seconds': 1800,
              'campaign_wall_seconds': 3600, 'additional_storage_bytes': 2 * 1024**3,
              'process_rss_bytes': 8 * 1024**3, 'allocated_vram_bytes': 8 * 1024**3}
TERMINAL = ('COMPLETE', 'FAILED_PRESERVED', 'BUDGET_HELD_PRESERVED')
LAST_CONSTRUCTOR_TERMINAL = None


class BudgetHeld(RuntimeError):
    pass


class AuthorizationRequired(RuntimeError):
    pass


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from fast_storage_scan_v1 import storage_bytes


def authorized_records():
    if not AUTHORIZATION.is_file() or not RUN_MANIFEST.is_file():
        raise AuthorizationRequired('EXPLICIT_HUMAN_APPROVAL_AND_FIXED_RUN_MANIFEST_REQUIRED')
    auth, run = read(AUTHORIZATION), read(RUN_MANIFEST)
    if (auth.get('schema') != 'smallk_continuation_authorization_v1'
            or auth.get('status') != 'APPROVED' or auth.get('origin') != 'human_user_response'
            or not auth.get('response_identifier') or not auth.get('user_response')):
        raise AuthorizationRequired('EXPLICIT_HUMAN_APPROVAL_RECORD_INVALID')
    limits = auth.get('limits', {})
    if set(limits) != set(MAX_LIMITS) or any(not isinstance(limits[k], (int, float))
            or not 0 < limits[k] <= MAX_LIMITS[k] for k in MAX_LIMITS):
        raise AuthorizationRequired('LIMITS_EXCEED_APPROVED_PROPOSAL_OR_ARE_MISSING')
    if (run.get('schema') != 'smallk_continuation_run_manifest_v1'
            or run.get('authorization_sha256') != sha(AUTHORIZATION)
            or run.get('runtime_source_sha256') != sha(CORE_RUNTIME)
            or run.get('expires_unix') != run.get('started_unix', 0) + limits['campaign_wall_seconds']
            or run.get('started_unix', 0) < auth.get('approved_unix', float('inf'))
            or not isinstance(run.get('baseline_storage_bytes'), int)):
        raise AuthorizationRequired('FIXED_AUTHORIZATION_BOUND_RUN_MANIFEST_INVALID')
    return auth, run


class _FastShared:
    def __init__(self, limit):
        self.limit = limit

    def check(self):
        return self.limit.base_check()

    def checkfast(self):
        return self.check()


class ContinuationLimit:
    def __init__(self, tag, gpu=True):
        global LAST_CONSTRUCTOR_TERMINAL
        LAST_CONSTRUCTOR_TERMINAL = None
        try:
            self.authorization, self.run_manifest = authorized_records()
        except Exception as error:
            LAST_CONSTRUCTOR_TERMINAL = {
                'terminal_status': 'FAILED_PRESERVED',
                'terminal_error': {'type': type(error).__name__, 'message': str(error)},
                'scope': 'pre-authorization failure; no resource receipt or scientific assets opened'}
            raise
        if not tag or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in tag):
            raise ValueError('SAFE_RESOURCE_TAG_REQUIRED')
        import psutil
        self.tag, self.gpu = tag, bool(gpu)
        self.limits = self.authorization['limits']
        self.authorization_record = {'path': str(AUTHORIZATION), 'sha256': sha(AUTHORIZATION)}
        self.run_manifest_record = {'path': str(RUN_MANIFEST), 'sha256': sha(RUN_MANIFEST)}
        self.guard_adapter_record = {'runtime_source': {'path': str(Path(__file__).resolve()), 'sha256': sha(__file__)}, 'approved_core_runtime': {'path': str(CORE_RUNTIME), 'sha256': sha(CORE_RUNTIME)}, 'storage_scanner': {'path': str(HERE / 'fast_storage_scan_v1.py'), 'sha256': sha(HERE / 'fast_storage_scan_v1.py')}, 'scope': 'same counters/caps/expiry/lease/full cadence; directory-canonical equivalent storage enumeration'}
        self.process = psutil.Process()
        self.started_unix = self.process.create_time()
        self.start = time.perf_counter()
        self.initial_elapsed = max(0., time.time() - self.started_unix)
        self.rss = self.vram = self.reserved = 0
        self.closed = False
        self.storage = self.run_manifest['baseline_storage_bytes']
        self.last_check = None
        self.path = RESOURCES / f'{os.getpid()}_{time.time_ns()}_{tag}.json'
        self.rt = SimpleNamespace(BudgetHeld=BudgetHeld, ROOT=ROOT, HERE=HERE,
                                  GPU_SECONDS=self.limits['gpu_process_wall_seconds'],
                                  CPU_SECONDS=self.limits['cpu_process_wall_seconds'])
        self.shared = _FastShared(self)
        RESOURCES.mkdir(exist_ok=True)
        self.previous = []
        for path in RESOURCES.glob('*.json'):
            if path == ACTIVE:
                raise RuntimeError('EXISTING_ACTIVE_LEASE_REQUIRES_AUTHORITATIVE_TERMINAL_REVIEW')
            receipt = read(path)
            if receipt.get('status') not in TERMINAL:
                raise RuntimeError('NONTERMINAL_CONTINUATION_RECEIPT:' + str(path))
            self.previous.append(receipt)
        self.lease = {'pid': os.getpid(), 'process_created_unix': self.started_unix,
                      'tag': tag, 'manifest_sha256': self.run_manifest_record['sha256']}
        write_new(ACTIVE, self.lease)
        self.lease_hash = sha(ACTIVE)
        try:
            self.check()
        except Exception as error:
            status = 'BUDGET_HELD_PRESERVED' if isinstance(error, BudgetHeld) else 'FAILED_PRESERVED'
            LAST_CONSTRUCTOR_TERMINAL = self.close(status, {'type': type(error).__name__, 'message': str(error)})
            raise

    def _elapsed(self):
        return self.initial_elapsed + time.perf_counter() - self.start

    def base_check(self):
        self.rss = max(self.rss, self.process.memory_info().rss)
        torch = sys.modules.get('torch')
        if self.gpu and torch is not None and torch.cuda.is_initialized():
            self.vram = max(self.vram, torch.cuda.max_memory_allocated())
            self.reserved = max(self.reserved, torch.cuda.max_memory_reserved())
        elapsed = self._elapsed()
        charged = sum(r['wall_seconds'] for r in self.previous if r['gpu'] == self.gpu) + elapsed
        cap = self.limits['gpu_process_wall_seconds' if self.gpu else 'cpu_process_wall_seconds']
        if charged > cap:
            raise BudgetHeld('CONTINUATION_PROCESS_WALL_BUDGET')
        if time.time() > self.run_manifest['expires_unix']:
            raise BudgetHeld('CONTINUATION_CAMPAIGN_WALL_BUDGET')
        if self.rss > self.limits['process_rss_bytes'] or self.vram > self.limits['allocated_vram_bytes']:
            raise BudgetHeld('CONTINUATION_RSS_OR_ALLOCATED_VRAM_BUDGET')
        return {'elapsed_seconds': elapsed, 'budget_counted_seconds': charged,
                'storage_bytes': self.storage, 'tag': self.tag, 'gpu': self.gpu}

    def check(self):
        value = self.base_check()
        if (sha(AUTHORIZATION) != self.authorization_record['sha256']
                or sha(RUN_MANIFEST) != self.run_manifest_record['sha256']):
            raise RuntimeError('AUTHORIZATION_OR_FIXED_MANIFEST_CHANGED')
        for record in self.guard_adapter_record.values():
            if isinstance(record, dict) and sha(record['path']) != record['sha256']:
                raise RuntimeError('GUARD_ADAPTER_SOURCE_CHANGED')
        for record in self.run_manifest.get('protected_prior_receipts', []):
            if sha(record['path']) != record['sha256']:
                raise RuntimeError('PROTECTED_PRIOR_RECEIPT_CHANGED')
        self.storage = storage_bytes(self.run_manifest['storage_roots'])
        growth = max(0, self.storage - self.run_manifest['baseline_storage_bytes'])
        if growth > self.limits['additional_storage_bytes'] or shutil.disk_usage(ROOT).free < 2 * 1024**3:
            raise BudgetHeld('CONTINUATION_STORAGE_GROWTH_OR_FREE_DISK_BUDGET')
        value.update(storage_bytes=self.storage, additional_storage_bytes=growth,
                     authorization=self.authorization_record, run_manifest=self.run_manifest_record, guard_adapter=self.guard_adapter_record)
        self.last_check = value
        self.base_check()
        return dict(value)

    def checkfull(self):
        return self.check()

    def close(self, status='COMPLETE', error=None):
        if self.closed:
            return dict(self.terminal)
        requested = status
        if status == 'COMPLETE' and error is not None:
            status = 'FAILED_PRESERVED'
        check_failure = None
        torch = sys.modules.get('torch')
        try:
            if self.gpu and torch is not None and torch.cuda.is_initialized():
                torch.cuda.synchronize()
            self.check()
        except Exception as failure:
            status = 'BUDGET_HELD_PRESERVED' if isinstance(failure, BudgetHeld) else 'FAILED_PRESERVED'
            check_failure = {'type': type(failure).__name__, 'message': str(failure)}
        if status not in TERMINAL:
            raise ValueError('TERMINAL_STATUS_REQUIRED')
        finished = time.time()
        elapsed = self._elapsed()
        receipt = {'tag': self.tag, 'gpu': self.gpu, 'pid': os.getpid(), 'status': status,
                   'started_unix': self.started_unix, 'finished_unix': finished, 'wall_seconds': elapsed,
                   'sampled_peak_rss': self.rss, 'peak_allocated_vram': self.vram,
                   'peak_reserved_vram': self.reserved, 'storage_bytes': self.storage,
                   'authorization': self.authorization_record, 'run_manifest': self.run_manifest_record, 'guard_adapter': self.guard_adapter_record,
                   'requested_status': requested, 'caller_error': error, 'check_failure': check_failure,
                   'previous_receipts': len(self.previous),
                   'wall_scope': 'process creation to terminal; CPU/model load/guards included, not GPU kernel time',
                   'original_ledger_preserved_and_not_reset': True}
        write_new(self.path, receipt)
        if sha(ACTIVE) != self.lease_hash:
            raise RuntimeError('OWN_ACTIVE_LEASE_CHANGED')
        ACTIVE.unlink()
        record = {'path': str(self.path.relative_to(ROOT)).replace('\\', '/'), 'sha256': sha(self.path),
                  'bytes': self.path.stat().st_size}
        terminal_error = check_failure or error
        self.terminal = {**record, 'terminal_status': status, 'requested_status': requested,
                         'terminal_error': terminal_error, 'check_failure': check_failure,
                         'continuation_limit': self.last_check, 'authorization': self.authorization_record,
                         'run_manifest': self.run_manifest_record}
        self.closed = True
        print(json.dumps({'resource_terminal': record, 'status': status, 'wall_seconds': elapsed}), flush=True)
        return dict(self.terminal)
