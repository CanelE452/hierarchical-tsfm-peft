"""Meter the explicitly uncapped follow-up while preserving original receipts."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = HERE.parents[2]
AUTH = HERE / 'AUTHORIZATION_SCOPE_v1.json'
MANIFEST = HERE / 'CONTINUATION_MANIFEST_v1.json'
RESOURCES = HERE / 'resources'
REPLY_ID = 'call_e1682a01be79412a9ffe80d525932115:0'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


class Meter:
    def __init__(self, tag, gpu=False):
        import psutil
        self.auth, self.manifest = read(AUTH), read(MANIFEST)
        if (self.auth.get('status') != 'APPROVED_HUMAN_UNCAPPED_CONTINUATION'
                or self.auth.get('gate_reply_id') != REPLY_ID
                or self.auth.get('gate_reply') != '게이트 수정 승인 — 기존 잔여 상한 내 실행'
                or self.auth.get('latest_human_override') != '상한내수행하지 말고 쭉해줘'
                or self.auth.get('original_auth_sha256') != sha(BASE / 'AUTHORIZATION_v1.json')
                or self.auth.get('original_manifest_sha256') != sha(BASE / 'CAMPAIGN_MANIFEST_v1.json')
                or self.auth.get('addendum_sha256') != sha(HERE / 'EXECUTION_SCOPE_ADDENDUM_KO_v2.md')
                or self.manifest.get('authorization_sha256') != sha(AUTH)
                or self.manifest.get('runtime_sha256') != sha(Path(__file__))):
            raise PermissionError('EXACT_HUMAN_SCOPE_AUTHORIZATION_REQUIRED')
        self.process = psutil.Process()
        self.started = time.perf_counter()
        self.pid = os.getpid()
        self.gpu = bool(gpu)
        self.closed = False
        RESOURCES.mkdir(exist_ok=True)
        self.path = RESOURCES / f'{self.pid}_{time.time_ns()}_{tag}.json'
        self.receipt = {'status': 'RUNNING', 'tag': tag, 'pid': self.pid, 'gpu_process': self.gpu,
                        'process_created_unix': self.process.create_time(), 'started_unix': time.time(),
                        'authorization': {'path': str(AUTH), 'sha256': sha(AUTH)},
                        'continuation_manifest': {'path': str(MANIFEST), 'sha256': sha(MANIFEST)},
                        'original_budget_records_unchanged': True,
                        'elapsed_or_cumulative_budget_caps': None,
                        'sampled_peak_rss_bytes': 0, 'peak_allocated_vram_bytes': 0,
                        'peak_reserved_vram_bytes': 0}
        write_new(self.path, self.receipt)
        self.check()

    def check(self, event=None):
        if self.closed:
            raise RuntimeError('METER_ALREADY_CLOSED')
        self.receipt['sampled_peak_rss_bytes'] = max(self.receipt['sampled_peak_rss_bytes'], self.process.memory_info().rss)
        torch = sys.modules.get('torch')
        if torch is not None and torch.cuda.is_initialized():
            self.receipt['peak_allocated_vram_bytes'] = max(self.receipt['peak_allocated_vram_bytes'], torch.cuda.max_memory_allocated())
            self.receipt['peak_reserved_vram_bytes'] = max(self.receipt['peak_reserved_vram_bytes'], torch.cuda.max_memory_reserved())
        self.receipt['wall_seconds'] = time.perf_counter() - self.started
        cpu = self.process.cpu_times()
        self.receipt['process_cpu_seconds'] = cpu.user + cpu.system
        if event is not None:
            self.receipt['latest_event'] = event
        self.receipt['heartbeat_unix'] = time.time()
        # This receipt belongs to this live job; historical receipts are never edited.
        self.path.write_text(json.dumps(self.receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        return self.receipt

    def protected_check(self):
        self.check()
        for row in self.manifest['protected_records']:
            if sha(row['path']) != row['sha256']:
                raise RuntimeError('PROTECTED_SOURCE_CHANGED:' + row['path'])

    def close(self, status, error=None):
        self.check()
        self.receipt.update(status=status, error=error, ended_unix=time.time())
        self.path.write_text(json.dumps(self.receipt, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        self.closed = True
        return {'path': str(self.path), 'sha256': sha(self.path), **self.receipt}
