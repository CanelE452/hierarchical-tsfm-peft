"""Same caps, with resource census and durable heartbeat at most one second apart."""
import time

import runtime_s0_reference as runtime


class FastBudget(runtime.Budget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ledger_cache = None
        self._resource_snapshot = None
        self._sampled_at = 0.
        self._persisted_at = 0.
        self._force_resources = False

    def _load_ledger(self):
        if self._ledger_cache is None:
            self._ledger_cache = super()._load_ledger()
        return self._ledger_cache

    def _persist(self, force=False):
        now = time.perf_counter()
        if self._ledger_cache is not None and (force or now-self._persisted_at >= 1.):
            runtime.save_json(self.ledger_path, self._ledger_cache)
            self._persisted_at = now

    def _save_ledger(self, ledger):
        self._ledger_cache = ledger
        self._persist()

    def _raw_snapshot(self):
        now_perf = time.perf_counter()
        if self._resource_snapshot is None or self._force_resources or now_perf-self._sampled_at >= 1.:
            self._resource_snapshot = super()._raw_snapshot()
            self._sampled_at = now_perf
        snap = dict(self._resource_snapshot)
        ledger = self._load_ledger()
        wall, cpu = self._current_elapsed()
        gpu_wall = cpu_wall = cpu_process = 0.
        for row in ledger['jobs']:
            if row['status'] == 'running':
                if row['id'] != self.job_id:
                    raise RuntimeError('STOP_RESOURCE: unexpected concurrent job')
                row_wall, row_cpu = wall, cpu
            else:
                row_wall, row_cpu = row['elapsed_wall_seconds'], row['process_time_seconds']
            if row['category'] == 'gpu':
                gpu_wall += row_wall
            else:
                cpu_wall += row_wall
            cpu_process += row_cpu
        snap.update(operation_elapsed_seconds=time.time()-ledger['operation_started_utc'],
                    gpu_wall_seconds=gpu_wall, cpu_wall_seconds=cpu_wall,
                    cpu_process_seconds=cpu_process, counters=dict(ledger['counters']))
        return snap

    def check(self, force=False):
        self._force_resources = force
        try:
            return super().check()
        except Exception:
            self._persist(force=True)
            raise
        finally:
            self._force_resources = False

    def tick(self, kind, n=1):
        result = super().tick(kind, n)
        if runtime._COUNTER_ALIASES[kind] in ('scientific_fits', 'new_test_predictions'):
            self._persist(force=True)
        return result

    def heartbeat(self, progress):
        self.check(force=True)
        ledger = self._load_ledger()
        ledger['jobs'][self.job_id]['progress'] = progress
        self._persist(force=True)

    def __enter__(self):
        try:
            result = super().__enter__()
            self._persist(force=True)
            return result
        except Exception:
            self._persist(force=True)
            raise

    def __exit__(self, exc_type, exc, tb):
        cap_error = None
        try:
            self.check(force=True)
        except Exception as error:
            cap_error = error
        try:
            if exc_type is None and cap_error is not None:
                super().__exit__(type(cap_error), cap_error, cap_error.__traceback__)
            else:
                super().__exit__(exc_type, exc, tb)
        finally:
            self._persist(force=True)
        if exc_type is None and cap_error is not None:
            raise cap_error
        return False
