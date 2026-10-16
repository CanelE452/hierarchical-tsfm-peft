"""Private-cache single-GPU ownership and conservative process-wall accounting.

This module imports no GPU framework. Call ``check`` at work boundaries; an
already submitted device operation cannot be interrupted by a Python budget.
"""
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import secrets
import socket
import threading
import time
import uuid

import psutil


class ResourceBusy(RuntimeError):
    pass


class BudgetExceeded(RuntimeError):
    pass


class _OSLock:
    def __init__(self, path):
        self.path = Path(path)
        self.file = None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(self.path, "a+b", buffering=0)
        if self.file.seek(0, 2) == 0:
            self.file.write(b"0")
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.file.close()
            self.file = None
            raise ResourceBusy("Another process holds the campaign GPU OS lock") from exc
        return self

    def close(self):
        if self.file is not None:
            self.file.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
            self.file.close()
            self.file = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *args):
        self.close()


def _atomic_json(path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with open(temporary, "x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _json_rows(path):
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _append(path, row):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _alive(pid, created):
    try:
        return abs(psutil.Process(int(pid)).create_time() - float(created)) < .001
    except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError, TypeError):
        return False


def gpu_usage_seconds(cache):
    """Existing probe entries are included without changing their source bytes."""
    charged = []
    seen = set()
    for row in _json_rows(Path(cache) / "GPU_USAGE.jsonl"):
        lease_id = row.get("lease_id")
        if lease_id and lease_id in seen:
            raise ValueError("Duplicated lease charge in GPU_USAGE.jsonl")
        seen.add(lease_id)
        value = float(row["charged_process_seconds"])
        if not math.isfinite(value) or value < 0:
            raise ValueError("Invalid GPU accounting entry")
        charged.append(value)
    return math.fsum(charged)


def _commit_receipt(cache, receipt):
    # Called only while holding the exclusive GPU OS lock. Immutable receipt
    # precedes journal append so a crash between the two can be recovered.
    directory = cache / "gpu_receipts"
    directory.mkdir(exist_ok=True)
    path = directory / (receipt["lease_id"] + ".json")
    if path.exists():
        receipt = json.loads(path.read_text(encoding="utf-8"))
    else:
        with open(path, "x", encoding="utf-8") as handle:
            json.dump(receipt, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
    if not any(x.get("lease_id") == receipt["lease_id"] for x in _json_rows(cache / "GPU_USAGE.jsonl")):
        _append(cache / "GPU_USAGE.jsonl", receipt)
    return receipt


class GPUReservation:
    def __init__(self, cache, total_gpu_seconds=43200, reserve_seconds=3600,
                 wall_deadline=1792050350 + 172800, purpose="training", kind=None):
        self.cache = Path(cache).resolve()
        self.total_gpu_seconds = float(total_gpu_seconds)
        self.reserve_seconds = float(reserve_seconds)
        self.wall_deadline = float(wall_deadline)
        self.purpose = purpose
        self.kind = kind or purpose
        if self.total_gpu_seconds <= 0 or self.reserve_seconds < 0:
            raise ValueError("Invalid GPU budget")
        if purpose not in ("training", "audit"):
            raise ValueError("purpose must be training or audit")
        self.lock = _OSLock(self.cache / "GPU_OWNER.lock")
        self.token = secrets.token_hex(32)
        self.lease_id = uuid.uuid4().hex
        self.pid = os.getpid()
        self.create_time = psutil.Process(self.pid).create_time()
        self.deadline = None
        self.started_unix = None
        self._entered = False
        self._server = None
        self._thread = None
        self._stop = threading.Event()
        self._authorized_children = {}

    def __enter__(self):
        self.lock.acquire()
        try:
            active_path = self.cache / "ACTIVE_GPU_LEASE.json"
            if active_path.exists():
                prior = json.loads(active_path.read_text(encoding="utf-8"))
                if _alive(prior["pid"], prior["create_time"]):
                    raise ResourceBusy("Recorded owner is still alive; manual investigation required")
                if any(_alive(x["pid"], x["create_time"]) for x in prior.get("authorized_children", [])):
                    raise ResourceBusy("A delegated GPU child remains alive; owner cannot be reclaimed")
                now = time.time()
                finish = min(now, float(prior["deadline"]))
                _commit_receipt(self.cache, dict(lease_id=prior["lease_id"],
                    kind="crash_recovery", status="CRASH_RECOVERED_UNKNOWN_CONSERVATIVE",
                    started_unix=prior["started_unix"], finished_unix=finish,
                    recovered_unix=now,
                    charged_process_seconds=max(0., finish-float(prior["started_unix"])),
                    accounting_scope="reserved_process_wall_unknown_activity"))
                active_path.unlink()
            used = gpu_usage_seconds(self.cache)
            available = self.total_gpu_seconds-used
            if self.purpose == "training":
                available -= self.reserve_seconds
            self.started_unix = time.time()
            self.deadline = min(self.wall_deadline, self.started_unix + available)
            if self.deadline <= self.started_unix:
                raise BudgetExceeded("GPU process-wall or campaign wall budget exhausted")
            self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._server.bind(("127.0.0.1", 0))
            self._server.listen(8)
            self._server.settimeout(.2)
            self._port = self._server.getsockname()[1]
            _atomic_json(active_path, dict(lease_id=self.lease_id, pid=self.pid,
                create_time=self.create_time, started_unix=self.started_unix,
                deadline=self.deadline, token=self.token, port=self._port,
                purpose=self.purpose, accounted_before_seconds=used, authorized_children=[]))
            self._entered = True
            self._thread = threading.Thread(target=self._serve, daemon=True)
            self._thread.start()
            return self
        except BaseException:
            if self._server:
                self._server.close()
            self.lock.close()
            raise

    @property
    def remaining(self):
        return max(0., self.deadline-time.time()) if self.deadline is not None else 0.

    def check(self):
        if not self._entered or self.lock.file is None:
            raise ResourceBusy("GPU reservation is not active")
        if time.time() >= self.deadline:
            raise BudgetExceeded("GPU reservation deadline reached")

    def child_environment(self):
        return dict(INFORMS_GPU_TOKEN=self.token, INFORMS_GPU_OWNER_PID=str(self.pid),
                    INFORMS_GPU_OWNER_CREATE_TIME=str(self.create_time),
                    INFORMS_GPU_LEASE_ID=self.lease_id, INFORMS_GPU_OWNER_PORT=str(self._port),
                    INFORMS_GPU_DEADLINE=str(self.deadline))

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                conn.settimeout(1)
                try:
                    request = json.loads(conn.recv(4096).decode())
                    self.check()
                    if not hmac.compare_digest(str(request.get("token", "")), self.token):
                        continue
                    child = psutil.Process(int(request["pid"]))
                    if abs(child.create_time()-float(request["create_time"])) >= .001:
                        continue
                    if not any(p.pid == self.pid and abs(p.create_time()-self.create_time) < .001
                               for p in child.parents()):
                        continue
                    self._authorized_children[child.pid] = dict(pid=child.pid, create_time=child.create_time())
                    active_path = self.cache / "ACTIVE_GPU_LEASE.json"
                    active = json.loads(active_path.read_text(encoding="utf-8"))
                    if active["lease_id"] != self.lease_id:
                        continue
                    active["authorized_children"] = list(self._authorized_children.values())
                    _atomic_json(active_path, active)
                    response = dict(lease_id=self.lease_id, pid=self.pid, create_time=self.create_time,
                                    child_pid=child.pid, nonce=request["nonce"], deadline=self.deadline)
                    payload = json.dumps(response, sort_keys=True)
                    signature = hmac.new(self.token.encode(), payload.encode(), hashlib.sha256).hexdigest()
                    conn.sendall(json.dumps(dict(payload=payload, signature=signature)).encode())
                except (ValueError, KeyError, OSError, psutil.Error, ResourceBusy, BudgetExceeded):
                    continue

    def __exit__(self, error_type, error, traceback):
        self._stop.set()
        if self._server:
            self._server.close()
        if self._thread:
            self._thread.join(timeout=2)
        finished = time.time()
        try:
            if self._entered:
                if any(_alive(x["pid"], x["create_time"]) for x in self._authorized_children.values()):
                    raise ResourceBusy("Join delegated GPU children before closing the parent reservation")
                _commit_receipt(self.cache, dict(lease_id=self.lease_id, kind=self.kind,
                    status="FAILED" if error_type else "CLOSED", started_unix=self.started_unix,
                    finished_unix=finished, charged_process_seconds=max(0., finished-self.started_unix),
                    deadline=self.deadline, deadline_overrun_seconds=max(0., finished-self.deadline),
                    accounting_scope="exclusive_reserved_process_wall_including_children",
                    exception_class=error_type.__name__ if error_type else None))
                active = self.cache / "ACTIVE_GPU_LEASE.json"
                if json.loads(active.read_text(encoding="utf-8"))["lease_id"] != self.lease_id:
                    raise ResourceBusy("Active GPU owner record was replaced unexpectedly")
                active.unlink()
        finally:
            self._entered = False
            self.lock.close()


class _ChildLease:
    def __init__(self, cache):
        self.cache = Path(cache).resolve()
        self.deadline = float(os.environ["INFORMS_GPU_DEADLINE"])

    @property
    def remaining(self):
        return max(0., self.deadline-time.time())

    def check(self):
        if time.time() >= self.deadline:
            raise BudgetExceeded("Parent GPU deadline reached")
        env = os.environ
        pid = int(env["INFORMS_GPU_OWNER_PID"])
        created = float(env["INFORMS_GPU_OWNER_CREATE_TIME"])
        if not _alive(pid, created):
            raise ResourceBusy("Parent GPU owner is not alive with the same process identity")
        active = json.loads((self.cache / "ACTIVE_GPU_LEASE.json").read_text(encoding="utf-8"))
        if (active["pid"] != pid or abs(active["create_time"]-created) >= .001
                or active["lease_id"] != env["INFORMS_GPU_LEASE_ID"]
                or active["deadline"] != self.deadline):
            raise ResourceBusy("Private owner record does not match child authorization")
        probe = _OSLock(self.cache / "GPU_OWNER.lock")
        try:
            probe.acquire()
        except ResourceBusy:
            pass
        else:
            probe.close()
            raise ResourceBusy("Owner OS lock is not held")
        nonce = secrets.token_hex(16)
        request = dict(token=env["INFORMS_GPU_TOKEN"], pid=os.getpid(),
                       create_time=psutil.Process().create_time(), nonce=nonce)
        with socket.create_connection(("127.0.0.1", int(env["INFORMS_GPU_OWNER_PORT"])), timeout=2) as conn:
            conn.sendall(json.dumps(request).encode())
            result = json.loads(conn.recv(4096).decode())
        expected = hmac.new(env["INFORMS_GPU_TOKEN"].encode(), result["payload"].encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, result["signature"]):
            raise ResourceBusy("GPU owner handshake signature mismatch")
        answer = json.loads(result["payload"])
        if (answer["nonce"] != nonce or answer["pid"] != pid or answer["child_pid"] != os.getpid()
                or abs(answer["create_time"]-created) >= .001
                or answer["lease_id"] != active["lease_id"] or answer["deadline"] != self.deadline):
            raise ResourceBusy("GPU owner handshake identity mismatch")
        return self

    def __enter__(self):
        return self.check()

    def __exit__(self, *args):
        return False


def validate_gpu_child(cache):
    """Borrow the live ancestor's lease; never create a child accounting budget."""
    return _ChildLease(cache)


def check_storage_budget(cache, max_bytes=30 * 1024**3, pending_bytes=0):
    total = int(pending_bytes)
    if total < 0:
        raise ValueError("Negative pending storage")
    for directory, subdirs, files in os.walk(Path(cache), followlinks=False):
        subdirs[:] = [x for x in subdirs if not Path(directory, x).is_symlink()]
        for name in files:
            path = Path(directory, name)
            if not path.is_symlink():
                total += path.stat().st_size
    if total > max_bytes:
        raise BudgetExceeded("New campaign storage cap exceeded")
    return total


def record_download(cache, source_id, byte_count, max_bytes=5 * 1024**3):
    """Reserve declared transfer bytes before starting a download; no refunds."""
    cache = Path(cache)
    byte_count = int(byte_count)
    if byte_count < 0:
        raise ValueError("Negative download size")
    with _OSLock(cache / "DOWNLOAD_BUDGET.lock"):
        baseline_path = cache / "DOWNLOAD_BASELINE.json"
        if baseline_path.exists():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        else:
            sources = cache / "sources"
            paths = list(sources.glob("*_download_receipt.json"))
            evidence = sources / "source_evidence_receipt.json"
            if evidence.exists():
                paths.append(evidence)
            records = []
            for receipt_path in sorted(paths):
                content = json.loads(receipt_path.read_text(encoding="utf-8"))
                items = [content] if "download_bytes_this_invocation" in content else list(content.values())
                count = sum(int(x["download_bytes_this_invocation"]) for x in items)
                if count < 0:
                    raise ValueError("Invalid prior download accounting")
                records.append(dict(receipt=receipt_path.name, bytes=count,
                    sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest()))
            baseline = dict(created_unix=time.time(), prior_download_bytes=sum(x["bytes"] for x in records),
                            source_receipts=records)
            with open(baseline_path, "x", encoding="utf-8") as handle:
                json.dump(baseline, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
        path = cache / "DOWNLOAD_USAGE.jsonl"
        used = baseline["prior_download_bytes"] + sum(int(x["reserved_bytes"]) for x in _json_rows(path))
        if used + byte_count > max_bytes:
            raise BudgetExceeded("New download cap exceeded")
        _append(path, dict(source_id=source_id, reserved_bytes=byte_count, reserved_unix=time.time()))
    return used + byte_count
