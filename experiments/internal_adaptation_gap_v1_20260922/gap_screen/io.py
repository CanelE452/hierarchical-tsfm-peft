from __future__ import annotations
import csv, hashlib, json, os, subprocess, time
from pathlib import Path
from typing import Any, Iterable
import numpy as np

class Blocked(RuntimeError):
    """An execution/data/resource problem; never a scientific negative result."""

def jsonable(x: Any) -> Any:
    if isinstance(x, dict): return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    if isinstance(x, (Path,)): return str(x)
    if isinstance(x, np.ndarray): return jsonable(x.tolist())
    if isinstance(x, np.generic): return jsonable(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x

def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path: Path, data: Any) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(jsonable(data), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""): h.update(b)
    return h.hexdigest()

def digest_object(data: Any) -> str:
    return hashlib.sha256(json.dumps(jsonable(data), sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns); w.writeheader()
        for r in rows: w.writerow({k: jsonable(v) for k, v in r.items()})

def run_command(args: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    p = subprocess.run(args, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if check and p.returncode:
        raise Blocked(f"COMMAND_FAILED {args[0]}: {p.stderr[-1800:]}")
    return p

def code_manifest(code_root: Path) -> dict[str, str]:
    files = []
    for p in code_root.rglob("*"):
        if not p.is_file() or any(v in p.parts for v in ["__pycache__", ".pytest_cache", ".venv"]): continue
        if p.suffix in {".py", ".json", ".md", ".txt"} and "test_output" not in p.parts:
            files.append(p)
    return {p.relative_to(code_root).as_posix(): sha(p) for p in sorted(files)}

def verify_code(code_root: Path, seal: dict) -> None:
    actual = code_manifest(code_root)
    if actual != seal["code"]: raise Blocked("SOURCE_SEAL_CHANGED: do not resume with changed experiment code/config")

class Ledger:
    def __init__(self, path: Path, cap: int):
        self.path = path; self.cap = cap; self.keys: set[tuple[str, int]] = set(); self.open_intents = set()
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                key = (r.get("fit", ""), int(r.get("step", -1)))
                if r["kind"] == "intent": self.keys.add(key); self.open_intents.add(key)
                elif r["kind"] == "commit": self.open_intents.discard(key)
        if self.open_intents: raise Blocked("AMBIGUOUS_OPTIMIZER_UPDATE: an intent lacks a commit; automatic replay forbidden")
    def emit(self, kind: str, **kw) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(jsonable(dict(kind=kind, utc=time.time(), **kw)), allow_nan=False) + "\n")
            f.flush(); os.fsync(f.fileno())
    def intent(self, fit: str, step: int) -> None:
        key = (fit, step)
        if key in self.keys: raise Blocked(f"DUPLICATE_UPDATE {key}")
        if len(self.keys) >= self.cap: raise Blocked("OPTIMIZER_CAP_EXCEEDED")
        self.keys.add(key); self.emit("intent", fit=fit, step=step)
    def commit(self, fit: str, step: int) -> None: self.emit("commit", fit=fit, step=step)

def require_complete_ledger(path: Path, expected: int) -> dict:
    rs = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    a = [(r["fit"], r["step"]) for r in rs if r["kind"] == "intent"]
    b = [(r["fit"], r["step"]) for r in rs if r["kind"] == "commit"]
    assert len(a) == len(set(a)) == expected and a == b, (len(a), len(b), expected)
    return dict(updates=expected, unique=True, unmatched_intents=0)
