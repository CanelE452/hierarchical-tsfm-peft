"""Isolated cost output writers; historical scientific runtime stays unchanged."""
import json
from pathlib import Path
import time
from research.ppt_justification_20261028_v1 import runtime as base

NAME = 'appliances_cost_20261028_v2'
REPO = base.REPO
REPORT = base.REPORT
CACHE = base.CACHE
OUTPUT_REPORT = REPO / 'reports' / NAME
OUTPUT_CACHE = REPO / '.cache' / NAME

def __getattr__(name):
    return getattr(base, name)

def output_path(path):
    path = Path(path).resolve()
    if not any(path.is_relative_to(root.resolve()) for root in (OUTPUT_REPORT, OUTPUT_CACHE)):
        raise ValueError('Output is outside the isolated cost campaign')
    return path

def write_new(path, value):
    path = output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')

def atomic(path, value):
    path = output_path(path)
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
