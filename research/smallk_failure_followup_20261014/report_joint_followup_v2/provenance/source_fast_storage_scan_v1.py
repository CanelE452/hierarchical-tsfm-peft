"""Physical-path deduplication with canonical directories and symlink files."""
import os
from pathlib import Path


def storage_bytes(roots):
    pending = [Path(root) for root in roots if Path(root).is_dir()]
    directories, files = set(), set()
    total = 0
    while pending:
        directory = pending.pop().resolve()
        if directory in directories:
            continue
        directories.add(directory)
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.is_dir(follow_symlinks=False):
                    if not entry.is_symlink():
                        pending.append(Path(entry.path))
                elif entry.is_file():
                    path = Path(entry.path)
                    canonical = path.resolve() if entry.is_symlink() else path
                    if canonical not in files:
                        files.add(canonical)
                        total += entry.stat().st_size
    return total
