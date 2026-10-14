from pathlib import Path
import pytest
from research.smallk_failure_followup_20261014 import continuation_runtime_v1 as old
from research.smallk_failure_followup_20261014 import fast_storage_scan_v1 as fast


def test_nested_overlapping_roots_preserve_physical_file_dedup(tmp_path):
    sub = tmp_path / 'nested'
    sub.mkdir()
    (tmp_path / 'a.bin').write_bytes(b'abc')
    (sub / 'b.bin').write_bytes(b'12345')
    roots = [tmp_path, sub, tmp_path / 'missing', tmp_path]
    assert old.storage_bytes(roots) == fast.storage_bytes(roots) == 8


def test_symlink_file_and_directory_match_original_enumeration(tmp_path):
    source = tmp_path / 'data'
    source.mkdir()
    target = source / 'target.bin'
    target.write_bytes(b'1234567')
    try:
        (tmp_path / 'alias.bin').symlink_to(target)
        (tmp_path / 'alias_dir').symlink_to(source, target_is_directory=True)
    except OSError as error:
        pytest.skip('Windows symlink creation unavailable: ' + type(error).__name__)
    roots = [tmp_path, source]
    assert old.storage_bytes(roots) == fast.storage_bytes(roots) == 7


def test_a_file_root_is_empty_like_original_glob(tmp_path):
    value = tmp_path / 'file.bin'
    value.write_bytes(b'1234')
    assert old.storage_bytes([value]) == fast.storage_bytes([value]) == 0
