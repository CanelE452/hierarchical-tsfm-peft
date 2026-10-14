from pathlib import Path
import hashlib
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import audit_stage_native_core_v3 as core


def test_original_npz_manifest_uses_raw_bytes_without_model_hash_metadata(tmp_path):
    value = np.array([[1., 2.]], np.float32)
    path = tmp_path / 'actual_schema.npz'
    np.savez_compressed(path, latent_past=value)
    raw_sha = hashlib.sha256(value.tobytes()).hexdigest()
    record = {'path': str(path), 'sha256': core.sha_file(path), 'arrays': {
        'latent_past': {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': raw_sha}}}
    arrays, _ = core.load_npz_record(record, role='fixture')
    assert np.array_equal(arrays['latent_past'], value)
    assert core.array_sha(value) != raw_sha
    record['arrays']['latent_past']['sha256'] = core.array_sha(value)
    with pytest.raises(core.AuditFailure, match='CONTENT_MISMATCH'):
        core.load_npz_record(record, role='fixture')
