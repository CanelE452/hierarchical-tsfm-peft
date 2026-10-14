import numpy as np
import audit_sequence_balanced_cost_v1 as audit
from audit_sequence_balanced_cost_v1 import close_arrays


def test_normalized_output_tolerance_does_not_accept_physical_threshold():
    ref = np.zeros((1, 48, 2), dtype=np.float32)
    assert close_arrays(ref + 0.000009, ref, 'normalized')
    assert not close_arrays(ref + 0.000011, ref, 'normalized')
    assert close_arrays(ref + 0.0019, ref, 'physical')
    assert not close_arrays(ref + 0.0021, ref, 'physical')


def test_shape_and_nonfinite_values_cannot_pass_output_parity():
    ref = np.ones((1, 48, 2), dtype=np.float32)
    assert not close_arrays(ref[:, :1], ref, 'normalized')
    altered = ref.copy()
    altered[0, 0, 0] = np.nan
    assert not close_arrays(altered, altered, 'normalized')
    assert not close_arrays(altered, ref, 'physical')


def test_original_bulk_false_is_diagnostic_not_protocol_failure():
    result = {
        'checks': 0,
        'failed_count': 0,
        'failures': [],
        'original_bulk_declared_false_count': 0,
    }

    def check(name, passed):
        result['checks'] += 1
        if not passed:
            result['failed_count'] += 1
            result['failures'].append(name)

    audit.record_original_bulk_flag(result, check, 'bulkfalse', {'pass': False, 'max_abs': 0.1})

    assert result['failed_count'] == 0
    assert result['original_bulk_declared_false_count'] == 1
    # Same-batch output parity remains the protocol gate and still fails when it exceeds tolerance.
    ref = np.zeros((1, 48, 1), dtype=np.float32)
    assert not audit.close_arrays(ref + 0.00002, ref, 'normalized')
