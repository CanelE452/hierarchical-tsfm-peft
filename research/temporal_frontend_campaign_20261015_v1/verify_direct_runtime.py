"""Recheck fixed B inputs after changing only the Windows interpreter entrypoint."""
import json
import time
from pathlib import Path

import torch

import cost_and_score as cs
import runtime as rt


def main():
    for name in ('FIRST_C_SCORE.json', 'ACTUAL_PARITY_DIRECT_INTERPRETER.json', 'DIRECT_PARITY_EXECUTION.json'):
        assert not (rt.HERE / name).exists(), name
    lock = json.loads((rt.HERE / 'COST_LOCK.json').read_text(encoding='utf-8'))
    assert lock['status'] == 'COMPLETE_VALID'
    cs.assert_no_training(check_exclusive=True)
    started = time.time()
    assert lock['time_unix'] < started
    routes = cs.rows()
    bolt = rt.Bolt()
    cs.parity(bolt, routes, *cs.make_context(bolt), output_name='ACTUAL_PARITY_DIRECT_INTERPRETER.json')
    assert rt.state_hash(bolt.model) == bolt.initial_hash
    assert not (rt.HERE / 'FIRST_C_SCORE.json').exists()
    receipt = dict(status='PASS', started_unix=started, finished_unix=time.time(),
        selection_lock_sha256=rt.sha(rt.HERE / 'SELECTION_LOCK.json'),
        cost_lock_sha256=rt.sha(rt.HERE / 'COST_LOCK.json'),
        parity_sha256=rt.sha(rt.HERE / 'ACTUAL_PARITY_DIRECT_INTERPRETER.json'),
        statistics_sha256=rt.sha(rt.HERE / 'FOLD_STATISTICS.json'),
        F_model_hash=bolt.initial_hash, C_before=False, C_after=False,
        scientific_sources_changed=False, driver_source_sha256=rt.sha(Path(__file__)),
        scope='Real B Raw physical/microbatch and CP0 checks after runtime entrypoint repair; no optimizer update or C score')
    rt.write(rt.HERE / 'DIRECT_PARITY_EXECUTION.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    main()
