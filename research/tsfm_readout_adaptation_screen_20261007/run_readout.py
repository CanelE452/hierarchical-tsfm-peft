"""One bounded S1 campaign: fixed caches, fits, joint choices, TEST, matched costs."""
import os
import time

ENTRY_UTC = time.time()
ENTRY_PERF = time.perf_counter()
ENTRY_CPU = time.process_time()
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '4'

import traceback

from runtime_readout import Budget, HERE, configure, save_json
from prepare_readout import prepare_all
from train_readout import select_all, train_all
from evaluate_readout import evaluate_all
from cost_readout import measure_all


def main():
    configure()
    state = {'schema': 'tsfm_readout_campaign_state_v1', 'status': 'running',
             'stage': 'cache', 'started_utc': ENTRY_UTC, 's2_executed': False,
             's3_executed': False, 's4_executed': False}
    save_json(HERE / 'campaign_state.json', state)
    try:
        with Budget(stage='s1', category='gpu', label='complete_fixed_s1_campaign') as budget:
            # Include interpreter/library startup in process and job costs.
            budget._start_perf, budget._start_process = ENTRY_PERF, ENTRY_CPU
            ledger = budget._load_ledger()
            ledger['jobs'][budget.job_id]['started_utc'] = ENTRY_UTC
            ledger['operation_started_utc'] = min(ledger['operation_started_utc'], ENTRY_UTC)
            budget.heartbeat('CACHE: sealed schedules and immutable prefix features')
            prepare_all(budget)
            state['stage'] = 'fits'
            save_json(HERE / 'campaign_state.json', state)
            budget.heartbeat('FITS: fixed 24-run TRAIN/VAL grid; TEST sealed')
            train_all(budget)
            state['stage'] = 'selection'
            save_json(HERE / 'campaign_state.json', state)
            select_all()
            state['stage'] = 'evaluation'
            save_json(HERE / 'campaign_state.json', state)
            budget.heartbeat('TEST: all 24 choices sealed, evaluate selected twelve models')
            evaluate_all(budget)
            state['stage'] = 'costs'
            save_json(HERE / 'campaign_state.json', state)
            budget.heartbeat('COSTS: disposable selected copies and complete-model deployment')
            measure_all(budget)
            state['stage'] = 'computed'
            state['budget'] = budget.snapshot()
            save_json(HERE / 'campaign_state.json', state)
        state.update(status='complete', elapsed_wall_seconds=time.perf_counter()-ENTRY_PERF,
                     process_cpu_seconds=time.process_time()-ENTRY_CPU, finished_utc=time.time())
        save_json(HERE / 'campaign_state.json', state)
        print('S1 COMPUTATION COMPLETE', flush=True)
    except BaseException as error:
        state.update(status='STOP_RESOURCE' if 'RESOURCE' in str(error) else 'STOP_DEBUG',
                     error=type(error).__name__+': '+str(error), traceback=traceback.format_exc(),
                     elapsed_wall_seconds=time.perf_counter()-ENTRY_PERF,
                     process_cpu_seconds=time.process_time()-ENTRY_CPU, finished_utc=time.time())
        save_json(HERE / 'campaign_state.json', state)
        print(state['status']+' '+state['error'], flush=True)
        raise


if __name__ == '__main__':
    main()
