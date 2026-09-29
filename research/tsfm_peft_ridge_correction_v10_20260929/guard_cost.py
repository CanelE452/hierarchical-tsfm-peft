"""Bound the approved cost job, including a shutdown margin under its hard cap."""
import time
from runtime_ridge import HERE, Job, read_json, save_json, transaction, digest


if __name__ == '__main__':
    with Job('cost_budget_guard01', 'cpu_check', 160, metadata={'purpose': 'own-job budget supervision'}) as guard:
        import psutil
        while True:
            ledger = read_json(HERE / 'ledger.json')
            row = next(r for r in ledger['jobs'] if r['label'] == 'jena_cost01')
            if row['status'] != 'running':
                save_json(HERE / 'cost_guard.json', {'status': 'natural_completion', 'cost_status': row['status'],
                          'cost_elapsed_s': row['elapsed_s'], 'stopped_processes': []})
                break
            if time.time() - row['started_utc'] >= 585:
                process = psutil.Process(row['pid'])
                command = process.cmdline()
                assert any('tsfm_peft_ridge_correction_v10_20260929' in arg and arg.endswith('cost_ridge.py') for arg in command)
                assert '--job' in command and command[command.index('--job') + 1] == 'jena_cost01'
                assert abs(process.create_time() - row['started_utc']) < 30
                save_json(HERE / 'cost_stop_request.json', {'reason': 'retain15s shutdown margin under approved600s cost cap',
                          'pid': row['pid'], 'command': command, 'requested_utc': time.time(),
                          'scope': 'only this verified v10 cost job; no other project process'})
                process.terminate()
                process.wait(timeout=10)
                stopped = time.time()
                with transaction() as ledger:
                    target = ledger['jobs'][row['id']]
                    target.update(status='stopped_budget', ended_utc=stopped,
                                  elapsed_s=stopped - target['started_utc'],
                                  reason='Supervisor stopped its own verified job before the600s cost ceiling; retain partial measurements')
                partial = HERE / 'cost_jena_jena_cost01_partial.json'
                parity = HERE / 'cost_jena_jena_cost01_parity.json'
                saved = read_json(partial)
                save_json(HERE / 'cost_guard.json', {'status': 'stopped_budget', 'cost_elapsed_s': stopped - row['started_utc'],
                          'stopped_processes': [row['pid']], 'completed_rows': len(saved['rows']), 'planned_rows': 78,
                          'partial': {'path': str(partial), 'sha256': digest(partial)},
                          'parity': {'path': str(parity), 'sha256': digest(parity)},
                          'final_speed_ranking': 'unconfirmed; incomplete planned3-block grid',
                          'new_fits': 0, 'accuracy_evaluations_remain_valid': True})
                break
            time.sleep(2)
