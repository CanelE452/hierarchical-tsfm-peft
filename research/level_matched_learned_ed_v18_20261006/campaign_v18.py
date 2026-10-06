"""Execute only the prescribed TRAIN/VAL grid and seal its joint selection."""
import argparse
import csv
import json
import subprocess
import sys
import time

from runtime_v18 import (CACHE, DATASETS, HERE, ROOT, SEEDS, artifact, budget_snapshot,
                         check_source_seal, read_json, save_json)
from train_v18 import require_preflight, run_id


def execute(args, label):
    log = CACHE / 'campaign.log'
    with log.open('a', encoding='utf-8') as handle:
        handle.write('\n' + label + '\n')
        handle.flush()
        result = subprocess.run([sys.executable, str(HERE / 'train_v18.py'), *args], cwd=ROOT,
                                stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        save_json(HERE / 'campaign_progress.json', {'status': 'stopped', 'failed_stage': label,
                       'returncode': result.returncode, 'log': artifact(log), 'budget': budget_snapshot()})
        raise RuntimeError('Campaign stopped; preserved child failure: ' + label)


def training_summary(results):
    fields = ['id','dataset','lr','seed','selected_epoch','best_val_mse','initial_val_mse',
              'epochs_completed','total_steps','microbatch','elapsed_s','peak_allocated_mib','peak_reserved_mib']
    with (HERE / 'training_summary.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({**{key: row[key] for key in fields[:-2]},
                             'peak_allocated_mib': row['training_peak']['allocated_bytes']/2**20,
                             'peak_reserved_mib': row['training_peak']['reserved_bytes']/2**20})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', required=True)
    parser.parse_args()
    check_source_seal()
    require_preflight()
    if (HERE / 'selection_seal.json').exists():
        raise FileExistsError('Joint selection already sealed; reuse completed artifacts')
    results = []
    for dataset in DATASETS:
        for lr in (1e-3,1e-4):
            for seed in SEEDS:
                identifier = run_id(dataset,lr,seed)
                path = HERE / 'runs' / identifier / 'result.json'
                if path.exists():
                    row = read_json(path)
                    if row['status']!='complete':
                        raise RuntimeError('Preserved partial attempt prevents duplicate fit')
                    artifact(row['checkpoint']['path'], row['checkpoint']['sha256'])
                else:
                    save_json(HERE / 'campaign_progress.json', {'status': 'fitting', 'current': identifier,
                               'complete_fits': len(results), 'budget': budget_snapshot(), 'updated_utc': time.time()})
                    print('START '+identifier, flush=True)
                    execute(['--dataset',dataset,'--lr',str(lr),'--seed',str(seed)], identifier)
                    row = read_json(path)
                results.append(row)
                print('COMPLETE '+identifier+' selected_epoch='+str(row['selected_epoch']), flush=True)
    training_summary(results)
    execute(['--select'], 'joint_VAL_only_selection')
    selection = read_json(HERE / 'selection.json')
    rows = [artifact(HERE / 'selection.json'), artifact(HERE / 'source_seal.json'), artifact(HERE / 'training_summary.csv')]
    for result in results:
        rows.append(artifact(HERE / 'runs' / result['id'] / 'result.json'))
        rows.extend(result[key] for key in ('checkpoint','best_val_prediction'))
    save_json(HERE / 'selection_seal.json', {'schema': 'matched_ed_joint_selection_seal_v18',
              'created_utc': time.time(), 'all_trainval_selection_complete_before_test': True,
              'test_predictions_at_seal': len(read_json(HERE / 'ledger.json')['test_predictions']),
              'fit_count': len(results), 'selected_run_ids': [identifier for unit in selection['units'].values() for identifier in unit['run_ids']],
              'prediction_ensemble': False, 'artifacts': rows})
    if read_json(HERE / 'selection_seal.json')['test_predictions_at_seal']!=0:
        raise RuntimeError('Joint seal created after new TEST exposure')
    save_json(HERE / 'campaign_progress.json', {'status': 'trainval_complete_selection_sealed',
              'complete_fits': len(results), 'new_test_predictions': 0, 'budget': budget_snapshot()})
    print(json.dumps({'status': 'joint_selection_sealed', 'fit_count': len(results), 'test_predictions': 0}))


if __name__=='__main__':
    main()
