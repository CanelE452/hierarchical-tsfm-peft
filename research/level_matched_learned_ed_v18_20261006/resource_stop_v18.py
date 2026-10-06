"""Record the mandatory resource stop and preserve preflight evidence without reruns."""
import json
import subprocess
import time

from runtime_v18 import (CACHE, DATASETS, HERE, LIMITS, ROOT, artifact, budget_snapshot,
                         digest, read_json, save_json)


def main():
    destination = HERE / 'resource_stop.json'
    if destination.exists():
        raise FileExistsError('Resource decision already recorded; preserve it')
    preflight = read_json(HERE / 'preflight.json')
    ledger = read_json(HERE / 'ledger.json')
    estimate = preflight['estimate']
    if preflight['status']!='STOP_RESOURCE' or estimate['projected_full_campaign_upper_s']<=LIMITS['gpu_seconds']:
        raise RuntimeError('No mandatory resource stop is evidenced')
    if ledger['fits'] or ledger['test_predictions']:
        raise RuntimeError('The stopped preflight must not be followed by any new fit or TEST')
    unchanged = [artifact(row['path'], row['sha256']) for row in preflight['source_artifacts']]
    protected = read_json(CACHE / 'protected_workspace_before.json')
    changed = [path for path, row in protected.items() if not (ROOT / path).is_file() or digest(path)!=row['sha256']]
    if changed:
        raise RuntimeError('Protected original file changed: '+str(changed[:8]))
    checks = {dataset: {
         'step0_full_canonical_val_parity': row['step0_COMPRESS_parity']['pass'],
         'finite_nonzero_encoder_decoder_gradients': all(
             gradient['present'] and gradient['finite'] and gradient['nonzero']>0
             for record in row['autograd_probe']['microbatch_records'] for gradient in record['gradients'].values()),
         'parent_seed_epoch_schedule_identity': row['sampling']['all_240_seed_epoch_schedules_literal_parent_identity'],
         'zero_update_state_preservation': row['state_preservation']['unchanged'],
         'optimizer_updates': row['state_preservation']['optimizer_updates'],
         'canonical_val_origins': row['canonical_VAL_origins'],
         'trainable_parameters': row['model_receipt']['trainable_parameter_count'],
         'basis_sha256': row['literal_PCA']['sha256'],
    } for dataset, row in preflight['units'].items()}
    if set(checks)!=set(DATASETS) or not all(all(row[k] for k in (
             'step0_full_canonical_val_parity','finite_nonzero_encoder_decoder_gradients',
             'parent_seed_epoch_schedule_identity','zero_update_state_preservation')) for row in checks.values()):
        raise RuntimeError('Scientific readiness checks incomplete')
    save_json(destination, {'schema': 'matched_ed_resource_stop_v18', 'status': 'BLOCKED_STOP_RESOURCE',
       'created_utc': time.time(), 'reason': 'Adopted REQUEST section15: full expected scope exceeds cumulative GPU3hour cap; no silent scope reduction',
       'full_campaign_complete': False, 'new_fits': 0, 'new_test_predictions': 0, 'optimizer_updates': 0,
       'scientific_readiness': checks, 'resource_gate_passed': False,
       'projected_full_scope_upper_s': estimate['projected_full_campaign_upper_s'], 'cap_s': LIMITS['gpu_seconds'],
       'projected_excess_s': estimate['projected_full_campaign_upper_s']-LIMITS['gpu_seconds'],
       'estimate_is_actual_fit_time': False, 'assumed_early_stopping': False,
       'preflight_actual_gpu_job_wall_s': sum(row['elapsed_s'] for row in ledger['jobs'] if row['category']=='gpu'),
       'mandatory_stop_preserved': True, 'preflight_repeated': False, 'estimate_retuned': False, 'scope_reduced': False,
       'preflight': artifact(HERE / 'preflight.json'), 'failed_attempt': artifact(HERE / 'preflight_attempt01.json'),
       'source_artifacts_unchanged': unchanged, 'protected_original_files': len(protected), 'protected_files_unchanged': True,
       'live_origin_main': subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=ROOT,text=True).split()[0],
       'unexecuted_after_gate': ['TRAIN/VAL fits','LR selection','joint selection seal','selected TEST predictions',
           'paired ED accuracy evaluation','matched training-path cost probe','matched inference cost','selected-checkpoint mechanism diagnostics'],
       'scientific_verdict': None, 'verdict_unavailable_reason': 'No trained/selected learnedED prediction exists; preflight cannot rank accuracy or methods',
       'minimum_next_plan': {'same_fit_grid': '3datasets x2learning_rates x2seeds =12fits', 'same_selected_test_predictions': 6,
          'recipe_or_scientific_scope_changes': False, 'only_required_external_change': 'Explicitly revise resource budget before any new fitting',
          'example_reviewable_gpu_cap_s': 14400, 'new_permission_granted': False},
       'self_check_not_independent_reproduction': True, 'pass_not_novelty_evidence': True})
    print(json.dumps({'status':'BLOCKED_STOP_RESOURCE','full_scope_estimate_s':estimate['projected_full_campaign_upper_s'],
                      'cap_s': LIMITS['gpu_seconds'],'new_fits':0,'new_test_predictions':0,'protected_files':len(protected)}))


if __name__=='__main__':
    main()
