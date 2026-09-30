"""Verify completed scope and provenance without rerunning a model or fit."""
import argparse
import ast
import csv
import subprocess
from pathlib import Path
from runtime_v12 import HERE,ROOT,OLD,Job,read_json,save_json,artifact,budget_snapshot,verify_provenance


def verify(label):
    checked={}
    def receipts(value):
        if isinstance(value,dict):
            if isinstance(value.get('path'),str) and 'sha256' in value:
                p=Path(value['path'])
                if not p.is_absolute(): p=ROOT/p
                key=str(p)
                if key not in checked: checked[key]=artifact(p)
                assert checked[key]['sha256']==value['sha256'],key
            for v in value.values(): receipts(v)
        elif isinstance(value,list):
            for v in value: receipts(v)
    verify_provenance()
    names=['reuse_manifest.json','q_contribution02.json','q_uncertainty01.json',
           'q_artifacts_check01.json','equivalence_audit01.json','equivalence_paired_backward_ledger.json',
           'equivalence_mask_policy01.json','equivalence_scheduler_replay01.json','figure_receipt.json',
           'report_values.json','cost_reference.json']
    for name in names: receipts(read_json(HERE/name))
    for p in HERE.glob('*.py'): ast.parse(p.read_text(encoding='utf-8'))
    q=read_json(HERE/'q_contribution02.json')
    checks=read_json(HERE/'q_artifacts_check01.json')
    audit=read_json(HERE/'equivalence_audit01.json')
    pairs=read_json(HERE/'equivalence_paired_backward_ledger.json')['pairs']
    decision=read_json(HERE/'NO_Q_RUN_DECISION.json')
    report=read_json(HERE/'report_values.json')
    assert checks['status']=='passed' and q['new_fits']==0
    assert len(pairs)==24 and all(x['status']=='complete' and x['optimizer_updates']==0 for x in pairs)
    assert audit['full_training_policy_equivalence_established'] is False
    assert audit['test_accessed'] is False
    assert all(not x['no_q_fit'] and not x['test_error_used_for_decision'] for x in decision['datasets'].values())
    assert report['original_detailed_plan_recovered'] is False and report['independent_confirmation'] is False
    for ds,data in q['datasets'].items():
        for family,summary in data['summary'].items():
            expected=summary['periods']['combined']
            actual=report['datasets'][ds]['scores'][family]
            assert expected==actual
    assert read_json(HERE/'cost_reference.json')['status']=='historical_v11_evidence_reused_not_v12_benchmark'
    old_changes=subprocess.check_output(['git','diff','a63d951','--name-only','--','research/tsfm_peft_internal_vs_subspace_v11_20260930'],cwd=ROOT,text=True).strip()
    assert not old_changes,'Original v11 changed'
    budget=budget_snapshot()
    assert budget['real_fit_attempts']==0 and budget['synthetic_sessions']==0
    assert all(budget[c+'_seconds']<budget['limits'][c+'_seconds'] for c in ('gpu','cpu_analysis','cpu_check'))
    assert budget['storage_bytes']<budget['limits']['storage_bytes']
    outputs={n:artifact(HERE/n) for n in names+['NO_Q_RUN_DECISION.json','TOPIC_DECISION.md','METHOD_UPDATE.md','CLAIM_EVIDENCE.md']}
    save_json(HERE/'final_checks.json',{
        'status':'passed_for_completed_old_data_scope',
        'overall_v12_confirmation_complete':False,
        'pending':'Actual preapproved new evaluation identities/source/columns/splits not recovered',
        'models_scored':45,'prior_scores_replayed':39,'q_parities':6,'paired_backward':24,
        'actual_optimizer_updates':0,'new_fits':0,'checks':checks['checks']+[
            'artifact_bytes_verified','original_v11_unchanged','no_test_driven_no_q_fit_decision',
            'actual_policy_not_asserted_equivalent','historical_cost_not_relabeled','figure_values_match',
            'missing_confirmation_explicit','budget_within_caps'],
        'verified_unique_receipts':len(checked),'outputs':outputs,'budget_snapshot_during_check':budget,
        'self_check_not_independent_reproduction':True,'job_label':label})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',default='final_check01');args=p.parse_args()
    with Job('cpu_check',args.job,reserve_s=120): verify(args.job)
