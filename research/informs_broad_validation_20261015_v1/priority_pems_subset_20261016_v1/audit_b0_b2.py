"""ROOT-triggered metadata/FP64 audit of exactly two original PeMS C32 routes.

Default zero I/O. No model, array, checkpoint, T value or OS query is decoded.
"""
import argparse
import ast
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import time
import types

NAME='priority_pems_subset_20261016_v1'
RAW_G_SHA='d12445d55207626400b3ebfacac5a08924ea442e4fe9ecd511720eabccece90b'
RAW_G_REFERENCE_AUDITOR_SHA='c2d99f40ffd7e5ed5ddff7c327aa3e003a5d06e660687f7fab9e8333c8d2e190'
RAW_G_REFERENCE_BUILDER_SHA='701dca855a385a49190a7583724f388753fa4cad31c7b8f4a815951283253dac'
RUNNER_SHA='caf332f445121ae03249c23023fde916fefb17ab0eb7685dbd9cefe7be566381'
DRAFT_SHA='b255fec55dcdfe49898d024193cbc7f0efe60100679b29e8e1cdd89bb7969693'
PREPARATION='priority_pause_validate_20261016_v1/FOLLOWUP_ROOT_v1/PEMS_B0_B2_DISPATCH_PREP_v1/'
DISPATCH_SHA='04264eeaf8de7df335cfb9caa39cb8c0fa6bd8b031f02bbc77a27ee52a4a7a79'
POST_SHA='675166ae91f07f4e3c5b05c776e779b64f35dffd1953b93814ffc8e85c84dd38'
PREPARATION_SHA='6002c1c0849fc81f66d4ff1dcb7207fceaeefec7ba97ad44d49de9867753ed81'
RUNS={f:'postT_cpu_pems08_E1_SUBSET_'+f+'_k8_s92601_C32_priority_v1' for f in ('B0','B2')}
ORIGINS=tuple(range(12499,14284-12+1,12))
BANDS=((0,3),(3,6),(6,12))
COHORT='pems08:C32_hash:primary:V_select:stride12'
DEADLINE=1792222550


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def counts(family,batches,full=False):
    if family not in RUNS:raise ValueError('Only registered B0/B2 K8 routes')
    total=sum(batches)*8
    return dict(streams=total,physical_streams=total,physical_calls=sum(math.ceil(b*8/128) for b in batches))


def expected_keys():return {(rid,mode,batch) for rid in RUNS.values() for mode,batch in (('fit',None),('rescore',None),('replay',None),('cost',1),('cost',8))}


def validate_backward(proof):
    keys=('trainable_F_input_tensors','F_input_gradient_callbacks','finite_callbacks','nonzero_callbacks')
    if not isinstance(proof,dict) or any(type(proof.get(k)) is not int or proof[k]<0 for k in keys) or proof['trainable_F_input_tensors']<1 or proof['finite_callbacks']<1 or proof['finite_callbacks']!=proof['F_input_gradient_callbacks'] or proof['nonzero_callbacks']>proof['finite_callbacks'] or proof.get('F_parameter_gradients_allowed') is not False or proof.get('first_update_all_gradients_nonzero_required') is not False:
        raise ValueError('Actual finite callbacks through frozen F input required; no invented all-nonzero gradient requirement')


def validate_training(receipt,search,refit):
    family=receipt.get('family');epoch=receipt.get('selected_epoch')
    want=dict(status='COMPLETE_VALID',run_id=RUNS.get(family),seed=92601,actual_C=32,L=512,H=12,registered_latent_k=8,A50_end=8928,A60_end=10713,full_parent_C=170,full_parent_identity_inherited=True,new_T_numeric_values_parsed=0,independent_generalization_claimed=False,old_C170_training_budget_matched=False,PCA_fit_time_separately_recorded=False,artifact_kind='TRAINED_SUBSET_BUNDLE',source_stage_API='original_fit_base._stage')
    if family not in RUNS or epoch not in (0.,.25,1.,3.) or any(receipt.get(k)!=v for k,v in want.items()) or not all(digest(receipt.get(k)) for k in ('artifact_sha256','model_state_sha','refit_scaler_sha','actual_subset_prefix_sha256','data_sha256')):raise ValueError('Exact original trained C32/K8 selected recipe and scaler/artifact lineage required')
    exposures=math.ceil(epoch*10190);updates=math.ceil(exposures/16)
    want=dict(search_updates=1577,search_exposures=25215,refit_updates=updates,refit_exposures=exposures,training_status='UNDERTRAINED_OR_BUDGET_LIMIT' if epoch==3 else 'SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED')
    if any(receipt.get(k)!=v for k,v in want.items()):raise ValueError('Original segmented search and fresh selected refit activity differs')
    for stage,want in ((search,dict(status='SEGMENT_COMPLETE',selected_epoch=epoch,completed_epochs=[0.,.25,1.,3.],train_origins=8405,exposures=25215,updates=1577)),(refit,dict(status='SEGMENT_COMPLETE',selected_epoch=epoch,completed_epochs=[0.]+([epoch] if epoch else []),train_origins=10190,exposures=exposures,updates=updates))):
        if not isinstance(stage,dict) or any(stage.get(k)!=v for k,v in want.items()):raise ValueError('Native search/refit endpoint and activity differs; zero refit is valid')
    for label in ('A50','A60'):
        init=receipt.get('initialization_by_stage',{}).get(label,{})
        if init.get('basis_initialization')!='A_fitted_centered_PCA' or set(init.get('fingerprints',{}))!={'mean','std','observed_counts','basis','pca_center'} or not all(digest(v) for v in init['fingerprints'].values()):raise ValueError('Actual A50/A60 PCA basis/center and normalization SHA metadata required')
    validate_backward(receipt.get('F_backward_observation'))


def validate_foundation_state(value,family,foundation_sha):
    if family not in RUNS or value.get('foundation_instances_created')!=1 or value.get('F_final_state_sha256')!=foundation_sha:raise ValueError('Actual outer frozen F state and single foundation instance required')


def same_cohort(left,right):
    if any(left.get(k)!=right.get(k) for k in ('reference_axis_fingerprints','band_count_binding_sha256','n_observed')) or any(left['fit_receipt'].get(k)!=right['fit_receipt'].get(k) for k in ('data_sha256','refit_scaler_sha')):raise ValueError('Reference target/mask/origin/channel/count/scaler/derived subset must match exactly')
    for batch in ('1','8'):
        if batch in left.get('costs',{}) and batch in right.get('costs',{}):
            if any(left['costs'][batch]['receipt'][k]!=right['costs'][batch]['receipt'][k] for k in ('physical_input_sha256','observation_mask_sha256','origins')):raise ValueError('Actual reused and new physical batch inputs/masks/origins differ')


def operations(public):
    source=Path(public)/NAME/'audit_raw_g.py'
    if sha(source)!=RAW_G_SHA:raise ValueError('Exact frozen Raw/G arithmetic/guard producer required')
    text=source.read_text(encoding='utf-8');tree=ast.parse(text)
    namespace=types.ModuleType('b0_b2_isolated_frozen_audit');namespace.__file__=str(source)
    nodes=[n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom,ast.Assign,ast.FunctionDef))]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source)+'::<isolated-functions>','exec'),namespace.__dict__)
    env=namespace.__dict__;env.update(RUNS=RUNS,PREPARATION=PREPARATION,counts=counts,expected_keys=expected_keys,validate_training=validate_training,validate_foundation_state=validate_foundation_state)
    env['ORCHESTRATION_PINS']={'dispatcher_source':(PREPARATION+'RUN_DISPATCH_DRAFT_v2.ps1',DISPATCH_SHA),'post_source':(PREPARATION+'RUN_POST_DRAFT_v2.ps1',POST_SHA)}
    def adapt(name,replacements):
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name);body=ast.get_source_segment(text,node)
        for old,new in replacements:
            if body.count(old)!=1:raise ValueError('Frozen '+name+' boundary changed: '+old)
            body=body.replace(old,new)
        exec(compile(body,str(source)+'::<explicit-B0-B2-'+name+'>','exec'),env)
    adapt('aggregate_trials',[("(\"CPU_preprocess\", \"host_to_device\", \"F\" if family == \"RAW\" else \"G_or_direct\", \"inverse_and_CPU\")",'("CPU_preprocess", "host_to_device", "P_E", "F", "D", "inverse_and_CPU")'),('four-stage 50-trial','six-stage 50-trial')])
    adapt('validate_validation',[('foundation_instances_created=1 if family == "RAW" else 0','foundation_instances_created=1')])
    adapt('collect_guards',[
        ('mode not in ("fit", "rescore", "replay", "raw_parity", "cost") or ((mode in ("raw_parity", "cost")) != (batch in (1, 8))) or (mode == "raw_parity" and run_id != RUNS["RAW"])','mode not in ("fit", "rescore", "replay", "cost") or ((mode == "cost") != (batch in (1, 8)))'),
        ('ROOT_PEMS_RAW_G_CURRENT_ABSENCE_v1','ROOT_PEMS_B0_B2_CURRENT_ABSENCE_v1')])
    adapt('accuracy_row',[
        (' if family == "DIRECT_G" else [None, None]',''),
        ('plan["foundation_tensor_sha256"] if family == "RAW" else "NO_FOUNDATION"','plan["foundation_tensor_sha256"]'),
        ('basis_sha=None, center_sha=None','basis_sha=receipt["initialization_by_stage"]["A50" if end == 8928 else "A60"]["fingerprints"]["basis"], center_sha=receipt["initialization_by_stage"]["A50" if end == 8928 else "A60"]["fingerprints"]["pca_center"]'),
        ('actual_C=32, H=12, K=0','actual_C=32, H=12, K=8'),
        ('full_C170_or_B2_comparison=False','full_C170_comparison=False, B0_B2_C32_structural_comparison=True')])
    raw_accuracy=env['accuracy_row']
    def checked_accuracy(inputs,helpers,plan,family,records,plan_sha):
        row,stats,identity,proofs=raw_accuracy(inputs,helpers,plan,family,records,plan_sha)
        backward=inputs.json('public',records[(RUNS[family],'fit',None)]['folder']+'/F_BACKWARD_OBSERVATION.json')
        if backward!=row['fit_receipt']['F_backward_observation']:raise ValueError('Actual fit gradient observation and BASE snapshot differ')
        validate_backward(backward);row['latent_k']=8
        return row,stats,identity,proofs
    env['accuracy_row']=checked_accuracy
    adapt('cost_row',[('(1 if family == "RAW" else 0)','1')])
    adapt('bind_native_execution',[
        ('PEMS_RAW_G_TWO_FIT_RESULT_v1','PEMS_B0_B2_TWO_FIT_RESULT_v1'),('PEMS_RAW_G_POST_RESULT_v1','PEMS_B0_B2_POST_RESULT_v1'),
        ('wrapper.get("B0_B2_automatically_started")','wrapper.get("other_conditions_automatically_started")'),
        ('[RUNS["RAW"], RUNS["DIRECT_G"]]','[RUNS["B0"], RUNS["B2"]]'),('PEMS_RAW_G_ADMISSION_v1','PEMS_B0_B2_ADMISSION_v1'),
        ('original_four_conditions_preserved=True, B0_B2_automatically_started=False','original_four_conditions_preserved=True, exact_B0_B2_explicitly_admitted=True, prior_new_training_conditions=1, new_trainable_conditions=2, cumulative_new_training_condition_ceiling=3'),
        ('PEMS_RAW_G_N5_PREDECESSOR_v1','PEMS_B0_B2_RAW_G_PREDECESSOR_v1')])
    raw_bind=env['bind_native_execution']
    def checked_native(inputs,execution,records):
        native,skipped=raw_bind(inputs,execution,records)
        if execution['preparation']['sha256']!=PREPARATION_SHA or any(wrapper.get('exact_B0_B2_explicitly_admitted') is not True or wrapper.get('prior_new_training_conditions')!=1 for wrapper in native.values()) or native['dispatch'].get('planned_trainable_conditions')!=2 or native['dispatch'].get('cumulative_new_training_condition_ceiling')!=3:raise ValueError('Exact original private preparation and separate B0/B2 two-condition admission required')
        return native,skipped
    env['bind_native_execution']=checked_native
    adapt('collect',[
        ('PEMS_RAW_G_ROOT_EXECUTION_BINDINGS_v1','PEMS_B0_B2_ROOT_EXECUTION_BINDINGS_v1'),('for family in ("RAW", "DIRECT_G")','for family in ("B0", "B2")'),
        ('COMPLETE_PEMS_RAW_G_FITS_AND_FULL_V','COMPLETE_PEMS_B0_B2_FITS_AND_FULL_V'),('COMPLETE_PEMS_RAW_G_TEN_POST_STEPS','COMPLETE_PEMS_B0_B2_EIGHT_POST_STEPS'),
        ('sum(r["family"] == "DIRECT_G" for r in rows)','len(rows)')])
    return namespace


def reference_rows(inputs,ops,execution,rows):
    predecessor=ops.bound_json(inputs,execution['predecessor'],'cache');ref=predecessor.get('audit')
    if not rows or ref is None:raise ValueError('No actual matched Raw/G canonical reference is available')
    audit_ref=dict(root='public',relative_path=ref['public_relative_path'],sha256=ref['sha256'])
    original=ops.bound_json(inputs,audit_ref,'public');report_ref=predecessor['report_bindings'];report_ref=dict(root='public',relative_path=report_ref['public_relative_path'],sha256=report_ref['sha256'])
    report=ops.bound_json(inputs,report_ref,'public');source=Path(inputs.roots['public'])/NAME/'build_raw_g_report_v2.py'
    if sha(source)!=RAW_G_REFERENCE_BUILDER_SHA:raise ValueError('Exact actual reference builder source required')
    inputs.capture('public',NAME+'/build_raw_g_report_v2.py',RAW_G_REFERENCE_BUILDER_SHA)
    module=types.ModuleType('b0_b2_reference_builder');module.__file__=str(source);exec(compile(source.read_text(encoding='utf-8'),str(source),'exec'),module.__dict__)
    flat=module.rows_from_audit(original)
    if original.get('audit_source_sha256')!=RAW_G_REFERENCE_AUDITOR_SHA or original.get('ROOT_bound_execution_sha256')!=predecessor['execution_bindings']['sha256'] or original.get('execution_plan_sha256')!=execution['plan_sha256'] or report.get('schema')!='PEMS_RAW_G_REPORT_BINDINGS_v1' or report.get('status')!='FROM_ACTUAL_PEMS_RAW_G_AUDIT_'+original['status'] or report.get('audit_sha256')!=audit_ref['sha256'] or report.get('builder_source_sha256')!=RAW_G_REFERENCE_BUILDER_SHA or report.get('auditor_source_sha256')!=RAW_G_REFERENCE_AUDITOR_SHA or report.get('input_SHA_before_after_equal') is not True or report.get('source48_preserved') is not True or report.get('whole_MASTER_complete') is not False:raise ValueError('Actual predecessor Raw/G audit/report identity must remain exact')
    csv_ref=next(r for r in report['outputs'] if r['relative_path']==NAME+'/RAW_G_RESULTS_v1.csv');actual=inputs.csv('public',csv_ref['relative_path'],csv_ref['sha256'])
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=module.FIELDS,lineterminator='\n');writer.writeheader()
    for row in flat:writer.writerow({k:json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':')) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    if actual!=list(csv.DictReader(io.StringIO(stream.getvalue()))):raise ValueError('Reused Raw/G CSV differs from its bound original aggregate audit')
    refs=[audit_ref,report_ref,dict(root='public',relative_path=csv_ref['relative_path'],sha256=csv_ref['sha256'])]
    for reference in original['rows']:
        same_cohort(rows[0],reference)
        if reference.get('actual_C')!=32 or reference.get('L')!=512 or reference.get('H')!=12 or reference.get('seed')!=92601 or reference.get('full_V_origins')!=148:raise ValueError('Only the same CPU C32/L512/H12/seed/V148 reference is comparable')
    return original['rows'],refs


def execute(public,cache,bindings_path,bindings_sha,audit_sha,output):
    target=Path(output).resolve()
    if target.exists():raise FileExistsError('CreateNew audit output only')
    if sha(__file__)!=audit_sha or not digest(bindings_sha) or time.time()>=DEADLINE:raise ValueError('Exact ROOT actual auditor/bindings source and original deadline required')
    public,cache=Path(public).resolve(),Path(cache).resolve()
    if not target.is_relative_to(public/NAME) or not Path(bindings_path).resolve().is_relative_to(cache):raise ValueError('Audit stays public and ROOT execution binding stays private')
    ops=operations(public);helpers=ops.load_helpers(public);inputs=helpers['inputs'].Inputs(public,cache)
    execution=inputs.json('cache',Path(bindings_path).resolve().relative_to(cache).as_posix(),bindings_sha)
    inputs.capture('public',NAME+'/audit_b0_b2.py',audit_sha);inputs.capture('public',NAME+'/audit_raw_g.py',RAW_G_SHA)
    inputs.capture('public',NAME+'/audit_raw_g_v2.py',RAW_G_REFERENCE_AUDITOR_SHA)
    inputs.capture('public',NAME+'/build_raw_g_report_v2.py',RAW_G_REFERENCE_BUILDER_SHA)
    result={};error=None
    try:
        result=ops.collect(inputs,helpers,execution)
        result.update(reference_rows=[],reference_inputs=[],reference_status='NOT_AVAILABLE_OR_NOT_MATCHED',reference_failures=[])
        try:
            references,refs=reference_rows(inputs,ops,execution,result['rows']);result.update(reference_rows=references,reference_inputs=refs,reference_status='MATCHED_ACTUAL_RAW_G_C32_REFERENCE')
        except Exception as exception:result['reference_failures'].append(dict(reason_class=type(exception).__name__,reason=str(exception) if type(exception) is ValueError else 'Reference metadata not accepted; new valid peers remain independent'))
    except Exception as exception:error=dict(reason_class=type(exception).__name__,failed_audit_check=str(exception) if type(exception) is ValueError else None,reason='Common source/resource/execution evidence rejected; original outputs preserved')
    preserved=inputs.preserved() and all(not inputs.path('public',name).exists() for name in result.get('missing_public_native_paths',[]))
    receipt=dict(result,schema='INDEPENDENT_PEMS_B0_B2_AUDIT_v1',status=result.get('status','NOT_COMPLETE_AUDIT_EVIDENCE_REJECTED') if error is None and preserved else 'NOT_COMPLETE_AUDIT_EVIDENCE_REJECTED',audit_source_sha256=audit_sha,ROOT_bound_execution_sha256=bindings_sha,execution_plan_sha256=execution.get('plan_sha256'),error=error,source48_preserved=error is None and preserved,input_SHA_before_after_equal=preserved,inputs=[dict(root=root,relative_path=name,sha256=d) for (root,name),d in sorted(inputs.files.items())],whole_MASTER_complete=False,independent_generalization_claimed=False,B2_causal_value_or_Bolt_necessity_proven=False,limitations=['One seed and repeated development V; convergence and generalization unresolved.','Callbacks cover search plus refit; nonzero first-update/all-parameter gradient proof is not claimed.','F-only resident bytes and isolated PCA fit time are not recorded; route memory and cold model load have separate scopes.','Reference Raw/G accuracy/cost are reused only after actual canonical metadata/cohort/input matching; no baseline retraining/rescore/cost rerun.','B0 and B2 differ structurally and in selected refit; this is not an isolated P component or equal-compute causal comparison.','Upstream interpolation/preprocessing ancestry remains UNKNOWN.'])
    helpers['runner'].new(target,receipt);return receipt


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--execute',action='store_true');parser.add_argument('--root-triggered',action='store_true')
    for name in ('public-root','cache-root','bindings','bindings-sha','audit-sha','output'):parser.add_argument('--'+name)
    args=parser.parse_args(argv)
    if not args.execute:print(json.dumps(dict(status='PREPARATION_ONLY_NOT_EXECUTED',input_reads=0,files_written=0,actual_experiment_operations=0)));return 0
    if not args.root_triggered or any(getattr(args,k) is None for k in ('public_root','cache_root','bindings','bindings_sha','audit_sha','output')):parser.error('Exact future ROOT actual execution/SHA and new output required')
    value=execute(args.public_root,args.cache_root,args.bindings,args.bindings_sha,args.audit_sha,args.output);print(json.dumps(dict(status=value['status'],error=value['error'])));return 0 if value['status']=='PASS' else 2


if __name__=='__main__':raise SystemExit(main())
