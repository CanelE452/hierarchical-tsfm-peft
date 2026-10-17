"""Invented metadata/statistic/trial fixtures; no actual result access."""
import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).with_name('audit_b0_b2.py')


def load():
    spec=importlib.util.spec_from_file_location('b0b2_audit_fixture',SOURCE)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def training(m,epoch):
    exposures=__import__('math').ceil(epoch*10190)
    init={label:dict(basis_initialization='A_fitted_centered_PCA',fingerprints={k:'a'*64 for k in ('mean','std','observed_counts','basis','pca_center')}) for label in ('A50','A60')}
    proof=dict(trainable_F_input_tensors=2,F_input_gradient_callbacks=2,finite_callbacks=2,nonzero_callbacks=0,F_parameter_gradients_allowed=False,first_update_all_gradients_nonzero_required=False)
    receipt=dict(status='COMPLETE_VALID',run_id=m.RUNS['B0'],family='B0',seed=92601,actual_C=32,L=512,H=12,registered_latent_k=8,A50_end=8928,A60_end=10713,full_parent_C=170,full_parent_identity_inherited=True,new_T_numeric_values_parsed=0,independent_generalization_claimed=False,old_C170_training_budget_matched=False,PCA_fit_time_separately_recorded=False,selected_epoch=epoch,search_updates=1577,search_exposures=25215,refit_updates=__import__('math').ceil(exposures/16),refit_exposures=exposures,artifact_kind='TRAINED_SUBSET_BUNDLE',source_stage_API='original_fit_base._stage',training_status='UNDERTRAINED_OR_BUDGET_LIMIT' if epoch==3 else 'SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED',initialization_by_stage=init,F_backward_observation=proof)
    receipt.update({k:'b'*64 for k in ('artifact_sha256','model_state_sha','refit_scaler_sha','actual_subset_prefix_sha256','data_sha256')})
    stages=[dict(status='SEGMENT_COMPLETE',selected_epoch=epoch,completed_epochs=[0.,.25,1.,3.],train_origins=8405,exposures=25215,updates=1577),dict(status='SEGMENT_COMPLETE',selected_epoch=epoch,completed_epochs=[0.]+([epoch] if epoch else []),train_origins=10190,exposures=exposures,updates=receipt['refit_updates'])]
    return receipt,*stages


def trials(m,batch):
    rows=[]
    for phase,count in (('warmup',10),('timed',20),('stage_probe',20)):
        for i in range(count):
            row=dict(run_id=m.RUNS['B2'],phase=phase,trial=str(i),order_position='0',batch_size=str(batch),whole_seconds=str(.01+i*.001),F_logical_streams=str(8*batch),F_physical_streams=str(8*batch),F_calls='1',prediction_sha256='c'*64,row_microbatch='128',stage_instrumentation=str(phase=='stage_probe'))
            row.update({'stage_'+key+'_seconds':'0.001' if phase=='stage_probe' else '' for key in ('CPU_preprocess','host_to_device','P_E','F','D','inverse_and_CPU')})
            rows.append(row)
    return rows


class Tests(unittest.TestCase):
    def setUp(self):
        if not SOURCE.exists():self.skipTest('implementation pending')
        self.m=load()

    def test_default_no_input_or_scientific_import(self):
        with patch.object(Path,'open',side_effect=AssertionError('input')),contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(self.m.main([]),0)
        self.assertIn('PREPARATION_ONLY_NOT_EXECUTED',out.getvalue())
        self.assertFalse(any(k in sys.modules for k in ('numpy','torch','matplotlib')))

    def test_registered_width_and_exact_ten_keys_have_no_Raw_parity(self):
        m=self.m;self.assertEqual(m.counts('B0',[4]*37),dict(streams=1184,physical_streams=1184,physical_calls=37))
        self.assertEqual(m.counts('B2',[4,2]),dict(streams=48,physical_streams=48,physical_calls=2))
        self.assertEqual(m.counts('B2',[8]),dict(streams=64,physical_streams=64,physical_calls=1))
        self.assertEqual(len(m.expected_keys()),10)
        with self.assertRaises(ValueError):m.counts('RAW',[1])

    def test_actual_selected_zero_quarter_one_three_and_PCA_hashes(self):
        for epoch in (0.,.25,1.,3.):self.m.validate_training(*training(self.m,epoch))
        receipt,search,refit=training(self.m,0)
        self.assertNotIn('epochs',receipt)
        receipt['initialization_by_stage']['A50']['fingerprints']['basis']=None
        with self.assertRaises(ValueError):self.m.validate_training(receipt,search,refit)

    def test_finite_input_gradients_do_not_require_first_nonzero_or_claim_parameter_gradient(self):
        receipt,search,refit=training(self.m,1)
        self.m.validate_training(receipt,search,refit)
        for key,value in (('finite_callbacks',0),('F_parameter_gradients_allowed',True),('first_update_all_gradients_nonzero_required',True)):
            bad=copy.deepcopy(receipt);bad['F_backward_observation'][key]=value
            with self.assertRaises(ValueError):self.m.validate_training(bad,search,refit)
        refit['updates']+=1
        with self.assertRaises(ValueError):self.m.validate_training(receipt,search,refit)

    def test_sufficient_statistics_are_channel_macro_and_bind_actual_observed_count(self):
        m=self.m;ops=m.operations(SOURCE.parent.parent);channels=[str(i) for i in range(32)];rows=[]
        for o in m.ORIGINS:
            for c in channels:
                for start,end in m.BANDS:
                    n=2 if c=='0' else 1
                    rows.append(dict(run_id=m.RUNS['B0'],split='V_select',cohort=m.COHORT,origin=str(o),channel=c,horizon_start=str(start),horizon_end=str(end),sse=str(n*(int(c)+1)),sae=str(n*(int(c)+1)/2),count=str(n)))
        result=ops.aggregate_statistics(rows,m.RUNS['B0'],channels,148*3*33)
        self.assertEqual(result['mse'],16.5);self.assertEqual(result['mae'],8.25);self.assertEqual(result['public_stats_rows'],14208)
        with self.assertRaises(ValueError):ops.aggregate_statistics(rows,m.RUNS['B0'],channels,56832)
        with self.assertRaises(ValueError):ops.aggregate_statistics(rows+[rows[0]],m.RUNS['B0'],channels,148*3*33)

    def test_native_six_stage_cost_and_derived_counter_scope(self):
        m=self.m;ops=m.operations(SOURCE.parent.parent);rows=trials(m,8)
        result=ops.aggregate_trials(rows,m.RUNS['B2'],'B2',8)
        self.assertEqual((result['trial_rows'],result['F_logical_streams'],result['F_calls']),(50,3200,50))
        self.assertEqual(len(result['actual_stage_fields']),6)
        self.assertEqual(result['summaries'][0]['median_seconds'],.019500000000000003)
        bad=copy.deepcopy(rows);bad[0].pop('stage_P_E_seconds')
        with self.assertRaises(ValueError):ops.aggregate_trials(bad,m.RUNS['B2'],'B2',8)
        bad=copy.deepcopy(rows);bad[0]['F_calls']='2'
        with self.assertRaises(ValueError):ops.aggregate_trials(bad,m.RUNS['B2'],'B2',8)

    def test_frozen_F_evidence_comes_from_actual_outer(self):
        value=dict(foundation_instances_created=1,F_final_state_sha256='a'*64)
        self.m.validate_foundation_state(value,'B2','a'*64)
        value['foundation_instances_created']=0
        with self.assertRaises(ValueError):self.m.validate_foundation_state(value,'B2','a'*64)

    def test_actual_rescore_has_no_F_requirement_but_outer_CPU6_has_one(self):
        m=self.m;ops=m.operations(SOURCE.parent.parent);identity=dict(run_id=m.RUNS['B0'],artifact_sha256='a'*64)
        value=dict(identity,status='PASS_INDEPENDENT_FULL_V_FP64',full_V_origins=148,actual_C=32,H=12,n_observed=123,public_stats_rows=14208,saved_predictions_rescored=True,new_model_inference=False,protected_T_target_operations=0,torch_imported=False,new_T_numeric_values_parsed=0)
        ops.validate_validation(value,'rescore','B0',identity,123)
        value=dict(identity,status='PASS_FRESH_CPU6_REPLAY',executed_now=True,microbatches=[4,2],replay_selection_seed=92601,F_counters=m.counts('B0',[4,2]),atol=1e-5,rtol=1e-4,full_V_reinference_claimed=False,foundation_instances_created=1,origins=[m.ORIGINS[i] for i in (0,1,2,74,146,147)],new_T_numeric_values_parsed=0)
        ops.validate_validation(value,'replay','B0',identity,123)
        value['F_counters']['streams']=192
        with self.assertRaises(ValueError):ops.validate_validation(value,'replay','B0',identity,123)

    def test_strong_guard_requires_effective_limits_and_current_exact_owner_absence(self):
        m=self.m;ops=m.operations(SOURCE.parent.parent);owner=dict(pid=12345,create_time=1.)
        with __import__('tempfile').TemporaryDirectory() as temporary:
            root=Path(temporary)
            process=dict(id='invented',command=['python','invented.py'],cwd=str(root),max_seconds=10)
            spec=dict(schema='CPU_GROUP_SUPERVISOR_v1',deadline_unix=m.DEADLINE,max_each_RSS=3221225472,max_group_RSS=3221225472,poll_seconds=1,terminate_grace_seconds=1,kill_grace_seconds=1,asset_root=str(root/'assets'),cache_root=str(root/'cache'),output_root=str(root/'cache'/'guard'),processes=[process])
            child=dict(id='invented',returncode=0,start_unix=2.,end_unix=3.,observed_tree_join_confirmed=True,process_tree_errors=[],remaining_owned_processes=[],owned_processes=[owner],root=owner,sampled_peak_RSS_bytes=123)
            guard=dict(schema='CPU_GROUP_SUPERVISOR_RESULT_v1',status='COMPLETE',helper_sha256=ops.GUARD_SHA,helper_unchanged=True,spec_file_sha256='a'*64,supervisor_failure_class=None,remaining_owned_processes=[],processes=[child],effective_spec=copy.deepcopy(spec),sampled_group_peak_RSS_bytes=123)
            closure=dict(schema='ROOT_OPAQUE_METADATA_OWNER_ABSENCE_v2',status='VERIFIED_STRONG_OBSERVED_TREE_CLOSURE_AND_CURRENT_ABSENCE',guard_sha256='b'*64,spec_sha256='a'*64,actual_PID_creation_time_check=True,remaining_exact_owners=[],errors=[],absent_identities=[owner],checked_unix=4.)
            values=dict(spec=spec,guard=guard,closure=closure)
            class Inputs:
                def json(self,root,name,digest):return values[name]
            binding={k:dict(root='cache',relative_path=k,sha256=v*64) for k,v in (('spec','a'),('guard','b'),('closure','c'))};binding['workers']=[dict(id='invented')]
            with patch.object(ops.time,'time',return_value=5.):
                ops.closed_binding(Inputs(),binding)
                closure['absent_identities']=[]
                with self.assertRaises(ValueError):ops.closed_binding(Inputs(),binding)
                closure['absent_identities']=[owner];guard['effective_spec']['max_each_RSS']=6442450944
                with self.assertRaises(ValueError):ops.closed_binding(Inputs(),binding)

    def test_same_C32_reference_requires_scaler_axes_count_and_batch_inputs(self):
        fresh=dict(fit_receipt=dict(data_sha256='a',refit_scaler_sha='b'),reference_axis_fingerprints=dict(observed='c'),band_count_binding_sha256='d',n_observed=42,costs={})
        reference=copy.deepcopy(fresh)
        self.m.same_cohort(fresh,reference)
        reference['fit_receipt']['refit_scaler_sha']='foreign'
        with self.assertRaises(ValueError):self.m.same_cohort(fresh,reference)

    def test_existing_output_rejected_before_any_helper_or_result_read(self):
        with patch.object(Path,'exists',return_value=True),patch.object(self.m,'operations',side_effect=AssertionError('inputs')):
            with self.assertRaises(FileExistsError):self.m.execute('public','cache','bindings','a'*64,'b'*64,'output.json')

    def test_actual_auditor_and_frozen_helper_are_both_mandatory_inputs_even_on_rejection(self):
        m=self.m;captured=[];written=[]
        class Inputs:
            def __init__(self):self.files={}
            def json(self,*args):return dict(plan_sha256='c'*64)
            def capture(self,root,name,digest):captured.append((root,name,digest));self.files[root,name]=digest
            def preserved(self):return True
        ops=types.SimpleNamespace(load_helpers=lambda public:dict(inputs=types.SimpleNamespace(Inputs=lambda public,cache:Inputs()),runner=types.SimpleNamespace(new=lambda target,receipt:written.append(receipt))))
        def rejected(*args):raise ValueError('SYNTHETIC_COMMON_EVIDENCE_REJECTED')
        ops.collect=rejected
        public=Path('synthetic_public').resolve();cache=Path('synthetic_cache').resolve()
        with patch.object(Path,'exists',return_value=False),patch.object(m,'sha',return_value='b'*64),patch.object(m.time,'time',return_value=0),patch.object(m,'operations',return_value=ops):
            receipt=m.execute(public,cache,cache/'binding.json','a'*64,'b'*64,public/m.NAME/'toy_audit.json')
        self.assertEqual(receipt['status'],'NOT_COMPLETE_AUDIT_EVIDENCE_REJECTED');self.assertEqual(len(written),1)
        self.assertIn(('public',m.NAME+'/audit_raw_g.py',m.RAW_G_SHA),captured)
        self.assertIn(('public',m.NAME+'/audit_raw_g_v2.py',m.RAW_G_REFERENCE_AUDITOR_SHA),captured)
        self.assertIn(('public',m.NAME+'/build_raw_g_report_v2.py',m.RAW_G_REFERENCE_BUILDER_SHA),captured)
        self.assertEqual(len(receipt['inputs']),len(captured))


class Exists(unittest.TestCase):
    def test_implementation_exists(self):self.assertTrue(SOURCE.exists(),'RED: source missing')


if __name__=='__main__':unittest.main()
