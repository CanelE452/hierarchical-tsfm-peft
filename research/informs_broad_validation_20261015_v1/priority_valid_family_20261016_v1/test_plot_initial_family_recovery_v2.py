import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).with_name('plot_initial_family_recovery_v2.py')
spec=importlib.util.spec_from_file_location('initial_family_recovery_plot_under_test',SOURCE)
plot=importlib.util.module_from_spec(spec);spec.loader.exec_module(plot)


def fixture():
    actual=[]
    for role,identifier,k,registered in zip(plot.ROLES,plot.IDS,(21,6),(0,6)):
        actual.append(dict(role=role,run_id=identifier,registered_K=registered,effective_F_width=k,MSE=.7 if k==21 else .6,MAE=.4,
            accuracy_device='cpu',training_device='cpu',CPU_threads=2,observed_labels=43344,V_origins=43,search_max_epoch=3,
            selected_refit_epoch=1,search_updates=1863,refit_updates=752,search_exposures=29769,refit_exposures=12020,
            search_seconds=10.,refit_seconds=7.,training_status='SCREEN_COMPLETE_CONVERGENCE_UNRESOLVED',accuracy_reused=False,
            F_resident_bytes=None,F_resident_bytes_scope='NOT_SEPARATELY_RECORDED',B1_median_seconds=.1,B8_median_seconds=.8))
    rows=[{k:str(v) if v is not None else '' for k,v in r.items()} for r in actual]
    audit=dict(schema='INDEPENDENT_INITIAL_FAMILY_AUDIT_v1',status='PASS',error=None,audit_source_sha256=plot.AUDITOR_SHA,
        input_SHA_before_after_equal=True,whole_MASTER_complete=False,independent_generalization_claimed=False,baseline_comparison_included=False,rows=actual)
    report=dict(schema='INITIAL_FAMILY_REPORT_BINDINGS_v1',status='FROM_ACTUAL_COMPLETE_INITIAL_FAMILY_RECOVERY_AUDIT',builder_source_sha256=plot.BUILDER_SHA,
        auditor_source_sha256=plot.AUDITOR_SHA,input_SHA_before_after_equal=True,source48_preserved=True,whole_MASTER_complete=False,
        new_accuracy_conditions=2,new_cost_conditions=4)
    activity=dict(selected_epoch=1,search_updates=1863,refit_updates=752,search_exposures=29769,refit_exposures=12020,search_fit_end=10482,refit_fit_end=12579,methods=[*plot.IDS,plot.B2_RUN])
    provenance=dict(original_POST_sha256='28911ed451a95ce4af157dd863914f9078f5b0f3f11ab82d0ee5ae2efd6d6080',original_POST_status='PARTIAL_VALIDATIONS_COSTS_PRESERVED',original_false_rows_count=4,original_receipt_mutated=False)
    audit.update(matched_training_activity=activity,recovery_provenance=provenance)
    report.update(matched_training_activity=activity,recovery_provenance=provenance)
    baseaudit=dict(schema='INDEPENDENT_EXISTING_B2_REFERENCE_ACTUAL_VALIDATION_v1',status='PASS',new_training=0,accuracy_reused=True,
        input_SHA_before_after_equal=True,new_T_target_operations=0,whole_MASTER_complete=False,independent_generalization_claimed=False,
        original_reference_fullV_FP64=dict(run_id=plot.B2_RUN,mse=.5,mae=.3,n_observed=43344,full_V_origins=43,
            artifact_sha256=plot.B2_ARTIFACT,manifest_sha256=plot.B2_MANIFEST),costs=[])
    base=[]
    for batch,latency in ((1,.09),(8,.65)):
        for phase in ('timed','stage_probe'):
            base.append(dict(run_id=plot.B2_RUN,dataset='jena',family='B2',seed='92601',C='21',K='6',L='512',H='48',
                accuracy_reused='True',new_training='0',normalized_MSE='0.5',normalized_MAE='0.3',n_origins='43',n_observed='43344',
                device='cpu',CPU_threads='2',worker_RSS_limit_bytes='3221225472',batch_size=str(batch),phase=phase,trials='20',
                median_seconds=str(latency),artifact_sha256=plot.B2_ARTIFACT,accuracy_manifest_sha256=plot.B2_MANIFEST,
                audit_sha256=plot.B2_AUDIT_SHA,F_resident_bytes='NA',F_resident_bytes_scope='NOT_SEPARATELY_RECORDED',model_loading_in_whole_trial_timer='False'))
        baseaudit['costs'].append(dict(batch=batch,summaries=[dict(phase='timed',batch_size=batch,trials=20,median_seconds=latency)]))
    binding=dict(schema='EXISTING_JENA_B2_REFERENCE_REPORT_BINDINGS_v1',status='REPORT_FROM_ACTUAL_AUDIT_PASS',audit_sha256=plot.B2_AUDIT_SHA,
        accuracy_reused=True,new_training=0,input_SHA_before_after_equal=True,whole_MASTER_complete=False,independent_generalization_claimed=False)
    return rows,audit,report,base,baseaudit,binding


class PlotPreparationTests(unittest.TestCase):
    def test_default_no_input_or_lazy_import(self):
        with patch.object(Path,'open',side_effect=AssertionError('input I/O')),patch.object(Path,'read_text',side_effect=AssertionError('input I/O')):
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream):self.assertEqual(plot.main([]),0)
        value=json.loads(stream.getvalue());self.assertEqual(value['input_reads'],0);self.assertEqual(value['actual_render'],0)
        self.assertFalse(any(name in sys.modules for name in ('numpy','torch','matplotlib')))

    def test_exact_three_source_rows_and_medians(self):
        data=plot.prepare_data(*fixture())
        self.assertEqual([r['role'] for r in data['rows']],['B2_REFERENCE',*plot.ROLES])
        self.assertEqual([float(r['B8_median_seconds']) for r in data['rows']],[.65,.8,.8])
        self.assertEqual(data['p90_confirmed'],{1:False,8:False})

    def test_csv_error_not_recomputed_or_tolerated(self):
        args=list(fixture());args[0][0]['MSE']='0.7000000000000001'
        with self.assertRaises(ValueError):plot.prepare_data(*args)

    def test_partial_audit_rejected(self):
        args=list(fixture());args[1]['status']='NOT_COMPLETE'
        with self.assertRaises(ValueError):plot.prepare_data(*args)

    def test_original_partial_cannot_be_hidden_by_recovery_figure(self):
        args=list(fixture());args[1]['recovery_provenance']=dict(args[1]['recovery_provenance'],original_POST_status='COMPLETE')
        with self.assertRaises(ValueError):plot.prepare_data(*args)

    def test_actual_matched_training_counts_are_required(self):
        args=list(fixture());args[1]['matched_training_activity']=dict(args[1]['matched_training_activity'],refit_exposures=12019)
        with self.assertRaises(ValueError):plot.prepare_data(*args)

    def test_wrong_registered_family_order_rejected(self):
        args=list(fixture());args[0].reverse()
        with self.assertRaises(ValueError):plot.prepare_data(*args)

    def test_refit_exposures_must_match_audited_metadata(self):
        args=list(fixture());args[0][1]['refit_exposures']='113'
        with self.assertRaises(ValueError):plot.prepare_data(*args)

    def test_foreign_b2_run_or_cohort_rejected(self):
        for field,value in (('run_id','foreign'),('n_observed','1'),('CPU_threads','4'),('K','21')):
            args=list(fixture());args[3][0][field]=value
            with self.assertRaises(ValueError):plot.prepare_data(*args)

    def test_b2_stage_value_cannot_replace_timed(self):
        args=list(fixture());args[3][1]['phase']='timed'
        with self.assertRaises(ValueError):plot.prepare_data(*args)

    def test_b2_copy_must_match_actual_audit(self):
        args=list(fixture());args[3][2]['median_seconds']='0.651'
        with self.assertRaises(ValueError):plot.prepare_data(*args)

    def test_foreign_b2_report_audit_rejected(self):
        args=list(fixture());args[5]['audit_sha256']='0'*64
        with self.assertRaises(ValueError):plot.prepare_data(*args)

    def test_negative_nonfinite_errors_rejected(self):
        for bad in ('-0.1','nan','inf'):
            args=list(fixture());args[0][0]['MSE']=bad
            with self.assertRaises(ValueError):plot.prepare_data(*args)

    def test_only_twelve_artist_scalars_allowed(self):
        data=plot.prepare_data(*fixture())
        values=[dict(column=k,values=[float(r[k]) for r in data['rows']]) for k in plot.COLUMNS]
        plot.validate_artists(values,data)
        foreign=copy.deepcopy(values);foreign[2]['values'][0]+=.01
        with self.assertRaises(ValueError):plot.validate_artists(foreign,data)
        with self.assertRaises(ValueError):plot.validate_artists(values+[values[0]],data)


if __name__=='__main__':unittest.main()
