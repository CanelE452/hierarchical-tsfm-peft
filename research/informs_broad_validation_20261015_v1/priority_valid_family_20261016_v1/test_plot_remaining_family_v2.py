import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


SOURCE=Path(__file__).with_name('plot_remaining_family_v2.py')
s=importlib.util.spec_from_file_location('plot_remaining_family',SOURCE)
m=importlib.util.module_from_spec(s)
s.loader.exec_module(m)


class PlotTests(unittest.TestCase):
    def setUp(self):
        self.rows=[]
        self.audit=dict(schema='INDEPENDENT_REMAINING_FAMILY_AUDIT_v1',status='PASS',error=None,audit_source_sha256=m.AUDITOR_SHA,input_SHA_before_after_equal=True,source48_preserved=True,whole_MASTER_complete=False,independent_generalization_claimed=False,rows=[])
        self.report=dict(schema='REMAINING_FAMILY_REPORT_BINDINGS_v1',status='FROM_ACTUAL_COMPLETE_REMAINING_FAMILY_AUDIT',builder_source_sha256=m.BUILDER_SHA,auditor_source_sha256=m.AUDITOR_SHA,input_SHA_before_after_equal=True,source48_preserved=True,whole_MASTER_complete=False,independent_generalization_claimed=False,new_accuracy_conditions=2,new_cost_conditions=4,reused_Jena_reference_rows=3,reused_Beijing_reference_rows=1,comparison_scope='WITHIN_EACH_DATASET_ONLY',cross_dataset_MSE_ranking=False,reference_audits_rerun=False,references=[])
        for i,method in enumerate(m.METHODS):
            row=dict(dataset=method['dataset'],row_scope='REUSED_REFERENCE' if method['reused'] else 'NEW_REMAINING_FAMILY',role=method['role'],run_id=method['run_id'],family=method['family'],K=str(method['registered_k']),registered_K=str(method['registered_k']),effective_F_width=str(method['width']),C=str(21 if method['dataset']=='jena' else 11),L='512',H='48',seed='92601',accuracy_device='cpu',training_device='cpu',CPU_threads='2',V_origins='43',observed_labels=str(43344 if method['dataset']=='jena' else 20539),accuracy_reused=str(method['reused']),cost_reused=str(method['reused']),F_resident_bytes='',F_resident_bytes_scope='NOT_SEPARATELY_RECORDED',selected_refit_epoch='0',refit_updates='0',refit_exposures='0',search_max_epoch='3',training_status='UNDERTRAINED_OR_BUDGET_LIMIT',MSE=str(.11+i*.013),MAE=str(.23+i*.017),B1_median_seconds=str(.031+i*.007),B8_median_seconds=str(.21+i*.012))
            self.rows.append(row)
            if method['reused']:
                self.report['references'].append(copy.deepcopy(row))
            else:
                record=dict(run_id=method['run_id'],mse=row['MSE'],mae=row['MAE'])
                for batch in (1,8):record[f'cost_B{batch}']=dict(summaries=[dict(phase='timed',batch_size=batch,trials=20,median_seconds=row[f'B{batch}_median_seconds'])])
                self.audit['rows'].append(record)

    def prepare(self):return m.prepare_data(self.rows,self.audit,self.report)

    def test_default_has_zero_io_lazy_renderer(self):
        with patch.object(Path,'read_bytes',side_effect=AssertionError('read')),contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(m.main([]),0)
        self.assertIn('PREPARATION_ONLY_NOT_EXECUTED',out.getvalue())
        self.assertFalse(any(k in sys.modules for k in ('numpy','torch','matplotlib')))

    def test_exact_two_datasets_six_rows_and_zero_refit_are_valid(self):
        data=self.prepare()
        self.assertEqual([len(data[d]) for d in ('jena','beijing')],[4,2])
        self.assertEqual(data['jena'][-1]['role'],'ADAPTS_BOLT_PORT_K6')
        self.assertEqual(data['beijing'][-1]['selected_refit_epoch'],'0')

    def test_plot_order_is_explicit_without_modifying_csv_order(self):
        self.rows.reverse()
        old=copy.deepcopy(self.rows)
        data=self.prepare()
        self.assertEqual([r['role'] for r in data['jena']],['B2_K6_REFERENCE','B4_K6_REFERENCE','LORA_R8_C21_REFERENCE','ADAPTS_BOLT_PORT_K6'])
        self.assertEqual(self.rows,old)

    def test_partial_audit_or_source_and_scope_changes_rejected(self):
        for target,key,value in [(self.audit,'status','NOT_COMPLETE'),(self.audit,'source48_preserved',False),(self.report,'builder_source_sha256','0'*64),(self.report,'cross_dataset_MSE_ranking',True)]:
            old=target[key];target[key]=value
            with self.assertRaises(ValueError):self.prepare()
            target[key]=old

    def test_foreign_dataset_duplicate_and_lora_width_rejected(self):
        for key,value in [('dataset','beijing'),('run_id',self.rows[1]['run_id']),('effective_F_width','6')]:
            row=self.rows[2];old=row[key];row[key]=value
            with self.assertRaises(ValueError):self.prepare()
            row[key]=old

    def test_audited_fresh_and_reused_csv_numbers_must_match_exactly(self):
        for i,key in [(3,'MSE'),(5,'B8_median_seconds'),(0,'MAE'),(2,'B1_median_seconds')]:
            old=self.rows[i][key];self.rows[i][key]=str(float(old)+.001)
            with self.assertRaises(ValueError):self.prepare()
            self.rows[i][key]=old

    def test_missing_nonfinite_or_wrong_reuse_flags_refused(self):
        for key,value in [('MSE','nan'),('MAE','-1'),('accuracy_reused','False'),('F_resident_bytes','123')]:
            old=self.rows[0][key];self.rows[0][key]=value
            with self.assertRaises(ValueError):self.prepare()
            self.rows[0][key]=old
        row=self.rows.pop()
        with self.assertRaises(ValueError):self.prepare()
        self.rows.append(row)

    def test_artist_binding_exact_eight_panels_twentyfour_scalars(self):
        data=self.prepare()
        artists=[dict(dataset=d,column=c,run_ids=[r['run_id'] for r in data[d]],values=[float(r[c]) for r in data[d]]) for d in ('jena','beijing') for c in m.COLUMNS]
        m.validate_artists(artists,data)
        self.assertEqual(sum(len(a['values']) for a in artists),24)
        artists[-1]['values'][0]/=8
        with self.assertRaises(ValueError):m.validate_artists(artists,data)

    def test_existing_output_refused_before_report_or_renderer(self):
        with patch.object(Path,'exists',return_value=True),patch.object(m,'module',side_effect=AssertionError('input helper')):
            with self.assertRaises(FileExistsError):m.execute('public','cache','s','r','audit','a','csv')

    def test_v2_byte_diff_only_binds_new_builder_and_own_filename(self):
        import hashlib
        original=SOURCE.with_name('plot_remaining_family.py').read_bytes()
        self.assertEqual(hashlib.sha256(original).hexdigest(),'4da404549722734e482375eedb416860acb686c429c0602f18d7b097a6c85fba')
        builder=SOURCE.with_name('build_remaining_family_report_v2.py')
        self.assertEqual(hashlib.sha256(builder.read_bytes()).hexdigest(),m.BUILDER_SHA)
        expected=original.replace(b'67d19ed46ee94192de20303aeb8480cb69641fdaf9246c507a5fccf68d12e35d',m.BUILDER_SHA.encode()).replace(b'build_remaining_family_report.py',b'build_remaining_family_report_v2.py').replace(b'plot_remaining_family.py',b'plot_remaining_family_v2.py')
        self.assertEqual(SOURCE.read_bytes(),expected)

    def test_pure_fake_axes_support_four_and_two_bars_and_zero_origins(self):
        class Bar:
            def __init__(self,height):self.height=height
            def get_height(self):return self.height
            def set_hatch(self,value):self.hatch=value
        class Axis:
            def bar(self,positions,values,**kwargs):
                self.count=len(list(positions));self.bars=[Bar(v) for v in values]
                self.assert_lengths=(self.count==len(values)==len(kwargs['color']))
                return self.bars
            def set_ylim(self,bottom,top):self.bottom=bottom
            def __getattr__(self,name):return lambda *args,**kwargs:None
        axes=[[Axis() for _ in range(4)] for _ in range(2)]
        artists=m.draw_panels(axes,self.prepare())
        self.assertEqual([a.count for row in axes for a in row],[4]*4+[2]*4)
        self.assertTrue(all(a.assert_lengths and a.bottom==0 for row in axes for a in row))
        self.assertEqual(sum(len(a['values']) for a in artists),24)


if __name__=='__main__':unittest.main()
