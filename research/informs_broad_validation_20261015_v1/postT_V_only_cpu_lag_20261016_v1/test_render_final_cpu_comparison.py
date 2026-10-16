"""Synthetic aggregate rows only; no campaign scores, costs, models or arrays."""
import copy
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent


def load():
    path=HERE/'render_final_cpu_comparison.py'
    spec=importlib.util.spec_from_file_location('tiny_final_cpu_renderer',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class FinalCPUFigures(unittest.TestCase):
    def setUp(self):
        self.assertTrue((HERE/'render_final_cpu_comparison.py').is_file(),'renderer is not implemented')
        self.h=load()

    def rows(self):
        result=[]
        for index,(run_id,label,kind,k) in enumerate(self.h.METHODS):
            neural=kind not in {'RAW','RAW_RIDGE'}
            row={name:'' for name in self.h.FIELDS}
            row.update(run_id=run_id,label=label,kind=kind,K=str(k),C='21',L='512',H='48',V_origins='43',seed='92601',
                dataset='jena',split='V_select',cohort='jena:primary:V_select:stride48',data_sha256='a'*64,scaler_sha256='b'*64,origins_sha256='c'*64,
                device='cpu',threads='2',selected_epoch='1' if neural else '',search_max_epoch='3' if neural else '',
                refit_epochs='1' if neural else '',refit_updates='10' if neural else '0',
                selection_rule='A_stop' if neural else ('NONE_FIXED' if kind=='RAW' else 'ORIGINAL_RIDGE'),
                metric_scope='NORMALIZED_CHANNEL_MACRO',MSE=str(.3+index*.01),MAE=str(.4+index*.01),FP64_rescore_status='PASS',
                replay_status='PASS',replay_device='cpu',replay_origin_count='6',replay_actual_forward='true',
                cost_scope='fresh_process_single_route',cost_timer_scope='physical_pipeline',cost_warmups='10',cost_timed_trials='20',cost_stage_trials='20',
                row_microbatch='128',checkpoint_inputs='false')
            for batch in (1,8):
                prefix='cost_B%d_'%batch
                row.update({prefix+'status':'COMPLETE_PROTOCOL',prefix+'median_seconds':str(.02*batch*(index+1)),
                    prefix+'p90_seconds':str(.025*batch*(index+1)),prefix+'min_seconds':str(.015*batch*(index+1)),
                    prefix+'max_seconds':str(.03*batch*(index+1)),prefix+'parity_status':'PASS' if kind=='RAW' else 'NOT_REQUIRED',prefix+'reason':''})
            result.append(row)
        return result

    def fixture(self,folder,rows=None):
        path=Path(folder)/'SYNTHETIC_AGGREGATE.csv'
        with path.open('x',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=self.h.FIELDS);writer.writeheader();writer.writerows(self.rows() if rows is None else rows)
        return path

    def test_schema_exact_ids_labels_and_duplicate_or_missing_rows(self):
        data=self.rows();self.assertEqual(len(self.h.validate_rows(data)),7)
        for change in ('duplicate','missing','unknown_G','label','schema'):
            rows=copy.deepcopy(data)
            if change=='duplicate':rows[-1]=dict(rows[0])
            elif change=='missing':rows.pop()
            elif change=='unknown_G':rows[2]['run_id']='primary_jena_E1_DIRECT_G_k0_s92601__e3'
            elif change=='label':rows[2]['label']='Different G'
            else:rows[0].pop('MAE')
            with self.subTest(change=change),self.assertRaises(ValueError):self.h.validate_rows(rows)

    def test_nonfinite_negative_metrics_and_invalid_timer_order_are_rejected(self):
        for field,value in (('MSE','nan'),('MAE','inf'),('MSE','-1'),('MAE','-1'),('cost_B1_median_seconds','0'),
            ('cost_B1_p90_seconds','.001'),('cost_B8_max_seconds','-1'),('refit_updates','1.5')):
            rows=self.rows();rows[0][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):self.h.validate_rows(rows)

    def test_mixed_data_scaler_origins_dimensions_device_and_protocol_are_rejected(self):
        for field,value in (('data_sha256','d'*64),('scaler_sha256','e'*64),('origins_sha256','f'*64),('H','12'),('V_origins','42'),
            ('device','cuda'),('threads','4'),('split','A_stop'),('seed','92602'),('cost_warmups','0'),('row_microbatch','32')):
            rows=self.rows();rows[-1][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.h.validate_rows(rows)

    def test_search_and_selected_fresh_refit_remain_distinct_and_not_V_selection(self):
        rows=self.rows();neural=rows[3]
        neural['selected_epoch']='3';neural['refit_epochs']='3'
        parsed=self.h.validate_rows(rows)
        self.assertEqual(parsed[3]['search_max_epoch'],3.)
        self.assertEqual(parsed[3]['selected_epoch'],3.)
        for field,value in (('selection_rule','V_best'),('search_max_epoch','1'),('refit_epochs','1')):
            changed=copy.deepcopy(rows);changed[3][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.h.validate_rows(changed)

    def test_failed_and_unmeasured_costs_are_explicit_missing_not_zero_or_dropped(self):
        rows=self.rows()
        for index,status in ((0,'FAILED'),(1,'NOT_MEASURED')):
            rows[index]['cost_B8_status']=status;rows[index]['cost_B8_reason']='PARITY_FAILED' if index==0 else 'NOT_RUN'
            rows[index]['cost_B8_parity_status']='FAIL' if index==0 else 'NOT_EXECUTED'
            for name in ('median_seconds','p90_seconds','min_seconds','max_seconds'):rows[index]['cost_B8_'+name]=''
        parsed=self.h.validate_rows(rows)
        self.assertEqual(len(parsed),7);self.assertIsNone(parsed[0]['costs'][8]['median_seconds'])
        self.assertEqual(parsed[0]['costs'][8]['status'],'FAILED')
        rows[0]['cost_B8_median_seconds']='0'
        with self.assertRaises(ValueError):self.h.validate_rows(rows)

    @unittest.skipUnless(os.environ.get('INFORMS_RENDER_TOY_SMOKE')=='1','Explicit opt-in after the CPU precision-cost quiet window')
    def test_actual_tiny_render_preserves_source_input_and_writes_all_formats_and_failure_labels(self):
        with tempfile.TemporaryDirectory() as folder:
            rows=self.rows()
            rows[0].update(cost_B8_status='FAILED',cost_B8_reason='PARITY_FAILED',cost_B8_parity_status='FAIL')
            for name in ('median_seconds','p90_seconds','min_seconds','max_seconds'):rows[0]['cost_B8_'+name]=''
            path=self.fixture(folder,rows);blob=path.read_bytes();source=self.h.sha(HERE/'render_final_cpu_comparison.py')
            output=Path(folder)/'figures'
            binding=self.h.render(path,output,self.h.sha(path))
            self.assertEqual(path.read_bytes(),blob);self.assertEqual((output/'INPUT_SNAPSHOT.csv').read_bytes(),blob)
            self.assertEqual(self.h.sha(HERE/'render_final_cpu_comparison.py'),source)
            self.assertEqual(len(list(output.glob('*.svg'))),3);self.assertEqual(len(list(output.glob('*.png'))),3)
            self.assertTrue(all(p.read_bytes().startswith(b'\x89PNG\r\n\x1a\n') for p in output.glob('*.png')))
            self.assertIn('FAILED',(output/'CPU_WHOLE_COST_B1_B8.svg').read_text(encoding='utf-8'))
            self.assertIn('FAILED',(output/'ACCURACY_CPU_COST_SCATTER.svg').read_text(encoding='utf-8'))
            self.assertFalse(binding['speedup_ratios_computed']);self.assertEqual(binding['PNG_dpi'],300)
            self.assertEqual(binding['row_count'],7);self.assertFalse(binding['new_scientific_validation_performed'])
            with self.assertRaises(FileExistsError):self.h.render(path,output,self.h.sha(path))

    def test_changed_sha_and_input_mutation_during_render_cannot_produce_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            path=self.fixture(folder);output=Path(folder)/'bad'
            with self.assertRaises(ValueError):self.h.render(path,output,'0'*64)
            self.assertFalse(output.exists())
            def mutate(rows,out):path.write_bytes(path.read_bytes()+b'\n');return []
            with patch.object(self.h,'draw',side_effect=mutate):
                with self.assertRaises(ValueError):self.h.render(path,Path(folder)/'mutated',self.h.sha(path))
            self.assertFalse((Path(folder)/'mutated/FIGURE_BINDING.json').exists())

    def test_original_symlink_input_and_output_parent_are_rejected_before_resolve(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            for kind in ('input_file','input_parent','output_parent'):
                linked=root/('linked_'+kind)
                csv_path=linked if kind=='input_file' else (linked/'aggregate.csv' if kind=='input_parent' else root/'aggregate.csv')
                output=linked/'figures' if kind=='output_parent' else root/'figures'
                with self.subTest(kind=kind),patch.object(Path,'is_symlink',lambda path:path==linked), \
                     patch.object(Path,'resolve',side_effect=AssertionError('Do not erase the original link before its guard')):
                    with self.assertRaises(ValueError):self.h.render(csv_path,output,'0'*64)
                    if kind!='output_parent':
                        with self.assertRaises(ValueError):self.h.read_rows(csv_path)

    def test_default_fresh_cli_is_stdlib_metadata_no_plot_import_or_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            result=subprocess.run([sys.executable,'-B',str(HERE/'render_final_cpu_comparison.py')],cwd=folder,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            value=json.loads(result.stdout)
            self.assertFalse(value['numpy_imported'] or value['torch_imported'] or value['matplotlib_imported'])
            self.assertEqual(value['files_written'],0);self.assertEqual(list(Path(folder).iterdir()),[])


if __name__=='__main__':unittest.main()
