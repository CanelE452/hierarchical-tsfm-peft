"""Small synthetic quantile/summary/mask/selection audit fixtures only."""
from pathlib import Path
import ast
import copy
import json
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

import audit_quantile_point_summary_v1 as audit
import run_quantile_point_summary_v1 as producer
from test_run_quantile_point_summary_v1 import fixture

audit.initialize_numpy()


def cell(raw_route=False):
    model,bolt,raw,cached,frozen=fixture(raw_route)
    outputs,calls=producer.infer_full_quantiles(model,bolt,raw["history"],lambda x:None)
    route="RAW_C" if raw_route else "RECON_CP0"
    row,independent=producer.evaluate_arrays(outputs,raw,cached,frozen,route)
    row.update(canonical_batch_size=32,stored_horizon=64,evaluated_horizon=48,actual_full_F_calls=calls)
    full={key:outputs[key] for key in ("latent_history","full_quantiles_FP32")};full["origins"]=raw["origins"].copy()
    candidates={s:{"prediction_FP32":outputs[s+"_prediction"],"latent_future_FP32":outputs[s+"_future"],"origins":raw["origins"].copy()} for s in audit.SUMMARIES}
    return row,full,candidates,independent,raw,cached,frozen,route


def selection_fixture():
    result={"all_4_locks_before_DEV":True,"resolved_selection_locks":4,"routes":{}}
    registry={"status":"ALL_FOUR_RESOLVED_BEFORE_DEV","DEV_opened_before_seal":False,"sealed_unix":20.,"conditions":[]};locks={}
    for route in audit.ROUTES:
        scores=[.1,.2,.3] if route=="RAW_C" else [.3,.2,.1]
        if route=="E_ONLY_s92601":scores=[.1,.1,.1]
        menu=dict(zip(audit.SUMMARIES,scores));winner=min(audit.SUMMARIES,key=lambda s:(menu[s],audit.SUMMARIES.index(s)))
        binding={"path":route+"_lock","sha256":"a"*64}
        entries={s:{"scores_all_observed":{"normalized":{"mse":mse}}} for s,mse in menu.items()}
        development={s:{"selected":s==winner,"nonselected_DEV_diagnostic_only":s!=winner} for s in audit.SUMMARIES}
        row={"status":"COMPLETE","selected_summary":winner,"selected_TUNE_MSE":menu[winner],"selection_lock":binding,
            "splits":{"tune":{"summaries":entries},"val":{"summaries":development}}}
        result['routes'][route]=row;registry['conditions'].append({"route":route,"status":"COMPLETE","selection_lock":binding})
        locks[binding['path']]={"route":route,"status":"COMPLETE","DEV_opened_before_lock":False,"sealed_unix":10.,"TUNE_menu":menu,"selected_summary":winner,"selected_TUNE_MSE":menu[winner]}
    return result,registry,locks


class QuantileIndependentAuditTests(unittest.TestCase):
    def test_raw_and_compressed_full64_nine_axes_point_sums_decoder_and_masks(self):
        for raw_route in (False,True):
            with self.subTest(raw_route=raw_route):
                values=cell(raw_route);result=audit.check_cell(*values)
                self.assertEqual(result['score_fields_checked'],72);self.assertEqual(set(result['summaries']),set(audit.SUMMARIES))
                self.assertEqual(result['full64_quantile_diagnostics']['shape'][1:], [64,6 if raw_route else 4,9])
                self.assertFalse(result['F_inference_independently_replayed']);json.dumps(result,allow_nan=False)

    def test_independent_nine_weights_are_fixed_proxies_and_median_is_exact_axis4(self):
        full=np.arange(9,dtype=np.float32).reshape(1,1,1,9)
        self.assertEqual(float(audit.summary(full,'MEDIAN')[0,0,0]),4.)
        np.testing.assert_allclose(audit.summary(full,'NINE_MEAN_PROXY'),[[[4.]]],atol=1e-6,rtol=0)
        expected=sum(float(v)*w for v,w in zip(range(9),[.15,.1,.1,.1,.1,.1,.1,.1,.15]))
        np.testing.assert_allclose(audit.summary(full,'TAIL_CLAMPED_PROXY'),[[[expected]]],atol=1e-6,rtol=0)
        with self.assertRaises(ValueError):audit.summary(full,'TRUE_MEAN')

    def test_crossings_include_unsorted_pairs_ties_and_all64_but_evaluation_only48(self):
        full=np.broadcast_to(np.arange(9,dtype=np.float32),(2,64,3,9)).copy();full[0,63,0,1]=-2;full[1,0,0,1]=0
        diagnostic=audit.crossing(full)
        self.assertEqual(diagnostic['crossing_pairs'],1);self.assertEqual(diagnostic['crossing_pairs_by_horizon'][63],1)
        self.assertEqual(diagnostic['maximum_adjacent_crossing'],2.);self.assertFalse(diagnostic['quantiles_sorted_after_inference'])
        self.assertEqual(audit.crossing(full[:,:48])['crossing_pairs'],0)
        changed=copy.deepcopy(diagnostic);changed['finite_nine_quantiles_identify_conditional_mean']=True
        with self.assertRaises(ValueError):audit.check_diagnostic(changed,full)

    def test_changed_layout_median_cache_summary_C_or_saved_independent_value_rejected(self):
        for kind in ('layout','baseline','future','C','independent','crossing'):
            values=copy.deepcopy(cell())
            if kind=='layout':values[1]['full_quantiles_FP32']=values[1]['full_quantiles_FP32'].transpose(0,2,1,3).copy()
            elif kind=='baseline':values[5]['prediction'][0,0,0]+=.1
            elif kind=='future':values[2]['TAIL_CLAMPED_PROXY']['latent_future_FP32'][0,0,0]+=.1
            elif kind=='C':values[2]['NINE_MEAN_PROXY']['prediction_FP32'][0,0,0]+=.1
            elif kind=='independent':values[3]['TAIL_CLAMPED_PROXY_NumPy_future'][0,0,0]+=.1
            else:values[0]['full64_quantile_diagnostics']['crossing_pairs']+=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):audit.check_cell(*values)

    def test_score_mask_origin_count_and_tolerances_cannot_be_relabelled(self):
        for kind in ('score','mask','origin','horizon','count','tolerance'):
            values=copy.deepcopy(cell(True))
            if kind=='score':values[0]['summaries']['MEDIAN']['scores_all_observed']['normalized']['mse']+=.1
            elif kind=='mask':values[4]['observed'][0,0,0]=True
            elif kind=='origin':values[2]['MEDIAN']['origins'][0]+=1
            elif kind=='horizon':values[0]['evaluated_horizon']=64
            elif kind=='count':values[0]['complete_target_positions']+=1
            else:values[0]['checks']['median_future_vs_accepted_cache']['atol']=.1
            with self.subTest(kind=kind),self.assertRaises(ValueError):audit.check_cell(*values)

    def test_unobserved_nan_inf_are_not_labels_and_observed_nonfinite_fails(self):
        values=cell();reference=audit.check_cell(*values);values[4]['target'][~values[4]['observed']]=np.inf
        self.assertEqual(reference,audit.check_cell(*values));values[4]['target'][values[4]['observed']]=np.nan
        with self.assertRaises(ValueError):audit.check_cell(*values)

    def test_same_raw_menu_own_TUNE_ties_and_nonselected_DEV_diagnostics(self):
        result,registry,locks=selection_fixture();verified=audit.check_selection(result,registry,lambda item:locks[item['path']])
        self.assertEqual(verified['RAW_C']['selected_summary'],'MEDIAN');self.assertEqual(verified['RECON_CP0']['selected_summary'],'TAIL_CLAMPED_PROXY')
        self.assertEqual(verified['E_ONLY_s92601']['selected_summary'],'MEDIAN');json.dumps(verified,allow_nan=False)
        for kind in ('raw_missing','duplicate','early_DEV','menu','reselect','DEVflag','late_lock'):
            r,g,l=copy.deepcopy((result,registry,locks))
            if kind=='raw_missing':g['conditions'].pop(0)
            elif kind=='duplicate':g['conditions'][-1]=copy.deepcopy(g['conditions'][0])
            elif kind=='early_DEV':r['all_4_locks_before_DEV']=False
            elif kind=='menu':l['RAW_C_lock']['TUNE_menu']['MEDIAN']+=.1
            elif kind=='reselect':r['routes']['RAW_C']['selected_summary']='TAIL_CLAMPED_PROXY'
            elif kind=='DEVflag':r['routes']['RAW_C']['splits']['val']['summaries']['NINE_MEAN_PROXY']['selected']=True
            else:l['RAW_C_lock']['sealed_unix']=21.
            with self.subTest(kind=kind),self.assertRaises(ValueError):audit.check_selection(r,g,lambda item:l[item['path']])

    def test_closed_actual_gate_precedes_input_IO_and_no_scientific_import_on_import(self):
        with patch.object(audit,'RESULT_SHA',None),patch.object(audit.base,'file_record',side_effect=AssertionError('IO forbidden')):
            with self.assertRaisesRegex(ValueError,'NOT_REGISTERED'):audit.audit_result('missing',None)
        tree=ast.parse(Path(audit.__file__).read_text(encoding='utf-8'))
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertNotIn('run_quantile_point_summary_v1',imports);self.assertNotIn('quantile_point_summary_v1',imports)
        command='import sys;sys.path.insert(0,'+repr(str(audit.HERE))+');import audit_quantile_point_summary_v1;assert "numpy" not in sys.modules and "torch" not in sys.modules;print("METADATA_IMPORT_PASS")'
        completed=subprocess.run([sys.executable,'-c',command],capture_output=True,text=True,check=True);self.assertIn('METADATA_IMPORT_PASS',completed.stdout)


if __name__=='__main__':unittest.main()
