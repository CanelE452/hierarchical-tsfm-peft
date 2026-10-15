"""Complete synthetic metadata tables/text/four-figure rendering; no science assets."""
from pathlib import Path
import copy
import csv
import json
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import make_methodology_diagnostics_report_v1 as report


def reference(name):return {"path":str(report.HERE/(name+".npz")),"sha256":"a"*64}


def scores(value):
    common={"observed_counts":[48,48],"active_channels":2}
    return {"normalized":{"mse":value,"mae":value**.5,"per_channel_mse":[value,value],"per_channel_mae":[value**.5]*2,**common},
        "physical_per_channel_diagnostic":{"mse":value*2.5,"mae":value**.5*1.5,"per_channel_mse":[value,value*4],"per_channel_mae":[value**.5,value**.5*2],**common}}


def masks(value):return {"all_observed":scores(value),"complete_C":scores(value)}


def entry(value,baseline,raw):
    return {"status":"COMPLETE","scores_all_observed":scores(value),"scores_complete_mask":scores(value),
        "baseline_Bolt_scores":masks(baseline),"RAW_BOLT_scores":masks(raw),"latent_commutator":{"max_abs":.1,"RMS":.02},"C_commutator":{"max_abs":.1,"RMS":.02},"artifact":reference('prediction'),"input_source":reference('input'),"cached_Bolt_source":reference('cache')}


def fixture_values():
    values={key:{"status":"PASS" if key.endswith('audit') else "COMPLETE","resource_terminal":{"status":"COMPLETE","wall_seconds":1.,"process_cpu_seconds":.5,"gpu_process":not key.endswith('audit')}} for key in report.FIXED_SHA}
    controls,quantile,learning=(values[k] for k in ('controls','quantile','learning'))
    controls['states']={};quantile['routes']={};learning['states']={};learning['matched_l7_native_audit']={"path":str(report.HERE/'fixture_space.json'),"sha256":"a"*64}
    space={"new_output_space_diagnostics":{}}
    for si,state in enumerate(report.STATES):
        controls['states'][state]={"controls":{}};learning['states'][state]={"learning":{}}
        space['new_output_space_diagnostics'][state]={"92601":{}}
        for control in report.CONTROLS:
            row={"splits":{},"Q_metadata":{"kind":control},"Q_artifact":reference('Q'),"Q_before":{"determinant":1.},"Q_after":{"determinant":1.},"control_lock":{"path":str(report.HERE/'fixture_lock.json'),"sha256":"a"*64}}
            for split in ('fit','tune','val'):
                baseline={'fit':.12,'tune':.10,'val':.49}[split]+si*.01;raw=.34 if split=='val' else .056
                value=baseline if control in ('IDENTITY','PERMUTATION') else baseline*1.5
                row['splits'][split]={"scores":masks(value),"baseline_Bolt_scores":masks(baseline),"RAW_BOLT_scores":masks(raw),"checks":{"parity":{"pass":True}},"latent_commutator":{"max_abs":.1},"C_commutator":{"max_abs":.2},"transport_future_gap":{"max_abs":1e-7},"transport_C_gap":{"max_abs":1e-7},"complete_target_positions":48,"artifact":reference('fixed'),"independent_artifact":reference('independent'),"actual_F_calls":1,"input_source":reference('input'),"cached_Bolt_source":reference('cache')}
            controls['states'][state]['controls'][control]=row
        for split in ('tune','val'):
            space['new_output_space_diagnostics'][state]['92601'][split]={"minimum_normalized_MSE":.10 if split=='tune' else .30,"RAW_BOLT_scores_complete_mask":scores(.056 if split=='tune' else .34),"complete_target_positions":48}
        for seed in report.SEEDS:
            cps=[]
            for update in (0,100,200,400,600):
                value=.10+si*.01-update*.000005
                cp={**entry(value,.10+si*.01,.056),"update":update,"state":reference(state+str(seed)+str(update)),"tune":{"mse":value,"mae":value**.5},"basis_diagnostics":{"determinant":1.,"orthogonality_max_abs":1e-7},"E_D_center_exact_initial_bytes":True}
                cps.append(cp)
            selected=cps[-1]
            probe={"status":"PASS","optimizer_updates_in_probe":0,"gradient_closure_pass":True,"F_input_path_nonzero_observed":True,"full_theta_gradient_norm":1.,"theta_F_input_chain_norm":.7,"theta_inverse_Q_contribution_norm":.3,"closure_residual_norm":1e-8,"artifact":reference('gradient')}
            learning['states'][state]['learning'][str(seed)]={"checkpoints":cps,"selected_update":600,"selected_tune":selected['tune'],"selected_state":selected['state'],"selected_basis_diagnostics":selected['basis_diagnostics'],"training_updates_completed":600,"post_optimizer_frozen_flag_checks":600,"frozen_state_checkpoint_byte_checks":5,"frozen_bytes_checked_at_CP_and_final_only":True,"independent_per_step_backward_optimizer_or_frozen_bytes_replayed":False,"final_frozen_status":{"trainable_parameter_names":["theta"]},"selection_lock":{"path":str(report.HERE/'fixture_learning_lock.json'),"sha256":"a"*64},"updates_log":{"path":str(report.HERE/'fixture_updates.jsonl'),"sha256":"a"*64},"first_batch_gradient_probe":probe,"global_clipping_fraction":.2,"order_sha256":"a"*64,"FIT":entry(.11,.12,.056),"TUNE":selected,"DEV":entry(.485 if si==0 else .51+si*.01,.49+si*.01,.34)}
    for route in ('RAW_C',)+report.STATES:
        row={"selected_summary":"TAIL_CLAMPED_PROXY","selection_lock":{"path":str(report.HERE/'fixture_summary_lock.json'),"sha256":"a"*64},"splits":{}}
        for split in ('fit','tune','val'):
            median=(.34 if split=='val' else .056) if route=='RAW_C' else (.49 if split=='val' else .10)+report.STATES.index(route)*.01
            summaries={}
            for j,summary in enumerate(report.SUMMARIES):
                difference=0 if j==0 else (-.005*j/2 if route=='RAW_C' or split!='val' else .001*j)
                summaries[summary]={"scores_all_observed":scores(median+difference),"scores_complete_mask":scores(median+difference),"checks":{"pass":True},"artifact":reference('summary')}
            row['splits'][split]={"summaries":summaries,"artifact":reference('quantile'),"full64_quantile_diagnostics":{"crossing_pairs":0},"evaluated48_quantile_diagnostics":{"crossing_pairs":0},"checks":{"median_cache":{"pass":True}},"input_source":reference('input')}
        quantile['routes'][route]=row
    values['space_context']=space
    return values


class MethodologyReportTests(unittest.TestCase):
    def test_full_six_table_row_counts_and_nested_source_values_preserved(self):
        values=fixture_values();tables=report.collect_tables(values)
        self.assertEqual([len(tables[k]) for k in report.TABLES],[45,36,30,18,6,6]);json.dumps(tables,allow_nan=False)
        row=next(r for r in tables['QUANTILE_SUMMARIES'] if r['route']=='RAW_C' and r['split']=='val' and r['selected_on_TUNE'])
        self.assertLess(row['MSE_minus_own_fresh_median'],0);self.assertEqual(row['summary'],'TAIL_CLAMPED_PROXY')
        q=next(r for r in tables['Q_SELECTED'] if r['split']=='val')
        self.assertFalse(q['existing_space_context']['new_geometry_computed']);self.assertFalse(q['existing_space_context']['universal_all_FP32_bound'])
        self.assertTrue(all(r['state_label'] in report.STATES and isinstance(r['state'],dict) for r in tables['Q_CHECKPOINTS']))

    def test_own_TUNE_checkpoint_and_exact_ties_earliest_not_DEV(self):
        values=fixture_values();row=values['learning']['states'][report.STATES[0]]['learning']['92601'];self.assertEqual(report.selected(row)['update'],600)
        for cp in row['checkpoints']:cp['tune']={'mse':.1,'mae':.2}
        row.update(selected_update=0,selected_tune=row['checkpoints'][0]['tune'],selected_state=row['checkpoints'][0]['state'])
        self.assertEqual(report.selected(row)['update'],0)
        row['selected_update']=600
        with self.assertRaises(ValueError):report.selected(row)

    def test_dynamic_Korean_text_keeps_failed_Raw_gap_proxy_and_cost_limits(self):
        values=fixture_values();tables=report.collect_tables(values);text=report.make_text(values,tables,{},report.HERE/'reports/fixture')
        self.assertIn('DEV 2/6개',text);self.assertIn('DEV 조건은 0/6개',text);self.assertIn('마지막 CP600 선정 6/6개',text)
        self.assertIn('3/3개는 자기 fresh median보다 DEV가 악화',text);self.assertIn('새 whole-deployment 비용은 미측정',text)
        self.assertIn('같은 값으로 대체하지 않았다',text);self.assertIn('decoded C 분위구간·CRPS를 주장하지 않는다',text)
        self.assertIn('독립 일반화·보호 TEST 결과가 아니다',text)

    def test_all_four_synthetic_figures_render_png_and_svg_and_tables_portable(self):
        values=fixture_values();tables=report.collect_tables(values)
        with tempfile.TemporaryDirectory(dir=report.HERE) as temp:
            output=Path(temp);paths=report.figures(tables,output);self.assertEqual(len(paths),8)
            for path in paths:
                self.assertGreater(path.stat().st_size,10000)
                if path.suffix=='.png':self.assertEqual(path.read_bytes()[:8],b'\x89PNG\r\n\x1a\n')
                else:self.assertEqual(ET.parse(path).getroot().tag.split('}')[-1],'svg')
            (output/'tables').mkdir()
            for name,rows in tables.items():report.write_csv(output/'tables'/(name+'.csv'),rows,output)
            with (output/'tables/Q_SELECTED.csv').open(encoding='utf-8-sig',newline='') as handle:
                rows=list(csv.DictReader(handle));self.assertEqual(len(rows),18)
                nested=json.loads(rows[0]['selected_state']);self.assertFalse(Path(nested['path']).is_absolute())

    def test_missing_CP_or_changed_frozen_flags_are_not_silently_accepted(self):
        for kind in ('CP','frozen','probe'):
            values=fixture_values();row=values['learning']['states'][report.STATES[0]]['learning']['92601']
            if kind=='CP':row['checkpoints'].pop()
            elif kind=='frozen':row['checkpoints'][0]['E_D_center_exact_initial_bytes']=False
            else:row['first_batch_gradient_probe']['optimizer_updates_in_probe']=1
            with self.subTest(kind=kind),self.assertRaises(ValueError):report.collect_tables(values)

    def test_closed_audit_gate_precedes_metadata_IO_and_existing_output_refused(self):
        with patch.dict(report.FIXED_SHA,learning_audit=None),patch.object(report,'read',side_effect=AssertionError('IO forbidden')):
            with self.assertRaisesRegex(ValueError,'NOT_YET_REGISTERED'):report.load_verified({key:'missing.json' for key in report.FIXED_SHA})
        with tempfile.TemporaryDirectory(dir=report.HERE) as temp:
            here=Path(temp);(here/'reports').mkdir();output=here/'reports/existing';output.mkdir()
            with patch.object(report,'HERE',here),patch.object(report,'load_verified',side_effect=AssertionError('IO forbidden')):
                with self.assertRaisesRegex(ValueError,'FRESH_LOCAL'):report.generate({},output)


if __name__=='__main__':unittest.main()
