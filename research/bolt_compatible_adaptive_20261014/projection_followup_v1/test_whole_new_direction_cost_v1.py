"""Synthetic CPU route/timer provenance checks; no model, GPU or scientific cohort."""
from pathlib import Path
import ast
import copy
import json
import tempfile
import unittest

import numpy as np
import whole_new_direction_cost_v1 as cost


def fixture(dataset):
    rng=np.random.default_rng(481);physical=rng.normal(size=(8,512,3)).astype(np.float32);observed=np.ones_like(physical,dtype=bool)
    observed[:,:3,0]=False;physical[~observed]=np.nan
    mean=np.array([1.,2.,3.],np.float32);std=np.array([2.,3.,4.],np.float32);origins=np.arange(8,dtype=np.int64)+512
    normalized=cost.preprocess(physical,observed,mean,std);counts={route:0 for route in cost.routes_for(dataset)}
    def callback(route):
        def predict(history,mask):
            counts[route]+=1
            return np.broadcast_to(history[:,-1:, :],(len(history),48,history.shape[-1])).copy()
        return predict
    predictors={route:callback(route) for route in counts};reference=np.broadcast_to(normalized[:,-1:],(8,48,3)).copy()
    references={route:reference for route in counts};metadata={route:{"synthetic_route":route} for route in counts}
    return physical,observed,mean,std,origins,predictors,references,metadata,counts


def memory():return {"RSS_bytes":1000,"allocated_VRAM_bytes":0,"reserved_VRAM_bytes":0,"max_allocated_VRAM_bytes":0,"max_reserved_VRAM_bytes":0,"scope":"Synthetic CPU only"}


class WholeNewDirectionCostTests(unittest.TestCase):
    def test_exact_dataset_specific_nineteen_routes_and_matched_denominators(self):
        for dataset,k in cost.DATASETS.items():
            routes=cost.routes_for(dataset)
            self.assertEqual(len(routes),19);self.assertEqual(len(set(routes)),19)
            self.assertEqual(sum(r.endswith('_BOLT') for r in routes),5)
            self.assertEqual(sum('_NATIVE_' in r for r in routes),8);self.assertEqual(sum('RIDGE' in r for r in routes),6)
            self.assertIn('PCA_K'+str(k)+'_BOLT',routes);self.assertIn('CROSS_MOMENT_K'+str(2*k)+'_RIDGE_HISTORY512',routes)
            for route in routes:self.assertIn(cost.matched_bolt(route),routes)
        with self.assertRaises(ValueError):cost.routes_for('unregistered')

    def test_five_rotations_have_five_distinct_positions_and_are_not_nineteen_latin(self):
        for dataset in cost.DATASETS:
            routes=cost.routes_for(dataset);order=cost.balanced_schedule(dataset)
            self.assertEqual(len(order),5);self.assertEqual(order,cost.balanced_schedule(dataset))
            for row in order:self.assertEqual(set(row),set(routes))
            for route in routes:self.assertEqual(len({r.index(route) for r in order}),5)

    def test_canonical_three_statement_timed_boundary_and_imported_math_unchanged(self):
        def timer(text):
            tree=ast.parse(text);function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='measure_cost')
            return [ast.dump(node) for node in next(n for n in ast.walk(function) if isinstance(n,ast.Try) and len(n.body)==3 and 'perf_end_ns, unix_end_ns' in ast.unparse(n)).body]
        parent=(cost.HERE/'whole_f_input_gradient_cost_v1.py').read_text(encoding='utf-8');source=Path(cost.__file__).read_text(encoding='utf-8')
        self.assertEqual(timer(source),timer(parent))
        self.assertEqual(cost.NORM_ATOL,1e-5);self.assertEqual(cost.NORM_RTOL,2e-6);self.assertEqual(cost.PHYSICAL_ATOL,2e-3);self.assertEqual(cost.PHYSICAL_RTOL,2e-5)
        self.assertIn('internal finite/shape guards included',source);self.assertIn('external meter guards',source)

    def test_two_synthetic_datasets_have_76_cells_380_integer_ns_samples_and_both_ratios(self):
        results=[]
        with tempfile.TemporaryDirectory(dir=cost.HERE) as temp:
            for dataset in cost.DATASETS:
                values=fixture(dataset);r=cost.measure_cost(dataset,*values[:8],Path(temp)/dataset,lambda:None,lambda:None,memory)
                self.assertEqual(r['status'],'COMPLETE');self.assertEqual((r['completed_cost_cells'],r['recorded_timer_samples']),(38,190))
                self.assertTrue(all(n==16 for n in values[8].values()))
                rows={(row['route'],row['batch']):row for row in r['rows']}
                for row in r['rows']:
                    self.assertEqual(row['median_seconds_per_window'],row['median_seconds']/row['batch'])
                    for label,base in (('RAW','RAW_BOLT'),('matched_Bolt',cost.matched_bolt(row['route']))):
                        ratios=[a/b for a,b in zip(row['seconds'],rows[base,row['batch']]['seconds'])]
                        self.assertEqual(row['paired_'+label+'_ratios'],ratios);self.assertEqual(row['paired_'+label+'_median_ratio'],float(np.median(ratios)))
                    for sample in row['samples']:
                        self.assertEqual((sample['perf_end_ns']-sample['perf_start_ns'])/1e9,sample['seconds']);self.assertGreater(sample['seconds'],0)
                    self.assertEqual(row['direct_C_readout_bypasses_D'],'RIDGE' in row['route'])
                json.dumps(r,allow_nan=False);results.append(r)
        self.assertEqual(sum(r['completed_cost_cells'] for r in results),76);self.assertEqual(sum(r['recorded_timer_samples'] for r in results),380)

    def test_failed_route_preserves_slots_and_other_routes_continue(self):
        values=fixture('robin');broken='PCA_K4_RIDGE_HISTORY512'
        values[5].pop(broken);values[6].pop(broken)
        with tempfile.TemporaryDirectory(dir=cost.HERE) as temp:
            result=cost.measure_cost('robin',*values[:8],Path(temp)/'failed',lambda:None,lambda:None,memory,{broken:{'reason':'synthetic unavailable'}})
        self.assertEqual(result['status'],'PARTIAL_COMPLETED');self.assertEqual((result['completed_cost_cells'],result['recorded_timer_samples']),(36,180))
        rows=[r for r in result['rows'] if r['route']==broken];self.assertEqual(len(rows),2);self.assertTrue(all(not r['samples'] and r['failure'] for r in rows));self.assertIn(broken,result['routes'])

    def test_bad_observed_values_scaler_and_unregistered_route_rejected(self):
        for kind in ('observed','scale','route'):
            values=fixture('robin')
            if kind=='observed':values[0][0,4,0]=np.nan
            elif kind=='scale':values[3][0]=0
            else:values[5]['unregistered']=lambda h,m:h
            with self.subTest(kind=kind),self.assertRaises(ValueError):cost.validate_inputs(*values[:5],values[5],values[6],values[7],{},'robin')


if __name__=='__main__':unittest.main()
