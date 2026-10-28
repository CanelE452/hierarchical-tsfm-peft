import tempfile
import unittest
from pathlib import Path

import numpy as np

from research.ppt_justification_20261028_v1 import postprocess as p


def rows(method, values, dataset='jena', cap=3, provisional=False):
    return [dict(method_id=method,dataset=dataset,seed=seed,MSE=value,MAE=value/2,
                 trainable_parameters=35717,training_cap=cap,extended=cap==10,
                 selected_epoch=1,provisional=provisional) for seed,value in zip(p.SEEDS,values)]


class ScientificAnalysisTests(unittest.TestCase):
    def test_paired_differences_follow_seed_not_input_order(self):
        data=rows('OURS',[1,2,3,4,5])+list(reversed(rows('NO_F_LINEAR',[2,4,6,8,10])))
        _,contrasts,paired=p.aggregate(data)
        self.assertEqual([r['delta_MSE'] for r in paired],[1,2,3,4,5])
        self.assertEqual(contrasts[0]['n'],5)

    def test_same_seed_missing_or_duplicate_rejected(self):
        data=rows('OURS',[1,1,1,1,1])
        for wrong in [data[:-1],data+data,data[:-1]+[data[0]]]:
            with self.assertRaises(ValueError):p.aggregate(wrong)

    def test_unclear_is_not_equal_and_bad_result_is_not_hidden(self):
        _,c,_=p.aggregate(rows('OURS',[1,2,3,4,5])+rows('NO_F_LINEAR',[1,2,3.1,4,5]))
        self.assertEqual(c[0]['descriptive_decision'],'UNCLEAR')
        _,c,_=p.aggregate(rows('OURS',[2]*5)+rows('NO_F_LINEAR',[1]*5))
        self.assertEqual(c[0]['descriptive_decision'],'VARIANT_LOWER')
        self.assertEqual(c[0]['effect_pct'],-50)

    def test_reference_provisional_is_propagated(self):
        _,c,_=p.aggregate(rows('OURS',[2]*5,provisional=True)+rows('NO_F_LINEAR',[1]*5))
        self.assertTrue(c[0]['descriptive_decision'].endswith('_PROVISIONAL'))

    def test_factorial_references_and_interaction(self):
        data=rows('OURS',[1]*5)+rows('A1',[3]*5,cap=10)+rows('NO_F_LINEAR',[4]*5)+rows('NO_F_LINEAR_NO_P',[9]*5)
        _,contrasts,_=p.aggregate(data)
        c={r['comparison_id']:r for r in contrasts}
        self.assertEqual(c['F_WITHOUT_P_JENA']['reference_method_id'],'A1')
        self.assertEqual(c['P_WITHOUT_F_JENA']['reference_method_id'],'NO_F_LINEAR')
        self.assertEqual(p.interaction_summary(data)['median'],3)
        self.assertEqual(p.interaction_summary(data)['status'],'DESCRIPTIVE_ONLY')

    def test_mixed_schedules_rejected(self):
        data=rows('OURS',[1]*5);data[2]['training_cap']=10
        with self.assertRaises(ValueError):p.aggregate(data)

    def selector_fixture(self,extension_count,state):
        base=[];ext=[];verified={}
        for method,dataset in p.ALL_FAMILIES:
            for seed in p.SEEDS:
                b=dict(job_id=f'{method}_{dataset}_{seed}',method_id=method,dataset=dataset,seed=seed,extended=False)
                e=dict(b,job_id=b['job_id']+'_ext',extended=True)
                base.append(b);ext.append(e)
                if (method,dataset)==p.MANDATORY[0]:
                    row=dict(job_id=b['job_id'],method_id=method,dataset=dataset,seed=seed,K=6,
                        trainable_parameters=35717,normalization_fit_end=10,MSE=1.,MAE=.5,
                        selected_epoch=3.,training_cap=3,extended=False)
                    verified[b['job_id']]={'row':row}
                    if seed-p.SEEDS[0]<extension_count:
                        verified[e['job_id']]={'row':dict(row,job_id=e['job_id'],selected_epoch=10.,training_cap=10,extended=True)}
        plan={'jobs':[j for j in base if (j['method_id'],j['dataset']) in p.MANDATORY],
              'optional_jobs':[j for j in base if (j['method_id'],j['dataset']) in p.OPTIONAL],'extension_jobs':ext}
        return plan,{'extension_status':{'NO_F_LINEAR:jena':state}},verified

    def test_intermediate_extension_state_rejected(self):
        args=self.selector_fixture(0,{'status':'ADMITTED','cap_selected_count':5})
        with self.assertRaises(ValueError):p.select_complete_families(*args)

    def test_partial_extension_keeps_complete_base_provisionally(self):
        args=self.selector_fixture(2,{'status':'INCOMPLETE','cap_selected_count':5})
        primary,all_new,coverage=p.select_complete_families(*args)
        self.assertEqual(len(primary),5);self.assertEqual(len(all_new),7)
        self.assertTrue(all(not r['extended'] and r['provisional'] for r in primary))
        self.assertEqual(coverage[1]['primary_n'],0)

    def test_complete_cap10_extension_replaces_base_provisionally(self):
        args=self.selector_fixture(5,{'status':'COMPLETE','cap_selected_count':5,'cap10_selected_count':5,'undertrained_at_10':True})
        primary,all_new,_=p.select_complete_families(*args)
        self.assertEqual(len(primary),5);self.assertEqual(len(all_new),10)
        self.assertTrue(all(r['extended'] and r['provisional'] for r in primary))

    def test_macro_channel_metric_and_protected_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'v.npz'
            pred=np.array([[[1.,3.],[1.,99.]]]);target=np.zeros_like(pred)
            observed=np.array([[[True,True],[True,False]]])
            arrays=dict(pred_normalized=pred,target_normalized=target,observed=observed,
                        origins=np.array([10]),channel_ids=np.array([0,1]),mean=np.zeros(2),std=np.ones(2))
            np.savez(path,**arrays)
            row=dict(job_id='TEST',dataset='jena',prediction_path=str(path),prediction_sha256=p.runtime.sha(path),MSE=5.,MAE=2.)
            _,checked=p.verify_arrays(row,None,{'shape':[1,2,2],'T_start':12})
            self.assertEqual(checked['MSE'],5.)
            with self.assertRaises(ValueError):p.verify_arrays(row,None,{'shape':[1,2,2],'T_start':11})
            changed=dict(arrays,target_normalized=np.ones_like(pred))
            with self.assertRaises(ValueError):p.verify_arrays(row,changed,{'shape':[1,2,2],'T_start':12})
            row['MSE']=4.
            with self.assertRaises(ValueError):p.verify_arrays(row,None,{'shape':[1,2,2],'T_start':12})


if __name__=='__main__':unittest.main()
