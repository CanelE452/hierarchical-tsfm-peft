import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location('v2_evaluator_under_test', Path(__file__).with_name('evaluate_v2.py'))
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)


def row(arm='residual', mode='current', seed=92601, value=1, latent=8, dataset='electricity'):
    return dict(id=f'{dataset}_{arm}_{mode}_k{latent}_{seed}', dataset=dataset, arm=arm,
                ed_mode=mode, seed=seed, best_val_mse=value, latent=latent)


class SelectionTests(unittest.TestCase):
    def test_selection_averages_losses_instead_of_choosing_seed(self):
        rows = [row(seed=s, value=v) for s, v in zip(ev.SEEDS, [1, 9])]
        rows += [row(mode='fixed_ed', seed=s, value=4) for s in ev.SEEDS]
        selection = ev.select_modes(rows)['electricity']
        self.assertEqual(selection['residual']['mode'], 'fixed_ed')
        self.assertEqual(selection['residual']['mean_best_val_mse'], 4)
        self.assertEqual(selection['selection_opportunity']['status'], 'pending')

    def test_raw_has_own_val_selection_but_equal_opportunities(self):
        rows = [row(a, m, s, 1 if (a, m) in [('residual', 'slow_ed'), ('raw_bypass', 'fixed_ed')] else 2)
                for a in ['residual', 'raw_bypass'] for m in ev.MODES for s in ev.SEEDS]
        selection = ev.select_modes(rows)['electricity']
        self.assertEqual(selection['residual']['mode'], 'slow_ed')
        self.assertEqual(selection['raw_bypass']['mode'], 'fixed_ed')
        self.assertTrue(selection['selection_opportunity']['final_fair_selection_comparison'])

    def test_equal_score_tie_keeps_current(self):
        selection = ev.select_modes([row(mode=m, seed=s) for m in ev.MODES for s in ev.SEEDS])
        self.assertEqual(selection['electricity']['residual']['mode'], 'current')

    def test_incomplete_seed_rejected(self):
        with self.assertRaisesRegex(ValueError, 'incomplete paired'):
            ev.select_modes([row()])

    def test_duplicate_selection_fit_rejected(self):
        with self.assertRaisesRegex(ValueError, 'multiple fits'):
            ev.select_modes([row(), row()])

    def test_nan_validation_rejected(self):
        with self.assertRaisesRegex(ValueError, 'validation'):
            ev.select_modes([row(value=float('nan'))])

    def test_same_mode_and_seed_at_distinct_latents_do_not_collide(self):
        rows = [row(a, m, s, value=(.5 if k == 8 else 1), latent=k, dataset='bull')
                for a in ['residual', 'raw_bypass'] for m, k in
                [('current',4), ('fixed_ed',4), ('slow_ed',4), ('fixed_ed',8)] for s in ev.SEEDS]
        result = ev.select_modes(rows)['bull']
        self.assertEqual(result['residual']['variant'], 'fixed_ed:k8')
        self.assertEqual(result['residual']['latent'], 8)
        self.assertEqual(len(result['residual']['available_variants']), 4)
        self.assertTrue(result['selection_opportunity']['final_fair_selection_comparison'])

    def test_same_mode_opportunities_but_different_latents_are_not_fair(self):
        rows = [row(a, 'fixed_ed', s, latent=k, dataset='bull')
                for a,k in [('residual',8),('raw_bypass',4)] for s in ev.SEEDS]
        result = ev.select_modes(rows)['bull']
        self.assertEqual(result['selection_opportunity']['status'], 'pending')

    def test_latent_tie_prefers_existing_base_configuration(self):
        rows = [row(mode=m,seed=s,latent=k,dataset='bull')
                for m,k in [('fixed_ed',8),('slow_ed',4),('fixed_ed',4),('current',4)] for s in ev.SEEDS]
        self.assertEqual(ev.select_modes(rows)['bull']['residual']['variant'],'current:k4')

    def test_latent_comparisons_keep_matched_seeds_and_do_not_average_k(self):
        target = np.zeros((14, 2, 2))
        mask = np.ones_like(target, bool)
        fits = [row(a,'fixed_ed',s,value=v,latent=k,dataset='bull')
                for a,k,v in [('residual',4,1),('residual',8,.5),('raw_bypass',4,4),('raw_bypass',8,2)]
                for s in ev.SEEDS]
        arrays, rows = {}, []
        for fit in fits:
            value = fit['best_val_mse']
            prediction = np.full(target.shape, np.sqrt(value))
            arrays[fit['id']] = prediction
            rows.append(dict(fit,fit=fit['id'],key=fit['id'],scores=ev.score_period(prediction,target,mask)))
        result = ev.period_comparisons(rows,arrays,ev.select_modes(fits)['bull'],target,mask)
        self.assertEqual(len(result['mean_scores']),4)
        self.assertAlmostEqual(result['matched_variant_residual_vs_raw']['fixed_ed:k8']['gain_pct'],75)
        self.assertAlmostEqual(result['fixed_ed_k8_vs_k4']['residual']['effect']['gain_pct'],50)
        self.assertAlmostEqual(result['selected_residual_vs_selected_raw']['gain_pct'],75)

    def test_incomplete_new_latent_seed_is_rejected(self):
        fits = [row(seed=s,latent=4,dataset='bull') for s in ev.SEEDS]
        fits += [row(mode='fixed_ed',latent=8,dataset='bull')]
        with self.assertRaisesRegex(ValueError,'incomplete paired'):
            ev.select_modes(fits)


class ScoreAndCacheTests(unittest.TestCase):
    def test_macro_weighting_and_missing_nan(self):
        target = np.array([[[1., 1.], [1., np.nan]]])
        mask = np.isfinite(target)
        prediction = np.array([[[3., 5.], [3., 100.]]])
        score = ev.scores(prediction, target, mask)
        self.assertEqual(score['mse'], 10)
        self.assertEqual(score['mae'], 3)
        self.assertEqual(score['target_counts'], [2, 1])

    def test_mask_does_not_hide_nonfinite_prediction(self):
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            ev.scores(np.array([[[np.nan]]]), np.ones((1, 1, 1)), np.zeros((1, 1, 1), bool))

    def test_effect_uses_mean_seed_losses_not_ensemble(self):
        target = np.zeros((14, 2, 2))
        mask = np.ones_like(target, dtype=bool)
        effect = ev.paired_effect([np.ones_like(target), -np.ones_like(target)], [2*np.ones_like(target)], target, mask, draws=30)
        self.assertEqual(effect['gain_pct'], 75)
        self.assertEqual(effect['conditional_block95_pct'], [75, 75])

    def test_effect_channel_macro_with_missing_targets(self):
        target = np.zeros((14, 2, 2))
        mask = np.ones_like(target, dtype=bool)
        mask[:, 1, 1] = False
        target[~mask] = np.nan
        effect = ev.paired_effect([np.ones_like(target)], [2*np.ones_like(target)], target, mask, draws=30)
        self.assertEqual(effect['gain_pct'], 75)

    def test_zero_reference_is_unavailable(self):
        target = np.zeros((14, 2, 2))
        effect = ev.paired_effect([np.ones_like(target)], [target], target, np.ones_like(target, bool), draws=3)
        self.assertEqual(effect['status'], 'unavailable')

    def test_origin_boundary_prevents_future_target_leak(self):
        data = dict(x=np.ones((800, 2)), val_origins=np.array([600, 613]))
        contract = {'datasets': {'electricity_first32': {'split_bounds': {'val': [600, 660]}, 'origin_counts': {'val': 2}}}}
        with self.assertRaisesRegex(ValueError, 'leaking'):
            ev.validate_origins(data, 'electricity', 'val', contract)
        data['val_origins'] = np.array([600, 612])
        np.testing.assert_array_equal(ev.validate_origins(data, 'electricity', 'val', contract), [600, 612])

    def test_cache_hash_origins_and_scores_are_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'p.npz'
            prediction = np.zeros((2, 2, 1))
            origins = np.array([512, 536])
            np.savez(path, prediction=prediction, origins=origins)
            target, mask = np.ones_like(prediction), np.ones_like(prediction, bool)
            ev.read_prediction(path, origins, target, mask, ev.digest(path), dict(mse=1, mae=1))
            for kwargs, message in [(dict(expected_hash='bad'), 'hash'),
                                    (dict(expected_scores=dict(mse=2, mae=1)), 'mse')]:
                with self.assertRaisesRegex(ValueError, message):
                    ev.read_prediction(path, origins, target, mask, **kwargs)
            with self.assertRaisesRegex(ValueError, 'origins'):
                ev.read_prediction(path, origins+1, target, mask)


class IntegrationTests(unittest.TestCase):
    def test_selection_precedes_inference_and_second_report_reuses_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            here, v1, cache = root/'v2', root/'v1', root/'cache'
            here.mkdir(); v1.mkdir()
            data, contract = {}, {'datasets': {}}
            for dataset in ev.PERIODS:
                path = root/(dataset+'.data')
                path.write_bytes(dataset.encode())
                item = dict(_path=path, x=np.ones((900, 2)), finite=np.ones((900, 2), bool),
                            columns=np.array(['a', 'b']), times=np.arange(900)*3600_000_000_000,
                            train_origins=np.array([512, 513]), val_origins=np.arange(600, 608))
                bounds = dict(train=[0, 562], val=[600, 660])
                if dataset == 'electricity':
                    item['dev_origins'] = np.arange(700, 714)
                    bounds['dev'] = [700, 780]
                    entry = dict(split_bounds=bounds, origin_counts=dict(train=2, val=8, dev=14))
                    contract['datasets']['electricity_first32'] = entry
                else:
                    for period, start in [('e1', 700), ('e2', 800)]:
                        item['eval_'+period+'_origins'] = np.arange(start, start+14)
                        bounds['eval_'+period] = [start, start+80]
                    entry = dict(split_periods={k:[str(np.datetime64('1970-01-01')+np.timedelta64(v,'h')) for v in pair]
                                               for k,pair in bounds.items()},
                                 origin_counts=dict(train=2, val=8, eval_e1=14, eval_e2=14))
                    contract['datasets']['bdg2_bull_office'] = entry
                data[dataset] = item
            ev.write_json(v1/'data_contract.json', contract)
            legacy, fitted, receipt, cohort, checkpoints = [], [], {}, [], {}
            for dataset, item in data.items():
                receipt[dataset] = dict(data_sha256=ev.digest(item['_path']))
                for arm in ['residual', 'raw_bypass']:
                    for seed in ev.SEEDS:
                        fit_id = f'{dataset}_{arm}_{seed}'
                        folder = root/fit_id; folder.mkdir()
                        checkpoint = folder/'best.pt'; checkpoint.write_bytes(b'legacy')
                        prediction = np.zeros((8,48,2))
                        np.savez(folder/'best_val.npz', prediction=prediction, origins=item['val_origins'])
                        fitted.append(dict(id=fit_id,dataset=dataset,arm=arm,seed=seed,ed_mode='current',
                                           latent=ev.BASE_LATENT[dataset],
                                           best_val_mse=1.,checkpoint=str(checkpoint),data_sha256=receipt[dataset]['data_sha256'],result_sha256=fit_id))
                        for period in ev.PERIODS[dataset]:
                            origin_key = ('eval_'+period if period!='dev' else period)+'_origins'
                            p=folder/(period+'.npz')
                            np.savez(p,prediction=np.zeros((14,48,2)),origins=item[origin_key])
                            legacy.append(dict(dataset=dataset,period=period,arm=arm,fit=fit_id,seed=seed,
                                               ed_mode='current',path=p,expected_hash=ev.digest(p),old_scores=dict(mse=1.,mae=1.)))
                for seed in ev.SEEDS:
                    spec=dict(id=f'{dataset}_fixed_{seed}',dataset=dataset,arm='residual',seed=seed,
                              ed_mode='fixed_ed',lr=.001,latent=ev.BASE_LATENT[dataset])
                    folder=root/spec['id']; folder.mkdir()
                    checkpoint=folder/'best.pt'; checkpoint.write_bytes(spec['id'].encode())
                    np.savez(folder/'best_val.npz',prediction=np.full((8,48,2),.5),origins=item['val_origins'])
                    result=dict(spec,status='complete',checkpoint=str(checkpoint),checkpoint_sha256=ev.digest(checkpoint),
                                data_sha256=receipt[dataset]['data_sha256'],best_val_mse=.25)
                    out=here/'runs'/spec['id']; out.mkdir(parents=True)
                    ev.write_json(out/'result.json',result)
                    checkpoints[str(checkpoint)]=dict(spec=spec,basis=np.zeros((2,1)),state={})
                    cohort.append(spec)
            cohort_path=here/'cohort.json'; ev.write_json(cohort_path,cohort)
            calls=[]
            class Model:
                def restore_adapter(self,state): pass
            class Job:
                record={'elapsed_s': .1}
                def __init__(self,*args): pass
                def __enter__(self): return self
                def __exit__(self,*args): pass
            def evaluate(model,item,origins,job,label):
                self.assertTrue((here/'first_selection.json').exists())
                calls.append(label)
                return {},np.full((len(origins),48,2),.5)
            fake_run=types.SimpleNamespace(GPUJob=Job,evaluate=evaluate,load_data=lambda d:data[d],
                                           make_model=lambda *a,**k:Model(),source_receipt=lambda:{})
            fake_torch=types.SimpleNamespace(load=lambda path,**k:checkpoints[str(path)],cuda=types.SimpleNamespace(empty_cache=lambda:None))
            with patch.object(ev,'HERE',here),patch.object(ev,'V1',v1),patch.object(ev,'CACHE',cache), \
                 patch.object(ev,'legacy_catalog',return_value=(legacy,fitted,receipt)), \
                 patch.dict(sys.modules,{'run':fake_run,'torch':fake_torch}):
                first=ev.run_evaluation(cohort_path,'first')
                self.assertEqual(first['schema_version'],2)
                self.assertEqual(len(calls),6)
                self.assertEqual(first['selection']['bull']['selection_opportunity']['status'],'pending')
                self.assertEqual(first['comparisons']['bull/e1']['selected_residual_vs_selected_raw']['status'],'pending')
                legacy_report=dict(first)
                legacy_report.pop('schema_version')
                legacy_report['rows']=[{k:v for k,v in r.items() if k!='latent'} for r in first['rows']]
                ev.write_json(here/'legacy_schema1.json',legacy_report)
                second=ev.run_evaluation(cohort_path,'second',[here/'legacy_schema1.json'])
                self.assertEqual(len(calls),6)
                self.assertEqual(second['gpu_accounted_seconds'],0)
                self.assertTrue(all(r['reused'] for r in second['rows']))
                self.assertTrue(all(r['latent']==ev.BASE_LATENT[r['dataset']] for r in second['rows']))
                bad=dict(first)
                bad['rows']=[dict(r,latent=999) if r['fit'] in {s['id'] for s in cohort} else r for r in first['rows']]
                ev.write_json(here/'bad_latent.json',bad)
                with self.assertRaisesRegex(ValueError,'latent differs'):
                    ev.run_evaluation(cohort_path,'bad_latent_reuse',[here/'bad_latent.json'])
                with self.assertRaises(FileExistsError): ev.run_evaluation(cohort_path,'first')


if __name__ == '__main__':
    unittest.main()
