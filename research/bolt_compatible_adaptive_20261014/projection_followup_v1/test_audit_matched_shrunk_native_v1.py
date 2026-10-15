"""Independent cached-head equations, causality, metric and lock fixtures."""
import copy
import inspect
import json
import unittest

import numpy as np

import audit_matched_shrunk_native_v1 as audit

audit.initialize_numpy()


def case(n=35):
    rng=np.random.default_rng(24)
    z=rng.normal(size=(n,512,2)).astype(np.float32)
    current=rng.normal(size=(n,3)).astype(np.float32)
    decoder=rng.normal(size=(n,2,3)).astype(np.float32)
    state={'head.weight':(rng.normal(size=(2,48,512))*.005).astype(np.float32),'head.bias':(rng.normal(size=(2,48))*.01).astype(np.float32)}
    payload={'latent_history':z,'decoder':decoder,'raw_last':current,'target':np.zeros((n,48,3),np.float32),'observed':np.ones((n,48,3),bool),'origins':np.arange(512,512+n,dtype=np.int64)}
    return payload,state


def registry_case():
    keys=[(data,route,str(seed)) for data in audit.DATASETS for route in ('PCA_PINV','FIT_TARGET_PINV') for seed in audit.SEEDS]
    locks={key:{'path':'synthetic_'+str(n),'sha256':'fixture'} for n,key in enumerate(keys)}
    registry={'status':'ALL_EIGHT_CONDITIONS_RESOLVED_BEFORE_ANY_NEW_DEV_INPUT_OR_PREDICTION','expected_conditions':8,
              'conditions':[{'dataset':key[0],'route':key[1],'seed':key[2],'status':'LOCKED','lock':locks[key]} for key in keys]}
    events=[{'sequence':n,'kind':'CONDITION_SELECTION_LOCK_SAVED','dataset':key[0],'route':key[1],'seed':key[2]} for n,key in enumerate(keys)]
    events.extend([{'sequence':8,'kind':'ALL_EIGHT_CONDITION_LOCKS_AND_REGISTRY_SAVED'},{'sequence':9,'kind':'DEV_INPUT_LOAD_AFTER_ALL_LOCKS'},{'sequence':10,'kind':'DEV_NATIVE_PREDICTION_AFTER_ALL_LOCKS'}])
    return events,registry,locks


class MatchedNativeIndependentFixtures(unittest.TestCase):
    def test_CPU32_components_match_independent_NumPy_equations(self):
        payload,state=case()
        prediction,future=audit.independent_head_forward(payload,state)
        alternative=audit.prior.native_latent_forecast(payload['latent_history'],state['head.weight'],state['head.bias'])
        expected=payload['raw_last'][:,None]+np.einsum('nhk,nkc->nhc',alternative-payload['latent_history'][:,-1:],payload['decoder'],optimize=False)
        np.testing.assert_allclose(future,alternative,atol=1e-5,rtol=2e-6)
        np.testing.assert_allclose(prediction,expected,atol=1e-5,rtol=2e-6)
        self.assertEqual(prediction.shape,(35,48,3))

    def test_zero_head_CP0_is_numeric_persistence_current_C(self):
        payload,state=case(2)
        for value in state.values():value[:]=0
        prediction,future=audit.independent_head_forward(payload,state)
        np.testing.assert_array_equal(prediction,np.broadcast_to(payload['raw_last'][:,None],prediction.shape))
        np.testing.assert_array_equal(future,np.broadcast_to(payload['latent_history'][:,-1:],future.shape))

    def test_signed_zero_CP0_byte_identity_is_not_claimed(self):
        payload,state=case(1)
        payload['raw_last'][:]=np.float32(-0.)
        for value in state.values():value[:]=0
        prediction,_=audit.independent_head_forward(payload,state)
        self.assertTrue(np.array_equal(prediction,np.broadcast_to(payload['raw_last'][:,None],prediction.shape)))

    def test_target_and_future_mask_are_absent_from_forward_dependencies(self):
        payload,state=case(2)
        original,_=audit.independent_head_forward(payload,state)
        changed={key:value.copy() for key,value in payload.items()};changed['target'][:]=np.nan;changed['observed'][:]=False
        again,_=audit.independent_head_forward(changed,state)
        audit.exact(original,again,'synthetic_future_label_and_mask_change')
        self.assertEqual(set(inspect.signature(audit.independent_head_forward).parameters),{'payload','state'})

    def test_each_origin_decoder_is_used_not_a_shared_static_substitute(self):
        payload,state=case(2)
        payload['latent_history'][1]=payload['latent_history'][0];payload['raw_last'][1]=payload['raw_last'][0]
        payload['decoder'][1]=2*payload['decoder'][0]
        prediction,_=audit.independent_head_forward(payload,state)
        np.testing.assert_allclose(prediction[1]-payload['raw_last'][1],2*(prediction[0]-payload['raw_last'][0]),atol=1e-6,rtol=2e-6)

    def test_CP_score_masks_NaN_and_uses_macro_active_channels(self):
        payload,state=case(1);prediction=np.zeros((1,48,3),np.float32)
        payload['target'][:]=np.nan;payload['observed'][:]=False
        payload['target'][0,0,0]=1;payload['observed'][0,0,0]=True
        payload['target'][0,:47,1]=2;payload['observed'][0,:47,1]=True
        recorded=audit.prior.scores(prediction,payload['target'],payload['observed'],np.ones(3))['normalized']
        recorded['per_channel_mse'][2]=None;recorded['per_channel_mae'][2]=None
        score,fields=audit.normalized_CP_score(recorded,prediction,payload,np.ones(3),'masked_nan_fixture')
        self.assertEqual(score['mse'],2.5);self.assertEqual(score['observed_counts'],[1,47,0]);self.assertEqual(fields,6)
        self.assertNotEqual(score['mse'],(1+47*4)/48)

    def test_exact_CP_tie_chooses_earliest_update(self):
        rows=[{'update':update,'mse':value} for update,value in zip(audit.CHECKPOINTS,[2.,.5,.5,.7,.8])]
        self.assertEqual(audit.selected_checkpoint(rows),100)
        for row in rows:row['mse']=0.
        self.assertEqual(audit.selected_checkpoint(rows),0)

    def test_missing_or_nonfinite_CP_does_not_get_selected(self):
        rows=[{'update':update,'mse':1.} for update in audit.CHECKPOINTS]
        rows[2]['mse']=float('nan')
        with self.assertRaises(ValueError):audit.selected_checkpoint(rows)
        with self.assertRaises(ValueError):audit.selected_checkpoint(rows[:4])

    def test_forward_rejects_extra_projection_buffer_or_FP64_parameters(self):
        payload,state=case(1);state['projection.encoder']=np.zeros((3,2),np.float32)
        with self.assertRaises(ValueError):audit.independent_head_forward(payload,state)
        del state['projection.encoder'];state['head.weight']=state['head.weight'].astype(np.float64)
        with self.assertRaises(ValueError):audit.independent_head_forward(payload,state)

    def test_state_hash_distinguishes_trained_head_and_frozen_payload_bytes(self):
        payload,state=case(1);initial=audit.prior.torch_state_sha(state)
        changed={key:value.copy() for key,value in state.items()};changed['head.bias'][0,0]+=.01
        self.assertNotEqual(initial,audit.prior.torch_state_sha(changed))
        frozen={key:payload[key] for key in ('latent_history','decoder','raw_last')}
        digest=audit.prior.torch_state_sha(frozen);modified={key:value.copy() for key,value in frozen.items()};modified['decoder'][0,0,0]+=.1
        self.assertNotEqual(digest,audit.prior.torch_state_sha(modified))

    def test_all_eight_registry_locks_before_any_DEV(self):
        result=audit.check_registry_events(*registry_case())
        self.assertTrue(result['all_eight_resolved_locks_before_any_new_DEV_input_or_prediction'])
        json.loads(json.dumps(result,allow_nan=False))

    def test_early_DEV_or_missing_condition_registry_is_rejected(self):
        events,registry,locks=registry_case();events[0]={'sequence':0,'kind':'DEV_INPUT_LOAD_AFTER_ALL_LOCKS'}
        with self.assertRaises(ValueError):audit.check_registry_events(events,registry,locks)
        events,registry,locks=registry_case();registry['conditions'].pop()
        with self.assertRaises(ValueError):audit.check_registry_events(events,registry,locks)

    def test_partial_resolved_condition_is_preserved_not_silently_called_complete(self):
        events,registry,locks=registry_case();registry['conditions'][0]['status']='FAILED_PATH_PRESERVED'
        self.assertTrue(audit.check_registry_events(events,registry,locks)['all_eight_resolved_locks_before_any_new_DEV_input_or_prediction'])

    def test_persistence_and_Raw_Bolt_are_distinct_forecast_controls(self):
        payload,state=case(1)
        payload['raw_last'][:]=0.
        current=np.broadcast_to(payload['raw_last'][:,None],payload['target'].shape)
        target=current+1
        persistence=audit.prior.scores(current,target,payload['observed'],np.ones(3))
        raw_bolt=audit.prior.scores(target,target,payload['observed'],np.ones(3))
        self.assertEqual(persistence['normalized']['mse'],1.)
        self.assertEqual(raw_bolt['normalized']['mse'],0.)

    def test_model_trainer_is_not_imported_by_independent_auditor(self):
        source=inspect.getsource(audit)
        self.assertNotIn('import native_shrunk_decoder_head_v1',source)
        self.assertNotIn('train_native_shrunk(',source)
        files=[{'path':str(audit.HERE/name)} for name in ('native_shrunk_decoder_head_v1.py','run_matched_shrunk_native_v1.py')]
        self.assertTrue(audit.source_contract(files)['head_only_optimizer_source_checked'])


if __name__=='__main__':unittest.main()
