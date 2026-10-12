"""CPU contract checks for the monitoring-only Robin wrapper."""
from pathlib import Path
import unittest
import numpy as np
import torch
from common import HERE,Budget,access_guard,write,sha


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue((HERE/'e5_robin_v2.py').exists(),'Robin monitoring wrapper not implemented')
        import e5_robin_v2 as wrapper
        self.wrapper = wrapper

    def test_fit_batch_keeps_two_nonoverlapping_instances(self):
        origins = self.wrapper.e5.training_origins((528,2121))
        np.testing.assert_array_equal(origins,[1040,1600])
        self.assertEqual(len(self.wrapper.e5.training_origins((2121,2520))),0)
        a = self.wrapper.e5.make_batches(origins,2,600,92601+31)
        self.assertEqual(a.shape,(600,2))
        self.assertTrue(all(set(row)=={1040,1600} for row in a))

    def test_tune_curve_unavailable_does_not_create_temporal_only_loss(self):
        called = []
        def objective(*args,**kwargs):
            called.append(np.asarray(args[4]).copy())
            return torch.tensor(1.5),{}
        data = {'fit_bounds':(528,2121),'tune_bounds':(2121,2520)}
        monitor = self.wrapper.monitor(None,None,'CAUSAL_CONTRAST',data,
                          np.array([[1040,1600]]),'cpu',92601,objective)
        self.assertEqual(len(called),1)
        np.testing.assert_array_equal(called[0],[1040,1600])
        self.assertEqual(monitor['fit_fixed_batch_objective_loss'],1.5)
        self.assertIsNone(monitor['tune_fixed_batch_objective_loss'])
        self.assertEqual(monitor['tune_objective_curve_status'],'UNAVAILABLE')

    def test_monkey_override_restores_original_lifecycle(self):
        before = (self.wrapper.e5.train_encoder,self.wrapper.e5.source_pins,self.wrapper.e5.write)
        with self.wrapper.install():
            self.assertIs(self.wrapper.e5.train_encoder,self.wrapper.train_encoder_v2)
            self.assertIn(str(HERE/'e5_robin_v2.py'),self.wrapper.e5.source_pins())
        self.assertEqual(before,(self.wrapper.e5.train_encoder,self.wrapper.e5.source_pins,self.wrapper.e5.write))


if __name__=='__main__':
    access_guard(); torch.set_num_threads(2)
    with Budget(2,gpu=False,label='E5 Robin monitor wrapper CPU unit') as budget:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests))
    write(HERE/'receipts'/f'E5_ROBIN_V2_UNIT_{budget.path.stem}.json',
          {'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
           'science_dispatches':0,'optimizer_updates':0,'GPU_calls':0,
           'budget_ledger':str(budget.path),'budget_ledger_sha256':sha(budget.path)})
    raise SystemExit(0 if result.wasSuccessful() else 1)
