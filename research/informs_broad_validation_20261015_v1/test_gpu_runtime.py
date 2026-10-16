"""Synthetic row batching and frozen-input-gradient checks, without real F/data."""
import unittest
import torch
try:
    from gpu_runtime import RowForecaster
except ImportError:
    RowForecaster = None

class FrozenRowTests(unittest.TestCase):
    def test_real_row_count_and_checkpoint_gradient(self):
        self.assertIsNotNone(RowForecaster, 'row runtime is not implemented')
        layer=torch.nn.Linear(9,4,bias=False).requires_grad_(False)
        x=torch.randn(2,9,5,requires_grad=True)
        f=RowForecaster(layer,4,row_microbatch=3,checkpoint_inputs=True)
        actual=f(x)
        expected=layer(x.transpose(1,2)).transpose(1,2)
        torch.testing.assert_close(actual,expected)
        self.assertEqual(f.streams,10)
        self.assertEqual(f.row_calls,4)
        actual.square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(torch.count_nonzero(x.grad).item(),0)
        self.assertIsNone(layer.weight.grad)

    def test_microbatch_parity(self):
        self.assertIsNotNone(RowForecaster, 'row runtime is not implemented')
        x=torch.randn(2,9,5)
        for size in (None,32,8,4):
            f=RowForecaster(lambda a:a[:,-1:].expand(-1,4),4,size,False)
            torch.testing.assert_close(f(x),x[:,-1:, :].expand(-1,4,-1))
            self.assertEqual(f.streams,10)

if __name__=='__main__':unittest.main()
