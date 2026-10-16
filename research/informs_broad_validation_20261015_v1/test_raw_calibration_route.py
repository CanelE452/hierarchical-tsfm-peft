import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

import deployment
import fit_base
import test_costs as fixtures


class BiasRouteTest(unittest.TestCase):
    def test_maximum_only_trained_model_can_really_infer_posthoc_K(self):
        task = fixtures.CostTests().task()
        pool = deployment.FoundationPool(None, "cpu", factory=lambda p, h, r, device: fixtures.ToyFoundation(h))
        with tempfile.TemporaryDirectory() as temp:
            spec = dict(run_id="max", family="B0", seed=9, device="cpu", latent_k=3, k_grid=[3])
            model, norm = fit_base.create_model(spec, task, task["split_ranges"]["A_stop"][1], pool.get(task["H"]))
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            state_sha = fit_base.training.fingerprint({key: fit_base.data_contract.array_sha256(value.numpy()) for key, value in state.items()})
            torch.save(dict(spec=spec, normalization=norm, model=state, model_state_sha=state_sha,
                            contract=dict(data_sha=task["data_sha256"], model_sha=fit_base._model_sha(pool.foundation),
                                          scaler_sha=fit_base.scaler_fingerprint(norm["mean"], norm["std"]))), Path(temp) / "max.pt")
            route = deployment.Route(dict(run_id="posthoc", family="B0", artifact_cache_relative_path="max.pt",
                                           latent_k=2, prefix_variant="posthoc"), task, temp, device="cpu", foundation_pool=pool)
            x = task["physical"][500:540][None]
            route.predict_physical(x, np.ones_like(x, bool))
            self.assertEqual(route.last_receipt["F_logical_streams"], 2)
            self.assertEqual(route.bundle["spec"]["k_grid"], [3])
            route.close(); pool.close()

    def test_actual_parent_and_bias_with_real_stream_count(self):
        task = fixtures.CostTests().task()
        mean, std, _ = fit_base.data_contract.fit_statistics(task, task["split_ranges"]["A_stop"][1])
        contract = dict(data_sha=task["data_sha256"], scaler_sha=fit_base.scaler_fingerprint(mean, std))
        with tempfile.TemporaryDirectory() as temp:
            with (Path(temp) / "bias.npz").open("xb") as handle:
                np.savez_compressed(handle, bias=np.array([.2, -.5, 1.]), mean=mean, std=std,
                                    contract_json=np.asarray(json.dumps(contract)))
            pool = deployment.FoundationPool(None, "cpu", factory=lambda p, h, r, device: fixtures.ToyFoundation(h))
            route = deployment.Route(dict(run_id="bias", family="RAW_CALIBRATED", artifact_cache_relative_path="bias.npz",
                                           b_route=dict(run_id="raw", family="RAW")), task, temp, device="cpu", foundation_pool=pool)
            x = task["physical"][500:540][None]
            result = route.predict_physical(x, np.ones_like(x, bool))
            expected = np.repeat(x[:, -1:], task["H"], axis=1) + np.array([.2, -.5, 1.]) * std
            np.testing.assert_allclose(result, expected, atol=1e-3, rtol=1e-6)
            self.assertEqual(route.last_receipt["F_logical_streams"], 3)
            self.assertEqual(pool.instances_created, 1)
            route.close(); pool.close()

    def test_nested_pca_calls_only_prefix_and_binds_foundation(self):
        task = fixtures.CostTests().task()
        end = task["split_ranges"]["A_stop"][1]
        mean, std, _ = fit_base.data_contract.fit_statistics(task, end)
        with tempfile.TemporaryDirectory() as temp:
            with (Path(temp) / "pca.npz").open("xb") as handle:
                np.savez_compressed(handle, basis=np.eye(3, dtype=np.float32), center=np.zeros(3, np.float32),
                                    mean=mean, std=std, fit_end=np.asarray(end), data_sha=np.asarray(task["data_sha256"]))
            pool = deployment.FoundationPool(None, "cpu", factory=lambda p, h, r, device: fixtures.ToyFoundation(h))
            foundation = pool.get(task["H"])
            spec = dict(run_id="nested", family="NESTED_PCA", artifact_cache_relative_path="pca.npz",
                        latent_k=2, foundation_sha=fit_base._model_sha(foundation))
            route = deployment.Route(spec, task, temp, device="cpu", foundation_pool=pool)
            x = task["physical"][500:540][None]
            result = route.predict_physical(x, np.ones_like(x, bool))
            self.assertEqual(route.last_receipt["F_logical_streams"], 2)
            np.testing.assert_allclose(result[:, :, 2], mean[2], atol=1e-5)
            route.close()
            with self.assertRaises(ValueError):
                deployment.Route(dict(spec, foundation_sha="wrong"), task, temp, device="cpu", foundation_pool=pool)
            pool.close()


if __name__ == "__main__":
    unittest.main()
