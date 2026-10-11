"""Installed official Chronos-Bolt-small, independent latent-coordinate forecasts."""
import inspect
import importlib.metadata

import torch

from common_direct import ROOT, HERE, digest, state_digest, write_json


MODEL_DIRECTORY = ROOT / '.cache/huggingface/models--amazon--chronos-bolt-small/snapshots/772f3d25d38aec6d914c8949dab4462e2d46f5d8'


class Foundation:
    def __init__(self, resources, max_rows=128):
        from chronos import ChronosBoltPipeline
        self.pipeline = ChronosBoltPipeline.from_pretrained(str(MODEL_DIRECTORY),
                                                            device_map='cuda', torch_dtype=torch.float32)
        self.model = self.pipeline.model.eval().requires_grad_(False)
        self.quantiles = list(self.model.chronos_config.quantiles)
        self.median = self.quantiles.index(.5)
        self.native_horizon = self.model.chronos_config.prediction_length
        self.max_rows = max_rows
        self.resources = resources
        self.initial_hash = state_digest(self.model)
        source = inspect.getfile(type(self.model))
        self.receipt = {'package_version': importlib.metadata.version('chronos-forecasting'),
                        'model_directory': MODEL_DIRECTORY,
                        'config_sha256': digest(MODEL_DIRECTORY / 'config.json'),
                        'model_state_sha256': self.initial_hash,
                        'installed_source_path': source, 'installed_source_sha256': digest(source),
                        'native_prediction_length': self.native_horizon,
                        'context_maximum': self.pipeline.model_context_length,
                        'quantiles': self.quantiles, 'point_quantile': .5,
                        'feature_coordinates': 'independent batch rows; time remains time',
                        'max_feature_rows_per_forward': self.max_rows,
                        'precision': 'FP32, no AMP, TF32 disabled', 'frozen': True,
                        'upstream_repository': 'https://github.com/amazon-science/chronos-forecasting'}
        write_json(HERE / 'FOUNDATION_PROVENANCE.json', self.receipt)

    @torch.no_grad()
    def forecast(self, latent, horizon):
        if latent.ndim < 3 or horizon > self.native_horizon:
            raise ValueError('Expected [B,time,features...] and a within-native horizon')
        batch, length = latent.shape[:2]
        shape = latent.shape[2:]
        rows = latent.reshape(batch, length, -1).transpose(1, 2).reshape(-1, length)
        predictions = []
        for start in range(0, len(rows), self.max_rows):
            prediction = self.model(context=rows[start:start + self.max_rows]).quantile_preds
            predictions.append(prediction[:, self.median, :horizon])
            self.resources.value['F_forward_calls'] += 1
        combined = torch.cat(predictions)
        return combined.reshape(batch, -1, horizon).transpose(1, 2).reshape(batch, horizon, *shape)

    @torch.no_grad()
    def verify_pipeline_parity(self):
        fixture = torch.linspace(-2, 3, 512, device='cuda').reshape(1, 512, 1)
        direct = self.forecast(fixture, 48).reshape(1, 48).cpu()
        official = self.pipeline.predict(fixture[:, :, 0], prediction_length=48)[:, self.median]
        difference = float((direct - official).abs().max())
        if not torch.equal(direct, official):
            raise RuntimeError('Direct median path differs from installed official pipeline')
        return {'status': 'PASS', 'max_abs_difference': difference,
                'new_optimizer_updates': 0, 'fixture': 'synthetic ramp, not science accuracy'}

    def verify_unchanged(self):
        return {'unchanged': state_digest(self.model) == self.initial_hash,
                'parameter_gradients_absent': all(p.grad is None for p in self.model.parameters()),
                'F_backward_calls': self.resources.value['F_backward_calls']}
