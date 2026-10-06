"""Pinned native Chronos-Bolt F0 inference, with one independent row per channel."""

import hashlib
import json
from pathlib import Path

import torch
from chronos import ChronosBoltPipeline


def state_hash(module):
    digest = hashlib.sha256()
    for kind, tensors in (("parameter", module.named_parameters()), ("buffer", module.named_buffers())):
        for name, tensor in tensors:
            array = tensor.detach().cpu().contiguous().numpy()
            digest.update(json.dumps([kind, name, str(array.dtype), list(array.shape)]).encode("utf-8"))
            digest.update(array.tobytes())
    return digest.hexdigest()


class BoltForecast:
    def __init__(self, checkpoint_dir, device="cuda"):
        self.checkpoint_dir = Path(checkpoint_dir).resolve(strict=True)
        if not self.checkpoint_dir.is_dir():
            raise ValueError("Bolt checkpoint must be a pinned local directory")
        self.device = device
        self.file_config = json.loads((self.checkpoint_dir / "config.json").read_text(encoding="utf-8"))
        if self.file_config.get("architectures") != ["ChronosBoltModelForForecasting"]:
            raise ValueError("Only the official native Chronos-Bolt forecasting architecture is allowed")
        self.pipeline = ChronosBoltPipeline.from_pretrained(
            str(self.checkpoint_dir), device_map=device, torch_dtype=torch.float32, local_files_only=True
        )
        self.model = self.pipeline.model.eval().requires_grad_(False)
        if type(self.model).__name__ != "ChronosBoltModelForForecasting":
            raise ValueError("Loaded model is not the declared Chronos-Bolt family")
        if any(p.is_floating_point() and p.dtype != torch.float32 for p in self.model.parameters()):
            raise ValueError("Chronos-Bolt parameters must be FP32")
        self.quantiles = tuple(float(q) for q in self.model.chronos_config.quantiles)
        if self.quantiles.count(0.5) != 1:
            raise ValueError("The native quantile configuration must contain exactly one q=0.5")
        self.median_index = self.quantiles.index(0.5)
        config = self.model.chronos_config
        declared = self.file_config["chronos_config"]
        if config.context_length < 512 or config.prediction_length < 48:
            raise ValueError("Checkpoint cannot provide L512/H48")
        if (config.context_length != declared["context_length"]
                or config.prediction_length != declared["prediction_length"]
                or self.quantiles != tuple(float(q) for q in declared["quantiles"])):
            raise ValueError("Loaded context/horizon/quantiles differ from the pinned config")

    @property
    def module(self):
        return self.model

    @torch.inference_mode()
    def __call__(self, standardized_cpu_x):
        x = standardized_cpu_x
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[1] != 512:
            raise ValueError("Expected CPU tensor [B,512,C]")
        if x.device.type != "cpu" or x.dtype != torch.float32:
            raise ValueError("Deployment starts with standardized CPU FP32 inputs")
        batch, length, channels = x.shape
        if batch <= 0 or channels <= 0:
            raise ValueError("Origin and channel axes must be nonempty")
        device_x = x.to(self.device)
        rows = device_x.transpose(1, 2).reshape(batch * channels, length)
        output = self.model(context=rows)
        quantile_preds = output.quantile_preds
        expected = (batch * channels, len(self.quantiles), self.model.chronos_config.prediction_length)
        if tuple(quantile_preds.shape) != expected:
            raise ValueError("Native output axes differ from [B*C,quantile,horizon]")
        point = quantile_preds[:, self.median_index, :48].clone(memory_format=torch.contiguous_format)
        return point.reshape(batch, channels, 48).transpose(1, 2).contiguous().float().cpu()

    def receipt(self, include_file_hashes=False, include_state_hash=False):
        parameters, buffers = list(self.model.parameters()), list(self.model.buffers())
        config, chronos_config = self.model.config, self.model.chronos_config
        parameter_bytes = sum(p.numel() * p.element_size() for p in parameters)
        buffer_bytes = sum(b.numel() * b.element_size() for b in buffers)
        result = {
            "checkpoint_dir": str(self.checkpoint_dir),
            "family": "Chronos-Bolt",
            "class": type(self.model).__name__,
            "architectures": list(config.architectures),
            "architecture": {name: getattr(config, name) for name in
                             ("d_model", "d_ff", "num_layers", "num_decoder_layers", "num_heads")},
            "context_length": 512, "prediction_length": 48,
            "model_context_length": chronos_config.context_length,
            "model_prediction_length": chronos_config.prediction_length,
            "input_patch_size": chronos_config.input_patch_size,
            "input_patch_stride": chronos_config.input_patch_stride,
            "use_reg_token": chronos_config.use_reg_token,
            "median_index": self.median_index, "quantiles": list(self.quantiles),
            "point_forecast": "native inverse-scaled q=0.5, first 48 observations",
            "task_rows_per_origin": "C original independent univariate channels",
            "reshape": "[B,L,C] -> [B,C,L] -> [B*C,L] -> [B*C,H] -> [B,H,C]",
            "forward_arguments": {"context": "[B*C,512] FP32", "mask": None,
                                  "target": None, "target_mask": None},
            "pipeline_predict_used": False, "row_chunk": None,
            "preprocessing": "existing TRAIN standardization and input imputation; native Bolt instance norm",
            "extra_normalization_patch": None,
            "parameter_count": sum(p.numel() for p in parameters),
            "trainable_parameter_count": sum(p.numel() for p in parameters if p.requires_grad),
            "parameter_bytes": parameter_bytes,
            "buffer_count": sum(b.numel() for b in buffers), "buffer_bytes": buffer_bytes,
            "total_tensor_bytes": parameter_bytes + buffer_bytes,
            "parameter_dtypes": sorted({str(p.dtype) for p in parameters}),
            "buffer_dtypes": sorted({str(b.dtype) for b in buffers}),
            "eval_mode": not self.model.training,
            "output_dtype": "torch.float32", "output_device": "cpu",
            "output_unit": "same TRAIN-standardized coordinate system as input",
            "load_arguments": {"device_map": str(self.device), "torch_dtype": "torch.float32",
                               "local_files_only": True},
        }
        if include_file_hashes:
            paths = [self.checkpoint_dir / "config.json", *sorted(self.checkpoint_dir.glob("*.safetensors"))]
            result["checkpoint_files"] = []
            for path in paths:
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
                        digest.update(block)
                result["checkpoint_files"].append(
                    {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}
                )
        if include_state_hash:
            result["state_sha256"] = state_hash(self.model)
        return result
