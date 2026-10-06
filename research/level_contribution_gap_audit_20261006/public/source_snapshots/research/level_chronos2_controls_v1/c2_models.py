"""Pinned native Chronos-2 inference for independent forecast origins."""

import hashlib
import json
from pathlib import Path

import torch
from chronos import Chronos2Pipeline


class C2Forecast:
    def __init__(self, checkpoint_dir, relation="MV", device="cuda"):
        if relation not in ("MV", "UNI"):
            raise ValueError("relation must be MV or UNI")
        self.checkpoint_dir = Path(checkpoint_dir).resolve(strict=True)
        self.relation = relation
        self.pipeline = Chronos2Pipeline.from_pretrained(
            str(self.checkpoint_dir),
            device_map=device,
            dtype=torch.float32,
            local_files_only=True,
        )
        self.model = self.pipeline.model.eval().requires_grad_(False)
        if any(p.is_floating_point() and p.dtype != torch.float32 for p in self.model.parameters()):
            raise ValueError("Chronos-2 parameters must be FP32")
        self.quantiles = tuple(float(q) for q in self.pipeline.quantiles)
        self.median_index = self.quantiles.index(0.5)
        if self.pipeline.model_context_length < 512 or self.pipeline.model_prediction_length < 48:
            raise ValueError("checkpoint cannot provide L512/H48")

    @property
    def module(self):
        return self.model

    @torch.inference_mode()
    def __call__(self, standardized_cpu_x):
        x = standardized_cpu_x
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[1] != 512:
            raise ValueError("expected CPU tensor [B,512,C]")
        if x.device.type != "cpu" or x.dtype != torch.float32:
            raise ValueError("expected standardized CPU FP32 input")
        batch, _, channels = x.shape
        task_input = x.permute(0, 2, 1).contiguous()
        if self.relation == "UNI":
            task_input = task_input.reshape(batch * channels, 1, 512)
        forecasts = self.pipeline.predict(
            task_input,
            prediction_length=48,
            context_length=512,
            cross_learning=False,
            batch_size=batch * channels,
        )
        expected_tasks = batch if self.relation == "MV" else batch * channels
        variates = channels if self.relation == "MV" else 1
        if len(forecasts) != expected_tasks or any(
            tuple(y.shape) != (variates, len(self.quantiles), 48) for y in forecasts
        ):
            raise ValueError("native forecast axes differ from [task,C,quantile,H48]")
        median = torch.stack([y[:, self.median_index, :] for y in forecasts])
        return median.reshape(batch, channels, 48).permute(0, 2, 1).contiguous().cpu()

    def capture_group_probe(self, standardized_cpu_x):
        """Run an untimed probe and record real model IDs and attention masks."""
        records = []

        def model_hook(module, args, kwargs):
            context = kwargs["context"]
            future_covariates = kwargs.get("future_covariates")
            records.append(
                {
                    "context_shape": list(context.shape),
                    "context_dtype": str(context.dtype),
                    "group_ids": kwargs["group_ids"].detach().cpu().tolist(),
                    "num_output_patches": int(kwargs["num_output_patches"]),
                    "future_target_provided": kwargs.get("future_target") is not None,
                    "future_covariates_shape": None
                    if future_covariates is None
                    else list(future_covariates.shape),
                    "known_future_values": 0
                    if future_covariates is None
                    else int(torch.isfinite(future_covariates).sum().item()),
                }
            )

        def encoder_hook(module, args, kwargs):
            records[-1]["encoder_group_ids"] = kwargs["group_ids"].detach().cpu().tolist()
            records[-1]["encoder_attention_mask"] = kwargs["attention_mask"].detach().cpu().tolist()

        def attention_hook(module, args, kwargs):
            mask = kwargs["attention_mask"].detach().cpu()
            records[-1]["actual_group_time_mask_shape"] = list(mask.shape)
            records[-1]["actual_allowed_any_time"] = (mask[:, 0] == 0).any(dim=0).tolist()
            records[-1]["actual_allowed_last_time"] = (mask[-1, 0] == 0).tolist()
            records[-1]["actual_mask_min"] = float(mask.min().item())
            records[-1]["actual_mask_max"] = float(mask.max().item())

        handles = [
            self.model.register_forward_pre_hook(model_hook, with_kwargs=True),
            self.model.encoder.register_forward_pre_hook(encoder_hook, with_kwargs=True),
            self.model.encoder.block[0].layer[1].register_forward_pre_hook(attention_hook, with_kwargs=True),
        ]
        try:
            prediction = self(standardized_cpu_x)
        finally:
            for handle in handles:
                handle.remove()
        return prediction, records

    def receipt(self, include_file_hashes=False, include_state_hash=False):
        """Collect provenance outside the measured inference path."""
        parameters = list(self.model.parameters())
        buffers = list(self.model.buffers())
        config = self.model.chronos_config
        result = {
            "checkpoint_dir": str(self.checkpoint_dir),
            "relation": self.relation,
            "context_length": 512,
            "prediction_length": 48,
            "cross_learning": False,
            "batch_size_rule": "B*C model rows",
            "median_index": self.median_index,
            "quantiles": list(self.quantiles),
            "model_context_length": self.pipeline.model_context_length,
            "model_prediction_length": self.pipeline.model_prediction_length,
            "input_patch_size": config.input_patch_size,
            "output_patch_size": config.output_patch_size,
            "use_arcsinh": config.use_arcsinh,
            "parameter_count": sum(p.numel() for p in parameters),
            "trainable_parameter_count": sum(p.numel() for p in parameters if p.requires_grad),
            "parameter_bytes": sum(p.numel() * p.element_size() for p in parameters),
            "buffer_count": sum(b.numel() for b in buffers),
            "buffer_bytes": sum(b.numel() * b.element_size() for b in buffers),
            "parameter_dtypes": sorted({str(p.dtype) for p in parameters}),
            "buffer_dtypes": sorted({str(b.dtype) for b in buffers}),
            "eval_mode": not self.model.training,
            "output_unit": "same coordinate system as standardized input",
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
                    {"path": str(path), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}
                )
        if include_state_hash:
            digest = hashlib.sha256()
            for name, tensor in self.model.state_dict().items():
                array = tensor.detach().cpu().contiguous().numpy()
                digest.update(name.encode("utf-8"))
                digest.update(json.dumps([str(array.dtype), list(array.shape)]).encode("utf-8"))
                digest.update(array.tobytes())
            result["state_sha256"] = digest.hexdigest()
        return result
