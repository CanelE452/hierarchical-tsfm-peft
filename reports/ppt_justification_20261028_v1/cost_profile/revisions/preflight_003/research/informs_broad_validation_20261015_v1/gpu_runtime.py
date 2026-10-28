"""Explicit row microbatching for gradient-enabled frozen Chronos-Bolt.

Counts separate logical forward rows from physical executions (including
checkpoint recomputation). No dataset is opened by importing this module.
"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
from pathlib import Path
import hashlib
import importlib.util
import time
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

class RowForecaster(nn.Module):
    def __init__(self, predict_rows, horizon, row_microbatch=8, checkpoint_inputs=True):
        super().__init__()
        self.predict_rows=predict_rows
        self.horizon=int(horizon)
        self.row_microbatch=row_microbatch
        self.checkpoint_inputs=checkpoint_inputs
        self.streams=self.row_calls=self.physical_streams=self.physical_calls=0

    def _invoke(self, rows):
        self.physical_streams+=len(rows)
        self.physical_calls+=1
        return self.predict_rows(rows)

    def forward(self, x):
        batch,length,channels=x.shape
        rows=x.transpose(1,2).reshape(batch*channels,length)
        size=self.row_microbatch or len(rows)
        parts=[]
        for start in range(0,len(rows),size):
            chunk=rows[start:start+size]
            self.streams+=len(chunk)
            self.row_calls+=1
            if self.checkpoint_inputs and torch.is_grad_enabled() and chunk.requires_grad:
                result=checkpoint(self._invoke,chunk,use_reentrant=False,preserve_rng_state=True)
            else:
                result=self._invoke(chunk)
            if result.shape!=(len(chunk),self.horizon):
                raise ValueError("Foundation row/horizon output does not match contract")
            parts.append(result)
        return torch.cat(parts).reshape(batch,channels,self.horizon).transpose(1,2)


def state_hash(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class FrozenBolt(RowForecaster):
    def __init__(self, model_path, horizon, row_microbatch=8, checkpoint_inputs=True, device="cuda"):
        from chronos import ChronosBoltPipeline
        started=time.perf_counter()
        path=Path(model_path)
        if not path.is_dir():raise FileNotFoundError("Pinned local Bolt snapshot is absent")
        pipeline=ChronosBoltPipeline.from_pretrained(str(path),device_map=device,
            torch_dtype=torch.float32,local_files_only=True)
        model=pipeline.model.eval().requires_grad_(False)
        # Reuse the already-tested zero-variance derivative correction verbatim.
        source=Path(__file__).resolve().parent.parent/"temporal_frontend_campaign_20261015_v1/runtime.py"
        spec=importlib.util.spec_from_file_location("informs_norm_reference",source)
        reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
        norm=model.instance_norm
        model.instance_norm=reference.SafeInstanceNorm(eps=norm.eps,use_arcsinh=norm.use_arcsinh)
        median=list(model.chronos_config.quantiles).index(.5)
        if horizon>model.chronos_config.prediction_length:raise ValueError("H exceeds actual local Bolt output")
        def predict(rows):return model(context=rows).quantile_preds[:,median,:horizon]
        super().__init__(predict,horizon,row_microbatch,checkpoint_inputs)
        self.model=model
        self.model_revision=path.name
        self.initial_hash=state_hash(model)
        self.load_seconds=time.perf_counter()-started
        self.device=device
        self.instance_norm_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()

    def assert_frozen_unchanged(self):
        if state_hash(self.model)!=self.initial_hash:raise AssertionError("Frozen backbone weights changed")
        if any(p.grad is not None for p in self.model.parameters()):raise AssertionError("Frozen backbone acquired parameter gradients")
