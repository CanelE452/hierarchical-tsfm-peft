"""Matched bias-free learned E/D around the literal parent frozen Bolt."""
import copy
import hashlib

import numpy as np
import torch
from torch import nn

from runtime_v18 import artifact, load_data, load_original, original_levels, resolve


SCHEMA = "matched_learned_ed_v18_model_1"
DIMENSIONS = {"robin": (17, 5), "peacock_education": (13, 4), "jena": (21, 6)}
PARITY = {"atol": 1e-5, "rtol": 1e-4}
REPLAY = {"atol": 1e-6, "rtol": 0}
GRADIENT_PARITY = {"atol": 1e-6, "rtol": 1e-4}


def tensor_hashes(state):
    return {name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
            for name, value in state.items()}


def state_hash(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def compute_loss(pred, target, mask, channel_count=None):
    if pred.ndim != 3 or pred.shape != target.shape or pred.shape != mask.shape:
        raise ValueError("Prediction/target/mask must share [B,48,C] axes")
    observed = mask.bool()
    count = observed.sum((0, 1)) if channel_count is None else channel_count
    valid = count > 0
    if not bool(valid.any()):
        raise ValueError("No observed target in effective batch")
    error = torch.where(observed, pred - target, torch.zeros_like(pred))
    return (error.square().sum((0, 1))[valid] / count[valid]).mean()


class LearnedED(nn.Module):
    def __init__(self, dataset, basis, backbone, parent):
        super().__init__()
        self.dataset = dataset
        self.channels, self.latent = DIMENSIONS[dataset]
        basis = torch.as_tensor(basis, dtype=torch.float32, device="cpu").clone()
        if tuple(basis.shape) != (self.channels, self.latent):
            raise ValueError("Literal parent PCA dimensions differ")
        self.register_buffer("basis0", basis)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(0)
            self.encoder = nn.Linear(self.channels, self.latent, bias=False)
            self.decoder = nn.Linear(self.latent, self.channels, bias=False)
        with torch.no_grad():
            self.encoder.weight.copy_(basis.T)
            self.decoder.weight.copy_(basis)
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.median = list(backbone.chronos_config.quantiles).index(0.5)
        self.parent = copy.deepcopy(parent)
        self.chunk_rows = None

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def backbone_point(self, z):
        batch, length, channels = z.shape
        if self.chunk_rows is not None:
            raise ValueError("Primary matched E/D uses the parent full independent-row path")
        output = self.backbone(context=z.transpose(1, 2).reshape(batch * channels, length))
        return output.quantile_preds[:, self.median, :48].reshape(batch, channels, 48).transpose(1, 2)

    def forward(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (512, self.channels) or x.dtype != torch.float32:
            raise ValueError("Expected standardized FP32 [B,512,C] input")
        return self.decoder(self.backbone_point(self.encoder(x)))

    def adapter_state(self):
        return {name: value.detach().cpu().clone() for name, value in
                (("encoder.weight", self.encoder.weight), ("decoder.weight", self.decoder.weight))}

    def restore_adapter(self, state):
        if set(state) != {"encoder.weight", "decoder.weight"}:
            raise ValueError("Only the two learned E/D weight tensors may be restored")
        self.load_state_dict(state, strict=False)

    def model_config(self):
        return {"schema": SCHEMA, "dataset": self.dataset, "channels": self.channels, "latent": self.latent,
                "context": 512, "horizon": 48, "precision": "float32", "bias": False,
                "point": "native inverse-scaled q0.5, first48; B*K independent latent rows",
                "backbone_source": self.parent, "backbone_eval": True, "backbone_weights_trainable": False,
                "backbone_autograd_operations": True, "G": None, "level_persistence": False,
                "reconstruction_auxiliary_coefficient": 0.0,
                "normalization_forward": self.backbone.instance_norm.forward.__func__.__name__}

    def receipt(self, include_state_hash=False):
        parameters = list(self.named_parameters())
        buffers = list(self.named_buffers())
        trainable = [(name, value) for name, value in parameters if value.requires_grad]
        parameter_bytes = sum(p.numel() * p.element_size() for _, p in parameters)
        buffer_bytes = sum(p.numel() * p.element_size() for _, p in buffers)
        receipt = {**self.model_config(), "parameter_count": sum(p.numel() for _, p in parameters),
                   "trainable_parameter_count": sum(p.numel() for _, p in trainable),
                   "trainable_parameter_names": [name for name, _ in trainable],
                   "parameter_bytes": parameter_bytes, "buffer_bytes": buffer_bytes,
                   "total_tensor_bytes": parameter_bytes + buffer_bytes,
                   "quantiles": list(self.backbone.chronos_config.quantiles), "median_index": self.median,
                   "model_context_length": self.backbone.chronos_config.context_length,
                   "model_prediction_length": self.backbone.chronos_config.prediction_length,
                   "basis_values_sha256": tensor_hashes({"basis": self.basis0})["basis"],
                   "eval_mode": not self.training, "backbone_eval_mode": not self.backbone.training,
                   "all_parameters_fp32": all(p.dtype == torch.float32 for _, p in parameters)}
        if include_state_hash:
            receipt.update(state_sha256=state_hash(self), backbone_state_sha256=state_hash(self.backbone),
                           adapter_tensor_sha256=tensor_hashes(self.adapter_state()))
        return receipt


def build_model(dataset, device="cuda"):
    data = load_data(dataset, include_test=False)
    rows = original_levels(dataset)
    if len(rows) != 2:
        raise ValueError("Exactly two selected ORIGINAL_LEVEL parent instances are required")
    original = load_original(rows[0], device=device, deploy=True)
    basis = torch.as_tensor(data["basis"], dtype=torch.float32)
    if not torch.equal(original.encoder.weight.detach().cpu(), basis.T):
        raise ValueError("Parent encoder does not use literal TRAIN PCA transpose")
    if not torch.equal(original.decoder.weight.detach().cpu(), basis):
        raise ValueError("Parent decoder does not use literal TRAIN PCA")
    backbone = original.backbone
    del original.backbone
    del original
    model = LearnedED(dataset, data["basis"], backbone, rows[0]).to(device)
    if {name for name, p in model.named_parameters() if p.requires_grad} != {"encoder.weight", "decoder.weight"}:
        raise ValueError("Trainable parameters must be E/D only")
    if model.receipt()["trainable_parameter_count"] != 2 * np.prod(DIMENSIONS[dataset]):
        raise ValueError("Actual learned E/D parameter enumeration differs")
    return model


def make_checkpoint(model, spec, **extra):
    return {"schema": SCHEMA, "spec": copy.deepcopy(spec), "basis": model.basis0.detach().cpu().clone(),
            "state": model.adapter_state(), "model_config": model.model_config(), **extra}


def restore_model(checkpoint, device="cuda", expected_sha256=None):
    receipt = artifact(checkpoint, expected_sha256)
    saved = torch.load(resolve(receipt["path"]), map_location="cpu", weights_only=False)
    if saved.get("schema") != SCHEMA:
        raise ValueError("Not a matched E/D checkpoint")
    model = build_model(saved["spec"]["dataset"], device=device)
    if not torch.equal(saved["basis"], model.basis0.cpu()) or saved["model_config"] != model.model_config():
        raise ValueError("Saved checkpoint PCA/backbone/model configuration changed")
    model.restore_adapter(saved["state"])
    return model.eval()
