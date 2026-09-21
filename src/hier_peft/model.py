import hashlib
import torch
from torch import nn

from .adapters import ResidualAdapter, context_embeddings
from .chronos_wrapper import load_pipeline, patch_embeddings, point_forecast
from .lora import attach_lora


def tensor_state_hash(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class ForecastModel(nn.Module):
    def __init__(self, snapshot, arm, seed, device="cuda"):
        super().__init__()
        self.arm = arm
        pipeline = load_pipeline(snapshot, device)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.backbone, self.lora_names = attach_lora(pipeline.model)
        self.core = self.backbone.get_base_model()
        self.adapter = None
        if arm != "LORA":
            # Same LoRA construction precedes this draw in all three adapter arms.
            self.adapter = ResidualAdapter().to(device)
        self.backbone.eval()

    def frozen_hash(self):
        return tensor_state_hash({n: p for n, p in self.backbone.named_parameters() if not p.requires_grad})

    def trainable_state(self):
        return {n: p.detach().cpu().clone() for n, p in self.named_parameters() if p.requires_grad}

    def restore_trainable(self, state):
        expected = {n: p for n, p in self.named_parameters() if p.requires_grad}
        if expected.keys() != state.keys():
            raise ValueError("Checkpoint trainable keys mismatch")
        with torch.no_grad():
            for name, param in expected.items():
                param.copy_(state[name])

    def prepare_context(self, context, parents, children):
        if self.adapter is None:
            return None
        with torch.no_grad():
            h = patch_embeddings(self.core, context)
            return context_embeddings(h, self.arm, parents, children)

    def forward_chunk(self, context, contexts=None):
        if self.adapter is None:
            return point_forecast(self.core, context)
        if contexts is None or len(contexts) != len(context):
            raise ValueError("Missing all-node past-only context summary")
        hook = self.core.input_patch_embedding.register_forward_hook(
            lambda module, args, output: self.adapter(output, contexts))
        try:
            return point_forecast(self.core, context)
        finally:
            hook.remove()

    def forward(self, context, parents, children, chunk_size=64):
        summaries = self.prepare_context(context, parents, children)
        return torch.cat([self.forward_chunk(context[i:i + chunk_size],
                          None if summaries is None else summaries[i:i + chunk_size])
                          for i in range(0, len(context), chunk_size)])
