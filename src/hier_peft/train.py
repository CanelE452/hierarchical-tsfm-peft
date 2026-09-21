import numpy as np
import torch

from .predict import past_context


def train_update(model, optimizer, context, target, scales, weights, parents, children, chunk_size=64):
    optimizer.zero_grad(set_to_none=True)
    summaries = model.prepare_context(context, parents, children)
    total_loss = 0.0
    for start in range(0, len(context), chunk_size):
        end = start + chunk_size
        prediction = model.forward_chunk(context[start:end], None if summaries is None else summaries[start:end])
        node_loss = (prediction - target[start:end]).square().mean(dim=1) / scales[start:end]
        loss = (node_loss * weights[start:end]).sum()
        if not torch.isfinite(loss).item():
            raise RuntimeError("BLOCKED_IMPLEMENTATION: nonfinite training loss")
        loss.backward()
        total_loss += loss.detach().item()
    parameters = [p for p in model.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
    optimizer.step()
    return total_loss, float(norm)


def training_tensors(values, origin, scales, weights, device="cuda"):
    context = past_context(values, origin, device)
    target = torch.as_tensor(np.array(values[:, origin:origin + 12], copy=True), device=device, dtype=torch.float32)
    return (context, target, torch.as_tensor(scales, device=device, dtype=torch.float32),
            torch.as_tensor(weights, device=device, dtype=torch.float32))
