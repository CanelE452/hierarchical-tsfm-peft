import numpy as np
import torch


def past_context(values, origin, device="cuda"):
    if origin < 48 or origin > values.shape[1]:
        raise ValueError("Origin cannot provide a full legal context")
    return torch.as_tensor(np.array(values[:, origin - 48:origin], copy=True), device=device, dtype=torch.float32)


@torch.no_grad()
def predict_origins(model, values, origins, parents, children, chunk_size=64):
    outputs = []
    for origin in origins:
        context = past_context(values, int(origin))
        outputs.append(model(context, parents, children, chunk_size).cpu().numpy())
    return np.stack(outputs)


def targets(values, origins):
    return np.stack([values[:, int(origin):int(origin) + 12] for origin in origins])
