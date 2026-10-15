"""Use the trainer's exact 32-window arithmetic for saved native predictions."""
import numpy as np
import torch

import native_head_v1 as nh


def native_components(model, history):
    latent = nh.cache_latents(model, {"history": history}, "deployment", None)
    futures, predictions = [], []
    with torch.no_grad():
        for start in range(0, len(history), 32):
            z = latent[start:start + 32]
            future = model.head(z)
            last = torch.from_numpy(np.ascontiguousarray(history[start:start + 32, -1:]))
            predictions.append(model.projection.forecast(last, future, z[:, -1:]))
            futures.append(future)
    return tuple(np.ascontiguousarray(value.cpu().numpy())
                 for value in (torch.cat(predictions), latent, torch.cat(futures)))
