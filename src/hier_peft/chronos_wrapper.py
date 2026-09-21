import inspect
import torch
from chronos import ChronosBoltPipeline
from huggingface_hub import HfApi, snapshot_download

from .audit import ROOT, RESULTS, sha256, utc_now, write_json

MODEL_ID = "amazon/chronos-bolt-small"


def download_model():
    info = HfApi().model_info(MODEL_ID)
    path = snapshot_download(MODEL_ID, revision=info.sha, cache_dir=ROOT / ".cache/huggingface",
                             allow_patterns=["*.json", "*.safetensors"])
    from pathlib import Path
    files = [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p), "bytes": p.stat().st_size}
             for p in Path(path).rglob("*") if p.is_file()]
    write_json(RESULTS / "MODEL_SOURCE_MANIFEST.json", {"model_id": MODEL_ID, "revision": info.sha,
               "download_timestamp": utc_now(), "snapshot_path": path, "files": files})
    return path


def load_pipeline(snapshot_path, device="cuda"):
    return ChronosBoltPipeline.from_pretrained(snapshot_path, device_map=device, torch_dtype=torch.float32)


def median_index(model):
    quantiles = list(model.chronos_config.quantiles)
    return quantiles.index(0.5)


def point_forecast(model, context):
    if context.ndim != 2 or context.shape[1] != 48:
        raise ValueError("Chronos screen requires exactly 48 past months")
    return model(context=context).quantile_preds[:, median_index(model), :12]


@torch.no_grad()
def patch_embeddings(model, context):
    if context.ndim != 2 or context.shape[1] != 48:
        raise ValueError("Context must have exactly 48 months")
    mask = ~torch.isnan(context)
    normalized, _ = model.instance_norm(context)
    patched = model.patch(normalized.to(model.dtype))
    patched_mask = torch.nan_to_num(model.patch(mask.to(model.dtype)), nan=0.0)
    patched = torch.where(patched_mask > 0, patched, 0)
    return model.input_patch_embedding(torch.cat([patched, patched_mask], dim=-1))
