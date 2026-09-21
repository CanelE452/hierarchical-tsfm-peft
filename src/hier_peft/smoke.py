import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from .audit import ROOT, RESULTS, sha256, utc_now, write_json
from .chronos_wrapper import load_pipeline, point_forecast, patch_embeddings, median_index
from .data import load_and_audit
from .metrics import seasonal_scales, level_weights
from .model import ForecastModel, tensor_state_hash
from .predict import past_context
from .train import train_update, training_tensors

ARMS = ["LORA", "LORA_SELF", "LORA_POOL", "LORA_HIER"]


def run_smoke():
    started = time.monotonic()
    attempt = utc_now()
    write_json(RESULTS / "SMOKE_AUDIT.json", {"status": "RUNNING", "attempt": attempt, "main_training_allowed": False})
    ds = load_and_audit("Labour", write_artifacts=False)
    snapshot = json.loads((RESULTS / "MODEL_SOURCE_MANIFEST.json").read_text())["snapshot_path"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pipeline = load_pipeline(snapshot)
    pipeline.model.eval()
    parity = []
    for origin in [48, 96, ds.split_manifest["origins"]["TRAIN"][-1]]:
        context = past_context(ds.values, origin)
        original = context.clone()
        with torch.no_grad():
            official = pipeline.predict(context, prediction_length=12)[:, median_index(pipeline.model)]
            local = point_forecast(pipeline.model, context).cpu()
            torch.testing.assert_close(local, official, rtol=1e-6, atol=1e-4)
            captured = []
            hook = pipeline.model.input_patch_embedding.register_forward_hook(lambda mod, args, out: captured.append(out.clone()))
            point_forecast(pipeline.model, context)
            hook.remove()
            extracted = patch_embeddings(pipeline.model, context)
            torch.testing.assert_close(extracted, captured[0], rtol=0, atol=0)
        assert torch.equal(context, original)
        parity.append({"origin": int(origin), "max_abs_diff": float((local - official).abs().max()),
                       "nodes": len(context), "horizon": 12, "patch_embedding_exact": True})
    del pipeline
    gc.collect()
    torch.cuda.empty_cache()
    train_end = ds.split_manifest["role_bounds"]["TRAIN"]["end_exclusive"]
    scales, scale_audit = seasonal_scales(ds.values[:, :train_end])
    weights = level_weights(len(ds.ids), ds.tags)
    origin = ds.split_manifest["origins"]["TRAIN"][0]
    x, y, d, w = training_tensors(ds.values, origin, scales, weights)
    init_checks, update_checks = [], []
    for seed in [92110, 92111, 92112]:
        lora_prediction = None
        lora_hash = None
        adapter_hash = None
        for arm in ARMS:
            model = ForecastModel(snapshot, arm, seed)
            frozen_before = model.frozen_hash()
            state0 = model.trainable_state()
            this_lora_hash = tensor_state_hash({n: p for n, p in state0.items() if "lora_" in n})
            with torch.no_grad():
                initial = model(x, ds.parents, ds.children)
            if arm == "LORA":
                lora_prediction = initial.clone()
                lora_hash = this_lora_hash
            else:
                this_adapter_hash = tensor_state_hash(model.adapter.state_dict())
                if adapter_hash is None:
                    adapter_hash = this_adapter_hash
                assert this_adapter_hash == adapter_hash
                assert this_lora_hash == lora_hash
                torch.testing.assert_close(initial, lora_prediction, rtol=1e-6, atol=1e-4)
            init_checks.append({"seed": seed, "arm": arm, "lora_hash": this_lora_hash,
                                "adapter_hash": adapter_hash if model.adapter else None,
                                "step0_max_abs_diff_to_lora": float((initial - lora_prediction).abs().max())})
            if seed == 92110:
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0)
                for step in range(2):
                    before_time = time.monotonic()
                    loss, norm = train_update(model, optimizer, x, y, d, w, ds.parents, ds.children, chunk_size=32)
                    torch.cuda.synchronize()
                    entry = {"kind": "smoke", "attempt": attempt, "seed": seed, "arm": arm,
                             "update": step + 1, "loss": loss, "grad_norm": norm,
                             "seconds": time.monotonic() - before_time, "main_budget_consumed": 0}
                    with (RESULTS / "UPDATE_LEDGER.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(entry) + "\n")
                state1 = model.trainable_state()
                changed = [n for n in state0 if not torch.equal(state0[n], state1[n])]
                assert any("lora_" in n for n in changed)
                if model.adapter:
                    assert any(n.startswith("adapter.") for n in changed)
                assert model.frozen_hash() == frozen_before
                with torch.no_grad():
                    after = model(x, ds.parents, ds.children)
                    perm = np.random.default_rng(123).permutation(len(x))
                    reverse = np.argsort(perm)
                    parents = {j: tuple(int(reverse[p]) for p in ds.parents[int(i)]) for j, i in enumerate(perm)}
                    children = {j: tuple(int(reverse[p]) for p in ds.children[int(i)]) for j, i in enumerate(perm)}
                    permuted = model(x[perm], parents, children)[reverse]
                    torch.testing.assert_close(after, permuted, rtol=2e-5, atol=2e-3)
                    chunked = model(x, ds.parents, ds.children, chunk_size=17)
                    torch.testing.assert_close(after, chunked, rtol=2e-5, atol=2e-3)
                ckpt = ROOT / ".cache" / f"smoke_{arm}.pt"
                torch.save(state1, ckpt)
                model.restore_trainable(state0)
                model.restore_trainable(torch.load(ckpt, weights_only=True))
                with torch.no_grad():
                    restored = model(x, ds.parents, ds.children)
                torch.testing.assert_close(after, restored, rtol=0, atol=0)
                assert model.frozen_hash() == frozen_before
                update_checks.append({"arm": arm, "updates": 2, "loss": loss, "grad_norm": norm,
                                      "changed_parameters": changed, "frozen_hash_before_after": frozen_before,
                                      "permutation_max_abs": float((after-permuted).abs().max()),
                                      "chunking_max_abs": float((after-chunked).abs().max()),
                                      "restore_max_abs": float((after-restored).abs().max()), "checkpoint_sha256": sha256(ckpt)})
                del optimizer
            del model
            gc.collect()
            torch.cuda.empty_cache()
    # Changing every value at and after origin cannot change the context tensor.
    poisoned = ds.values.copy()
    poisoned[:, origin:] = 1e20
    torch.testing.assert_close(x, past_context(poisoned, origin), rtol=0, atol=0)
    torch.testing.assert_close(x, past_context(ds.values, origin), rtol=0, atol=0)
    report = {"status": "PASS", "attempt": attempt, "main_training_allowed": True,
              "actual_dataset": "Labour", "f0_parity": parity, "initialization_checks": init_checks,
              "optimizer_checks": update_checks, "scale_audit": scale_audit,
              "context_future_poison_invariance": True, "input_unchanged": True,
              "precision": "float32", "tf32": False,
              "tolerances": {"f0_and_step0": {"rtol": 1e-6, "atol": 1e-4},
                             "permutation_and_chunking": {"rtol": 2e-5, "atol": 2e-3}, "restore": "exact"},
              "smoke_updates_this_attempt": 8, "main_updates": 0,
              "seconds": time.monotonic() - started, "peak_cuda_bytes": torch.cuda.max_memory_allocated()}
    write_json(RESULTS / "SMOKE_AUDIT.json", report)
    return report
