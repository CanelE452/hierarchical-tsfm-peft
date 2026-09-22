"""Shared paths, sealing, timing and forecasting helpers.

Reuses the existing gap_screen (internal_adaptation_gap_v1_20260922) for data, model
loading, native loss, metrics and the update journal. Nothing here trains a model.
"""
from __future__ import annotations
import contextlib, gc, hashlib, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]            # experiments/<eid>
REPO = HERE.parents[1]
CFG = json.loads((HERE/'config.json').read_text(encoding='utf-8'))
SET = json.loads((HERE/'settings.json').read_text(encoding='utf-8'))
EID = CFG['experiment_id']
GAP = REPO/'experiments'/SET['reused_experiment']
GAP_RESULTS = REPO/'results'/SET['reused_experiment']
OUT = REPO/'results'/EID
CACHE = REPO/'.cache'/EID
for p in (str(GAP), str(HERE)):
    if p not in sys.path: sys.path.insert(0, p)

from gap_screen.io import (Blocked, read_json, write_json, sha, digest_object, write_csv,
                           run_command, Ledger, require_complete_ledger)
from gap_screen.data import read_panel, prepare_frame, load_raw, save_panel
from gap_screen.model import (configure_runtime, autocast, sync, normalized_loss, raw_inverse,
                              to_quantiles, freeze_stochastic, fingerprint, load_base, lora_targets,
                              seed_all, enable_checkpointing)
from gap_screen.metrics import score, scalar_primary, gain, paired_bootstrap, choose_checkpoint
import reference_core as rc

BASECFG = json.loads((GAP/'config.json').read_text(encoding='utf-8'))   # data split, model pin, precision
DEVICE = 'cuda'
ARMS = CFG['arms']
F_ARMS = ('F_FULL', 'F_TOP3')
E_ARMS = ('E_EMLOC', 'E_DELTA', 'E_VALUE_DELTA', 'E_RESPONSE_DELTA')
DELTA_ARMS = ('E_DELTA', 'E_VALUE_DELTA', 'E_RESPONSE_DELTA')
EMULATOR_OF = {'E_EMLOC': 'svd', 'E_DELTA': 'svd', 'E_VALUE_DELTA': 'value', 'E_RESPONSE_DELTA': 'response'}
SCALING = CFG['lora_alpha']/CFG['lora_rank']

TRAINING_CODE = ['config.json', 'settings.json', 'reference_core.py', 'run.py',
                 'emulator_screen/__init__.py', 'emulator_screen/common.py',
                 'emulator_screen/modules.py', 'emulator_screen/stages.py',
                 'emulator_screen/decision.py']
REPORTING_CODE = ['emulator_screen/report.py', 'emulator_screen/verify.py', 'figures.py', 'publish.py']


def code_hashes(names):
    return {n: sha(HERE/n) for n in names if (HERE/n).exists()}


def reused_hashes():
    files = {f'experiments/{SET["reused_experiment"]}/{n}': sha(GAP/n) for n in SET['reused_gap_screen_files']}
    files.update({f'results/{SET["reused_experiment"]}/{n}': sha(GAP_RESULTS/n) for n in SET['reused_result_files']})
    return files


def assert_seal():
    """Training/selection code and config must equal the sealed state before any GPU stage."""
    seal = read_json(OUT/'SOURCE_SEAL.json')
    if code_hashes(TRAINING_CODE) != seal['training_code']:
        raise Blocked('SOURCE_SEAL_CHANGED: training/selection code differs from the seal')
    if reused_hashes() != seal['reused']:
        raise Blocked('REUSED_SOURCE_CHANGED: gap_screen or prior Jena audit changed')
    if digest_object(CFG) != seal['config_hash'] or digest_object(SET) != seal['settings_hash']:
        raise Blocked('CONFIG_CHANGED_AFTER_SEAL')


def stage_done(name):
    return (CACHE/'stages'/f'{name}.json').exists()


def mark_stage(name, info):
    write_json(CACHE/'stages'/f'{name}.json', dict(info, complete=True, utc=time.time()))


class Clock:
    """Synchronized wall-clock parts in seconds; names are the cost-ledger components."""
    def __init__(self):
        self.parts = {}

    @contextlib.contextmanager
    def part(self, name):
        sync(DEVICE); t = time.perf_counter()
        try:
            yield
        finally:
            sync(DEVICE); self.parts[name] = self.parts.get(name, 0.) + time.perf_counter() - t


def gpu_reset():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


def gpu_peak():
    return dict(peak_allocated_bytes=int(torch.cuda.max_memory_allocated()),
                peak_reserved_bytes=int(torch.cuda.max_memory_reserved()))


def process_rss():
    import psutil
    return int(psutil.Process().memory_info().rss)


class Forecaster(torch.nn.Module):
    """Same encode -> output head -> raw inverse path as gap_screen.model.Predictor, without injecting PEFT."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.nq = model.num_quantiles
        self.patch = model.chronos_config.output_patch_size
        self.arcsinh = model.chronos_config.use_arcsinh

    def forward(self, x, groups, horizon):
        count = math.ceil(horizon/self.patch)
        enc, (loc, scale), _, _ = self.model.encode(context=x, group_ids=groups, num_output_patches=count)
        h = enc.last_hidden_state[:, -count:]
        q = to_quantiles(self.model.output_patch_embedding(h), self.nq, self.patch)
        return q, raw_inverse(q, loc, scale, self.arcsinh)[..., :horizon], loc, scale


def tensor_batch(panel, origins, context=None):
    x, y, g = panel.batch(origins, context)
    return [torch.as_tensor(z, device=DEVICE) for z in (x, y, g)]


@torch.no_grad()
def predict(fc, panel, origins, horizon=None):
    """Raw quantile forecasts [N, C, Q, H] plus loc/scale [N, C, 1], one origin (all channels) per forward."""
    horizon = horizon or panel.horizon
    raws, locs, scales = [], [], []
    for o in origins:
        x, _, g = tensor_batch(panel, [o])
        with autocast(DEVICE, BASECFG['precision']):
            _, raw, loc, scale = fc(x, g, horizon)
        raws.append(raw.float().cpu().numpy()); locs.append(loc.float().cpu().numpy()); scales.append(scale.float().cpu().numpy())
    return np.stack(raws), np.stack(locs), np.stack(scales)


def levels_of(model):
    return model.quantiles.detach().float().cpu().numpy()


def sigma_mse(delta, sigma):
    """reference_core.weighted_mse over numpy [C,Q,H] or batched [N,C,Q,H] (mean over all but series scale)."""
    d = np.asarray(delta, dtype=np.float64)/np.asarray(sigma, dtype=np.float64)[..., :, None, None]
    return float((d**2).mean())


def lora_bytes(state):
    return int(sum(v.numel()*v.element_size() for v in state.values()))


def tensor_digest(state):
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode()); h.update(state[k].detach().cpu().contiguous().float().numpy().tobytes())
    return h.hexdigest()
