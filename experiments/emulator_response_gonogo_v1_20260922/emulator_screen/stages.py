"""Stages of the emulator-response go/no-go screen. Each GPU stage asserts the source seal first.

Cold-start cost rule (fixed before any fit; see PROTOCOL_KO.md section 8):
  every arm      : data_prepare + (fit process) model load(s) + 128 updates + checkpoint I/O
                   + transfer/correction + full-model VAL at 0/16/32/64/128 + selection/export
  every E arm    : + full_model_load + activation_collection + svd_factorization + emulator_build (compress stage)
  VALUE/RESPONSE : + teacher forwards on calibration origins + initial-error forwards + 16 gamma updates
  delta arms     : + F0 TRAIN/VAL cache + own E0 TRAIN/VAL cache (contract section 5)
Model reloads inside the calibrate/caches stages are process artifacts (the cold pipeline already holds
those models) and are recorded only in the experiment ledger. Diagnostics (emulator-only VAL, naive EMLoC
transfer, heldout response probes, preflight) are never charged.

Every preflight stage runs all journal-free checks first; the limited smoke updates are the last action.
Calibration forwards (teacher and emulator) run in FP32: under BF16 autocast the 1e-4/1e-3 probe responses
were measured to be rounding noise before sealing (COMPAT_REVISION_KO.md).
"""
from __future__ import annotations
import contextlib, gc, math, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from .common import *
from .common import rc
from .modules import *

H_CKPTS = CFG['checkpoints']


def _meta():
    return read_json(CACHE/'emulator'/'meta.json')


def _panel():
    return read_panel(CACHE/'panel.npz')


def calib_ctx():
    """Calibration precision (settings.calibration_precision): 'fp32' = no autocast for teacher and emulator alike."""
    if SET['calibration_precision'] == 'fp32': return contextlib.nullcontext()
    return autocast(DEVICE, BASECFG['precision'])


def select_origin_sets(panel):
    tr = panel.origins['train']; n = len(tr)
    cal_idx = np.linspace(0, n-1, CFG['calibration_origins']).round().astype(int)
    if len(set(cal_idx.tolist())) != CFG['calibration_origins']: raise Blocked('CALIBRATION_ORIGIN_DUPLICATE')
    rest = np.setdiff1d(np.arange(n), cal_idx)
    diag_idx = rest[np.linspace(0, len(rest)-1, CFG['diagnostic_origins']).round().astype(int)]
    if len(set(diag_idx.tolist())) != CFG['diagnostic_origins'] or set(diag_idx.tolist()) & set(cal_idx.tolist()):
        raise Blocked('DIAGNOSTIC_ORIGIN_OVERLAP')
    return tr[cal_idx], tr[diag_idx], cal_idx, diag_idx


# ============================================================ data
def stage_prepare():
    assert_seal(); clock = Clock()
    src = Path(SET['jena_file'])
    if not src.is_file(): raise Blocked('BLOCKED_DATA: local Jena 2024 CSV missing')
    digest = sha(src)
    if digest != SET['jena_sha256']: raise Blocked('BLOCKED_DATA: Jena source hash differs from the audited file')
    with clock.part('data_prepare'):
        panel = prepare_frame(load_raw(src, 'jena'), 'jena', BASECFG,
                              dict(source='explicit local official archive CSV', sha256=digest, bytes=src.stat().st_size))
    old = read_json(GAP_RESULTS/'jena'/'DATA_AUDIT.json')
    prior = read_panel(REPO/SET['reused_panel_cache'])
    checks = dict(values_sha256=panel.metadata['values_sha256'] == old['values_sha256'],
                  scale_sha256=panel.metadata['scale_sha256'] == old['scale_sha256'],
                  columns=panel.columns == old['columns'],
                  bounds={k: tuple(v) for k, v in panel.metadata['bounds'].items()} == {k: tuple(v) for k, v in old['bounds'].items()},
                  origins={r: bool(np.array_equal(panel.origins[r], prior.origins[r])) for r in ('train', 'val', 'test')},
                  values_equal_prior_cache=bool(np.array_equal(panel.values, prior.values, equal_nan=True)),
                  sigma_equal_prior_cache=bool(np.array_equal(panel.sigma, prior.sigma)),
                  context=panel.context == CFG['context'], horizon=panel.horizon == CFG['horizon'])
    flat = [v for v in checks.values() if not isinstance(v, dict)] + list(checks['origins'].values())
    if not all(flat): raise Blocked(f'BLOCKED_DATA: panel differs from the prior audited Jena panel {checks}')
    panel.metadata['data_prepare_s'] = clock.parts['data_prepare']
    save_panel(panel, CACHE/'panel.npz')
    write_json(OUT/'DATA_MANIFEST.json', dict(
        source=str(src), source_sha256=digest, source_bytes=src.stat().st_size,
        provenance='official Jena MPI roof 2024 archive CSV (both official URLs returned 404 on 2026-09-22; see prior COMPAT_REVISION_KO.md)',
        reused_prior_audit=f'results/{SET["reused_experiment"]}/jena/DATA_AUDIT.json', prior_match=checks,
        rows=panel.metadata['rows'], channels=len(panel.columns), columns=panel.columns,
        context=panel.context, horizon=panel.horizon, bounds=panel.metadata['bounds'],
        origins={r: len(panel.origins[r]) for r in ('train', 'val', 'test')}, eligibility=panel.metadata['eligibility'],
        values_sha256=panel.metadata['values_sha256'], scale_sha256=panel.metadata['scale_sha256'],
        data_prepare_s=clock.parts['data_prepare'], panel_cache_sha256=sha(CACHE/'panel.npz'),
        note='Jena TEST was used in earlier development; this is not independent confirmation.'))
    mark_stage('prepare', dict(parts=clock.parts))


# ============================================================ smoke helper (the only optimizer updates outside main/calibration)
def _smoke_updates(model, fcm, panel, arm, ledger, levels, delta=None):
    """Two real optimizer updates on a throwaway copy: B-grad at zero init, then A and B both connected."""
    opt = loraplus_optimizer(model); rep = {}
    origins = panel.origins['train'][:2]
    for step, o in enumerate(origins, 1):
        opt.zero_grad(set_to_none=True)
        x, y, g = tensor_batch(panel, [o])
        with autocast(DEVICE, BASECFG['precision']):
            q, raw, loc, scale = fcm(x, g, panel.horizon)
            if delta is None:
                loss = normalized_loss(q, y, loc, scale, levels, fcm.arcsinh)
            else:
                f0, e0 = delta[int(o)]      # precomputed zero-LoRA forecasts; never recomputed inside a live graph
                qn = rc.raw_to_native(rc.anchored_raw(f0, raw.float(), e0), loc, scale, fcm.arcsinh)
                loss = normalized_loss(qn, y, loc, scale, levels, fcm.arcsinh)
        loss.backward()
        a = [p.grad for n, p in model.named_parameters() if p.requires_grad and '.lora_A.' in n]
        b = [p.grad for n, p in model.named_parameters() if p.requires_grad and '.lora_B.' in n]
        if any(t is None or not torch.isfinite(t).all() for t in a+b): raise Blocked(f'SMOKE_BAD_GRADIENT {arm}')
        a_nz = sum(bool(t.abs().sum() > 0) for t in a); b_nz = sum(bool(t.abs().sum() > 0) for t in b)
        rep[f'step{step}'] = dict(loss=float(loss), A_nonzero_modules=a_nz, B_nonzero_modules=b_nz, modules=len(b))
        if step == 1 and (b_nz != len(b) or a_nz != 0): raise Blocked(f'SMOKE_ZERO_INIT_GRADIENT_PATTERN {arm}: A{a_nz} B{b_nz}')
        if step == 2 and (a_nz != len(a) or b_nz != len(b)): raise Blocked(f'SMOKE_AB_NOT_CONNECTED {arm}: A{a_nz} B{b_nz}')
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], CFG['gradient_clip'])
        ledger.intent(f'smoke_{arm}', step); opt.step(); sync(DEVICE); ledger.commit(f'smoke_{arm}', step)
    return dict(smoke=rep, smoke_updates=len(origins))


# ============================================================ preflight on the original model
def stage_preflight_base():
    assert_seal(); panel = _panel(); rt, at = SET['prediction_rtol'], SET['prediction_atol']
    full, _ = load_base(BASECFG, DEVICE); fc = Forecaster(full); levels = full.quantiles.detach()
    patch = full.chronos_config.output_patch_size; H = panel.horizon; rep = {}
    canon = canonical_map(full); tops = top3_names(canon)
    shapes = {n: (get(full, n).out_features, get(full, n).in_features) for n in canon}
    if len(canon) != 97 or len([n for n in canon if n.startswith('encoder.')]) != 96: raise Blocked('LORA_MODULE_COUNT_CHANGED')
    (CACHE/'init').mkdir(parents=True, exist_ok=True); inits = {}
    for s in CFG['seeds']:
        path = CACHE/'init'/f'seed_{s}.pt'
        if not path.exists(): torch.save(canonical_init(s, shapes), path)
        inits[s] = dict(path=path.relative_to(REPO).as_posix(), sha256=sha(path))
    o1 = panel.origins['train'][:1]; x, y, g = tensor_batch(panel, o1)
    with torch.no_grad(), autocast(DEVICE, BASECFG['precision']):
        q, raw, loc, scale = fc(x, g, H)
        native = full(context=x, group_ids=g, num_output_patches=H//patch, future_target=y)
        torch.testing.assert_close(raw, native.quantile_preds[..., :H], rtol=rt, atol=at)
        ours = normalized_loss(q, y, loc, scale, levels, full.chronos_config.use_arcsinh)
        torch.testing.assert_close(ours, native.loss, rtol=1e-5, atol=1e-5)
        from_raw = rc.native_loss_from_raw(raw.float(), y, loc, scale, levels, full.chronos_config.use_arcsinh)
        torch.testing.assert_close(from_raw, ours.float(), rtol=rt, atol=at)
        back = rc.raw_to_native(raw.float(), loc, scale, full.chronos_config.use_arcsinh)
        roundtrip = float((back - q.float()).abs().max())
        torch.testing.assert_close(back, q.float(), rtol=rt, atol=at)
        poisoned = full(context=x, group_ids=g, num_output_patches=H//patch, future_target=y+987.)
        torch.testing.assert_close(native.quantile_preds, poisoned.quantile_preds, rtol=0, atol=0)
        xx, _, gg = tensor_batch(panel, panel.origins['train'][:2]); c = len(panel.columns)
        a1 = fc(xx, gg, H)[1]; xx2 = xx.clone(); xx2[c:] += 1000.; b1 = fc(xx2, gg, H)[1]
        torch.testing.assert_close(a1[:c], b1[:c], rtol=rt, atol=at)
    f0 = raw.detach().clone()
    rep['native'] = dict(max_abs_raw=float((raw - native.quantile_preds[..., :H]).abs().max()),
                         loss_ours=float(ours), loss_official=float(native.loss), loss_from_raw=float(from_raw),
                         raw_native_roundtrip_max_abs=roundtrip, future_target_poison_invariant=True, group_isolation=True)
    # PEFT parity: additive adapter vs PEFT LoRA with identical A/B and alpha/r
    attach_adapters(full, canon); st = canonical_init(CFG['seeds'][0], shapes)
    gen = torch.Generator().manual_seed(7)
    for n in canon: st[n+'.lora_B.weight'] = torch.randn(st[n+'.lora_B.weight'].shape, generator=gen)*1e-3
    load_lora(full, st)
    with torch.no_grad(), autocast(DEVICE, BASECFG['precision']): ours_lora = fc(x, g, H)[1].detach().clone()
    del fc, full; gc.collect(); torch.cuda.empty_cache()
    from peft import LoraConfig, inject_adapter_in_model
    ref, _ = load_base(BASECFG, DEVICE)
    inject_adapter_in_model(LoraConfig(r=CFG['lora_rank'], lora_alpha=CFG['lora_alpha'], lora_dropout=0., bias='none',
                                       target_modules=canon), ref)
    with torch.no_grad():
        for n in canon:
            m = get(ref, n)
            if float(m.scaling['default']) != SCALING: raise Blocked('PEFT_SCALING_MISMATCH')
            m.lora_A['default'].weight.copy_(st[n+'.lora_A.weight']); m.lora_B['default'].weight.copy_(st[n+'.lora_B.weight'])
    with torch.no_grad(), autocast(DEVICE, BASECFG['precision']): peft_lora = Forecaster(ref)(x, g, H)[1]
    torch.testing.assert_close(ours_lora, peft_lora, rtol=rt, atol=at)
    rep['peft_parity'] = dict(max_abs=float((ours_lora - peft_lora).abs().max()), peft_scaling=SCALING, modules=len(canon))
    del ref; gc.collect(); torch.cuda.empty_cache()
    # journal-free checks for both F arms first, smoke updates last
    models = {}
    for arm, names in (('F_FULL', canon), ('F_TOP3', tops)):
        model, _ = load_base(BASECFG, DEVICE); bb0 = backbone_digest(model)
        init0 = torch.load(CACHE/'init'/f'seed_{CFG["seeds"][0]}.pt', weights_only=True)
        attach_adapters(model, names); load_lora(model, {k: v for k, v in init0.items() if k.rsplit('.', 2)[0] in names})
        trainable = set_trainable_lora(model, names)
        if arm == 'F_TOP3': install_top3_prefix(model)
        else: enable_checkpointing(model)
        fcm = Forecaster(model)
        with torch.no_grad(), autocast(DEVICE, BASECFG['precision']): init_pred = fcm(x, g, H)[1]
        torch.testing.assert_close(init_pred, f0, rtol=rt, atol=at)
        rep[arm] = dict(initial_equals_f0=True, trainable_tensors=len(trainable),
                        trainable_parameters=int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
                        official_loraplus=check_loraplus_official(model))
        models[arm] = (model, fcm, bb0)
    rep['top3_prefix'] = dict(no_grad_blocks=list(range(SET['top3_first_trainable_block'])),
                              checkpointed_blocks=list(range(SET['top3_first_trainable_block'], 12)))
    smoke = Ledger(OUT/'SMOKE_LEDGER.jsonl', CFG['max_smoke_updates'])
    for arm, (model, fcm, bb0) in models.items():
        rep[arm].update(_smoke_updates(model, fcm, panel, arm, smoke, levels))
        if backbone_digest(model) != bb0: raise Blocked('BACKBONE_CHANGED_DURING_SMOKE')
    write_json(OUT/'PREFLIGHT_BASE.json', dict(status='PASS', checks=rep, init_files=inits, canonical_modules=len(canon),
                                               top3_modules=len(tops)))
    mark_stage('preflight_base', {})


# ============================================================ compression
def stage_compress():
    assert_seal(); panel = _panel(); clock = Clock(); gpu_reset()
    with clock.part('full_model_load'):
        full, _ = load_base(BASECFG, DEVICE)
    model_fp = fingerprint(full); fc = Forecaster(full)
    targets = compression_targets(full); canon = canonical_map(full)
    if len(targets) != 120: raise Blocked(f'COMPRESSION_TARGET_COUNT_CHANGED {len(targets)}')
    shapes = {n: [get(full, n).out_features, get(full, n).in_features] for n in sorted(set(targets) | set(canon))}
    weight_norms = {n: float(torch.linalg.matrix_norm(get(full, n).weight.detach().double())) for n in canon}
    cal, diag, cal_idx, diag_idx = select_origin_sets(panel)
    with clock.part('activation_collection'):
        moments, rows = collect_second_moments(fc, full, panel, cal, targets)
    factors, excluded, info = make_factors(full, moments, targets, clock)
    prep_peak = gpu_peak()
    (CACHE/'emulator').mkdir(parents=True, exist_ok=True)
    torch.save({n: dict(left=l, right=r) for n, (l, r) in factors.items()}, CACHE/'emulator'/'factors.pt')
    full_params = sum(p.numel() for p in full.parameters()); full_bytes = sum(p.numel()*p.element_size() for p in full.parameters())
    del moments, fc, full; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    with clock.part('emulator_build'):
        emu, _ = build_emulator(factors)
    emu_fp = fingerprint(emu)
    if compression_targets(emu): raise Blocked('DENSE_LAYER_LEFT_IN_EMULATOR')
    emu_bytes = sum(t.numel()*t.element_size() for t in list(emu.parameters()) + list(emu.buffers()))
    write_json(CACHE/'emulator'/'meta.json', dict(
        calibration_origins=cal.tolist(), diagnostic_origins=diag.tolist(), calibration_index=cal_idx.tolist(),
        diagnostic_index=diag_idx.tolist(), shapes=shapes, weight_norms=weight_norms, canonical=canon,
        top3=top3_names(canon), targets=targets, model_fingerprint=model_fp, emulator_svd_fingerprint=emu_fp,
        factor_file_sha256=sha(CACHE/'emulator'/'factors.pt')))
    write_json(OUT/'MODULE_MAP.json', dict(
        compression_targets=targets, compression_count=len(targets), excluded=excluded,
        canonical_lora_modules=canon, canonical_count=len(canon), top3_modules=top3_names(canon),
        top3_count=len(top3_names(canon)), shapes=shapes,
        factorized_lora_targets=[n for n in canon if n in factors],
        dense_lora_targets=[n for n in canon if n not in factors],
        compressed_without_lora=[n for n in targets if n not in canon],
        note='LoRA A/B live on original in->out projections; the emulator keeps in/out dimensions, adapters are additive branches. '
             'MLP wi/wo are compressed but carry no LoRA in the contract map, so EMLoC correction cannot act on them.'))
    total_rank = int(sum(v['rank'] for v in info.values()))
    write_json(OUT/'EMULATOR_MANIFEST.json', dict(
        method='activation-aware weighted SVD, uncentered X^T X / n from 16 TRAIN origins (F0 forward, no future y)',
        keep_fraction=CFG['compression_weight_entry_fraction'], eigen_floor=CFG['svd_relative_eigen_floor'],
        modules=info, excluded=excluded, total_retained_rank=total_rank,
        calibration_gamma_parameters=total_rank, activation_rows=rows,
        full_model_parameters=int(full_params), full_model_parameter_bytes=int(full_bytes),
        emulator_tensor_bytes=int(emu_bytes),
        replaced_dense_entries=int(sum(v['dense_entries'] for v in info.values())),
        replaced_factor_entries=int(sum(v['factor_entries'] for v in info.values())),
        forward_form='main fits/caches: exp(gamma) folded into left (FrozenFactorizedLinear); calibration: reference_core.FactorizedLinear',
        model_fingerprint=model_fp, emulator_fingerprint=emu_fp, factor_file_sha256=sha(CACHE/'emulator'/'factors.pt'),
        seconds=clock.parts, compress_peak=prep_peak, emulator_build_peak=gpu_peak()))
    mark_stage('compress', dict(parts=clock.parts, peak=prep_peak))


# ============================================================ preflight on the emulator path
def stage_preflight_emulator():
    assert_seal(); panel = _panel(); meta = _meta(); rt, at = SET['prediction_rtol'], SET['prediction_atol']
    canon = meta['canonical']; shapes = {k: tuple(v) for k, v in meta['shapes'].items()}; H = panel.horizon
    factors = load_factors(); rep = {}
    full, _ = load_base(BASECFG, DEVICE); fcF = Forecaster(full); levels = full.quantiles.detach()
    cal = np.asarray(meta['calibration_origins'])
    o1 = panel.origins['train'][:1]; x, y, g = tensor_batch(panel, o1)
    with torch.no_grad(), autocast(DEVICE, BASECFG['precision']):
        f0_q, f0, f_loc, f_scale = fcF(x, g, H)
    smoke_origins = panel.origins['train'][:2]
    f0_smoke = {int(o): torch.as_tensor(r, device=DEVICE) for o, r in zip(smoke_origins, predict(fcF, panel, smoke_origins)[0])}
    # (c) full-rank mapping: exact algebra in float64 for every module, then a fp32 full-rank model forward
    moments, _ = collect_second_moments(fcF, full, panel, cal, list(factors))
    worst = 0.; full_rank = {}
    for n in factors:
        w = get(full, n).weight.detach().double(); r = min(w.shape)
        l, r_ = rc.activation_factors(w, moments[n].double(), r, CFG['svd_relative_eigen_floor'])
        rel = float(torch.linalg.matrix_norm(l @ r_ - w)/torch.linalg.matrix_norm(w)); worst = max(worst, rel)
        full_rank[n] = (l.float().cpu(), r_.float().cpu())
    if worst > SET['full_rank_weight_rel_tol_fp64']: raise Blocked(f'FULL_RANK_MAPPING_FAILED {worst}')
    del moments; gc.collect(); torch.cuda.empty_cache()
    with torch.no_grad():
        dense32 = fcF(x, g, H)[0].float()
    fr, _ = build_emulator(full_rank)
    with torch.no_grad():
        fr32 = Forecaster(fr)(x, g, H)[0].float()
    fr_err = float((fr32 - dense32).abs().max())
    if fr_err > SET['full_rank_prediction_abs_tol_native_fp32']: raise Blocked(f'FULL_RANK_PREDICTION_MISMATCH {fr_err}')
    rep['full_rank_mapping'] = dict(max_weight_rel_error_fp64=worst, max_native_prediction_abs_fp32=fr_err,
                                   note='full rank must reproduce the original; the half-rank emulator is not forced to')
    del fr, full_rank; gc.collect(); torch.cuda.empty_cache()
    # (d) no dense reconstruction inside either compressed forward form
    from torch.utils._python_dispatch import TorchDispatchMode
    emu, _ = build_emulator(factors)
    emu_t, _ = build_emulator(factors, trainable_gamma=True)
    probe_name = next(n for n in factors if n.endswith('self_attention.q'))
    rep['no_dense_forward'] = {}
    for tag, model in (('frozen_folded', emu), ('calibration_gamma', emu_t)):
        mod = get(model, probe_name); seen = []
        class Record(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                if isinstance(out, torch.Tensor): seen.append((str(func), tuple(out.shape)))
                return out
        with torch.no_grad(), Record():
            mod(torch.randn(3, 5, mod.in_features, device=DEVICE))
        dense_like = [s for s in seen if s[1] in {(mod.out_features, mod.in_features), (mod.in_features, mod.out_features)}]
        if dense_like: raise Blocked(f'DENSE_WEIGHT_MATERIALIZED {tag} {dense_like}')
        rep['no_dense_forward'][tag] = dict(module=probe_name, ops=seen, dense_shaped_outputs=0)
    emu_bytes = sum(t.numel()*t.element_size() for t in list(emu.parameters()) + list(emu.buffers()))
    full_bytes = sum(t.numel()*t.element_size() for t in list(full.parameters()) + list(full.buffers()))
    rep['no_dense_forward'].update(emulator_tensor_bytes=int(emu_bytes), full_tensor_bytes=int(full_bytes),
                                   remaining_dense_linear_in_blocks=len(compression_targets(emu)))
    if compression_targets(emu): raise Blocked('DENSE_LINEAR_REMAINS_IN_EMULATOR')
    # gamma save -> rebuild round trip (the path the main fits use), before any journal entry
    gen = torch.Generator().manual_seed(5)
    gtest = {n: (torch.rand(l.shape[1], generator=gen)*2-1)*.1 for n, (l, _) in factors.items()}
    for n, m in factorized_modules(emu_t).items():
        with torch.no_grad(): m.gamma.copy_(gtest[n])
    tmp = CACHE/'emulator'/'gamma_roundtrip_check.pt'; torch.save(gtest, tmp)
    emu_g, _ = build_emulator(factors, torch.load(tmp, weights_only=True))
    dense_rel = 0.
    for n in factors:
        wt = get(emu_t, n).dense_weight_for_audit_only(); fm = get(emu_g, n)
        dense_rel = max(dense_rel, float(torch.linalg.matrix_norm(wt - fm.left @ fm.right)/torch.linalg.matrix_norm(wt)))
    if dense_rel > SET['gamma_roundtrip_dense_rel_tol']: raise Blocked(f'GAMMA_ROUNDTRIP_MAP_MISMATCH {dense_rel}')
    with torch.no_grad():
        a_ = Forecaster(emu_t)(x, g, H)[0]; b_ = Forecaster(emu_g)(x, g, H)[0]
    native_abs = float((a_ - b_).abs().max())
    if native_abs > SET['full_rank_prediction_abs_tol_native_fp32']: raise Blocked(f'GAMMA_ROUNDTRIP_PREDICTION_MISMATCH {native_abs}')
    emu_g2, _ = build_emulator(factors, torch.load(tmp, weights_only=True))
    if fingerprint(emu_g) != fingerprint(emu_g2): raise Blocked('EMULATOR_FINGERPRINT_NOT_STABLE')
    rep['gamma_roundtrip'] = dict(dense_map_max_rel=dense_rel, native_fp32_max_abs=native_abs, fingerprint_stable=True,
                                  note='same algebra, different rounding order (folded exp(gamma)); raw-space gaps are sinh-amplified')
    tmp.unlink(); del emu_g, emu_g2, emu_t; gc.collect(); torch.cuda.empty_cache()
    # loc/scale identity; zero-LoRA transfer
    fcE = Forecaster(emu)
    with torch.no_grad(), autocast(DEVICE, BASECFG['precision']):
        e_q, e0, e_loc, e_scale = fcE(x, g, H)
    if not (torch.equal(e_loc, f_loc) and torch.equal(e_scale, f_scale)): raise Blocked('BLOCKED_MAPPING: loc/scale differ')
    rep['gamma'] = dict(parameters=int(sum(l.shape[1] for l, _ in factors.values())), frozen_in_main='FrozenFactorizedLinear has no parameters')
    rep['half_rank_emulator_vs_f0'] = dict(sigma_mse=sigma_mse(e0.float().cpu().numpy() - f0.float().cpu().numpy(), panel.sigma))
    init0 = torch.load(CACHE/'init'/f'seed_{CFG["seeds"][0]}.pt', weights_only=True)
    bb0 = backbone_digest(full); attach_adapters(full, canon); load_lora(full, init0)
    with torch.no_grad(), autocast(DEVICE, BASECFG['precision']): t0 = fcF(x, g, H)[1]
    torch.testing.assert_close(t0, f0, rtol=rt, atol=at)
    if tensor_digest(lora_state(full)) != tensor_digest(init0): raise Blocked('TRANSFER_HASH_MISMATCH')
    rep['transfer_theta0_equals_f0'] = True
    rep['emloc'] = _emloc_checks(full, factors, canon)
    if backbone_digest(full) != bb0: raise Blocked('BACKBONE_CHANGED')
    del full, fcF; gc.collect(); torch.cuda.empty_cache()
    # calibration path without updates: FP32 vs BF16 probe resolution, gamma gradients on value and plus/minus paths
    rep['calibration_checks'] = _calibration_checks(panel, meta, factors)
    # emulator-side main arms: anchored train0, LoRA+ groups (journal-free)
    attach_adapters(emu, canon); enable_checkpointing(emu); fcE = Forecaster(emu)
    load_lora(emu, init0)
    e0_smoke = {int(o): torch.as_tensor(r, device=DEVICE) for o, r in zip(smoke_origins, predict(fcE, panel, smoke_origins)[0])}
    for arm in E_ARMS:
        load_lora(emu, init0); trainable = set_trainable_lora(emu, canon)
        if any(p.requires_grad for n, p in emu.named_parameters() if '.lora_' not in n): raise Blocked('NON_LORA_TRAINABLE_IN_MAIN')
        if arm in DELTA_ARMS:
            with autocast(DEVICE, BASECFG['precision']):
                _, raw, _, _ = fcE(x, g, H)
                tr0 = rc.anchored_raw(f0.float(), raw.float(), e0.float())
            torch.testing.assert_close(tr0, f0.float(), rtol=rt, atol=at)
        rep[arm] = dict(trainable_tensors=len(trainable), trainable_parameters=int(sum(p.numel() for p in emu.parameters() if p.requires_grad)),
                        official_loraplus=check_loraplus_official(emu), anchored_train0_equals_f0=arm in DELTA_ARMS)
    # smoke updates last: 8 main-arm connection updates, then 4 calibration connection updates
    smoke = Ledger(OUT/'SMOKE_LEDGER.jsonl', CFG['max_smoke_updates'])
    for arm in E_ARMS:
        load_lora(emu, init0); set_trainable_lora(emu, canon)
        delta = {o: (f0_smoke[o], e0_smoke[o]) for o in f0_smoke} if arm in DELTA_ARMS else None
        rep[arm].update(smoke_emulator='svd (gamma 0); VALUE/RESPONSE arms share this code path and load sealed gammas in the fit',
                        **_smoke_updates(emu, fcE, panel, arm, smoke, levels, delta))
    del emu, fcE; gc.collect(); torch.cuda.empty_cache()
    rep['calibration_smoke'] = _calibration_smoke(panel, meta, factors, smoke)
    rep['optimizer_separation'] = dict(calibration_targets='gamma only (checked)', main_targets='lora_A/lora_B only (checked)')
    write_json(OUT/'PREFLIGHT_EMULATOR.json', dict(status='PASS', checks=rep))
    mark_stage('preflight_emulator', {})


@torch.no_grad()
def _emloc_checks(full, factors, canon):
    name = next(n for n in canon if n in factors and n.endswith('self_attention.q'))
    w_full = get(full, name).base_layer.weight.detach().float()
    left, right = factors[name]; w_em = emulator_dense_weight(left.to(DEVICE).float(), right.to(DEVICE).float())
    gen = torch.Generator().manual_seed(11)
    a = (torch.rand(CFG['lora_rank'], w_full.shape[1], generator=gen)*2-1).to(DEVICE)/math.sqrt(w_full.shape[1])
    b = (torch.randn(w_full.shape[0], CFG['lora_rank'], generator=gen)*1e-3).to(DEVICE)
    out = {}
    aa, bb = rc.emloc_correct(w_full.double(), w_em.double(), a.double(), b.double(), scaling=SCALING, correction_limit=math.inf)
    u = torch.linalg.svd(a.double().T, full_matrices=False)[0]
    lhs = (w_full.double() + SCALING*bb @ aa) @ u; rhs = (w_em.double() + SCALING*b.double() @ a.double()) @ u
    out['active_subspace_identity_rel'] = float(torch.linalg.matrix_norm(lhs - rhs)/torch.linalg.matrix_norm(rhs))
    if out['active_subspace_identity_rel'] > 1e-10: raise Blocked('EMLOC_IDENTITY_FAILED')
    for tag, dtype, tol in (('fp32', torch.float32, SET['official_emloc_fp32_rel_tol']),
                            ('bf16', torch.bfloat16, SET['official_emloc_bf16_rel_tol'])):
        oa, ob = official_get_corrected(a, b, w_em, w_full, CFG['emloc_correction_limit'], dtype)
        pa, pb = rc.emloc_correct(w_full, w_em, a, b, scaling=1., correction_limit=CFG['emloc_correction_limit'])
        rel = float(torch.linalg.matrix_norm(ob.float() @ oa.float() - pb @ pa)/torch.linalg.matrix_norm(pb @ pa))
        out[f'official_vs_port_scaling1_{tag}_rel'] = rel
        if rel > tol: raise Blocked(f'EMLOC_OFFICIAL_PARITY_{tag} {rel}')
    oa, ob = official_get_corrected(a, b, w_em, w_full, math.inf, torch.float32)
    lhs_off = (w_full + SCALING*ob @ oa) @ u.float()
    out['verbatim_official_at_scaling2_subspace_rel_mismatch'] = float(torch.linalg.matrix_norm(lhs_off - rhs.float())/torch.linalg.matrix_norm(rhs.float()))
    bu = torch.linalg.svd(a.T, full_matrices=False)
    b_eff = SCALING*(b @ bu[2].T)*bu[1][None, :]; diff = (w_full - w_em) @ bu[0]
    out['clamped_basis_fraction_L3_random_B'] = float(((diff.abs().mean(0)/b_eff.abs().mean(0)) > CFG['emloc_correction_limit']).float().mean())
    za, zb = rc.emloc_correct(w_full, w_em, a, torch.zeros_like(b), scaling=SCALING)
    out['zero_B_noop'] = bool(torch.equal(zb, torch.zeros_like(b)) and torch.equal(za, a))
    la, lb = rc.emloc_correct(w_full, w_em, a, b, scaling=SCALING, correction_limit=0.)
    out['zero_limit_preserves_rel'] = float(torch.linalg.matrix_norm(lb @ la - b @ a)/torch.linalg.matrix_norm(b @ a))
    sa, sb = rc.emloc_correct(w_full, w_full, a, b, scaling=SCALING)
    out['same_base_preserves_rel'] = float(torch.linalg.matrix_norm(sb @ sa - b @ a)/torch.linalg.matrix_norm(b @ a))
    out['module'] = name; out['scaling'] = SCALING; out['official_commit'] = '332d70e9189f8a2bec3dc8a238eef1439b30e938'
    out['official_scaling_note'] = ('official config/peft/8B.json sets r=8 without lora_alpha -> PEFT default alpha=8, '
                                    'scaling 1; its B_-correction is exact only at scaling 1, so the port divides by alpha/r=2')
    out['official_lora_targets'] = 'language_model.model.layers.*w([o123]|qkv): attention wqkv/wo AND MLP w1/w2/w3'
    out['baseline_status'] = ('LIMITED_BASELINE: the contract map puts LoRA only on attention q/k/v/o + output head, '
                              'so compressed MLP wi/wo errors are not correctable by EMLoC here (official corrects MLP too)')
    return out


def _teacher_forwards(full, fcF, panel, canon, origins, plist, clock=None, zero_part=None, probe_part=None):
    f0s, fps, fms = [], [], []
    for j, o in enumerate(origins):
        x, _, g = tensor_batch(panel, [o]); set_probe(full, plist[j % 4])
        with torch.no_grad(), calib_ctx():
            with (clock.part(zero_part) if clock else contextlib.nullcontext()):
                set_sign(full, canon, 0); f0s.append(fcF(x, g, panel.horizon)[1].float())
            with (clock.part(probe_part) if clock else contextlib.nullcontext()):
                set_sign(full, canon, 1); fps.append(fcF(x, g, panel.horizon)[1].float())
                set_sign(full, canon, -1); fms.append(fcF(x, g, panel.horizon)[1].float())
        set_sign(full, canon, 0)
    return f0s, fps, fms


def _calibration_checks(panel, meta, factors):
    """Journal-free: probe resolution under BF16 vs FP32, and gamma gradients on value and response paths."""
    canon = meta['canonical']; shapes = {k: tuple(v) for k, v in meta['shapes'].items()}
    probes = make_probes(CFG['probe_seed'], canon, shapes, meta['weight_norms'])
    cal = np.asarray(meta['calibration_origins'])[:2]; sigma = torch.as_tensor(panel.sigma, device=DEVICE, dtype=torch.float32)
    full, _ = load_base(BASECFG, DEVICE); attach_probes(full, canon); fcF = Forecaster(full); out = {'resolution': []}
    for p in (0, 1):
        for o in cal:
            set_probe(full, probes[p]); x, _, g = tensor_batch(panel, [o]); d = {}
            for tag, ctx in (('bf16', lambda: autocast(DEVICE, BASECFG['precision'])), ('fp32', contextlib.nullcontext)):
                with torch.no_grad(), ctx():
                    set_sign(full, canon, 1); fp = fcF(x, g, panel.horizon)[1].float()
                    set_sign(full, canon, -1); fm = fcF(x, g, panel.horizon)[1].float(); set_sign(full, canon, 0)
                d[tag] = (fp - fm)*.5
            out['resolution'].append(dict(probe_index=p, origin=int(o),
                rel_error_bf16_vs_fp32=float(torch.linalg.vector_norm(d['bf16']-d['fp32'])/torch.linalg.vector_norm(d['fp32'])),
                zero_fraction_bf16=float((d['bf16'] == 0).float().mean()), zero_fraction_fp32=float((d['fp32'] == 0).float().mean()),
                target_mse_fp32=float(rc.weighted_mse(d['fp32'], sigma)), target_mse_bf16=float(rc.weighted_mse(d['bf16'], sigma))))
    teach = _teacher_forwards(full, fcF, panel, canon, cal, probes)
    del full, fcF; gc.collect(); torch.cuda.empty_cache()
    emu, _ = build_emulator(factors, trainable_gamma=True); attach_probes(emu, canon); fcE = Forecaster(emu)
    mods = factorized_modules(emu)
    for m in mods.values(): m.set_calibration(True)
    gammas = [m.gamma for m in mods.values()]
    trainable = {n for n, p in emu.named_parameters() if p.requires_grad}
    if trainable != {n+'.gamma' for n in mods} or any('.lora_' in n for n in trainable): raise Blocked('CALIBRATION_TARGETS_MIXED')
    x, _, g = tensor_batch(panel, [cal[0]]); set_probe(emu, probes[0])
    with calib_ctx():
        set_sign(emu, canon, 0); e0 = fcE(x, g, panel.horizon)[1].float()
        lv = rc.value_loss(e0, teach[0][0], rc.weighted_mse(e0.detach() - teach[0][0], sigma), sigma)
        set_sign(emu, canon, 1); ep = fcE(x, g, panel.horizon)[1].float()
        set_sign(emu, canon, -1); em = fcE(x, g, panel.horizon)[1].float(); set_sign(emu, canon, 0)
        lr_ = rc.response_loss(ep, em, teach[1][0], teach[2][0], sigma)
    gv = torch.autograd.grad(lv, gammas, retain_graph=True, allow_unused=True)
    gr = torch.autograd.grad(lr_, gammas, allow_unused=True)
    out['value_path_modules_with_gradient'] = sum(bool(t is not None and t.abs().sum() > 0) for t in gv)
    out['response_path_modules_with_gradient'] = sum(bool(t is not None and t.abs().sum() > 0) for t in gr)
    if out['value_path_modules_with_gradient'] == 0 or out['response_path_modules_with_gradient'] == 0:
        raise Blocked('CALIBRATION_GAMMA_GRADIENT_MISSING')
    out['gamma_parameters'] = int(sum(t.numel() for t in gammas)); out['precision'] = SET['calibration_precision']
    return out


def _calibration_smoke(panel, meta, factors, smoke):
    """2 real gamma updates per kind on a throwaway gamma, same precision and code path as the calibrate stage."""
    canon = meta['canonical']; shapes = {k: tuple(v) for k, v in meta['shapes'].items()}
    probes = make_probes(CFG['probe_seed'], canon, shapes, meta['weight_norms'])
    cal = np.asarray(meta['calibration_origins'])[:2]; sigma = torch.as_tensor(panel.sigma, device=DEVICE, dtype=torch.float32)
    full, _ = load_base(BASECFG, DEVICE); attach_probes(full, canon)
    F0c, Fpc, Fmc = _teacher_forwards(full, Forecaster(full), panel, canon, cal, probes)
    del full; gc.collect(); torch.cuda.empty_cache()
    emu, _ = build_emulator(factors, trainable_gamma=True); attach_probes(emu, canon); fcE = Forecaster(emu); out = {}
    mods = factorized_modules(emu); gammas = [m.gamma for m in mods.values()]
    with torch.no_grad(), calib_ctx():
        init = []
        for j, o in enumerate(cal):
            x, _, g = tensor_batch(panel, [o]); set_sign(emu, canon, 0)
            init.append(rc.weighted_mse(fcE(x, g, panel.horizon)[1].float() - F0c[j], sigma))
    for kind in CFG['gamma_kinds']:
        for m in mods.values():
            with torch.no_grad(): m.gamma.zero_()
            m.set_calibration(True)
        opt = torch.optim.AdamW(gammas, lr=CFG['gamma_lr'], betas=(.9, .999), eps=1e-8, weight_decay=0.)
        steps = {}
        for j, o in enumerate(cal):
            opt.zero_grad(set_to_none=True); set_probe(emu, probes[j % 4]); x, _, g = tensor_batch(panel, [o])
            lv, lr_, loss = _calibration_loss(emu, fcE, canon, x, g, panel.horizon, F0c[j], Fpc[j], Fmc[j], init[j], sigma, kind)
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in gammas): raise Blocked('CALIBRATION_BAD_GRADIENT')
            gn = float(torch.nn.utils.clip_grad_norm_(gammas, CFG['gradient_clip']))
            smoke.intent(f'smoke_calibration_{kind}', j+1); opt.step(); sync(DEVICE); smoke.commit(f'smoke_calibration_{kind}', j+1)
            for m in mods.values(): m.clamp_calibration()
            steps[f'step{j+1}'] = dict(L_value=float(lv), L_response=float(lr_), gamma_grad_norm=gn)
        out[kind] = steps
        for m in mods.values():
            with torch.no_grad(): m.gamma.zero_()
            m.set_calibration(False)
    out['probe_effective_rel_norms'] = _probe_norm_check(probes, meta)
    return out


def _calibration_loss(emu, fcE, canon, x, g, horizon, f0, fp, fm, init, sigma, kind):
    with calib_ctx():
        set_sign(emu, canon, 0); e0 = fcE(x, g, horizon)[1].float()
        lv = rc.value_loss(e0, f0, init, sigma); lr_ = torch.zeros((), device=DEVICE)
        if kind == 'RESPONSE':
            set_sign(emu, canon, 1); ep = fcE(x, g, horizon)[1].float()
            set_sign(emu, canon, -1); em = fcE(x, g, horizon)[1].float(); set_sign(emu, canon, 0)
            lr_ = rc.response_loss(ep, em, fp, fm, sigma)
        loss = lv + CFG['response_loss_weight']*lr_
    return lv, lr_, loss


def _probe_norm_check(probes, meta):
    rels = []
    for p in probes:
        vals = [float(SCALING*torch.linalg.matrix_norm(b.double() @ a.double())/meta['weight_norms'][n]) for n, (a, b) in p.items()]
        rels.append([min(vals), max(vals)])
    return rels


# ============================================================ gamma calibration (VALUE / RESPONSE)
def stage_calibrate():
    assert_seal(); panel = _panel(); meta = _meta(); clock = Clock()
    canon = meta['canonical']; shapes = {k: tuple(v) for k, v in meta['shapes'].items()}
    ledger = Ledger(OUT/'CALIBRATION_LEDGER.jsonl', CFG['max_calibration_updates'])
    probes = make_probes(CFG['probe_seed'], canon, shapes, meta['weight_norms'])
    dprobes = make_probes(CFG['diagnostic_probe_seed'], canon, shapes, meta['weight_norms'])
    cal = np.asarray(meta['calibration_origins']); diag = np.asarray(meta['diagnostic_origins'])
    sigma = torch.as_tensor(panel.sigma, device=DEVICE, dtype=torch.float32); H = panel.horizon
    gpu_reset()
    with clock.part('artifact_teacher_full_load'):
        full, _ = load_base(BASECFG, DEVICE)
    attach_probes(full, canon); fcF = Forecaster(full)
    F0c, Fpc, Fmc = _teacher_forwards(full, fcF, panel, canon, cal, probes, clock, 'teacher_zero_calibration', 'teacher_probe_calibration')
    F0d, Fpd, Fmd = _teacher_forwards(full, fcF, panel, canon, diag, dprobes, clock, 'diag_teacher_zero', 'diag_teacher_probe')
    teacher_peak = gpu_peak()
    del full, fcF; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    factors = load_factors()
    with clock.part('artifact_emulator_load'):
        emu, _ = build_emulator(factors, trainable_gamma=True)
    attach_probes(emu, canon); fcE = Forecaster(emu); mods = factorized_modules(emu)
    gammas = [m.gamma for m in mods.values()]

    @torch.no_grad()
    def evaluate(origins, plist, f0s, fps, fms, label):
        """Diagnostic only (never charged): value and response errors of the current emulator."""
        rows = []
        for j, o in enumerate(origins):
            x, _, g = tensor_batch(panel, [o]); set_probe(emu, plist[j % 4])
            with calib_ctx():
                set_sign(emu, canon, 0); e0 = fcE(x, g, H)[1].float()
                set_sign(emu, canon, 1); ep = fcE(x, g, H)[1].float()
                set_sign(emu, canon, -1); em = fcE(x, g, H)[1].float(); set_sign(emu, canon, 0)
            df = (fps[j]-fms[j])*.5; de = (ep-em)*.5
            rows.append(dict(origin=int(o), probe_index=j % 4, value_numerator=float(rc.weighted_mse(e0-f0s[j], sigma)),
                             response_numerator=float(rc.weighted_mse(de-df, sigma)),
                             response_target_mse=float(rc.weighted_mse(df, sigma)), label=label))
        return rows
    init_err = []
    with torch.no_grad():
        for j, o in enumerate(cal):
            x, _, g = tensor_batch(panel, [o])
            with clock.part('initial_error_calibration'), calib_ctx():
                set_sign(emu, canon, 0); init_err.append(rc.weighted_mse(fcE(x, g, H)[1].float() - F0c[j], sigma))
    base_cal = evaluate(cal, probes, F0c, Fpc, Fmc, 'calibration')
    base_diag = evaluate(diag, dprobes, F0d, Fpd, Fmd, 'heldout')
    init_diag = [r['value_numerator'] for r in base_diag]
    trace, response_rows, fingerprints, stats = [], [], {}, {}

    def finish_rows(rows, variant, denominators):
        for r, d in zip(rows, denominators):
            r.update(variant=variant, L_value=r['value_numerator']/(d+1e-8),
                     L_response=r['response_numerator']/(r['response_target_mse']+1e-8))
            response_rows.append(r)
    finish_rows([dict(r) for r in base_cal], 'svd', [float(t) for t in init_err])
    finish_rows([dict(r) for r in base_diag], 'svd', init_diag)
    for kind in CFG['gamma_kinds']:
        for m in mods.values():
            with torch.no_grad(): m.gamma.zero_()
            m.set_calibration(True)
        opt = torch.optim.AdamW(gammas, lr=CFG['gamma_lr'], betas=(.9, .999), eps=1e-8, weight_decay=0.)
        trainable = {n for n, p in emu.named_parameters() if p.requires_grad}
        if trainable != {n+'.gamma' for n in mods}: raise Blocked('CALIBRATION_TARGETS_MIXED')
        torch.cuda.reset_peak_memory_stats()
        for j, o in enumerate(cal):
            sync(DEVICE); t = time.perf_counter()
            opt.zero_grad(set_to_none=True); set_probe(emu, probes[j % 4]); x, _, g = tensor_batch(panel, [o])
            lv, lr_, loss = _calibration_loss(emu, fcE, canon, x, g, H, F0c[j], Fpc[j], Fmc[j], init_err[j], sigma, kind)
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in gammas): raise Blocked('CALIBRATION_BAD_GRADIENT')
            gn = float(torch.nn.utils.clip_grad_norm_(gammas, CFG['gradient_clip']))
            sync(DEVICE); fb = time.perf_counter() - t
            ledger.intent(kind, j+1); sync(DEVICE); tick = time.perf_counter()
            opt.step()
            for m in mods.values(): m.clamp_calibration()
            sync(DEVICE); dt = fb + time.perf_counter() - tick; ledger.commit(kind, j+1)
            clock.parts[f'calibration_{kind}'] = clock.parts.get(f'calibration_{kind}', 0.) + dt
            trace.append(dict(kind=kind, step=j+1, origin=int(o), probe_index=j % 4, L_value=float(lv),
                              L_response=float(lr_), loss=float(loss), gamma_grad_norm=gn, active_step_s=dt))
        stats[kind] = dict(peak=gpu_peak())
        for m in mods.values(): m.set_calibration(False)
        g_state = {n.replace('.base_layer', ''): m.gamma.detach().cpu().clone() for n, m in mods.items()}
        if set(g_state) != set(factors): raise Blocked('GAMMA_KEYS_DO_NOT_MATCH_FACTORS')
        torch.save(g_state, CACHE/'emulator'/f'gamma_{kind.lower()}.pt')
        allg = torch.cat([v.flatten() for v in g_state.values()]); bound = math.log(2)
        stats[kind].update(gamma_min=float(allg.min()), gamma_max=float(allg.max()), gamma_mean_abs=float(allg.abs().mean()),
                           fraction_at_bound=float((allg.abs() >= bound-1e-6).float().mean()), gamma_count=int(allg.numel()),
                           file_sha256=sha(CACHE/'emulator'/f'gamma_{kind.lower()}.pt'))
        variant = kind.lower()
        finish_rows(evaluate(cal, probes, F0c, Fpc, Fmc, 'calibration'), variant, [float(t) for t in init_err])
        finish_rows(evaluate(diag, dprobes, F0d, Fpd, Fmd, 'heldout'), variant, init_diag)
    del emu, fcE; gc.collect(); torch.cuda.empty_cache()
    for kind in ('svd', 'value', 'response'):
        e, _ = build_emulator(factors, load_gammas(kind)); fingerprints[kind] = fingerprint(e); del e
        gc.collect(); torch.cuda.empty_cache()
    write_csv(OUT/'CALIBRATION_RESPONSE.csv', response_rows)
    write_csv(OUT/'CALIBRATION_TRACE.csv', trace)
    write_json(OUT/'PROBE_MANIFEST.json', dict(
        seed=CFG['probe_seed'], diagnostic_seed=CFG['diagnostic_probe_seed'], order='index = 2*direction + scale_index',
        scales=CFG['probe_relative_weight_norms'], directions=CFG['probe_independent_directions'],
        plus_minus='B sign flipped only', calibration_probe_of_origin='j mod 4', precision=SET['calibration_precision'],
        effective_rel_norm_range=_probe_norm_check(probes, meta),
        diagnostic_effective_rel_norm_range=_probe_norm_check(dprobes, meta),
        teacher_forwards=dict(calibration_zero=len(cal), calibration_probe=2*len(cal), heldout_zero=len(diag), heldout_probe=2*len(diag))))
    write_json(OUT/'CALIBRATION_MANIFEST.json', dict(
        kinds=CFG['gamma_kinds'], updates_per_kind=CFG['gamma_updates_per_kind'], lr=CFG['gamma_lr'],
        bound='[-log 2, log 2] after each step', clip=CFG['gradient_clip'], response_weight=CFG['response_loss_weight'],
        precision=SET['calibration_precision'], gamma_parameters=int(sum(m.numel() for m in gammas)),
        calibration_origins=cal.tolist(), heldout_origins=diag.tolist(),
        emulator_fingerprints=fingerprints, stats=stats, seconds=clock.parts, teacher_peak=teacher_peak,
        forwards_per_update=dict(VALUE=1, RESPONSE=3),
        note='Both calibrations start from the same E_svd; the two main seeds share them (not a calibration-population replicate).'))
    mark_stage('calibrate', dict(parts=clock.parts))


# ============================================================ cost basis (fixed rule, computed from stage receipts before fits)
CHARGE_RULE = {
    'all': [('prepare', 'data_prepare')],
    'E': [('compress', 'full_model_load'), ('compress', 'activation_collection'),
          ('compress', 'svd_factorization'), ('compress', 'emulator_build')],
    'VALUE': [('calibrate', 'teacher_zero_calibration'), ('calibrate', 'initial_error_calibration'),
              ('calibrate', 'calibration_VALUE')],
    'RESPONSE': [('calibrate', 'teacher_zero_calibration'), ('calibrate', 'teacher_probe_calibration'),
                 ('calibrate', 'initial_error_calibration'), ('calibrate', 'calibration_RESPONSE')],
    'delta': [('caches', 'f0_cache_train'), ('caches', 'f0_cache_val')],
}


def stage_cost_basis():
    assert_seal()
    parts = {s: read_json(CACHE/'stages'/f'{s}.json')['parts'] for s in ('prepare', 'compress', 'calibrate', 'caches')}
    charged = {}
    for arm in ARMS:
        items = list(CHARGE_RULE['all'])
        if arm in E_ARMS: items += CHARGE_RULE['E']
        if arm == 'E_VALUE_DELTA': items += CHARGE_RULE['VALUE']
        if arm == 'E_RESPONSE_DELTA': items += CHARGE_RULE['RESPONSE']
        if arm in DELTA_ARMS:
            k = EMULATOR_OF[arm]
            items += CHARGE_RULE['delta'] + [('caches', f'e0_cache_{k}_train'), ('caches', f'e0_cache_{k}_val')]
        charged[arm] = {f'{s}:{k}': float(parts[s][k]) for s, k in items}
    write_json(OUT/'COST_BASIS.json', dict(rule=CHARGE_RULE, charged=charged, stage_parts=parts,
        note='cold-start charge per independent adaptation; physically shared stages are counted once in the experiment ledger'))


# ============================================================ F0 / E0 TRAIN+VAL caches
def stage_caches():
    assert_seal(); panel = _panel(); clock = Clock(); factors = load_factors(); rec = {}
    (CACHE/'caches').mkdir(parents=True, exist_ok=True)
    gpu_reset()
    with clock.part('artifact_full_load'):
        full, _ = load_base(BASECFG, DEVICE)
    fcF = Forecaster(full); ref = {}
    for role in ('train', 'val'):
        with clock.part(f'f0_cache_{role}'):
            raw, loc, scale = predict(fcF, panel, panel.origins[role])
            np.save(CACHE/'caches'/f'F0_{role}_raw.npy', raw); np.save(CACHE/'caches'/f'F0_{role}_loc.npy', loc)
            np.save(CACHE/'caches'/f'F0_{role}_scale.npy', scale)
        ref[role] = (loc, scale); np.save(CACHE/'caches'/f'origins_{role}.npy', panel.origins[role])
    rec['F0'] = dict(fingerprint=fingerprint(full), peak=gpu_peak())
    del full, fcF; gc.collect(); torch.cuda.empty_cache()
    for kind in ('svd', 'value', 'response'):
        torch.cuda.reset_peak_memory_stats()
        with clock.part(f'artifact_emulator_load_{kind}'):
            emu, _ = build_emulator(factors, load_gammas(kind))
        fp = fingerprint(emu)
        if fp != read_json(OUT/'CALIBRATION_MANIFEST.json')['emulator_fingerprints'][kind]: raise Blocked('EMULATOR_CHANGED_AFTER_CALIBRATION')
        fcE = Forecaster(emu); gap = {}
        for role in ('train', 'val'):
            with clock.part(f'e0_cache_{kind}_{role}'):
                e_raw, e_loc, e_scale = predict(fcE, panel, panel.origins[role])
                np.save(CACHE/'caches'/f'E0_{kind}_{role}_raw.npy', e_raw)
            if not (np.array_equal(e_loc, ref[role][0]) and np.array_equal(e_scale, ref[role][1])):
                raise Blocked('BLOCKED_MAPPING: emulator loc/scale differ')
            gap[role] = sigma_mse(e_raw - np.load(CACHE/'caches'/f'F0_{role}_raw.npy'), panel.sigma)
        rec[kind] = dict(fingerprint=fp, peak=gpu_peak(), sigma_mse_vs_f0=gap)
        del emu, fcE; gc.collect(); torch.cuda.empty_cache()
    files = {p.name: sha(p) for p in sorted((CACHE/'caches').glob('*.npy'))}
    write_json(OUT/'CACHE_MANIFEST.json', dict(roles=['train', 'val'], origins={r: len(panel.origins[r]) for r in ('train', 'val')},
        files=files, bytes={p.name: p.stat().st_size for p in (CACHE/'caches').glob('*.npy')}, seconds=clock.parts, models=rec,
        contract='no future y enters any cached forecast; E0 computed after calibration with frozen gamma; loc/scale identical to F0; '
                 'TRAIN and VAL caches charged in full (contract section 5); VAL caches serve only the uncharged emulator-side diagnostic'))
    mark_stage('caches', dict(parts=clock.parts))


# ============================================================ main fit worker
def stage_fit(arm, seed):
    assert_seal(); panel = _panel(); meta = _meta(); H = panel.horizon
    key = f'{arm}_{seed}'; outp = OUT/'fits'/f'{key}.json'; work = CACHE/'fits'/key
    if outp.exists() and read_json(outp).get('complete'): return
    if work.exists(): raise Blocked(f'PARTIAL_FIT {key}: automatic replay is prohibited')
    work.mkdir(parents=True); clock = Clock(); seed_all(seed); gpu_reset(); start_all = time.perf_counter()
    canon = meta['canonical']; names = top3_names(canon) if arm == 'F_TOP3' else canon
    init = torch.load(CACHE/'init'/f'seed_{seed}.pt', weights_only=True)
    init = {k: v for k, v in init.items() if k.rsplit('.', 2)[0] in names}
    kind = EMULATOR_OF.get(arm)
    if arm in F_ARMS:
        with clock.part('full_model_load'): model, _ = load_base(BASECFG, DEVICE)
    else:
        with clock.part('emulator_load'): model, _ = build_emulator(load_factors(), load_gammas(kind))
        if fingerprint(model) != read_json(OUT/'CALIBRATION_MANIFEST.json')['emulator_fingerprints'][kind]:
            raise Blocked('EMULATOR_FINGERPRINT_MISMATCH')
    attach_adapters(model, names); load_lora(model, init); trainable = set_trainable_lora(model, names)
    if arm == 'F_TOP3': install_top3_prefix(model)
    else: enable_checkpointing(model)
    fc = Forecaster(model); opt = loraplus_optimizer(model); levels = model.quantiles.detach()
    frozen0 = fingerprint(model, frozen_only=True); initial = lora_state(model)
    if arm in DELTA_ARMS:
        row = {int(o): i for i, o in enumerate(np.load(CACHE/'caches'/'origins_train.npy'))}
        F0raw = np.load(CACHE/'caches'/'F0_train_raw.npy', mmap_mode='r'); E0raw = np.load(CACHE/'caches'/f'E0_{kind}_train_raw.npy', mmap_mode='r')
        F0loc = np.load(CACHE/'caches'/'F0_train_loc.npy'); F0scale = np.load(CACHE/'caches'/'F0_train_scale.npy')
        F0val = np.load(CACHE/'caches'/'F0_val_raw.npy', mmap_mode='r'); E0val = np.load(CACHE/'caches'/f'E0_{kind}_val_raw.npy', mmap_mode='r')
    schedule = panel.schedule(seed, CFG['main_steps'], CFG['origins_per_update']); np.save(work/'schedule.npy', schedule)
    ledger = Ledger(OUT/'UPDATE_LEDGER.jsonl', CFG['max_main_updates'])
    curves, trace = [], []; val_y = panel.targets('val'); levels_np = levels.float().cpu().numpy()
    updates_so_far = 0.

    def val_record(pred, label):
        m, _ = score(pred, val_y, panel.sigma, levels_np)
        return {f'{label}_primary': m['primary'], f'{label}_nmae': m['nmae'], f'{label}_crossing': m['crossing']}

    def checkpoint(step):
        with clock.part('checkpoint_io'):
            st = lora_state(model); p = work/f'step_{step}.pt'; torch.save(st, p)
        rec = dict(step=step, state_sha256=sha(p), updates_s_so_far=updates_so_far,
                   checkpoint_io_s_so_far=clock.parts.get('checkpoint_io', 0.))
        if arm in F_ARMS:
            t0 = clock.parts.get('val_eval', 0.)
            with clock.part('val_eval'):
                pred, _, _ = predict(fc, panel, panel.origins['val'])
            np.save(work/f'val_full_{step}.npy', pred)
            rec.update(val_record(pred, 'val')); rec['val_eval_s'] = clock.parts['val_eval'] - t0; rec['transfer_s'] = 0.
        else:
            with clock.part('diag_emulator_val'):
                pred, _, _ = predict(fc, panel, panel.origins['val'])
                if arm in DELTA_ARMS: pred = np.asarray(F0val) + pred - np.asarray(E0val)
                np.save(work/f'val_emulator_train_{step}.npy', pred)
            rec.update(val_record(pred, 'emulator_val'))
        curves.append(rec)
        print(f'{key} step={step} ' + ' '.join(f'{k}={v:.6g}' for k, v in rec.items() if k.endswith('_primary')), flush=True)

    checkpoint(0)
    torch.cuda.reset_peak_memory_stats()
    for step, origins in enumerate(schedule, 1):
        sync(DEVICE); t = time.perf_counter(); opt.zero_grad(set_to_none=True); total = 0.; seen = []
        for o in origins:
            x, y, g = tensor_batch(panel, [o])
            with autocast(DEVICE, BASECFG['precision']):
                q, raw, loc, scale = fc(x, g, H)
                if arm in DELTA_ARMS:
                    r = row[int(o)]; seen.append((r, loc.detach(), scale.detach()))
                    f0 = torch.as_tensor(np.asarray(F0raw[r]), device=DEVICE); e0 = torch.as_tensor(np.asarray(E0raw[r]), device=DEVICE)
                    qn = rc.raw_to_native(rc.anchored_raw(f0, raw.float(), e0), loc, scale, fc.arcsinh)
                    loss = normalized_loss(qn, y, loc, scale, levels, fc.arcsinh)/len(origins)
                else:
                    loss = normalized_loss(q, y, loc, scale, levels, fc.arcsinh)/len(origins)
            loss.backward(); total += loss.detach()
        params = [p for p in model.parameters() if p.requires_grad]
        gn = torch.nn.utils.clip_grad_norm_(params, CFG['gradient_clip'])
        sync(DEVICE); fb = time.perf_counter() - t
        # integrity checks outside the timed region: finite loss/gradients and exact F0 loc/scale mapping
        if not torch.isfinite(total): raise FloatingPointError('NONFINITE_LOSS')
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params): raise FloatingPointError('BAD_GRADIENT')
        for r, loc_, scale_ in seen:
            if not (np.array_equal(loc_.float().cpu().numpy(), F0loc[r]) and np.array_equal(scale_.float().cpu().numpy(), F0scale[r])):
                raise Blocked('BLOCKED_MAPPING: live loc/scale differ from the F0 cache')
        ledger.intent(key, step); sync(DEVICE); tick = time.perf_counter(); opt.step(); sync(DEVICE)
        dt = fb + time.perf_counter() - tick; ledger.commit(key, step)
        clock.parts['main_updates'] = clock.parts.get('main_updates', 0.) + dt; updates_so_far += dt
        trace.append(dict(step=step, loss=float(total), gradient_norm=float(gn), active_step_s=dt))
        if step in H_CKPTS:
            if step == H_CKPTS[-1]: train_peak = gpu_peak()
            checkpoint(step)
    if fingerprint(model, frozen_only=True) != frozen0: raise Blocked('FROZEN_WEIGHTS_CHANGED')
    final = lora_state(model); changed = sum(not torch.equal(initial[n], v) for n, v in final.items())
    if changed == 0: raise Blocked('TRAINABLES_DID_NOT_CHANGE')
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    del fc, opt, model; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    transfer = dict(method='none (trained on the original model)')
    if arm in E_ARMS:
        with clock.part('full_model_load_transfer'): full, _ = load_base(BASECFG, DEVICE)
        bb0 = backbone_digest(full); attach_adapters(full, canon); fcF = Forecaster(full)
        factors = load_factors() if arm == 'E_EMLOC' else None
        for rec in curves:
            p = work/f'step_{rec["step"]}.pt'
            if sha(p) != rec['state_sha256']: raise Blocked('CHECKPOINT_CHANGED')
            st = torch.load(p, weights_only=True)
            t0 = clock.parts.get('transfer', 0.)
            with clock.part('transfer'):
                exported = corrected_state(st, canon, full, factors) if arm == 'E_EMLOC' else st
                load_lora(full, exported)
            if tensor_digest(lora_state(full)) != tensor_digest(exported): raise Blocked('TRANSFER_HASH_MISMATCH')
            torch.save(exported, work/f'transferred_{rec["step"]}.pt')
            rec['transferred_sha256'] = sha(work/f'transferred_{rec["step"]}.pt'); rec['transfer_s'] = clock.parts['transfer'] - t0
            t1 = clock.parts.get('val_eval', 0.)
            with clock.part('val_eval'):
                pred, _, _ = predict(fcF, panel, panel.origins['val'])
            np.save(work/f'val_full_{rec["step"]}.npy', pred)
            rec.update(val_record(pred, 'val')); rec['val_eval_s'] = clock.parts['val_eval'] - t1
            em = np.load(work/f'val_emulator_train_{rec["step"]}.npy')
            rec['train_vs_actual_sigma_rmse'] = math.sqrt(sigma_mse(em - pred, panel.sigma))
            if arm == 'E_EMLOC':
                with clock.part('diag_naive_transfer'):
                    load_lora(full, st); npred, _, _ = predict(fcF, panel, panel.origins['val'])
                rec.update(val_record(npred, 'naive_val'))
        if backbone_digest(full) != bb0: raise Blocked('BACKBONE_CHANGED_BY_TRANSFER')
        transfer = dict(method='EMLoC-style correction L=3 with explicit alpha/r' if arm == 'E_EMLOC' else 'direct LoRA copy',
                        backbone_unchanged=True, hash_checked=True)
        del full, fcF; gc.collect(); torch.cuda.empty_cache()
    else:
        for rec in curves:
            rec['transferred_sha256'] = rec['state_sha256']
    selected = choose_checkpoint(curves)
    with clock.part('selection_export'):
        src = work/(f'transferred_{selected["step"]}.pt' if arm in E_ARMS else f'step_{selected["step"]}.pt')
        exported = torch.load(src, weights_only=True); torch.save(exported, work/'selected_export.pt')
    basis = read_json(OUT/'COST_BASIS.json')
    info = dict(complete=True, key=key, arm=arm, seed=seed, emulator=kind, lora_modules=len(names),
                trainable_parameters=trainable_params, adapter_bytes=lora_bytes(exported), selected_step=selected['step'],
                selected_val_primary=selected['val_primary'], curves=curves, trace=trace, parts=clock.parts,
                charged_preparation=basis['charged'][arm], train_peak=train_peak, frozen_unchanged=True,
                changed_trainable_tensors=changed, initial_lora_sha256=tensor_digest(initial), transfer=transfer,
                schedule_sha256=sha(work/'schedule.npy'), unique_train_origins=int(len(np.unique(schedule))),
                selected_export_sha256=sha(work/'selected_export.pt'),
                full_process_wall_s=time.perf_counter()-start_all, rss_bytes=process_rss(),
                checkpoint_dir=work.relative_to(REPO).as_posix())
    write_json(outp, info)


# ============================================================ selection seal and TEST
def stage_select():
    assert_seal(); path = OUT/'MODEL_SELECTION.json'
    if path.exists(): return
    fits = {f'{a}_{s}': read_json(OUT/'fits'/f'{a}_{s}.json') for s in CFG['seeds'] for a in ARMS}
    for k, f in fits.items():
        if not f.get('complete'): raise Blocked('INCOMPLETE_FIT '+k)
        if f['selected_step'] != choose_checkpoint(f['curves'])['step']: raise Blocked('SELECTION_RULE_VIOLATED '+k)
    full_best = [min(r['val_primary'] for r in fits[f'F_FULL_{s}']['curves']) for s in CFG['seeds']]
    f0_val = fits[f'F_FULL_{CFG["seeds"][0]}']['curves'][0]['val_primary']
    target, already = rc.common_quality_target([float(np.mean(full_best))], f0_val, .005)
    write_json(path, dict(rule='transferred full-model VAL minimum; tie -> earlier step',
        choices={k: dict(selected_step=f['selected_step'], val_primary=f['selected_val_primary']) for k, f in fits.items()},
        fit_hashes={f'{k}.json': sha(OUT/'fits'/f'{k}.json') for k in fits},
        time_quality_target=dict(target=target, rule='1.005 x mean over the two F_FULL seeds of each seed best VAL checkpoint',
                                 f_full_best_val_by_seed=full_best, f0_val=f0_val, f0_meets_target=already),
        test_scoring_started=False, created_utc=time.time()))


def stage_test():
    assert_seal(); panel = _panel(); meta = _meta(); canon = meta['canonical']
    sel_path = OUT/'MODEL_SELECTION.json'; sel = read_json(sel_path); sel_hash = sha(sel_path)
    for fn, h in sel['fit_hashes'].items():
        if sha(OUT/'fits'/fn) != h: raise Blocked('FIT_CHANGED_AFTER_SELECTION')
    pred_dir = CACHE/'test_predictions'; pred_dir.mkdir(parents=True, exist_ok=True); clock = Clock()
    full, _ = load_base(BASECFG, DEVICE); bb0 = backbone_digest(full); attach_adapters(full, canon); fc = Forecaster(full)
    zero = {n+'.lora_A.weight': torch.zeros(CFG['lora_rank'], tuple(meta['shapes'][n])[1]) for n in canon}
    zero.update({n+'.lora_B.weight': torch.zeros(tuple(meta['shapes'][n])[0], CFG['lora_rank']) for n in canon})
    items = [('F0', zero)]
    for s in CFG['seeds']:
        for a in ARMS:
            f = read_json(OUT/'fits'/f'{a}_{s}.json'); work = REPO/f['checkpoint_dir']
            if sha(work/'selected_export.pt') != f['selected_export_sha256']: raise Blocked('EXPORT_CHANGED')
            items.append((f'{a}_{s}', torch.load(work/'selected_export.pt', weights_only=True)))
            if a == 'E_EMLOC':
                items.append((f'{a}_{s}_NAIVE', torch.load(work/f'step_{f["selected_step"]}.pt', weights_only=True)))
    for label, st in items:
        dest = pred_dir/f'{label}.npz'
        if dest.exists(): continue
        load_lora(full, st)
        with clock.part('test_prediction'):
            pred, _, _ = predict(fc, panel, panel.origins['test'])
        np.savez_compressed(dest, q=pred, origins=panel.origins['test'])
        with torch.no_grad(), autocast(DEVICE, BASECFG['precision']):
            x, _, g = tensor_batch(panel, panel.origins['test'][:1]); v = fc(x, g, panel.horizon)[1].float().cpu().numpy()
        np.testing.assert_allclose(v, pred[0], rtol=SET['prediction_rtol'], atol=SET['prediction_atol'])
    if backbone_digest(full) != bb0: raise Blocked('BACKBONE_CHANGED_DURING_TEST')
    if sha(sel_path) != sel_hash: raise Blocked('SELECTION_CHANGED_DURING_TEST')
    write_json(OUT/'PREDICTIONS_MANIFEST.json', dict(complete=True, selection_sha256=sel_hash,
        predictions={p.relative_to(REPO).as_posix(): sha(p) for p in sorted(pred_dir.glob('*.npz'))},
        all_saved_before_scoring=True, restore_first_origin_verified=True, evaluation_seconds=clock.parts,
        note='all TEST forecasts come from the original Chronos-2 plus the exported LoRA; *_NAIVE = EMLoC diagnostic without correction'))
