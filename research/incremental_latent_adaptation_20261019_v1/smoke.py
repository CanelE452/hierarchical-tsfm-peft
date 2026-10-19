"""Actual parent/F input-gradient and short-profile gate, without V access."""
import time

import numpy as np
import torch

from research.adapter_diagnostics_20261017_v2.existing_jobs import _foundation
from research.informs_broad_validation_20261015_v1.networks import module_state_hash
from research.informs_broad_validation_20261015_v1.training import macro_loss, PermutationStream
from . import common, data_contract, models


def run(oldctx, dataset='jena'):
    data = data_contract.load(oldctx, dataset, 92601)
    device = oldctx['device']
    torch.set_num_threads(2)
    pool, foundation = _foundation(oldctx, oldctx['modules'], data['task']['H'], device)
    foundation.row_microbatch, foundation.checkpoint_inputs = 128, False
    model = models.make(data['s_model'], 'RESIDUAL_F', 92601).to(device)
    g = data['g_model'].to(device)
    g_hash = module_state_hash(g)
    s = data['s_model'].to(device)
    x = torch.tensor(data['x'][:4], device=device)
    with torch.no_grad():
        gp = g(x).cpu().numpy()
        sp = s(x, foundation).cpu().numpy()
        np.testing.assert_allclose(gp, data['g'][:4], atol=1e-5, rtol=1e-4)
        np.testing.assert_allclose(sp, data['s'][:4], atol=1e-5, rtol=1e-4)
    del s
    fit, select = (data['split'][k] for k in ('fit_indices', 'select_indices'))
    arrays = {k: torch.as_tensor(data[k], device=device) for k in ('x','g','y','mask')}
    counts = arrays['mask'][fit].sum((0,1))
    sampler = PermutationStream(fit, 92601)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.001, weight_decay=0)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    norms = []
    first_E_zero = False
    initial_state = module_state_hash(model)
    for update in range(30):
        batch = sampler.draw(16)
        optimizer.zero_grad(set_to_none=True)
        for lo in range(0, len(batch), 4):
            idx = batch[lo:lo+4]
            c = model(arrays['x'][idx], foundation)
            loss = macro_loss(arrays['g'][idx]+c, arrays['y'][idx], arrays['mask'][idx], counts, len(fit), len(batch))
            loss.backward()
        if update == 0:
            first_E_zero = bool(torch.count_nonzero(model.E.grad) == 0)
        norms.append(dict(E=float(model.E.grad.norm()),
                          P=float(sum(p.grad.square().sum() for p in model.front.parameters() if p.grad is not None).sqrt())))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    torch.cuda.synchronize()
    update_seconds = time.perf_counter()-started
    began = time.perf_counter()
    pred = []
    with torch.no_grad():
        for lo in range(0, len(select), 4):
            ix = select[lo:lo+4]
            pred.append((arrays['g'][ix]+model(arrays['x'][ix], foundation)).cpu().numpy())
    torch.cuda.synchronize()
    eval_seconds = time.perf_counter()-began
    foundation.assert_frozen_unchanged()
    assert module_state_hash(g) == g_hash
    assert first_E_zero and max(v['E'] for v in norms[1:]) > 0 and max(v['P'] for v in norms[1:]) > 0
    assert module_state_hash(model) != initial_state
    result = dict(status='PASS', dataset=dataset, updates=30, exposures=480,
        first_zero_D_P_E_gradient_zero=first_E_zero, later_gradient_norms=norms,
        thirty_update_seconds=update_seconds, seconds_per_update=update_seconds/30,
        Mselect_evaluation_seconds=eval_seconds,
        smoke_only_score=common.score(np.concatenate(pred),data['y'][select],data['mask'][select]),
        peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,
        peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
        parent_reload_parity='PASS', F_frozen_unchanged=True, G_frozen_unchanged=True,
        F_tensor_sha256=foundation.initial_hash, F_weight_parameter_gradient_count=sum(p.grad is not None for p in foundation.model.parameters()),
        F_rows_forward=foundation.physical_streams, F_calls=foundation.physical_calls,
        F_row_microbatch=128, checkpoint_inputs=False, new_V_numeric_reads=0,
        smoke_states_used_as_research_results=False,
        torch_version=torch.__version__, GPU=torch.cuda.get_device_name(), dtype='float32')
    pool.close()
    common.atomic(common.REPORT/f'SMOKE_{dataset}.json', result)
    common.event('SMOKE', dataset=dataset, status='PASS', seconds_per_update=update_seconds/30,
                 Mselect_evaluation_seconds=eval_seconds, peak_allocated_MiB=result['peak_allocated_MiB'])
    return result
