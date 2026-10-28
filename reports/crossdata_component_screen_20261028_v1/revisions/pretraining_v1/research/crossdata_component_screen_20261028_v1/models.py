"""Dispatch the unchanged B2 and three previously registered interventions."""
from contextlib import contextmanager, nullcontext

import torch

from research.jena_p_branch_20261027_v1 import models as branches
from research.ppt_justification_20261028_v1 import models as decoder
from research.ppt_e_fixed_20261028_v1 import models as fixed_e


METHODS = ('OURS', 'TIME_ONLY', 'A5_LAG0', 'E_FIXED')
_active = False


def _check_model(model, norm, task, spec, method_id):
    c, k = task['physical'].shape[1], spec['latent_k']
    if (model.length, model.horizon, model.channels, model.latent_k) != (task['L'], task['H'], c, k):
        raise ValueError('Original B2 axes differ from the requested task')
    if model.front.kind != 'B2' or model.encoder_kind != 'pointwise' or model.k_grid is not None:
        raise ValueError('Only the original pointwise B2 with one fixed K is allowed')
    if not isinstance(model.E, torch.nn.Parameter) or model.E.requires_grad != (method_id != 'E_FIXED'):
        raise ValueError('E trainability differs from the selected intervention')
    if not torch.equal(model.E.detach().cpu(), torch.as_tensor(norm['basis']).float()):
        raise ValueError('E differs from its stage-local PCA initialization')
    if any(not p.requires_grad for name, p in model.named_parameters() if name != 'E'):
        raise ValueError('Only E_FIXED may freeze an adapter parameter')
    if method_id == 'TIME_ONLY':
        if not isinstance(model.front.mixer, branches.SingleBranchMixer) or model.front.mixer.method != 'P_TIME_ONLY':
            raise ValueError('The retained branch is not the registered temporal branch')
    elif isinstance(model.front.mixer, branches.SingleBranchMixer):
        raise ValueError('A full P was required')
    if method_id == 'A5_LAG0':
        if not isinstance(model, decoder.DecoderIntervention) or tuple(model.D0.shape) != (k, c):
            raise ValueError('The decoder is not the registered anchored lag0 intervention')
    elif tuple(model.D.shape) != (7, k, c):
        raise ValueError('The original anchored lag7 decoder was required')


@contextmanager
def evaluation_scope(model, deterministic=False):
    with branches.evaluation_scope(model, deterministic=deterministic):
        yield model


@contextmanager
def patch_core(fit, training, method_id):
    global _active
    if method_id not in METHODS:
        raise ValueError('Unknown cross-data intervention')
    inherited_modules = {branches.__name__, decoder.__name__, fixed_e.__name__}
    if (_active or branches._original_factory is not None or decoder._original_factory is not None
            or getattr(fit.create_model, '__module__', None) in inherited_modules):
        raise RuntimeError('Only one model patch context may be active')
    _active = True
    try:
        if method_id == 'TIME_ONLY':
            scope = branches.patch_core(fit)
        elif method_id == 'A5_LAG0':
            scope = decoder.patch_core(fit)
        elif method_id == 'E_FIXED':
            scope = fixed_e.patch_core(fit, training)
        else:
            scope = nullcontext([])
        with scope as stage_audits:
            factory = fit.create_model

            def create_model(spec, task, fit_end, foundation):
                if spec.get('method_id') != method_id or spec.get('family') != 'B2':
                    raise ValueError('Model creation differs from the active B2 intervention')
                internal_spec = dict(spec)
                if method_id == 'TIME_ONLY':
                    internal_spec['method_id'] = 'P_TIME_ONLY'
                model, norm = factory(internal_spec, task, fit_end, foundation)
                _check_model(model, norm, task, spec, method_id)
                model.method_id = method_id
                return model, norm

            fit.create_model = create_model
            try:
                yield stage_audits if stage_audits is not None else []
            finally:
                fit.create_model = factory
    finally:
        _active = False
