"""Original B2 with only the stage-local PCA E frozen, plus an invariance audit."""
from contextlib import contextmanager
import hashlib

import torch


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


@contextmanager
def evaluation_scope(model, *, deterministic=False):
    modes = {module: module.training for module in model.modules()}
    try:
        model.eval()
        with torch.no_grad():
            yield model
    finally:
        for module, mode in modes.items():
            module.training = mode


@contextmanager
def patch_core(fit, training):
    original_factory, original_body, original_segment = fit.create_model, fit._stage_body, training.train_segment
    stages, active = [], []

    def create_model(spec, task, fit_end, foundation):
        if spec['method_id'] != 'E_FIXED' or spec['family'] != 'B2':
            raise ValueError('Only the registered original B2 E-fixed model is allowed')
        model, norm = original_factory(spec, task, fit_end, foundation)
        if not model.E.requires_grad or tuple(model.E.shape) != (task['physical'].shape[1], spec['latent_k']):
            raise ValueError('Original E parameter differs')
        model.E.requires_grad_(False)
        if not torch.equal(model.E.detach().cpu(), torch.as_tensor(norm['basis']).float()):
            raise ValueError('Fixed E is not the stage-local fitted PCA initialization')
        return model, norm

    def train_segment(model, windows, origins, sampler, optimizer, *args, **kwargs):
        if not active or active[-1]['model'] is not model:
            raise ValueError('Training occurred outside the E invariance audit')
        audit = active[-1]
        if any(parameter is model.E for group in optimizer.param_groups for parameter in group['params']):
            raise ValueError('Frozen E was included in optimizer parameter groups')
        if model.E.requires_grad or model.E.grad is not None:
            raise ValueError('Frozen E has a gradient or is trainable')
        audit['optimizer_parameter_group_checks'] += 1
        result = original_segment(model, windows, origins, sampler, optimizer, *args, **kwargs)
        if not torch.equal(model.E.detach().cpu(), audit['initial']):
            raise ValueError('E changed during an optimizer segment')
        return result

    def stage_body(spec, task, fit_end, stop_bounds, epochs, stage_dir, public_dir,
                   foundation, deadline, contract, model, norm):
        initial = model.E.detach().cpu().clone()
        audit = dict(model=model, initial=initial, optimizer_parameter_group_checks=0)
        active.append(audit)
        try:
            result = original_body(spec, task, fit_end, stop_bounds, epochs, stage_dir, public_dir,
                                   foundation, deadline, contract, model, norm)
            if (model.E.requires_grad or model.E.grad is not None
                    or not torch.equal(model.E.detach().cpu(), initial)
                    or not torch.equal(result['model']['E'], initial)):
                raise ValueError('Live or selected E differs from its stage initialization')
            expected_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            if result['summary']['trainable_parameters'] != expected_count:
                raise ValueError('Optimizer trainable parameter count differs')
            if result['summary']['updates'] and not audit['optimizer_parameter_group_checks']:
                raise ValueError('Optimizer E exclusion was not observed')
            record = dict(status='PASS', fit_end=int(fit_end), phase='search' if stop_bounds else 'refit',
                          initial_E_sha256=tensor_sha(initial), live_E_sha256=tensor_sha(model.E),
                          selected_E_sha256=tensor_sha(result['model']['E']),
                          E_requires_grad=False, E_gradient_absent=True,
                          frozen_parameters=initial.numel(), trainable_parameters=expected_count,
                          optimizer_parameter_group_checks=audit['optimizer_parameter_group_checks'],
                          updates=result['summary']['updates'],
                          optimizer_evidence='Actual optimizer group checks at every train_segment; zero-update stage needs no optimizer step.')
            result['summary']['E_INVARIANCE'] = record
            stages.append(record)
            return result
        finally:
            active.pop()

    fit.create_model, fit._stage_body, training.train_segment = create_model, stage_body, train_segment
    try:
        yield stages
    finally:
        fit.create_model, fit._stage_body, training.train_segment = original_factory, original_body, original_segment
