"""Registered structural variants around the unchanged INFORMS model factory."""
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


BASE_FAMILY = dict(OURS='B2', C1='PCA_FIXED', C2='ADAPTS', C3='ADAPTS',
                   A1='B0', A2='B2', A3='B2', A4='B2', A5='B2')
_original_factory = None


class PointwiseMLP(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.input = nn.Linear(channels, hidden)
        self.output = nn.Linear(hidden, channels)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, x):
        return self.output(F.gelu(self.input(x)))


class PointwiseCapacityMixer(nn.Module):
    def __init__(self, original, channels, seed):
        super().__init__()
        temporal_parameters = sum(p.numel() for p in original.temporal.parameters())
        self.hidden = (temporal_parameters - channels) // (2 * channels + 1)
        self.temporal_norm = original.temporal_norm
        self.feature_norm = original.feature_norm
        self.feature = original.feature
        # CPU initialization is isolated without reseeding any CUDA generator.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(int(seed))
            self.temporal = PointwiseMLP(channels, self.hidden)
        reference = next(original.feature.parameters())
        self.temporal.to(device=reference.device, dtype=reference.dtype)

    def forward(self, x):
        y = x + self.temporal(self.temporal_norm(x))
        return y + self.feature(self.feature_norm(y))


class EncoderDropout(nn.Dropout):
    def forward(self, x, *, enabled):
        return F.dropout(x, self.p, training=enabled, inplace=False)


class DropoutLinearAdapter(nn.Module):
    def __init__(self, original, seed):
        super().__init__()
        self.encoder = original.encoder
        self.dropout = EncoderDropout(p=.1)
        self.decoder = original.decoder
        self.length, self.horizon = original.length, original.horizon
        self.channels, self.latent_k = original.channels, original.latent_k
        self.seed = int(seed)
        self._evaluation = None

    def forward(self, x, foundation):
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.length, self.channels):
            raise ValueError('Expected the registered [B,L,C] history')
        if not self.training and self._evaluation is None:
            raise RuntimeError('C3 evaluation requires evaluation_scope around the whole evaluation')
        samples = 10 if self._evaluation == 'mc' else 1
        enabled = self._evaluation != 'deterministic' and (self.training or self._evaluation == 'mc')
        encoded = self.encoder(x)
        forecasts = []
        for _ in range(samples):
            future = foundation(self.dropout(encoded, enabled=enabled))
            if tuple(future.shape) != (x.shape[0], self.horizon, self.latent_k):
                raise ValueError('Foundation output must be [B,H,K]')
            forecasts.append(future)
        latent = forecasts[0] if samples == 1 else torch.stack(forecasts).mean(0)
        return self.decoder(latent)


class DecoderVariant(nn.Module):
    def __init__(self, original, method):
        super().__init__()
        self.front, self.E = original.front, original.E
        self.register_buffer('center', original.center)
        self.length, self.horizon = original.length, original.horizon
        self.channels, self.latent_k = original.channels, original.latent_k
        self.kmax, self.k_grid = original.kmax, original.k_grid
        self.encoder_kind = original.encoder_kind
        self.method_id = method
        if method == 'A4':
            self.W = nn.Parameter(torch.linalg.pinv(self.E.detach()).clone())
            self.b = nn.Parameter(self.center.detach().clone())
        elif method == 'A5':
            self.D0 = nn.Parameter(original.D[0].detach().clone())
        else:
            raise ValueError('Only A4/A5 replace the decoder')

    @property
    def pca_center(self):
        return self.center

    def _k(self, latent_k):
        k = self.latent_k if latent_k is None else latent_k
        if type(k) is not int or k != self.latent_k:
            raise ValueError('This campaign uses one fixed K per dataset')
        return k

    def encode(self, x, latent_k=None):
        self._k(latent_k)
        if x.ndim != 3 or tuple(x.shape[1:]) != (self.length, self.channels):
            raise ValueError('Expected the registered [B,L,C] history')
        return (self.front(x) - self.center) @ self.E

    def decode(self, x, z, future, latent_k=None):
        k = self._k(latent_k)
        if tuple(z.shape) != (x.shape[0], self.length, k) or tuple(future.shape) != (x.shape[0], self.horizon, k):
            raise ValueError('Latent history and foundation forecast shapes differ')
        if self.method_id == 'A4':
            return future @ self.W + self.b
        return x[:, -1:] + (future - z[:, -1:]) @ self.D0

    def forward(self, x, foundation, latent_k=None):
        z = self.encode(x, latent_k)
        return self.decode(x, z, foundation(z), latent_k)


def create_model(spec, task, fit_end, foundation):
    import fit_base
    method = spec['method_id']
    if method not in BASE_FAMILY:
        raise ValueError('Unregistered comparative/ablation method')
    factory = _original_factory or fit_base.create_model
    model, normalization = factory(dict(spec, family=BASE_FAMILY[method]), task, fit_end, foundation)
    if method == 'C3':
        model = DropoutLinearAdapter(model, spec['seed'])
    elif method == 'A2':
        model.front.mixer = PointwiseCapacityMixer(model.front.mixer, model.channels, spec['seed'])
    elif method == 'A3':
        from research.adapter_diagnostics_20261017_v2.new_jobs import qr_basis
        basis = qr_basis(model.channels, model.latent_k, 10000000 + int(spec['seed']))
        with torch.no_grad():
            model.E.copy_(torch.as_tensor(basis, device=model.E.device))
            model.D.zero_()
            model.D[0].copy_(torch.linalg.pinv(model.E))
        normalization['basis'] = basis
        normalization['basis_initialization'] = 'Gaussian_reduced_QR_separate_CPU_generator_fixed_signs'
    elif method in ('A4', 'A5'):
        model = DecoderVariant(model, method)
    model.method_id = method
    return model, normalization


def predict_model(model, x, foundation, family, latent_k=None):
    if model.method_id in ('C2', 'C3'):
        return model(x, foundation)
    return model(x, foundation, latent_k)


@contextmanager
def evaluation_scope(model, *, deterministic=False):
    """Reset C3's MC stream once, enclosing every batch in one evaluation."""
    modes = {module: module.training for module in model.modules()}
    previous = getattr(model, '_evaluation', None)
    devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})
    is_dropout = isinstance(model, DropoutLinearAdapter)
    try:
        model.eval()
        if is_dropout:
            model._evaluation = 'deterministic' if deterministic else 'mc'
            with torch.random.fork_rng(devices=devices):
                seed = model.seed + 7000000
                torch.random.default_generator.manual_seed(seed)
                for index in devices:
                    torch.cuda.default_generators[index].manual_seed(seed)
                with torch.no_grad():
                    yield model
        else:
            with torch.no_grad():
                yield model
    finally:
        if is_dropout:
            model._evaluation = previous
        for module, training in modes.items():
            module.training = training


@contextmanager
def patch_core(fit_module):
    """Worker-process-local hooks; existing source files remain unchanged."""
    global _original_factory
    if _original_factory is not None:
        raise RuntimeError('The model hooks are already installed')
    originals = fit_module.create_model, fit_module.predict_model, fit_module._evaluate
    _original_factory = originals[0]

    def evaluate(model, *args, **kwargs):
        if isinstance(model, DropoutLinearAdapter) and model._evaluation is None:
            with evaluation_scope(model):
                return originals[2](model, *args, **kwargs)
        return originals[2](model, *args, **kwargs)

    fit_module.create_model = create_model
    fit_module.predict_model = predict_model
    fit_module._evaluate = evaluate
    try:
        yield
    finally:
        fit_module.create_model, fit_module.predict_model, fit_module._evaluate = originals
        _original_factory = None
