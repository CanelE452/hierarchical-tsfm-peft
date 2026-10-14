"""Learned two-branch binary alignment with explicitly causal future-pattern memory.

This is an equations 13--16 inspired adaptation, not an ExoTimer reproduction.
Prediction never accepts the current query's future target. FIT futures enter the
prediction path only as dated historical memory, with a mask at every input row.
"""
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

OLD = Path(__file__).resolve().parents[1] / 'smallk_predictive_alignment_20261013'
sys.path.insert(0, str(OLD))
import engine as en

METHODS = (
    'NO_ALIGN', 'B_COSINE', 'B_TANH', 'B_GUMBEL', 'B_ST',
    'B_FIXED_GUMBEL', 'B_UNIFORM', 'A_CONTINUOUS', 'A_BIHASH_GUMBEL',
    'A_BIHASH_SHUFFLE', 'A_BIHASH_DETACH_E', 'A_BIHASH_FIXED_MAPS',
)


def binary_class_logits(logits):
    """Each hash bit is its own two-class decision, not a categorical bit index."""
    return torch.stack((logits, -logits), dim=-1)


def binary_code(logits, training, temperature, mode, generator=None):
    if mode == 'continuous':
        return logits
    if mode == 'tanh':
        return torch.tanh(logits / temperature) if training else torch.where(logits >= 0, 1., -1.)
    if not training:
        return torch.where(logits >= 0, 1., -1.)
    if mode not in ('gumbel', 'st'):
        raise ValueError('UNKNOWN_BINARY_MODE: ' + mode)
    if generator is None:
        raise ValueError('DEDICATED_GENERATOR_REQUIRED')
    classes = binary_class_logits(logits)
    uniform = torch.rand(classes.shape, device=classes.device, dtype=classes.dtype, generator=generator)
    uniform = uniform.clamp(torch.finfo(uniform.dtype).eps, 1 - torch.finfo(uniform.dtype).eps)
    gumbel = -torch.log(-torch.log(uniform))
    probabilities = torch.softmax((classes + gumbel) / temperature, dim=-1)
    soft = 2 * probabilities[..., 0] - 1
    if mode == 'gumbel':
        return soft
    hard = torch.where(probabilities[..., 0] >= probabilities[..., 1], 1., -1.)
    return hard.detach() - soft.detach() + soft


def masked_attention(query, keys, eligible, temperature=.2, uniform=False):
    """Cosine retrieval, exact zero for disallowed or fully masked memory rows."""
    if keys.shape[0] == 0:
        return query.new_zeros((*query.shape[:2], 0))
    if uniform:
        weights = eligible.to(query.dtype)
        return weights / weights.sum(-1, keepdim=True).clamp_min(1)
    similarity = F.normalize(query, dim=-1, eps=1e-6) @ F.normalize(keys, dim=-1, eps=1e-6).T
    allowed = eligible.any(-1, keepdim=True)
    scores = (similarity / temperature).masked_fill(~eligible, -torch.inf)
    scores = torch.where(allowed, scores, torch.zeros_like(scores))
    return torch.softmax(scores, dim=-1) * eligible.to(query.dtype)


def masked_macro_loss(prediction, target, observed):
    count = observed.sum((0, 1))
    desired = torch.where(observed, target, prediction.detach())
    error = (prediction - desired).square().sum((0, 1))
    valid = count > 0
    if not valid.any():
        return prediction.sum() * 0
    return (error / count.clamp_min(1))[valid].mean()


class FutureAutoencoder(nn.Module):
    def __init__(self, channels, horizon=48, representation=32):
        super().__init__()
        self.channels, self.horizon = channels, horizon
        self.encoder = nn.Sequential(nn.Linear(2 * horizon * channels, 64), nn.GELU(), nn.Linear(64, representation))
        self.decoder = nn.Sequential(nn.Linear(representation, 64), nn.GELU(), nn.Linear(64, horizon * channels))

    def encode(self, delta, observed):
        value = torch.where(observed, delta, torch.zeros_like(delta)).flatten(1)
        features = torch.cat((value, observed.to(delta.dtype).flatten(1)), dim=1)
        return self.encoder(features)

    def decode(self, representation):
        return self.decoder(representation).reshape(*representation.shape[:-1], self.horizon, self.channels)

    def forward(self, delta, observed):
        representation = self.encode(delta, observed)
        return self.decode(representation), representation

    def reconstruction_loss(self, delta, observed):
        reconstruction, representation = self(delta, observed)
        return masked_macro_loss(reconstruction, delta, observed), representation


class FutureBiHash(nn.Module):
    def __init__(self, mean, basis, seed, method, bank_delta=None, bank_observed=None,
                 bank_origins=None, horizon=48, representation=32, bits=32):
        super().__init__()
        if method not in METHODS:
            raise ValueError('UNKNOWN_METHOD: ' + method)
        self.method, self.seed = method, int(seed)
        self.channels, self.k = basis.shape
        self.horizon, self.representation, self.bits = horizon, representation, bits
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            self.codec = en.Codec(mean, basis, seed, 'ANCHOR_NONLINEAR_D')
            self.student_projection = nn.Linear(self.k, representation)
            self.future_autoencoder = FutureAutoencoder(self.channels, horizon, representation)
            self.student_hash = nn.Linear(representation, bits)
            self.future_hash = nn.Linear(representation, bits)
            self.student_fusion = nn.Sequential(nn.Linear(2 * representation, 64), nn.GELU(), nn.Linear(64, self.k))
            self.memory_fusion = nn.Sequential(nn.Linear(2 * representation, 64), nn.GELU(), nn.Linear(64, representation))
            self.alpha_logit = nn.Parameter(torch.zeros(()))
            with torch.no_grad():
                self.student_fusion[-1].weight.mul_(.01)
                self.student_fusion[-1].bias.mul_(.01)
                self.memory_fusion[-1].weight.mul_(.01)
                self.memory_fusion[-1].bias.mul_(.01)
        if bank_delta is None:
            bank_delta = np.empty((0, horizon, self.channels), np.float32)
            bank_observed = np.empty_like(bank_delta, dtype=bool)
            bank_origins = np.empty(0, np.int64)
        delta = torch.as_tensor(np.array(bank_delta, copy=True), dtype=torch.float32)
        observed = torch.as_tensor(np.array(bank_observed, copy=True), dtype=torch.bool)
        origins = torch.as_tensor(np.array(bank_origins, copy=True), dtype=torch.long)
        if delta.shape != observed.shape or delta.shape != (len(origins), horizon, self.channels):
            raise ValueError('BANK_SHAPE_CONTRACT')
        if len(origins) > 1 and not bool((origins[1:] > origins[:-1]).all()):
            raise ValueError('BANK_ORIGINS_MUST_BE_UNIQUE_SORTED')
        if not bool(torch.isfinite(delta[observed]).all()):
            raise ValueError('NONFINITE_OBSERVED_BANK_TARGET')
        self.register_buffer('bank_delta', torch.where(observed, delta, torch.zeros_like(delta)))
        self.register_buffer('bank_observed', observed)
        self.register_buffer('bank_valid', observed.flatten(1).any(-1))
        self.register_buffer('bank_origins', origins)
        self.register_buffer('bank_end_exclusive', origins + horizon)
        self.register_buffer('memory_embedding', torch.zeros(len(origins), representation))
        self.register_buffer('memory_hash_logits', torch.zeros(len(origins), bits))
        self.register_buffer('memory_is_frozen', torch.tensor(False))
        self.register_buffer('noise_calls', torch.zeros((), dtype=torch.long))
        if method in ('B_FIXED_GUMBEL', 'A_BIHASH_FIXED_MAPS'):
            self.student_hash.requires_grad_(False)
            self.future_hash.requires_grad_(False)

    @staticmethod
    def temperature(update):
        return 1. - .75 * min(max(float(update), 0.), 600.) / 600.

    def _generator(self, device):
        generator = torch.Generator(device=device)
        generator.manual_seed(self.seed * 1000003 + int(self.noise_calls))
        self.noise_calls.add_(1)
        return generator

    def _mode(self):
        if self.method in ('B_COSINE', 'A_CONTINUOUS'):
            return 'continuous'
        if self.method == 'B_TANH':
            return 'tanh'
        return 'st' if self.method == 'B_ST' else 'gumbel'

    def _codes(self, student, future, update, future_logits=None):
        student_logits = self.student_hash(student)
        if future_logits is None:
            future_logits = self.future_hash(future)
        mode, temperature = self._mode(), self.temperature(update)
        query = binary_code(student_logits, self.training, temperature, mode,
            self._generator(student.device) if self.training and mode in ('gumbel', 'st') else None)
        keys = binary_code(future_logits, self.training, temperature, mode,
            self._generator(future.device) if self.training and mode in ('gumbel', 'st') else None)
        return query, keys

    def eligibility(self, origins, length):
        origins = torch.as_tensor(origins, dtype=torch.long, device=self.bank_origins.device)
        row_times = origins[:, None] - length + torch.arange(length, device=origins.device)
        eligible = self.bank_end_exclusive[None, None] <= row_times[:, :, None] + 1
        return eligible & self.bank_valid[None, None]

    def _memory(self):
        if not self.training and bool(self.memory_is_frozen):
            return self.memory_embedding, self.memory_hash_logits
        representation = self.future_autoencoder.encode(self.bank_delta, self.bank_observed)
        return representation, self.future_hash(representation)

    def encode_with_memory(self, x, origins, update=0, variant=None):
        z = self.codec.encode(x)
        if not self.method.startswith('B_'):
            return z, {'z': z, 'augmented_z': z}
        student = self.student_projection(z)
        memory, future_logits = self._memory()
        eligible = self.eligibility(origins, x.shape[1])
        if self.method == 'B_UNIFORM' or variant == 'uniform':
            attention = masked_attention(student, memory, eligible, uniform=True)
            query, keys = student, memory
        elif variant == 'tanh':
            query = torch.tanh(self.student_hash(student) / self.temperature(600))
            keys = torch.tanh(future_logits / self.temperature(600))
            attention = masked_attention(query, keys, eligible)
        else:
            query, keys = self._codes(student, memory, update, future_logits)
            attention = masked_attention(query, keys, eligible)
        retrieved = attention @ memory
        correction = self.student_fusion(torch.cat((student, retrieved), dim=-1))
        has_memory = eligible.any(-1, keepdim=True)
        augmented = z + torch.where(has_memory, correction, torch.zeros_like(correction))
        return augmented, {'z': z, 'student': student, 'augmented_z': augmented, 'memory': memory,
            'eligible': eligible, 'attention': attention, 'query_code': query, 'memory_code': keys,
            'has_memory': has_memory, 'student_correction': correction}

    def forward(self, x, origins, forecaster, update=0, return_details=False, variant=None):
        if variant not in (None, 'student', 'memory', 'uniform', 'tanh'):
            raise ValueError('UNKNOWN_INFERENCE_ABLATION')
        if variant is not None and self.training:
            raise ValueError('ABLATION_IS_INFERENCE_ONLY')
        augmented, details = self.encode_with_memory(x, origins, update, variant)
        future = forecaster(augmented)
        student_prediction = x[:, -1:] + self.codec.decode(future) - self.codec.decode(augmented[:, -1:])
        prediction = student_prediction
        if self.method.startswith('B_'):
            attention, student, memory = details['attention'], details['student'], details['memory']
            column_mass = attention.sum(1).unsqueeze(-1)
            pooled_student = (attention.transpose(1, 2) @ student) / column_mass.clamp_min(1e-6)
            expanded_memory = memory.unsqueeze(0).expand(len(x), -1, -1)
            correction = self.memory_fusion(torch.cat((expanded_memory, pooled_student), dim=-1))
            memory_delta = self.future_autoencoder.decode(expanded_memory + correction)
            weighted_delta = torch.einsum('bj,bjhc->bhc', attention[:, -1], memory_delta)
            memory_prediction = x[:, -1:] + weighted_delta
            alpha = torch.sigmoid(self.alpha_logit)
            fused = alpha * student_prediction + (1 - alpha) * memory_prediction
            final_has_memory = details['has_memory'][:, -1:].expand(-1, self.horizon, self.channels)
            prediction = torch.where(final_has_memory, fused, student_prediction)
            if variant == 'student': prediction = student_prediction
            if variant == 'memory': prediction = torch.where(final_has_memory, memory_prediction, student_prediction)
            details.update(memory_prediction=memory_prediction, alpha=alpha, memory_delta=memory_delta,
                memory_column_mass=column_mass, memory_pooled_student=pooled_student)
        details.update(student_prediction=student_prediction, prediction=prediction)
        return (prediction, details) if return_details else prediction

    def _alignment_loss(self, student, target, observed, x, origins, update, pair_permutation):
        delta = torch.where(observed, target - x[:, -1:], torch.zeros_like(target))
        future = self.future_autoencoder.encode(delta, observed)
        valid = observed.flatten(1).any(-1)
        if self.method == 'A_BIHASH_SHUFFLE':
            if pair_permutation is None:
                raise ValueError('SHUFFLE_REQUIRES_EXPLICIT_NONOVERLAP_BATCH_PERMUTATION')
            p = torch.as_tensor(pair_permutation, dtype=torch.long, device=x.device)
            if p.shape != (len(x),) or not torch.equal(torch.sort(p).values, torch.arange(len(x), device=x.device)):
                raise ValueError('SHUFFLE_NOT_BIJECTION')
            o = torch.as_tensor(origins, dtype=torch.long, device=x.device)
            if bool((torch.abs(o - o[p]) < self.horizon).any()):
                raise ValueError('SHUFFLE_TARGET_WINDOWS_OVERLAP')
            future, valid = future[p], valid & valid[p]
        student, future = student[valid], future[valid]
        if len(student) < 2:
            return student.sum() * 0, {'valid_pair_count': int(len(student)), 'contrastive_logits': student.new_empty((0, 0))}
        query, keys = self._codes(student, future, update)
        logits = F.normalize(query, dim=-1, eps=1e-6) @ F.normalize(keys, dim=-1, eps=1e-6).T / .2
        labels = torch.arange(len(student), device=x.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / (2 * np.log(len(student)))
        return loss, {'valid_pair_count': len(student), 'contrastive_logits': logits,
            'alignment_student_code': query, 'alignment_future_code': keys}

    def losses(self, x, origins, target, observed, forecaster, update=0, pair_permutation=None):
        prediction, details = self(x, origins, forecaster, update, True)
        main = masked_macro_loss(prediction, target, observed)
        delta = torch.where(observed, target - x[:, -1:], torch.zeros_like(target))
        reconstruction, future = self.future_autoencoder(delta, observed)
        rec = masked_macro_loss(reconstruction, delta, observed)
        auxiliary = main * 0
        if self.method.startswith('A_'):
            z = details['z'].detach() if self.method == 'A_BIHASH_DETACH_E' else details['z']
            student_rows = self.student_projection(z)
            summary = student_rows[:, -1] + student_rows.mean(1)
            auxiliary, alignment_details = self._alignment_loss(summary, target, observed, x, origins, update, pair_permutation)
            details.update(alignment_details)
        details.update(current_future_representation=future, current_future_reconstruction=reconstruction)
        return {'main': main, 'aux': auxiliary, 'rec': rec, 'prediction': prediction, 'details': details}

    @torch.no_grad()
    def freeze_memory(self):
        self.memory_embedding.copy_(self.future_autoencoder.encode(self.bank_delta, self.bank_observed))
        self.memory_hash_logits.copy_(self.future_hash(self.memory_embedding))
        self.memory_is_frozen.fill_(True)

    def load_future_pretrain(self, state):
        self.future_autoencoder.load_state_dict(state)
        self.memory_is_frozen.fill_(False)

    def metadata(self):
        codec = self.codec.metadata()
        registered = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if not self.method.startswith('B_'):
            inference_parameters = codec['active_structural_parameters']
            deployment_buffers = 0
        else:
            modules = (self.codec, self.student_projection, self.student_fusion, self.memory_fusion, self.future_autoencoder.decoder)
            inference_parameters = sum(p.numel() for m in modules for p in m.parameters()) + 1
            inference_parameters -= codec['structurally_cancelled_decoder_bias_parameters']
            if self.method != 'B_UNIFORM':
                inference_parameters += sum(p.numel() for p in self.student_hash.parameters())
            deployment_buffers = self.memory_embedding.numel() * self.memory_embedding.element_size()
            if self.method != 'B_UNIFORM':
                deployment_buffers += self.memory_hash_logits.numel() * self.memory_hash_logits.element_size()
            deployment_buffers += self.bank_end_exclusive.numel() * self.bank_end_exclusive.element_size() + len(self.bank_origins)
        return {'method': self.method, 'registered_parameters': registered, 'trainable_parameters': trainable,
            'deployment_active_structural_parameters': inference_parameters,
            'deployment_memory_bytes': deployment_buffers, 'memory_entries': len(self.bank_origins),
            'future_target_input_at_prediction': False, 'hash_bits': self.bits, 'representation_dim': self.representation,
            'hash_class_axis': 'last axis, 2 choices per bit', 'training_temperature': '1 -> .25 over 600 updates',
            'inference_binary_tie_rule': 'I >= 0 gives +1', 'retrieval': 'cosine normalized eps=1e-6, softmax / .2',
            'causal_memory_rule': 'bank origin + H <= raw row time + 1',
            'fixed_map_caveat': 'maps frozen; feature distributions E and phi remain learnable' if 'FIXED' in self.method else None,
            'codec': codec}
