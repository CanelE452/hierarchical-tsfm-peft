"""Fixed FIT future coordinates and auxiliary losses, removed at deployment."""
import hashlib
import numpy as np
import torch
from torch import nn
from sklearn.cluster import KMeans

METHODS = ('BASE_Q', 'LATENT_FUTURE_MSE', 'SUMMARY_MSE', 'SHUFFLED_SUMMARY', 'FIXED_REGIME', 'LEARNED_REGIME', 'SHUFFLED_REGIME')
WEIGHTS = (.03, .1)


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def future_coordinates(target, observed, last, basis, return_diagnostics=False):
    """Observed weighted LS; full observations equal orthonormal projection."""
    n, h, _ = target.shape
    k = basis.shape[1]
    coordinates, valid = np.zeros((n, h, k), np.float32), np.zeros((n, h), bool)
    groups = {}
    for i in range(n):
        for t in range(h):
            groups.setdefault(observed[i, t].tobytes(), []).append((i, t))
    conditions=[]
    for rows in groups.values():
        mask = observed[rows[0][0], rows[0][1]]
        b = basis[mask].astype(np.float64)
        if mask.sum() < k or np.linalg.matrix_rank(b, tol=1e-8) < k:
            continue
        ij = np.array(rows)
        delta = target[ij[:, 0], ij[:, 1]][:, mask]-last[ij[:, 0]][:, mask]
        coordinates[ij[:, 0], ij[:, 1]] = delta@np.linalg.pinv(b).T
        valid[ij[:, 0], ij[:, 1]] = True
        conditions.extend([float(np.linalg.cond(b))]*len(rows))
    if return_diagnostics:
        finite=coordinates[valid]
        return coordinates,valid,{'observed_basis_condition_quantiles':np.quantile(conditions,[0,.5,.9,.99,1]).tolist() if conditions else [],
            'target_latent_absolute_quantiles':np.quantile(np.abs(finite),[0,.5,.9,.99,1]).tolist() if len(finite) else []}
    return coordinates, valid


def summary_numpy(coordinates, valid):
    n, h, k = coordinates.shape
    count = valid.sum(1)
    mean = (coordinates*valid[:, :, None]).sum(1)/np.maximum(count[:, None], 1)
    times = np.arange(h, dtype=np.float64)
    tmean = (times[None, :]*valid).sum(1)/np.maximum(count, 1)
    centered = times[None, :]-tmean[:, None]
    denominator = (centered**2*valid).sum(1)
    slope = (coordinates*centered[:, :, None]*valid[:, :, None]).sum(1)/np.where(denominator[:,None]>0,denominator[:,None],1)
    result = np.concatenate((mean, coordinates[:, -1], slope), axis=1).astype(np.float32)
    mask = np.concatenate((np.repeat((count>0)[:, None], k, 1), np.repeat(valid[:, -1:], k, 1),
                           np.repeat((denominator>0)[:, None], k, 1)), axis=1)
    return result, mask


def summary_torch(coordinates, valid):
    count = valid.sum(1).clamp_min(1)
    mean = (coordinates*valid[:, :, None]).sum(1)/count[:, None]
    times = torch.arange(coordinates.shape[1], device=coordinates.device, dtype=coordinates.dtype)
    tmean = (times[None, :]*valid).sum(1)/count
    centered = times[None, :]-tmean[:, None]
    denominator = (centered.square()*valid).sum(1)
    denominator = torch.where(denominator>0,denominator,torch.ones_like(denominator))
    slope = (coordinates*centered[:, :, None]*valid[:, :, None]).sum(1)/denominator[:, None]
    return torch.cat((mean, coordinates[:, -1], slope), dim=1)


def nonoverlap_permutation(origins, seed, horizon=48):
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        permutation = rng.permutation(len(origins))
        if np.all(np.abs(origins-origins[permutation]) >= horizon):
            return permutation
    # Deterministic bipartite assignment if rejection is too restrictive.
    from scipy.optimize import linear_sum_assignment
    cost = rng.random((len(origins), len(origins)))
    cost[np.abs(origins[:, None]-origins[None, :]) < horizon] = 1e9
    rows, cols = linear_sum_assignment(cost)
    if np.any(cost[rows, cols] >= 1e9):
        raise ValueError('NO_BIJECTIVE_NONOVERLAP_FIT_SHUFFLE')
    return cols


class Alignment(nn.Module):
    def __init__(self, target, observed, history, basis, origins, seed):
        super().__init__()
        k = basis.shape[1]
        self.q = nn.Linear(k, k)
        with torch.no_grad():
            self.q.weight.copy_(torch.eye(k))
            self.q.bias.zero_()
        coord, valid,projection_diagnostic = future_coordinates(target, observed, history[:, -1], basis,True)
        summ, smask = summary_numpy(coord, valid)
        usable = smask.all(1)
        means = np.array([summ[smask[:, j], j].mean() if smask[:,j].any() else 0. for j in range(3*k)], np.float32)
        stds = np.array([summ[smask[:, j], j].std() if smask[:,j].any() else 1. for j in range(3*k)], np.float32)
        stds = np.where(stds>1e-6, stds, 1).astype(np.float32)
        normalized = (summ-means)/stds
        latent_mean = coord[valid].mean(0) if valid.any() else np.zeros(k,np.float32)
        latent_std = coord[valid].std(0) if valid.any() else np.ones(k,np.float32)
        latent_std = np.where(latent_std>1e-6, latent_std, 1).astype(np.float32)
        rng = np.random.default_rng(seed)
        fixed = rng.standard_normal((8, 3*k)).astype(np.float32)
        model = KMeans(n_clusters=8, n_init=10, random_state=seed).fit(normalized[usable]) if usable.sum()>=8 else None
        learned = model.cluster_centers_.astype(np.float32) if model is not None else np.zeros((8,3*k),np.float32)
        usable_indices=np.flatnonzero(usable)
        permutation=np.arange(len(origins))
        shuffle_error=None
        try:
            if len(usable_indices)<2: raise ValueError('INSUFFICIENT_COMPLETE_ORIGINS_FOR_SHUFFLE')
            permutation[usable_indices]=usable_indices[nonoverlap_permutation(origins[usable], seed)]
        except ValueError as error: shuffle_error=str(error)
        for name, array in {'target': coord, 'valid': valid, 'summary_target': normalized,
            'summary_valid': smask, 'means': means, 'stds': stds, 'latent_mean': latent_mean,
            'latent_std': latent_std, 'fixed': fixed, 'learned': learned, 'shuffle': permutation}.items():
            self.register_buffer(name, torch.from_numpy(array))
        self.metadata = {'basis_sha256': digest(basis), 'target_sha256': digest(coord),**projection_diagnostic,
            'valid_sha256': digest(valid), 'summary_mean': means.tolist(), 'summary_std': stds.tolist(),
            'summary_valid_dimensions': smask.sum(0).tolist(), 'valid_horizons': int(valid.sum()),
            'total_horizons': int(valid.size), 'complete_summary_origins': int(usable.sum()),
            'projection_rule': 'Observed weighted least squares; >=K observed channels and rank K; full observations equal (Y-x_last)B',
            'P': 8, 'temperature': 1., 'kmeans_n_init': 10, 'prototype_fitting_scope': 'FIT_FUTURE_ONLY',
            'prototype_fixed_sha256': digest(fixed), 'prototype_learned_sha256': digest(learned),
            'shuffle_indices': permutation.tolist(), 'shuffle_sha256': digest(permutation),
            'shuffle_min_origin_distance_complete_subset': int(np.min(np.abs(origins[usable]-origins[permutation[usable]]))) if usable.any() else None,
            'shuffle_available':shuffle_error is None,'shuffle_failure_reason':shuffle_error,
            'learned_prototypes_available':model is not None,
            'shuffle_usable_subset_indices':usable_indices.tolist(),
            'shuffle_is_bijection': bool(len(np.unique(permutation))==len(origins)),
            'fixed_occupancy': np.bincount(np.argmin(((normalized[usable, None]-fixed)**2).sum(2),1),minlength=8).tolist(),
            'learned_occupancy': np.bincount(model.labels_,minlength=8).tolist() if model is not None else None,
            'fixed_center_norms':np.linalg.norm(fixed,axis=1).tolist(),
            'learned_center_norms':np.linalg.norm(learned,axis=1).tolist(),
            'fixed_pairwise_distances':np.sqrt(((fixed[:,None]-fixed[None])**2).sum(2)).tolist(),
            'learned_pairwise_distances':np.sqrt(((learned[:,None]-learned[None])**2).sum(2)).tolist(),
            'deployment_head_removed': True}

    @staticmethod
    def distribution(summary, prototypes):
        return torch.softmax(-(summary[:, None]-prototypes).square().sum(-1), dim=-1)

    def loss(self, method, delta, indices):
        student = self.q(delta)
        valid = self.valid[indices]
        if method == 'BASE_Q':
            return student.sum()*0
        if method == 'LATENT_FUTURE_MSE':
            if not self.valid.any(): raise ValueError('LATENT_TARGET_HAS_NO_VALID_FIT_HORIZON')
            error = ((student-self.target[indices])/self.latent_std).square()
            return (error*valid[:, :, None]).sum()/(valid.sum().clamp_min(1)*student.shape[-1])
        summary = (summary_torch(student, valid)-self.means)/self.stds
        if not self.summary_valid.all(1).any(): raise ValueError('SUMMARY_TARGET_HAS_NO_COMPLETE_FIT_ORIGIN')
        if method.startswith('SHUFFLED') and not self.metadata['shuffle_available']: raise ValueError(self.metadata['shuffle_failure_reason'])
        if method in ('SUMMARY_MSE','SHUFFLED_SUMMARY'):
            target_indices=self.shuffle[indices] if method=='SHUFFLED_SUMMARY' else indices
            mask=self.summary_valid[indices].all(1)[:,None].expand_as(summary)
            return ((summary-self.summary_target[target_indices]).square()*mask).sum()/mask.sum().clamp_min(1)
        prototypes = self.fixed if method == 'FIXED_REGIME' else self.learned
        if method!='FIXED_REGIME' and not self.metadata['learned_prototypes_available']: raise ValueError('INSUFFICIENT_COMPLETE_FIT_FUTURE_SUMMARIES_FOR_P8')
        target_indices = self.shuffle[indices] if method == 'SHUFFLED_REGIME' else indices
        usable = self.summary_valid[indices].all(1)&self.summary_valid[target_indices].all(1)
        p = self.distribution(self.summary_target[target_indices], prototypes).detach()
        logq = torch.log_softmax(-(summary[:, None]-prototypes).square().sum(-1),dim=-1)
        kl = (p*(p.clamp_min(1e-12).log()-logq)).sum(1)
        return (kl*usable).sum()/usable.sum().clamp_min(1)

    def diagnostics(self, delta):
        with torch.no_grad():
            value = self.q(delta)
            summ = (summary_torch(value,self.valid)-self.means)/self.stds
            result = {}
            for name in ('fixed','learned'):
                if name=='learned' and not self.metadata['learned_prototypes_available']:
                    result[name]={'unavailable':'Fewer than 8 complete FIT summaries'}
                    continue
                teacher = self.distribution(self.summary_target,getattr(self,name))
                student = self.distribution(summ,getattr(self,name))
                usable = self.summary_valid.all(1)
                if not usable.any():
                    result[name]={'unavailable':'No complete FIT summaries'}
                    continue
                teacher,student = teacher[usable],student[usable]
                constant = teacher.mean(0)
                kl = lambda q: (teacher*(teacher.clamp_min(1e-12).log()-q.clamp_min(1e-12).log())).sum(1).mean()
                result[name] = {'teacher_mean_distribution': constant.cpu().tolist(),
                    'student_mean_distribution': student.mean(0).cpu().tolist(),
                    'teacher_entropy': float(-(teacher*teacher.clamp_min(1e-12).log()).sum(1).mean()),
                    'student_entropy': float(-(student*student.clamp_min(1e-12).log()).sum(1).mean()),
                    'student_vs_teacher_kl': float(kl(student)), 'constant_distribution_kl':float(kl(constant[None].expand_as(teacher)))}
            flattened = value.reshape(-1,value.shape[-1]).double()
            cov = torch.cov(flattened.T)
            if value.shape[-1] == 1: cov = cov.reshape(1,1)
            eig = torch.linalg.eigvalsh(cov).clamp_min(0)
            weights = eig/eig.sum().clamp_min(1e-20)
            result['student_variance'] = flattened.var(0).cpu().tolist()
            result['student_effective_rank'] = float(torch.exp(-(weights*weights.clamp_min(1e-20).log()).sum()))
            for label,coordinates in (('before_q',delta),('after_q',value)):
                mask=self.valid[:,:,None]
                error=((coordinates-self.target)/self.latent_std).square()
                result[label+'_latent_future_normalized_mse']=float((error*mask).sum()/(self.valid.sum().clamp_min(1)*value.shape[-1]))
                summary=(summary_torch(coordinates,self.valid)-self.means)/self.stds
                usable=self.summary_valid.all(1)
                result[label+'_complete_summary_normalized_mse']=float((summary[usable]-self.summary_target[usable]).square().mean()) if usable.any() else None
            return result
