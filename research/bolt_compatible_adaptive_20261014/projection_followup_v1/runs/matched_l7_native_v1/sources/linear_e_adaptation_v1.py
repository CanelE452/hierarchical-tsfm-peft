"""Pure Torch linear E/causal-lag D; no model/data loaders or execution on import."""
from operator import index

import torch
from torch import nn


def require(value, message):
    if not value: raise ValueError(message)


class LinearLagAdapter(nn.Module):
    def __init__(self, center, encoder, decoder, lags, learn_E=True):
        super().__init__()
        try:
            values = tuple(lags)
            require(not any(isinstance(v, bool) for v in values), "INTEGER_LAGS_REQUIRED")
            values = tuple(index(v) for v in values)
        except TypeError as error:
            raise ValueError("INTEGER_LAGS_REQUIRED") from error
        require(bool(values) and values == tuple(sorted(set(values))) and values[0] >= 0, "ASCENDING_DISTINCT_NONNEGATIVE_LAGS_REQUIRED")
        c, e, d = (torch.as_tensor(value, dtype=torch.float32).detach().clone() for value in (center, encoder, decoder))
        require(c.ndim == 1 and c.numel() > 0 and e.ndim == 2 and e.shape[0] == len(c)
            and 0 < e.shape[1] <= len(c) and d.shape == (len(values)*e.shape[1], len(c))
            and c.device == e.device == d.device and all(torch.isfinite(v).all() for v in (c,e,d)), "FINITE_MATCHED_C_CK_LKC_INITIAL_STATE_REQUIRED")
        self.lags, self.C, self.K, self.learn_E = values, len(c), e.shape[1], bool(learn_E)
        self.register_buffer("center", c)
        self.E = nn.Parameter(e, requires_grad=self.learn_E)
        self.D = nn.Parameter(d)

    def encode(self, x):
        require(isinstance(x, torch.Tensor) and x.dtype == torch.float32 and x.ndim == 3 and x.numel() > 0
            and x.shape[2] == self.C and x.device == self.E.device and torch.isfinite(x).all(), "FINITE_FP32_NTC_ENCODER_INPUT_REQUIRED")
        result = (x-self.center) @ self.E
        require(torch.isfinite(result).all(), "NONFINITE_LINEAR_ENCODING")
        return result

    def lag_features(self, zpast, zfuture, *, include_origin=False):
        require(isinstance(zpast, torch.Tensor) and isinstance(zfuture, torch.Tensor)
            and zpast.dtype == zfuture.dtype == torch.float32 and zpast.ndim == zfuture.ndim == 3
            and zpast.numel() > 0 and zfuture.numel() > 0 and zpast.shape[0] == zfuture.shape[0]
            and zpast.shape[2] == zfuture.shape[2] == self.K and zpast.shape[1] > self.lags[-1]
            and zpast.device == zfuture.device and torch.isfinite(zpast).all() and torch.isfinite(zfuture).all(), "MATCHED_FINITE_PAST_FUTURE_K_AND_LAG_BUFFER_REQUIRED")
        joined = torch.cat((zpast,zfuture), dim=1)
        L, H, origin = zpast.shape[1], zfuture.shape[1], int(bool(include_origin))
        pieces = [joined[:, L-lag-origin:L-lag+H]-zpast[:, -1-lag:-lag if lag else None] for lag in self.lags]
        result = torch.cat(pieces, dim=2)
        require(torch.isfinite(result).all(), "NONFINITE_CAUSAL_LAG_FEATURES")
        return result

    def decode(self, currentC, zpast, zfuture, *, include_origin=False):
        require(isinstance(currentC, torch.Tensor) and currentC.dtype == torch.float32 and currentC.shape == (len(zpast),self.C)
            and currentC.device == self.D.device == zpast.device and torch.isfinite(currentC).all(), "FINITE_FP32_CURRENT_C_AND_MATCHED_DEVICE_REQUIRED")
        result = currentC[:,None] + self.lag_features(zpast,zfuture,include_origin=include_origin) @ self.D
        require(torch.isfinite(result).all(), "NONFINITE_LINEAR_LAG_FORECAST")
        return result

    def forward(self, x, bolt, *, return_components=False):
        zpast = self.encode(x)
        zfuture = bolt(zpast)
        prediction = self.decode(x[:,-1], zpast, zfuture)
        return (prediction,zpast,zfuture) if return_components else prediction


def global_fit_macro_mse(prediction, target, observed, fit_counts, fit_windows):
    require(isinstance(prediction,torch.Tensor) and isinstance(target,torch.Tensor) and isinstance(observed,torch.Tensor)
        and prediction.dtype == target.dtype == torch.float32 and prediction.ndim == 3 and prediction.numel() > 0
        and prediction.shape == target.shape == observed.shape and observed.dtype == torch.bool
        and prediction.device == target.device == observed.device and torch.isfinite(prediction).all(), "MATCHED_FINITE_FP32_FORECAST_AND_MASK_REQUIRED")
    require(torch.isfinite(target[observed]).all(), "FINITE_OBSERVED_TARGET_REQUIRED")
    counts = torch.as_tensor(fit_counts,device=prediction.device)
    require(counts.dtype == torch.int64 and counts.shape == (prediction.shape[2],) and (counts >= 0).all()
        and type(fit_windows) is int and fit_windows > 0, "GLOBAL_FIT_INT64_COUNTS_AND_WINDOW_COUNT_REQUIRED")
    active = counts > 0
    require(active.any(), "AT_LEAST_ONE_ACTIVE_FIT_CHANNEL_REQUIRED")
    require(not observed[:,:,~active].any(), "OBSERVATIONS_IN_GLOBALLY_EMPTY_FIT_CHANNEL")
    safe_target = torch.where(observed,target,prediction.detach())
    error = torch.where(observed,prediction-safe_target,torch.zeros_like(prediction))
    per_channel_SSE = error.square().sum(dim=(0,1))
    loss = (fit_windows/len(prediction))*(per_channel_SSE[active]/counts[active].to(prediction.dtype)).mean()
    require(torch.isfinite(loss), "NONFINITE_GLOBAL_FIT_WEIGHTED_LOSS")
    return loss


def gradient_diagnostics(model, loss, zpast, zfuture):
    require(loss.ndim == 0 and loss.requires_grad and torch.isfinite(loss), "FINITE_SCALAR_DIFFERENTIABLE_LOSS_REQUIRED")
    def grad(output, variable, upstream=None):
        if not output.requires_grad or not variable.requires_grad: return None
        return torch.autograd.grad(output,variable,grad_outputs=upstream,retain_graph=True,create_graph=False,allow_unused=True)[0]
    total_z, future_g = grad(loss,zpast), grad(loss,zfuture)
    total_E = grad(loss,model.E)
    via_F = grad(zfuture,zpast,future_g.detach()) if future_g is not None else None
    available = via_F is not None and future_g is not None
    direct_z = (total_z if total_z is not None else torch.zeros_like(zpast))-via_F if available else None
    via_E = grad(zpast,model.E,via_F.detach()) if available else None
    direct_E = grad(zpast,model.E,direct_z.detach()) if available else None
    tensors = (total_z,future_g,total_E,via_F,direct_z,via_E,direct_E)
    require(all(torch.isfinite(value).all() for value in tensors if value is not None), "NONFINITE_GRADIENT_DIAGNOSTIC")
    def norm(value): return float(torch.linalg.vector_norm(value.detach().double()).cpu()) if value is not None else None
    residual = total_E-via_E-direct_E if all(v is not None for v in (total_E,via_E,direct_E)) else None
    reason = None
    if not zpast.requires_grad: reason = "ENCODED_PAST_NOT_REQUIRES_GRAD"
    elif not zfuture.requires_grad: reason = "F_FUTURE_NOT_REQUIRES_GRAD"
    elif future_g is None: reason = "LOSS_DISCONNECTED_FROM_F_FUTURE"
    elif via_F is None: reason = "F_FUTURE_DISCONNECTED_FROM_PAST"
    return {"finite":True,"E_trainable":model.E.requires_grad,"F_path_available":available,"F_path_unavailable_reason":reason,
        "F_future_gradient_norm":norm(future_g),"F_input_via_future_gradient_norm":norm(via_F),
        "past_total_gradient_norm":norm(total_z),"past_direct_gradient_norm":norm(direct_z),
        "E_total_gradient_norm":norm(total_E),"E_F_path_gradient_norm":norm(via_E),"E_direct_past_gradient_norm":norm(direct_E),
        "E_gradient_sum_residual_max_abs":float(residual.detach().abs().max().cpu()) if residual is not None else None,
        "parameter_grad_fields_populated_by_helper":False,"graph_retained_for_subsequent_backward":True,
        "direct_E_path_scope":"Encoded past lag baselines/known prefix; raw currentC anchor is independent of E",
        "nonzero_total_E_gradient_alone_proves_F_path":False}


def parameter_diagnostics(model, latent_history=None):
    e, d, center = (value.detach().double().cpu() for value in (model.E,model.D,model.center))
    require(all(torch.isfinite(value).all() for value in (e,d,center)), "NONFINITE_ADAPTER_PARAMETERS")
    e_singular, d_singular = torch.linalg.svdvals(e), torch.linalg.svdvals(d)
    variance = None
    if latent_history is not None:
        require(isinstance(latent_history,torch.Tensor) and latent_history.ndim == 3 and latent_history.numel() > 0
            and latent_history.shape[2] == model.K and torch.isfinite(latent_history).all(), "FINITE_NLK_PARAMETER_DIAGNOSTIC_LATENTS_REQUIRED")
        variance = latent_history.detach().double().cpu().var(dim=(0,1),unbiased=False).tolist()
    return {"finite":True,"E_l2_norm":float(torch.linalg.vector_norm(e)),"D_l2_norm":float(torch.linalg.vector_norm(d)),
        "center_l2_norm":float(torch.linalg.vector_norm(center)),"E_singularvalues":e_singular.tolist(),
        "D_singularvalues":d_singular.tolist(),"latent_variance":variance,
        "D_feature_to_C_operator_gain":float(d_singular[0]),"D_column_l2":torch.linalg.vector_norm(d,dim=0).tolist(),
        "gain_scope":"Euclidean lag-feature-vector to C operator; not a joint-horizon or future-error bound",
        "latent_variance_unbiased":False,"lags":list(model.lags),"E_trainable":model.E.requires_grad}
