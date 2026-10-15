"""FIT-past PCA/ForeCA-principle projections and one shared masked affine decoder."""
from dataclasses import dataclass
import hashlib
import json

import numpy as np


LENGTH = 512
EIGENVALUE_RELATIVE_CUT = 1e-8
SPECTRAL_SMOOTHING_BINS = 7
MAX_ITERATIONS = 200


def _array_sha(value):
    array = np.ascontiguousarray(value)
    record = hashlib.sha256()
    record.update(str(array.dtype).encode())
    record.update(str(array.shape).encode())
    record.update(array.tobytes())
    return record.hexdigest()


def one_sided_cross_spectrum(history):
    """Return real cross-spectra; frequency bins keep DC/Nyquist single weight."""
    x = np.ascontiguousarray(history, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] != LENGTH or not len(x) or not x.shape[2] or not np.isfinite(x).all():
        raise ValueError("FINITE_NONEMPTY_N512C_SPECTRAL_HISTORY_REQUIRED")
    hann = np.hanning(LENGTH)
    transformed = np.fft.rfft(x * hann[None, :, None], axis=1)
    spectrum = np.einsum("nfi,nfj->fij", transformed.conj(), transformed, optimize=True).real
    weights = np.full(LENGTH // 2 + 1, 2., dtype=np.float64)
    weights[[0, -1]] = 1.
    spectrum *= weights[:, None, None] / (len(x) * LENGTH * np.sum(hann * hann))
    return (spectrum + spectrum.transpose(0, 2, 1)) * .5


def smooth_cross_spectrum(spectrum):
    value = np.asarray(spectrum, dtype=np.float64)
    if value.ndim != 3 or value.shape[0] != LENGTH // 2 + 1 or value.shape[1] != value.shape[2] or not np.isfinite(value).all():
        raise ValueError("FINITE_257_SQUARE_CROSS_SPECTRUM_REQUIRED")
    padded = np.pad(value, ((3, 3), (0, 0), (0, 0)), mode="reflect")
    return sum(padded[offset:offset + len(value)] for offset in range(SPECTRAL_SMOOTHING_BINS)) / SPECTRAL_SMOOTHING_BINS


def spectral_entropy(spectrum, direction, *, gradient=False):
    w = np.asarray(direction, dtype=np.float64)
    products = np.einsum("fij,j->fi", spectrum, w, optimize=True)
    power = products @ w
    scale = max(float(np.max(np.abs(power))), np.finfo(np.float64).tiny)
    if not np.isfinite(power).all() or np.min(power) < -1e-12 * scale:
        raise ValueError("NONFINITE_OR_NON_PSD_DIRECTIONAL_SPECTRUM")
    power = np.maximum(power, 0.)
    total = float(power.sum())
    if total <= np.finfo(np.float64).tiny:
        raise ValueError("COLLAPSED_DIRECTIONAL_SPECTRUM")
    probability = power / total
    positive = probability > 0
    natural_entropy = -float(np.sum(probability[positive] * np.log(probability[positive])))
    entropy = natural_entropy / np.log(len(probability))
    if not gradient:
        return entropy
    logs = np.log(np.maximum(probability, np.finfo(np.float64).tiny))
    coefficients = -(logs + natural_entropy) / (total * np.log(len(probability)))
    derivative = 2. * np.einsum("f,fi->i", coefficients, products, optimize=True)
    return entropy, derivative


def _canonical_sign(direction):
    return -direction if direction[np.argmax(np.abs(direction))] < 0 else direction


def foreca_directions(spectrum, k):
    """Deterministic two-start projected-gradient entropy search, not official reproduction."""
    rank = spectrum.shape[1]
    directions = []
    records = []
    for component in range(k):
        previous = np.stack(directions, axis=1) if directions else np.empty((rank, 0))
        complement = np.eye(rank) - previous @ previous.T
        canonical = complement[:, int(np.argmax(np.sum(complement * complement, axis=0)))]
        random = complement @ np.random.default_rng(20261014 + component).normal(size=rank)
        candidates = []
        for start_name, initial in (("canonical_projected_unit_vector", canonical), ("fixed_rng_projected_direction", random)):
            norm = float(np.linalg.norm(initial))
            if norm <= 1e-12:
                raise ValueError("DEFLATED_INITIAL_DIRECTION_COLLAPSED")
            w = initial / norm
            initial_entropy = spectral_entropy(spectrum, w)
            entropy = initial_entropy
            iterations = 0
            accepted = 0
            for iteration in range(MAX_ITERATIONS):
                value, derivative = spectral_entropy(spectrum, w, gradient=True)
                tangent = complement @ derivative
                tangent -= w * float(w @ tangent)
                norm = float(np.linalg.norm(tangent))
                if not np.isfinite(norm):
                    raise ValueError("NONFINITE_SPECTRAL_GRADIENT")
                if norm <= 1e-10:
                    break
                iterations = iteration + 1
                step = min(1., 1. / norm)
                found = False
                for _ in range(30):
                    proposal = complement @ (w - step * tangent)
                    proposal_norm = float(np.linalg.norm(proposal))
                    if proposal_norm <= 1e-12:
                        step *= .5
                        continue
                    proposal /= proposal_norm
                    proposal_entropy = spectral_entropy(spectrum, proposal)
                    if proposal_entropy < value - 1e-12:
                        w, entropy = proposal, proposal_entropy
                        accepted += 1
                        found = True
                        break
                    step *= .5
                if not found:
                    break
            candidates.append((entropy, w, {"initialization": start_name,
                               "initial_entropy": initial_entropy, "final_entropy": entropy,
                               "iterations": iterations, "accepted_steps": accepted}))
        selected = min(range(2), key=lambda index: candidates[index][0])
        direction = _canonical_sign(candidates[selected][1])
        directions.append(direction)
        records.append({"component": component, "selected_start": selected,
                        "starts": [candidate[2] for candidate in candidates]})
    result = np.stack(directions, axis=1)
    if not np.allclose(result.T @ result, np.eye(k), atol=1e-9, rtol=0):
        raise ValueError("LATENT_DIRECTIONS_NOT_ORTHONORMAL")
    return result, records


@dataclass(frozen=True)
class ProjectionState:
    center: np.ndarray
    encoder: np.ndarray
    decoder: np.ndarray
    decoder_bias: np.ndarray
    fit_metadata: dict

    def metadata(self):
        return {**self.fit_metadata, "state_array_sha256": {
            name: _array_sha(getattr(self, name))
            for name in ("center", "encoder", "decoder", "decoder_bias")}}

    def encode(self, history):
        x = np.ascontiguousarray(history)
        if x.dtype != np.float32 or x.ndim != 3 or x.shape[2] != len(self.center) or not np.isfinite(x).all():
            raise ValueError("FINITE_FP32_NTC_ENCODER_INPUT_REQUIRED")
        return np.ascontiguousarray((x - self.center.astype(np.float32)) @ self.encoder.astype(np.float32))

    def decode(self, latents):
        z = np.ascontiguousarray(latents)
        if z.dtype != np.float32 or z.ndim != 3 or z.shape[2] != self.encoder.shape[1] or not np.isfinite(z).all():
            raise ValueError("FINITE_FP32_NTK_DECODER_INPUT_REQUIRED")
        return np.ascontiguousarray(z @ self.decoder.astype(np.float32) + self.decoder_bias.astype(np.float32))

    def forecast(self, raw_last, latent_future, latent_last):
        last = np.asarray(raw_last)
        if last.ndim == 2:
            last = last[:, None, :]
        future = np.asarray(latent_future)
        zlast = np.asarray(latent_last)
        if zlast.ndim == 2:
            zlast = zlast[:, None, :]
        if (last.dtype != np.float32 or last.shape != (len(future), 1, len(self.center))
                or future.dtype != np.float32 or future.ndim != 3 or future.shape[2] != self.encoder.shape[1]
                or zlast.dtype != np.float32 or zlast.shape != (len(future), 1, self.encoder.shape[1])
                or not all(np.isfinite(value).all() for value in (last, future, zlast))):
            raise ValueError("FP32_CURRENT_C_AND_LAST_K_ANCHOR_SHAPES_REQUIRED")
        return np.ascontiguousarray(last + (future - zlast) @ self.decoder.astype(np.float32))

    def to_npz(self):
        return {**{name: np.ascontiguousarray(getattr(self, name), dtype=np.float64)
                   for name in ("center", "encoder", "decoder", "decoder_bias")},
                "metadata_json": np.array(json.dumps(self.fit_metadata, sort_keys=True, allow_nan=False))}

    @classmethod
    def from_npz(cls, arrays):
        values = {name: np.ascontiguousarray(arrays[name], dtype=np.float64)
                  for name in ("center", "encoder", "decoder", "decoder_bias")}
        center, encoder, decoder, bias = (values[name] for name in ("center", "encoder", "decoder", "decoder_bias"))
        if (center.ndim != 1 or encoder.ndim != 2 or encoder.shape[0] != len(center)
                or decoder.shape != (encoder.shape[1], len(center)) or bias.shape != center.shape
                or not all(np.isfinite(value).all() for value in values.values())):
            raise ValueError("SERIALIZED_PROJECTION_ARRAY_SHAPE_OR_FINITE_ERROR")
        metadata = json.loads(str(np.asarray(arrays["metadata_json"]).item()))
        if metadata["channels"] != len(center) or metadata["K"] != encoder.shape[1]:
            raise ValueError("SERIALIZED_PROJECTION_METADATA_SHAPE_ERROR")
        return cls(**values, fit_metadata=metadata)


def fit_projection(route, history, history_observed, k):
    route = str(route).upper()
    x = np.ascontiguousarray(history)
    mask = np.ascontiguousarray(history_observed)
    if route not in ("PCA", "FORECA"):
        raise ValueError("PROJECTION_ROUTE_MUST_BE_PCA_OR_FORECA")
    if (x.dtype != np.float32 or x.ndim != 3 or x.shape[1] != LENGTH or not len(x)
            or not x.shape[2] or not np.isfinite(x).all()):
        raise ValueError("FINITE_NONEMPTY_FP32_N512C_FIT_PAST_REQUIRED")
    if mask.dtype != np.bool_ or mask.shape != x.shape:
        raise ValueError("BOOL_OBSERVED_PAST_MASK_REQUIRED")
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or not 1 <= k <= x.shape[2]:
        raise ValueError("VALID_INTEGER_K_REQUIRED")
    flat = x.astype(np.float64).reshape(-1, x.shape[2])
    center = flat.mean(axis=0)
    centered = flat - center
    covariance = centered.T @ centered / len(flat)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    maximum = float(eigenvalues[-1])
    if maximum <= 0 or not np.isfinite(eigenvalues).all():
        raise ValueError("ZERO_OR_NONFINITE_FIT_VARIANCE")
    selected = np.flatnonzero(eigenvalues > maximum * EIGENVALUE_RELATIVE_CUT)[::-1]
    rank = len(selected)
    if rank < k:
        raise ValueError(f"WHITENING_RANK_BELOW_K:{rank}<{k}")
    vectors = eigenvectors[:, selected].copy()
    for column in range(rank):
        vectors[:, column] = _canonical_sign(vectors[:, column])
    whitening = vectors / np.sqrt(eigenvalues[selected])[None, :]
    search = []
    if route == "PCA":
        directions = np.eye(rank, k)
    else:
        white = (centered @ whitening).reshape(x.shape[0], LENGTH, rank)
        spectrum = smooth_cross_spectrum(one_sided_cross_spectrum(white))
        directions, search = foreca_directions(spectrum, k)
    encoder = whitening @ directions
    latent = centered @ encoder
    latent_covariance = latent.T @ latent / len(latent)
    if not np.allclose(latent_covariance, np.eye(k), atol=1e-7, rtol=0):
        raise ValueError("WHITENED_LATENT_VARIANCE_OR_ORTHOGONALITY_ERROR")
    design = np.column_stack((latent, np.ones(len(latent), dtype=np.float64)))
    flat_mask = mask.reshape(flat.shape)
    decoder = np.empty((k, x.shape[2]), dtype=np.float64)
    bias = np.empty(x.shape[2], dtype=np.float64)
    counts, ranks, conditions = [], [], []
    for channel in range(x.shape[2]):
        observed = flat_mask[:, channel]
        count = int(observed.sum())
        if not count:
            raise ValueError(f"DECODER_CHANNEL_WITHOUT_OBSERVED_FIT_PAST:{channel}")
        coefficient, _, design_rank, singular = np.linalg.lstsq(design[observed], flat[observed, channel], rcond=None)
        decoder[:, channel], bias[channel] = coefficient[:-1], coefficient[-1]
        counts.append(count)
        ranks.append(int(design_rank))
        conditions.append(float(singular[0] / singular[-1]) if singular[-1] > 0 else None)
    if not np.isfinite(decoder).all() or not np.isfinite(bias).all():
        raise ValueError("NONFINITE_MASKED_DECODER_SOLUTION")
    metadata = {"route": route, "channels": x.shape[2], "K": int(k), "length": LENGTH,
                "fit_windows": len(x), "fit_positions_with_overlap": len(flat), "whitening_rank": rank,
                "eigenvalue_relative_cut": EIGENVALUE_RELATIVE_CUT,
                "retained_whitening_eigenvalues": eigenvalues[selected].tolist(),
                "latent_fit_covariance": latent_covariance.tolist(),
                "history_sha256": _array_sha(x), "history_observed_sha256": _array_sha(mask),
                "center_covariance_scope": "All causal-forward-filled FIT past positions, including filled values; repeated overlapping positions retained",
                "decoder_target_scope": "Only true observed FIT past targets per original channel; unobserved filled targets excluded",
                "decoder_objective": "Unregularized per-channel masked affine least squares; NumPy lstsq rcond=None",
                "decoder_observed_counts": counts, "decoder_design_ranks": ranks,
                "decoder_design_conditions": conditions,
                "encoder_fit_dtype": "float64", "serialized_state_dtype": "float64", "deployment_dtype": "float32",
                "forecast_formula": "raw_current_C + (predicted_future_K - last_observed_K) @ decoder_FP32; decoder bias reconstruction only",
                "strict_K_only_compression": False, "actual_future_target_used_in_projection_or_decoder_fit": False,
                "foreca_variant": "Spectral normalized-entropy projected-gradient deflation; principle adaptation, not official reproduction",
                "spectral_contract": {"window": "NumPy symmetric Hann512", "fft": "rfft512",
                                      "bins": 257, "DC_weight": 1, "Nyquist_weight": 1, "interior_weight": 2,
                                      "averaging": "Mean cross-spectrum across FIT windows",
                                      "smoothing": "7-bin arithmetic mean with reflect boundary padding",
                                      "normalization": "Directional power divided by directional total; entropy divided by log257"},
                "foreca_max_iterations_per_start": MAX_ITERATIONS, "foreca_search": search}
    return ProjectionState(center, encoder, decoder, bias, metadata)
