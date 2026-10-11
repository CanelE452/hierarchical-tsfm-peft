"""Paper-based GTConvAE reimplementation; native graph-time E and D are retained."""

from __future__ import annotations

from itertools import chain
import math

import torch
from torch import nn


FILTER_ORDER = 4
TIME_REDUCTION = 4
LATENT_FEATURES = 2


class ProductGraphShift(nn.Module):
    """Eq. (1), with a non-wrapping directed past-to-present temporal line."""

    def __init__(self, graph: torch.Tensor, no_spatial: bool = False):
        super().__init__()
        graph = torch.as_tensor(graph, dtype=torch.float32).clone().contiguous()
        if graph.ndim != 2 or graph.shape[0] != graph.shape[1]:
            raise ValueError("graph must be a square C x C matrix")
        if not torch.isfinite(graph).all():
            raise ValueError("graph contains non-finite entries")
        self.register_buffer("graph", graph)
        self.no_spatial = bool(no_spatial)
        self.s00 = nn.Parameter(torch.tensor(0.0))
        self.s10 = nn.Parameter(torch.tensor(0.5))
        if self.no_spatial:
            self.register_buffer("s01", torch.tensor(0.0))
            self.register_buffer("s11", torch.tensor(0.0))
        else:
            self.s01 = nn.Parameter(torch.tensor(0.5))
            self.s11 = nn.Parameter(torch.tensor(0.0))

    @property
    def coefficients(self) -> torch.Tensor:
        return torch.stack([self.s00, self.s01, self.s10, self.s11])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        past = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        out = self.s00 * x + self.s10 * past
        if not self.no_spatial:
            spatial = torch.einsum("btcf,dc->btdf", x, self.graph)
            past_spatial = torch.cat(
                [torch.zeros_like(spatial[:, :1]), spatial[:, :-1]], dim=1
            )
            out = out + self.s01 * spatial + self.s11 * past_spatial
        return out


class PolynomialGTFilter(nn.Module):
    """Eq. (2)-(3): order K has K+1 feature-bank taps, including identity."""

    def __init__(self, input_features: int, output_features: int, order: int = FILTER_ORDER):
        super().__init__()
        self.order = int(order)
        self.weight = nn.Parameter(
            torch.empty(self.order + 1, input_features, output_features)
        )
        nn.init.normal_(self.weight, std=math.sqrt(2.0 / ((self.order + 1) * input_features)))

    def forward(self, x: torch.Tensor, product: ProductGraphShift) -> torch.Tensor:
        current = x
        out = current @ self.weight[0]
        for degree in range(1, self.order + 1):
            current = product(current)
            out = out + current @ self.weight[degree]
        return out


def temporal_downsample(x: torch.Tensor, rate: int = 2) -> torch.Tensor:
    """Eq. (4), max summary of complete non-overlapping consecutive blocks."""
    if x.shape[1] % rate:
        raise ValueError("Temporal downsampling requires complete blocks; no target padding")
    blocks = x.reshape(x.shape[0], x.shape[1] // rate, rate, *x.shape[2:])
    return blocks.amax(dim=2)


def temporal_upsample(x: torch.Tensor, rate: int = 2) -> torch.Tensor:
    """Eq. (8): insert at one-based t=kr, hence zero-based positions r-1,2r-1,..."""
    out = x.new_zeros(x.shape[0], x.shape[1] * rate, *x.shape[2:])
    out[:, rate - 1::rate] = x
    return out


class GTConvAE(nn.Module):
    family = "GTCONVAE_PAPER_REIMPLEMENTATION_NATIVE_ED"
    time_reduction = TIME_REDUCTION
    latent_features = LATENT_FEATURES
    filter_order = FILTER_ORDER
    pooling = "max_nonoverlapping_right_edge_timestamp"
    decoder_upsampling_phase = "one_based_t_equals_kr"
    decoder_operation_order = "zero_insertion_then_GT_filter"
    reconstruction_zero_startup_steps = 3

    def __init__(self, graph: torch.Tensor, no_spatial: bool = False):
        super().__init__()
        self.product = ProductGraphShift(graph, no_spatial=no_spatial)
        self.channels = self.product.graph.shape[0]
        self.no_spatial = bool(no_spatial)
        self.encoder = nn.ModuleList([
            PolynomialGTFilter(1, 8), PolynomialGTFilter(8, 2)
        ])
        self.decoder = nn.ModuleList([
            PolynomialGTFilter(2, 8), PolynomialGTFilter(8, 1)
        ])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.channels:
            raise ValueError("GT encoder input must be [B,T,C]")
        if x.shape[1] % self.time_reduction:
            raise ValueError("GT context length must contain complete four-timestamp blocks")
        features = x.unsqueeze(-1)
        for layer in self.encoder:
            features = torch.relu(temporal_downsample(layer(features, self.product)))
        return features

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 4 or z.shape[-2:] != (self.channels, self.latent_features):
            raise ValueError("GT decoder latent must be [B,Tz,C,2]")
        features = z
        for index, layer in enumerate(self.decoder):
            features = layer(temporal_upsample(features), self.product)
            if index + 1 < len(self.decoder):
                features = torch.relu(features)
        return features.squeeze(-1)

    def decode_future(
        self, latent_history: torch.Tensor, latent_future: torch.Tensor,
        horizon: int | None = None,
    ) -> torch.Tensor:
        """Retain decoder state from observed latent history, then crop physical future."""
        decoded = self.decode(torch.cat([latent_history, latent_future], dim=1))
        start = latent_history.shape[1] * self.time_reduction
        available = latent_future.shape[1] * self.time_reduction
        horizon = available if horizon is None else int(horizon)
        if horizon < 1 or horizon > available:
            raise ValueError("Request enough F-predicted latent blocks for the physical horizon")
        return decoded[:, start:start + horizon]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def sparsity_penalty(self) -> torch.Tensor:
        return self.product.coefficients.abs().sum()

    def encoder_parameters(self):
        """Product coefficients are shared by E/D and therefore frozen with E in calibration."""
        return chain(self.encoder.parameters(), self.product.parameters())

    def decoder_parameters(self):
        """Native polynomial D filters only; this never includes a replacement linear readout."""
        return self.decoder.parameters()

    def parameter_report(self) -> dict:
        return {
            "encoder_filter_coefficients": sum(p.numel() for p in self.encoder.parameters()),
            "decoder_filter_coefficients": sum(p.numel() for p in self.decoder.parameters()),
            "product_coefficients_total": 4,
            "product_coefficients_trainable": sum(p.numel() for p in self.product.parameters()),
            "total_trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "taps_per_feature_pair": self.filter_order + 1,
            "bias_coefficients": 0,
            "product_shared_encoder_decoder": True,
            "no_spatial": self.no_spatial,
            "product_coefficients": self.product.coefficients.detach().cpu().tolist(),
            "pooling": self.pooling,
            "decoder_operation_order": self.decoder_operation_order,
            "final_output_activation": "linear_for_signed_standardized_data",
            "structural_zero_reconstruction_prefix_steps": self.reconstruction_zero_startup_steps,
            "forecast_boundary": "decode_concatenated_latent_history_and_predicted_future_then_crop",
        }

    @staticmethod
    def latent_forecast_steps(horizon: int) -> int:
        return math.ceil(int(horizon) / TIME_REDUCTION)

    @staticmethod
    def alignment_report(history_length: int = 512, horizon: int = 48) -> dict:
        if history_length % TIME_REDUCTION:
            raise ValueError("Encode only a declared context of complete four-timestamp blocks")
        ticks = []
        for tick in range(math.ceil(horizon / TIME_REDUCTION)):
            first = history_length + TIME_REDUCTION * tick
            ticks.append({
                "future_latent_tick_zero_based": tick,
                "physical_block_indices_zero_based": [first, first + TIME_REDUCTION - 1],
                "representative_timestamp_zero_based": first + TIME_REDUCTION - 1,
                "forecast_target_indices_zero_based": [TIME_REDUCTION * tick,
                    min(TIME_REDUCTION * (tick + 1), horizon) - 1],
            })
        return {
            "history_length": history_length, "history_latent_ticks": history_length // 4,
            "physical_horizon": horizon, "future_latent_ticks": math.ceil(horizon / 4),
            "encoder_tick_j_block": "[4*j,4*j+3]; representative timestamp 4*j+3",
            "decoder_zero_insert_phase": "right edge at each factor-two stage (Eq8 one-based t=kr)",
            "forecast_crop_start": history_length,
            "forecast_crop_stop_exclusive": history_length + horizon,
            "target_padding": False, "future_blocks": ticks,
        }
