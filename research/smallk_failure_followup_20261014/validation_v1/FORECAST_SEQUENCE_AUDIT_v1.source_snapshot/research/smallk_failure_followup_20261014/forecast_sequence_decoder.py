"""Source-only RF1/RF3 residual decoder comparison; no asset/model loader."""
import argparse
import json

import torch
from torch import nn


DATASETS = {
    'robin': {'C': 17, 'K': 4, 'static_width': 22},
    'beijing_aotizhongxin': {'C': 11, 'K': 2, 'static_width': 21},
}
SEEDS = (92601, 92602)
HEADS = ('STATIC_RF1', 'TEMPORAL_RF3')
TEMPORAL_WIDTH = 16
HORIZON = 48
UPDATES = 600
BATCH_SIZE = 4
CHECKPOINTS = (0, 100, 200, 400, 600)
LOCAL_FIT_SECONDS = 150
PROPOSED_COST_GPU_SECONDS = 180
PROPOSED_AUDIT_CPU_SECONDS = 300
PARITY_ATOL = 1e-5
PARITY_RTOL = 2e-6


def _check_latents(past_z, predicted_future_z, latent_channels):
    if past_z.ndim != 3 or predicted_future_z.ndim != 3:
        raise ValueError('LATENTS_MUST_BE_BATCH_TIME_K')
    if past_z.shape[0] != predicted_future_z.shape[0]:
        raise ValueError('LATENT_BATCH_MISMATCH')
    if past_z.shape[1] < 2 or predicted_future_z.shape[1] < 1:
        raise ValueError('TWO_PAST_POINTS_AND_NONEMPTY_FORECAST_REQUIRED')
    if past_z.shape[2] != latent_channels or predicted_future_z.shape[2] != latent_channels:
        raise ValueError('LATENT_K_MISMATCH')


class StaticResidualHead(nn.Module):
    """Pointwise control: R(z_future[h]) - R(z_last), receptive field one."""
    def __init__(self, latent_channels, output_channels, width):
        super().__init__()
        self.latent_channels = latent_channels
        self.hidden = nn.Linear(latent_channels, width)
        self.output = nn.Linear(width, output_channels)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def response(self, values):
        return self.output(torch.nn.functional.gelu(self.hidden(values)))

    def forward(self, past_z, predicted_future_z):
        _check_latents(past_z, predicted_future_z, self.latent_channels)
        repeated_last = past_z[:, -1:].expand(-1, predicted_future_z.shape[1], -1)
        return self.response(predicted_future_z) - self.response(repeated_last)


class TemporalResidualHead(nn.Module):
    """Causal RF3 over encoded past prefix plus predicted future latents."""
    def __init__(self, latent_channels, output_channels, width=TEMPORAL_WIDTH):
        super().__init__()
        self.latent_channels = latent_channels
        self.hidden = nn.Conv1d(latent_channels, width, kernel_size=3, padding=0)
        self.output = nn.Conv1d(width, output_channels, kernel_size=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def response(self, past_z, predicted_future_z):
        sequence = torch.cat((past_z[:, -2:], predicted_future_z), dim=1)
        hidden = torch.nn.functional.gelu(self.hidden(sequence.transpose(1, 2)))
        return self.output(hidden).transpose(1, 2)

    def forward(self, past_z, predicted_future_z):
        _check_latents(past_z, predicted_future_z, self.latent_channels)
        repeated_last = past_z[:, -1:].expand(-1, predicted_future_z.shape[1], -1)
        return self.response(past_z, predicted_future_z) - self.response(past_z, repeated_last)


def make_head(dataset, head, seed):
    if dataset not in DATASETS or head not in HEADS or seed not in SEEDS:
        raise ValueError('LOCKED_DATASET_HEAD_SEED_REQUIRED')
    spec = DATASETS[dataset]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        if head == 'STATIC_RF1':
            return StaticResidualHead(spec['K'], spec['C'], spec['static_width'])
        return TemporalResidualHead(spec['K'], spec['C'])


def forecast_sequence_trace(codec, frozen_forecaster, head, history):
    """Only historical raw input enters E; targets never enter this path."""
    past_z = codec.encode(history)
    predicted_future_z = frozen_forecaster(past_z)
    original_prediction = (history[:, -1:] + codec.decode(predicted_future_z)
                           - codec.decode(past_z[:, -1:]))
    residual = head(past_z, predicted_future_z)
    return original_prediction + residual, past_z, predicted_future_z, residual


def capacity_summary(dataset):
    spec = DATASETS[dataset]
    k, c, width = spec['K'], spec['C'], spec['static_width']
    static = (k + 1) * width + (width + 1) * c
    temporal = (3 * k + 1) * TEMPORAL_WIDTH + (TEMPORAL_WIDTH + 1) * c
    return {
        'registered': {'STATIC_RF1': static, 'TEMPORAL_RF3': temporal},
        'structurally_cancelled_output_bias': c,
        'potentially_active': {'STATIC_RF1': static - c, 'TEMPORAL_RF3': temporal - c},
        'registered_symmetric_relative_difference': abs(static - temporal) / ((static + temporal) / 2),
        'active_symmetric_relative_difference': abs(static - temporal) / ((static + temporal) / 2 - c),
        'matching_scope': 'new-head registered parameters; symmetric relative difference <2%; not exact',
        'zero_initialization': 'all output weights/biases zero; hidden gradients may be zero at first update',
    }


def conditions():
    return [{'dataset': dataset, 'seed': seed, 'head': head, **spec}
            for dataset, spec in DATASETS.items() for seed in SEEDS for head in HEADS]


def describe():
    return {
        'status': 'SOURCE_ONLY_NOT_EXECUTED', 'actual_data_opened': False,
        'GPU_executed': False, 'actual_training_executed': False,
        'conditions': conditions(), 'capacity': {name: capacity_summary(name) for name in DATASETS},
        'training_proposal': {'new_updates': UPDATES, 'batch_size': BATCH_SIZE,
                              'checkpoints': list(CHECKPOINTS), 'optimizer': 'AdamW',
                              'lr': .001, 'weight_decay': 0., 'clip_norm': 1.,
                              'loss': 'normalized observed channel-macro actual-Y MSE',
                              'selection': 'actual-Y TUNE MSE, earliest exact tie, includes update0'},
        'execution_gate': 'resource extension required if current wall/caps exhausted; no runner/auto-execution',
        'proposed_process_caps': {'fits_gpu_process_seconds': len(conditions()) * LOCAL_FIT_SECONDS,
                                  'each_fit_seconds': LOCAL_FIT_SECONDS,
                                  'cost_gpu_process_seconds': PROPOSED_COST_GPU_SECONDS,
                                  'audit_cpu_process_seconds': PROPOSED_AUDIT_CPU_SECONDS},
        'forward_inputs': ['past raw history to existing causal E', 'encoded past last two',
                           'frozen F predicted future latents'],
        'extra_raw_C_side_information': False, 'actual_future_inputs': False,
        'existing_current_C_anchor': True, 'original_sources_modified': False,
        'parity': {'atol': PARITY_ATOL, 'rtol': PARITY_RTOL,
                   'zero_head': 'adds exact zero to original MAIN operation order',
                   'constant_future': 'head residual exact zero; old anchor sum/subtraction retains FP rounding'},
        'hypothesis_status': 'RF3 restoration hypothesis; not an established real-data failure cause',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(describe(), indent=2, ensure_ascii=False))
