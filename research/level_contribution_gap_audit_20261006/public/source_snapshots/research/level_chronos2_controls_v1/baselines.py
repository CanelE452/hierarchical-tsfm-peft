"""Restore the selected presentation baselines without fitting or reselection."""
from pathlib import Path
import sys

import numpy as np
import torch

from runtime import HERE, artifact, import_file, read_json


def load_data(dataset, include_test=False):
    unit = read_json(HERE / 'input_contract.json')['units'][dataset]
    entry = unit['test' if include_test else 'trainval']
    artifact(entry['path'], entry['sha256'])
    with np.load(entry['path'], allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if data['x'].shape[1] != unit['c'] or data['basis'].shape != (unit['c'], unit['k']):
        raise ValueError('Original-channel / latent dimensions changed')
    if data['columns'].tolist() != unit['columns'] or data['finite'].shape != data['x'].shape:
        raise ValueError('Column order / observation mask changed')
    data.update(_path=entry['path'], _C=unit['c'], _K=unit['k'], _dataset=dataset)
    return data


def inputs(data, origins):
    return torch.from_numpy(np.stack([data['x'][int(o)-512:int(o)] for o in origins]).astype(np.float32))


def targets(data, origins):
    return (np.stack([data['x'][int(o):int(o)+48] for o in origins]),
            np.stack([data['finite'][int(o):int(o)+48] for o in origins]))


def baseline_rows(dataset):
    return read_json(HERE / 'reuse_manifest.json')['units'][dataset]['rows']


def source_module(row):
    path = Path(row['restore_module'])
    artifact(path, row['restore_module_artifact']['sha256'])
    if path.name == 'model_confirmation_v12.py':
        import_file(path.parent / 'runtime_v12.py', 'runtime_v12')
        import_file(path.parent / 'runtime_confirmation_v12.py', 'runtime_confirmation_v12')
    return import_file(path, 'c2_baseline_' + path.parent.name + '_' + path.stem)


class BaselineForecast:
    def __init__(self, module, row, device):
        self.module = module.eval().requires_grad_(False)
        self.row, self.device, self.chunk_rows = row, device, None

    def set_chunk(self, chunk_rows):
        self.chunk_rows = chunk_rows
        if chunk_rows is not None and chunk_rows <= 0:
            raise ValueError('Chunk size must be positive')

    @torch.inference_mode()
    def __call__(self, cpu_x):
        if cpu_x.device.type != 'cpu' or cpu_x.dtype != torch.float32:
            raise ValueError('Deployment starts with standardized CPU FP32 inputs')
        x = cpu_x.to(self.device)
        if self.row['method'] in ('F0', 'MSE_LORA'):
            b, l, c = x.shape
            rows = x.transpose(1, 2).reshape(b*c, l)
            chunks = (rows,) if self.chunk_rows is None else rows.split(self.chunk_rows)
            values = []
            for chunk in chunks:
                output = self.module.backbone(context=chunk)
                values.append(output.quantile_preds[:, self.module.median, :48].clone(memory_format=torch.contiguous_format))
                del output
            value = torch.cat(values) if len(values) > 1 else values[0]
            value = value.reshape(b, c, 48).transpose(1, 2)
        elif self.row.get('output_component') == 'main':
            value = self.module.components(x)[0]
        else:
            value = self.module(x)
        return value.contiguous().cpu()

    def receipt(self):
        params, buffers = list(self.module.parameters()), list(self.module.buffers())
        return {'dataset': self.row['dataset'], 'method': self.row['method'], 'id': self.row['id'],
                'seed': self.row['seed'], 'checkpoint': self.row.get('checkpoint'),
                'parameter_count': sum(p.numel() for p in params),
                'inference_trainable_parameter_count': sum(p.numel() for p in params if p.requires_grad),
                'parameter_bytes': sum(p.numel()*p.element_size() for p in params),
                'buffer_bytes': sum(p.numel()*p.element_size() for p in buffers),
                'original_fitted_parameter_count': 24624 if self.row['method'] == 'DIRECT_NLINEAR' else
                    17920 if self.row['method'] == 'LEVEL' else 294912 if self.row['method'] == 'MSE_LORA' else 0,
                'dtype': sorted({str(p.dtype) for p in params}), 'eval': not self.module.training,
                'output_component': self.row.get('output_component'), 'row_chunk': self.chunk_rows}


def load_baseline(row, device='cuda', deploy=True):
    source = source_module(row)
    checkpoint = row.get('checkpoint')
    if checkpoint:
        artifact(checkpoint['path'], checkpoint['sha256'])
        model = getattr(source, row['restore_function'])(checkpoint['path'], device=device)
    else:
        model = getattr(source, row['build_function'])(**row['build_kwargs'], device=device)
    if deploy and row.get('deploy_function'):
        model = getattr(source, row['deploy_function'])(model)
    return BaselineForecast(model, row, device)
