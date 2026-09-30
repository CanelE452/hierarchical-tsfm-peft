"""Prospective confirmation contract sharing the original v12 resource ledger."""
from pathlib import Path
import os
import numpy as np
import torch

from runtime_v12 import (HERE, ROOT, CACHE, SEEDS, LIMITS, Job, read_json, save_json,
                        digest, artifact, import_file, budget_snapshot, require_storage,
                        environment_receipt, confirmation_protocol, verify_provenance)

OLD = ROOT / 'research/tsfm_peft_internal_vs_subspace_v11_20260930'
_numeric = import_file(OLD / 'runtime_v11.py', 'v12_reused_numeric_runtime')
tensors = _numeric.tensors
batches = _numeric.batches
phase_sample = _numeric.phase_sample
masked_macro_loss = _numeric.masked_macro_loss
score_arrays = _numeric.score_arrays


def protocol():
    verify_provenance()
    return confirmation_protocol()


def configure():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


def load_data(dataset, include_test=False):
    p = protocol()
    unit = p['units'][dataset]
    contract_path = HERE / f'data_contract_confirmation_{dataset}.json'
    contract = read_json(contract_path)
    manifest = read_json(HERE / 'confirmation_data_manifest.json')
    if manifest[dataset]['contract_sha256'] != digest(contract_path):
        raise RuntimeError('Confirmation data contract changed')
    if include_test:
        seal_path = HERE / 'confirmation_selection_seal.json'
        exposure = read_json(HERE / 'confirmation_test_exposure.json')
        seal = read_json(seal_path)
        if exposure['seal_sha256'] != digest(seal_path):
            raise RuntimeError('Joint TEST exposure seal mismatch')
        for name in ('selected_confirmation.json', 'confirmation_protocol.json',
                     'confirmation_data_manifest.json'):
            if seal['files'][name] != digest(HERE / name):
                raise RuntimeError('Joint selection seal changed: ' + name)
        if set(seal['datasets']) != set(p['units']):
            raise RuntimeError('All units must be selected before TEST')
    entry = contract['test' if include_test else 'trainval']
    path = Path(entry['path'])
    if digest(path) != entry['sha256']:
        raise RuntimeError('Confirmation array bytes changed')
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    c, k = unit['C'], unit['K']
    if data['x'].shape[1] != c or data['basis'].shape != (c, k):
        raise ValueError('Confirmation dimensions changed')
    if data['finite'].shape != data['x'].shape or data['columns'].tolist() != unit['columns']:
        raise ValueError('Mask or channel order mismatch')
    data.update(_path=path, _dataset=dataset, _phase_period=unit['phase_period'], _C=c, _K=k)
    return data


def sources_for(dataset, seed):
    parents = read_json(HERE / 'parents_selected.json')['selected'][dataset]['level']
    index = list(SEEDS).index(int(seed))
    path, sha = parents['checkpoints'][index], parents['checkpoint_sha256'][index]
    artifact(path, sha)
    return {'checkpoint': path, 'sha256': sha, 'level_checkpoint': path,
            'level_checkpoint_sha256': sha}
