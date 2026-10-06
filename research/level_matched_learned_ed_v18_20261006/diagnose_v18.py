"""Descriptive selected E/D diagnostics; no fitting, FM forward or selection use."""
import json
import time

import numpy as np
import torch

from runtime_v18 import (DATASETS, HERE, Job, artifact, array_hash, check_selection_seal,
                         dataset_contract, load_data, read_json, resolve, save_json, score_arrays)


def spectrum(weight):
    singular = torch.linalg.svdvals(weight.double()).numpy()
    tolerance = max(weight.shape) * np.finfo(np.float32).eps * singular[0]
    return {'singular_values': singular.tolist(), 'numerical_rank': int(np.count_nonzero(singular>tolerance)),
            'rank_tolerance': float(tolerance), 'rank_rule': 'max(shape) * FP32 eps * largest singular value'}


def main():
    check_selection_seal()
    destination = HERE / 'mechanism_diagnostics.json'
    if destination.exists():
        raise FileExistsError('Diagnostics already completed; preserve and verify')
    torch.set_num_threads(4)
    selection = read_json(HERE / 'selection.json')
    records = {'schema': 'matched_learned_ed_mechanism_v18', 'created_utc': time.time(),
               'scope': 'Descriptive only, after sealed selection; no new fitting or backbone forward',
               'selection_use': False, 'selection_seal': artifact(HERE / 'selection_seal.json'), 'units': {}}
    with Job('cpu', 'Selected E/D mechanism diagnostics') as job:
        for dataset in DATASETS:
            data = load_data(dataset, include_test=False)
            contract = dataset_contract(dataset)
            unit = selection['units'][dataset]
            basis = torch.from_numpy(data['basis']).float()
            records['units'][dataset] = []
            for index, checkpoint in enumerate(unit['checkpoints']):
                artifact(checkpoint['path'], checkpoint['sha256'])
                saved = torch.load(resolve(checkpoint['path']), map_location='cpu', weights_only=False)
                encoder = saved['state']['encoder.weight'].float()
                decoder = saved['state']['decoder.weight'].float()
                if not torch.equal(saved['basis'], basis):
                    raise RuntimeError('Selected checkpoint PCA changed')
                if not torch.isfinite(encoder).all() or not torch.isfinite(decoder).all():
                    raise ValueError('Nonfinite selected adapter')
                row = {'seed': unit['seeds'][index], 'id': unit['run_ids'][index], 'checkpoint': checkpoint,
                       'encoder_relative_frobenius_from_pca': float(torch.linalg.norm(encoder-basis.T)/torch.linalg.norm(basis.T)),
                       'decoder_relative_frobenius_from_pca': float(torch.linalg.norm(decoder-basis)/torch.linalg.norm(basis)),
                       'decoder_relative_frobenius_from_encoder_transpose': float(torch.linalg.norm(decoder-encoder.T)/torch.linalg.norm(decoder)),
                       'encoder': spectrum(encoder), 'decoder': spectrum(decoder), 'periods': {}}
                for period in ('train', 'val'):
                    start, end = contract['bounds'][period]
                    x = torch.from_numpy(data['x'][start:end]).float()
                    mask = data['finite'][start:end].astype(bool)
                    with torch.no_grad():
                        fixed = x @ basis
                        latent = x @ encoder.T
                        reconstructed = latent @ decoder.T
                    observed = score_arrays(reconstructed.numpy()[:,None,:], x.numpy()[:,None,:], mask[:,None,:])
                    row['periods'][period] = {'bounds': [start,end], 'rows': len(x),
                         'input_values_sha256': array_hash(x.numpy()), 'observation_mask_sha256': array_hash(mask),
                         'reconstruction_mse_all_imputed_coordinates': float((reconstructed.double()-x.double()).square().mean()),
                         'reconstruction_mse_observed_channel_macro': observed['mse'],
                         'fixed_latent_mean': fixed.double().mean(0).tolist(),
                         'fixed_latent_std_population': fixed.double().std(0,unbiased=False).tolist(),
                         'learned_latent_mean': latent.double().mean(0).tolist(),
                         'learned_latent_std_population': latent.double().std(0,unbiased=False).tolist()}
                result = read_json(HERE / 'run_results' / (unit['run_ids'][index]+'.json'))
                initial, selected = result['initial_val_mse'], result['best_val_mse']
                row['full_val_forecast_improvement_from_step0_compress'] = {
                    'step0_mse': initial, 'selected_mse': selected, 'difference': selected-initial,
                    'relative_percent': 100*(selected/initial-1), 'selected_epoch': result['selected_epoch'],
                    'source_result': artifact(HERE / 'run_results' / (unit['run_ids'][index]+'.json'))}
                records['units'][dataset].append(row)
            job.heartbeat(dataset)
        save_json(destination, records)
    print(json.dumps({'status': 'complete', 'diagnostics': list(DATASETS), 'new_fits': 0, 'backbone_forward': 0}))


if __name__=='__main__':
    main()
