"""Guard-overhead repair verification using the six remaining disposable updates."""
import gc
import time

import numpy as np
import torch

from calibrate_readout import backward, feature_slice, full_val, step, sync
from model_readout import Readout, load_backbone
from runtime_readout import (Budget, CACHE, HERE, ReferenceBudget, configure,
                             digest, load_data, save_json)


def main():
    configure()
    result = {'schema': 'tsfm_readout_guard_repair_v1', 'scientific_fits': 0,
              'test_access': False, 'selection': False, 'timing_scope':
              'Limited repair diagnostic: one optimizer update and one full cached VAL per arm; not a replacement 20-update calibration block',
              'source': {name: digest(HERE / name) for name in ('runtime_s0_reference.py', 'runtime_readout.py', 'budget_guard.py')},
              'guard_check': {}, 'datasets': {}}
    for name, cls in (('original', ReferenceBudget), ('repaired', Budget)):
        with cls(stage='s0', category='cpu', label='guard_checks_'+name) as budget:
            times = []
            for _ in range(30):
                start = time.perf_counter()
                budget.check()
                times.append(time.perf_counter()-start)
            result['guard_check'][name] = {'checks': 30, 'seconds': times,
                'median_seconds': float(np.median(times)), 'maximum_seconds': max(times)}
    with Budget(stage='s0', category='gpu', label='guard_repair_integrity') as budget:
        for dataset in ('robin', 'peacock_education', 'jena'):
            print('GUARD REPAIR '+dataset, flush=True)
            data = load_data(dataset)
            with np.load(CACHE / (dataset+'_sample_features.npz'), allow_pickle=False) as archive:
                origins = archive['origins'].copy()
                features = {key: torch.from_numpy(archive[key].copy()) for key in ('h', 'loc', 'scale', 'point')}
            backbone = load_backbone(dataset)
            result['datasets'][dataset] = {}
            for arm in ('OUTPUT_AFFINE', 'NATIVE_HEAD'):
                model = Readout(backbone, data['_C'], arm).eval()
                saved = torch.load(CACHE / ('restore_'+dataset+'_'+arm+'.pt'), map_location='cpu', weights_only=False)
                model.load_trainable_state(saved['model'])
                frozen = model.frozen_hash()
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0.)
                optimizer.load_state_dict(saved['optimizer'])
                # Warm forward/backward without extra optimizer updates; loaded Adam moments already exist.
                for _ in range(10):
                    backward(model, data, origins[:4], features, np.arange(4), budget, 'cached')
                sync()
                start = time.perf_counter()
                step(model, optimizer, data, origins[:4], features, np.arange(4), budget, 'cached')
                sync()
                step_time = time.perf_counter()-start
                start = time.perf_counter()
                full_val(model, data, data['val_origins'], features, 24, budget, 'cached')
                sync()
                val_time = time.perf_counter()-start
                if model.frozen_hash() != frozen:
                    raise AssertionError('Guard repair changed frozen state')
                result['datasets'][dataset][arm] = {'one_step_seconds': step_time,
                    'one_full_cached_val_seconds': val_time, 'loaded_adam_state': True,
                    'additional_optimizer_updates': 1, 'frozen_state': 'PASS'}
                model.zero_grad(set_to_none=True)
                del optimizer, model, saved
            del backbone, data, features
            gc.collect()
            torch.cuda.empty_cache()
        result['budget'] = budget.snapshot()
    result['status'] = 'PASS'
    save_json(HERE / 'guard_repair.json', result)
    print('GUARD REPAIR PASS', flush=True)


if __name__ == '__main__':
    main()
