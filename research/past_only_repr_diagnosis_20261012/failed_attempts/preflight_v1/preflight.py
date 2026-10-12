import json
import platform
import shutil
import subprocess
import time

import numpy as np
import psutil
from common import *


def main():
    access_guard()
    with Budget(1, False, 'asset_and_data_preflight') as budget:
        assets = []
        manifest = read(DIRECT / 'runs.json')
        for entry in manifest['runs']:
            if entry['job']['family'] not in ('TS2', 'GT'):
                continue
            result_path = ROOT / entry['result_path']
            r = read(result_path)
            if entry['job']['family'] == 'TS2':
                record = r['training']['encoder']
                path = ROOT / record['checkpoint']
                expected = record['checkpoint_sha256']
            else:
                record = r['training']['reconstruction']
                path = ROOT / record['checkpoint_path']
                expected = None
            actual = sha(path) if path.is_file() else None
            assets.append({'job': entry['job'], 'path': path, 'exists': path.is_file(),
                           'bytes': path.stat().st_size if path.is_file() else None,
                           'checkpoint_hash': actual, 'expected_hash': expected,
                           'hash_matches': actual == expected if expected else None,
                           'result_path': result_path, 'result_hash': sha(result_path),
                           'prior_training_scope': 'TRAIN_ALL; repeated VAL selected existing weights'})
        datasets = {}
        for ds in ('jena', 'robin'):
            new, frozen = load_data(ds), load_data(ds, frozen=True)
            observed = new['mask']
            recovery = frozen['x']*frozen['std']+frozen['mean']
            maxerr = float(np.max(np.abs(recovery[observed]-frozen['physical'][observed])))
            datasets[ds] = {k: new[k] for k in ('fit_bounds','tune_bounds','val_bounds','columns',
                                              'interval_seconds','source','source_hash','raw_recovery_note')}
            datasets[ds].update(shape=list(new['x'].shape), observed_mask_hash=ahash(observed),
                recovered_observed_all_finite=bool(np.isfinite(new['physical'][observed]).all()),
                missing_are_nan=bool(np.isnan(new['physical'][~observed]).all()),
                frozen_roundtrip_max_abs=maxerr,
                origins={k:{'count':len(v),'first':int(v[0]),'last':int(v[-1]),'hash':ahash(v),
                    'unique_target_timestamps':len(np.unique(v[:,None]+np.arange(H))),
                    'duplicated_target_rows':len(v)*H-len(np.unique(v[:,None]+np.arange(H)))}
                    for k,v in new['origins'].items()},
                new_fit_mean_hash=ahash(new['mean']),new_fit_std_hash=ahash(new['std']),
                frozen_train_mean_hash=ahash(frozen['mean']),frozen_train_std_hash=ahash(frozen['std']))
            budget.check()
        model = ROOT / '.cache/huggingface/models--amazon--chronos-bolt-small/snapshots/772f3d25d38aec6d914c8949dab4462e2d46f5d8'
        config = read(model/'config.json')
        weights = model/'model.safetensors'
        processes = []
        for p in psutil.process_iter(['pid','create_time','cmdline']):
            try:
                args = p.info['cmdline'] or []
                if any('professor_compressor_followup_20261012_run01' in a for a in args):
                    processes.append({'pid':p.pid,'created':p.info['create_time'],
                                      'source_arguments':[a for a in args if a.endswith('.py')]})
            except (psutil.NoSuchProcess,psutil.AccessDenied):
                pass
        pre = {'status':'COMPLETE_VERIFIED', 'captured_unix':time.time(),
               'HEAD':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
               'branch':subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
               'existing_changes':subprocess.check_output(['git','status','--short'],text=True).splitlines(),
               'platform':platform.platform(), 'python':sys.executable,
               'ram_available':psutil.virtual_memory().available, 'disk_free':shutil.disk_usage(ROOT).free,
               'gpu_inventory':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip(),
               'existing_processes_preserved':processes,'data':datasets,'assets':assets,
               'Bolt':{'directory':model,'weights_actual_target':weights.resolve(),'weights_exists':weights.is_file(),
                       'weights_bytes':weights.stat().st_size,'weights_sha':sha(weights),
                       'config_sha':sha(model/'config.json'),'chronos_config':config.get('chronos_config')},
               'protected_TEST_new_access':0,'old_files_overwritten':0}
        write(HERE/'ASSETS_AND_PREFLIGHT.json',pre)
        protocol = {'approved_instructions_sha':sha(HERE/'APPROVED_INSTRUCTIONS.md'),'plan_sha':sha(HERE/'PLAN.md'),
                    'HEAD':pre['HEAD'],'L':L,'H':H,'first_future_origin_rule':'history[o-512:o], target[o:o+48]',
                    'seeds':SEEDS,'lambdas':LAMBDAS,'selection':'TUNE physical original-channel macro MSE; grid-order tie',
                    'scale':'physical per-channel and within-dataset macro; normalized companion; no inter-dataset pool',
                    'new_fit':'TRAIN start+floor(.8*length); observed FIT stats; causal ffill',
                    'frozen':'original TRAIN_ALL preprocessing; historically exposed SWA diagnostic',
                    'gpu_limits_seconds':{'1':5400,'2':5400},'cpu_numeric_wall_seconds':3600,
                    'memory_limit_bytes':8*1024**3,'new_cache_conservative_limit_bytes':4*1024**3,
                    'E1_samples':{'representation':16,'cold_full_forecast':2,'fixed_uniform_selection':True},
                    'ridge_objective':'mean observed output SSE + lambda*coef_l2; intercept unpenalized; alpha=nobs*lambda',
                    'E3_scope':'pending COST_ONLY preflight; FULL or predeclared DIAG_SUBSET',
                    'E3_decoder':'d->128 ReLU->C, Adam1e-3,600update,0/100TUNE reconstruction selection',
                    'E6_objective_gate':'two-seed Jena TUNE direct-ridge MSE: REC vsFORECAST, tieREC; incompleteE5→FORECAST',
                    'E6_real_gate':'ALL_LAG fullMSE<Self and3of4chronologicalblocks; MAE-onlytrade named',
                    'scientific_status':'NO_SCIENCE_RESULTS_YET', 'TEST_new_access':0}
        write(HERE/'PROTOCOL.json', protocol)
        write(HERE/'STATUS.json', {f'E{i}':{'status':'READY' if i==1 else 'NOT_STARTED'} for i in range(1,7)})
        print(json.dumps({'status':'PREFLIGHT_PASS','assets':len(assets),'datasets':datasets,'resources':budget.record},default=jsonable))


if __name__ == '__main__':
    main()
