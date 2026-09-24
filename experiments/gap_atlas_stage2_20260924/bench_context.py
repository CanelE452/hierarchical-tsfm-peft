"""How long does one optimizer step take at each fit context? Needed to plan within the contract's
ten-hour limit (section 8). Steps here are recorded in the ledger as benchmark entries."""
import time, torch, numpy as np
import ga, arms
from periods import Period

STEPS = 12
for cid in ('C4', 'C1'):
    cfg = ga.BY_ID[cid]
    per = Period(cfg, 'P2')
    series = [{'target': np.asarray(e['target'], dtype=np.float32)} for e in per.training_series()]
    for ctx in (512, 2048, 8192):
        pipe = ga.pipeline(dtype='float32')
        torch.cuda.reset_peak_memory_stats(); torch.manual_seed(0)
        t0 = time.perf_counter()
        try:
            pipe.fit(series, prediction_length=per.H, finetune_mode='lora', lora_config=None,
                     context_length=ctx, learning_rate=1e-5, num_steps=STEPS, batch_size=32,
                     validation_inputs=None, output_dir=str(ga.SCRATCH/f'bench_{cid}_{ctx}'),
                     save_strategy='no', seed=0, remove_printer_callback=True)
            dt = time.perf_counter()-t0
            peak = torch.cuda.max_memory_allocated()/2**30
            print(f'{cid} variates={per.target_dim} H={per.H} ctx={ctx:5d}: {dt/STEPS:.3f} s/step '
                  f'-> 1000 steps {dt/STEPS*1000/60:.1f} min, peak {peak:.2f} GiB', flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f'{cid} ctx={ctx}: OOM', flush=True)
        del pipe; torch.cuda.empty_cache()
