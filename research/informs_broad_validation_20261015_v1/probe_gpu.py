"""Actual GPU sizing probe on synthetic arrays; never a scientific result."""
import argparse,json,time,hashlib,os
from pathlib import Path
import numpy as np
import torch
from gpu_runtime import FrozenBolt

p=argparse.ArgumentParser();p.add_argument("--model",required=True);p.add_argument("--cache",required=True);args=p.parse_args()
started=time.time();cache=Path(args.cache);torch.set_num_threads(2);torch.manual_seed(92601)
f=FrozenBolt(args.model,48,row_microbatch=8,checkpoint_inputs=True)
results=[]
for c,k in ((11,3),(321,81),(321,321)):
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    x=torch.randn(16,512,c,device="cuda");y=torch.randn(16,48,c,device="cuda")
    basis=torch.linalg.qr(torch.randn(c,c),mode="reduced").Q[:,:k]
    E=torch.nn.Parameter(basis.to("cuda"));D=torch.nn.Parameter(basis.T.to("cuda"))
    opt=torch.optim.AdamW([E,D],lr=.001,weight_decay=0)
    before=(f.streams,f.physical_streams);timings=[];losses=[]
    for step in range(3):
        torch.cuda.synchronize();t=time.perf_counter();opt.zero_grad(set_to_none=True);loss_value=0.
        for j in range(16):
            z=x[j:j+1]@E;forecast=f(z)
            prediction=(forecast-z[:,-1:])@D+x[j:j+1,-1:]
            loss=(prediction-y[j:j+1]).square().mean()/16
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite sizing probe")
            loss.backward();loss_value+=loss.item()
        norm=torch.nn.utils.clip_grad_norm_([E,D],1)
        if not torch.isfinite(norm):raise RuntimeError("Nonfinite sizing gradient")
        opt.step();torch.cuda.synchronize();timings.append(time.perf_counter()-t);losses.append(loss_value)
    result=dict(channels=c,latent_k=k,length=512,horizon=48,effective_batch=16,microbatch=1,
        updates=3,seconds=timings,median_update_seconds=float(np.median(timings)),
        logical_F_rows=f.streams-before[0],physical_F_rows_including_recompute=f.physical_streams-before[1],
        peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,
        loss=losses,scope="SYNTHETIC_AFFINE_FRONT_ACTUAL_F_BACKWARD_LOWER_BOUND_NOT_DATASET_SCORE")
    results.append(result);print(json.dumps(result),flush=True)
    with (cache/"PROBE_PROGRESS.json").open("w") as out:json.dump(dict(completed=len(results),results=results),out,indent=2)
    del x,y,E,D,opt;torch.cuda.empty_cache()
f.assert_frozen_unchanged()
receipt=dict(status="ACTUAL_GPU_PROBE_COMPLETE",scientific_fits=0,synthetic_optimizer_updates=9,
    model_revision=f.model_revision,model_state_sha256=f.initial_hash,model_load_seconds=f.load_seconds,
    results=results,process_wall_seconds=time.time()-started,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    caveat="Affine front synthetic sizing lower bound; new real-task frontend timings still required")
with (Path(__file__).parent/"PROBE_ACTUAL_GPU.json").open("x") as out:json.dump(receipt,out,indent=2)
with (cache/"GPU_USAGE.jsonl").open("a") as out:out.write(json.dumps(dict(kind="probe",started_unix=started,finished_unix=time.time(),charged_process_seconds=time.time()-started))+"\n")
print("PROBE_COMPLETE",flush=True)
