"""Actual B2/C321 GPU batching probe; synthetic data, no research score."""
import argparse,json,time,hashlib
from pathlib import Path
import numpy as np
import torch
from gpu_runtime import FrozenBolt
from networks import ForecastAdapter
p=argparse.ArgumentParser();p.add_argument("--model",required=True);p.add_argument("--cache",required=True);a=p.parse_args()
t=time.time();torch.set_num_threads(2);torch.manual_seed(92601)
f=FrozenBolt(a.model,48,row_microbatch=128,checkpoint_inputs=False)
model=ForecastAdapter("B2",512,48,321,321,np.zeros(321,np.float32),np.eye(321,dtype=np.float32)).cuda()
x=torch.randn(16,512,321,device="cuda");y=torch.randn(16,48,321,device="cuda")
opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0)
with torch.no_grad():
 f.row_microbatch=8;small=f(x[:1]);f.row_microbatch=128;large=f(x[:1])
 parity=dict(atol=1e-5,rtol=1e-4,max_abs=float((small-large).abs().max()),pass_=bool(torch.allclose(small,large,atol=1e-5,rtol=1e-4)))
 assert parity['pass_']
torch.cuda.reset_peak_memory_stats();seconds=[]
for step in range(3):
 torch.cuda.synchronize();s=time.perf_counter();opt.zero_grad(set_to_none=True)
 for i in range(0,16,4):
  loss=(model(x[i:i+4],f)-y[i:i+4]).square().mean()/4
  loss.backward()
 norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1)
 if not torch.isfinite(norm):raise RuntimeError('Probe gradient is not finite')
 opt.step();torch.cuda.synchronize();seconds.append(time.perf_counter()-s)
 print(json.dumps(dict(step=step+1,seconds=seconds[-1],peak_MiB=torch.cuda.max_memory_allocated()/2**20)),flush=True)
f.assert_frozen_unchanged()
r=dict(status='PASS',scope='SYNTHETIC_B2_ACTUAL_F_BACKWARD',scientific_fits=0,updates=3,channels=321,latent_k=321,microbatch=4,effective_batch=16,F_row_microbatch=128,checkpoint_inputs=False,seconds=seconds,median_update_seconds=float(np.median(seconds)),peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,peak_reserved_MiB=torch.cuda.max_memory_reserved()/2**20,parity=parity,model_sha256=f.initial_hash,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),process_seconds=time.time()-t)
with (Path(__file__).parent/'PROBE_BATCHING.json').open('x') as out:json.dump(r,out,indent=2)
with (Path(a.cache)/'GPU_USAGE.jsonl').open('a') as out:out.write(json.dumps(dict(kind='batching_probe',started_unix=t,finished_unix=time.time(),charged_process_seconds=time.time()-t))+'\n')
print('BATCHING_PROBE_COMPLETE',flush=True)
