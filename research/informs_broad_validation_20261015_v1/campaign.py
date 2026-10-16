"""Single-GPU registered campaign queue; final T scoring is a separate locked phase."""
import os
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("MKL_NUM_THREADS","2")
import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import data_contract as data
import fit_base
import registry_extension
from gpu_runtime import FrozenBolt
from resources import GPUReservation, check_storage_budget

HERE=Path(__file__).resolve().parent
CACHE=Path(os.environ.get("INFORMS_CACHE_ROOT",HERE.parents[1]/".cache"/HERE.name))
ASSET_ROOT=Path(os.environ.get("INFORMS_ASSET_ROOT",HERE.parents[1]))
SMALL_REV="772f3d25d38aec6d914c8949dab4462e2d46f5d8"
FAMILIES={"B0":"B0","B2":"B2","B4":"B4","DIRECT_G":"DIRECT_G","DIRECT_B2":"DIRECT_B2","DIRECT_B4":"DIRECT_B4","LORA_R8":"LORA","ADAPTS_BOLT_PORT":"ADAPTS","PCA_FIXED_BOLT":"PCA_FIXED_BOLT","LAG_CAUSAL":"LAG_CAUSAL","LAG_BIDIR":"LAG_BIDIR"}

def new_json(path,value):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("x",encoding="utf-8") as f:json.dump(value,f,indent=2,allow_nan=False)

def registry_rows():
 with (HERE/"EXPERIMENT_REGISTRY.csv").open(encoding="utf-8",newline="") as f:return list(csv.DictReader(f))

def record_status(run_id,**fields):
 path=HERE/"EXPERIMENT_REGISTRY.csv";rows=registry_rows();matches=[r for r in rows if r['experiment_id']==run_id]
 if len(matches)!=1:raise ValueError('Unregistered or duplicate experiment')
 matches[0].update({k:str(v) for k,v in fields.items()})
 with path.open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def used_gpu_seconds():
 path=CACHE/'GPU_USAGE.jsonl'
 return sum(json.loads(line)['charged_process_seconds'] for line in path.read_text().splitlines() if line.strip()) if path.exists() else 0.

def require_pre_selection(operation='new campaign fitting'):
 registry_extension.require_pre_selection(HERE,operation)

def execute_fits(ids,epochs,deadline,lease=None):
 require_pre_selection()
 torch.set_num_threads(2);registry={r['experiment_id']:r for r in registry_rows()};foundation=None;task_cache={}
 results=[]
 for run_id in ids:
  row=registry[run_id]
  if lease is not None:lease.check()
  check_storage_budget(CACHE,pending_bytes=128*1024**2)
  if row['family'] not in FAMILIES:raise ValueError('This phase is only for registered base fits')
  if time.time()>=deadline:break
  ds=row['dataset']
  if ds not in task_cache:task_cache[ds]=data.load_task(ds)
  task=task_cache[ds]
  family=FAMILIES[row['family']]
  direct=family in fit_base.DIRECT
  if not direct and foundation is None:
   foundation=FrozenBolt(ASSET_ROOT/'.cache/huggingface/models--amazon--chronos-bolt-small/snapshots'/SMALL_REV,task['H'],128,False)
  elif not direct and foundation.horizon!=task['H']:
   del foundation;gc.collect();torch.cuda.empty_cache()
   foundation=FrozenBolt(ASSET_ROOT/'.cache/huggingface/models--amazon--chronos-bolt-small/snapshots'/SMALL_REV,task['H'],128,False)
  # G/direct fits see no F. Cost measurements later launch fresh processes.
  active_f=None if direct else foundation
  k=int(row['latent_k']) or task['physical'].shape[1]
  spec=dict(run_id=run_id,dataset=ds,family=family,latent_k=k,seed=int(row['seed']),fold=row['fold'],
   epochs=list(epochs),lr=.0001 if family=='LORA' else .001,device='cuda',microbatch=4,effective_batch=16,
   F_row_microbatch=128,checkpoint_inputs=False,lora_rank=8,lora_alpha=16,
   stage_label='registered_initial_breadth' if max(epochs)<3 else 'registered_screen')
  record_status(run_id,execution_status='RUNNING',reason='search A_fit, A_stop selection then new A60 refit')
  started=time.time();print(json.dumps(dict(event='FIT_START',run_id=run_id,epochs=epochs)),flush=True)
  torch.cuda.reset_peak_memory_stats()
  try:
   receipt=fit_base.fit_base(spec,task,CACHE,HERE,active_f,deadline)
   record_status(run_id,execution_status=('PARTIAL_SCREEN025' if receipt['status']=='COMPLETE_VALID' and max(epochs)<3 else receipt['status']),training_status=receipt['training_status'],validity_status='PENDING_INDEPENDENT_VALIDATION',reason=('0.25 screen and selected A60 refit complete; epochs1/3 pending; no T evaluation' if max(epochs)<3 else 'No T evaluation yet'))
   results.append(receipt);print(json.dumps(dict(event='FIT_DONE',run_id=run_id,status=receipt['status'],selected_epoch=receipt.get('selected_epoch'),search_updates=receipt.get('search_updates'),refit_updates=receipt.get('refit_updates'))),flush=True)
  except Exception as exc:
   error=dict(run_id=run_id,status='EXECUTION_FAILED',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),started_unix=started,finished_unix=time.time())
   directory=CACHE/'failures';directory.mkdir(exist_ok=True)
   new_json(directory/(run_id+'_'+str(time.time_ns())+'.json'),error)
   record_status(run_id,execution_status='EXECUTION_FAILED',validity_status='INVALID_OR_NOT_COMPLETED',reason=type(exc).__name__)
   print(json.dumps(dict(event='FIT_FAILED',run_id=run_id,error_type=type(exc).__name__,error=str(exc))),flush=True)
   if foundation is not None:
    foundation.assert_frozen_unchanged()
  gc.collect();torch.cuda.empty_cache()
 return results

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--run-id',action='append');p.add_argument('--dataset');p.add_argument('--family',action='append')
 p.add_argument('--epochs',type=float,nargs='+',default=[.25]);p.add_argument('--max-jobs',type=int)
 p.add_argument('--reserve-gpu-seconds',type=float,default=3600)
 p.add_argument('--minimum-only',action='store_true')
 p.add_argument('--resume',action='store_true',help='Resume unchanged-contract BUDGET_LIMIT only; epoch extension requires a registered new continuation')
 a=p.parse_args();rows=registry_rows();selected=[]
 for row in rows:
  accepted={'PENDING_QUEUE'}|({'BUDGET_LIMIT'} if a.resume else set())
  if row['family'] not in FAMILIES or row['execution_status'] not in accepted:continue
  if a.minimum_only and row['requirement']!='MANDATORY_MINIMUM':continue
  if a.dataset and row['dataset']!=a.dataset:continue
  if a.family and row['family'] not in a.family:continue
  if a.run_id and row['experiment_id'] not in a.run_id:continue
  selected.append(row['experiment_id'])
 if a.run_id and not selected:raise ValueError('Requested run is absent, not pending, or unsupported by this phase')
 if a.max_jobs:selected=selected[:a.max_jobs]
 if not selected:print('No pending matching base fits');return
 require_pre_selection()
 with GPUReservation(CACHE,total_gpu_seconds=43200,reserve_seconds=a.reserve_gpu_seconds,
                     wall_deadline=1792050350+172800,purpose='training',kind='base_fit_queue') as lease:
  lease.check()
  execute_fits(selected,a.epochs,lease.deadline-5,lease)

if __name__=='__main__':main()
