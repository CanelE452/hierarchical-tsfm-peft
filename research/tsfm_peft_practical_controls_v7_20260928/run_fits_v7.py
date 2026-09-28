"""Sequential launch of approved fits; subprocess failures stop this launcher."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from runtime_v7 import HERE, CACHE, digest

parser=argparse.ArgumentParser()
parser.add_argument('--dataset',choices=['robin','jena'],required=True)
args=parser.parse_args()
protocol=json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))
for spec in (s for s in protocol['fits'] if s['dataset']==args.dataset):
    out=HERE/'runs'/spec['id']
    result_path=out/'result.json'
    if result_path.exists():
        result=json.loads(result_path.read_text(encoding='utf-8'))
        if result['status']!='complete' or result['spec']!=spec or digest(result['checkpoint'])!=result['checkpoint_sha256']:
            raise RuntimeError('Existing result is not valid for automatic reuse: '+spec['id'])
        print('Reused completed attempt '+spec['id'],flush=True)
        continue
    if out.exists():
        raise RuntimeError('Interrupted/failed attempt requires explicit new retry spec; not restarted: '+spec['id'])
    command=[sys.executable,'-B',str(HERE/'train_v7.py'),'--dataset',args.dataset,'--id',spec['id'],
             '--job','gpu_'+spec['id'],'--reserve-seconds','1200']
    print('Launching '+spec['id'],flush=True)
    logs=CACHE/'logs'
    logs.mkdir(parents=True,exist_ok=True)
    log=logs/(spec['id']+'.log')
    with log.open('x',encoding='utf-8') as handle:
        outcome=subprocess.run(command,stdout=handle,stderr=subprocess.STDOUT)
    print('Finished '+spec['id']+' exit='+str(outcome.returncode)+' log='+str(log),flush=True)
    if outcome.returncode:
        raise SystemExit(outcome.returncode)
print('All requested base fits complete',flush=True)

