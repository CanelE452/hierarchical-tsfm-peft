"""Scoped CLI; GPU stages run serially in owned fresh processes."""
import argparse
import os
import subprocess
import sys
import time

from . import common as c


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','validate','features','smoke','phase_a','phase_b','cost','report','verify'))
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.stage in ('report','verify'):
        module=__import__('research.gt_ts2_fastcheck_20261018_v1.reporting' if args.stage=='report' else 'research.gt_ts2_fastcheck_20261018_v1.verify',fromlist=['main'])
        return module.main()
    if args.stage=='prepare' or args.worker:
        if args.stage=='phase_b':
            from .admission import check
            admission=check(c.read_json(c.REPORT/'UPDATE_DECISION.json'),c.current_budget())
            admission.update(checked_unix=time.time(),pid=os.getpid(),
                             wrapper_source_sha256=c.file_sha(c.CODE/'run.py'),
                             gate_source_sha256=c.file_sha(c.CODE/'admission.py'))
            path=c.REPORT/'WORKER_PHASE_B_ADMISSION.json'
            if path.exists():
                path=c.REPORT/'admission_attempts'/f'worker_{time.time_ns()}.json'
            c.seal_json(path,admission)
            if admission['status']!='ADMITTED':
                print(c.json_bytes(admission).decode(),flush=True)
                return 3
        from . import execution
        if args.stage in ('smoke','phase_a','phase_b','cost'):
            from . import campaign
            module=campaign
        else:
            module=execution
        result=getattr(module,args.stage)()
        print(c.json_bytes(result).decode(),flush=True)
        return 0
    stamp=str(time.time_ns())
    log=c.CACHE/f'{args.stage}_{stamp}.log'
    receipt=dict(stage=args.stage,started_unix=time.time(),status='FAILED',command=f'python -m research.{c.CODE.name}.run {args.stage}',
                 log_reference='cache://'+log.name)
    try:
        if args.stage=='phase_b':
            from .admission import check
            admission=check(c.read_json(c.REPORT/'UPDATE_DECISION.json'),c.current_budget())
            admission.update(checked_unix=time.time(),gate_source_sha256=c.file_sha(c.CODE/'admission.py'))
            admission_path=c.REPORT/'PHASE_B_ADMISSION.json'
            if admission_path.exists():
                admission_path=c.REPORT/'admission_attempts'/f'{stamp}.json'
            c.seal_json(admission_path,admission)
            if admission['status']!='ADMITTED':
                receipt['status']='NOT_RUN_BUDGET'
                print(c.json_bytes(admission).decode(),flush=True)
                return 0
        with c.gpu_lease(args.stage):
            c.progress(args.stage,status='RUNNING')
            env=os.environ.copy()
            env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
            env['PYTHONDONTWRITEBYTECODE']='1'
            with log.open('xb') as stream:
                result=subprocess.run([sys.executable,'-m',f'research.{c.CODE.name}.run',args.stage,'--worker'],
                                      cwd=c.ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            receipt['exit_code']=result.returncode
            if result.returncode==3 and args.stage=='phase_b':
                receipt['status']='NOT_RUN_BUDGET'
                return 0
            if result.returncode:
                raise RuntimeError('Owned GPU worker failed; original log preserved: '+log.name)
        receipt['status']='COMPLETE'
        return 0
    finally:
        receipt.update(ended_unix=time.time(),wall_seconds=time.time()-receipt['started_unix'],log_sha256=c.file_sha(log) if log.exists() else None)
        c.seal_json(c.REPORT/'stage_receipts'/f'{args.stage}_{stamp}.json',receipt)
        print(c.json_bytes(receipt).decode(),flush=True)


if __name__=='__main__':
    raise SystemExit(main())
