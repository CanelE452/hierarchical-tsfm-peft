#!/usr/bin/env python
"""Single entry point. No training occurs merely by importing the package."""
from __future__ import annotations
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
import argparse, json, sys, time, traceback
from pathlib import Path
from gap_screen.io import *


def main():
    p=argparse.ArgumentParser(description='Run a bounded Chronos-2 adaptation-gap study')
    p.add_argument('command',choices=['all','prepare','preflight','train','predict','report','verify','fit-worker'])
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    p.add_argument('--panel',choices=['all','ettm2','jena'],default='all')
    p.add_argument('--device',default='cuda')
    p.add_argument('--model-dir',default=None)
    p.add_argument('--ettm2-file',default=None);p.add_argument('--jena-file',default=None)
    p.add_argument('--arm',choices=['HEAD','LORA','LORAPLUS']);p.add_argument('--seed',type=int)
    args=p.parse_args(); config=read_json(args.config)
    code=Path(__file__).parent.resolve();repo=args.repo.resolve()
    if not (repo/'.git').exists():raise Blocked('NOT_A_GIT_WORKTREE: install in the existing repository')
    from gap_screen.runner import Context,environment,preflight_panel,fit,seal_selection,predict_test
    from gap_screen.data import acquire
    from gap_screen.report import render_panel,summary_report
    from gap_screen.verify import verify
    ctx=Context(repo,code,config,args.device,args.model_dir)
    if args.command=='fit-worker':
        if args.panel=='all' or args.arm is None or args.seed not in config['seeds']: raise Blocked('INVALID_WORKER_ARGUMENTS')
        from gap_screen.model import configure_runtime
        configure_runtime(args.device,config['precision']);fit(ctx,args.panel,args.arm,args.seed);return
    panels=config['panels'] if args.panel=='all' else [args.panel]
    lock=ctx.cache/'CONTROLLER.lock'
    try:
        with lock.open('x') as f:f.write(json.dumps(dict(pid=os.getpid(),utc=time.time())))
    except FileExistsError:raise Blocked('CONTROLLER_LOCK_EXISTS: check ownership; do not delete a live process lock')
    errors=[]
    try:
        if args.command in ('all','prepare'):
            if not (ctx.out/'SOURCE_SEAL.json').exists():environment(ctx)
            else:ctx.assert_seal()
        elif not (ctx.out/'SOURCE_SEAL.json').exists():raise Blocked('RUN_PREPARE_FIRST')
        for panel in panels:
            result,cache=ctx.paths(panel);result.mkdir(parents=True,exist_ok=True)
            try:
                if args.command in ('all','prepare') and not (cache/'panel.npz').exists():
                    acquire(repo,cache,result,panel,config,getattr(args,panel+'_file'))
                if args.command in ('all','preflight'):preflight_panel(ctx,panel)
                if args.command in ('all','train'):
                    if not (result/'PREFLIGHT.json').exists():raise Blocked('PREFLIGHT_REQUIRED')
                    for i,seed in enumerate(config['seeds']):
                        arms=config['arms'] if i==0 else list(reversed(config['arms']))
                        for arm in arms:
                            fp=result/'fits'/f'{panel}_{arm}_{seed}.json'
                            if fp.exists() and read_json(fp).get('complete'):continue
                            command=[sys.executable,str(code/'run.py'),'fit-worker','--repo',str(repo),'--config',str(args.config.resolve()),
                                '--panel',panel,'--arm',arm,'--seed',str(seed),'--device',args.device]
                            if args.model_dir:command+=['--model-dir',args.model_dir]
                            # Inherit logs so the operator can see progress. Only our child is managed.
                            import subprocess
                            rc=subprocess.run(command,cwd=repo).returncode
                            if rc:raise Blocked(f'FIT_WORKER_FAILED: {panel}/{arm}/{seed}; inspect preserved work directory')
                    seal_selection(ctx,panel)
                if args.command in ('all','predict'):
                    if not (result/'PREDICTIONS_MANIFEST.json').exists():predict_test(ctx,panel)
                if args.command in ('all','report'):
                    render_panel(ctx,panel)
            except Exception as e:
                classification='BLOCKED_RESOURCE' if any(t in str(e).lower() for t in ['cuda','memory','gpu_busy']) else 'BLOCKED_EXECUTION'
                write_json(result/'ERROR.json',dict(classification=classification,error=str(e),traceback=traceback.format_exc(),utc=time.time()))
                errors.append(panel);print(f'{panel}: {classification}: {e}',file=sys.stderr)
                # Continue another panel only when data/implementation failure is local; no rescues or settings edits.
        if args.command in ('all','report','verify'):
            summary_report(ctx)
            if all((ctx.paths(x)[0]/'REPORT_RECEIPT.json').exists() for x in config['panels']): verify(ctx)
            else:write_json(ctx.out/'VERIFICATION.json',dict(status='INCOMPLETE',errors=errors,missing=[x for x in config['panels'] if not (ctx.paths(x)[0]/'REPORT_RECEIPT.json').exists()]))
        if errors:raise SystemExit(2)
    finally:
        lock.unlink(missing_ok=True)

if __name__=='__main__':
    try:main()
    except Blocked as e:
        print(f'BLOCKED: {e}',file=sys.stderr);raise SystemExit(2)
