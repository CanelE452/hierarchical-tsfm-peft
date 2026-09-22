#!/usr/bin/env python
"""Publish only this experiment's code/results. Never force-push or stage unrelated files."""
from __future__ import annotations
import argparse,json,re,sys,time
from pathlib import Path
from gap_screen.io import *

ALLOWED_EXT={'.py','.json','.jsonl','.csv','.md','.txt','.png','.yml','.yaml'}
DENIED_EXT={'.npz','.npy','.pt','.pth','.safetensors','.bin','.zip','.whl'}
SECRET=re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|hf_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|https?://[^\s/]+:[^\s@]+@)',re.I)


def git(repo,*args,check=True):return run_command(['git',*args],cwd=repo,check=check).stdout.strip()


def publish(repo:Path,code:Path,config:dict):
    repo=repo.resolve();code=code.resolve();out=repo/'results'/config['experiment_id']
    verify_report=read_json(out/'VERIFICATION.json')
    if verify_report['status']!='PASS':raise Blocked('PUBLISH_REQUIRES_COMPLETE_VERIFICATION')
    if read_json(out/'ENVIRONMENT.json').get('authenticity')!='REAL_MODEL_RUN':raise Blocked('SYNTHETIC_RESULTS_CANNOT_BE_PUBLISHED_AS_REAL')
    # Fresh CPU artifact verification: a stale PASS file is not sufficient.
    from gap_screen.runner import Context
    from gap_screen.verify import verify
    verify(Context(repo,code,config,device='cpu'),require_all=True)
    code_rel=code.relative_to(repo).as_posix();out_rel=out.relative_to(repo).as_posix()
    if not code_rel.startswith('experiments/'):raise Blocked('CODE_NOT_INSTALLED_UNDER_EXPERIMENTS')
    if git(repo,'diff','--cached','--name-only'):raise Blocked('UNRELATED_STAGED_FILES: do not mix commits')
    remote=git(repo,'remote','get-url','origin')
    if not re.search(r'[:/]CanelE452/hierarchical-tsfm-peft(?:\.git)?$',remote):raise Blocked('WRONG_REMOTE')
    branch=git(repo,'branch','--show-current')
    if not branch:raise Blocked('DETACHED_HEAD')
    # Refuse to push pre-existing unrelated local commits.
    git(repo,'fetch','origin',branch)
    upstream='FETCH_HEAD'
    ahead=git(repo,'rev-list',f'{upstream}..HEAD')
    if ahead:raise Blocked('PREEXISTING_UNPUSHED_COMMITS: publishing would include unrelated work')
    if git(repo,'merge-base','HEAD',upstream)!=git(repo,'rev-parse',upstream):raise Blocked('REMOTE_ADVANCED: no automatic merge/rebase')
    files=[]
    for root in (code,out):
        for p in root.rglob('*'):
            if not p.is_file() or any(x in p.parts for x in ('__pycache__','.pytest_cache','test_output')):continue
            if p.is_symlink():raise Blocked('SYMLINK_IN_PUBLISH_SCOPE '+p.name)
            if p.suffix in DENIED_EXT:raise Blocked('LARGE_PRIVATE_ARTIFACT_IN_PUBLISH_SCOPE '+p.name)
            if p.suffix not in ALLOWED_EXT:continue
            if p.stat().st_size>config['max_single_artifact_bytes']:raise Blocked('ARTIFACT_TOO_LARGE '+p.name)
            if p.suffix!='.png' and SECRET.search(p.read_text(encoding='utf-8',errors='replace')):raise Blocked('POSSIBLE_SECRET '+p.name)
            files.append(p.relative_to(repo).as_posix())
    # A scoped add may include intentional ignored results, but never .cache/raw/model weights.
    for i in range(0,len(files),100):git(repo,'add','-f','--',*files[i:i+100])
    staged=git(repo,'diff','--cached','--name-only').splitlines()
    if not all(p.startswith(code_rel+'/') or p.startswith(out_rel+'/') for p in staged):raise Blocked('UNSCOPED_STAGING')
    if staged:git(repo,'commit','-m','Complete bounded internal adaptation gap screen with reports and verification')
    artifact_commit=git(repo,'rev-parse','HEAD')
    git(repo,'push','origin',f'HEAD:refs/heads/{branch}')
    remote_head=git(repo,'ls-remote','origin',f'refs/heads/{branch}').split()[0]
    if remote_head!=artifact_commit:raise Blocked('REMOTE_HASH_NOT_CONFIRMED')
    receipt=dict(artifact_commit=artifact_commit,remote_confirmed=True,branch=branch,utc=time.time(),
        report=f'https://github.com/CanelE452/hierarchical-tsfm-peft/blob/{artifact_commit}/{out_rel}/SUMMARY_KO.md',
        note='This receipt references the data/report commit, not its own later receipt commit.')
    write_json(out/'PUSH_RECEIPT.json',receipt)
    git(repo,'add','-f','--',out_rel+'/PUSH_RECEIPT.json');git(repo,'commit','-m','Record verified report artifact commit')
    git(repo,'push','origin',f'HEAD:refs/heads/{branch}')
    final=git(repo,'rev-parse','HEAD');remote_final=git(repo,'ls-remote','origin',f'refs/heads/{branch}').split()[0]
    if final!=remote_final:raise Blocked('FINAL_RECEIPT_PUSH_NOT_CONFIRMED')
    print(json.dumps(dict(report=receipt['report'],artifact_commit=artifact_commit,final_remote_commit=final),ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    a=p.parse_args()
    try:publish(a.repo,Path(__file__).parent,read_json(a.config))
    except Blocked as e:print('PUSH_BLOCKED:',e,file=sys.stderr);raise SystemExit(2)
