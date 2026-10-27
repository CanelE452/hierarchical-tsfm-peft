"""Package campaign reports and source with explicit preservation/ZIP checks."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

from . import runtime


def build_bundle(*, preparation=False):
    suffix='_preparation' if preparation else '_bundle'
    target=runtime.REPORT/(runtime.NAME+suffix+'.zip')
    receipt_name='PREPARATION_BUNDLE_RECEIPT.json' if preparation else 'BUNDLE_RECEIPT.json'
    manifest_name='PREPARATION_BUNDLE_MANIFEST.json' if preparation else 'BUNDLE_MANIFEST.json'
    if target.exists():
        with zipfile.ZipFile(target) as archive:
            if archive.testzip():
                raise ValueError('Interrupted/corrupt bundle preserved; use a new explicitly named packaging attempt')
            saved=json.loads(archive.read('reports/'+runtime.NAME+'/'+manifest_name))
            for row in saved['files']:
                if hashlib.sha256(archive.read(row['path'])).hexdigest()!=row['sha256']:
                    raise ValueError('Existing bundle content mismatch')
            count=len(archive.infolist())
        receipt=dict(path=str(target),bytes=target.stat().st_size,sha256=runtime.sha(target),
                     entries=count,zip_crc='PASS',archived_content_hashes='PASS',
                     existing_files_unchanged=runtime.read(runtime.REPORT/(
                         'PREPARATION_POST_HASHES.json' if preparation else 'POST_HASHES.json'))['status']=='PASS',
                     kind='preparation_only' if preparation else 'execution_results')
        if (runtime.REPORT/receipt_name).exists():
            if runtime.read(runtime.REPORT/receipt_name)!=receipt:
                raise ValueError('Existing bundle receipt mismatch')
        else:
            runtime.write_new(runtime.REPORT/receipt_name,receipt)
        return receipt
    before=runtime.read(runtime.REPORT/'PRE_HASHES.json')
    files=[]
    changed=[]
    for row in before['files']:
        path=Path(row['path'])
        actual=runtime.sha(path) if path.is_file() else None
        files.append(dict(row,post_sha256=actual,matches=actual==row['sha256']))
        if actual!=row['sha256']:
            changed.append(str(path))
    post_name='PREPARATION_POST_HASHES.json' if preparation else 'POST_HASHES.json'
    post=dict(
        status='PASS' if not changed else 'FAIL_EXISTING_FILE_CHANGED',
        scope=before['scope'],checked=len(files),changed=changed,files=files,
        checked_utc=datetime.now(timezone.utc).isoformat())
    if (runtime.REPORT/post_name).exists():
        old=runtime.read(runtime.REPORT/post_name)
        if any(old[key]!=post[key] for key in ('status','scope','checked','changed','files')):
            raise ValueError('Preservation state changed after packaging began')
    else:
        runtime.write_new(runtime.REPORT/post_name,post)
    members=[]
    for root in (runtime.REPORT,runtime.REPO/'research'/runtime.NAME):
        members.extend(path for path in root.rglob('*') if path.is_file()
                       and path.suffix not in ('.zip','.pyc') and '__pycache__' not in path.parts
                       and path.name not in ('BUNDLE_RECEIPT.json','PREPARATION_BUNDLE_RECEIPT.json',manifest_name))
    manifest=runtime.REPORT/manifest_name
    records=[dict(path=p.relative_to(runtime.REPO).as_posix(),bytes=p.stat().st_size,sha256=runtime.sha(p))
             for p in sorted(set(members))]
    content=dict(kind='preparation_only' if preparation else 'execution_results',
                 files=records,private_prediction_arrays_included=False,
                 note='Reports and source only; source data, frozen model and training checkpoints remain local.')
    if manifest.exists():
        if runtime.read(manifest)!=content:
            raise ValueError('Packaging inputs changed after manifest was sealed')
    else:
        runtime.write_new(manifest,content)
    members.append(manifest)
    with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as archive:
        for path in sorted(set(members)):
            archive.write(path,path.relative_to(runtime.REPO).as_posix())
    with zipfile.ZipFile(target) as archive:
        bad=archive.testzip()
        if bad:
            raise ValueError('ZIP CRC failure: '+bad)
        for row in records:
            if hashlib.sha256(archive.read(row['path'])).hexdigest()!=row['sha256']:
                raise ValueError('ZIP content hash mismatch: '+row['path'])
        count=len(archive.infolist())
    receipt=dict(path=str(target),bytes=target.stat().st_size,sha256=runtime.sha(target),
                 entries=count,zip_crc='PASS',archived_content_hashes='PASS',
                 existing_files_unchanged=not changed,kind='preparation_only' if preparation else 'execution_results')
    runtime.write_new(runtime.REPORT/receipt_name,receipt)
    return receipt


if __name__=='__main__':
    print(json.dumps(build_bundle(),ensure_ascii=False))
