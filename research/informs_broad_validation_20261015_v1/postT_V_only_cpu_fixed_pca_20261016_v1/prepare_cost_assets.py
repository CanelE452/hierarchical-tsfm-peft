"""원 CPU loader용 승인 캐시 4파일의 portable opaque-byte 준비 도구.

기본은 metadata 확인만 한다. --execute에서만 새 파일을 xb로 복사하며,
기존 파일은 같은 bytes/SHA면 검증하고 다르면 거부한다. NPZ 해석,
원 raw source 접근, 모델/학습/추론/TEST API, 자식 실행은 없다.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import time

HERE=Path(__file__).resolve().parent
NAME="postT_V_only_cpu_fixed_pca_20261016_v1"
PLAN_SHA="b4207edec63331cf95b24909c24c692091251dd8e29316f8867b9f66c65a0930"
RUNNER_SHA="d73d866a6d03061d00fb7ca015a8f3b41e273b73011b22f0c1cd0fbbfb073dd9"
FILE_LOCKS=[
    dict(dataset="electricity",relative_path="data/electricity_permitted.npz",sha256="4d31bcc7584e48971a354230514fd505648ae82bc64c9bb71fba03c61ac5f64e",bytes=16885858),
    dict(dataset="electricity",relative_path="data/electricity_receipt.json",sha256="2c32b308e521ea223580555eeadf2b61694eba6cd0f52aae8f8e32eb4a4df944",bytes=6925),
    dict(dataset="pems08",relative_path="data/pems08_permitted.npz",sha256="44276ac16c3812a8e075a763f1faa20dd3f290f15d4617315494597719b8f24e",bytes=4880142),
    dict(dataset="pems08",relative_path="data/pems08_receipt.json",sha256="bc3b48bf4637728808fe0fbc550792756a08bfe6b2f3db3f7d2be2af4106d737",bytes=2048)]


def sha(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def new(path,value):
    with Path(path).open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")


def no_links(path):
    path=Path(path)
    if any(p.is_symlink() or (getattr(p,"is_junction",lambda:False)()) for p in [path]+list(path.parents)):
        raise ValueError("Symlink/junction 경로는 허용하지 않음")
    return path.resolve()


def relative(root,name):
    root=no_links(root);part=Path(name)
    if part.is_absolute() or ".." in part.parts:raise ValueError("명시 root 내부 상대경로 필요")
    result=no_links(root/part)
    if not result.is_relative_to(root) or result==root:raise ValueError("Root 밖 경로 거부")
    return result


def load_runner():
    path=HERE/"fixed_pca_runner.py"
    if sha(path)!=RUNNER_SHA:raise ValueError("검토된 PCA metadata runner SHA 변경")
    spec=importlib.util.spec_from_file_location("opaque_cost_asset_metadata",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def metadata(public,cache,asset,plan_sha=None):
    public,cache,asset=(no_links(p) for p in (public,cache,asset))
    if plan_sha is not None and plan_sha!=PLAN_SHA:raise ValueError("고정 PCA PLAN SHA 불일치")
    runner=load_runner();plan=runner.metadata(public,cache,asset,PLAN_SHA)
    expected={"data/"+ds+suffix for ds in ("electricity","pems08") for suffix in ("_permitted.npz","_receipt.json")}
    if len(FILE_LOCKS)!=4 or {r["relative_path"] for r in FILE_LOCKS}!=expected:
        raise ValueError("승인된 정확한 네 캐시 파일만 허용")
    files=[]
    for lock in FILE_LOCKS:
        source=relative(cache,lock["relative_path"])
        if (not source.is_file() or source.stat().st_size!=lock["bytes"] or sha(source)!=lock["sha256"]
                or lock["dataset"] not in ("electricity","pems08")
                or not source.name.startswith(lock["dataset"]+"_")):
            raise ValueError("승인 원 캐시 파일의 SHA/bytes/자료명 불일치")
        destination_relative=NAME+"/data/"+source.name
        target=relative(cache,destination_relative)
        if target.exists():
            if not target.is_file() or target.stat().st_size!=lock["bytes"] or sha(target)!=lock["sha256"]:
                raise ValueError("기존 목적 캐시가 다름; 덮어쓰지 않음")
            status="EXISTING_IDENTICAL"
        else:status="ABSENT"
        files.append(dict(dataset=lock["dataset"],source_cache_relative_path=lock["relative_path"],destination_cache_relative_path=destination_relative,
            sha256=lock["sha256"],bytes=lock["bytes"],destination_status=status))
    for dataset in ("electricity","pems08"):
        receipt=read(relative(cache,"data/"+dataset+"_receipt.json"))
        task=plan["task_contracts"][dataset]
        archive=next(row for row in files if row["source_cache_relative_path"]=="data/"+dataset+"_permitted.npz")
        if (receipt.get("name")!=dataset or receipt.get("shape")!=task["shape"] or receipt.get("metadata")!=task["metadata"]
                or receipt.get("interval_seconds")!=task["interval_seconds"]
                or receipt.get("cache_reference")!=archive["source_cache_relative_path"]
                or receipt.get("cache_sha256")!=archive["sha256"] or receipt.get("cache_bytes")!=archive["bytes"]
                or receipt.get("source_sha256")!=task["metadata"].get("source_sha256")):
            raise ValueError("Task/직접 loader receipt 필드/승인 archive 연결 불일치")
    return dict(plan=plan,files=files,source39_and_nine_immutable_pins_checked=True,
        shape_is_receipt_metadata_not_numeric_array_verification=True)


def execute(public,cache,asset,plan_sha,output,*,prior_proof_sha=None):
    public,cache,asset=(no_links(p) for p in (public,cache,asset))
    if plan_sha!=PLAN_SHA:raise ValueError("실행에는 고정 PLAN SHA 필요")
    if prior_proof_sha is not None and not re.fullmatch(r"[0-9a-f]{64}",prior_proof_sha):raise ValueError("선행 proof SHA 형식 불일치")
    state=metadata(public,cache,asset,plan_sha);plan=state["plan"]
    post=public/NAME;output=no_links(output)
    if output==post or not output.is_relative_to(post) or output.exists():raise FileExistsError("새 별도 public 진단 출력 필요")
    added=sum(r["bytes"] for r in state["files"] if r["destination_status"]=="ABSENT")
    used=sum(p.stat().st_size for root in (public,cache) for p in root.rglob("*") if p.is_file())
    reserve=256*1024**2
    if used+added+reserve>plan["storage_limit_bytes"] or shutil.disk_usage(cache).free<added+reserve:
        raise RuntimeError("C+R30GiB/free-space 입장 한도 초과")
    if time.time()>=plan["deadline_unix"]:raise TimeoutError("원 campaign deadline 경과")
    helper_sha=sha(__file__)
    output.mkdir(parents=True,exist_ok=False)
    new(output/"STARTED.json",dict(plan_sha256=plan_sha,helper_sha256=helper_sha,source_file_count=4,created_unix=time.time(),numeric_deserializations=0))
    copied,verified,error,postflight,preserved=[],[],None,None,False
    try:
        for row in state["files"]:
            if time.time()>=plan["deadline_unix"]:raise TimeoutError("파일 처리 전 deadline 경과")
            source=relative(cache,row["source_cache_relative_path"])
            target=relative(cache,row["destination_cache_relative_path"])
            if sha(source)!=row["sha256"] or source.stat().st_size!=row["bytes"]:raise ValueError("처리 중 원 캐시 변경")
            if row["destination_status"]=="EXISTING_IDENTICAL":
                if sha(target)!=row["sha256"] or target.stat().st_size!=row["bytes"]:raise ValueError("검증 중 기존 목적 캐시 변경")
                verified.append(row["destination_cache_relative_path"])
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with source.open("rb") as stream,target.open("xb") as destination:shutil.copyfileobj(stream,destination,length=1048576)
                copied.append(row["destination_cache_relative_path"])
                if sha(source)!=row["sha256"] or sha(target)!=row["sha256"] or target.stat().st_size!=row["bytes"]:
                    raise ValueError("복사 중 source/destination bytes 변경")
        if time.time()>=plan["deadline_unix"]:raise TimeoutError("검증 종료 전 deadline 경과")
    except Exception as failure:
        error=failure
    finally:
        try:
            after=metadata(public,cache,asset,plan_sha)
            if (sha(__file__)!=helper_sha or any(r["destination_status"]!="EXISTING_IDENTICAL" for r in after["files"])
                    or [{k:v for k,v in r.items() if k!="destination_status"} for r in after["files"]]
                    !=[{k:v for k,v in r.items() if k!="destination_status"} for r in state["files"]]):
                raise ValueError("입력/helper/목적 바이트 전후 불일치")
            preserved=True
        except Exception as failure:
            postflight=failure
        used_after=sum(p.stat().st_size for root in (public,cache) for p in root.rglob("*") if p.is_file())
        result=dict(status="PASS_OPAQUE_PERMITTED_CACHE_ACCESS_PREPARED_NOT_COST_EXECUTED" if error is None and postflight is None and preserved else "FAILED_PRESERVED",
            plan_sha256=plan_sha,helper_sha256=helper_sha,runner_sha256=RUNNER_SHA,loader_source_sha256=plan["source39"]["data_contract.py"],
            files=state["files"],copied_files=len(copied),verified_existing_files=len(verified),
            copied_bytes=sum(r["bytes"] for r in state["files"] if r["destination_cache_relative_path"] in copied),
            original_source_registry_selection_score_cost_T_pins_preserved=preserved,existing_destinations_overwritten=0,
            original_source39_sha256=plan["source39"],original_immutable_public_sha256=plan["immutable_public"],
            copied_cache_relative_paths=copied,verified_existing_cache_relative_paths=verified,
            error_class=type(error).__name__ if error else None,postflight_error_class=type(postflight).__name__ if postflight else None,
            prior_proof_sha256_supplied_without_opening=prior_proof_sha,prior_private_proof_read=False,
            combined_bytes_before=used,combined_bytes_after=used_after,storage_limit_bytes=plan["storage_limit_bytes"],admission_reserve_bytes=reserve,
            numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules,archive_numeric_deserializations=0,
            actual_models_or_forecasts_or_cost_trials=0,child_processes_started=0,new_T_target_or_scoring=False,created_unix=time.time(),
            shape_evidence_scope="Pinned receipt/Task metadata only; NPZ contents remain opaque and arrays are not revalidated",
            access_scope="Only existing permitted NPZ and receipt bytes; no raw source/protected Electricity tail file is opened")
        new(output/"COST_CACHE_ASSET_ACCESS.json",result)
    if error is not None:raise error
    if postflight is not None:raise postflight
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("public","cache","asset"):parser.add_argument("--"+name+"-root",required=True)
    parser.add_argument("--plan-sha");parser.add_argument("--output");parser.add_argument("--prior-proof-sha")
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    public,cache,asset=(no_links(value) for value in (args.public_root,args.cache_root,args.asset_root))
    if args.execute:
        if not args.plan_sha or not args.output:parser.error("--execute에는 --plan-sha와 새 --output 필요")
        value=execute(public,cache,asset,args.plan_sha,Path(args.output),prior_proof_sha=args.prior_proof_sha)
        print(json.dumps(dict(status=value["status"],copied_files=value["copied_files"],verified_existing_files=value["verified_existing_files"],copied_bytes=value["copied_bytes"])))
    else:
        state=metadata(public,cache,asset,args.plan_sha)
        print(json.dumps(dict(status="METADATA_ONLY_OPAQUE_ASSET_ACCESS",files=state["files"],files_written=0,
            numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules,child_processes_started=0,numeric_deserializations=0,
            helper_sha256=sha(__file__),plan_sha256=PLAN_SHA)))


if __name__=="__main__":main()
