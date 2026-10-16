"""명시한 post-T namespace 한 곳의 공개 충분통계를 원형 보존해 포장한다.

기본 metadata는 통계 본문을 읽거나 압축하지 않는다. --execute는 ROOT의
quiet-chain 확인 후 사용한다. 원 stats_pack의 bounded codec/복원/검사
함수를 재사용하며 원 모델/NPZ/가중치/원자료 및 기존 파일을 수정하지 않는다.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import time

HERE=Path(__file__).resolve().parent
ORIGINAL_SHA="0d8c4e4cc9881b70907acf7a154133a64cb582227c1df357521a7668e7fde3fd"
NAMES=dict(lag="postT_V_only_cpu_lag_20261016_v1",fixed_pca="postT_V_only_cpu_fixed_pca_20261016_v1",multik="postT_V_only_cpu_multik_20261016_v1")
PLAN_SHAS=dict(lag="fedf00ebb1807528514c0685a6b226fd6a466149531adc7a5791ffa07d92af35",
    fixed_pca="b4207edec63331cf95b24909c24c692091251dd8e29316f8867b9f66c65a0930",
    multik="2051573b8dee48200eb34e96e37e7af0d197442619ea1ba52a36a80ad4e27aec")
DEADLINE=1792222550


def load_original(public):
    import hashlib
    path=Path(public)/"stats_pack.py"
    if hashlib.sha256(path.read_bytes()).hexdigest()!=ORIGINAL_SHA:raise ValueError("원 stats_pack source SHA 변경")
    spec=importlib.util.spec_from_file_location("postT_original_stats_pack",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def new(path,value):
    with Path(path).open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")


def metadata(repo,public,cache,namespace,selection):
    repo,public,cache=(Path(p).resolve() for p in (repo,public,cache))
    if namespace not in NAMES or not repo.is_dir() or public.parents[1]!=repo or public.parent.name!="research" or not cache.is_dir():
        raise ValueError("명시 repo와 원 research/campaign root, cache, namespace 필요")
    core=load_original(public)
    prefix=(public/NAMES[namespace]).relative_to(repo).as_posix()
    plan_path=core._inside(repo,prefix+"/PLAN.json")
    if core.sha_file(plan_path)!=PLAN_SHAS[namespace]:raise ValueError("고정 post-T PLAN SHA 변경")
    plan=read(plan_path)
    if len(plan["source39"])!=39 or len(plan["immutable_public"])!=9:raise ValueError("원 source39/9immutable guard 범위 불일치")
    pins={}
    for mapping in (plan["source39"],plan["immutable_public"]):
        for name,digest in mapping.items():
            path=core._inside(public,name)
            if core.sha_file(path)!=digest:raise ValueError("원 과학/registry/score/cost/T input 변경")
            pins[path.relative_to(repo).as_posix()]=digest
    registry=core._inside(repo,prefix+"/"+plan["postT_registry_relative_path"])
    if core.sha_file(registry)!=plan["postT_registry_sha256"]:raise ValueError("별도 registry 변경")
    pins[plan_path.relative_to(repo).as_posix()]=PLAN_SHAS[namespace]
    pins[registry.relative_to(repo).as_posix()]=plan["postT_registry_sha256"]
    pins[(public/"stats_pack.py").relative_to(repo).as_posix()]=ORIGINAL_SHA
    selected=read(selection)
    if not isinstance(selected,dict) or set(selected)!={"files"} or not isinstance(selected["files"],list) or not 1<=len(selected["files"])<=10000:
        raise ValueError("files 배열만 가진 명시적 선택 JSON 필요")
    seen=set();records=[]
    for record in selected["files"]:
        if set(record)!={"repository_relative_path","gzip_bytes","gzip_sha256"}:raise ValueError("입력 파일의 정확한 path/bytes/SHA 필요")
        name=core._safe_name(record["repository_relative_path"])
        relative=PurePosixPath(name).relative_to(prefix)
        if (relative.parts[0] not in {"predictions","ordered_predictions","route_predictions","route_replays","raw_calibration"}
                or relative.name!="PUBLIC_ERROR_STATS.csv.gz" or "T_final" in relative.parts
                or name.casefold() in seen):raise ValueError("한 namespace의 중복 없는 공개 V 통계만 허용")
        seen.add(name.casefold())
        size=record["gzip_bytes"]
        if isinstance(size,bool) or not isinstance(size,int) or not 0<size<=core.MAX_ARCHIVE or not re.fullmatch(r"[0-9a-f]{64}",record["gzip_sha256"]):
            raise ValueError("입력 gzip SHA/byte 한도 불일치")
        path=core._inside(repo,name)
        if not path.is_file() or path.stat().st_size!=size:raise ValueError("명시 통계 파일 없음/크기 변경")
        records.append(dict(record))
    if sum(r["gzip_bytes"] for r in records)>core.MAX_OUTPUT:raise ValueError("원 bounded restore 출력 상한 초과")
    return dict(repo=repo,public=public,cache=cache,namespace=namespace,prefix=prefix,core=core,plan=plan,
        records=records,pins=pins,selection_sha256=core.sha_file(selection))


def storage_admission(state,pending):
    used=sum(p.stat().st_size for root in (state["public"],state["cache"]) for p in root.rglob("*") if p.is_file())
    reserve=256*1024**2
    if used+pending+reserve>30*1024**3 or shutil.disk_usage(state["cache"]).free<pending+reserve:
        raise RuntimeError("C+R30GiB 또는256MiB 입장 여유 부족")
    return used


def execute(repo,public,cache,namespace,selection,output,*,wall_seconds=1100):
    if isinstance(wall_seconds,bool) or not isinstance(wall_seconds,(int,float)) or not math.isfinite(wall_seconds) or not 0<wall_seconds<=1100:
        raise ValueError("wall_seconds는 0초 초과1100초 이하")
    state=metadata(repo,public,cache,namespace,selection);core=state["core"]
    repo,public,cache=state["repo"],state["public"],state["cache"]
    output=Path(output)
    if not output.is_absolute():raise ValueError("명시된 절대 output 경로 필요")
    output=core._inside(repo,output.resolve().relative_to(repo).as_posix())
    if output.parent!=public/"stats_packs" or not output.name.startswith("pack_") or output.exists():raise FileExistsError("원R/stats_packs/pack_* 새 output만 허용")
    started=time.time();deadline=min(started+wall_seconds,DEADLINE)
    if deadline<=started:raise TimeoutError("원 campaign deadline 경과")
    original_bytes=sum(r["gzip_bytes"] for r in state["records"])
    used=storage_admission(state,original_bytes)
    wrapper_sha=core.sha_file(__file__)
    output.mkdir(parents=True,exist_ok=False)
    records=[];archives=[];group=[];seen=set();assignment={}
    for selected in sorted(state["records"],key=lambda r:r["repository_relative_path"]):
        if time.time()>=deadline:raise TimeoutError("포장 wall 한도 경과; 부분 output 보존")
        path=core._inside(repo,selected["repository_relative_path"]);blob=path.read_bytes()
        if len(blob)!=selected["gzip_bytes"] or core.sha(blob)!=selected["gzip_sha256"]:raise ValueError("선택된 원 gzip SHA 변경")
        raw=core._gunzip(blob);_,row_count=core._csv_schema(raw)
        recipe=core.find_gzip_recipe(blob,raw);lines=raw.splitlines(keepends=True);prefix=lines[1].split(b",",1)[0]+b","
        constant=all(line.startswith(prefix) for line in lines[1:]) and b'"' not in prefix
        if recipe is not None and constant:
            payload=lines[0].split(b",",1)[1]+b"".join(line[len(prefix):] for line in lines[1:]);mode="csv_common9";suffix=".csv"
        else:payload=blob;mode="opaque_gzip";suffix=".csv.gz";recipe=None
        member="members/"+core.sha(payload)+suffix
        if member not in seen:
            if group and sum(len(v) for n,v in group)+len(payload)>64*1024**2:
                assignment.update(core._emit(group,output,archives));group=[]
                storage_admission(state,original_bytes)
            group.append((member,payload));seen.add(member)
        record=dict(repository_relative_path=selected["repository_relative_path"],member=member,storage_mode=mode,
            gzip_bytes=len(blob),gzip_sha256=core.sha(blob),csv_bytes=len(raw),csv_sha256=core.sha(raw),row_count=row_count,
            gzip_recipe=recipe,run_id_prefix=prefix.decode("utf-8") if mode=="csv_common9" else None,
            same_environment_original_gzip_restore_verified=True)
        if core._restore_blob(record,payload)!=blob:raise ValueError("원 codec의 gzip byte 재생성 실패")
        records.append(record)
    if group:assignment.update(core._emit(group,output,archives))
    for record in records:record["archive"]=assignment[record["member"]]
    if time.time()>=deadline:raise TimeoutError("별도 복원 전 wall 한도 경과")
    manifest=dict(format=core.SCHEMA,status="PENDING_RESTORE_VERIFICATION",source_sha256=ORIGINAL_SHA,wrapper_source_sha256=wrapper_sha,
        campaign_repository_relative=state["prefix"],namespace=namespace,records=records,archives=archives,
        selection_sha256=state["selection_sha256"],original_total_bytes=original_bytes,packed_total_bytes=sum(a["bytes"] for a in archives),
        zlib_compile_version=core.zlib.ZLIB_VERSION,zlib_runtime_version=core.zlib.ZLIB_RUNTIME_VERSION,
        max_archive_bytes=core.MAX_ARCHIVE,max_expanded_tar_bytes=core.MAX_TAR,max_member_bytes=core.MAX_MEMBER,
        original_files_modified=False,NPZ_or_weights_included=False,raw_array_values_read=False,public_aggregate_statistics_read=True,
        original_gzip_SHA_requires_reproducible_zlib=True,separate_temporary_restore_all_sha_verified=False)
    draft=output/".MANIFEST_VERIFY.json";new(draft,manifest)
    storage_admission(state,original_bytes)
    with tempfile.TemporaryDirectory(prefix="postT_stats_verify_",dir=cache) as folder:
        temporary=Path(folder).resolve()
        if temporary==cache or not temporary.is_relative_to(cache):raise ValueError("검증 temp 경로 범위 불일치")
        restored=core.restore(draft,temporary,deadline=deadline)
        if restored["restored"]!=len(records):raise ValueError("별도 복원 수 불일치")
        for record in records:
            blob=core._inside(temporary,record["repository_relative_path"]).read_bytes();raw=core._gunzip(blob)
            if (core.sha(blob)!=record["gzip_sha256"] or core.sha(raw)!=record["csv_sha256"]
                    or core._csv_schema(raw)[1]!=record["row_count"]):raise ValueError("별도 temp의 원 gzip/CSV/모든 행 검산 불일치")
        repeated=core.restore(draft,temporary,deadline=deadline)
        if repeated["restored"]!=0 or repeated["already_matching"]!=len(records):raise ValueError("반복 복원은 기존 exact 파일 읽기만 해야 함")
    if temporary.exists():raise ValueError("별도 임시 복원 경로 정리 실패")
    after=metadata(repo,public,cache,namespace,selection)
    if after["pins"]!=state["pins"] or after["selection_sha256"]!=state["selection_sha256"] or core.sha_file(__file__)!=wrapper_sha:
        raise ValueError("wrapper/원 source/PLAN/registry/선택 입력 변경")
    for record in records:
        if core.sha_file(core._inside(repo,record["repository_relative_path"]))!=record["gzip_sha256"]:raise ValueError("원 충분통계 변경")
    storage_admission(state,0)
    if time.time()>=deadline:raise TimeoutError("완료 검산 wall 한도 경과")
    manifest.update(status="PASS",separate_temporary_restore_all_sha_verified=True)
    manifest_path=output/"MANIFEST.json";new(manifest_path,manifest)
    for archive in archives:core.inspect_archive(output/archive["file"])
    manifest_name=manifest_path.relative_to(repo).as_posix()
    core_name=(public/"stats_pack.py").relative_to(repo).as_posix()
    readme=("# post-T 공개 충분통계 원형 복원\n\n이 pack은 `"+namespace+"` 한 namespace만 포함한다. 원 CSV의 모든 행과 숫자, run ID 및 원 gzip bytes를 보존한다. 원 파일은 로컬에서 변경·삭제하지 않았다.\n\n"
        "- [원 path·bytes·SHA와 archive manifest](MANIFEST.json)\n- [실제 복원 검산 범위](PROOF.json)\n\n저장소 루트에서 실행한다.\n\n```text\npython "+core_name+" restore --manifest "+manifest_name+" --destination .\n```\n\n"
        "기존 일치 파일은 읽기만 하고 다른 파일은 덮어쓰지 않는다. 다른 zlib 환경에서 원 gzip 재생성이 다르면 실패하며 CSV/gzip SHA를 생략하지 않는다. 별도 temp에서 전 파일 gzip·CSV SHA/행 수와 반복 복원을 확인했다. 이는 공개 집계 통계의 저장·복원 검산이며 새 예측·학습·독립 일반화 검증이 아니다.\n")
    with (output/"README_KO.md").open("x",encoding="utf-8",newline="\n") as stream:stream.write(readme)
    proof=dict(status="PASS",namespace=namespace,manifest_repository_relative_path=manifest_name,manifest_sha256=core.sha_file(manifest_path),
        original_stats_pack_source_sha256=ORIGINAL_SHA,wrapper_source_sha256=wrapper_sha,selection_sha256=state["selection_sha256"],
        verified_original_gzip_and_CSV_files=len(records),verified_CSV_rows=sum(r["row_count"] for r in records),
        separate_restore=restored,repeated_restore=repeated,separate_temp_removed=True,source_and_original_statistics_unchanged=True,
        original_source_and_immutable_pins=state["pins"],original_gzip_bytes=original_bytes,archive_bytes=manifest["packed_total_bytes"],
        archive_count=len(archives),storage_before_bytes=used,actual_arrays_models_or_child_operations=0,
        wall_seconds=time.time()-started,deadline_unix=deadline,new_T_target_or_scoring=False,
        publication_total_size_and_link_gate_not_claimed=True,original_helper_pack_entry_was_not_called=True)
    new(output/"PROOF.json",proof)
    additions=[(output/name).relative_to(repo).as_posix() for name in ("MANIFEST.json","README_KO.md","PROOF.json","ADD_LIST.txt")]
    additions.extend((output/a["file"]).relative_to(repo).as_posix() for a in archives)
    with (output/"ADD_LIST.txt").open("x",encoding="utf-8",newline="\n") as stream:stream.write("\n".join(sorted(additions))+"\n")
    return proof


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("repo","campaign","cache"):parser.add_argument("--"+name+"-root")
    parser.add_argument("--namespace",choices=tuple(NAMES));parser.add_argument("--selection")
    parser.add_argument("--output");parser.add_argument("--wall-seconds",type=float,default=1100);parser.add_argument("--execute",action="store_true")
    args=parser.parse_args();roots=(args.repo_root,args.campaign_root,args.cache_root)
    if args.execute:
        if not all(roots+(args.namespace,args.selection,args.output)):parser.error("실행에는 roots/namespace/선택JSON/새output 필요")
        value=execute(*(Path(p) for p in roots),args.namespace,Path(args.selection),Path(args.output),wall_seconds=args.wall_seconds)
        print(json.dumps(dict(status=value["status"],verified_files=value["verified_original_gzip_and_CSV_files"],archive_bytes=value["archive_bytes"])))
        return
    count=None
    if any(roots) or args.selection or args.namespace:
        if not all(roots+(args.namespace,args.selection)):parser.error("metadata 검산에는 roots/namespace/명시 선택JSON 필요")
        state=metadata(*(Path(p) for p in roots),args.namespace,Path(args.selection));count=len(state["records"])
    print(json.dumps(dict(status="METADATA_ONLY_POST_T_STATS_PACK",selected_files=count,one_namespace_per_manifest=True,
        original_stats_pack_sha256=ORIGINAL_SHA,statistics_content_reads=0,compressions=0,files_written=0,
        numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules,actual_arrays_models_or_child_operations=0)))


if __name__=="__main__":main()
