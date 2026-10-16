"""Explicit numeric-aware publication inspection; frozen legacy sources unchanged.

This provides inspect/check_index, not legacy prepare/verify or scientific audit.
The default CLI reads no inputs. Actual inspection requires --execute and a
byte-pinned explicit file list; numeric statistics are decoded only then.
"""
import argparse
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import posixpath
import re
import sys


HERE=Path(__file__).resolve().parent
PUBLICATION_SHA="697476f0ba518c3d0d6888cc9744aac9e2e46cb614221a425e949803934cd1d6"
CORE_SHA="0d8c4e4cc9881b70907acf7a154133a64cb582227c1df357521a7668e7fde3fd"
CODEC_SHA="c9c42c780306b6e75eb68c939db1c3a9658b3e2680a3d57c15c9971cbefd8743"
TRANSPORT_SHA="5824cc8bebeb21bd1c818c4fd1fb2d5d0b2cd38ceb1d588228f6f69a05742a85"
TRANSPORT_DIRECTORY="publication_derivatives/transport_legacy_V_part0_20261016_v1"
SCHEMA="numeric_aware_publication_inspection_v1"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(path):
    path=Path(path).absolute()
    if any(p.is_symlink() or getattr(p,"is_junction",lambda:False)() for p in (path,*path.parents)):
        raise ValueError("Symlink or junction in publication path")
    return path.resolve()


def source_pins():
    expected={"publication_check.py":PUBLICATION_SHA,"stats_pack.py":CORE_SHA,"stats_pack_numeric.py":CODEC_SHA,
        "stats_pack_transport.py":TRANSPORT_SHA}
    for name,digest in expected.items():
        if sha(plain(HERE/name))!=digest:raise ValueError("Frozen publication/codec source SHA differs")
    return expected


def _load(name,filename,*,importer=None):
    spec=importlib.util.spec_from_file_location(name,HERE/filename)
    value=importlib.util.module_from_spec(spec)
    if importer is not None:value.__dict__["__builtins__"]=dict(vars(builtins),__import__=importer)
    spec.loader.exec_module(value)
    return value


def load_dependencies():
    source_pins();core=_load("numeric_publication_frozen_stats_core","stats_pack.py")
    def legacy_import(name,globals=None,locals=None,fromlist=(),level=0):
        if name=="stats_pack" and level==0:return core
        return builtins.__import__(name,globals,locals,fromlist,level)
    pc=_load("numeric_publication_frozen_inspector","publication_check.py",importer=legacy_import)
    codec=_load("numeric_publication_exact_codec","stats_pack_numeric.py")
    return pc,codec


def load_transport():
    source_pins();return _load("numeric_publication_exact_archive_transport","stats_pack_transport.py")


def inspect(campaign,files,tracked=None):
    """Inspect the complete explicit union; no writes, staging or model calls."""
    pc,codec=load_dependencies();before=source_pins();campaign=plain(campaign);root=pc._root(campaign)
    if not campaign.is_dir():raise ValueError("Existing campaign directory required")
    tracked=pc.tracked_files(root) if tracked is None else set(tracked)
    names=[];numeric=[];transported=[];normal=[];seen=set();failures=[];paths={};pre={}
    for raw in files:
        try:
            name,path=pc._canonical_name(root,raw)
            if name!=str(raw):raise ValueError("Canonical repository-relative names required")
            path=plain(path)
        except (ValueError,OSError):
            pc._failure(failures,"<invalid-relative-name>","FILE_PATH_UNSAFE");continue
        if name.casefold() in seen:
            pc._failure(failures,name,"DUPLICATE_FILE");continue
        seen.add(name.casefold());names.append(name);paths[name]=path
        if path.is_file():pre[name]=(path.stat().st_size,sha(path))
        (numeric if name.endswith(".numeric.xz") else transported if name.endswith(".transport.xz") else normal).append(name)
    base=pc.inspect(campaign,normal,tracked=tracked,virtual_files=numeric+transported)
    failures.extend(base["failures"]);inventory=list(base["files"]);numeric_records=[]
    published=set(names);case_targets=pc._link_case_inventory(published|tracked)
    links=[]
    # Re-evaluate every successful legacy link against the real complete union.
    for link in base["links"]:
        if link["target"] not in published|tracked:
            pc._failure(failures,link["file"],"LINK_NOT_PUBLISHED",target=link["target"])
        links.append(link)
    for name in numeric:
        path=paths[name]
        if name in pre:inventory.append(dict(file=name,bytes=pre[name][0],sha256=pre[name][1]))
        try:
            rel=path.relative_to(campaign)
            if (len(rel.parts)!=3 or rel.parts[0]!="stats_packs" or not rel.parts[1].startswith("pack_numeric")
                    or re.fullmatch(r"part_\d{4}\.numeric\.xz",rel.name) is None):
                raise ValueError("Numeric archive is outside its exact public namespace")
            if not path.is_file():raise ValueError("Numeric archive missing")
            size=path.stat().st_size
            if not 0<size<=min(pc.MAX_FILE_BYTES,codec.MAX_ARCHIVE):raise ValueError("Numeric archive exceeds individual limit")
            manifest=plain(path.with_name("MANIFEST.json"));manifest_name=manifest.relative_to(root).as_posix()
            if manifest_name not in published:raise ValueError("Numeric manifest must be explicitly selected")
            manifest_sha=sha(manifest)
            declaration=codec.load_manifest(manifest)
            pc._scan_text(manifest_name+"::records",json.dumps(declaration["records"],ensure_ascii=False),failures)
            rows=codec.inspect_archive(path)
            for member in rows:
                pc._scan_text(name+"::"+member["member_name"],member["csv_text"],failures)
            if sha(manifest)!=manifest_sha:raise ValueError("Numeric manifest changed during inspection")
            numeric_records.append(dict(file=name,manifest=manifest_name,manifest_sha256=manifest_sha,
                records=len(rows),format=codec.SCHEMA,byte_exact_csv_and_gzip_inspected=True))
        except (Exception,) as error:
            pc._failure(failures,name,"NUMERIC_ARCHIVE_INVALID",error_type=type(error).__name__)
    transport_records=[]
    for name in transported:
        path=paths[name]
        if name in pre:inventory.append(dict(file=name,bytes=pre[name][0],sha256=pre[name][1]))
        try:
            transport=load_transport()
            if path.parent!=campaign/TRANSPORT_DIRECTORY or re.fullmatch(r"part_\d{4}\.transport\.xz",path.name) is None:
                raise ValueError("Transport archive is outside the approved exact public namespace")
            if not path.is_file() or not 0<path.stat().st_size<=min(pc.MAX_FILE_BYTES,transport.MAX_ARCHIVE):
                raise ValueError("Transport archive missing or exceeds individual limit")
            manifest=plain(path.with_name("MANIFEST.json"));manifest_name=manifest.relative_to(root).as_posix()
            if manifest_name not in published:raise ValueError("Transport manifest must be explicitly selected")
            manifest_sha=sha(manifest);declaration=transport._load_manifest(manifest);binding=declaration["binding"]
            original_name=binding["original_manifest_repository_relative_path"]
            if original_name not in normal:raise ValueError("Original legacy manifest must be a selected normal file")
            original=plain(paths[original_name])
            if (not original.is_file() or original.stat().st_size!=binding["original_manifest_bytes"]
                    or sha(original)!=binding["original_manifest_sha256"]):
                raise ValueError("Published original legacy manifest differs from exact embedded binding")
            pc._scan_text(manifest_name+"::binding_and_members",json.dumps(dict(binding=binding,members=declaration["members"]),ensure_ascii=False),failures)
            rows=transport.inspect_archive(path)
            for member in rows:pc._scan_text(name+"::"+member["member_name"],member["csv_text"],failures)
            if sha(manifest)!=manifest_sha or sha(original)!=binding["original_manifest_sha256"]:
                raise ValueError("Transport or original manifest changed during inspection")
            transport_records.append(dict(file=name,manifest=manifest_name,manifest_sha256=manifest_sha,format=transport.SCHEMA,
                original_manifest=original_name,original_manifest_sha256=binding["original_manifest_sha256"],
                original_archive=binding["original_archive_repository_relative_path"],original_archive_sha256=binding["original_archive_sha256"],
                members=len(rows),original_archive_tar_members_and_XZ_exactly_rebuilt=True,
                original_archive_restore_required_before_original_stats_restore=True))
        except Exception as error:
            pc._failure(failures,name,"TRANSPORT_ARCHIVE_INVALID",error_type=type(error).__name__)
    # The original link checker is reused with no virtual paths at this final pass.
    for name in normal:
        path=paths[name]
        if not path.is_file() or path.suffix.lower() not in {".md",".svg"}:continue
        # These files were already fully parsed/scanned by the frozen inspector.
        for target in [x["target"] for x in base["links"] if x["file"]==name]:
            from_posix=posixpath.relpath(target,posixpath.dirname(name))
            pc._link(root,name,from_posix,published,tracked,set(),failures,[],case_targets)
    total=sum(item["bytes"] for item in inventory)
    if total>pc.MAX_BUNDLE_BYTES:pc._failure(failures,"<bundle>","BUNDLE_SIZE_LIMIT",bytes=total,limit=pc.MAX_BUNDLE_BYTES)
    for name,(size,digest) in pre.items():
        path=plain(paths[name])
        if not path.is_file() or path.stat().st_size!=size or sha(path)!=digest:
            pc._failure(failures,name,"INPUT_CHANGED_DURING_INSPECTION")
    if source_pins()!=before:raise ValueError("Source pins changed during inspection")
    return dict(status="FAIL" if failures else "PASS",schema=SCHEMA,files=sorted(inventory,key=lambda x:x["file"]),
        file_count=len(inventory),selected_bytes=total,public_increment_upper_bound_bytes=total,links=links,failures=failures,
        numeric_archives=numeric_records,transport_archives=transport_records,source_pins=before,explicit_files_only=True,automatic_discovery=False,
        frozen_legacy_inspector_used_for_nonnumeric_files=True,legacy_full_union_PASS_claimed=False,
        legacy_prepare_verify_compatibility_claimed=False,scientific_audit_or_model_rerun=False,files_written=0)


def check_index(campaign,files):
    pc,_=load_dependencies();return pc.check_index(pc._root(plain(campaign)),files)


def _bound_input(path,digest):
    path=plain(path)
    if not re.fullmatch(r"[0-9a-f]{64}",digest or "") or not path.is_file() or path.stat().st_size>20*1024**2 or sha(path)!=digest:
        raise ValueError("Explicit bounded input and exact SHA required")
    return path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign");parser.add_argument("--file-list");parser.add_argument("--file-list-sha")
    parser.add_argument("--selection");parser.add_argument("--selection-sha");parser.add_argument("--output")
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dict(status="METADATA_ONLY_NUMERIC_PUBLICATION_INSPECTOR",input_reads=0,files_written=0,
            numeric_archives_inspected=0,numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules,
            model_array_Task_or_child_operations=0,legacy_full_union_PASS_claimed=False)));return 0
    if not all((args.campaign,args.file_list,args.file_list_sha)):parser.error("Explicit campaign/list/SHA required")
    if bool(args.selection)!=bool(args.selection_sha):parser.error("Selection and exact SHA must be supplied together")
    listing=_bound_input(args.file_list,args.file_list_sha)
    selection=_bound_input(args.selection,args.selection_sha) if args.selection else None
    files=listing.read_text(encoding="utf-8").splitlines()
    if not files or any(not x for x in files):raise ValueError("Nonempty explicit file list required")
    output=plain(args.output) if args.output else None
    if output is not None and (output.exists() or output.is_relative_to(plain(args.campaign).parents[1])):
        raise ValueError("Fresh output outside the public repository required")
    result=inspect(Path(args.campaign),files)
    _bound_input(listing,args.file_list_sha)
    if selection is not None:_bound_input(selection,args.selection_sha)
    result.update(input_file_list_sha256=args.file_list_sha,selection_sha256=args.selection_sha)
    if output is not None:
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open("x",encoding="utf-8",newline="\n") as stream:json.dump(result,stream,ensure_ascii=False,indent=2);stream.write("\n")
    print(json.dumps(dict(status=result["status"],schema=SCHEMA,file_count=result["file_count"],selected_bytes=result["selected_bytes"],failures=result["failures"])))
    return 0 if result["status"]=="PASS" else 1


if __name__=="__main__":raise SystemExit(main())
