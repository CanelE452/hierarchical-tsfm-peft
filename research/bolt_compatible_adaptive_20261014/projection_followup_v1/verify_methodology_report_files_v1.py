"""Limited report file/row/link checks; no scientific metric or array replay."""
from pathlib import Path
import argparse
import csv
import json
import re
from urllib.parse import unquote,urlsplit
import xml.etree.ElementTree as ET

from make_linear_e_adaptation_report_v2 import require,record,read,sha

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
METADATA_SUFFIXES={".json",".jsonl",".md",".py"}
PAYLOAD_SUFFIXES={".json",".md",".csv",".png",".svg"}


def local_path(value,base=None,*,directory=False):
    path=Path(value);path=path if path.is_absolute() else (base or ROOT)/path
    require(not any(item.is_symlink() for item in (path,*path.parents)),"SYMLINK_PATH_REJECTED")
    resolved=path.resolve();require(resolved.is_relative_to(ROOT.resolve()),"LOCAL_PATH_CONTAINMENT")
    require(resolved.is_dir() if directory else resolved.is_file(),"EXISTING_LOCAL_DIRECTORY" if directory else "EXISTING_LOCAL_FILE:"+str(path));return resolved


def metadata_bindings(value):
    if isinstance(value,dict):
        if "path" in value and "sha256" in value:yield value
        else:
            for item in value.values():yield from metadata_bindings(item)
    elif isinstance(value,list):
        for item in value:yield from metadata_bindings(item)


def markdown_targets(text):
    text=re.sub(r"^\s*```[^\n]*\n.*?^\s*```\s*$","",text,flags=re.MULTILINE|re.DOTALL)
    inline=re.findall(r"!?\[[^\]]*\]\(\s*(<[^>]+>|[^)\s]+)(?:\s+[\"'][^\"']*[\"'])?\s*\)",text)
    definitions=re.findall(r"^\s{0,3}\[[^\]]+\]:\s*(<[^>]+>|\S+)",text,flags=re.MULTILINE)
    return [target[1:-1] if target.startswith("<") else target for target in inline+definitions]


def verify(report_directory,root_viewed_png_count=None,check=lambda:None):
    folder=local_path(report_directory,directory=True);manifest_path=local_path(folder/"MANIFEST.json");manifest_binding=record(manifest_path);manifest=read(manifest_path)
    require(manifest["status"]=="COMPLETE" and isinstance(manifest["files"],list) and manifest["files"],"COMPLETE_REPORT_MANIFEST_REQUIRED");payload={};guarded=[manifest_binding]
    for entry in manifest["files"]:
        check();require(not Path(entry["path"]).is_absolute(),"RELATIVE_MANIFEST_PAYLOAD_PATH");path=local_path(entry["path"],folder)
        require(path.is_relative_to(folder) and path.relative_to(folder).as_posix()==entry["path"].replace("\\","/") and path!=manifest_path,"REPORT_PAYLOAD_CONTAINMENT")
        require(path not in payload and path.suffix.lower() in PAYLOAD_SUFFIXES,"UNIQUE_REPORT_METADATA_PAYLOAD");require(sha(path)==entry["sha256"],"PAYLOAD_SHA:"+entry["path"]);require(type(entry["bytes"]) is int and path.stat().st_size==entry["bytes"],"PAYLOAD_SIZE:"+entry["path"]);payload[path]=entry;guarded.append({**entry,"path":str(path)})
    entries=list(folder.rglob("*"));require(not any(path.is_symlink() for path in entries),"SYMLINK_REPORT_ENTRY");require({path.resolve() for path in entries if path.is_file() and path!=manifest_path}==set(payload),"EXACT_MANIFEST_PAYLOAD_SET")
    source_bindings=list(metadata_bindings(manifest["source"]));require(bool(source_bindings),"BOUND_REPORT_SOURCE_REQUIRED");bindings=source_bindings+list(metadata_bindings(manifest["inputs"]))
    for binding in bindings:
        check();path=local_path(binding["path"]);require(path.suffix.lower() in METADATA_SUFFIXES,"BOUND_METADATA_ONLY_NO_ARRAY_MODEL_READ");require(sha(path)==binding["sha256"],"BOUND_METADATA_SHA:"+str(path))
        if "bytes" in binding:require(type(binding["bytes"]) is int and path.stat().st_size==binding["bytes"],"BOUND_METADATA_SIZE")
        guarded.append({**binding,"path":str(path)})
    csv_paths={path for path in payload if path.suffix.lower()==".csv"};counted={};table_paths=set()
    for name,expected in manifest["table_row_counts"].items():
        require(type(expected) is int and expected>=0,"NONNEGATIVE_CSV_ROW_COUNT");matches=[path for path in csv_paths if path.relative_to(folder).as_posix()==name] if name.lower().endswith(".csv") else [path for path in csv_paths if path.stem==name]
        require(len(matches)==1,"UNIQUE_MANIFEST_CSV_TABLE:"+name);path=matches[0]
        with path.open(encoding="utf-8-sig",newline="") as handle:actual=sum(1 for _ in csv.DictReader(handle))
        require(actual==expected,"CSV_ROW_COUNT:"+name);counted[name]=actual;table_paths.add(path)
    require(table_paths==csv_paths,"ALL_PAYLOAD_CSV_COUNTS_REGISTERED")
    local_links=external_links=0;md_paths=[path for path in payload if path.suffix.lower()==".md"]
    require(bool(md_paths),"MARKDOWN_REPORT_PAYLOAD_REQUIRED")
    for path in md_paths:
        for target in markdown_targets(path.read_text(encoding="utf-8-sig")):
            parsed=urlsplit(target)
            if parsed.scheme.lower() in ("https","http"):external_links+=1;continue
            if not parsed.path and parsed.fragment:continue
            require(not parsed.scheme and not parsed.netloc,"LOCAL_OR_HTTPS_MARKDOWN_DESTINATION");local_path(unquote(parsed.path),path.parent);local_links+=1
    png_paths=[path for path in payload if path.suffix.lower()==".png"];svg_paths=[path for path in payload if path.suffix.lower()==".svg"]
    for field,paths in (("PNG_count",png_paths),("SVG_count",svg_paths)):require(type(manifest[field]) is int and manifest[field]==len(paths),"EXACT_"+field)
    for path in png_paths:
        with path.open("rb") as handle:require(handle.read(8)==b"\x89PNG\r\n\x1a\n","PNG_FILE_SIGNATURE")
    for path in svg_paths:require(ET.parse(path).getroot().tag.split("}")[-1]=="svg","SVG_XML_ROOT")
    if root_viewed_png_count is not None:require(type(root_viewed_png_count) is int and root_viewed_png_count==len(png_paths),"ROOT_VIEWED_PNG_COUNT")
    for binding in guarded:check();require(sha(binding["path"])==binding["sha256"],"CHECKED_FILE_UNCHANGED")
    return {"status":"PASS","report_directory":str(folder),"manifest":manifest_binding,"report_source":manifest["source"],"bound_inputs":manifest["inputs"],"payload_files_checked":len(payload),"bound_metadata_files_checked":len(bindings),"CSV_table_row_counts_checked":counted,"local_links_checked":local_links,"external_HTTP_links_ignored":external_links,"Markdown_files_checked":len(md_paths),"PNG_count":len(png_paths),"SVG_count":len(svg_paths),"root_viewed_png_count_declared":root_viewed_png_count,"root_view_scope":"Optional caller declaration only; this checker does not inspect image readability.","visual_review_performed_by_checker":False,"scientific_metrics_recomputed":False,"CSV_primitives_independently_reconstructed":False,"scientific_NPZ_model_F_GPU_or_timer_replayed":False,"protected_TEST_accessed":False,"scope":"File SHA/size, existing metadata source/input bindings, CSV row counts, local Markdown destinations and image formats/counts only. Not a scientific audit."}


def execute(report_directory,output,root_viewed_png_count):
    import continuation_runtime_v1 as runtime
    folder=local_path(report_directory,directory=True);destination=Path(output);destination=destination if destination.is_absolute() else ROOT/destination;local_path(destination.parent,directory=True);require(not destination.is_symlink(),"SYMLINK_OUTPUT_REJECTED");destination=destination.resolve();require(destination.is_relative_to(ROOT.resolve()) and destination.suffix.lower()==".json" and not destination.exists() and not destination.is_relative_to(folder),"FRESH_JSON_OUTPUT_OUTSIDE_REPORT_REQUIRED")
    meter=runtime.Meter("methodology_report_files_"+destination.stem,gpu=False);failure=None;source=record(__file__);result={"status":"RUNNING","report_directory":str(folder)}
    try:meter.protected_check();result=verify(folder,root_viewed_png_count,meter.check);require(record(__file__)==source,"VERIFIER_SOURCE_UNCHANGED")
    except Exception as error:failure={"type":type(error).__name__,"message":str(error)};result.update(status="FAILED_PRESERVED",failure=failure)
    finally:
        try:meter.protected_check()
        except Exception as error:failure={"type":type(error).__name__,"message":str(error)};result.update(status="FAILED_PRESERVED",failure=failure)
        result["verifier_source"]=source;result["resource_terminal"]=meter.close(result["status"],failure);runtime.write_new(destination,result)
    print(json.dumps({"status":result["status"],"output":str(destination),"failure":failure}),flush=True);return 0 if result["status"]=="PASS" else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--report-directory",required=True);parser.add_argument("--output",required=True);parser.add_argument("--root-viewed-png-count",type=int);args=parser.parse_args();raise SystemExit(execute(args.report_directory,args.output,args.root_viewed_png_count))
