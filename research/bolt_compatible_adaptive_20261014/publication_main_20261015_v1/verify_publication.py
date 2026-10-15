"""Read-only Git-index publication checks using only Python's standard library."""
from pathlib import Path,PurePosixPath
from urllib.parse import unquote,urlsplit
from collections import Counter
import argparse
import ast
import csv
import gzip
import hashlib
import io
import json
import re
import struct
import subprocess
import xml.etree.ElementTree as ET
import zlib

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BASE=HERE.parent
TEXT={".py",".md",".json",".jsonl",".txt",".csv",".svg",".toml",".yaml",".yml"}
FORBIDDEN={".npz",".npy",".pt",".pth",".safetensors",".ckpt",".bin",".pkl",".pickle",".parquet",".arrow",".h5",".hdf5",".sqlite",".db"}
SECRET={
    "private_key":re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "openai_key":re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{24,}\b"),
    "github_token":re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{24,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    "huggingface_token":re.compile(rb"\bhf_[A-Za-z0-9]{24,}\b"),
    "aws_access_key":re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "labelled_Korean_phone":re.compile(rb"(?:phone|telephone|\btel\b|\xec\xa0\x84\xed\x99\x94|\xec\x97\xb0\xeb\x9d\xbd)[^\r\n]{0,60}?(?<![A-Za-z0-9])01[016789][ -]?\d{3,4}[ -]?\d{4}(?![A-Za-z0-9])",re.I),
}


def git(*args): return subprocess.check_output(["git",*args],cwd=ROOT)
def sha(data): return hashlib.sha256(data).hexdigest()
def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()


def load_json(data):
    def invalid(value): raise ValueError("NONSTANDARD_JSON_NUMBER:"+value)
    return json.loads(data.decode("utf-8-sig"),parse_constant=invalid)


def markdown_links(text):
    text=re.sub(r"(?ms)^\s*(```|~~~).*?^\s*\1[^\n]*$","",text)
    links=[]
    for match in re.finditer(r"!?\[[^\]\n]*\]\(\s*",text):
        start=match.end(); end=start; level=1; escaped=False
        if text[start:start+1]=="<":
            end=text.find(">",start+1)
            if end>=0: links.append(text[start+1:end])
            continue
        while end<len(text) and level:
            char=text[end]
            if escaped: escaped=False
            elif char=="\\": escaped=True
            elif char=="(": level+=1
            elif char==")": level-=1
            if level: end+=1
        if level==0: links.append(re.split(r"\s+[\"']",text[start:end].strip(),maxsplit=1)[0])
    definitions={m.group(1).casefold():m.group(2).strip("<>") for m in re.finditer(r"(?m)^\s*\[([^\]]+)\]:\s*(<[^>]+>|\S+)",text)}
    for match in re.finditer(r"!?\[([^\]\n]+)\]\[([^\]\n]*)\]",text):
        key=(match.group(2) or match.group(1)).casefold()
        if key in definitions: links.append(definitions[key])
    return links


def resolved_link(source,target):
    if not target or target.startswith("#"): return None
    if urlsplit(target).scheme in ("http","https","mailto","data"): return None
    path=unquote(target.split("#",1)[0].split("?",1)[0]).replace("\\","/")
    if not path: return None
    if path.startswith(("/","file:")) or re.match(r"^[A-Za-z]:",path): raise ValueError("LOCAL_ABSOLUTE_MARKDOWN_LINK")
    destination=(source.parent/path).resolve()
    if not destination.is_relative_to(ROOT): raise ValueError("MARKDOWN_LINK_ESCAPES_REPOSITORY")
    return destination.relative_to(ROOT).as_posix()


def check_png(data):
    if not data.startswith(b"\x89PNG\r\n\x1a\n"): raise ValueError("PNG_SIGNATURE")
    offset=8; width=height=depth=color=interlace=None; decoded=0; inflater=zlib.decompressobj(); idats=0; ended=False
    while offset<len(data):
        if offset+12>len(data): raise ValueError("PNG_TRUNCATED_CHUNK")
        size=struct.unpack(">I",data[offset:offset+4])[0]; kind=data[offset+4:offset+8]; end=offset+12+size
        if end>len(data): raise ValueError("PNG_TRUNCATED_DATA")
        value=data[offset+8:offset+8+size]; crc=struct.unpack(">I",data[offset+8+size:end])[0]
        if zlib.crc32(kind+value)&0xffffffff!=crc: raise ValueError("PNG_CRC")
        if kind==b"IHDR":
            if offset!=8 or size!=13: raise ValueError("PNG_IHDR")
            width,height,depth,color,compression,filtering,interlace=struct.unpack(">IIBBBBB",value)
            if not width or not height or color not in (0,2,3,4,6) or depth not in (1,2,4,8,16) or compression or filtering or interlace not in (0,1): raise ValueError("PNG_HEADER_VALUES")
        elif kind==b"IDAT": decoded+=len(inflater.decompress(value)); idats+=1
        elif kind==b"IEND":
            if size or end!=len(data): raise ValueError("PNG_IEND_OR_TRAILING_BYTES")
            ended=True
        offset=end
    decoded+=len(inflater.flush())
    if width is None or not ended or not idats or not inflater.eof or inflater.unused_data: raise ValueError("PNG_INCOMPLETE_ZLIB_OR_STRUCTURE")
    if interlace==0:
        channels={0:1,2:3,3:1,4:2,6:4}[color]; expected=height*(1+(width*channels*depth+7)//8)
        if decoded!=expected: raise ValueError("PNG_DECODED_SCANLINE_BYTES")
    return {"width":width,"height":height}


def indexed_blobs(entries,paths):
    process=subprocess.Popen(["git","cat-file","--batch"],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    try:
        for path in paths:
            oid=entries[path][1]; process.stdin.write((oid+"\n").encode("ascii")); process.stdin.flush()
            header=process.stdout.readline().split()
            if len(header)!=3 or header[0].decode()!=oid or header[1]!=b"blob": raise ValueError("GIT_INDEX_BLOB_HEADER:"+path)
            size=int(header[2]); data=process.stdout.read(size)
            if len(data)!=size or process.stdout.read(1)!=b"\n": raise ValueError("GIT_INDEX_BLOB_LENGTH:"+path)
            yield path,data
    finally:
        process.stdin.close(); process.stdout.close(); process.wait()


def verify(manifest_path,output,max_bytes):
    errors=[]; warnings=[]; counters=Counter(); sensitive=[]
    def error(kind,path,detail=None): errors.append({"kind":kind,"path":str(path),**({"detail":detail} if detail else {})})
    entries={}
    for row in git("ls-files","--stage","-z").split(b"\0"):
        if not row: continue
        meta,name=row.split(b"\t",1); mode,oid,stage=meta.decode().split(); path=name.decode("utf-8")
        if stage!="0": error("UNMERGED_GIT_INDEX",path)
        entries[path]=(mode,oid)
    staged={p.decode("utf-8") for p in git("diff","--cached","--name-only","--diff-filter=ACMR","-z").split(b"\0") if p}
    deleted=[p.decode("utf-8") for p in git("diff","--cached","--name-only","--diff-filter=D","-z").split(b"\0") if p]
    for path in deleted: error("STAGED_DELETION",path)
    prefix=BASE.relative_to(ROOT).as_posix()+"/"; pub_prefix=HERE.relative_to(ROOT).as_posix()+"/"
    allowed={"README.md","_docs/history/2026-10-15.md"}
    for path in staged:
        if not path.startswith(prefix) and path not in allowed: error("UNRELATED_STAGED_PATH",path)
    manifest=load_json(Path(manifest_path).read_bytes()); records=manifest["files"]; record_map={r["repo_path"]:r for r in records}
    if len(record_map)!=len(records): error("DUPLICATE_MANIFEST_RECORD",manifest_path)
    for path,row in record_map.items():
        local=(ROOT/path).resolve()
        if not local.is_relative_to(ROOT) or not local.is_file(): error("MANIFEST_LOCAL_FILE_MISSING_OR_OUTSIDE",path); continue
        if local.stat().st_size!=row["bytes"] or file_sha(local)!=row["sha256"]: error("MANIFEST_LOCAL_BYTE_SHA_MISMATCH",path)
        if path not in entries: error("MANIFEST_FILE_NOT_IN_GIT_INDEX",path)
    manifest_rel=Path(manifest_path).resolve().relative_to(ROOT).as_posix(); output=Path(output).resolve()
    exclusions={manifest_rel}
    if output.is_relative_to(ROOT): exclusions.add(output.relative_to(ROOT).as_posix())
    exclusions.update(manifest.get("self_hash_exclusions",[]))
    for path in staged-set(record_map)-exclusions: error("STAGED_PATH_MISSING_FROM_PUBLICATION_MANIFEST",path)
    paths=sorted((set(record_map)|{p for p in entries if p.startswith(pub_prefix)})&set(entries))
    cached_MD={}; index_hashes={}; format_counts=Counter(); largest=[]
    for path,data in indexed_blobs(entries,paths):
        counters["indexed_files_checked"]+=1; counters["indexed_payload_bytes"]+=len(data); index_hashes[path]=sha(data); largest.append((len(data),path))
        suffix=PurePosixPath(path).suffix.lower()
        if entries[path][0] not in ("100644","100755"): error("NON_REGULAR_GIT_MODE",path)
        if suffix in FORBIDDEN or PurePosixPath(path).name in (".env","credentials.json","id_rsa","id_ed25519"): error("FORBIDDEN_BINARY_DATA_OR_CREDENTIAL_FILE",path)
        if len(data)>max_bytes: error("FILE_ABOVE_PUBLICATION_SIZE_LIMIT",path,str(len(data)))
        if path in record_map and (len(data)!=record_map[path]["bytes"] or sha(data)!=record_map[path]["sha256"]): error("GIT_INDEX_DIFFERS_FROM_MANIFEST_LOCAL_BYTES",path)
        if suffix in TEXT:
            found={name:len(pattern.findall(data)) for name,pattern in SECRET.items()}; found={name:count for name,count in found.items() if count}
            if found: sensitive.append({"path":path,"patterns":found}); error("SENSITIVE_PATTERN_REQUIRES_REVIEW",path)
        try:
            if suffix==".py": ast.parse(data.decode("utf-8-sig"),filename=path); format_counts["Python_AST"]+=1
            elif suffix==".json": load_json(data); format_counts["JSON"]+=1
            elif suffix==".jsonl":
                for line in data.splitlines():
                    if line.strip(): load_json(line); counters["JSONL_records_checked"]+=1
                format_counts["JSONL"]+=1
            elif suffix==".csv":
                rows=csv.reader(io.StringIO(data.decode("utf-8-sig"),newline=""),strict=True)
                for row in rows: counters["CSV_rows_checked"]+=1
                format_counts["CSV"]+=1
            elif suffix==".png": check_png(data); format_counts["PNG"]+=1
            elif suffix==".svg":
                if ET.fromstring(data).tag.split("}")[-1]!="svg": raise ValueError("SVG_ROOT_TAG")
                format_counts["SVG"]+=1
            elif suffix==".md": cached_MD[path]=data.decode("utf-8-sig"); format_counts["Markdown"]+=1
        except Exception as exc: error("FORMAT_PARSE_FAILURE",path,type(exc).__name__+":"+str(exc))
    archive_path=HERE/"EVIDENCE_ARCHIVES.json"; archives=load_json(archive_path.read_bytes())
    if archives["status"]!="PASS" or archives["original_files_modified"] is not False: error("EVIDENCE_ARCHIVE_STATUS",archive_path)
    for row in archives["records"]:
        published,original=row["published_repo_path"],row["original_repo_path"]; archive=(ROOT/published).resolve(); raw=(ROOT/original).resolve()
        try:
            if not archive.is_relative_to(ROOT) or not raw.is_relative_to(ROOT): raise ValueError("ARCHIVE_PATH_OUTSIDE_REPOSITORY")
            if published not in entries or index_hashes.get(published)!=row["archive_sha256"]: raise ValueError("ARCHIVE_INDEX_SHA")
            if archive.stat().st_size!=row["archive_bytes"] or file_sha(archive)!=row["archive_sha256"] or raw.stat().st_size!=row["original_bytes"] or file_sha(raw)!=row["original_sha256"]: raise ValueError("ARCHIVE_OR_ORIGINAL_LOCAL_SHA")
            digest=hashlib.sha256(); size=0
            with gzip.open(archive,"rb") as handle:
                for block in iter(lambda:handle.read(1024*1024),b""): digest.update(block); size+=len(block)
            if digest.hexdigest()!=row["original_sha256"] or size!=row["original_bytes"]: raise ValueError("GZIP_DECOMPRESSED_BYTE_IDENTITY")
            with gzip.open(archive,"rb") as handle: uncompressed=handle.read()
            found={name:len(pattern.findall(uncompressed)) for name,pattern in SECRET.items()}; found={name:count for name,count in found.items() if count}
            if found: sensitive.append({"path":published,"patterns":found}); error("SENSITIVE_ARCHIVE_PATTERN_REQUIRES_REVIEW",published)
            load_json(uncompressed); del uncompressed; counters["gzip_JSON_archives_checked"]+=1; counters["decompressed_original_bytes_checked"]+=size
        except Exception as exc: error("ARCHIVE_VERIFICATION_FAILURE",published,type(exc).__name__+":"+str(exc))
    publication_links=0; historical_issues=[]
    for path,text in cached_MD.items():
        is_copy=path.startswith(pub_prefix)
        for target in markdown_links(text):
            try:
                resolved=resolved_link(ROOT/path,target)
                if resolved is None: continue
                issue=None
                if not (ROOT/resolved).exists(): issue="LOCAL_TARGET_MISSING"
                elif resolved not in entries and not any(p.startswith(resolved.rstrip("/")+"/") for p in entries): issue="TARGET_NOT_IN_GIT_INDEX"
                if is_copy:
                    publication_links+=1
                    if issue: error("PUBLICATION_MARKDOWN_"+issue,path,resolved)
                elif issue: historical_issues.append({"source":path,"target":resolved,"issue":issue})
            except Exception as exc:
                if is_copy: error("PUBLICATION_MARKDOWN_UNUSABLE_LINK",path,str(exc))
                else: historical_issues.append({"source":path,"issue":str(exc)})
    if historical_issues: warnings.append({"kind":"PRESERVED_ORIGINAL_MD_LINK_ISSUES","count":len(historical_issues),"scope":"Historical immutable originals are not the zero-broken-link publication copies","issues":historical_issues})
    source_rel=Path(__file__).resolve().relative_to(ROOT).as_posix()
    if source_rel not in index_hashes or index_hashes[source_rel]!=file_sha(__file__): error("RUNNING_VERIFIER_SOURCE_NOT_EXACTLY_INDEXED",source_rel)
    result={"status":"PASS" if not errors else "FAILED","scope":"Publication filesystem, Git index, formats and links only; no scientific arrays/model/F/fit/inference replay","manifest":{"path":str(Path(manifest_path).resolve()),"sha256":file_sha(manifest_path),"records_checked":len(records)},"Git_index_inventory_sha256":sha(json.dumps(entries,sort_keys=True).encode()),"Git_index_manifest_paths_checked":len(paths),"staged_paths_checked":len(staged),"counts":dict(counters),"format_counts":dict(format_counts),"publication_copy_MD_files_checked":sum(p.startswith(pub_prefix) for p in cached_MD),"publication_relative_links_checked":publication_links,"source":{"path":source_rel,"sha256":file_sha(__file__)},"largest_indexed_files":[{"path":path,"bytes":size} for size,path in sorted(largest,reverse=True)[:12]],"self_hash_exclusions":sorted(exclusions),"sensitive_findings":sensitive,"errors":errors,"warnings":warnings,"scientific_reexecution":False,"protected_TEST_or_binary_arrays_opened":False}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8",newline="\n") as handle: json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False); handle.write("\n")
    print(json.dumps({"status":result["status"],"files":len(paths),"staged":len(staged),"publication_links":publication_links,"errors":errors[:20],"warning_groups":len(warnings),"receipt":str(output)},ensure_ascii=False))
    return 0 if not errors else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--manifest",type=Path,default=HERE/"FILES.json"); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--max-file-mib",type=int,default=10)
    args=parser.parse_args(); raise SystemExit(verify(args.manifest,args.output,args.max_file_mib*1024*1024))
