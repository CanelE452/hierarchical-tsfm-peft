"""Byte-exact numeric transport for public sufficient-statistic CSVs only.

This separate format does not extend or impersonate the frozen stats_pack v1.
Only --execute pack reads selected statistics. Restore verifies every original
CSV and gzip byte; no numeric rounding, float32 conversion or row removal occurs.
"""
import argparse
import csv
import importlib.util
import io
import json
import lzma
import math
from pathlib import Path, PurePosixPath
import re
import struct
import sys
import tarfile
import tempfile
import time
import zlib

HERE=Path(__file__).resolve().parent
CORE_SHA="0d8c4e4cc9881b70907acf7a154133a64cb582227c1df357521a7668e7fde3fd"
WRAPPER_SHA="0c48b6243ba43498bdc79c4633d97ec4d0741b651a1c8808e3f30e04919aea13"
SCHEMA="public_error_stats_numeric_pack_v1"
NUMERIC_SCHEMA="canonical_numeric_common9_v1"
MAGIC=b"STNUMC1\x00"
MAX_ARCHIVE=20*1024**2-1
MAX_MEMBER=64*1024**2
MAX_TAR=128*1024**2
MAX_OUTPUT=3*1024**3
MAX_MANIFEST=8*1024**2
MAX_ROWS=1000000
MAX_HEADER=1024**2
DEADLINE=1792222550
NUMBER_COLUMNS={2,4,5,6,7,8}


def load_core():
    path=HERE/"stats_pack.py"
    import hashlib
    if hashlib.sha256(path.read_bytes()).hexdigest()!=CORE_SHA:raise ValueError("Frozen original stats_pack source changed")
    spec=importlib.util.spec_from_file_location("numeric_pack_frozen_stats_core",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def _check(deadline):
    if deadline is not None and time.time()>=deadline:raise TimeoutError("Numeric codec wall limit reached; partial files preserved")


def _new(path,value):
    with Path(path).open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")


def _plain(path):
    path=Path(path).absolute()
    if any(p.is_symlink() or getattr(p,"is_junction",lambda:False)() for p in (path,*path.parents)):
        raise ValueError("Nonsymlink and nonjunction paths required")
    return path.resolve()


def _integer(value,maximum,*,positive=False):
    if isinstance(value,bool) or not isinstance(value,int) or value<0 or value>maximum or (positive and value==0):raise ValueError("Invalid bounded integer")
    return value


def _digest(value):
    if not isinstance(value,str) or re.fullmatch("[0-9a-f]{64}",value) is None:raise ValueError("SHA256 binding required")
    return value


def _shuffle_xor(raw,width):
    if len(raw)%width:raise ValueError("Fixed-width word alignment differs")
    result=bytearray(len(raw));previous=0
    for offset in range(0,len(raw),width):
        current=int.from_bytes(raw[offset:offset+width],"little")
        result[offset:offset+width]=(current^previous).to_bytes(width,"little");previous=current
    return b"".join(result[column::width] for column in range(width))


def _unshuffle_xor(planes,width):
    if len(planes)%width:raise ValueError("Plane alignment differs")
    count=len(planes)//width;interleaved=bytearray(len(planes));result=bytearray(len(planes))
    for column in range(width):interleaved[column::width]=planes[column*count:(column+1)*count]
    previous=0
    for offset in range(0,len(planes),width):
        current=int.from_bytes(interleaved[offset:offset+width],"little")^previous
        result[offset:offset+width]=current.to_bytes(width,"little");previous=current
    return bytes(result)


def _literal_tokens(tokens):
    data=bytearray()
    for token in tokens:
        if len(token)>MAX_MEMBER:raise ValueError("Literal token limit exceeded")
        data.extend(struct.pack("<I",len(token)));data.extend(token)
    return bytes(data)


def _read_literals(raw,count):
    values=[];offset=0
    for _ in range(count):
        if offset+4>len(raw):raise ValueError("Truncated literal length")
        length=struct.unpack_from("<I",raw,offset)[0];offset+=4
        if length>MAX_MEMBER or offset+length>len(raw):raise ValueError("Truncated or oversized literal token")
        token=raw[offset:offset+length];token.decode("utf-8");values.append(token);offset+=length
    if offset!=len(raw):raise ValueError("Unbound trailing literal bytes")
    return values


def _number_token(token):
    text=token.decode("utf-8")
    try:
        integer=int(text)
        if str(integer)==text and -(2**63)<=integer<2**63:return 1,struct.pack("<q",integer)
    except ValueError:pass
    try:
        value=float(text)
        if math.isfinite(value) and str(value)==text:return 2,struct.pack("<d",value)
    except (ValueError,OverflowError):pass
    return 0,b"\x00"*8


def _lines(raw):
    lines=raw.splitlines(keepends=True);bodies=[];endings=[]
    for index,line in enumerate(lines):
        if line.endswith(b"\r\n"):body,ending=line[:-2],2
        elif line.endswith(b"\n"):body,ending=line[:-1],1
        elif line.endswith(b"\r"):body,ending=line[:-1],3
        else:body,ending=line,0
        if (b"\r" in body or b"\n" in body or (ending==0 and index!=len(lines)-1)
                or body+{0:b"",1:b"\n",2:b"\r\n",3:b"\r"}[ending]!=line):return None
        bodies.append(body);endings.append(ending)
    return lines,bodies,endings


def encode_common9(raw,*,deadline=None):
    core=load_core()
    if not 0<len(raw)<=MAX_MEMBER:raise ValueError("Common9 CSV limit exceeded")
    core._csv_schema(raw,common=True)
    if b'"' in raw:return None
    parsed=_lines(raw)
    if parsed is None:return None
    lines,bodies,endings=parsed
    if b"".join(lines)!=raw or not 1<=len(lines)-1<=MAX_ROWS:return None
    rows=[body.split(b",") for body in bodies[1:]]
    if any(len(row)!=9 for row in rows):return None
    n=len(rows);descriptors=[];parts=[bytes(endings[1:])]
    for index,name in enumerate(core.COLUMNS[1:]):
        _check(deadline);tokens=[row[index] for row in rows]
        if index in NUMBER_COLUMNS:
            modes=bytearray();words=bytearray();literals=[]
            for row_index,token in enumerate(tokens):
                if row_index%4096==0:_check(deadline)
                mode,word=_number_token(token);modes.append(mode);words.extend(word)
                if mode==0:literals.append(token)
            literal=_literal_tokens(literals);planes=_shuffle_xor(words,8)
            descriptors.append(dict(name=name,kind="numeric",modes_bytes=n,planes_bytes=len(planes),literal_bytes=len(literal),literal_count=len(literals)))
            parts.extend((bytes(modes),planes,literal))
        else:
            unique=[];lookup={};indices=bytearray()
            for token in tokens:
                if token not in lookup:lookup[token]=len(unique);unique.append(token)
                indices.extend(struct.pack("<I",lookup[token]))
            dictionary=_literal_tokens(unique);planes=_shuffle_xor(indices,4)
            descriptors.append(dict(name=name,kind="dictionary",dictionary_bytes=len(dictionary),entries=len(unique),planes_bytes=len(planes)))
            parts.extend((dictionary,planes))
    header=dict(format=NUMERIC_SCHEMA,rows=n,header=lines[0].decode("utf-8"),line_endings_bytes=n,columns=descriptors,
        common_csv_bytes=len(raw),common_csv_sha256=core.sha(raw),encoding="Exact canonical int64/IEEE754 binary64; column XOR previous word then byte planes; all other tokens literal UTF8")
    encoded=json.dumps(header,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode("utf-8")
    payload=MAGIC+struct.pack("<I",len(encoded))+encoded+b"".join(parts)
    if len(encoded)>MAX_HEADER or len(payload)>MAX_MEMBER:return None
    if decode_common9(payload)!=raw:raise ValueError("Numeric parser roundtrip changed original CSV bytes")
    return payload


def decode_common9(payload):
    core=load_core()
    if not 12<len(payload)<=MAX_MEMBER or payload[:8]!=MAGIC:raise ValueError("Invalid bounded numeric member")
    length=struct.unpack_from("<I",payload,8)[0]
    if not 0<length<=MAX_HEADER or 12+length>len(payload):raise ValueError("Numeric header size limit exceeded")
    header=json.loads(payload[12:12+length].decode("utf-8"));n=_integer(header.get("rows"),MAX_ROWS,positive=True)
    if (header.get("format")!=NUMERIC_SCHEMA or header.get("line_endings_bytes")!=n or len(header.get("columns",[]))!=9
            or not isinstance(header.get("header"),str)):raise ValueError("Numeric member schema differs")
    _integer(header.get("common_csv_bytes"),MAX_MEMBER,positive=True);_digest(header.get("common_csv_sha256"))
    offset=12+length
    def take(size):
        nonlocal offset
        _integer(size,MAX_MEMBER)
        if offset+size>len(payload):raise ValueError("Truncated numeric column")
        value=payload[offset:offset+size];offset+=size;return value
    endings=take(n)
    if any(e not in (0,1,2,3) or (e==0 and i!=n-1) for i,e in enumerate(endings)):raise ValueError("Invalid row ending encoding")
    columns=[]
    for index,(name,info) in enumerate(zip(core.COLUMNS[1:],header["columns"])):
        if info.get("name")!=name:raise ValueError("Column order differs")
        if index in NUMBER_COLUMNS:
            if info.get("kind")!="numeric" or info.get("modes_bytes")!=n or info.get("planes_bytes")!=8*n:raise ValueError("Numeric column width differs")
            modes=take(n)
            if any(v not in (0,1,2) for v in modes):raise ValueError("Unknown numeric token mode")
            words=_unshuffle_xor(take(8*n),8)
            literal_count=_integer(info.get("literal_count"),n)
            if modes.count(0)!=literal_count:raise ValueError("Numeric literal count differs")
            literals=iter(_read_literals(take(info.get("literal_bytes")),literal_count));tokens=[]
            for row,mode in enumerate(modes):
                word=words[8*row:8*(row+1)]
                if mode==0:
                    if word!=b"\x00"*8:raise ValueError("Noncanonical literal placeholder")
                    token=next(literals)
                else:
                    value=struct.unpack("<q" if mode==1 else "<d",word)[0]
                    if mode==2 and not math.isfinite(value):raise ValueError("Nonfinite encoded numeric token")
                    token=str(value).encode("utf-8")
                    if _number_token(token)!=(mode,word):raise ValueError("Noncanonical numeric representation")
                tokens.append(token)
        else:
            if info.get("kind")!="dictionary" or info.get("planes_bytes")!=4*n:raise ValueError("Dictionary column width differs")
            entries=_integer(info.get("entries"),n,positive=True)
            dictionary=_read_literals(take(info.get("dictionary_bytes")),entries)
            if len(set(dictionary))!=entries:raise ValueError("Duplicate dictionary value")
            indices=_unshuffle_xor(take(4*n),4);tokens=[]
            for row in range(n):
                choice=struct.unpack_from("<I",indices,4*row)[0]
                if choice>=entries:raise ValueError("Dictionary index out of range")
                tokens.append(dictionary[choice])
        if any(b'"' in token or b"," in token or b"\n" in token or b"\r" in token for token in tokens):raise ValueError("Complex CSV tokens require opaque gzip storage")
        columns.append(tokens)
    if offset!=len(payload):raise ValueError("Unbound trailing numeric member bytes")
    result=bytearray(header["header"].encode("utf-8"));newline={0:b"",1:b"\n",2:b"\r\n",3:b"\r"}
    for row in range(n):
        result.extend(b",".join(column[row] for column in columns));result.extend(newline[endings[row]])
        if len(result)>header["common_csv_bytes"]:raise ValueError("Declared CSV expansion limit exceeded")
    raw=bytes(result)
    if len(raw)!=header["common_csv_bytes"] or core.sha(raw)!=header["common_csv_sha256"]:raise ValueError("Common9 CSV exact byte SHA differs")
    if core._csv_schema(raw,common=True)[1]!=n:raise ValueError("Decoded numeric row count differs")
    return raw


def encode_blob(blob,*,deadline=None):
    core=load_core()
    if not 0<len(blob)<=MAX_ARCHIVE:raise ValueError("Original gzip size limit exceeded")
    raw=core._gunzip(blob);_,rows=core._csv_schema(raw);recipe=core.find_gzip_recipe(blob,raw);payload=None;prefix=None
    lines=raw.splitlines(keepends=True)
    if recipe is not None and b'"' not in raw and len(lines)>1:
        prefix=lines[1].split(b",",1)[0]+b","
        if prefix.count(b",")==1 and all(line.startswith(prefix) for line in lines[1:]):
            common=lines[0].split(b",",1)[1]+b"".join(line[len(prefix):] for line in lines[1:])
            payload=encode_common9(common,deadline=deadline)
    mode="numeric_common9" if payload is not None else "opaque_gzip"
    if payload is None:payload=blob;prefix=None;recipe=None
    record=dict(storage_mode=mode,gzip_bytes=len(blob),gzip_sha256=core.sha(blob),csv_bytes=len(raw),csv_sha256=core.sha(raw),row_count=rows,
        gzip_recipe=recipe,run_id_prefix=prefix.decode("utf-8") if prefix is not None else None)
    if restore_blob(record,payload)!=blob:raise ValueError("Original CSV or gzip parser roundtrip changed bytes")
    return record,payload


def restore_blob(record,payload):
    core=load_core()
    _integer(record.get("csv_bytes"),MAX_MEMBER,positive=True)
    _integer(record.get("gzip_bytes"),MAX_ARCHIVE,positive=True)
    rows=_integer(record.get("row_count"),MAX_ROWS,positive=True)
    if record.get("storage_mode")=="opaque_gzip":
        blob=core._restore_blob(record,payload)
        if core._csv_schema(core._gunzip(blob))[1]!=rows:raise ValueError("Opaque row count differs")
        return blob
    if record.get("storage_mode")!="numeric_common9":raise ValueError("Unsupported numeric pack storage mode")
    prefix=record.get("run_id_prefix")
    if not isinstance(prefix,str) or not 1<len(prefix)<=record["csv_bytes"]:raise ValueError("Bounded UTF8 run ID prefix required")
    prefix=prefix.encode("utf-8")
    if (not 1<len(prefix)<=record["csv_bytes"] or not prefix.endswith(b",") or prefix.count(b",")!=1
            or b"\n" in prefix or b"\r" in prefix or b'"' in prefix):raise ValueError("Invalid bounded run ID prefix")
    common=decode_common9(payload)
    if core._csv_schema(common,common=True)[1]!=rows:raise ValueError("Decomposed common9 row count differs")
    expected_bytes=len(common)+len(b"run_id,")+rows*len(prefix)
    if expected_bytes>MAX_MEMBER or expected_bytes!=record["csv_bytes"]:raise ValueError("Original CSV expansion size differs before prefix join")
    projection=dict(record,storage_mode="csv_common9")
    raw=core._original_csv(projection,common)
    if core._csv_schema(raw)[1]!=rows:raise ValueError("Restored row count differs")
    blob=core.regenerate_gzip(raw,record["gzip_recipe"])
    if len(blob)!=record["gzip_bytes"] or core.sha(blob)!=record["gzip_sha256"]:raise ValueError("Original gzip exact byte SHA differs; reproducible zlib required")
    return blob


def _emit(group,directory,archives,deadline=None):
    core=load_core();_check(deadline)
    if not group or len({name for name,payload in group})!=len(group):raise ValueError("Unique nonempty numeric archive members required")
    if any(not 0<len(payload)<=MAX_MEMBER or re.fullmatch(r"members/[0-9a-f]{64}\.(?:numeric|csv\.gz)",name) is None for name,payload in group):raise ValueError("Unsafe numeric member name/size")
    raw=core._tar_bytes(group)
    if len(raw)>MAX_TAR:raise ValueError("Numeric expanded tar limit exceeded")
    packed=lzma.compress(raw,format=lzma.FORMAT_XZ,preset=6)
    if len(packed)>MAX_ARCHIVE:
        if len(group)==1:raise ValueError("Single numeric statistic exceeds compressed archive limit")
        mid=len(group)//2
        return {**_emit(group[:mid],directory,archives,deadline),**_emit(group[mid:],directory,archives,deadline)}
    filename="part_%04d.numeric.xz"%len(archives)
    with (Path(directory)/filename).open("xb") as stream:stream.write(packed)
    archives.append(dict(file=filename,bytes=len(packed),sha256=core.sha(packed),tar_bytes=len(raw),
        members=[dict(name=name,bytes=len(payload),sha256=core.sha(payload)) for name,payload in group]))
    return {name:filename for name,payload in group}


def load_manifest(path):
    core=load_core();path=_plain(path)
    if path.stat().st_size>MAX_MANIFEST:raise ValueError("Numeric manifest limit exceeded")
    value=json.loads(path.read_text(encoding="utf-8"))
    if (value.get("format")!=SCHEMA or value.get("original_stats_pack_sha256")!=CORE_SHA
            or not 1<=len(value.get("records",[]))<=10000 or not 1<=len(value.get("archives",[]))<=10000):raise ValueError("Unsupported numeric pack manifest")
    prefix=core._safe_name(value["campaign_repository_relative"]);archive_names=[];member_map={}
    for archive in value["archives"]:
        name=archive["file"]
        if re.fullmatch(r"part_\d{4}\.numeric\.xz",name) is None or name in archive_names:raise ValueError("Invalid or duplicate numeric archive filename")
        archive_names.append(name);_integer(archive["bytes"],MAX_ARCHIVE,positive=True);_integer(archive["tar_bytes"],MAX_TAR,positive=True);_digest(archive["sha256"])
        if not 1<=len(archive["members"])<=10000:raise ValueError("Bounded member list required")
        for info in archive["members"]:
            if re.fullmatch(r"members/[0-9a-f]{64}\.(?:numeric|csv\.gz)",info["name"]) is None or info["name"] in member_map:raise ValueError("Unsafe or duplicate numeric member")
            _integer(info["bytes"],MAX_MEMBER,positive=True);_digest(info["sha256"]);member_map[info["name"]]=(name,info)
            if info["name"].split("/")[1].split(".")[0]!=info["sha256"]:raise ValueError("Content-addressed member SHA differs")
    seen=set();bound=set()
    for rec in value["records"]:
        name=core._safe_name(rec["repository_relative_path"]);rel=PurePosixPath(name).relative_to(prefix)
        if (not rel.parts or rel.parts[0] not in {"predictions","ordered_predictions","route_predictions","route_replays","raw_calibration"}
                or rel.name!="PUBLIC_ERROR_STATS.csv.gz" or "T_final" in rel.parts or name.casefold() in seen):raise ValueError("Only explicit public V sufficient statistics are supported")
        seen.add(name.casefold());_integer(rec["gzip_bytes"],MAX_ARCHIVE,positive=True);_integer(rec["csv_bytes"],MAX_MEMBER,positive=True);_integer(rec["row_count"],MAX_ROWS,positive=True)
        _digest(rec["gzip_sha256"]);_digest(rec["csv_sha256"])
        mode=rec.get("storage_mode")
        if mode not in {"numeric_common9","opaque_gzip"} or rec.get("member") not in member_map or rec.get("archive")!=member_map[rec["member"]][0]:raise ValueError("Unbound numeric record/member")
        expected_suffix=".numeric" if mode=="numeric_common9" else ".csv.gz"
        if not rec["member"].endswith(expected_suffix):raise ValueError("Numeric member mode differs")
        bound.add(rec["member"])
    if bound!=set(member_map):raise ValueError("Archive contains unbound numeric members")
    return value


def _archive_members(manifest,path):
    core=load_core();path=_plain(path)
    found=[row for row in manifest["archives"] if row["file"]==path.name]
    if len(found)!=1:raise ValueError("Unique archive binding required")
    info=found[0]
    if path.stat().st_size!=info["bytes"] or core.sha_file(path)!=info["sha256"]:raise ValueError("Numeric archive byte SHA differs")
    raw=core.bounded_xz_decompress(path.read_bytes(),info["tar_bytes"])
    if len(raw)!=info["tar_bytes"]:raise ValueError("Numeric tar expansion differs")
    expected={row["name"]:row for row in info["members"]};result={}
    with tarfile.open(fileobj=io.BytesIO(raw),mode="r:") as tar:
        for member in tar:
            if member.name not in expected or member.name in result or not member.isfile() or member.pax_headers:raise ValueError("Unexpected nonregular numeric tar member")
            declared=expected[member.name]
            if not 0<member.size<=MAX_MEMBER or member.size!=declared["bytes"]:raise ValueError("Numeric tar member size differs")
            payload=tar.extractfile(member).read(member.size+1)
            if len(payload)!=member.size or core.sha(payload)!=declared["sha256"]:raise ValueError("Numeric member byte SHA differs")
            result[member.name]=payload
    if set(result)!=set(expected) or core._tar_bytes(list(result.items()))!=raw:raise ValueError("Noncanonical tar metadata/padding/trailing bytes")
    return result


def inspect_archive(path):
    path=_plain(path);manifest=load_manifest(path.with_name("MANIFEST.json"));members=_archive_members(manifest,path);core=load_core();out=[]
    for record in [rec for rec in manifest["records"] if rec["archive"]==path.name]:
        blob=restore_blob(record,members[record["member"]]);raw=core._gunzip(blob);text,rows=core._csv_schema(raw)
        if rows!=record["row_count"]:raise ValueError("Inspected row count differs")
        out.append(dict(member_name=record["member"],repository_relative_path=record["repository_relative_path"],csv_text=text,
            row_count=rows,csv_sha256=core.sha(raw),original_gzip_sha256=core.sha(blob),storage_mode=record["storage_mode"]))
    return out


def restore(manifest_path,destination,*,max_output_bytes=MAX_OUTPUT,deadline=None):
    core=load_core();manifest_path=_plain(manifest_path);manifest=load_manifest(manifest_path);destination=_plain(destination)
    _integer(max_output_bytes,MAX_OUTPUT,positive=True)
    if sum(rec["gzip_bytes"] for rec in manifest["records"])>max_output_bytes:raise ValueError("Restore output budget exceeded")
    for rec in manifest["records"]:
        target=core._inside(destination,rec["repository_relative_path"])
        if target.exists() and (not target.is_file() or target.stat().st_size!=rec["gzip_bytes"] or core.sha_file(target)!=rec["gzip_sha256"]):raise ValueError("Existing destination differs; no overwrite")
    restored=matching=0
    for info in manifest["archives"]:
        _check(deadline);members=_archive_members(manifest,core._inside(manifest_path.parent,info["file"]))
        for rec in [r for r in manifest["records"] if r["archive"]==info["file"]]:
            _check(deadline);blob=restore_blob(rec,members[rec["member"]]);target=core._inside(destination,rec["repository_relative_path"])
            if target.exists():
                if target.stat().st_size!=rec["gzip_bytes"] or core.sha_file(target)!=rec["gzip_sha256"]:raise ValueError("Existing destination changed")
                matching+=1;continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open("xb") as stream:stream.write(blob)
            restored+=1
    return dict(status="PASS",restored=restored,already_matching=matching,CSV_and_original_gzip_SHA_verified=True)


def metadata(repo,campaign,cache,namespace,selection,selection_sha):
    core=load_core();selection=_plain(selection)
    if _digest(selection_sha)!=core.sha_file(selection):raise ValueError("Exact selection JSON SHA required")
    path=HERE/"pack_postT_stats.py"
    if core.sha_file(path)!=WRAPPER_SHA:raise ValueError("Frozen namespace/39+9 metadata guard changed")
    spec=importlib.util.spec_from_file_location("numeric_pack_frozen_postT_guard",path)
    wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper)
    state=wrapper.metadata(_plain(repo),_plain(campaign),_plain(cache),namespace,selection)
    if state["selection_sha256"]!=selection_sha:raise ValueError("Selection changed during metadata check")
    state["wrapper"]=wrapper
    return state


def pack(repo,campaign,cache,namespace,selection,selection_sha,output,*,wall_seconds=1100):
    if isinstance(wall_seconds,bool) or not isinstance(wall_seconds,(int,float)) or not math.isfinite(wall_seconds) or not 0<wall_seconds<=1100:raise ValueError("Positive bounded1100s wall required")
    state=metadata(repo,campaign,cache,namespace,selection,selection_sha);core=state["core"];repo,campaign,cache=state["repo"],state["public"],state["cache"]
    output=_plain(output)
    if output.parent!=campaign/"stats_packs" or not output.name.startswith("pack_numeric") or output.exists():raise FileExistsError("New R/stats_packs/pack_numeric* output required")
    started=time.time();deadline=min(started+wall_seconds,DEADLINE);_check(deadline)
    source_sha=core.sha_file(__file__);source_bytes=sum(row["gzip_bytes"] for row in state["records"])
    state["wrapper"].storage_admission(state,source_bytes)
    output.mkdir(parents=True,exist_ok=False);records=[];archives=[];group=[];assignment={};seen=set()
    for selected in sorted(state["records"],key=lambda row:row["repository_relative_path"]):
        _check(deadline);blob=core._inside(repo,selected["repository_relative_path"]).read_bytes()
        if len(blob)!=selected["gzip_bytes"] or core.sha(blob)!=selected["gzip_sha256"]:raise ValueError("Selected original gzip changed")
        record,payload=encode_blob(blob,deadline=deadline)
        suffix=".numeric" if record["storage_mode"]=="numeric_common9" else ".csv.gz"
        member="members/"+core.sha(payload)+suffix
        if member not in seen:
            if group and sum(len(p) for _,p in group)+len(payload)>MAX_MEMBER:
                assignment.update(_emit(group,output,archives,deadline));group=[]
            group.append((member,payload));seen.add(member)
        records.append(dict(record,repository_relative_path=selected["repository_relative_path"],member=member))
    if group:assignment.update(_emit(group,output,archives,deadline))
    for record in records:record["archive"]=assignment[record["member"]]
    manifest=dict(format=SCHEMA,status="PENDING_RESTORE_VERIFICATION",original_stats_pack_sha256=CORE_SHA,namespace_guard_sha256=WRAPPER_SHA,
        numeric_source_sha256=source_sha,campaign_repository_relative=state["prefix"],namespace=namespace,selection_sha256=selection_sha,
        records=records,archives=archives,original_total_bytes=source_bytes,packed_total_bytes=sum(a["bytes"] for a in archives),
        encoding="Common9 columns: canonical int64/binary64 exact tokens, XOR previous word, byte planes, XZ preset6; literal UTF8 otherwise; complex CSV opaque gzip",
        python_version=".".join(map(str,sys.version_info[:3])),numeric_word_byte_order="little",
        zlib_compile_version=zlib.ZLIB_VERSION,zlib_runtime_version=zlib.ZLIB_RUNTIME_VERSION,original_gzip_SHA_requires_reproducible_zlib=True,
        original_files_modified=False,public_error_aggregate_statistics_read=True,raw_series_or_model_arrays_read=False,numeric_values_changed=False,
        full_rows_or_conditions_dropped=False,legacy_stats_pack_FORMAT_or_source_changed=False,original_publication_inspector_compatibility=False)
    draft=output/".MANIFEST_VERIFY.json";_new(draft,manifest)
    state["wrapper"].storage_admission(state,source_bytes)
    with tempfile.TemporaryDirectory(prefix="numeric_stats_verify_",dir=cache) as folder:
        temporary=_plain(folder)
        if not temporary.is_relative_to(cache) or temporary==cache:raise ValueError("Private verification temp must stay below campaign cache")
        first=restore(draft,temporary,deadline=deadline)
        if first["restored"]!=len(records):raise ValueError("Separate full restore missed a source file")
        for rec in records:
            blob=core._inside(temporary,rec["repository_relative_path"]).read_bytes();raw=core._gunzip(blob)
            if core.sha(blob)!=rec["gzip_sha256"] or core.sha(raw)!=rec["csv_sha256"] or core._csv_schema(raw)[1]!=rec["row_count"]:raise ValueError("Independent original gzip/CSV/row verification differs")
        repeated=restore(draft,temporary,deadline=deadline)
        if repeated["restored"]!=0 or repeated["already_matching"]!=len(records):raise ValueError("Repeated restore must only skip byte-exact originals")
    if temporary.exists():raise ValueError("Disposable verification tree not removed")
    after=metadata(repo,campaign,cache,namespace,selection,selection_sha)
    if after["pins"]!=state["pins"] or core.sha_file(__file__)!=source_sha:raise ValueError("Frozen source/metadata/selection changed during pack")
    for rec in records:
        if core.sha_file(core._inside(repo,rec["repository_relative_path"]))!=rec["gzip_sha256"]:raise ValueError("Original statistic gzip changed")
    _check(deadline);state["wrapper"].storage_admission(state,0)
    manifest.update(status="PASS",separate_full_restore_verified=True,repeated_restore_verified=True)
    manifest_path=output/"MANIFEST.json";_new(manifest_path,manifest)
    for archive in archives:inspect_archive(output/archive["file"])
    proof=dict(status="PASS",manifest_repository_relative_path=manifest_path.relative_to(repo).as_posix(),manifest_sha256=core.sha_file(manifest_path),
        numeric_source_sha256=source_sha,original_stats_pack_sha256=CORE_SHA,namespace_guard_sha256=WRAPPER_SHA,selection_sha256=selection_sha,
        verified_original_gzip_and_CSV_files=len(records),verified_CSV_rows=sum(r["row_count"] for r in records),separate_restore=first,repeated_restore=repeated,
        separate_temp_removed=True,original_source_and_statistics_unchanged=True,original_source_and_immutable_pins=state["pins"],
        original_gzip_bytes=source_bytes,numeric_archive_bytes=manifest["packed_total_bytes"],wall_seconds=time.time()-started,
        actual_raw_series_model_array_or_child_operations=0,new_training_inference_T_or_selection=False,publication_budget_or_existing_inspector_PASS_claimed=False)
    _new(output/"PROOF.json",proof)
    helper_name=(campaign/"stats_pack_numeric.py").relative_to(repo).as_posix();manifest_name=manifest_path.relative_to(repo).as_posix()
    readme=("# 공개 충분통계 숫자 토큰의 무손실 인코딩\n\n원 CSV의 모든 행·순서·token·header·개행과 원 gzip bytes를 보존했다. 부동소수점은 원문이 `str(float(token))`과 EXACT 같을 때만 IEEE754 binary64를 쓰며 float32·반올림을 사용하지 않는다. 정수는 canonical signed int64만 사용하고 다른 원문은 literal UTF8로 남겼다. 복잡한 quoted CSV 또는 재현 불가능한 gzip은 원 gzip 그대로 저장한다. 원 CSV/gzip SHA가 다르면 복원 PASS가 아니다.\n\n"
        "- [새 FORMAT manifest](MANIFEST.json)\n- [전수·별도 temp·반복 복원 검산](PROOF.json)\n\n저장소 루트에서 실행한다.\n\n```text\npython "+helper_name+" restore --manifest "+manifest_name+" --destination .\n```\n\n"
        "같은 원문 파일은 읽기만 하고 다른 기존 파일은 덮어쓰지 않는다. 동일 zlib 재현이 필요하며 SHA 검사를 생략하거나 공차로 대체하지 않는다. 새 FORMAT은 원 stats_pack/publication inspector를 변경하거나 위장하지 않는다. 최종 게시에는 별도 numeric inspect_archive 및 명시 allowlist 검사가 필요하다. 이 검산은 공개 SSE/SAE/count 집계의 저장 포맷 검산이며 새 학습·모델·시계열 원자료·정답 배열 평가가 아니다.\n")
    with (output/"README_KO.md").open("x",encoding="utf-8",newline="\n") as stream:stream.write(readme)
    additions=[(output/name).relative_to(repo).as_posix() for name in ("MANIFEST.json","PROOF.json","README_KO.md","ADD_LIST.txt")]+[(output/a["file"]).relative_to(repo).as_posix() for a in archives]
    with (output/"ADD_LIST.txt").open("x",encoding="utf-8",newline="\n") as stream:stream.write("\n".join(sorted(additions))+"\n")
    return proof


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest="command")
    p=sub.add_parser("pack")
    for name in ("repo","campaign","cache"):p.add_argument("--"+name+"-root",required=True)
    p.add_argument("--namespace",required=True,choices=("lag","fixed_pca","multik"));p.add_argument("--selection",required=True);p.add_argument("--selection-sha",required=True)
    p.add_argument("--output");p.add_argument("--wall-seconds",type=float,default=1100);p.add_argument("--execute",action="store_true")
    p=sub.add_parser("restore");p.add_argument("--manifest",required=True);p.add_argument("--destination",required=True);p.add_argument("--max-output-bytes",type=int,default=MAX_OUTPUT);p.add_argument("--deadline",type=float)
    p=sub.add_parser("inspect");p.add_argument("--archive",required=True)
    args=parser.parse_args(argv)
    if args.command=="pack":
        if args.execute:
            if not args.output:parser.error("Actual pack requires a fresh explicit output")
            result=pack(Path(args.repo_root),Path(args.campaign_root),Path(args.cache_root),args.namespace,Path(args.selection),args.selection_sha,Path(args.output),wall_seconds=args.wall_seconds)
            print(json.dumps(result));return 0
        state=metadata(Path(args.repo_root),Path(args.campaign_root),Path(args.cache_root),args.namespace,Path(args.selection),args.selection_sha)
        print(json.dumps(dict(status="METADATA_ONLY_NUMERIC_PACK",selected_files=len(state["records"]),statistics_content_read=False,compressions=0,files_written=0)));return 0
    if args.command=="restore":print(json.dumps(restore(Path(args.manifest),Path(args.destination),max_output_bytes=args.max_output_bytes,deadline=args.deadline)));return 0
    if args.command=="inspect":print(json.dumps(inspect_archive(Path(args.archive)),ensure_ascii=False));return 0
    print(json.dumps(dict(status="METADATA_ONLY_NUMERIC_CODEC",format=SCHEMA,files_written=0,statistics_content_read=False,compressions=0,
        original_core_sha256=CORE_SHA,namespace_guard_sha256=WRAPPER_SHA,numpy_imported="numpy" in sys.modules,torch_imported="torch" in sys.modules,
        model_Task_or_array_operations=0,legacy_FORMAT_changed=False,publication_inspector_compatibility=False)));return 0


if __name__=="__main__":raise SystemExit(main())
