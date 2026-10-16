"""Materialize only Jena's pre-T prefix; default CLI reads no inputs.

C/F-order .npy primitive arrays are streamed. For 2D Fortran storage each
column's requested row segment alone is converted; intervening tail bytes
are read/discarded opaquely. ZIP decompression may buffer additional bytes;
this is a numeric-parse boundary, not a claim of zero physical tail I/O.
"""
import argparse
import ast
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import zipfile

SOURCE_REL=".cache/tsfm_peft_practical_controls_v7_20260928/data/jena_trainval.npz"
SOURCE_SHA="29cfae52d7cd7128e551ef85da11138ed4d3f7b3ccf27af359c4fc1dedf4a933"
SOURCE_BYTES=1561064
CONTRACT_REL="research/level_matched_learned_ed_v18_20261006/data_contract_jena.json"
CONTRACT_SHA="678bd4cbd63265290c0eeed8a78b09545a05219b666fd3b818682ce385906820"
SOURCE_ROWS,PERMITTED_ROWS,PREFIX_ROWS,CHANNELS=35135,20965,16772,21
BOUNDS=dict(A_fit=[0,10482],A_stop=[10482,12579],M_cal=[12579,14675],V_select=[14675,16772],T_final=[16772,20965])
MAX_HEADER=65536
MAX_ARRAY_BYTES=64*1024**2
MAX_METADATA_BYTES=65536


def sha(path):
    value=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):value.update(block)
    return value.hexdigest()


def plain(path):
    path=Path(path).absolute()
    if any(p.is_symlink() or getattr(p,"is_junction",lambda:False)() for p in (path,*path.parents)):raise ValueError("Nonsymlink source paths required")
    return path.resolve()


def read_exact(stream,count):
    if not isinstance(count,int) or isinstance(count,bool) or not 0<=count<=MAX_ARRAY_BYTES:raise ValueError("Bounded nonnegative byte count required")
    pieces=[];remaining=count
    while remaining:
        part=stream.read(min(remaining,65536))
        if not part:raise ValueError("Truncated NPY header or requested prefix")
        pieces.append(part);remaining-=len(part)
    return b"".join(pieces)


def primitive_size(descr):
    if not isinstance(descr,str):raise ValueError("Object/structured/subarray dtype forbidden")
    match=re.fullmatch(r"[<>=|]([biufSUMm])(\d+)(?:\[([a-zA-Z0-9]+)\])?",descr)
    if match is None:raise ValueError("Primitive fixed-width dtype required")
    kind,width,unit=match.groups();width=int(width)
    allowed={"b":{1},"i":{1,2,4,8},"u":{1,2,4,8},"f":{2,4,8},"M":{8},"m":{8}}
    if kind in allowed and width not in allowed[kind]:raise ValueError("Unsupported primitive width")
    if kind in {"S","U"} and not 1<=width<=4096:raise ValueError("Bounded string dtype required")
    if (kind in {"M","m"})!=(unit is not None):raise ValueError("Datetime unit descriptor differs")
    return width*(4 if kind=="U" else 1)


def read_header(stream):
    magic=read_exact(stream,8)
    if magic[:6]!=b"\x93NUMPY" or magic[6:8] not in (b"\x01\x00",b"\x02\x00",b"\x03\x00"):raise ValueError("Supported NPY version required")
    size_width=2 if magic[6]==1 else 4
    count=int.from_bytes(read_exact(stream,size_width),"little")
    if not 0<count<=MAX_HEADER:raise ValueError("NPY header exceeds64KiB bound")
    header=ast.literal_eval(read_exact(stream,count).decode("utf-8" if magic[6]==3 else "latin1").strip())
    if not isinstance(header,dict) or set(header)!={"descr","fortran_order","shape"}:raise ValueError("Exact primitive NPY header required")
    shape=header["shape"]
    if (not isinstance(shape,tuple) or len(shape)>2 or any(isinstance(n,bool) or not isinstance(n,int) or n<0 for n in shape)
            or not isinstance(header["fortran_order"],bool)):raise ValueError("Bounded scalar/vector/matrix shape and explicit order required")
    itemsize=primitive_size(header["descr"]);payload=math.prod(shape)*itemsize
    if payload>MAX_ARRAY_BYTES:raise ValueError("Declared NPY expansion exceeds bound")
    return dict(descr=header["descr"],shape=list(shape),fortran_order=header["fortran_order"],itemsize=itemsize,
        header_total_bytes=8+size_width+count,payload_bytes=payload)


def decode_numeric(blob,descr):
    import numpy as np
    dtype=np.dtype(descr)
    if dtype.hasobject or dtype.fields or dtype.subdtype:raise ValueError("Primitive numeric/string dtype only")
    return np.frombuffer(blob,dtype=dtype).copy()


def read_member(archive,name,rows=None):
    info=archive.getinfo(name)
    with archive.open(info) as stream:
        header=read_header(stream);shape=header["shape"];size=header["itemsize"]
        if info.file_size!=header["header_total_bytes"]+header["payload_bytes"]:raise ValueError("Declared NPY member bytes differ or are truncated")
        if rows is None:
            if header["payload_bytes"]>MAX_METADATA_BYTES:raise ValueError("Only small metadata may be fully decoded")
            blob=read_exact(stream,header["payload_bytes"]);array=decode_numeric(blob,header["descr"]).reshape(shape)
            converted=len(blob);skipped=unread=0
        else:
            if (not shape or isinstance(rows,bool) or not isinstance(rows,int) or not 0<=rows<=shape[0]):raise ValueError("Requested row prefix outside declared source")
            if len(shape)==2 and header["fortran_order"]:
                import numpy as np
                array=np.empty((rows,shape[1]),dtype=np.dtype(header["descr"]))
                skipped=0;tail=(shape[0]-rows)*size
                for channel in range(shape[1]):
                    if channel:
                        remaining=tail
                        while remaining:
                            chunk=read_exact(stream,min(remaining,65536));remaining-=len(chunk);skipped+=len(chunk)
                    array[:,channel]=decode_numeric(read_exact(stream,rows*size),header["descr"])
                converted=rows*shape[1]*size;unread=tail if shape[1] else 0
            else:
                count=rows*(shape[1] if len(shape)==2 else 1)*size
                array=decode_numeric(read_exact(stream,count),header["descr"]).reshape([rows]+shape[1:])
                converted=count;skipped=0;unread=header["payload_bytes"]-count
        evidence=dict(member=name,**header,materialized_shape=list(array.shape),numeric_payload_bytes_converted=converted,
            opaque_payload_bytes_discarded=skipped,payload_bytes_left_unread=unread,tail_numeric_values_parsed=0)
        return array,evidence


def array_sha(array):
    import numpy as np
    value=np.ascontiguousarray(array);digest=hashlib.sha256()
    digest.update(str(value.dtype).encode());digest.update(json.dumps(list(value.shape)).encode());digest.update(value.tobytes())
    return digest.hexdigest()


def load_jena_prefix(asset_root,old_task_contract):
    """Return (task, access_receipt); real numeric execution belongs to ROOT."""
    root=plain(asset_root);source=plain(root/SOURCE_REL);contract_path=plain(root/CONTRACT_REL)
    if not source.is_relative_to(root) or not contract_path.is_relative_to(root):raise ValueError("Asset root escape")
    if source.stat().st_size!=SOURCE_BYTES or sha(source)!=SOURCE_SHA or sha(contract_path)!=CONTRACT_SHA:raise ValueError("Exact legacy source/contract bytes changed")
    old=copy.deepcopy(old_task_contract)
    if (old.get("shape")!=[PERMITTED_ROWS,CHANNELS] or old.get("L")!=512 or old.get("H")!=48
            or old.get("primary_split_ranges")!=BOUNDS or len(old.get("columns",[]))!=CHANNELS
            or set(old.get("array_sha256",{}))!={"physical","observed","times","columns"}
            or not re.fullmatch("[0-9a-f]{64}",old.get("data_sha256",""))):raise ValueError("Explicit sealed original full20965 task contract required")
    contract=json.loads(contract_path.read_text(encoding="utf-8-sig"))
    if (contract["trainval"]["sha256"]!=SOURCE_SHA or contract["trainval"]["bytes"]!=SOURCE_BYTES
            or contract["bounds"]["train"]!=[0,26207] or int(.8*26207)!=PERMITTED_ROWS):raise ValueError("Original permitted prefix ancestry differs")
    import numpy as np
    values={};members=[]
    with zipfile.ZipFile(source) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)):raise ValueError("Duplicate ZIP member names forbidden")
        for name in ("x","finite","times","mean","std","columns"):
            array,evidence=read_member(archive,name+".npy",PREFIX_ROWS if name in {"x","finite","times"} else None)
            values[name]=array;members.append(evidence)
    expected={"x":([PREFIX_ROWS,CHANNELS],np.dtype("float32")),"finite":([PREFIX_ROWS,CHANNELS],np.dtype("bool")),
        "times":([PREFIX_ROWS],np.dtype("int64")),"mean":([CHANNELS],np.dtype("float32")),"std":([CHANNELS],np.dtype("float32"))}
    for name,(shape,dtype) in expected.items():
        if list(values[name].shape)!=shape or values[name].dtype!=dtype:raise ValueError("Actual primitive source axes/dtype differ")
    if any(row["shape"][0]!=SOURCE_ROWS for row in members if row["member"] in {"x.npy","finite.npy","times.npy"}):raise ValueError("Actual source35135 row count differs")
    columns=values["columns"].tolist()
    if columns!=old["columns"] or columns!=contract["columns"]:raise ValueError("Original channel names/order differ")
    physical=(values["x"]*values["std"].astype(np.float32)+values["mean"].astype(np.float32)).astype(np.float32)
    observed=values["finite"].astype(bool,copy=True);times=values["times"].astype(np.int64,copy=True)
    if (not np.isfinite(values["mean"]).all() or not np.isfinite(values["std"]).all()
            or not np.isfinite(physical[observed]).all() or not np.all(np.diff(times)==600*1000000000)):
        raise ValueError("Observed prefix values, legacy normalization or time grid invalid")
    physical[~observed]=np.nan
    hashes=dict(physical=array_sha(physical),observed=array_sha(observed),times=array_sha(times),
        columns=hashlib.sha256(json.dumps(columns,ensure_ascii=False).encode()).hexdigest())
    prefix_sha=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()
    task=dict(name="jena",dataset="jena",physical=physical,observed=observed,times=times,columns=columns,L=512,H=48,
        interval_seconds=600,period=144,data_sha256=old["data_sha256"],array_sha256=old["array_sha256"],
        data_identity_provenance="INHERITED_NOT_RECOMPUTED",materialized_prefix_sha256=prefix_sha,materialized_prefix_array_sha256=hashes,
        original_total_rows=PERMITTED_ROWS,materialized_rows=PREFIX_ROWS,split_ranges=copy.deepcopy(BOUNDS),
        secondary_split_ranges=copy.deepcopy(old.get("secondary_split_ranges")),metadata=copy.deepcopy(old["metadata"]))
    task["metadata"]["current_priority_loader_scope"]="Only pre-T prefix converted; inherited legacy deserialization note describes earlier ancestry"
    if sha(source)!=SOURCE_SHA or sha(contract_path)!=CONTRACT_SHA:raise ValueError("Original source/contract changed during prefix materialization")
    receipt=dict(status="MATERIALIZED_PRE_T_PREFIX_ONLY",source_relative_path=SOURCE_REL,source_sha256=SOURCE_SHA,source_bytes=SOURCE_BYTES,
        source_rows=SOURCE_ROWS,old_permitted_rows=PERMITTED_ROWS,materialized_rows=PREFIX_ROWS,materialized_bounds=[0,PREFIX_ROWS],
        contract_relative_path=CONTRACT_REL,contract_sha256=CONTRACT_SHA,original_primary_split_ranges=BOUNDS,members=members,
        inherited_full_task_data_sha256=old["data_sha256"],inherited_full_task_array_sha256=old["array_sha256"],
        inherited_identity_status="INHERITED_NOT_RECOMPUTED",materialized_prefix_sha256=prefix_sha,materialized_prefix_array_sha256=hashes,
        old_task_contract_sha256=hashlib.sha256(json.dumps(old,sort_keys=True,ensure_ascii=False).encode()).hexdigest(),
        new_T_numeric_values_parsed=0,raw_values_published=False,original_source_bytes_preserved=True,
        physical_IO_scope="Source file opaque SHA reads whole compressed bytes; ZIP may buffer/decompress opaque tail while skipping; only selected prefix payload bytes reach numeric decoding")
    return task,receipt


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(json.dumps(dict(status="LOADER_LIBRARY_ONLY_NOT_EXECUTED",input_reads=0,writes=0,numpy_imported="numpy" in sys.modules,
        torch_imported="torch" in sys.modules,actual_source_numeric_values_parsed=0,source_rows=SOURCE_ROWS,
        original_permitted_rows=PERMITTED_ROWS,requested_materialized_rows=PREFIX_ROWS)))
    return 0


if __name__=="__main__":raise SystemExit(main())
