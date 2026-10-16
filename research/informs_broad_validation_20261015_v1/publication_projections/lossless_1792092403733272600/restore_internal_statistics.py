from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import stats_pack as sp
def restore(destination):
 root=Path(destination).resolve();items=json.loads(Path(__file__).with_name("INTERNAL_DUPLICATES32.json").read_text(encoding="utf-8"));todo=[];matched=0
 for x in items:
  source=(root/x["canonical_stats_relative_path"]).resolve();target=(root/x["original"]["repository_relative_path"]).resolve()
  if not source.is_relative_to(root) or not target.is_relative_to(root):raise ValueError("Unsafe restoration path")
  raw=sp._gunzip(source.read_bytes())
  if sp.sha(raw)!=x["exact_rows"]["replacement_uncompressed_sha256"]:raise ValueError("Canonical CSV differs")
  lines=raw.splitlines(keepends=True);prefix=(x["canonical_run_id"]+",").encode()
  if not all(v.startswith(prefix) for v in lines[1:]):raise ValueError("Canonical ID differs")
  payload=lines[0].split(b",",1)[1]+b"".join(v[len(prefix):] for v in lines[1:]);blob=sp._restore_blob(x["original"],payload)
  if target.exists():
   if target.read_bytes()!=blob:raise ValueError("Existing destination differs; never overwrite")
   matched+=1
  else:todo.append((target,blob))
 for target,blob in todo:
  target.parent.mkdir(parents=True,exist_ok=True)
  with target.open("xb") as f:f.write(blob)
 return dict(status="PASS",restored=len(todo),already_matching=matched)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--destination",type=Path,required=True);a=p.parse_args();print(json.dumps(restore(a.destination)))
