from pathlib import Path
import argparse,gzip,hashlib,json
def restore(manifest_path,destination):
 m=Path(manifest_path);spec=json.loads(m.read_text(encoding="utf-8"));root=Path(destination).resolve();pending=[];matched=0
 for r in spec["items"]:
  source=(m.parent/r["gzip_file"]).resolve();target=(root/r["original_file"]).resolve()
  if not source.is_relative_to(m.parent.resolve()) or not target.is_relative_to(root):raise ValueError("Unsafe restoration path")
  blob=source.read_bytes()
  if len(blob)!=r["gzip_bytes"] or hashlib.sha256(blob).hexdigest()!=r["gzip_sha256"]:raise ValueError("Gzip byte SHA differs")
  with gzip.GzipFile(fileobj=__import__("io").BytesIO(blob)) as f:raw=f.read(r["original_bytes"]+1)
  if len(raw)!=r["original_bytes"] or hashlib.sha256(raw).hexdigest()!=r["original_sha256"]:raise ValueError("Original byte SHA differs")
  if target.exists():
   if not target.is_file() or target.read_bytes()!=raw:raise ValueError("Existing destination differs; never overwrite")
   matched+=1
  else:pending.append((target,raw))
 for target,raw in pending:
  target.parent.mkdir(parents=True,exist_ok=True)
  with target.open("xb") as f:f.write(raw)
 return dict(status="PASS",restored=len(pending),already_matching=matched)
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--destination",type=Path,required=True);a=p.parse_args();print(json.dumps(restore(Path(__file__).with_name("GZIP_MANIFEST.json"),a.destination)))
