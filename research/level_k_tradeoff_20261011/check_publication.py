"""Check published paths, original science seals, and an optional staged bundle."""

from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import os
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LINK = re.compile(r"(!?\[[^\]]*\]\()([^\)]+)(\))")
OUTPUT = HERE / "publication_checks.json"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def text_without_destinations(text):
    return LINK.sub(lambda m: m[1] + "<DESTINATION>" + m[3], text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--staged", action="store_true")
    parser.add_argument("--local", action="store_true", help="Also check unpublished local snapshots/cache preservation")
    parser.add_argument("--write-receipt", action="store_true")
    args = parser.parse_args()
    manifest = read(HERE / "publication_files.json")
    selected = {Path(p) for p in manifest["files"]}
    checks, links = [], []

    def check(name, passed, **detail):
        checks.append({"name": name, "passed": bool(passed), **detail})

    for path in sorted(selected):
        absolute = ROOT / path
        if absolute == OUTPUT and args.write_receipt and not OUTPUT.exists():
            continue
        check(f"bundle file:{path.as_posix()}", absolute.is_file())
        check(f"no raw asset:{path.as_posix()}", path.suffix not in
              (".pt", ".pth", ".ckpt", ".safetensors", ".npz", ".npy", ".pkl", ".log", ".tmp")
              and ".cache" not in path.parts and "__pycache__" not in path.parts)
        if not absolute.is_file() or path.suffix != ".md":
            continue
        text = absolute.read_text(encoding="utf-8-sig")
        if path.as_posix() == "README.md":
            text = text.split("## 동결 TSFM의 K·표현·디코더 비용–오차 비교 (2026-10-11)", 1)[1]
            text = text.split("\n## ", 1)[0]
        for match in LINK.finditer(text):
            value = match[2]
            if value.startswith(("http://", "https://", "mailto:", "#")):
                continue
            target = (absolute.parent / value).resolve()
            relative_ok = not re.match(r"^[A-Za-z]:[\\/]", value) and not value.startswith(("/", "file:"))
            inside = target.is_relative_to(ROOT)
            pending_receipt = target == OUTPUT and args.write_receipt and not OUTPUT.exists()
            present = inside and (target.is_file() or pending_receipt)
            tracked = inside and (target.relative_to(ROOT) in selected or
                subprocess.run(["git", "ls-files", "--error-unmatch", "--", target.relative_to(ROOT).as_posix()],
                               cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0)
            check(f"Markdown target:{path.as_posix()}:{value}", relative_ok and present and tracked)
            links.append({"document": path.as_posix(), "destination": value,
                          "target": target.relative_to(ROOT).as_posix() if inside else str(target),
                          "image": match[1].startswith("!")})
            if match[1].startswith("!") and present:
                check(f"PNG bytes:{value}", target.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n")
    for name in ("k", "continuation", "method"):
        sealed = read(HERE / f"source_seal.{name}.json")
        check(f"science seal:{name}", all(sha(ROOT / r["path"]) == r["sha256"] for r in sealed["artifacts"]))
    summary = read(HERE / "summary.json")
    for name, expected in summary["inputs"].items():
        check(f"science result unchanged:{name}", sha(HERE / name) == expected)
    before = read(HERE / "publication_before.json")
    check("original final audit receipt unchanged", sha(HERE / "final_verification.json")
          == before["original_final_audit_sha256"])
    check("original independent audit retained", read(HERE / "verification.json")["status"] == "PASS"
          and len(read(HERE / "verification.json")["checks"]) == 2518)
    context = read(HERE / "context/provenance.json")
    for row in context["artifacts"]:
        published = ROOT / row["published"]
        check(f"context copy:{row['published']}", sha(published) == row["published_sha256"])
        if args.local:
            source = ROOT / row["source"]
            check(f"context original unchanged:{row['source']}", sha(source) == row["source_sha256"])
            if source.suffix == ".md":
                check(f"context text preserved:{row['source']}",
                      text_without_destinations(source.read_text(encoding="utf-8-sig"))
                      == text_without_destinations(published.read_text(encoding="utf-8-sig")))
    if args.local:
        backups = {r["source"]: r for r in before["preserved_originals"]}
        for row in before["preserved_originals"]:
            check(f"original document preserved:{row['source']}", sha(ROOT / row["backup"]) == row["sha256"])
        for row in before["original_artifacts"]:
            original = ROOT / backups[row["path"]]["backup"] if row["path"] in backups else ROOT / row["path"]
            check(f"original artifact preserved:{row['path']}", sha(original) == row["sha256"])
        for name in ("REPORT_KO.md", "EXECUTION_NOTE_KO.md"):
            path = (HERE / name).relative_to(ROOT).as_posix()
            check(f"report text and metrics preserved:{name}",
                  text_without_destinations((ROOT / backups[path]["backup"]).read_text(encoding="utf-8-sig"))
                  == text_without_destinations((HERE / name).read_text(encoding="utf-8-sig")))
        prior = read(HERE / "parent_preservation.before.json")
        check("preceding 4662 artifacts preserved", len(prior["artifacts"]) == 4662
              and all(sha(ROOT / backups[r["path"]]["backup"] if r["path"] in backups else Path(r["path"])
                          if Path(r["path"]).is_absolute() else ROOT / r["path"]) == r["sha256"]
                      for r in prior["artifacts"]))
        history = prior["history_append_only"]
        check("prior history prefix preserved", hashlib.sha256((ROOT / history["path"]).read_bytes()
              [:history["bytes"]]).hexdigest() == history["prefix_sha256"])
    if args.staged:
        staged = {Path(p) for p in subprocess.check_output(["git", "diff", "--cached", "--name-only"],
                  cwd=ROOT, text=True, encoding="utf-8").splitlines()}
        expected = selected - ({OUTPUT.relative_to(ROOT)} if args.write_receipt and not OUTPUT.exists() else set())
        check("exact staged publication scope", staged == expected,
              missing=sorted(p.as_posix() for p in expected - staged),
              extra=sorted(p.as_posix() for p in staged - expected))
        for path in sorted(staged):
            blob = subprocess.check_output(["git", "show", f":{path.as_posix()}"], cwd=ROOT)
            check(f"staged bytes:{path.as_posix()}", hashlib.sha256(blob).hexdigest() == sha(ROOT / path))
    failed = [row for row in checks if not row["passed"]]
    files = [{"path": p.as_posix(), "sha256": sha(ROOT / p), "bytes": (ROOT / p).stat().st_size}
             for p in sorted(selected) if (ROOT / p).is_file() and ROOT / p != OUTPUT]
    receipt = {"status": "FAIL" if failed else "PASS", "checked_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "Publication paths and preserved artifacts; no training, GPU/F call, TEST, or benchmark",
               "base_commit": before["base_commit"], "local_preservation_checked": args.local,
               "staged_checked": args.staged, "check_count": len(checks), "failure_count": len(failed),
               "checks": checks, "failures": failed, "links": links, "files": files,
               "self_exclusion": "publication_checks.json cannot hash itself"}
    print(f"Publication {receipt['status']}: {len(checks)} checks, {len(links)} local links, {len(files)} files")
    for failure in failed:
        print(failure)
    if args.write_receipt:
        with OUTPUT.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
