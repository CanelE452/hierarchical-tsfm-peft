"""Preserve local receipts and prepare a narrowly scoped GitHub publication."""

from pathlib import Path
import hashlib
import json
import os
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONTEXT = HERE / "context"
BACKUP = ROOT / ".cache" / HERE.name / "publication_before"
LINK = re.compile(r"(!?\[[^\]]*\]\()([^\)]+)(\))")


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def relative(path, start):
    return Path(os.path.relpath(path, start)).as_posix()


def local_target(value, source):
    if value.startswith(("https://", "http://", "mailto:", "#")):
        return None
    path = Path(value)
    return path.resolve() if path.is_absolute() else (source.parent / path).resolve()


def main():
    receipt = HERE / "publication_before.json"
    if receipt.exists() or BACKUP.exists():
        raise FileExistsError("A publication snapshot already exists; preserve it rather than rerun")
    final = json.loads((HERE / "final_verification.json").read_text(encoding="utf-8"))
    if final["status"] != "PASS":
        raise RuntimeError("The original final audit must pass")
    for row in final["artifacts"]:
        if sha(ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError(f"Pre-publication artifact changed: {row['path']}")
    changed = [ROOT / "README.md", ROOT / "_docs/history/2026-10-11.md",
               HERE / "write_report.py", HERE / "REPORT_KO.md", HERE / "EXECUTION_NOTE_KO.md"]
    preserved = []
    for source in changed:
        target = BACKUP / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        preserved.append({"source": source.relative_to(ROOT).as_posix(),
                          "backup": target.relative_to(ROOT).as_posix(),
                          "sha256": sha(source), "bytes": source.stat().st_size})
    receipt.write_text(json.dumps({"scope": "Snapshot before authorized GitHub path/index publication; original science audit retained",
                                  "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                                  "original_final_audit_sha256": sha(HERE / "final_verification.json"),
                                  "original_artifacts": final["artifacts"], "preserved_originals": preserved},
                                 ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    pending = [ROOT / "research/level_encoder_contract_20261010/METHODS_KO.md"]
    sources = {}
    while pending:
        source = pending.pop()
        if source in sources:
            continue
        if not source.is_relative_to(ROOT / "research") or not source.is_file():
            raise ValueError(f"Unsupported or missing context source: {source}")
        if source.suffix not in (".md", ".json", ".png", ".pdf"):
            raise ValueError(f"Do not publish a raw array/checkpoint: {source}")
        target = CONTEXT / source.relative_to(ROOT / "research")
        sources[source] = target
        if source.suffix == ".md":
            for match in LINK.finditer(source.read_text(encoding="utf-8-sig")):
                linked = local_target(match[2], source)
                if linked is not None:
                    pending.append(linked)
    copied = []
    for source, target in sorted(sources.items()):
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".md":
            text = source.read_text(encoding="utf-8-sig")

            def replace_link(match):
                linked = local_target(match[2], source)
                if linked is None:
                    return match[0]
                return match[1] + relative(sources[linked], target.parent) + match[3]

            target.write_text(LINK.sub(replace_link, text), encoding="utf-8")
        else:
            target.write_bytes(source.read_bytes())
        copied.append({"source": source.relative_to(ROOT).as_posix(), "source_sha256": sha(source),
                       "published": target.relative_to(ROOT).as_posix(), "published_sha256": sha(target),
                       "change": "Markdown destinations only" if source.suffix == ".md" else "byte-identical copy"})
    (CONTEXT / "provenance.json").write_text(json.dumps({"scope": "Historical 2026-10-10 documents; latest study state is REPORT_KO.md",
                                                        "artifacts": copied}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    note = HERE / "EXECUTION_NOTE_KO.md"
    old = note.read_text(encoding="utf-8")
    note.write_text(old.replace(f"({(HERE / 'REPORT_KO.md').as_posix()})", "(REPORT_KO.md)"), encoding="utf-8")
    print(f"Preserved {len(final['artifacts'])} original receipts/artifacts; copied {len(copied)} context targets")


if __name__ == "__main__":
    main()
