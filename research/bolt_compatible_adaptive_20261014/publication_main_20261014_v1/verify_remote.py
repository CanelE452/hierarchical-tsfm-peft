"""Check published bytes and GitHub Markdown rendering; no scientific execution."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import subprocess
import time
from urllib.parse import quote, unquote, urlsplit
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PREFIX = HERE.relative_to(ROOT).as_posix()
REPOSITORY = "CanelE452/hierarchical-tsfm-peft"
RAW_BASE = "https://raw.githubusercontent.com/" + REPOSITORY
WEB_BASE = "https://github.com/" + REPOSITORY
ALLOWED_SUFFIXES = {".py", ".md", ".json", ".csv", ".png", ".svg", ".txt"}
MARKDOWN_LINK = re.compile(r"(!?)\[[^\]\n]*\]\(([^)\n]+)\)")
HTML_PATHS = [
    "RESEARCH_RESULTS_20261014_KO.md",
    PREFIX + "/README_KO.md",
    PREFIX + "/reports/adaptive_REPORT_KO.md",
    PREFIX + "/reports/PRESENTATION_BRIEF_KO.md",
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, timeout=30)


def checked_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("Non-repository path: " + value)
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError("Non-public artifact type: " + value)
    return path.as_posix()


def fetch(url):
    last_error = None
    for attempt in range(1, 4):
        try:
            request = Request(url, headers={"User-Agent": "LEVEL-publication-verifier/1.0"})
            with urlopen(request, timeout=20) as response:
                data = response.read()
                return {"http_status": response.status, "attempts": attempt,
                        "bytes": len(data), "sha256": digest(data)}, data
        except Exception as error:
            last_error = type(error).__name__ + ": " + str(error)
            if attempt < 3:
                time.sleep(attempt)
    return {"http_status": None, "attempts": 3, "error": last_error}, None


class RenderedMarkdown(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.article_depth = 0
        self.article_count = 0
        self.images = []
        self.links = []
        self.json_scripts = []
        self.script_text = None

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "article":
            self.article_depth += 1
            self.article_count += 1
        if self.article_depth:
            if tag == "img" and values.get("src"):
                self.images.append(values["src"])
            if tag == "a" and values.get("href"):
                self.links.append(values["href"])
        if tag == "script" and values.get("type") == "application/json":
            self.script_text = []

    def handle_endtag(self, tag):
        if tag == "article" and self.article_depth:
            self.article_depth -= 1
        if tag == "script" and self.script_text is not None:
            self.json_scripts.append("".join(self.script_text))
            self.script_text = None

    def handle_data(self, data):
        if self.script_text is not None:
            self.script_text.append(data)


def rich_text_fragments(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"richText", "renderedFile"} and isinstance(item, str):
                if "<article" in item:
                    yield item
            yield from rich_text_fragments(item)
    elif isinstance(value, list):
        for item in value:
            yield from rich_text_fragments(item)


def rendered_markdown(document):
    parsed = RenderedMarkdown()
    parsed.feed(document)
    if parsed.article_count:
        return parsed, "article"
    for script in parsed.json_scripts:
        try:
            payload = json.loads(script)
        except json.JSONDecodeError:
            continue
        for fragment in rich_text_fragments(payload):
            candidate = RenderedMarkdown()
            candidate.feed(fragment)
            if candidate.article_count:
                return candidate, "react-app-richText"
    return parsed, "NO_RENDERED_MARKDOWN"


def repository_url_path(url, commit):
    value = urlsplit(html.unescape(url))
    path = unquote(value.path)
    if value.netloc == "raw.githubusercontent.com":
        prefix = "/" + REPOSITORY + "/" + commit + "/"
        return path[len(prefix):] if path.startswith(prefix) else None
    if value.netloc not in {"", "github.com"}:
        return None
    for kind in ("blob", "raw"):
        prefix = "/" + REPOSITORY + "/" + kind + "/" + commit + "/"
        if path.startswith(prefix):
            return path[len(prefix):]
    return None


def expected_links(path, document):
    images, links = [], []
    for marker, target in MARKDOWN_LINK.findall(document):
        target = target.strip().strip("<>")
        url = urlsplit(target)
        if url.scheme or url.netloc or not url.path:
            continue
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(path), unquote(url.path)))
        resolved = checked_path(resolved)
        (images if marker else links).append(resolved)
    return images, links


def verify_raw(row, commit):
    result, data = fetch(RAW_BASE + "/" + commit + "/" + quote(row["path"], safe="/"))
    result.update({"path": row["path"], "expected_sha256": row["sha256"],
                   "expected_bytes": row["bytes"]})
    result["pass"] = (data is not None and result["http_status"] == 200
                      and result["bytes"] == row["bytes"]
                      and result["sha256"] == row["sha256"])
    if data is not None and row["path"].endswith(".png"):
        result["png_signature"] = data.startswith(b"\x89PNG\r\n\x1a\n")
        result["pass"] = result["pass"] and result["png_signature"]
    return result


def verify_html(path, commit):
    source = git("show", commit + ":" + path).decode("utf-8-sig")
    expected_images, expected_local_links = expected_links(path, source)
    fetched, data = fetch(WEB_BASE + "/blob/" + commit + "/" + quote(path, safe="/"))
    record = {"path": path, **fetched, "pass": False,
              "expected_images": expected_images, "expected_local_links": expected_local_links}
    if data is None:
        return record
    rendered, mode = rendered_markdown(data.decode("utf-8"))
    image_paths = [repository_url_path(url, commit) for url in rendered.images]
    link_paths = [repository_url_path(url, commit) for url in rendered.links]
    record.update({"render_source": mode, "rendered_article_count": rendered.article_count,
                   "rendered_images": image_paths,
                   "rendered_local_links": [value for value in link_paths if value],
                   "missing_images": [value for value in expected_images if value not in image_paths],
                   "missing_local_links": [value for value in expected_local_links if value not in link_paths]})
    expected_count = {PREFIX + "/README_KO.md": 2,
                      PREFIX + "/reports/adaptive_REPORT_KO.md": 8}.get(path)
    record["required_image_count"] = expected_count
    record["pass"] = (fetched["http_status"] == 200 and rendered.article_count > 0
                      and not record["missing_images"] and not record["missing_local_links"]
                      and (expected_count is None or len(expected_images) == expected_count)
                      and (expected_count is None or len(image_paths) == expected_count))
    return record


def self_test():
    article = '<article><a href="/CanelE452/hierarchical-tsfm-peft/blob/' + "a" * 40 + '/x.md">x</a><img src="/CanelE452/hierarchical-tsfm-peft/raw/' + "a" * 40 + '/y.png"></article>'
    parsed, mode = rendered_markdown(article)
    assert mode == "article" and parsed.article_count == 1 and len(parsed.images) == 1
    payload = '<script type="application/json" data-target="react-app.embeddedData">' + json.dumps({"payload": {"richText": article}}) + '</script>'
    parsed, mode = rendered_markdown(payload)
    assert mode == "react-app-richText" and len(parsed.links) == 1
    parsed, mode = rendered_markdown('<html><img src="y.png"></html>')
    assert mode == "NO_RENDERED_MARKDOWN" and not parsed.images
    assert repository_url_path(parsed.images[0] if parsed.images else "/other.png", "a" * 40) is None
    assert repository_url_path("/CanelE452/hierarchical-tsfm-peft/raw/" + "a" * 40 + "/y.png", "a" * 40) == "y.png"
    assert expected_links("d/r.md", '![x](../f.png) [a](t.md#part) [web](https://example.org)') == (["f.png"], ["d/t.md"])
    print(json.dumps({"status": "PASS", "scope": "five parser/path fixtures; no HTTP/Git/scientific execution"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not args.commit or not re.fullmatch(r"[0-9a-f]{40}", args.commit):
        parser.error("--commit must be the full published commit SHA")
    commit = args.commit
    report = {"status": "FAIL", "commit": commit,
              "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "GitHub published bytes and actual Markdown rendering only; no scientific execution",
              "protected_TEST_opened": False, "failures": []}
    try:
        manifest_path = HERE / "FILES.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8-sig"))
        rows = [{"path": checked_path(row["path"]), "sha256": row["sha256"],
                 "bytes": row["bytes"]} for row in manifest["files"]]
        rows.append({"path": PREFIX + "/FILES.json", "sha256": digest(manifest_bytes),
                     "bytes": len(manifest_bytes)})
        if len({row["path"] for row in rows}) != len(rows):
            raise ValueError("Duplicate publication manifest paths")
        report["base_commit"] = manifest["base_commit"]
        report["local_head_begin"] = git("rev-parse", "HEAD").decode().strip()
        report["remote_main_begin"] = git("ls-remote", "origin", "refs/heads/main").decode().split()[0]
        if report["local_head_begin"] != commit or report["remote_main_begin"] != commit:
            raise ValueError("Published SHA differs from local HEAD or remote main")
        with ThreadPoolExecutor(max_workers=6) as executor:
            report["raw_files"] = list(executor.map(lambda row: verify_raw(row, commit), rows))
            report["html_pages"] = list(executor.map(lambda path: verify_html(path, commit), HTML_PATHS))
        expected_paths = {row["path"] for row in rows}
        linked_paths = set()
        for page in report["html_pages"]:
            linked_paths.update(page["expected_images"])
            linked_paths.update(page["expected_local_links"])
        extra_rows = []
        for path in sorted(linked_paths - expected_paths):
            checked_path(path)
            data = git("show", commit + ":" + path)
            extra_rows.append({"path": path, "sha256": digest(data), "bytes": len(data)})
        with ThreadPoolExecutor(max_workers=6) as executor:
            report["additional_linked_raw_files"] = list(executor.map(lambda row: verify_raw(row, commit), extra_rows))
        report["local_head_end"] = git("rev-parse", "HEAD").decode().strip()
        report["remote_main_end"] = git("ls-remote", "origin", "refs/heads/main").decode().split()[0]
        checks = report["raw_files"] + report["additional_linked_raw_files"] + report["html_pages"]
        report["failures"] = [{"path": row["path"], "detail": row} for row in checks if not row["pass"]]
        if report["local_head_end"] != commit or report["remote_main_end"] != commit:
            report["failures"].append({"detail": "Local HEAD or remote main changed during verification"})
        report.update({"raw_file_checks": len(report["raw_files"]),
                       "raw_SHA_matches": sum(row["pass"] for row in report["raw_files"]),
                       "additional_linked_raw_checks": len(report["additional_linked_raw_files"]),
                       "html_checks": len(report["html_pages"]),
                       "rendered_image_tags": sum(len(row.get("rendered_images", [])) for row in report["html_pages"]),
                       "failed_count": len(report["failures"]),
                       "status": "PASS" if not report["failures"] else "FAIL"})
    except Exception as error:
        report["failures"].append({"detail": type(error).__name__ + ": " + str(error)})
        report["failed_count"] = len(report["failures"])
    output = ROOT / ".cache/bolt_compatible_adaptive_20261014/publication_main_20261014_v1/REMOTE_VERIFICATION.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        previous = output.with_name("REMOTE_VERIFICATION_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
        previous.write_bytes(output.read_bytes())
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report.get(key) for key in ("status", "commit", "raw_file_checks", "raw_SHA_matches", "html_checks", "rendered_image_tags", "failed_count")}, ensure_ascii=False))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
