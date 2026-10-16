"""Acquire two public pinned Bolt assets without loading a model or a GPU."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import urllib.request
import uuid

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
from huggingface_hub import HfApi
from resources import check_storage_budget, record_download

REPOSITORIES = ("amazon/chronos-bolt-mini", "amazon/chronos-bolt-tiny")
REQUIRED_FILES = ("README.md", "config.json", "model.safetensors")


def write_new_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)


def metadata(url):
    request = urllib.request.Request(url, headers={"User-Agent": "tsfm-research-reproducibility/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read(512 * 1024 + 1)
    if len(body) > 512 * 1024:
        raise ValueError("Unexpectedly large metadata response")
    return json.loads(body)


def hashes(path):
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(block)
            git_blob.update(block)
    return dict(bytes=size, sha256=sha256.hexdigest(), git_blob_sha1=git_blob.hexdigest())


def acquire_file(cache, target, url, expected_size, expected_sha256=None, expected_git_blob=None):
    expected_size = int(expected_size)
    reused = target.exists()
    if not reused:
        check_storage_budget(cache, pending_bytes=expected_size)
        record_download(cache, "model_asset:" + str(target.relative_to(cache)).replace("\\", "/"), expected_size)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + "." + uuid.uuid4().hex + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "tsfm-research-reproducibility/1.0"})
        downloaded = 0
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("xb") as output:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                downloaded += len(block)
                if downloaded > expected_size:
                    raise ValueError("Asset response exceeds pinned byte length")
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        actual = hashes(temporary)
        if (actual["bytes"] != expected_size
                or (expected_sha256 and actual["sha256"] != expected_sha256)
                or (expected_git_blob and actual["git_blob_sha1"] != expected_git_blob)):
            raise ValueError("Downloaded asset does not match pinned metadata; partial preserved")
        temporary.rename(target)
    actual = hashes(target)
    if (actual["bytes"] != expected_size
            or (expected_sha256 and actual["sha256"] != expected_sha256)
            or (expected_git_blob and actual["git_blob_sha1"] != expected_git_blob)):
        raise ValueError("Existing asset does not match pinned metadata")
    return dict(**actual, reused_existing=reused, downloaded_bytes=0 if reused else expected_size,
                expected_lfs_sha256=expected_sha256, expected_git_blob_sha1=expected_git_blob,
                source_url=url, cache_relative_path=str(target.relative_to(cache)).replace("\\", "/"))


def safetensors_header(path):
    with path.open("rb") as stream:
        header_length = int.from_bytes(stream.read(8), "little")
        if not 2 <= header_length <= 16 * 1024**2:
            raise ValueError("Unexpected safetensors header length")
        header = json.loads(stream.read(header_length))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    body_size = path.stat().st_size - 8 - header_length
    for value in tensors.values():
        start, end = value["data_offsets"]
        if not 0 <= start <= end <= body_size:
            raise ValueError("Safetensors data offset exceeds pinned file")
    # safe_open validates the format and exposes metadata; no tensor is loaded.
    from safetensors import safe_open
    with safe_open(str(path), framework="numpy") as stream:
        keys = list(stream.keys())
        if set(keys) != set(tensors):
            raise ValueError("Safetensors metadata key mismatch")
        for key in keys:
            if list(stream.get_slice(key).get_shape()) != tensors[key]["shape"]:
                raise ValueError("Safetensors metadata shape mismatch")
    return dict(header_bytes=header_length, tensor_count=len(tensors),
                parameter_elements=sum(math.prod(x["shape"]) for x in tensors.values()),
                stored_dtypes=sorted({x["dtype"] for x in tensors.values()}),
                header_sha256=hashlib.sha256(json.dumps(header, sort_keys=True).encode()).hexdigest(),
                tensor_payload_loaded=False)


def acquire(cache, output):
    cache = Path(cache).resolve()
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve existing model manifest; pass a new --output for another acquisition")
    journal = cache / "DOWNLOAD_USAGE.jsonl"
    prior = [json.loads(x) for x in journal.read_text(encoding="utf-8").splitlines()] if journal.exists() else []
    if not any(x["source_id"] == "initial_header_api_conservative" for x in prior):
        record_download(cache, "initial_header_api_conservative", 2 * 1024**2)
    api = HfApi(token=False)
    result = dict(status="ACQUIRING", started_unix=time.time(), token_used=False,
                  terms_accepted_on_user_behalf=False, models={}, GPU_calls=0,
                  model_loads=0, tensor_payload_loads=0)
    source_repo = "amazon-science/chronos-forecasting"
    source_revision = metadata(f"https://api.github.com/repos/{source_repo}/commits/main")["sha"]
    license_meta = metadata(f"https://api.github.com/repos/{source_repo}/contents/LICENSE?ref={source_revision}")
    license_url = f"https://raw.githubusercontent.com/{source_repo}/{source_revision}/LICENSE"
    license_target = cache / "models" / "official_source" / source_revision / "LICENSE"
    license_record = acquire_file(cache, license_target, license_url, license_meta["size"],
                                  expected_git_blob=license_meta["sha"])
    license_text = license_target.read_text(encoding="utf-8")
    if "Apache License" not in license_text or "Version 2.0, January 2004" not in license_text:
        raise ValueError("Official project license did not match Apache-2.0")
    result["official_source_license"] = dict(repository=source_repo, revision=source_revision,
                                            license="Apache-2.0", full_text_read=True, **license_record)
    for repo_id in REPOSITORIES:
        info = api.model_info(repo_id, files_metadata=True, token=False)
        if info.private or info.gated:
            raise PermissionError("Only public ungated models may be acquired")
        card = info.card_data.to_dict() if info.card_data else {}
        if card.get("license") != "apache-2.0":
            raise PermissionError("Model card license requires review")
        revision = info.sha
        directory = cache / "models" / repo_id / revision
        siblings = {file.rfilename: file for file in info.siblings}
        if not set(REQUIRED_FILES).issubset(siblings):
            raise FileNotFoundError("Pinned model lacks a required safe asset")
        pinned = dict(repository=repo_id, revision=revision, private=info.private, gated=info.gated,
                      model_card=card, files={})
        for name in REQUIRED_FILES:
            entry = siblings[name]
            sha256 = entry.lfs.sha256 if entry.lfs else None
            pinned["files"][name] = acquire_file(cache, directory / name,
                f"https://huggingface.co/{repo_id}/resolve/{revision}/{name}", entry.size,
                expected_sha256=sha256, expected_git_blob=None if sha256 else entry.blob_id)
            print(json.dumps(dict(repository=repo_id, file=name, status="HASH_VERIFIED",
                                  bytes=entry.size)), flush=True)
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        card_text = (directory / "README.md").read_text(encoding="utf-8")
        if "Apache-2.0" not in card_text and "apache-2.0" not in card_text:
            raise PermissionError("Downloaded model card license declaration missing")
        chronos = config["chronos_config"]
        if chronos["context_length"] < 512 or chronos["prediction_length"] < 48 or .5 not in chronos["quantiles"]:
            raise ValueError("Pinned Bolt config does not meet registered L/H/median requirements")
        pinned.update(status="READY_PINNED_NOT_INFERENCE_TESTED", config=config,
                      model_card_full_text_read=True, context_length=chronos["context_length"],
                      prediction_length=chronos["prediction_length"], quantiles=chronos["quantiles"],
                      cache_relative_path=str(directory.relative_to(cache)).replace("\\", "/"),
                      safe_tensor_metadata=safetensors_header(directory / "model.safetensors"),
                      official_source_license_revision=source_revision)
        result["models"][repo_id.rsplit("-", 1)[-1]] = pinned
        receipt_path = directory / "ACQUISITION_RECEIPT.json"
        if receipt_path.exists():
            existing = json.loads(receipt_path.read_text(encoding="utf-8"))
            if existing["repository"] != repo_id or existing["revision"] != revision or existing["config"] != config:
                raise ValueError("An existing acquisition receipt has a different model contract")
            for name in REQUIRED_FILES:
                if existing["files"][name]["sha256"] != pinned["files"][name]["sha256"]:
                    raise ValueError("An existing acquisition receipt has different file hashes")
        else:
            write_new_json(receipt_path, pinned)
    result.update(status="TWO_ASSETS_READY_PINNED_NOT_INFERENCE_TESTED", finished_unix=time.time(),
                  new_download_file_bytes=sum(f["downloaded_bytes"] for x in result["models"].values() for f in x["files"].values())+license_record["downloaded_bytes"],
                  cache_storage_bytes_at_completion=check_storage_budget(cache))
    write_new_json(output, result)
    return result


def verify_existing(cache, manifest):
    """Verify immutable assets against the saved pin; no requests or downloads."""
    cache = Path(cache).resolve()
    result = json.loads(Path(manifest).read_text(encoding="utf-8"))
    files = [result["official_source_license"]]
    for model in result["models"].values():
        files.extend(model["files"].values())
    for expected in files:
        path = (cache / expected["cache_relative_path"]).resolve()
        if not path.is_relative_to(cache):
            raise ValueError("Model manifest path escapes private cache")
        actual = hashes(path)
        if any(actual[key] != expected[key] for key in ("bytes", "sha256", "git_blob_sha1")):
            raise ValueError("Pinned local model asset hash mismatch")
    for model in result["models"].values():
        header = safetensors_header(cache / model["cache_relative_path"] / "model.safetensors")
        if header != model["safe_tensor_metadata"]:
            raise ValueError("Pinned safetensors metadata mismatch")
    return dict(status="ALL_SEVEN_ASSET_FILES_REVERIFIED", files=len(files), GPU_calls=0,
                model_loads=0, network_requests=0, tensor_payload_loads=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--output", default=str(Path(__file__).with_name("MODEL_ASSETS.json")))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.verify_only:
            print(json.dumps(verify_existing(args.cache, args.output)), flush=True)
        else:
            result = acquire(args.cache, args.output)
            print(json.dumps(dict(status=result["status"], models=list(result["models"]),
                                  new_download_file_bytes=result["new_download_file_bytes"])), flush=True)
    except Exception as error:
        # Avoid logging signed redirect URLs, credentials, or local traceback paths.
        print(json.dumps(dict(status="ASSET_ACQUISITION_FAILED", error_class=type(error).__name__)), flush=True)
        raise SystemExit(1)
