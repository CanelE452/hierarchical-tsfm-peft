import importlib
import importlib.metadata as metadata
import inspect
import json
import platform
import shutil
import subprocess
import sys

from .audit import ROOT, RESULTS, sha256, utc_now, write_json


def command(args):
    p = subprocess.run(args, capture_output=True, text=True, errors="replace")
    return {"args": args, "returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}


def precheck():
    gpu = command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.free", "--format=csv"])
    hardware = command(["powershell", "-NoProfile", "-Command",
        "[pscustomobject]@{CPU=(Get-CimInstance Win32_Processor).Name; RAMBytes=(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory} | ConvertTo-Json"])
    result = {"timestamp_utc": utc_now(), "os": platform.platform(), "kernel": platform.version(),
              "hardware": hardware, "python": sys.version, "python_executable": sys.executable,
              "python_base_prefix": sys.base_prefix, "isolated_venv": sys.prefix != sys.base_prefix,
              "python_candidates": {"base": "3.13.9", "conda_mlts": "3.11.16", "conda_dante_val": "3.11.15"},
              "nvidia_smi": gpu, "nvidia_smi_full": command(["nvidia-smi"]),
              "git": command(["git", "--version"]), "gh": command(["gh", "--version"]),
              "disk": {str(d): dict(zip(("total", "used", "free"), shutil.disk_usage(d))) for d in [ROOT, ROOT.anchor]},
              "torch_cuda_test": "pending_installation"}
    write_json(RESULTS / "ENV_PRECHECK.json", result)
    return result


def verify_environment():
    import psutil
    import torch
    imports = ["chronos", "peft", "transformers", "accelerate", "datasetsforecast.hierarchical",
               "hierarchicalforecast.methods", "statsforecast", "utilsforecast", "pandas", "numpy",
               "scipy", "sklearn", "pyarrow", "matplotlib", "yaml", "pytest", "safetensors"]
    modules = {name: importlib.import_module(name) for name in imports}
    assert sys.version_info[:2] == (3, 11)
    assert sys.prefix != sys.base_prefix
    assert torch.cuda.is_available(), "BLOCKED_NO_CUDA_FOR_SCREEN"
    a = torch.arange(16, device="cuda", dtype=torch.float32).reshape(4, 4)
    assert torch.isfinite(a @ a).all().item()
    torch.cuda.synchronize()
    versions = {d.metadata["Name"]: d.version for d in metadata.distributions()}
    sources = {}
    for name, module in modules.items():
        file = inspect.getfile(module)
        sources[name] = {"installed_source": file, "sha256": sha256(file)}
    reports = {}
    for name in ["torch-install.json", "dependencies-install.json"]:
        source = ROOT / ".cache" / name
        report = json.loads(source.read_text(encoding="utf-8"))
        reports[name] = [{"name": p["metadata"]["name"], "version": p["metadata"]["version"],
                          "download_info": p["download_info"]} for p in report["install"]]
    check = command([sys.executable, "-m", "pip", "check"])
    assert check["returncode"] == 0, check
    freeze = command([sys.executable, "-m", "pip", "freeze"])
    assert freeze["returncode"] == 0
    lines = [line for line in freeze["stdout"].splitlines() if not line.startswith(("#", "-e ", "hierarchical-tsfm-peft"))]
    (ROOT / "requirements-lock.txt").write_text("--extra-index-url https://download.pytorch.org/whl/cu128\n" + "\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    result = {"timestamp_utc": utc_now(), "python": sys.version, "python_executable": sys.executable,
              "packages": versions, "pip_check": check, "import_smoke": list(modules), "source_hashes": sources,
              "cuda_available": torch.cuda.is_available(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
              "gpu_count": torch.cuda.device_count(), "gpu": torch.cuda.get_device_name(0),
              "compute_capability": list(torch.cuda.get_device_capability(0)), "cuda_matmul_finite": True,
              "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
              "ram_bytes": psutil.virtual_memory().total, "wheel_provenance": reports,
              "torch_install_reference": "https://pytorch.org/get-started/previous-versions/#v2100",
              "reference_upstream_shas_not_installed_commits": {
                  "chronos-forecasting": "10afa9ebe016e514f9d7dc1aa873f66af57e116b",
                  "datasetsforecast": "f85c26352bc5c2e37b70be729e3e2cfb89b590a2",
                  "hierarchicalforecast": "671cbbc8bf52a4c10a2f22ebfa77f72043677622"},
              "provenance_note": "Installed PyPI/CUDA wheels are identified by exact versions, wheel URLs and hashes; reference SHAs are not asserted as wheel commits.",
              "lock_sha256": sha256(ROOT / "requirements-lock.txt")}
    write_json(RESULTS / "ENVIRONMENT.json", result)
    return result


if __name__ == "__main__":
    if "--verify" in sys.argv:
        report = verify_environment()
        print(json.dumps({k: report[k] for k in ["python", "torch", "torch_cuda", "gpu", "cuda_available"]}))
    else:
        precheck()
        print("ENV_PRECHECK.json recorded")
