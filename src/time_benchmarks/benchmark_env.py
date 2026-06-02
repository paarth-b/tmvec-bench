import json
import os
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _run_command(command, timeout=5):
    if not command:
        return None
    binary = command[0]
    if shutil.which(binary) is None:
        return None
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    if result.returncode == 0:
        return stdout or None
    if stdout or stderr:
        return {
            "returncode": result.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
    return None


def _read_os_release():
    os_release = Path("/etc/os-release")
    if not os_release.exists():
        return None
    data = {}
    for line in os_release.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = value.strip().strip('"')
    return data or None


def _parse_lscpu_json(raw_output):
    if not raw_output:
        return None
    try:
        payload = json.loads(raw_output)
    except json.JSONDecodeError:
        return None
    rows = payload.get("lscpu", [])
    parsed = {}
    for row in rows:
        field = row.get("field", "").strip().rstrip(":")
        value = row.get("data")
        if field:
            parsed[field] = value
    return parsed or None


def capture_benchmark_environment(requested_threads=None, accelerator=None):
    os_release = _read_os_release()
    lscpu = _parse_lscpu_json(_run_command(["lscpu", "-J"]))
    git_commit = _run_command(["git", "rev-parse", "HEAD"])
    git_branch = _run_command(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    git_status = _run_command(["git", "status", "--short"])
    gpu_query = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )

    slurm = {
        key: os.environ.get(key)
        for key in (
            "SLURM_JOB_ID",
            "SLURM_JOB_NAME",
            "SLURM_CPUS_PER_TASK",
            "SLURM_JOB_NODELIST",
            "SLURM_SUBMIT_HOST",
            "SLURM_CLUSTER_NAME",
        )
        if os.environ.get(key)
    }

    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "requested_threads": requested_threads,
        "accelerator": accelerator,
        "working_directory": str(Path.cwd()),
        "os_release": os_release,
        "cpu": {
            "model_name": lscpu.get("Model name") if lscpu else None,
            "architecture": lscpu.get("Architecture") if lscpu else None,
            "logical_cpus": lscpu.get("CPU(s)") if lscpu else None,
            "threads_per_core": lscpu.get("Thread(s) per core") if lscpu else None,
            "cores_per_socket": lscpu.get("Core(s) per socket") if lscpu else None,
            "sockets": lscpu.get("Socket(s)") if lscpu else None,
            "numa_nodes": lscpu.get("NUMA node(s)") if lscpu else None,
            "raw_lscpu": lscpu,
        },
        "gpu": {
            "nvidia_smi": gpu_query,
        },
        "compiler": {
            "gcc_version": _run_command(["gcc", "--version"]),
        },
        "git": {
            "commit": git_commit,
            "branch": git_branch,
            "status_short": git_status.splitlines() if isinstance(git_status, str) and git_status else [],
        },
        "numa": {
            "numactl_hardware": _run_command(["numactl", "--hardware"]),
        },
        "slurm": slurm,
        "environment": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }


def write_json(path, payload):
    output_path = Path(path)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
