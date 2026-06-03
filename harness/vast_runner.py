"""One-shot Vast.ai GPU evaluation runner."""

from __future__ import annotations

import ast
import json
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_IMAGE = "pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel"
DEFAULT_TEMPLATE_HASH = "3ba4addf2b917a405583ebb21dfd3f72"
DEFAULT_DISK_GB = 40
DEFAULT_REMOTE_DIR = "/root/shape_bench_CUDA"
DEFAULT_LOCAL_RUNS_DIR = "results/vast_runs"
DEFAULT_MAX_SSH_AUTH_FAILURES = 4
SSH_OPTIONS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "StrictHostKeyChecking=accept-new",
    "-o",
    "ConnectTimeout=10",
    "-o",
    "ServerAliveInterval=30",
    "-o",
    "ServerAliveCountMax=4",
    "-o",
    "LogLevel=ERROR",
]


@dataclass(frozen=True)
class VastRunConfig:
    offer_id: int
    project_root: Path
    image: str | None = None
    template_hash: str | None = DEFAULT_TEMPLATE_HASH
    disk_gb: int = DEFAULT_DISK_GB
    remote_dir: str = DEFAULT_REMOTE_DIR
    local_runs_dir: str = DEFAULT_LOCAL_RUNS_DIR
    repo_ref: str = "HEAD"
    poll_seconds: int = 10
    max_wait_seconds: int = 600
    keep_instance: bool = False
    allow_dirty: bool = False
    skip_tests: bool = False
    max_ssh_auth_failures: int = DEFAULT_MAX_SSH_AUTH_FAILURES


@dataclass(frozen=True)
class VastRunResult:
    instance_id: int
    local_run_dir: Path
    remote_exit_code: int
    destroyed: bool


def run_vast_eval(config: VastRunConfig) -> VastRunResult:
    """Launch a Vast.ai instance, run the GPU batch, fetch results, and destroy it."""
    _log("checking local git tree")
    _ensure_clean_repo(config.project_root, allow_dirty=config.allow_dirty)
    local_run_dir = _local_run_dir(config.project_root, config.local_runs_dir)
    local_run_dir.mkdir(parents=True, exist_ok=True)
    _log(f"local run directory: {local_run_dir}")
    instance_id: int | None = None
    cleanup_error: BaseException | None = None
    destroyed = False
    remote_exit_code = 1

    try:
        _log(f"creating Vast instance from offer {config.offer_id}")
        instance_id = create_instance(config)
        _log(f"created Vast instance {instance_id}")
        ssh_args = wait_for_ssh(
            instance_id,
            poll_seconds=config.poll_seconds,
            max_wait_seconds=config.max_wait_seconds,
            max_auth_failures=config.max_ssh_auth_failures,
        )
        _log("uploading committed project archive")
        upload_git_archive(config, ssh_args)
        _log("starting remote GPU evaluation")
        remote_exit_code = run_remote_eval(config, ssh_args, local_run_dir)
        _log(f"remote GPU evaluation finished with exit code {remote_exit_code}")
        _log("downloading result artifacts")
        download_results(config, ssh_args, local_run_dir)
    finally:
        if instance_id is not None and not config.keep_instance:
            _log(f"destroying Vast instance {instance_id}; do not interrupt cleanup")
            try:
                _destroy_instance_uninterruptible(instance_id)
                destroyed = True
                _log(f"destroy request sent for Vast instance {instance_id}")
            except BaseException as exc:
                cleanup_error = exc
                _log(f"destroy request failed for Vast instance {instance_id}: {_shorten(str(exc))}")
        if instance_id is not None:
            metadata = {
                "created_at": datetime.now(UTC).isoformat(),
                "instance_id": instance_id,
                "offer_id": config.offer_id,
                "image": config.image,
                "template_hash": config.template_hash,
                "disk_gb": config.disk_gb,
                "repo_ref": config.repo_ref,
                "remote_exit_code": remote_exit_code,
                "destroyed": destroyed,
                "cleanup_error": str(cleanup_error) if cleanup_error is not None else None,
            }
            (local_run_dir / "vast_run_metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        if cleanup_error is not None:
            raise cleanup_error
    if instance_id is None:
        raise RuntimeError("Vast instance was not created")
    return VastRunResult(
        instance_id=instance_id,
        local_run_dir=local_run_dir,
        remote_exit_code=remote_exit_code,
        destroyed=destroyed,
    )


def create_instance(config: VastRunConfig) -> int:
    command = ["vastai", "create", "instance", str(config.offer_id)]
    if config.template_hash:
        _log(f"using Vast template hash {config.template_hash}")
        command.extend(["--template_hash", config.template_hash])
    elif config.image:
        _log(f"using raw Docker image {config.image}")
        command.extend(["--image", config.image, "--ssh", "--direct"])
    else:
        raise ValueError("either template_hash or image must be configured")
    command.extend(
        [
            "--disk",
            str(config.disk_gb),
            "--cancel-unavail",
            "--raw",
        ]
    )
    output = _run_checked(
        command,
        cwd=config.project_root,
    )
    return parse_instance_id(output.stdout)


def wait_for_ssh(
    instance_id: int,
    *,
    poll_seconds: int,
    max_wait_seconds: int,
    max_auth_failures: int = DEFAULT_MAX_SSH_AUTH_FAILURES,
) -> list[str]:
    deadline = time.monotonic() + max_wait_seconds
    last_error = ""
    publickey_denials = 0
    _log(f"waiting up to {max_wait_seconds}s for SSH on Vast instance {instance_id}")
    while time.monotonic() < deadline:
        status = _run(["vastai", "show", "instance", str(instance_id), "--raw"])
        if status.returncode == 0:
            try:
                instance = _parse_cli_object(status.stdout)
                actual_status = instance.get("actual_status") or instance.get("cur_state")
                intended_status = instance.get("intended_status")
                status_msg = instance.get("status_msg")
            except ValueError:
                actual_status = None
                intended_status = None
                status_msg = None
            if intended_status == "stopped" and actual_status in {"loading", "stopped"}:
                detail = status_msg or "instance intended status became stopped before SSH was ready"
                raise RuntimeError(f"Vast instance {instance_id} is not starting: {detail}")
            _log(
                "poll: "
                f"status={actual_status or 'unknown'} "
                f"intended={intended_status or 'unknown'}"
            )
            ssh_url = _run(["vastai", "ssh-url", str(instance_id)])
            if ssh_url.returncode == 0 and ssh_url.stdout.strip():
                ssh_args = parse_ssh_args(ssh_url.stdout)
                probe = _run(ssh_command(ssh_args, "echo shapebench-ready"), timeout=20)
                if probe.returncode == 0:
                    _log("SSH is ready")
                    return ssh_args
                last_error = probe.stderr.strip() or probe.stdout.strip()
                if "Permission denied (publickey)" in last_error:
                    publickey_denials += 1
                    if publickey_denials >= max_auth_failures:
                        raise RuntimeError(
                            "Vast SSH public-key authentication failed repeatedly. "
                            "Stop this offer and use a Vast template or verify the account SSH key."
                        )
                else:
                    publickey_denials = 0
            else:
                last_error = ssh_url.stderr.strip()
            if actual_status:
                last_error = f"status={actual_status}; {last_error}".strip()
        else:
            last_error = status.stderr.strip()
        if last_error:
            _log(f"SSH not ready yet: {_shorten(last_error)}")
        time.sleep(poll_seconds)
    raise TimeoutError(f"Vast instance {instance_id} did not become SSH-ready: {last_error}")


def upload_git_archive(config: VastRunConfig, ssh_args: list[str]) -> None:
    remote_command = (
        f"rm -rf {shlex.quote(config.remote_dir)} && "
        f"mkdir -p {shlex.quote(config.remote_dir)} && "
        f"tar -xf - -C {shlex.quote(config.remote_dir)}"
    )
    archive = subprocess.Popen(
        ["git", "archive", config.repo_ref],
        cwd=config.project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )
    ssh = subprocess.Popen(
        ssh_command(ssh_args, remote_command),
        stdin=archive.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )
    if archive.stdout is not None:
        archive.stdout.close()
    _, archive_stderr = archive.communicate()
    ssh_stdout, ssh_stderr = ssh.communicate()
    if archive.returncode != 0:
        raise RuntimeError(f"git archive failed: {archive_stderr.decode(errors='replace')}")
    if ssh.returncode != 0:
        raise RuntimeError(
            "remote archive upload failed: "
            f"{ssh_stdout.decode(errors='replace')} {ssh_stderr.decode(errors='replace')}"
        )


def run_remote_eval(config: VastRunConfig, ssh_args: list[str], local_run_dir: Path) -> int:
    remote_script = build_remote_eval_script(config)
    log_path = local_run_dir / "remote_eval.log"
    command = ssh_command(ssh_args, f"bash -lc {shlex.quote(remote_script)}")
    _log(f"remote log: {log_path}")
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(f"[remote] {line}", end="", flush=True)
                log_file.write(line)
                log_file.flush()
            return process.wait()
        except BaseException:
            process.kill()
            process.wait()
            raise


def download_results(config: VastRunConfig, ssh_args: list[str], local_run_dir: Path) -> None:
    remote_command = (
        f"cd {shlex.quote(config.remote_dir)} && "
        "tar -czf - results/raw results/tables 2>/dev/null || true"
    )
    remote_tar = subprocess.Popen(
        ssh_command(ssh_args, remote_command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )
    extract = subprocess.Popen(
        ["tar", "-xzf", "-", "-C", str(local_run_dir)],
        stdin=remote_tar.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )
    if remote_tar.stdout is not None:
        remote_tar.stdout.close()
    _, remote_stderr = remote_tar.communicate()
    _, extract_stderr = extract.communicate()
    if remote_tar.returncode != 0:
        raise RuntimeError(f"remote results archive failed: {remote_stderr.decode(errors='replace')}")
    if extract.returncode != 0:
        raise RuntimeError(f"local results extraction failed: {extract_stderr.decode(errors='replace')}")


def destroy_instance(instance_id: int) -> None:
    completed = _run(["vastai", "destroy", "instance", str(instance_id), "--yes", "--raw"], timeout=120)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Vast destroy failed for instance {instance_id}: "
            f"{completed.stdout.strip()} {completed.stderr.strip()}".strip()
        )


def _destroy_instance_uninterruptible(instance_id: int) -> None:
    old_handler = signal.getsignal(signal.SIGINT)
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        destroy_instance(instance_id)
    finally:
        signal.signal(signal.SIGINT, old_handler)


def build_remote_eval_script(config: VastRunConfig) -> str:
    test_command = "" if config.skip_tests else "python -m pytest -q\n"
    return f"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export TORCH_EXTENSIONS_DIR=/tmp/shape_bench_torch_extensions
cd {shlex.quote(config.remote_dir)}
if ! command -v git >/dev/null 2>&1; then
  apt-get update
  apt-get install -y git
fi
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
nvidia-smi
nvcc --version
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
{test_command}python scripts/run_gpu_eval_batch.py
"""


def parse_instance_id(output: str) -> int:
    data = _parse_cli_object(output)
    for key in ("new_contract", "instance_id", "id"):
        value = data.get(key)
        if isinstance(value, int):
            return value
    raise ValueError(f"could not find instance id in Vast output: {output}")


def parse_ssh_args(output: str) -> list[str]:
    tokens = shlex.split(output.strip())
    if tokens and tokens[0] == "ssh":
        tokens = tokens[1:]
    if not tokens:
        raise ValueError("empty Vast SSH URL output")
    if len(tokens) == 1 and tokens[0].startswith("ssh://"):
        parsed = urlparse(tokens[0])
        if not parsed.hostname or parsed.port is None or not parsed.username:
            raise ValueError(f"could not parse Vast SSH URL: {tokens[0]}")
        return ["-p", str(parsed.port), f"{parsed.username}@{parsed.hostname}"]
    return tokens


def ssh_command(ssh_args: list[str], remote_command: str) -> list[str]:
    return ["ssh", *SSH_OPTIONS, *ssh_args, remote_command]


def _parse_cli_object(output: str) -> dict[str, Any]:
    text = output.strip()
    if not text:
        raise ValueError("empty Vast CLI output")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = ast.literal_eval(text)
    if isinstance(value, list):
        if len(value) != 1 or not isinstance(value[0], dict):
            raise ValueError(f"expected one object, got: {text}")
        return value[0]
    if not isinstance(value, dict):
        raise ValueError(f"expected object, got: {text}")
    return value


def _ensure_clean_repo(project_root: Path, *, allow_dirty: bool) -> None:
    if allow_dirty:
        return
    status = _run_checked(["git", "status", "--porcelain"], cwd=project_root)
    if status.stdout.strip():
        raise RuntimeError(
            "working tree has uncommitted changes; commit first or pass --allow-dirty "
            "if you intentionally want to archive only committed files"
        )


def _local_run_dir(project_root: Path, local_runs_dir: str) -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return project_root / local_runs_dir / timestamp


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[vast-runner {timestamp}] {message}", flush=True)


def _shorten(text: str, *, limit: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _run_checked(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    completed = _run(command, cwd=cwd)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed: {shlex.join(command)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return completed


def _run(
    command: list[str],
    *,
    cwd: Path | None = None,
    timeout: int | None = 60,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
