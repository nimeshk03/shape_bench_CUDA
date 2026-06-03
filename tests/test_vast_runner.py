from __future__ import annotations

from pathlib import Path

import pytest

from harness import vast_runner
from harness.vast_runner import (
    VastRunConfig,
    build_remote_eval_script,
    create_instance,
    destroy_instance,
    parse_instance_id,
    parse_ssh_args,
    ssh_command,
)


def test_parse_instance_id_accepts_raw_json() -> None:
    assert parse_instance_id('{"success": true, "new_contract": 12345}') == 12345


def test_parse_instance_id_accepts_python_repr() -> None:
    assert parse_instance_id("{'success': True, 'new_contract': 67890}") == 67890


def test_parse_instance_id_rejects_missing_id() -> None:
    with pytest.raises(ValueError, match="instance id"):
        parse_instance_id('{"success": false}')


def test_parse_ssh_args_accepts_args_or_full_command() -> None:
    assert parse_ssh_args("-p 1234 root@example.com") == ["-p", "1234", "root@example.com"]
    assert parse_ssh_args("ssh -p 1234 root@example.com") == ["-p", "1234", "root@example.com"]


def test_parse_ssh_args_accepts_vast_ssh_url() -> None:
    assert parse_ssh_args("ssh://root@79.117.120.96:20182") == ["-p", "20182", "root@79.117.120.96"]


def test_ssh_command_disables_interactive_prompts() -> None:
    command = ssh_command(["-p", "20182", "root@79.117.120.96"], "echo ok")

    assert command[:2] == ["ssh", "-o"]
    assert "BatchMode=yes" in command
    assert "StrictHostKeyChecking=accept-new" in command
    assert "ConnectTimeout=10" in command
    assert command[-4:] == ["-p", "20182", "root@79.117.120.96", "echo ok"]


def test_build_remote_eval_script_runs_gpu_batch_and_tests() -> None:
    script = build_remote_eval_script(
        VastRunConfig(
            offer_id=1,
            project_root=Path("/tmp/project"),
            remote_dir="/root/project",
        )
    )

    assert "python -m pytest -q" in script
    assert "python scripts/run_gpu_eval_batch.py" in script
    assert "TORCH_EXTENSIONS_DIR=/tmp/shape_bench_torch_extensions" in script
    assert "torch.cuda.is_available()" in script


def test_build_remote_eval_script_can_skip_tests() -> None:
    script = build_remote_eval_script(
        VastRunConfig(
            offer_id=1,
            project_root=Path("/tmp/project"),
            skip_tests=True,
        )
    )

    assert "python -m pytest -q" not in script
    assert "python scripts/run_gpu_eval_batch.py" in script


def test_create_instance_asks_vast_to_cancel_unavailable(monkeypatch, tmp_path) -> None:
    commands: list[list[str]] = []

    def fake_run_checked(command, *, cwd=None):
        commands.append(command)

        class Result:
            stdout = '{"success": true, "new_contract": 123}'

        return Result()

    monkeypatch.setattr(vast_runner, "_run_checked", fake_run_checked)

    instance_id = create_instance(VastRunConfig(offer_id=99, project_root=tmp_path))

    assert instance_id == 123
    assert "--cancel-unavail" in commands[0]


def test_destroy_instance_skips_confirmation(monkeypatch) -> None:
    commands: list[list[str]] = []

    def fake_run(command, *, cwd=None, timeout=60):
        commands.append(command)

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(vast_runner, "_run", fake_run)

    destroy_instance(123)

    assert commands == [["vastai", "destroy", "instance", "123", "--yes", "--raw"]]


def test_destroy_instance_raises_when_cli_fails(monkeypatch) -> None:
    def fake_run(command, *, cwd=None, timeout=60):
        class Result:
            returncode = 1
            stdout = ""
            stderr = "cannot destroy"

        return Result()

    monkeypatch.setattr(vast_runner, "_run", fake_run)

    with pytest.raises(RuntimeError, match="cannot destroy"):
        destroy_instance(123)
