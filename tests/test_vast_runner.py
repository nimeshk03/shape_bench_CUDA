from __future__ import annotations

from pathlib import Path

import pytest

from harness.vast_runner import VastRunConfig, build_remote_eval_script, parse_instance_id, parse_ssh_args


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
