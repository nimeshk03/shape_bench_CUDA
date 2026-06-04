"""Experiment config loading for repeatable GPU batches."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    description: str
    attempts: tuple[str, ...]


def load_experiment_config(path: str | Path, *, project_root: str | Path) -> ExperimentConfig:
    root = Path(project_root).resolve()
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = root / config_path
    data = _read_config_json(config_path)

    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{config_path}: field 'name' must be a non-empty string")

    description = data.get("description", "")
    if not isinstance(description, str):
        raise ValueError(f"{config_path}: field 'description' must be a string")

    raw_attempts = data.get("attempts")
    if not isinstance(raw_attempts, list) or not raw_attempts:
        raise ValueError(f"{config_path}: field 'attempts' must be a non-empty list")

    attempts = []
    for index, attempt in enumerate(raw_attempts, start=1):
        if not isinstance(attempt, str) or not attempt.strip():
            raise ValueError(f"{config_path}: attempts[{index}] must be a non-empty string")
        attempts.append(_normalize_attempt_path(attempt, project_root=root, config_path=config_path, index=index))
    _validate_attempts(tuple(attempts), project_root=root, config_path=config_path)
    return ExperimentConfig(name=name, description=description, attempts=tuple(attempts))


def validate_experiment_config_in_git(
    config: ExperimentConfig,
    *,
    project_root: str | Path,
    repo_ref: str,
) -> None:
    """Verify configured attempts exist in the git ref that will be archived."""
    root = Path(project_root)
    for attempt in config.attempts:
        contract_path = f"{attempt}/extracted/eval_contract.json"
        if not _git_path_exists(contract_path, project_root=root, repo_ref=repo_ref):
            raise ValueError(
                f"experiment {config.name!r}: archived git ref {repo_ref!r} "
                f"does not contain {contract_path}; commit the prepared attempt "
                "or choose a matching --repo-ref"
            )


def _read_config_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing experiment config: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected JSON object")
    return data


def _normalize_attempt_path(attempt: str, *, project_root: Path, config_path: Path, index: int) -> str:
    attempt_path = Path(attempt)
    if attempt_path.is_absolute():
        raise ValueError(f"{config_path}: attempts[{index}] must be repo-relative, got absolute path: {attempt}")

    resolved_attempt = (project_root / attempt_path).resolve()
    try:
        relative_attempt = resolved_attempt.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"{config_path}: attempts[{index}] escapes project root: {attempt}") from exc

    normalized = relative_attempt.as_posix()
    if normalized == ".":
        raise ValueError(f"{config_path}: attempts[{index}] must point to an attempt directory")
    return normalized


def _validate_attempts(attempts: tuple[str, ...], *, project_root: Path, config_path: Path) -> None:
    seen: set[str] = set()
    for attempt in attempts:
        if attempt in seen:
            raise ValueError(f"{config_path}: duplicate attempt entry: {attempt}")
        seen.add(attempt)

        attempt_dir = Path(attempt)
        if not attempt_dir.is_absolute():
            attempt_dir = project_root / attempt_dir
        contract_path = attempt_dir / "extracted" / "eval_contract.json"
        if not contract_path.is_file():
            raise ValueError(f"{config_path}: attempt is not prepared for evaluation: {attempt}")


def _git_path_exists(path: str, *, project_root: Path, repo_ref: str) -> bool:
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{repo_ref}:{path}"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0
