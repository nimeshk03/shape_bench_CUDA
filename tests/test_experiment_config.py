from __future__ import annotations

import json

import pytest

from harness import experiment_config
from harness.experiment_config import load_experiment_config, validate_experiment_config_in_git


def test_load_experiment_config_returns_validated_attempts(tmp_path) -> None:
    first = _make_prepared_attempt(tmp_path, "generated/baseline/task_002/attempt_001")
    second = _make_prepared_attempt(tmp_path, "generated/shape_aware/task_002/attempt_001")
    config_path = tmp_path / "configs" / "experiment.json"
    config_path.parent.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "name": "demo",
                "description": "demo batch",
                "attempts": [str(first.relative_to(tmp_path)), str(second.relative_to(tmp_path))],
            }
        ),
        encoding="utf-8",
    )

    config = load_experiment_config(config_path, project_root=tmp_path)

    assert config.name == "demo"
    assert config.description == "demo batch"
    assert config.attempts == (
        "generated/baseline/task_002/attempt_001",
        "generated/shape_aware/task_002/attempt_001",
    )


def test_load_experiment_config_normalizes_repo_relative_attempts(tmp_path) -> None:
    _make_prepared_attempt(tmp_path, "generated/baseline/task_002/attempt_001")
    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "demo",
                "attempts": ["./generated/../generated/baseline/task_002/attempt_001"],
            }
        ),
        encoding="utf-8",
    )

    config = load_experiment_config(config_path, project_root=tmp_path)

    assert config.attempts == ("generated/baseline/task_002/attempt_001",)


def test_load_experiment_config_rejects_absolute_attempt_path(tmp_path) -> None:
    attempt = _make_prepared_attempt(tmp_path, "generated/baseline/task_002/attempt_001")
    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "demo",
                "attempts": [str(attempt)],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be repo-relative"):
        load_experiment_config(config_path, project_root=tmp_path)


def test_load_experiment_config_rejects_attempt_path_outside_repo(tmp_path) -> None:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    _make_prepared_attempt(tmp_path, "outside/attempt_001")
    config_path = project_root / "experiment.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "demo",
                "attempts": ["../outside/attempt_001"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="escapes project root"):
        load_experiment_config(config_path, project_root=project_root)


def test_load_experiment_config_rejects_unprepared_attempt(tmp_path) -> None:
    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "demo",
                "attempts": ["generated/baseline/task_002/attempt_001"],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="attempt is not prepared for evaluation"):
        load_experiment_config(config_path, project_root=tmp_path)


def test_load_experiment_config_rejects_duplicate_attempts(tmp_path) -> None:
    _make_prepared_attempt(tmp_path, "generated/baseline/task_002/attempt_001")
    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        json.dumps(
            {
                "name": "demo",
                "attempts": [
                    "generated/baseline/task_002/attempt_001",
                    "generated/../generated/baseline/task_002/attempt_001",
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate attempt entry"):
        load_experiment_config(config_path, project_root=tmp_path)


def test_validate_experiment_config_in_git_checks_archived_contracts(monkeypatch, tmp_path) -> None:
    config = experiment_config.ExperimentConfig(
        name="demo",
        description="",
        attempts=("generated/baseline/task_002/attempt_001",),
    )
    checked_paths: list[tuple[str, str]] = []

    def fake_git_path_exists(path, *, project_root, repo_ref):
        checked_paths.append((repo_ref, path))
        return True

    monkeypatch.setattr(experiment_config, "_git_path_exists", fake_git_path_exists)

    validate_experiment_config_in_git(config, project_root=tmp_path, repo_ref="abc123")

    assert checked_paths == [("abc123", "generated/baseline/task_002/attempt_001/extracted/eval_contract.json")]


def test_validate_experiment_config_in_git_rejects_missing_archived_contract(monkeypatch, tmp_path) -> None:
    config = experiment_config.ExperimentConfig(
        name="demo",
        description="",
        attempts=("generated/baseline/task_002/attempt_001",),
    )
    monkeypatch.setattr(experiment_config, "_git_path_exists", lambda path, *, project_root, repo_ref: False)

    with pytest.raises(ValueError, match="archived git ref 'HEAD' does not contain"):
        validate_experiment_config_in_git(config, project_root=tmp_path, repo_ref="HEAD")


def _make_prepared_attempt(tmp_path, relative_path: str):
    attempt_dir = tmp_path / relative_path
    extracted_dir = attempt_dir / "extracted"
    extracted_dir.mkdir(parents=True)
    (extracted_dir / "eval_contract.json").write_text("{}", encoding="utf-8")
    return attempt_dir
