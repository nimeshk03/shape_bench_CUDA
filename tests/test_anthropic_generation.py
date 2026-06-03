from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.anthropic_generation import (
    ANTHROPIC_VERSION,
    DEFAULT_MODEL,
    DEFAULT_SMOKE_TEST_MODEL,
    DEFAULT_TEMPERATURE,
    AnthropicGenerationConfig,
    build_anthropic_payload,
    extract_text_response,
    generation_attempt_dir,
    get_anthropic_api_key,
    load_env_file,
    write_generation_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "tasks" / "task_001"


def test_default_models_are_current() -> None:
    assert DEFAULT_MODEL == "claude-sonnet-4-6"
    assert DEFAULT_SMOKE_TEST_MODEL == "claude-haiku-4-5-20251001"
    assert DEFAULT_TEMPERATURE == 0.1


def test_build_anthropic_payload() -> None:
    payload = build_anthropic_payload("hello", model="claude-sonnet-4-20250514", max_tokens=128)

    assert payload["model"] == "claude-sonnet-4-20250514"
    assert payload["max_tokens"] == 128
    assert payload["temperature"] == 0.1
    assert payload["messages"] == [{"role": "user", "content": "hello"}]


def test_build_anthropic_payload_rejects_empty_prompt() -> None:
    with pytest.raises(ValueError, match="prompt"):
        build_anthropic_payload("", model="claude-sonnet-4-20250514", max_tokens=128)


def test_build_anthropic_payload_rejects_invalid_temperature() -> None:
    with pytest.raises(ValueError, match="temperature"):
        build_anthropic_payload("hello", model="claude-sonnet-4-20250514", max_tokens=128, temperature=1.1)


def test_load_env_file_and_get_key(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    env_file = tmp_path / ".env.local"
    env_file.write_text("ANTHROPIC_API_KEY = 'test-key'\nOTHER=value\n", encoding="utf-8")

    assert load_env_file(env_file)["ANTHROPIC_API_KEY"] == "test-key"
    assert get_anthropic_api_key(env_file) == "test-key"


def test_get_key_accepts_claude_api_key_alias(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("CLAUDE_API_KEY", raising=False)
    env_file = tmp_path / ".env.local"
    env_file.write_text("export CLAUDE_API_KEY = alias-key\n", encoding="utf-8")

    assert get_anthropic_api_key(env_file) == "alias-key"


def test_environment_variable_takes_precedence(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    env_file = tmp_path / ".env.local"
    env_file.write_text("ANTHROPIC_API_KEY=file-key\n", encoding="utf-8")

    assert get_anthropic_api_key(env_file) == "env-key"


def test_extract_text_response() -> None:
    response = {
        "content": [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
    }

    assert extract_text_response(response) == "first\nsecond"


def test_generation_attempt_dir() -> None:
    config = AnthropicGenerationConfig(
        task_dir=TASK_DIR,
        prompt_mode="baseline",
        attempt=2,
        output_root=Path("generated"),
    )

    assert generation_attempt_dir(config) == Path("generated") / "baseline" / "task_001" / "attempt_002"


def test_write_generation_artifacts(tmp_path) -> None:
    config = AnthropicGenerationConfig(
        task_dir=TASK_DIR,
        prompt_mode="shape_aware",
        attempt=1,
        temperature=0.2,
        output_root=tmp_path,
    )
    raw_response = {
        "id": "msg_test",
        "content": [{"type": "text", "text": "code"}],
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "stop_reason": "end_turn",
    }

    write_generation_artifacts(
        tmp_path / "attempt_001",
        config=config,
        prompt="prompt",
        raw_response=raw_response,
        response_text="code",
    )

    metadata = json.loads((tmp_path / "attempt_001" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["provider"] == "anthropic"
    assert metadata["anthropic_version"] == ANTHROPIC_VERSION
    assert metadata["temperature"] == 0.2
    assert metadata["usage"] == {"input_tokens": 10, "output_tokens": 5}
    assert (tmp_path / "attempt_001" / "prompt.md").read_text(encoding="utf-8") == "prompt"
    assert (tmp_path / "attempt_001" / "response.md").read_text(encoding="utf-8") == "code\n"
