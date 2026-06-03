"""Anthropic API generation helpers for ShapeBench-CUDA."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.prompt_renderer import PROJECT_ROOT, PromptMode, render_prompt
from harness.task_loader import load_task


ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_SMOKE_TEST_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_TEMPERATURE = 0.1
API_KEY_ENV_NAMES = ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY")


@dataclass(frozen=True)
class AnthropicGenerationConfig:
    task_dir: Path
    prompt_mode: PromptMode
    attempt: int
    model: str = DEFAULT_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    env_file: Path = PROJECT_ROOT / ".env.local"
    output_root: Path = PROJECT_ROOT / "generated"
    api_url: str = ANTHROPIC_API_URL
    timeout_seconds: float = 120.0

    @property
    def task_id(self) -> str:
        return load_task(self.task_dir).metadata["task_id"]


def load_env_file(path: str | Path) -> dict[str, str]:
    """Load simple KEY=VALUE lines from an env file without mutating os.environ."""
    env_path = Path(path)
    if not env_path.is_file():
        return {}

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            raise ValueError(f"invalid env line {line_number} in {env_path}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            raise ValueError(f"invalid env line {line_number} in {env_path}: empty key")
        values[key] = value
    return values


def get_anthropic_api_key(env_file: str | Path = PROJECT_ROOT / ".env.local") -> str:
    """Return the Anthropic API key from environment variables or an env file."""
    for env_name in API_KEY_ENV_NAMES:
        api_key = os.environ.get(env_name)
        if api_key:
            return api_key

    values = load_env_file(env_file)
    for env_name in API_KEY_ENV_NAMES:
        api_key = values.get(env_name)
        if api_key:
            return api_key

    raise RuntimeError(
        "Anthropic API key is not set. Add ANTHROPIC_API_KEY or CLAUDE_API_KEY "
        "to the environment or to .env.local. Do not commit .env.local."
    )


def build_anthropic_payload(
    prompt: str,
    *,
    model: str,
    max_tokens: int,
    temperature: float = DEFAULT_TEMPERATURE,
) -> dict[str, Any]:
    """Build the Anthropic Messages API request body."""
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if not 0.0 <= temperature <= 1.0:
        raise ValueError("temperature must be between 0.0 and 1.0")
    if not model.strip():
        raise ValueError("model must be non-empty")
    if not prompt.strip():
        raise ValueError("prompt must be non-empty")
    return {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }


def call_anthropic_messages(
    prompt: str,
    *,
    api_key: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    api_url: str = ANTHROPIC_API_URL,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Call Anthropic Messages API and return the decoded JSON response."""
    payload = build_anthropic_payload(
        prompt,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic API request failed with HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Anthropic API request failed: {exc.reason}") from exc

    decoded = json.loads(body)
    if not isinstance(decoded, dict):
        raise ValueError("Anthropic API response must be a JSON object")
    return decoded


def run_anthropic_smoke_test(
    *,
    env_file: str | Path = PROJECT_ROOT / ".env.local",
    model: str = DEFAULT_SMOKE_TEST_MODEL,
    api_url: str = ANTHROPIC_API_URL,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Run a tiny Anthropic request to validate API key and network access."""
    api_key = get_anthropic_api_key(env_file)
    return call_anthropic_messages(
        "Reply exactly with OK.",
        api_key=api_key,
        model=model,
        max_tokens=8,
        temperature=0.0,
        api_url=api_url,
        timeout_seconds=timeout_seconds,
    )


def extract_text_response(response: dict[str, Any]) -> str:
    """Extract concatenated text blocks from an Anthropic Messages API response."""
    content = response.get("content")
    if not isinstance(content, list):
        raise ValueError("Anthropic response missing content list")

    text_blocks: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str):
            text_blocks.append(block["text"])
    if not text_blocks:
        raise ValueError("Anthropic response did not contain text blocks")
    return "\n".join(text_blocks).strip()


def generation_attempt_dir(config: AnthropicGenerationConfig) -> Path:
    """Return the directory for one generated-code attempt."""
    return (
        config.output_root
        / config.prompt_mode
        / config.task_id
        / f"attempt_{config.attempt:03d}"
    )


def run_anthropic_generation(config: AnthropicGenerationConfig) -> Path:
    """Render a prompt, call Anthropic, and save generation artifacts."""
    if config.attempt <= 0:
        raise ValueError("attempt must be positive")

    prompt = render_prompt(config.task_dir, config.prompt_mode)
    api_key = get_anthropic_api_key(config.env_file)
    response = call_anthropic_messages(
        prompt,
        api_key=api_key,
        model=config.model,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        api_url=config.api_url,
        timeout_seconds=config.timeout_seconds,
    )
    response_text = extract_text_response(response)

    output_dir = generation_attempt_dir(config)
    write_generation_artifacts(
        output_dir,
        config=config,
        prompt=prompt,
        raw_response=response,
        response_text=response_text,
    )
    return output_dir


def write_generation_artifacts(
    output_dir: str | Path,
    *,
    config: AnthropicGenerationConfig,
    prompt: str,
    raw_response: dict[str, Any],
    response_text: str,
) -> None:
    """Write prompt, response, and metadata for one generation attempt."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metadata = {
        "task_id": config.task_id,
        "prompt_mode": config.prompt_mode,
        "attempt": config.attempt,
        "provider": "anthropic",
        "model": config.model,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "anthropic_version": ANTHROPIC_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "usage": raw_response.get("usage", {}),
        "stop_reason": raw_response.get("stop_reason"),
        "response_id": raw_response.get("id"),
    }

    (destination / "prompt.md").write_text(prompt, encoding="utf-8")
    (destination / "response.md").write_text(response_text + "\n", encoding="utf-8")
    (destination / "raw_response.json").write_text(
        json.dumps(raw_response, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
