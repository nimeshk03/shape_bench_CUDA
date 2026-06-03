#!/usr/bin/env python
"""Generate CUDA code with Anthropic Claude from a rendered ShapeBench prompt."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.anthropic_generation import (  # noqa: E402
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    AnthropicGenerationConfig,
    generation_attempt_dir,
    run_anthropic_generation,
)
from harness.prompt_renderer import PROMPT_FILES  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-dir", required=True, help="Task directory, for example tasks/task_001")
    parser.add_argument("--mode", required=True, choices=sorted(PROMPT_FILES), help="Prompt mode")
    parser.add_argument("--attempt", type=int, default=1, help="Positive attempt number")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Anthropic model id")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS, help="Maximum output tokens")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help="Sampling temperature")
    parser.add_argument("--env-file", default=str(ROOT / ".env.local"), help="Env file containing ANTHROPIC_API_KEY")
    parser.add_argument("--output-root", default=str(ROOT / "generated"), help="Generated output root")
    parser.add_argument("--dry-run", action="store_true", help="Print target directory without calling the API")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = AnthropicGenerationConfig(
        task_dir=Path(args.task_dir),
        prompt_mode=args.mode,
        attempt=args.attempt,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        env_file=Path(args.env_file),
        output_root=Path(args.output_root),
    )

    if args.dry_run:
        print(f"Dry run: would write generation artifacts to {generation_attempt_dir(config)}")
        return 0

    output_dir = run_anthropic_generation(config)
    print(f"Generation artifacts written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
