#!/usr/bin/env python
"""Run a tiny Anthropic API smoke test."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.anthropic_generation import (  # noqa: E402
    DEFAULT_SMOKE_TEST_MODEL,
    extract_text_response,
    run_anthropic_smoke_test,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=str(ROOT / ".env.local"), help="Env file containing ANTHROPIC_API_KEY")
    parser.add_argument("--model", default=DEFAULT_SMOKE_TEST_MODEL, help="Anthropic model id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    response = run_anthropic_smoke_test(env_file=Path(args.env_file), model=args.model)
    text = extract_text_response(response)
    usage = response.get("usage", {})

    print("Anthropic API smoke test passed.")
    print(f"Model: {args.model}")
    print(f"Response: {text}")
    print(f"Usage: {usage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
