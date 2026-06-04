#!/usr/bin/env python
"""Export compact, versionable artifacts from a completed run directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.experiment_artifact import export_experiment_artifact  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="Run directory, for example results/vast_runs/20260603T191105Z")
    parser.add_argument("--output", help="Output directory. Defaults to results/experiments/<run_id>.")
    parser.add_argument(
        "--source-commit",
        help="Exact source commit for legacy run metadata that only recorded repo_ref=HEAD.",
    )
    parser.add_argument(
        "--exported-at",
        help="Optional export timestamp to include. Omit for deterministic artifacts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = export_experiment_artifact(
        args.run_dir,
        output_path=args.output,
        source_commit=args.source_commit,
        exported_at=args.exported_at,
    )
    print(f"Exported experiment artifact summary: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
