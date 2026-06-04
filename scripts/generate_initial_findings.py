#!/usr/bin/env python
"""Generate the Stage 9 initial findings report from exported artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.experiment_analysis import (  # noqa: E402
    analyze_experiments,
    load_experiment_artifacts,
    load_task_names,
    render_initial_findings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiments-dir",
        default=ROOT / "results" / "experiments",
        type=Path,
        help="Directory containing exported experiment artifacts.",
    )
    parser.add_argument(
        "--tasks-dir",
        default=ROOT / "tasks",
        type=Path,
        help="Directory containing task metadata.",
    )
    parser.add_argument(
        "--output",
        default=ROOT / "report" / "initial_findings.md",
        type=Path,
        help="Markdown output path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts = load_experiment_artifacts(args.experiments_dir)
    task_names = load_task_names(args.tasks_dir)
    summary = analyze_experiments(artifacts, task_names=task_names)
    report = render_initial_findings(summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote initial findings report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
