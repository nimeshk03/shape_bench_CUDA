"""Analysis helpers for exported ShapeBench-CUDA experiment artifacts."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SUMMARY_FILENAME = "summary.json"
RAW_RESULTS_FILENAME = "raw_results.jsonl"
ORIGINAL_SHAPE_CATEGORY = "original"


@dataclass(frozen=True)
class ExperimentArtifact:
    """Loaded compact experiment artifact."""

    run_id: str
    path: Path
    summary: dict[str, Any]
    raw_results: list[dict[str, Any]]

    @property
    def source_commit(self) -> str:
        value = self.summary.get("source_commit")
        return value if isinstance(value, str) else ""

    @property
    def gpu_name(self) -> str:
        preflight = self.summary.get("gpu_eval_batch_summary", {}).get("preflight", {})
        torch_info = preflight.get("torch", {})
        value = torch_info.get("gpu_name")
        return value if isinstance(value, str) else ""

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(sorted({str(row["task_id"]) for row in self.raw_results}))

    @property
    def attempt_keys(self) -> set[tuple[str, str, int | str]]:
        return {_attempt_key(row) for row in self.raw_results}


@dataclass(frozen=True)
class RateStats:
    """Aggregate prompt-mode rates."""

    prompt_mode: str
    attempts: int
    shape_checks: int
    passed_shape_checks: int
    original_passed: int
    original_total: int
    multi_shape_passed: int
    shape_variant_only_failures: int
    mean_speedup: float | None
    median_speedup: float | None

    @property
    def robustness_score(self) -> float:
        return _rate(self.passed_shape_checks, self.shape_checks)

    @property
    def original_pass_rate(self) -> float:
        return _rate(self.original_passed, self.original_total)

    @property
    def multi_shape_pass_rate(self) -> float:
        return _rate(self.multi_shape_passed, self.attempts)


@dataclass(frozen=True)
class TaskPromptStats:
    """Aggregate task and prompt-mode metrics."""

    task_id: str
    task_name: str
    prompt_mode: str
    attempts: int
    shape_checks: int
    passed_shape_checks: int
    original_passed: int
    original_total: int
    multi_shape_passed: int
    mean_speedup: float | None
    median_speedup: float | None
    mean_generated_ms: float | None


@dataclass(frozen=True)
class FailureCase:
    """One failed attempt summarized by shape categories."""

    run_id: str
    task_id: str
    task_name: str
    prompt_mode: str
    attempt: int | str
    original_passed: bool
    passed_shapes: int
    total_shapes: int
    failure_reasons: dict[str, int]
    failed_shape_categories: tuple[str, ...]
    max_abs_error: float | None


@dataclass(frozen=True)
class AnalysisSummary:
    """Full analysis used by the report renderer."""

    artifacts: tuple[ExperimentArtifact, ...]
    task_names: dict[str, str]
    prompt_stats: tuple[RateStats, ...]
    task_prompt_stats: tuple[TaskPromptStats, ...]
    failure_cases: tuple[FailureCase, ...]
    total_attempts: int
    total_shape_checks: int
    passed_shape_checks: int


def load_experiment_artifacts(experiments_dir: str | Path) -> tuple[ExperimentArtifact, ...]:
    """Load all exported experiment artifacts below ``experiments_dir``."""
    root = Path(experiments_dir)
    artifacts: list[ExperimentArtifact] = []
    for summary_path in sorted(root.glob(f"*/{SUMMARY_FILENAME}")):
        run_dir = summary_path.parent
        raw_path = run_dir / RAW_RESULTS_FILENAME
        if not raw_path.is_file():
            raise FileNotFoundError(f"missing raw result artifact: {raw_path}")
        summary = _read_json(summary_path)
        raw_results = _read_jsonl(raw_path)
        artifacts.append(
            ExperimentArtifact(
                run_id=run_dir.name,
                path=run_dir,
                summary=summary,
                raw_results=raw_results,
            )
        )
    if not artifacts:
        raise ValueError(f"no exported experiment artifacts found in {root}")
    return tuple(artifacts)


def load_task_names(tasks_dir: str | Path) -> dict[str, str]:
    """Load task display names from task metadata files."""
    names: dict[str, str] = {}
    for path in sorted(Path(tasks_dir).glob("task_*/metadata.json")):
        data = _read_json(path)
        task_id = data.get("task_id")
        name = data.get("name")
        if isinstance(task_id, str) and isinstance(name, str):
            names[task_id] = name
    return names


def analyze_experiments(
    artifacts: Iterable[ExperimentArtifact],
    *,
    task_names: dict[str, str] | None = None,
) -> AnalysisSummary:
    """Compute prompt, task, and failure summaries from artifacts."""
    artifact_tuple = tuple(artifacts)
    names = dict(task_names or {})
    rows = [row for artifact in artifact_tuple for row in artifact.raw_results]
    if not rows:
        raise ValueError("cannot analyze experiments with no raw result rows")

    prompt_groups = _group_by(rows, lambda row: str(row["prompt_mode"]))
    prompt_stats = tuple(
        _rate_stats(prompt_mode, group)
        for prompt_mode, group in sorted(prompt_groups.items())
    )

    task_prompt_groups = _group_by(rows, lambda row: (str(row["task_id"]), str(row["prompt_mode"])))
    task_prompt_stats = tuple(
        _task_prompt_stats(task_id, prompt_mode, group, names)
        for (task_id, prompt_mode), group in sorted(task_prompt_groups.items())
    )

    failure_cases = _failure_cases(artifact_tuple, names)
    return AnalysisSummary(
        artifacts=artifact_tuple,
        task_names=names,
        prompt_stats=prompt_stats,
        task_prompt_stats=task_prompt_stats,
        failure_cases=tuple(failure_cases),
        total_attempts=len({_attempt_key(row) for row in rows}),
        total_shape_checks=len(rows),
        passed_shape_checks=sum(1 for row in rows if row.get("correct") is True),
    )


def render_initial_findings(summary: AnalysisSummary) -> str:
    """Render a Markdown initial-findings report."""
    task_ids = sorted({row.task_id for row in summary.task_prompt_stats})
    experiment_lines = [
        f"- `{artifact.run_id}`: {len(artifact.attempt_keys)} attempts, "
        f"{_passed_count(artifact.raw_results)}/{len(artifact.raw_results)} shape checks passed, "
        f"tasks `{', '.join(artifact.task_ids)}`, commit `{artifact.source_commit[:7]}`"
        for artifact in summary.artifacts
    ]

    lines = [
        "# ShapeBench-CUDA Initial Findings",
        "",
        "## Motivation",
        "",
        (
            "ShapeBench-CUDA evaluates whether LLM-generated CUDA kernels remain "
            "correct and performant when tensor shapes change beyond the original "
            "benchmark shape."
        ),
        "",
        "The central comparison is baseline prompting versus shape-aware prompting.",
        "",
        "## Experiment Setup",
        "",
        "Exported experiment artifacts included in this report:",
        "",
        *experiment_lines,
        "",
        "Aggregate scope:",
        "",
        "```text",
        f"Tasks analyzed: {len(task_ids)} ({', '.join(task_ids)})",
        f"Generated attempts: {summary.total_attempts}",
        f"Shape evaluations: {summary.passed_shape_checks}/{summary.total_shape_checks} passed",
        "Shape categories: original, smaller, larger, odd, batch_variant, non_power_of_two",
        "GPU platform: Vast.ai RTX 4090 runs",
        "```",
        "",
        "## Task List",
        "",
        "| Task | Name |",
        "|---|---|",
    ]
    for task_id in task_ids:
        lines.append(f"| `{task_id}` | {summary.task_names.get(task_id, task_id)} |")

    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Metric | Meaning |",
            "|---|---|",
            "| Original pass rate | Fraction of attempts that pass the original benchmark shape. |",
            "| Multi-shape pass rate | Fraction of attempts that pass every configured shape. |",
            "| Robustness score | Fraction of all per-shape evaluations that pass. |",
            "| Shape-variant-only failures | Attempts that pass original shape but fail at least one variant. |",
            "| Mean/median speedup | Speedup versus PyTorch eager for correctness-passing rows only. |",
            "",
            "## Prompt-Mode Results",
            "",
            "| Prompt mode | Attempts | Original pass | Multi-shape pass | Robustness | Variant-only failures | Mean speedup | Median speedup |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.prompt_stats:
        lines.append(
            f"| `{row.prompt_mode}` | {row.attempts} | "
            f"{_pct(row.original_pass_rate)} | {_pct(row.multi_shape_pass_rate)} | "
            f"{_pct(row.robustness_score)} | {row.shape_variant_only_failures} | "
            f"{_fmt_number(row.mean_speedup)} | {_fmt_number(row.median_speedup)} |"
        )

    lines.extend(
        [
            "",
            "## Task And Prompt Breakdown",
            "",
            "| Task | Prompt mode | Attempts | Checks passed | Original pass | Multi-shape pass | Mean speedup | Median speedup | Mean generated ms |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary.task_prompt_stats:
        lines.append(
            f"| `{row.task_id}` | `{row.prompt_mode}` | {row.attempts} | "
            f"{row.passed_shape_checks}/{row.shape_checks} | "
            f"{row.original_passed}/{row.original_total} | "
            f"{row.multi_shape_passed}/{row.attempts} | "
            f"{_fmt_number(row.mean_speedup)} | {_fmt_number(row.median_speedup)} | "
            f"{_fmt_number(row.mean_generated_ms)} |"
        )

    lines.extend(
        [
            "",
            "## Failure Cases",
            "",
        ]
    )
    if summary.failure_cases:
        lines.extend(
            [
                "| Run | Task | Prompt mode | Attempt | Original passed | Shapes passed | Failure reasons | Failed shapes | Max abs error |",
                "|---|---|---|---:|---|---:|---|---|---:|",
            ]
        )
        for case in summary.failure_cases:
            reasons = ", ".join(f"{key}: {value}" for key, value in sorted(case.failure_reasons.items()))
            failed_shapes = ", ".join(case.failed_shape_categories)
            lines.append(
                f"| `{case.run_id}` | `{case.task_id}` | `{case.prompt_mode}` | {case.attempt} | "
                f"{'yes' if case.original_passed else 'no'} | "
                f"{case.passed_shapes}/{case.total_shapes} | {reasons} | {failed_shapes} | "
                f"{_fmt_number(case.max_abs_error)} |"
            )
    else:
        lines.append("No correctness failures were observed in the exported artifacts.")

    lines.extend(
        [
            "",
            "## Lessons Learned",
            "",
            "- The current evidence does not show a robustness advantage for shape-aware prompting.",
            "- Baseline attempts passed all exported shape evaluations in the current artifact set.",
            "- Shape-aware failures so far failed the original shape too, so they are generated-code correctness failures rather than shape-variant-only failures.",
            "- Performance varies strongly by task family; reductions and elementwise tasks can show speedups, while generated matmul-like kernels are often slower than PyTorch eager.",
            "- Mean speedup can be distorted by microsecond-scale timing outliers, so median speedup should be reported alongside mean speedup.",
            "",
            "## Next Steps",
            "",
            "1. Add a generated CSV/Markdown table output for paper figures and quick review.",
            "2. Repeat timing-sensitive batches to estimate run-to-run variance.",
            "3. Add tasks that are more likely to create shape-variant-only failures, such as randomized shape sampling and stronger non-contiguous stride cases.",
            "4. Inspect failed generated kernels to classify root causes beyond the current high-level failure taxonomy.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object: {path}")
    return data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"expected JSON object in {path}:{line_number}")
        records.append(record)
    return records


def _group_by(rows: Iterable[dict[str, Any]], key_fn) -> dict[Any, list[dict[str, Any]]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    return groups


def _rate_stats(prompt_mode: str, rows: list[dict[str, Any]]) -> RateStats:
    attempt_groups = _group_by(rows, _attempt_key)
    original_rows = [row for row in rows if row.get("shape_category") == ORIGINAL_SHAPE_CATEGORY]
    return RateStats(
        prompt_mode=prompt_mode,
        attempts=len(attempt_groups),
        shape_checks=len(rows),
        passed_shape_checks=_passed_count(rows),
        original_passed=_passed_count(original_rows),
        original_total=len(original_rows),
        multi_shape_passed=_multi_shape_passed(attempt_groups.values()),
        shape_variant_only_failures=_shape_variant_only_failures(attempt_groups.values()),
        mean_speedup=_mean(_correct_numeric_values(rows, "speedup_vs_eager")),
        median_speedup=_median(_correct_numeric_values(rows, "speedup_vs_eager")),
    )


def _task_prompt_stats(
    task_id: str,
    prompt_mode: str,
    rows: list[dict[str, Any]],
    task_names: dict[str, str],
) -> TaskPromptStats:
    attempt_groups = _group_by(rows, _attempt_key)
    original_rows = [row for row in rows if row.get("shape_category") == ORIGINAL_SHAPE_CATEGORY]
    return TaskPromptStats(
        task_id=task_id,
        task_name=task_names.get(task_id, task_id),
        prompt_mode=prompt_mode,
        attempts=len(attempt_groups),
        shape_checks=len(rows),
        passed_shape_checks=_passed_count(rows),
        original_passed=_passed_count(original_rows),
        original_total=len(original_rows),
        multi_shape_passed=_multi_shape_passed(attempt_groups.values()),
        mean_speedup=_mean(_correct_numeric_values(rows, "speedup_vs_eager")),
        median_speedup=_median(_correct_numeric_values(rows, "speedup_vs_eager")),
        mean_generated_ms=_mean(_correct_numeric_values(rows, "generated_ms")),
    )


def _failure_cases(
    artifacts: tuple[ExperimentArtifact, ...],
    task_names: dict[str, str],
) -> list[FailureCase]:
    cases: list[FailureCase] = []
    for artifact in artifacts:
        attempt_groups = _group_by(artifact.raw_results, _attempt_key)
        for (task_id, prompt_mode, attempt), rows in sorted(attempt_groups.items()):
            failed_rows = [row for row in rows if row.get("correct") is not True]
            if not failed_rows:
                continue
            original_rows = [row for row in rows if row.get("shape_category") == ORIGINAL_SHAPE_CATEGORY]
            max_errors = _numeric_values(failed_rows, "max_abs_error")
            cases.append(
                FailureCase(
                    run_id=artifact.run_id,
                    task_id=str(task_id),
                    task_name=task_names.get(str(task_id), str(task_id)),
                    prompt_mode=str(prompt_mode),
                    attempt=attempt,
                    original_passed=bool(original_rows and original_rows[0].get("correct") is True),
                    passed_shapes=_passed_count(rows),
                    total_shapes=len(rows),
                    failure_reasons=dict(Counter(str(row.get("failure_reason")) for row in failed_rows)),
                    failed_shape_categories=tuple(str(row["shape_category"]) for row in failed_rows),
                    max_abs_error=max(max_errors) if max_errors else None,
                )
            )
    return cases


def _attempt_key(row: dict[str, Any]) -> tuple[str, str, int | str]:
    extra = row.get("extra")
    attempt = extra.get("attempt") if isinstance(extra, dict) else None
    if not isinstance(attempt, (int, str)):
        attempt = "unknown"
    return (str(row["task_id"]), str(row["prompt_mode"]), attempt)


def _passed_count(rows: Iterable[dict[str, Any]]) -> int:
    return sum(1 for row in rows if row.get("correct") is True)


def _multi_shape_passed(attempt_groups: Iterable[list[dict[str, Any]]]) -> int:
    return sum(1 for rows in attempt_groups if rows and all(row.get("correct") is True for row in rows))


def _shape_variant_only_failures(attempt_groups: Iterable[list[dict[str, Any]]]) -> int:
    count = 0
    for rows in attempt_groups:
        original_rows = [row for row in rows if row.get("shape_category") == ORIGINAL_SHAPE_CATEGORY]
        if not original_rows or original_rows[0].get("correct") is not True:
            continue
        if any(row.get("correct") is not True for row in rows if row.get("shape_category") != ORIGINAL_SHAPE_CATEGORY):
            count += 1
    return count


def _correct_numeric_values(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    return _numeric_values((row for row in rows if row.get("correct") is True), key)


def _numeric_values(rows: Iterable[dict[str, Any]], key: str) -> list[float]:
    values = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return values


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _fmt_number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"
