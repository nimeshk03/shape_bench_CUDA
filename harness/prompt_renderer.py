"""Render concrete LLM prompts from prompt modes and task definitions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from harness.task_loader import TaskDefinition, load_task


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PromptMode = Literal["baseline", "shape_aware"]

PROMPT_FILES: dict[PromptMode, str] = {
    "baseline": "baseline_prompt.md",
    "shape_aware": "shape_aware_prompt.md",
}


def render_prompt(
    task_dir: str | Path,
    prompt_mode: PromptMode,
    *,
    prompts_dir: str | Path | None = None,
) -> str:
    """Render a complete prompt for one task and one prompt mode."""
    task = load_task(task_dir)
    prompt_text = load_prompt_text(prompt_mode, prompts_dir=prompts_dir)
    model_code = task.model_path.read_text(encoding="utf-8")

    return "\n".join(
        [
            "# ShapeBench-CUDA Rendered Prompt",
            "",
            f"Prompt mode: `{prompt_mode}`",
            f"Task id: `{task.metadata['task_id']}`",
            "",
            "## Generation Instructions",
            "",
            prompt_text.strip(),
            "",
            "## Task Metadata",
            "",
            "```json",
            json.dumps(task.metadata, indent=2, sort_keys=True),
            "```",
            "",
            "## Shape Variants",
            "",
            "```json",
            _shape_registry_json(task),
            "```",
            "",
            "## PyTorch Reference Model",
            "",
            "The generated CUDA implementation should match `Model.forward` and `reference`.",
            "`create_inputs` is included only to show how local tests create inputs.",
            "",
            "```python",
            model_code.rstrip(),
            "```",
            "",
            "## Generated Code Expectations",
            "",
            "- Implement CUDA/C++ extension code that matches the PyTorch reference.",
            "- Keep the generated implementation self-contained for the harness.",
            "- Do not include unrelated prose, training code, or experiment orchestration.",
            "",
        ]
    )


def load_prompt_text(
    prompt_mode: PromptMode,
    *,
    prompts_dir: str | Path | None = None,
) -> str:
    """Load the generic prompt text for a prompt mode."""
    prompt_path = _prompt_path(prompt_mode, prompts_dir)
    if not prompt_path.is_file():
        raise FileNotFoundError(f"missing prompt file: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def write_rendered_prompt(
    task_dir: str | Path,
    prompt_mode: PromptMode,
    output_path: str | Path | None = None,
    *,
    prompts_dir: str | Path | None = None,
    output_dir: str | Path = "generated/prompts",
) -> Path:
    """Render and write a prompt, returning the output path."""
    task = load_task(task_dir)
    destination = Path(output_path) if output_path else default_output_path(task, prompt_mode, output_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        render_prompt(task.task_dir, prompt_mode, prompts_dir=prompts_dir),
        encoding="utf-8",
    )
    return destination


def default_output_path(
    task: TaskDefinition,
    prompt_mode: PromptMode,
    output_dir: str | Path = "generated/prompts",
) -> Path:
    """Return the default rendered prompt path for a task and prompt mode."""
    directory = Path(output_dir)
    if not directory.is_absolute():
        directory = PROJECT_ROOT / directory
    return directory / f"{task.metadata['task_id']}_{prompt_mode}.md"


def _prompt_path(prompt_mode: PromptMode, prompts_dir: str | Path | None) -> Path:
    if prompt_mode not in PROMPT_FILES:
        raise ValueError(f"unknown prompt mode: {prompt_mode}")
    directory = Path(prompts_dir) if prompts_dir is not None else PROJECT_ROOT / "prompts"
    return directory / PROMPT_FILES[prompt_mode]


def _shape_registry_json(task: TaskDefinition) -> str:
    shapes = {name: list(shape) for name, shape in task.shapes.items()}
    return json.dumps(shapes, indent=2, sort_keys=True)
