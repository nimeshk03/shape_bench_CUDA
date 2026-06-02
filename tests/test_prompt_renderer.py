from __future__ import annotations

from pathlib import Path

from harness.prompt_renderer import PROJECT_ROOT, render_prompt, write_rendered_prompt


ROOT = Path(__file__).resolve().parents[1]
TASK_DIR = ROOT / "tasks" / "task_001"


def test_render_baseline_prompt_includes_task_context() -> None:
    rendered = render_prompt(TASK_DIR, "baseline")

    assert "Prompt mode: `baseline`" in rendered
    assert '"task_id": "task_001"' in rendered
    assert "elementwise_add_relu" in rendered
    assert "class Model" in rendered
    assert '"original"' in rendered
    assert "create_inputs` is included only to show how local tests create inputs" in rendered
    assert "shape variants" not in rendered.split("## Generation Instructions", 1)[1].split(
        "## Task Metadata", 1
    )[0].lower()


def test_render_shape_aware_prompt_includes_shape_robustness_instructions() -> None:
    rendered = render_prompt(TASK_DIR, "shape_aware")

    assert "Prompt mode: `shape_aware`" in rendered
    assert "hardcoded dimensions" in rendered.lower()
    assert "non-power-of-two" in rendered.lower()
    assert '"odd"' in rendered
    assert "1007" in rendered
    assert "1013" in rendered


def test_write_rendered_prompt_uses_default_name(tmp_path) -> None:
    output_path = write_rendered_prompt(TASK_DIR, "baseline", output_dir=tmp_path)

    assert output_path == tmp_path / "task_001_baseline.md"
    assert output_path.is_file()
    assert "ShapeBench-CUDA Rendered Prompt" in output_path.read_text(encoding="utf-8")


def test_write_rendered_prompt_default_output_is_project_relative() -> None:
    output_path = write_rendered_prompt(TASK_DIR, "baseline")

    assert output_path == PROJECT_ROOT / "generated" / "prompts" / "task_001_baseline.md"
