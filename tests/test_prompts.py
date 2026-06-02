from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = ROOT / "prompts"


def test_prompt_files_exist() -> None:
    assert (PROMPTS_DIR / "baseline_prompt.md").is_file()
    assert (PROMPTS_DIR / "shape_aware_prompt.md").is_file()


def test_baseline_prompt_emphasizes_performance_and_correctness() -> None:
    text = (PROMPTS_DIR / "baseline_prompt.md").read_text(encoding="utf-8").lower()

    assert "cuda/c++ extension" in text
    assert "numerical correctness" in text
    assert "performance" in text
    assert "shape variants" not in text
    assert "hardcoded dimensions" not in text


def test_shape_aware_prompt_emphasizes_shape_robustness() -> None:
    text = (PROMPTS_DIR / "shape_aware_prompt.md").read_text(encoding="utf-8").lower()

    assert "shape variants" in text
    assert "runtime tensor shape" in text
    assert "hardcoded dimensions" in text
    assert "odd dimensions" in text
    assert "non-power-of-two" in text
    assert "boundary checks" in text
