from __future__ import annotations

import json

import pytest

from harness.attempt_contract import prepare_attempt_contract


def test_prepare_attempt_contract_keeps_existing_solution(tmp_path) -> None:
    root = _make_project_root(tmp_path)
    attempt_dir = root / "generated" / "baseline" / "task_001" / "attempt_001"
    _write_attempt_metadata(attempt_dir, prompt_mode="baseline", attempt=1)
    extracted_dir = attempt_dir / "extracted"
    extracted_dir.mkdir(parents=True)
    (extracted_dir / "solution.py").write_text(
        'from torch.utils.cpp_extension import load\n'
        '_ext = load(name="custom_existing_ext", sources=[])\n'
        "def forward(x, y):\n"
        "    return _ext.add_relu(x, y)\n",
        encoding="utf-8",
    )
    _write_cuda_source(extracted_dir / "add_relu.cu", extension_function="add_relu")
    (extracted_dir / "manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {"filename": "add_relu.cu"},
                    {"filename": "solution.py"},
                ]
            }
        ),
        encoding="utf-8",
    )

    contract = prepare_attempt_contract(attempt_dir)

    assert contract.created_fallback_solution is False
    assert contract.cuda_source == "add_relu.cu"
    assert contract.extension_function == "add_relu"
    assert contract.extension_name == "custom_existing_ext"
    assert contract.input_names == ["x", "y"]
    contract_json = json.loads((extracted_dir / "eval_contract.json").read_text(encoding="utf-8"))
    assert contract_json["task_id"] == "task_001"
    assert contract_json["prompt_mode"] == "baseline"
    assert contract_json["attempt"] == 1
    assert contract_json["extension_name"] == "custom_existing_ext"


def test_prepare_attempt_contract_records_existing_solution_extension_function(tmp_path) -> None:
    root = _make_project_root(tmp_path)
    attempt_dir = root / "generated" / "shape_aware" / "task_001" / "attempt_007"
    _write_attempt_metadata(attempt_dir, prompt_mode="shape_aware", attempt=7)
    extracted_dir = attempt_dir / "extracted"
    extracted_dir.mkdir(parents=True)
    (extracted_dir / "solution.py").write_text(
        'from torch.utils.cpp_extension import load\n'
        '_ext = load(name="multi_ext", sources=[])\n'
        "def forward(x, y):\n"
        "    ext = _ext\n"
        "    return ext.add_relu(x, y)\n",
        encoding="utf-8",
    )
    (extracted_dir / "multi.cu").write_text(
        """
#include <torch/extension.h>

torch::Tensor add_relu(torch::Tensor x, torch::Tensor y) {
    return torch::relu(x + y);
}

torch::Tensor debug_add(torch::Tensor x, torch::Tensor y) {
    return x + y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_relu", &add_relu);
    m.def("debug_add", &debug_add);
}
""",
        encoding="utf-8",
    )
    (extracted_dir / "manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {"filename": "multi.cu"},
                    {"filename": "solution.py"},
                ]
            }
        ),
        encoding="utf-8",
    )

    contract = prepare_attempt_contract(attempt_dir)

    contract_json = json.loads((extracted_dir / "eval_contract.json").read_text(encoding="utf-8"))
    assert contract.created_fallback_solution is False
    assert contract.extension_function == "add_relu"
    assert contract_json["extension_function"] == "add_relu"


def test_prepare_attempt_contract_creates_fallback_solution(tmp_path) -> None:
    root = _make_project_root(tmp_path)
    attempt_dir = root / "generated" / "shape_aware" / "task_001" / "attempt_002"
    _write_attempt_metadata(attempt_dir, prompt_mode="shape_aware", attempt=2)
    extracted_dir = attempt_dir / "extracted"
    extracted_dir.mkdir(parents=True)
    _write_cuda_source(extracted_dir / "add_relu.cu", extension_function="add_relu")
    (extracted_dir / "manifest.json").write_text(
        json.dumps({"files": [{"filename": "add_relu.cu"}]}),
        encoding="utf-8",
    )

    contract = prepare_attempt_contract(attempt_dir)

    solution = (extracted_dir / "solution.py").read_text(encoding="utf-8")
    contract_json = json.loads((extracted_dir / "eval_contract.json").read_text(encoding="utf-8"))
    assert contract.created_fallback_solution is True
    assert '"add_relu.cu"' in solution
    assert 'name="shapebench_task_001_shape_aware_attempt_002"' in solution
    assert "def forward" in solution
    assert 'getattr(_load_ext(), "add_relu")(x, y)' in solution
    assert contract_json["entrypoint_file"] == "solution.py"
    assert contract_json["entrypoint_function"] == "forward"
    assert contract_json["input_names"] == ["x", "y"]
    assert contract_json["extension_function"] == "add_relu"
    assert contract_json["extension_name"] == "shapebench_task_001_shape_aware_attempt_002"


def test_prepare_attempt_contract_can_force_fallback_over_existing_solution(tmp_path) -> None:
    root = _make_project_root(tmp_path)
    attempt_dir = root / "generated" / "shape_aware" / "task_001" / "attempt_006"
    _write_attempt_metadata(attempt_dir, prompt_mode="shape_aware", attempt=6)
    extracted_dir = attempt_dir / "extracted"
    extracted_dir.mkdir(parents=True)
    (extracted_dir / "solution.py").write_text(
        "import missing_prebuilt_ext\n\n"
        "def forward(x, y):\n"
        "    return missing_prebuilt_ext.add_relu(x, y)\n",
        encoding="utf-8",
    )
    _write_cuda_source(extracted_dir / "add_relu.cu", extension_function="add_relu")
    (extracted_dir / "manifest.json").write_text(
        json.dumps(
            {
                "files": [
                    {"filename": "add_relu.cu"},
                    {"filename": "solution.py"},
                ]
            }
        ),
        encoding="utf-8",
    )

    contract = prepare_attempt_contract(attempt_dir, force_fallback=True)

    solution = (extracted_dir / "solution.py").read_text(encoding="utf-8")
    contract_json = json.loads((extracted_dir / "eval_contract.json").read_text(encoding="utf-8"))
    assert contract.created_fallback_solution is True
    assert "missing_prebuilt_ext" not in solution
    assert "Fallback generated-attempt entrypoint" in solution
    assert 'name="shapebench_task_001_shape_aware_attempt_006"' in solution
    assert contract_json["created_fallback_solution"] is True
    assert contract_json["extension_name"] == "shapebench_task_001_shape_aware_attempt_006"


def test_prepare_attempt_contract_preserves_fallback_status_when_rerun(tmp_path) -> None:
    root = _make_project_root(tmp_path)
    attempt_dir = root / "generated" / "baseline" / "task_001" / "attempt_003"
    _write_attempt_metadata(attempt_dir, prompt_mode="baseline", attempt=3)
    extracted_dir = attempt_dir / "extracted"
    extracted_dir.mkdir(parents=True)
    _write_cuda_source(extracted_dir / "add_relu.cu", extension_function="add_relu")
    (extracted_dir / "manifest.json").write_text(
        json.dumps({"files": [{"filename": "add_relu.cu"}]}),
        encoding="utf-8",
    )

    first_contract = prepare_attempt_contract(attempt_dir)
    second_contract = prepare_attempt_contract(attempt_dir)

    contract_json = json.loads((extracted_dir / "eval_contract.json").read_text(encoding="utf-8"))
    assert first_contract.created_fallback_solution is True
    assert second_contract.created_fallback_solution is True
    assert contract_json["created_fallback_solution"] is True
    assert contract_json["extension_name"] == "shapebench_task_001_baseline_attempt_003"


def test_prepare_attempt_contract_rejects_solution_without_top_level_forward(tmp_path) -> None:
    root = _make_project_root(tmp_path)
    attempt_dir = root / "generated" / "baseline" / "task_001" / "attempt_003"
    _write_attempt_metadata(attempt_dir, prompt_mode="baseline", attempt=3)
    extracted_dir = attempt_dir / "extracted"
    extracted_dir.mkdir(parents=True)
    (extracted_dir / "solution.py").write_text(
        "class Model:\n    def forward(self, x, y):\n        return x\n",
        encoding="utf-8",
    )
    (extracted_dir / "manifest.json").write_text(
        json.dumps({"files": [{"filename": "solution.py"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="top-level forward"):
        prepare_attempt_contract(attempt_dir)


def test_prepare_attempt_contract_generates_fallback_from_task_metadata(tmp_path) -> None:
    root = _make_project_root(tmp_path, input_names=["left", "right", "bias"])
    attempt_dir = root / "generated" / "shape_aware" / "task_001" / "attempt_004"
    _write_attempt_metadata(attempt_dir, prompt_mode="shape_aware", attempt=4)
    extracted_dir = attempt_dir / "extracted"
    extracted_dir.mkdir(parents=True)
    _write_cuda_source(extracted_dir / "fused.cu", extension_function="fused_add_bias")
    (extracted_dir / "manifest.json").write_text(
        json.dumps({"files": [{"filename": "fused.cu"}]}),
        encoding="utf-8",
    )

    contract = prepare_attempt_contract(attempt_dir)

    solution = (extracted_dir / "solution.py").read_text(encoding="utf-8")
    contract_json = json.loads((extracted_dir / "eval_contract.json").read_text(encoding="utf-8"))
    assert contract.created_fallback_solution is True
    assert "def forward(left, right, bias) -> Tensor:" in solution
    assert 'getattr(_load_ext(), "fused_add_bias")(left, right, bias)' in solution
    assert contract_json["input_names"] == ["left", "right", "bias"]
    assert contract_json["cuda_source"] == "fused.cu"
    assert contract_json["extension_function"] == "fused_add_bias"


def test_prepare_attempt_contract_rejects_keyword_input_names_before_writing_fallback(tmp_path) -> None:
    root = _make_project_root(tmp_path, input_names=["class"])
    attempt_dir = root / "generated" / "baseline" / "task_001" / "attempt_005"
    _write_attempt_metadata(attempt_dir, prompt_mode="baseline", attempt=5)
    extracted_dir = attempt_dir / "extracted"
    extracted_dir.mkdir(parents=True)
    _write_cuda_source(extracted_dir / "add_relu.cu", extension_function="add_relu")
    (extracted_dir / "manifest.json").write_text(
        json.dumps({"files": [{"filename": "add_relu.cu"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="valid Python identifier"):
        prepare_attempt_contract(attempt_dir)

    assert not (extracted_dir / "solution.py").exists()


def _make_project_root(tmp_path, *, input_names: list[str] | None = None):
    task_dir = tmp_path / "tasks" / "task_001"
    task_dir.mkdir(parents=True)
    (tmp_path / "generated").mkdir()
    (task_dir / "metadata.json").write_text(
        json.dumps(
            {
                "task_id": "task_001",
                "name": "demo",
                "description": "demo task",
                "category": "elementwise",
                "input_kind": "matrix",
                "input_names": input_names or ["x", "y"],
                "original_shape": [8, 8],
                "dtype": "float32",
                "atol": 1e-4,
                "rtol": 1e-4,
                "expected_output": "same shape tensor",
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def _write_cuda_source(path, *, extension_function: str) -> None:
    path.write_text(
        f"""
#include <torch/extension.h>

torch::Tensor {extension_function}(torch::Tensor x, torch::Tensor y) {{
    return x + y;
}}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {{
    m.def("{extension_function}", &{extension_function});
}}
""",
        encoding="utf-8",
    )


def _write_attempt_metadata(attempt_dir, *, prompt_mode: str, attempt: int) -> None:
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "metadata.json").write_text(
        json.dumps(
            {
                "task_id": "task_001",
                "prompt_mode": prompt_mode,
                "attempt": attempt,
                "provider": "anthropic",
                "model": "claude-sonnet-4-6",
            }
        ),
        encoding="utf-8",
    )
