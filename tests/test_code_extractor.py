from __future__ import annotations

import json

import pytest

from harness.code_extractor import (
    extract_attempt_code,
    infer_filename,
    infer_referenced_cuda_filename,
    parse_code_blocks,
)


def test_parse_code_blocks() -> None:
    markdown = """Text
```cpp
int main() { return 0; }
```
```python
# setup.py
setup()
```
"""

    blocks = parse_code_blocks(markdown)

    assert len(blocks) == 2
    assert blocks[0].language == "cpp"
    assert blocks[1].language == "python"


def test_infer_filenames() -> None:
    blocks = parse_code_blocks(
        """```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
```
```python
# setup.py
from setuptools import setup
```
```python
# solution.py
def forward(x): return x
```
"""
    )

    assert infer_filename(blocks[0]).filename == "extension.cu"
    assert infer_filename(blocks[1]).filename == "setup.py"
    assert infer_filename(blocks[2]).filename == "solution.py"


def test_infer_referenced_cuda_filename() -> None:
    blocks = parse_code_blocks(
        """```cpp
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
```
```python
# setup.py
CUDAExtension(name="x", sources=["add_relu.cu"])
```
```python
# solution.py
sources=[str(path / "add_relu.cu")]
```
"""
    )

    assert infer_referenced_cuda_filename(blocks) == "add_relu.cu"


def test_extract_attempt_code_writes_files_and_manifest(tmp_path) -> None:
    attempt_dir = tmp_path / "attempt_001"
    attempt_dir.mkdir()
    (attempt_dir / "response.md").write_text(
        """```cpp
__global__ void kernel() {}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
```
```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension
setup()
```
```python
# solution.py
def call():
    pass
```
""",
        encoding="utf-8",
    )

    extracted = extract_attempt_code(attempt_dir)

    assert [item.filename for item in extracted] == ["extension.cu", "setup.py", "solution.py"]
    assert (attempt_dir / "extracted" / "extension.cu").is_file()
    manifest = json.loads((attempt_dir / "extracted" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_response"] == "response.md"
    assert len(manifest["files"]) == 3


def test_extract_attempt_code_refuses_overwrite(tmp_path) -> None:
    attempt_dir = tmp_path / "attempt_001"
    attempt_dir.mkdir()
    (attempt_dir / "response.md").write_text("```cpp\n__global__ void kernel() {}\n```\n", encoding="utf-8")

    extract_attempt_code(attempt_dir)
    with pytest.raises(FileExistsError):
        extract_attempt_code(attempt_dir)

    extract_attempt_code(attempt_dir, overwrite=True)


def test_extract_attempt_code_overwrite_removes_stale_manifest_files(tmp_path) -> None:
    attempt_dir = tmp_path / "attempt_001"
    attempt_dir.mkdir()
    response_path = attempt_dir / "response.md"
    response_path.write_text("```cpp\n__global__ void kernel() {}\n```\n", encoding="utf-8")

    extract_attempt_code(attempt_dir)
    stale_file = attempt_dir / "extracted" / "extension.cu"
    assert stale_file.is_file()

    response_path.write_text(
        """```cpp
__global__ void kernel() {}
```
```python
# setup.py
CUDAExtension(name="x", sources=["add_relu.cu"])
```
""",
        encoding="utf-8",
    )
    extract_attempt_code(attempt_dir, overwrite=True)

    assert not stale_file.exists()
    assert (attempt_dir / "extracted" / "add_relu.cu").is_file()


def test_extract_attempt_code_uses_referenced_cuda_filename(tmp_path) -> None:
    attempt_dir = tmp_path / "attempt_001"
    attempt_dir.mkdir()
    (attempt_dir / "response.md").write_text(
        """```cpp
__global__ void kernel() {}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
```
```python
# setup.py
from torch.utils.cpp_extension import CUDAExtension
CUDAExtension(name="x", sources=["add_relu.cu"])
```
```python
# solution.py
sources=[str(pathlib.Path(__file__).parent / "add_relu.cu")]
```
""",
        encoding="utf-8",
    )

    extracted = extract_attempt_code(attempt_dir)

    assert [item.filename for item in extracted] == ["add_relu.cu", "setup.py", "solution.py"]
    assert (attempt_dir / "extracted" / "add_relu.cu").is_file()
