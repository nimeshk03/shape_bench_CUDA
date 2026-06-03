"""Prepare generated attempts for evaluation."""

from __future__ import annotations

import ast
import json
import keyword
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.task_loader import load_metadata


DEFAULT_ENTRYPOINT_FILE = "solution.py"
DEFAULT_ENTRYPOINT_FUNCTION = "forward"


@dataclass(frozen=True)
class AttemptContract:
    attempt_dir: Path
    extracted_dir: Path
    entrypoint_file: str
    entrypoint_function: str
    cuda_source: str | None
    extension_function: str | None
    extension_name: str | None
    task_id: str
    prompt_mode: str
    attempt: int
    input_names: list[str]
    created_fallback_solution: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_dir": str(self.attempt_dir),
            "extracted_dir": str(self.extracted_dir.relative_to(self.attempt_dir)),
            "entrypoint_file": self.entrypoint_file,
            "entrypoint_function": self.entrypoint_function,
            "cuda_source": self.cuda_source,
            "extension_function": self.extension_function,
            "extension_name": self.extension_name,
            "task_id": self.task_id,
            "prompt_mode": self.prompt_mode,
            "attempt": self.attempt,
            "input_names": self.input_names,
            "created_fallback_solution": self.created_fallback_solution,
        }


def prepare_attempt_contract(
    attempt_dir: str | Path,
    *,
    overwrite: bool = False,
    force_fallback: bool = False,
) -> AttemptContract:
    """Ensure an extracted attempt has a solution.py evaluation entrypoint."""
    attempt_path = Path(attempt_dir)
    extracted_dir = attempt_path / "extracted"
    manifest = _load_extraction_manifest(extracted_dir)
    attempt_metadata = _load_attempt_metadata(attempt_path)
    task_metadata = _load_task_metadata_for_attempt(attempt_path)
    files = manifest.get("files", [])
    if not isinstance(files, list):
        raise ValueError(f"invalid extracted manifest files list: {extracted_dir / 'manifest.json'}")

    cuda_source = _find_cuda_source(files)
    extension_function = _find_extension_function(extracted_dir, cuda_source)
    generated_extension_name = _extension_name(attempt_metadata)
    solution_path = extracted_dir / DEFAULT_ENTRYPOINT_FILE
    is_fallback_solution = solution_path.exists() and _is_generated_fallback(solution_path)
    created_fallback = is_fallback_solution

    if force_fallback:
        _write_fallback_solution(
            solution_path,
            cuda_source,
            extension_function,
            generated_extension_name,
            task_metadata["input_names"],
        )
        created_fallback = True
    elif solution_path.exists():
        if overwrite and is_fallback_solution:
            _write_fallback_solution(
                solution_path,
                cuda_source,
                extension_function,
                generated_extension_name,
                task_metadata["input_names"],
            )
            created_fallback = True
        elif not _has_top_level_forward(solution_path):
            if overwrite:
                _write_fallback_solution(
                    solution_path,
                    cuda_source,
                    extension_function,
                    generated_extension_name,
                    task_metadata["input_names"],
                )
                created_fallback = True
            else:
                raise ValueError(f"existing solution.py does not define top-level forward: {solution_path}")
    else:
        if cuda_source is None:
            raise ValueError(f"cannot create fallback solution.py without a CUDA source in {extracted_dir}")
        _write_fallback_solution(
            solution_path,
            cuda_source,
            extension_function,
            generated_extension_name,
            task_metadata["input_names"],
        )
        created_fallback = True

    extension_name = (
        generated_extension_name
        if created_fallback
        else _find_solution_extension_name(solution_path)
    )

    contract = AttemptContract(
        attempt_dir=attempt_path,
        extracted_dir=extracted_dir,
        entrypoint_file=DEFAULT_ENTRYPOINT_FILE,
        entrypoint_function=DEFAULT_ENTRYPOINT_FUNCTION,
        cuda_source=cuda_source,
        extension_function=extension_function,
        extension_name=extension_name,
        task_id=attempt_metadata["task_id"],
        prompt_mode=attempt_metadata["prompt_mode"],
        attempt=int(attempt_metadata["attempt"]),
        input_names=list(task_metadata["input_names"]),
        created_fallback_solution=created_fallback,
    )
    _write_contract(extracted_dir, contract)
    return contract


def _load_extraction_manifest(extracted_dir: Path) -> dict[str, Any]:
    manifest_path = extracted_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing extraction manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"extraction manifest must be a JSON object: {manifest_path}")
    return manifest


def _load_attempt_metadata(attempt_dir: Path) -> dict[str, Any]:
    metadata_path = attempt_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"missing attempt metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"attempt metadata must be a JSON object: {metadata_path}")
    for field in ("task_id", "prompt_mode", "attempt"):
        if field not in metadata:
            raise ValueError(f"attempt metadata missing {field!r}: {metadata_path}")
    return metadata


def _load_task_metadata_for_attempt(attempt_dir: Path) -> dict[str, Any]:
    metadata = _load_attempt_metadata(attempt_dir)
    task_dir = _project_root_from_attempt(attempt_dir) / "tasks" / metadata["task_id"]
    return load_metadata(task_dir)


def _find_cuda_source(files: list[Any]) -> str | None:
    cuda_files = [
        item.get("filename")
        for item in files
        if isinstance(item, dict)
        and isinstance(item.get("filename"), str)
        and item["filename"].endswith(".cu")
    ]
    unique = sorted(set(cuda_files))
    if len(unique) == 1:
        return unique[0]
    return None


def _write_fallback_solution(
    solution_path: Path,
    cuda_source: str | None,
    extension_function: str | None,
    extension_name: str,
    input_names: list[str],
) -> None:
    if cuda_source is None:
        raise ValueError("cuda_source is required for fallback solution.py")
    if extension_function is None:
        raise ValueError("extension_function is required for fallback solution.py")
    solution_path.write_text(
        _fallback_solution_source(cuda_source, extension_function, extension_name, input_names),
        encoding="utf-8",
    )


def _fallback_solution_source(
    cuda_source: str,
    extension_function: str,
    extension_name: str,
    input_names: list[str],
) -> str:
    _validate_python_identifiers(input_names)
    args = ", ".join(input_names)
    return f'''"""Fallback generated-attempt entrypoint.

This file was created by ShapeBench-CUDA because the generated attempt did not
provide an extracted solution.py. It loads the extracted CUDA source and exposes
forward(*inputs) for the evaluator contract.
"""

from __future__ import annotations

import pathlib

from torch import Tensor
from torch.utils.cpp_extension import load


_EXT = None


def _load_ext():
    global _EXT
    if _EXT is None:
        source = pathlib.Path(__file__).parent / "{cuda_source}"
        _EXT = load(
            name="{extension_name}",
            sources=[str(source)],
            verbose=False,
        )
    return _EXT


def forward({args}) -> Tensor:
    return getattr(_load_ext(), "{extension_function}")({args})
'''


def _is_generated_fallback(solution_path: Path) -> bool:
    return "Fallback generated-attempt entrypoint" in solution_path.read_text(encoding="utf-8")


def _write_contract(extracted_dir: Path, contract: AttemptContract) -> None:
    (extracted_dir / "eval_contract.json").write_text(
        json.dumps(contract.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _extension_name(attempt_metadata: dict[str, Any]) -> str:
    raw_name = (
        f"shapebench_{attempt_metadata['task_id']}_"
        f"{attempt_metadata['prompt_mode']}_attempt_{int(attempt_metadata['attempt']):03d}"
    )
    return re.sub(r"[^A-Za-z0-9_]", "_", raw_name)


def _project_root_from_attempt(attempt_dir: Path) -> Path:
    current = attempt_dir.resolve()
    for parent in (current, *current.parents):
        if (parent / "tasks").is_dir() and (parent / "generated").is_dir():
            return parent
    raise ValueError(f"could not locate project root from attempt directory: {attempt_dir}")


def _has_top_level_forward(solution_path: Path) -> bool:
    tree = ast.parse(solution_path.read_text(encoding="utf-8"), filename=str(solution_path))
    return any(isinstance(node, ast.FunctionDef) and node.name == DEFAULT_ENTRYPOINT_FUNCTION for node in tree.body)


def _find_solution_extension_name(solution_path: Path) -> str | None:
    tree = ast.parse(solution_path.read_text(encoding="utf-8"), filename=str(solution_path))
    names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_load_call(node.func):
            continue
        for keyword_arg in node.keywords:
            if keyword_arg.arg == "name" and isinstance(keyword_arg.value, ast.Constant):
                if isinstance(keyword_arg.value.value, str):
                    names.append(keyword_arg.value.value)
        if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            names.append(node.args[0].value)
    unique = sorted(set(names))
    if len(unique) == 1:
        return unique[0]
    return None


def _is_load_call(func: ast.expr) -> bool:
    if isinstance(func, ast.Name):
        return func.id == "load"
    return isinstance(func, ast.Attribute) and func.attr == "load"


def _find_extension_function(extracted_dir: Path, cuda_source: str | None) -> str | None:
    if cuda_source is None:
        return None
    cuda_path = extracted_dir / cuda_source
    if not cuda_path.is_file():
        return None
    matches = re.findall(
        r'm\.def\(\s*["\'](?P<name>[A-Za-z_][A-Za-z0-9_]*)["\']',
        cuda_path.read_text(encoding="utf-8"),
    )
    unique = sorted(set(matches))
    if len(unique) == 1:
        return unique[0]
    return None


def _validate_python_identifiers(input_names: list[str]) -> None:
    if not input_names:
        raise ValueError("input_names must be non-empty for fallback solution.py")
    for name in input_names:
        if not isinstance(name, str) or not name.isidentifier() or keyword.iskeyword(name):
            raise ValueError(f"input name is not a valid Python identifier: {name!r}")
