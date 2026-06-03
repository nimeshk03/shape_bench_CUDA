"""Extract generated code blocks from LLM response markdown."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


Confidence = Literal["high", "medium", "low"]

CODE_BLOCK_RE = re.compile(r"```(?P<label>[^\n`]*)\n(?P<code>.*?)```", re.DOTALL)
CU_REFERENCE_RE = re.compile(r"['\"](?P<filename>[A-Za-z0-9_.-]+\.cu)['\"]")


@dataclass(frozen=True)
class CodeBlock:
    index: int
    language: str
    code: str


@dataclass(frozen=True)
class FileInference:
    filename: str
    reason: str
    confidence: Confidence


@dataclass(frozen=True)
class ExtractedFile:
    block_index: int
    language: str
    filename: str
    reason: str
    confidence: Confidence
    path: Path

    def to_manifest_entry(self, root: Path) -> dict[str, str | int]:
        return {
            "block_index": self.block_index,
            "language": self.language,
            "filename": self.filename,
            "path": str(self.path.relative_to(root)),
            "reason": self.reason,
            "confidence": self.confidence,
        }


def parse_code_blocks(markdown: str) -> list[CodeBlock]:
    """Parse fenced Markdown code blocks."""
    blocks: list[CodeBlock] = []
    for index, match in enumerate(CODE_BLOCK_RE.finditer(markdown), start=1):
        label = match.group("label").strip().lower()
        language = label.split(maxsplit=1)[0] if label else "text"
        blocks.append(
            CodeBlock(
                index=index,
                language=language,
                code=match.group("code").strip() + "\n",
            )
        )
    return blocks


def infer_filename(block: CodeBlock, *, preferred_cuda_filename: str | None = None) -> FileInference:
    """Infer a filename for a generated code block."""
    first_line = _first_nonempty_line(block.code).lower()
    code_lower = block.code.lower()

    explicit = _filename_from_comment(first_line)
    if explicit:
        return FileInference(explicit, "filename indicated by leading comment", "high")

    if block.language in {"cpp", "c++", "cuda", "cu"}:
        if preferred_cuda_filename:
            return FileInference(
                preferred_cuda_filename,
                "CUDA filename referenced by generated Python/build block",
                "high",
            )
        if "pybind11_module" in code_lower or "__global__" in code_lower:
            return FileInference("extension.cu", "C++/CUDA extension block", "high")
        return FileInference("source.cpp", "C++ block without stronger filename hint", "low")

    if block.language == "python":
        if "setup(" in code_lower and "cudaextension" in code_lower:
            return FileInference("setup.py", "Python setup/CUDAExtension block", "high")
        if "torch.utils.cpp_extension" in code_lower or "def forward" in code_lower or "class model" in code_lower:
            return FileInference("solution.py", "Python wrapper/model block", "medium")
        return FileInference("snippet.py", "Python block without stronger filename hint", "low")

    return FileInference(f"block_{block.index:02d}.txt", "unsupported language block", "low")


def extract_attempt_code(
    attempt_dir: str | Path,
    *,
    overwrite: bool = False,
    output_subdir: str = "extracted",
) -> list[ExtractedFile]:
    """Extract code blocks from an attempt response.md into files."""
    attempt_path = Path(attempt_dir)
    response_path = attempt_path / "response.md"
    if not response_path.is_file():
        raise FileNotFoundError(f"missing response file: {response_path}")

    blocks = parse_code_blocks(response_path.read_text(encoding="utf-8"))
    if not blocks:
        raise ValueError(f"no fenced code blocks found in {response_path}")

    output_dir = attempt_path / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        _remove_previous_extracted_files(output_dir)

    preferred_cuda_filename = infer_referenced_cuda_filename(blocks)
    extracted: list[ExtractedFile] = []
    used_names: set[str] = set()
    for block in blocks:
        inference = infer_filename(block, preferred_cuda_filename=preferred_cuda_filename)
        filename = _dedupe_filename(inference.filename, used_names)
        used_names.add(filename)
        destination = output_dir / filename
        if destination.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing file: {destination}")
        destination.write_text(block.code, encoding="utf-8")
        extracted.append(
            ExtractedFile(
                block_index=block.index,
                language=block.language,
                filename=filename,
                reason=inference.reason,
                confidence=inference.confidence,
                path=destination,
            )
        )

    write_manifest(attempt_path, output_dir, extracted, response_path)
    return extracted


def infer_referenced_cuda_filename(blocks: list[CodeBlock]) -> str | None:
    """Infer a CUDA filename referenced by generated Python/setup blocks."""
    filenames: list[str] = []
    for block in blocks:
        if block.language != "python":
            continue
        for match in CU_REFERENCE_RE.finditer(block.code):
            filenames.append(match.group("filename"))

    unique = sorted(set(filenames))
    if len(unique) == 1:
        return unique[0]
    return None


def write_manifest(
    attempt_dir: Path,
    output_dir: Path,
    extracted: list[ExtractedFile],
    response_path: Path,
) -> Path:
    """Write an extraction manifest."""
    manifest = {
        "attempt_dir": str(attempt_dir),
        "source_response": str(response_path.relative_to(attempt_dir)),
        "output_dir": str(output_dir.relative_to(attempt_dir)),
        "files": [item.to_manifest_entry(attempt_dir) for item in extracted],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _filename_from_comment(line: str) -> str | None:
    normalized = line.removeprefix("#").removeprefix("//").strip()
    if normalized.startswith(("setup.py", "solution.py")):
        return normalized.split(maxsplit=1)[0]
    return None


def _dedupe_filename(filename: str, used_names: set[str]) -> str:
    if filename not in used_names:
        return filename
    path = Path(filename)
    stem = path.stem
    suffix = path.suffix
    counter = 2
    while True:
        candidate = f"{stem}_{counter}{suffix}"
        if candidate not in used_names:
            return candidate
        counter += 1


def _remove_previous_extracted_files(output_dir: Path) -> None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in manifest.get("files", []):
        path_value = item.get("path")
        if not isinstance(path_value, str):
            continue
        candidate = output_dir.parent / path_value
        if candidate.is_file():
            candidate.unlink()
