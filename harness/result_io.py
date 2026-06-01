"""JSON and JSONL helpers for ShapeBench-CUDA results."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def to_dict(record: Any) -> dict[str, Any]:
    """Convert a supported record object into a dictionary."""
    if hasattr(record, "to_dict"):
        return record.to_dict()
    if isinstance(record, dict):
        return dict(record)
    raise TypeError(f"object does not support to_dict conversion: {type(record).__name__}")


def from_dict(data: dict[str, Any], record_type: type[Any]) -> Any:
    """Create a record object from a dictionary."""
    if hasattr(record_type, "from_dict"):
        return record_type.from_dict(data)
    return record_type(**data)


def write_jsonl(path: str | Path, records: Iterable[Any]) -> None:
    """Write records to a JSONL file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(to_dict(record), sort_keys=True))
            handle.write("\n")


def read_jsonl(path: str | Path, record_type: type[Any] | None = None) -> list[Any]:
    """Read a JSONL file as dictionaries or typed records."""
    records: list[Any] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}") from exc
            records.append(from_dict(data, record_type) if record_type else data)
    return records

