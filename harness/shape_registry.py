"""Shape variant helpers for ShapeBench-CUDA tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


Shape = tuple[int, ...]
ShapeRegistry = dict[str, Shape]

REQUIRED_SHAPE_CATEGORIES = (
    "original",
    "smaller",
    "larger",
    "odd",
    "batch_variant",
    "non_power_of_two",
)


def validate_shape(shape: Sequence[int]) -> Shape:
    """Return a normalized shape tuple or raise ValueError."""
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
        raise ValueError(f"shape must be a sequence of positive integers, got {shape!r}")
    if len(shape) == 0:
        raise ValueError("shape must contain at least one dimension")

    normalized: list[int] = []
    for dim in shape:
        if isinstance(dim, bool) or not isinstance(dim, int):
            raise ValueError(f"shape dimensions must be integers, got {shape!r}")
        if dim <= 0:
            raise ValueError(f"shape dimensions must be positive, got {shape!r}")
        normalized.append(dim)
    return tuple(normalized)


def validate_shape_registry(registry: Mapping[str, Sequence[int]]) -> ShapeRegistry:
    """Validate and normalize a registry of named shape variants."""
    missing = [name for name in REQUIRED_SHAPE_CATEGORIES if name not in registry]
    if missing:
        raise ValueError(f"shape registry is missing categories: {', '.join(missing)}")

    normalized = {name: validate_shape(shape) for name, shape in registry.items()}
    ranks = {len(shape) for shape in normalized.values()}
    if len(ranks) != 1:
        raise ValueError(f"all shape variants must have the same rank, got ranks {sorted(ranks)}")
    return normalized


def default_vector_shapes(original: Sequence[int]) -> ShapeRegistry:
    """Create basic variants for a 1D tensor shape."""
    (n,) = _expect_rank(original, rank=1)
    return validate_shape_registry(
        {
            "original": (n,),
            "smaller": (_half(n),),
            "larger": (_double(n),),
            "odd": (_odd_near(n),),
            "batch_variant": (_quarter(n),),
            "non_power_of_two": (_non_power_of_two_near(n),),
        }
    )


def default_matrix_shapes(original: Sequence[int]) -> ShapeRegistry:
    """Create basic variants for a 2D matrix-like tensor shape."""
    rows, cols = _expect_rank(original, rank=2)
    return validate_shape_registry(
        {
            "original": (rows, cols),
            "smaller": (_half(rows), cols),
            "larger": (_double(rows), cols),
            "odd": (_odd_near(rows), _odd_near(cols)),
            "batch_variant": (_quarter(rows), cols),
            "non_power_of_two": (
                _non_power_of_two_near(rows),
                _non_power_of_two_near(cols),
            ),
        }
    )


def default_batch_feature_shapes(original: Sequence[int]) -> ShapeRegistry:
    """Create basic variants for a batch-feature tensor shape."""
    batch, features = _expect_rank(original, rank=2)
    return validate_shape_registry(
        {
            "original": (batch, features),
            "smaller": (_half(batch), features),
            "larger": (_double(batch), features),
            "odd": (_odd_near(batch), _odd_near(features)),
            "batch_variant": (_quarter(batch), features),
            "non_power_of_two": (
                _non_power_of_two_near(batch),
                _non_power_of_two_near(features),
            ),
        }
    )


def _expect_rank(shape: Sequence[int], rank: int) -> Shape:
    normalized = validate_shape(shape)
    if len(normalized) != rank:
        raise ValueError(f"expected rank {rank} shape, got {normalized}")
    return normalized


def _half(value: int) -> int:
    return max(1, value // 2)


def _quarter(value: int) -> int:
    return max(1, value // 4)


def _double(value: int) -> int:
    return max(1, value * 2)


def _odd_near(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def _non_power_of_two_near(value: int) -> int:
    if value <= 2:
        return 3
    if value & (value - 1) == 0:
        return value - 1
    return value

