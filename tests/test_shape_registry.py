from __future__ import annotations

import pytest

from harness.shape_registry import (
    REQUIRED_SHAPE_CATEGORIES,
    default_batch_feature_shapes,
    default_matrix_shapes,
    default_vector_shapes,
    validate_shape,
    validate_shape_registry,
)


def test_validate_shape_normalizes_valid_shape() -> None:
    assert validate_shape([1024, 512]) == (1024, 512)


@pytest.mark.parametrize("shape", [[], [0], [-1], [True], ["1024"]])
def test_validate_shape_rejects_invalid_shape(shape: list[object]) -> None:
    with pytest.raises(ValueError):
        validate_shape(shape)  # type: ignore[arg-type]


def test_default_matrix_shapes_include_required_categories() -> None:
    shapes = default_matrix_shapes((1024, 1024))

    assert set(REQUIRED_SHAPE_CATEGORIES).issubset(shapes)
    assert shapes["original"] == (1024, 1024)
    assert shapes["smaller"] == (512, 1024)
    assert shapes["larger"] == (2048, 1024)
    assert shapes["odd"] == (1025, 1025)
    assert shapes["non_power_of_two"] == (1023, 1023)


def test_default_vector_shapes() -> None:
    shapes = default_vector_shapes((1024,))

    assert shapes["original"] == (1024,)
    assert shapes["batch_variant"] == (256,)


def test_default_batch_feature_shapes_changes_batch_dimension() -> None:
    shapes = default_batch_feature_shapes((128, 768))

    assert shapes["batch_variant"] == (32, 768)
    assert shapes["larger"] == (256, 768)


def test_validate_shape_registry_requires_categories_and_same_rank() -> None:
    valid = {name: (8, 8) for name in REQUIRED_SHAPE_CATEGORIES}
    assert validate_shape_registry(valid)["original"] == (8, 8)

    missing = dict(valid)
    missing.pop("odd")
    with pytest.raises(ValueError, match="missing categories"):
        validate_shape_registry(missing)

    mixed_rank = dict(valid)
    mixed_rank["odd"] = (7,)
    with pytest.raises(ValueError, match="same rank"):
        validate_shape_registry(mixed_rank)

