# ShapeBench-CUDA Initial Findings

## Motivation

ShapeBench-CUDA evaluates whether LLM-generated CUDA kernels remain correct and performant when tensor shapes change beyond the original benchmark shape.

The central comparison is baseline prompting versus shape-aware prompting.

## Experiment Setup

Exported experiment artifacts included in this report:

- `20260603T191105Z`: 12 attempts, 72/72 shape checks passed, tasks `task_002, task_003`, commit `63f68d4`
- `20260604T035522Z`: 30 attempts, 174/180 shape checks passed, tasks `task_004, task_005, task_006, task_007, task_008`, commit `da7385a`
- `20260604T081929Z`: 24 attempts, 138/144 shape checks passed, tasks `task_009, task_010, task_011, task_012`, commit `d2caa64`

Aggregate scope:

```text
Tasks analyzed: 11 (task_002, task_003, task_004, task_005, task_006, task_007, task_008, task_009, task_010, task_011, task_012)
Generated attempts: 66
Shape evaluations: 384/396 passed
Shape categories: original, smaller, larger, odd, batch_variant, non_power_of_two
GPU platform: Vast.ai RTX 4090 runs
```

## Task List

| Task | Name |
|---|---|
| `task_002` | rowwise_sum |
| `task_003` | matrix_transpose |
| `task_004` | matrix_multiply |
| `task_005` | rowwise_softmax |
| `task_006` | rowwise_layer_norm |
| `task_007` | broadcast_affine_clamp |
| `task_008` | batched_transpose |
| `task_009` | noncontiguous_affine_relu |
| `task_010` | dynamic_lastdim_sum_squares |
| `task_011` | batched_matrix_multiply |
| `task_012` | strided_batched_transpose |

## Metrics

| Metric | Meaning |
|---|---|
| Original pass rate | Fraction of attempts that pass the original benchmark shape. |
| Multi-shape pass rate | Fraction of attempts that pass every configured shape. |
| Robustness score | Fraction of all per-shape evaluations that pass. |
| Shape-variant-only failures | Attempts that pass original shape but fail at least one variant. |
| Mean/median speedup | Speedup versus PyTorch eager for correctness-passing rows only. |

## Prompt-Mode Results

| Prompt mode | Attempts | Original pass | Multi-shape pass | Robustness | Variant-only failures | Mean speedup | Median speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| `baseline` | 33 | 100.0% | 100.0% | 100.0% | 0 | 1.881 | 1.246 |
| `shape_aware` | 33 | 93.9% | 93.9% | 93.9% | 0 | 2.267 | 1.027 |

## Task And Prompt Breakdown

| Task | Prompt mode | Attempts | Checks passed | Original pass | Multi-shape pass | Mean speedup | Median speedup | Mean generated ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `task_002` | `baseline` | 3 | 18/18 | 3/3 | 3/3 | 0.930 | 0.917 | 0.007 |
| `task_002` | `shape_aware` | 3 | 18/18 | 3/3 | 3/3 | 0.740 | 0.870 | 0.012 |
| `task_003` | `baseline` | 3 | 18/18 | 3/3 | 3/3 | 3.067 | 2.463 | 0.009 |
| `task_003` | `shape_aware` | 3 | 18/18 | 3/3 | 3/3 | 3.175 | 2.443 | 0.009 |
| `task_004` | `baseline` | 3 | 18/18 | 3/3 | 3/3 | 0.557 | 0.571 | 0.013 |
| `task_004` | `shape_aware` | 3 | 18/18 | 3/3 | 3/3 | 0.748 | 0.794 | 0.011 |
| `task_005` | `baseline` | 3 | 18/18 | 3/3 | 3/3 | 0.535 | 0.516 | 0.012 |
| `task_005` | `shape_aware` | 3 | 18/18 | 3/3 | 3/3 | 0.623 | 0.685 | 0.011 |
| `task_006` | `baseline` | 3 | 18/18 | 3/3 | 3/3 | 5.624 | 4.812 | 0.007 |
| `task_006` | `shape_aware` | 3 | 18/18 | 3/3 | 3/3 | 5.396 | 4.695 | 0.007 |
| `task_007` | `baseline` | 3 | 18/18 | 3/3 | 3/3 | 2.461 | 2.423 | 0.005 |
| `task_007` | `shape_aware` | 3 | 18/18 | 3/3 | 3/3 | 2.430 | 2.404 | 0.005 |
| `task_008` | `baseline` | 3 | 18/18 | 3/3 | 3/3 | 2.721 | 1.914 | 0.007 |
| `task_008` | `shape_aware` | 3 | 12/18 | 2/3 | 2/3 | 2.112 | 1.863 | 0.010 |
| `task_009` | `baseline` | 3 | 18/18 | 3/3 | 3/3 | 1.443 | 1.753 | 0.278 |
| `task_009` | `shape_aware` | 3 | 18/18 | 3/3 | 3/3 | 6.578 | 1.641 | 0.018 |
| `task_010` | `baseline` | 3 | 18/18 | 3/3 | 3/3 | 1.564 | 1.382 | 0.012 |
| `task_010` | `shape_aware` | 3 | 18/18 | 3/3 | 3/3 | 0.759 | 0.846 | 0.031 |
| `task_011` | `baseline` | 3 | 18/18 | 3/3 | 3/3 | 0.866 | 0.876 | 0.019 |
| `task_011` | `shape_aware` | 3 | 18/18 | 3/3 | 3/3 | 0.732 | 0.651 | 0.021 |
| `task_012` | `baseline` | 3 | 18/18 | 3/3 | 3/3 | 0.919 | 0.969 | 0.040 |
| `task_012` | `shape_aware` | 3 | 12/18 | 2/3 | 2/3 | 1.251 | 1.017 | 0.161 |

## Failure Cases

| Run | Task | Prompt mode | Attempt | Original passed | Shapes passed | Failure reasons | Failed shapes | Max abs error |
|---|---|---|---:|---|---:|---|---|---:|
| `20260604T035522Z` | `task_008` | `shape_aware` | 1 | no | 0/6 | original_shape_correctness_failure: 1, shape_variant_correctness_failure: 5 | original, smaller, larger, odd, batch_variant, non_power_of_two | 8.475 |
| `20260604T081929Z` | `task_012` | `shape_aware` | 1 | no | 0/6 | original_shape_correctness_failure: 1, shape_variant_correctness_failure: 5 | original, smaller, larger, odd, batch_variant, non_power_of_two | 7.904 |

## Lessons Learned

- The current evidence does not show a robustness advantage for shape-aware prompting.
- Baseline attempts passed all exported shape evaluations in the current artifact set.
- Shape-aware failures so far failed the original shape too, so they are generated-code correctness failures rather than shape-variant-only failures.
- Performance varies strongly by task family; reductions and elementwise tasks can show speedups, while generated matmul-like kernels are often slower than PyTorch eager.
- Mean speedup can be distorted by microsecond-scale timing outliers, so median speedup should be reported alongside mean speedup.

## Next Steps

1. Add a generated CSV/Markdown table output for paper figures and quick review.
2. Repeat timing-sensitive batches to estimate run-to-run variance.
3. Add tasks that are more likely to create shape-variant-only failures, such as randomized shape sampling and stronger non-contiguous stride cases.
4. Inspect failed generated kernels to classify root causes beyond the current high-level failure taxonomy.
