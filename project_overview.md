# ShapeBench-CUDA Project Overview

## Summary

ShapeBench-CUDA is a lightweight research project for evaluating whether LLM-generated CUDA kernels are robust across input shape variations.

Recent systems such as CUDA Agent show that language-model agents can generate high-performance CUDA kernels through a compile, verify, profile, and iterate workflow. ShapeBench-CUDA focuses on a narrower follow-up question:

> Do LLM-generated CUDA kernels remain correct and performant under realistic input shape variations?

The initial goal is not to train a new model or reproduce CUDA Agent's full reinforcement learning pipeline. The goal is to build a small benchmark and evaluation harness for shape robustness.

## Motivation

LLM-generated CUDA kernels may perform well for one benchmark shape while failing or slowing down when dimensions change. Common risks include:

- hardcoded tensor sizes
- assumptions about power-of-two dimensions
- missing boundary checks
- fixed indexing patterns
- poor behavior on odd, smaller, larger, or batch-varied shapes
- correctness tests that are too narrow

For real use, a generated CUDA kernel should be fast and correct across reasonable shape variations, not only the original benchmark input.

## Main Research Question

> How well do LLM-generated CUDA kernels generalize across input shape variations, and can simple shape-aware prompting improve their robustness?

Sub-questions:

- What percentage of generated kernels pass correctness checks on the original benchmark shape?
- What percentage continue to pass on varied shapes?
- How stable is speedup across shapes?
- What kinds of shape changes cause failures?
- Does explicitly prompting for variable-shape support improve generalization?
- Is there a tradeoff between original-shape performance and multi-shape robustness?

## Relationship to CUDA Agent

CUDA Agent demonstrates an LLM-driven CUDA development loop:

1. Inspect a PyTorch model.
2. Write CUDA/C++ extension code.
3. Compile the generated implementation.
4. Verify numerical correctness.
5. Profile runtime performance.
6. Iteratively improve the implementation.

ShapeBench-CUDA reuses the general compile/verify/profile idea, but evaluates a different property:

```text
CUDA Agent:       Can an agent generate high-performance CUDA kernels?
ShapeBench-CUDA: Are generated kernels robust when tensor shapes change?
```

The project is complementary to CUDA Agent, not a reproduction of it.

## Proposed Evaluation Pipeline

```text
Selected PyTorch task
        ↓
Generate CUDA implementation
        ↓
Compile CUDA extension
        ↓
Run correctness test on original shape
        ↓
Run correctness tests on shape variants
        ↓
Benchmark against PyTorch eager and/or torch.compile
        ↓
Save structured results
        ↓
Analyze pass rate, speedup, and failure modes
```

## Shape Variation Strategy

Each task should define shape variants such as:

- `original`: the benchmark shape
- `smaller`: reduced workload with compatible dimensions
- `larger`: increased workload
- `odd`: dimensions that stress boundary handling
- `non_power_of_two`: dimensions that break power-of-two assumptions
- `batch_variant`: changed batch size with similar feature dimensions

Example:

```json
{
  "original": [1024, 1024],
  "smaller": [512, 1024],
  "larger": [2048, 1024],
  "odd": [1007, 1013],
  "batch_variant": [256, 1024]
}
```

Later extensions may test non-contiguous tensors, different dtypes, adversarial values, numerical stability, and randomized shapes.

## Prompting Modes

The MVP compares two prompt modes.

Baseline prompt:

- asks for the fastest correct CUDA implementation
- follows normal CUDA optimization constraints
- does not add special shape-generalization instructions

Shape-aware prompt:

- explicitly asks for dynamic shape handling
- forbids hardcoded dimensions
- requires odd and non-power-of-two support
- requires boundary checks
- asks to preserve performance where possible

## Evaluation Metrics

Core metrics:

- `original_shape_pass_rate`: fraction of kernels correct on the original shape
- `multi_shape_pass_rate`: fraction correct on every tested shape
- `shape_robustness_score`: fraction of shape variants passed by a kernel
- `speedup_vs_eager`: PyTorch eager runtime divided by generated runtime
- `speedup_vs_compile`: `torch.compile` runtime divided by generated runtime, when available
- `speedup_stability`: minimum speedup across shapes divided by maximum speedup across shapes
- `failure_reason`: categorized failure mode

Example shape robustness score:

```text
passed_shape_variants / total_shape_variants
```

If a kernel passes 3 of 5 shape variants, its score is `0.60`.

## Failure Categories

Failures should be categorized when possible:

- compilation failure
- original-shape correctness failure
- shape-variant correctness failure
- runtime error
- out-of-bounds memory access
- incorrect indexing
- assumes fixed dimension
- assumes power-of-two size
- boundary condition bug
- performance regression
- timeout

## Initial Experiment Design

The first experiment should stay small:

```text
5-10 tasks
2 prompt modes
3-5 shape variants per task
1 AWS GPU instance
```

Recommended AWS instances:

```text
g5.xlarge   NVIDIA A10G 24GB
g6.xlarge   NVIDIA L4 24GB
g6.2xlarge  NVIDIA L4 24GB
```

Avoid expensive A100/H100-class instances during Phase 1.

Expected result table:

| Prompt Mode | Original Pass Rate | Multi-Shape Pass Rate | Avg Robustness Score | Avg Speedup |
| --- | ---: | ---: | ---: | ---: |
| Baseline | TBD | TBD | TBD | TBD |
| Shape-aware | TBD | TBD | TBD | TBD |

## Result Format

Each per-shape run should be saved as structured JSON/JSONL.

Example:

```json
{
  "task_id": "example_task_001",
  "prompt_mode": "shape_aware",
  "gpu_name": "NVIDIA A10G",
  "cuda_version": "12.x",
  "torch_version": "2.x",
  "shape": [1024, 1024],
  "shape_category": "original",
  "correct": true,
  "max_abs_error": 0.0003,
  "mean_abs_error": 0.00001,
  "pytorch_eager_ms": 1.42,
  "torch_compile_ms": 0.91,
  "generated_ms": 0.73,
  "speedup_vs_eager": 1.95,
  "speedup_vs_compile": 1.25,
  "failure_reason": null
}
```

## Expected Contributions

The project should produce:

- a lightweight benchmark for shape generalization in LLM-generated CUDA kernels
- a correctness and performance harness for multiple shape variants
- an empirical comparison of baseline and shape-aware prompts
- a taxonomy of common shape-generalization failures
- initial evidence about whether robustness-focused prompting improves generated kernel reliability

## Phase 1 Success Criteria

Minimum viable result:

```text
10 tasks x 2 prompt modes x 3 shape variants
```

Strong result:

```text
20-30 tasks x 2 prompt modes x 5 shape variants
```

Phase 1 is successful if it produces:

- a working local CPU-compatible harness
- an AWS GPU environment ready for CUDA runs
- selected tasks with original shapes and variants
- baseline and shape-aware results
- summary tables comparing robustness and speed
- a short report explaining observed failure modes

## Future Extensions

After the MVP, possible directions include:

- bottleneck-aware feedback for correctness and performance failures
- lightweight profiler-guided optimization
- robust evaluation with non-contiguous tensors, dtypes, adversarial values, NaN/Inf handling, randomized shapes, and multiple GPU architectures
- domain-specific CUDA benchmarks, such as packet feature extraction, entropy calculation, Bloom filter lookup, string matching, log aggregation, or graph-based risk scoring
