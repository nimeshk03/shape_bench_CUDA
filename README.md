# ShapeBench-CUDA

ShapeBench-CUDA is a lightweight research project for evaluating whether LLM-generated CUDA kernels remain correct and performant across input shape variations.

The Phase 1 goal is a local, CPU-compatible harness plus AWS-ready CUDA scripts. Local development should not require an NVIDIA GPU.

## Research Question

How well do LLM-generated CUDA kernels generalize across input shape variations, and can shape-aware prompting improve robustness?

The MVP compares normal CUDA-generation prompts against shape-aware prompts across original, smaller, larger, odd, non-power-of-two, and batch-varied tensor shapes.

## Local Setup

Create and activate the project environment:

```bash
conda env create -f environment.yml
conda activate shapebench-cuda
```

Run the local setup check:

```bash
python scripts/check_local_setup.py
```

CUDA availability may be `False` locally. That is expected.

Run local tests:

```bash
pytest
```

## Current Harness

The local harness currently includes:

- `harness/shape_registry.py`: default shape variants and validation.
- `harness/result_schema.py`: dataclasses for per-shape results and task summaries.
- `harness/result_io.py`: JSONL read/write helpers.
- `harness/task_loader.py`: task metadata and shape loading.
- `harness/cuda_checks.py`: CUDA checks that fail gracefully on CPU-only machines.
- `harness/compare_outputs.py`: tensor-like output comparison.
- `harness/run_benchmark.py`: simple callable timing helper.

These pieces are intentionally CPU-compatible. CUDA compilation and benchmarking are expected to run later on an AWS GPU instance.

## Initial Benchmark Task

The first local benchmark task is:

- `tasks/task_001`: elementwise `relu(x + y)` for two same-shaped tensors.

It includes task metadata, fixed shape variants, a PyTorch `Model`, deterministic input generation, and a functional reference. This task is intentionally simple so the harness can be validated before moving to generated CUDA kernels.

## Current Phase

Phase 1 focuses on:

- project structure
- task metadata
- shape variant generation
- result schemas
- CPU-compatible tests
- CUDA-aware scripts that fail gracefully when CUDA is unavailable

## References

- Technical reference list: `docs/references.md`
- NVIDIA CUDA C++ Programming Guide: https://docs.nvidia.com/cuda/cuda-programming-guide/
- PyTorch C++/CUDA extensions: https://docs.pytorch.org/docs/stable/cpp_extension.html
- PyTorch custom C++ and CUDA operators tutorial: https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
- `torch.compile` API: https://docs.pytorch.org/docs/stable/generated/torch.compile.html
- CUDA Agent reference paper: https://arxiv.org/abs/2602.24286
