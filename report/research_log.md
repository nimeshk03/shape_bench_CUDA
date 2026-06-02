# ShapeBench-CUDA Research Log

This log records project decisions, implementation milestones, validation results, and experiment notes that may later become paper material.

## 2026-06-03

### Project Direction

ShapeBench-CUDA is being developed as an independent research project about shape generalization in LLM-generated CUDA kernels.

Main research question:

> How well do LLM-generated CUDA kernels generalize across input shape variations, and can shape-aware prompting improve robustness?

Current phase:

- Phase 1: local MVP plus AWS-ready evaluation harness.
- Local work must remain CPU-compatible.
- CUDA compilation and benchmarking are reserved for later AWS GPU runs.

### Workflow Decision

The user wants the assistant to act as a research assistant:

- Present options when research or implementation decisions are needed.
- Let the user choose the direction.
- Implement in small reviewable steps.
- Explain each implemented step simply.
- Maintain this research log as paper-preparation material.

### Public-Readiness Decision

The repository should be private now but suitable for becoming public later.

Actions taken:

- Removed personal local paths from public-facing files.
- Moved local setup details and workflow preferences outside the repository.
- Kept `AGENTS.md` because it contains useful project constraints for collaborators and agents.
- Ignored local-only files such as caches, local planning notes, and local PDFs.

### Local Harness Milestone

Implemented the Phase 1 local harness skeleton:

- `harness/shape_registry.py`: validates shapes and creates default vector/matrix/batch-feature variants.
- `harness/result_schema.py`: defines structured per-shape results and task summaries.
- `harness/result_io.py`: writes and reads JSONL result records.
- `harness/task_loader.py`: loads task metadata and shape definitions.
- `harness/cuda_checks.py`: checks CUDA availability safely on CPU-only machines.
- `harness/compare_outputs.py`: compares tensor-like outputs.
- `harness/run_benchmark.py`: provides a simple callable timing helper.

Validation result:

```text
pytest -q
20 passed
```

Commit:

```text
061f20e Initialize ShapeBench-CUDA local harness
```

### GitHub Milestone

Created a private GitHub repository and pushed the initial local harness:

```text
https://github.com/nimeshk03/shape_bench_CUDA
```

Repository visibility:

```text
PRIVATE
```

### First Benchmark Task Decision

The first benchmark task should be intentionally simple, so the task format and harness can be validated before CUDA generation.

Chosen task:

```text
task_001: elementwise relu(x + y)
```

Reasoning:

- Easy to understand.
- Easy to test locally.
- Shape-general by nature.
- A generated CUDA kernel for it should still expose common shape bugs such as hardcoded sizes, missing boundary checks, or power-of-two assumptions.

Files added:

- `tasks/task_001/metadata.json`
- `tasks/task_001/shapes.json`
- `tasks/task_001/model.py`
- `tests/test_task_001.py`

Shape variants:

- original: `[1024, 1024]`
- smaller: `[512, 1024]`
- larger: `[2048, 1024]`
- odd: `[1007, 1013]`
- batch variant: `[256, 1024]`
- non-power-of-two: `[1000, 1007]`

Validation result:

```text
pytest -q
23 passed in 3.53s
```

Commit:

```text
85cfb4c Add first elementwise benchmark task
```

### Current Open Questions

- Should the next step be prompt files, AWS readiness scripts, or a CPU-only correctness runner for task definitions?
- Should the first prompt pair be generic for all tasks or task-specific?
- Should task metadata include tolerance settings such as `atol` and `rtol` from the beginning?
- Should shape variants remain fixed per task or be generated from metadata by the harness?
