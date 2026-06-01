# ShapeBench-CUDA Agent Instructions

## Project Identity

ShapeBench-CUDA is a research project for evaluating shape generalization in LLM-generated CUDA kernels.

The project is inspired by CUDA Agent, but it is not a reproduction of CUDA Agent's full reinforcement learning system. CUDA Agent asks whether an agent can generate high-performance CUDA kernels. ShapeBench-CUDA asks whether generated kernels remain correct and performant when tensor shapes change.

Treat this repository as an independent project. The sibling `../CUDA-Agent` repository, if present, is reference material only and must not be modified unless the user explicitly asks.

## Research Question

Main question:

> How well do LLM-generated CUDA kernels generalize across input shape variations, and can shape-aware prompting improve robustness?

Sub-questions:

- Do generated kernels pass correctness on the original benchmark shape?
- Do they remain correct on smaller, larger, odd, non-power-of-two, and batch-varied shapes?
- How stable is speedup across shape variants?
- Which shape changes cause failures?
- Does shape-aware prompting improve multi-shape correctness?
- Is there a tradeoff between peak original-shape speed and general robustness?

## Current Phase

Current priority:

```text
Phase 1: Local MVP + AWS-ready evaluation harness
```

Phase 1 should build a small, clean, testable framework before any large GPU experiments.

Required Phase 1 capabilities:

- project structure
- task metadata
- shape variant generation
- result schemas
- CPU-compatible local tests
- CUDA-aware scripts that fail gracefully when CUDA is unavailable
- AWS-ready scripts for later CUDA compile/run/benchmark work

Do not jump ahead to full-scale experiments, reinforcement learning, large KernelBench runs, heavy profiling, or CUDA-Agent reproduction work unless explicitly requested.

## Scope

In scope:

- shape generalization evaluation
- baseline prompt versus shape-aware prompt comparison
- correctness testing across multiple shapes
- runtime benchmarking on AWS GPU later
- JSON/JSONL/CSV result logging
- failure reason categorization
- simple report generation

Out of scope for Phase 1:

- training or fine-tuning an LLM
- reinforcement learning
- reproducing CUDA Agent's training pipeline
- large-scale KernelBench evaluation
- heavy Nsight Compute profiling on every task
- multi-GPU benchmarking
- energy-aware optimization
- cybersecurity benchmark design
- retrieval-augmented CUDA generation
- modifying `../CUDA-Agent`

## Expected Repository Layout

```text
shape_bench_CUDA/
├── README.md
├── project_overview.md
├── AGENTS.md
├── environment.yml
├── requirements.txt
├── .gitignore
├── paper/
├── tasks/
├── prompts/
├── generated/
├── harness/
├── scripts/
├── results/
├── report/
└── docs/
```

Important directories:

```text
tasks/                  Task definitions and metadata
prompts/                Baseline and shape-aware prompts
generated/              Generated CUDA code outputs
harness/                Evaluation and benchmarking code
scripts/                Setup and execution scripts
results/raw/            Raw JSON/JSONL results
results/tables/         Summarized CSV/Markdown tables
results/figures/        Plots and visualizations
report/                 Research notes and findings
docs/                   Development notes
```

Generated CUDA outputs should be separated by prompt mode:

```text
generated/baseline/
generated/shape_aware/
```

## Local Development Rules

Local development is for writing code, editing prompts, validating schemas, running CPU-compatible tests, generating reports, and analyzing results.

The local laptop is assumed not to have an NVIDIA GPU:

```text
torch.cuda.is_available() == False
```

This is expected locally and must not be treated as an error.

Local rules:

- Use Python 3.11.
- Use a project-specific Python environment.
- Prefer creating the environment from `environment.yml` or installing from `requirements.txt`.
- Activate your environment before running project Python commands.
- Keep machine-specific paths, credentials, AWS details, and personal workflow preferences outside the public repository.

```bash
conda env create -f environment.yml
conda activate shapebench-cuda
```

- Local tests must not require CUDA.
- CUDA compilation and benchmarking must not be required locally.
- CUDA-specific code should detect missing CUDA and skip or fail gracefully with clear messages.

Useful local checks:

```bash
conda activate shapebench-cuda
python scripts/check_local_setup.py
pytest
```

## AWS GPU Rules

AWS is only for CUDA execution:

- compiling CUDA extensions
- running CUDA correctness checks
- benchmarking generated kernels
- collecting runtime results
- saving experiment logs

Do not use AWS for long manual editing, paper reading, idle notebooks, or non-GPU development.

Preferred Phase 1 instances:

```text
g5.xlarge   NVIDIA A10G 24GB
g6.xlarge   NVIDIA L4 24GB
g6.2xlarge  NVIDIA L4 24GB
```

Avoid expensive A100/H100/P-series instances during Phase 1 unless explicitly requested.

Before GPU experiments on AWS, verify:

```bash
nvidia-smi
nvcc --version
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY
```

On AWS, `torch.cuda.is_available()` must be `True`. If it is not, stop and fix the environment before running experiments.

## AWS Cost Control

- Prepare code locally before starting a GPU instance.
- Start AWS only when ready to run a batch.
- Run small experiments first.
- Save all logs and results.
- Stop the instance immediately after CUDA experiments finish.
- Avoid idle notebooks, unused Elastic IPs, unnecessary snapshots, and oversized EBS volumes.
- Do not rerun experiments unless prior results were invalid or incomplete.

Preferred workflow:

```text
local laptop:
  edit code, prompts, task configs
  run CPU-compatible tests
  commit/push or otherwise sync code

AWS GPU instance:
  pull/sync latest code
  conda activate shapebench-cuda
  run a small experiment batch
  save results
  push/download results
  stop instance
```

## Technical References

For CUDA, PyTorch extension, benchmarking, and `torch.compile` details, refer to:

```text
docs/references.md
```

Use official documentation first. Do not guess CUDA/PyTorch APIs when documentation is available.

## Coding Style

Prefer:

- simple Python
- readable functions
- type hints
- dataclasses
- small modules
- JSON/JSONL output
- explicit error messages
- CPU-compatible tests

Avoid:

- over-engineering
- hidden global state
- hardcoded absolute paths
- requiring CUDA for imports
- modifying unrelated files
- unnecessary frameworks
- RL or distributed-training dependencies in Phase 1

## Testing Rules

Local tests must pass without CUDA:

```bash
pytest
```

Testable Phase 1 components:

- shape registry
- task metadata parsing
- result schema serialization
- failure reason categories
- local setup script
- benchmark runner CUDA detection

CUDA-specific tests should use `pytest.mark.skipif` or verify graceful failure when CUDA is unavailable.

## Prompt Files

Prompt files should live in:

```text
prompts/baseline_prompt.md
prompts/shape_aware_prompt.md
```

Baseline prompt:

- optimize for CUDA performance
- enforce normal correctness constraints
- do not add special shape robustness emphasis

Shape-aware prompt:

- avoid hardcoded dimensions
- handle variable shapes
- handle odd and non-power-of-two sizes
- include boundary checks
- preserve performance where possible

## Agent Workflow

When working in this repo:

1. Read `AGENTS.md`.
2. Inspect the current file tree.
3. Make small, reviewable changes.
4. Keep local tests CPU-compatible.
5. Do not assume CUDA locally.
6. Do not modify sibling reference repositories.
7. Run tests after changes when possible.
8. Summarize what changed.
9. Clearly mention any work that still needs AWS GPU validation.
