# ShapeBench-CUDA

ShapeBench-CUDA is a lightweight research project for evaluating whether LLM-generated CUDA kernels remain correct and performant across input shape variations.

The Phase 1 goal is a local, CPU-compatible harness plus GPU-ready CUDA scripts. Local development should not require an NVIDIA GPU.

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

These pieces are intentionally CPU-compatible. CUDA compilation and benchmarking are expected to run later on a rented GPU instance.

## Initial Benchmark Task

The first local benchmark task is:

- `tasks/task_001`: elementwise `relu(x + y)` for two same-shaped tensors.

It includes task metadata, fixed shape variants, a PyTorch `Model`, deterministic input generation, and a functional reference. This task is intentionally simple so the harness can be validated before moving to generated CUDA kernels.

## Prompt Modes

The initial prompt modes are:

- `prompts/baseline_prompt.md`: asks for a fast correct CUDA implementation without special shape-generalization emphasis.
- `prompts/shape_aware_prompt.md`: asks for a fast correct CUDA implementation that explicitly handles shape variants, odd dimensions, non-power-of-two sizes, and batch changes.

Rendered baseline prompts include only the original task shape. Shape variants are included only in rendered shape-aware prompts, so the baseline-vs-shape-aware comparison stays clean.

Render a concrete prompt for a task:

```bash
python scripts/render_prompt.py --task-dir tasks/task_001 --mode baseline
python scripts/render_prompt.py --task-dir tasks/task_001 --mode shape_aware
```

Rendered prompts are written to `generated/prompts/`.

## Anthropic Generation

Set your Anthropic API key in `.env.local`:

```bash
ANTHROPIC_API_KEY=...
```

`CLAUDE_API_KEY=...` is also accepted.

`.env.local` is ignored by Git.

Generate code for a task and prompt mode:

```bash
python scripts/check_anthropic_api.py
python scripts/generate_with_anthropic.py --task-dir tasks/task_001 --mode baseline --attempt 1
python scripts/generate_with_anthropic.py --task-dir tasks/task_001 --mode shape_aware --attempt 1
```

Generation uses `temperature=0.1` by default for reproducibility.

Generation artifacts are saved under:

```text
generated/<prompt_mode>/<task_id>/attempt_<number>/
```

Each attempt stores `metadata.json`, `prompt.md`, `response.md`, and `raw_response.json`.

Extract generated code blocks from an attempt:

```bash
python scripts/extract_generated_code.py generated/baseline/task_001/attempt_002
python scripts/extract_generated_code.py generated/shape_aware/task_001/attempt_002
```

Extracted files are written to the attempt's `extracted/` directory with a `manifest.json`.

Prepare an extracted attempt for evaluation:

```bash
python scripts/prepare_attempt_contract.py generated/baseline/task_001/attempt_003
python scripts/prepare_attempt_contract.py generated/shape_aware/task_001/attempt_002
```

The evaluation contract is:

```text
extracted/solution.py exposes forward(*inputs)
```

If an attempt has CUDA code but no `solution.py`, the prep step creates a fallback wrapper.
The contract records `task_id`, `prompt_mode`, `attempt`, `input_names`, `cuda_source`, `extension_function`, and a unique extension name.

Evaluate a prepared attempt:

```bash
python scripts/evaluate_attempt.py generated/baseline/task_001/attempt_003
python scripts/evaluate_attempt.py generated/shape_aware/task_001/attempt_002
```

The evaluator writes per-shape correctness records to `results/raw/`. On a CPU-only local machine, generated CUDA attempts are expected to record compilation failures instead of passing. On a GPU instance, the same command should compile and run the generated CUDA entrypoint.

If `TORCH_EXTENSIONS_DIR` is not set, the evaluator uses `/tmp/shape_bench_torch_extensions` for PyTorch extension build artifacts.

## GPU Evaluation

On a GPU instance, first verify the CUDA environment:

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

Run the first prepared evaluation batch on any CUDA GPU machine:

```bash
python scripts/run_gpu_eval_batch.py
```

The GPU batch script requires CUDA by default. For a local CPU-only smoke test, use:

```bash
python scripts/run_gpu_eval_batch.py --allow-cpu --device auto --summary-output /tmp/shapebench_gpu_eval_batch_summary.json
```

The default GPU batch evaluates:

- `generated/baseline/task_001/attempt_003`
- `generated/shape_aware/task_001/attempt_002`

It writes per-shape JSONL results under `results/raw/` and a summary JSON under `results/tables/`.

## Vast.ai Manual-Offer Automation

For cheaper early GPU experiments, use Vast.ai with a manually selected offer.

Local setup:

```bash
python -m pip install --upgrade vastai
vastai set api-key <your_vast_api_key>
vastai create ssh-key "$(cat ~/.ssh/id_ed25519.pub)"
vastai search offers 'gpu_name=RTX_4090 num_gpus=1' --limit 10
```

If RTX 4090 offers are too expensive, search RTX 3090:

```bash
vastai search offers 'gpu_name=RTX_3090 num_gpus=1' --limit 10
```

After choosing an `offer_id`, commit the local repo first, then run:

```bash
python scripts/run_vast_eval.py --offer-id <offer_id>
```

By default this uses the Vast.ai `PyTorch (cuDNN Devel)` template instead of raw image mode:

```text
3ba4addf2b917a405583ebb21dfd3f72
```

The Vast runner:

- creates an SSH-capable Vast instance from the template,
- uses noninteractive SSH options, including automatic first-use host-key acceptance,
- aborts after repeated SSH public-key failures instead of waiting until the full timeout,
- uploads a committed `git archive` of this repo,
- installs Python requirements,
- runs CUDA preflight checks,
- runs `pytest -q`,
- runs `python scripts/run_gpu_eval_batch.py`,
- downloads result files and logs to `results/vast_runs/<timestamp>/`,
- streams remote output to the terminal and saves it as `remote_eval.log`,
- destroys the Vast instance automatically unless `--keep-instance` is passed.

Raw Docker image fallback:

```bash
python scripts/run_vast_eval.py --offer-id <offer_id> --template-hash "" --image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
```

Use `--keep-instance` only for debugging. Otherwise the script destroys the instance to control cost.

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
