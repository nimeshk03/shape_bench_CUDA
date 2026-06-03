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

### Prompt Mode Decision

Decision:

```text
Use generic prompt files first.
```

Files added:

- `prompts/baseline_prompt.md`
- `prompts/shape_aware_prompt.md`
- `tests/test_prompts.py`

Reasoning:

- The core experiment compares baseline prompting against shape-aware prompting.
- Generic prompts are reusable across future tasks.
- Task-specific prompt rendering can be added later by combining these files with task metadata and model code.

Baseline prompt intent:

- Optimize CUDA performance.
- Preserve normal numerical correctness.
- Avoid special shape-robustness emphasis.

Shape-aware prompt intent:

- Avoid hardcoded dimensions.
- Use runtime shape information.
- Handle smaller, larger, odd, non-power-of-two, and batch-varied shapes.
- Preserve performance where possible without sacrificing multi-shape correctness.

### Prompt Renderer Milestone

Implemented a local prompt renderer that combines:

- a generic prompt mode,
- task metadata,
- shape variants,
- and the PyTorch reference model.

Files added:

- `harness/prompt_renderer.py`
- `scripts/render_prompt.py`
- `tests/test_prompt_renderer.py`
- `generated/prompts/task_001_baseline.md`
- `generated/prompts/task_001_shape_aware.md`

Task metadata update:

- Added `atol` and `rtol` to `tasks/task_001/metadata.json`.

Reasoning:

- This creates concrete prompts ready to give to an LLM.
- The workflow is still local and CPU-compatible.
- API integration can be delayed until the manual prompt workflow is validated.

Validation result:

```text
pytest -q
29 passed in 2.62s
```

Current generated prompt commands:

```bash
python scripts/render_prompt.py --task-dir tasks/task_001 --mode baseline
python scripts/render_prompt.py --task-dir tasks/task_001 --mode shape_aware
```

### Skeptic Review Cleanup

Reviewed the Phase 1 implementation before starting CUDA generation.

Issues addressed:

- Prompt renderer default output paths are now resolved relative to the project root.
- Rendered prompts now clarify that generated CUDA should match `Model.forward` and `reference`; `create_inputs` is only testing context.
- Task metadata validation now checks required fields, dtype, tolerances, input names, and original shape format.
- Task loading now verifies `metadata.original_shape` matches `shapes.original`.
- `TaskSummary.from_results()` now requires exactly one `original` shape result to avoid confusing missing original-shape data with original-shape failure.
- `.env.local` is ignored so local API keys are not committed accidentally.
- Rendered prompts for `task_001` were regenerated and kept as reproducible prompt artifacts.

Rationale:

- These fixes reduce silent failure modes before generated CUDA code enters the workflow.
- The project remains local and CPU-compatible.

### Anthropic API Generation Decision

Decision:

```text
Use the Anthropic API for first generated CUDA attempts.
```

Implementation:

- Added `harness/anthropic_generation.py`.
- Added `scripts/generate_with_anthropic.py`.
- Added `tests/test_anthropic_generation.py`.

API details:

- Uses Anthropic Messages API.
- Sends `anthropic-version: 2023-06-01`.
- Default generation model: `claude-sonnet-4-6`.
- Smoke-test model: `claude-haiku-4-5-20251001`.
- Default maximum output tokens: `4096`.
- Default generation temperature: `0.1`.
- API key is read from `ANTHROPIC_API_KEY` or `CLAUDE_API_KEY` in the environment or `.env.local`.
- `.env.local` remains ignored and must not be committed.

Generation artifact layout:

```text
generated/<prompt_mode>/<task_id>/attempt_<number>/
├── metadata.json
├── prompt.md
├── raw_response.json
└── response.md
```

Reasoning:

- This records model, prompt mode, attempt number, token usage, raw response, and extracted text for paper traceability.
- Tests avoid real API calls and do not read the user's actual `.env.local`.

### Anthropic API Smoke Test

Ran a tiny Anthropic API smoke test before sending full CUDA-generation prompts.

Command:

```bash
python scripts/check_anthropic_api.py
```

Result:

```text
Anthropic API smoke test passed.
Model: claude-haiku-4-5-20251001
Response: OK.
Usage: input_tokens=12, output_tokens=5
```

Notes:

- `.env.local` used `CLAUDE_API_KEY`, so the loader was updated to accept both `ANTHROPIC_API_KEY` and `CLAUDE_API_KEY`.
- The first sandboxed network attempt failed due DNS restrictions, then the same smoke test passed outside the sandbox.

### First CUDA Generation Attempts

Generated first CUDA responses for `task_001` with Anthropic Claude.

Commands:

```bash
python scripts/generate_with_anthropic.py --task-dir tasks/task_001 --mode baseline --attempt 1
python scripts/generate_with_anthropic.py --task-dir tasks/task_001 --mode shape_aware --attempt 1
```

Artifacts:

```text
generated/baseline/task_001/attempt_001/
generated/shape_aware/task_001/attempt_001/
```

Baseline attempt metadata:

```text
model: claude-sonnet-4-6
response_id: msg_01JgYx8vjSrKKxwFRLDHA9AY
input_tokens: 1018
output_tokens: 1205
stop_reason: end_turn
```

Shape-aware attempt metadata:

```text
model: claude-sonnet-4-6
response_id: msg_01Vy3eDW3Tb2JHJKZ1Nh9fH5
input_tokens: 1092
output_tokens: 894
stop_reason: end_turn
```

Status:

- Generation only.
- No CUDA compilation yet.
- No correctness or benchmark results yet.

### Low-Temperature Regeneration

Decision:

```text
Regenerate task_001 before code extraction using explicit temperature=0.1.
```

Reasoning:

- Attempt 1 used Anthropic's default/unspecified temperature.
- Research runs should use explicit generation settings for reproducibility.
- The extractor should target the generation style we plan to use.

Commands:

```bash
python scripts/generate_with_anthropic.py --task-dir tasks/task_001 --mode baseline --attempt 2 --temperature 0.1
python scripts/generate_with_anthropic.py --task-dir tasks/task_001 --mode shape_aware --attempt 2 --temperature 0.1
```

Baseline attempt 2 metadata:

```text
model: claude-sonnet-4-6
temperature: 0.1
response_id: msg_01UDatMNmZHQZ8jo8oSogk
input_tokens: 1018
output_tokens: 1096
stop_reason: end_turn
```

Shape-aware attempt 2 metadata:

```text
model: claude-sonnet-4-6
temperature: 0.1
response_id: msg_01YMUSYzKsqCL4KwUw12rqV9
input_tokens: 1092
output_tokens: 912
stop_reason: end_turn
```

Status:

- Generation only.
- No CUDA compilation yet.
- Use attempt 2 as the main target for the code extractor.

### Code Extractor Milestone

Implemented an automatic extractor for fenced code blocks in generated LLM responses.

Files added:

- `harness/code_extractor.py`
- `scripts/extract_generated_code.py`
- `tests/test_code_extractor.py`

Extractor behavior:

- Reads `response.md` from a generation attempt directory.
- Parses fenced Markdown code blocks.
- Infers filenames using conservative rules.
- Writes extracted files to `extracted/`.
- Writes `extracted/manifest.json`.
- Refuses to overwrite existing extracted files unless `--overwrite` is passed.

Commands run:

```bash
python scripts/extract_generated_code.py generated/baseline/task_001/attempt_002
python scripts/extract_generated_code.py generated/shape_aware/task_001/attempt_002
```

Extraction results:

```text
baseline attempt 2:
  extracted/extension.cu
  extracted/setup.py
  extracted/manifest.json

shape-aware attempt 2:
  extracted/extension.cu
  extracted/setup.py
  extracted/solution.py
  extracted/manifest.json
```

Observed follow-up issue:

- The shape-aware `solution.py` references `add_relu.cu`, while the extractor named the CUDA block `extension.cu`.
- This should be handled before AWS compilation, either by improving filename inference or adding a compile-prep step that reconciles filenames.

### Code Extractor Filename Fix

Addressed the extractor review feedback:

- The extractor now scans generated Python/setup blocks for referenced `.cu` filenames.
- If exactly one CUDA filename is referenced, the CUDA block is saved with that filename.
- Re-extracted attempt 2 artifacts now use `add_relu.cu`, matching generated `setup.py` and `solution.py` references.
- `--overwrite` now removes previous manifest-tracked extracted files before writing new ones.

Re-extraction commands:

```bash
python scripts/extract_generated_code.py generated/baseline/task_001/attempt_002 --overwrite
python scripts/extract_generated_code.py generated/shape_aware/task_001/attempt_002 --overwrite
```

Updated extraction results:

```text
baseline attempt 2:
  extracted/add_relu.cu
  extracted/setup.py
  extracted/manifest.json

shape-aware attempt 2:
  extracted/add_relu.cu
  extracted/setup.py
  extracted/solution.py
  extracted/manifest.json
```

### Evaluation Contract Decision

Decision:

```text
Use extracted/solution.py:forward as the generated-attempt evaluation entrypoint.
```

Contract:

```text
attempt/extracted/eval_contract.json
attempt/extracted/solution.py
entrypoint function: forward(*inputs)
```

Implementation:

- Added `harness/attempt_contract.py`.
- Added `scripts/prepare_attempt_contract.py`.
- Added `tests/test_attempt_contract.py`.

Fallback behavior:

- If an extracted attempt already has `solution.py`, keep it.
- If an extracted attempt has a single CUDA source but no `solution.py`, create a fallback wrapper that loads the CUDA source with `torch.utils.cpp_extension.load` and exposes `forward(x, y)`.

Commands run:

```bash
python scripts/prepare_attempt_contract.py generated/baseline/task_001/attempt_002
python scripts/prepare_attempt_contract.py generated/shape_aware/task_001/attempt_002
```

Results:

```text
baseline attempt 2:
  entrypoint: extracted/solution.py:forward
  cuda_source: add_relu.cu
  fallback solution created: true

shape-aware attempt 2:
  entrypoint: extracted/solution.py:forward
  cuda_source: add_relu.cu
  fallback solution created: false
```

Status:

- Contract prepared locally.
- No CUDA compilation yet.

### Evaluation Contract Review Cleanup

Addressed follow-up concerns from review:

- Fallback `solution.py` now uses a unique extension name derived from task id, prompt mode, and attempt number.
- `eval_contract.json` now records `task_id`, `prompt_mode`, `attempt`, `input_names`, `cuda_source`, and `extension_name`.

Updated baseline attempt 2 contract:

```text
entrypoint: solution.py:forward
input_names: x, y
extension_name: shapebench_task_001_baseline_attempt_002
created_fallback_solution: true
```

Updated shape-aware attempt 2 contract:

```text
entrypoint: solution.py:forward
input_names: x, y
extension_name: shapebench_task_001_shape_aware_attempt_002
created_fallback_solution: false
```

Remaining concern:

- Generated `solution.py` files can still compile at import time or at `forward()` time depending on the LLM response. The evaluator should catch both import-time and runtime failures.

### Evaluation Contract Design Fixes

Addressed review feedback that found two evaluation-contract risks and one research-design blocker.

Prompt contamination fix:

- Baseline rendered prompts now include only the original shape.
- Shape variants are shown only in shape-aware prompts.
- Previously generated baseline attempts 1 and 2 were produced from prompts that included shape variants, so they should not be treated as clean baseline evidence.

Evaluation contract fixes:

- Existing extracted `solution.py` files must define a top-level `forward` function before a contract is written.
- Fallback wrappers now derive `forward(*inputs)` from task metadata instead of hardcoding `forward(x, y)`.
- Fallback wrappers now infer the CUDA extension function from a single `m.def("...")` binding in the extracted `.cu` file.
- `eval_contract.json` now records `extension_function`.

Clean baseline regeneration:

```bash
python scripts/render_prompt.py --task-dir tasks/task_001 --mode baseline
python scripts/render_prompt.py --task-dir tasks/task_001 --mode shape_aware
python scripts/generate_with_anthropic.py --task-dir tasks/task_001 --mode baseline --attempt 3 --temperature 0.1
python scripts/extract_generated_code.py generated/baseline/task_001/attempt_003
python scripts/prepare_attempt_contract.py generated/baseline/task_001/attempt_003
```

Baseline attempt 3 metadata:

```text
model: claude-sonnet-4-6
temperature: 0.1
response_id: msg_01W2aw8X2DoNbs4zoG3JvLfd
input_tokens: 932
output_tokens: 1245
stop_reason: end_turn
```

Baseline attempt 3 contract:

```text
entrypoint: solution.py:forward
input_names: x, y
cuda_source: add_relu.cu
extension_function: add_relu
extension_name: shapebench_task_001_baseline_attempt_003
created_fallback_solution: true
```

Validation result:

```text
pytest -q
53 passed in 2.37s
```

### Evaluation Contract Idempotency Fix

Addressed another review pass on the contract-preparation flow.

Issues fixed:

- Re-running `prepare_attempt_contract.py` on an existing fallback wrapper now preserves `created_fallback_solution: true`.
- Contracts for LLM-provided `solution.py` files now record the extension name actually used by the entrypoint, when it can be inferred from `load(name=...)`.
- Fallback wrapper generation now rejects Python keywords such as `class` or `from` before writing `solution.py`.

Updated artifact check:

```text
baseline attempt 2:
  created_fallback_solution: true
  extension_name: shapebench_task_001_baseline_attempt_002

baseline attempt 3:
  created_fallback_solution: true
  extension_name: shapebench_task_001_baseline_attempt_003

shape-aware attempt 2:
  created_fallback_solution: false
  extension_name: elementwise_add_relu_ext
```

Validation result:

```text
pytest -q
55 passed in 2.42s
```

### Correctness Evaluator Milestone

Implemented the first evaluation runner for prepared generated attempts.

Files added:

- `harness/evaluator.py`
- `scripts/evaluate_attempt.py`
- `tests/test_evaluator.py`

Evaluator behavior:

- Reads `attempt/extracted/eval_contract.json`.
- Loads the matching task metadata, shape variants, and reference model.
- Imports `extracted/solution.py` and calls the configured `forward` entrypoint.
- Allows generated Python entrypoints to import helper files from their `extracted/` directory.
- Uses `/tmp/shape_bench_torch_extensions` as the default PyTorch extension build cache when `TORCH_EXTENSIONS_DIR` is not already set.
- Runs deterministic correctness checks for every shape variant.
- Records one `ShapeBenchResult` per shape in JSONL format under `results/raw/`.
- Converts import-time compile failures and forward-time runtime failures into structured failed results instead of crashing the run.

Local validation:

```text
pytest -q
60 passed in 2.49s
```

Local generated-attempt smoke tests:

```bash
python scripts/evaluate_attempt.py generated/baseline/task_001/attempt_003 --device auto
python scripts/evaluate_attempt.py generated/shape_aware/task_001/attempt_002 --device auto
```

Results on the local CPU-only machine:

```text
baseline attempt 3:
  passed shapes: 0/6
  original passed: false
  failure reasons: compilation_failure=6

shape-aware attempt 2:
  passed shapes: 0/6
  original passed: false
  failure reasons: compilation_failure=6
```

Interpretation:

- These local failures are expected because CUDA compilation is not available locally.
- The important Phase 1 result is that the evaluator produces structured failure records instead of crashing.
- The same evaluator command is now ready for rented GPU correctness runs.

### GPU Evaluation Batch Script

Implemented the first GPU-ready batch command for the prepared generated attempts.

Files added:

- `harness/gpu_eval_batch.py`
- `scripts/run_gpu_eval_batch.py`
- `tests/test_gpu_eval_batch.py`

Batch behavior:

- Runs preflight checks for `nvidia-smi`, `nvcc --version`, and `torch.cuda.is_available()`.
- Requires CUDA by default, so a misconfigured GPU instance fails before running experiments.
- Evaluates the first clean task pair:
  - `generated/baseline/task_001/attempt_003`
  - `generated/shape_aware/task_001/attempt_002`
- Writes per-shape JSONL results under `results/raw/`.
- Writes a batch summary JSON under `results/tables/`.
- Supports local smoke testing with `--allow-cpu`.

GPU command:

```bash
python scripts/run_gpu_eval_batch.py
```

Local smoke command:

```bash
python scripts/run_gpu_eval_batch.py --allow-cpu --device auto --summary-output /tmp/shapebench_gpu_eval_batch_summary.json
```

Local smoke result:

```text
baseline attempt 3:
  passed shapes: 0/6
  failure reasons: compilation_failure=6

shape-aware attempt 2:
  passed shapes: 0/6
  failure reasons: compilation_failure=6
```

Strict local preflight result:

```text
GPU preflight failed: nvidia-smi is not available; nvcc is not available; torch.cuda.is_available() is false
```

Validation result:

```text
pytest -q
61 passed in 2.37s
```

### Vast.ai Manual-Offer Automation

Decision:

```text
Use Vast.ai for early CUDA experiments to reduce GPU cost, with manual offer selection for cost control.
```

Implementation:

- Added provider-neutral GPU batch helpers in `harness/gpu_eval_batch.py`.
- Added `scripts/run_gpu_eval_batch.py` as the primary CUDA-machine batch command.
- Added `harness/vast_runner.py` and `scripts/run_vast_eval.py` for one-shot Vast runs.
- Added `tests/test_vast_runner.py`.
- Installed the `vastai` CLI in the local `shapebench-cuda` environment and added it to `requirements.txt` and `environment.yml`.

Vast runner behavior:

- User manually searches Vast offers and chooses an `offer_id`.
- The script creates an SSH/direct Vast instance using a PyTorch CUDA devel image.
- The script uploads a committed `git archive` of the repo instead of cloning from GitHub, avoiding GitHub credentials on the rented machine.
- The remote machine installs requirements, checks CUDA, runs `pytest -q`, then runs `scripts/run_gpu_eval_batch.py`.
- Results and logs are downloaded to `results/vast_runs/<timestamp>/`.
- The Vast instance is destroyed automatically unless `--keep-instance` is passed.

Command:

```bash
python scripts/run_vast_eval.py --offer-id <offer_id>
```

Default image:

```text
pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel
```

Open requirement:

- The local repo must be committed before running the Vast script because the uploaded archive uses committed Git content.

Validation result:

```text
pytest -q
67 passed in 2.37s
```

Local smoke results:

```bash
python scripts/run_gpu_eval_batch.py --allow-cpu --device auto --summary-output /tmp/shapebench_gpu_eval_batch_summary.json
```

The local smoke command recorded the expected CPU-only `compilation_failure=6` for baseline attempt 3 and shape-aware attempt 2.

### Vast.ai Runner Hardening

Observation:

```text
The first manual Vast run exposed two practical issues: SSH could prompt for
host-key confirmation, and the runner stayed mostly silent while waiting or
running remote commands. A failed or interrupted setup can waste paid GPU time.
```

Implementation:

- Normalized Vast `ssh://user@host:port` outputs into explicit SSH arguments.
- Added noninteractive SSH options, including batch mode, automatic first-use host-key acceptance, and connection/server keepalive timeouts.
- Added timestamped runner progress logs for instance creation, SSH polling, upload, remote execution, result download, and destroy.
- Streamed remote command output live to the terminal while saving it to `results/vast_runs/<timestamp>/remote_eval.log`.
- Made automatic destroy ignore Ctrl-C during the destroy request, reducing the chance of leaving an instance running after a local interruption.
- Made destroy fail loudly if the Vast CLI does not accept the destroy request.

Validation result:

```text
conda run -n shapebench-cuda pytest -q
72 passed in 2.14s
```

### Vast.ai Template Launch Fix

Observation:

```text
The first RTX 4090 Vast offer reached a running/loading state but repeatedly
rejected SSH public-key authentication. The runner treated this as normal
startup delay, so it kept waiting until the user interrupted the run. Cleanup
then destroyed the instance, and `vastai show instances` reported no active
instances.
```

Implementation:

- Switched the default Vast launch path from raw Docker image mode to Vast template mode.
- Default template: `PyTorch (cuDNN Devel)`, hash `3ba4addf2b917a405583ebb21dfd3f72`.
- Kept raw Docker image mode as an explicit fallback using `--template-hash "" --image ...`.
- Added `LogLevel=ERROR` to SSH options to suppress noisy post-quantum warnings during readiness probes.
- Added fast failure after repeated `Permission denied (publickey)` SSH responses.
- Added a clean `KeyboardInterrupt` path in the CLI so Ctrl-C does not print a full traceback after cleanup.

Open question:

```text
Need to retry with the Vast template launch path. If SSH still fails, the next
suspect is account-level SSH key registration rather than the container image.
```

Validation result:

```text
conda run -n shapebench-cuda pytest -q
74 passed in 2.30s
```

### Vast.ai SSH Setup Check

Observation:

```text
The local SSH public key and the key registered in Vast.ai matched by fingerprint:
SHA256:bQ+ktm7AgBCvBeyBwA4+B14pIadMLtNgTFjGiwQ7z+0.
The interrupted template-mode run showed `Connection refused` while the instance
was still booting, not a public-key rejection.
```

Implementation:

- Added `scripts/check_vast_setup.py` to verify Vast CLI availability, local SSH key fingerprint, registered Vast key fingerprints, and active instance count without launching an instance.
- Improved Vast runner SSH readiness messages so boot-time `Connection refused` is reported separately from `Permission denied (publickey)`.

Validation target:

```text
Run `python scripts/check_vast_setup.py` before the next paid Vast launch.
```

Validation result:

```text
conda run -n shapebench-cuda pytest -q
75 passed in 2.44s

conda run -n shapebench-cuda python scripts/check_vast_setup.py
Vast setup check passed; active Vast instances: 0.
```

### Vast.ai First Successful SSH Run

Observation:

```text
The template-mode Vast runner successfully created instance 39251492, waited
for SSH, uploaded the committed archive, streamed remote logs, downloaded
artifacts, and destroyed the instance. The run failed during CUDA preflight
because `nvidia-smi` worked but `nvcc` was missing.
```

Result:

```text
GPU: NVIDIA GeForce RTX 4090
Driver: 590.48.01
CUDA reported by driver: 13.1
Failure: bash: line 13: nvcc: command not found
Remote exit code: 127
Destroyed: True
Active Vast instances after run: 0
```

Implementation:

- Changed the default Vast template hash to `e4c5e88bc289f4eecb0c955c4fe7430d`, a PyTorch CUDA devel template tagged `2.2.0-cuda12.1-cudnn8-devel`.
- Moved remote CUDA preflight checks before `pip install` so missing `nvcc` fails before spending time installing dependencies.

Open question:

```text
Need to retry with the devel template and confirm `nvcc --version` succeeds
before running the full GPU batch.
```

Validation result:

```text
conda run -n shapebench-cuda pytest -q
76 passed in 2.28s
```

### Vast.ai Python 3.10 Compatibility Fix

Observation:

```text
The devel template successfully passed `nvidia-smi`, `nvcc --version`, and
`torch.cuda.is_available()`. The run then failed during pytest collection
because the Vast template uses Python 3.10 and the harness imported
`datetime.UTC`, which was introduced in Python 3.11.
```

Result:

```text
Template: e4c5e88bc289f4eecb0c955c4fe7430d
GPU: NVIDIA GeForce RTX 4090
nvcc: CUDA compilation tools 12.1, V12.1.105
PyTorch: 2.2.0
CUDA available: True
Failure: ImportError: cannot import name 'UTC' from 'datetime'
Remote exit code: 2
Destroyed: True
Active Vast instances after run: 0
```

Implementation:

- Replaced `datetime.UTC` with `datetime.timezone.utc` in GPU/Vast harness code so it works on Python 3.10 and 3.11.
- Removed `vastai` from project `requirements.txt` and `environment.yml`; the CLI remains a local setup dependency but GPU workers do not need it.
- This should also avoid remote dependency churn from installing the Vast CLI package on the CUDA worker.

Validation result:

```text
conda run -n shapebench-cuda pytest -q
76 passed in 2.32s
```

### Vast.ai First Full Evaluator Run

Observation:

```text
The Vast runner completed a full remote evaluation cycle on instance 39253518.
The instance passed CUDA preflight, installed project requirements, ran the
test suite, ran `scripts/run_gpu_eval_batch.py`, downloaded result artifacts,
and destroyed the instance.
```

Result:

```text
Remote exit code: 0
Destroyed: True
Active Vast instances after run: 0
Remote pytest: 76 passed in 4.24s
GPU: NVIDIA GeForce RTX 4090
Python: 3.10.13
PyTorch: 2.2.0
CUDA available: True
nvcc: CUDA compilation tools 12.1, V12.1.105
```

Evaluation result:

```text
baseline attempt_003: 0/6 shapes passed; original_passed=False; failures={'compilation_failure': 6}
shape_aware attempt_002: 0/6 shapes passed; original_passed=False; failures={'compilation_failure': 6}
```

Root cause:

```text
The first shape in each attempt failed with `Ninja is required to load C++
extensions`, so PyTorch extension compilation could not start. Later shapes
then failed because the compiled extension shared object did not exist.
```

Implementation:

- Added `ninja` to project requirements and environment files so CUDA extension compilation can run on the GPU worker.

Validation result:

```text
conda run -n shapebench-cuda pytest -q
76 passed in 2.26s
```

### Vast.ai Successful GPU Evaluation

Observation:

```text
After adding `ninja`, the guarded Vast.ai run completed the full GPU evaluation
successfully. The instance was created, passed CUDA preflight, ran remote tests,
ran the GPU batch evaluator, downloaded artifacts, and destroyed itself.
```

Run metadata:

```text
Local run dir: results/vast_runs/20260603T062646Z
Instance id: 39254037
Offer id: 36368079
Template hash: e4c5e88bc289f4eecb0c955c4fe7430d
Remote exit code: 0
Destroyed: True
Cleanup error: None
Active Vast instances after run: 0
```

CUDA preflight:

```text
GPU: NVIDIA GeForce RTX 4090
Python: 3.10.13
PyTorch: 2.2.0
torch CUDA: 12.1
torch.cuda.is_available(): True
nvcc: CUDA compilation tools 12.1, V12.1.105
Remote pytest: 76 passed in 5.20s
```

Evaluation result:

```text
baseline attempt_003: 6/6 shapes passed; original_passed=True; failures={}
shape_aware attempt_002: 6/6 shapes passed; original_passed=True; failures={}
```

Notes:

```text
This proves the Vast.ai automation path is working for a small CUDA evaluation
batch. The result should be treated as a first infrastructure smoke run, not a
final research conclusion, because only one task and one attempt per prompt
mode were evaluated.
```

## 2026-06-03 - Task 001 First Experiment Generation Batch

Decision:

```text
Begin the first real comparison batch with three new baseline attempts and
three new shape-aware attempts for task_001. Use Anthropic generation with
temperature 0.1 to reduce unnecessary sampling variance while still allowing
different attempts.
```

Generated attempts:

```text
baseline attempt_004: model=claude-sonnet-4-6, temperature=0.1, input_tokens=932, output_tokens=1402, stop=end_turn
baseline attempt_005: model=claude-sonnet-4-6, temperature=0.1, input_tokens=932, output_tokens=1250, stop=end_turn
baseline attempt_006: model=claude-sonnet-4-6, temperature=0.1, input_tokens=932, output_tokens=1309, stop=end_turn
shape_aware attempt_003: model=claude-sonnet-4-6, temperature=0.1, input_tokens=1092, output_tokens=806, stop=end_turn
shape_aware attempt_004: model=claude-sonnet-4-6, temperature=0.1, input_tokens=1092, output_tokens=995, stop=end_turn
shape_aware attempt_005: model=claude-sonnet-4-6, temperature=0.1, input_tokens=1092, output_tokens=799, stop=end_turn
```

Extraction and evaluation-contract prep:

```text
generated/baseline/task_001/attempt_004: solution.py:forward, cuda=add_relu_cuda.cu, function=add_relu, extension=add_relu_cuda_ext, fallback=false
generated/baseline/task_001/attempt_005: solution.py:forward, cuda=add_relu.cu, function=add_relu, extension=shapebench_task_001_baseline_attempt_005, fallback=true
generated/baseline/task_001/attempt_006: solution.py:forward, cuda=add_relu_cuda.cu, function=add_relu, extension=shapebench_task_001_baseline_attempt_006, fallback=true
generated/shape_aware/task_001/attempt_003: solution.py:forward, cuda=elementwise_add_relu.cu, function=elementwise_add_relu, extension=shapebench_task_001_shape_aware_attempt_003, fallback=true
generated/shape_aware/task_001/attempt_004: solution.py:forward, cuda=elementwise_add_relu.cu, function=elementwise_add_relu, extension=shapebench_task_001_shape_aware_attempt_004, fallback=true
generated/shape_aware/task_001/attempt_005: solution.py:forward, cuda=elementwise_add_relu.cu, function=elementwise_add_relu, extension=null, fallback=false
```

Tooling update before GPU run:

```text
The Vast runner now supports repeated --attempt arguments. This is required
because the old Vast flow ran only the default smoke-test pair, while this
experiment needs to evaluate the six newly generated attempts exactly.
```

Local validation:

```text
conda run -n shapebench-cuda pytest -q
77 passed in 2.25s
```

Open next step:

```text
Commit and push this batch, then run one guarded Vast.ai evaluation over the
six explicit attempts. The generated CUDA should be treated as experiment
data; GPU correctness/performance failures should be recorded, not patched
away, unless the failure comes from the harness itself.
```

### Task 001 Six-Attempt GPU Correctness Result

Validated run:

```text
Commit: 8b494c7 Add task 001 experiment generation batch
Run artifacts: results/vast_runs/20260603T142019Z
GPU: NVIDIA GeForce RTX 4090
PyTorch: 2.2.0
CUDA available: true
Remote tests: 77 passed
Evaluator exit code: 0
```

Correctness result:

```text
baseline attempt_004: 6/6 shapes passed
baseline attempt_005: 6/6 shapes passed
baseline attempt_006: 6/6 shapes passed
shape_aware attempt_003: 6/6 shapes passed
shape_aware attempt_004: 6/6 shapes passed
shape_aware attempt_005: 6/6 shapes passed
```

Shape coverage:

```text
original: 1024 x 1024
smaller: 512 x 1024
larger: 2048 x 1024
odd: 1007 x 1013
batch_variant: 256 x 1024
non_power_of_two: 1000 x 1007
```

Interpretation:

```text
For task_001, both baseline and shape-aware prompting produced generated CUDA
attempts that generalized correctly across all configured shape variants in
this first six-attempt batch. This task is an elementwise add+ReLU case, so it
may be too simple to expose meaningful shape-generalization differences.
```

Limitation:

```text
This run established correctness but did not produce runtime or speedup
measurements; generated_ms, pytorch_eager_ms, torch_compile_ms, and speedup
fields were null. The next harness improvement should add reliable timing so
we can compare robustness and performance together.
```

## 2026-06-03 - Evaluator Timing Instrumentation

Decision:

```text
Add timing to the evaluator before drawing prompt-mode or performance
conclusions. Correctness remains the gate: benchmark timings are collected only
for shapes that pass correctness.
```

Implementation:

```text
The evaluator now records PyTorch eager time, generated-kernel time, and
speedup_vs_eager in each per-shape JSONL result. Batch summaries also record
the benchmark warmup and iteration settings so later analysis can distinguish
full timing runs from smoke-test timing runs.
```

Local validation:

```text
conda run -n shapebench-cuda pytest -q
78 passed in 2.32s
```

Open next step:

```text
Run the six-attempt task_001 batch again on a GPU worker with timing enabled,
then compare correctness and speedup across baseline and shape-aware attempts.
```

### Task 001 Six-Attempt GPU Timing Result

Validated run:

```text
Commit: 96d2cb2 Add evaluator timing measurements
Run artifacts: results/vast_runs/20260603T182712Z
GPU: NVIDIA GeForce RTX 4090
PyTorch: 2.2.0
Benchmark settings: warmup=10, iterations=50
Remote tests: 78 passed
Evaluator exit code: 0
```

Correctness:

```text
baseline attempts 004, 005, 006: 18/18 shape rows passed
shape-aware attempts 003, 004, 005: 18/18 shape rows passed
total: 36/36 shape rows passed
```

Timing summary:

```text
baseline mean generated_ms: 0.01791
baseline mean pytorch_eager_ms: 0.02585
baseline mean speedup_vs_eager: 1.543x

shape-aware mean generated_ms: 0.02153
shape-aware mean pytorch_eager_ms: 0.02481
shape-aware mean speedup_vs_eager: 1.165x
```

Attempt-level speedup:

```text
baseline attempt_004: mean 2.096x, min 1.520x, max 2.351x
baseline attempt_005: mean 1.305x, min 1.120x, max 1.614x
baseline attempt_006: mean 1.229x, min 1.102x, max 1.420x
shape-aware attempt_003: mean 1.200x, min 1.116x, max 1.259x
shape-aware attempt_004: mean 1.215x, min 1.124x, max 1.274x
shape-aware attempt_005: mean 1.080x, min 0.916x, max 1.454x
```

Interpretation:

```text
For this simple elementwise task, all attempts generalized correctly across the
configured shape variants. The baseline group was faster on average in this
single timed run, mainly because baseline attempt_004 was substantially faster
than the other attempts. This is not enough evidence for prompt-mode
superiority because task_001 is simple, the sample is small, and there is only
one timed run on one GPU host.
```

Open next step:

```text
Add a harder task where shape assumptions are more likely to matter, such as a
reduction, transpose, tiled matmul-like operation, or non-contiguous/batched
case, then repeat the same baseline versus shape-aware generation and timed
GPU evaluation flow.
```

## 2026-06-04 - Task 002 Reduction Benchmark Setup

Decision:

```text
Add task_002 as a row-wise sum reduction benchmark. This is harder than the
elementwise add+ReLU task because generated kernels must reduce across a row
dimension and handle odd/non-power-of-two row widths without assuming a fixed
block-aligned size.
```

Task definition:

```text
task_id: task_002
name: rowwise_sum
operation: torch.sum(x, dim=1)
input: one float32 2D tensor
output: one float32 1D tensor with one sum per input row
shape variants: original, smaller, larger, odd, batch_variant, non_power_of_two
```

Generation batch:

```text
baseline attempt_001: model=claude-sonnet-4-6, temperature=0.1, input_tokens=940, output_tokens=1260
baseline attempt_002: model=claude-sonnet-4-6, temperature=0.1, input_tokens=940, output_tokens=1239
baseline attempt_003: model=claude-sonnet-4-6, temperature=0.1, input_tokens=940, output_tokens=1260
shape_aware attempt_001: model=claude-sonnet-4-6, temperature=0.1, input_tokens=1100, output_tokens=944
shape_aware attempt_002: model=claude-sonnet-4-6, temperature=0.1, input_tokens=1100, output_tokens=1077
shape_aware attempt_003: model=claude-sonnet-4-6, temperature=0.1, input_tokens=1100, output_tokens=1097
```

Extraction and contract prep:

```text
baseline attempt_001: solution.py:forward, cuda=rowwise_sum.cu, function=rowwise_sum, fallback=true
baseline attempt_002: solution.py:forward, cuda=rowwise_sum.cu, function=rowwise_sum, fallback=true
baseline attempt_003: solution.py:forward, cuda=rowwise_sum.cu, function=rowwise_sum, fallback=true
shape_aware attempt_001: solution.py:forward, cuda=rowwise_sum.cu, function=rowwise_sum, fallback=true
shape_aware attempt_002: solution.py:forward, cuda=rowwise_sum.cu, function=rowwise_sum, fallback=false
shape_aware attempt_003: solution.py:forward, cuda=rowwise_sum.cu, function=rowwise_sum, fallback=false
```

Local validation:

```text
conda run -n shapebench-cuda pytest -q
83 passed in 2.61s
```

Open next step:

```text
Run a guarded Vast.ai GPU evaluation for these six task_002 attempts with
correctness and timing enabled. Treat compilation/correctness failures as
research data unless the harness itself is at fault.
```

## 2026-06-04 - Task 002 Contract Cleanup and Task 003 Transpose Setup

Task 002 cleanup:

```text
The first task_002 GPU run passed correctness, but shape_aware attempts 002 and
003 used the same generated extension name while their CUDA sources differed.
To keep timing and per-attempt attribution clean, both were converted to the
standard fallback wrapper with unique extension names:

shape_aware attempt_002: shapebench_task_002_shape_aware_attempt_002
shape_aware attempt_003: shapebench_task_002_shape_aware_attempt_003
```

Decision:

```text
Add task_003 as a rectangular 2D transpose benchmark. This is harder than
elementwise add and different from row-wise reduction because generated kernels
must map row/column indices correctly and write an output tensor with swapped
dimensions. The original shape is rectangular rather than square so transposed
indexing mistakes are easier to detect.
```

Task definition:

```text
task_id: task_003
name: matrix_transpose
operation: torch.transpose(x, 0, 1).contiguous()
input: one float32 2D tensor
output: one float32 2D tensor with rows and columns swapped
shape variants: original, smaller, larger, odd, batch_variant, non_power_of_two
```

Generation batch:

```text
baseline attempt_001: model=claude-sonnet-4-6, temperature=0.1, input_tokens=956, output_tokens=976
baseline attempt_002: model=claude-sonnet-4-6, temperature=0.1, input_tokens=956, output_tokens=1007
baseline attempt_003: model=claude-sonnet-4-6, temperature=0.1, input_tokens=956, output_tokens=1055
shape_aware attempt_001: model=claude-sonnet-4-6, temperature=0.1, input_tokens=1116, output_tokens=2054
shape_aware attempt_002: model=claude-sonnet-4-6, temperature=0.1, input_tokens=1116, output_tokens=1108
shape_aware attempt_003: model=claude-sonnet-4-6, temperature=0.1, input_tokens=1116, output_tokens=1099
```

Extraction and contract prep:

```text
All six task_003 attempts produced one CUDA source. All six were prepared with
the standard fallback wrapper so each attempt has a unique extension name before
GPU benchmarking.
```

Local validation:

```text
conda run -n shapebench-cuda pytest -q
87 passed in 2.48s
```

Open next step:

```text
Run a guarded GPU evaluation for task_002 and task_003 together after committing
the setup, then compare correctness and timing across the two harder tasks.
```

## 2026-06-04 - Task 002 and Task 003 GPU Timing Result

GPU run:

```text
run directory: results/vast_runs/20260603T191105Z
GPU: NVIDIA GeForce RTX 4090
PyTorch: 2.2.0
CUDA available: true
nvcc: 12.1
remote tests: 87 passed
evaluator exit code: 0
instance cleanup: destroyed
benchmark warmup: 10 iterations
benchmark timing: 50 iterations
```

Correctness:

```text
task_002 rowwise_sum: 36/36 shape checks passed
task_003 matrix_transpose: 36/36 shape checks passed

All six shape categories passed for both tasks:
original, smaller, larger, odd, batch_variant, non_power_of_two

task_002 max_abs_error across this run: 1.52587890625e-05
task_003 max_abs_error across this run: 0.0
```

Mean timing by task and prompt mode:

```text
task_id   prompt_mode   checks_passed   generated_ms   pytorch_eager_ms   mean_speedup_vs_eager
task_002  baseline      18/18           0.007          0.007              0.930
task_002  shape_aware   18/18           0.012          0.008              0.740
task_003  baseline      18/18           0.009          0.034              3.067
task_003  shape_aware   18/18           0.009          0.034              3.175
```

Mean timing by attempt:

```text
task_002 baseline attempt_001: 6/6 passed, generated_ms=0.006, pytorch_eager_ms=0.005, speedup=0.935
task_002 baseline attempt_002: 6/6 passed, generated_ms=0.006, pytorch_eager_ms=0.005, speedup=0.924
task_002 baseline attempt_003: 6/6 passed, generated_ms=0.011, pytorch_eager_ms=0.010, speedup=0.932
task_002 shape_aware attempt_001: 6/6 passed, generated_ms=0.015, pytorch_eager_ms=0.009, speedup=0.659
task_002 shape_aware attempt_002: 6/6 passed, generated_ms=0.015, pytorch_eager_ms=0.009, speedup=0.681
task_002 shape_aware attempt_003: 6/6 passed, generated_ms=0.007, pytorch_eager_ms=0.006, speedup=0.881
task_003 baseline attempt_001: 6/6 passed, generated_ms=0.009, pytorch_eager_ms=0.034, speedup=3.227
task_003 baseline attempt_002: 6/6 passed, generated_ms=0.010, pytorch_eager_ms=0.034, speedup=2.815
task_003 baseline attempt_003: 6/6 passed, generated_ms=0.009, pytorch_eager_ms=0.034, speedup=3.158
task_003 shape_aware attempt_001: 6/6 passed, generated_ms=0.009, pytorch_eager_ms=0.034, speedup=3.238
task_003 shape_aware attempt_002: 6/6 passed, generated_ms=0.009, pytorch_eager_ms=0.033, speedup=3.156
task_003 shape_aware attempt_003: 6/6 passed, generated_ms=0.009, pytorch_eager_ms=0.033, speedup=3.131
```

Interpretation:

```text
The harder task_003 transpose benchmark produced clean correctness across all
configured shape variants for both prompt modes. Unlike task_002, transpose also
showed a clear speedup over PyTorch eager execution in this run, around 3x on
average. The prompt-mode difference is still not conclusive: both baseline and
shape-aware attempts passed all shapes, and their task_003 timing was very
similar. Current evidence supports that the benchmark harness can capture both
correctness and timing on a layout-sensitive task, but it still does not show a
shape-aware robustness advantage.
```

Open next step:

```text
Add a task where shape-aware prompting is more likely to matter for correctness,
such as tiled matrix multiplication or non-contiguous/batched inputs. Also add
per-attempt progress logging to the remote evaluator so long CUDA compilation
phases are easier to monitor during larger batches.
```
