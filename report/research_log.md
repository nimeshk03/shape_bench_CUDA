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
