#!/usr/bin/env bash
set -euo pipefail

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is not available on PATH. Install Conda or activate the environment manually." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate shapebench-cuda

python scripts/check_local_setup.py
