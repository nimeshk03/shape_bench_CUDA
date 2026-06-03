#!/usr/bin/env python
"""Check local Vast.ai CLI and SSH setup without launching an instance."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_PUBLIC_KEY = Path.home() / ".ssh" / "id_ed25519.pub"


def main() -> int:
    public_key_path = DEFAULT_PUBLIC_KEY
    failures: list[str] = []

    print("ShapeBench-CUDA Vast.ai setup check")
    print(f"Local public key: {public_key_path}")

    vast_version = _run(["vastai", "--version"])
    if vast_version.returncode != 0:
        print("vastai CLI: MISSING or not runnable")
        print(vast_version.stderr.strip())
        return 1
    print(f"vastai CLI: {vast_version.stdout.strip() or 'installed'}")

    if not public_key_path.exists():
        print("Local SSH key: MISSING")
        return 1

    local_public_key = public_key_path.read_text(encoding="utf-8").strip()
    local_fingerprint = _fingerprint(local_public_key)
    print(f"Local SSH key fingerprint: {local_fingerprint}")

    keys_result = _run(["vastai", "show", "ssh-keys", "--raw"])
    if keys_result.returncode != 0:
        print("Vast SSH keys: could not read")
        print(keys_result.stderr.strip())
        return 1

    registered_keys = _parse_cli_value(keys_result.stdout)
    if not isinstance(registered_keys, list):
        print("Vast SSH keys: unexpected CLI output")
        return 1

    registered_fingerprints = []
    print("Registered Vast SSH key fingerprints:")
    for key in registered_keys:
        if not isinstance(key, dict) or not key.get("public_key"):
            continue
        fingerprint = _fingerprint(str(key["public_key"]).strip())
        registered_fingerprints.append(fingerprint)
        print(f"  id={key.get('id')}: {fingerprint}")

    if local_fingerprint not in registered_fingerprints:
        failures.append("local public key is not registered with Vast.ai")

    instances_result = _run(["vastai", "show", "instances", "--raw"])
    if instances_result.returncode == 0:
        active_instances = _parse_cli_value(instances_result.stdout)
        active_count = len(active_instances) if isinstance(active_instances, list) else "unknown"
        print(f"Active Vast instances: {active_count}")
    else:
        print("Active Vast instances: could not read")

    if failures:
        print("Vast setup check failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("Vast setup check passed.")
    return 0


def _fingerprint(public_key: str) -> str:
    completed = subprocess.run(
        ["ssh-keygen", "-lf", "-"],
        input=public_key + "\n",
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"ssh-keygen failed: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _parse_cli_value(output: str) -> Any:
    text = output.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=False, capture_output=True, text=True)


if __name__ == "__main__":
    raise SystemExit(main())
