from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_local_setup_script_passes() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "check_local_setup.py"

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "Local setup check passed." in completed.stdout
    assert "torch.cuda.is_available()" in completed.stdout or "torch: not installed" in completed.stdout

