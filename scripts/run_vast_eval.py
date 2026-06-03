#!/usr/bin/env python
"""Launch a Vast.ai GPU instance, run ShapeBench-CUDA evaluation, and destroy it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.vast_runner import (  # noqa: E402
    DEFAULT_DISK_GB,
    DEFAULT_IMAGE,
    DEFAULT_MAX_SSH_AUTH_FAILURES,
    DEFAULT_TEMPLATE_HASH,
    VastRunConfig,
    run_vast_eval,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offer-id", type=int, required=True, help="Vast.ai offer id from `vastai search offers`.")
    parser.add_argument(
        "--template-hash",
        default=DEFAULT_TEMPLATE_HASH,
        help="Vast.ai template hash to launch. Pass an empty string with --image to use raw image mode.",
    )
    parser.add_argument(
        "--image",
        default=None,
        help=f"Raw Docker image fallback. Defaults to {DEFAULT_IMAGE!r} only when --template-hash is empty.",
    )
    parser.add_argument("--disk", type=int, default=DEFAULT_DISK_GB, help="Disk size in GB.")
    parser.add_argument("--repo-ref", default="HEAD", help="Git ref to archive and upload.")
    parser.add_argument("--remote-dir", default="/root/shape_bench_CUDA", help="Remote project directory.")
    parser.add_argument(
        "--attempt",
        action="append",
        dest="attempts",
        help="Attempt directory to evaluate on the GPU worker. Can be passed multiple times.",
    )
    parser.add_argument("--poll-seconds", type=int, default=10, help="Seconds between SSH readiness checks.")
    parser.add_argument("--max-wait-seconds", type=int, default=600, help="Maximum wait for SSH readiness.")
    parser.add_argument(
        "--max-ssh-auth-failures",
        type=int,
        default=DEFAULT_MAX_SSH_AUTH_FAILURES,
        help="Abort after this many repeated SSH public-key failures.",
    )
    parser.add_argument(
        "--keep-instance",
        action="store_true",
        help="Do not destroy the Vast instance after the run. Use only for debugging.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow a dirty local working tree. The uploaded archive still uses the committed git ref.",
    )
    parser.add_argument("--skip-tests", action="store_true", help="Skip remote pytest before GPU evaluation.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = VastRunConfig(
        offer_id=args.offer_id,
        project_root=ROOT,
        image=args.image or (DEFAULT_IMAGE if not args.template_hash else None),
        template_hash=args.template_hash or None,
        disk_gb=args.disk,
        remote_dir=args.remote_dir,
        repo_ref=args.repo_ref,
        poll_seconds=args.poll_seconds,
        max_wait_seconds=args.max_wait_seconds,
        keep_instance=args.keep_instance,
        allow_dirty=args.allow_dirty,
        skip_tests=args.skip_tests,
        attempt_dirs=tuple(args.attempts or ()),
        max_ssh_auth_failures=args.max_ssh_auth_failures,
    )
    try:
        result = run_vast_eval(config)
    except KeyboardInterrupt:
        print("Vast run interrupted after cleanup attempt.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Vast run failed: {exc}", file=sys.stderr)
        return 1

    print(f"Vast instance: {result.instance_id}")
    print(f"Local run dir: {result.local_run_dir}")
    print(f"Remote exit code: {result.remote_exit_code}")
    print(f"Destroyed: {result.destroyed}")
    return result.remote_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
