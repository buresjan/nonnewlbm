#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

import yaml

from run_case import build_command, repo_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or print all configured TCPC cases.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--materials", nargs="*", default=None)
    parser.add_argument("--regimes", nargs="*", default=None)
    args = parser.parse_args()

    root = repo_root()
    sim_cfg = yaml.safe_load((root / "configs/simulation.yaml").read_text(encoding="utf-8"))
    materials = args.materials or sim_cfg["cases"]["materials"]
    regimes = args.regimes or sim_cfg["cases"]["regimes"]

    for regime in regimes:
        for material in materials:
            cmd = build_command(root, regime, material)
            print(" ".join(shlex.quote(part) for part in cmd))
            if not args.dry_run:
                rc = subprocess.call(cmd, cwd=root)
                if rc != 0:
                    return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
