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
    parser.add_argument("--geometry", default=None)
    parser.add_argument("--final-time", type=float, default=None)
    parser.add_argument("--vtk-period", type=float, default=None)
    parser.add_argument("--print-period", type=float, default=None)
    parser.add_argument("--stat-reset-period", type=float, default=None)
    parser.add_argument("--lbm-viscosity", type=float, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    args = parser.parse_args()

    root = repo_root()
    sim_cfg = yaml.safe_load((root / "configs/simulation.yaml").read_text(encoding="utf-8"))
    materials = args.materials or sim_cfg["cases"]["materials"]
    regimes = args.regimes or sim_cfg["cases"]["regimes"]

    for regime in regimes:
        for material in materials:
            cmd = build_command(
                root,
                regime,
                material,
                geometry=args.geometry,
                final_time=args.final_time,
                vtk_period=args.vtk_period,
                print_period=args.print_period,
                stat_reset_period=args.stat_reset_period,
                lbm_viscosity=args.lbm_viscosity,
                block_size=args.block_size,
            )
            print(" ".join(shlex.quote(part) for part in cmd))
            if not args.dry_run:
                rc = subprocess.call(cmd, cwd=root)
                if rc != 0:
                    return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
