#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_command(root: Path, regime: str, material: str, case_id: str | None = None) -> list[str]:
    sim_cfg = load_yaml(root / "configs/simulation.yaml")
    regimes = load_yaml(root / "configs/regimes.yaml")["regimes"]
    if regime not in regimes:
        raise ValueError(f"Unknown regime '{regime}'. Expected one of {sorted(regimes)}")
    if material not in sim_cfg["cases"]["materials"]:
        raise ValueError(f"Unknown material '{material}'. Expected one of {sim_cfg['cases']['materials']}")

    solver_cfg = sim_cfg["solver"]
    geometry = root / sim_cfg["geometry"]["runtime_vtk"]
    executable = root / solver_cfg["executable"]
    rid = case_id or f"{regime}_{material}"
    flow = regimes[regime]
    return [
        str(executable),
        "--geometry", str(geometry),
        "--material", material,
        "--case-id", rid,
        "--ivc-mls", str(flow["ivc_mls"]),
        "--svc-mls", str(flow["svc_mls"]),
        "--final-time", str(solver_cfg["final_time_s"]),
        "--vtk-period", str(solver_cfg["vtk_period_s"]),
        "--print-period", str(solver_cfg["print_period_s"]),
        "--stat-reset-period", str(solver_cfg["stat_reset_period_s"]),
        "--lbm-viscosity", str(solver_cfg["lbm_viscosity"]),
        "--block-size", str(solver_cfg["block_size"]),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or print one TCPC non-Newtonian case.")
    parser.add_argument("--regime", required=True)
    parser.add_argument("--material", required=True, choices=["newtonian", "sx", "gx"])
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    cmd = build_command(root, args.regime, args.material, args.case_id)
    print(" ".join(shlex.quote(part) for part in cmd))
    if args.dry_run:
        return 0
    return subprocess.call(cmd, cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
