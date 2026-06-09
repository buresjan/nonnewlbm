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


def build_command(
    root: Path,
    regime: str,
    material: str,
    case_id: str | None = None,
    *,
    geometry: str | None = None,
    final_time: float | None = None,
    vtk_period: float | None = None,
    print_period: float | None = None,
    stat_reset_period: float | None = None,
    lbm_viscosity: float | None = None,
    block_size: int | None = None,
) -> list[str]:
    sim_cfg = load_yaml(root / "configs/simulation.yaml")
    regimes = load_yaml(root / "configs/regimes.yaml")["regimes"]
    if regime not in regimes:
        raise ValueError(f"Unknown regime '{regime}'. Expected one of {sorted(regimes)}")
    if material not in sim_cfg["cases"]["materials"]:
        raise ValueError(f"Unknown material '{material}'. Expected one of {sim_cfg['cases']['materials']}")

    solver_cfg = sim_cfg["solver"]
    geometry_path = root / (geometry or sim_cfg["geometry"]["runtime_vtk"])
    executable = root / solver_cfg["executable"]
    rid = case_id or f"{regime}_{material}"
    flow = regimes[regime]
    return [
        str(executable),
        "--geometry", str(geometry_path),
        "--material", material,
        "--case-id", rid,
        "--ivc-mls", str(flow["ivc_mls"]),
        "--svc-mls", str(flow["svc_mls"]),
        "--final-time", str(final_time if final_time is not None else solver_cfg["final_time_s"]),
        "--vtk-period", str(vtk_period if vtk_period is not None else solver_cfg["vtk_period_s"]),
        "--print-period", str(print_period if print_period is not None else solver_cfg["print_period_s"]),
        "--stat-reset-period", str(stat_reset_period if stat_reset_period is not None else solver_cfg["stat_reset_period_s"]),
        "--lbm-viscosity", str(lbm_viscosity if lbm_viscosity is not None else solver_cfg["lbm_viscosity"]),
        "--block-size", str(block_size if block_size is not None else solver_cfg["block_size"]),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or print one TCPC non-Newtonian case.")
    parser.add_argument("--regime", required=True)
    parser.add_argument("--material", required=True, choices=["newtonian", "sx", "gx"])
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--geometry", default=None)
    parser.add_argument("--final-time", type=float, default=None)
    parser.add_argument("--vtk-period", type=float, default=None)
    parser.add_argument("--print-period", type=float, default=None)
    parser.add_argument("--stat-reset-period", type=float, default=None)
    parser.add_argument("--lbm-viscosity", type=float, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = repo_root()
    cmd = build_command(
        root,
        args.regime,
        args.material,
        args.case_id,
        geometry=args.geometry,
        final_time=args.final_time,
        vtk_period=args.vtk_period,
        print_period=args.print_period,
        stat_reset_period=args.stat_reset_period,
        lbm_viscosity=args.lbm_viscosity,
        block_size=args.block_size,
    )
    print(" ".join(shlex.quote(part) for part in cmd))
    if args.dry_run:
        return 0
    return subprocess.call(cmd, cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
