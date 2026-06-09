#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate TCPC25_03 LBM geometry.vtk from STL.")
    parser.add_argument("--config", default="configs/simulation.yaml")
    parser.add_argument("--longest-axis-voxels", type=int, default=None)
    parser.add_argument("--output", default=None, help="Output VTK path; default from simulation config.")
    parser.add_argument("--test-output", default=None, help="Output pre-PE label VTK path.")
    args = parser.parse_args()

    root = repo_root()
    with (root / args.config).open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    geom_cfg = config["geometry"]

    geometry_dir = root / "geometry"
    sys.path.insert(0, str(geometry_dir))
    import tcpc_geometry  # type: ignore

    output_vtk = root / (args.output or geom_cfg["runtime_vtk"])
    test_vtk = root / (args.test_output or geom_cfg["test_vtk"])
    tapered_stl = root / geom_cfg["tapered_stl"]
    output_vtk.parent.mkdir(parents=True, exist_ok=True)
    test_vtk.parent.mkdir(parents=True, exist_ok=True)
    tapered_stl.parent.mkdir(parents=True, exist_ok=True)

    tcpc_geometry.INPUT_STL = str(root / geom_cfg["input_stl"])
    tcpc_geometry.OUTPUT_STL = str(tapered_stl)
    tcpc_geometry.LONGEST_AXIS_VOXELS = int(
        args.longest_axis_voxels or geom_cfg["longest_axis_voxels"]
    )
    tcpc_geometry.PE_EXTENSION_LEFT_M = float(geom_cfg["pe_extension_left_m"])
    tcpc_geometry.PE_EXTENSION_RIGHT_M = float(geom_cfg["pe_extension_right_m"])
    tcpc_geometry.PE_OUTLET_RADIUS_M = float(geom_cfg["pe_outlet_radius_m"])
    tcpc_geometry.PE_BLOCK_SIZE = int(geom_cfg["pe_block_size"])
    tcpc_geometry.OUTPUT_DIR = str(output_vtk.parent / "mesh_triplet")
    tcpc_geometry.SHOW_MAYAVI = False

    old_cwd = Path.cwd()
    os.chdir(output_vtk.parent)
    try:
        tcpc_geometry.main()
    finally:
        os.chdir(old_cwd)

    generated_geometry = output_vtk.parent / "geometry.vtk"
    generated_test = output_vtk.parent / "test.vtk"
    if generated_geometry != output_vtk:
        generated_geometry.replace(output_vtk)
    if generated_test.exists() and generated_test != test_vtk:
        generated_test.replace(test_vtk)

    print(f"geometry_vtk={output_vtk}")
    print(f"test_vtk={test_vtk}")
    print(f"tapered_stl={tapered_stl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
