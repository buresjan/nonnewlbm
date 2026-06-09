#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import vtk


DEFAULT_LABELS = {
    3: "IVC",
    4: "SVC",
    5: "RPA",
    6: "LPA",
}


@dataclass
class VtkArrays:
    wall: vtk.vtkDataArray
    velocity: vtk.vtkDataArray
    rho: vtk.vtkDataArray | None
    dims: tuple[int, int, int]
    dx: float


def get_array(data, name: str):
    arr = data.GetArray(name)
    if arr is None:
        raise KeyError(name)
    return arr


def read_vtk(path: Path) -> VtkArrays:
    reader = vtk.vtkRectilinearGridReader()
    reader.SetFileName(str(path))
    reader.ReadAllScalarsOn()
    reader.ReadAllVectorsOn()
    reader.Update()
    grid = reader.GetOutput()
    dims = grid.GetDimensions()
    dx = abs(grid.GetPoint(1)[0] - grid.GetPoint(0)[0]) if dims[0] > 1 else 1.0

    point = grid.GetPointData()
    cell = grid.GetCellData()
    wall = point.GetArray("wall") or cell.GetArray("wall")
    velocity = point.GetArray("velocity") or cell.GetArray("velocity")
    rho = point.GetArray("lbm_rho") or cell.GetArray("lbm_rho")
    if wall is None:
        raise RuntimeError(f"{path} has no 'wall' array")
    if velocity is None:
        raise RuntimeError(f"{path} has no 'velocity' array")
    return VtkArrays(wall=wall, velocity=velocity, rho=rho, dims=dims, dx=dx)


def flat_index(x: int, y: int, z: int, dims: tuple[int, int, int]) -> int:
    nx, ny, _ = dims
    return x + nx * (y + ny * z)


def label_coords(arrays: VtkArrays, label: int) -> np.ndarray:
    nx, ny, nz = arrays.dims
    coords = []
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                i = flat_index(x, y, z, arrays.dims)
                if int(round(arrays.wall.GetComponent(i, 0))) == label:
                    coords.append((x, y, z))
    return np.asarray(coords, dtype=int)


def cap_axis_and_sign(coords: np.ndarray, dims: tuple[int, int, int]) -> tuple[int, float]:
    if coords.size == 0:
        return 0, 1.0
    spread = coords.max(axis=0) - coords.min(axis=0)
    axis = int(np.argmin(spread))
    center = float(coords[:, axis].mean())
    domain_mid = 0.5 * (dims[axis] - 1)
    sign = -1.0 if center < domain_mid else 1.0
    return axis, sign


def integrate_label(arrays: VtkArrays, label: int, density_weighted: bool = False) -> tuple[float, int]:
    coords = label_coords(arrays, label)
    if coords.size == 0:
        return 0.0, 0
    axis, normal_sign = cap_axis_and_sign(coords, arrays.dims)
    area = arrays.dx * arrays.dx
    flux = 0.0
    for x, y, z in coords:
        i = flat_index(int(x), int(y), int(z), arrays.dims)
        value = arrays.velocity.GetComponent(i, axis) * normal_sign * area
        if density_weighted and arrays.rho is not None:
            value *= arrays.rho.GetComponent(i, 0)
        flux += value
    return flux, int(coords.shape[0])


def liter_per_min(m3_s: float) -> float:
    return m3_s * 1000.0 * 60.0


def compute_file(path: Path, label_names: dict[int, str]) -> dict[str, float | int | str]:
    arrays = read_vtk(path)
    row: dict[str, float | int | str] = {"file": str(path)}
    signed: dict[str, float] = {}
    for label, name in label_names.items():
        flux, count = integrate_label(arrays, label)
        # Inlet labels have outward-negative flux; report positive inflow.
        if name in {"IVC", "SVC"}:
            reported = -flux
        else:
            reported = flux
        signed[name] = reported
        row[f"{name}_l_min"] = abs(liter_per_min(reported))
        row[f"{name}_cells"] = count
    inlet = abs(signed.get("IVC", 0.0)) + abs(signed.get("SVC", 0.0))
    outlet = abs(signed.get("RPA", 0.0)) + abs(signed.get("LPA", 0.0))
    row["mass_balance_l_min"] = liter_per_min(inlet - outlet)
    denom = abs(signed.get("RPA", 0.0)) + abs(signed.get("LPA", 0.0))
    row["lpa_split_percent"] = 100.0 * abs(signed.get("LPA", 0.0)) / denom if denom > 0 else 0.0
    row["rpa_split_percent"] = 100.0 * abs(signed.get("RPA", 0.0)) / denom if denom > 0 else 0.0
    return row


def parse_label_map(text: str | None) -> dict[int, str]:
    if not text:
        return DEFAULT_LABELS.copy()
    out: dict[int, str] = {}
    for part in text.split(","):
        label, name = part.split(":", 1)
        out[int(label)] = name
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute TCPC inlet/outlet flows and LPA/RPA split.")
    parser.add_argument("paths", nargs="+", help="VTK files or glob patterns.")
    parser.add_argument("--labels", default=None, help="Comma list like 3:IVC,4:SVC,5:RPA,6:LPA.")
    parser.add_argument("--output", default="flows.csv")
    args = parser.parse_args()

    files: list[Path] = []
    for item in args.paths:
        matches = sorted(glob.glob(item))
        files.extend(Path(m) for m in (matches or [item]))
    label_names = parse_label_map(args.labels)
    rows = [compute_file(path, label_names) for path in files]
    if not rows:
        return 0
    fieldnames = list(rows[0].keys())
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
