# TCPC25_03 Non-Newtonian LBM Setup

This repository combines the working TCPC phantom LBM setup with the older Carreau-Yasuda non-Newtonian LBM implementation. It is configured for the `#04 PE simulace` regimes `v1`, `v2`, `v3`, `v4a`, and `v4b`, each runnable as:

- `newtonian`
- `sx` (saccharose/SX constants from the historical non-Newtonian code)
- `gx` (glycerol/GX constants from the historical non-Newtonian code)

Generated `geometry.vtk`, solver binaries, and simulation results are intentionally not tracked.

## Prerequisites

Python:

```bash
python3 -m pip install numpy scipy trimesh vtk pyyaml
```

Solver machine:

- CUDA compiler (`nvcc`)
- MPI compiler wrapper (`mpicxx`)
- VTK C++ development headers/libraries
- TNL checkout compatible with the historical solver

The build uses `TNL_DIR` to find TNL. Example:

```bash
scripts/build_solver.sh TNL_DIR=/home/bures/geraldine/Eichler/2026/2026_02_29-nonNew/Glycerol-nonNew/tnl_submodule
```

If VTK is not discoverable through `pkg-config`, pass `VTK_CONFIG`:

```bash
scripts/build_solver.sh \
  TNL_DIR=/path/to/tnl_submodule \
  VTK_CONFIG="-I/usr/include/vtk -lvtkCommonCore -lvtkIOLegacy -lvtkCommonDataModel -lvtkIOXML"
```

## Geometry

Generate the runtime geometry from the tracked STL:

```bash
python3 scripts/generate_geometry.py
```

This writes:

- `runtime/geometry.vtk`: solver input, with `wall` labels
- `runtime/test.vtk`: pre-PE label check
- `runtime/TCPC25_03_tapered.stl`: tapered/capped intermediate STL

The default geometry reproduces the historical PE setup: 25 mm outlet extensions and 4 mm outlet diameter.

The generated VTK file is used as the wall mask. To match the historical working TCPC solver, the simulation imposes open boundary planes itself: x-min is IVC, x-max is SVC, y-min is outlet label `5`, and y-max is outlet label `6`. Wall cells from the VTK geometry override those plane labels.

## Build

```bash
scripts/build_solver.sh TNL_DIR=/path/to/tnl_submodule
```

The solver executable is:

```bash
solver/sim_tcpc/sim_tcpc
```

The build still needs TNL headers for the lattice arrays, but the unused
immersed-boundary TNL sparse-matrix branch is disabled by default. Re-enable it
only for IBM/Lagrange experiments with `use_TNL_LAGRANGE=yes`.

## Run Cases

Print all configured commands:

```bash
python3 scripts/run_all_cases.py --dry-run
```

Run one case:

```bash
python3 scripts/run_case.py --regime v1 --material sx
```

The solver writes results under `results_<case-id>/`, for example `results_v1_sx/`.

The command generated for each case passes the prescribed IVC/SVC flow rates from `configs/regimes.yaml` directly in ml/s.

## Outputs

VTK output includes:

- `wall`
- `velocity`
- `mean_velocity`
- `lbm_rho`
- `pressure_pa`
- `gamma_dot`
- `nu_phys_m2_s`
- `mu_phys_pa_s`
- `turbulence_intensity`
- `rms_velocity`
- `strain_diag_1_s`
- `strain_shear_1_s`
- `wss_proxy_pa`

`wss_proxy_pa` is an approximate voxel-gradient wall-shear proxy on fluid cells adjacent to wall cells. It is not an exact stress projected onto the original STL surface.

## Flow Split Postprocessing

Compute inlet/outlet flows and LPA/RPA split from VTK files:

```bash
python3 postprocess/compute_flows.py 'results_v1_sx/vtk3D/data_*.vtk' --output results_v1_sx/flows.csv
```

Default label names are:

- `3:IVC`
- `4:SVC`
- `5:RPA`
- `6:LPA`

Override if visual inspection of the generated geometry shows the pulmonary labels are swapped:

```bash
python3 postprocess/compute_flows.py 'results_v1_sx/vtk3D/data_*.vtk' \
  --labels 3:IVC,4:SVC,5:LPA,6:RPA \
  --output results_v1_sx/flows.csv
```

## Source Directories Used

- Newtonian TCPC geometry/flow setup: `2026_02_20-simulationSetup`
- Non-Newtonian Carreau-Yasuda LBM implementation: `2026_02_29-nonNew`
- Historical flow postprocessing reference: `2026_03_18-zpracFlows`
