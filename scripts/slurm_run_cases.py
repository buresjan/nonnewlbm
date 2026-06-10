#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

import yaml

from run_case import build_command, repo_root


COMPLETED_STATES = {"COMPLETED"}
FAILED_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "STOPPED",
    "TIMEOUT",
}
TERMINAL_STATES = COMPLETED_STATES | FAILED_STATES
ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass
class CaseJob:
    regime: str
    material: str
    case_id: str
    run_dir: Path
    command: list[str]
    job_name: str
    sbatch_path: Path
    manifest_path: Path
    result_dir: Path
    job_id: str | None = None
    state: str = "NOT_SUBMITTED"


def iso_timestamp() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def run_id() -> str:
    return "run-" + datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def natural_key(path: str | Path) -> list[int | str]:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", str(path))]


def safe_token(text: str, *, max_len: int = 64) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in text)
    cleaned = cleaned.strip("-_") or "case"
    return cleaned[:max_len]


def shell_array(name: str, values: Sequence[str]) -> list[str]:
    lines = [f"{name}=("]
    lines.extend(f"    {shlex.quote(value)}" for value in values)
    lines.append(")")
    return lines


def read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def update_manifest(job: CaseJob, updates: dict) -> None:
    data = read_json(job.manifest_path) if job.manifest_path.exists() else {}
    data.update(updates)
    write_json(job.manifest_path, data)


def slurm_state_head(raw_state: str) -> str:
    return raw_state.strip().upper().split()[0] if raw_state.strip() else "UNKNOWN"


def render_sbatch(
    *,
    root: Path,
    job: CaseJob,
    args: argparse.Namespace,
) -> str:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job.job_name}",
        "#SBATCH --output=stdout.log",
        "#SBATCH --error=stderr.log",
        f"#SBATCH --time={args.time}",
        f"#SBATCH --cpus-per-task={args.cpus_per_task}",
        f"#SBATCH --mem={args.mem}",
    ]
    if args.partition:
        lines.append(f"#SBATCH --partition={args.partition}")
    if args.account:
        lines.append(f"#SBATCH --account={args.account}")
    if args.qos:
        lines.append(f"#SBATCH --qos={args.qos}")
    if args.constraint:
        lines.append(f"#SBATCH --constraint={args.constraint}")
    if args.gres:
        lines.append(f"#SBATCH --gres={args.gres}")
    elif args.gpus is not None and args.gpus > 0:
        lines.append(f"#SBATCH --gpus={args.gpus}")
    if args.threads_per_core is not None:
        lines.append(f"#SBATCH --threads-per-core={args.threads_per_core}")
    for extra in args.extra_sbatch:
        lines.append(f"#SBATCH {extra}")

    lines.extend(
        [
            "",
            "set -euo pipefail",
            f"REPO_ROOT={shlex.quote(str(root))}",
            f"SOLVER_BINARY={shlex.quote(job.command[0])}",
            f"RESULT_DIR={shlex.quote(str(job.result_dir))}",
            "",
            'cd "$REPO_ROOT"',
            'export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"',
            "export OMPI_MCA_accelerator=cuda",
            "",
            'if [ ! -x "$SOLVER_BINARY" ]; then',
            '    echo "Solver binary $SOLVER_BINARY is missing or not executable." >&2',
            "    exit 3",
            "fi",
        ]
    )

    if args.overwrite_results:
        lines.extend(
            [
                "",
                'if [ -e "$RESULT_DIR" ]; then',
                '    echo "Removing existing $RESULT_DIR" >&2',
                '    rm -rf "$RESULT_DIR"',
                "fi",
            ]
        )
    else:
        lines.extend(
            [
                "",
                'if [ -e "$RESULT_DIR" ]; then',
                '    echo "Result directory $RESULT_DIR already exists; use --overwrite-results to replace it." >&2',
                "    exit 4",
                "fi",
            ]
        )

    for item in args.env:
        name, _, value = item.partition("=")
        if not name or not _:
            raise ValueError(f"--env must be NAME=VALUE, got {item!r}")
        if ENV_NAME_RE.fullmatch(name) is None:
            raise ValueError(f"Invalid environment variable name in --env: {name!r}")
        lines.append(f"export {name}={shlex.quote(value)}")

    lines.extend(
        [
            "",
            *shell_array("CMD", job.command),
            "",
            'echo "case_id: ' + job.case_id + '"',
            'echo "start: $(date --iso-8601=seconds)"',
            'printf "command:"',
            'printf " %q" "${CMD[@]}"',
            'printf "\\n"',
            '"${CMD[@]}"',
            'echo "end: $(date --iso-8601=seconds)"',
        ]
    )
    return "\n".join(lines) + "\n"


def build_jobs(root: Path, args: argparse.Namespace) -> tuple[Path, list[CaseJob]]:
    sim_cfg = read_yaml(root / "configs" / "simulation.yaml")
    regimes_cfg = read_yaml(root / "configs" / "regimes.yaml")["regimes"]
    regimes = args.regimes or sim_cfg["cases"]["regimes"]
    materials = args.materials or sim_cfg["cases"]["materials"]

    unknown_regimes = [regime for regime in regimes if regime not in regimes_cfg]
    if unknown_regimes:
        raise ValueError(f"Unknown regimes: {', '.join(unknown_regimes)}")
    unknown_materials = [material for material in materials if material not in sim_cfg["cases"]["materials"]]
    if unknown_materials:
        raise ValueError(f"Unknown materials: {', '.join(unknown_materials)}")

    parent = (root / args.runs_root).resolve() if not Path(args.runs_root).is_absolute() else Path(args.runs_root)
    batch_dir = parent / run_id()
    batch_dir.mkdir(parents=True, exist_ok=False)

    jobs: list[CaseJob] = []
    for regime in regimes:
        for material in materials:
            case_id = f"{args.case_prefix}{regime}_{material}{args.case_suffix}"
            command = build_command(
                root,
                regime,
                material,
                case_id,
                geometry=args.geometry,
                final_time=args.final_time,
                vtk_period=args.vtk_period,
                print_period=args.print_period,
                stat_reset_period=args.stat_reset_period,
                lbm_viscosity=args.lbm_viscosity,
                block_size=args.block_size,
            )
            case_run_dir = batch_dir / safe_token(case_id)
            case_run_dir.mkdir(parents=True, exist_ok=False)
            job = CaseJob(
                regime=regime,
                material=material,
                case_id=case_id,
                run_dir=case_run_dir,
                command=command,
                job_name=safe_token(f"{args.job_prefix}{regime}-{material}", max_len=48),
                sbatch_path=case_run_dir / "job.sbatch",
                manifest_path=case_run_dir / "manifest.json",
                result_dir=root / f"results_{case_id}",
            )
            job.sbatch_path.write_text(render_sbatch(root=root, job=job, args=args), encoding="utf-8")
            write_json(
                job.manifest_path,
                {
                    "regime": regime,
                    "material": material,
                    "case_id": case_id,
                    "job_name": job.job_name,
                    "created_at": iso_timestamp(),
                    "command": command,
                    "sbatch_path": str(job.sbatch_path),
                    "result_dir": str(job.result_dir),
                    "state": job.state,
                },
            )
            jobs.append(job)

    write_jobs_csv(batch_dir, jobs)
    return batch_dir, jobs


def load_jobs(batch_dir: Path) -> list[CaseJob]:
    jobs: list[CaseJob] = []
    for manifest_path in sorted(batch_dir.glob("*/manifest.json"), key=natural_key):
        data = read_json(manifest_path)
        case_run_dir = manifest_path.parent
        job = CaseJob(
            regime=data["regime"],
            material=data["material"],
            case_id=data["case_id"],
            run_dir=case_run_dir,
            command=list(data["command"]),
            job_name=data["job_name"],
            sbatch_path=Path(data["sbatch_path"]),
            manifest_path=manifest_path,
            result_dir=Path(data["result_dir"]),
            job_id=data.get("job_id"),
            state=data.get("state", "UNKNOWN"),
        )
        jobs.append(job)
    if not jobs:
        raise RuntimeError(f"No job manifests found under {batch_dir}")
    return jobs


def submit_job(job: CaseJob) -> None:
    proc = subprocess.run(
        ["sbatch", "--parsable", job.sbatch_path.name],
        cwd=job.run_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    job.job_id = proc.stdout.strip().split(";", 1)[0]
    job.state = "SUBMITTED"
    update_manifest(job, {"job_id": job.job_id, "state": job.state, "submitted_at": iso_timestamp()})


def query_job_states(job_ids: Sequence[str]) -> dict[str, str]:
    ids = [str(job_id) for job_id in job_ids if str(job_id)]
    states = {job_id: "UNKNOWN" for job_id in ids}
    if not ids:
        return states

    try:
        proc = subprocess.run(
            ["squeue", "-h", "-j", ",".join(ids), "-o", "%i|%T"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        proc = None
    if proc is not None and proc.returncode == 0:
        for line in proc.stdout.splitlines():
            raw_job_id, _, raw_state = line.partition("|")
            job_id = raw_job_id.strip()
            if job_id in states:
                states[job_id] = slurm_state_head(raw_state)

    unresolved = [job_id for job_id, state in states.items() if state == "UNKNOWN"]
    if not unresolved:
        return states

    try:
        proc = subprocess.run(
            [
                "sacct",
                "-j",
                ",".join(unresolved),
                "--format=JobIDRaw,State",
                "--parsable2",
                "--noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        proc = None
    if proc is None or proc.returncode != 0:
        return states

    unresolved_set = set(unresolved)
    for line in proc.stdout.splitlines():
        raw_job_id, _, raw_state = line.partition("|")
        job_id = raw_job_id.strip()
        if job_id in unresolved_set and states[job_id] == "UNKNOWN":
            states[job_id] = slurm_state_head(raw_state)
    return states


def wait_for_jobs(batch_dir: Path, jobs: list[CaseJob], poll_interval: float) -> None:
    missing = [job.case_id for job in jobs if not job.job_id]
    if missing:
        raise RuntimeError(f"Cannot wait for jobs without Slurm ids: {', '.join(missing)}")

    last_line = ""
    while True:
        states = query_job_states([job.job_id or "" for job in jobs])
        changed = False
        for job in jobs:
            assert job.job_id is not None
            state = states.get(job.job_id, "UNKNOWN")
            if state != job.state:
                job.state = state
                update_manifest(job, {"state": state, "state_updated_at": iso_timestamp()})
                changed = True

        counts: dict[str, int] = {}
        for job in jobs:
            counts[job.state] = counts.get(job.state, 0) + 1
        line = ", ".join(f"{state}={count}" for state, count in sorted(counts.items()))
        if changed or line != last_line:
            print(f"[wait] {iso_timestamp()} {line}", flush=True)
            write_jobs_csv(batch_dir, jobs)
            last_line = line

        if all(job.state in TERMINAL_STATES for job in jobs):
            return
        time.sleep(max(1.0, poll_interval))


def write_jobs_csv(batch_dir: Path, jobs: list[CaseJob]) -> None:
    path = batch_dir / "jobs.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "case_id",
                "regime",
                "material",
                "job_id",
                "state",
                "run_dir",
                "result_dir",
            ],
        )
        writer.writeheader()
        for job in jobs:
            writer.writerow(
                {
                    "case_id": job.case_id,
                    "regime": job.regime,
                    "material": job.material,
                    "job_id": job.job_id or "",
                    "state": job.state,
                    "run_dir": str(job.run_dir),
                    "result_dir": str(job.result_dir),
                }
            )


def run_postprocess(root: Path, job: CaseJob) -> dict[str, str]:
    pattern = job.result_dir / "vtk3D" / "rank000_data_*.vtk"
    vtk_files = sorted(glob.glob(str(pattern)), key=natural_key)
    if not vtk_files:
        raise FileNotFoundError(f"No VTK files found for {job.case_id}: {pattern}")
    output = job.result_dir / "flows.csv"
    cmd = [
        sys.executable,
        str(root / "postprocess" / "compute_flows.py"),
        *vtk_files,
        "--output",
        str(output),
    ]
    subprocess.run(cmd, cwd=root, check=True)
    with output.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No rows written to {output}")
    last = rows[-1]
    last.update({"flow_csv": str(output), "snapshot_count": str(len(vtk_files))})
    return last


def harvest(root: Path, batch_dir: Path, jobs: list[CaseJob]) -> int:
    rows: list[dict[str, str]] = []
    failures = 0
    for job in jobs:
        row: dict[str, str] = {
            "case_id": job.case_id,
            "regime": job.regime,
            "material": job.material,
            "job_id": job.job_id or "",
            "state": job.state,
        }
        if job.state != "COMPLETED":
            row["harvest_status"] = "skipped_non_completed"
            failures += 1
            rows.append(row)
            continue
        try:
            row.update(run_postprocess(root, job))
            row["harvest_status"] = "ok"
            update_manifest(job, {"flow_csv": row["flow_csv"], "harvested_at": iso_timestamp()})
            print(f"[harvest] {job.case_id}: {row['flow_csv']}", flush=True)
        except Exception as exc:
            row["harvest_status"] = "failed"
            row["harvest_error"] = str(exc)
            failures += 1
            print(f"[harvest] {job.case_id}: failed: {exc}", flush=True)
        rows.append(row)

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    summary = batch_dir / "flow_summary.csv"
    with summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[harvest] summary: {summary}", flush=True)
    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Submit TCPC material/regime cases to Slurm, wait for completion, and harvest flows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--regimes", nargs="*", default=None)
    parser.add_argument("--materials", nargs="*", default=None)
    parser.add_argument("--case-prefix", default="")
    parser.add_argument("--case-suffix", default="")
    parser.add_argument("--geometry", default=None)
    parser.add_argument("--final-time", type=float, default=None)
    parser.add_argument("--vtk-period", type=float, default=None)
    parser.add_argument("--print-period", type=float, default=None)
    parser.add_argument("--stat-reset-period", type=float, default=None)
    parser.add_argument("--lbm-viscosity", type=float, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--runs-root", default="runtime/slurm")
    parser.add_argument("--resume", type=Path, default=None, help="Existing runtime/slurm/run-* directory.")
    parser.add_argument("--dry-run", action="store_true", help="Prepare scripts/manifests but do not submit.")
    parser.add_argument("--submit-only", action="store_true", help="Submit jobs and exit without waiting.")
    parser.add_argument("--harvest-only", action="store_true", help="Do not submit or wait; only harvest an existing --resume run.")
    parser.add_argument("--no-harvest", action="store_true")
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument("--overwrite-results", action="store_true")
    parser.add_argument("--job-prefix", default="nnlbm-")
    parser.add_argument("--partition", default="gp")
    parser.add_argument("--time", default="12:00:00")
    parser.add_argument("--cpus-per-task", type=int, default=8)
    parser.add_argument("--mem", default="32G")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--gres", default=None, help="Use #SBATCH --gres=VALUE instead of --gpus.")
    parser.add_argument("--account", default=None)
    parser.add_argument("--qos", default=None)
    parser.add_argument("--constraint", default=None)
    parser.add_argument("--threads-per-core", type=int, default=1)
    parser.add_argument("--extra-sbatch", action="append", default=[], help="Extra raw SBATCH option, e.g. '--mail-type=END'.")
    parser.add_argument("--env", action="append", default=[], help="Extra job environment variable, NAME=VALUE.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()

    if args.harvest_only and args.resume is None:
        raise SystemExit("--harvest-only requires --resume")

    if args.resume is not None:
        batch_dir = args.resume.resolve()
        jobs = load_jobs(batch_dir)
        print(f"[resume] loaded {len(jobs)} jobs from {batch_dir}", flush=True)
    else:
        batch_dir, jobs = build_jobs(root, args)
        print(f"[prepare] batch_dir={batch_dir}", flush=True)
        for job in jobs:
            print(f"[prepare] {job.case_id}: {job.sbatch_path}", flush=True)

    if args.dry_run:
        print("[dry-run] no jobs submitted", flush=True)
        return 0

    if not args.resume and not args.harvest_only:
        for job in jobs:
            submit_job(job)
            print(f"[submit] {job.case_id}: job_id={job.job_id}", flush=True)
        write_jobs_csv(batch_dir, jobs)

    if args.submit_only:
        print(f"[submit-only] batch_dir={batch_dir}", flush=True)
        return 0

    if not args.harvest_only:
        wait_for_jobs(batch_dir, jobs, args.poll_interval)

    write_jobs_csv(batch_dir, jobs)
    if args.no_harvest:
        return 0 if all(job.state == "COMPLETED" for job in jobs) else 1

    failures = harvest(root, batch_dir, jobs)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
