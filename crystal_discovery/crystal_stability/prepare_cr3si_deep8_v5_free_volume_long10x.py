#!/usr/bin/env python3
"""Prepare a 10x-total CUDA continuation of Cr3Si A15 deep8 free-volume v5.

The source checkpoint and final saved trajectory frame are both at 10,000,000
steps. PACSim interprets ``run_steps`` as additional steps after loading a
checkpoint, so this continuation adds 90,000,000 steps for 100,000,000 total
steps (15 microseconds at dt=0.15 ps). The completed source run is preserved.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import gsd.hoomd


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "runs/cr3si_a15_deep8_v5_free_volume"
DESTINATION = HERE / "runs/cr3si_a15_deep8_v5_free_volume_long10x"
SOURCE_STEPS = 10_000_000
ADDITIONAL_STEPS = 90_000_000
TARGET_TOTAL_STEPS = SOURCE_STEPS + ADDITIONAL_STEPS
TIMESTEP_PS = 0.15


def main() -> None:
    if DESTINATION.exists():
        raise FileExistsError(f"refusing to overwrite continuation: {DESTINATION}")
    required = (
        "configuration.yaml",
        "run.yaml",
        "first_frame.gsd",
        "trajectory.gsd",
        "state_data.csv",
        "checkpoint.chk",
    )
    for name in required:
        if not (SOURCE / name).exists():
            raise FileNotFoundError(SOURCE / name)

    with gsd.hoomd.open(str(SOURCE / "trajectory.gsd"), "r") as trajectory:
        last_saved_step = int(trajectory[-1].configuration.step)
    if last_saved_step != SOURCE_STEPS:
        raise RuntimeError(
            f"expected source trajectory at {SOURCE_STEPS}, found {last_saved_step}"
        )

    DESTINATION.mkdir(parents=True, exist_ok=False)
    for name in required:
        shutil.copy2(SOURCE / name, DESTINATION / name)
    for cif in SOURCE.glob("*.cif"):
        shutil.copy2(cif, DESTINATION / cif.name)

    run_path = DESTINATION / "run.yaml"
    text = run_path.read_text()
    text = re.sub(
        r"^run_steps:.*$",
        f"run_steps: {ADDITIONAL_STEPS}",
        text,
        flags=re.MULTILINE,
    )
    # Reporter thinning changes only output cadence, not the simulated dynamics.
    # This retains 900 additional full-particle frames across the continuation.
    text = re.sub(
        r"^trajectory_interval:.*$",
        "trajectory_interval: 100000",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^state_data_interval:.*$",
        "state_data_interval: 10000",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^checkpoint_interval:.*$",
        "checkpoint_interval: 1000000",
        text,
        flags=re.MULTILINE,
    )
    run_path.write_text(text)

    job = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_146_chemistry
#SBATCH --comment="gpu_mps=yes"
#SBATCH --job-name=cr3si_d8_v5fv_long10x
#SBATCH --output={HERE}/logs/%x.%j.out
#SBATCH --error={HERE}/logs/%x.%j.err
set -euo pipefail
cd {HERE}
source env.sh
cd {DESTINATION}
pacsim-run run.yaml -c checkpoint.chk
echo "done: $(pwd)/trajectory.gsd"
"""
    (DESTINATION / "job.sbatch").write_text(job)

    manifest = {
        "protocol": "checkpoint_continuation",
        "source_run": str(SOURCE.resolve()),
        "source_protocol": "v5_large_free_volume_no_walls_no_gravity",
        "source_checkpoint_step": SOURCE_STEPS,
        "additional_steps": ADDITIONAL_STEPS,
        "target_total_steps": TARGET_TOTAL_STEPS,
        "timestep_ps": TIMESTEP_PS,
        "source_duration_ps": SOURCE_STEPS * TIMESTEP_PS,
        "additional_duration_ps": ADDITIONAL_STEPS * TIMESTEP_PS,
        "target_total_duration_ps": TARGET_TOTAL_STEPS * TIMESTEP_PS,
        "target_total_duration_microseconds": TARGET_TOTAL_STEPS * TIMESTEP_PS / 1e6,
        "platform": "CUDA",
        "interaction_parameters_changed": False,
        "temperature_schedule_changed": False,
        "box_changed": False,
        "walls_changed": False,
        "gravity_changed": False,
        "reporter_intervals_only_changed": True,
        "original_run_preserved": True,
    }
    (DESTINATION / "protocol_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(DESTINATION)


if __name__ == "__main__":
    main()
