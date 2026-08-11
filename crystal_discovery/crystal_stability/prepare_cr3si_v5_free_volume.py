#!/usr/bin/env python3
"""Prepare the two Cr3Si A15 large-free-volume stability reruns.

The v4 periodic-bulk cells put the crystal directly against its periodic images
(box 3608 nm, crystal extent 3495 nm, particle volume fraction about 0.449).
These v5 runs keep the same 4096-particle crystal and v4 dynamics, but enlarge
the box to volume fraction 0.0087.  Walls and gravity remain off, so this is a
finite crystal in periodic vacuum rather than a densely caged periodic bulk.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
TARGET_VOLUME_FRACTION = 0.0087
RUN_STEPS = 10_000_000
TRAJECTORY_INTERVAL = 20_000
TIMESTEP_PS = 0.15
CONDITIONS = {
    "cr3si_a15_v5_free_volume": "cr3si_a15",
    "cr3si_a15_deep8_v5_free_volume": "cr3si_a15_deep8",
}


def null_quantity(text: str, key: str) -> str:
    return re.sub(
        rf"^{key}:\s*!Quantity\s*\n(?:^[ \t]+.*\n)*",
        f"{key}: null\n",
        text,
        flags=re.MULTILINE,
    )


def v5_run_yaml(source: str) -> str:
    text = source
    target = re.search(
        r"^potential_temperature:.*?\n\s*unit:.*?\n\s*value:\s*([\d.]+)",
        text,
        flags=re.DOTALL | re.MULTILINE,
    ).group(1)
    text = re.sub(
        r"(integrator_parameters:.*?temperature:)\s*!Copy\s*\n\s*key:\s*\w+",
        r"\1 !Quantity\n    unit: kelvin\n    value: " + target,
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"(integrator_parameters:.*?temperature:\s*!Quantity\s*\n"
        r"\s*unit:\s*kelvin\s*\n\s*value:\s*)[\d.]+",
        r"\g<1>" + target,
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"^update_reporter_parameters:\s*\n"
        r"(?:^[ \t]+.*\n|^\s*\n(?=[ \t]))*",
        "",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(r"^update_reporter:.*\n", "", text, flags=re.MULTILINE)
    text = re.sub(
        r"^wall_directions:\s*\n(?:^\s*-\s*\w+\s*\n)+",
        "wall_directions:\n  - false\n  - false\n  - false\n",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(r"^use_gravity:.*$", "use_gravity: false", text, flags=re.MULTILINE)
    text = null_quantity(text, "epsilon")
    text = re.sub(r"^alpha:.*$", "alpha: null", text, flags=re.MULTILINE)
    for key in ("gravitational_acceleration", "water_density", "particle_density"):
        text = null_quantity(text, key)
    text = re.sub(
        r"(stepSize:\s*!Quantity\s*\n\s*unit:\s*picosecond\s*\n\s*value:\s*)[\d.]+",
        rf"\g<1>{TIMESTEP_PS}",
        text,
    )
    text = re.sub(r"^run_steps:.*$", f"run_steps: {RUN_STEPS}", text, flags=re.MULTILINE)
    text = re.sub(
        r"^trajectory_interval:.*$",
        f"trajectory_interval: {TRAJECTORY_INTERVAL}",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^state_data_interval:.*$",
        f"state_data_interval: {TRAJECTORY_INTERVAL // 10}",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^checkpoint_interval:.*$",
        f"checkpoint_interval: {TRAJECTORY_INTERVAL * 10}",
        text,
        flags=re.MULTILINE,
    )
    return text.rstrip() + "\n\nupdate_reporter: null\nupdate_reporter_parameters: null\n"


def job_text(destination: Path) -> str:
    name = destination.name
    return f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_146_chemistry
#SBATCH --comment="gpu_mps=yes"
#SBATCH --job-name=v5fv_{name.replace('cr3si_a15_', '')}
#SBATCH --output={HERE}/logs/%x.%j.out
#SBATCH --error={HERE}/logs/%x.%j.err
set -euo pipefail
cd {HERE}
source env.sh
cd {destination}
pacsim-create configuration.yaml first_frame.gsd
python {HERE}/set_box_density.py first_frame.gsd --target-vf {TARGET_VOLUME_FRACTION}
pacsim-run run.yaml
echo "done: $(pwd)/trajectory.gsd"
"""


def main() -> None:
    for destination_name, source_name in CONDITIONS.items():
        source = RUNS / source_name
        destination = RUNS / destination_name
        if (destination / "trajectory.gsd").exists():
            raise FileExistsError(f"refusing to overwrite completed run: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / "configuration.yaml", destination / "configuration.yaml")
        for cif in source.glob("*.cif"):
            shutil.copy2(cif, destination / cif.name)
        (destination / "run.yaml").write_text(v5_run_yaml((source / "run.yaml").read_text()))
        (destination / "job.sbatch").write_text(job_text(destination))
        manifest = {
            "protocol": "v5_large_free_volume_no_walls_no_gravity",
            "source_condition": source_name,
            "source_configuration": str((source / "configuration.yaml").resolve()),
            "source_run_yaml": str((source / "run.yaml").resolve()),
            "target_volume_fraction": TARGET_VOLUME_FRACTION,
            "run_steps": RUN_STEPS,
            "timestep_ps": TIMESTEP_PS,
            "trajectory_interval": TRAJECTORY_INTERVAL,
            "temperature_K": 298.0,
            "temperature_ramp": False,
            "walls": False,
            "gravity": False,
            "interpretation": (
                "finite crystal in a large periodic-vacuum box; tests free-surface "
                "retention and removes dense periodic caging"
            ),
        }
        (destination / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(destination)


if __name__ == "__main__":
    main()
