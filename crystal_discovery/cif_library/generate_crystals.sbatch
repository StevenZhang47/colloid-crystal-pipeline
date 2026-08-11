#!/bin/bash
# Generate one colloidal crystal per library CIF, as a SLURM array.
#
#   sbatch --array=0-42 generate_crystals.sbatch
#
# Array index -> the i-th slug of `python generate_crystals.py --list`.
# Each task: design.py -> build.py -> pacsim-create, then a cohesion check.
# Output lands in crystal_stability/runs/lib_<slug>/first_frame.gsd, which is
# exactly what the EMD matcher reads as a family anchor.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=04:00:00
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_146_general
#SBATCH --job-name=libcrystal
#SBATCH --output=logs/%x.%A_%a.out
#SBATCH --error=logs/%x.%A_%a.err
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source ../crystal_stability/env.sh
export OPENMM_CPU_THREADS=${SLURM_CPUS_PER_TASK}

# The anisotropic LatticeBuilder (PACSim >= 582a2d0) must shadow the main checkout.
# generate_crystals.py asserts this rather than trusting it.
export PYTHONPATH=/projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/PACSim_aniso

# FORCE=1 re-designs and rebuilds even where first_frame.gsd already exists.
python generate_crystals.py --index "${SLURM_ARRAY_TASK_ID}" ${FORCE:+--force}
