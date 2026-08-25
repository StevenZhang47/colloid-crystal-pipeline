#!/bin/bash
# Thermalise the CIF-ideal anchor lattices.
#
# WHY: generate_crystals.py stops after `pacsim-create`, which optimises the LATTICE
# CONSTANTS at zero temperature and writes first_frame.gsd. It never calls `pacsim-run`.
# So every lib_*/first_frame.gsd is a mathematically perfect lattice -- on lib_cscl the
# standard deviation of the nearest-neighbour distance is 4e-5 nm, i.e. every particle is
# exactly on its site. The matcher then compares a 298 K thermally-disordered CANDIDATE
# against a 0 K anchor, and part of every reported distance is that temperature gap
# rather than a structural difference.
#
# The fix is to run the run.yaml that build.py already wrote for each anchor. That is the
# SAME protocol the candidates were produced with (30 K -> 298 K ramp over 200k steps,
# 500k steps total, 500 frames), so anchor and candidate end up on equal footing.
#
# first_frame.gsd is NOT overwritten -- it stays as the 0 K reference so the before/after
# comparison remains available.
#
#   sbatch --array=0-41 thermalize_anchors.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=04:00:00
#SBATCH --account=torch_pr_146_chemistry
#SBATCH --comment="gpu_mps=yes"
#SBATCH --job-name=thermanchor
#SBATCH --output=logs/%x.%A_%a.out
#SBATCH --error=logs/%x.%A_%a.err
set -euo pipefail

CS=/projects/rps/gmh4/hockygroup/sz5808/crystal_discovery/crystal_stability
cd "$CS"
source env.sh

mapfile -t RUNS < <(for d in runs/lib_*/; do [ -f "$d/first_frame.gsd" ] && basename "$d"; done | sort)
run="${RUNS[$SLURM_ARRAY_TASK_ID]}"
echo "task $SLURM_ARRAY_TASK_ID / ${#RUNS[@]} -> $run"

cd "$CS/runs/$run"
if [ -f trajectory.gsd ] && [ -f final_frame.gsd ]; then
  echo "already thermalised, skipping"; exit 0
fi
pacsim-run run.yaml
echo "done: $(pwd)/trajectory.gsd"
