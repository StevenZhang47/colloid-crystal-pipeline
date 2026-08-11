#!/bin/bash
# Build and relax one crystal on a GPU. This is the one you want: the 6750-particle
# CsCl run is ~38 h on 8 CPU threads, but well under an hour on a GPU.
#
#   sbatch submit_gpu.sbatch systems/cscl.yaml --platform CUDA
#   sbatch submit_gpu.sbatch systems/cscl.yaml --platform CUDA --debye 4.1 --tag melt_some
#
# Extra arguments pass straight through to build.py, so a melting scan is just
# three submissions with different --debye/--tag.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=08:00:00
#SBATCH --account=torch_pr_146_chemistry
#SBATCH --comment="gpu_mps=yes"
#SBATCH --job-name=stability
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source env.sh

run_dir=$(python build.py "$@" | sed -n 's|^wrote \(.*\)/configuration.yaml$|\1|p')
echo "run directory: ${run_dir}"
cd "${run_dir}"

pacsim-create configuration.yaml first_frame.gsd
pacsim-run run.yaml

echo "done: ${run_dir}/trajectory.gsd"
