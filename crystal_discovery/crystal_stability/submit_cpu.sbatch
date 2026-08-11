#!/bin/bash
# Build and relax one crystal.
#
#   sbatch submit.sbatch systems/cscl.yaml
#   sbatch submit.sbatch systems/cscl.yaml --debye 4.1 --tag melt_some
#
# Any extra arguments are passed straight through to build.py, so a melting scan
# is just three submissions with different --debye/--tag.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=12:00:00
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_146_general
#SBATCH --job-name=stability
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source env.sh
export OPENMM_CPU_THREADS=${SLURM_CPUS_PER_TASK}

# build.py prints the run directory it wrote; capture it so the run lands there.
run_dir=$(python build.py "$@" | sed -n 's|^wrote \(.*\)/configuration.yaml$|\1|p')
echo "run directory: ${run_dir}"
cd "${run_dir}"

pacsim-create configuration.yaml first_frame.gsd
pacsim-run run.yaml

echo "done: ${run_dir}/trajectory.gsd"
