#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=06:00:00
#SBATCH --partition=cpu_short
#SBATCH --account=torch_pr_146_general
#SBATCH --job-name=recalibd
#SBATCH --output=logs/%x.%j.out
#SBATCH --error=logs/%x.%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/sz5808/.conda/envs/pacsim
export OPENMM_PLUGIN_DIR=/scratch/sz5808/.conda/envs/pacsim/lib/plugins
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
python recalibrate_accept_d.py
