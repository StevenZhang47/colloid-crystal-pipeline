#!/bin/bash
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=8 --mem=48GB
#SBATCH --time=03:00:00 --partition=cpu_short --account=torch_pr_146_general
#SBATCH --job-name=refbank --output=logs/%x.%j.out --error=logs/%x.%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/sz5808/.conda/envs/pacsim
export OPENMM_PLUGIN_DIR=/scratch/sz5808/.conda/envs/pacsim/lib/plugins
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
python build_reference_bank.py
